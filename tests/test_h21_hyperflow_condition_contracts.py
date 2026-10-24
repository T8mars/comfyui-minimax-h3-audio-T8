"""Actual tiny native50+2 HyperFlow conditioning and inpaint continuation.

Only adapter tensors and text embeddings are synthetic. No FakeForward,
sampler kernel, Core inpaint function or learned weight loader is substituted.
This tests CPU numerical contracts, never trained H3 media quality.
"""
from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import comfy.nested_tensor
import comfy.sample
import pytest
import torch

from h3_audio_t8_pkg import hyperflow_sampling_advanced as native
from h3_audio_t8_pkg import hyperflow_two_pass_advanced as combined
from h3_audio_t8_pkg import hyperflow_runtime_advanced as runtime
from h3_audio_t8_pkg import prompt_relay_advanced as relay
from h3_audio_t8_pkg.modular_sampling import hyperflow as separate
from h3_audio_t8_pkg.modular_sampling import hyperflow_effects as effects, eav
from h3_audio_t8_pkg.modular_sampling import hyperflow_storage as storage
from h3_audio_t8_pkg.nodes_hyperflow_advanced import MiniMaxH3HyperFlowSplitT8Advanced
from h3_audio_t8_pkg.progressive_sampling_runtime import _restore_stage_objects
from test_modular_hyperflow import inputs


def parts(value):
    return value["samples"].unbind()


def equal_av(left, right):
    for actual, expected in zip(parts(left), parts(right), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def nonzero_source(source, mask=None):
    video, audio = parts(source)
    result = {"samples": comfy.nested_tensor.NestedTensor((
        torch.linspace(.1, .4, video.numel()).reshape(video.shape),
        torch.linspace(-.2, .3, audio.numel()).reshape(audio.shape)))}
    if mask:
        vm, am = (torch.ones_like(part) for part in parts(result))
        if mask in {"binary", "soft"}:
            vm[:, :, 0, :, 0] = 0
            am[..., :3] = 0
        if mask == "soft":
            vm[:, :, 1, :, 1] = .5
            am[..., 3:5] = .5
        if mask == "audio_locked":
            am.zero_()
        result["noise_mask"] = comfy.nested_tensor.NestedTensor((vm, am))
    return result


def full8(model, source, positive, *, seed=21102193):
    branch, sampler, sigmas = native.setup_hyperflow_sampler(
        model, source, native.build_hyperflow_plan(model))
    try:
        output = comfy.sample.sample_custom(branch, comfy.sample.prepare_noise(source["samples"], seed),
            1., sampler, sigmas, positive, [], source["samples"],
            noise_mask=source.get("noise_mask"), disable_pbar=True, seed=seed)
    finally:
        _restore_stage_objects(branch)
    return {**source, "samples": output}


def explicit_independent_pair(low, high, source, positive, high_positive, *, seed=21102193):
    """Reference: actual two sampler calls with captured state and distinct text."""
    captured = {}
    head_plan = native.build_hyperflow_plan(low, 0, 4)
    head, sampler, sigmas = native.setup_hyperflow_sampler(low, source, head_plan,
        internal_continuation=True, endpoint_capture=lambda value: captured.update(x=value))
    try:
        boundary = comfy.sample.sample_custom(head, comfy.sample.prepare_noise(source["samples"], seed),
            1., sampler, sigmas, positive, [], source["samples"], disable_pbar=True, seed=seed)
    finally:
        _restore_stage_objects(head)
    tail, sampler, sigmas = native.setup_hyperflow_sampler(high, source,
        native.build_hyperflow_plan(high, 4, 8), internal_continuation=True,
        x_sigma_override=lambda: captured["x"])
    try:
        output = comfy.sample.sample_custom(tail, comfy.sample.prepare_empty_noise(boundary),
            1., sampler, sigmas, high_positive, [], boundary, disable_pbar=True, seed=seed)
    finally:
        _restore_stage_objects(tail)
    return {**source, "samples": output}


def test_combined_accepts_independent_high_conditioning_and_preserves_legacy_defaults(monkeypatch):
    low, high, source, positive = inputs(monkeypatch, distinct=True)
    source = nonzero_source(source)
    high_positive = deepcopy(positive)
    high_positive[0][0].add_(.4)
    expected = explicit_independent_pair(low, high, source, positive, high_positive)
    source_before = [value.clone() for value in parts(source)]
    weights_before = [value.detach().clone() for value in low.model.parameters()]
    options_before = deepcopy(low.model_options)
    callbacks = []
    actual, report = combined.sample_hyperflow_split(low, high, positive, source,
        positive_high=high_positive, negative_high=[], seed=21102193,
        callback=lambda step, _p, _x, total: callbacks.append((step, total)))
    equal_av(actual, expected)
    legacy, _ = combined.sample_hyperflow_split(low, high, positive, source, seed=21102193)
    defaulted, _ = combined.sample_hyperflow_split(low, high, positive, source,
        positive_high=None, negative_high=None, seed=21102193)
    equal_av(legacy, defaulted)
    assert any(not torch.equal(a, b) for a, b in zip(parts(actual), parts(legacy), strict=True))
    assert callbacks == [(i, 8) for i in range(8)]
    assert json.loads(report)["total_nfe"] == 8
    assert low.model_options == options_before
    assert all(torch.equal(a, b) for a, b in zip(source_before, parts(source), strict=True))
    assert all(torch.equal(a, b) for a, b in zip(weights_before, low.model.parameters(), strict=True))
    assert runtime.active_step() is None and runtime._ACTIVE_FORWARD.get() is None
    assert not torch.cuda.is_initialized()


def test_split_node_appends_optional_independent_condition_inputs():
    schema = MiniMaxH3HyperFlowSplitT8Advanced.define_schema()
    names = [item.id for item in schema.inputs]
    assert names[:8] == ["model_low", "model_high", "positive", "av_latent", "seed", "split_interval", "cfg", "negative"]
    assert names[8:] == ["positive_high", "negative_high"]
    assert all(item.optional for item in schema.inputs[8:])


@pytest.mark.parametrize("split", [1, 4, 7])
@pytest.mark.parametrize("mask", ["ones", "binary", "soft", "audio_locked"])
@pytest.mark.parametrize("route", ["combined", "separate"])
def test_valid_masks_match_actual_uninterrupted_single8(monkeypatch, request, split, mask, route):
    low, high, source, positive = inputs(monkeypatch)
    source = nonzero_source(source, mask)
    source_before = [value.clone() for value in parts(source)]
    mask_before = [value.clone() for value in source["noise_mask"].unbind()]
    expected = full8(low, source, positive)
    calls = []
    if route == "combined":
        actual, raw = combined.sample_hyperflow_split(low, high, positive, source,
            seed=21102193, split_interval=split,
            callback=lambda step, _p, _x, total: calls.append((step, total)))
        assert json.loads(raw)["stage_nfe"] == [split, 8 - split]
    else:
        boundary = separate.sample_head(low, source, positive, [],
            comfy.sample.prepare_noise(source["samples"], 21102193),
            seed=21102193, split_interval=split,
            callback=lambda step, _p, _x, total: calls.append((step, total)))
        assert boundary.verify()["execution"]["actual_apply_intervals"] == list(range(split))
        actual, raw = separate.sample_tail(boundary, high, positive, [], seed=21102193,
            callback=lambda step, _p, _x, total: calls.append((step, total)))
        result = json.loads(raw)
        assert result["execution"]["actual_apply_intervals"] == list(range(split, 8))
        assert result["head_executed"] is result["fresh_noise"] is False
    equal_av(actual, expected)
    assert calls == [(i, 8) for i in range(8)]
    if mask == "audio_locked":
        # The original uninterrupted Core path itself differs from this source
        # by up to7.45e-8 in FP32 masked Euler arithmetic. Direct process-in/out
        # roundtrip alone is exact. Keep the stronger bit-exact full8 comparison
        # above, while bounding source preservation to one FP32 epsilon.
        native_delta = (parts(expected)[1] - parts(source)[1]).abs()
        request.node.user_properties.append(("native_locked_audio_source_max_abs", float(native_delta.max())))
        torch.testing.assert_close(parts(actual)[1], parts(source)[1],
            rtol=0, atol=torch.finfo(torch.float32).eps)
    assert all(torch.equal(a, b) for a, b in zip(source_before, parts(source), strict=True))
    assert all(torch.equal(a, b) for a, b in zip(mask_before, source["noise_mask"].unbind(), strict=True))
    assert all(torch.isfinite(part).all() for part in parts(actual))
    assert runtime.active_step() is None and runtime._ACTIVE_FORWARD.get() is None
    assert not torch.cuda.is_initialized()


def resized_source(source, kind):
    source = nonzero_source(source)
    raw = torch.tensor([[0., 1., .5], [.5, 0., 1.]])
    if kind == "video_only_2d":
        source["noise_mask"] = raw
    elif kind == "video_only_4d":
        source["noise_mask"] = raw.expand(1, 3, 2, 3).clone()
    elif kind == "single_nested":
        source["noise_mask"] = comfy.nested_tensor.NestedTensor((raw,))
    else:
        source["noise_mask"] = comfy.nested_tensor.NestedTensor(tuple(torch.zeros_like(part) for part in parts(source)))
    return source


@pytest.mark.parametrize("kind", ["video_only_2d", "video_only_4d", "single_nested", "all_zero"])
@pytest.mark.parametrize("route", ["combined", "separate"])
def test_actual_core_mask_resize_defaults_and_all_zero_are_not_rejected(monkeypatch, kind, route):
    low, high, source, positive = inputs(monkeypatch)
    source = resized_source(source, kind)
    expected = full8(low, source, positive)
    if route == "combined":
        actual, _ = combined.sample_hyperflow_split(low, high, positive, source, seed=21102193)
    else:
        boundary = separate.sample_head(low, source, positive, [],
            comfy.sample.prepare_noise(source["samples"], 21102193), seed=21102193)
        actual, _ = separate.sample_tail(boundary, high, positive, [], seed=21102193)
    equal_av(actual, expected)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("fault", ["rank1", "empty", "nan", "inf", "negative", "above_one", "too_many_channels"])
def test_real_invalid_mask_contract_is_still_rejected(monkeypatch, fault):
    low, high, source, positive = inputs(monkeypatch)
    source = nonzero_source(source)
    source["noise_mask"] = {
        "rank1": torch.ones(1), "empty": torch.empty(0, 2),
        "nan": torch.full((2, 2), float("nan")), "inf": torch.full((2, 2), float("inf")),
        "negative": torch.full((2, 2), -.01), "above_one": torch.full((2, 2), 1.01),
        "too_many_channels": torch.ones(1, 25, 2, 2, 2),
    }[fault]
    with pytest.raises(ValueError, match="mask"):
        combined.sample_hyperflow_split(low, high, positive, source, seed=21102193)
    assert runtime.active_step() is None and runtime._ACTIVE_FORWARD.get() is None


def masked_head(monkeypatch, mask="binary"):
    low, high, source, positive = inputs(monkeypatch)
    source = nonzero_source(source, mask)
    boundary = separate.sample_head(low, source, positive, [],
        comfy.sample.prepare_noise(source["samples"], 21102193), seed=21102193)
    return boundary, low, high, source, positive


def clone_latent(value):
    cloned = dict(value)
    cloned["samples"] = comfy.nested_tensor.NestedTensor(tuple(part.clone() for part in parts(value)))
    if "noise_mask" in value:
        mask = value["noise_mask"]
        cloned["noise_mask"] = (comfy.nested_tensor.NestedTensor(tuple(part.clone() for part in mask.unbind()))
                                if mask.is_nested else mask.clone())
    return cloned


@pytest.mark.parametrize("fault", ["source", "noise", "mask"])
def test_self_consistent_output_receipt_cannot_replace_original_head_inpaint_request(monkeypatch, fault):
    boundary, _, _, _, _ = masked_head(monkeypatch)
    source = clone_latent(boundary.inpaint["source"])
    noise = comfy.nested_tensor.NestedTensor(tuple(part.clone() for part in boundary.inpaint["noise"].unbind()))
    scaffold = clone_latent(boundary.scaffold)
    with torch.inference_mode():
        if fault == "source":
            source["samples"].unbind()[0].flatten()[0] += .05
        elif fault == "noise":
            noise.unbind()[1].flatten()[0] += .05
        else:
            source["noise_mask"].unbind()[0].flatten()[0] = .25
            scaffold["noise_mask"].unbind()[0].flatten()[0] = .25
    changed = replace(boundary, scaffold=scaffold, inpaint={"source": source, "noise": noise})
    # Recompute the output/outer checksums, retaining the original actual HEAD
    # request. Otherwise this would only retest a trivial SHA mismatch.
    receipt = json.loads(boundary.receipt_json)
    receipt["outputs"] = separate.snapshot(changed.state())
    receipt.pop("receipt_sha256")
    receipt["receipt_sha256"] = separate.sha(receipt)
    changed = replace(changed, receipt_json=separate.canonical(receipt))
    with pytest.raises(ValueError, match="actual HEAD request"):
        changed.verify()
    boundary.verify()


def test_masked_context_is_required_and_old_unmasked_state_remains_two_fields(monkeypatch):
    boundary, low, _, _, positive = masked_head(monkeypatch)
    with pytest.raises(ValueError, match="inpaint"):
        replace(boundary, inpaint=None).verify()
    source = {"samples": comfy.nested_tensor.NestedTensor(tuple(torch.zeros_like(part) for part in parts(boundary.scaffold)))}
    unmasked = separate.sample_head(low, source, positive, [],
        comfy.sample.prepare_noise(source["samples"], 21102193), seed=21102193)
    assert unmasked.inpaint is None
    assert set(unmasked.state()) == {"x_sigma", "scaffold"}
    assert "inpaint_reference" not in unmasked.verify()["request"]


@pytest.mark.parametrize("mask", ["binary", "audio_locked"])
def test_frozen_masked_head_replays_real_cold_tail_without_drawing_noise(tmp_path, monkeypatch, mask):
    boundary, _, high, _, positive = masked_head(monkeypatch, mask)
    expected, _ = separate.sample_tail(boundary, high, positive, [], seed=21102193)
    path, digest, _ = storage.save_boundary(boundary, tmp_path)
    loaded, _ = storage.load_boundary(tmp_path, path, digest)
    assert loaded.receipt_json == boundary.receipt_json
    assert separate.snapshot(loaded.inpaint) == separate.snapshot(boundary.inpaint)
    code = r'''
import gc,json,runpy,sys
runpy.run_path('tools/check_director_d1_cpu.py',run_name='config')
runpy.run_path('tests/conftest.py')
import pytest,torch,comfy.sample
from test_modular_hyperflow import inputs
from h3_audio_t8_pkg.modular_sampling import hyperflow as stages,hyperflow_storage as storage
def forbidden(*a,**kw):
    raise AssertionError('HEAD or fresh NOISE ran during cold masked TAIL')
with pytest.MonkeyPatch.context() as patch:
    low,high,source,positive=inputs(patch)
    boundary,_=storage.load_boundary(sys.argv[1],sys.argv[2],sys.argv[3])
    patch.setattr(stages,'sample_head',forbidden)
    patch.setattr(comfy.sample,'prepare_noise',forbidden)
    output,text=stages.sample_tail(boundary,high,positive,[],seed=21102193)
    report=json.loads(text)
    assert report['head_executed'] is report['fresh_noise'] is False
    assert report['execution']['actual_apply_intervals']==[4,5,6,7]
    assert not torch.cuda.is_initialized()
    print('RESULT='+json.dumps(stages.snapshot(output),sort_keys=True))
del low,high,source,positive,boundary,output
gc.collect()
'''
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    import folder_paths
    env["PYTHONPATH"] = str(Path(folder_paths.__file__).resolve().parent) + os.pathsep + env.get("PYTHONPATH", "")
    child = subprocess.run([sys.executable, "-X", "utf8", "-B", "-c", code, str(tmp_path), path, digest],
        cwd=root, env=env, capture_output=True, text=True, encoding="utf8", timeout=90)
    assert child.returncode == 0, child.stdout + child.stderr
    actual = json.loads(next(line[7:] for line in child.stdout.splitlines() if line.startswith("RESULT=")))
    assert actual == separate.snapshot(expected)
    assert storage.file_sha(tmp_path / path) == digest
    assert not torch.cuda.is_initialized()


def real_paired_relay(model, positive, source, name):
    """Synthetic text/event fixture, actual layout/router/owner and attention."""
    positive = deepcopy(positive)
    positive[0][0][0, 0] = torch.linspace(-1, 1, 8)
    positive[0][0][0, 1] = torch.linspace(1, -1, 8)
    video, audio = parts(source)
    layout = relay.build_packed_layout(2, video.shape[2], video.shape[3], video.shape[4],
        audio.shape[-1], keyframes=[], refs=[], frame_count=5)
    binding = relay._bind_layout_contract({"schema": relay.PROMPT_RELAY_PATCH_VERSION,
        "plan_hash": name, "text_len": 2, "query_route": "joint_av_exp",
        "events": [{"text_key_start": i, "text_key_end": i+1, "midpoint": float(i*4),
                    "window": .1, "sigma": .2} for i in range(2)]},
        layout, resolved_task="t2va", keyframes=[], refs=[])
    marked = [[value, {**metadata, relay.PROMPT_RELAY_BINDING_KEY: binding}] for value, metadata in positive]
    marked = relay._attach_binding_model_cond(marked, binding["binding_hash"])
    bound, _ = relay.patch_prompt_relay_model(model, binding, 32)
    return bound, marked


@pytest.mark.parametrize("with_relay", [False, True])
@pytest.mark.parametrize("mode", ["report_only", "apply_exp"])
def test_masked_eav_and_independently_paired_relay_observe_actual_interval_calls(monkeypatch, mode, with_relay):
    low, high, source, positive = inputs(monkeypatch)
    source = nonzero_source(source, "binary")
    high_positive = deepcopy(positive)
    if with_relay:
        low, positive = real_paired_relay(low, positive, source, "head-events")
        high, high_positive = real_paired_relay(high, high_positive, source, "tail-events")
    bound = effects.bind_head(low, source, positive, [], 4)
    config = eav.EAVConfig(mode, .2, 0., 1., 32, 3.)
    low, _, _ = eav.apply_stage_eav(bound[0], bound[4], source, bound[5], config)
    boundary = separate.sample_head(low, source, positive, [],
        comfy.sample.prepare_noise(source["samples"], 21102193), seed=21102193)
    if with_relay:
        with pytest.raises(ValueError, match="not paired"):
            effects.bind_tail(boundary, high, positive, [])
    bound = effects.bind_tail(boundary, high, high_positive, [])
    high, _, _ = eav.apply_stage_eav(bound[0], bound[4], boundary.scaffold, bound[5], config)
    result, _ = separate.sample_tail_result(boundary, high, high_positive, [], seed=21102193)
    for value, phase in ((boundary, "head"), (result, "tail")):
        audit = effects.audit(value, phase)
        assert audit["composition_verified"] is True
        assert audit["eav"]["selector_calls"] == 200
        assert audit["eav"]["completed_forwards"] == 4
        if with_relay:
            assert audit["relay"]["routed_attention_calls"] == 200
            assert audit["relay"]["completed_forwards"] == 4
    assert all(torch.isfinite(part).all() for part in parts(result.output))
    assert not torch.cuda.is_initialized()


def test_failed_split_restores_owners_then_next_independent_request_is_exact(monkeypatch):
    low, high, source, positive = inputs(monkeypatch)
    source = nonzero_source(source)
    high_positive = deepcopy(positive)
    high_positive[0][0].add_(.2)
    expected = explicit_independent_pair(low, high, source, positive, high_positive)

    def abort(index, *_args):
        if index == 5:
            raise RuntimeError("independent tail cancelled")

    with pytest.raises(RuntimeError, match="independent tail cancelled"):
        combined.sample_hyperflow_split(low, high, positive, source,
            positive_high=high_positive, seed=21102193, callback=abort)
    assert runtime.active_step() is None and runtime._ACTIVE_FORWARD.get() is None
    actual, _ = combined.sample_hyperflow_split(low, high, positive, source,
        positive_high=high_positive, seed=21102193)
    equal_av(actual, expected)
    assert not low.object_patches_backup and not high.object_patches_backup
    assert not torch.cuda.is_initialized()
