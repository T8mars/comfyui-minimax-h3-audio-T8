"""Actual native50+2 tiny HyperFlow, synthetic adapter; no trained GPU claim."""
from dataclasses import replace
import json

import pytest
import torch
import comfy.nested_tensor
import comfy.patcher_extension
import comfy.sample

from h3_audio_t8_pkg.modular_sampling import hyperflow as stages
from h3_audio_t8_pkg import hyperflow_sampling_advanced as native
from h3_audio_t8_pkg import hyperflow_runtime_advanced as runtime
from h3_audio_t8_pkg import hyperflow_two_pass_advanced as legacy
from h3_audio_t8_pkg.progressive_sampling_runtime import _restore_stage_objects
from test_hyperflow_advanced import _tiny_native_model, _tiny_weights
from test_progressive_sampling_runtime import tiny_model, conditioning


def inputs(monkeypatch, *, distinct=False):
    _, diffusion = _tiny_native_model(monkeypatch)
    base = tiny_model()
    base.model.diffusion_model = diffusion
    weights = _tiny_weights(diffusion)
    low, _, _ = runtime.install_hyperflow(base, weights)
    high, _, _ = runtime.install_hyperflow(base, weights)
    if distinct:
        key, value = next((key, value) for key, value in base.model.named_parameters() if value.ndim == 2)
        low.add_patches({key: ("diff", (torch.full_like(value, .01),))}, .7)
        high.add_patches({key: ("diff", (torch.full_like(value, -.02),))}, .4)
    source = {"samples": comfy.nested_tensor.NestedTensor((
        torch.zeros(1, 24, 2, 2, 2), torch.zeros(1, 32, 2, 8)))}
    return low, high, source, conditioning()


def head(low, source, positive, split=4, **kwargs):
    return stages.sample_head(low, source, positive, [], comfy.sample.prepare_noise(source["samples"], 7),
                              seed=7, split_interval=split, **kwargs)


def equal(left, right):
    for a, b in zip(left["samples"].unbind(), right["samples"].unbind()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("split", [1, 4, 7])
@pytest.mark.parametrize("distinct", [False, True])
def test_native_split_parity_with_two_real_loader_owners(monkeypatch, split, distinct):
    low, high, source, positive = inputs(monkeypatch, distinct=distinct)
    original, _ = legacy.sample_hyperflow_split(low, high, positive, source, seed=7, split_interval=split)
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Separate stages cannot call the old whole-split executor")
    monkeypatch.setattr(legacy, "sample_hyperflow_split", forbidden)
    notifications = []
    def callback(index, _prediction, _state, total):
        notifications.append((index, total))
    boundary = head(low, source, positive, split, callback=callback)
    receipt = boundary.verify()
    assert receipt["execution"]["actual_apply_intervals"] == list(range(split))
    output, text = stages.sample_tail(boundary, high, positive, [], seed=7, callback=callback)
    equal(original, output)
    report = json.loads(text)
    assert report["execution"]["actual_apply_intervals"] == list(range(split, 8))
    assert report["head_executed"] is report["fresh_noise"] is report["learned_upscale"] is False
    assert notifications == [(i, 8) for i in range(8)]
    assert runtime.active_step() is None
    assert not low.object_patches_backup and not high.object_patches_backup


@pytest.mark.parametrize("split", [1, 4, 7])
def test_continuation_matches_true_single8_and_high_never_draws_noise(monkeypatch, split):
    low, high, source, positive = inputs(monkeypatch)
    branch, sampler, sigmas = native.setup_hyperflow_sampler(low, source, native.build_hyperflow_plan(low))
    try:
        full = comfy.sample.sample_custom(branch, comfy.sample.prepare_noise(source["samples"], 7),
            1., sampler, sigmas, positive, [], source["samples"], disable_pbar=True, seed=7)
    finally:
        _restore_stage_objects(branch)
    boundary = head(low, source, positive, split)
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Continuous HIGH cannot draw fresh noise")
    monkeypatch.setattr(comfy.sample, "prepare_noise", forbidden)
    output, _ = stages.sample_tail(boundary, high, positive, [], seed=7)
    equal({"samples": full}, output)


def test_high_conditions_are_independent_without_reexecuting_head(monkeypatch):
    low, high, source, positive = inputs(monkeypatch, distinct=True)
    boundary = head(low, source, positive)
    before = boundary.receipt_json
    high_positive = conditioning()
    high_positive[0][0].add_(.4)
    plan = native.build_hyperflow_plan(high, 4, 8)
    branch, sampler, sigmas = native.setup_hyperflow_sampler(high, boundary.scaffold, plan,
        internal_continuation=True, x_sigma_override=lambda: boundary.x_sigma)
    try:
        expected = comfy.sample.sample_custom(branch, comfy.sample.prepare_empty_noise(boundary.scaffold["samples"]),
            1., sampler, sigmas, high_positive, [], boundary.scaffold["samples"], disable_pbar=True, seed=19)
    finally:
        _restore_stage_objects(branch)
    monkeypatch.setattr(stages, "sample_head", lambda *_args, **_kwargs: pytest.fail("Hidden HEAD rerun"))
    actual, report = stages.sample_tail(boundary, high, high_positive, [], seed=19)
    equal({"samples": expected}, actual)
    assert before == boundary.receipt_json
    assert json.loads(report)["request"]["seed"] == 19


@pytest.mark.parametrize("fault", ["raw", "scaffold", "receipt", "other_base", "artifact", "clock"])
def test_bad_boundary_or_incompatible_tail_rejected_before_forward(monkeypatch, fault):
    low, high, source, positive = inputs(monkeypatch)
    boundary = head(low, source, positive)
    with torch.inference_mode():
        if fault == "raw":
            boundary.x_sigma.flatten()[0] += .1
        elif fault == "scaffold":
            boundary.scaffold["samples"].unbind()[0].flatten()[0] += .1
        elif fault == "receipt":
            receipt = json.loads(boundary.receipt_json)
            receipt["execution"]["callbacks"] = []
            boundary = replace(boundary, receipt_json=json.dumps(receipt))
        elif fault == "other_base":
            _, high, _, _ = inputs(monkeypatch)
            # A rebuilt byte-identical native base is now deliberately valid
            # for persistent TAIL. This negative case must differ in content.
            high.model.diffusion_model.blocks[0].attn.qkv_proj.weight.add_(.1)
        else:
            binding = high.get_attachment(runtime.ATTACHMENT_KEY)
            changed = {"sha256": "d" * 64} if fault == "artifact" else {"audio_shift": 2.}
            high.set_attachments(runtime.ATTACHMENT_KEY, replace(binding, **changed))
    monkeypatch.setattr(stages, "_run", lambda *_a, **_k: pytest.fail("Invalid boundary reached sampling"))
    with pytest.raises(ValueError):
        stages.sample_tail(boundary, high, positive, [], seed=7)


@pytest.mark.parametrize("phase", ["head", "tail"])
def test_cancel_cleanup_retry_and_unknown_wrapper_delegate(monkeypatch, phase):
    low, high, source, positive = inputs(monkeypatch)
    calls = []
    def wrapper(executor, *args, **kwargs):
        calls.append(runtime.active_step().absolute_index)
        return executor(*args, **kwargs)
    selected = low if phase == "head" else high
    selected.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.APPLY_MODEL, "user_unknown", wrapper)
    boundary = None if phase == "head" else head(low, source, positive)
    def cancel(index, *_):
        if index % 4 == 1:
            raise RuntimeError("cancel this stage")
    with pytest.raises(RuntimeError, match="cancel this stage"):
        if phase == "head":
            head(low, source, positive, callback=cancel)
        else:
            stages.sample_tail(boundary, high, positive, [], seed=7, callback=cancel)
    assert runtime.active_step() is None and runtime._ACTIVE_FORWARD.get() is None
    calls.clear()
    if phase == "head":
        result = head(low, source, positive)
        assert result.verify()["execution"]["portable_identity"] is False
    else:
        _, report = stages.sample_tail(boundary, high, positive, [], seed=7)
        assert json.loads(report)["execution"]["portable_identity"] is False
    assert calls == list(range(0, 4) if phase == "head" else range(4, 8))
    assert "user_unknown" in selected.wrappers[comfy.patcher_extension.WrappersMP.APPLY_MODEL]


@pytest.mark.parametrize("fault", ["mask", "bad_noise", "bad_split", "bad_cfg", "bad_seed", "bad_reserve"])
def test_invalid_head_input_never_invokes_native_stage(monkeypatch, fault):
    low, high, source, positive = inputs(monkeypatch)
    noise = comfy.sample.prepare_noise(source["samples"], 7)
    kwargs = dict(seed=7, split_interval=4)
    if fault == "mask":
        source["noise_mask"] = torch.ones(1)
    elif fault == "bad_noise":
        noise.unbind()[1][..., 0] = float("nan")
    else:
        kwargs.update({"bad_split": {"split_interval": True}, "bad_cfg": {"cfg": float("nan")},
            "bad_seed": {"seed": -1}, "bad_reserve": {"reserve_vram_mib": 128}}[fault])
    monkeypatch.setattr(stages.cleanup, "_native_stage", lambda *_a, **_k: pytest.fail("Bad input sampled"))
    with pytest.raises(ValueError):
        stages.sample_head(low, source, positive, [], noise, **kwargs)


@pytest.mark.parametrize("phase", ["head", "tail"])
@pytest.mark.parametrize("fault", ["condition", "model"])
def test_changes_during_execution_do_not_receive_completed_receipt(monkeypatch, phase, fault):
    low, high, source, positive = inputs(monkeypatch)
    boundary = None if phase == "head" else head(low, source, positive)
    selected = low if phase == "head" else high
    def mutate(index, *_):
        if index % 4 != 1:
            return
        if fault == "condition":
            positive[0][0].add_(.1)
        else:
            key, value = next((key, value) for key, value in selected.model.named_parameters() if value.ndim == 2)
            selected.add_patches({key: ("diff", (torch.full_like(value, .01),))}, .5)
    with pytest.raises(ValueError, match="changed during"):
        if phase == "head":
            head(low, source, positive, callback=mutate)
        else:
            stages.sample_tail(boundary, high, positive, [], seed=7, callback=mutate)
    assert runtime.active_step() is None and runtime._ACTIVE_FORWARD.get() is None


def test_resource_failure_occurs_before_forward_and_can_retry(monkeypatch):
    low, high, source, positive = inputs(monkeypatch)
    original = stages.cleanup._resource_snapshot
    def fail(*_args, **_kwargs):
        raise RuntimeError("insufficient reserved workspace")
    monkeypatch.setattr(stages.cleanup, "_resource_snapshot", fail)
    with pytest.raises(RuntimeError, match="reserved workspace"):
        head(low, source, positive)
    assert runtime.active_step() is None and not low.object_patches_backup
    monkeypatch.setattr(stages.cleanup, "_resource_snapshot", original)
    boundary = head(low, source, positive)
    assert boundary.verify()["execution"]["actual_apply_intervals"] == [0, 1, 2, 3]
