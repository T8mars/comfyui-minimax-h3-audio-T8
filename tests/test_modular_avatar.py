"""Actual tiny native Avatar stages; source/PCM binding, not trained quality/VAE provenance."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import comfy.nested_tensor
import comfy.sample
import pytest
import torch

from h3_audio_t8_pkg.avatar_progressive_entry import sample_avatar_progressive
from h3_audio_t8_pkg.modular_sampling import avatar
from h3_audio_t8_pkg.modular_sampling import avatar_nodes as nodes
from h3_audio_t8_pkg.modular_sampling import progressive as stages
from h3_audio_t8_pkg.modular_sampling import progressive_high_result as results
from h3_audio_t8_pkg.modular_sampling import progressive_storage as storage
from h3_audio_t8_pkg.modular_sampling.eav import EAVConfig
from test_modular_progressive_stages import inputs, initialized
from test_modular_progressive_effects import selected
from test_progressive_sampling_runtime import tiny_model
from test_progressive_relay import paired
import test_progressive_sampling_runtime as fixtures

stub_lifter = fixtures.stub_lifter


def recording():
    return {"waveform": torch.linspace(-.3, .3, 13334).reshape(1, 2, 6667), "sample_rate": 32000}


def case_inputs(task="t2va", steps=4, low=2):
    case = inputs(task, steps, low, source=initialized("avatar"), mode="initialized_av_exp")
    source = avatar.bind_source(case["source"], recording())
    case["source"] = source.source
    case["low_source"] = stages.prepare_low_source(source.source, case["plan"], "initialized_av_exp")
    case["avatar"] = source
    return case


def low_only(case, args=None, **options):
    model, pos, neg = args or (case["model"], case["lp"], case["ln"])
    return avatar.sample_low(case["avatar"], model, case["sampler"], case["plan"], case["low_source"],
        pos, neg, comfy.sample.prepare_noise(case["low_source"]["samples"], 7), seed=7, **options)


def handoff(case, boundary, model=None, **options):
    model = case["model"] if model is None else model
    value = stages.lift_input(boundary, model, case["sampler"])
    video, audio = value["samples"].unbind()
    lifted, _ = stages.legacy._lift_video(video, audio, case["plan"], {"labelled_interpolation_double": True})
    return avatar.prepare_high(case["avatar"], boundary, model, case["sampler"],
        {"samples": comfy.nested_tensor.NestedTensor((lifted, audio))},
        comfy.sample.prepare_noise(lifted, 8), **options)


def high_only(case, state, args=None, **options):
    model, pos, neg = args or (case["model"], case["hp"], case["hn"])
    return results.sample_high_result(state, model, case["sampler"], pos, neg, seed=7, **options)[0]


@pytest.mark.parametrize("task", ["t2va", "i2va"])
@pytest.mark.parametrize("separate", [False, True])
@pytest.mark.parametrize("kind", ["plain", "eav", "combined"])
def test_exact_original_avatar_math_and_each_phase_effects(stub_lifter, task, separate, kind):
    case = case_inputs(task, 8, 4)
    high = tiny_model() if separate else case["model"]
    if separate:
        key, weight = next(iter(high.model.named_parameters()))
        assert high.add_patches({key: ("diff", (torch.ones_like(weight) * .01,))})
    model, positive = case["model"], case["positive"]
    if kind == "combined":
        model, positive, _ = paired(model, task)
    before = stages.snapshot(case["avatar"].recording), stages.snapshot(case["source"])
    expected, old_report = sample_avatar_progressive(model=model, model_hires=high if separate else None,
        positive=positive, negative=case["negative"], av_latent=case["source"], sampler=case["sampler"],
        sigmas=case["sigmas"], upscaler_model="labelled-interpolation", seed=7, low_evaluations=4, task=task,
        eav_mode="disabled" if kind == "plain" else "apply_exp", eav_tau=.2,
        eav_start_video_progress=0., eav_end_video_progress=1.)
    config = EAVConfig("apply_exp", .2, 0., 1.) if kind != "plain" else None
    boundary = low_only(case, selected(case, "low", config=config, with_relay=kind == "combined"))
    state = handoff(case, boundary, high)
    result = high_only(case, state, selected(case, "high", source=state, model=high,
        config=config, with_relay=kind == "combined"))
    output, audio, text = avatar.deliver(result, case["avatar"].recording)
    for actual, wanted in zip(output["samples"].unbind(), expected["samples"].unbind()):
        torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
    assert audio is case["avatar"].recording
    assert before == (stages.snapshot(audio), stages.snapshot(case["source"]))
    assert boundary.verify()["execution"]["actual_apply_calls"] == 4
    assert result.verify()["sampling"]["execution"]["actual_apply_calls"] == 4
    assert boundary.verify()["portable_identity"] and result.verify()["portable_identity"]
    assert json.loads(text)["source_audio_latent_max_abs_difference"] < 1e-6
    assert json.loads(old_report)["counts"]["actual_forwards"] == {"low": 4, "high": 4}
    if config:
        from h3_audio_t8_pkg.modular_sampling.progressive_effects import audit
        for phase, actual in (("low", boundary), ("high", result)):
            report = audit(actual, phase)
            assert report["eav"]["model_forward_count"] == 4
            assert report["eav"]["verified_native_mask_forwards"] == 4
            if kind == "combined":
                assert report["relay"]["completed_calls"] == {"forward": 4, "routed_attention": 4}


@pytest.mark.parametrize("fault", ["missing_mask", "video_mask", "generated", "fractional", "nan_av",
    "nan_pcm", "empty_pcm", "rate_zero", "rate_float", "already_bound"])
def test_source_validation_before_sampling(fault):
    source, audio = initialized("avatar"), recording()
    v, a = source["samples"].unbind()
    if fault == "missing_mask":
        source.pop("noise_mask")
    elif fault == "video_mask":
        source["noise_mask"] = torch.ones_like(v)
    elif fault in ("generated", "fractional"):
        source["noise_mask"] = comfy.nested_tensor.NestedTensor((torch.ones_like(v), torch.full_like(a, 1. if fault == "generated" else .01)))
    elif fault == "nan_av":
        a.fill_(float("nan"))
    elif fault == "nan_pcm":
        audio["waveform"].fill_(float("nan"))
    elif fault == "empty_pcm":
        audio["waveform"] = torch.zeros(1, 2, 0)
    elif fault.startswith("rate"):
        audio["sample_rate"] = 0 if fault == "rate_zero" else 32000.1
    else:
        source = avatar.bind_source(source, audio).source
    with pytest.raises(ValueError):
        avatar.bind_source(source, audio)


@pytest.mark.parametrize("fault", ["pcm", "source_audio", "low_audio", "low_video", "low_tag"])
def test_low_wrong_binding_never_executes_model(monkeypatch, fault):
    case = case_inputs()
    if fault == "pcm":
        case["avatar"].recording["waveform"].add_(.1)
    elif fault == "source_audio":
        case["source"]["samples"].unbind()[1].add_(.1)
    elif fault == "low_tag":
        case["low_source"].pop(avatar.KEY)
    else:
        case["low_source"]["samples"].unbind()[0 if fault == "low_video" else 1].add_(.1)
    monkeypatch.setattr(stages, "sample_low", lambda *a, **k: pytest.fail("Invalid Avatar reached LOW"))
    with pytest.raises(ValueError):
        low_only(case)


@pytest.mark.parametrize("fault", ["ordinary_low", "different_recording", "high_audio", "high_mask", "other_tag"])
def test_high_wrong_source_rejected_before_handoff(stub_lifter, monkeypatch, fault):
    case = case_inputs()
    boundary = low_only(case)
    options = {}
    if fault == "ordinary_low":
        receipt = boundary.verify()
        receipt["request"].pop(avatar.KEY)
        receipt["request_sha256"] = stages.sha(receipt["request"])
        receipt.pop("receipt_sha256")
        receipt["receipt_sha256"] = stages.sha(receipt)
        boundary = replace(boundary, receipt_json=stages.canonical(receipt))
    elif fault == "different_recording":
        new_audio = recording()
        new_audio["waveform"].mul_(.5)
        case["avatar"] = avatar.bind_source({k: v for k, v in case["source"].items() if k != avatar.KEY}, new_audio)
    else:
        source = deepcopy(case["source"])
        if fault == "high_audio":
            source["samples"].unbind()[1].add_(.02)
        elif fault == "high_mask":
            source["noise_mask"].unbind()[1].fill_(.01)
        else:
            source[avatar.KEY] = "f" * 64
        options["high_source"] = source
    monkeypatch.setattr(stages, "prepare_high", lambda *a, **k: pytest.fail("Invalid Avatar reached handoff"))
    with pytest.raises(ValueError):
        handoff(case, boundary, **options)


def test_save_load_low_only_high_and_completed_high_no_source_encoder(stub_lifter, tmp_path, monkeypatch):
    case = case_inputs()
    boundary = low_only(case)
    path, digest, _ = storage.save_boundary(boundary, tmp_path)
    loaded, _ = storage.load_boundary(tmp_path, path, digest)
    def forbidden(*args, **kwargs):
        pytest.fail("LOW/whole executor ran during HIGH-only recovery")
    monkeypatch.setattr(stages, "sample_low", forbidden)
    monkeypatch.setattr(stages.legacy, "sample_progressive_h3", forbidden)
    high = tiny_model()
    key, weight = next(iter(high.model.named_parameters()))
    assert high.add_patches({key: ("diff", (torch.ones_like(weight) * .02,))})
    result = high_only(case, handoff(case, loaded, high), (high, case["hp"], case["hn"]))
    path, digest, _ = results.save_result(result, tmp_path)
    monkeypatch.setattr(stages, "sample_high", forbidden)
    monkeypatch.setattr(avatar, "bind_source", forbidden)
    loaded_high, _ = results.load_result(tmp_path, path, digest)
    pcm = recording()
    output, audio, text = avatar.deliver(loaded_high, pcm)
    assert audio is pcm and torch.equal(audio["waveform"], case["avatar"].recording["waveform"])
    assert all(torch.equal(a, b) for a, b in zip(output["samples"].unbind(), result.output["samples"].unbind()))
    assert json.loads(text)["encoding_calls"] == 0
    pcm["waveform"].mul_(.9)
    with pytest.raises(ValueError, match="recording"):
        avatar.deliver(loaded_high, pcm)


@pytest.mark.parametrize("phase", ["low", "high"])
def test_cancel_release_and_retry_recording_unchanged(stub_lifter, phase):
    case = case_inputs()
    before = stages.snapshot(case["avatar"].recording)
    state = handoff(case, low_only(case)) if phase == "high" else None
    def cancel(*args):
        raise InterruptedError("owned Avatar cancel")
    with pytest.raises(InterruptedError, match="owned Avatar cancel"):
        high_only(case, state, callback=cancel) if state else low_only(case, callback=cancel)
    result = high_only(case, state or handoff(case, low_only(case)))
    avatar.deliver(result, case["avatar"].recording)
    assert before == stages.snapshot(case["avatar"].recording)
    assert not case["model"].wrappers


def test_public_nodes_same_stage_contracts(stub_lifter):
    from comfy_extras.nodes_custom_sampler import Noise_RandomNoise
    source = initialized("avatar")
    bound = nodes.MiniMaxH3AvatarSourceBindEXPT8.execute(source, recording()).result
    case = inputs(source=bound[0], mode="initialized_av_exp")
    low = nodes.MiniMaxH3AvatarLowStageEXPT8.execute(case["model"], case["sampler"], Noise_RandomNoise(7),
        case["plan"], case["low_source"], case["lp"], case["ln"], bound[1]).result[0]
    case["avatar"] = bound[1]
    lift_input = stages.lift_input(low, case["model"], case["sampler"])
    v, a = lift_input["samples"].unbind()
    lifted, _ = stages.legacy._lift_video(v, a, case["plan"], {"labelled_interpolation_double": True})
    state = nodes.MiniMaxH3AvatarHighHandoffEXPT8.execute(bound[1], low, case["model"], case["sampler"],
        {"samples": comfy.nested_tensor.NestedTensor((lifted, a))}, Noise_RandomNoise(8)).result[0]
    high = high_only(case, state)
    delivered = nodes.MiniMaxH3AvatarDeliveryAuditEXPT8.execute(high, bound[1].recording).result
    assert delivered[1] is bound[1].recording
    ids = [node.define_schema().node_id for node in nodes.NODES]
    assert len(ids) == len(set(ids)) == 4


@pytest.mark.parametrize("phase", ["low", "high"])
@pytest.mark.parametrize("kind", ["eav", "relay", "combined"])
def test_single_phase_external_effects_and_disabled_audio_anchor(stub_lifter, phase, kind):
    from h3_audio_t8_pkg.modular_sampling.progressive_effects import audit
    case = case_inputs()
    config = EAVConfig("apply_exp", .2, 0., 1.) if kind != "relay" else None
    args = selected(case, "low", config=config, with_relay=kind != "eav") if phase == "low" else None
    boundary = low_only(case, args)
    state = handoff(case, boundary)
    args = selected(case, "high", source=state, config=config, with_relay=kind != "eav") if phase == "high" else None
    result = high_only(case, state, args)
    avatar.deliver(result, case["avatar"].recording)
    active = audit(boundary if phase == "low" else result, phase)
    inactive = audit(result if phase == "low" else boundary, "high" if phase == "low" else "low")
    assert inactive["status"] == "no_external_stage_effects"
    assert active["status"] == "verified_stage_execution_quality_unverified"
    assert active["assigned_nfe"] == 2


def test_unknown_user_wrapper_kept_and_no_false_persistent_identity(stub_lifter, tmp_path):
    case = case_inputs()
    calls = []
    def wrapper(execute, args):
        calls.append(1)
        return execute(args["input"], args["timestep"], **args["c"])
    case["model"].set_model_unet_function_wrapper(wrapper)
    boundary = low_only(case)
    result = high_only(case, handoff(case, boundary))
    avatar.deliver(result, case["avatar"].recording)
    assert len(calls) == 4 and not boundary.verify()["portable_identity"]
    assert case["model"].model_options["model_function_wrapper"] is wrapper
    path, digest, _ = storage.save_boundary(boundary, tmp_path)
    with pytest.raises(ValueError, match="Unverified"):
        storage.load_boundary(tmp_path, path, digest)


@pytest.mark.parametrize("target", ["pcm", "source"])
def test_upstream_recording_changed_during_low_cannot_issue_receipt(target):
    case = case_inputs()
    def mutate(*args):
        if target == "pcm":
            case["avatar"].recording["waveform"].add_(.1)
        else:
            case["source"]["samples"].unbind()[0].add_(.1)
    with pytest.raises(ValueError, match="changed"):
        low_only(case, callback=mutate)
    assert not case["model"].wrappers


@pytest.mark.parametrize("kind", ["plain", "combined"])
@pytest.mark.parametrize("change_high", [False, True])
def test_new_process_recording_bound_low_only_high_and_completed_delivery(stub_lifter, tmp_path, kind, change_high):
    case = case_inputs()
    config = EAVConfig("apply_exp", .2, 0., 1.) if kind == "combined" else None
    boundary = low_only(case, selected(case, "low", config=config, with_relay=kind == "combined"))
    path, digest, _ = storage.save_boundary(boundary, tmp_path)
    if change_high:
        key, weight = next(iter(case["model"].model.named_parameters()))
        assert case["model"].add_patches({key: ("diff", (torch.ones_like(weight) * .02,))})
    state = handoff(case, boundary)
    expected = high_only(case, state, selected(case, "high", source=state, config=config, with_relay=kind == "combined"))
    code = """
import json, runpy, sys
runpy.run_path('tools/check_director_d1_cpu.py', run_name='config')
runpy.run_path('tests/conftest.py')
import pytest, torch
from test_modular_avatar import case_inputs, handoff, high_only, selected, EAVConfig, recording
from test_progressive_sampling_runtime import stub_lifter
from h3_audio_t8_pkg.modular_sampling import avatar, progressive as stages, progressive_storage as storage
from h3_audio_t8_pkg.modular_sampling import progressive_high_result as results
boundary, _ = storage.load_boundary(sys.argv[1], sys.argv[2], sys.argv[3])
case = case_inputs()
if sys.argv[5] == 'True':
    key, weight = next(iter(case['model'].model.named_parameters()))
    case['model'].add_patches({key: ('diff', (torch.ones_like(weight) * .02,))})
def forbidden(*a, **k):
    raise AssertionError('LOW/whole executor/source encoder unexpectedly executed')
with pytest.MonkeyPatch.context() as patch:
    stub_lifter.__wrapped__(patch)
    patch.setattr(stages, 'sample_low', forbidden)
    patch.setattr(stages.legacy, 'sample_progressive_h3', forbidden)
    state = handoff(case, boundary)
    config = EAVConfig('apply_exp', .2, 0., 1.) if sys.argv[4] == 'combined' else None
    result = high_only(case, state, selected(case, 'high', source=state, config=config, with_relay=sys.argv[4] == 'combined'))
assert result.verify()['sampling']['execution']['actual_apply_calls'] == 2
path, digest, _ = results.save_result(result, sys.argv[1])
stages.sample_low = stages.sample_high = avatar.bind_source = forbidden
loaded, _ = results.load_result(sys.argv[1], path, digest)
pcm = recording()
output, delivered, text = avatar.deliver(loaded, pcm)
assert delivered is pcm and json.loads(text)['encoding_calls'] == 0 and not torch.cuda.is_initialized()
print('RESULT=' + json.dumps(stages.snapshot(output)))
"""
    child = subprocess.run([sys.executable, "-c", code, str(tmp_path), path, digest, kind, str(change_high)],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=60)
    assert child.returncode == 0, child.stdout + child.stderr
    actual = json.loads(next(line[7:] for line in child.stdout.splitlines() if line.startswith("RESULT=")))
    assert actual == stages.snapshot(expected.output)
