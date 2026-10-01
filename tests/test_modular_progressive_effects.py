"""Actual native tiny effects, single-stage execution and cancellation; CPU only."""
from copy import deepcopy
import json

import comfy.sample
import pytest
from legacy_schema_review import assert_reviewed_legacy_schemas
import torch

from h3_audio_t8_pkg.modular_sampling import progressive as stages
from h3_audio_t8_pkg.modular_sampling import progressive_effects as effects
from h3_audio_t8_pkg.modular_sampling import progressive_high_result as high_results
from h3_audio_t8_pkg.modular_sampling.eav import EAVConfig
from h3_audio_t8_pkg import progressive_sampling_runtime as legacy
from h3_audio_t8_pkg import prompt_relay_advanced as relay
from test_modular_progressive_stages import inputs, low_only, restart, initialized
from test_progressive_relay import paired
from test_progressive_sampling_runtime import conditioning
import test_progressive_sampling_runtime as fixtures

stub_lifter = fixtures.stub_lifter


def selected(case, phase, *, source=None, config=None, with_relay=False, model=None, mode="apply_exp"):
    model = case["model"] if model is None else model
    pos, neg = (case["lp"], case["ln"]) if phase == "low" else (case["hp"], case["hn"])
    if with_relay:
        bound, positive, _ = paired(model, case["task"])
        model, pos, neg = effects.apply_relay(bound, positive, conditioning(), case["plan"], phase, mode=mode)
    if config is not None:
        model = effects.apply_eav(model, config, case["plan"], phase,
                                 case["low_source"] if phase == "low" else source)
    return model, pos, neg


def sample(case, phase, model, pos, neg, *, state=None, **kwargs):
    if phase == "low":
        noise = comfy.sample.prepare_noise(case["low_source"]["samples"], 7)
        return stages.sample_low(model, case["sampler"], case["plan"], case["low_source"], pos, neg, noise, seed=7, **kwargs)
    return high_results.sample_high_result(state, model, case["sampler"], pos, neg, seed=7, **kwargs)[0]


@pytest.mark.parametrize("task", ["t2va", "i2va"])
@pytest.mark.parametrize("with_relay", [False, True])
def test_separate_external_effects_match_original_composer_exactly(stub_lifter, task, with_relay):
    case = inputs(task, steps=8, low=4)
    model, positive = case["model"], case["positive"]
    if with_relay:
        model, positive, _ = paired(model, task)
    expected, _ = legacy.sample_progressive_h3(model, positive, case["negative"], case["source"], case["sampler"],
        case["sigmas"], upscaler_model="test", seed=7, low_evaluations=4, task=task,
        eav_mode="apply_exp", eav_tau=.2, eav_start_video_progress=0., eav_end_video_progress=1.)
    config = EAVConfig("apply_exp", .2, 0., 1.)
    low = sample(case, "low", *selected(case, "low", config=config, with_relay=with_relay))
    state = restart(case, low)
    high = sample(case, "high", *selected(case, "high", source=state, config=config, with_relay=with_relay), state=state)
    for actual, wanted in zip(high.output["samples"].unbind(), expected["samples"].unbind()):
        torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
    for phase, result in (("low", low), ("high", high)):
        report = effects.audit(result, phase)
        assert report["status"] == "verified_stage_execution_quality_unverified"
        assert report["eav"]["model_forward_count"] == 4
        assert report["eav"]["attention_calls_per_active_forward"] == [1] * 4
        assert report["eav"]["g_min"] > 1
        if with_relay:
            assert report["relay"]["completed_calls"] == {"forward": 4, "routed_attention": 4}
        assert result.verify()["portable_identity"] is True


@pytest.mark.parametrize("phase", ["low", "high"])
@pytest.mark.parametrize("with_relay", [False, True])
def test_only_one_phase_enabled_and_report_only_exact_bypass(stub_lifter, phase, with_relay):
    case = inputs(steps=8, low=4)
    baseline_low = low_only(case)
    state = restart(case, baseline_low)
    args = selected(case, phase, source=state, with_relay=with_relay)
    baseline = sample(case, phase, *args, state=state)
    control = sample(case, phase, *selected(case, phase, source=state, with_relay=with_relay,
        config=EAVConfig("report_only", .2, 0., 1.)), state=state)
    applied = sample(case, phase, *selected(case, phase, source=state, with_relay=with_relay,
        config=EAVConfig("apply_exp", .2, 0., 1.)), state=state)
    def parts(result):
        return result.tensors.values() if phase == "low" else result.output["samples"].unbind()
    assert all(torch.equal(a, b) for a, b in zip(parts(baseline), parts(control)))
    assert any(not torch.equal(a, b) for a, b in zip(parts(control), parts(applied)))
    assert baseline_low.verify()["execution"]["actual_apply_calls"] == 4
    report = effects.audit(applied, phase)
    assert report["assigned_nfe"] == 4
    if phase == "high":
        assert report["eav"]["forwards"][0]["progress_video"] > 0
        assert not applied.verify()["sampling"]["low_executed"]


@pytest.mark.parametrize("kind", ["binary", "fractional", "avatar", "zero"])
def test_initialized_masks_and_partial_clock_equal_original(stub_lifter, kind):
    from h3_audio_t8_pkg.sampling import native_flow_sigmas
    case = inputs(source=initialized(kind), mode="initialized_av_exp", sigmas=native_flow_sigmas(5, 12.)[1:])
    expected, _ = legacy.sample_progressive_h3(case["model"], case["positive"], case["negative"],
        case["source"], case["sampler"], case["sigmas"], upscaler_model="test", seed=7,
        low_evaluations=2, input_mode=case["mode"], eav_mode="apply_exp", eav_tau=.2)
    config = EAVConfig("apply_exp", .2, 0., 1.)
    low = sample(case, "low", *selected(case, "low", config=config))
    state = restart(case, low)
    high = sample(case, "high", *selected(case, "high", source=state, config=config), state=state)
    for a, b in zip(high.output["samples"].unbind(), expected["samples"].unbind()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert effects.audit(high, "high")["eav"]["verified_native_mask_forwards"] == 2


@pytest.mark.parametrize("phase", ["low", "high"])
@pytest.mark.parametrize("kind", ["relay", "eav", "combined"])
def test_same_apply_owner_reuse_cancellation_and_audit_snapshot_is_immutable(stub_lifter, phase, kind):
    case = inputs()
    state = restart(case, low_only(case))
    args = selected(case, phase, source=state, with_relay=kind != "eav",
                    config=EAVConfig("apply_exp", .2, 0., 1.) if kind != "relay" else None)
    one = sample(case, phase, *args, state=state)
    first = deepcopy(effects.audit(one, phase))
    two = sample(case, phase, *args, state=state)
    assert effects.audit(two, phase)["assigned_nfe"] == 2
    def cancel(*_):
        raise RuntimeError("cancel own stage")
    with pytest.raises(RuntimeError, match="cancel own stage"):
        sample(case, phase, *args, state=state, callback=cancel)
    assert effects.owner(args[0]).last_report["status"] == "aborted"
    assert not effects.owner(args[0]).lock.locked()
    retried = sample(case, phase, *args, state=state)
    assert effects.audit(retried, phase)["status"] == "verified_stage_execution_quality_unverified"
    assert effects.audit(one, phase) == first
    assert not case["model"].wrappers and effects.owner(case["model"]) is None


def test_disabled_eav_preserves_object_and_no_effects_is_not_verified():
    case = inputs()
    assert effects.apply_eav(case["model"], EAVConfig("disabled"), case["plan"], "low", case["low_source"]) is case["model"]
    assert effects.audit(low_only(case), "low")["status"] == "no_external_stage_effects"


@pytest.mark.parametrize("change", ["phase", "mask", "config", "busy", "cfg"])
def test_bound_stage_errors_before_any_model_call(stub_lifter, monkeypatch, change):
    case = inputs()
    state = restart(case, low_only(case))
    args = selected(case, "low", config=EAVConfig("apply_exp"))
    runtime = effects.owner(args[0])
    kwargs = {}
    if change == "config":
        runtime.eav_runtime.config["tau"] = 3.
    elif change == "mask":
        import comfy.nested_tensor
        case["low_source"]["noise_mask"] = comfy.nested_tensor.NestedTensor(
            [torch.zeros_like(x) for x in case["low_source"]["samples"].unbind()])
    elif change == "busy":
        runtime.lock.acquire()
    elif change == "cfg":
        kwargs["cfg"] = 2.
    monkeypatch.setattr(stages.legacy, "_native_stage", lambda *a, **k: pytest.fail("sampling must not start"))
    with pytest.raises((ValueError, RuntimeError)):
        sample(case, "high" if change == "phase" else "low", *args, state=state, **kwargs)
    if change == "busy":
        runtime.lock.release()


def test_relay_disabled_strips_only_authenticated_owner():
    case = inputs()
    model, positive, _ = paired(case["model"])
    original = deepcopy(relay.prompt_relay_model_contract(model)["binding"])
    selected_model, pos, neg = effects.apply_relay(model, positive, conditioning(), case["plan"], "low", mode="disabled")
    assert relay.PROMPT_RELAY_BINDING_KEY not in pos[0][1]
    assert effects.owner(selected_model) is None
    assert relay.prompt_relay_model_contract(model)["binding"] == original
    assert sample(case, "low", selected_model, pos, neg).verify()["execution"]["actual_apply_calls"] == 2


def test_wrong_relay_pair_and_double_apply_refused():
    case = inputs()
    model, positive, _ = paired(case["model"])
    wrong = deepcopy(positive)
    wrong[0][1][relay.PROMPT_RELAY_BINDING_KEY]["plan_hash"] = "wrong"
    with pytest.raises(ValueError, match="not paired"):
        effects.apply_relay(model, wrong, conditioning(), case["plan"], "low")
    selected_model, pos, neg = effects.apply_relay(model, positive, conditioning(), case["plan"], "low")
    with pytest.raises(ValueError, match="do not stack"):
        effects.apply_relay(selected_model, pos, neg, case["plan"], "low")


def test_external_high_only_never_calls_low_or_whole_executor(stub_lifter, monkeypatch):
    case = inputs()
    state = restart(case, low_only(case))
    args = selected(case, "high", source=state, with_relay=True, config=EAVConfig("apply_exp", .2, 0., 1.))
    def forbidden(*a, **k):
        pytest.fail("HIGH effects may not rerun LOW or a hidden whole runner")
    monkeypatch.setattr(stages, "sample_low", forbidden)
    monkeypatch.setattr(legacy, "sample_progressive_h3", forbidden)
    result = sample(case, "high", *args, state=state)
    assert effects.audit(result, "high")["relay"]["completed_calls"]["forward"] == 2


def test_public_nodes_execute_and_old_schema_prefix_is_preserved(stub_lifter):
    from h3_audio_t8_pkg.modular_sampling import progressive_effect_nodes as nodes
    from tools import build_modular_progressive_workflow as builder
    info = builder.load_live_info()
    before = json.loads((builder.ROOT / "artifacts/development/modular-sampling-m3-progressive-effects-20260923/before-registration.json").read_text(encoding="utf-8"))
    assert_reviewed_legacy_schemas(before["nodes"], info)
    assert len(nodes.NODES) == 5
    assert all(node.__name__ in info for node in nodes.NODES)
    case = inputs()
    model = nodes.MiniMaxH3ProgressiveLowEAVApplyEXPT8.execute(
        case["model"], EAVConfig("apply_exp"), case["plan"], case["low_source"]).result[0]
    result = sample(case, "low", model, case["lp"], case["ln"])
    returned, text = nodes.MiniMaxH3ProgressiveLowEffectsAuditEXPT8.execute(result).result
    assert returned is result and json.loads(text)["eav"]["model_forward_count"] == 2


@pytest.mark.parametrize("phase", ["low", "high"])
def test_unknown_effect_stack_archives_but_does_not_fake_portable_load(stub_lifter, tmp_path, phase):
    from h3_audio_t8_pkg.modular_sampling import progressive_storage
    case = inputs()
    case["model"].add_wrapper_with_key("apply_model", "user-unverified",
        lambda executor, *args, **kwargs: executor(*args, **kwargs))
    state = restart(case, low_only(case))
    result = sample(case, phase, *selected(case, phase, source=state, with_relay=True,
        config=EAVConfig("apply_exp", .2, 0., 1.)), state=state)
    save = progressive_storage.save_boundary if phase == "low" else high_results.save_result
    load = progressive_storage.load_boundary if phase == "low" else high_results.load_result
    path, digest, text = save(result, tmp_path)
    assert json.loads(text)["status"] == "archived_unverified_identity"
    with pytest.raises(ValueError, match="[Uu]nverified"):
        load(tmp_path, path, digest)
