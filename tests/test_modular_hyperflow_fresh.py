"""Real original Core full8/partial4/fresh4, actual50+2 and synthetic adapter."""
from dataclasses import replace
import json

import comfy.model_management
import comfy.nested_tensor
import comfy.patcher_extension
import pytest
import torch
from comfy_extras.nodes_custom_sampler import BasicGuider, RandomNoise, SamplerCustomAdvanced

from h3_audio_t8_pkg import nodes_hyperflow_advanced as old
from h3_audio_t8_pkg.modular_sampling import hyperflow_fresh as fresh, eav, hyperflow
from h3_audio_t8_pkg.modular_sampling.results import sample_stage, StageResult, canonical, sha
from h3_audio_t8_pkg.modular_sampling.storage import save_stage, load_stage
from h3_audio_t8_pkg.progressive_sampling_runtime import _restore_stage_objects
from test_modular_hyperflow import inputs, equal
from test_progressive_relay import paired


ORIGINAL = (old.MiniMaxH3HyperFlowSamplerT8Advanced, old.MiniMaxH3HyperFlowCoarseSamplerT8Advanced,
            old.MiniMaxH3HyperFlowRefineSamplerT8Advanced, old.MiniMaxH3HyperFlowPartialRefineSamplerT8Advanced)


def setup(monkeypatch, stage, *, mode=None, with_relay=False, masked=False):
    low, high, _, positive = inputs(monkeypatch, distinct=True)
    model = low if fresh.SPECS[stage][1] == 0 else high
    source = {"samples": comfy.nested_tensor.NestedTensor((torch.full((1, 24, 2, 4, 8), .1),
                                                          torch.full((1, 32, 2, 8), -.2))), "source_note": "preserve"}
    if masked:
        vm, am = (torch.ones_like(part) for part in source["samples"].unbind())
        vm[:, :, :1] = 0
        am[..., :4] = 0
        source["noise_mask"] = comfy.nested_tensor.NestedTensor((vm, am))
    if with_relay:
        model, positive, _ = paired(model)
    prepared, sampler, sigmas, context, _ = fresh.build_stage(model, source, stage)
    telemetry = None
    if mode is not None:
        prepared, telemetry, _ = eav.apply_stage_eav(prepared, sigmas, source, context,
            eav.EAVConfig(mode, .2, 0., 1., 32, 3.))
    args = (RandomNoise.execute(31).result[0], BasicGuider.execute(prepared, positive).result[0],
            sampler, sigmas, source, context)
    return args, model, positive, telemetry


@pytest.mark.parametrize("stage", fresh.STAGES)
@pytest.mark.parametrize("masked", [False, True])
def test_both_core_outputs_bit_exact_to_original_public_sampler(monkeypatch, stage, masked):
    args, model, positive, _ = setup(monkeypatch, stage, masked=masked)
    plan = fresh.native.build_hyperflow_plan(model, *fresh.SPECS[stage][1:])
    branch, sampler, sigmas, _ = ORIGINAL[fresh.STAGES.index(stage)].execute(model, args[4], plan).result
    reference = SamplerCustomAdvanced.execute(args[0], BasicGuider.execute(branch, positive).result[0], sampler, sigmas, args[4]).result
    _restore_stage_objects(branch)
    actual = sample_stage(*args)
    for value, expected in zip(actual[:2], reference):
        equal(value, expected)
    receipt = actual[2].verify()
    context = args[5]
    assert receipt["verified_recipe_completion"] and receipt["portable_identity"], receipt["request"]["model"]
    assert receipt["execution"]["callbacks"] == list(range(context.steps))
    assert receipt["execution"]["hyperflow_fresh"]["absolute_apply_intervals"] == list(range(context.start, context.end))
    assert receipt["execution"]["hyperflow_fresh"]["fresh_noise_restart"] == (context.start > 0)
    assert actual[0]["source_note"] == "preserve"
    if masked:
        assert actual[0]["noise_mask"] is args[4]["noise_mask"]
    if stage in fresh.STAGES[:2]:
        lifted, report = fresh.lift_input(actual[2])
        assert lifted is actual[0 if stage == fresh.STAGES[0] else 1]
        assert json.loads(report)["sampling_calls"] == 0
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("stage", fresh.STAGES)
@pytest.mark.parametrize("with_relay", [False, True])
@pytest.mark.parametrize("mode", ["disabled", "report_only", "apply_exp"])
def test_fresh_stage_actual_effects_and_cold_hot_portable_storage(monkeypatch, tmp_path, stage, with_relay, mode):
    args, _, _, telemetry = setup(monkeypatch, stage, mode=mode, with_relay=with_relay)
    first = sample_stage(*args)[2]
    again = sample_stage(*args)[2]
    equal(first.output, again.output)
    receipt = first.verify()
    assert receipt["verified_recipe_completion"] and receipt["portable_identity"], receipt["request"]["model"]
    observed = receipt["execution"]["hyperflow_fresh"]
    assert observed["composition_verified"]
    assert receipt["request_sha256"] == again.verify()["request_sha256"]
    if mode != "disabled":
        assert observed["eav"]["completed_forwards"] == args[5].steps
        assert observed["eav"]["selector_calls"] == args[5].steps * 50
        assert observed["eav"]["observed_absolute_sigmas"] == pytest.approx(
            args[5].trajectory_sigmas[args[5].start:args[5].end], abs=2e-6)
    else:
        assert "eav" not in observed
    if with_relay:
        assert observed["relay"]["routed_attention_calls"] == args[5].steps * 50
    path, digest, _ = save_stage(first, tmp_path)
    loaded = load_stage(tmp_path, path, digest, stage)
    assert loaded[3].receipt_json == first.receipt_json
    telemetry.completed_forwards = 999
    if mode != "disabled":
        assert first.verify()["execution"]["hyperflow_fresh"]["eav"]["completed_forwards"] == args[5].steps


@pytest.mark.parametrize("stage", fresh.STAGES)
def test_disabled_and_report_only_match_while_apply_changes_video(monkeypatch, stage):
    results = [sample_stage(*setup(monkeypatch, stage, mode=mode, with_relay=True)[0])
               for mode in ("disabled", "report_only", "apply_exp")]
    for slot in (0, 1):
        equal(results[0][slot], results[1][slot])
    assert not torch.equal(results[0][0]["samples"].unbind()[0], results[2][0]["samples"].unbind()[0])


def test_unknown_user_wrapper_preserved_without_portable_claim(monkeypatch):
    args, _, _, _ = setup(monkeypatch, fresh.STAGES[2], mode="apply_exp", with_relay=True)
    calls = []
    def user(executor, *a, **kw):
        calls.append(True)
        return executor(*a, **kw)
    model = args[1].model_patcher
    model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.APPLY_MODEL, "user-fresh", user)
    result = sample_stage(*args)[2]
    assert calls and result.verify()["portable_identity"] is False
    assert result.verify()["execution"]["hyperflow_fresh"]["composition_verified"]
    assert model.get_wrappers(comfy.patcher_extension.WrappersMP.APPLY_MODEL, "user-fresh") == [user]


@pytest.mark.parametrize("fault", ["sigmas", "clock", "pair", "sampler", "receipt", "busy"])
def test_invalid_own_state_cannot_acquire_completed_result(monkeypatch, fault):
    args, _, _, _ = setup(monkeypatch, fresh.STAGES[3], mode="apply_exp", with_relay=True)
    args = list(args)
    model = args[1].model_patcher
    if fault == "sigmas":
        args[3] = args[3] + .1
        with pytest.raises(ValueError, match="SIGMAS"):
            sample_stage(*args)
    elif fault == "clock":
        model.model_options["transformer_options"]["minimax_h3_sigma_shift_audio"] = 7.
        with pytest.raises(ValueError, match="clocks"):
            sample_stage(*args)
    elif fault == "pair":
        args[1].original_conds["positive"][0][fresh.relay.PROMPT_RELAY_BINDING_KEY] = {"bad": "pair"}
        with pytest.raises(ValueError, match="paired"):
            sample_stage(*args)
    elif fault == "sampler":
        original = args[2].sampler_function
        args[2].sampler_function = lambda *a, **kw: original(*a, **kw)
        result = sample_stage(*args)[2]
        assert not result.verify()["verified_recipe_completion"] and not result.verify()["portable_identity"]
    elif fault == "receipt":
        result = sample_stage(*args)[2]
        receipt = result.verify()
        receipt["execution"]["hyperflow_fresh"]["absolute_apply_intervals"] = []
        receipt.pop("receipt_sha256")
        receipt["receipt_sha256"] = sha(receipt)
        with pytest.raises(ValueError, match="completion"):
            replace(result, receipt_json=canonical(receipt)).verify()
    else:
        lock = fresh.capture_owner(model).lock
        lock.acquire()
        try:
            with pytest.raises(RuntimeError, match="busy"):
                sample_stage(*args)
            assert lock.locked()
        finally:
            lock.release()
        assert sample_stage(*args)[2].verify()["portable_identity"]


def test_only_low_result_can_supply_learned_lift_input(monkeypatch):
    high = sample_stage(*setup(monkeypatch, fresh.STAGES[2])[0])[2]
    with pytest.raises(ValueError, match="LOW"):
        fresh.lift_input(high)
    with pytest.raises(ValueError, match="Stage Result"):
        fresh.lift_input(high.output)
    assert type(high) is StageResult and type(high) is not hyperflow.ContinuousBoundary
