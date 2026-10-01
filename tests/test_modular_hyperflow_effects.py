"""Actual50+2 tiny native HyperFlow effects; synthetic weights, no GPU claim."""
from dataclasses import replace
import json

import comfy.model_management
import comfy.nested_tensor
import comfy.patcher_extension
import pytest
import torch

from h3_audio_t8_pkg.modular_sampling import hyperflow as stages, hyperflow_effects as effects, eav
from test_modular_hyperflow import inputs, head, equal
from test_progressive_relay import paired


def case(monkeypatch, mode="report_only", relay=False, split=4, phase="both"):
    low, high, source, positive = inputs(monkeypatch, distinct=True)
    source = {"samples": comfy.nested_tensor.NestedTensor((torch.zeros(1, 24, 2, 4, 8), torch.zeros(1, 32, 2, 8)))}
    hp = positive
    if relay:
        low, positive, _ = paired(low)
        high, hp, _ = paired(high)
    bound = effects.bind_head(low, source, positive, [], split)
    low = bound[0]
    config = eav.EAVConfig(mode, .2, 0., 1., 32, 3.)
    lr = None
    if phase in ("both", "head"):
        low, lr, _ = eav.apply_stage_eav(low, bound[4], source, bound[5], config)
    boundary = head(low, source, positive, split)
    bound_high = effects.bind_tail(boundary, high, hp, [])
    high = bound_high[0]
    hr = None
    if phase in ("both", "tail"):
        high, hr, _ = eav.apply_stage_eav(high, bound_high[4], boundary.scaffold, bound_high[5], config)
    return low, high, source, positive, hp, boundary, lr, hr


@pytest.mark.parametrize("split", [1, 4, 7])
@pytest.mark.parametrize("with_relay", [False, True])
@pytest.mark.parametrize("mode", ["report_only", "apply_exp"])
def test_actual_effects_per_phase_clocks_and_portable_results(monkeypatch, split, with_relay, mode):
    low, high, source, positive, hp, boundary, lr, hr = case(monkeypatch, mode, with_relay, split)
    result, _ = stages.sample_tail_result(boundary, high, hp, [], seed=7)
    for value, phase, start, stop, telemetry in ((boundary, "head", 0, split, lr), (result, "tail", split, 8, hr)):
        audit = effects.audit(value, phase)
        assert audit["composition_verified"] is True, audit
        assert audit["assigned_nfe"] == stop - start
        assert audit["eav"]["completed_forwards"] == stop - start
        assert audit["eav"]["selector_calls"] == (stop - start) * 50
        assert audit["eav"]["status"] == "observed_" + mode
        assert audit["eav"]["observed_absolute_sigmas"] == pytest.approx(
            effects.capture_owner(low if phase == "head" else high).context.trajectory_sigmas[start:stop], abs=2e-6)
        if with_relay:
            assert audit["relay"]["completed_forwards"] == stop - start
            assert audit["relay"]["routed_attention_calls"] == (stop - start) * 50
        receipt = value.verify()
        execution = receipt["execution"] if phase == "head" else receipt["sampling"]["execution"]
        assert execution["portable_identity"] is True
        assert telemetry.snapshot()["completed_forwards"] == stop - start
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("phase", ["head", "tail", "both"])
@pytest.mark.parametrize("with_relay", [False, True])
def test_report_and_disabled_are_exact_and_only_selected_phase_has_eav(monkeypatch, phase, with_relay):
    _, high, _, _, hp, boundary, _, _ = case(monkeypatch, "disabled", with_relay, phase=phase)
    expected, _ = stages.sample_tail_result(boundary, high, hp, [], seed=7)
    _, high, _, _, hp, reported, _, _ = case(monkeypatch, "report_only", with_relay, phase=phase)
    result, _ = stages.sample_tail_result(reported, high, hp, [], seed=7)
    assert torch.equal(boundary.x_sigma, reported.x_sigma)
    equal(expected.output, result.output)
    assert ("eav" in effects.audit(reported, "head")) == (phase in ("head", "both"))
    assert ("eav" in effects.audit(result, "tail")) == (phase in ("tail", "both"))


def test_repeated_run_reports_own_counts_not_previous_or_mutable_ui_state(monkeypatch):
    low, high, source, positive, hp, boundary, _, hr = case(monkeypatch, "apply_exp", True)
    original = json.dumps(effects.audit(boundary, "head"), sort_keys=True)
    again = head(low, source, positive)
    assert effects.audit(again, "head") == effects.audit(boundary, "head")
    first, _ = stages.sample_tail_result(boundary, high, hp, [], seed=7)
    second, _ = stages.sample_tail_result(boundary, high, hp, [], seed=7)
    assert effects.audit(first, "tail") == effects.audit(second, "tail")
    assert json.dumps(effects.audit(boundary, "head"), sort_keys=True) == original
    hr.completed_forwards = 999
    assert effects.audit(first, "tail")["eav"]["completed_forwards"] == 4


def test_cancel_and_retry_only_tail_with_fresh_execution_audit(monkeypatch):
    _, high, _, _, hp, boundary, _, hr = case(monkeypatch, "apply_exp", True)
    def cancel(*args):
        raise comfy.model_management.InterruptProcessingException()
    with pytest.raises(comfy.model_management.InterruptProcessingException):
        stages.sample_tail_result(boundary, high, hp, [], seed=7, callback=cancel)
    assert hr.snapshot()["status"] == "aborted"
    monkeypatch.setattr(stages, "sample_head", lambda *a, **kw: pytest.fail("HEAD rerun"))
    result, _ = stages.sample_tail_result(boundary, high, hp, [], seed=7)
    assert effects.audit(result, "tail")["composition_verified"] is True
    assert effects.audit(result, "tail")["relay"]["completed_forwards"] == 4


@pytest.mark.parametrize("change", ["role", "source", "split", "pair", "cfg", "config"])
def test_bad_phase_or_paired_conditions_never_get_completed_receipt(monkeypatch, change):
    low, high, source, positive, hp, boundary, lr, hr = case(monkeypatch, "apply_exp", True)
    if change == "role":
        with pytest.raises(ValueError, match="phase|interval"):
            stages.sample_tail_result(boundary, low, positive, [], seed=7)
    elif change == "source":
        source["samples"] = comfy.nested_tensor.NestedTensor(tuple(part[..., :1] for part in source["samples"].unbind()))
        with pytest.raises(ValueError, match="geometry|interval"):
            head(low, source, positive)
    elif change == "split":
        with pytest.raises(ValueError, match="phase|interval"):
            head(low, source, positive, 3)
    elif change == "pair":
        from h3_audio_t8_pkg import prompt_relay_advanced as relay
        hp[0][1][relay.PROMPT_RELAY_BINDING_KEY] = {"foreign": "plan"}
        with pytest.raises(ValueError, match="not paired"):
            stages.sample_tail_result(boundary, high, hp, [], seed=7)
    elif change == "cfg":
        with pytest.raises(ValueError, match="CFG1"):
            stages.sample_tail_result(boundary, high, hp, [], seed=7, cfg=2.)
    else:
        def mutate(*args):
            hr.config = replace(hr.config, tau=.3)
        with pytest.raises(ValueError, match="configuration|MODEL"):
            stages.sample_tail_result(boundary, high, hp, [], seed=7, callback=mutate)


def test_unknown_wrapper_still_delegates_without_portable_claim(monkeypatch):
    low, high, source, positive, hp, boundary, _, _ = case(monkeypatch, "apply_exp", True)
    calls = []
    def user(executor, *a, **kw):
        calls.append(True)
        return executor(*a, **kw)
    high.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.APPLY_MODEL, "user", user)
    result, _ = stages.sample_tail_result(boundary, high, hp, [], seed=7)
    assert calls and result.verify()["sampling"]["execution"]["portable_identity"] is False
    assert effects.audit(result, "tail")["composition_verified"] is True


def test_bypassed_effects_preserve_user_selector_and_cannot_claim_portable_success(tmp_path, monkeypatch):
    from h3_audio_t8_pkg.modular_sampling import hyperflow_storage as storage
    _, high, _, _, hp, boundary, _, hr = case(monkeypatch, "apply_exp", True)
    calls = []
    def custom(func, *args, **kwargs):
        calls.append(True)
        return func(*args, **kwargs)
    high.model_options["transformer_options"]["optimized_attention_override"] = custom
    result, _ = stages.sample_tail_result(boundary, high, hp, [], seed=7)
    report = effects.audit(result, "tail")
    assert calls and high.model_options["transformer_options"]["optimized_attention_override"] is custom
    assert hr.completed_forwards == 4 and hr.selector_calls == 0
    assert report["composition_verified"] is False and report["relay"]["routed_attention_calls"] == 0
    assert result.verify()["sampling"]["execution"]["portable_identity"] is False
    path, digest, _ = storage.save_result(result, tmp_path)
    with pytest.raises(ValueError, match="Unverified executable"):
        storage.load_result(tmp_path, path, digest)


@pytest.mark.parametrize("lease", ["stage", "relay"])
def test_busy_owners_do_not_reset_other_execution_counters(monkeypatch, lease):
    _, high, _, _, hp, boundary, _, hr = case(monkeypatch, "apply_exp", True)
    owner = effects.capture_owner(high)
    lock = owner.lock if lease == "stage" else effects._relay_lock(owner.relay_wrapper)
    before = dict(owner.relay_counts)
    hr.completed_forwards = 123
    lock.acquire()
    try:
        with pytest.raises(RuntimeError, match="busy"):
            stages.sample_tail_result(boundary, high, hp, [], seed=7)
        assert owner.relay_counts == before and hr.completed_forwards == 123
        assert lock.locked()
    finally:
        lock.release()
    result, _ = stages.sample_tail_result(boundary, high, hp, [], seed=7)
    assert effects.audit(result, "tail")["eav"]["completed_forwards"] == 4


def test_actual_gain_changes_video_and_absolute_window_does_not_restart(monkeypatch):
    _, high, _, _, hp, boundary, _, _ = case(monkeypatch, "disabled", True)
    expected, _ = stages.sample_tail_result(boundary, high, hp, [], seed=7)
    _, high, _, _, hp, boundary, _, _ = case(monkeypatch, "apply_exp", True)
    applied, _ = stages.sample_tail_result(boundary, high, hp, [], seed=7)
    assert not torch.equal(expected.output["samples"].unbind()[0], applied.output["samples"].unbind()[0])
    low, _, source, positive = inputs(monkeypatch)
    bound = effects.bind_head(low, source, positive, [], 1)
    selected, _, _ = eav.apply_stage_eav(bound[0], bound[4], bound[3], bound[5], eav.EAVConfig("apply_exp", .2, .15, .9, 32, 3.))
    result = head(selected, source, positive, 1)
    assert effects.audit(result, "head")["eav"]["status"] == "observed_no_steps_in_effect_window"
