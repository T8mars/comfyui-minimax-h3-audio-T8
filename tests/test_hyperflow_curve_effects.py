"""Native tiny51 projection execution with real external EAV and paired Relay."""
from dataclasses import replace
import json

import comfy.nested_tensor
import comfy.sample
import pytest
import torch

from h3_audio_t8_pkg.modular_sampling import hyperflow_curve as stages, hyperflow_curve_effects as effects, eav
from h3_audio_t8_pkg.modular_sampling import hyperflow_curve_storage as storage
from test_hyperflow_curve_stages import inputs, head, equal
from test_progressive_relay import paired


def case(tmp_path, monkeypatch, mode="report_only", relay=False):
    low, high, _, positive, _ = inputs(tmp_path, monkeypatch, distinct=True)
    source = {"samples": comfy.nested_tensor.NestedTensor((torch.zeros(1, 24, 2, 4, 8), torch.zeros(1, 32, 2, 8)))}
    hp = positive
    if relay:
        low, positive, _ = paired(low)
        high, hp, _ = paired(high)
    bound = effects.bind_head(low, source, positive, [])
    config = eav.EAVConfig(mode, .2, 0., 1., 32, 3.)
    low, lr, _ = eav.apply_stage_eav(bound[0], bound[4], source, bound[5], config)
    boundary = head(low, source, positive)
    bound_high = effects.bind_tail(boundary, high, hp, [])
    high, hr, _ = eav.apply_stage_eav(bound_high[0], bound_high[4], boundary.scaffold, bound_high[5], config)
    return low, high, source, positive, hp, boundary, lr, hr


@pytest.mark.parametrize("with_relay", [False, True])
def test_actual_phase_eav_relay_coverage_and_portable_store(tmp_path, monkeypatch, with_relay):
    low, high, _, _, hp, boundary, lr, hr = case(tmp_path, monkeypatch, "apply_exp", with_relay)
    result, _ = stages.sample_tail(boundary, high, hp, [], seed=7)
    for value, phase, begin, end, telemetry in ((boundary, "head", 0, 4, lr), (result, "tail", 4, 8, hr)):
        report = effects.audit(value, phase)
        assert report["composition_verified"] is True
        assert report["eav"]["completed_forwards"] == 4
        assert report["eav"]["selector_calls"] == 200
        assert report["eav"]["status"] == "observed_apply_exp"
        assert report["eav"]["observed_absolute_sigmas"] == pytest.approx(
            effects.capture_owner(low if phase == "head" else high).context.trajectory_sigmas[begin:end], abs=2e-6)
        if with_relay:
            assert report["relay"]["routed_attention_calls"] == 200
            assert report["relay"]["completed_forwards"] == 4
        telemetry.completed_forwards = 999
        assert effects.audit(value, phase)["eav"]["completed_forwards"] == 4
    assert boundary.verify()["execution"]["portable_identity"] is True
    assert result.verify()["sampling"]["execution"]["portable_identity"] is True
    path, digest, _ = storage.save_result(result, str(tmp_path / "store"))
    frozen, _ = storage.load_result(str(tmp_path / "store"), path, digest)
    equal(frozen.output, result.output)
    assert not torch.cuda.is_initialized()


def test_report_only_and_disabled_math_exact_on_same_artifact(tmp_path, monkeypatch):
    low, high, source, positive, _ = inputs(tmp_path, monkeypatch)
    config = eav.EAVConfig("disabled", .2, 0., 1., 32, 3.)
    bound = effects.bind_head(low, source, positive, [])
    disabled, _, _ = eav.apply_stage_eav(bound[0], bound[4], source, bound[5], config)
    baseline = head(disabled, source, positive)
    expected, _ = stages.sample_tail(baseline, high, positive, [], seed=7)
    reported, _, _ = eav.apply_stage_eav(bound[0], bound[4], source, bound[5], replace(config, mode="report_only"))
    boundary = head(reported, source, positive)
    bound_high = effects.bind_tail(boundary, high, positive, [])
    selected, _, _ = eav.apply_stage_eav(bound_high[0], bound_high[4], boundary.scaffold,
        bound_high[5], replace(config, mode="report_only"))
    actual, _ = stages.sample_tail(boundary, selected, positive, [], seed=7)
    torch.testing.assert_close(boundary.x_sigma, baseline.x_sigma, rtol=0, atol=0)
    equal(actual.output, expected.output)


def test_cancel_then_tail_only_retry_has_own_frozen_effect_audit(tmp_path, monkeypatch):
    _, high, _, _, hp, boundary, _, hr = case(tmp_path, monkeypatch, "apply_exp", True)
    original = json.dumps(effects.audit(boundary, "head"), sort_keys=True)
    def cancel(*_args):
        raise RuntimeError("real curve effect cancellation")
    with pytest.raises(RuntimeError, match="real curve effect cancellation"):
        stages.sample_tail(boundary, high, hp, [], seed=7, callback=cancel)
    assert hr.snapshot()["status"] == "aborted"
    monkeypatch.setattr(stages, "sample_head", lambda *_a, **_kw: pytest.fail("Hidden HEAD rerun"))
    result, _ = stages.sample_tail(boundary, high, hp, [], seed=7)
    assert effects.audit(result, "tail")["composition_verified"] is True
    assert effects.audit(result, "tail")["relay"]["routed_attention_calls"] == 200
    assert json.dumps(effects.audit(boundary, "head"), sort_keys=True) == original


def test_changed_eav_config_has_no_completed_portable_receipt(tmp_path, monkeypatch):
    _, high, _, _, hp, boundary, _, hr = case(tmp_path, monkeypatch, "apply_exp", True)
    def mutate(*_args):
        hr.config = replace(hr.config, tau=.4)
    with pytest.raises(ValueError, match="configuration changed"):
        stages.sample_tail(boundary, high, hp, [], seed=7, callback=mutate)
    assert hr.snapshot()["status"] == "aborted"
