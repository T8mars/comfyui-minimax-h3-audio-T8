"""v5 external EAV must bind the actual joint AV window and its overlap mask."""
import json

import pytest
import torch

from h3_audio_t8_pkg import sampling
from h3_audio_t8_pkg.modular_sampling.chunked_v5 import (
    _window_input, lift_standard_joint, prepare_standard_joint, sample_standard_window,
)
from h3_audio_t8_pkg.modular_sampling.chunked_v5_effects import (
    assert_v5_eav_binding, audit_v5_eav, bind_v5_eav,
)
from h3_audio_t8_pkg.modular_sampling.eav import EAVConfig
from test_modular_chunked_v5 import v5_harness as _base_v5_harness
from test_modular_speed_stages import _model
from test_progressive_sampling_runtime import conditioning


@pytest.fixture
def v5_case(monkeypatch):
    return _base_v5_harness.__wrapped__(monkeypatch)


def _prepared(v5_case):
    _calls, source, positive, noise, sampler, sigmas, plan = v5_case
    lifted, receipt, _ = lift_standard_joint(source, plan)
    prepared, _ = prepare_standard_joint(source, lifted, receipt, plan, noise)
    model, _other_sampler, _other_sigmas = sampling.setup_dual_clock_sampling(
        _model(), lifted, 4, 12., 3.,
    )
    return model, source, lifted, prepared, positive, noise, sampler, sigmas, plan


def test_v5_eav_binds_exact_first_window_and_audits_unexecuted_calls(v5_case):
    model, source, lifted, prepared, positive, noise, sampler, sigmas, plan = _prepared(v5_case)
    patched, runtime, report, context = bind_v5_eav(
        model, sigmas, source, lifted, prepared, plan, 0, None,
        EAVConfig(mode="report_only"),
    )
    assert patched is not model and context.stage == "pass2_window_0"
    assert runtime.context is context
    assert json.loads(report)["status"] == "unverified_incomplete_stage_coverage"
    first = _window_input(source, lifted, prepared, plan, 0, None)
    assert_v5_eav_binding(patched, first, prepared, 0, sigmas)
    _out, result, _ = sample_standard_window(
        patched, positive, source, lifted, prepared, plan, noise, sampler, sigmas, 0,
    )
    audited, audit_json = audit_v5_eav(result, prepared, runtime)
    assert audited is result.output_latent
    assert json.loads(audit_json)["status"] == "unverified_incomplete_stage_coverage"
    with pytest.raises(ValueError, match="exact joint AV window"):
        assert_v5_eav_binding(patched, first, prepared, 1, sigmas)


def test_v5_eav_second_window_uses_locked_video_and_audio_masks(v5_case):
    model, source, lifted, prepared, positive, noise, sampler, sigmas, plan = _prepared(v5_case)
    _first, previous, _ = sample_standard_window(
        model, positive, source, lifted, prepared, plan, noise, sampler, sigmas, 0,
    )
    window = _window_input(source, lifted, prepared, plan, 1, previous)
    video_mask, audio_mask = window.piece["noise_mask"].unbind()
    assert window.video_overlap > 0 and window.audio_overlap > 0
    assert torch.count_nonzero(video_mask[:, :, :window.video_overlap]) == 0
    assert torch.count_nonzero(audio_mask[..., :window.audio_overlap]) == 0
    patched, runtime, _report, _ = bind_v5_eav(
        model, sigmas, source, lifted, prepared, plan, 1, previous,
        EAVConfig(mode="report_only"),
    )
    assert_v5_eav_binding(patched, window, prepared, 1, sigmas)
    _final, result, _ = sample_standard_window(
        patched, positive, source, lifted, prepared, plan, noise, sampler, sigmas, 1,
        previous,
    )
    assert result.index == 1
    _audited, audit_json = audit_v5_eav(result, prepared, runtime)
    assert json.loads(audit_json)["chunked_v5_window_index"] == 1
    with pytest.raises(ValueError, match="previous PASS2 result"):
        bind_v5_eav(model, sigmas, source, lifted, prepared, plan, 1, None, EAVConfig())


def test_v5_eav_tiny_real_h3_report_only_forwards(v5_case, monkeypatch):
    model, source, lifted, prepared, _positive, noise, _sampler, sigmas, plan = _prepared(v5_case)
    # The fixture substitutes only upscaling/noise/sampling for parity tests.
    # Keep its already-produced lift, then restore the native tiny H3 sampler.
    monkeypatch.undo()
    _model_again, native_sampler, _unused = sampling.setup_dual_clock_sampling(
        _model(), lifted, 4, 12., 3.,
    )
    base, *_ = sample_standard_window(
        model, conditioning(), source, lifted, prepared, plan, noise,
        native_sampler, sigmas, 0,
    )
    patched, runtime, _report, _context = bind_v5_eav(
        model, sigmas, source, lifted, prepared, plan, 0, None,
        EAVConfig(mode="report_only", start_video_progress=0.1),
    )
    actual, result, _ = sample_standard_window(
        patched, conditioning(), source, lifted, prepared, plan, noise,
        native_sampler, sigmas, 0,
    )
    for new, old in zip(actual["samples"].unbind(), base["samples"].unbind()):
        assert torch.equal(new, old)
    _audited, report_json = audit_v5_eav(result, prepared, runtime)
    report = json.loads(report_json)
    assert report["completed_forwards"] == report["planned_forwards"] == 4
    assert report["clock_match"] is True
    assert report["status"] == "observed_report_only", report
    assert report["selector_calls"] > 0


def test_v5_eav_tiny_second_window_apply_observes_locked_av_masks(v5_case, monkeypatch):
    model, source, lifted, prepared, _positive, noise, _sampler, sigmas, plan = _prepared(v5_case)
    monkeypatch.undo()
    _unused_model, native_sampler, _unused_sigmas = sampling.setup_dual_clock_sampling(
        _model(), lifted, 4, 12., 3.,
    )
    _first, previous, _ = sample_standard_window(
        model, conditioning(), source, lifted, prepared, plan, noise,
        native_sampler, sigmas, 0,
    )
    patched, runtime, _report, _context = bind_v5_eav(
        model, sigmas, source, lifted, prepared, plan, 1, previous,
        EAVConfig(mode="apply_exp", start_video_progress=0.1, g_hard_limit=3.0),
    )
    _final, result, _ = sample_standard_window(
        patched, conditioning(), source, lifted, prepared, plan, noise,
        native_sampler, sigmas, 1, previous,
    )
    _audited, report_json = audit_v5_eav(result, prepared, runtime)
    report = json.loads(report_json)
    assert report["completed_forwards"] == report["planned_forwards"] == 4
    assert report["status"] == "observed_apply_exp", report
    assert report["feta"]["active_forward_count"] > 0
    assert report["feta"]["forwards"][0]["progressive_mask_contract"]["mask_present"] is True
