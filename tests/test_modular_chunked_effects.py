"""Chunked PASS2 external EAV is single-tile only and identity-bound."""
import json

import pytest
import torch
import comfy.nested_tensor

from h3_audio_t8_pkg import chunked_two_pass_upscale_advanced as old
from h3_audio_t8_pkg import sampling
from h3_audio_t8_pkg.modular_sampling import chunked_effects
from h3_audio_t8_pkg.modular_sampling.chunked_source import slice_chunked_source
from h3_audio_t8_pkg.modular_sampling.chunked_stages import (
    lift_chunked_segment, prepare_chunked_pass2, sample_chunked_pass2,
)
from h3_audio_t8_pkg.modular_sampling.eav import EAVConfig
from test_chunked_two_pass_global_noise_advanced import _CountingCoordinateNoise, _plan
from test_modular_chunked_source import _latent
from test_modular_chunked_stages import _fake_lift
from test_modular_speed_stages import _model
from test_progressive_sampling_runtime import conditioning


def _prepared(monkeypatch, builder, strategy="full_frame_safe"):
    monkeypatch.setattr(old, "learned_upscale_h3_av_latent", _fake_lift)
    plan = _plan(builder, strategy=strategy)
    source = _latent(masked=builder is old.build_chunked_two_pass_masked_low_sigma_plan)
    noise = _CountingCoordinateNoise()
    context, _ = prepare_chunked_pass2(source, plan, noise)
    segment, spec, _ = slice_chunked_source(source, plan, 0)
    lifted, _ = lift_chunked_segment(segment, spec, context, plan)
    model, sampler, sigmas = sampling.setup_dual_clock_sampling(
        _model(), lifted, 1, 12., 3.,
    )
    return model, sampler, sigmas, segment, lifted, spec, context, plan, noise


@pytest.mark.parametrize("builder", [
    old.build_chunked_two_pass_plan,
    old.build_chunked_two_pass_global_noise_plan,
    old.build_chunked_two_pass_low_sigma_plan,
    old.build_chunked_two_pass_masked_low_sigma_plan,
])
def test_chunked_eav_single_tile_binds_and_rejects_wrong_identity(monkeypatch, builder):
    model, _sampler, sigmas, segment, lifted, spec, context, plan, _noise = _prepared(
        monkeypatch, builder,
    )
    patched, runtime, report, stage = chunked_effects.bind_chunked_eav(
        model, sigmas, segment, lifted, spec, context, plan, EAVConfig(mode="report_only"),
    )
    assert patched is not model
    assert stage.stage == f"pass2_segment_{spec.index}"
    assert stage.video_shape == tuple(lifted["samples"].tensors[0].shape)
    assert json.loads(report)["status"] == "unverified_incomplete_stage_coverage"
    assert runtime.context is stage
    chunked_effects.assert_effect_binding(patched, lifted, spec)
    wrong = dict(lifted)
    video, audio = lifted["samples"].unbind()
    changed = video.clone()
    changed[0, 0, 0, 0, 0] += 1
    wrong["samples"] = type(lifted["samples"])((changed, audio))
    with pytest.raises(ValueError, match="not bound"):
        chunked_effects.assert_effect_binding(patched, wrong, spec)


def test_chunked_eav_rejects_multi_tile_and_unbound_clock(monkeypatch):
    model, _sampler, sigmas, segment, lifted, spec, context, plan, _noise = _prepared(
        monkeypatch, old.build_chunked_two_pass_global_noise_plan, "independent_tiles_exp",
    )
    with pytest.raises(ValueError, match="full_frame_safe"):
        chunked_effects.bind_chunked_eav(
            model, sigmas, segment, lifted, spec, context, plan, EAVConfig(),
        )
    plan["spatial_strategy"] = "full_frame_safe"
    with pytest.raises(ValueError, match="identity mismatch"):
        chunked_effects.bind_chunked_eav(
            model, sigmas, segment, lifted, spec, context, plan, EAVConfig(),
        )


def test_chunked_eav_audit_requires_matching_live_pass2_result(monkeypatch):
    model, sampler, sigmas, segment, lifted, spec, context, plan, noise = _prepared(
        monkeypatch, old.build_chunked_two_pass_global_noise_plan,
    )
    patched, runtime, _report, _stage = chunked_effects.bind_chunked_eav(
        model, sigmas, segment, lifted, spec, context, plan, EAVConfig(mode="report_only"),
    )
    monkeypatch.setattr(old, "_spatial_resample", lambda video, *_args, **_kwargs: (video, {}))
    _output, result, _ = sample_chunked_pass2(
        patched, [], segment, lifted, spec, context, plan, noise, sampler, sigmas,
    )
    audited, report = chunked_effects.audit_chunked_eav(result, spec, runtime)
    assert audited is result.output_latent
    assert json.loads(report)["status"] == "unverified_incomplete_stage_coverage"
    result.output_latent["samples"].tensors[0][0, 0, 0, 0, 0] += 1
    with pytest.raises(ValueError, match="identity mismatch"):
        chunked_effects.audit_chunked_eav(result, spec, runtime)


def test_chunked_eav_real_tiny_forward_and_report_only_parity(monkeypatch):
    model, sampler, _sigmas, segment, lifted, spec, context, plan, _noise = _prepared(
        monkeypatch, old.build_chunked_two_pass_plan,
    )
    sigmas = sampling.native_flow_sigmas(3, 12.)

    class ZeroNoise:
        seed = 7

        def generate_noise(self, latent):
            return comfy.nested_tensor.NestedTensor(tuple(
                torch.zeros_like(value) for value in latent["samples"].unbind()
            ))

    noise = ZeroNoise()
    base, *_ = sample_chunked_pass2(
        model, conditioning(), segment, lifted, spec, context, plan, noise, sampler, sigmas,
    )
    patched, runtime, _report, _stage = chunked_effects.bind_chunked_eav(
        model, sigmas, segment, lifted, spec, context, plan,
        EAVConfig(mode="report_only", start_video_progress=0.1),
    )
    actual, result, _ = sample_chunked_pass2(
        patched, conditioning(), segment, lifted, spec, context, plan, noise, sampler, sigmas,
    )
    for a, b in zip(actual["samples"].unbind(), base["samples"].unbind()):
        assert torch.equal(a, b)
    _audited, report_json = chunked_effects.audit_chunked_eav(result, spec, runtime)
    report = json.loads(report_json)
    assert report["completed_forwards"] == report["planned_forwards"] == 3
    assert report["clock_match"] is True
    assert report["status"] == "observed_report_only", json.dumps(report["feta"]["forwards"], indent=2)
    assert report["selector_calls"] > 0


def test_chunked_v4_masked_eav_real_tiny_apply_calls(monkeypatch):
    model, sampler, _sigmas, segment, lifted, spec, context, plan, noise = _prepared(
        monkeypatch, old.build_chunked_two_pass_masked_low_sigma_plan,
    )
    sigmas = sampling.native_flow_sigmas(3, 12.)
    patched, runtime, _report, _stage = chunked_effects.bind_chunked_eav(
        model, sigmas, segment, lifted, spec, context, plan,
        EAVConfig(mode="apply_exp", start_video_progress=0.1),
    )
    _output, result, _ = sample_chunked_pass2(
        patched, conditioning(), segment, lifted, spec, context, plan, noise, sampler, sigmas,
    )
    _latent_output, report_json = chunked_effects.audit_chunked_eav(result, spec, runtime)
    report = json.loads(report_json)
    assert report["status"] == "observed_apply_exp", report
    assert report["completed_forwards"] == report["planned_forwards"] == 3
    assert report["feta"]["active_forward_count"] > 0
    assert report["feta"]["forwards"][0]["progressive_mask_contract"]["mask_present"] is True
