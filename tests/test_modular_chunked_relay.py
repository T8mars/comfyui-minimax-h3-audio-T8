"""Real tiny external Prompt Relay execution on one full-clip PASS2."""
import json

import comfy.nested_tensor
import pytest
import torch

from h3_audio_t8_pkg import chunked_two_pass_upscale_advanced as old
from h3_audio_t8_pkg import sampling
from h3_audio_t8_pkg.modular_sampling import chunked_effects, chunked_relay
from h3_audio_t8_pkg.modular_sampling.eav import EAVConfig
from h3_audio_t8_pkg.modular_sampling.chunked_source import slice_chunked_source
from h3_audio_t8_pkg.modular_sampling.chunked_stages import (
    lift_chunked_segment, prepare_chunked_pass2, sample_chunked_pass2,
)
from h3_audio_t8_pkg.prompt_relay_advanced import (
    build_prompt_relay_conditioning, build_prompt_relay_plan,
)
from helpers import FakeAudioVAE, FakeVideoVAE
from test_chunked_two_pass_global_noise_advanced import _CountingCoordinateNoise, _plan
from test_modular_chunked_stages import _fake_lift
from test_modular_speed_stages import _model
from test_prompt_relay_advanced import NativeLikeFakeClip


def _prepared(monkeypatch, *, with_keyframes=False):
    monkeypatch.setattr(old, "learned_upscale_h3_av_latent", _fake_lift)
    relay_plan, *_ = build_prompt_relay_plan(
        global_prompt=("One continuous room with <Picture 1> first and <Picture 2> last"
                       if with_keyframes else "One continuous room"),
        local_prompts="A woman waves.\nShe walks away.",
        length=22, timing_mode="auto_equal", time_ranges="", math_profile="paper_v1",
        epsilon=0.1, allow_gaps=False, allow_overlaps=False,
    )
    plan = _plan(old.build_chunked_two_pass_global_noise_plan, strategy="full_frame_safe")
    original = {"samples": comfy.nested_tensor.NestedTensor((
        torch.zeros(1, 24, 7, 2, 2), torch.zeros(1, 32, 2, 37),
    ))}
    noise = _CountingCoordinateNoise()
    context, _ = prepare_chunked_pass2(original, plan, noise)
    segment, spec, _ = slice_chunked_source(original, plan, 0)
    lifted, _ = lift_chunked_segment(segment, spec, context, plan)
    model, sampler, sigmas = sampling.setup_dual_clock_sampling(_model(), lifted, 3, 12., 3.)
    relay_model, positive, relay_latent, *_ = build_prompt_relay_conditioning(
        model=model, clip=NativeLikeFakeClip(), video_vae=FakeVideoVAE(),
        audio_vae=FakeAudioVAE(), prompt_relay_plan=relay_plan,
        width=64, height=64, task_type="FL2VA" if with_keyframes else "T2VA",
        audio_mode="native",
        audio_denoise_strength=0.35, add_source_as_reference=False,
        prompt_primary_audio_ordinal=0, strict_prompt_tags=True,
        ref_image_size="match", reference_video_policy="official_2_to_15s",
        execution_mode="apply_exp", query_chunk_rows=64,
        **({"first_frame": torch.zeros((1, 64, 64, 3)),
            "last_frame": torch.ones((1, 64, 64, 3))} if with_keyframes else {}),
    )
    return (relay_model, positive, relay_latent, relay_plan, segment, lifted,
            spec, context, plan, sigmas, sampler, noise)


def test_external_relay_full_clip_real_tiny_calls_and_binding(monkeypatch):
    inputs = _prepared(monkeypatch)
    (relay_model, positive, relay_latent, relay_plan, segment, lifted,
     spec, context, plan, sigmas, sampler, noise) = inputs
    selected, paired, runtime, report = chunked_relay.bind_full_clip_relay(
        relay_model, positive, relay_latent, relay_plan,
        segment, lifted, spec, context, plan, sigmas,
    )
    assert json.loads(report)["sampled"] is False
    assert paired is positive and selected is not relay_model
    output, result, _ = sample_chunked_pass2(
        selected, paired, segment, lifted, spec, context, plan, noise, sampler, sigmas,
    )
    audited, audit_json = chunked_relay.audit_full_clip_relay(result, spec, runtime)
    assert audited is result.output_latent is output
    audit = json.loads(audit_json)
    assert audit["status"] == "observed_relay_calls_quality_unverified", audit
    assert audit["actual_calls"] == audit["expected_calls"]
    assert audit["actual_calls"]["routed_attention_calls"] > 0
    _second, second_result, _ = sample_chunked_pass2(
        selected, paired, segment, lifted, spec, context, plan, noise, sampler, sigmas,
    )
    _, second_json = chunked_relay.audit_full_clip_relay(second_result, spec, runtime)
    assert json.loads(second_json)["actual_calls"] == audit["expected_calls"]


def test_external_relay_rejects_wrong_plan_and_layout(monkeypatch):
    inputs = _prepared(monkeypatch)
    (relay_model, positive, relay_latent, relay_plan, segment, lifted,
     spec, context, plan, sigmas, _sampler, _noise) = inputs
    wrong_plan, *_ = build_prompt_relay_plan(
        global_prompt="Changed scene", local_prompts="A woman waves.\nShe walks away.",
        length=22, timing_mode="auto_equal", time_ranges="", math_profile="paper_v1",
        epsilon=0.1, allow_gaps=False, allow_overlaps=False,
    )
    with pytest.raises(ValueError, match="another Plan"):
        chunked_relay.bind_full_clip_relay(
            relay_model, positive, relay_latent, wrong_plan,
            segment, lifted, spec, context, plan, sigmas,
        )
    video, audio = relay_latent["samples"].unbind()
    wrong_latent = {"samples": comfy.nested_tensor.NestedTensor((video[:, :, :, :2, :], audio))}
    with pytest.raises(ValueError, match="layout differs"):
        chunked_relay.bind_full_clip_relay(
            relay_model, positive, wrong_latent, relay_plan,
            segment, lifted, spec, context, plan, sigmas,
        )
    unpaired = [[item[0], dict(item[1])] for item in positive]
    unpaired[0][1].pop("minimax_prompt_relay_binding")
    with pytest.raises(ValueError, match="MODEL and CONDITIONING"):
        chunked_relay.bind_full_clip_relay(
            relay_model, unpaired, relay_latent, relay_plan,
            segment, lifted, spec, context, plan, sigmas,
        )


def test_external_relay_fl2va_keyframes_run_through_real_tiny_pass2(monkeypatch):
    (relay_model, positive, relay_latent, relay_plan, segment, lifted,
     spec, context, plan, sigmas, sampler, noise) = _prepared(
         monkeypatch, with_keyframes=True,
     )
    assert len(positive[0][1]["minimax_keyframes"]) == 2
    selected, paired, runtime, _ = chunked_relay.bind_full_clip_relay(
        relay_model, positive, relay_latent, relay_plan,
        segment, lifted, spec, context, plan, sigmas,
    )
    _output, result, _report = sample_chunked_pass2(
        selected, paired, segment, lifted, spec, context, plan, noise, sampler, sigmas,
    )
    _latent, audit_json = chunked_relay.audit_full_clip_relay(result, spec, runtime)
    audit = json.loads(audit_json)
    assert audit["status"] == "observed_relay_calls_quality_unverified", audit
    assert audit["actual_calls"] == audit["expected_calls"]


def test_external_relay_rejects_post_bind_condition_or_sigmas_rewire(monkeypatch):
    (relay_model, positive, relay_latent, relay_plan, segment, lifted,
     spec, context, plan, sigmas, sampler, noise) = _prepared(monkeypatch)
    selected, paired, _runtime, _ = chunked_relay.bind_full_clip_relay(
        relay_model, positive, relay_latent, relay_plan,
        segment, lifted, spec, context, plan, sigmas,
    )
    unpaired = [[item[0], dict(item[1])] for item in paired]
    unpaired[0][1].pop("minimax_prompt_relay_binding")
    with pytest.raises(ValueError, match="MODEL and CONDITIONING"):
        sample_chunked_pass2(selected, unpaired, segment, lifted, spec, context,
                             plan, noise, sampler, sigmas)
    with pytest.raises(ValueError, match="not bound to this lifted segment"):
        sample_chunked_pass2(selected, paired, segment, lifted, spec, context,
                             plan, noise, sampler, sigmas * 0.9)


def test_external_relay_and_eav_combined_real_tiny_pass2(monkeypatch):
    (relay_model, positive, relay_latent, relay_plan, segment, lifted,
     spec, context, plan, sigmas, sampler, noise) = _prepared(monkeypatch)
    selected, paired, _relay_runtime, _ = chunked_relay.bind_full_clip_relay(
        relay_model, positive, relay_latent, relay_plan,
        segment, lifted, spec, context, plan, sigmas,
    )
    eav_model, eav_runtime, _report, _stage = chunked_effects.bind_chunked_eav(
        selected, sigmas, segment, lifted, spec, context, plan,
        EAVConfig(mode="apply_exp", start_video_progress=0.1, g_hard_limit=3.0),
    )
    _output, result, _ = sample_chunked_pass2(
        eav_model, paired, segment, lifted, spec, context, plan, noise, sampler, sigmas,
    )
    _latent, audit_json = chunked_effects.audit_chunked_eav(result, spec, eav_runtime)
    audit = json.loads(audit_json)
    assert audit["status"] == "observed_apply_exp", audit
    assert audit["relay_required"] is True
    assert audit["relay_attention_calls"] > 0
    assert audit["completed_forwards"] == audit["planned_forwards"] == 3
