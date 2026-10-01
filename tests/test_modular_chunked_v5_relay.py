"""Project an external full-clip Relay onto v5's non-native window grid."""
import json

import comfy.nested_tensor
import pytest
import torch

from h3_audio_t8_pkg import chunked_two_pass_parity as parity
from h3_audio_t8_pkg import chunked_two_pass_upscale_advanced as legacy
from h3_audio_t8_pkg import sampling
from h3_audio_t8_pkg.modular_sampling.chunked_v5 import (
    lift_standard_joint, prepare_standard_joint, sample_standard_window,
)
from h3_audio_t8_pkg.modular_sampling.chunked_v5_relay import (
    audit_v5_relay, project_v5_relay,
)
from h3_audio_t8_pkg.modular_sampling.chunked_v5_effects import bind_v5_eav, audit_v5_eav
from h3_audio_t8_pkg.modular_sampling.eav import EAVConfig
from h3_audio_t8_pkg.prompt_relay_advanced import (
    PROMPT_RELAY_BINDING_KEY, build_prompt_relay_conditioning,
    build_prompt_relay_plan,
)
from helpers import FakeAudioVAE, FakeVideoVAE
from test_chunked_two_pass_parity import _plan
from test_modular_chunked_stages import _fake_lift
from test_modular_speed_stages import _model
from test_prompt_relay_advanced import NativeLikeFakeClip


def _build_case(monkeypatch, *, with_keyframes=False, window_count=2):
    monkeypatch.setattr(legacy, "learned_upscale_h3_av_latent", _fake_lift)
    plan = _plan(
        temporal_chunk_frames={2: 136, 3: 102, 4: 85}[window_count],
        temporal_overlap_frames=34,
    )
    source = {"samples": comfy.nested_tensor.NestedTensor((
        torch.zeros(1, 24, 57, 2, 2), torch.zeros(1, 32, 2, 320),
    ))}

    class ZeroNoise:
        seed = 23

        def generate_noise(self, latent):
            return comfy.nested_tensor.NestedTensor(tuple(
                torch.zeros_like(value) for value in latent["samples"].unbind()
            ))

    noise = ZeroNoise()
    lifted, receipt, _ = lift_standard_joint(source, plan)
    expected_frames = {
        2: [(0, 136), (102, 192)],
        3: [(0, 102), (68, 170), (136, 192)],
        4: [(0, 85), (51, 136), (102, 187), (153, 192)],
    }
    assert [(item[1], item[3]) for item in receipt.segments] == expected_frames[window_count]
    prepared, _ = prepare_standard_joint(source, lifted, receipt, plan, noise)
    raw_model, sampler, _ = sampling.setup_dual_clock_sampling(_model(), lifted, 4, 12., 3.)
    relay_plan, *_ = build_prompt_relay_plan(
        global_prompt=("One continuous room with <Picture 1> first and <Picture 2> last"
                       if with_keyframes else "One continuous room"),
        local_prompts="A woman waves.\nShe walks away.",
        length=192, timing_mode="auto_equal", time_ranges="", math_profile="paper_v1",
        epsilon=0.1, allow_gaps=False, allow_overlaps=False,
    )
    relay_model, positive, relay_latent, *_ = build_prompt_relay_conditioning(
        model=raw_model, clip=NativeLikeFakeClip(), video_vae=FakeVideoVAE(),
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
    sigmas = torch.tensor(parity.UPSTREAM_REFINE_VIDEO_SIGMAS[4], dtype=torch.float32)
    return (raw_model, relay_model, positive, relay_latent, relay_plan,
            source, lifted, prepared, plan, sigmas, sampler, noise)


@pytest.fixture
def v5_relay_case(monkeypatch):
    return _build_case(monkeypatch)


def test_v5_relay_projects_first_window_and_runs_actual_native_calls(v5_relay_case):
    (raw_model, relay_model, positive, relay_latent, relay_plan,
     source, lifted, prepared, plan, sigmas, sampler, noise) = v5_relay_case
    selected, paired, runtime, report_json = project_v5_relay(
        raw_model, relay_model, positive, relay_latent, relay_plan,
        source, lifted, prepared, plan, sigmas, 0,
    )
    report = json.loads(report_json)
    assert report["window_frames"] == [0, 136]
    assert report["sampled"] is False
    with pytest.raises(ValueError, match="Project full-clip Prompt Relay"):
        sample_standard_window(relay_model, positive, source, lifted, prepared,
                               plan, noise, sampler, sigmas, 0)
    _output, result, _ = sample_standard_window(
        selected, paired, source, lifted, prepared, plan, noise, sampler, sigmas, 0,
    )
    _audited, audit_json = audit_v5_relay(result, prepared, runtime)
    audit = json.loads(audit_json)
    assert audit["status"] == "observed_relay_calls_quality_unverified", audit
    assert audit["actual_calls"] == audit["expected_calls"]
    assert audit["actual_calls"]["completed_forwards"] == 4
    assert audit["actual_calls"]["routed_attention_calls"] > 0


def test_v5_relay_second_window_rebases_events_and_preserves_refined_audio(v5_relay_case):
    (raw_model, relay_model, positive, relay_latent, relay_plan,
     source, lifted, prepared, plan, sigmas, sampler, noise) = v5_relay_case
    first_model, first_positive, _runtime, _ = project_v5_relay(
        raw_model, relay_model, positive, relay_latent, relay_plan,
        source, lifted, prepared, plan, sigmas, 0,
    )
    _first, previous, _ = sample_standard_window(
        first_model, first_positive, source, lifted, prepared, plan, noise, sampler, sigmas, 0,
    )
    second_model, second_positive, runtime, report_json = project_v5_relay(
        raw_model, relay_model, positive, relay_latent, relay_plan,
        source, lifted, prepared, plan, sigmas, 1, previous,
    )
    assert json.loads(report_json)["window_frames"] == [102, 192]
    first_binding = first_positive[0][1]["minimax_prompt_relay_binding"]
    second_binding = second_positive[0][1]["minimax_prompt_relay_binding"]
    assert second_binding["events"][0]["midpoint"] == pytest.approx(
        first_binding["events"][0]["midpoint"] - 170.0,
    )
    final, result, _ = sample_standard_window(
        second_model, second_positive, source, lifted, prepared, plan,
        noise, sampler, sigmas, 1, previous,
    )
    assert final["samples"].tensors[1].shape[-1] == 320
    assert result.index == 1
    _audited, audit_json = audit_v5_relay(result, prepared, runtime)
    audit = json.loads(audit_json)
    assert audit["status"] == "observed_relay_calls_quality_unverified", audit
    assert audit["actual_calls"] == audit["expected_calls"]


def test_v5_relay_rejects_global_plan_and_paired_conditioning_mismatch(v5_relay_case):
    (raw_model, relay_model, positive, relay_latent, relay_plan,
     source, lifted, prepared, plan, sigmas, _sampler, _noise) = v5_relay_case
    wrong_plan, *_ = build_prompt_relay_plan(
        global_prompt="Changed room", local_prompts="A woman waves.\nShe walks away.",
        length=192, timing_mode="auto_equal", time_ranges="", math_profile="paper_v1",
        epsilon=0.1, allow_gaps=False, allow_overlaps=False,
    )
    with pytest.raises(ValueError, match="another Plan"):
        project_v5_relay(raw_model, relay_model, positive, relay_latent,
                         wrong_plan, source, lifted, prepared, plan, sigmas, 0)
    unpaired = [[tensor, dict(metadata)] for tensor, metadata in positive]
    unpaired[0][1].pop("minimax_prompt_relay_binding")
    with pytest.raises(ValueError, match="MODEL and CONDITIONING"):
        project_v5_relay(raw_model, relay_model, unpaired, relay_latent,
                         relay_plan, source, lifted, prepared, plan, sigmas, 0)


def test_v5_relay_fl2va_first_last_frames_survive_both_window_layouts(monkeypatch):
    (raw_model, relay_model, positive, relay_latent, relay_plan,
     source, lifted, prepared, plan, sigmas, sampler, noise) = _build_case(
         monkeypatch, with_keyframes=True,
     )
    assert len(positive[0][1]["minimax_keyframes"]) == 2
    first_model, first_positive, first_runtime, _ = project_v5_relay(
        raw_model, relay_model, positive, relay_latent, relay_plan,
        source, lifted, prepared, plan, sigmas, 0,
    )
    _first, previous, _ = sample_standard_window(
        first_model, first_positive, source, lifted, prepared, plan,
        noise, sampler, sigmas, 0,
    )
    first_audit = json.loads(audit_v5_relay(previous, prepared, first_runtime)[1])
    assert first_audit["status"] == "observed_relay_calls_quality_unverified"
    second_model, second_positive, second_runtime, _ = project_v5_relay(
        raw_model, relay_model, positive, relay_latent, relay_plan,
        source, lifted, prepared, plan, sigmas, 1, previous,
    )
    _final, result, _ = sample_standard_window(
        second_model, second_positive, source, lifted, prepared, plan,
        noise, sampler, sigmas, 1, previous,
    )
    second_audit = json.loads(audit_v5_relay(result, prepared, second_runtime)[1])
    assert second_audit["status"] == "observed_relay_calls_quality_unverified", second_audit
    assert second_audit["actual_calls"] == second_audit["expected_calls"]


def test_v5_relay_and_eav_one_window_tiny_actual_combined_calls(v5_relay_case):
    (raw_model, relay_model, positive, relay_latent, relay_plan,
     source, lifted, prepared, plan, sigmas, sampler, noise) = v5_relay_case
    selected, paired, _standalone_runtime, _ = project_v5_relay(
        raw_model, relay_model, positive, relay_latent, relay_plan,
        source, lifted, prepared, plan, sigmas, 0,
    )
    eav_model, eav_runtime, _report, _stage = bind_v5_eav(
        selected, sigmas, source, lifted, prepared, plan, 0, None,
        EAVConfig(mode="apply_exp", start_video_progress=0.1, g_hard_limit=3.0),
        paired,
    )
    _output, result, _ = sample_standard_window(
        eav_model, paired, source, lifted, prepared, plan,
        noise, sampler, sigmas, 0,
    )
    _audited, report_json = audit_v5_eav(result, prepared, eav_runtime)
    report = json.loads(report_json)
    assert report["status"] == "observed_apply_exp", report
    assert report["relay_required"] is True
    assert report["relay_attention_calls"] > 0
    assert report["completed_forwards"] == report["planned_forwards"] == 4


def test_v5_relay_and_eav_second_window_actual_combined_calls(v5_relay_case):
    (raw_model, relay_model, positive, relay_latent, relay_plan,
     source, lifted, prepared, plan, sigmas, sampler, noise) = v5_relay_case
    first_model, first_positive, _first_runtime, _ = project_v5_relay(
        raw_model, relay_model, positive, relay_latent, relay_plan,
        source, lifted, prepared, plan, sigmas, 0,
    )
    _first, previous, _ = sample_standard_window(
        first_model, first_positive, source, lifted, prepared, plan,
        noise, sampler, sigmas, 0,
    )
    second_model, second_positive, _second_runtime, _ = project_v5_relay(
        raw_model, relay_model, positive, relay_latent, relay_plan,
        source, lifted, prepared, plan, sigmas, 1, previous,
    )
    eav_model, eav_runtime, _report, _stage = bind_v5_eav(
        second_model, sigmas, source, lifted, prepared, plan, 1, previous,
        EAVConfig(mode="apply_exp", start_video_progress=0.1, g_hard_limit=3.0),
        second_positive,
    )
    final, result, _ = sample_standard_window(
        eav_model, second_positive, source, lifted, prepared, plan,
        noise, sampler, sigmas, 1, previous,
    )
    assert final["samples"].tensors[0].shape[2] == 57
    assert final["samples"].tensors[1].shape[-1] == 320
    _audited, report_json = audit_v5_eav(result, prepared, eav_runtime)
    report = json.loads(report_json)
    assert report["status"] == "observed_apply_exp", report
    assert report["relay_required"] is True
    assert report["relay_attention_calls"] > 0
    assert report["completed_forwards"] == report["planned_forwards"] == 4


@pytest.mark.parametrize("window_count,with_keyframes", [
    (3, False), (4, False), (4, True),
])
def test_v5_multi_window_geometry_runs_relay_and_eav_on_every_window(
        monkeypatch, window_count, with_keyframes):
    (raw_model, relay_model, positive, relay_latent, relay_plan,
     source, lifted, prepared, plan, sigmas, sampler, noise) = _build_case(
         monkeypatch, window_count=window_count, with_keyframes=with_keyframes,
     )
    previous = None
    for index in range(window_count):
        projected_model, projected_positive, _relay_runtime, projection_json = (
            project_v5_relay(
                raw_model, relay_model, positive, relay_latent, relay_plan,
                source, lifted, prepared, plan, sigmas, index, previous,
            )
        )
        assert json.loads(projection_json)["window_index"] == index
        if with_keyframes:
            assert projected_positive[0][1][PROMPT_RELAY_BINDING_KEY]["task"] == (
                ["i2va", "t2va", "t2va", "l2va"][index])
        selected, runtime, _report, _context = bind_v5_eav(
            projected_model, sigmas, source, lifted, prepared, plan, index,
            previous, EAVConfig(mode="report_only", start_video_progress=0.1),
            projected_positive,
        )
        final, previous, _ = sample_standard_window(
            selected, projected_positive, source, lifted, prepared, plan,
            noise, sampler, sigmas, index, previous,
        )
        _audited, audit_json = audit_v5_eav(previous, prepared, runtime)
        audit = json.loads(audit_json)
        assert audit["chunked_v5_window_index"] == index
        assert audit["status"] == "observed_report_only", audit
        assert audit["relay_required"] is True
        assert audit["relay_attention_calls"] > 0
        assert audit["completed_forwards"] == audit["planned_forwards"] == 4
    assert final["samples"].tensors[0].shape[2] == 57
    assert final["samples"].tensors[1].shape[-1] == 320
