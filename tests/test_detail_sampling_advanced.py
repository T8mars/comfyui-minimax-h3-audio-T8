from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import comfy.nested_tensor
from comfy.ldm.minimax.model import MiniMaxH3Model

from h3_audio_t8_pkg.detail_sampling_advanced import (
    apply_h3_spatiotemporal_guidance,
    build_tail_detail_schedule,
    model_time_bias_sigma,
    setup_detail_mixer_sampling,
    setup_model_time_bias_sampling,
    setup_rectified_flow_restart_sampling,
    setup_two_pass_detail_mixer_sampling,
    temporal_detail_enhance,
)
from h3_audio_t8_pkg.nodes_detail_sampling_advanced import (
    DETAIL_SAMPLING_ADVANCED_NODE_CLASSES,
    MiniMaxH3AVTailDetailScheduleT8Advanced,
    MiniMaxH3DetailMixerSamplerT8Advanced,
    MiniMaxH3ModelTimeBiasSamplerT8Advanced,
    MiniMaxH3RectifiedFlowRestartSamplerT8Advanced,
    MiniMaxH3SpatioTemporalGuidanceT8Advanced,
    MiniMaxH3TemporalDetailEnhanceT8Advanced,
    MiniMaxH3TwoPassDetailMixerT8Advanced,
)
from h3_audio_t8_pkg.sampling import native_flow_sigmas, time_shift_sigma


def test_tail_detail_default_one_step_uses_video_sigma_midpoint():
    sigmas = native_flow_sigmas(8, 12.0)
    output, actual_nfe, report_json = build_tail_detail_schedule(
        sigmas,
        extra_tail_steps=1,
        spacing="video_sigma_linear",
        shift_video=12.0,
        shift_audio=3.0,
        profile="turbo_standard8",
    )
    report = json.loads(report_json)

    assert actual_nfe == 9
    assert output.shape == (10,)
    assert output[-3].item() == pytest.approx(12 / 19)
    assert output[-2].item() == pytest.approx(6 / 19)
    assert output[-1].item() == 0.0
    assert report["inserted_video_sigmas"] == pytest.approx([6 / 19])
    assert report["inserted_audio_sigmas"] == pytest.approx([3 / 29])
    assert report["final_zero_is_endpoint_not_model_call"] is True


def test_tail_detail_three_steps_descend_linearly_and_map_audio_clock():
    sigmas = native_flow_sigmas(8, 12.0).to(torch.float64)
    output, actual_nfe, report_json = build_tail_detail_schedule(
        sigmas,
        extra_tail_steps=3,
        spacing="video_sigma_linear",
        shift_video=12.0,
        shift_audio=3.0,
        profile="turbo_standard8",
    )
    report = json.loads(report_json)
    expected_video = [9 / 19, 6 / 19, 3 / 19]
    expected_audio = [9 / 49, 3 / 29, 3 / 67]

    assert actual_nfe == 11
    assert output[-5:-1].tolist() == pytest.approx([12 / 19, *expected_video])
    assert report["inserted_video_sigmas"] == pytest.approx(expected_video)
    assert report["inserted_audio_sigmas"] == pytest.approx(expected_audio)
    assert torch.all(time_shift_sigma(output, 12.0, 3.0)[:-1] > time_shift_sigma(output, 12.0, 3.0)[1:])


def test_tail_detail_zero_is_exact_object_identity_and_invalid_schedules_fail_closed():
    sigmas = native_flow_sigmas(8, 12.0)
    output, actual_nfe, report_json = build_tail_detail_schedule(
        sigmas,
        extra_tail_steps=0,
        spacing="video_sigma_linear",
        shift_video=12.0,
        shift_audio=3.0,
        profile="turbo_standard8",
    )
    assert output is sigmas
    assert actual_nfe == 8
    assert json.loads(report_json)["status"] == "noop"
    with pytest.raises(ValueError, match="strictly descending"):
        build_tail_detail_schedule(
            torch.tensor([1.0, 0.5, 0.6, 0.0]),
            extra_tail_steps=1,
            spacing="video_sigma_linear",
            shift_video=12.0,
            shift_audio=3.0,
            profile="custom_strict",
        )


def _two_pass_mixer_kwargs():
    return {
        "shift_video": 6.0,
        "shift_audio": 3.0,
        "enable_tail": False,
        "extra_tail_steps": 3,
        "tail_spacing": "video_sigma_linear",
        "enable_model_time_bias": False,
        "bias": -0.025,
        "bias_start_progress": 0.70,
        "bias_end_progress": 0.95,
        "bias_domain": "video_sigma",
        "enable_stg": False,
        "stg_scale": 0.35,
        "stg_double_blocks": "25",
        "stg_start_progress": 0.25,
        "stg_end_progress": 0.85,
        "enable_restart": False,
        "restart_video_sigma": 0.15,
        "restart_steps": 3,
        "restart_seed": 2608193401,
    }


def test_two_pass_detail_mixer_preserves_external_refine_schedule_when_disabled(monkeypatch):
    sigmas = torch.tensor([0.85, 0.6316, 0.3158, 0.0])
    sampler = object()
    model = SimpleNamespace(model_options={})
    patched = SimpleNamespace(model_options={})
    monkeypatch.setattr(
        "h3_audio_t8_pkg.detail_sampling_advanced.setup_dual_clock_sampling",
        lambda *_args, **_kwargs: (patched, sampler, torch.linspace(1.0, 0.0, 4)),
    )
    output_model, output_sampler, output_sigmas, nfe, forwards, report_text = (
        setup_two_pass_detail_mixer_sampling(
            model,
            {},
            sigmas,
            **_two_pass_mixer_kwargs(),
        )
    )
    assert output_model is patched
    assert output_sampler is sampler
    assert output_sigmas is sigmas
    assert nfe == 3
    assert forwards == 3
    report = json.loads(report_text)
    assert report["status"] == "parity_passthrough"
    assert report["external_refine_schedule_authoritative"] is True
    assert report["phase_scope"] == "high_resolution_refine_only"
    assert "rebases" in report["partial_start_clock_contract"]
    assert report["audio_completion_owner"] == "pass_2_when_audio_is_unmasked"


def test_two_pass_detail_mixer_tail_three_augments_only_refine_schedule(monkeypatch):
    sigmas = torch.tensor([0.85, 0.6316, 0.3158, 0.0])
    sampler = object()
    model = SimpleNamespace(model_options={})
    patched = SimpleNamespace(model_options={})
    monkeypatch.setattr(
        "h3_audio_t8_pkg.detail_sampling_advanced.setup_dual_clock_sampling",
        lambda *_args, **_kwargs: (patched, sampler, torch.linspace(1.0, 0.0, 4)),
    )
    kwargs = _two_pass_mixer_kwargs()
    kwargs["enable_tail"] = True
    output_model, output_sampler, output_sigmas, nfe, forwards, report_text = (
        setup_two_pass_detail_mixer_sampling(model, {}, sigmas, **kwargs)
    )
    assert output_model is patched
    assert output_sampler is sampler
    assert output_sigmas.shape == (7,)
    assert output_sigmas[-5:].tolist() == pytest.approx(
        [0.3158, 0.23685, 0.1579, 0.07895, 0.0]
    )
    assert nfe == 6
    assert forwards == 6
    report = json.loads(report_text)
    assert report["children"]["tail"]["phase_scope"] == "high_resolution_refine_only"
    assert report["enabled_mechanisms"] == ["tail_subdivision"]


def test_two_pass_detail_mixer_rejects_full_trajectory_sigmas():
    with pytest.raises(ValueError, match="starting below sigma 1.0"):
        setup_two_pass_detail_mixer_sampling(
            SimpleNamespace(model_options={}),
            {},
            torch.tensor([1.0, 0.5, 0.0]),
            **_two_pass_mixer_kwargs(),
        )


def test_two_pass_detail_mixer_composes_all_sampling_mechanisms(monkeypatch):
    video = torch.zeros((1, 24, 1, 1, 1))
    audio = torch.zeros((1, 32, 2, 1))
    latent = {"samples": comfy.nested_tensor.NestedTensor((video, audio))}
    refine_sigmas = torch.tensor([0.85, 0.6316, 0.3158, 0.0])
    source = _FakePatchModel()
    monkeypatch.setattr(
        "h3_audio_t8_pkg.detail_sampling_advanced.setup_dual_clock_sampling",
        lambda model, *_args, **_kwargs: (
            model,
            object(),
            torch.linspace(1.0, 0.0, 4),
        ),
    )
    monkeypatch.setattr(
        "h3_audio_t8_pkg.detail_sampling_advanced.model_uses_raw_audio_velocity",
        lambda _model: True,
    )
    kwargs = _two_pass_mixer_kwargs()
    kwargs.update(
        enable_tail=True,
        enable_model_time_bias=True,
        enable_stg=True,
        enable_restart=True,
    )
    model, sampler, sigmas, nfe, forwards, report_text = (
        setup_two_pass_detail_mixer_sampling(
            source,
            latent,
            refine_sigmas,
            **kwargs,
        )
    )
    report = json.loads(report_text)
    assert model.post_cfg is not None
    assert sigmas.numel() - 1 == 6
    assert sampler._reported_total_steps == 9
    assert nfe == 9
    assert forwards >= nfe
    assert report["model_time_biased_calls"] > 0
    assert report["enabled_mechanisms"] == [
        "tail_subdivision",
        "model_time_bias",
        "spatiotemporal_guidance",
        "rectified_flow_restart",
    ]


def test_model_time_bias_is_smooth_endpoint_zero_and_integrator_schedule_is_not_modified(monkeypatch):
    sigmas = native_flow_sigmas(8, 12.0)
    biased = model_time_bias_sigma(
        sigmas,
        bias=-0.1,
        start_progress=0.0,
        end_progress=1.0,
        shift_video=12.0,
        domain="video_sigma",
    )
    assert biased[0].item() == pytest.approx(sigmas[0].item())
    assert biased[-1].item() == pytest.approx(0.0)
    assert torch.any(biased[1:-1] < sigmas[1:-1])

    class FakeModel:
        def __init__(self):
            self.model_options = {"transformer_options": {}}
            self.wrapper = None

        def set_model_unet_function_wrapper(self, wrapper):
            self.wrapper = wrapper
            self.model_options["model_function_wrapper"] = wrapper

    fake_model = FakeModel()
    fake_sampler = object()
    monkeypatch.setattr(
        "h3_audio_t8_pkg.detail_sampling_advanced.setup_dual_clock_sampling",
        lambda *_args, **_kwargs: (fake_model, fake_sampler, sigmas),
    )
    model, sampler, returned_sigmas, report_json = setup_model_time_bias_sampling(
        object(),
        {},
        steps=8,
        shift_video=12.0,
        shift_audio=3.0,
        bias=-0.05,
        start_progress=0.7,
        end_progress=1.0,
        bias_domain="video_sigma",
    )
    assert model is fake_model
    assert sampler is fake_sampler
    assert returned_sigmas is sigmas
    assert fake_model.wrapper is not None
    report = json.loads(report_json)
    assert report["integrator_schedule_unchanged"] is True
    assert len(report["model_visible_video_call_sigmas"]) == 8
    assert report["final_zero_is_endpoint_not_model_call"] is True
    with pytest.raises(ValueError, match="between -0.5 and 0.0"):
        model_time_bias_sigma(
            sigmas,
            bias=0.1,
            start_progress=0.0,
            end_progress=1.0,
            shift_video=12.0,
            domain="video_sigma",
        )


class _FakePatchModel:
    def __init__(self, model_options=None, diffusion_model=None):
        self.model_options = {} if model_options is None else model_options
        self.post_cfg = None
        if diffusion_model is None:
            diffusion_model = MiniMaxH3Model.__new__(MiniMaxH3Model)
        self.model = SimpleNamespace(diffusion_model=diffusion_model)

    def clone(self):
        return _FakePatchModel(
            copy.deepcopy(self.model_options),
            self.model.diffusion_model,
        )

    def set_model_sampler_post_cfg_function(self, callback):
        self.post_cfg = callback
        self.model_options.setdefault("sampler_post_cfg_function", []).append(callback)

    def set_model_unet_function_wrapper(self, callback):
        self.model_options["model_function_wrapper"] = callback


def test_h3_stg_zero_scale_is_identity_and_patch_conflicts_fail_closed():
    source = _FakePatchModel()
    output, report_json = apply_h3_spatiotemporal_guidance(
        source,
        scale=0.0,
        double_blocks="25",
        start_progress=0.25,
        end_progress=0.85,
        shift_video=12.0,
        rescale=0.0,
    )
    assert output is source
    assert json.loads(report_json)["status"] == "noop"

    conflict = _FakePatchModel(
        {
            "transformer_options": {
                "patches_replace": {"dit": {("double_block", 25): object()}}
            }
        }
    )
    with pytest.raises(ValueError, match="overwrite existing"):
        apply_h3_spatiotemporal_guidance(
            conflict,
            scale=0.6,
            double_blocks="25",
            start_progress=0.25,
            end_progress=0.85,
            shift_video=12.0,
            rescale=0.0,
        )
    with pytest.raises(ValueError, match="between 0 and 49"):
        apply_h3_spatiotemporal_guidance(
            source,
            scale=0.6,
            double_blocks="50",
            start_progress=0.25,
            end_progress=0.85,
            shift_video=12.0,
            rescale=0.0,
        )

    with pytest.raises(ValueError, match="requires a native ComfyUI MiniMax H3"):
        apply_h3_spatiotemporal_guidance(
            _FakePatchModel(diffusion_model=object()),
            scale=0.6,
            double_blocks="25",
            start_progress=0.25,
            end_progress=0.85,
            shift_video=12.0,
            rescale=0.0,
        )


def test_h3_stg_rechecks_runtime_replacement_conflicts():
    source = _FakePatchModel()
    patched, _ = apply_h3_spatiotemporal_guidance(
        source,
        scale=0.6,
        double_blocks="25",
        start_progress=0.25,
        end_progress=0.85,
        shift_video=12.0,
        rescale=0.0,
    )
    with pytest.raises(RuntimeError, match="runtime double-block replacement conflicts"):
        patched.post_cfg(
            {
                "sigma": torch.tensor([0.8]),
                "cond": object(),
                "model_options": {
                    "transformer_options": {
                        "patches_replace": {
                            "dit": {("double_block", 25): object()},
                        }
                    }
                },
            }
        )

    with pytest.raises(ValueError, match="currently requires rescale=0"):
        apply_h3_spatiotemporal_guidance(
            source,
            scale=0.6,
            double_blocks="25",
            start_progress=0.25,
            end_progress=0.85,
            shift_video=12.0,
            rescale=0.1,
        )


def test_temporal_detail_noop_is_exact_and_upscale_is_multiple_of_32():
    frames = torch.rand((3, 65, 99, 3), generator=torch.Generator().manual_seed(7))
    output, report_json = temporal_detail_enhance(
        frames,
        upscale_factor=1.0,
        strength=0.0,
        blur_radius=2,
        blur_sigma=1.2,
        motion_threshold=0.04,
        temporal_guard=0.85,
    )
    assert output is frames
    assert output.shape[1:3] == (65, 99)
    assert json.loads(report_json)["multiple_of_32_applies_when_upscaling"] is True

    aligned = torch.rand((2, 64, 96, 3))
    exact, exact_report = temporal_detail_enhance(
        aligned,
        upscale_factor=1.0,
        strength=0.0,
        blur_radius=2,
        blur_sigma=1.2,
        motion_threshold=0.04,
        temporal_guard=0.85,
    )
    assert exact is aligned
    assert json.loads(exact_report)["status"] == "noop"

    enlarged, enlarged_report = temporal_detail_enhance(
        frames,
        upscale_factor=1.5,
        strength=0.0,
        blur_radius=2,
        blur_sigma=1.2,
        motion_threshold=0.04,
        temporal_guard=0.85,
    )
    assert enlarged.shape[1] % 32 == 0 and enlarged.shape[2] % 32 == 0
    enlarged_payload = json.loads(enlarged_report)
    assert enlarged_payload["output_multiple_of_32"] is True
    assert enlarged_payload["frame_chunk_size"] == 8

    tiny_growth, tiny_report = temporal_detail_enhance(
        frames,
        upscale_factor=1.01,
        strength=0.0,
        blur_radius=2,
        blur_sigma=1.2,
        motion_threshold=0.04,
        temporal_guard=0.85,
    )
    assert tiny_growth.shape[1] >= frames.shape[1]
    assert tiny_growth.shape[2] >= frames.shape[2]
    assert json.loads(tiny_report)["aspect_ratio_error_percent"] >= 0.0

    with pytest.raises(ValueError, match="safety budget"):
        temporal_detail_enhance(
            torch.rand((2, 640, 1152, 3)),
            upscale_factor=2.0,
            strength=0.0,
            blur_radius=2,
            blur_sigma=1.2,
            motion_threshold=0.04,
            temporal_guard=0.85,
            maximum_output_megapixels=2.1,
        )


def test_temporal_guard_reduces_detail_in_moving_regions():
    frames = torch.zeros((3, 64, 64, 3))
    frames[:, 16:48, 16:48] = 0.5
    frames[1, 24:40, 24:40] = 1.0
    guarded, report_json = temporal_detail_enhance(
        frames,
        upscale_factor=1.0,
        strength=0.8,
        blur_radius=2,
        blur_sigma=1.2,
        motion_threshold=0.01,
        temporal_guard=1.0,
    )
    unguarded, _ = temporal_detail_enhance(
        frames,
        upscale_factor=1.0,
        strength=0.8,
        blur_radius=2,
        blur_sigma=1.2,
        motion_threshold=0.01,
        temporal_guard=0.0,
    )
    assert torch.isfinite(guarded).all()
    assert (guarded - frames).abs().mean() < (unguarded - frames).abs().mean()
    assert json.loads(report_json)["audio_touched"] is False

    chunked, _ = temporal_detail_enhance(
        frames,
        upscale_factor=1.0,
        strength=0.8,
        blur_radius=2,
        blur_sigma=1.2,
        motion_threshold=0.01,
        temporal_guard=1.0,
        frame_chunk_size=1,
    )
    assert torch.equal(chunked, guarded)


def test_joint_av_rectified_flow_restart_is_deterministic_and_reports_real_extra_calls(monkeypatch):
    video = torch.zeros((1, 24, 1, 1, 1))
    audio = torch.zeros((1, 32, 2, 1))
    latent = {"samples": comfy.nested_tensor.NestedTensor((video, audio))}
    base_sigmas = native_flow_sigmas(2, 12.0)
    monkeypatch.setattr(
        "h3_audio_t8_pkg.detail_sampling_advanced.setup_dual_clock_sampling",
        lambda *_args, **_kwargs: (object(), object(), base_sigmas),
    )
    monkeypatch.setattr(
        "h3_audio_t8_pkg.detail_sampling_advanced.model_uses_raw_audio_velocity",
        lambda _model: True,
    )
    _model, sampler, returned_sigmas, report_json = setup_rectified_flow_restart_sampling(
        object(),
        latent,
        steps=2,
        shift_video=12.0,
        shift_audio=3.0,
        restart_video_sigma=0.15,
        restart_steps=3,
        restart_seed=1234,
    )

    class IdentityDenoiser:
        sigmas = None

        def __call__(self, x, _sigma, **_kwargs):
            return x

    packed_values = 24 + 64
    initial = torch.zeros((1, 1, packed_values))
    callbacks = []
    first = sampler.sampler_function(
        IdentityDenoiser(),
        initial.clone(),
        returned_sigmas,
        callback=callbacks.append,
        disable=True,
    )
    second = sampler.sampler_function(
        IdentityDenoiser(),
        initial.clone(),
        returned_sigmas,
        disable=True,
    )
    report = json.loads(report_json)

    assert torch.equal(first, second)
    assert not torch.equal(first, initial)
    assert [item["i"] for item in callbacks] == [0, 1, 2, 3, 4]
    assert report["base_nfe"] == 2
    assert report["restart_nfe"] == 3
    assert report["actual_total_nfe"] == 5
    assert report["restart_modalities"] == "joint_audio_video"


def test_joint_av_restart_rejects_locked_audio_and_fractional_masks(monkeypatch):
    video = torch.zeros((1, 24, 1, 1, 2))
    audio = torch.zeros((1, 32, 2, 2))
    base_sigmas = native_flow_sigmas(2, 12.0)
    monkeypatch.setattr(
        "h3_audio_t8_pkg.detail_sampling_advanced.setup_dual_clock_sampling",
        lambda *_args, **_kwargs: (object(), object(), base_sigmas),
    )
    monkeypatch.setattr(
        "h3_audio_t8_pkg.detail_sampling_advanced.model_uses_raw_audio_velocity",
        lambda _model: True,
    )

    locked_audio = {
        "samples": comfy.nested_tensor.NestedTensor((video, audio)),
        "noise_mask": comfy.nested_tensor.NestedTensor(
            (torch.ones_like(video), torch.zeros_like(audio))
        ),
    }
    with pytest.raises(ValueError, match="complete audio latent"):
        setup_rectified_flow_restart_sampling(
            object(),
            locked_audio,
            steps=2,
            shift_video=12.0,
            shift_audio=3.0,
            restart_video_sigma=0.15,
            restart_steps=3,
            restart_seed=1234,
        )

    fractional_video = {
        "samples": comfy.nested_tensor.NestedTensor((video, audio)),
        "noise_mask": comfy.nested_tensor.NestedTensor(
            (torch.full_like(video, 0.5), torch.ones_like(audio))
        ),
    }
    with pytest.raises(ValueError, match="binary video mask"):
        setup_rectified_flow_restart_sampling(
            object(),
            fractional_video,
            steps=2,
            shift_video=12.0,
            shift_audio=3.0,
            restart_video_sigma=0.15,
            restart_steps=3,
            restart_seed=1234,
        )


def test_joint_av_restart_allows_binary_conditioned_video_rows_with_full_audio(monkeypatch):
    video = torch.zeros((1, 24, 1, 1, 2))
    audio = torch.zeros((1, 32, 2, 2))
    video_mask = torch.ones_like(video)
    video_mask[..., 0] = 0.0
    latent = {
        "samples": comfy.nested_tensor.NestedTensor((video, audio)),
        "noise_mask": comfy.nested_tensor.NestedTensor(
            (video_mask, torch.ones_like(audio))
        ),
    }
    base_sigmas = native_flow_sigmas(2, 12.0)
    monkeypatch.setattr(
        "h3_audio_t8_pkg.detail_sampling_advanced.setup_dual_clock_sampling",
        lambda *_args, **_kwargs: (object(), object(), base_sigmas),
    )
    monkeypatch.setattr(
        "h3_audio_t8_pkg.detail_sampling_advanced.model_uses_raw_audio_velocity",
        lambda _model: True,
    )
    _model, _sampler, _sigmas, report_json = setup_rectified_flow_restart_sampling(
        object(),
        latent,
        steps=2,
        shift_video=12.0,
        shift_audio=3.0,
        restart_video_sigma=0.15,
        restart_steps=3,
        restart_seed=1234,
    )
    report = json.loads(report_json)
    assert report["video_mask_contract"]["active_fraction"] == pytest.approx(0.5)
    assert report["audio_mask_contract"]["all_active"] is True
    assert report["conditioned_binary_video_rows_preserved"] is True


def test_detail_mixer_all_disabled_uses_stable_route_and_reports_true_cost(monkeypatch):
    sigmas = native_flow_sigmas(8, 12.0)
    source = _FakePatchModel()
    sampler_marker = object()
    monkeypatch.setattr(
        "h3_audio_t8_pkg.detail_sampling_advanced.setup_dual_clock_sampling",
        lambda *_args, **_kwargs: (source, sampler_marker, sigmas),
    )
    model, sampler, output_sigmas, actual_nfe, forwards, report_json = (
        setup_detail_mixer_sampling(
            source,
            {},
            steps=8,
            shift_video=12.0,
            shift_audio=3.0,
            enable_tail=False,
            extra_tail_steps=1,
            tail_spacing="video_sigma_linear",
            profile="turbo_standard8",
            enable_model_time_bias=False,
            bias=-0.025,
            bias_start_progress=0.70,
            bias_end_progress=0.95,
            bias_domain="video_sigma",
            enable_stg=False,
            stg_scale=0.35,
            stg_double_blocks="25",
            stg_start_progress=0.25,
            stg_end_progress=0.85,
            enable_restart=False,
            restart_video_sigma=0.15,
            restart_steps=3,
            restart_seed=1234,
        )
    )
    report = json.loads(report_json)
    assert model is source
    assert sampler is sampler_marker
    assert output_sigmas is sigmas
    assert actual_nfe == forwards == 8
    assert report["status"] == "noop"
    assert report["enabled_mechanisms"] == []
    assert report["temporal_detail_external"] is True


def test_detail_mixer_composes_tail_bias_stg_and_restart_with_honest_nfe(monkeypatch):
    video = torch.zeros((1, 24, 1, 1, 1))
    audio = torch.zeros((1, 32, 2, 1))
    latent = {"samples": comfy.nested_tensor.NestedTensor((video, audio))}
    sigmas = native_flow_sigmas(8, 12.0)
    source = _FakePatchModel()
    monkeypatch.setattr(
        "h3_audio_t8_pkg.detail_sampling_advanced.setup_dual_clock_sampling",
        lambda model, *_args, **_kwargs: (model, object(), sigmas),
    )
    monkeypatch.setattr(
        "h3_audio_t8_pkg.detail_sampling_advanced.model_uses_raw_audio_velocity",
        lambda _model: True,
    )

    model, sampler, output_sigmas, actual_nfe, forwards, report_json = (
        setup_detail_mixer_sampling(
            source,
            latent,
            steps=8,
            shift_video=12.0,
            shift_audio=3.0,
            enable_tail=True,
            extra_tail_steps=3,
            tail_spacing="video_sigma_linear",
            profile="turbo_standard8",
            enable_model_time_bias=True,
            bias=-0.025,
            bias_start_progress=0.70,
            bias_end_progress=0.95,
            bias_domain="video_sigma",
            enable_stg=True,
            stg_scale=0.35,
            stg_double_blocks="25",
            stg_start_progress=0.25,
            stg_end_progress=0.85,
            enable_restart=True,
            restart_video_sigma=0.15,
            restart_steps=3,
            restart_seed=1234,
        )
    )
    report = json.loads(report_json)
    assert model.post_cfg is not None
    assert output_sigmas.numel() - 1 == 11
    assert sampler._reported_total_steps == 14
    assert actual_nfe == 14
    # Runtime comparisons use the actual float32 schedule; the nominal 0.25
    # boundary lands just below 0.25, so four calls are active, not five.
    assert report["stg_extra_weak_forwards"] == 4
    assert forwards == 18
    assert report["planned_joint_av_transformer_forwards"] == 18
    assert report["random_restart_applied"] is True
    assert report["applied_mechanisms"] == [
        "tail_subdivision",
        "model_time_bias",
        "spatiotemporal_guidance",
        "rectified_flow_restart",
    ]


def test_detail_mixer_stg_rejects_existing_post_cfg_hook():
    source = _FakePatchModel(
        {"sampler_post_cfg_function": [lambda args: args["denoised"]]}
    )
    with pytest.raises(ValueError, match="existing sampler_post_cfg_function"):
        setup_detail_mixer_sampling(
            source,
            {},
            steps=8,
            shift_video=12.0,
            shift_audio=3.0,
            enable_tail=False,
            extra_tail_steps=1,
            tail_spacing="video_sigma_linear",
            profile="turbo_standard8",
            enable_model_time_bias=False,
            bias=-0.025,
            bias_start_progress=0.70,
            bias_end_progress=0.95,
            bias_domain="video_sigma",
            enable_stg=True,
            stg_scale=0.35,
            stg_double_blocks="25",
            stg_start_progress=0.25,
            stg_end_progress=0.85,
            enable_restart=False,
            restart_video_sigma=0.15,
            restart_steps=3,
            restart_seed=1234,
        )


def test_detail_mixer_api_example_keeps_temporal_detail_after_decode_and_audio_bypass():
    root = Path(__file__).resolve().parents[1]
    prompt = json.loads(
        (root / "tests" / "fixtures" / "api" / "detail_mixer_advanced_api.json")
        .read_text(encoding="utf-8")
    )
    mixer = prompt["8"]
    assert mixer["class_type"] == "MiniMaxH3DetailMixerSamplerT8Advanced"
    assert mixer["inputs"]["enable_tail"] is True
    assert mixer["inputs"]["enable_model_time_bias"] is True
    assert mixer["inputs"]["enable_stg"] is True
    assert mixer["inputs"]["enable_restart"] is False
    assert prompt["13"]["class_type"] == "MiniMaxH3TemporalDetailEnhanceT8Advanced"
    assert prompt["13"]["inputs"]["frames"] == ["12", 0]
    assert prompt["14"]["inputs"]["images"] == ["13", 0]
    assert prompt["14"]["inputs"]["audio"] == ["12", 1]


def test_all_six_advanced_nodes_are_registered_in_isolated_order():
    assert DETAIL_SAMPLING_ADVANCED_NODE_CLASSES == [
        MiniMaxH3AVTailDetailScheduleT8Advanced,
        MiniMaxH3ModelTimeBiasSamplerT8Advanced,
        MiniMaxH3RectifiedFlowRestartSamplerT8Advanced,
        MiniMaxH3SpatioTemporalGuidanceT8Advanced,
        MiniMaxH3TemporalDetailEnhanceT8Advanced,
        MiniMaxH3DetailMixerSamplerT8Advanced,
    ]

    schema = MiniMaxH3DetailMixerSamplerT8Advanced.define_schema()
    inputs = {item.id: item for item in schema.inputs}
    assert inputs["enable_tail"].default is False
    assert inputs["enable_model_time_bias"].default is False
    assert inputs["enable_stg"].default is False
    assert inputs["enable_restart"].default is False

    two_pass_schema = MiniMaxH3TwoPassDetailMixerT8Advanced.define_schema()
    two_pass_inputs = {item.id: item for item in two_pass_schema.inputs}
    assert two_pass_schema.is_experimental is True
    assert two_pass_inputs["enable_tail"].default is False
    assert two_pass_inputs["extra_tail_steps"].default == 3
    assert two_pass_inputs["enable_model_time_bias"].default is False
    assert two_pass_inputs["enable_stg"].default is False
    assert two_pass_inputs["enable_restart"].default is False
