from __future__ import annotations

import copy

import pytest
import torch

import comfy.model_sampling
import comfy.nested_tensor

from h3_audio_t8_pkg.sampling_multirate_exp import (
    MiniMaxH3MultiRateFlowSamplingEXP,
    balanced_microstep_counts,
    model_uses_raw_audio_velocity,
    multirate_flow_schedule,
    sample_minimax_h3_multirate_exp_euler,
    setup_multirate_exp_sampling,
    time_shift_slope,
)


def test_balanced_schedule_preserves_all_four_video_macro_boundaries():
    sigmas_8, macro_indices_8, counts_8 = multirate_flow_schedule(4, 8, 12.0)
    sigmas_10, macro_indices_10, counts_10 = multirate_flow_schedule(4, 10, 12.0)
    expected_macro_sigmas = torch.tensor([1.0, 36 / 37, 12 / 13, 0.8, 0.0])

    assert counts_8 == (2, 2, 2, 2)
    assert macro_indices_8 == (0, 2, 4, 6, 8)
    assert len(sigmas_8) == 9
    assert torch.allclose(sigmas_8[list(macro_indices_8)], expected_macro_sigmas)

    assert counts_10 == (2, 3, 2, 3)
    assert macro_indices_10 == (0, 2, 5, 7, 10)
    assert len(sigmas_10) == 11
    assert torch.allclose(sigmas_10[list(macro_indices_10)], expected_macro_sigmas)


def test_audio_steps_cannot_be_lower_than_video_steps():
    with pytest.raises(ValueError, match="greater than or equal"):
        balanced_microstep_counts(4, 3)


def test_multirate_integrates_constant_velocities_and_uses_audio_step_nfe():
    sigmas, macro_indices, _ = multirate_flow_schedule(4, 10, 12.0)
    calls = 0
    callbacks = []
    raw_video = 2.0
    raw_audio = 3.0

    def model(x, sigma, **_kwargs):
        nonlocal calls
        calls += 1
        slope = time_shift_slope(sigma, 12.0, 3.0).reshape(-1, 1, 1)
        derivative = torch.cat((
            torch.full_like(x[..., :2], raw_video),
            torch.full_like(x[..., 2:], raw_audio) * slope,
        ), dim=-1)
        return x - derivative * sigma.reshape(-1, 1, 1)

    output = sample_minimax_h3_multirate_exp_euler(
        model,
        torch.zeros((1, 1, 4)),
        sigmas,
        callback=callbacks.append,
        disable=True,
        video_values=2,
        packed_values=4,
        shift_video=12.0,
        shift_audio=3.0,
        macro_indices=macro_indices,
    )

    assert calls == 10
    assert [item["i"] for item in callbacks] == list(range(10))
    assert torch.allclose(output[..., :2], torch.full((1, 1, 2), -raw_video))
    assert torch.allclose(output[..., 2:], torch.full((1, 1, 2), -raw_audio))
    assert torch.allclose(callbacks[-1]["denoised"], output)


def test_current_h3_multirate_integrates_raw_audio_velocity_without_legacy_slope():
    sigmas, macro_indices, _ = multirate_flow_schedule(4, 10, 12.0)
    raw_video = 2.0
    raw_audio = 3.0
    callbacks = []

    def model(x, sigma, **_kwargs):
        derivative = torch.cat((
            torch.full_like(x[..., :2], raw_video),
            torch.full_like(x[..., 2:], raw_audio),
        ), dim=-1)
        return x - derivative * sigma.reshape(-1, 1, 1)

    output = sample_minimax_h3_multirate_exp_euler(
        model,
        torch.zeros((1, 1, 4)),
        sigmas,
        callback=callbacks.append,
        disable=True,
        video_values=2,
        packed_values=4,
        shift_video=12.0,
        shift_audio=3.0,
        macro_indices=macro_indices,
        audio_velocity_is_raw=True,
    )

    assert torch.allclose(output[..., :2], torch.full((1, 1, 2), -raw_video))
    assert torch.allclose(output[..., 2:], torch.full((1, 1, 2), -raw_audio))
    assert torch.allclose(callbacks[-1]["denoised"], output)


def test_video_commits_only_the_first_derivative_of_each_macro_step():
    sigmas, macro_indices, _ = multirate_flow_schedule(4, 10, 12.0)

    def model(x, sigma, **_kwargs):
        derivative = torch.cat((1.0 + x[..., :1], torch.zeros_like(x[..., 1:])), dim=-1)
        return x - derivative * sigma.reshape(-1, 1, 1)

    output = sample_minimax_h3_multirate_exp_euler(
        model,
        torch.zeros((1, 1, 2)),
        sigmas,
        disable=True,
        video_values=1,
        packed_values=2,
        shift_video=12.0,
        shift_audio=3.0,
        macro_indices=macro_indices,
    )

    expected = torch.zeros((1, 1, 1))
    for macro in range(4):
        sigma_start = sigmas[macro_indices[macro]]
        sigma_end = sigmas[macro_indices[macro + 1]]
        expected = expected + (1.0 + expected) * (sigma_end - sigma_start)
    assert torch.allclose(output[..., :1], expected)


def test_zero_audio_mask_keeps_flat_inpaint_endpoint_semantics():
    sigmas, macro_indices, _ = multirate_flow_schedule(4, 10, 12.0)

    def model(x, sigma, **_kwargs):
        denoised = x.clone()
        denoised[..., 2:] = 0.25
        return denoised

    output = sample_minimax_h3_multirate_exp_euler(
        model,
        torch.ones((1, 1, 4)),
        sigmas,
        extra_args={"denoise_mask": torch.tensor([[[1.0, 1.0, 0.0, 0.0]]])},
        disable=True,
        video_values=2,
        packed_values=4,
        shift_video=12.0,
        shift_audio=3.0,
        macro_indices=macro_indices,
    )
    assert torch.allclose(output[..., 2:], torch.full((1, 1, 2), 0.25))


class FakeModelConfig:
    sampling_settings = {"shift": 1.0, "multiplier": 1000}


class FakeBaseModel:
    model_config = FakeModelConfig()


class FakeCurrentBaseModel(FakeBaseModel):
    def audio_scale(self):
        return self.model_sampling.audio_scale


class FakeModelPatcher:
    def __init__(self, base_model=None):
        self.model = FakeBaseModel() if base_model is None else base_model
        self.model_options = {}
        self.objects = {
            "model_sampling": comfy.model_sampling.ModelSamplingDiscreteFlow(
                self.model.model_config
            )
        }

    def clone(self):
        cloned = FakeModelPatcher.__new__(FakeModelPatcher)
        cloned.model = self.model
        cloned.model_options = copy.deepcopy(self.model_options)
        cloned.objects = self.objects.copy()
        return cloned

    def get_model_object(self, name):
        return self.objects[name]

    def add_object_patch(self, name, value):
        self.objects[name] = value


def test_setup_keeps_exp_model_sampler_schedule_and_nfe_coherent():
    video = torch.zeros((1, 24, 2, 2, 2))
    audio = torch.zeros((1, 32, 2, 8))
    latent = {"samples": comfy.nested_tensor.NestedTensor((video, audio))}

    model, sampler, sigmas = setup_multirate_exp_sampling(
        FakeModelPatcher(), latent, 4, 10, 12.0, 3.0
    )

    assert model.get_model_object("model_sampling").shift == 12.0
    assert model.get_model_object("model_sampling").audio_scale == 1.0
    assert model.model_options["transformer_options"] == {
        "minimax_h3_sigma_shift_video": 12.0,
        "minimax_h3_sigma_shift_audio": 3.0,
    }
    assert sampler.sampler_function.__name__ == "sample_minimax_h3_multirate_exp_euler"
    assert len(sigmas) == 11


def test_exp_setup_detects_current_and_legacy_h3_audio_velocity_protocols():
    assert model_uses_raw_audio_velocity(FakeModelPatcher()) is False
    assert model_uses_raw_audio_velocity(FakeModelPatcher(FakeCurrentBaseModel())) is True


def test_exp_custom_sampling_exposes_neutral_audio_scale_for_current_comfy():
    sampling = MiniMaxH3MultiRateFlowSamplingEXP(FakeModelConfig())
    assert sampling.audio_scale == 1.0
