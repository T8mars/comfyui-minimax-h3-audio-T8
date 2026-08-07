from __future__ import annotations

import copy
import types

import pytest
import torch

import comfy.model_sampling
import comfy.nested_tensor

from h3_audio_t8_pkg.sampling import (
    MiniMaxH3FlowSampling,
    model_uses_raw_audio_velocity,
    native_flow_sigmas,
    sample_minimax_h3_dual_clock_euler,
    setup_dual_clock_sampling,
    time_shift_sigma,
    time_shift_slope,
)


def test_native_four_step_grid_matches_h3_reference_schedule():
    sigmas = native_flow_sigmas(4, 12.0)
    assert sigmas.tolist() == pytest.approx([1.0, 36 / 37, 12 / 13, 0.8, 0.0])
    audio_sigmas = time_shift_sigma(sigmas, 12.0, 3.0)
    assert audio_sigmas.tolist() == pytest.approx([1.0, 0.9, 0.75, 0.5, 0.0])


def test_dual_clock_euler_integrates_each_raw_velocity_on_its_own_clock():
    sigmas = native_flow_sigmas(4, 12.0)
    raw_video = 2.0
    raw_audio = 3.0

    def model(x, sigma, **_kwargs):
        slope = time_shift_slope(sigma, 12.0, 3.0).reshape(-1, 1, 1)
        derivative = torch.cat((
            torch.full_like(x[..., :2], raw_video),
            torch.full_like(x[..., 2:], raw_audio) * slope,
        ), dim=-1)
        return x - derivative * sigma.reshape(-1, 1, 1)

    output = sample_minimax_h3_dual_clock_euler(
        model,
        torch.zeros((1, 1, 4)),
        sigmas,
        disable=True,
        video_values=2,
        packed_values=4,
        shift_video=12.0,
        shift_audio=3.0,
    )
    assert output[..., :2] == pytest.approx(torch.full((1, 1, 2), -raw_video))
    assert output[..., 2:] == pytest.approx(torch.full((1, 1, 2), -raw_audio))


def test_current_h3_raw_audio_velocity_uses_audio_clock_without_legacy_slope():
    sigmas = native_flow_sigmas(4, 12.0)
    raw_video = 2.0
    raw_audio = 3.0
    callbacks = []

    def model(x, sigma, **_kwargs):
        derivative = torch.cat((
            torch.full_like(x[..., :2], raw_video),
            torch.full_like(x[..., 2:], raw_audio),
        ), dim=-1)
        return x - derivative * sigma.reshape(-1, 1, 1)

    output = sample_minimax_h3_dual_clock_euler(
        model,
        torch.zeros((1, 1, 4)),
        sigmas,
        callback=callbacks.append,
        disable=True,
        video_values=2,
        packed_values=4,
        shift_video=12.0,
        shift_audio=3.0,
        audio_velocity_is_raw=True,
    )
    assert output[..., :2] == pytest.approx(torch.full((1, 1, 2), -raw_video))
    assert output[..., 2:] == pytest.approx(torch.full((1, 1, 2), -raw_audio))
    assert callbacks[-1]["denoised"] == pytest.approx(output)


def test_zero_audio_denoise_mask_keeps_flat_inpaint_endpoint_semantics():
    sigmas = native_flow_sigmas(4, 12.0)

    def model(x, sigma, **_kwargs):
        denoised = x.clone()
        denoised[..., 2:] = 0.25
        return denoised

    output = sample_minimax_h3_dual_clock_euler(
        model,
        torch.ones((1, 1, 4)),
        sigmas,
        extra_args={"denoise_mask": torch.tensor([[[1.0, 1.0, 0.0, 0.0]]])},
        disable=True,
        video_values=2,
        packed_values=4,
        shift_video=12.0,
        shift_audio=3.0,
    )
    assert output[..., 2:] == pytest.approx(torch.full((1, 1, 2), 0.25))


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
        self.objects = {"model_sampling": comfy.model_sampling.ModelSamplingDiscreteFlow(self.model.model_config)}

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


def test_setup_keeps_model_sampler_and_schedule_shifts_coherent():
    video = torch.zeros((1, 24, 2, 2, 2))
    audio = torch.zeros((1, 32, 2, 8))
    latent = {"samples": comfy.nested_tensor.NestedTensor((video, audio))}
    model, sampler, sigmas = setup_dual_clock_sampling(FakeModelPatcher(), latent, 4, 12.0, 3.0)

    assert model.get_model_object("model_sampling").shift == 12.0
    assert model.get_model_object("model_sampling").audio_scale == 1.0
    assert model.model_options["transformer_options"] == {
        "minimax_h3_sigma_shift_video": 12.0,
        "minimax_h3_sigma_shift_audio": 3.0,
    }
    assert sampler.sampler_function.__name__ == "sample_minimax_h3_dual_clock_euler"
    assert len(sigmas) == 5


def test_setup_detects_current_and_legacy_h3_audio_velocity_protocols():
    assert model_uses_raw_audio_velocity(FakeModelPatcher()) is False
    assert model_uses_raw_audio_velocity(FakeModelPatcher(FakeCurrentBaseModel())) is True


def test_current_comfy_h3_audio_scale_access_accepts_custom_sampling():
    from comfy.model_base import MiniMaxH3

    sampling = MiniMaxH3FlowSampling(FakeModelConfig())
    holder = types.SimpleNamespace(
        latent_shapes=[(1, 24, 1, 1, 1), (1, 32, 2, 1)],
        model_sampling=sampling,
    )
    assert MiniMaxH3.audio_scale(holder) == 1.0
