from __future__ import annotations

import copy
import types

import pytest
import torch

import comfy.model_sampling
import comfy.nested_tensor
import comfy.samplers

from h3_audio_t8_pkg.sampling import (
    BETA57_ALPHA,
    BETA57_BETA,
    BETA57_SCHEDULER_NAME,
    DEFAULT_SAMPLER_NAME,
    DEFAULT_SCHEDULER_NAME,
    MiniMaxH3FlowSampling,
    SAMPLER_OPTIONS,
    SCHEDULER_OPTIONS,
    model_uses_raw_audio_velocity,
    native_flow_sigmas,
    rebind_dual_clock_sampler,
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


def test_partial_schedule_rebases_custom_audio_start_onto_audio_clock():
    sigmas = torch.tensor([0.9, 0.5, 0.0])
    latent_image = torch.tensor([[[2.0, 2.0]]])
    noise = torch.tensor([[[6.0, 6.0]]])
    flat_start = sigmas[0] * noise + (1.0 - sigmas[0]) * latent_image
    first_model_input = []

    class FakeSamplerModel:
        def __init__(self):
            self.noise = noise
            self.latent_image = latent_image

        def __call__(self, x, _sigma, **_kwargs):
            first_model_input.append(x.detach().clone())
            return x

    sample_minimax_h3_dual_clock_euler(
        FakeSamplerModel(),
        flat_start,
        sigmas,
        disable=True,
        video_values=1,
        packed_values=2,
        shift_video=12.0,
        shift_audio=3.0,
        audio_velocity_is_raw=True,
    )

    sigma_audio = time_shift_sigma(sigmas[0], 12.0, 3.0)
    expected_audio = sigma_audio * noise[..., 1:] + (1.0 - sigma_audio) * latent_image[..., 1:]
    assert first_model_input[0][..., :1] == pytest.approx(flat_start[..., :1])
    assert first_model_input[0][..., 1:] == pytest.approx(expected_audio)
    assert not torch.equal(first_model_input[0][..., 1:], flat_start[..., 1:])


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


def test_comfy_inpaint_wrapper_limits_zero_mask_audio_change_to_float_roundoff():
    sigmas = native_flow_sigmas(4, 12.0)
    latent_image = torch.tensor([[[0.0, 0.0, 7.0, -3.0]]])
    noise = torch.tensor([[[1.0, -1.0, 2.0, 4.0]]])
    denoise_mask = torch.tensor([[[1.0, 1.0, 0.0, 0.0]]])

    class FakeBaseModel:
        @staticmethod
        def scale_latent_inpaint(*, latent_image, **_kwargs):
            return latent_image

    class FakeDenoiser:
        inner_model = FakeBaseModel()

        @staticmethod
        def __call__(x, _sigma, **_kwargs):
            return torch.full_like(x, 0.25)

    inpaint_model = comfy.samplers.KSamplerX0Inpaint(FakeDenoiser(), sigmas)
    inpaint_model.latent_image = latent_image
    inpaint_model.noise = noise
    output = sample_minimax_h3_dual_clock_euler(
        inpaint_model,
        noise,
        sigmas,
        extra_args={"denoise_mask": denoise_mask},
        disable=True,
        video_values=2,
        packed_values=4,
        shift_video=12.0,
        shift_audio=3.0,
        audio_velocity_is_raw=True,
    )

    audio_roundoff = torch.amax(torch.abs(output[..., 2:] - latent_image[..., 2:]))
    assert float(audio_roundoff) <= 1e-6
    assert not torch.equal(output[..., :2], latent_image[..., :2])


def test_partial_schedule_rebase_preserves_explicit_zero_mask_audio_lock():
    sigmas = torch.tensor([0.9, 0.5, 0.0])
    latent_image = torch.tensor([[[0.0, 0.0, 7.0, -3.0]]])
    noise = torch.tensor([[[1.0, -1.0, 2.0, 4.0]]])
    denoise_mask = torch.tensor([[[1.0, 1.0, 0.0, 0.0]]])
    flat_start = sigmas[0] * noise + (1.0 - sigmas[0]) * latent_image

    class FakeBaseModel:
        @staticmethod
        def scale_latent_inpaint(*, latent_image, **_kwargs):
            return latent_image

    class FakeDenoiser:
        inner_model = FakeBaseModel()

        @staticmethod
        def __call__(x, _sigma, **_kwargs):
            return torch.full_like(x, 0.25)

    inpaint_model = comfy.samplers.KSamplerX0Inpaint(FakeDenoiser(), sigmas)
    inpaint_model.latent_image = latent_image
    inpaint_model.noise = noise
    output = sample_minimax_h3_dual_clock_euler(
        inpaint_model,
        flat_start,
        sigmas,
        extra_args={"denoise_mask": denoise_mask},
        disable=True,
        video_values=2,
        packed_values=4,
        shift_video=12.0,
        shift_audio=3.0,
        audio_velocity_is_raw=True,
    )

    assert output[..., 2:] == pytest.approx(latent_image[..., 2:], abs=1e-6)
    assert not torch.equal(output[..., :2], latent_image[..., :2])


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
    assert DEFAULT_SAMPLER_NAME == "dual_clock_euler"
    assert DEFAULT_SCHEDULER_NAME == "native_flow"


def test_custom_dual_clock_sampler_rebinds_to_upscaled_packed_geometry():
    low_video = torch.zeros((1, 24, 2, 2, 2))
    audio = torch.zeros((1, 32, 2, 8))
    low = {"samples": comfy.nested_tensor.NestedTensor((low_video, audio))}
    model, sampler, _sigmas = setup_dual_clock_sampling(
        FakeModelPatcher(), low, 4, 12.0, 3.0
    )
    high_video = torch.zeros((1, 24, 2, 4, 4))
    high = {"samples": comfy.nested_tensor.NestedTensor((high_video, audio))}

    rebound = rebind_dual_clock_sampler(model, high, sampler)

    assert rebound is not sampler
    assert rebound.sampler_function._minimax_h3_video_values == high_video[0].numel()
    assert rebound.sampler_function._minimax_h3_packed_values == (
        high_video[0].numel() + audio[0].numel()
    )
    assert rebound.sampler_function._minimax_h3_shift_video == 12.0
    assert rebound.sampler_function._minimax_h3_shift_audio == 3.0


def test_native_sampler_does_not_require_shape_rebinding():
    video = torch.zeros((1, 24, 2, 2, 2))
    audio = torch.zeros((1, 32, 2, 8))
    latent = {"samples": comfy.nested_tensor.NestedTensor((video, audio))}
    model, sampler, _sigmas = setup_dual_clock_sampling(
        FakeModelPatcher(FakeCurrentBaseModel()),
        latent,
        4,
        12.0,
        3.0,
        "euler",
        "native_flow",
    )

    assert rebind_dual_clock_sampler(model, latent, sampler) is sampler


def test_explicit_defaults_are_identical_to_the_legacy_five_argument_call():
    video = torch.zeros((1, 24, 2, 2, 2))
    audio = torch.zeros((1, 32, 2, 8))
    latent = {"samples": comfy.nested_tensor.NestedTensor((video, audio))}

    implicit = setup_dual_clock_sampling(FakeModelPatcher(), latent, 4, 12.0, 3.0)
    explicit = setup_dual_clock_sampling(
        FakeModelPatcher(),
        latent,
        4,
        12.0,
        3.0,
        "dual_clock_euler",
        "native_flow",
    )

    implicit_sampling = implicit[0].get_model_object("model_sampling")
    explicit_sampling = explicit[0].get_model_object("model_sampling")
    assert implicit_sampling.__class__ is explicit_sampling.__class__
    assert implicit[1].sampler_function.__name__ == explicit[1].sampler_function.__name__
    assert torch.equal(implicit[2], explicit[2])


def test_custom_euler_accepts_comfyui_scheduler_without_changing_audio_protocol():
    video = torch.zeros((1, 24, 2, 2, 2))
    audio = torch.zeros((1, 32, 2, 8))
    latent = {"samples": comfy.nested_tensor.NestedTensor((video, audio))}

    model, sampler, sigmas = setup_dual_clock_sampling(
        FakeModelPatcher(),
        latent,
        4,
        12.0,
        3.0,
        "dual_clock_euler",
        "normal",
    )
    model_sampling = model.get_model_object("model_sampling")

    assert model_sampling.audio_scale == 1.0
    assert sampler.sampler_function.__name__ == "sample_minimax_h3_dual_clock_euler"
    assert torch.equal(
        sigmas,
        comfy.samplers.calculate_sigmas(model_sampling, "normal", 4).cpu(),
    )
    assert not torch.equal(sigmas, native_flow_sigmas(4, 12.0))


def test_beta57_uses_current_comfyui_beta_scheduler_without_global_registration():
    video = torch.zeros((1, 24, 2, 2, 2))
    audio = torch.zeros((1, 32, 2, 8))
    latent = {"samples": comfy.nested_tensor.NestedTensor((video, audio))}

    registered_before = tuple(comfy.samplers.SCHEDULER_NAMES)
    model, sampler, sigmas = setup_dual_clock_sampling(
        FakeModelPatcher(),
        latent,
        4,
        12.0,
        3.0,
        "dual_clock_euler",
        BETA57_SCHEDULER_NAME,
    )
    model_sampling = model.get_model_object("model_sampling")

    assert tuple(comfy.samplers.SCHEDULER_NAMES) == registered_before
    assert SCHEDULER_OPTIONS[:2] == ["native_flow", "beta57"]
    assert sampler.sampler_function.__name__ == "sample_minimax_h3_dual_clock_euler"
    assert torch.equal(
        sigmas,
        comfy.samplers.beta_scheduler(
            model_sampling,
            4,
            alpha=BETA57_ALPHA,
            beta=BETA57_BETA,
        ).cpu(),
    )


def test_native_av_sampler_accepts_beta57_schedule():
    video = torch.zeros((1, 24, 2, 2, 2))
    audio = torch.zeros((1, 32, 2, 8))
    latent = {"samples": comfy.nested_tensor.NestedTensor((video, audio))}

    model, sampler, sigmas = setup_dual_clock_sampling(
        FakeModelPatcher(FakeCurrentBaseModel()),
        latent,
        4,
        12.0,
        3.0,
        "euler",
        BETA57_SCHEDULER_NAME,
    )
    model_sampling = model.get_model_object("model_sampling")

    assert isinstance(model_sampling, comfy.model_sampling.ModelSamplingAV)
    assert sampler.sampler_function.__name__ == "sample_euler"
    assert torch.equal(
        sigmas,
        comfy.samplers.beta_scheduler(
            model_sampling,
            4,
            alpha=BETA57_ALPHA,
            beta=BETA57_BETA,
        ).cpu(),
    )


def test_beta57_fails_clearly_when_comfyui_beta_scheduler_is_unavailable(monkeypatch):
    video = torch.zeros((1, 24, 2, 2, 2))
    audio = torch.zeros((1, 32, 2, 8))
    latent = {"samples": comfy.nested_tensor.NestedTensor((video, audio))}
    monkeypatch.delattr(comfy.samplers, "beta_scheduler")

    with pytest.raises(RuntimeError, match="update ComfyUI"):
        setup_dual_clock_sampling(
            FakeModelPatcher(),
            latent,
            4,
            12.0,
            3.0,
            "dual_clock_euler",
            BETA57_SCHEDULER_NAME,
        )


def test_standard_sampler_uses_current_comfyui_native_flow_av_protocol():
    video = torch.zeros((1, 24, 2, 2, 2))
    audio = torch.zeros((1, 32, 2, 8))
    latent = {"samples": comfy.nested_tensor.NestedTensor((video, audio))}

    model, sampler, sigmas = setup_dual_clock_sampling(
        FakeModelPatcher(FakeCurrentBaseModel()),
        latent,
        4,
        12.0,
        3.0,
        "euler",
        "native_flow",
    )
    model_sampling = model.get_model_object("model_sampling")

    assert isinstance(model_sampling, comfy.model_sampling.ModelSamplingAV)
    assert model_sampling.audio_scale == 4.0
    assert sampler.sampler_function.__name__ == "sample_euler"
    assert torch.equal(sigmas, native_flow_sigmas(4, 12.0))


def test_standard_sampler_fails_clearly_on_legacy_comfyui_h3_protocol():
    video = torch.zeros((1, 24, 2, 2, 2))
    audio = torch.zeros((1, 32, 2, 8))
    latent = {"samples": comfy.nested_tensor.NestedTensor((video, audio))}

    with pytest.raises(RuntimeError, match="FLOW_AV"):
        setup_dual_clock_sampling(
            FakeModelPatcher(),
            latent,
            4,
            12.0,
            3.0,
            "euler",
            "native_flow",
        )


def test_sampler_and_scheduler_choices_reject_unknown_api_values():
    video = torch.zeros((1, 24, 2, 2, 2))
    audio = torch.zeros((1, 32, 2, 8))
    latent = {"samples": comfy.nested_tensor.NestedTensor((video, audio))}

    assert SAMPLER_OPTIONS[0] == "dual_clock_euler"
    assert SCHEDULER_OPTIONS[0] == "native_flow"
    assert SCHEDULER_OPTIONS[1] == "beta57"
    with pytest.raises(ValueError, match="Unknown sampler"):
        setup_dual_clock_sampling(
            FakeModelPatcher(), latent, 4, 12.0, 3.0, "not_a_sampler", "native_flow"
        )
    with pytest.raises(ValueError, match="Unknown scheduler"):
        setup_dual_clock_sampling(
            FakeModelPatcher(), latent, 4, 12.0, 3.0, "dual_clock_euler", "not_a_scheduler"
        )


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
