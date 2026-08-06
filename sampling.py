from __future__ import annotations

import math

import torch

import comfy.model_sampling
import comfy.samplers
import comfy.utils
from comfy.k_diffusion.sampling import to_d

from .core import nested_av_parts


class MiniMaxH3FlowSampling(comfy.model_sampling.ModelSamplingDiscreteFlow, comfy.model_sampling.CONST):
    pass


def shift_sigma(base_sigma, shift: float):
    return shift * base_sigma / (1.0 + (shift - 1.0) * base_sigma)


def time_shift_sigma(sigma, from_shift: float, to_shift: float):
    base_sigma = sigma / (from_shift + sigma * (1.0 - from_shift))
    return shift_sigma(base_sigma, to_shift)


def time_shift_slope(sigma, from_shift: float, to_shift: float):
    base_sigma = sigma / (from_shift + sigma * (1.0 - from_shift))
    numerator = to_shift * (1.0 + (from_shift - 1.0) * base_sigma) ** 2
    denominator = from_shift * (1.0 + (to_shift - 1.0) * base_sigma) ** 2
    return numerator / denominator


def native_flow_sigmas(steps: int, shift_video: float) -> torch.Tensor:
    base_sigmas = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float32)
    return shift_sigma(base_sigmas, shift_video)


def _audio_step_scale(sigma_video, sigma_audio, slope_audio, denoise_mask):
    flat_scale = -sigma_video
    dual_scale = -sigma_audio / slope_audio
    if denoise_mask is None:
        return dual_scale
    return flat_scale + denoise_mask * (dual_scale - flat_scale)


def sample_minimax_h3_dual_clock_euler(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    *,
    video_values: int,
    packed_values: int,
    shift_video: float,
    shift_audio: float,
):
    extra_args = {} if extra_args is None else extra_args
    if x.shape[-1] != packed_values:
        raise ValueError(
            "MiniMax H3 packed latent changed after sampler setup: "
            f"expected {packed_values} values, got {x.shape[-1]}"
        )

    denoise_mask = extra_args.get("denoise_mask")
    audio_mask = None
    if denoise_mask is not None:
        if denoise_mask.shape[-1] != packed_values:
            raise ValueError("MiniMax H3 denoise mask does not match the packed AV latent")
        audio_mask = denoise_mask[..., video_values:]

    s_in = x.new_ones([x.shape[0]])
    for step in comfy.utils.model_trange(len(sigmas) - 1, disable=disable):
        sigma_video = sigmas[step]
        sigma_video_next = sigmas[step + 1]
        denoised = model(x, sigma_video * s_in, **extra_args)
        derivative = to_d(x, sigma_video, denoised)

        sigma_audio = time_shift_sigma(sigma_video, shift_video, shift_audio)
        sigma_audio_next = time_shift_sigma(sigma_video_next, shift_video, shift_audio)
        slope_audio = time_shift_slope(sigma_video, shift_video, shift_audio)
        video_delta = sigma_video_next - sigma_video
        audio_delta = (sigma_audio_next - sigma_audio) / slope_audio
        if audio_mask is not None:
            audio_delta = video_delta + audio_mask * (audio_delta - video_delta)

        if callback is not None:
            endpoint_scale = _audio_step_scale(sigma_video, sigma_audio, slope_audio, audio_mask)
            denoised[..., video_values:] = x[..., video_values:] + derivative[..., video_values:] * endpoint_scale
            callback({
                "x": x,
                "i": step,
                "sigma": sigma_video,
                "sigma_hat": sigma_video,
                "denoised": denoised,
            })

        x = torch.cat((
            x[..., :video_values] + derivative[..., :video_values] * video_delta,
            x[..., video_values:] + derivative[..., video_values:] * audio_delta,
        ), dim=-1)
    return x


def setup_dual_clock_sampling(model, av_latent: dict, steps: int, shift_video: float, shift_audio: float):
    video, audio = nested_av_parts(av_latent)
    if video.shape[1] != 24 or audio.shape[1] != 32 or audio.shape[2] != 2:
        raise ValueError(
            "Unexpected MiniMax H3 AV channels: "
            f"video={tuple(video.shape)}, audio={tuple(audio.shape)}"
        )

    patched_model = model.clone()
    original_sampling = model.get_model_object("model_sampling")
    model_sampling = MiniMaxH3FlowSampling(model.model.model_config)
    model_sampling.set_parameters(shift=shift_video)
    if hasattr(original_sampling, "noise_scale"):
        model_sampling.set_noise_scale(original_sampling.noise_scale)
    patched_model.add_object_patch("model_sampling", model_sampling)

    transformer_options = patched_model.model_options.get("transformer_options", {}).copy()
    transformer_options["minimax_h3_sigma_shift_video"] = shift_video
    transformer_options["minimax_h3_sigma_shift_audio"] = shift_audio
    patched_model.model_options["transformer_options"] = transformer_options

    video_values = math.prod(video.shape[1:])
    packed_values = video_values + math.prod(audio.shape[1:])

    def sampler_function(model_wrap, x, sigmas, extra_args=None, callback=None, disable=None):
        return sample_minimax_h3_dual_clock_euler(
            model_wrap,
            x,
            sigmas,
            extra_args=extra_args,
            callback=callback,
            disable=disable,
            video_values=video_values,
            packed_values=packed_values,
            shift_video=shift_video,
            shift_audio=shift_audio,
        )

    sampler_function.__name__ = "sample_minimax_h3_dual_clock_euler"
    sampler = comfy.samplers.KSAMPLER(sampler_function)
    sigmas = native_flow_sigmas(steps, shift_video)
    return patched_model, sampler, sigmas
