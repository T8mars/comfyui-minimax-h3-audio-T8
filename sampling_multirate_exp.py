from __future__ import annotations

import math

import torch

import comfy.model_sampling
import comfy.samplers
import comfy.utils
from comfy.k_diffusion.sampling import to_d

from .core import nested_av_parts


class MiniMaxH3MultiRateFlowSamplingEXP(
    comfy.model_sampling.ModelSamplingDiscreteFlow,
    comfy.model_sampling.CONST,
):
    @property
    def audio_scale(self):
        # This sampler owns the separate audio clock, so ComfyUI must not also
        # carry/scale the audio stream onto the video schedule.
        return 1.0


def model_uses_raw_audio_velocity(model) -> bool:
    """Detect the post-2026-08-06 ComfyUI MiniMax H3 sampling protocol."""
    base_model = getattr(model, "model", None)
    return callable(getattr(base_model, "audio_scale", None))


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


def balanced_microstep_counts(video_steps: int, audio_steps: int) -> tuple[int, ...]:
    if video_steps < 1:
        raise ValueError("video_steps must be at least 1")
    if audio_steps < video_steps:
        raise ValueError(
            "audio_steps must be greater than or equal to video_steps; "
            "the EXP sampler needs at least one joint H3 call per video macro step"
        )
    return tuple(
        ((macro + 1) * audio_steps) // video_steps
        - (macro * audio_steps) // video_steps
        for macro in range(video_steps)
    )


def multirate_flow_schedule(
    video_steps: int,
    audio_steps: int,
    shift_video: float,
) -> tuple[torch.Tensor, tuple[int, ...], tuple[int, ...]]:
    counts = balanced_microstep_counts(video_steps, audio_steps)
    base_sigmas = [1.0]
    macro_indices = [0]

    for macro, microsteps in enumerate(counts):
        macro_start = 1.0 - macro / video_steps
        macro_end = 1.0 - (macro + 1) / video_steps
        for micro in range(1, microsteps + 1):
            fraction = micro / microsteps
            base_sigmas.append(macro_start + (macro_end - macro_start) * fraction)
        macro_indices.append(len(base_sigmas) - 1)

    sigmas = shift_sigma(torch.tensor(base_sigmas, dtype=torch.float32), shift_video)
    return sigmas, tuple(macro_indices), counts


def _audio_step_scale(
    sigma_video,
    sigma_audio,
    slope_audio,
    denoise_mask,
    audio_velocity_is_raw: bool,
):
    flat_scale = -sigma_video
    dual_scale = -sigma_audio if audio_velocity_is_raw else -sigma_audio / slope_audio
    if denoise_mask is None:
        return dual_scale
    return flat_scale + denoise_mask * (dual_scale - flat_scale)


def sample_minimax_h3_multirate_exp_euler(
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
    macro_indices: tuple[int, ...],
    audio_velocity_is_raw: bool = False,
):
    extra_args = {} if extra_args is None else extra_args
    if x.shape[-1] != packed_values:
        raise ValueError(
            "MiniMax H3 packed latent changed after EXP sampler setup: "
            f"expected {packed_values} values, got {x.shape[-1]}"
        )
    if len(macro_indices) < 2 or macro_indices[0] != 0:
        raise ValueError("Invalid MiniMax H3 EXP macro-step boundaries")
    if macro_indices[-1] != len(sigmas) - 1:
        raise ValueError(
            "MiniMax H3 EXP sigma schedule changed after sampler setup: "
            f"expected {macro_indices[-1] + 1} values, got {len(sigmas)}"
        )

    denoise_mask = extra_args.get("denoise_mask")
    audio_mask = None
    if denoise_mask is not None:
        if denoise_mask.shape[-1] != packed_values:
            raise ValueError("MiniMax H3 denoise mask does not match the packed AV latent")
        audio_mask = denoise_mask[..., video_values:]

    video = x[..., :video_values]
    audio = x[..., video_values:]
    s_in = x.new_ones([x.shape[0]])
    progress = iter(comfy.utils.model_trange(len(sigmas) - 1, disable=disable))

    for macro in range(len(macro_indices) - 1):
        first_micro = macro_indices[macro]
        after_last_micro = macro_indices[macro + 1]
        sigma_macro_start = sigmas[first_micro]
        sigma_macro_end = sigmas[after_last_micro]
        video_start = video
        video_derivative = None

        for expected_step in range(first_micro, after_last_micro):
            step = next(progress)
            if step != expected_step:
                raise RuntimeError("MiniMax H3 EXP progress iterator returned an unexpected step")

            sigma_video = sigmas[step]
            sigma_video_next = sigmas[step + 1]
            if video_derivative is None:
                video_eval = video_start
            else:
                video_eval = video_start + video_derivative * (sigma_video - sigma_macro_start)

            x_eval = torch.cat((video_eval, audio), dim=-1)
            denoised = model(x_eval, sigma_video * s_in, **extra_args)
            derivative = to_d(x_eval, sigma_video, denoised)
            if video_derivative is None:
                video_derivative = derivative[..., :video_values]
            audio_derivative = derivative[..., video_values:]

            sigma_audio = time_shift_sigma(sigma_video, shift_video, shift_audio)
            sigma_audio_next = time_shift_sigma(sigma_video_next, shift_video, shift_audio)
            slope_audio = time_shift_slope(sigma_video, shift_video, shift_audio)
            video_delta = sigma_video_next - sigma_video
            audio_delta = sigma_audio_next - sigma_audio
            if not audio_velocity_is_raw:
                # Legacy H3 returned slope-scaled audio velocity; current H3
                # returns the raw audio-clock velocity.
                audio_delta = audio_delta / slope_audio
            if audio_mask is not None:
                audio_delta = video_delta + audio_mask * (audio_delta - video_delta)

            if callback is not None:
                video_endpoint = video_start - video_derivative * sigma_macro_start
                audio_endpoint_scale = _audio_step_scale(
                    sigma_video,
                    sigma_audio,
                    slope_audio,
                    audio_mask,
                    audio_velocity_is_raw,
                )
                audio_endpoint = audio + audio_derivative * audio_endpoint_scale
                callback({
                    "x": x_eval,
                    "i": step,
                    "sigma": sigma_video,
                    "sigma_hat": sigma_video,
                    "denoised": torch.cat((video_endpoint, audio_endpoint), dim=-1),
                })

            audio = audio + audio_derivative * audio_delta

        video = video_start + video_derivative * (sigma_macro_end - sigma_macro_start)

    return torch.cat((video, audio), dim=-1)


def setup_multirate_exp_sampling(
    model,
    av_latent: dict,
    video_steps: int,
    audio_steps: int,
    shift_video: float,
    shift_audio: float,
):
    video, audio = nested_av_parts(av_latent)
    if video.shape[1] != 24 or audio.shape[1] != 32 or audio.shape[2] != 2:
        raise ValueError(
            "Unexpected MiniMax H3 AV channels: "
            f"video={tuple(video.shape)}, audio={tuple(audio.shape)}"
        )

    sigmas, macro_indices, _counts = multirate_flow_schedule(
        video_steps, audio_steps, shift_video
    )

    patched_model = model.clone()
    audio_velocity_is_raw = model_uses_raw_audio_velocity(model)
    original_sampling = model.get_model_object("model_sampling")
    model_sampling = MiniMaxH3MultiRateFlowSamplingEXP(model.model.model_config)
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

    def sampler_function(model_wrap, x, schedule, extra_args=None, callback=None, disable=None):
        return sample_minimax_h3_multirate_exp_euler(
            model_wrap,
            x,
            schedule,
            extra_args=extra_args,
            callback=callback,
            disable=disable,
            video_values=video_values,
            packed_values=packed_values,
            shift_video=shift_video,
            shift_audio=shift_audio,
            macro_indices=macro_indices,
            audio_velocity_is_raw=audio_velocity_is_raw,
        )

    sampler_function.__name__ = "sample_minimax_h3_multirate_exp_euler"
    sampler = comfy.samplers.KSAMPLER(sampler_function)
    return patched_model, sampler, sigmas
