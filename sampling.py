from __future__ import annotations

import math

import torch

import comfy.model_sampling
import comfy.samplers
import comfy.utils
from comfy.k_diffusion.sampling import to_d

from .core import nested_av_parts


DEFAULT_SAMPLER_NAME = "dual_clock_euler"
DEFAULT_SCHEDULER_NAME = "native_flow"
SAMPLER_OPTIONS = [DEFAULT_SAMPLER_NAME]
if hasattr(comfy.model_sampling, "ModelSamplingAV"):
    SAMPLER_OPTIONS.extend(
        name for name in comfy.samplers.SAMPLER_NAMES if name != DEFAULT_SAMPLER_NAME
    )
SCHEDULER_OPTIONS = [
    DEFAULT_SCHEDULER_NAME,
    *(name for name in comfy.samplers.SCHEDULER_NAMES if name != DEFAULT_SCHEDULER_NAME),
]


class MiniMaxH3FlowSampling(comfy.model_sampling.ModelSamplingDiscreteFlow, comfy.model_sampling.CONST):
    @property
    def audio_scale(self):
        # The custom sampler advances the unpacked audio stream on its own clock.
        # New ComfyUI H3 builds require this property, but must not additionally
        # carry/scale audio onto the video clock (that would apply the transform twice).
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


def native_flow_sigmas(steps: int, shift_video: float) -> torch.Tensor:
    base_sigmas = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float32)
    return shift_sigma(base_sigmas, shift_video)


def _scheduler_sigmas(model_sampling, scheduler: str, steps: int, shift_video: float) -> torch.Tensor:
    if scheduler == DEFAULT_SCHEDULER_NAME:
        return native_flow_sigmas(steps, shift_video)
    if scheduler not in comfy.samplers.SCHEDULER_NAMES:
        raise ValueError(
            f"Unknown scheduler {scheduler!r}; available schedulers: "
            f"{', '.join(SCHEDULER_OPTIONS)}"
        )
    return comfy.samplers.calculate_sigmas(model_sampling, scheduler, steps).cpu()


def _make_sampling(
    model,
    original_sampling,
    shift_video: float,
    shift_audio: float,
    use_native_av: bool,
):
    if use_native_av:
        native_av_cls = getattr(comfy.model_sampling, "ModelSamplingAV", None)
        if native_av_cls is None or not model_uses_raw_audio_velocity(model):
            raise RuntimeError(
                "The selected ComfyUI sampler requires native MiniMax H3 FLOW_AV support. "
                "Update ComfyUI, or select dual_clock_euler for cross-version compatibility."
            )

        class MiniMaxH3NativeAVSampling(native_av_cls, comfy.model_sampling.CONST):
            pass

        model_sampling = MiniMaxH3NativeAVSampling(model.model.model_config)
        model_sampling.set_parameters(shift=shift_video, audio_shift=shift_audio)
    else:
        model_sampling = MiniMaxH3FlowSampling(model.model.model_config)
        model_sampling.set_parameters(shift=shift_video)

    if hasattr(original_sampling, "noise_scale"):
        model_sampling.set_noise_scale(original_sampling.noise_scale)
    return model_sampling


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
    audio_velocity_is_raw: bool = False,
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
        audio_delta = sigma_audio_next - sigma_audio
        if not audio_velocity_is_raw:
            # Older ComfyUI H3 builds returned an audio velocity multiplied by
            # d(sigma_audio)/d(sigma_video), so their update needs the inverse.
            audio_delta = audio_delta / slope_audio
        if audio_mask is not None:
            audio_delta = video_delta + audio_mask * (audio_delta - video_delta)

        if callback is not None:
            endpoint_scale = _audio_step_scale(
                sigma_video,
                sigma_audio,
                slope_audio,
                audio_mask,
                audio_velocity_is_raw,
            )
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


def setup_dual_clock_sampling(
    model,
    av_latent: dict,
    steps: int,
    shift_video: float,
    shift_audio: float,
    sampler_name: str = DEFAULT_SAMPLER_NAME,
    scheduler: str = DEFAULT_SCHEDULER_NAME,
):
    video, audio = nested_av_parts(av_latent)
    if video.shape[1] != 24 or audio.shape[1] != 32 or audio.shape[2] != 2:
        raise ValueError(
            "Unexpected MiniMax H3 AV channels: "
            f"video={tuple(video.shape)}, audio={tuple(audio.shape)}"
        )

    if sampler_name not in SAMPLER_OPTIONS:
        raise ValueError(
            f"Unknown sampler {sampler_name!r}; available samplers: "
            f"{', '.join(SAMPLER_OPTIONS)}"
        )

    use_native_av = sampler_name != DEFAULT_SAMPLER_NAME
    patched_model = model.clone()
    original_sampling = model.get_model_object("model_sampling")
    model_sampling = _make_sampling(
        model,
        original_sampling,
        shift_video,
        shift_audio,
        use_native_av,
    )
    patched_model.add_object_patch("model_sampling", model_sampling)

    transformer_options = patched_model.model_options.get("transformer_options", {}).copy()
    transformer_options["minimax_h3_sigma_shift_video"] = shift_video
    transformer_options["minimax_h3_sigma_shift_audio"] = shift_audio
    patched_model.model_options["transformer_options"] = transformer_options

    if use_native_av:
        sampler = comfy.samplers.sampler_object(sampler_name)
    else:
        audio_velocity_is_raw = model_uses_raw_audio_velocity(model)
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
                audio_velocity_is_raw=audio_velocity_is_raw,
            )

        sampler_function.__name__ = "sample_minimax_h3_dual_clock_euler"
        sampler = comfy.samplers.KSAMPLER(sampler_function)

    sigmas = _scheduler_sigmas(model_sampling, scheduler, steps, shift_video)
    return patched_model, sampler, sigmas
