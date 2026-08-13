from __future__ import annotations

import json
import math
from collections.abc import Mapping

import torch

import comfy.samplers
import comfy.utils
from comfy.k_diffusion.sampling import to_d

from .core import encode_audio_once, fit_audio_latent, nested_av_parts, replace_audio_latent
from .sampling import (
    DEFAULT_SCHEDULER_NAME,
    SCHEDULER_OPTIONS,
    model_uses_raw_audio_velocity,
    setup_dual_clock_sampling,
    time_shift_sigma,
    time_shift_slope,
)


SCHEMA = "t8.minimax_h3.scheduled_drive_audio_injection.v1"
MODES = ("report_only", "scheduled_injection")
ENVELOPES = ("constant", "fade_out", "fade_in")


def _report(**values) -> str:
    return json.dumps({"schema": SCHEMA, **values}, ensure_ascii=False, indent=2, sort_keys=True)


def _progress_weight(progress: float, start: float, end: float, envelope: str) -> float:
    if progress < start or progress > end:
        return 0.0
    if envelope == "constant" or math.isclose(start, end):
        return 1.0
    local = (progress - start) / (end - start)
    if envelope == "fade_out":
        return 1.0 - local
    if envelope == "fade_in":
        return local
    raise ValueError(f"Unknown injection envelope {envelope!r}")


def _patch_stack_summary(model) -> list[str]:
    markers: list[str] = []
    options = getattr(model, "model_options", {})
    transformer = options.get("transformer_options", {}) if isinstance(options, Mapping) else {}
    patches = transformer.get("patches_replace", {}) if isinstance(transformer, Mapping) else {}
    if isinstance(patches, Mapping) and any(bool(value) for value in patches.values()):
        markers.append("patches_replace")

    base = getattr(model, "model", None)
    extra_conds = getattr(base, "extra_conds", None)
    forward = getattr(getattr(base, "diffusion_model", None), "forward", None)
    extra_fn = getattr(extra_conds, "__func__", extra_conds)
    forward_fn = getattr(forward, "__func__", forward)
    if getattr(extra_fn, "_t8_long_video_patch_version", None) is not None:
        markers.append("long_video")
    if getattr(extra_fn, "_t8_multikeyframe_patch_version", None) is not None:
        markers.append("multikeyframe")
    if getattr(forward_fn, "_t8_multikeyframe_patch_version", None) is not None:
        markers.append("multikeyframe_forward")
    return sorted(set(markers))


def _validate_contract(
    model,
    av_latent: dict,
    mode: str,
    start_percent: float,
    end_percent: float,
    strength: float,
    envelope: str,
    allow_unverified_patch_stack: bool,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    if mode not in MODES:
        raise ValueError(f"Unknown scheduled audio injection mode {mode!r}")
    if envelope not in ENVELOPES:
        raise ValueError(f"Unknown injection envelope {envelope!r}")
    if not 0.0 <= start_percent <= end_percent <= 1.0:
        raise ValueError("Injection window must satisfy 0 <= start_percent <= end_percent <= 1")
    if not 0.0 <= strength <= 1.0:
        raise ValueError("Injection strength must be between 0 and 1")

    video, audio = nested_av_parts(av_latent)
    if video.shape[0] != 1 or audio.shape[0] != 1:
        raise ValueError("Scheduled drive-audio injection currently requires batch size 1")
    if video.shape[1] != 24 or audio.shape[1] != 32 or audio.shape[2] != 2:
        raise ValueError(
            "Unexpected MiniMax H3 AV latent layout: "
            f"video={tuple(video.shape)}, audio={tuple(audio.shape)}"
        )

    conflicts = _patch_stack_summary(model)
    if mode == "scheduled_injection" and conflicts and not allow_unverified_patch_stack:
        raise RuntimeError(
            "Scheduled drive-audio injection found an unverified model patch stack "
            f"({', '.join(conflicts)}). Keep this Advanced probe isolated, or explicitly "
            "enable allow_unverified_patch_stack after accepting the compatibility risk."
        )
    return video, audio, conflicts


def _audio_step_scale(
    sigma_video,
    sigma_audio,
    slope_audio,
    audio_mask,
    audio_velocity_is_raw: bool,
):
    flat_scale = -sigma_video
    dual_scale = -sigma_audio if audio_velocity_is_raw else -sigma_audio / slope_audio
    if audio_mask is None:
        return dual_scale
    return flat_scale + audio_mask * (dual_scale - flat_scale)


def sample_minimax_h3_scheduled_audio_euler(
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
    audio_velocity_is_raw: bool,
    source_audio_x0: torch.Tensor,
    fixed_audio_noise: torch.Tensor,
    noise_scale: float,
    start_percent: float,
    end_percent: float,
    strength: float,
    envelope: str,
    lock_final_audio: bool,
):
    extra_args = {} if extra_args is None else extra_args
    if x.shape[-1] != packed_values:
        raise ValueError(
            "MiniMax H3 packed latent changed after Scheduled Audio setup: "
            f"expected {packed_values} values, got {x.shape[-1]}"
        )
    if x.shape[0] != 1:
        raise ValueError("Scheduled drive-audio injection currently requires batch size 1")

    source = source_audio_x0.to(device=x.device, dtype=x.dtype).reshape(1, -1)
    fixed_noise = fixed_audio_noise.to(device=x.device, dtype=x.dtype).reshape(1, -1)
    audio_values = packed_values - video_values
    if source.shape[-1] != audio_values or fixed_noise.shape[-1] != audio_values:
        raise ValueError(
            "Scheduled drive-audio latent no longer matches the sampler audio stream: "
            f"expected {audio_values}, source={source.shape[-1]}, noise={fixed_noise.shape[-1]}"
        )

    denoise_mask = extra_args.get("denoise_mask")
    audio_mask = None
    if denoise_mask is not None:
        if denoise_mask.shape[-1] != packed_values:
            raise ValueError("MiniMax H3 denoise mask does not match the packed AV latent")
        audio_mask = denoise_mask[..., video_values:]

    total_steps = len(sigmas) - 1
    s_in = x.new_ones([x.shape[0]])
    for step in comfy.utils.model_trange(total_steps, disable=disable):
        sigma_video = sigmas[step]
        sigma_video_next = sigmas[step + 1]
        sigma_audio = time_shift_sigma(sigma_video, shift_video, shift_audio)
        sigma_audio_next = time_shift_sigma(sigma_video_next, shift_video, shift_audio)
        slope_audio = time_shift_slope(sigma_video, shift_video, shift_audio)

        progress = 1.0 if total_steps <= 1 else step / (total_steps - 1)
        weight = strength * _progress_weight(progress, start_percent, end_percent, envelope)
        if weight > 0.0:
            target_audio = (
                (1.0 - sigma_audio) * source
                + sigma_audio * (float(noise_scale) * fixed_noise)
            )
            current_audio = x[..., video_values:]
            injected_audio = current_audio + weight * (target_audio - current_audio)
            x = torch.cat((x[..., :video_values], injected_audio), dim=-1)

        denoised = model(x, sigma_video * s_in, **extra_args)
        derivative = to_d(x, sigma_video, denoised)
        video_delta = sigma_video_next - sigma_video
        audio_delta = sigma_audio_next - sigma_audio
        if not audio_velocity_is_raw:
            audio_delta = audio_delta / slope_audio
        if audio_mask is not None:
            audio_delta = video_delta + audio_mask * (audio_delta - video_delta)

        if callback is not None:
            endpoint_scale = _audio_step_scale(
                sigma_video, sigma_audio, slope_audio, audio_mask, audio_velocity_is_raw
            )
            callback_denoised = denoised.clone()
            callback_denoised[..., video_values:] = (
                x[..., video_values:]
                + derivative[..., video_values:] * endpoint_scale
            )
            callback(
                {
                    "x": x,
                    "i": step,
                    "sigma": sigma_video,
                    "sigma_hat": sigma_video,
                    "denoised": callback_denoised,
                }
            )

        x = torch.cat(
            (
                x[..., :video_values] + derivative[..., :video_values] * video_delta,
                x[..., video_values:] + derivative[..., video_values:] * audio_delta,
            ),
            dim=-1,
        )

    if lock_final_audio:
        x = torch.cat((x[..., :video_values], source), dim=-1)
    return x


def setup_scheduled_drive_audio_injection(
    model,
    av_latent: dict,
    drive_audio,
    audio_vae,
    steps: int,
    shift_video: float,
    shift_audio: float,
    mode: str = "report_only",
    start_percent: float = 0.0,
    end_percent: float = 1.0,
    strength: float = 1.0,
    envelope: str = "constant",
    injection_seed: int = 0,
    lock_final_audio: bool = False,
    scheduler: str = DEFAULT_SCHEDULER_NAME,
    allow_unverified_patch_stack: bool = False,
    final_audio=None,
):
    video, template_audio, conflicts = _validate_contract(
        model,
        av_latent,
        mode,
        float(start_percent),
        float(end_percent),
        float(strength),
        envelope,
        bool(allow_unverified_patch_stack),
    )
    if scheduler not in SCHEDULER_OPTIONS:
        raise ValueError(f"Unknown scheduler {scheduler!r}")

    patched_model, sampler, sigmas = setup_dual_clock_sampling(
        model,
        av_latent,
        int(steps),
        float(shift_video),
        float(shift_audio),
        scheduler=scheduler,
    )
    mux_audio = final_audio if final_audio is not None else drive_audio
    common = {
        "mode": mode,
        "status": "bypass" if mode == "report_only" else "experimental",
        "steps": int(steps),
        "shift_video": float(shift_video),
        "shift_audio": float(shift_audio),
        "scheduler": scheduler,
        "patch_stack": conflicts,
        "bit_exact_bypass_claim": mode == "report_only",
        "scientific_limits": [
            "Injection controls the complete supplied drive-audio latent, not speech alone.",
            "It cannot selectively remove unrequested speech while preserving newly generated ambience.",
            "mux_audio is an unchanged supplied AUDIO passthrough; it does not contain model-generated sound.",
            "lock_final_audio locks the encoded VAE latent, not bit-exact source PCM.",
        ],
    }
    if mode == "report_only":
        return patched_model, sampler, sigmas, av_latent, mux_audio, _report(**common)

    encoded = encode_audio_once(audio_vae, drive_audio)
    fitted = fit_audio_latent(encoded, template_audio).detach().to("cpu", torch.float32)
    controlled_latent = replace_audio_latent(av_latent, encoded, 1.0)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(injection_seed) & ((1 << 63) - 1))
    fixed_noise = torch.randn(fitted.shape, generator=generator, device="cpu", dtype=torch.float32)
    video_values = math.prod(video.shape[1:])
    packed_values = video_values + math.prod(template_audio.shape[1:])
    sampling = patched_model.get_model_object("model_sampling")
    noise_scale = float(getattr(sampling, "noise_scale", 1.0))
    audio_velocity_is_raw = model_uses_raw_audio_velocity(model)

    def sampler_function(model_wrap, x, sigmas, extra_args=None, callback=None, disable=None):
        return sample_minimax_h3_scheduled_audio_euler(
            model_wrap,
            x,
            sigmas,
            extra_args=extra_args,
            callback=callback,
            disable=disable,
            video_values=video_values,
            packed_values=packed_values,
            shift_video=float(shift_video),
            shift_audio=float(shift_audio),
            audio_velocity_is_raw=audio_velocity_is_raw,
            source_audio_x0=fitted,
            fixed_audio_noise=fixed_noise,
            noise_scale=noise_scale,
            start_percent=float(start_percent),
            end_percent=float(end_percent),
            strength=float(strength),
            envelope=envelope,
            lock_final_audio=bool(lock_final_audio),
        )

    sampler_function.__name__ = "sample_minimax_h3_scheduled_audio_euler"
    sampler = comfy.samplers.KSAMPLER(sampler_function)
    return (
        patched_model,
        sampler,
        sigmas,
        controlled_latent,
        mux_audio,
        _report(
            **common,
            start_percent=float(start_percent),
            end_percent=float(end_percent),
            strength=float(strength),
            envelope=envelope,
            injection_seed=int(injection_seed),
            lock_final_audio=bool(lock_final_audio),
            source_audio_latent_shape=list(fitted.shape),
            encoded_once=True,
            audio_mask_forced_to_one=True,
            deterministic_fixed_noise=True,
            recommended_status="EXP_until_real_AV_AB_validation",
        ),
    )
