from __future__ import annotations

import json
import math
from collections.abc import Mapping

import torch

import comfy.nested_tensor
import comfy.utils

from .core import (
    AUDIO_LATENT_FPS,
    classify_h3_vae,
    encode_audio_once,
    nested_av_parts,
    split_noise_masks,
)


TIMED_AUDIO_BED_SCHEMA = "minimax_h3_t8_timed_audio_bed_lock_v1"
AUDIO_LATENT_FIT_POLICIES = ("strict", "fit_reported")


def _fit_encoded_bed(
    encoded: torch.Tensor,
    template: torch.Tensor,
    policy: str,
) -> tuple[torch.Tensor, str]:
    if policy not in AUDIO_LATENT_FIT_POLICIES:
        raise ValueError(f"unknown audio latent fit policy: {policy}")
    if encoded.ndim != 4:
        raise ValueError(
            "The audio VAE must return [B,C,stereo,T] latent data; "
            f"got {tuple(encoded.shape)}"
        )
    if encoded.shape[:-1] != template.shape[:-1]:
        raise ValueError(
            "Background-bed latent layout does not match the H3 AV audio stream: "
            f"encoded={tuple(encoded.shape)}, target={tuple(template.shape)}"
        )
    current_t = int(encoded.shape[-1])
    target_t = int(template.shape[-1])
    if current_t == target_t:
        return encoded.to(device=template.device, dtype=template.dtype), "exact"
    if policy == "strict":
        raise ValueError(
            "Background audio encoded to a different H3 latent duration: "
            f"encoded T={current_t}, AV target T={target_t}. Supply an exact-duration bed "
            "or explicitly choose fit_reported."
        )
    if current_t > target_t:
        fitted = encoded[..., :target_t]
        action = f"trimmed_{current_t - target_t}_latent_steps"
    else:
        padding = encoded.new_zeros((*encoded.shape[:-1], target_t - current_t))
        fitted = torch.cat((encoded, padding), dim=-1)
        action = f"zero_padded_{target_t - current_t}_latent_steps"
    return fitted.to(device=template.device, dtype=template.dtype), action


def _mask_like(mask, samples: torch.Tensor) -> torch.Tensor:
    if mask is None:
        return torch.ones_like(samples)
    if not isinstance(mask, torch.Tensor):
        raise ValueError("H3 noise masks must be torch.Tensor values")
    mask = mask.to(device=samples.device, dtype=samples.dtype)
    if tuple(mask.shape) != tuple(samples.shape):
        mask = comfy.utils.reshape_mask(mask, samples.shape)
    return mask.clamp(0.0, 1.0)


def build_timed_audio_bed_lock(
    av_latent: Mapping,
    background_audio: Mapping,
    audio_vae,
    tail_lock_start_seconds: float,
    head_denoise_strength: float = 1.0,
    tail_denoise_strength: float = 0.0,
    transition_seconds: float = 0.0,
    audio_latent_fit_policy: str = "strict",
):
    if not isinstance(av_latent, Mapping):
        raise ValueError("av_latent must be a connected H3 joint AV LATENT value")
    if audio_vae is None:
        raise ValueError("audio_vae must be connected")
    if classify_h3_vae(audio_vae) != "audio":
        raise ValueError("audio_vae must be the MiniMax H3 audio VAE, not the video VAE")
    numeric = {
        "tail_lock_start_seconds": tail_lock_start_seconds,
        "head_denoise_strength": head_denoise_strength,
        "tail_denoise_strength": tail_denoise_strength,
        "transition_seconds": transition_seconds,
    }
    for name, value in numeric.items():
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    if tail_lock_start_seconds < 0.0:
        raise ValueError("tail_lock_start_seconds must be nonnegative")
    if transition_seconds < 0.0 or transition_seconds > 10.0:
        raise ValueError("transition_seconds must be between 0 and 10")
    if not 0.0 <= head_denoise_strength <= 1.0:
        raise ValueError("head_denoise_strength must be between 0 and 1")
    if not 0.0 <= tail_denoise_strength <= 1.0:
        raise ValueError("tail_denoise_strength must be between 0 and 1")
    if tail_denoise_strength > head_denoise_strength:
        raise ValueError(
            "tail_denoise_strength cannot exceed head_denoise_strength; this node must not "
            "silently increase generation freedom after the dialogue boundary"
        )

    video, template_audio = nested_av_parts(dict(av_latent))
    encoded = encode_audio_once(audio_vae, background_audio)
    fitted, fit_action = _fit_encoded_bed(
        encoded,
        template_audio,
        audio_latent_fit_policy,
    )
    target_t = int(fitted.shape[-1])
    requested_start_step = math.ceil(float(tail_lock_start_seconds) * AUDIO_LATENT_FPS)
    if requested_start_step >= target_t:
        raise ValueError(
            "tail_lock_start_seconds is outside the H3 audio latent timeline: "
            f"requested step {requested_start_step}, available T={target_t}"
        )
    start_step = max(0, requested_start_step)
    transition_steps = round(float(transition_seconds) * AUDIO_LATENT_FPS)
    transition_end = min(target_t, start_step + transition_steps)

    desired_1d = fitted.new_full((target_t,), float(tail_denoise_strength))
    if start_step:
        desired_1d[:start_step] = float(head_denoise_strength)
    if transition_end > start_step:
        # A transition begins at the explicit boundary and reaches the tail
        # strength later. Default zero avoids an implicit softened interval.
        desired_1d[start_step:transition_end] = torch.linspace(
            float(head_denoise_strength),
            float(tail_denoise_strength),
            transition_end - start_step + 1,
            device=fitted.device,
            dtype=fitted.dtype,
        )[:-1]
    desired_mask = desired_1d.reshape(1, 1, 1, -1).expand_as(fitted)

    video_mask, input_audio_mask = split_noise_masks(
        dict(av_latent),
        video,
        template_audio,
    )
    video_mask = _mask_like(video_mask, video)
    existing_audio_cap = _mask_like(input_audio_mask, fitted)
    effective_audio_mask = torch.minimum(existing_audio_cap, desired_mask)

    output = dict(av_latent)
    output["samples"] = comfy.nested_tensor.NestedTensor((video, fitted))
    output["noise_mask"] = comfy.nested_tensor.NestedTensor(
        (video_mask, effective_audio_mask)
    )
    audio_output = {
        key: value
        for key, value in output.items()
        if key not in {"samples", "noise_mask"}
    }
    audio_output["samples"] = fitted
    audio_output["noise_mask"] = effective_audio_mask

    fully_locked_start_step = transition_end if transition_steps else start_step
    report = {
        "schema": TIMED_AUDIO_BED_SCHEMA,
        "status": "experimental_timed_background_bed_ready",
        "facts": {
            "audio_latent_fps": AUDIO_LATENT_FPS,
            "audio_latent_t": target_t,
            "timeline_seconds": target_t / AUDIO_LATENT_FPS,
            "requested_tail_lock_start_seconds": float(tail_lock_start_seconds),
            "requested_start_step_ceil": requested_start_step,
            "actual_boundary_seconds_on_40hz_grid": start_step / AUDIO_LATENT_FPS,
            "transition_seconds_requested": float(transition_seconds),
            "transition_steps": transition_steps,
            "fully_locked_start_step": fully_locked_start_step,
            "fully_locked_start_seconds_on_40hz_grid": (
                fully_locked_start_step / AUDIO_LATENT_FPS
            ),
            "head_denoise_strength": float(head_denoise_strength),
            "tail_denoise_strength": float(tail_denoise_strength),
            "audio_latent_fit_policy": audio_latent_fit_policy,
            "audio_latent_fit_action": fit_action,
            "existing_audio_mask_was_cap": input_audio_mask is not None,
            "video_mask_preserved_as_constraint": True,
        },
        "warnings": [
            "This requires an independent full-duration background bed; it does not separate "
            "dialogue from a mixed H3 master.",
            "The boundary is quantized to the 40Hz H3 audio-latent grid and is not a "
            "sample-exact decoded speech endpoint.",
            "A nonzero transition delays the fully locked tail and may permit changes inside "
            "that transition interval.",
        ],
        "claims": {
            "source_separation_performed": False,
            "decoded_tail_quality_verified": False,
            "mouth_stop_guaranteed": False,
            "denoise_strength_is_calibrated_linear_weight": False,
            "locked_tail_latent_endpoint_requested": tail_denoise_strength == 0.0,
            "video_samples_modified": False,
            "memory_safe": False,
        },
    }
    return output, audio_output, json.dumps(report, ensure_ascii=False, indent=2)
