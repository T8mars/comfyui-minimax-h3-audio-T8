from __future__ import annotations

import inspect
import json
from typing import Any, Mapping

import torch

import comfy.nested_tensor

from .core import AUDIO_LATENT_FPS, FPS, nested_av_parts, split_noise_masks
from .long_video import (
    CONTEXT_FRAME_STEPS,
    LONG_VIDEO_SCHEMA,
    pixel_frames_from_latent_t,
)


NATIVE_MASKED_CONTEXT_SCHEMA = "t8.native_masked_video_context.v1"


def _has_parameters(function, *names: str) -> bool:
    try:
        parameters = inspect.signature(function).parameters
    except (TypeError, ValueError):
        return False
    return all(name in parameters for name in names)


def require_native_h3_av_mask_support() -> dict[str, Any]:
    """Require current ComfyUI's native MiniMax H3 video/audio mask contract."""
    import comfy.ldm.minimax.model as minimax_model
    import comfy.model_base as model_base

    base = getattr(model_base, "MiniMaxH3", None)
    diffusion = getattr(minimax_model, "MiniMaxH3Model", None)
    missing: list[str] = []
    if base is None:
        missing.append("comfy.model_base.MiniMaxH3")
    else:
        for name in ("_token_grid_masks", "_denoise_mask_conds"):
            if not callable(getattr(base, name, None)):
                missing.append(f"MiniMaxH3.{name}")
        scale = getattr(base, "scale_latent_inpaint", None)
        if not callable(scale) or not _has_parameters(scale, "x", "denoise_mask"):
            missing.append("MiniMaxH3.scale_latent_inpaint(x, denoise_mask)")

    if not callable(getattr(minimax_model, "mask_row_values", None)):
        missing.append("minimax.model.mask_row_values")
    forward = getattr(diffusion, "forward", None) if diffusion is not None else None
    if not callable(forward) or not _has_parameters(
        forward, "denoise_mask", "audio_denoise_mask"
    ):
        missing.append("MiniMaxH3Model.forward video/audio masks")
    if missing:
        raise RuntimeError(
            "Native Masked Video Context requires current ComfyUI MiniMax H3 AV-mask "
            "support. Missing: " + ", ".join(missing)
        )
    return {
        "supported": True,
        "runtime_patch_installed": False,
        "backend": "comfyui_native_minimax_h3_av_mask",
    }


def _report_object(value: str, *, label: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} must be valid JSON") from error
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return parsed


def _mask_range(mask: torch.Tensor, *, label: str) -> tuple[float, float]:
    if mask.dtype == torch.bool:
        return 0.0, 1.0
    if not mask.dtype.is_floating_point:
        raise ValueError(f"{label} must use a floating or boolean dtype")
    if not bool(torch.isfinite(mask).all()):
        raise ValueError(f"{label} contains NaN or Inf")
    minimum = float(mask.amin().item())
    maximum = float(mask.amax().item())
    if minimum < 0.0 or maximum > 1.0:
        raise ValueError(f"{label} values must stay within [0,1]")
    return minimum, maximum


def _native_video_mask(
    mask: torch.Tensor | None,
    video: torch.Tensor,
) -> torch.Tensor | None:
    if mask is None:
        return None
    if not isinstance(mask, torch.Tensor) or mask.ndim != 5:
        raise ValueError(
            "Native Masked Video Context requires a native latent-aligned video mask"
        )
    if (
        int(mask.shape[0]) != 1
        or int(mask.shape[1]) not in {1, int(video.shape[1])}
        or tuple(mask.shape[2:]) != tuple(video.shape[2:])
    ):
        raise ValueError(
            "Native Masked Video Context requires a native latent-aligned video mask "
            "with matching T/H/W and one or 24 channels"
        )
    _mask_range(mask, label="existing video noise_mask")
    return mask


def _native_audio_mask(
    mask: torch.Tensor | None,
    audio: torch.Tensor,
) -> torch.Tensor | None:
    if mask is None:
        return None
    if not isinstance(mask, torch.Tensor) or mask.ndim != 4:
        raise ValueError(
            "Native Masked Video Context requires a native latent-aligned audio mask"
        )
    if (
        int(mask.shape[0]) != 1
        or int(mask.shape[1]) not in {1, int(audio.shape[1])}
        or tuple(mask.shape[2:]) != tuple(audio.shape[2:])
    ):
        raise ValueError(
            "Native Masked Video Context requires a native latent-aligned audio mask "
            "with matching stereo/T and one or 32 channels"
        )
    _mask_range(mask, label="existing audio noise_mask")
    return mask


def _validated_context(
    context: Mapping[str, Any],
    *,
    chain_id: str,
    segment_index: int,
    context_frames: int,
    target_video: torch.Tensor,
) -> torch.Tensor:
    if not isinstance(context, Mapping) or int(context.get("schema", -1)) != LONG_VIDEO_SCHEMA:
        raise ValueError("Connect a validated H3 T8 Previous Context output")
    if bool(context.get("empty", False)):
        raise ValueError("Native Masked Video Context only runs on continuation segments")
    metadata = context.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("Previous Context metadata is missing")
    if str(metadata.get("chain_id", "")) != chain_id:
        raise ValueError("Previous Context chain_id does not match the Planner report")
    if int(metadata.get("target_segment_index", -1)) != segment_index:
        raise ValueError("Previous Context target segment does not match the Planner report")
    if int(metadata.get("source_segment_index", -2)) != segment_index - 1:
        raise ValueError("Previous Context is not from the immediately previous segment")
    maximum = int(metadata.get("max_context_frames", 0))
    if maximum not in CONTEXT_FRAME_STEPS or maximum < context_frames:
        raise ValueError("Previous Context does not contain the requested context_frames")
    video_tail = context.get("video_tail")
    if not isinstance(video_tail, torch.Tensor) or video_tail.ndim != 5:
        raise ValueError("Previous Context video tail must be [B,C,T,H,W]")
    if int(video_tail.shape[0]) != 1 or int(video_tail.shape[1]) != 24:
        raise ValueError("Previous Context video tail must use batch 1 and 24 channels")
    required_steps = CONTEXT_FRAME_STEPS[context_frames]
    if int(video_tail.shape[2]) < required_steps:
        raise ValueError("Previous Context video tail is shorter than context_frames")
    if tuple(video_tail.shape[-2:]) != tuple(target_video.shape[-2:]):
        raise ValueError("Native masked continuation requires the same latent canvas")
    if not video_tail.dtype.is_floating_point or not bool(torch.isfinite(video_tail).all()):
        raise ValueError("Previous Context video tail must contain finite floating values")
    return video_tail[:, :, -required_steps:].contiguous()


def apply_native_masked_video_context(
    av_latent: dict,
    context: Mapping[str, Any],
    planner_report_json: str,
    conditioning_report_json: str,
) -> tuple[dict, int, str]:
    """Overlay a validated previous video-latent tail without touching target audio."""
    capability = require_native_h3_av_mask_support()
    planner = _report_object(planner_report_json, label="planner_report_json")
    conditioning = _report_object(
        conditioning_report_json, label="conditioning_report_json"
    )

    chain_id = str(planner.get("chain_id", ""))
    segment_index = int(planner.get("segment_index", -1))
    if segment_index <= 0:
        raise ValueError("Native Masked Video Context only runs on continuation segments")
    context_frames = int(planner.get("context_frames", -1))
    if context_frames not in CONTEXT_FRAME_STEPS:
        raise ValueError("Planner context_frames must be 5, 22, or 39")
    if int(conditioning.get("schema", -1)) != LONG_VIDEO_SCHEMA:
        raise ValueError("Conditioning report schema is not the H3 T8 Long Video contract")
    if int(conditioning.get("segment_index", -1)) != segment_index:
        raise ValueError("Planner and Conditioning segment_index values do not match")
    if int(conditioning.get("context_frames", -1)) != context_frames:
        raise ValueError("Planner and Conditioning context_frames values do not match")
    if not bool(conditioning.get("context_active", False)):
        raise ValueError("Native Masked Video Context requires active previous context")
    if conditioning.get("context_audio") != "video_only":
        raise ValueError(
            "Native Masked Video Context requires context_audio=video_only so target audio "
            "and Vocal Lock masks remain untouched"
        )
    if bool(conditioning.get("timeline_audio_ref", False)):
        raise ValueError("video-only masked context must not contain a timeline audio reference")
    required_steps = CONTEXT_FRAME_STEPS[context_frames]
    if int(conditioning.get("motion_keyframes", -1)) != required_steps:
        raise ValueError("Conditioning motion_keyframes do not match context_frames")

    video, audio = nested_av_parts(av_latent)
    target_frames = pixel_frames_from_latent_t(int(video.shape[2]))
    render_frames = int(planner.get("render_frames", -1))
    if render_frames != target_frames or int(conditioning.get("render_frames", -1)) != target_frames:
        raise ValueError("Planner, Conditioning and target render_frames do not match")
    if context_frames >= target_frames or required_steps >= int(video.shape[2]):
        raise ValueError("Masked context must leave a non-empty generated video region")
    expected_audio_t = round(target_frames / FPS * AUDIO_LATENT_FPS)
    if int(audio.shape[-1]) != expected_audio_t:
        raise ValueError(
            f"Target audio has {audio.shape[-1]} ticks for {target_frames} frames; "
            f"expected {expected_audio_t}"
        )
    trim_start = float(planner.get("trim_start_seconds", -1.0))
    if abs(trim_start * FPS - context_frames) > 1e-6:
        raise ValueError("Planner trim_start_seconds does not match context_frames")

    context_video = _validated_context(
        context,
        chain_id=chain_id,
        segment_index=segment_index,
        context_frames=context_frames,
        target_video=video,
    )
    existing_video_mask, existing_audio_mask = split_noise_masks(av_latent, video, audio)
    existing_video_mask = _native_video_mask(existing_video_mask, video)
    existing_audio_mask = _native_audio_mask(existing_audio_mask, audio)
    if existing_video_mask is not None:
        locked_prefix = existing_video_mask[..., :required_steps, :, :]
        if not bool((locked_prefix >= 1.0 - 1e-6).all()):
            raise ValueError(
                "Target prefix already contains locked or partial video mask values; "
                "masked context refuses to overwrite another visual owner"
            )

    output_video = video.clone()
    output_video[:, :, :required_steps] = context_video.to(
        device=output_video.device,
        dtype=output_video.dtype,
    )
    if existing_video_mask is None:
        output_video_mask = torch.ones(
            (1, 1, *video.shape[2:]),
            device=video.device,
            dtype=torch.float32,
        )
        video_mask_policy = "created_native_latent_mask"
    else:
        output_video_mask = existing_video_mask.clone()
        video_mask_policy = "preserved_existing_outside_context"
    output_video_mask[..., :required_steps, :, :] = 0

    if existing_audio_mask is None:
        output_audio_mask = torch.ones(
            (1, 1, *audio.shape[2:]),
            device=audio.device,
            dtype=torch.float32,
        )
        audio_mask_policy = "created_all_generate_equivalent"
        audio_mask_reused = False
    else:
        output_audio_mask = existing_audio_mask
        audio_mask_policy = "reused_exact_existing_object"
        audio_mask_reused = True

    output = av_latent.copy()
    output["samples"] = comfy.nested_tensor.NestedTensor((output_video, audio))
    output["noise_mask"] = comfy.nested_tensor.NestedTensor(
        (output_video_mask, output_audio_mask)
    )
    output["t8_native_masked_video_context"] = {
        "schema": NATIVE_MASKED_CONTEXT_SCHEMA,
        "chain_id": chain_id,
        "segment_index": segment_index,
        "context_frames": context_frames,
        "context_video_latent_steps": required_steps,
        "audio_policy": "target_audio_untouched",
    }
    video_mask_min, video_mask_max = _mask_range(
        output_video_mask, label="output video noise_mask"
    )
    audio_mask_min, audio_mask_max = _mask_range(
        output_audio_mask, label="output audio noise_mask"
    )
    report = {
        "schema": NATIVE_MASKED_CONTEXT_SCHEMA,
        "status": "VIDEO_CONTEXT_APPLIED_AUDIO_UNTOUCHED",
        "chain_id": chain_id,
        "segment_index": segment_index,
        "context_frames": context_frames,
        "context_video_latent_steps": required_steps,
        "render_frames": target_frames,
        "generated_visible_frames_after_trim": target_frames - context_frames,
        "trim_start_seconds": trim_start,
        "video_mask_policy": video_mask_policy,
        "video_noise_mask_range": [video_mask_min, video_mask_max],
        "audio_samples_reused": tuple(output["samples"].unbind())[1] is audio,
        "audio_noise_mask_reused": audio_mask_reused,
        "audio_noise_mask_policy": audio_mask_policy,
        "audio_noise_mask_range": [audio_mask_min, audio_mask_max],
        "audio_context_from_previous_segment": False,
        "source_video_vae_roundtrip": False,
        "runtime_patch_installed": capability["runtime_patch_installed"],
        "quality_validated": False,
        "plan_b_experimental": True,
    }
    return output, context_frames, json.dumps(report, ensure_ascii=False, indent=2)
