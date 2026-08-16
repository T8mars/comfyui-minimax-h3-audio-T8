from __future__ import annotations

import json
import math
from typing import Any

import torch

import comfy.nested_tensor
import comfy.utils


PIXEL_ALIGNMENT = 32
MAX_PIXEL_DIMENSION = 16384
SUPPORTED_PIXEL_SCALES = (8, 16)
ALIGNMENT_POLICIES = ("best_aspect", "nearest", "ceil", "floor")
UPSCALE_METHODS = ("nearest-exact", "bilinear", "area", "bicubic", "bislerp")


def _ceil_multiple(value: float, multiple: int) -> int:
    return max(multiple, math.ceil(value / multiple) * multiple)


def _floor_multiple(value: float, multiple: int) -> int:
    return max(multiple, math.floor(value / multiple) * multiple)


def _nearest_multiple(value: float, multiple: int) -> int:
    return max(multiple, math.floor((value / multiple) + 0.5) * multiple)


def aligned_upscale_geometry(
    latent_width: int,
    latent_height: int,
    scale_by: float,
    pixels_per_latent: int,
    alignment_policy: str,
) -> dict[str, float | int | str]:
    """Choose a legal pixel-space target for a latent resize.

    An arbitrary fractional scale cannot generally preserve both the exact source
    aspect ratio and a 32-pixel grid. ``best_aspect`` therefore evaluates the four
    nearest legal width/height pairs, then chooses the smallest aspect error and
    uses size error as the tie-breaker. The residual error is always reported.
    """
    latent_width = int(latent_width)
    latent_height = int(latent_height)
    pixels_per_latent = int(pixels_per_latent)
    scale_by = float(scale_by)
    if latent_width <= 0 or latent_height <= 0:
        raise ValueError("Latent width and height must be positive")
    if not math.isfinite(scale_by) or scale_by < 1.0 or scale_by > 8.0:
        raise ValueError("scale_by must be finite and within [1.0, 8.0]")
    if pixels_per_latent not in SUPPORTED_PIXEL_SCALES:
        raise ValueError(
            f"pixels_per_latent must be one of {SUPPORTED_PIXEL_SCALES}, "
            f"got {pixels_per_latent}"
        )
    if PIXEL_ALIGNMENT % pixels_per_latent:
        raise ValueError("pixels_per_latent must divide the 32-pixel alignment")
    if alignment_policy not in ALIGNMENT_POLICIES:
        raise ValueError(
            f"Unknown alignment_policy {alignment_policy!r}; "
            f"expected one of {ALIGNMENT_POLICIES}"
        )

    source_width = latent_width * pixels_per_latent
    source_height = latent_height * pixels_per_latent
    ideal_width = source_width * scale_by
    ideal_height = source_height * scale_by

    minimum_width = _ceil_multiple(source_width, PIXEL_ALIGNMENT)
    minimum_height = _ceil_multiple(source_height, PIXEL_ALIGNMENT)

    if alignment_policy == "ceil":
        output_width = max(minimum_width, _ceil_multiple(ideal_width, PIXEL_ALIGNMENT))
        output_height = max(minimum_height, _ceil_multiple(ideal_height, PIXEL_ALIGNMENT))
    elif alignment_policy == "floor":
        output_width = max(minimum_width, _floor_multiple(ideal_width, PIXEL_ALIGNMENT))
        output_height = max(minimum_height, _floor_multiple(ideal_height, PIXEL_ALIGNMENT))
    elif alignment_policy == "nearest":
        output_width = max(minimum_width, _nearest_multiple(ideal_width, PIXEL_ALIGNMENT))
        output_height = max(minimum_height, _nearest_multiple(ideal_height, PIXEL_ALIGNMENT))
    else:
        width_options = {
            max(minimum_width, _floor_multiple(ideal_width, PIXEL_ALIGNMENT)),
            max(minimum_width, _ceil_multiple(ideal_width, PIXEL_ALIGNMENT)),
        }
        height_options = {
            max(minimum_height, _floor_multiple(ideal_height, PIXEL_ALIGNMENT)),
            max(minimum_height, _ceil_multiple(ideal_height, PIXEL_ALIGNMENT)),
        }
        source_aspect = source_width / source_height

        def candidate_score(candidate: tuple[int, int]) -> tuple[float, float, float]:
            width, height = candidate
            aspect_error = abs(math.log((width / height) / source_aspect))
            normalized_size_error = math.hypot(
                (width - ideal_width) / ideal_width,
                (height - ideal_height) / ideal_height,
            )
            area_error = abs(math.log((width * height) / (ideal_width * ideal_height)))
            return aspect_error, normalized_size_error, area_error

        output_width, output_height = min(
            ((width, height) for width in width_options for height in height_options),
            key=candidate_score,
        )

    if output_width > MAX_PIXEL_DIMENSION or output_height > MAX_PIXEL_DIMENSION:
        raise ValueError(
            "Aligned output exceeds the 16384-pixel ComfyUI node limit: "
            f"{output_width}x{output_height}"
        )
    if output_width % PIXEL_ALIGNMENT or output_height % PIXEL_ALIGNMENT:
        raise AssertionError("Internal geometry error: output is not divisible by 32")

    output_latent_width = output_width // pixels_per_latent
    output_latent_height = output_height // pixels_per_latent
    source_aspect = source_width / source_height
    output_aspect = output_width / output_height
    return {
        "alignment_policy": alignment_policy,
        "pixels_per_latent": pixels_per_latent,
        "source_width": source_width,
        "source_height": source_height,
        "ideal_width": ideal_width,
        "ideal_height": ideal_height,
        "output_width": output_width,
        "output_height": output_height,
        "output_latent_width": output_latent_width,
        "output_latent_height": output_latent_height,
        "actual_scale_x": output_width / source_width,
        "actual_scale_y": output_height / source_height,
        "source_aspect_ratio": source_aspect,
        "output_aspect_ratio": output_aspect,
        "aspect_ratio_error_percent": abs((output_aspect / source_aspect) - 1.0) * 100.0,
    }


def _resize_spatial_tensor(
    tensor: torch.Tensor,
    latent_width: int,
    latent_height: int,
    method: str,
) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor) or tensor.ndim < 4:
        shape = getattr(tensor, "shape", None)
        raise ValueError(f"Expected a latent tensor with at least 4 dimensions, got {shape}")
    if tuple(tensor.shape[-2:]) == (latent_height, latent_width):
        return tensor
    return comfy.utils.common_upscale(
        tensor,
        latent_width,
        latent_height,
        method,
        "disabled",
    )


def _resize_mask_tensor(
    mask: torch.Tensor,
    source_width: int,
    source_height: int,
    output_width: int,
    output_height: int,
) -> tuple[torch.Tensor, str]:
    if not isinstance(mask, torch.Tensor) or mask.ndim < 2:
        return mask, "preserved_non_tensor_or_rank_lt_2"
    if tuple(mask.shape[-2:]) != (source_height, source_width):
        return mask, "preserved_nonmatching_spatial_shape"
    if (source_width, source_height) == (output_width, output_height):
        return mask, "preserved_noop"

    original_dtype = mask.dtype
    if mask.ndim == 2:
        work = mask.unsqueeze(0).unsqueeze(0)
        restore_shape = "hw"
    elif mask.ndim == 3:
        work = mask.unsqueeze(1)
        restore_shape = "bhw"
    else:
        work = mask
        restore_shape = "unchanged"
    if not torch.is_floating_point(work):
        work = work.to(torch.float32)
    resized = comfy.utils.common_upscale(
        work,
        output_width,
        output_height,
        "nearest-exact",
        "disabled",
    )
    if restore_shape == "hw":
        resized = resized[0, 0]
    elif restore_shape == "bhw":
        resized = resized[:, 0]
    if original_dtype == torch.bool:
        resized = resized >= 0.5
    elif resized.dtype != original_dtype:
        resized = resized.round().to(original_dtype)
    return resized, "resized_nearest_exact"


def _nested_parts(value: Any, field_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    if not getattr(value, "is_nested", False):
        raise ValueError(f"{field_name} is not a nested tensor")
    parts = tuple(value.unbind())
    if len(parts) != 2:
        raise ValueError(
            f"Expected {field_name} to contain exactly video and audio parts, got {len(parts)}"
        )
    video, audio = parts
    if not isinstance(video, torch.Tensor) or video.ndim != 5:
        raise ValueError(
            f"Expected nested H3 video latent [B,C,T,H,W], got {getattr(video, 'shape', None)}"
        )
    if not isinstance(audio, torch.Tensor) or audio.ndim != 4:
        raise ValueError(
            f"Expected nested H3 audio latent [B,C,F,T], got {getattr(audio, 'shape', None)}"
        )
    return video, audio


def upscale_latent_by_32(
    latent: dict,
    upscale_method: str,
    scale_by: float,
    pixels_per_latent: int,
    alignment_policy: str,
) -> tuple[dict, int, int, str]:
    if not isinstance(latent, dict) or "samples" not in latent:
        raise ValueError("Expected a LATENT dictionary containing 'samples'")
    if upscale_method not in UPSCALE_METHODS:
        raise ValueError(
            f"Unknown upscale_method {upscale_method!r}; expected one of {UPSCALE_METHODS}"
        )

    samples = latent["samples"]
    is_joint_av = bool(getattr(samples, "is_nested", False))
    if is_joint_av:
        video, audio = _nested_parts(samples, "samples")
        spatial_source = video
        if int(pixels_per_latent) != 16:
            raise ValueError(
                "Nested MiniMax H3 AV latent requires pixels_per_latent=16; "
                "the 8-pixel option is only for plain SD/SDXL-style latents"
            )
    else:
        if not isinstance(samples, torch.Tensor) or samples.ndim < 4:
            raise ValueError(
                "Expected plain LATENT samples with at least 4 dimensions or a nested H3 AV latent"
            )
        spatial_source = samples
        audio = None

    source_latent_height = int(spatial_source.shape[-2])
    source_latent_width = int(spatial_source.shape[-1])
    geometry = aligned_upscale_geometry(
        source_latent_width,
        source_latent_height,
        scale_by,
        pixels_per_latent,
        alignment_policy,
    )
    output_latent_width = int(geometry["output_latent_width"])
    output_latent_height = int(geometry["output_latent_height"])

    resized_video = _resize_spatial_tensor(
        spatial_source,
        output_latent_width,
        output_latent_height,
        upscale_method,
    )
    output = latent.copy()
    if is_joint_av:
        if resized_video is spatial_source:
            output["samples"] = samples
        else:
            output["samples"] = comfy.nested_tensor.NestedTensor((resized_video, audio))
    else:
        output["samples"] = resized_video

    mask_status = "absent"
    if "noise_mask" in latent and latent["noise_mask"] is not None:
        mask = latent["noise_mask"]
        if is_joint_av:
            if not getattr(mask, "is_nested", False):
                raise ValueError(
                    "A nested H3 AV latent with noise_mask requires a nested video/audio noise_mask"
                )
            video_mask, audio_mask = _nested_parts(mask, "noise_mask")
            resized_video_mask, mask_status = _resize_mask_tensor(
                video_mask,
                source_latent_width,
                source_latent_height,
                output_latent_width,
                output_latent_height,
            )
            if resized_video_mask is video_mask:
                output["noise_mask"] = mask
            else:
                output["noise_mask"] = comfy.nested_tensor.NestedTensor(
                    (resized_video_mask, audio_mask)
                )
        else:
            if getattr(mask, "is_nested", False):
                raise ValueError("Plain LATENT samples cannot be paired with a nested noise_mask")
            output["noise_mask"], mask_status = _resize_mask_tensor(
                mask,
                source_latent_width,
                source_latent_height,
                output_latent_width,
                output_latent_height,
            )

    report = {
        "schema_version": 1,
        "node": "MiniMaxH3LatentUpscaleBy32T8",
        "status": "ok",
        "latent_kind": "minimax_h3_joint_av" if is_joint_av else "plain",
        "upscale_method": upscale_method,
        "requested_scale_by": float(scale_by),
        "alignment_pixels": PIXEL_ALIGNMENT,
        "geometry": geometry,
        "noise_mask": mask_status,
        "audio_latent_preserved": is_joint_av,
        "metadata_keys_preserved": sorted(key for key in latent if key not in {"samples", "noise_mask"}),
        "notes": [
            "Both output pixel dimensions are exactly divisible by 32.",
            "An arbitrary fractional scale cannot always preserve an exact aspect ratio on a 32-pixel grid; inspect aspect_ratio_error_percent.",
            "For nested H3 AV latent, only the video spatial axes are resized; audio values and timing are preserved.",
        ],
    }
    return (
        output,
        int(geometry["output_width"]),
        int(geometry["output_height"]),
        json.dumps(report, ensure_ascii=False, sort_keys=True),
    )
