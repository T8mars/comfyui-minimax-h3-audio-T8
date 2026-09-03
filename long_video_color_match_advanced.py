from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import kornia
from safetensors import safe_open
from safetensors.torch import save_file
import torch
import torch.nn.functional as torch_functional

from .long_video import LONG_VIDEO_SCHEMA, context_state_path, sanitize_chain_id


COLOR_MATCH_SCHEMA = "t8.minimax_h3.long_video_color_match.v2"
COLOR_STATE_SCHEMA = 2
SPATIAL_GRID_HEIGHT = 5
SPATIAL_GRID_WIDTH = 8
LAB_SCALE_MINIMUM = 0.85
LAB_SCALE_MAXIMUM = 1.18
MOMENT_EPSILON = 1e-6


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)


def _tensor_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _rgb_to_lab(rgb: torch.Tensor) -> torch.Tensor:
    return kornia.color.rgb_to_lab(rgb.permute(0, 3, 1, 2).contiguous()).permute(
        0, 2, 3, 1
    )


def _lab_to_rgb(lab: torch.Tensor) -> torch.Tensor:
    return kornia.color.lab_to_rgb(lab.permute(0, 3, 1, 2).contiguous()).permute(
        0, 2, 3, 1
    )


def _frame_statistics(frames: torch.Tensor) -> dict[str, torch.Tensor]:
    """Calculate small pooled statistics without materializing a full Lab video batch."""
    frame_count = int(frames.shape[0])
    rgb_means: list[torch.Tensor] = []
    rgb_thumbnail_sum = torch.zeros(
        (3, SPATIAL_GRID_HEIGHT, SPATIAL_GRID_WIDTH),
        device=frames.device,
        dtype=torch.float32,
    )
    lab_mean = torch.zeros(3, device=frames.device, dtype=torch.float32)
    lab_m2 = torch.zeros(3, device=frames.device, dtype=torch.float32)
    pixel_count = 0
    for frame in frames:
        rgb = frame[None, ..., :3].float().clamp(0.0, 1.0)
        rgb_means.append(rgb.mean(dim=(0, 1, 2)))
        rgb_thumbnail_sum += torch_functional.adaptive_avg_pool2d(
            rgb.permute(0, 3, 1, 2),
            (SPATIAL_GRID_HEIGHT, SPATIAL_GRID_WIDTH),
        )[0]
        lab = _rgb_to_lab(rgb)
        frame_pixels = int(lab.shape[1] * lab.shape[2])
        frame_mean = lab.mean(dim=(0, 1, 2))
        frame_variance = lab.var(dim=(0, 1, 2), correction=0)
        if pixel_count == 0:
            lab_mean = frame_mean
            lab_m2 = frame_variance * frame_pixels
        else:
            combined_pixels = pixel_count + frame_pixels
            delta = frame_mean - lab_mean
            lab_mean = lab_mean + delta * (frame_pixels / combined_pixels)
            lab_m2 = (
                lab_m2
                + frame_variance * frame_pixels
                + delta.square() * (pixel_count * frame_pixels / combined_pixels)
            )
        pixel_count += frame_pixels
    lab_variance = (lab_m2 / pixel_count).clamp_min(0.0)
    return {
        "rgb_means": torch.stack(rgb_means).float(),
        "rgb_thumbnail": (rgb_thumbnail_sum / frame_count).float(),
        "lab_mean": lab_mean,
        "lab_std": lab_variance.sqrt(),
    }


def color_state_path(chain_id: str, source_segment_index: int) -> Path:
    context_path = context_state_path(chain_id, source_segment_index)
    return context_path.with_name(
        f"segment_{int(source_segment_index):05d}.color.safetensors"
    )


def _validate_context_binding(
    context: Mapping[str, Any],
    chain_id: str,
    segment_index: int,
) -> None:
    if not isinstance(context, Mapping) or int(context.get("schema", -1)) != LONG_VIDEO_SCHEMA:
        raise ValueError("Connect the matching H3 T8 Long Video Context Load output")
    if segment_index == 0:
        if not bool(context.get("empty")):
            raise ValueError("segment 0 Color Match requires an empty segment-0 context")
        if sanitize_chain_id(context.get("chain_id", "")) != chain_id:
            raise ValueError("Color Match context chain_id does not match the planner")
        if int(context.get("target_segment_index", -1)) != 0:
            raise ValueError("Color Match context target segment does not match segment 0")
        return
    if bool(context.get("empty")):
        raise ValueError("Continuation Color Match requires the previous long-video context")
    metadata = context.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("Continuation Color Match context metadata is missing")
    checks = {
        "chain_id": str(metadata.get("chain_id", "")) == chain_id,
        "source_segment_index": int(metadata.get("source_segment_index", -1))
        == segment_index - 1,
        "target_segment_index": int(metadata.get("target_segment_index", -1))
        == segment_index,
    }
    if not all(checks.values()):
        failed = ", ".join(name for name, passed in checks.items() if not passed)
        raise ValueError(f"Color Match context/planner binding failed: {failed}")


def _load_reference(
    path: Path,
    *,
    chain_id: str,
    source_segment_index: int,
    width: int,
    height: int,
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(
            "Long-video Color Match needs the preceding segment RGB state. "
            "Run segment 0 and every continuation in order with the Color Match node connected. "
            f"Missing: {path}"
        )
    expected_keys = {
        "tail_rgb_means",
        "tail_rgb_thumbnail",
        "tail_lab_mean",
        "tail_lab_std",
    }
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
        keys = set(handle.keys())
        if keys != expected_keys:
            raise ValueError(f"Invalid Color Match state tensor keys: {sorted(keys)}")
        tensors = {key: handle.get_tensor(key) for key in expected_keys}
    required = {
        "schema",
        "chain_id",
        "source_segment_index",
        "width",
        "height",
        "tail_frame_count",
        *(f"{key}_sha256" for key in expected_keys),
    }
    missing = sorted(required - set(metadata))
    if missing:
        raise ValueError(f"Color Match state metadata is incomplete: {', '.join(missing)}")
    if int(metadata["schema"]) != COLOR_STATE_SCHEMA:
        raise ValueError("Unsupported Color Match state schema")
    if metadata["chain_id"] != chain_id:
        raise ValueError("Color Match state chain_id does not match the current chain")
    if int(metadata["source_segment_index"]) != source_segment_index:
        raise ValueError("Color Match state segment index does not match the requested predecessor")
    if int(metadata["width"]) != width or int(metadata["height"]) != height:
        raise ValueError("Color Match state canvas does not match the current segment")
    means = tensors["tail_rgb_means"]
    if means.ndim != 2 or means.shape[1] != 3 or int(means.shape[0]) < 1:
        raise ValueError("Color Match tail_rgb_means must have shape [N,3]")
    if int(metadata["tail_frame_count"]) != int(means.shape[0]):
        raise ValueError("Color Match tail frame-count metadata does not match its tensor")
    expected_shapes = {
        "tail_rgb_thumbnail": (3, SPATIAL_GRID_HEIGHT, SPATIAL_GRID_WIDTH),
        "tail_lab_mean": (3,),
        "tail_lab_std": (3,),
    }
    for key, shape in expected_shapes.items():
        if tuple(tensors[key].shape) != shape:
            raise ValueError(f"Color Match {key} must have shape {list(shape)}")
    for key, tensor in tensors.items():
        if metadata[f"{key}_sha256"] != _tensor_sha256(tensor):
            raise ValueError(f"Color Match {key} checksum failed")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"Color Match {key} contains non-finite data")
    if float(means.min()) < 0.0 or float(means.max()) > 1.0:
        raise ValueError("Color Match tail RGB state is not SDR data in [0,1]")
    thumbnail = tensors["tail_rgb_thumbnail"]
    if float(thumbnail.min()) < 0.0 or float(thumbnail.max()) > 1.0:
        raise ValueError("Color Match RGB thumbnail is not SDR data in [0,1]")
    return {key: tensor.float().contiguous() for key, tensor in tensors.items()}, metadata


def _save_reference(
    path: Path,
    *,
    frames: torch.Tensor,
    chain_id: str,
    source_segment_index: int,
    reference_frames: int,
    report: Mapping[str, Any],
) -> None:
    count = min(int(reference_frames), int(frames.shape[0]))
    statistics = {
        f"tail_{key}": value.detach().cpu().contiguous()
        for key, value in _frame_statistics(frames[-count:]).items()
    }
    metadata = {
        "schema": str(COLOR_STATE_SCHEMA),
        "chain_id": chain_id,
        "source_segment_index": str(source_segment_index),
        "width": str(int(frames.shape[2])),
        "height": str(int(frames.shape[1])),
        "tail_frame_count": str(count),
        "report_json": _json(report),
        **{f"{key}_sha256": _tensor_sha256(value) for key, value in statistics.items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        save_file(statistics, str(temporary), metadata=metadata)
        with open(temporary, "r+b") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _fade_weights(
    frame_count: int,
    reference_frames: int,
    transition_frames: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    count = min(frame_count, transition_frames)
    weights = torch.zeros(frame_count, device=device, dtype=torch.float32)
    held = min(count, reference_frames)
    weights[:held] = 1.0
    remaining = count - held
    if remaining > 0:
        weights[held:count] = torch.linspace(
            1.0 - 1.0 / (remaining + 1),
            1.0 / (remaining + 1),
            remaining,
            device=device,
            dtype=torch.float32,
        )
    return weights


def _global_lab_transform(
    rgb: torch.Tensor,
    scale: torch.Tensor,
    offset: torch.Tensor,
) -> torch.Tensor:
    lab = _rgb_to_lab(rgb.float().clamp(0.0, 1.0))
    transformed = lab * scale.view(1, 1, 1, 3) + offset.view(1, 1, 1, 3)
    return _lab_to_rgb(transformed).clamp(0.0, 1.0)


def _global_transform_thumbnail(
    frames: torch.Tensor,
    scale: torch.Tensor,
    offset: torch.Tensor,
) -> torch.Tensor:
    thumbnail_sum = torch.zeros(
        (3, SPATIAL_GRID_HEIGHT, SPATIAL_GRID_WIDTH),
        device=frames.device,
        dtype=torch.float32,
    )
    for frame in frames:
        matched = _global_lab_transform(frame[None].float(), scale, offset)
        thumbnail_sum += torch_functional.adaptive_avg_pool2d(
            matched.permute(0, 3, 1, 2),
            (SPATIAL_GRID_HEIGHT, SPATIAL_GRID_WIDTH),
        )[0]
    return thumbnail_sum / int(frames.shape[0])


def _vector(tensor: torch.Tensor) -> list[float]:
    return [float(value) for value in tensor.detach().cpu().flatten()]


def _safe_std_ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    return numerator / denominator.clamp_min(MOMENT_EPSILON)


def process_long_video_color_match(
    frames: torch.Tensor,
    context: Mapping[str, Any],
    chain_id: str,
    segment_index: int,
    enabled: bool = True,
    reference_frames: int = 5,
    transition_frames: int = 24,
    strength: float = 1.0,
    minimum_jump: float = 0.0005,
    maximum_offset: float = 0.02,
    scene_cut_threshold: float = 0.18,
) -> tuple[torch.Tensor, str, str]:
    if not isinstance(frames, torch.Tensor) or frames.ndim != 4 or frames.shape[-1] < 3:
        raise ValueError("frames must be an IMAGE batch [T,H,W,C>=3]")
    if int(frames.shape[0]) < 1:
        raise ValueError("Color Match requires at least one output frame")
    safe_chain = sanitize_chain_id(chain_id)
    segment_index = int(segment_index)
    if segment_index < 0:
        raise ValueError("segment_index cannot be negative")
    reference_frames = int(reference_frames)
    transition_frames = int(transition_frames)
    if not 1 <= reference_frames <= 24:
        raise ValueError("reference_frames must be 1..24")
    if not 1 <= transition_frames <= 240:
        raise ValueError("transition_frames must be 1..240")
    values = (strength, minimum_jump, maximum_offset, scene_cut_threshold)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("Color Match controls must be finite")
    if not 0.0 <= float(strength) <= 2.0:
        raise ValueError("strength must be between 0 and 2")
    if min(float(minimum_jump), float(maximum_offset), float(scene_cut_threshold)) < 0.0:
        raise ValueError("Color Match thresholds cannot be negative")
    if float(minimum_jump) >= float(scene_cut_threshold):
        raise ValueError("minimum_jump must be smaller than scene_cut_threshold")
    _validate_context_binding(context, safe_chain, segment_index)

    finite = bool(torch.isfinite(frames).all())
    in_range = finite and float(frames.min()) >= 0.0 and float(frames.max()) <= 1.0
    if not finite or not in_range:
        raise ValueError("Color Match only accepts finite SDR IMAGE data in [0,1]")

    output = frames
    status = "REFERENCE_INITIALIZED"
    applied = False
    compare_count = 0
    jump_vector = torch.zeros(3, dtype=torch.float32)
    effective_rgb_offset = torch.zeros(3, dtype=torch.float32)
    lab_scale = torch.ones(3, dtype=torch.float32)
    lab_offset = torch.zeros(3, dtype=torch.float32)
    lab_mean_jump_before = torch.zeros(3, dtype=torch.float32)
    lab_mean_jump_after = torch.zeros(3, dtype=torch.float32)
    lab_std_ratio_before = torch.ones(3, dtype=torch.float32)
    lab_std_ratio_after = torch.ones(3, dtype=torch.float32)
    jump_before = 0.0
    jump_after = 0.0
    spatial_jump_before = 0.0
    spatial_residual_after_global = 0.0
    spatial_jump_after = 0.0
    maximum_total_rgb_delta = 0.0
    clamped_channel_fraction = 0.0
    reference_path = None

    if segment_index > 0:
        reference_path = color_state_path(safe_chain, segment_index - 1)
        if not bool(enabled):
            status = "DISABLED_SOURCE_IDENTITY"
        else:
            reference, _metadata = _load_reference(
                reference_path,
                chain_id=safe_chain,
                source_segment_index=segment_index - 1,
                width=int(frames.shape[2]),
                height=int(frames.shape[1]),
            )
            compare_count = min(
                reference_frames,
                int(reference["tail_rgb_means"].shape[0]),
                int(frames.shape[0]),
            )
            ref_rgb_mean = reference["tail_rgb_means"][-compare_count:].mean(dim=0).to(
                device=frames.device, dtype=torch.float32
            )
            ref_thumbnail = reference["tail_rgb_thumbnail"].to(
                device=frames.device, dtype=torch.float32
            )
            ref_lab_mean = reference["tail_lab_mean"].to(
                device=frames.device, dtype=torch.float32
            )
            ref_lab_std = reference["tail_lab_std"].to(
                device=frames.device, dtype=torch.float32
            )
            current_head = frames[:compare_count, ..., :3]
            current_stats = _frame_statistics(current_head)
            current_rgb_mean = current_stats["rgb_means"].mean(dim=0)
            jump_vector = current_rgb_mean - ref_rgb_mean
            jump_before = float(jump_vector.abs().max())
            spatial_jump_before = float(
                (current_stats["rgb_thumbnail"] - ref_thumbnail).abs().max()
            )
            lab_mean_jump_before = current_stats["lab_mean"] - ref_lab_mean
            lab_std_ratio_before = _safe_std_ratio(
                current_stats["lab_std"], ref_lab_std
            )
            lab_scale = _safe_std_ratio(
                ref_lab_std, current_stats["lab_std"]
            ).clamp(LAB_SCALE_MINIMUM, LAB_SCALE_MAXIMUM)
            lab_offset = ref_lab_mean - lab_scale * current_stats["lab_mean"]
            moment_jump = float((lab_scale - 1.0).abs().max())
            material_jump = max(jump_before, spatial_jump_before, moment_jump)
            if jump_before >= float(scene_cut_threshold):
                status = "ABSTAIN_SCENE_CUT_OR_LARGE_COLOR_JUMP"
                jump_after = jump_before
                spatial_jump_after = spatial_jump_before
                lab_mean_jump_after = lab_mean_jump_before
                lab_std_ratio_after = lab_std_ratio_before
            elif material_jump <= float(minimum_jump) or float(strength) == 0.0:
                status = "NO_MATERIAL_COLOR_JUMP_SOURCE_IDENTITY"
                jump_after = jump_before
                spatial_jump_after = spatial_jump_before
                lab_mean_jump_after = lab_mean_jump_before
                lab_std_ratio_after = lab_std_ratio_before
            else:
                spatial_residual = (
                    ref_thumbnail
                    - _global_transform_thumbnail(current_head, lab_scale, lab_offset)
                ).clamp(-float(maximum_offset), float(maximum_offset))
                spatial_residual_after_global = float(spatial_residual.abs().max())
                spatial_field = torch_functional.interpolate(
                    spatial_residual[None],
                    size=(int(frames.shape[1]), int(frames.shape[2])),
                    mode="bilinear",
                    align_corners=False,
                ).permute(0, 2, 3, 1)
                weights = _fade_weights(
                    int(frames.shape[0]),
                    reference_frames,
                    transition_frames,
                    device=frames.device,
                )
                candidate = frames.clone()
                clamped_channels = 0
                processed_channels = 0
                affected = min(int(frames.shape[0]), transition_frames)
                for frame_index in range(affected):
                    source_rgb = frames[frame_index : frame_index + 1, ..., :3].float()
                    globally_matched = _global_lab_transform(
                        source_rgb, lab_scale, lab_offset
                    )
                    raw_delta = (
                        globally_matched + spatial_field - source_rgb
                    ) * float(strength)
                    clamped_channels += int(
                        (raw_delta.abs() > float(maximum_offset)).sum().item()
                    )
                    processed_channels += int(raw_delta.numel())
                    bounded_delta = raw_delta.clamp(
                        -float(maximum_offset), float(maximum_offset)
                    )
                    maximum_total_rgb_delta = max(
                        maximum_total_rgb_delta, float(bounded_delta.abs().max())
                    )
                    corrected = (
                        source_rgb
                        + weights[frame_index].float() * bounded_delta
                    ).clamp(0.0, 1.0)
                    candidate[frame_index : frame_index + 1, ..., :3] = corrected.to(
                        dtype=frames.dtype
                    )
                output = candidate
                output_stats = _frame_statistics(output[:compare_count, ..., :3])
                output_rgb_mean = output_stats["rgb_means"].mean(dim=0)
                effective_rgb_offset = output_rgb_mean - current_rgb_mean
                jump_after = float((output_rgb_mean - ref_rgb_mean).abs().max())
                spatial_jump_after = float(
                    (output_stats["rgb_thumbnail"] - ref_thumbnail).abs().max()
                )
                lab_mean_jump_after = output_stats["lab_mean"] - ref_lab_mean
                lab_std_ratio_after = _safe_std_ratio(
                    output_stats["lab_std"], ref_lab_std
                )
                clamped_channel_fraction = (
                    clamped_channels / processed_channels if processed_channels else 0.0
                )
                applied = True
                status = "COLOR_MATCH_APPLIED"
    elif not bool(enabled):
        status = "REFERENCE_INITIALIZED_COLOR_MATCH_DISABLED"

    state_path = color_state_path(safe_chain, segment_index)
    report: dict[str, Any] = {
        "schema": COLOR_MATCH_SCHEMA,
        "state_schema": COLOR_STATE_SCHEMA,
        "status": status,
        "enabled": bool(enabled),
        "chain_id": safe_chain,
        "segment_index": segment_index,
        "frame_count": int(frames.shape[0]),
        "width": int(frames.shape[2]),
        "height": int(frames.shape[1]),
        "reference_frames_requested": reference_frames,
        "comparison_frame_count": compare_count,
        "transition_frames": transition_frames,
        "strength": float(strength),
        "minimum_jump": float(minimum_jump),
        "maximum_offset": float(maximum_offset),
        "scene_cut_threshold": float(scene_cut_threshold),
        "reference_state_path": str(reference_path) if reference_path is not None else None,
        "written_state_path": str(state_path),
        "rgb_jump_vector_before": _vector(jump_vector),
        "maximum_rgb_jump_before": jump_before,
        "effective_rgb_offset": _vector(effective_rgb_offset),
        "applied_rgb_offset": _vector(effective_rgb_offset),
        "maximum_rgb_jump_after": jump_after,
        "lab_mean_jump_before": _vector(lab_mean_jump_before),
        "lab_mean_jump_after": _vector(lab_mean_jump_after),
        "lab_std_ratio_before": _vector(lab_std_ratio_before),
        "lab_std_ratio_after": _vector(lab_std_ratio_after),
        "applied_lab_scale": _vector(lab_scale),
        "applied_lab_offset": _vector(lab_offset),
        "lab_scale_bounds": [LAB_SCALE_MINIMUM, LAB_SCALE_MAXIMUM],
        "spatial_grid": [SPATIAL_GRID_HEIGHT, SPATIAL_GRID_WIDTH],
        "maximum_spatial_rgb_jump_before": spatial_jump_before,
        "maximum_spatial_residual_after_global": spatial_residual_after_global,
        "maximum_spatial_rgb_jump_after": spatial_jump_after,
        "maximum_total_rgb_delta": maximum_total_rgb_delta,
        "clamped_channel_fraction": clamped_channel_fraction,
        "applied": applied,
        "source_identity": output is frames,
        "audio_touched": False,
        "latent_touched": False,
        "detail_generation": False,
        "method": "bounded_uniform_reinhard_lab_spatial_rgb_with_fade",
        "reference_design": (
            "ComfyUI built-in ColorTransfer reinhard_lab pooled mean/std matching plus "
            "WanAnimatePlus auto_drift-inspired five-frame seam comparison; T8 adds an "
            "8x5 local RGB residual field, a 0.02 total-delta bound, temporal fade and "
            "strict state/checksum validation"
        ),
        "contract": (
            "post-decode/post-trim SDR RGB correction only; preceding output tail is the "
            "reference; auxiliary channels, audio and native AV latent are untouched"
        ),
    }
    report["report_sha256"] = hashlib.sha256(
        json.dumps(report, sort_keys=True, allow_nan=False).encode("utf-8")
    ).hexdigest()
    _save_reference(
        state_path,
        frames=output,
        chain_id=safe_chain,
        source_segment_index=segment_index,
        reference_frames=reference_frames,
        report=report,
    )
    return output, status, _json(report)
