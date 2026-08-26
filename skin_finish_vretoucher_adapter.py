from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from typing import Any

import torch
from torch.nn import functional as torch_functional


VRETOUCHER_ADAPTER_SCHEMA = "h3_t8_skin_finish_vretoucher_adapter/v1"
VRETOUCHER_CONTEXT_FRAMES = 6
VRETOUCHER_CANVAS_SIZE = 512


class VRetouchAdapterUnavailable(RuntimeError):
    def __init__(self, status: str, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail


def _canonical_json(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hashed_report(value: dict[str, Any]) -> dict[str, Any]:
    output = dict(value)
    output["sha256"] = hashlib.sha256(_canonical_json(output).encode("utf-8")).hexdigest()
    return output


def _validate_frames(frames: torch.Tensor) -> tuple[int, int, int, int]:
    if not isinstance(frames, torch.Tensor) or frames.ndim != 4:
        raise VRetouchAdapterUnavailable(
            "ABSTAIN_INVALID_SOURCE_SHAPE",
            "frames must be a ComfyUI IMAGE tensor shaped [T,H,W,C]",
        )
    count, height, width, channels = [int(value) for value in frames.shape]
    if count < 1 or height < 8 or width < 8 or channels < 3:
        raise VRetouchAdapterUnavailable(
            "ABSTAIN_INVALID_SOURCE_SHAPE",
            "frames require at least one 8x8 RGB image",
        )
    if not frames.is_floating_point():
        raise VRetouchAdapterUnavailable(
            "ABSTAIN_INVALID_SOURCE_DTYPE",
            "frames must use a floating-point ComfyUI IMAGE dtype",
        )
    return count, height, width, channels


def causal_context_indices(
    current_frame: int,
    shot_start: int,
    shot_end: int,
    context_frames: int = VRETOUCHER_CONTEXT_FRAMES,
) -> list[int]:
    current = int(current_frame)
    start = int(shot_start)
    end = int(shot_end)
    count = int(context_frames)
    if count != VRETOUCHER_CONTEXT_FRAMES:
        raise ValueError("VRetouchEr requires exactly six context frames")
    if start < 0 or end < start or current < start or current > end:
        raise ValueError("current_frame must lie inside a finite inclusive shot range")
    first = current - count + 1
    return [max(start, first + offset) for offset in range(count)]


def _validated_box(value: Sequence[float] | None) -> tuple[float, float, float, float]:
    if value is None or len(value) != 4:
        raise ValueError("a tracked face box [left,top,right,bottom] is required")
    left, top, right, bottom = [float(item) for item in value]
    if not all(math.isfinite(item) for item in (left, top, right, bottom)):
        raise ValueError("face box contains a non-finite coordinate")
    if right - left < 4.0 or bottom - top < 4.0:
        raise ValueError("face box is too small or inverted")
    return left, top, right, bottom


def square_face_crop_record(
    face_box: Sequence[float],
    *,
    frame_width: int,
    frame_height: int,
    context_factor: float = 1.45,
    canvas_size: int = VRETOUCHER_CANVAS_SIZE,
) -> dict[str, Any]:
    if int(frame_width) < 8 or int(frame_height) < 8:
        raise ValueError("frame geometry must be at least 8x8")
    if not 1.0 <= float(context_factor) <= 3.0:
        raise ValueError("context_factor must stay within 1.0..3.0")
    if int(canvas_size) != VRETOUCHER_CANVAS_SIZE:
        raise ValueError("the audited VRetouchEr representation uses a fixed 512 canvas")
    left, top, right, bottom = _validated_box(face_box)
    center_x = (left + right) * 0.5
    center_y = (top + bottom) * 0.5
    side = max(8, int(math.ceil(max(right - left, bottom - top) * float(context_factor))))
    crop_left = int(math.floor(center_x - side * 0.5))
    crop_top = int(math.floor(center_y - side * 0.5))
    crop_right = crop_left + side
    crop_bottom = crop_top + side
    clipped_left = max(0, crop_left)
    clipped_top = max(0, crop_top)
    clipped_right = min(int(frame_width), crop_right)
    clipped_bottom = min(int(frame_height), crop_bottom)
    if clipped_right <= clipped_left or clipped_bottom <= clipped_top:
        raise ValueError("expanded face crop does not intersect the source frame")
    scale = float(canvas_size) / float(side)
    return {
        "face_box_xyxy": [left, top, right, bottom],
        "square_crop_xyxy_unclipped": [crop_left, crop_top, crop_right, crop_bottom],
        "square_crop_side_px": side,
        "clipped_crop_xyxy": [clipped_left, clipped_top, clipped_right, clipped_bottom],
        "padding_ltrb": [
            max(0, -crop_left),
            max(0, -crop_top),
            max(0, crop_right - int(frame_width)),
            max(0, crop_bottom - int(frame_height)),
        ],
        "canvas_size": int(canvas_size),
        "scale_x": scale,
        "scale_y": scale,
        "anisotropy": 1.0,
        "whole_frame_is_never_squashed": True,
    }


def build_vretoucher_context_plan(
    frames: torch.Tensor,
    *,
    current_frame: int,
    shot_start: int,
    shot_end: int,
    track_key: str,
    frame_track_keys: Sequence[str | None],
    face_boxes: Sequence[Sequence[float] | None],
    context_factor: float = 1.45,
) -> dict[str, Any]:
    frame_count, height, width, channels = _validate_frames(frames)
    if int(shot_end) >= frame_count:
        raise VRetouchAdapterUnavailable(
            "ABSTAIN_SHOT_RANGE_OUTSIDE_SOURCE",
            "shot_end exceeds the source frame count",
        )
    if len(frame_track_keys) != frame_count or len(face_boxes) != frame_count:
        raise VRetouchAdapterUnavailable(
            "ABSTAIN_TRACK_EVIDENCE_LENGTH_MISMATCH",
            "track labels and face boxes must contain one item per source frame",
        )
    key = str(track_key).strip()
    if not key:
        raise VRetouchAdapterUnavailable(
            "ABSTAIN_TRACK_KEY_MISSING",
            "a reviewed shot-local track key is required",
        )
    try:
        indices = causal_context_indices(current_frame, shot_start, shot_end)
    except ValueError as error:
        raise VRetouchAdapterUnavailable(
            "ABSTAIN_INVALID_CAUSAL_WINDOW", str(error)
        ) from error

    records: list[dict[str, Any]] = []
    for position, source_index in enumerate(indices):
        observed_key = frame_track_keys[source_index]
        if observed_key != key:
            raise VRetouchAdapterUnavailable(
                "ABSTAIN_TRACK_DISCONTINUITY",
                f"frame {source_index} belongs to {observed_key!r}, not reviewed track {key!r}",
            )
        try:
            crop = square_face_crop_record(
                face_boxes[source_index],
                frame_width=width,
                frame_height=height,
                context_factor=float(context_factor),
            )
        except (TypeError, ValueError) as error:
            raise VRetouchAdapterUnavailable(
                "ABSTAIN_FACE_LOCALIZATION_MISSING",
                f"frame {source_index} cannot form a safe face crop: {error}",
            ) from error
        records.append(
            {
                "context_position": position,
                "source_frame_index": source_index,
                "left_padded_from_shot_start": source_index == int(shot_start)
                and source_index != int(current_frame) - (VRETOUCHER_CONTEXT_FRAMES - 1 - position),
                **crop,
            }
        )

    return _hashed_report(
        {
            "schema": VRETOUCHER_ADAPTER_SCHEMA,
            "status": "READY_WEIGHT_AND_MODEL_NOT_YET_VALIDATED",
            "source": {
                "frame_count": frame_count,
                "height": height,
                "width": width,
                "channels": channels,
            },
            "current_frame": int(current_frame),
            "shot": {"start_frame": int(shot_start), "end_frame": int(shot_end)},
            "track_key": key,
            "context_frame_count": VRETOUCHER_CONTEXT_FRAMES,
            "context_indices": indices,
            "context_records": records,
            "causal_only": True,
            "tail_wrap": False,
            "shot_boundary_reset": True,
            "identity_scope": "one_reviewed_shot_local_track",
            "model_output_scope": "current_newest_frame_only",
            "automatic_accept": False,
        }
    )


def _extract_square(frame: torch.Tensor, record: dict[str, Any]) -> torch.Tensor:
    left, top, right, bottom = [
        int(value) for value in record["square_crop_xyxy_unclipped"]
    ]
    side = int(record["square_crop_side_px"])
    image = frame[..., :3].detach().to(device="cpu", dtype=torch.float32).movedim(-1, 0)
    pad_left, pad_top, pad_right, pad_bottom = [
        int(value) for value in record["padding_ltrb"]
    ]
    if any((pad_left, pad_top, pad_right, pad_bottom)):
        image = torch_functional.pad(
            image,
            (pad_left, pad_right, pad_top, pad_bottom),
            mode="replicate",
        )
    shifted_left = left + pad_left
    shifted_top = top + pad_top
    crop = image[:, shifted_top : shifted_top + side, shifted_left : shifted_left + side]
    if crop.shape[-2:] != (side, side):
        raise VRetouchAdapterUnavailable(
            "ABSTAIN_CROP_GEOMETRY_MISMATCH",
            "padded face crop did not produce the planned square geometry",
        )
    return torch_functional.interpolate(
        crop.unsqueeze(0),
        size=(VRETOUCHER_CANVAS_SIZE, VRETOUCHER_CANVAS_SIZE),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )[0]


def extract_vretoucher_context(
    frames: torch.Tensor,
    plan: dict[str, Any],
    *,
    normalize_to_minus_one_one: bool = True,
) -> torch.Tensor:
    _validate_frames(frames)
    if not isinstance(plan, dict) or plan.get("schema") != VRETOUCHER_ADAPTER_SCHEMA:
        raise VRetouchAdapterUnavailable(
            "ABSTAIN_CONTEXT_PLAN_INVALID",
            f"plan must use {VRETOUCHER_ADAPTER_SCHEMA}",
        )
    records = plan.get("context_records")
    if not isinstance(records, list) or len(records) != VRETOUCHER_CONTEXT_FRAMES:
        raise VRetouchAdapterUnavailable(
            "ABSTAIN_CONTEXT_PLAN_INVALID", "plan must contain six crop records"
        )
    crops = torch.stack(
        [
            _extract_square(frames[int(record["source_frame_index"])], record)
            for record in records
        ]
    ).clamp_(0.0, 1.0)
    if normalize_to_minus_one_one:
        crops = crops.mul(2.0).sub(1.0)
    return crops


def _gaussian_blur_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if int(radius) <= 0:
        return mask
    radius = int(radius)
    sigma = max(float(radius) / 3.0, 0.5)
    axis = torch.arange(-radius, radius + 1, dtype=torch.float32)
    kernel = torch.exp(-(axis * axis) / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum()
    value = mask[None, None].float()
    value = torch_functional.pad(value, (radius, radius, 0, 0), mode="replicate")
    value = torch_functional.conv2d(value, kernel.view(1, 1, 1, -1))
    value = torch_functional.pad(value, (0, 0, radius, radius), mode="replicate")
    value = torch_functional.conv2d(value, kernel.view(1, 1, -1, 1))
    return value[0, 0]


def _proposal_chw(proposal: torch.Tensor) -> torch.Tensor:
    if not isinstance(proposal, torch.Tensor) or proposal.ndim != 3:
        raise VRetouchAdapterUnavailable(
            "ABSTAIN_PROPOSAL_SHAPE_INVALID",
            "proposal_current_crop must be [3,H,W] or [H,W,3]",
        )
    if proposal.shape[0] == 3:
        output = proposal
    elif proposal.shape[-1] == 3:
        output = proposal.movedim(-1, 0)
    else:
        raise VRetouchAdapterUnavailable(
            "ABSTAIN_PROPOSAL_SHAPE_INVALID",
            "proposal_current_crop must contain exactly three RGB channels",
        )
    if not output.is_floating_point() or not bool(torch.isfinite(output).all()):
        raise VRetouchAdapterUnavailable(
            "ABSTAIN_PROPOSAL_VALUES_INVALID",
            "proposal_current_crop must contain finite floating-point RGB values",
        )
    return output.detach().to(device="cpu", dtype=torch.float32)


def compose_vretoucher_current_frame(
    source_frame: torch.Tensor,
    proposal_current_crop: torch.Tensor,
    current_record: dict[str, Any],
    semantic_skin_mask: torch.Tensor,
    *,
    person_mask: torch.Tensor | None = None,
    amount: float = 1.0,
    feather_px: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    if not isinstance(source_frame, torch.Tensor) or source_frame.ndim != 3:
        raise VRetouchAdapterUnavailable(
            "ABSTAIN_INVALID_SOURCE_SHAPE", "source_frame must be [H,W,C]"
        )
    height, width, channels = [int(value) for value in source_frame.shape]
    if height < 8 or width < 8 or channels < 3 or not source_frame.is_floating_point():
        raise VRetouchAdapterUnavailable(
            "ABSTAIN_INVALID_SOURCE_SHAPE", "source_frame must contain floating RGB"
        )
    if not 0.0 <= float(amount) <= 1.0:
        raise ValueError("amount must stay within 0..1")
    if not 0 <= int(feather_px) <= 64:
        raise ValueError("feather_px must stay within 0..64")
    mask = semantic_skin_mask.detach().to(device="cpu", dtype=torch.float32).squeeze()
    if mask.shape != (height, width) or not bool(torch.isfinite(mask).all()):
        raise VRetouchAdapterUnavailable(
            "ABSTAIN_SEMANTIC_MASK_INVALID",
            "semantic_skin_mask must be one finite full-frame [H,W] mask",
        )
    mask = mask.clamp(0.0, 1.0)
    if person_mask is not None:
        person = person_mask.detach().to(device="cpu", dtype=torch.float32).squeeze()
        if person.shape != (height, width) or not bool(torch.isfinite(person).all()):
            raise VRetouchAdapterUnavailable(
                "ABSTAIN_PERSON_MASK_INVALID",
                "person_mask must be one finite full-frame [H,W] mask",
            )
        mask = mask * person.clamp(0.0, 1.0)
    hard = mask > 0.0
    soft = _gaussian_blur_mask(mask, int(feather_px)) * hard.float()
    soft = soft.clamp(0.0, 1.0).mul(float(amount))

    proposal = _proposal_chw(proposal_current_crop)
    side = int(current_record["square_crop_side_px"])
    proposal = torch_functional.interpolate(
        proposal.unsqueeze(0),
        size=(side, side),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )[0].movedim(0, -1).clamp(0.0, 1.0)
    left, top, right, bottom = [
        int(value) for value in current_record["square_crop_xyxy_unclipped"]
    ]
    target_left = max(0, left)
    target_top = max(0, top)
    target_right = min(width, right)
    target_bottom = min(height, bottom)
    if target_right <= target_left or target_bottom <= target_top:
        raise VRetouchAdapterUnavailable(
            "ABSTAIN_CROP_GEOMETRY_MISMATCH", "current crop misses the source frame"
        )
    proposal_left = target_left - left
    proposal_top = target_top - top
    proposal_region = proposal[
        proposal_top : proposal_top + (target_bottom - target_top),
        proposal_left : proposal_left + (target_right - target_left),
    ]
    if proposal_region.shape[:2] != (target_bottom - target_top, target_right - target_left):
        raise VRetouchAdapterUnavailable(
            "ABSTAIN_CROP_GEOMETRY_MISMATCH", "proposal paste region is inconsistent"
        )

    source_cpu = source_frame.detach().to(device="cpu")
    output = source_cpu.clone()
    alpha = soft[target_top:target_bottom, target_left:target_right, None].to(
        dtype=source_cpu.dtype
    )
    source_rgb = source_cpu[target_top:target_bottom, target_left:target_right, :3]
    output[target_top:target_bottom, target_left:target_right, :3] = (
        source_rgb * (1.0 - alpha)
        + proposal_region.to(dtype=source_cpu.dtype) * alpha
    )
    changed = torch.any(output[..., :3] != source_cpu[..., :3], dim=-1)
    outside = soft <= 0.0
    exterior_exact = bool(torch.equal(output[..., :3][outside], source_cpu[..., :3][outside]))
    auxiliary_exact = bool(torch.equal(output[..., 3:], source_cpu[..., 3:]))
    report = _hashed_report(
        {
            "schema": VRETOUCHER_ADAPTER_SCHEMA,
            "status": "CANDIDATE_REQUIRES_IDENTITY_AND_HUMAN_REVIEW",
            "changed_pixel_count": int(torch.count_nonzero(changed)),
            "effective_mask_pixel_count": int(torch.count_nonzero(soft > 0.0)),
            "semantic_skin_only": True,
            "person_track_intersection": person_mask is not None,
            "exterior_exact": exterior_exact,
            "auxiliary_channels_exact": auxiliary_exact,
            "automatic_accept": False,
            "weight_and_model_inference_validated": False,
        }
    )
    return output, soft, _canonical_json(report)
