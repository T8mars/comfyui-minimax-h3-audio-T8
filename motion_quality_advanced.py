from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import torch
import torch.nn.functional as torch_functional

from .sampling import shift_sigma, time_shift_sigma


TURBO_DUAL_CLOCK_TEST_STEPS = 8
_PROFILE_NFE = {
    "stock20": 20,
    "turbo_standard8": TURBO_DUAL_CLOCK_TEST_STEPS,
    "turbo_ema8": TURBO_DUAL_CLOCK_TEST_STEPS,
    "turbo_fl2v8": TURBO_DUAL_CLOCK_TEST_STEPS,
    "custom_strict": None,
}
_TURBO_PROFILES = {name for name in _PROFILE_NFE if name.startswith("turbo_")}


def canonical_json(value: Any, *, indent: int | None = None) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        indent=indent,
        separators=None if indent is not None else (",", ":"),
        sort_keys=True,
    )


def _schedule_sha(values: torch.Tensor) -> str:
    canonical = canonical_json([format(float(value), ".17g") for value in values])
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _validate_h3_sigmas(sigmas: torch.Tensor, profile: str) -> torch.Tensor:
    if not isinstance(sigmas, torch.Tensor):
        raise TypeError("sigmas must be a torch.Tensor")
    if sigmas.ndim != 1 or sigmas.numel() < 2:
        raise ValueError("H3 sigmas must be a one-dimensional tensor with at least two values")
    if not sigmas.dtype.is_floating_point:
        raise TypeError("H3 sigmas must use a floating-point dtype")

    values = sigmas.detach().to(device="cpu", dtype=torch.float64)
    if not bool(torch.isfinite(values).all()):
        raise ValueError("H3 sigmas contain NaN or Inf")
    if float(values.min()) < -1e-9 or float(values.max()) > 1.0 + 1e-9:
        raise ValueError("H3 Flow sigmas must stay inside the normalized [0, 1] range")
    if abs(float(values[-1])) > 1e-9:
        raise ValueError("H3 sigma schedule must end at exactly zero")
    if not bool(torch.all(values[:-1] > values[1:])):
        raise ValueError("H3 sigmas must be strictly descending with no duplicate steps")
    if profile not in _PROFILE_NFE:
        raise ValueError(f"unsupported H3 schedule profile: {profile!r}")

    expected_nfe = _PROFILE_NFE[profile]
    actual_nfe = int(values.numel() - 1)
    if expected_nfe is not None and actual_nfe != expected_nfe:
        raise ValueError(
            f"profile {profile!r} requires {expected_nfe} steps, got {actual_nfe}"
        )
    return values


def _inverse_shift_sigma(sigmas: torch.Tensor, shift: float) -> torch.Tensor:
    if not math.isfinite(shift) or shift <= 0.0:
        raise ValueError("sigma shift must be finite and greater than zero")
    denominator = shift + sigmas * (1.0 - shift)
    if not bool(torch.all(denominator > 0.0)):
        raise ValueError("sigma shift produced an invalid base-flow denominator")
    base = sigmas / denominator
    if not bool(torch.isfinite(base).all()):
        raise ValueError("inverse sigma shift produced NaN or Inf")
    return base


def _selected_intervals(
    base_sigmas: torch.Tensor,
    range_mode: str,
    tail_intervals: int,
    start_progress: float,
    end_progress: float,
) -> list[int]:
    interval_count = int(base_sigmas.numel() - 1)
    if range_mode == "tail_intervals":
        if tail_intervals < 1 or tail_intervals > interval_count:
            raise ValueError(
                f"tail_intervals must be between 1 and {interval_count}, got {tail_intervals}"
            )
        return list(range(interval_count - tail_intervals, interval_count))
    if range_mode != "base_progress":
        raise ValueError(f"unsupported range_mode: {range_mode!r}")
    if not (
        math.isfinite(start_progress)
        and math.isfinite(end_progress)
        and 0.0 <= start_progress < end_progress <= 1.0
    ):
        raise ValueError("base progress must satisfy 0 <= start < end <= 1")

    selected = []
    for interval in range(interval_count):
        midpoint_base = float((base_sigmas[interval] + base_sigmas[interval + 1]) * 0.5)
        midpoint_progress = 1.0 - midpoint_base
        if start_progress <= midpoint_progress <= end_progress:
            selected.append(interval)
    if not selected:
        raise ValueError("the requested base-progress range selects no sigma intervals")
    return selected


def _balanced_insert_counts(extra_substeps: int, selected_count: int) -> list[int]:
    if extra_substeps < 0:
        raise ValueError("extra_substeps cannot be negative")
    if selected_count < 1:
        raise ValueError("at least one interval must be selected")
    return [
        ((index + 1) * extra_substeps) // selected_count
        - (index * extra_substeps) // selected_count
        for index in range(selected_count)
    ]


def build_av_sigma_tail_schedule(
    sigmas: torch.Tensor,
    mode: str,
    extra_substeps: int,
    range_mode: str,
    tail_intervals: int,
    start_progress: float,
    end_progress: float,
    spacing: str,
    shift_video: float,
    shift_audio: float,
    profile: str,
    sampling_route: str,
    accept_turbo_schedule_ood: bool,
) -> tuple[torch.Tensor, int, str]:
    """Build a fail-closed H3 AV schedule while preserving every input knot."""
    values = _validate_h3_sigmas(sigmas, profile)
    if mode not in {"report_only", "apply_exp"}:
        raise ValueError(f"unsupported schedule mode: {mode!r}")
    if spacing not in {"base_time_linear", "base_time_cosine"}:
        raise ValueError(f"unsupported spacing: {spacing!r}")
    if sampling_route not in {
        "dual_clock_euler",
        "native_flow_av_unverified",
        "multirate_exp_unsupported",
        "unknown",
    }:
        raise ValueError(f"unsupported sampling_route: {sampling_route!r}")
    if extra_substeps < 0 or extra_substeps > 32:
        raise ValueError("extra_substeps must be between 0 and 32")

    base_sigmas = _inverse_shift_sigma(values, shift_video)
    selected = _selected_intervals(
        base_sigmas,
        range_mode,
        tail_intervals,
        start_progress,
        end_progress,
    )
    blockers = []
    if sampling_route != "dual_clock_euler":
        blockers.append(
            "P0 apply supports only the explicit dual_clock_euler route; other routes remain unverified"
        )
    if profile in _TURBO_PROFILES and extra_substeps > 0 and not accept_turbo_schedule_ood:
        blockers.append(
            "Turbo dual-clock uses the project 8-step test baseline; inserted times are experimental OOD points"
        )

    applied = mode == "apply_exp" and extra_substeps > 0
    if applied and blockers:
        raise ValueError("schedule apply blocked: " + "; ".join(blockers))

    output = sigmas
    inserted: list[dict[str, Any]] = []
    if applied:
        counts = _balanced_insert_counts(extra_substeps, len(selected))
        counts_by_interval = dict(zip(selected, counts, strict=True))
        pieces = []
        for interval in range(values.numel() - 1):
            pieces.append(sigmas[interval : interval + 1])
            count = counts_by_interval.get(interval, 0)
            if count == 0:
                continue
            fractions = torch.arange(1, count + 1, dtype=torch.float64) / (count + 1)
            if spacing == "base_time_cosine":
                fractions = (1.0 - torch.cos(fractions * math.pi)) * 0.5
            start_base = base_sigmas[interval]
            end_base = base_sigmas[interval + 1]
            inserted_base = start_base + (end_base - start_base) * fractions
            inserted_video_cpu = shift_sigma(inserted_base, shift_video)
            inserted_video = inserted_video_cpu.to(device=sigmas.device, dtype=sigmas.dtype)
            pieces.append(inserted_video)
            inserted_audio_cpu = shift_sigma(inserted_base, shift_audio)
            for local_index in range(count):
                inserted.append(
                    {
                        "after_input_index": interval,
                        "base_sigma": float(inserted_base[local_index]),
                        "video_sigma": float(inserted_video_cpu[local_index]),
                        "audio_sigma": float(inserted_audio_cpu[local_index]),
                    }
                )
        pieces.append(sigmas[-1:])
        output = torch.cat(pieces)

    output_values = _validate_h3_sigmas(output, "custom_strict")
    input_cursor = 0
    for value in output_values:
        if input_cursor < values.numel() and float(value) == float(values[input_cursor]):
            input_cursor += 1
    if input_cursor != values.numel():
        raise RuntimeError("internal error: output schedule did not preserve every original sigma knot")

    audio_values = time_shift_sigma(output_values, shift_video, shift_audio)
    actual_nfe = int(output_values.numel() - 1)
    original_nfe = int(values.numel() - 1)
    report = {
        "schema": "minimax_h3_av_sigma_tail_subdivision_t8_v1",
        "status": "applied_exp" if applied else "report_only",
        "applied": applied,
        "noop_reason": (
            "report_only"
            if mode == "report_only"
            else "extra_substeps_is_zero"
            if extra_substeps == 0
            else None
        ),
        "profile": profile,
        "sampling_route": sampling_route,
        "turbo_dual_clock_test_standard_steps": TURBO_DUAL_CLOCK_TEST_STEPS,
        "turbo_schedule_ood_accepted": bool(accept_turbo_schedule_ood),
        "shift_video": float(shift_video),
        "shift_audio": float(shift_audio),
        "range_mode": range_mode,
        "selected_intervals": selected,
        "spacing": spacing,
        "requested_extra_substeps": int(extra_substeps),
        "inserted_substeps": len(inserted) if applied else 0,
        "original_nfe": original_nfe,
        "actual_nfe": actual_nfe,
        "estimated_sampler_time_increase_percent": (
            100.0 * (actual_nfe - original_nfe) / original_nfe
        ),
        "all_original_knots_preserved": True,
        "input_schedule_sha256": _schedule_sha(values),
        "output_schedule_sha256": _schedule_sha(output_values),
        "base_sigmas": [float(value) for value in _inverse_shift_sigma(output_values, shift_video)],
        "video_sigmas": [float(value) for value in output_values],
        "audio_sigmas": [float(value) for value in audio_values],
        "inserted_points": inserted if applied else [],
        "apply_blockers": blockers,
        "joint_av_forward_notice": (
            "Every added sigma is one additional full joint H3 audio-video DiT evaluation."
        ),
        "quality_validated": False,
        "memory_safe_claim": False,
    }
    return output, actual_nfe, canonical_json(report, indent=2)


def _frame_to_gray(frame: torch.Tensor) -> torch.Tensor:
    frame = frame.detach().to(device="cpu", dtype=torch.float32)
    if frame.ndim != 3 or frame.shape[-1] not in {1, 3, 4}:
        raise ValueError("IMAGE frames must have shape [frames, height, width, 1|3|4]")
    if not bool(torch.isfinite(frame).all()):
        raise ValueError("IMAGE frames contain NaN or Inf")
    minimum = float(frame.min())
    maximum = float(frame.max())
    if minimum < -0.01 or maximum > 1.01:
        raise ValueError("IMAGE values must be normalized to the ComfyUI [0, 1] range")
    frame = frame.clamp(0.0, 1.0)
    if frame.shape[-1] == 1:
        return frame[..., 0]
    return (
        frame[..., 0] * 0.2126
        + frame[..., 1] * 0.7152
        + frame[..., 2] * 0.0722
    )


def _prepare_masks(
    face_mask: torch.Tensor | None,
    frame_count: int,
    height: int,
    width: int,
) -> torch.Tensor | None:
    if face_mask is None:
        return None
    masks = face_mask.detach().to(device="cpu", dtype=torch.float32)
    if masks.ndim == 2:
        masks = masks.unsqueeze(0)
    elif masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim != 3 or masks.shape[0] not in {1, frame_count}:
        raise ValueError("face_mask must have shape [H,W], [1,H,W], or [frames,H,W]")
    if not bool(torch.isfinite(masks).all()):
        raise ValueError("face_mask contains NaN or Inf")
    masks = torch_functional.interpolate(
        masks.unsqueeze(1),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )[:, 0].clamp(0.0, 1.0)
    if masks.shape[0] == 1:
        masks = masks.expand(frame_count, -1, -1)
    if bool(torch.any(masks.flatten(1).sum(dim=1) < 1.0)):
        raise ValueError("face_mask contains an empty frame")
    return masks


def _manual_roi_bounds(
    height: int,
    width: int,
    roi_x: float,
    roi_y: float,
    roi_width: float,
    roi_height: float,
) -> tuple[int, int, int, int]:
    values = (roi_x, roi_y, roi_width, roi_height)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("manual ROI values must be finite")
    if roi_width <= 0.0 or roi_height <= 0.0:
        raise ValueError("manual ROI width and height must be greater than zero")
    if roi_x < 0.0 or roi_y < 0.0 or roi_x + roi_width > 1.0 or roi_y + roi_height > 1.0:
        raise ValueError("manual ROI must stay inside normalized frame coordinates")
    left = min(width - 1, int(math.floor(roi_x * width)))
    top = min(height - 1, int(math.floor(roi_y * height)))
    right = max(left + 1, int(math.ceil((roi_x + roi_width) * width)))
    bottom = max(top + 1, int(math.ceil((roi_y + roi_height) * height)))
    return top, min(bottom, height), left, min(right, width)


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor | None) -> float:
    if weights is None:
        return float(values.mean())
    denominator = float(weights.sum())
    if denominator <= 0.0:
        raise ValueError("quality audit ROI has no weighted pixels")
    return float((values * weights).sum() / denominator)


def _contiguous_ranges(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    ranges = []
    start = previous = indices[0]
    for index in indices[1:]:
        if index != previous + 1:
            ranges.append((start, previous))
            start = index
        previous = index
    ranges.append((start, previous))
    return ranges


def _legal_repair_window(
    raw_start: int,
    raw_end: int,
    frame_count: int,
    context_frames: int,
) -> tuple[int, int, bool]:
    expanded_start = max(0, raw_start - context_frames)
    expanded_end = min(frame_count - 1, raw_end + context_frames)
    needed = expanded_end - expanded_start + 1
    legal_lengths = [length for length in range(5, frame_count + 1, 17) if length >= needed]
    if not legal_lengths:
        return expanded_start, expanded_end, False
    length = legal_lengths[0]
    center = (raw_start + raw_end) // 2
    start = max(0, min(center - length // 2, frame_count - length))
    return start, start + length - 1, True


def audit_motion_quality(
    frames: torch.Tensor,
    fps: float,
    roi_mode: str,
    roi_x: float,
    roi_y: float,
    roi_width: float,
    roi_height: float,
    sharpness_ratio_floor: float,
    temporal_instability_multiplier: float,
    high_motion_delta_floor: float,
    freeze_delta_ceiling: float,
    repair_context_frames: int,
    face_mask: torch.Tensor | None = None,
) -> tuple[bool, int, str, str]:
    """Run dependency-free temporal proxies without claiming face identity accuracy."""
    if not isinstance(frames, torch.Tensor) or frames.ndim != 4:
        raise ValueError("frames must be a ComfyUI IMAGE tensor [frames,H,W,C]")
    frame_count, height, width, _channels = frames.shape
    if frame_count < 2 or height < 2 or width < 2:
        raise ValueError("motion audit requires at least two frames of at least 2x2 pixels")
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError("fps must be finite and greater than zero")
    if roi_mode not in {"full_frame", "manual_static_roi", "connected_mask"}:
        raise ValueError(f"unsupported roi_mode: {roi_mode!r}")
    if roi_mode == "connected_mask" and face_mask is None:
        raise ValueError("roi_mode=connected_mask requires face_mask")
    if not 0.0 < sharpness_ratio_floor <= 1.0:
        raise ValueError("sharpness_ratio_floor must be in (0, 1]")
    if temporal_instability_multiplier < 1.0:
        raise ValueError("temporal_instability_multiplier must be at least 1")
    if high_motion_delta_floor < 0.0 or freeze_delta_ceiling < 0.0:
        raise ValueError("motion thresholds cannot be negative")
    if repair_context_frames < 0:
        raise ValueError("repair_context_frames cannot be negative")

    masks = _prepare_masks(face_mask, frame_count, height, width)
    if roi_mode == "manual_static_roi":
        top, bottom, left, right = _manual_roi_bounds(
            height,
            width,
            roi_x,
            roi_y,
            roi_width,
            roi_height,
        )
    else:
        top, bottom, left, right = 0, height, 0, width

    sharpness = []
    motion = [0.0]
    instability = [0.0, 0.0]
    previous = None
    previous_previous = None
    previous_mask = None
    for frame_index in range(frame_count):
        gray = _frame_to_gray(frames[frame_index])[top:bottom, left:right]
        mask = None
        if masks is not None and roi_mode == "connected_mask":
            mask = masks[frame_index, top:bottom, left:right]

        grad_x = (gray[:, 1:] - gray[:, :-1]).abs()
        grad_y = (gray[1:, :] - gray[:-1, :]).abs()
        weight_x = None if mask is None else mask[:, 1:] * mask[:, :-1]
        weight_y = None if mask is None else mask[1:, :] * mask[:-1, :]
        sharpness.append(
            0.5 * (_weighted_mean(grad_x, weight_x) + _weighted_mean(grad_y, weight_y))
        )

        if previous is not None:
            pair_mask = None
            if mask is not None and previous_mask is not None:
                pair_mask = mask * previous_mask
            motion.append(_weighted_mean((gray - previous).abs(), pair_mask))
        if previous_previous is not None:
            triple_mask = None
            if mask is not None and previous_mask is not None:
                triple_mask = mask * previous_mask
            instability.append(
                _weighted_mean((gray - 2.0 * previous + previous_previous).abs(), triple_mask)
            )

        previous_previous = previous
        previous = gray
        previous_mask = mask

    sharpness_tensor = torch.tensor(sharpness, dtype=torch.float64)
    motion_tensor = torch.tensor(motion, dtype=torch.float64)
    instability_tensor = torch.tensor(instability[:frame_count], dtype=torch.float64)
    median_sharpness = float(torch.median(sharpness_tensor))
    median_motion = float(torch.median(motion_tensor[1:]))
    median_instability = float(torch.median(instability_tensor[2:])) if frame_count > 2 else 0.0
    sharpness_threshold = median_sharpness * sharpness_ratio_floor
    instability_threshold = max(0.01, median_instability * temporal_instability_multiplier)

    risk_indices = []
    per_frame = []
    for frame_index in range(frame_count):
        reasons = []
        low_sharpness = sharpness[frame_index] < sharpness_threshold
        high_motion = motion[frame_index] > high_motion_delta_floor
        unstable = instability_tensor[frame_index] > instability_threshold
        frozen_in_dynamic_sequence = (
            frame_index > 0
            and median_motion > high_motion_delta_floor
            and motion[frame_index] < freeze_delta_ceiling
        )
        if low_sharpness and high_motion:
            reasons.append("high_motion_low_sharpness_proxy")
        if bool(unstable):
            reasons.append("temporal_instability_proxy")
        if frozen_in_dynamic_sequence:
            reasons.append("possible_frozen_frame_in_dynamic_sequence")
        if reasons:
            risk_indices.append(frame_index)
        per_frame.append(
            {
                "frame": frame_index,
                "time_seconds": frame_index / fps,
                "sharpness_proxy": sharpness[frame_index],
                "motion_delta_proxy": motion[frame_index],
                "temporal_instability_proxy": float(instability_tensor[frame_index]),
                "risk_reasons": reasons,
            }
        )

    raw_ranges = _contiguous_ranges(risk_indices)
    ranges = []
    for raw_start, raw_end in raw_ranges:
        repair_start, repair_end, legal = _legal_repair_window(
            raw_start,
            raw_end,
            frame_count,
            repair_context_frames,
        )
        ranges.append(
            {
                "raw_start_frame": raw_start,
                "raw_end_frame": raw_end,
                "raw_start_seconds": raw_start / fps,
                "raw_end_seconds": raw_end / fps,
                "suggested_repair_start_frame": repair_start,
                "suggested_repair_end_frame": repair_end,
                "suggested_length": repair_end - repair_start + 1,
                "suggested_length_is_17n_plus_5": legal,
            }
        )

    scope = (
        "connected_mask"
        if roi_mode == "connected_mask"
        else "manual_static_roi"
        if roi_mode == "manual_static_roi"
        else "full_frame"
    )
    report = {
        "schema": "minimax_h3_motion_quality_audit_t8_v1",
        "status": "risk_detected" if risk_indices else "no_proxy_risk_detected",
        "frame_count": int(frame_count),
        "width": int(width),
        "height": int(height),
        "fps": float(fps),
        "analysis_scope": scope,
        "risk_frame_count": len(risk_indices),
        "risk_range_count": len(ranges),
        "median_sharpness_proxy": median_sharpness,
        "median_motion_delta_proxy": median_motion,
        "median_temporal_instability_proxy": median_instability,
        "thresholds": {
            "sharpness_ratio_floor": sharpness_ratio_floor,
            "sharpness_absolute_threshold": sharpness_threshold,
            "temporal_instability_multiplier": temporal_instability_multiplier,
            "temporal_instability_absolute_threshold": instability_threshold,
            "high_motion_delta_floor": high_motion_delta_floor,
            "freeze_delta_ceiling": freeze_delta_ceiling,
        },
        "risk_ranges": ranges,
        "per_frame": per_frame,
        "identity_metric_valid": False,
        "face_detection_valid": False,
        "quality_guarantee": False,
        "scientific_limits": [
            "Dependency-free proxies do not identify a person or prove face identity.",
            "A static ROI does not track a moving face; connect a reviewed per-frame mask when available.",
            "Sharpness alone cannot distinguish real detail from oversharpening or reduced motion.",
            "Suggested repair windows are plans only and must be reviewed before non-destructive repair.",
        ],
    }
    return bool(risk_indices), len(ranges), canonical_json(ranges), canonical_json(report, indent=2)
