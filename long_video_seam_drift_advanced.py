from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
from typing import Any

import torch
import torch.nn.functional as F


SEAM_DRIFT_SCHEMA = "t8.minimax_h3.long_video_seam_drift.v1"


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)


def _parse_boundaries(value: str, frame_count: int) -> list[int]:
    text = str(value or "").strip()
    if not text:
        return []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"boundary_frames_json is invalid JSON: {error}") from error
    if isinstance(payload, Mapping):
        payload = payload.get("boundary_frames", payload.get("boundaries"))
    if not isinstance(payload, list):
        raise ValueError("boundary_frames_json must be a list or {boundary_frames:[...]}")
    result = sorted({int(item) for item in payload})
    invalid = [item for item in result if not 1 <= item < frame_count]
    if invalid:
        raise ValueError(
            "boundary frame(s) must identify the first frame after a seam: "
            + ", ".join(map(str, invalid))
        )
    return result


def _fit_mask(mask, frames: torch.Tensor) -> torch.Tensor | None:
    if mask is None:
        return None
    if not isinstance(mask, torch.Tensor):
        raise TypeError("person_roi must be a MASK tensor")
    value = mask.detach().to(device=frames.device, dtype=torch.float32)
    if value.ndim == 4 and value.shape[-1] == 1:
        value = value[..., 0]
    if value.ndim != 3:
        raise ValueError("person_roi must have shape [T,H,W], [1,H,W], or [T,H,W,1]")
    if value.shape[0] not in {1, frames.shape[0]}:
        raise ValueError("person_roi frame count must be 1 or match the IMAGE batch")
    if tuple(value.shape[1:]) != tuple(frames.shape[1:3]):
        value = F.interpolate(
            value[:, None], size=tuple(frames.shape[1:3]), mode="bilinear", align_corners=False
        )[:, 0]
    if value.shape[0] == 1:
        value = value.expand(frames.shape[0], -1, -1)
    return value.clamp(0.0, 1.0)


def _weighted_mean_rgb(frames: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    rgb = frames[..., :3].float()
    if mask is None:
        return rgb.mean(dim=(0, 1, 2))
    weights = mask[..., None].float()
    denominator = weights.sum().clamp_min(1e-6)
    return (rgb * weights).sum(dim=(0, 1, 2)) / denominator


def _luma(frames: torch.Tensor) -> torch.Tensor:
    return (
        frames[..., 0].float() * 0.2126
        + frames[..., 1].float() * 0.7152
        + frames[..., 2].float() * 0.0722
    )


def _weighted_luma_mean(frames: torch.Tensor, mask: torch.Tensor | None) -> float:
    values = _luma(frames)
    if mask is None:
        return float(values.mean())
    denominator = mask.sum().clamp_min(1e-6)
    return float((values * mask).sum() / denominator)


def _high_frequency_energy(frame: torch.Tensor) -> float:
    rgb = frame[..., :3].float()
    dx = rgb[:, 1:] - rgb[:, :-1]
    dy = rgb[1:] - rgb[:-1]
    return float((dx.square().mean() + dy.square().mean()) * 0.5)


def _frame_mad(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left[..., :3].float() - right[..., :3].float()).abs().mean())


def _bounded_correction(
    before_mean: torch.Tensor,
    after_mean: torch.Tensor,
    max_gain: float,
    max_offset: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    epsilon = 1e-4
    # Keep texture contrast close to source: use only the square root of the
    # raw channel ratio, then close the remaining low-frequency mean gap with
    # a bounded offset. A full ratio would scale gradient energy twice as far.
    gain = torch.sqrt((before_mean + epsilon) / (after_mean + epsilon)).clamp(
        1.0 / max_gain, max_gain
    )
    offset = (before_mean - after_mean * gain).clamp(-max_offset, max_offset)
    return gain, offset


def process_long_video_seam_drift(
    frames: torch.Tensor,
    boundary_frames_json: str,
    mode: str = "report_only",
    color_contract: str = "sdr_rec709_0_to_1",
    analysis_window_frames: int = 3,
    transition_frames: int = 24,
    scene_cut_threshold: float = 0.18,
    minimum_drift: float = 0.008,
    maximum_gain: float = 1.08,
    maximum_offset: float = 0.04,
    maximum_frame_change: float = 0.06,
    maximum_texture_ratio_deviation: float = 0.12,
    person_roi=None,
) -> tuple[torch.Tensor, str, str]:
    if not isinstance(frames, torch.Tensor) or frames.ndim != 4 or frames.shape[-1] < 3:
        raise ValueError("frames must be an IMAGE batch [T,H,W,C>=3]")
    if mode not in {"report_only", "bounded_candidate_exp"}:
        raise ValueError("mode must be report_only or bounded_candidate_exp")
    if color_contract not in {"sdr_rec709_0_to_1", "unknown_or_hdr"}:
        raise ValueError("unknown color_contract")
    window = int(analysis_window_frames)
    transition = int(transition_frames)
    if not 1 <= window <= 24 or not 1 <= transition <= 240:
        raise ValueError("analysis_window_frames must be 1..24 and transition_frames 1..240")
    values = [
        scene_cut_threshold,
        minimum_drift,
        maximum_gain,
        maximum_offset,
        maximum_frame_change,
        maximum_texture_ratio_deviation,
    ]
    if not all(math.isfinite(float(item)) for item in values):
        raise ValueError("all seam thresholds must be finite")
    if maximum_gain < 1.0 or min(
        scene_cut_threshold,
        minimum_drift,
        maximum_offset,
        maximum_frame_change,
        maximum_texture_ratio_deviation,
    ) < 0:
        raise ValueError("seam thresholds are outside their supported range")

    boundaries = _parse_boundaries(boundary_frames_json, int(frames.shape[0]))
    source = frames.detach()
    finite = bool(torch.isfinite(source).all())
    in_range = finite and float(source.min()) >= 0.0 and float(source.max()) <= 1.0
    roi = _fit_mask(person_roi, source)
    candidate = source.clone() if mode == "bounded_candidate_exp" else frames
    reports: list[dict[str, Any]] = []
    previous_candidate_stop = -1
    applied_count = 0

    for boundary in boundaries:
        before_start = max(0, boundary - window)
        after_stop = min(int(source.shape[0]), boundary + window)
        before = source[before_start:boundary]
        after = source[boundary:after_stop]
        before_mask = roi[before_start:boundary] if roi is not None else None
        after_mask = roi[boundary:after_stop] if roi is not None else None
        before_rgb = _weighted_mean_rgb(before, None)
        after_rgb = _weighted_mean_rgb(after, None)
        before_luma = _weighted_luma_mean(before, None)
        after_luma = _weighted_luma_mean(after, None)
        roi_before_luma = (
            _weighted_luma_mean(before, before_mask) if roi is not None else None
        )
        roi_after_luma = (
            _weighted_luma_mean(after, after_mask) if roi is not None else None
        )
        seam_mad = _frame_mad(source[boundary - 1], source[boundary])
        drift = float((before_rgb - after_rgb).abs().mean())
        after_lumas = _luma(after).mean(dim=(1, 2))
        flash_range = float(after_lumas.max() - after_lumas.min()) if len(after_lumas) else 0.0
        black = max(before_luma, after_luma) < 0.01
        reason = "report_only"
        eligible = True
        if color_contract != "sdr_rec709_0_to_1":
            eligible, reason = False, "abstain_unknown_or_hdr_color_contract"
        elif not finite:
            eligible, reason = False, "abstain_nonfinite_input"
        elif not in_range:
            eligible, reason = False, "abstain_out_of_sdr_range"
        elif black:
            eligible, reason = False, "abstain_black_or_near_black_seam"
        elif seam_mad >= float(scene_cut_threshold):
            eligible, reason = False, "abstain_scene_cut_or_large_flash"
        elif flash_range >= float(scene_cut_threshold) * 0.75:
            eligible, reason = False, "abstain_transient_flash_after_seam"
        elif drift < float(minimum_drift):
            eligible, reason = False, "abstain_no_material_drift"
        elif boundary < previous_candidate_stop:
            eligible, reason = False, "abstain_overlapping_transition"

        gain = torch.ones(3)
        offset = torch.zeros(3)
        corrected_frames = 0
        rolled_back_frames = 0
        max_observed_change = 0.0
        texture_ratios: list[float] = []
        if mode == "bounded_candidate_exp" and eligible:
            gain, offset = _bounded_correction(
                before_rgb, after_rgb, float(maximum_gain), float(maximum_offset)
            )
            stop = min(int(source.shape[0]), boundary + transition)
            for frame_index in range(boundary, stop):
                alpha = 1.0 - (frame_index - boundary) / max(1, transition)
                original = source[frame_index]
                corrected_rgb = (
                    original[..., :3].float() * gain.to(original.device)
                    + offset.to(original.device)
                ).clamp(0.0, 1.0)
                proposed = original.clone()
                proposed[..., :3] = torch.lerp(
                    original[..., :3].float(), corrected_rgb, float(alpha)
                ).to(original.dtype)
                change = float((proposed[..., :3] - original[..., :3]).abs().max())
                before_hf = _high_frequency_energy(original)
                after_hf = _high_frequency_energy(proposed)
                ratio = after_hf / max(before_hf, 1e-12)
                max_observed_change = max(max_observed_change, change)
                texture_ratios.append(ratio)
                if (
                    change > float(maximum_frame_change)
                    or abs(ratio - 1.0) > float(maximum_texture_ratio_deviation)
                ):
                    rolled_back_frames += 1
                    continue
                candidate[frame_index] = proposed
                corrected_frames += 1
            if corrected_frames:
                applied_count += 1
                previous_candidate_stop = stop
                reason = "bounded_candidate_applied"
            else:
                eligible = False
                reason = "abstain_all_candidate_frames_rolled_back"

        post_mad = (
            _frame_mad(candidate[boundary - 1], candidate[boundary])
            if mode == "bounded_candidate_exp"
            else seam_mad
        )
        reports.append(
            {
                "boundary_frame": boundary,
                "before_window": [before_start, boundary],
                "after_window": [boundary, after_stop],
                "eligible": eligible,
                "status": reason,
                "seam_mad_before": seam_mad,
                "seam_mad_after": post_mad,
                "rgb_mean_before": [float(item) for item in before_rgb],
                "rgb_mean_after": [float(item) for item in after_rgb],
                "luma_before": before_luma,
                "luma_after": after_luma,
                "roi_luma_before": roi_before_luma,
                "roi_luma_after": roi_after_luma,
                "drift_mean_abs_rgb": drift,
                "after_flash_range": flash_range,
                "gain": [float(item) for item in gain],
                "offset": [float(item) for item in offset],
                "corrected_frames": corrected_frames,
                "rolled_back_frames": rolled_back_frames,
                "maximum_observed_change": max_observed_change,
                "texture_ratio_min": min(texture_ratios) if texture_ratios else None,
                "texture_ratio_max": max(texture_ratios) if texture_ratios else None,
            }
        )

    status = (
        "source_identity_report_only"
        if mode == "report_only"
        else "candidate_applied"
        if applied_count
        else "source_identity_abstain"
    )
    source_identity = bool(torch.equal(candidate, source))
    report = {
        "schema": SEAM_DRIFT_SCHEMA,
        "status": status,
        "mode": mode,
        "frame_count": int(source.shape[0]),
        "boundary_count": len(boundaries),
        "applied_boundary_count": applied_count,
        "source_identity": source_identity,
        "color_contract": color_contract,
        "finite": finite,
        "in_0_to_1_range": in_range,
        "person_roi_connected": roi is not None,
        "audio_touched": False,
        "detail_generation": False,
        "boundaries": reports,
        "contract": (
            "display-domain bounded low-frequency gain/offset with per-frame rollback; "
            "scene cuts, flashes, black frames, HDR/unknown transfer and unsafe texture changes abstain"
        ),
    }
    report["report_sha256"] = hashlib.sha256(
        json.dumps(report, sort_keys=True, allow_nan=False).encode("utf-8")
    ).hexdigest()
    return candidate, status, _json(report)
