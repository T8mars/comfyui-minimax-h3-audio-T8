from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import torchaudio

from .core import (
    AUDIO_LATENT_FPS,
    FPS,
    align_frame_count,
    classify_h3_vae,
    encode_audio_once,
    fit_audio_latent,
    nested_av_parts,
    validate_audio,
    video_latent_t,
)


MOTION_PLAN_SCHEMA = "h3_t8_motion_recovery_plan/v1"
MOTION_WINDOW_SCHEMA = "h3_t8_motion_window_plan/v1"
ALLOWED_ANALYSIS_MODES = {"report_only", "auto_conservative_exp", "manual_ranges"}
ALLOWED_AUDIO_SEED_MODES = {
    "none_invent_exp": 1.0,
    "follow_original_0p5": 0.5,
    "follow_original_0p7": 0.7,
    "lock_original_0p0": 0.0,
}
ALLOWED_RECOVERY_AUDIO_MODES = {
    "pass1_original",
    "pass2_recovered_exp",
    "blend_exp",
}
RECOVERY_AUDIO_DELIVERY_STATUS = {
    "pass1_original": "stable_default_exact_pass1",
    "pass2_recovered_exp": "diagnostic_only_failed_single_clip_listening",
    "blend_exp": "opt_in_single_clip_pass_at_mix_0p8",
}
RECOVERY_AUDIO_KNOWN_RISK = {
    "pass1_original": "none_observed; deliverable audio bypasses pass-2 recovery",
    "pass2_recovered_exp": (
        "human review heard the middle window suddenly become distant before returning; "
        "do not use as a delivery default"
    ),
    "blend_exp": (
        "one I2VA Mandarin clip sounded normal at pass1_mix=0.8; this does not generalize "
        "to other prompts, windows, voices, or mix values"
    ),
}


def canonical_json(value: Any, *, indent: int | None = None) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        indent=indent,
        separators=None if indent is not None else (",", ":"),
        sort_keys=True,
    )


def _sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _available_system_memory_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except (ImportError, AttributeError, OSError, ValueError):
        return None


def _cpu_frame_memory_estimate(
    *, frame_count: int, height: int, width: int
) -> dict[str, Any]:
    frame_bytes = int(frame_count) * int(height) * int(width) * 3 * 4
    available = _available_system_memory_bytes()
    ratio = None if available is None or available <= 0 else frame_bytes / available
    if ratio is None:
        risk = "unknown"
    elif ratio >= 0.5:
        risk = "high"
    elif ratio >= 0.25:
        risk = "elevated"
    else:
        risk = "low_estimate_only"
    return {
        "float32_frame_mib": round(frame_bytes / (1024**2), 3),
        "available_system_mib": (
            None if available is None else round(available / (1024**2), 3)
        ),
        "available_ratio": None if ratio is None else round(ratio, 6),
        "risk": risk,
        "note": "Estimate excludes model weights, VAE workspaces, tensors and allocator overhead.",
    }


def _signed_plan(plan: dict[str, Any]) -> dict[str, Any]:
    output = dict(plan)
    output.pop("plan_sha256", None)
    output["plan_sha256"] = _sha256(output)
    return output


def validate_motion_plan(plan: Mapping[str, Any], *, allow_window: bool = True) -> dict[str, Any]:
    if not isinstance(plan, Mapping):
        raise ValueError("motion_plan must be a connected H3 motion recovery plan")
    output = dict(plan)
    schema = output.get("schema")
    allowed = {MOTION_PLAN_SCHEMA}
    if allow_window:
        allowed.add(MOTION_WINDOW_SCHEMA)
    if schema not in allowed:
        raise ValueError(f"Unsupported motion plan schema {schema!r}")
    expected_sha = output.get("plan_sha256")
    unsigned = dict(output)
    unsigned.pop("plan_sha256", None)
    actual_sha = _sha256(unsigned)
    if not isinstance(expected_sha, str) or expected_sha != actual_sha:
        raise ValueError("motion plan SHA-256 does not match its payload")
    holds = output.get("holds")
    if not isinstance(holds, list) or not holds:
        raise ValueError("motion plan must contain a non-empty holds list")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in holds):
        raise ValueError("motion plan holds must be positive integers")
    world_length = int(output.get("world_length", -1))
    expanded_length = int(output.get("expanded_length", -1))
    if world_length != len(holds):
        raise ValueError(
            f"motion plan world_length/holds mismatch: {world_length} vs {len(holds)}"
        )
    if expanded_length != sum(holds):
        raise ValueError(
            f"motion plan expanded_length/holds mismatch: {expanded_length} vs {sum(holds)}"
        )
    if align_frame_count(expanded_length) != expanded_length:
        raise ValueError(f"expanded length {expanded_length} is not on the H3 17n+5 grid")
    if float(output.get("fps", 0.0)) != float(FPS):
        raise ValueError(f"motion plan must use native H3 {FPS}fps")
    if schema == MOTION_WINDOW_SCHEMA:
        parent_sha = output.get("parent_plan_sha256")
        if not isinstance(parent_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", parent_sha):
            raise ValueError("window plan must contain a valid parent_plan_sha256")
        parent_world_length = int(output.get("parent_world_length", -1))
        window = output.get("window")
        if parent_world_length < 1 or not isinstance(window, Mapping):
            raise ValueError("window plan is missing its parent length or window payload")
        required = {
            "index",
            "count",
            "source_start",
            "source_end",
            "core_start",
            "core_end",
            "local_core_start",
            "local_core_end",
        }
        if not required.issubset(window):
            raise ValueError("window plan payload is incomplete")
        index = int(window["index"])
        count = int(window["count"])
        source_start = int(window["source_start"])
        source_end = int(window["source_end"])
        core_start = int(window["core_start"])
        core_end = int(window["core_end"])
        if not (0 <= index < count and count >= 1):
            raise ValueError("window plan index/count is invalid")
        if not (
            0 <= source_start <= core_start <= core_end <= source_end < parent_world_length
        ):
            raise ValueError("window plan source/core spans are invalid")
        if world_length != source_end - source_start + 1:
            raise ValueError("window plan world_length does not match its source span")
        if int(window["local_core_start"]) != core_start - source_start or int(
            window["local_core_end"]
        ) != core_end - source_start:
            raise ValueError("window plan local/global core spans do not agree")
    return output


def _frame_count_from_latent_t(latent_t: int) -> int:
    latent_t = int(latent_t)
    if latent_t < 2 or (latent_t - 2) % 5:
        raise ValueError(
            "H3 video latent time must follow 5k+2, corresponding to 17k+5 pixel frames; "
            f"got {latent_t}"
        )
    return ((latent_t - 2) // 5) * 17 + 5


def _token_start_frame(token_index: int) -> int:
    cycle, phase = divmod(int(token_index), 5)
    return cycle * 17 + (0 if phase == 0 else 4 * (phase - 1) + 1)


def _token_spans(latent_t: int, frame_count: int) -> list[tuple[int, int]]:
    spans = []
    for token_index in range(latent_t):
        start = _token_start_frame(token_index)
        end = min(frame_count - 1, _token_start_frame(token_index + 1) - 1)
        if start > end:
            raise ValueError("H3 token/frame mapping produced an empty span")
        spans.append((start, end))
    return spans


def _profile(values: torch.Tensor, order: int) -> torch.Tensor:
    length = int(values.shape[2])
    if length <= order:
        return torch.zeros(length, dtype=torch.float32)
    raw = torch.diff(values, n=order, dim=2).abs().mean(dim=(0, 1, 3, 4))
    lead = order // 2
    output = torch.empty(length, dtype=torch.float32)
    output[lead : lead + raw.numel()] = raw.float()
    output[:lead] = raw[0]
    output[lead + raw.numel() :] = raw[-1]
    return output


def _phase_normalize(profile: torch.Tensor) -> torch.Tensor:
    output = profile.clone().float()
    for phase in range(5):
        phase_values = output[phase::5]
        if phase_values.numel() == 0:
            continue
        mean = float(phase_values.mean().item())
        if mean > 1e-12:
            output[phase::5] /= mean
    return output


def _trajectory_profile(values: torch.Tensor) -> torch.Tensor:
    energy = values.abs().mean(dim=(0, 1)).float()
    energy = energy - energy.amin(dim=(1, 2), keepdim=True)
    totals = energy.sum(dim=(1, 2)).clamp_min(1e-8)
    ys = torch.arange(energy.shape[1], dtype=torch.float32).view(1, -1, 1)
    xs = torch.arange(energy.shape[2], dtype=torch.float32).view(1, 1, -1)
    cy = (energy * ys).sum(dim=(1, 2)) / totals
    cx = (energy * xs).sum(dim=(1, 2)) / totals
    path = torch.stack((cy, cx), dim=1)
    if path.shape[0] <= 3:
        return torch.zeros(path.shape[0], dtype=torch.float32)
    raw = torch.linalg.vector_norm(torch.diff(path, n=3, dim=0), dim=1)
    output = torch.empty(path.shape[0], dtype=torch.float32)
    output[1 : 1 + raw.numel()] = raw
    output[:1] = raw[0]
    output[1 + raw.numel() :] = raw[-1]
    return output


def _robust_positive_z(values: torch.Tensor) -> torch.Tensor:
    values = values.float()
    median = values.median()
    mad = (values - median).abs().median()
    scale = torch.clamp(mad * 1.4826, min=1e-6)
    return torch.relu((values - median) / scale).clamp(max=20.0)


def _decoded_residual_motion(frames: torch.Tensor, analysis_side: int = 96) -> torch.Tensor:
    """Cheap global-translation-compensated frame residual on CPU.

    This is a diagnostic baseline, not optical flow. Each frame is compared
    against small integer translations of the preceding frame and the least
    residual is retained, which discounts simple camera pans without adding a
    model dependency.
    """
    x = frames.detach().float().cpu()[..., :3].movedim(-1, 1)
    height, width = int(x.shape[-2]), int(x.shape[-1])
    scale = min(1.0, float(analysis_side) / max(height, width))
    target_h = max(8, int(round(height * scale)))
    target_w = max(8, int(round(width * scale)))
    x = F.interpolate(x, size=(target_h, target_w), mode="area")
    gray = 0.2126 * x[:, 0] + 0.7152 * x[:, 1] + 0.0722 * x[:, 2]
    output = torch.zeros(gray.shape[0], dtype=torch.float32)
    for frame_index in range(1, gray.shape[0]):
        current = gray[frame_index]
        previous = gray[frame_index - 1]
        candidates = []
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                shifted = torch.roll(previous, shifts=(dy, dx), dims=(0, 1))
                y0 = max(0, dy)
                y1 = min(target_h, target_h + dy)
                x0 = max(0, dx)
                x1 = min(target_w, target_w + dx)
                if y1 <= y0 or x1 <= x0:
                    continue
                candidates.append((current[y0:y1, x0:x1] - shifted[y0:y1, x0:x1]).abs().mean())
        output[frame_index] = torch.stack(candidates).amin() if candidates else 0.0
    if output.numel() > 1:
        output[0] = output[1]
    return output


def _frame_profile_from_tokens(
    token_values: torch.Tensor,
    spans: list[tuple[int, int]],
    frame_count: int,
) -> torch.Tensor:
    output = torch.zeros(frame_count, dtype=torch.float32)
    for token_index, (start, end) in enumerate(spans):
        output[start : end + 1] = token_values[token_index]
    return output


def _token_profile_from_frames(
    frame_values: torch.Tensor,
    spans: list[tuple[int, int]],
) -> torch.Tensor:
    return torch.stack([frame_values[start : end + 1].mean() for start, end in spans])


def _ranges_from_mask(mask: torch.Tensor) -> list[tuple[int, int]]:
    indices = torch.nonzero(mask, as_tuple=False).flatten().tolist()
    if not indices:
        return []
    ranges: list[tuple[int, int]] = []
    start = previous = int(indices[0])
    for index in indices[1:]:
        index = int(index)
        if index != previous + 1:
            ranges.append((start, previous))
            start = index
        previous = index
    ranges.append((start, previous))
    return ranges


_MANUAL_RANGE = re.compile(
    r"^\s*(?P<start>\d+(?:\.\d+)?)(?P<start_s>s?)\s*-\s*"
    r"(?P<end>\d+(?:\.\d+)?)(?P<end_s>s?)(?:\s*:\s*(?P<hold>\d+))?\s*$"
)


def _parse_manual_ranges(
    value: str,
    *,
    frame_count: int,
    fps: float,
    default_hold: int,
) -> list[tuple[int, int, int]]:
    if not value.strip():
        raise ValueError("manual_ranges mode requires at least one range")
    output: list[tuple[int, int, int]] = []
    for raw in value.split(","):
        match = _MANUAL_RANGE.match(raw)
        if not match:
            raise ValueError(
                f"Invalid manual motion range {raw!r}; use '36-60:4' or '1.5s-2.4s:3'"
            )
        start = float(match.group("start"))
        end = float(match.group("end"))
        if bool(match.group("start_s")) != bool(match.group("end_s")):
            raise ValueError("Both endpoints of a manual range must use the same frame/second unit")
        if match.group("start_s"):
            start *= fps
            end *= fps
        start_i = max(0, min(frame_count - 1, int(math.floor(start))))
        end_i = max(0, min(frame_count - 1, int(math.ceil(end))))
        if end_i < start_i:
            start_i, end_i = end_i, start_i
        hold = int(match.group("hold") or default_hold)
        if hold < 2 or hold > 8:
            raise ValueError("manual hold must be between 2 and 8")
        output.append((start_i, end_i, hold))
    return output


def _apply_token_ramp(token_holds: torch.Tensor) -> torch.Tensor:
    output = token_holds.clone().to(torch.int64)
    peak = int(output.max().item())
    for _ in range(max(0, peak - 1)):
        left = torch.cat((torch.ones(1, dtype=output.dtype), output[:-1]))
        right = torch.cat((output[1:], torch.ones(1, dtype=output.dtype)))
        output = torch.maximum(output, torch.maximum(left, right) - 1)
    return output.clamp(min=1)


def _legalize_holds(holds: list[int]) -> tuple[list[int], int, int | None]:
    output = [int(value) for value in holds]
    raw_length = sum(output)
    legal_length = align_frame_count(raw_length)
    padding = legal_length - raw_length
    anchor = None
    if padding:
        hot = [index for index, value in enumerate(output) if value > 1]
        anchor = hot[-1] if hot else len(output) - 1
        output[anchor] += padding
    return output, padding, anchor


def _plan_ranges(holds: list[int], frame_scores: torch.Tensor) -> list[dict[str, Any]]:
    mask = torch.tensor([value > 1 for value in holds], dtype=torch.bool)
    output = []
    for start, end in _ranges_from_mask(mask):
        output.append(
            {
                "start_frame": start,
                "end_frame": end,
                "max_hold": max(holds[start : end + 1]),
                "peak_score": round(float(frame_scores[start : end + 1].max().item()), 6),
            }
        )
    return output


def _heat_preview(frames: torch.Tensor, scores: torch.Tensor, threshold: float) -> torch.Tensor:
    output = frames.detach().float().cpu().clone()
    strength = (scores / max(float(threshold), 1e-6)).clamp(0.0, 2.0) * 0.28
    red = output.new_tensor([1.0, 0.08, 0.02])
    output[..., :3] = output[..., :3] * (1.0 - strength[:, None, None, None])
    output[..., :3] += red * strength[:, None, None, None]
    bar_h = max(2, int(output.shape[1]) // 48)
    bar = (scores / max(float(scores.max().item()), 1e-6)).clamp(0.0, 1.0)
    output[:, -bar_h:, :, :3] = red * bar[:, None, None, None]
    return output.clamp(0.0, 1.0)


def analyze_motion_overload(
    av_latent: Mapping[str, Any],
    frames: torch.Tensor,
    mode: str,
    manual_ranges: str,
    overload_threshold: float,
    minimum_profile_contrast: float,
    minimum_residual_motion: float,
    max_hold: int,
    bridge_tokens: int,
    minimum_hot_frames: int,
    fps: float,
) -> tuple[dict[str, Any], torch.Tensor, bool, int, str]:
    if mode not in ALLOWED_ANALYSIS_MODES:
        raise ValueError(f"Unsupported analysis mode {mode!r}")
    if float(fps) != float(FPS):
        raise ValueError(f"MiniMax H3 motion recovery requires native {FPS}fps")
    if not isinstance(frames, torch.Tensor) or frames.ndim != 4 or frames.shape[-1] < 3:
        raise ValueError("frames must be a connected IMAGE [T,H,W,C]")
    if not torch.isfinite(frames).all():
        raise ValueError("frames contain NaN or Inf")
    video, _audio = nested_av_parts(dict(av_latent))
    video = video.detach().float().cpu()
    frame_count = _frame_count_from_latent_t(video.shape[2])
    if int(frames.shape[0]) != frame_count:
        raise ValueError(
            f"decoded frames/latent clock mismatch: frames={frames.shape[0]}, latent={frame_count}"
        )
    height, width = int(frames.shape[1]), int(frames.shape[2])
    if height % 32 or width % 32:
        raise ValueError(
            "decoded MiniMax H3 frames must use a canvas divisible by 32; "
            f"got {width}x{height}"
        )
    if tuple(video.shape[-2:]) != (height // 16, width // 16):
        raise ValueError(
            "decoded frames and video latent spatial clocks do not match: "
            f"frames={width}x{height}, latent={tuple(video.shape[-2:])}"
        )
    if int(max_hold) < 2 or int(max_hold) > 8:
        raise ValueError("max_hold must be between 2 and 8")
    spans = _token_spans(video.shape[2], frame_count)
    d1 = _phase_normalize(_profile(video, 1))
    d3 = _phase_normalize(_profile(video, 3))
    trajectory = _phase_normalize(_trajectory_profile(video))
    residual_frames = _decoded_residual_motion(frames)
    residual = _token_profile_from_frames(residual_frames, spans)
    composite_tokens = (
        0.20 * _robust_positive_z(d1)
        + 0.35 * _robust_positive_z(d3)
        + 0.20 * _robust_positive_z(trajectory)
        + 0.25 * _robust_positive_z(residual)
    )
    composite_frames = _frame_profile_from_tokens(composite_tokens, spans, frame_count)
    profile_contrast = float(
        composite_tokens.max().item() / max(float(composite_tokens.mean().item()), 1e-6)
    )
    residual_peak = float(residual_frames.max().item())

    recommended_token_holds = torch.ones(video.shape[2], dtype=torch.int64)
    auto_gate = (
        float(composite_tokens.max().item()) >= float(overload_threshold)
        and profile_contrast >= float(minimum_profile_contrast)
        and residual_peak >= float(minimum_residual_motion)
    )
    if auto_gate:
        hot = composite_tokens >= float(overload_threshold)
        hot_indices = torch.nonzero(hot, as_tuple=False).flatten().tolist()
        if bridge_tokens > 0:
            for left, right in zip(hot_indices[:-1], hot_indices[1:]):
                if 1 < right - left <= int(bridge_tokens):
                    hot[left : right + 1] = True
        recommended_token_holds[hot] = int(max_hold)
        recommended_token_holds = _apply_token_ramp(recommended_token_holds)
    recommended_holds = [
        int(recommended_token_holds[token_index].item())
        for token_index, (start, end) in enumerate(spans)
        for _ in range(end - start + 1)
    ]
    if len(recommended_holds) != frame_count:
        raise RuntimeError("internal H3 token/frame plan length mismatch")
    recommended_mask = torch.tensor([value > 1 for value in recommended_holds])
    recommended_hot_frames = int(recommended_mask.sum().item())
    if recommended_hot_frames < int(minimum_hot_frames):
        recommended_holds = [1] * frame_count
        auto_gate = False

    if mode == "manual_ranges":
        selected_holds = [1] * frame_count
        for start, end, hold in _parse_manual_ranges(
            manual_ranges,
            frame_count=frame_count,
            fps=fps,
            default_hold=max_hold,
        ):
            for frame_index in range(start, end + 1):
                selected_holds[frame_index] = max(selected_holds[frame_index], hold)
        status = "ready"
        decision_reason = "manual_ranges"
    elif mode == "auto_conservative_exp" and auto_gate:
        selected_holds = recommended_holds
        status = "ready"
        decision_reason = "automatic_gate_passed"
    elif mode == "auto_conservative_exp":
        selected_holds = [1] * frame_count
        status = "abstained"
        decision_reason = "automatic_gate_failed"
    else:
        selected_holds = [1] * frame_count
        status = "report_only"
        decision_reason = "analysis_only"

    selected_holds, legal_padding, pad_anchor = _legalize_holds(selected_holds)
    recommended_legal, recommended_padding, recommended_anchor = _legalize_holds(
        recommended_holds
    )
    expanded_length = sum(selected_holds)
    pixel_megapixels = float(frames.shape[1] * frames.shape[2]) / 1_000_000.0
    estimated_video_tokens = video_latent_t(expanded_length) * (frames.shape[1] // 32) * (
        frames.shape[2] // 32
    )
    should_repair = status == "ready" and expanded_length > frame_count
    plan = _signed_plan(
        {
            "schema": MOTION_PLAN_SCHEMA,
            "status": status,
            "mode": mode,
            "decision_reason": decision_reason,
            "fps": float(fps),
            "world_length": frame_count,
            "expanded_length": expanded_length,
            "holds": selected_holds,
            "legal_padding_frames": legal_padding,
            "legal_padding_anchor": pad_anchor,
            "recommended": {
                "holds": recommended_legal,
                "expanded_length": sum(recommended_legal),
                "legal_padding_frames": recommended_padding,
                "legal_padding_anchor": recommended_anchor,
                "ranges": _plan_ranges(recommended_legal, composite_frames),
            },
            "selected_ranges": _plan_ranges(selected_holds, composite_frames),
            "source": {
                "video_latent_shape": list(video.shape),
                "frame_shape": list(frames.shape),
                "canvas_width": int(frames.shape[2]),
                "canvas_height": int(frames.shape[1]),
                "pixel_megapixels": round(pixel_megapixels, 6),
            },
            "analysis": {
                "overload_threshold": float(overload_threshold),
                "minimum_profile_contrast": float(minimum_profile_contrast),
                "minimum_residual_motion": float(minimum_residual_motion),
                "profile_contrast": round(profile_contrast, 6),
                "residual_motion_peak": round(residual_peak, 8),
                "automatic_gate": bool(auto_gate),
                "max_hold": int(max_hold),
                "bridge_tokens": int(bridge_tokens),
                "minimum_hot_frames": int(minimum_hot_frames),
                "curves": {
                    "latent_d1": [round(float(value), 6) for value in d1],
                    "latent_d3": [round(float(value), 6) for value in d3],
                    "trajectory_d3": [round(float(value), 6) for value in trajectory],
                    "decoded_residual_motion": [
                        round(float(value), 8) for value in residual_frames
                    ],
                    "composite_frame_score": [
                        round(float(value), 6) for value in composite_frames
                    ],
                },
            },
            "estimate": {
                "source_video_latent_tokens": int(video.shape[2]),
                "expanded_video_latent_tokens": video_latent_t(expanded_length),
                "approx_spatiotemporal_tokens": int(estimated_video_tokens),
                "frame_multiplier": round(expanded_length / frame_count, 6),
                "cpu_held_frames": _cpu_frame_memory_estimate(
                    frame_count=expanded_length,
                    height=height,
                    width=width,
                ),
                "requires_second_pass": bool(should_repair),
                "memory_safe": False,
                "note": (
                    "This is a structural estimate, not a universal VRAM guarantee. "
                    "Use Motion Segment Plan before a large expanded pass."
                ),
            },
            "audio_default": "pass1_original",
        }
    )
    preview = _heat_preview(frames, composite_frames, overload_threshold)
    report = {
        "schema_version": 1,
        "node": "MiniMaxH3MotionOverloadAnalyzeT8Advanced",
        "status": status,
        "should_repair": should_repair,
        "decision_reason": decision_reason,
        "world_length": frame_count,
        "expanded_length": expanded_length,
        "recommended_expanded_length": sum(recommended_legal),
        "profile_contrast": profile_contrast,
        "residual_motion_peak": residual_peak,
        "automatic_gate": auto_gate,
        "cpu_held_frames": plan["estimate"]["cpu_held_frames"],
        "selected_ranges": plan["selected_ranges"],
        "recommended_ranges": plan["recommended"]["ranges"],
        "plan_sha256": plan["plan_sha256"],
        "claims": {
            "read_only": True,
            "physical_jerk": False,
            "quality_improvement": False,
            "memory_safe": False,
        },
    }
    return plan, preview, should_repair, expanded_length, canonical_json(report, indent=2)


def _runs(holds: list[int]) -> list[tuple[int, int]]:
    output: list[list[int]] = []
    for hold in holds:
        if output and output[-1][0] == hold:
            output[-1][1] += 1
        else:
            output.append([int(hold), 1])
    return [(hold, count) for hold, count in output]


def _phase_advance(n_fft: int, hop: int) -> torch.Tensor:
    return torch.linspace(0.0, math.pi * hop, n_fft // 2 + 1)[..., None]


def _fit_audio_samples(audio: Mapping[str, Any], target_samples: int) -> dict[str, Any]:
    waveform, sample_rate = validate_audio(audio)
    waveform = waveform.detach().float().cpu()
    if waveform.shape[-1] > target_samples:
        waveform = waveform[..., :target_samples]
    elif waveform.shape[-1] < target_samples:
        waveform = F.pad(waveform, (0, target_samples - waveform.shape[-1]))
    return {"waveform": waveform.contiguous(), "sample_rate": sample_rate}


def _stretch_audio_by_holds(
    audio: Mapping[str, Any],
    holds: list[int],
    *,
    inverse: bool,
    fps: float,
) -> dict[str, Any]:
    waveform, sample_rate = validate_audio(audio)
    waveform = waveform.detach().float().cpu()
    batch, channels, sample_count = waveform.shape
    flat = waveform.reshape(batch * channels, sample_count)
    n_fft = 2048
    hop = 512
    window = torch.hann_window(n_fft)
    phase_advance = _phase_advance(n_fft, hop)
    samples_per_frame = sample_rate / float(fps)
    cursor = 0.0
    pieces = []
    for hold, count in _runs(holds):
        if inverse:
            source_count = hold * count * samples_per_frame
            target_count = int(round(count * samples_per_frame))
            rate = float(hold)
        else:
            source_count = count * samples_per_frame
            target_count = int(round(hold * count * samples_per_frame))
            rate = 1.0 / float(hold)
        start = int(round(cursor))
        end = int(round(cursor + source_count))
        cursor += source_count
        segment = flat[:, start:min(end, sample_count)]
        if segment.shape[-1] == 0:
            pieces.append(torch.zeros(flat.shape[0], target_count))
            continue
        if hold > 1 and segment.shape[-1] >= n_fft:
            spectrum = torch.stft(
                segment,
                n_fft,
                hop,
                window=window,
                return_complex=True,
            )
            spectrum = torchaudio.functional.phase_vocoder(
                spectrum,
                rate,
                phase_advance,
            )
            segment = torch.istft(
                spectrum,
                n_fft,
                hop,
                window=window,
                length=target_count,
            )
        else:
            segment = F.interpolate(
                segment[:, None], size=target_count, mode="linear", align_corners=False
            )[:, 0]
        if segment.shape[-1] < target_count:
            segment = F.pad(segment, (0, target_count - segment.shape[-1]))
        pieces.append(segment[:, :target_count])
    output = torch.cat(pieces, dim=-1) if pieces else flat[:, :0]
    expected = round((len(holds) if inverse else sum(holds)) / fps * sample_rate)
    if output.shape[-1] > expected:
        output = output[:, :expected]
    elif output.shape[-1] < expected:
        output = F.pad(output, (0, expected - output.shape[-1]))
    return {
        "waveform": output.reshape(batch, channels, -1).contiguous(),
        "sample_rate": sample_rate,
    }


def prepare_motion_retiming(
    frames: torch.Tensor,
    motion_plan: Mapping[str, Any],
    video_vae,
    audio_vae,
    source_audio: Mapping[str, Any],
    audio_seed_mode: str,
) -> tuple[dict[str, Any], torch.Tensor, dict[str, Any], dict[str, Any], str]:
    plan = validate_motion_plan(motion_plan)
    if plan["status"] != "ready" or plan["expanded_length"] <= plan["world_length"]:
        raise ValueError(
            "motion plan is not ready for retiming; use auto_conservative_exp/manual_ranges "
            "and confirm that it selected a non-empty repair range"
        )
    if not isinstance(frames, torch.Tensor) or frames.ndim != 4:
        raise ValueError("frames must be a connected IMAGE")
    if int(frames.shape[0]) != int(plan["world_length"]):
        raise ValueError("frames do not match motion plan world_length")
    source_shape = plan.get("source", {}).get("frame_shape")
    if source_shape and list(frames.shape) != source_shape:
        raise ValueError("frames shape does not match the source shape signed by the motion plan")
    if classify_h3_vae(video_vae) != "video":
        raise ValueError("video_vae must be a MiniMax H3 video VAE")
    if audio_seed_mode not in ALLOWED_AUDIO_SEED_MODES:
        raise ValueError(f"Unsupported audio_seed_mode {audio_seed_mode!r}")
    holds = plan["holds"]
    indices = torch.tensor(
        [frame_index for frame_index, hold in enumerate(holds) for _ in range(hold)],
        dtype=torch.long,
    )
    source_cpu = frames.detach().cpu()
    smeared_frames = source_cpu.index_select(0, indices)
    video_latent = video_vae.encode(smeared_frames)
    if not isinstance(video_latent, torch.Tensor) or video_latent.ndim != 5:
        raise ValueError("MiniMax H3 video VAE did not return [B,24,T,H,W]")
    expected_video_t = video_latent_t(plan["expanded_length"])
    if tuple(video_latent.shape[:2]) != (1, 24) or video_latent.shape[2] != expected_video_t:
        raise ValueError(
            "Encoded smeared video does not match the H3 clock: "
            f"shape={tuple(video_latent.shape)}, expected T={expected_video_t}"
        )
    audio_strength = float(ALLOWED_AUDIO_SEED_MODES[audio_seed_mode])
    target_audio_t = round(plan["expanded_length"] / FPS * AUDIO_LATENT_FPS)
    if audio_seed_mode == "none_invent_exp":
        smeared_audio = {
            "waveform": torch.zeros(1, 2, round(plan["expanded_length"] / FPS * 32000)),
            "sample_rate": 32000,
        }
        audio_latent = video_latent.new_zeros((1, 32, 2, target_audio_t))
    else:
        if source_audio is None:
            raise ValueError(
                f"audio_seed_mode={audio_seed_mode} requires connected source_audio"
            )
        if classify_h3_vae(audio_vae) != "audio":
            raise ValueError("audio_vae must be a MiniMax H3 audio VAE")
        validate_audio(source_audio, "source_audio")
        smeared_audio = _stretch_audio_by_holds(
            source_audio,
            holds,
            inverse=False,
            fps=FPS,
        )
        encoded_audio = encode_audio_once(audio_vae, smeared_audio)
        template = video_latent.new_zeros((1, 32, 2, target_audio_t))
        audio_latent = fit_audio_latent(encoded_audio, template)
    import comfy.nested_tensor

    output = {
        "samples": comfy.nested_tensor.NestedTensor((video_latent, audio_latent)),
        "noise_mask": comfy.nested_tensor.NestedTensor(
            (torch.ones_like(video_latent), torch.full_like(audio_latent, audio_strength))
        ),
        "h3_t8_motion_plan_sha256": plan["plan_sha256"],
        "h3_t8_motion_world_length": int(plan["world_length"]),
        "h3_t8_motion_expanded_length": int(plan["expanded_length"]),
    }
    report = {
        "schema_version": 1,
        "node": "MiniMaxH3MotionRetimingPrepareT8Advanced",
        "status": "ready",
        "plan_sha256": plan["plan_sha256"],
        "world_length": plan["world_length"],
        "expanded_length": plan["expanded_length"],
        "video_latent_shape": list(video_latent.shape),
        "audio_latent_shape": list(audio_latent.shape),
        "audio_seed_mode": audio_seed_mode,
        "audio_noise_strength": audio_strength,
        "audio_warning": (
            "The smeared audio seed is a phase-vocoder guide for the shared AV Transformer. "
            "The safe final default remains the original pass-1 audio."
        ),
    }
    return output, smeared_frames, smeared_audio, plan, canonical_json(report, indent=2)


def compose_motion_recovery(
    av_latent: Mapping[str, Any],
    sigmas: torch.Tensor,
    motion_plan: Mapping[str, Any],
    mode: str,
    denoise_fraction: float,
    minimum_second_pass_nfe: int,
) -> tuple[dict[str, Any], torch.Tensor, dict[str, Any], int, str]:
    plan = validate_motion_plan(motion_plan)
    if not isinstance(av_latent, Mapping) or "samples" not in av_latent:
        raise ValueError("av_latent must be the output of Motion Retiming Prepare")
    if av_latent.get("h3_t8_motion_plan_sha256") != plan["plan_sha256"]:
        raise ValueError("prepared AV latent and motion plan SHA-256 do not match")
    if not isinstance(sigmas, torch.Tensor) or sigmas.ndim != 1 or sigmas.numel() < 2:
        raise ValueError("sigmas must be a one-dimensional schedule with at least one model call")
    if not torch.isfinite(sigmas).all() or bool(torch.any(sigmas[1:] > sigmas[:-1])):
        raise ValueError("sigmas must be finite and monotonically non-increasing")
    if abs(float(sigmas[-1].item())) > 1e-7:
        raise ValueError("motion recovery sigma schedule must end at zero")
    if mode not in {"report_only", "apply_exp"}:
        raise ValueError(f"Unsupported composer mode {mode!r}")
    denoise_fraction = float(denoise_fraction)
    if not 0.05 <= denoise_fraction <= 1.0:
        raise ValueError("denoise_fraction must be between 0.05 and 1.0")
    input_nfe = int(sigmas.numel() - 1)
    run_nfe = max(int(minimum_second_pass_nfe), int(round(input_nfe * denoise_fraction)))
    run_nfe = min(input_nfe, run_nfe)
    output_sigmas = sigmas if mode == "report_only" else sigmas[input_nfe - run_nfe :]
    actual_nfe = input_nfe if mode == "report_only" else run_nfe
    video, audio = nested_av_parts(dict(av_latent))
    if video.shape[2] != video_latent_t(plan["expanded_length"]):
        raise ValueError("prepared AV video latent no longer matches the motion plan")
    expected_audio_t = round(plan["expanded_length"] / FPS * AUDIO_LATENT_FPS)
    if audio.shape[-1] != expected_audio_t:
        raise ValueError("prepared AV audio latent no longer matches the motion plan")
    report = {
        "schema_version": 1,
        "node": "MiniMaxH3MotionRecoveryComposerT8Advanced",
        "status": "report_only" if mode == "report_only" else "ready",
        "plan_sha256": plan["plan_sha256"],
        "input_nfe": input_nfe,
        "actual_second_pass_nfe": actual_nfe,
        "denoise_fraction": denoise_fraction,
        "expanded_length": plan["expanded_length"],
        "automatic_stacking": False,
        "quality_improvement": False,
        "memory_safe": False,
    }
    return dict(av_latent), output_sigmas, plan, actual_nfe, canonical_json(report, indent=2)


def recover_motion_av(
    generated_frames: torch.Tensor,
    generated_audio: Mapping[str, Any],
    pass1_audio: Mapping[str, Any],
    motion_plan: Mapping[str, Any],
    audio_mode: str,
    pass1_mix: float,
) -> tuple[torch.Tensor, dict[str, Any], str]:
    plan = validate_motion_plan(motion_plan)
    if audio_mode not in ALLOWED_RECOVERY_AUDIO_MODES:
        raise ValueError(f"Unsupported audio_mode {audio_mode!r}")
    if not 0.0 <= float(pass1_mix) <= 1.0:
        raise ValueError("pass1_mix must be between 0 and 1")
    if not isinstance(generated_frames, torch.Tensor) or generated_frames.ndim != 4:
        raise ValueError("generated_frames must be a connected IMAGE")
    if int(generated_frames.shape[0]) != int(plan["expanded_length"]):
        raise ValueError(
            "generated frame count does not match the expanded motion plan: "
            f"{generated_frames.shape[0]} vs {plan['expanded_length']}"
        )
    starts = []
    cursor = 0
    for hold in plan["holds"]:
        starts.append(cursor)
        cursor += hold
    recovered_frames = generated_frames.index_select(
        0, torch.tensor(starts, dtype=torch.long, device=generated_frames.device)
    )
    pass1_waveform, pass1_rate = validate_audio(pass1_audio, "pass1_audio")
    target_pass1_samples = round(plan["world_length"] / FPS * pass1_rate)
    original = _fit_audio_samples(pass1_audio, target_pass1_samples)
    vocoder_used = False
    if audio_mode == "pass1_original":
        output_audio = original
    else:
        validate_audio(generated_audio, "generated_audio")
        recovered_pass2 = _stretch_audio_by_holds(
            generated_audio,
            plan["holds"],
            inverse=True,
            fps=FPS,
        )
        vocoder_used = True
        if audio_mode == "pass2_recovered_exp":
            output_audio = recovered_pass2
        else:
            generated_waveform, generated_rate = validate_audio(recovered_pass2)
            reference = original["waveform"]
            if pass1_rate != generated_rate:
                reference = torchaudio.functional.resample(reference, pass1_rate, generated_rate)
            target = generated_waveform.shape[-1]
            if reference.shape[-1] > target:
                reference = reference[..., :target]
            elif reference.shape[-1] < target:
                reference = F.pad(reference, (0, target - reference.shape[-1]))
            if reference.shape[1] != generated_waveform.shape[1]:
                reference = reference[:, :1].expand(-1, generated_waveform.shape[1], -1)
            mixed = (
                float(pass1_mix) * reference
                + (1.0 - float(pass1_mix)) * generated_waveform
            )
            output_audio = {"waveform": mixed.contiguous(), "sample_rate": generated_rate}
    output_audio["h3_t8_motion_audio_mode"] = audio_mode
    output_audio["h3_t8_motion_plan_sha256"] = plan["plan_sha256"]
    report = {
        "schema_version": 1,
        "node": "MiniMaxH3MotionRecoverAVT8Advanced",
        "status": "ok",
        "plan_sha256": plan["plan_sha256"],
        "input_frames": int(generated_frames.shape[0]),
        "output_frames": int(recovered_frames.shape[0]),
        "audio_mode": audio_mode,
        "pass1_mix": float(pass1_mix),
        "phase_vocoder_used": vocoder_used,
        "safe_default": audio_mode == "pass1_original",
        "delivery_status": RECOVERY_AUDIO_DELIVERY_STATUS[audio_mode],
        "known_audio_risk": RECOVERY_AUDIO_KNOWN_RISK[audio_mode],
        "output_audio_samples": int(output_audio["waveform"].shape[-1]),
        "output_audio_rate": int(output_audio["sample_rate"]),
    }
    return recovered_frames, output_audio, canonical_json(report, indent=2)


def route_automatic_motion_recovery(
    should_repair: bool,
    baseline_frames: torch.Tensor,
    baseline_audio: Mapping[str, Any],
    motion_plan: Mapping[str, Any],
    repaired_frames: torch.Tensor | None,
    repaired_audio: Mapping[str, Any] | None,
) -> tuple[torch.Tensor, dict[str, Any], bool, str]:
    """Select the baseline or repaired branch after ComfyUI lazy evaluation.

    The caller is responsible for declaring the repaired inputs as lazy.  When the
    analyzer abstains those values therefore remain ``None`` and the expensive
    second-pass graph is never requested by ComfyUI.
    """
    plan = validate_motion_plan(motion_plan, allow_window=False)
    expected_repair = (
        plan["status"] == "ready"
        and int(plan["expanded_length"]) > int(plan["world_length"])
    )
    requested_repair = bool(should_repair)
    if requested_repair != expected_repair:
        raise ValueError(
            "should_repair does not match the signed analyzer plan; reconnect both "
            "outputs from the same Motion Overload Analyze node"
        )
    if not isinstance(baseline_frames, torch.Tensor) or baseline_frames.ndim != 4:
        raise ValueError("baseline_frames must be a connected IMAGE")
    if int(baseline_frames.shape[0]) != int(plan["world_length"]):
        raise ValueError("baseline_frames do not match the signed analyzer plan")
    validate_audio(baseline_audio, "baseline_audio")

    if requested_repair:
        if not isinstance(repaired_frames, torch.Tensor) or repaired_frames.ndim != 4:
            raise ValueError("repaired_frames must be connected when should_repair is true")
        if tuple(repaired_frames.shape) != tuple(baseline_frames.shape):
            raise ValueError("repaired_frames must match the baseline frame clock and canvas")
        if not torch.isfinite(repaired_frames).all():
            raise ValueError("repaired_frames contain NaN or Inf")
        if repaired_audio is None:
            raise ValueError("repaired_audio must be connected when should_repair is true")
        validate_audio(repaired_audio, "repaired_audio")
        frames_out = repaired_frames
        audio_out = dict(repaired_audio)
        branch = "repair"
    else:
        # Preserve the actual objects on ABSTAIN.  Besides avoiding pass 2, this
        # prevents unnecessary CPU copies or float operations on the safe audio.
        frames_out = baseline_frames
        audio_out = baseline_audio if isinstance(baseline_audio, dict) else dict(baseline_audio)
        branch = "abstain_passthrough"

    report = {
        "schema_version": 1,
        "node": "MiniMaxH3MotionAutoGateT8Advanced",
        "status": "repaired" if requested_repair else "abstained",
        "selected_branch": branch,
        "should_repair": requested_repair,
        "second_pass_requested": requested_repair,
        "baseline_object_passthrough": not requested_repair,
        "plan_sha256": plan["plan_sha256"],
        "decision_reason": plan["decision_reason"],
        "world_length": int(plan["world_length"]),
    }
    return frames_out, audio_out, requested_repair, canonical_json(report, indent=2)


def _window_plan_from_parent(
    parent: dict[str, Any],
    *,
    start: int,
    end: int,
    core_start: int,
    core_end: int,
    window_index: int,
    window_count: int,
    hot_in: bool,
    hot_out: bool,
) -> dict[str, Any]:
    local_holds = []
    for frame_index in range(start, end + 1):
        if core_start <= frame_index <= core_end:
            local_holds.append(int(parent["holds"][frame_index]))
        else:
            local_holds.append(1)
    local_holds, padding, anchor = _legalize_holds(local_holds)
    return _signed_plan(
        {
            "schema": MOTION_WINDOW_SCHEMA,
            "status": "ready",
            "mode": "window",
            "decision_reason": "budgeted_segment",
            "fps": float(FPS),
            "world_length": len(local_holds),
            "expanded_length": sum(local_holds),
            "holds": local_holds,
            "legal_padding_frames": padding,
            "legal_padding_anchor": anchor,
            "parent_plan_sha256": parent["plan_sha256"],
            "parent_world_length": int(parent["world_length"]),
            "window": {
                "index": int(window_index),
                "count": int(window_count),
                "source_start": int(start),
                "source_end": int(end),
                "core_start": int(core_start),
                "core_end": int(core_end),
                "local_core_start": int(core_start - start),
                "local_core_end": int(core_end - start),
                "hot_in": bool(hot_in),
                "hot_out": bool(hot_out),
            },
            "source": {
                "frame_shape": [len(local_holds), *parent["source"]["frame_shape"][1:]],
                "canvas_width": parent["source"]["canvas_width"],
                "canvas_height": parent["source"]["canvas_height"],
                "pixel_megapixels": parent["source"]["pixel_megapixels"],
            },
            "audio_default": "pass1_original",
        }
    )


def _build_budgeted_windows(
    holds: list[int],
    budget: int,
    handle_frames: int,
    coverage: str,
) -> list[tuple[int, int, int, int, bool, bool]]:
    frame_count = len(holds)
    hot_indices = [index for index, hold in enumerate(holds) if hold > 1]
    if coverage == "hot_ranges_only":
        if not hot_indices:
            raise ValueError("motion plan has no selected hot ranges")
        coverage_start, coverage_end = hot_indices[0], hot_indices[-1]
    elif coverage == "full_clip":
        coverage_start, coverage_end = 0, frame_count - 1
    else:
        raise ValueError(f"Unsupported coverage {coverage!r}")
    windows = []
    core_start = coverage_start
    while core_start <= coverage_end:
        core_end = core_start
        while core_end < coverage_end:
            candidate_end = core_end + 1
            start = max(0, core_start - handle_frames)
            end = min(frame_count - 1, candidate_end + handle_frames)
            expanded = sum(
                holds[index] if core_start <= index <= candidate_end else 1
                for index in range(start, end + 1)
            )
            if align_frame_count(expanded) > budget:
                break
            core_end = candidate_end
        start = max(0, core_start - handle_frames)
        end = min(frame_count - 1, core_end + handle_frames)
        expanded = sum(
            holds[index] if core_start <= index <= core_end else 1
            for index in range(start, end + 1)
        )
        if align_frame_count(expanded) > budget:
            raise ValueError(
                "window budget cannot fit one core frame plus handles; raise "
                "max_expanded_frames or lower handle_frames/max_hold"
            )
        hot_in = core_start > coverage_start and holds[core_start] > 1
        hot_out = core_end < coverage_end and holds[core_end] > 1
        windows.append((start, end, core_start, core_end, hot_in, hot_out))
        core_start = core_end + 1
    return windows


def plan_motion_segment(
    frames: torch.Tensor,
    baseline_audio: Mapping[str, Any],
    motion_plan: Mapping[str, Any],
    max_expanded_frames: int,
    window_index: int,
    handle_frames: int,
    coverage: str,
) -> tuple[
    torch.Tensor,
    dict[str, Any],
    torch.Tensor,
    torch.Tensor,
    dict[str, Any],
    int,
    int,
    str,
]:
    parent = validate_motion_plan(motion_plan, allow_window=False)
    if parent["status"] != "ready":
        raise ValueError("parent motion plan must be ready")
    if int(frames.shape[0]) != int(parent["world_length"]):
        raise ValueError("frames do not match parent motion plan")
    waveform, sample_rate = validate_audio(baseline_audio, "baseline_audio")
    max_expanded_frames = int(max_expanded_frames)
    if max_expanded_frames < 39:
        raise ValueError("max_expanded_frames must be at least 39")
    max_expanded_frames = align_frame_count(max_expanded_frames)
    windows = _build_budgeted_windows(
        parent["holds"], max_expanded_frames, int(handle_frames), coverage
    )
    selected = max(0, min(int(window_index), len(windows) - 1))
    start, end, core_start, core_end, hot_in, hot_out = windows[selected]
    window_plan = _window_plan_from_parent(
        parent,
        start=start,
        end=end,
        core_start=core_start,
        core_end=core_end,
        window_index=selected,
        window_count=len(windows),
        hot_in=hot_in,
        hot_out=hot_out,
    )
    segment = frames[start : end + 1].detach().cpu()
    start_sample = round(start / FPS * sample_rate)
    end_sample = round((end + 1) / FPS * sample_rate)
    segment_audio = {
        "waveform": waveform.detach().float().cpu()[..., start_sample:end_sample].contiguous(),
        "sample_rate": sample_rate,
    }
    target_samples = round(len(window_plan["holds"]) / FPS * sample_rate)
    segment_audio = _fit_audio_samples(segment_audio, target_samples)
    report = {
        "schema_version": 1,
        "node": "MiniMaxH3MotionSegmentPlanT8Advanced",
        "status": "ready",
        "parent_plan_sha256": parent["plan_sha256"],
        "window_plan_sha256": window_plan["plan_sha256"],
        "window_index": selected,
        "window_count": len(windows),
        "source_span": [start, end],
        "core_span": [core_start, core_end],
        "audio_sample_span": [start_sample, end_sample],
        "audio_samples": int(segment_audio["waveform"].shape[-1]),
        "expanded_frames": window_plan["expanded_length"],
        "max_expanded_frames": max_expanded_frames,
        "cpu_held_frames": _cpu_frame_memory_estimate(
            frame_count=window_plan["expanded_length"],
            height=int(frames.shape[1]),
            width=int(frames.shape[2]),
        ),
        "hot_in": hot_in,
        "hot_out": hot_out,
        "warning": (
            "A hot cut lies inside an active motion burst; raising the budget or lowering "
            "max_hold is usually safer than adding more windows."
            if hot_in or hot_out
            else None
        ),
    }
    return (
        segment,
        window_plan,
        segment[:1].clone(),
        segment[-1:].clone(),
        segment_audio,
        len(windows),
        selected,
        canonical_json(report, indent=2),
    )


def _safe_run_name(value: str) -> str:
    output = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("._")
    if not output:
        raise ValueError("run_name must contain at least one safe character")
    return output[:128]


def _resolve_store_root(store_dir: str) -> Path:
    if store_dir.strip():
        return Path(store_dir).expanduser().resolve()
    import folder_paths

    return (Path(folder_paths.get_output_directory()) / "MiniMaxH3_Motion_Windows").resolve()


def _blend_segment(
    baseline: torch.Tensor,
    segment: torch.Tensor,
    window: Mapping[str, Any],
    feather_frames: int,
) -> torch.Tensor:
    start = int(window["source_start"])
    end = int(window["source_end"])
    if segment.shape[0] != end - start + 1:
        raise ValueError("stored segment length does not match its signed source span")
    local_core_start = int(window["local_core_start"])
    local_core_end = int(window["local_core_end"])
    weights = torch.ones(segment.shape[0], dtype=torch.float32)
    left = min(int(feather_frames), local_core_start)
    right = min(int(feather_frames), segment.shape[0] - 1 - local_core_end)
    if left > 0:
        weights[:left] = torch.linspace(0.0, 1.0, left + 2)[1:-1]
    if right > 0:
        weights[-right:] = torch.linspace(1.0, 0.0, right + 2)[1:-1]
    weights[: max(0, local_core_start - left)] = 0.0
    weights[min(segment.shape[0], local_core_end + right + 1) :] = 0.0
    target = baseline[start : end + 1]
    w = weights[:, None, None, None].to(dtype=target.dtype)
    output = baseline.clone()
    output[start : end + 1] = target * (1.0 - w) + segment.to(target.dtype) * w
    return output


def _blend_audio_segment(
    baseline_audio: dict[str, Any],
    segment_audio: dict[str, Any],
    window: Mapping[str, Any],
    feather_frames: int,
) -> dict[str, Any]:
    base_waveform, base_rate = validate_audio(baseline_audio, "baseline_audio")
    segment_waveform, segment_rate = validate_audio(segment_audio, "segment_audio")
    base_waveform = base_waveform.detach().float().cpu()
    segment_waveform = segment_waveform.detach().float().cpu()
    if segment_rate != base_rate:
        segment_waveform = torchaudio.functional.resample(
            segment_waveform, segment_rate, base_rate
        )
    start_frame = int(window["source_start"])
    end_frame = int(window["source_end"])
    start_sample = round(start_frame / FPS * base_rate)
    end_sample = round((end_frame + 1) / FPS * base_rate)
    target_len = end_sample - start_sample
    if segment_waveform.shape[-1] > target_len:
        segment_waveform = segment_waveform[..., :target_len]
    elif segment_waveform.shape[-1] < target_len:
        segment_waveform = F.pad(segment_waveform, (0, target_len - segment_waveform.shape[-1]))
    if segment_waveform.shape[1] != base_waveform.shape[1]:
        segment_waveform = segment_waveform[:, :1].expand(-1, base_waveform.shape[1], -1)
    weights = torch.ones(target_len, dtype=torch.float32)
    local_core_start = int(window["local_core_start"])
    left_samples = min(
        round(feather_frames / FPS * base_rate),
        round(local_core_start / FPS * base_rate),
    )
    right_frames = end_frame - (int(window["core_end"]))
    right_samples = min(
        round(feather_frames / FPS * base_rate),
        round(right_frames / FPS * base_rate),
    )
    if left_samples > 0:
        weights[:left_samples] = torch.linspace(0.0, 1.0, left_samples + 2)[1:-1]
    if right_samples > 0:
        weights[-right_samples:] = torch.linspace(1.0, 0.0, right_samples + 2)[1:-1]
    output = base_waveform.clone()
    target = output[..., start_sample:end_sample]
    w = weights[None, None]
    output[..., start_sample:end_sample] = target * (1.0 - w) + segment_waveform * w
    return {"waveform": output.contiguous(), "sample_rate": base_rate}


def _audio_segment_is_exact_baseline_slice(
    baseline_audio: Mapping[str, Any],
    segment_audio: Mapping[str, Any],
    window: Mapping[str, Any],
) -> bool:
    """Detect the safe pass-1 path without performing any float blend math."""
    base_waveform, base_rate = validate_audio(baseline_audio, "baseline_audio")
    segment_waveform, segment_rate = validate_audio(segment_audio, "segment_audio")
    if segment_rate != base_rate:
        return False
    start_sample = round(int(window["source_start"]) / FPS * base_rate)
    end_sample = round((int(window["source_end"]) + 1) / FPS * base_rate)
    expected = base_waveform.detach().float().cpu()[..., start_sample:end_sample].contiguous()
    candidate = segment_waveform.detach().float().cpu().contiguous()
    return candidate.shape == expected.shape and torch.equal(candidate, expected)


def collect_motion_window(
    baseline_frames: torch.Tensor,
    baseline_audio: Mapping[str, Any],
    recovered_segment: torch.Tensor,
    recovered_audio: Mapping[str, Any],
    window_plan: Mapping[str, Any],
    run_name: str,
    store_dir: str,
    write_window: bool,
    store_dtype: str,
    feather_frames: int,
) -> tuple[torch.Tensor, dict[str, Any], bool, int, str]:
    plan = validate_motion_plan(window_plan)
    if plan["schema"] != MOTION_WINDOW_SCHEMA:
        raise ValueError("window_plan must be the output of Motion Segment Plan")
    window = plan["window"]
    if not isinstance(baseline_frames, torch.Tensor) or baseline_frames.ndim != 4:
        raise ValueError("baseline_frames must be a connected IMAGE")
    if int(baseline_frames.shape[0]) != int(plan["parent_world_length"]):
        raise ValueError("baseline_frames do not match window parent_world_length")
    if not isinstance(recovered_segment, torch.Tensor) or recovered_segment.ndim != 4:
        raise ValueError("recovered_segment must be a connected IMAGE")
    if int(recovered_segment.shape[0]) != int(plan["world_length"]):
        raise ValueError("recovered segment does not match window world_length")
    if tuple(recovered_segment.shape[1:]) != tuple(baseline_frames.shape[1:]):
        raise ValueError("recovered segment canvas does not match baseline frames")
    if not torch.isfinite(recovered_segment).all():
        raise ValueError("recovered_segment contains NaN or Inf")
    _baseline_waveform, baseline_rate = validate_audio(baseline_audio, "baseline_audio")
    _recovered_waveform, recovered_rate = validate_audio(recovered_audio, "recovered_audio")
    baseline_audio_fit = _fit_audio_samples(
        baseline_audio,
        round(plan["parent_world_length"] / FPS * baseline_rate),
    )
    recovered_audio_fit = _fit_audio_samples(
        recovered_audio,
        round(plan["world_length"] / FPS * recovered_rate),
    )
    recovered_audio_fit["h3_t8_motion_audio_mode"] = str(
        recovered_audio.get("h3_t8_motion_audio_mode", "")
    )
    recovered_audio_fit["h3_t8_motion_plan_sha256"] = str(
        recovered_audio.get("h3_t8_motion_plan_sha256", "")
    )
    if store_dtype not in {"float32_exact", "float16_half_disk"}:
        raise ValueError(f"Unsupported store_dtype {store_dtype!r}")
    run = _safe_run_name(run_name)
    root = (
        _resolve_store_root(store_dir)
        / run
        / str(plan["parent_plan_sha256"])[:16]
    )
    root.mkdir(parents=True, exist_ok=True)
    index = int(window["index"])
    count = int(window["count"])
    path = root / f"w{index:03d}.pt"
    if write_window:
        torch.save(
            {
                "plan": plan,
                "segment": recovered_segment.detach()
                .to(torch.float32 if store_dtype == "float32_exact" else torch.float16)
                .cpu(),
                "audio": {
                    "waveform": recovered_audio_fit["waveform"],
                    "sample_rate": int(recovered_audio_fit["sample_rate"]),
                    "h3_t8_motion_audio_mode": recovered_audio_fit[
                        "h3_t8_motion_audio_mode"
                    ],
                    "h3_t8_motion_plan_sha256": recovered_audio_fit[
                        "h3_t8_motion_plan_sha256"
                    ],
                },
            },
            path,
        )
    banked: dict[int, Path] = {}
    for candidate in root.glob("w[0-9][0-9][0-9].pt"):
        try:
            candidate_index = int(candidate.stem[1:])
        except ValueError:
            continue
        banked[candidate_index] = candidate
    missing = [candidate for candidate in range(count) if candidate not in banked]
    if missing:
        report = {
            "schema_version": 1,
            "node": "MiniMaxH3MotionWindowCollectT8Advanced",
            "status": "waiting",
            "run_dir": str(root),
            "windows_on_disk": sorted(banked),
            "missing_windows": missing,
            "parent_plan_sha256": plan["parent_plan_sha256"],
            "store_dtype": store_dtype,
        }
        return (
            baseline_frames.detach().cpu(),
            baseline_audio_fit,
            False,
            len(banked),
            canonical_json(report, indent=2),
        )
    output_frames = baseline_frames.detach().float().cpu()
    output_audio = {
        "waveform": baseline_audio_fit["waveform"],
        "sample_rate": int(baseline_audio_fit["sample_rate"]),
    }
    loaded = []
    exact_audio_bypass_windows = []
    for candidate_index in range(count):
        payload = torch.load(banked[candidate_index], weights_only=False, map_location="cpu")
        candidate_plan = validate_motion_plan(payload["plan"])
        if candidate_plan["parent_plan_sha256"] != plan["parent_plan_sha256"]:
            raise ValueError("window bank contains a result from a different parent motion plan")
        candidate_window = candidate_plan["window"]
        if int(candidate_window["index"]) != candidate_index or int(candidate_window["count"]) != count:
            raise ValueError("window bank index/count contract mismatch")
        output_frames = _blend_segment(
            output_frames,
            payload["segment"].float(),
            candidate_window,
            feather_frames,
        )
        audio_policy = str(payload["audio"].get("h3_t8_motion_audio_mode", ""))
        if audio_policy == "pass1_original" or _audio_segment_is_exact_baseline_slice(
            baseline_audio_fit,
            payload["audio"],
            candidate_window,
        ):
            # pass1_original is already the desired final waveform. Re-blending an
            # identical slice can still change float bits and therefore AAC output.
            exact_audio_bypass_windows.append(candidate_index)
        else:
            output_audio = _blend_audio_segment(
                output_audio,
                payload["audio"],
                candidate_window,
                feather_frames,
            )
        loaded.append(candidate_plan["plan_sha256"])
    report = {
        "schema_version": 1,
        "node": "MiniMaxH3MotionWindowCollectT8Advanced",
        "status": "complete",
        "run_dir": str(root),
        "window_count": count,
        "window_plan_sha256": loaded,
        "parent_plan_sha256": plan["parent_plan_sha256"],
        "frames": int(output_frames.shape[0]),
        "audio_samples": int(output_audio["waveform"].shape[-1]),
        "resume_capable": True,
        "store_dtype": store_dtype,
        "exact_pass1_audio_bypass_windows": exact_audio_bypass_windows,
    }
    return output_frames, output_audio, True, count, canonical_json(report, indent=2)
