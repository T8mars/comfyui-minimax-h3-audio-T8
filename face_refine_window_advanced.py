from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable

import torch

from .core import validate_audio
from .face_refine_advanced import (
    _scene_ranges,
    _validate_frames,
    canonical_json,
    source_proxy_sha256,
)


WINDOW_PLAN_SCHEMA = "h3_t8_face_refine_window_plan/v1"
WINDOW_MAPPING_SCHEMA = "h3_t8_face_refine_window_mapping/v1"
H3_FPS = 24.0
DEFAULT_AUDIO_SAMPLE_RATE = 32_000
_RANGE_RE = re.compile(
    r"^\s*(?P<start>\d+(?:\.\d+)?)\s*(?:-|:|\.\.)\s*(?P<end>\d+(?:\.\d+)?)\s*$"
)


def _signed(payload: dict, key: str) -> dict:
    result = dict(payload)
    result[key] = hashlib.sha256(canonical_json(result).encode("utf-8")).hexdigest()
    return result


def _validate_signed(payload: dict, schema: str, hash_key: str, name: str) -> dict:
    if not isinstance(payload, dict) or payload.get("schema") != schema:
        raise ValueError(f"{name} must use {schema}")
    unsigned = {key: value for key, value in payload.items() if key != hash_key}
    expected = hashlib.sha256(canonical_json(unsigned).encode("utf-8")).hexdigest()
    if payload.get(hash_key) != expected:
        raise ValueError(f"{name} hash mismatch; it may be stale or modified")
    return payload


def _legal_h3_length_at_least(value: int, maximum: int) -> int:
    value = max(1, int(value))
    n = max(1, math.ceil((value - 5) / 17))
    candidate = 17 * n + 5
    if candidate > int(maximum):
        raise ValueError(
            f"No legal 17n+5 H3 window can satisfy {value} frames within max_render_frames={maximum}"
        )
    return candidate


def _parse_ranges(
    value: str,
    *,
    range_mode: str,
    fps: float,
    frame_count: int,
) -> list[tuple[int, int]]:
    text = str(value or "").strip()
    if not text:
        return []
    if range_mode not in {"frames_inclusive", "seconds_inclusive"}:
        raise ValueError(f"Unsupported range_mode: {range_mode}")

    ranges: list[tuple[int, int]] = []
    for item in re.split(r"[,;\n]+", text):
        if not item.strip():
            continue
        match = _RANGE_RE.match(item)
        if match is None:
            raise ValueError(
                f"Invalid repair range {item!r}; use inclusive start-end pairs such as 0-23,50-60"
            )
        raw_start = float(match.group("start"))
        raw_end = float(match.group("end"))
        if range_mode == "frames_inclusive":
            if not raw_start.is_integer() or not raw_end.is_integer():
                raise ValueError("Frame ranges must use integer 0-based inclusive indices")
            start, end = int(raw_start), int(raw_end)
        else:
            start, end = round(raw_start * fps), round(raw_end * fps)
        if start > end:
            raise ValueError(f"Repair range is reversed: {item!r}")
        if start < 0 or end >= frame_count:
            raise ValueError(
                f"Repair range {start}-{end} is outside source frame bounds 0-{frame_count - 1}"
            )
        ranges.append((start, end))
    return ranges


def _normalise_ranges(
    ranges: Iterable[tuple[int, int]], overlap_policy: str
) -> list[tuple[int, int]]:
    if overlap_policy not in {"reject", "merge"}:
        raise ValueError(f"Unsupported overlap_policy: {overlap_policy}")
    ordered = sorted((int(start), int(end)) for start, end in ranges)
    if not ordered:
        return []
    output = [ordered[0]]
    for start, end in ordered[1:]:
        previous_start, previous_end = output[-1]
        if start <= previous_end:
            if overlap_policy == "reject":
                raise ValueError(
                    f"Repair ranges {previous_start}-{previous_end} and {start}-{end} overlap or duplicate"
                )
            output[-1] = (previous_start, max(previous_end, end))
        else:
            output.append((start, end))
    return output


def _shot_for_range(
    start: int, end: int, shots: list[tuple[int, int]]
) -> tuple[int, int, int]:
    for shot_id, (shot_start, shot_end) in enumerate(shots):
        if shot_start <= start <= end <= shot_end:
            return shot_id, shot_start, shot_end
    raise ValueError(
        f"Repair range {start}-{end} crosses a detected hard cut; split it into shot-local ranges"
    )


def _frame_boundary_sample(frame_boundary: int, sample_rate: int) -> int:
    return round(int(frame_boundary) * int(sample_rate) / int(H3_FPS))


def build_face_refine_window_plan(
    base_frames: torch.Tensor,
    fps: float,
    repair_ranges: str,
    range_mode: str,
    context_before_frames: int,
    context_after_frames: int,
    min_render_frames: int,
    max_render_frames: int,
    scene_cut_threshold: float,
    overlap_policy: str,
    short_shot_policy: str,
    enabled: bool,
):
    frame_count, height, width = _validate_frames(base_frames, name="base_frames")
    if abs(float(fps) - H3_FPS) > 0.01:
        raise ValueError(f"Face Refine Window requires exact 24fps input; got {fps:.6g}fps")
    if int(context_before_frames) < 0 or int(context_after_frames) < 0:
        raise ValueError("Context frame counts cannot be negative")
    if int(min_render_frames) < 1 or int(max_render_frames) < int(min_render_frames):
        raise ValueError("Render limits must satisfy 1 <= min_render_frames <= max_render_frames")
    if short_shot_policy not in {"reject", "edge_hold_exp"}:
        raise ValueError(f"Unsupported short_shot_policy: {short_shot_policy}")

    requested = (
        _parse_ranges(
            repair_ranges,
            range_mode=range_mode,
            fps=float(fps),
            frame_count=frame_count,
        )
        if enabled
        else []
    )
    normalised = _normalise_ranges(requested, overlap_policy)
    if normalised:
        shots, scene_deltas = _scene_ranges(base_frames, float(scene_cut_threshold))
    else:
        shots, scene_deltas = [(0, frame_count - 1)], [0.0]
    source = {
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "channels": int(base_frames.shape[-1]),
        "dtype": str(base_frames.dtype),
        "fps": float(fps),
        "proxy_sha256": source_proxy_sha256(base_frames),
    }

    windows: list[dict] = []
    if enabled:
        for repair_start, repair_end in normalised:
            shot_id, shot_start, shot_end = _shot_for_range(repair_start, repair_end, shots)
            desired_start = repair_start - int(context_before_frames)
            desired_end = repair_end + int(context_after_frames)
            required = desired_end - desired_start + 1
            render_count = _legal_h3_length_at_least(
                max(required, int(min_render_frames)), int(max_render_frames)
            )
            shot_length = shot_end - shot_start + 1

            if shot_length >= render_count:
                latest_start = shot_end - render_count + 1
                render_source_start = min(max(desired_start, shot_start), latest_start)
                render_source_end = render_source_start + render_count - 1
                pre_pad = 0
                post_pad = 0
            else:
                if short_shot_policy != "edge_hold_exp":
                    raise ValueError(
                        f"Shot {shot_id} has only {shot_length} frames but the smallest legal window is "
                        f"{render_count}; select edge_hold_exp explicitly to pad context"
                    )
                render_source_start = shot_start
                render_source_end = shot_end
                pre_pad = 0
                post_pad = render_count - shot_length

            accept_relative_ranges = [
                [
                    pre_pad + repair_start - render_source_start,
                    pre_pad + repair_end - render_source_start,
                ]
            ]
            if accept_relative_ranges[0][0] < 0 or accept_relative_ranges[0][1] >= render_count:
                raise ValueError("Internal window mapping failed to contain the requested repair range")

            windows.append(
                {
                    "window_index": len(windows),
                    "shot_id": shot_id,
                    "shot_start_frame": shot_start,
                    "shot_end_frame": shot_end,
                    "repair_ranges_abs": [[repair_start, repair_end]],
                    "render_source_start_frame": render_source_start,
                    "render_source_end_frame": render_source_end,
                    "render_frame_count": render_count,
                    "pre_pad_frames": pre_pad,
                    "post_pad_frames": post_pad,
                    "padding_policy": short_shot_policy if pre_pad or post_pad else "none",
                    "accept_relative_ranges": accept_relative_ranges,
                    "source_start_seconds": render_source_start / H3_FPS,
                    "render_duration_seconds": render_count / H3_FPS,
                    "default_32khz_audio": {
                        "source_start_sample": _frame_boundary_sample(
                            render_source_start, DEFAULT_AUDIO_SAMPLE_RATE
                        ),
                        "source_end_sample_exclusive": _frame_boundary_sample(
                            render_source_end + 1, DEFAULT_AUDIO_SAMPLE_RATE
                        ),
                        "render_start_sample": _frame_boundary_sample(
                            pre_pad, DEFAULT_AUDIO_SAMPLE_RATE
                        ),
                        "render_end_sample_exclusive": _frame_boundary_sample(
                            pre_pad + render_source_end - render_source_start + 1,
                            DEFAULT_AUDIO_SAMPLE_RATE,
                        ),
                        "target_sample_count": _frame_boundary_sample(
                            render_count, DEFAULT_AUDIO_SAMPLE_RATE
                        ),
                    },
                }
            )

    mask = base_frames.new_zeros((frame_count, height, width))
    for start, end in normalised if enabled else []:
        mask[start : end + 1] = 1.0

    status = "disabled_noop" if not enabled else ("empty_noop" if not normalised else "planned")
    plan = _signed(
        {
            "schema": WINDOW_PLAN_SCHEMA,
            "status": status,
            "source": source,
            "requested_ranges_text": str(repair_ranges or ""),
            "range_mode": range_mode,
            "requested_ranges_frames": [[start, end] for start, end in requested],
            "normalised_ranges_frames": [[start, end] for start, end in normalised],
            "settings": {
                "enabled": bool(enabled),
                "context_before_frames": int(context_before_frames),
                "context_after_frames": int(context_after_frames),
                "min_render_frames": int(min_render_frames),
                "max_render_frames": int(max_render_frames),
                "scene_cut_threshold": float(scene_cut_threshold),
                "overlap_policy": overlap_policy,
                "short_shot_policy": short_shot_policy,
            },
            "shots": [
                {"shot_id": index, "start_frame": start, "end_frame": end}
                for index, (start, end) in enumerate(shots)
            ],
            "windows": windows,
            "window_count": len(windows),
            "metrics": {
                "scene_cut_count": max(0, len(shots) - 1),
                "max_scene_delta": max(scene_deltas),
            },
            "limits": {
                "single_shot_per_window": True,
                "single_primary_person": True,
                "automatic_queue": False,
                "automatic_accept": False,
                "quality_guaranteed": False,
                "identity_verified": False,
                "audio_replacement_allowed": False,
            },
        },
        "plan_sha256",
    )
    return plan, mask, len(windows), canonical_json(plan)


def _validate_source_binding(frames: torch.Tensor, source: dict, name: str) -> None:
    frame_count, height, width = _validate_frames(frames, name=name)
    expected = (
        int(source.get("frame_count", -1)),
        int(source.get("height", -1)),
        int(source.get("width", -1)),
        int(source.get("channels", -1)),
    )
    actual = (frame_count, height, width, int(frames.shape[-1]))
    if actual != expected:
        raise ValueError(f"{name} shape {actual} does not match plan source {expected}")
    if source_proxy_sha256(frames) != source.get("proxy_sha256"):
        raise ValueError(f"{name} does not match the source-bound plan")


def _window_base_frames(base_frames: torch.Tensor, window: dict) -> torch.Tensor:
    start = int(window["render_source_start_frame"])
    end = int(window["render_source_end_frame"])
    pre = int(window["pre_pad_frames"])
    post = int(window["post_pad_frames"])
    section = base_frames[start : end + 1]
    parts = []
    if pre:
        parts.append(section[:1].expand(pre, -1, -1, -1))
    parts.append(section)
    if post:
        parts.append(section[-1:].expand(post, -1, -1, -1))
    result = torch.cat(parts, dim=0)
    if int(result.shape[0]) != int(window["render_frame_count"]):
        raise ValueError("Window frame mapping produced an unexpected length")
    return result


def extract_face_refine_window(
    base_frames: torch.Tensor,
    window_plan: dict,
    window_index: int,
    pad_policy: str,
    source_audio: dict | None = None,
):
    plan = _validate_signed(window_plan, WINDOW_PLAN_SCHEMA, "plan_sha256", "window_plan")
    _validate_source_binding(base_frames, plan["source"], "base_frames")
    windows = plan.get("windows")
    if not isinstance(windows, list) or not windows:
        raise ValueError("window_plan has no render window; disabled and empty plans are exact bypasses")
    index = int(window_index)
    if index < 0 or index >= len(windows):
        raise ValueError(f"window_index {index} is outside 0-{len(windows) - 1}")
    if pad_policy not in {"reject", "edge_hold_exp"}:
        raise ValueError(f"Unsupported pad_policy: {pad_policy}")
    window = dict(windows[index])
    if (window["pre_pad_frames"] or window["post_pad_frames"]) and pad_policy != "edge_hold_exp":
        raise ValueError("This window requires explicit edge_hold_exp padding")

    render_frames = _window_base_frames(base_frames, window)
    frame_map = []
    source_start = int(window["render_source_start_frame"])
    source_end = int(window["render_source_end_frame"])
    pre = int(window["pre_pad_frames"])
    for relative in range(int(window["render_frame_count"])):
        if relative < pre:
            frame_map.append(
                {
                    "render_frame": relative,
                    "source_frame": None,
                    "edge_source_frame": source_start,
                    "kind": "padding",
                }
            )
        elif relative < pre + source_end - source_start + 1:
            source_index = source_start + relative - pre
            frame_map.append(
                {
                    "render_frame": relative,
                    "source_frame": source_index,
                    "edge_source_frame": source_index,
                    "kind": "source",
                }
            )
        else:
            frame_map.append(
                {
                    "render_frame": relative,
                    "source_frame": None,
                    "edge_source_frame": source_end,
                    "kind": "padding",
                }
            )

    render_audio = None
    audio_mapping = {"connected": False, "warning": "No source AUDIO connected; lipsync is not verifiable."}
    if source_audio is not None:
        waveform, sample_rate = validate_audio(source_audio, "source_audio")
        source_sample_start = _frame_boundary_sample(source_start, sample_rate)
        source_sample_end = _frame_boundary_sample(source_end + 1, sample_rate)
        target_sample_count = _frame_boundary_sample(int(window["render_frame_count"]), sample_rate)
        render_sample_start = _frame_boundary_sample(pre, sample_rate)
        render_sample_end = _frame_boundary_sample(
            pre + source_end - source_start + 1, sample_rate
        )
        output = waveform.new_zeros(
            (int(waveform.shape[0]), int(waveform.shape[1]), target_sample_count)
        )
        source_slice = waveform[..., source_sample_start:source_sample_end]
        writable = min(
            int(source_slice.shape[-1]),
            max(0, render_sample_end - render_sample_start),
            max(0, target_sample_count - render_sample_start),
        )
        if writable:
            output[..., render_sample_start : render_sample_start + writable] = source_slice[
                ..., :writable
            ]
        render_audio = {"waveform": output, "sample_rate": sample_rate}
        audio_mapping = {
            "connected": True,
            "sample_rate": sample_rate,
            "source_start_sample": source_sample_start,
            "source_end_sample_exclusive": source_sample_end,
            "render_start_sample": render_sample_start,
            "render_end_sample_exclusive": render_sample_end,
            "target_sample_count": target_sample_count,
            "padding_samples_are_zero": True,
        }

    mapping = _signed(
        {
            "schema": WINDOW_MAPPING_SCHEMA,
            "window_plan_sha256": plan["plan_sha256"],
            "source": dict(plan["source"]),
            "window": window,
            "frame_map": frame_map,
            "audio": audio_mapping,
            "limits": {
                "candidate_audio_discarded": True,
                "final_full_source_audio_required": True,
                "padding_never_accepted": True,
                "context_never_accepted_implicitly": True,
                "automatic_accept": False,
            },
        },
        "mapping_sha256",
    )
    report = {
        "status": "window_extracted",
        "window_index": index,
        "render_frame_count": int(render_frames.shape[0]),
        "legal_17n_plus_5": (int(render_frames.shape[0]) - 5) % 17 == 0,
        "source_start_seconds": source_start / H3_FPS,
        "render_duration_seconds": int(render_frames.shape[0]) / H3_FPS,
        "accept_relative_ranges": window["accept_relative_ranges"],
        "window_plan_sha256": plan["plan_sha256"],
        "mapping_sha256": mapping["mapping_sha256"],
        "audio": audio_mapping,
        "warning": (
            None
            if source_audio is not None
            else "Visual-only probe: connect source AUDIO before judging speech or lipsync."
        ),
    }
    return (
        render_frames,
        render_audio,
        mapping,
        source_start / H3_FPS,
        int(render_frames.shape[0]) / H3_FPS,
        json.dumps(window["accept_relative_ranges"], ensure_ascii=False),
        canonical_json(report),
    )


def _normalise_mask(mask: torch.Tensor, frame_count: int, height: int, width: int) -> torch.Tensor:
    if not isinstance(mask, torch.Tensor):
        raise ValueError("changed_mask must be a MASK tensor")
    value = mask
    if value.ndim == 4 and value.shape[-1] == 1:
        value = value[..., 0]
    elif value.ndim == 4 and value.shape[1] == 1:
        value = value[:, 0]
    if value.ndim != 3 or tuple(value.shape) != (frame_count, height, width):
        raise ValueError(
            f"changed_mask must use [{frame_count},{height},{width}], got {tuple(mask.shape)}"
        )
    if not torch.isfinite(value).all():
        raise ValueError("changed_mask contains NaN or Inf")
    if bool((value < 0).any()) or bool((value > 1).any()):
        raise ValueError("changed_mask values must stay within 0..1")
    return value


def _ranges_contained(
    requested: list[tuple[int, int]], allowed: list[tuple[int, int]]
) -> bool:
    return all(any(a_start <= start <= end <= a_end for a_start, a_end in allowed) for start, end in requested)


def _temporal_weights(ranges: list[tuple[int, int]], fade: int) -> dict[int, float]:
    weights: dict[int, float] = {}
    for start, end in ranges:
        length = end - start + 1
        for offset, frame in enumerate(range(start, end + 1)):
            if fade <= 0 or length == 1:
                value = 1.0
            else:
                value = min(
                    1.0,
                    (offset + 1) / (fade + 1),
                    (length - offset) / (fade + 1),
                )
            weights[frame] = max(weights.get(frame, 0.0), float(value))
    return weights


def _safe_reject_outputs(
    base_frames: torch.Tensor,
    review_frames: torch.Tensor,
    reason: str,
    decision: str,
):
    frame_count, height, width = _validate_frames(base_frames, name="base_frames")
    zero = base_frames.new_zeros((frame_count, height, width))
    report = {
        "status": "rejected_contract",
        "decision": decision,
        "reason": reason,
        "accepted_frame_count": 0,
        "rejected_frame_count": 0,
        "source_preserved_bit_exact": True,
        "automatic_accept": False,
        "quality_guaranteed": False,
        "identity_verified": False,
    }
    return review_frames, base_frames, zero, zero.clone(), 0, 0, canonical_json(report)


def apply_face_refine_manual_review(
    base_frames: torch.Tensor,
    candidate_window_frames: torch.Tensor,
    changed_mask: torch.Tensor,
    window_mapping: dict,
    decision: str,
    accepted_subranges: str,
    confirm_accept: bool,
    edge_fade_frames: int,
):
    frame_count, height, width = _validate_frames(base_frames, name="base_frames")
    fallback_review = torch.cat((base_frames, base_frames), dim=2)
    if decision not in {"preview_only", "reject", "accept_selected"}:
        return _safe_reject_outputs(base_frames, fallback_review, "unsupported decision", decision)

    try:
        mapping = _validate_signed(
            window_mapping, WINDOW_MAPPING_SCHEMA, "mapping_sha256", "window_mapping"
        )
        _validate_source_binding(base_frames, mapping["source"], "base_frames")
        window = mapping["window"]
        render_count = int(window["render_frame_count"])
        candidate_count, candidate_height, candidate_width = _validate_frames(
            candidate_window_frames, name="candidate_window_frames"
        )
        if (
            candidate_count != render_count
            or candidate_height != height
            or candidate_width != width
            or int(candidate_window_frames.shape[-1]) != int(base_frames.shape[-1])
        ):
            raise ValueError("candidate window shape does not match the source-bound mapping")
        render_base = _window_base_frames(base_frames, window)
        review_frames = torch.cat((render_base, candidate_window_frames), dim=2)
        mask = _normalise_mask(changed_mask, render_count, height, width).to(
            device=candidate_window_frames.device, dtype=candidate_window_frames.dtype
        )
        outside = mask <= 0
        outside_channels = outside.unsqueeze(-1).expand_as(candidate_window_frames)
        if not torch.equal(
            candidate_window_frames[outside_channels], render_base.to(candidate_window_frames.device)[outside_channels]
        ):
            raise ValueError("candidate modified pixels outside changed_mask")
        shots = mapping.get("window", {})
        if int(shots["shot_start_frame"]) > int(shots["render_source_start_frame"]):
            raise ValueError("window mapping starts before its shot")
        if int(shots["render_source_end_frame"]) > int(shots["shot_end_frame"]):
            raise ValueError("window mapping ends after its shot")
    except (KeyError, TypeError, ValueError) as error:
        return _safe_reject_outputs(base_frames, fallback_review, str(error), decision)

    result = base_frames.clone()
    accepted_mask = base_frames.new_zeros((frame_count, height, width))
    rejected_mask = base_frames.new_zeros((frame_count, height, width))
    allowed = [tuple(map(int, item)) for item in window["repair_ranges_abs"]]
    selected: list[tuple[int, int]] = []
    status = decision
    reason = None
    if decision == "accept_selected":
        if not bool(confirm_accept):
            status = "rejected_unconfirmed"
            reason = "accept_selected requires confirm_accept=true"
        else:
            try:
                selected = _normalise_ranges(
                    _parse_ranges(
                        accepted_subranges,
                        range_mode="frames_inclusive",
                        fps=H3_FPS,
                        frame_count=frame_count,
                    )
                    if str(accepted_subranges or "").strip()
                    else allowed,
                    "merge",
                )
                if not selected:
                    raise ValueError("accept_selected requires at least one accepted frame range")
                if not _ranges_contained(selected, allowed):
                    raise ValueError("accepted_subranges must stay inside the planned repair ranges")
            except ValueError as error:
                status = "rejected_invalid_selection"
                reason = str(error)
                selected = []

    weights = _temporal_weights(selected, int(edge_fade_frames)) if status == "accept_selected" else {}
    mapped_changed_frames: set[int] = set()
    for record in mapping["frame_map"]:
        relative = int(record["render_frame"])
        source_index = record.get("source_frame")
        if source_index is None:
            continue
        source_index = int(source_index)
        source_mask = mask[relative].to(device=base_frames.device, dtype=base_frames.dtype)
        if bool((source_mask > 0).any()):
            mapped_changed_frames.add(source_index)
        temporal = weights.get(source_index, 0.0)
        if temporal > 0:
            alpha = source_mask * float(temporal)
            source_candidate = candidate_window_frames[relative].to(
                device=base_frames.device, dtype=base_frames.dtype
            )
            result[source_index] = (
                base_frames[source_index] * (1.0 - alpha.unsqueeze(-1))
                + source_candidate * alpha.unsqueeze(-1)
            )
            accepted_mask[source_index] = torch.maximum(accepted_mask[source_index], alpha)
        elif bool((source_mask > 0).any()):
            rejected_mask[source_index] = torch.maximum(rejected_mask[source_index], source_mask)

    accepted_frames = int((accepted_mask.flatten(1).amax(dim=1) > 0).sum().item())
    rejected_frames = int((rejected_mask.flatten(1).amax(dim=1) > 0).sum().item())
    if decision in {"preview_only", "reject"} or status != "accept_selected":
        if not torch.equal(result, base_frames):
            raise AssertionError("Reject and preview paths must preserve the source tensor bit-exact")
    outside_accepted = accepted_mask <= 0
    if not torch.equal(
        result[outside_accepted.unsqueeze(-1).expand_as(result)],
        base_frames[outside_accepted.unsqueeze(-1).expand_as(base_frames)],
    ):
        raise AssertionError("Manual review changed pixels outside the accepted mask")

    report = {
        "status": status,
        "decision": decision,
        "reason": reason,
        "selected_ranges_abs": [[start, end] for start, end in selected],
        "planned_repair_ranges_abs": [list(item) for item in allowed],
        "accepted_frame_count": accepted_frames,
        "rejected_frame_count": rejected_frames,
        "candidate_changed_source_frames": len(mapped_changed_frames),
        "source_preserved_outside_accepted_mask_bit_exact": True,
        "candidate_audio_discarded": True,
        "final_full_source_audio_required": True,
        "mapping_sha256": mapping["mapping_sha256"],
        "window_plan_sha256": mapping["window_plan_sha256"],
        "automatic_accept": False,
        "quality_guaranteed": False,
        "identity_verified": False,
        "metrics_are_diagnostic_only": True,
    }
    return (
        review_frames,
        result,
        accepted_mask,
        rejected_mask,
        accepted_frames,
        rejected_frames,
        canonical_json(report),
    )
