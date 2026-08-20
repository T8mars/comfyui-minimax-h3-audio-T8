from __future__ import annotations

import json
import math
from collections.abc import Mapping

from .core import FPS, align_frame_count
from .prompt_relay_advanced import _validate_plan


MAX_PREVIEW_CHARS = 12000


def _short_text(value: str, limit: int = 240) -> str:
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _ranges_for_count(counts: list[int], predicate) -> list[list[int]]:
    ranges: list[list[int]] = []
    start = None
    for index, value in enumerate(counts + [0]):
        active = index < len(counts) and predicate(value)
        if active and start is None:
            start = index
        elif not active and start is not None:
            ranges.append([start, index - 1])
            start = None
    return ranges


def preview_prompt_relay_plan(prompt_relay_plan: Mapping) -> tuple[dict, bool, int, str, str]:
    plan = _validate_plan(prompt_relay_plan)
    frame_count = plan.get("frame_count")
    if isinstance(frame_count, bool) or not isinstance(frame_count, int):
        raise ValueError("Prompt Relay Preview requires an integer frame_count")
    if frame_count < 5 or frame_count > 3600 or align_frame_count(frame_count) != frame_count:
        raise ValueError(
            "Prompt Relay Preview requires a 5..3600 frame H3 17n+5 timeline"
        )
    if not math.isclose(float(plan.get("fps", 0.0)), float(FPS), abs_tol=1e-12):
        raise ValueError("Prompt Relay Preview requires the native 24fps timeline")

    events = plan.get("events")
    if not isinstance(events, list):
        raise ValueError("Prompt Relay Preview requires an event list")
    if not events:
        report = {
            "status": "prompt_relay_plan_bypass",
            "ready": True,
            "plan_hash": plan["plan_hash"],
            "frame_count": frame_count,
            "fps": FPS,
            "duration_seconds": frame_count / FPS,
            "event_count": 0,
            "timing_mode": plan.get("timing_mode"),
            "math_profile": plan.get("math_profile"),
            "query_route": plan.get("query_route", "video_only_paper"),
            "covered_frame_count": 0,
            "gap_ranges_inclusive": [],
            "overlap_ranges_inclusive": [],
            "events": [],
            "bypass_reason": "no_active_local_events",
            "model_loaded": False,
            "sampling_executed": False,
        }
        timeline_text = "\n".join(
            [
                "Prompt Relay Plan: BYPASS (no active local events)",
                (
                    f"{frame_count} frames @ {FPS}fps = "
                    f"{frame_count / FPS:.6f}s | global prompt only"
                ),
                f"plan_hash: {plan['plan_hash']}",
            ]
        )
        return (
            plan,
            True,
            0,
            timeline_text,
            json.dumps(report, ensure_ascii=False, indent=2),
        )
    counts = [0] * frame_count
    summaries = []
    previous_start = -1
    for index, event in enumerate(events, 1):
        if not isinstance(event, Mapping):
            raise ValueError(f"Prompt Relay Preview event {index} must be an object")
        if event.get("event_index") != index:
            raise ValueError("Prompt Relay Preview event indices must be sequential")
        start = event.get("start_frame")
        end = event.get("end_frame_exclusive")
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
            or start < 0
            or end > frame_count
            or end <= start
        ):
            raise ValueError(
                f"Prompt Relay Preview event {index} has an invalid frame interval"
            )
        if start < previous_start:
            raise ValueError(
                "Prompt Relay Preview events must be listed in chronological order"
            )
        if end - start < 5:
            raise ValueError(
                f"Prompt Relay Preview event {index} is shorter than 5 frames"
            )
        previous_start = start
        prompt = event.get("local_prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Prompt Relay Preview event {index} prompt cannot be empty")
        for frame in range(start, end):
            counts[frame] += 1
        summaries.append(
            {
                "event_index": index,
                "start_frame": start,
                "end_frame_inclusive": end - 1,
                "start_seconds": start / FPS,
                "end_seconds": end / FPS,
                "frame_count": end - start,
                "prompt_preview": _short_text(prompt),
            }
        )

    gap_ranges = _ranges_for_count(counts, lambda value: value == 0)
    overlap_ranges = _ranges_for_count(counts, lambda value: value > 1)
    if gap_ranges and not bool(plan.get("allow_gaps")):
        raise ValueError("Prompt Relay Preview found gaps while allow_gaps is disabled")
    if overlap_ranges and not bool(plan.get("allow_overlaps")):
        raise ValueError(
            "Prompt Relay Preview found overlaps while allow_overlaps is disabled"
        )

    report = {
        "status": "prompt_relay_plan_ready",
        "ready": True,
        "plan_hash": plan["plan_hash"],
        "frame_count": frame_count,
        "fps": FPS,
        "duration_seconds": frame_count / FPS,
        "event_count": len(events),
        "timing_mode": plan.get("timing_mode"),
        "math_profile": plan.get("math_profile"),
        "query_route": plan.get("query_route", "video_only_paper"),
        "covered_frame_count": sum(value > 0 for value in counts),
        "gap_ranges_inclusive": gap_ranges,
        "overlap_ranges_inclusive": overlap_ranges,
        "events": summaries,
        "model_loaded": False,
        "sampling_executed": False,
    }
    lines = [
        "Prompt Relay Plan: READY",
        (
            f"{frame_count} frames @ {FPS}fps = {frame_count / FPS:.6f}s | "
            f"{len(events)} events | {plan.get('math_profile')} | "
            f"{plan.get('query_route', 'video_only_paper')}"
        ),
        f"plan_hash: {plan['plan_hash']}",
    ]
    for event in summaries:
        lines.append(
            f"[{event['event_index']}] frames {event['start_frame']}-"
            f"{event['end_frame_inclusive']} | {event['start_seconds']:.3f}-"
            f"{event['end_seconds']:.3f}s | {event['prompt_preview']}"
        )
    if gap_ranges:
        lines.append(f"allowed gaps: {gap_ranges}")
    if overlap_ranges:
        lines.append(f"allowed overlaps: {overlap_ranges}")
    timeline_text = "\n".join(lines)
    if len(timeline_text) > MAX_PREVIEW_CHARS:
        timeline_text = timeline_text[: MAX_PREVIEW_CHARS - 1] + "…"
    return (
        plan,
        True,
        len(events),
        timeline_text,
        json.dumps(report, ensure_ascii=False, indent=2),
    )
