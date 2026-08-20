from __future__ import annotations

import json
import math
from collections.abc import Mapping

from .core import FPS, align_frame_count
from .prompt_relay_advanced import build_prompt_relay_plan
from .prompt_relay_events_advanced import (
    MAX_RELAY_EVENTS,
    PROMPT_RELAY_EVENTS_SCHEMA,
    PROMPT_RELAY_EVENTS_TYPE,
    build_prompt_relay_event,
    json_hash,
    prompt_relay_events_to_inputs,
)
from .studio_advanced import PROMPT_PACKET_SCHEMA


__all__ = [
    "MAX_RELAY_EVENTS",
    "PROMPT_RELAY_EVENTS_SCHEMA",
    "PROMPT_RELAY_EVENTS_TYPE",
    "build_prompt_relay_event",
    "build_prompt_relay_plan_from_packet",
]


MAX_RELAY_FRAMES = 3600


def _validate_prompt_packet(prompt_packet: Mapping) -> dict:
    if not isinstance(prompt_packet, Mapping):
        raise ValueError("Prompt Packet → Relay requires an H3_T8_PROMPT_PACKET mapping")
    packet = dict(prompt_packet)
    if packet.get("schema") != PROMPT_PACKET_SCHEMA:
        raise ValueError("Prompt Packet → Relay received an unsupported packet schema")
    if packet.get("backend") != "minimax_h3":
        raise ValueError(
            "Prompt Packet → Relay only supports packets compiled for minimax_h3"
        )
    if packet.get("compiler_only") is not True:
        raise ValueError(
            "Prompt Packet → Relay requires an unmodified Prompt Compiler packet"
        )

    claimed_hash = packet.pop("packet_hash", None)
    actual_hash = json_hash(packet)
    if not isinstance(claimed_hash, str) or claimed_hash != actual_hash:
        raise ValueError(
            "Prompt Packet hash mismatch; rebuild it with T8 Video Prompt Compiler"
        )
    packet["packet_hash"] = claimed_hash

    compiled_prompt = packet.get("compiled_prompt")
    if not isinstance(compiled_prompt, str) or not compiled_prompt.strip():
        raise ValueError("Prompt Packet compiled_prompt cannot be empty")
    duration = packet.get("duration_seconds")
    if isinstance(duration, bool):
        raise ValueError("Prompt Packet duration_seconds must be a finite number")
    try:
        duration = float(duration)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Prompt Packet duration_seconds must be a finite number"
        ) from error
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("Prompt Packet duration_seconds must be finite and positive")
    packet["duration_seconds"] = duration
    return packet


def _event_inputs(events_json: str, timing_mode: str) -> tuple[str, str, int]:
    try:
        payload = json.loads(str(events_json))
    except json.JSONDecodeError as error:
        raise ValueError(f"Prompt Relay events_json is invalid JSON: {error}") from error
    if isinstance(payload, Mapping):
        unknown_wrapper = set(payload) - {"events"}
        if unknown_wrapper:
            raise ValueError(
                "Prompt Relay events_json wrapper has unsupported keys: "
                f"{sorted(unknown_wrapper)}"
            )
        payload = payload.get("events")
    if not isinstance(payload, list):
        raise ValueError("Prompt Relay events_json must contain an event list")
    if len(payload) > MAX_RELAY_EVENTS:
        raise ValueError(
            f"Prompt Relay supports at most {MAX_RELAY_EVENTS} packet events"
        )

    prompts: list[str] = []
    ranges: list[str] = []
    for index, raw in enumerate(payload, 1):
        if not isinstance(raw, Mapping):
            raise ValueError(f"Prompt Relay event {index} must be an object")
        unknown = set(raw) - {"prompt", "start", "end"}
        if unknown:
            raise ValueError(
                f"Prompt Relay event {index} has unsupported keys: {sorted(unknown)}"
            )
        prompt = raw.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Prompt Relay event {index} prompt cannot be empty")
        prompt = prompt.strip()
        if "\n" in prompt or "\r" in prompt or "|" in prompt:
            raise ValueError(
                f"Prompt Relay event {index} prompt cannot contain a newline or |"
            )
        if len(prompt) > 8000:
            raise ValueError(f"Prompt Relay event {index} prompt is too long")
        prompts.append(prompt)

        has_timing = "start" in raw or "end" in raw
        if timing_mode == "auto_equal":
            if has_timing:
                raise ValueError(
                    f"Prompt Relay event {index} includes start/end while timing_mode "
                    "is auto_equal; remove them or select an explicit timing mode"
                )
            continue
        if "start" not in raw or "end" not in raw:
            raise ValueError(
                f"Prompt Relay event {index} requires start and end for {timing_mode}"
            )
        if isinstance(raw["start"], bool) or isinstance(raw["end"], bool):
            raise ValueError(f"Prompt Relay event {index} start/end must be numbers")
        try:
            start, end = float(raw["start"]), float(raw["end"])
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Prompt Relay event {index} start/end must be numbers"
            ) from error
        if not math.isfinite(start) or not math.isfinite(end):
            raise ValueError(f"Prompt Relay event {index} start/end must be finite")
        ranges.append(f"{start:.12g}-{end:.12g}")

    if sum(len(prompt) for prompt in prompts) > 30000:
        raise ValueError("Prompt Relay packet events exceed 30000 prompt characters")
    return "\n".join(prompts), "\n".join(ranges), len(prompts)


def build_prompt_relay_plan_from_packet(
    prompt_packet: Mapping,
    events_json: str,
    timing_mode: str,
    math_profile: str,
    epsilon: float,
    allow_gaps: bool,
    allow_overlaps: bool,
    prompt_relay_events: Mapping | None = None,
) -> tuple[dict, str, int, str, str]:
    packet = _validate_prompt_packet(prompt_packet)
    event_collection_hash = None
    if prompt_relay_events is None:
        local_prompts, time_ranges, event_count = _event_inputs(
            events_json,
            str(timing_mode),
        )
        event_source = "events_json"
    else:
        local_prompts, time_ranges, event_count, event_collection_hash = (
            prompt_relay_events_to_inputs(
                prompt_relay_events,
                str(timing_mode),
            )
        )
        event_source = "typed_event_chain"

    requested_frames = max(1, int(round(packet["duration_seconds"] * FPS)))
    aligned_frames = align_frame_count(requested_frames)
    if aligned_frames > MAX_RELAY_FRAMES:
        raise ValueError(
            "Prompt Packet duration resolves to "
            f"{aligned_frames} aligned frames, above the Prompt Relay limit "
            f"of {MAX_RELAY_FRAMES}"
        )

    plan, compiled_prompt, frame_count, timeline_json, report_json = (
        build_prompt_relay_plan(
            global_prompt=packet["compiled_prompt"],
            local_prompts=local_prompts,
            length=aligned_frames,
            timing_mode=str(timing_mode),
            time_ranges=time_ranges,
            math_profile=str(math_profile),
            epsilon=float(epsilon),
            allow_gaps=bool(allow_gaps),
            allow_overlaps=bool(allow_overlaps),
        )
    )

    plan = dict(plan)
    plan.pop("plan_hash", None)
    plan["source_prompt_packet"] = {
        "schema": packet["schema"],
        "packet_hash": packet["packet_hash"],
        "backend": packet["backend"],
        "duration_seconds": packet["duration_seconds"],
        "aspect_ratio": packet.get("aspect_ratio"),
    }
    plan["source_events"] = {
        "source": event_source,
        "events_hash": event_collection_hash,
        "event_count": event_count,
    }
    plan["plan_hash"] = json_hash(plan)

    report = json.loads(report_json)
    report.update(
        {
            "status": (
                "packet_relay_plan_ready"
                if event_count
                else "packet_relay_plan_bypass_no_events"
            ),
            "source_packet_schema": packet["schema"],
            "source_packet_hash": packet["packet_hash"],
            "source_backend": packet["backend"],
            "source_compiler_only": True,
            "source_duration_seconds": packet["duration_seconds"],
            "requested_frame_count": requested_frames,
            "aligned_frame_count": frame_count,
            "aligned_duration_seconds": frame_count / FPS,
            "duration_alignment_delta_seconds": (
                frame_count / FPS - packet["duration_seconds"]
            ),
            "event_count": event_count,
            "event_source": event_source,
            "event_collection_hash": event_collection_hash,
            "prompt_source": "packet.compiled_prompt",
            "automatic_event_rewrite": False,
            "plan_hash": plan["plan_hash"],
        }
    )
    report["notes"] = list(report.get("notes", [])) + [
        "the Prompt Compiler packet remains the authoritative global H3 visual/audio prompt",
        "the selected event source is explicit and human-auditable; this node does not infer or rewrite events",
        "duration is rounded at 24fps and then aligned upward to the H3 17n+5 frame grid",
        "zero active events keep the packet prompt global and install no Relay attention patch",
    ]
    return (
        plan,
        compiled_prompt,
        frame_count,
        timeline_json,
        json.dumps(report, ensure_ascii=False, indent=2),
    )
