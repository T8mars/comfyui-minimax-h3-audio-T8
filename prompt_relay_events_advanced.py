from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping


MAX_RELAY_EVENTS = 32
PROMPT_RELAY_EVENTS_TYPE = "H3_T8_PROMPT_RELAY_EVENTS"
PROMPT_RELAY_EVENTS_SCHEMA = 1


def json_hash(value) -> str:
    packed = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(packed.encode("utf-8")).hexdigest()


def _empty_event_collection() -> dict:
    result = {
        "type": PROMPT_RELAY_EVENTS_TYPE,
        "schema": PROMPT_RELAY_EVENTS_SCHEMA,
        "events": [],
    }
    result["events_hash"] = json_hash(result)
    return result


def validate_prompt_relay_events(value: Mapping | None) -> dict:
    if value is None:
        return _empty_event_collection()
    if not isinstance(value, Mapping):
        raise ValueError("Prompt Relay events must come from the chainable Event node")
    result = dict(value)
    try:
        schema = int(result.get("schema", 0))
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Prompt Relay event collection has an unsupported type/schema"
        ) from error
    if (
        result.get("type") != PROMPT_RELAY_EVENTS_TYPE
        or schema != PROMPT_RELAY_EVENTS_SCHEMA
    ):
        raise ValueError("Prompt Relay event collection has an unsupported type/schema")
    claimed_hash = result.pop("events_hash", None)
    actual_hash = json_hash(result)
    if not isinstance(claimed_hash, str) or claimed_hash != actual_hash:
        raise ValueError(
            "Prompt Relay event collection hash mismatch; rebuild the Event chain"
        )
    events = result.get("events")
    if not isinstance(events, list) or len(events) > MAX_RELAY_EVENTS:
        raise ValueError(
            f"Prompt Relay event collection supports at most {MAX_RELAY_EVENTS} events"
        )
    normalized = []
    for index, event in enumerate(events, 1):
        if not isinstance(event, Mapping):
            raise ValueError(f"Prompt Relay chained event {index} must be an object")
        if set(event) != {"event_index", "prompt", "start", "end"}:
            raise ValueError(
                f"Prompt Relay chained event {index} has an invalid field contract"
            )
        if event.get("event_index") != index:
            raise ValueError("Prompt Relay chained event indices must be sequential")
        prompt = event.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Prompt Relay chained event {index} prompt cannot be empty")
        if prompt != prompt.strip():
            raise ValueError(
                f"Prompt Relay chained event {index} prompt must not have outer whitespace"
            )
        if "\n" in prompt or "\r" in prompt or "|" in prompt:
            raise ValueError(
                f"Prompt Relay chained event {index} prompt cannot contain a newline or |"
            )
        if len(prompt) > 8000:
            raise ValueError(f"Prompt Relay chained event {index} prompt is too long")
        if isinstance(event.get("start"), bool) or isinstance(event.get("end"), bool):
            raise ValueError(
                f"Prompt Relay chained event {index} start/end must be finite numbers"
            )
        try:
            start, end = float(event["start"]), float(event["end"])
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Prompt Relay chained event {index} start/end must be finite numbers"
            ) from error
        if not math.isfinite(start) or not math.isfinite(end):
            raise ValueError(
                f"Prompt Relay chained event {index} start/end must be finite numbers"
            )
        normalized.append(dict(event))
    result["events"] = normalized
    result["events_hash"] = claimed_hash
    return result


def build_prompt_relay_event(
    prompt: str,
    start: float,
    end: float,
    enabled: bool,
    previous_events: Mapping | None = None,
) -> tuple[dict, str, str]:
    collection = validate_prompt_relay_events(previous_events)
    events = [dict(event) for event in collection["events"]]
    if bool(enabled):
        prompt = str(prompt).strip()
        if not prompt:
            raise ValueError("Prompt Relay event prompt cannot be empty")
        if "\n" in prompt or "\r" in prompt or "|" in prompt:
            raise ValueError("Prompt Relay event prompt cannot contain a newline or |")
        if len(prompt) > 8000:
            raise ValueError("Prompt Relay event prompt is too long")
        if len(events) >= MAX_RELAY_EVENTS:
            raise ValueError(f"Prompt Relay supports at most {MAX_RELAY_EVENTS} events")
        if isinstance(start, bool) or isinstance(end, bool):
            raise ValueError("Prompt Relay event start/end must be finite numbers")
        start, end = float(start), float(end)
        if not math.isfinite(start) or not math.isfinite(end):
            raise ValueError("Prompt Relay event start/end must be finite numbers")
        events.append(
            {
                "event_index": len(events) + 1,
                "prompt": prompt,
                "start": start,
                "end": end,
            }
        )

    result = {
        "type": PROMPT_RELAY_EVENTS_TYPE,
        "schema": PROMPT_RELAY_EVENTS_SCHEMA,
        "events": events,
    }
    result["events_hash"] = json_hash(result)
    preview = {
        "event_count": len(events),
        "events": events,
        "timing_note": (
            "start/end are ignored by auto_equal; frames uses an inclusive end, "
            "while seconds/percent use an end boundary; an all-disabled chain is a "
            "global-only no-patch bypass"
        ),
    }
    report = {
        "status": "event_chain_ready",
        "enabled": bool(enabled),
        "event_count": len(events),
        "events_hash": result["events_hash"],
        "maximum_events": MAX_RELAY_EVENTS,
    }
    return (
        result,
        json.dumps(preview, ensure_ascii=False, indent=2),
        json.dumps(report, ensure_ascii=False, indent=2),
    )


def prompt_relay_events_to_inputs(
    prompt_relay_events: Mapping,
    timing_mode: str,
) -> tuple[str, str, int, str]:
    collection = validate_prompt_relay_events(prompt_relay_events)
    events = collection["events"]
    local_prompts = "\n".join(event["prompt"] for event in events)
    if str(timing_mode) == "auto_equal":
        time_ranges = ""
    else:
        time_ranges = "\n".join(
            f"{float(event['start']):.12g}-{float(event['end']):.12g}"
            for event in events
        )
    return local_prompts, time_ranges, len(events), collection["events_hash"]
