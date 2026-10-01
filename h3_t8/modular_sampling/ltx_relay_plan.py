"""A separate LTX output-frame timeline for opt-in external Prompt Relay.

H3's 17n+5 planner cannot describe the 8n+1 frames represented by an LTX
video latent. This module deliberately plans time only: it neither encodes
text nor claims to have changed attention or sampled a candidate.
"""
from collections.abc import Mapping
import json
import math

from ..prompt_relay_advanced import (
    _auto_equal_ranges,
    _local_prompt_lines,
    _parse_range,
    _sha256_json,
    _validate_ranges,
)
from ..prompt_relay_events_advanced import prompt_relay_events_to_inputs
from .ltx_eav import _latent_shape


PLAN_TYPE = "T8_LTX_PROMPT_RELAY_PLAN"
PLAN_SCHEMA = 1
TIMING_MODES = ("auto_equal", "frames", "seconds", "percent")


def _ranges(mode, raw, count, frames, fps):
    if mode == "auto_equal":
        return _auto_equal_ranges(count, frames)
    lines = [line.strip() for line in str(raw).splitlines() if line.strip()]
    if len(lines) != count:
        raise ValueError(f"LTX Relay needs {count} time ranges, got {len(lines)}")
    ranges = []
    for line in lines:
        start, end = _parse_range(line)
        if mode == "frames":
            if not start.is_integer() or not end.is_integer():
                raise ValueError("LTX Relay frame ranges require integer indices")
            ranges.append((int(start), int(end) + 1))
        elif mode == "seconds":
            ranges.append((round(start * fps), round(end * fps)))
        elif mode == "percent":
            if not 0 <= start <= end <= 100:
                raise ValueError("LTX Relay percent ranges require 0..100")
            ranges.append((round(frames * start / 100), round(frames * end / 100)))
        else:
            raise ValueError(f"Unknown LTX Relay timing mode {mode!r}")
    return ranges


def build_ltx_relay_plan(
    ltx_latent,
    global_prompt,
    local_prompts,
    timing_mode,
    time_ranges,
    fps,
    epsilon,
    allow_gaps,
    allow_overlaps,
    prompt_relay_events=None,
):
    shape = _latent_shape(ltx_latent)
    frames = (shape[2] - 1) * 8 + 1
    global_prompt = str(global_prompt)
    if not global_prompt.strip():
        raise ValueError("LTX Relay global prompt cannot be empty")
    if timing_mode not in TIMING_MODES:
        raise ValueError(f"Unknown LTX Relay timing mode {timing_mode!r}")
    if isinstance(fps, bool) or not math.isfinite(float(fps)) or not 0 < float(fps) <= 1000:
        raise ValueError("LTX Relay fps must be finite and within (0,1000]")
    if isinstance(epsilon, bool) or not math.isfinite(float(epsilon)) or not 0 < float(epsilon) < 1:
        raise ValueError("LTX Relay epsilon must be strictly between 0 and 1")
    fps, epsilon = float(fps), float(epsilon)

    source_events = None
    if prompt_relay_events is not None:
        local_prompts, time_ranges, count, events_hash = prompt_relay_events_to_inputs(
            prompt_relay_events, timing_mode
        )
        source_events = {"events_hash": events_hash, "event_count": count}
    prompts = _local_prompt_lines(local_prompts)
    if not prompts and str(time_ranges).strip():
        raise ValueError("LTX Relay global-only plan cannot include time ranges")
    ranges = _ranges(timing_mode, time_ranges, len(prompts), frames, fps) if prompts else []
    if ranges:
        _validate_ranges(ranges, frames, bool(allow_gaps), bool(allow_overlaps))

    compiled = f"Global scene: {global_prompt}"
    events = []
    for index, (prompt, (start, end)) in enumerate(zip(prompts, ranges), 1):
        compiled += "\n"
        char_start = len(compiled)
        compiled += f"Event {index}: {prompt}"
        events.append({"event_index": index, "local_prompt": prompt,
                       "start_frame": start, "end_frame_exclusive": end,
                       "start_seconds": start / fps, "end_seconds": end / fps,
                       "prompt_char_start": char_start, "prompt_char_end": len(compiled)})
    plan = {
        "type": PLAN_TYPE, "schema": PLAN_SCHEMA,
        "temporal_contract": "ltx_output_frames_8n_plus_1",
        "math_profile": "paper_v1_ltx_adaptation_exp",
        "latent_shape": list(shape), "frame_count": frames,
        "fps": fps, "epsilon": epsilon, "timing_mode": timing_mode,
        "global_prompt": global_prompt, "compiled_prompt": compiled,
        "allow_gaps": bool(allow_gaps), "allow_overlaps": bool(allow_overlaps),
        "events": events,
    }
    if source_events is not None:
        plan["source_events"] = source_events
    plan["plan_hash"] = _sha256_json(plan)
    timeline = {"frame_count": frames, "fps": fps, "events": events}
    report = {"status": "timeline_ready" if events else "global_only_bypass",
              "plan_hash": plan["plan_hash"], "event_count": len(events),
              "frame_count": frames, "latent_frames": shape[2],
              "attention_applied": False, "sampling_verified": False,
              "paper_qualified_for_ltx": False,
              "note": "LTX timeline only; text encoding and attention bias require separate nodes."}
    return (plan, compiled, frames, json.dumps(timeline, ensure_ascii=False, indent=2),
            json.dumps(report, ensure_ascii=False, indent=2))


def validate_ltx_relay_plan(plan: Mapping, ltx_latent=None):
    if not isinstance(plan, Mapping):
        raise ValueError("LTX Relay plan must be a mapping")
    checked = dict(plan)
    claimed = checked.pop("plan_hash", None)
    if (checked.get("type") != PLAN_TYPE or checked.get("schema") != PLAN_SCHEMA
            or claimed != _sha256_json(checked)):
        raise ValueError("LTX Relay plan type/schema/hash mismatch; rebuild it")
    shape = checked.get("latent_shape")
    if (not isinstance(shape, list) or len(shape) != 5 or shape[1] != 128
            or not all(type(value) is int and value > 0 for value in shape)
            or shape[2] < 2 or checked.get("frame_count") != (shape[2] - 1) * 8 + 1):
        raise ValueError("LTX Relay plan has invalid latent frame geometry")
    if ltx_latent is not None and list(_latent_shape(ltx_latent)) != shape:
        raise ValueError("LTX Relay plan does not match the connected LTX latent")
    checked["plan_hash"] = claimed
    return checked
