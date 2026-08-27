from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
import re
from typing import Any

from .core import FPS
from .long_video_orchestration import build_long_video_chain_plan


VOICE_CONTEXT_PLAN_TYPE = "H3_T8_LONG_VIDEO_VOICE_CONTEXT_PLAN"
VOICE_CONTEXT_SCHEMA = "t8.minimax_h3.long_video_voice_context.v1"
VOICE_CONTEXT_GATE_SCHEMA = "t8.minimax_h3.long_video_voice_context_gate.v1"
_PUNCTUATION_RE = re.compile(r"[。！？!?；;.!?]\s*$")


def _canonical_json(value: object, *, indent: int | None = None) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":") if indent is None else None,
        indent=indent,
        allow_nan=False,
    )


def _hash(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _load_json(value: str, name: str) -> Any:
    text = str(value or "").strip()
    if not text:
        return {} if name == "voice_bindings_json" else []
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"{name} is invalid JSON: {error}") from error


def _voice_bindings(value: str) -> dict[str, int]:
    payload = _load_json(value, "voice_bindings_json")
    if isinstance(payload, Mapping) and "bindings" in payload:
        payload = payload["bindings"]
    result: dict[str, int] = {}
    if isinstance(payload, Mapping):
        source = payload.items()
    elif isinstance(payload, list):
        source = []
        for index, item in enumerate(payload):
            if not isinstance(item, Mapping):
                raise ValueError(f"voice binding {index} must be an object")
            source.append((item.get("character_id", item.get("speaker")), item.get("audio_ordinal")))
    else:
        raise ValueError("voice_bindings_json must be an object or a list of binding objects")
    for raw_character, raw_ordinal in source:
        character = str(raw_character or "").strip()
        if not character:
            raise ValueError("every voice binding needs a non-empty character_id")
        if character in result:
            raise ValueError(f"duplicate voice binding for {character}")
        ordinal = int(raw_ordinal)
        if not 1 <= ordinal <= 9:
            raise ValueError(f"audio ordinal for {character} must be between 1 and 9")
        result[character] = ordinal
    return result


def _dialogue_turns(value: str, total_frames: int) -> list[dict[str, Any]]:
    payload = _load_json(value, "dialogue_timeline_json")
    if isinstance(payload, Mapping) and "turns" in payload:
        payload = payload["turns"]
    if not isinstance(payload, list):
        raise ValueError("dialogue_timeline_json must be a list or {turns:[...]}")
    turns: list[dict[str, Any]] = []
    for index, raw in enumerate(payload):
        if not isinstance(raw, Mapping):
            raise ValueError(f"dialogue turn {index} must be an object")
        character = str(raw.get("character_id", raw.get("speaker", "")) or "").strip()
        text = str(raw.get("text", "") or "").strip()
        if not character or not text:
            raise ValueError(f"dialogue turn {index} needs character_id/speaker and text")
        start_seconds = float(raw.get("start_seconds", -1.0))
        end_seconds = float(raw.get("end_seconds", -1.0))
        if not math.isfinite(start_seconds) or not math.isfinite(end_seconds):
            raise ValueError(f"dialogue turn {index} times must be finite")
        start_frame = round(start_seconds * FPS)
        end_frame = round(end_seconds * FPS)
        if start_frame < 0 or end_frame <= start_frame or end_frame > total_frames:
            raise ValueError(
                f"dialogue turn {index} must satisfy 0 <= start < end <= total duration"
            )
        turns.append(
            {
                "turn_index": index,
                "character_id": character,
                "text": text,
                "start_seconds": start_frame / FPS,
                "end_seconds": end_frame / FPS,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "sentence_terminal": bool(_PUNCTUATION_RE.search(text)),
            }
        )
    turns.sort(key=lambda item: (item["start_frame"], item["end_frame"], item["turn_index"]))
    return turns


def _segment_bounds(segment) -> tuple[int, int, int]:
    start = round(segment.plan.timeline_start_seconds * FPS)
    stop = round(segment.plan.timeline_end_seconds * FPS)
    render_start = max(0, start - int(segment.plan.context_frames))
    return start, stop, render_start


def build_long_video_voice_context_plan(
    chain_id: str,
    total_duration_seconds: float,
    render_window_frames: int,
    context_frames: int,
    global_prompt: str,
    dialogue_timeline_json: str,
    voice_bindings_json: str,
    cross_boundary_policy: str = "abstain",
    first_shot_review_required: bool = True,
) -> tuple[dict, str, str, bool, bool, str]:
    if cross_boundary_policy not in {"abstain", "duplicate_exact_text_exp"}:
        raise ValueError("cross_boundary_policy must be abstain or duplicate_exact_text_exp")
    segments = build_long_video_chain_plan(
        chain_id,
        total_duration_seconds,
        render_window_frames,
        context_frames,
        global_prompt,
        "",
        0,
        "fixed",
    )
    total_frames = sum(item.plan.final_frame_count for item in segments)
    bindings = _voice_bindings(voice_bindings_json)
    turns = _dialogue_turns(dialogue_timeline_json, total_frames)
    missing_bindings = sorted({item["character_id"] for item in turns} - set(bindings))
    if missing_bindings:
        raise ValueError("missing voice binding(s): " + ", ".join(missing_bindings))

    boundaries = [round(item.plan.timeline_end_seconds * FPS) for item in segments[:-1]]
    crossed: list[dict[str, Any]] = []
    for turn in turns:
        hit = [value for value in boundaries if turn["start_frame"] < value < turn["end_frame"]]
        if hit:
            crossed.append({**turn, "crossed_boundaries": hit})

    segment_items: list[dict[str, Any]] = []
    prompt_overrides: list[dict[str, Any]] = []
    audio_pins: list[dict[str, Any]] = []
    for segment in segments:
        start, stop, render_start = _segment_bounds(segment)
        overlaps = [
            turn for turn in turns
            if turn["start_frame"] < stop and turn["end_frame"] > start
        ]
        prompt_lines = [str(global_prompt or "").strip()]
        segment_pins = []
        for turn in overlaps:
            ordinal = bindings[turn["character_id"]]
            overlap_start = max(start, turn["start_frame"])
            overlap_stop = min(stop, turn["end_frame"])
            is_crossed = turn["start_frame"] < start or turn["end_frame"] > stop
            if not is_crossed or cross_boundary_policy == "duplicate_exact_text_exp":
                prompt_lines.append(
                    f'<Audio {ordinal}> {turn["character_id"]} says exactly: "{turn["text"]}"'
                )
            pin = {
                "turn_index": turn["turn_index"],
                "character_id": turn["character_id"],
                "audio_ordinal": ordinal,
                "global_start_frame": turn["start_frame"],
                "global_end_frame": turn["end_frame"],
                "segment_overlap_start_frame": overlap_start,
                "segment_overlap_end_frame": overlap_stop,
                "render_local_start_frame": max(0, overlap_start - render_start),
                "render_local_end_frame": min(
                    int(segment.plan.render_frames), overlap_stop - render_start
                ),
                "crosses_segment_boundary": is_crossed,
                "audio_injection": "reference_tag_only_no_waveform_reinjection",
            }
            segment_pins.append(pin)
            audio_pins.append({"segment_index": segment.index, **pin})
        prompt = "\n".join(line for line in prompt_lines if line).strip()
        prompt_overrides.append(
            {
                "prompt": prompt,
                "note": (
                    "voice context compiled from exact dialogue frames; set "
                    "prompt_primary_audio_ordinal=0 so explicit <Audio N> tags are not remapped"
                ),
            }
        )
        segment_items.append(
            {
                "segment_index": segment.index,
                "timeline_start_frame": start,
                "timeline_end_frame": stop,
                "render_start_frame": render_start,
                "render_frame_count": int(segment.plan.render_frames),
                "context_frames": int(segment.plan.context_frames),
                "dialogue_turn_indices": [item["turn_index"] for item in overlaps],
                "audio_pin_frames": segment_pins,
                "prompt": prompt,
            }
        )

    ready = not crossed or cross_boundary_policy == "duplicate_exact_text_exp"
    status = "ready" if ready else "abstain_cross_boundary_sentence"
    plan = {
        "schema": VOICE_CONTEXT_SCHEMA,
        "chain_id": str(chain_id),
        "status": status,
        "ready": ready,
        "first_shot_review_required": bool(first_shot_review_required),
        "cross_boundary_policy": cross_boundary_policy,
        "fixed_h3_segment_boundaries": True,
        "adaptive_boundary_shift": False,
        "adaptive_boundary_reason": (
            "non-final H3 continuation context must end at the sampled and delivered latent tail"
        ),
        "prompt_primary_audio_ordinal_required": 0,
        "voice_bindings": bindings,
        "turns": turns,
        "cross_boundary_turns": crossed,
        "segments": segment_items,
        "audio_reference_contract": (
            "audio_pin_frames are an identity-reference/prompt routing plan only; drive/final audio "
            "continues to use the existing independent global timeline window"
        ),
        "review_route": (
            "use the existing Background/Accepted long-video workflow when first-shot review is "
            "required; the one-call in-node loop cannot pause inside one node execution"
        ),
    }
    plan["plan_hash"] = _hash(plan)
    prompt_json = json.dumps({"segments": prompt_overrides}, ensure_ascii=False, indent=2)
    pins_json = json.dumps(
        {"schema": VOICE_CONTEXT_SCHEMA, "audio_pin_frames": audio_pins},
        ensure_ascii=False,
        indent=2,
    )
    report = {
        "schema": VOICE_CONTEXT_SCHEMA,
        "status": status,
        "ready": ready,
        "segment_count": len(segments),
        "turn_count": len(turns),
        "voice_count": len(bindings),
        "cross_boundary_turn_count": len(crossed),
        "sentence_terminal_count": sum(item["sentence_terminal"] for item in turns),
        "no_dialogue_is_valid": not turns,
        "first_shot_review_required": bool(first_shot_review_required),
        "plan_hash": plan["plan_hash"],
        "warnings": (
            [
                "one or more dialogue turns cross a fixed H3 segment boundary; split the timeline "
                "at a natural punctuation/time point or explicitly choose the experimental duplicate policy"
            ]
            if crossed and not ready
            else []
        ),
    }
    return (
        plan,
        prompt_json,
        pins_json,
        ready,
        bool(first_shot_review_required),
        json.dumps(report, ensure_ascii=False, indent=2),
    )


def release_long_video_voice_context_plan(
    plan: Mapping[str, Any], first_shot_approved: bool
) -> tuple[str, str, bool, str]:
    if not isinstance(plan, Mapping) or plan.get("schema") != VOICE_CONTEXT_SCHEMA:
        raise TypeError("plan must come from MiniMax H3 Long Video Voice Context")
    plan_hash = str(plan.get("plan_hash", ""))
    expected = _hash({key: value for key, value in plan.items() if key != "plan_hash"})
    if plan_hash != expected:
        raise ValueError("voice context plan hash does not match its contents")
    mechanically_ready = bool(plan.get("ready"))
    review_required = bool(plan.get("first_shot_review_required"))
    approved = bool(first_shot_approved)
    released = mechanically_ready and (approved or not review_required)
    prompts = [
        {"prompt": str(item.get("prompt", "")), "note": "released voice context plan"}
        for item in plan.get("segments", [])
    ]
    prompt_json = (
        json.dumps({"segments": prompts}, ensure_ascii=False, indent=2) if released else ""
    )
    pins = [
        {"segment_index": item.get("segment_index"), **dict(pin)}
        for item in plan.get("segments", [])
        for pin in item.get("audio_pin_frames", [])
    ]
    pins_json = (
        json.dumps(
            {"schema": VOICE_CONTEXT_GATE_SCHEMA, "audio_pin_frames": pins},
            ensure_ascii=False,
            indent=2,
        )
        if released
        else ""
    )
    status = (
        "released"
        if released
        else "awaiting_first_shot_review"
        if mechanically_ready and review_required
        else "abstain_plan_not_ready"
    )
    report = {
        "schema": VOICE_CONTEXT_GATE_SCHEMA,
        "status": status,
        "released": released,
        "mechanically_ready": mechanically_ready,
        "first_shot_review_required": review_required,
        "first_shot_approved": approved,
        "plan_hash": plan_hash,
        "execution_note": (
            "connect released segment_prompts_json to the existing long-video runner and set "
            "prompt_primary_audio_ordinal=0; use Background/Accepted for a real pause-after-shot-0 gate"
        ),
    }
    return prompt_json, pins_json, released, json.dumps(report, ensure_ascii=False, indent=2)
