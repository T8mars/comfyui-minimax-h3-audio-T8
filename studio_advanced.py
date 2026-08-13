from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence

from .core import FPS, MAX_TRAINED_FRAMES, align_frame_count


CAST_SCHEMA = "t8.minimax_h3.unified_cast.v1"
SOUND_CANVAS_SCHEMA = "t8.minimax_h3.sound_canvas.v1"
PROMPT_PACKET_SCHEMA = "t8.video_prompt_packet.v1"
STUDIO_TIMELINE_SCHEMA = "t8.minimax_h3.studio_timeline.v1"
REPAIR_PLAN_SCHEMA = "t8.minimax_h3.selective_repair.v1"
MAX_UINT64 = 0xFFFFFFFFFFFFFFFF
MIN_STUDIO_FRAMES = 22
BACKENDS = ("minimax_h3", "wan_2_2", "ltx_video", "generic_cinematic")
SOUND_ROLES = ("dialogue", "music", "ambience", "sfx")
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)


def _hash(value: Any) -> str:
    packed = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(packed.encode("utf-8")).hexdigest()


def _text(value: Any, name: str, *, required=False, maximum=16000) -> str:
    value = "" if value is None else value
    if not isinstance(value, str):
        raise ValueError(f"{name} must be text")
    value = value.strip()
    if required and not value:
        raise ValueError(f"{name} cannot be empty")
    if len(value) > maximum:
        raise ValueError(f"{name} exceeds {maximum} characters")
    return value


def _json(value: str, name: str, empty: Any) -> Any:
    text = _text(value, name, maximum=1_000_000)
    if not text:
        return deepcopy(empty)
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"{name} is invalid JSON: {error}") from error


def _safe_id(value: Any, name: str) -> str:
    value = _text(value, name, required=True, maximum=64)
    if not SAFE_ID.fullmatch(value):
        raise ValueError(f"{name} contains unsupported characters")
    return value


def _text_list(value: Any, name: str, maximum_items=32) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [item.strip() for item in value.split(";") if item.strip()]
    if not isinstance(value, list) or len(value) > maximum_items:
        raise ValueError(f"{name} must be a list with at most {maximum_items} items")
    return [_text(item, f"{name}[]", required=True, maximum=1000) for item in value]


def build_unified_cast(cast_json: str, strict_identity=True) -> dict[str, Any]:
    payload = _json(cast_json, "cast_json", [])
    if isinstance(payload, Mapping):
        payload = payload.get("characters", payload)
        if isinstance(payload, Mapping):
            payload = [dict(value, id=key) for key, value in payload.items()]
    if not isinstance(payload, list) or not payload:
        raise ValueError("cast_json must contain at least one character")
    if len(payload) > 32:
        raise ValueError("Unified Cast supports at most 32 characters")
    characters: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(payload):
        if not isinstance(raw, Mapping):
            raise ValueError(f"character {index} must be an object")
        character_id = _safe_id(raw.get("id"), f"character {index} id")
        if character_id in characters:
            raise ValueError(f"duplicate character id: {character_id}")
        characters[character_id] = {
            "id": character_id,
            "name": _text(raw.get("name", character_id), "name", required=True, maximum=128),
            "visual_identity": _text(raw.get("visual_identity", ""), f"{character_id}.visual_identity", required=bool(strict_identity), maximum=4000),
            "wardrobe_default": _text(raw.get("wardrobe_default", ""), "wardrobe", maximum=2000),
            "voice_direction": _text(raw.get("voice_direction", ""), "voice_direction", maximum=2000),
            "behavior_rules": _text_list(raw.get("behavior_rules", []), "behavior_rules"),
            "forbidden_changes": _text_list(raw.get("forbidden_changes", []), "forbidden_changes"),
            "reference_slots": _text_list(raw.get("reference_slots", []), "reference_slots"),
        }
    fragments = []
    for character in characters.values():
        parts = [f"{character['name']} [{character['id']}]: {character['visual_identity']}"]
        if character["wardrobe_default"]:
            parts.append("wardrobe: " + character["wardrobe_default"])
        if character["behavior_rules"]:
            parts.append("continuity: " + "; ".join(character["behavior_rules"]))
        if character["forbidden_changes"]:
            parts.append("do not change: " + "; ".join(character["forbidden_changes"]))
        fragments.append("; ".join(parts))
    result = {
        "schema": CAST_SCHEMA,
        "characters": characters,
        "character_order": list(characters),
        "prompt_fragment": "\n".join(fragments),
        "strict_identity": bool(strict_identity),
        "runtime_voice_profiles_attached": False,
    }
    result["cast_hash"] = _hash(result)
    return result


def validate_cast(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or value.get("schema") != CAST_SCHEMA:
        raise ValueError("cast must be a MiniMax H3 Unified Cast object")
    return dict(value)


def build_sound_canvas(
    events_json: str,
    total_duration_seconds: float,
    no_unrequested_speech=True,
    allow_dialogue_overlap=False,
) -> dict[str, Any]:
    duration = float(total_duration_seconds)
    if not math.isfinite(duration) or not 0 < duration <= 86400:
        raise ValueError("total_duration_seconds must be finite and between 0 and 86400")
    payload = _json(events_json, "events_json", [])
    if isinstance(payload, Mapping):
        payload = payload.get("events", [])
    if not isinstance(payload, list) or len(payload) > 1024:
        raise ValueError("events_json must contain at most 1024 events")
    events, ids, warnings = [], set(), []
    for index, raw in enumerate(payload):
        if not isinstance(raw, Mapping):
            raise ValueError(f"sound event {index} must be an object")
        event_id = _safe_id(raw.get("id", f"sound_{index:03d}"), "sound event id")
        if event_id in ids:
            raise ValueError(f"duplicate sound event id: {event_id}")
        ids.add(event_id)
        role = str(raw.get("role", "ambience")).strip().lower()
        if role not in SOUND_ROLES:
            raise ValueError(f"sound event {event_id} has unsupported role")
        start = float(raw.get("start_seconds", 0))
        end = float(raw.get("end_seconds", duration))
        if not all(math.isfinite(v) for v in (start, end)) or start < 0 or end <= start or end > duration + 1e-9:
            raise ValueError(f"sound event {event_id} has an invalid time range")
        exact = _text(raw.get("exact_text", ""), "exact_text", maximum=8000)
        description = _text(raw.get("description", ""), "description", maximum=4000)
        if role == "dialogue" and not exact:
            raise ValueError(f"dialogue event {event_id} requires exact_text")
        if role != "dialogue" and not description:
            raise ValueError(f"{role} event {event_id} requires description")
        gain = float(raw.get("gain_db", 0))
        pan = float(raw.get("pan", 0))
        if not math.isfinite(gain) or not -60 <= gain <= 24 or not math.isfinite(pan) or not -1 <= pan <= 1:
            raise ValueError(f"sound event {event_id} gain/pan is outside the supported range")
        events.append({
            "id": event_id,
            "role": role,
            "start_seconds": start,
            "end_seconds": end,
            "speaker_id": _text(raw.get("speaker_id", ""), "speaker_id", maximum=64),
            "exact_text": exact,
            "description": description,
            "gain_db": gain,
            "pan": pan,
            "duck_under_dialogue": bool(raw.get("duck_under_dialogue", role != "dialogue")),
        })
    events.sort(key=lambda item: (item["start_seconds"], item["end_seconds"], item["id"]))
    dialogues = [event for event in events if event["role"] == "dialogue"]
    for previous, current in zip(dialogues, dialogues[1:]):
        if current["start_seconds"] < previous["end_seconds"]:
            message = f"dialogue events {previous['id']} and {current['id']} overlap"
            if not allow_dialogue_overlap:
                raise ValueError(message)
            warnings.append(message + "; joint speaker identity is not guaranteed")
    lines = ["Audio timeline (times are relative to this canvas):"]
    for event in events:
        timing = f"{event['start_seconds']:.3f}-{event['end_seconds']:.3f}s"
        if event["role"] == "dialogue":
            speaker = f" by {event['speaker_id']}" if event["speaker_id"] else ""
            lines.append(f"- {timing} dialogue{speaker}; say exactly: {event['exact_text']}")
        else:
            lines.append(f"- {timing} {event['role']}: {event['description']}")
    if no_unrequested_speech:
        lines.append(
            "Do not add, repeat, paraphrase, mumble, whisper, or improvise speech outside the exact dialogue events. "
            "When dialogue ends, continue only the requested music, ambience, and sound effects until the scene ends."
        )
    result = {
        "schema": SOUND_CANVAS_SCHEMA,
        "total_duration_seconds": duration,
        "events": events,
        "no_unrequested_speech": bool(no_unrequested_speech),
        "allow_dialogue_overlap": bool(allow_dialogue_overlap),
        "h3_audio_prompt": "\n".join(lines),
        "warnings": warnings,
        "source_separation_claim": False,
        "exact_timing_claim": False,
    }
    result["sound_canvas_hash"] = _hash(result)
    return result


def validate_sound_canvas(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or value.get("schema") != SOUND_CANVAS_SCHEMA:
        raise ValueError("sound_canvas must be a MiniMax H3 Sound Canvas object")
    return dict(value)


def sound_canvas_window(canvas, start_seconds: float, end_seconds: float):
    canvas = validate_sound_canvas(canvas)
    if canvas is None:
        return None
    events = []
    for source in canvas["events"]:
        if source["end_seconds"] <= start_seconds or source["start_seconds"] >= end_seconds:
            continue
        event = dict(source)
        event["start_seconds"] = max(0.0, source["start_seconds"] - start_seconds)
        event["end_seconds"] = min(end_seconds, source["end_seconds"]) - start_seconds
        events.append(event)
    return build_sound_canvas(
        canonical_json(events),
        end_seconds - start_seconds,
        canvas["no_unrequested_speech"],
        canvas["allow_dialogue_overlap"],
    )


def _cast_fragment(cast, cast_ids: Sequence[str]) -> str:
    cast = validate_cast(cast)
    if cast is None:
        if cast_ids:
            raise ValueError("shot has cast_ids but no Unified Cast is connected")
        return ""
    ids = list(cast_ids) if cast_ids else list(cast["character_order"])
    fragments = []
    for character_id in ids:
        if character_id not in cast["characters"]:
            raise ValueError(f"unknown cast id: {character_id}")
        character = cast["characters"][character_id]
        value = f"{character['name']} [{character_id}]: {character['visual_identity']}"
        if character["wardrobe_default"]:
            value += "; wardrobe: " + character["wardrobe_default"]
        if character["forbidden_changes"]:
            value += "; preserve: " + "; ".join(character["forbidden_changes"])
        fragments.append(value)
    return "\n".join(fragments)


def compile_prompt_packet(
    prompt: str,
    backend: str,
    duration_seconds: float,
    aspect_ratio: str,
    dialogue="",
    negative_prompt="",
    strict_exact_dialogue=True,
    cast=None,
    cast_ids: Sequence[str] = (),
    sound_canvas=None,
    creative_brief=None,
) -> dict[str, Any]:
    prompt = _text(prompt, "prompt", required=True, maximum=30000)
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}")
    duration = float(duration_seconds)
    if not math.isfinite(duration) or not 0 < duration <= 86400:
        raise ValueError("duration_seconds must be finite and between 0 and 86400")
    aspect = _text(aspect_ratio, "aspect_ratio", required=True, maximum=32)
    dialogue = _text(dialogue, "dialogue", maximum=12000)
    negative = _text(negative_prompt, "negative_prompt", maximum=12000)
    brief = dict(creative_brief or {})
    allowed = {"camera", "action", "scene", "lighting", "style", "continuity", "notes"}
    unknown = set(brief) - allowed
    if unknown:
        raise ValueError(f"creative_brief contains unsupported keys: {sorted(unknown)}")
    visual = [prompt]
    for key in ("scene", "action", "camera", "lighting", "style", "continuity"):
        value = _text(brief.get(key, ""), key, maximum=4000)
        if value:
            visual.append(f"{key.title()}: {value}")
    cast_text = _cast_fragment(cast, cast_ids)
    if cast_text:
        visual.append("Cast continuity:\n" + cast_text)
    visual.append(f"Target duration: {duration:.3f}s. Aspect ratio: {aspect}.")
    visual_prompt = "\n".join(visual)
    canvas = validate_sound_canvas(sound_canvas)
    audio = []
    if dialogue:
        audio.append("Spoken dialogue, exact wording: " + dialogue)
        if strict_exact_dialogue:
            audio.append(
                "Do not add, repeat, paraphrase, mumble, whisper, or improvise speech after this line. "
                "Continue only requested non-speech audio."
            )
    if canvas:
        audio.append(canvas["h3_audio_prompt"])
    audio_prompt = "\n".join(audio)
    if backend == "minimax_h3":
        compiled = visual_prompt + (("\n\nAudio direction:\n" + audio_prompt) if audio_prompt else "")
        note = "H3 joint visual/audio prompt; timing and exact speech remain generative, not guaranteed."
    elif backend == "wan_2_2":
        compiled = visual_prompt
        note = "Visual prompt only; Sound Canvas remains sidecar metadata."
    elif backend == "ltx_video":
        compiled = "Cinematic shot specification:\n" + visual_prompt
        note = "Visual prompt with sidecar audio plan; runtime support must be validated."
    else:
        compiled = visual_prompt
        note = "Backend-neutral visual prompt with structured audio sidecar."
    result = {
        "schema": PROMPT_PACKET_SCHEMA,
        "backend": backend,
        "compiled_prompt": compiled,
        "visual_prompt": visual_prompt,
        "audio_prompt": audio_prompt,
        "negative_prompt": negative,
        "duration_seconds": duration,
        "aspect_ratio": aspect,
        "dialogue": dialogue,
        "strict_exact_dialogue": bool(strict_exact_dialogue),
        "cast_ids": list(cast_ids),
        "cast_hash": cast.get("cast_hash") if cast else None,
        "sound_canvas_hash": canvas.get("sound_canvas_hash") if canvas else None,
        "backend_note": note,
        "compiler_only": True,
        "quality_guarantee": False,
    }
    result["packet_hash"] = _hash(result)
    return result


def _derived_seed(project_id: str, base_seed: int, index: int, policy: str) -> int:
    if not 0 <= int(base_seed) <= MAX_UINT64:
        raise ValueError("base_seed must be between 0 and 2^64-1")
    if policy == "fixed":
        return int(base_seed)
    if policy == "increment":
        return (int(base_seed) + index) & MAX_UINT64
    if policy == "hash_project_shot":
        payload = f"{project_id}\0{int(base_seed)}\0{index}".encode("utf-8")
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    raise ValueError("seed_policy must be fixed, increment, or hash_project_shot")


def _shot_parts(raw: Mapping[str, Any], default_duration: float, split_long_shots: bool):
    requested = float(raw.get("duration_seconds", default_duration))
    if not math.isfinite(requested) or requested <= 0:
        raise ValueError("shot duration_seconds must be finite and positive")
    max_seconds = MAX_TRAINED_FRAMES / FPS
    if requested > max_seconds and not split_long_shots:
        raise ValueError(
            f"shot duration {requested:g}s exceeds one H3 window ({max_seconds:.3f}s); enable split_long_shots"
        )
    count = max(1, math.ceil(requested / max_seconds))
    for part in range(count):
        yield part, count, requested / count


def build_studio_timeline(
    project_id: str,
    shots_json: str,
    default_backend: str,
    default_duration_seconds: float,
    default_aspect_ratio: str,
    base_seed: int,
    seed_policy: str,
    split_long_shots: bool,
    strict_exact_dialogue: bool,
    cast=None,
    sound_canvas=None,
) -> dict[str, Any]:
    project_id = _safe_id(project_id, "project_id")
    cast = validate_cast(cast)
    sound_canvas = validate_sound_canvas(sound_canvas)
    payload = _json(shots_json, "shots_json", [])
    if isinstance(payload, Mapping):
        payload = payload.get("shots", [])
    if not isinstance(payload, list) or not payload:
        raise ValueError("shots_json must contain at least one shot")
    if len(payload) > 512:
        raise ValueError("Studio Timeline supports at most 512 source shots")
    if default_backend not in BACKENDS:
        raise ValueError(f"default_backend must be one of {BACKENDS}")
    default_duration = float(default_duration_seconds)
    if not math.isfinite(default_duration) or default_duration <= 0:
        raise ValueError("default_duration_seconds must be finite and positive")
    shots, source_ids = [], set()
    timeline_frame, timeline_seconds = 0, 0.0
    for source_index, raw in enumerate(payload):
        raw = {"prompt": raw} if isinstance(raw, str) else raw
        if not isinstance(raw, Mapping):
            raise ValueError(f"shot {source_index} must be text or an object")
        source_id = _safe_id(raw.get("id", f"shot_{source_index:03d}"), "shot id")
        if source_id in source_ids:
            raise ValueError(f"duplicate shot id: {source_id}")
        source_ids.add(source_id)
        prompt = _text(raw.get("prompt"), f"shot {source_id} prompt", required=True, maximum=30000)
        backend = str(raw.get("backend", default_backend)).strip()
        aspect = _text(raw.get("aspect_ratio", default_aspect_ratio), "aspect_ratio", required=True, maximum=32)
        cast_ids = _text_list(raw.get("cast_ids", []), "cast_ids")
        brief = raw.get("creative_brief", {})
        if not isinstance(brief, Mapping):
            raise ValueError(f"shot {source_id} creative_brief must be an object")
        dialogue = _text(raw.get("dialogue", ""), "dialogue", maximum=12000)
        parts = list(_shot_parts(raw, default_duration, bool(split_long_shots)))
        if len(parts) > 1 and dialogue:
            raise ValueError(
                f"shot {source_id} exceeds one H3 window and contains dialogue; split it explicitly so exact lines stay assigned"
            )
        for part, part_count, requested_duration in parts:
            frame_count = min(MAX_TRAINED_FRAMES, align_frame_count(max(MIN_STUDIO_FRAMES, round(requested_duration * FPS))))
            render_duration = frame_count / FPS
            window_canvas = sound_canvas_window(sound_canvas, timeline_seconds, timeline_seconds + render_duration)
            part_prompt = prompt
            if part_count > 1:
                part_prompt += (
                    f"\nContinuation part {part + 1} of {part_count} for {source_id}; preserve identity, wardrobe, "
                    "scene geography, lighting direction and motion continuity."
                )
            packet = compile_prompt_packet(
                part_prompt,
                backend,
                render_duration,
                aspect,
                dialogue,
                _text(raw.get("negative_prompt", ""), "negative_prompt", maximum=12000),
                strict_exact_dialogue,
                cast,
                cast_ids,
                window_canvas,
                brief,
            )
            index = len(shots)
            shot_id = source_id if part_count == 1 else f"{source_id}.part{part + 1}"
            seed = int(raw["seed"]) if "seed" in raw else _derived_seed(project_id, base_seed, index, seed_policy)
            if not 0 <= seed <= MAX_UINT64:
                raise ValueError(f"shot {shot_id} seed must be between 0 and 2^64-1")
            shots.append({
                "index": index,
                "id": shot_id,
                "source_id": source_id,
                "source_index": source_index,
                "part_index": part,
                "part_count": part_count,
                "status": "planned",
                "start_frame": timeline_frame,
                "end_frame_exclusive": timeline_frame + frame_count,
                "start_seconds": timeline_seconds,
                "end_seconds": timeline_seconds + render_duration,
                "requested_duration_seconds": requested_duration,
                "render_duration_seconds": render_duration,
                "frame_count": frame_count,
                "seed": seed,
                "prompt_packet": packet,
                "notes": _text(raw.get("notes", ""), "notes", maximum=4000),
            })
            timeline_frame += frame_count
            timeline_seconds += render_duration
    timeline = {
        "schema": STUDIO_TIMELINE_SCHEMA,
        "project_id": project_id,
        "shots": shots,
        "shot_count": len(shots),
        "source_shot_count": len(payload),
        "total_frames": timeline_frame,
        "total_duration_seconds": timeline_seconds,
        "fps": FPS,
        "frame_grid": "17n+5 per generated shot",
        "default_backend": default_backend,
        "seed_policy": seed_policy,
        "base_seed": int(base_seed),
        "split_long_shots": bool(split_long_shots),
        "cast_hash": cast.get("cast_hash") if cast else None,
        "sound_canvas_hash": sound_canvas.get("sound_canvas_hash") if sound_canvas else None,
        "compiler_only": True,
        "generation_complete": False,
    }
    timeline["timeline_hash"] = _hash(timeline)
    return timeline


def validate_timeline(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or value.get("schema") != STUDIO_TIMELINE_SCHEMA:
        raise ValueError("timeline must be a MiniMax H3 Studio Timeline object")
    if not isinstance(value.get("shots"), list) or not value["shots"]:
        raise ValueError("Studio Timeline contains no shots")
    return dict(value)


def select_studio_shot(timeline: Mapping[str, Any], shot_index: int):
    timeline = validate_timeline(timeline)
    index = int(shot_index)
    if index < 0 or index >= len(timeline["shots"]):
        raise ValueError(f"shot_index must be between 0 and {len(timeline['shots']) - 1}")
    shot = deepcopy(timeline["shots"][index])
    report = {
        "schema": STUDIO_TIMELINE_SCHEMA,
        "project_id": timeline["project_id"],
        "timeline_hash": timeline["timeline_hash"],
        "selected_index": index,
        "selected_id": shot["id"],
        "next_index": index + 1 if index + 1 < len(timeline["shots"]) else None,
        "is_last": index + 1 == len(timeline["shots"]),
        "frame_count": shot["frame_count"],
        "seed": shot["seed"],
        "backend": shot["prompt_packet"]["backend"],
        "compiler_only": True,
    }
    return shot, report


def _index_set(value: str, maximum: int) -> set[int]:
    text = _text(value, "manual_indices", maximum=8000)
    if not text:
        return set()
    result: set[int] = set()
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"invalid descending index range: {token}")
            result.update(range(start, end + 1))
        else:
            result.add(int(token))
    if any(index < 0 or index >= maximum for index in result):
        raise ValueError(f"manual_indices must stay between 0 and {maximum - 1}")
    return result


def _threshold_violations(scores: Mapping[str, Any], thresholds: Mapping[str, Any]) -> list[str]:
    violations = []
    for metric, rule in thresholds.items():
        if metric not in scores:
            continue
        value = float(scores[metric])
        if not math.isfinite(value):
            violations.append(f"{metric}:non_finite")
            continue
        if isinstance(rule, Mapping):
            if "min" in rule and value < float(rule["min"]):
                violations.append(f"{metric}<min")
            if "max" in rule and value > float(rule["max"]):
                violations.append(f"{metric}>max")
        elif value < float(rule):
            violations.append(f"{metric}<min")
    return violations


def build_selective_repair_plan(
    timeline: Mapping[str, Any],
    quality_report_json: str,
    selection_policy: str,
    manual_indices: str,
    thresholds_json: str,
    repair_mode: str,
    prompt_addendum: str,
    seed_stride: int,
    context_before_frames: int,
    context_after_frames: int,
) -> dict[str, Any]:
    timeline = validate_timeline(timeline)
    original_snapshot = canonical_json(timeline)
    policies = {"manual", "failed_status", "score_threshold", "failed_or_score"}
    if selection_policy not in policies:
        raise ValueError(f"selection_policy must be one of {sorted(policies)}")
    modes = {"auto", "seed_retry", "prompt_tighten", "reference_refresh", "full_regenerate"}
    if repair_mode not in modes:
        raise ValueError(f"repair_mode must be one of {sorted(modes)}")
    report = _json(quality_report_json, "quality_report_json", {"segments": []})
    report = {"segments": report} if isinstance(report, list) else report
    if not isinstance(report, Mapping) or not isinstance(report.get("segments", []), list):
        raise ValueError("quality_report_json must be a list or {segments: [...]} object")
    thresholds = _json(thresholds_json, "thresholds_json", {})
    if not isinstance(thresholds, Mapping):
        raise ValueError("thresholds_json must be an object")
    maximum = len(timeline["shots"])
    selected = _index_set(manual_indices, maximum) if selection_policy == "manual" else set()
    reasons = {index: ["manual"] for index in selected}
    report_items: dict[int, Mapping[str, Any]] = {}
    for item in report.get("segments", []):
        if not isinstance(item, Mapping) or "index" not in item:
            raise ValueError("each quality segment must be an object with index")
        index = int(item["index"])
        if index < 0 or index >= maximum or index in report_items:
            raise ValueError(f"invalid or duplicate quality segment index: {index}")
        report_items[index] = item
        status_failed = str(item.get("status", "")).lower() in {"fail", "failed", "rejected"}
        scores = item.get("scores", {})
        if not isinstance(scores, Mapping):
            raise ValueError(f"quality segment {index} scores must be an object")
        item_reasons = []
        if selection_policy in {"failed_status", "failed_or_score"} and status_failed:
            item_reasons.append("failed_status")
        if selection_policy in {"score_threshold", "failed_or_score"}:
            item_reasons.extend(_threshold_violations(scores, thresholds))
        if item_reasons:
            selected.add(index)
            reasons[index] = item_reasons
    if selection_policy != "manual" and not report_items:
        raise ValueError("quality_report_json contains no segment evidence for this policy")
    addendum = _text(prompt_addendum, "prompt_addendum", maximum=8000)
    stride = int(seed_stride)
    before, after = int(context_before_frames), int(context_after_frames)
    if not 1 <= stride <= MAX_UINT64:
        raise ValueError("seed_stride must be between 1 and 2^64-1")
    if min(before, after) < 0 or max(before, after) > MAX_TRAINED_FRAMES:
        raise ValueError("repair context frames must be between 0 and 362")
    repairs = []
    for index in sorted(selected):
        shot = timeline["shots"][index]
        evidence = report_items.get(index, {})
        issues = [str(value).lower() for value in evidence.get("issues", [])]
        resolved_mode = repair_mode
        if repair_mode == "auto":
            joined = " ".join(issues + reasons.get(index, []))
            if any(word in joined for word in ("identity", "character", "speaker")):
                resolved_mode = "reference_refresh"
            elif any(word in joined for word in ("speech", "dialogue", "prompt", "extra")):
                resolved_mode = "prompt_tighten"
            elif any(word in joined for word in ("seam", "motion", "flicker", "black")):
                resolved_mode = "full_regenerate"
            else:
                resolved_mode = "seed_retry"
        attempt = max(0, int(evidence.get("attempt", 0))) + 1
        packet = shot["prompt_packet"]
        repair_prompt = packet["compiled_prompt"]
        if addendum:
            repair_prompt += "\n\nRepair instruction:\n" + addendum
        repairs.append({
            "repair_index": len(repairs), "shot_index": index, "shot_id": shot["id"],
            "mode": resolved_mode, "attempt": attempt, "reasons": reasons.get(index, []),
            "source_seed": shot["seed"],
            "repair_seed": (int(shot["seed"]) + stride * attempt) & MAX_UINT64,
            "frame_count": shot["frame_count"], "prompt": repair_prompt,
            "negative_prompt": packet["negative_prompt"], "backend": packet["backend"],
            "context_before_frames": before, "context_after_frames": after,
            "requires_reference_refresh": resolved_mode == "reference_refresh",
        })
    result = {
        "schema": REPAIR_PLAN_SCHEMA,
        "project_id": timeline["project_id"],
        "source_timeline_hash": timeline["timeline_hash"],
        "selection_policy": selection_policy,
        "repair_count": len(repairs),
        "repairs": repairs,
        "selected_indices": sorted(selected),
        "unchanged_indices": [i for i in range(maximum) if i not in selected],
        "non_destructive": True,
        "accepted_media_mutated": False,
        "automatic_accept": False,
        "notes": [
            "This node creates a repair plan only; it never deletes, overwrites, or automatically accepts media.",
            "Unselected segments remain outside this plan.",
            "Context frames are hints and require a compatible continuation workflow.",
        ],
    }
    result["repair_plan_hash"] = _hash(result)
    if canonical_json(timeline) != original_snapshot:
        raise RuntimeError("Selective repair unexpectedly mutated the source timeline")
    return result


def select_repair_segment(plan: Mapping[str, Any], repair_index: int):
    if not isinstance(plan, Mapping) or plan.get("schema") != REPAIR_PLAN_SCHEMA:
        raise ValueError("repair_plan must be a MiniMax H3 Selective Repair object")
    repairs = plan.get("repairs", [])
    index = int(repair_index)
    if not repairs:
        raise ValueError("repair plan contains no selected segments")
    if index < 0 or index >= len(repairs):
        raise ValueError(f"repair_index must be between 0 and {len(repairs) - 1}")
    repair = deepcopy(repairs[index])
    report = {
        "schema": REPAIR_PLAN_SCHEMA,
        "repair_plan_hash": plan["repair_plan_hash"],
        "selected_repair_index": index,
        "shot_index": repair["shot_index"],
        "shot_id": repair["shot_id"],
        "next_repair_index": index + 1 if index + 1 < len(repairs) else None,
        "is_last": index + 1 == len(repairs),
        "non_destructive": True,
    }
    return repair, report
