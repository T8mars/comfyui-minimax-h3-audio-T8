from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import time

import torch

import comfy.model_management

from .audio_ops import decode_av_latent, trim_av_output
from .conditioning import build_conditioning
from .core import FPS, MAX_TRAINED_FRAMES, MIN_TRAINED_FRAMES, align_frame_count, validate_audio
from .long_video_delivery import (
    _cleanup_temporary,
    _mux_video_with_raw_audio,
    _normalize_audio,
    _sha256_file,
    _write_planar_audio_raw,
    accept_long_video_candidate,
    compose_accepted_long_video,
    load_delivery_manifest,
    long_video_chain_root,
    save_long_video_candidate,
)
from .long_video_in_node_loop_advanced import (
    _atomic_write_json,
    _available_candidate_id,
    _candidate_base_id,
    _exclusive_loop_lock,
    _has_saved_candidate,
    _media_signature,
    _release_segment_memory,
    _reusable_candidate,
    _sample_one_segment,
)
from .long_video import sanitize_chain_id
from .prompt_relay_events_advanced import (
    MAX_RELAY_EVENTS,
    PROMPT_RELAY_EVENTS_SCHEMA,
    PROMPT_RELAY_EVENTS_TYPE,
    json_hash,
)


MV_SCENE_PLAN_TYPE = "H3_T8_MV_SCENE_PLAN"
MV_PROMPT_PLAN_TYPE = "H3_T8_MV_PROMPT_PLAN"
MV_VOCAL_LOCK_PROMPT_PLAN_TYPE = "H3_T8_MV_VOCAL_LOCK_PROMPT_PLAN"
MV_SCENE_SCHEMA = "t8.minimax_h3.mv_scene_plan.v1"
MV_PROMPT_SCHEMA = "t8.minimax_h3.mv_prompt_plan.v1"
MV_VOCAL_LOCK_PROMPT_SCHEMA = "t8.minimax_h3.mv_vocal_lock_prompt_plan.v2"
MV_LOOP_STATE_NAME = "mv_in_node_loop_state.json"
MV_VOCAL_LOCK_LOOP_STATE_NAME = "mv_vocal_lock_in_node_loop_state.json"


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


def _load_json(value: str, name: str):
    text = str(value or "").strip()
    if not text:
        return []
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"{name} is invalid JSON: {error}") from error


def _manual_boundary_frames(value: str, total_frames: int) -> list[int]:
    payload = _load_json(value, "manual_boundaries_json")
    if isinstance(payload, Mapping):
        payload = payload.get("boundaries", payload.get("seconds", []))
    if not isinstance(payload, list):
        raise ValueError("manual_boundaries_json must be a list or {boundaries:[...]}")
    frames: list[int] = []
    for index, item in enumerate(payload):
        if isinstance(item, Mapping):
            item = item.get("seconds", item.get("time_seconds"))
        if isinstance(item, bool):
            raise ValueError(f"manual boundary {index} must be a finite time in seconds")
        try:
            seconds = float(item)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"manual boundary {index} must be a finite time in seconds"
            ) from error
        if not math.isfinite(seconds):
            raise ValueError(f"manual boundary {index} must be finite")
        frame = round(seconds * FPS)
        if not 0 < frame < total_frames:
            raise ValueError(
                f"manual boundary {index} must land strictly inside the song duration"
            )
        frames.append(frame)
    if len(set(frames)) != len(frames):
        raise ValueError("manual boundaries collapse to duplicate 24fps frames")
    return sorted(frames)


def _block_rms(waveform: torch.Tensor, sample_rate: int, hop_ms: int) -> torch.Tensor:
    # Audio normally arrives on CPU. Work on one detached mono copy only and cap the
    # analysis rate; the original full-rate waveform remains authoritative for rendering.
    mono = waveform[0].detach().float().mean(dim=0).cpu()
    stride = max(1, int(sample_rate) // 8000)
    mono = mono[::stride]
    analysis_rate = float(sample_rate) / stride
    hop = max(1, round(analysis_rate * int(hop_ms) / 1000.0))
    block_count = max(1, math.ceil(int(mono.numel()) / hop))
    padded = torch.nn.functional.pad(mono, (0, block_count * hop - int(mono.numel())))
    return padded.reshape(block_count, hop).square().mean(dim=1).sqrt()


def _candidate_scores(
    waveform: torch.Tensor,
    sample_rate: int,
    hop_ms: int,
    total_frames: int,
) -> dict[int, float]:
    rms = _block_rms(waveform, sample_rate, hop_ms)
    if int(rms.numel()) == 1:
        return {}
    low = torch.quantile(rms, 0.10)
    high = torch.quantile(rms, 0.90)
    spread = max(float(high - low), 1e-8)
    level = ((rms - low) / spread).clamp(0.0, 1.0)
    quiet = 1.0 - level
    previous = torch.cat((rms[:1], rms[:-1]))
    drop = ((previous - rms) / spread).clamp(0.0, 1.0)
    change = ((rms - previous).abs() / spread).clamp(0.0, 1.0)
    combined = 0.65 * quiet + 0.25 * drop + 0.10 * change
    result: dict[int, float] = {}
    for index, raw_score in enumerate(combined.tolist()):
        frame = round(index * int(hop_ms) * FPS / 1000.0)
        if 0 < frame < total_frames:
            result[frame] = max(result.get(frame, 0.0), float(raw_score))
    return result


def _fallback_equal_boundaries(total_frames: int, target: int, maximum: int) -> list[int]:
    count = max(1, math.ceil(total_frames / max(1, maximum)))
    count = max(count, round(total_frames / max(1, target)))
    return [round(index * total_frames / count) for index in range(count + 1)]


def _dynamic_boundaries(
    total_frames: int,
    scores: Mapping[int, float],
    minimum: int,
    target: int,
    maximum: int,
) -> list[int]:
    if total_frames <= maximum:
        return [0, total_frames]
    candidates = sorted({0, total_frames, *scores})
    best = {0: 0.0}
    previous: dict[int, int] = {}
    for end in candidates[1:]:
        chosen_cost = math.inf
        chosen_start = None
        for start in reversed(candidates):
            if start >= end or start not in best:
                continue
            duration = end - start
            is_final = end == total_frames
            if duration > maximum:
                break
            if duration < minimum and not (is_final and start > 0):
                continue
            duration_cost = ((duration - target) / max(1, target)) ** 2
            boundary_reward = 0.0 if is_final else 0.45 * float(scores.get(end, 0.0))
            short_final_cost = 0.0
            if is_final and duration < minimum:
                short_final_cost = ((minimum - duration) / max(1, minimum)) ** 2
            cost = best[start] + duration_cost + short_final_cost - boundary_reward
            if cost < chosen_cost:
                chosen_cost, chosen_start = cost, start
        if chosen_start is not None:
            best[end] = chosen_cost
            previous[end] = chosen_start
    if total_frames not in best:
        return _fallback_equal_boundaries(total_frames, target, maximum)
    boundaries = [total_frames]
    while boundaries[-1] != 0:
        boundaries.append(previous[boundaries[-1]])
    boundaries.reverse()
    if len(boundaries) >= 3 and boundaries[-1] - boundaries[-2] < minimum:
        merged = boundaries[-1] - boundaries[-3]
        if merged <= maximum:
            boundaries.pop(-2)
    return boundaries


def _scene_activity(
    rms: torch.Tensor,
    hop_ms: int,
    start_frame: int,
    end_frame: int,
) -> float:
    start = max(0, math.floor(start_frame / FPS * 1000 / hop_ms))
    end = min(int(rms.numel()), math.ceil(end_frame / FPS * 1000 / hop_ms))
    if end <= start:
        return 0.0
    low = float(torch.quantile(rms, 0.10))
    high = float(torch.quantile(rms, 0.90))
    value = float(rms[start:end].mean())
    return max(0.0, min(1.0, (value - low) / max(high - low, 1e-8)))


def validate_mv_scene_plan(value: Mapping) -> dict:
    if not isinstance(value, Mapping):
        raise ValueError("MV Scene Plan must come from the local MV Vocal Scene Planner node")
    plan = dict(value)
    claimed = plan.pop("plan_hash", None)
    if plan.get("type") != MV_SCENE_PLAN_TYPE or plan.get("schema") != MV_SCENE_SCHEMA:
        raise ValueError("MV Scene Plan has an unsupported type/schema")
    actual = _hash(plan)
    if not isinstance(claimed, str) or claimed != actual:
        raise ValueError("MV Scene Plan hash mismatch; rebuild the local scene plan")
    scenes = plan.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("MV Scene Plan contains no scenes")
    cursor = 0
    for index, scene in enumerate(scenes):
        if not isinstance(scene, Mapping) or int(scene.get("index", -1)) != index:
            raise ValueError("MV Scene Plan scene indices must be contiguous")
        start = int(scene.get("start_frame", -1))
        end = int(scene.get("end_frame", -1))
        render = int(scene.get("render_frame_count", -1))
        if start != cursor or end <= start:
            raise ValueError("MV Scene Plan timeline must be contiguous and non-empty")
        if not MIN_TRAINED_FRAMES <= render <= MAX_TRAINED_FRAMES:
            raise ValueError("MV Scene Plan render windows must stay in the trained H3 range")
        if align_frame_count(render) != render or render < end - start:
            raise ValueError("MV Scene Plan contains an invalid 17n+5 render window")
        cursor = end
    if cursor != int(plan.get("total_frames", -1)):
        raise ValueError("MV Scene Plan final boundary does not match total_frames")
    plan["plan_hash"] = claimed
    return plan


def build_mv_scene_plan(
    full_song,
    min_scene_seconds: float = 5.0,
    target_scene_seconds: float = 7.0,
    max_scene_seconds: float = 10.0,
    analysis_hop_ms: int = 100,
    vocal_policy: str = "assume_vocal",
    manual_boundaries_json: str = "",
    vocal_stem=None,
) -> tuple[dict, int, float, str, str]:
    waveform, sample_rate = validate_audio(full_song, "full_song")
    if vocal_policy not in {"assume_vocal", "energy_proxy", "vocal_stem_required"}:
        raise ValueError(
            "vocal_policy must be assume_vocal, energy_proxy, or vocal_stem_required"
        )
    if vocal_policy == "vocal_stem_required" and vocal_stem is None:
        raise ValueError("vocal_stem_required needs a connected local vocal_stem AUDIO")
    for name, value in (
        ("min_scene_seconds", min_scene_seconds),
        ("target_scene_seconds", target_scene_seconds),
        ("max_scene_seconds", max_scene_seconds),
    ):
        if not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(f"{name} must be a finite positive value")
    if not float(min_scene_seconds) <= float(target_scene_seconds) <= float(max_scene_seconds):
        raise ValueError("scene seconds must satisfy min <= target <= max")
    if not 20 <= int(analysis_hop_ms) <= 500:
        raise ValueError("analysis_hop_ms must be between 20 and 500")

    total_frames = max(1, round(int(waveform.shape[-1]) / sample_rate * FPS))
    minimum = max(1, round(float(min_scene_seconds) * FPS))
    target = max(minimum, round(float(target_scene_seconds) * FPS))
    maximum = min(MAX_TRAINED_FRAMES, round(float(max_scene_seconds) * FPS))
    if target > maximum or minimum > maximum:
        raise ValueError(
            f"scene timing exceeds the current H3 maximum of {MAX_TRAINED_FRAMES / FPS:.3f}s"
        )

    analysis_audio = vocal_stem if vocal_stem is not None else full_song
    analysis_waveform, analysis_rate = validate_audio(analysis_audio, "analysis_audio")
    analysis_rms = _block_rms(analysis_waveform, analysis_rate, int(analysis_hop_ms))
    manual = _manual_boundary_frames(manual_boundaries_json, total_frames)
    if manual:
        boundaries = [0, *manual, total_frames]
        source = "manual"
    else:
        scores = _candidate_scores(
            analysis_waveform, analysis_rate, int(analysis_hop_ms), total_frames
        )
        boundaries = _dynamic_boundaries(total_frames, scores, minimum, target, maximum)
        source = "local_audio_dp"

    scenes = []
    warnings = []
    for index, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        frame_count = end - start
        render = align_frame_count(max(MIN_TRAINED_FRAMES, frame_count))
        if render > MAX_TRAINED_FRAMES:
            raise ValueError(
                f"scene {index} needs {render} render frames; add a boundary before "
                f"{start / FPS:.3f}s"
            )
        activity = _scene_activity(analysis_rms, int(analysis_hop_ms), start, end)
        if vocal_stem is not None:
            performance_state = "vocal_active" if activity >= 0.12 else "non_vocal"
        elif vocal_policy == "assume_vocal":
            performance_state = "vocal_active"
        else:
            performance_state = "vocal_active" if activity >= 0.22 else "non_vocal"
        if frame_count < minimum and len(boundaries) > 2:
            warnings.append(
                f"scene {index} is shorter than the preferred minimum after exact final alignment"
            )
        scenes.append(
            {
                "index": index,
                "start_frame": start,
                "end_frame": end,
                "frame_count": frame_count,
                "start_seconds": start / FPS,
                "end_seconds": end / FPS,
                "duration_seconds": frame_count / FPS,
                "render_frame_count": render,
                "trim_frame_count": frame_count,
                "audio_activity_score": round(activity, 6),
                "performance_state": performance_state,
            }
        )
    plan = {
        "type": MV_SCENE_PLAN_TYPE,
        "schema": MV_SCENE_SCHEMA,
        "fps": FPS,
        "total_frames": total_frames,
        "duration_seconds": total_frames / FPS,
        "scene_count": len(scenes),
        "boundary_source": source,
        "analysis_source": "vocal_stem" if vocal_stem is not None else "full_song",
        "vocal_policy": vocal_policy,
        "analysis_hop_ms": int(analysis_hop_ms),
        "scene_seconds": {
            "minimum": float(min_scene_seconds),
            "target": float(target_scene_seconds),
            "maximum": float(max_scene_seconds),
        },
        "scenes": scenes,
        "warnings": warnings,
        "external_api_used": False,
    }
    plan["plan_hash"] = _hash(plan)
    report = {
        "status": "ready",
        "plan_hash": plan["plan_hash"],
        "scene_count": len(scenes),
        "duration_seconds": total_frames / FPS,
        "boundary_source": source,
        "analysis_source": plan["analysis_source"],
        "notes": [
            "analysis is local CPU tensor math; no remote LLM, TTS, music or video API is used",
            "every render length is quantized before H3 generation to the legal 17n+5 grid",
            "the final original song remains authoritative and is muxed once after video assembly",
        ],
        "warnings": warnings,
    }
    return (
        plan,
        len(scenes),
        total_frames / FPS,
        _canonical_json({"scenes": scenes}, indent=2),
        _canonical_json(report, indent=2),
    )


def build_mv_vocal_lock_scene_plan(
    full_song,
    vocal_lock_audio,
    min_scene_seconds: float = 5.0,
    target_scene_seconds: float = 7.0,
    max_scene_seconds: float = 10.0,
    analysis_hop_ms: int = 50,
    vocal_active_ratio: float = 0.12,
    manual_boundaries_json: str = "",
) -> tuple[dict, int, float, str, str]:
    if not math.isfinite(float(vocal_active_ratio)) or not 0.0 <= float(vocal_active_ratio) <= 1.0:
        raise ValueError("vocal_active_ratio must be finite and between 0 and 1")
    base = build_mv_scene_plan(
        full_song,
        min_scene_seconds=min_scene_seconds,
        target_scene_seconds=target_scene_seconds,
        max_scene_seconds=max_scene_seconds,
        analysis_hop_ms=analysis_hop_ms,
        vocal_policy="vocal_stem_required",
        manual_boundaries_json=manual_boundaries_json,
        vocal_stem=vocal_lock_audio,
    )[0]
    audio_contract = _validate_vocal_lock_audio_contract(
        full_song, vocal_lock_audio, int(base["total_frames"])
    )
    waveform, sample_rate = validate_audio(vocal_lock_audio, "vocal_lock_audio")
    rms = _block_rms(waveform, sample_rate, int(analysis_hop_ms))
    global_peak = float(rms.max()) if int(rms.numel()) else 0.0
    active_threshold = max(1e-5, global_peak * 0.08)
    plan = json.loads(_canonical_json(base))
    plan.pop("plan_hash", None)
    for scene in plan["scenes"]:
        start = max(0, math.floor(scene["start_frame"] / FPS * 1000 / analysis_hop_ms))
        end = min(
            int(rms.numel()),
            math.ceil(scene["end_frame"] / FPS * 1000 / analysis_hop_ms),
        )
        window = rms[start:end]
        ratio = (
            float((window >= active_threshold).float().mean())
            if int(window.numel())
            else 0.0
        )
        scene["vocal_activity_ratio"] = round(ratio, 6)
        scene["vocal_activity_threshold_rms"] = round(active_threshold, 8)
        scene["performance_state"] = (
            "vocal_active" if ratio >= float(vocal_active_ratio) else "non_vocal"
        )
    plan.update(
        {
            "analysis_source": "required_vocal_lock_audio",
            "vocal_policy": "vocal_lock_v2",
            "vocal_active_ratio": float(vocal_active_ratio),
            "audio_timeline_contract": audio_contract,
            "external_api_used": False,
        }
    )
    plan["plan_hash"] = _hash(plan)
    validate_mv_scene_plan(plan)
    report = {
        "status": "ready",
        "plan_hash": plan["plan_hash"],
        "scene_count": int(plan["scene_count"]),
        "duration_seconds": float(plan["duration_seconds"]),
        "boundary_source": plan["boundary_source"],
        "analysis_source": plan["analysis_source"],
        "vocal_active_scene_count": sum(
            scene["performance_state"] == "vocal_active" for scene in plan["scenes"]
        ),
        "audio_timeline_contract": audio_contract,
        "notes": [
            "scene boundaries and vocal activity are derived locally from the required vocal_lock_audio",
            "both audio inputs must resolve to the same exact 24fps scene-plan duration",
            "this planner detects vocal energy intervals; it does not infer or transcribe words",
        ],
    }
    return (
        plan,
        int(plan["scene_count"]),
        float(plan["duration_seconds"]),
        _canonical_json({"scenes": plan["scenes"]}, indent=2),
        _canonical_json(report, indent=2),
    )


def validate_mv_prompt_plan(value: Mapping) -> dict:
    if not isinstance(value, Mapping):
        raise ValueError("MV Prompt Plan must come from the local Ref2VA Prompt Compiler node")
    plan = dict(value)
    claimed = plan.pop("prompt_plan_hash", None)
    if plan.get("type") != MV_PROMPT_PLAN_TYPE or plan.get("schema") != MV_PROMPT_SCHEMA:
        raise ValueError("MV Prompt Plan has an unsupported type/schema")
    if not isinstance(claimed, str) or claimed != _hash(plan):
        raise ValueError("MV Prompt Plan hash mismatch; rebuild the local prompt plan")
    validate_mv_scene_plan(plan.get("scene_plan"))
    prompts = plan.get("segments")
    if not isinstance(prompts, list) or len(prompts) != plan["scene_plan"]["scene_count"]:
        raise ValueError("MV Prompt Plan segment count does not match its scene plan")
    plan["prompt_plan_hash"] = claimed
    return plan


def _single_line(value: str) -> str:
    return " ".join(str(value or "").replace("|", " ").split())


def build_mv_prompt_plan(
    scene_plan: Mapping,
    global_creative_prompt: str,
    performer_description: str,
    visual_style: str,
    camera_pattern: str,
    non_vocal_action: str,
    exact_lyrics_json: str = "",
) -> tuple[dict, str, dict, str, str]:
    scene_plan = validate_mv_scene_plan(scene_plan)
    lyric_payload = _load_json(exact_lyrics_json, "exact_lyrics_json")
    if lyric_payload and not isinstance(lyric_payload, list):
        raise ValueError("exact_lyrics_json must be a list of scene lyric strings")
    if any(not isinstance(item, str) for item in lyric_payload):
        raise ValueError("exact_lyrics_json must contain only scene lyric strings")
    if len(lyric_payload) > int(scene_plan["scene_count"]):
        raise ValueError("exact_lyrics_json contains more entries than the scene plan")

    global_prompt = _single_line(global_creative_prompt)
    performer = _single_line(performer_description) or "the same lead performer"
    style = _single_line(visual_style) or "cinematic music video, natural skin and lighting"
    cameras = [
        item.strip()
        for item in str(camera_pattern or "").replace("|", "\n").splitlines()
        if item.strip()
    ] or ["stable medium shot with restrained natural camera movement"]
    non_vocal = _single_line(non_vocal_action) or "keeps the mouth naturally closed and breathes"
    segments = []
    relay_events = []
    segment_json = []
    for scene in scene_plan["scenes"]:
        index = int(scene["index"])
        camera = _single_line(cameras[index % len(cameras)])
        lyric = _single_line(lyric_payload[index]) if index < len(lyric_payload) else ""
        vocal = scene["performance_state"] == "vocal_active"
        if vocal:
            performance = (
                "The performer visibly sings in time with <Audio 1>; mouth openings, closures, "
                "breaths and facial emphasis follow the connected audio naturally."
            )
            if lyric:
                performance += f' The exact supplied lyric for this scene is: "{lyric}".'
            else:
                performance += " Do not invent, print or subtitle any lyric text."
        else:
            performance = (
                f"This is a non-vocal passage: the performer {non_vocal}; do not create speaking "
                "or singing mouth shapes."
            )
        prompt = "\n".join(
            (
                f"REFERENCE — <Picture 1> is {performer}; preserve identity, face, hair and wardrobe.",
                f"SCENE — {global_prompt or 'A coherent performance continuing through the song.'}",
                f"CAMERA — {camera}.",
                f"PERFORMANCE — {performance}",
                (
                    f"TIMING — local scene {index + 1}, song time "
                    f"{scene['start_seconds']:.3f}s to {scene['end_seconds']:.3f}s; respond only "
                    "to the connected <Audio 1> window."
                ),
                (
                    f"FINISH — {style}; stable anatomy and background, coherent motion, no text, "
                    "no watermark, no duplicate face, no drifting identity."
                ),
            )
        )
        item = {
            "index": index,
            "prompt": prompt,
            "start_frame": int(scene["start_frame"]),
            "end_frame": int(scene["end_frame"]),
            "performance_state": scene["performance_state"],
            "exact_lyrics_supplied": bool(lyric),
        }
        segments.append(item)
        segment_json.append({"prompt": prompt, "note": f"MV scene {index + 1}"})
        relay_events.append(
            {
                "event_index": index + 1,
                "prompt": _single_line(prompt),
                "start": float(scene["start_seconds"]),
                "end": float(scene["end_seconds"]),
            }
        )
    relay_available = len(relay_events) <= MAX_RELAY_EVENTS
    events = {
        "type": PROMPT_RELAY_EVENTS_TYPE,
        "schema": PROMPT_RELAY_EVENTS_SCHEMA,
        "events": relay_events if relay_available else [],
    }
    events["events_hash"] = json_hash(events)
    prompt_plan = {
        "type": MV_PROMPT_PLAN_TYPE,
        "schema": MV_PROMPT_SCHEMA,
        "scene_plan": scene_plan,
        "segments": segments,
        "prompt_relay_events_hash": events["events_hash"],
        "prompt_relay_events_available": relay_available,
        "external_api_used": False,
        "compiler": "deterministic_local_six_section_ref2va_v1",
    }
    prompt_plan["prompt_plan_hash"] = _hash(prompt_plan)
    report = {
        "status": "ready",
        "prompt_plan_hash": prompt_plan["prompt_plan_hash"],
        "scene_count": len(segments),
        "exact_lyric_scene_count": sum(item["exact_lyrics_supplied"] for item in segments),
        "external_api_used": False,
        "prompt_relay_events_available": relay_available,
        "notes": [
            "prompts are compiled deterministically on the local machine",
            "lyrics are never guessed; only exact_lyrics_json text is quoted",
            "<Picture 1> and <Audio 1> match the project conditioning media contract",
        ],
    }
    if not relay_available:
        report["notes"].append(
            f"typed Prompt Relay events are empty because this MV has more than "
            f"{MAX_RELAY_EVENTS} scenes; MV rendering remains available"
        )
    return (
        prompt_plan,
        json.dumps(segment_json, ensure_ascii=False, indent=2),
        events,
        "\n\n--- SCENE ---\n\n".join(item["prompt"] for item in segments),
        _canonical_json(report, indent=2),
    )


_OFFICIAL_REF2VA_SECTIONS = (
    "subject_definitions:",
    "summary:",
    "retention_analysis:",
    "detailed_description:",
    "overall_soundscape:",
    "non_diegetic_music:",
)


def _validate_official_ref2va_prompt(prompt: str, *, vocal_active: bool) -> None:
    positions = [prompt.find(section) for section in _OFFICIAL_REF2VA_SECTIONS]
    if any(position < 0 for position in positions) or positions != sorted(positions):
        raise ValueError("Vocal Lock prompt must contain the official six Ref2VA sections in order")
    if any(prompt.count(section) != 1 for section in _OFFICIAL_REF2VA_SECTIONS):
        raise ValueError("Vocal Lock prompt must contain each official Ref2VA section exactly once")
    required = (
        "<Subject 1>",
        "<Picture 1>",
        "<Audio 1>",
        "[reference generation + audio reuse]",
        "fully_preserved",
        "fully_copy",
        "[Shot 1]",
    )
    if any(token not in prompt for token in required):
        raise ValueError("Vocal Lock prompt is missing a required Ref2VA reference relationship")
    if vocal_active and (
        "unobstructed" not in prompt.lower() or "phoneme" not in prompt.lower()
    ):
        raise ValueError("Vocal Lock prompt must keep the mouth visible and request phoneme articulation")


def validate_mv_vocal_lock_prompt_plan(value: Mapping) -> dict:
    if not isinstance(value, Mapping):
        raise ValueError(
            "MV Vocal Lock Prompt Plan must come from the V2 Ref2VA Prompt Compiler node"
        )
    plan = dict(value)
    claimed = plan.pop("prompt_plan_hash", None)
    if (
        plan.get("type") != MV_VOCAL_LOCK_PROMPT_PLAN_TYPE
        or plan.get("schema") != MV_VOCAL_LOCK_PROMPT_SCHEMA
    ):
        raise ValueError("MV Vocal Lock Prompt Plan has an unsupported type/schema")
    if not isinstance(claimed, str) or claimed != _hash(plan):
        raise ValueError("MV Vocal Lock Prompt Plan hash mismatch; rebuild the V2 prompt plan")
    scene_plan = validate_mv_scene_plan(plan.get("scene_plan"))
    prompts = plan.get("segments")
    if not isinstance(prompts, list) or len(prompts) != scene_plan["scene_count"]:
        raise ValueError("MV Vocal Lock Prompt Plan segment count does not match its scene plan")
    for index, item in enumerate(prompts):
        if not isinstance(item, Mapping) or int(item.get("index", -1)) != index:
            raise ValueError("MV Vocal Lock Prompt Plan segment indices must be contiguous")
        _validate_official_ref2va_prompt(
            str(item.get("prompt", "")),
            vocal_active=item.get("performance_state") == "vocal_active",
        )
    audio_contract = plan.get("audio_contract")
    if not isinstance(audio_contract, Mapping) or audio_contract.get("drive") != "vocal_lock_audio":
        raise ValueError("MV Vocal Lock Prompt Plan is missing the isolated-audio drive contract")
    if audio_contract.get("delivery") != "full_song_muxed_once":
        raise ValueError("MV Vocal Lock Prompt Plan is missing the one-time full-song delivery contract")
    plan["prompt_plan_hash"] = claimed
    return plan


def _safe_vocal_camera_direction(value: str) -> str:
    direction = _single_line(value) or (
        "locked-off static camera with stable framing, no camera movement, and continuous "
        "mouth visibility"
    )
    replacements = (
        (r"\b(?:extreme )?wide shot\b", "stable performance framing"),
        (r"\b(?:full-body|full body|long) shot\b", "stable performance framing"),
        (r"\bover-the-shoulder shot\b", "front three-quarter framing"),
        (r"\bfrom behind\b", "from a front three-quarter angle"),
        (r"\bprofile(?: shot)?\b", "three-quarter face view"),
        (r"\beye-only shot\b", "face-visible performance framing"),
    )
    for pattern, replacement in replacements:
        direction = re.sub(pattern, replacement, direction, flags=re.IGNORECASE)
    return direction


def build_mv_vocal_lock_prompt_plan(
    scene_plan: Mapping,
    global_creative_prompt: str,
    performer_description: str,
    visual_style: str,
    camera_pattern: str,
    vocal_content_type: str = "singing",
    vocal_language: str = "English",
    exact_vocal_text_json: str = "",
    non_vocal_action: str = "keeps the mouth naturally closed and breathes with the rhythm",
) -> tuple[dict, str, dict, str, str]:
    scene_plan = validate_mv_scene_plan(scene_plan)
    if vocal_content_type not in {"singing", "spoken_dialogue"}:
        raise ValueError("vocal_content_type must be singing or spoken_dialogue")
    text_payload = _load_json(exact_vocal_text_json, "exact_vocal_text_json")
    if text_payload and not isinstance(text_payload, list):
        raise ValueError("exact_vocal_text_json must be a list of scene text strings")
    if any(not isinstance(item, str) for item in text_payload):
        raise ValueError("exact_vocal_text_json must contain only scene text strings")
    if len(text_payload) > int(scene_plan["scene_count"]):
        raise ValueError("exact_vocal_text_json contains more entries than the scene plan")

    global_prompt = _single_line(global_creative_prompt) or "A coherent cinematic performance."
    performer = _single_line(performer_description) or "the same lead performer"
    style = _single_line(visual_style) or "cinematic, realistic light and natural skin texture"
    language = _single_line(vocal_language) or "Unknown"
    cameras = [
        _safe_vocal_camera_direction(item)
        for item in str(camera_pattern or "").replace("|", "\n").splitlines()
        if item.strip()
    ] or [
        "locked-off static camera with stable framing, no camera movement, and continuous "
        "mouth visibility"
    ]
    non_vocal = _single_line(non_vocal_action) or "keeps the mouth naturally closed"
    segments = []
    relay_events = []
    segment_json = []
    for scene in scene_plan["scenes"]:
        index = int(scene["index"])
        exact_text = _single_line(text_payload[index]) if index < len(text_payload) else ""
        vocal_active = scene["performance_state"] == "vocal_active"
        camera = cameras[index % len(cameras)]
        if vocal_active:
            action = "sings the supplied isolated vocal" if vocal_content_type == "singing" else "speaks the supplied isolated dialogue"
            summary_action = "performs the supplied vocal with clearly visible synchronized mouth movement"
            vocal_detail = (
                f"<Subject 1> (S1) {action} naturally. The full face and unobstructed mouth remain "
                "visible for the entire shot; lips, jaw, tongue visibility, cheeks, and facial muscles "
                "articulate every audible phoneme in synchronization with <Audio 1>."
            )
            if exact_text:
                vocal_detail += f" The exact supplied words are <d>[{language}] {exact_text}</d>"
            else:
                vocal_detail += (
                    " No words or lyrics are added, removed, guessed, paraphrased, printed, or subtitled."
                )
        else:
            summary_action = "remains visible with a relaxed closed mouth during the non-vocal interval"
            vocal_detail = (
                f"<Audio 1> remains the synchronized timing signal while <Subject 1> (S1) {non_vocal}. "
                "No unrelated speaking or singing mouth shapes are introduced."
            )
        prompt = "\n".join(
            (
                "subject_definitions:",
                (
                    f"<Subject 1> is the exact performer shown in <Picture 1>: {performer}. The same "
                    "identity, facial structure, hair, clothing, and visual style remain unchanged."
                ),
                (
                    "<Audio 1> is the supplied isolated vocal-lock signal for this entire target clip "
                    "and is reused as the synchronized vocal track for <Subject 1> (S1)."
                ),
                "summary:",
                (
                    "[reference generation + audio reuse] The target video is one continuous "
                    f"performance shot in which <Subject 1> {summary_action}; <Audio 1> supplies the "
                    "complete synchronized vocal timing."
                ),
                "retention_analysis:",
                (
                    "<Subject 1> (appears in [Shot 1]): fully_preserved - identity, face, hair, "
                    "clothing, and visual style from <Picture 1> are retained."
                ),
                (
                    "<Audio 1>: fully_copy - the isolated vocal-lock signal is reused 1:1 across the "
                    "complete generated scene timeline."
                ),
                "detailed_description:",
                (
                    f"The target video uses {style}. {global_prompt} It contains one continuous shot "
                    "with no internal cut and no visible text."
                ),
                (
                    "[Shot 1] Medium close-up performance shot, front-facing or three-quarter face "
                    f"view, with {camera}. <Subject 1> stays clearly identifiable and the full mouth "
                    f"region is never cropped, covered, turned away, or motion-blurred. {vocal_detail}"
                ),
                (
                    "The complete performer stays in optical focus. Hair strands, face contour, "
                    "shoulders, clothing edges, and the subject silhouette remain sharply resolved "
                    "and temporally stable, without haloing, edge smearing, double contours, "
                    "synthetic depth haze, or motion blur. Natural depth of field may soften only "
                    "the distant background."
                ),
                "overall_soundscape:",
                (
                    "No additional dialogue, crowd voices, ambience, or sound effects are invented; "
                    "the synchronized vocal content comes only from <Audio 1>."
                ),
                "non_diegetic_music:",
                "N/A",
            )
        )
        _validate_official_ref2va_prompt(prompt, vocal_active=vocal_active)
        item = {
            "index": index,
            "prompt": prompt,
            "start_frame": int(scene["start_frame"]),
            "end_frame": int(scene["end_frame"]),
            "performance_state": scene["performance_state"],
            "exact_vocal_text_supplied": bool(exact_text),
        }
        segments.append(item)
        segment_json.append({"prompt": prompt, "note": f"MV Vocal Lock scene {index + 1}"})
        relay_events.append(
            {
                "event_index": index + 1,
                "prompt": _single_line(prompt),
                "start": float(scene["start_seconds"]),
                "end": float(scene["end_seconds"]),
            }
        )
    relay_available = len(relay_events) <= MAX_RELAY_EVENTS
    events = {
        "type": PROMPT_RELAY_EVENTS_TYPE,
        "schema": PROMPT_RELAY_EVENTS_SCHEMA,
        "events": relay_events if relay_available else [],
    }
    events["events_hash"] = json_hash(events)
    prompt_plan = {
        "type": MV_VOCAL_LOCK_PROMPT_PLAN_TYPE,
        "schema": MV_VOCAL_LOCK_PROMPT_SCHEMA,
        "scene_plan": scene_plan,
        "segments": segments,
        "audio_contract": {
            "drive": "vocal_lock_audio",
            "conditioning_mode": "lock_source",
            "delivery": "full_song_muxed_once",
        },
        "prompt_relay_events_hash": events["events_hash"],
        "prompt_relay_events_available": relay_available,
        "external_api_used": False,
        "compiler": "deterministic_local_official_ref2va_six_section_v2",
    }
    prompt_plan["prompt_plan_hash"] = _hash(prompt_plan)
    report = {
        "status": "ready",
        "prompt_plan_hash": prompt_plan["prompt_plan_hash"],
        "scene_count": len(segments),
        "exact_vocal_text_scene_count": sum(
            item["exact_vocal_text_supplied"] for item in segments
        ),
        "official_ref2va_sections": list(_OFFICIAL_REF2VA_SECTIONS),
        "audio_contract": prompt_plan["audio_contract"],
        "external_api_used": False,
        "prompt_relay_events_available": relay_available,
        "notes": [
            "the six Ref2VA sections and reference markers follow the official MiniMax H3 format",
            "only exact_vocal_text_json is emitted inside <d>; missing words are never guessed",
            "vocal_lock_audio drives H3 while full_song is reserved for the final delivery mux",
        ],
    }
    if not relay_available:
        report["notes"].append(
            f"typed Prompt Relay events are empty above {MAX_RELAY_EVENTS} scenes; rendering remains available"
        )
    return (
        prompt_plan,
        json.dumps(segment_json, ensure_ascii=False, indent=2),
        events,
        "\n\n--- SCENE ---\n\n".join(item["prompt"] for item in segments),
        _canonical_json(report, indent=2),
    )


def _audio_window(audio, start_frame: int, frame_count: int, *, name: str) -> dict:
    waveform, sample_rate = validate_audio(audio, name)
    start_sample = round(int(start_frame) * sample_rate / FPS)
    end_sample = round((int(start_frame) + int(frame_count)) * sample_rate / FPS)
    target_samples = end_sample - start_sample
    output = waveform.new_zeros((waveform.shape[0], waveform.shape[1], target_samples))
    available = min(max(0, int(waveform.shape[-1]) - start_sample), target_samples)
    if available:
        output[..., :available] = waveform[..., start_sample : start_sample + available]
    return {"waveform": output, "sample_rate": sample_rate}


def _validate_vocal_lock_audio_contract(full_song, vocal_lock_audio, total_frames: int) -> dict:
    result = {}
    for name, audio in (("full_song", full_song), ("vocal_lock_audio", vocal_lock_audio)):
        waveform, sample_rate = validate_audio(audio, name)
        duration_seconds = int(waveform.shape[-1]) / int(sample_rate)
        resolved_frames = round(duration_seconds * FPS)
        if resolved_frames != int(total_frames):
            raise ValueError(
                f"{name} resolves to {resolved_frames} frames at {FPS}fps but the MV scene plan "
                f"requires {int(total_frames)}; align both audio inputs before Vocal Lock rendering"
            )
        rms = float(waveform.detach().float().square().mean().sqrt().cpu())
        peak = float(waveform.detach().float().abs().max().cpu())
        if name == "vocal_lock_audio" and (rms < 1e-5 or peak < 1e-4):
            raise ValueError("vocal_lock_audio is effectively silent and cannot drive assessable lip sync")
        result[name] = {
            "sample_rate": int(sample_rate),
            "samples": int(waveform.shape[-1]),
            "duration_seconds": duration_seconds,
            "resolved_frames": resolved_frames,
            "rms": rms,
            "peak": peak,
        }
    return result


def _mv_audio_sources(full_song, vocal_lock_audio=None) -> dict:
    if vocal_lock_audio is None:
        return {
            "conditioning": full_song,
            "candidate_preview": full_song,
            "final_delivery": full_song,
            "policy": "v1_full_song_conditioning_and_final_mux",
        }
    return {
        "conditioning": vocal_lock_audio,
        "candidate_preview": vocal_lock_audio,
        "final_delivery": full_song,
        "policy": "v2_isolated_vocal_conditioning_full_song_final_mux",
    }


def _mux_master_audio(
    video_path: str,
    master_audio,
    total_frames: int,
    filename_prefix: str,
) -> tuple[str, dict]:
    input_path = Path(video_path).resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"assembled MV video is missing: {input_path}")
    waveform, sample_rate = validate_audio(master_audio, "full_song")
    target_samples = round(int(total_frames) * sample_rate / FPS)
    audio_array, audio_report = _normalize_audio(master_audio, target_samples)
    safe_name = "".join(
        char if char.isalnum() or char in "._-" else "_"
        for char in str(filename_prefix or "H3_Local_MV")
    ).strip("._-") or "H3_Local_MV"
    output = input_path.parent / f"{safe_name}_master_audio.mp4"
    descriptor, raw_name = tempfile.mkstemp(
        prefix=f".{output.stem}.", suffix=".audio.f32.tmp", dir=output.parent
    )
    os.close(descriptor)
    raw = Path(raw_name)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.stem}.", suffix=".mp4.tmp", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        _write_planar_audio_raw(raw, audio_array)
        _mux_video_with_raw_audio(
            input_path,
            raw,
            temporary,
            sample_rate=sample_rate,
        )
        with temporary.open("r+b") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        active_error = sys.exc_info()[1]
        _cleanup_temporary(raw, active_error)
        _cleanup_temporary(temporary, active_error)
    return str(output), {
        "output_path": str(output),
        "output_sha256": _sha256_file(output),
        "sample_rate": sample_rate,
        "audio_samples": target_samples,
        "full_song_muxed_once": True,
        "video_stream_copied": True,
        "audio_encoder_process": "isolated_ffmpeg_subprocess",
        "audio_adjustment": audio_report,
    }


def _state_payload(**kwargs) -> dict:
    return {
        "schema": "t8.minimax_h3.mv_in_node_loop.v1",
        "updated_unix": time.time(),
        **kwargs,
    }


def _contract(
    prompt_plan: Mapping,
    *,
    width: int,
    height: int,
    base_seed: int,
    steps: int,
    shift_video: float,
    shift_audio: float,
    sampler_name: str,
    scheduler: str,
    model_id: str,
    reference_image,
    full_song,
    vocal_lock_audio=None,
    route_revision: str = "v1",
) -> dict:
    contract = {
        "schema": 1,
        "prompt_plan_hash": prompt_plan["prompt_plan_hash"],
        "width": int(width),
        "height": int(height),
        "base_seed": int(base_seed),
        "steps": int(steps),
        "shift_video": float(shift_video),
        "shift_audio": float(shift_audio),
        "sampler_name": str(sampler_name),
        "scheduler": str(scheduler),
        "model_id": str(model_id or "unknown"),
        "reference_image": _media_signature(reference_image),
        "full_song": _media_signature(full_song),
        "external_api_used": False,
    }
    if route_revision != "v1":
        contract.update(
            {
                "schema": 2,
                "route_revision": str(route_revision),
                "vocal_lock_audio": _media_signature(vocal_lock_audio),
                "conditioning_audio_policy": "isolated_vocal_lock_audio",
                "delivery_audio_policy": "full_song_muxed_once",
            }
        )
    return contract


def _accepted_matches_plan(
    manifest: Mapping,
    prompt_plan: Mapping,
    sampling_summary: str,
    *,
    base_seed: int,
    model_id: str,
    width: int,
    height: int,
) -> None:
    scenes = prompt_plan["scene_plan"]["scenes"]
    prompts = prompt_plan["segments"]
    accepted = manifest.get("segments", [])
    if len(accepted) > len(scenes):
        raise ValueError("accepted MV manifest has more segments than the current scene plan")
    for index, item in enumerate(accepted):
        scene = scenes[index]
        expected = {
            "index": index,
            "timeline_start_frame": int(scene["start_frame"]),
            "timeline_end_frame": int(scene["end_frame"]),
            "frame_count": int(scene["frame_count"]),
            "is_final_segment": index == len(scenes) - 1,
            "model_id": str(model_id or "unknown"),
            "sampling_summary": sampling_summary,
            "seed": (int(base_seed) + index) & 0xFFFFFFFFFFFFFFFF,
            "width": int(width),
            "height": int(height),
        }
        for field, value in expected.items():
            if item.get(field) != value:
                raise ValueError(
                    f"accepted MV segment {index} field {field} does not match the current plan"
                )
        if str(item.get("prompt", "")) != str(prompts[index]["prompt"]):
            raise ValueError(
                f"accepted MV segment {index} prompt does not match the current prompt plan"
            )


def run_local_mv_in_node_loop(
    model,
    clip,
    video_vae,
    audio_vae,
    reference_image,
    full_song,
    mv_prompt_plan: Mapping,
    *,
    chain_id: str,
    width: int,
    height: int,
    base_seed: int,
    steps: int,
    shift_video: float,
    shift_audio: float,
    sampler_name: str,
    scheduler: str,
    resume_existing: bool,
    filename_prefix: str,
    bit_depth: int,
    crf: int,
    model_id: str,
    vocal_lock_audio=None,
    prompt_plan_validator=validate_mv_prompt_plan,
    route_revision: str = "v1",
    loop_state_name: str = MV_LOOP_STATE_NAME,
    state_schema: str = "t8.minimax_h3.mv_in_node_loop.v1",
) -> tuple[str, str, int, str, str]:
    if width <= 0 or height <= 0 or width % 32 or height % 32:
        raise ValueError("MiniMax H3 MV width and height must be positive and divisible by 32")
    if not 1 <= int(steps) <= 1000:
        raise ValueError("MiniMax H3 MV steps must be between 1 and 1000")
    if not math.isfinite(float(shift_video)) or float(shift_video) <= 0:
        raise ValueError("MiniMax H3 MV shift_video must be finite and positive")
    if not math.isfinite(float(shift_audio)) or float(shift_audio) <= 0:
        raise ValueError("MiniMax H3 MV shift_audio must be finite and positive")
    if not 0 <= int(base_seed) <= 0xFFFFFFFFFFFFFFFF:
        raise ValueError("MiniMax H3 MV base_seed must be an unsigned 64-bit integer")
    if bit_depth not in {8, 10} or not 0 <= int(crf) <= 51:
        raise ValueError("bit_depth must be 8/10 and crf must be between 0 and 51")
    prompt_plan = prompt_plan_validator(mv_prompt_plan)
    scene_plan = prompt_plan["scene_plan"]
    scenes = scene_plan["scenes"]
    prompts = prompt_plan["segments"]
    audio_sources = _mv_audio_sources(full_song, vocal_lock_audio)
    safe_chain = sanitize_chain_id(chain_id)
    root = long_video_chain_root(safe_chain)
    state_path = root / loop_state_name
    sampling_summary = (
        f"local_mv_{route_revision} {int(steps)}-step {sampler_name}/{scheduler} "
        f"shift{float(shift_video):g}/{float(shift_audio):g}"
    )
    contract = _contract(
        prompt_plan,
        width=width,
        height=height,
        base_seed=base_seed,
        steps=steps,
        shift_video=shift_video,
        shift_audio=shift_audio,
        sampler_name=sampler_name,
        scheduler=scheduler,
        model_id=model_id,
        reference_image=reference_image,
        full_song=full_song,
        vocal_lock_audio=vocal_lock_audio,
        route_revision=route_revision,
    )
    contract_sha256 = _hash(contract)

    with _exclusive_loop_lock(root):
        manifest, _source = load_delivery_manifest(safe_chain, allow_new=True)
        accepted_count = len(manifest.get("segments", []))
        if not resume_existing and (
            state_path.is_file() or accepted_count or _has_saved_candidate(root)
        ):
            raise ValueError(
                "resume_existing is false but this MV chain_id already has saved state or candidates"
            )
        if state_path.is_file():
            saved_state = json.loads(state_path.read_text(encoding="utf-8"))
            if not isinstance(saved_state, Mapping):
                raise ValueError("this MV chain_id has an invalid local state document")
            if (
                saved_state.get("schema") != state_schema
                or saved_state.get("chain_id") != safe_chain
            ):
                raise ValueError("this MV chain_id has an incompatible local state document")
            if saved_state.get("contract_sha256") != contract_sha256:
                if accepted_count:
                    raise ValueError(
                        "this MV chain_id contains accepted segments from a different contract; "
                        "restore the original settings or use a new chain_id"
                    )
                saved_state = {}
        else:
            if accepted_count:
                raise ValueError(
                    "this MV chain_id has accepted segments but its contract state is missing; "
                    "restore mv_in_node_loop_state.json or use a new chain_id"
                )
            saved_state = {}
        _accepted_matches_plan(
            manifest,
            prompt_plan,
            sampling_summary,
            base_seed=base_seed,
            model_id=model_id,
            width=width,
            height=height,
        )

        final_path = str(saved_state.get("final_video_path", ""))
        final_sha = str(saved_state.get("final_video_sha256", ""))
        if (
            len(manifest.get("segments", [])) == len(scenes)
            and final_path
            and Path(final_path).is_file()
            and _sha256_file(Path(final_path)) == final_sha
        ):
            report = dict(saved_state)
            report["resume_action"] = "returned_verified_existing_master_audio_output"
            return (
                final_path,
                str(root / "manifest.json"),
                len(scenes),
                "complete",
                _canonical_json(report, indent=2),
            )

        state = _state_payload(
            schema=state_schema,
            chain_id=safe_chain,
            contract_sha256=contract_sha256,
            status="running",
            scene_count=len(scenes),
            accepted_count=len(manifest.get("segments", [])),
            current_scene_index=len(manifest.get("segments", [])),
            external_api_used=False,
        )
        _atomic_write_json(state_path, state)
        try:
            for index in range(len(manifest.get("segments", [])), len(scenes)):
                comfy.model_management.throw_exception_if_processing_interrupted()
                manifest, _source = load_delivery_manifest(safe_chain, allow_new=True)
                if len(manifest["segments"]) != index:
                    raise RuntimeError("accepted MV manifest changed during local rendering")
                scene = scenes[index]
                prompt_item = prompts[index]
                parent_id = (
                    str(manifest["segments"][-1]["candidate_id"])
                    if manifest["segments"]
                    else ""
                )
                parent_revision = int(manifest["revision"])
                seed = (int(base_seed) + index) & 0xFFFFFFFFFFFFFFFF
                expected = {
                    "chain_id": safe_chain,
                    "index": index,
                    "parent_candidate_id": parent_id,
                    "parent_manifest_revision": parent_revision,
                    "frame_count": int(scene["frame_count"]),
                    "timeline_start_frame": int(scene["start_frame"]),
                    "timeline_end_frame": int(scene["end_frame"]),
                    "is_final_segment": index == len(scenes) - 1,
                    "model_id": str(model_id or "unknown"),
                    "sampling_summary": sampling_summary,
                    "prompt": str(prompt_item["prompt"]),
                    "seed": seed,
                    "width": int(width),
                    "height": int(height),
                }
                base_candidate_id = _candidate_base_id(expected, contract_sha256)
                candidate_json = _reusable_candidate(root, expected, base_candidate_id)
                if candidate_json is None:
                    candidate_id = _available_candidate_id(root, index, base_candidate_id)
                    try:
                        drive_audio = audio_sources["conditioning"]
                        render_audio = _audio_window(
                            drive_audio,
                            int(scene["start_frame"]),
                            int(scene["render_frame_count"]),
                            name=(
                                "vocal_lock_audio"
                                if vocal_lock_audio is not None
                                else "full_song"
                            ),
                        )
                        delivery_audio = _audio_window(
                            audio_sources["candidate_preview"],
                            int(scene["start_frame"]),
                            int(scene["frame_count"]),
                            name=(
                                "vocal_lock_audio"
                                if vocal_lock_audio is not None
                                else "full_song"
                            ),
                        )
                        (
                            positive,
                            av_latent,
                            _mux_audio,
                            conditioned_prompt,
                            _media_map,
                            _conditioning_report,
                        ) = build_conditioning(
                            clip,
                            video_vae,
                            audio_vae,
                            str(prompt_item["prompt"]),
                            int(width),
                            int(height),
                            int(scene["render_frame_count"]),
                            "Ref2VA",
                            "lock_source",
                            0.0,
                            True,
                            1,
                            True,
                            "match",
                            "official_2_to_15s",
                            render_audio,
                            render_audio,
                            None,
                            None,
                            {"ref_image_1": reference_image},
                            None,
                            None,
                            None,
                        )
                        sampled = _sample_one_segment(
                            model,
                            positive,
                            av_latent,
                            seed=seed,
                            steps=int(steps),
                            shift_video=float(shift_video),
                            shift_audio=float(shift_audio),
                            sampler_name=str(sampler_name),
                            scheduler=str(scheduler),
                            segment_index=index,
                            sampling_plan=None,
                        )
                        sampled.pop("_h3_t8_long_video_sampling_report", None)
                        frames, _generated_audio, _video_latent, _audio_latent = (
                            decode_av_latent(sampled, video_vae, audio_vae)
                        )
                        trimmed_frames, _unused_audio, _trim_report = trim_av_output(
                            frames,
                            0.0,
                            int(scene["frame_count"]) / FPS,
                            None,
                            FPS,
                        )
                        candidate_json, _video, _save_report = save_long_video_candidate(
                            trimmed_frames,
                            delivery_audio,
                            sampled,
                            safe_chain,
                            index,
                            int(scene["start_frame"]) / FPS,
                            index != len(scenes) - 1,
                            parent_id,
                            parent_revision,
                            candidate_id,
                            str(model_id or "unknown"),
                            sampling_summary,
                            conditioned_prompt,
                            seed,
                            FPS,
                            int(bit_depth),
                            int(crf),
                        )
                    finally:
                        _release_segment_memory()
                _preview, accepted, _manifest_path, _accept_report = (
                    accept_long_video_candidate(candidate_json, True, "reject_existing", True)
                )
                if not accepted:
                    raise RuntimeError(f"local MV scene {index} was not accepted")
                manifest, _source = load_delivery_manifest(safe_chain)
                state = _state_payload(
                    schema=state_schema,
                    chain_id=safe_chain,
                    contract_sha256=contract_sha256,
                    status="composing" if len(manifest["segments"]) == len(scenes) else "running",
                    scene_count=len(scenes),
                    accepted_count=len(manifest["segments"]),
                    current_scene_index=(
                        None if len(manifest["segments"]) == len(scenes) else len(manifest["segments"])
                    ),
                    external_api_used=False,
                )
                _atomic_write_json(state_path, state)

            assembled_path, assembled_report_json = compose_accepted_long_video(
                safe_chain,
                f"{filename_prefix}_segments",
                True,
                "none",
                0.0,
                int(crf),
            )
            final_path, master_report = _mux_master_audio(
                assembled_path,
                audio_sources["final_delivery"],
                int(scene_plan["total_frames"]),
                filename_prefix,
            )
            manifest, _source = load_delivery_manifest(safe_chain)
            state = _state_payload(
                schema=state_schema,
                chain_id=safe_chain,
                contract_sha256=contract_sha256,
                status="complete",
                scene_count=len(scenes),
                accepted_count=len(scenes),
                current_scene_index=None,
                final_video_path=final_path,
                final_video_sha256=master_report["output_sha256"],
                manifest_revision=int(manifest["revision"]),
                external_api_used=False,
                source_audio_policy=(
                    "isolated_vocal_lock_conditioning_full_song_muxed_once"
                    if vocal_lock_audio is not None
                    else "full_original_song_muxed_once"
                ),
            )
            _atomic_write_json(state_path, state)
            report = {
                **state,
                "state_path": str(state_path),
                "manifest_path": str(root / "manifest.json"),
                "sampling_summary": sampling_summary,
                "assembled_segments": json.loads(assembled_report_json),
                "master_audio": master_report,
                "notes": [
                    "all H3 scenes were sampled locally through the connected ComfyUI MODEL",
                    "no HTTP prompt queue, remote H3 gateway, remote LLM, TTS or external video API was used",
                    (
                        "isolated vocal_lock_audio drove H3 lock_source conditioning; the full "
                        "original song was used only by the final delivery mux"
                        if vocal_lock_audio is not None
                        else "the full original song replaced segment AAC in one final mux"
                    ),
                ],
            }
            return (
                final_path,
                str(root / "manifest.json"),
                len(scenes),
                "complete",
                _canonical_json(report, indent=2),
            )
        except BaseException as error:
            try:
                manifest, _source = load_delivery_manifest(safe_chain, allow_new=True)
                accepted_count = len(manifest["segments"])
            except BaseException:
                accepted_count = 0
            state = _state_payload(
                schema=state_schema,
                chain_id=safe_chain,
                contract_sha256=contract_sha256,
                status=(
                    "interrupted"
                    if isinstance(error, comfy.model_management.InterruptProcessingException)
                    else "failed"
                ),
                scene_count=len(scenes),
                accepted_count=accepted_count,
                current_scene_index=accepted_count if accepted_count < len(scenes) else None,
                last_error=f"{type(error).__name__}: {error}",
                external_api_used=False,
            )
            try:
                _atomic_write_json(state_path, state)
            except OSError as state_error:
                add_note = getattr(error, "add_note", None)
                if callable(add_note):
                    add_note(f"could not persist local MV failure state: {state_error}")
            raise
        finally:
            _release_segment_memory()


def run_local_mv_vocal_lock_in_node_loop(
    model,
    clip,
    video_vae,
    audio_vae,
    reference_image,
    full_song,
    vocal_lock_audio,
    mv_vocal_lock_prompt_plan: Mapping,
    *,
    chain_id: str,
    width: int,
    height: int,
    base_seed: int,
    steps: int,
    shift_video: float,
    shift_audio: float,
    sampler_name: str,
    scheduler: str,
    resume_existing: bool,
    filename_prefix: str,
    bit_depth: int,
    crf: int,
    model_id: str,
) -> tuple[str, str, int, str, str]:
    prompt_plan = validate_mv_vocal_lock_prompt_plan(mv_vocal_lock_prompt_plan)
    audio_contract = _validate_vocal_lock_audio_contract(
        full_song,
        vocal_lock_audio,
        int(prompt_plan["scene_plan"]["total_frames"]),
    )
    result = run_local_mv_in_node_loop(
        model,
        clip,
        video_vae,
        audio_vae,
        reference_image,
        full_song,
        prompt_plan,
        chain_id=chain_id,
        width=width,
        height=height,
        base_seed=base_seed,
        steps=steps,
        shift_video=shift_video,
        shift_audio=shift_audio,
        sampler_name=sampler_name,
        scheduler=scheduler,
        resume_existing=resume_existing,
        filename_prefix=filename_prefix,
        bit_depth=bit_depth,
        crf=crf,
        model_id=model_id,
        vocal_lock_audio=vocal_lock_audio,
        prompt_plan_validator=validate_mv_vocal_lock_prompt_plan,
        route_revision="vocal_lock_v2",
        loop_state_name=MV_VOCAL_LOCK_LOOP_STATE_NAME,
        state_schema="t8.minimax_h3.mv_vocal_lock_in_node_loop.v2",
    )
    report = json.loads(result[4])
    report.update(
        {
            "route": "local_mv_vocal_lock_v2",
            "vocal_lock_audio_validation": audio_contract,
            "conditioning_audio": "vocal_lock_audio",
            "delivery_audio": "full_song_muxed_once",
            "lipsync_claim": (
                "audio-conditioned Vocal Lock candidate; visual synchronization still requires "
                "objective offset review and normal-speed human review"
            ),
        }
    )
    return (*result[:4], _canonical_json(report, indent=2))
