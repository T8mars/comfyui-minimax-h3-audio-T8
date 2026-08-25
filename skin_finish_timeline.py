from __future__ import annotations

import gc
import math
import time
from collections import defaultdict
from typing import Any, Mapping

import torch

from .multiface_refine_advanced import _mask_at_source
from .skin_finish import (
    SKIN_FINISH_REPORT_SCHEMA,
    SKIN_FINISH_STATE_SCHEMA,
    _audio_contract,
    _interrupt_and_progress,
    _json_hash,
    _memory_snapshot,
    _normalize_mask,
    _process_chunk,
    _progress_bar,
    _tensor_proxy_sha256,
    _validate_frames,
    canonical_json,
)
from .skin_finish_multiface_parser import _identity_labels, _validate_track_plan
from .skin_finish_p1 import _shot_for_frame
from .skin_finish_person_profiles import (
    _PREVIEW_COLORS,
    _PerPersonUnavailable,
    _parse_semantic_report,
    _preview_indices,
    _render_ownership_preview,
    _validate_parameter_values,
)
from .studio_advanced import _hash as _studio_hash
from .studio_advanced import validate_timeline


SKIN_FINISH_TIMELINE_PLAN_SCHEMA = "h3_t8_skin_finish_timeline_plan/v1"
SKIN_FINISH_TIMELINE_PLAN_REPORT_SCHEMA = "h3_t8_skin_finish_timeline_plan_report/v1"
MAX_SKIN_FINISH_KEYFRAMES = 96
_SELECTOR_TYPES = {"global", "character_id", "shot_track"}
_CURVES = {"hold", "linear", "smoothstep"}
_NUMERIC_PARAMETERS = ("amount", "texture_keep", "shine_control", "tone_adjust")


def _validate_studio_timeline(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        timeline = validate_timeline(value)
    except Exception as error:
        raise _PerPersonUnavailable(
            "ABSTAIN_STUDIO_TIMELINE_INVALID", str(error)
        ) from error
    supplied_hash = timeline.get("timeline_hash")
    unsigned = {key: item for key, item in timeline.items() if key != "timeline_hash"}
    if not isinstance(supplied_hash, str) or supplied_hash != _studio_hash(unsigned):
        raise _PerPersonUnavailable(
            "ABSTAIN_STUDIO_TIMELINE_INVALID",
            "Studio Timeline hash mismatch; the timeline may be stale or modified",
        )
    shots = timeline["shots"]
    cursor = 0
    for expected_index, shot in enumerate(shots):
        if not isinstance(shot, Mapping):
            raise _PerPersonUnavailable(
                "ABSTAIN_STUDIO_TIMELINE_INVALID", "Studio Timeline shot must be an object"
            )
        try:
            index = int(shot["index"])
            start = int(shot["start_frame"])
            end = int(shot["end_frame_exclusive"])
            count = int(shot["frame_count"])
        except (KeyError, TypeError, ValueError) as error:
            raise _PerPersonUnavailable(
                "ABSTAIN_STUDIO_TIMELINE_INVALID",
                "Studio Timeline shot geometry is incomplete",
            ) from error
        if index != expected_index or start != cursor or count <= 0 or end != start + count:
            raise _PerPersonUnavailable(
                "ABSTAIN_STUDIO_TIMELINE_INVALID",
                "Studio Timeline shots must be ordered, contiguous and internally consistent",
            )
        cursor = end
    fps = float(timeline.get("fps", 0.0))
    if (
        not math.isfinite(fps)
        or fps <= 0.0
        or int(timeline.get("total_frames", -1)) != cursor
        or int(timeline.get("shot_count", -1)) != len(shots)
    ):
        raise _PerPersonUnavailable(
            "ABSTAIN_STUDIO_TIMELINE_INVALID",
            "Studio Timeline total frame count, shot count or fps is invalid",
        )
    return timeline


def _normalize_selector(selector_type: str, selector: str) -> str:
    selector_type = str(selector_type)
    if selector_type not in _SELECTOR_TYPES:
        raise ValueError(f"Unsupported selector_type: {selector_type}")
    if selector_type == "global":
        return "*"
    value = str(selector).strip()
    if not value or len(value) > 96 or any(ord(character) < 32 for character in value):
        raise ValueError("selector must be a non-empty printable value of at most 96 characters")
    if selector_type == "shot_track":
        parts = value.split(":")
        if (
            len(parts) != 2
            or not all(part.isdigit() for part in parts)
            or any(str(int(part)) != part for part in parts)
        ):
            raise ValueError("shot_track selector must use canonical SAM shot:track syntax, for example 0:1")
    return value


def _keyframe_payload(
    *,
    timeline: dict[str, Any],
    selector_type: str,
    selector: str,
    studio_shot_index: int,
    frame_in_shot: int,
    interpolation_to_next: str,
    preset: str,
    amount: float,
    texture_keep: float,
    shine_control: float,
    tone_adjust: float,
) -> dict[str, Any]:
    selector_type = str(selector_type)
    selector = _normalize_selector(selector_type, selector)
    shot_index = int(studio_shot_index)
    if shot_index < 0 or shot_index >= len(timeline["shots"]):
        raise ValueError(
            f"studio_shot_index must be between 0 and {len(timeline['shots']) - 1}"
        )
    shot = timeline["shots"][shot_index]
    local_frame = int(frame_in_shot)
    if local_frame < 0 or local_frame >= int(shot["frame_count"]):
        raise ValueError(
            f"frame_in_shot must be between 0 and {int(shot['frame_count']) - 1} "
            f"for Studio shot {shot_index}"
        )
    curve = str(interpolation_to_next)
    if curve not in _CURVES:
        raise ValueError(f"Unsupported interpolation_to_next: {curve}")
    parameters = _validate_parameter_values(
        preset=str(preset),
        amount=amount,
        texture_keep=texture_keep,
        shine_control=shine_control,
        tone_adjust=tone_adjust,
    )
    return {
        "selector_type": selector_type,
        "selector": selector,
        "studio_shot_index": shot_index,
        "frame_in_shot": local_frame,
        "absolute_frame": int(shot["start_frame"]) + local_frame,
        "interpolation_to_next": curve,
        **parameters,
    }


def _validate_timeline_plan(
    value: dict | None,
    *,
    timeline: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict) or value.get("schema") != SKIN_FINISH_TIMELINE_PLAN_SCHEMA:
        raise _PerPersonUnavailable(
            "ABSTAIN_SKIN_FINISH_TIMELINE_PLAN_INVALID",
            f"timeline_plan must use {SKIN_FINISH_TIMELINE_PLAN_SCHEMA}",
        )
    expected_keys = {
        "schema",
        "status",
        "timeline_hash",
        "project_id",
        "fps",
        "total_frames",
        "keyframes",
        "keyframe_count",
        "interpolation_contract",
        "automatic_accept",
        "sha256",
    }
    if set(value) != expected_keys:
        raise _PerPersonUnavailable(
            "ABSTAIN_SKIN_FINISH_TIMELINE_PLAN_INVALID",
            "timeline_plan contains missing or unknown top-level fields",
        )
    unsigned = {key: item for key, item in value.items() if key != "sha256"}
    if value.get("sha256") != _json_hash(unsigned):
        raise _PerPersonUnavailable(
            "ABSTAIN_SKIN_FINISH_TIMELINE_PLAN_INVALID",
            "timeline_plan hash mismatch; the plan may be stale or modified",
        )
    if (
        value.get("status") != "skin_finish_timeline_plan_ready"
        or value.get("automatic_accept") is not False
    ):
        raise _PerPersonUnavailable(
            "ABSTAIN_SKIN_FINISH_TIMELINE_PLAN_INVALID",
            "timeline_plan is not in the ready, non-automatic state",
        )
    if timeline is not None and (
        value.get("timeline_hash") != timeline.get("timeline_hash")
        or int(value.get("total_frames", -1)) != int(timeline.get("total_frames", -2))
        or abs(float(value.get("fps", 0.0)) - float(timeline.get("fps", -1.0))) > 1.0e-9
    ):
        raise _PerPersonUnavailable(
            "ABSTAIN_SKIN_FINISH_TIMELINE_MISMATCH",
            "timeline_plan belongs to a different Studio Timeline",
        )
    keyframes = value.get("keyframes")
    if not isinstance(keyframes, list) or not 1 <= len(keyframes) <= MAX_SKIN_FINISH_KEYFRAMES:
        raise _PerPersonUnavailable(
            "ABSTAIN_SKIN_FINISH_TIMELINE_PLAN_INVALID",
            f"timeline_plan must contain 1..{MAX_SKIN_FINISH_KEYFRAMES} keyframes",
        )
    if int(value.get("keyframe_count", -1)) != len(keyframes):
        raise _PerPersonUnavailable(
            "ABSTAIN_SKIN_FINISH_TIMELINE_PLAN_INVALID",
            "timeline_plan keyframe_count does not match the entries",
        )
    normalized = []
    seen: set[tuple[str, str, int, int]] = set()
    try:
        validation_timeline = timeline
        if validation_timeline is None:
            # A prior plan is only accepted by the builder when the current timeline is
            # supplied, so executor-side validation always has the authoritative timeline.
            raise ValueError("authoritative Studio Timeline is required")
        for raw in keyframes:
            if not isinstance(raw, dict) or set(raw) != {
                "selector_type",
                "selector",
                "studio_shot_index",
                "frame_in_shot",
                "absolute_frame",
                "interpolation_to_next",
                "preset",
                "amount",
                "texture_keep",
                "shine_control",
                "tone_adjust",
            }:
                raise ValueError("keyframe contains missing or unknown fields")
            item = _keyframe_payload(
                timeline=validation_timeline,
                selector_type=raw["selector_type"],
                selector=raw["selector"],
                studio_shot_index=raw["studio_shot_index"],
                frame_in_shot=raw["frame_in_shot"],
                interpolation_to_next=raw["interpolation_to_next"],
                preset=raw["preset"],
                amount=raw["amount"],
                texture_keep=raw["texture_keep"],
                shine_control=raw["shine_control"],
                tone_adjust=raw["tone_adjust"],
            )
            if item["absolute_frame"] != int(raw["absolute_frame"]):
                raise ValueError("keyframe absolute_frame does not match Studio Timeline geometry")
            key = (
                item["selector_type"],
                item["selector"],
                item["studio_shot_index"],
                item["frame_in_shot"],
            )
            if key in seen:
                raise ValueError("duplicate selector keyframe at the same Studio shot frame")
            seen.add(key)
            normalized.append(item)
    except (TypeError, ValueError) as error:
        raise _PerPersonUnavailable(
            "ABSTAIN_SKIN_FINISH_TIMELINE_PLAN_INVALID", str(error)
        ) from error
    expected_order = sorted(
        normalized,
        key=lambda item: (
            item["studio_shot_index"],
            item["selector_type"],
            item["selector"],
            item["frame_in_shot"],
        ),
    )
    if normalized != expected_order:
        raise _PerPersonUnavailable(
            "ABSTAIN_SKIN_FINISH_TIMELINE_PLAN_INVALID",
            "timeline_plan keyframes are not in canonical order",
        )
    return {**value, "keyframes": normalized}


def build_skin_finish_timeline_keyframe(
    studio_timeline: dict,
    selector_type: str,
    selector: str,
    studio_shot_index: int,
    frame_in_shot: int,
    interpolation_to_next: str,
    preset: str,
    amount: float,
    texture_keep: float,
    shine_control: float,
    tone_adjust: float,
    previous_plan: dict | None = None,
) -> tuple[dict, str]:
    timeline = _validate_studio_timeline(studio_timeline)
    previous = _validate_timeline_plan(previous_plan, timeline=timeline)
    item = _keyframe_payload(
        timeline=timeline,
        selector_type=selector_type,
        selector=selector,
        studio_shot_index=studio_shot_index,
        frame_in_shot=frame_in_shot,
        interpolation_to_next=interpolation_to_next,
        preset=preset,
        amount=amount,
        texture_keep=texture_keep,
        shine_control=shine_control,
        tone_adjust=tone_adjust,
    )
    keyframes = list((previous or {}).get("keyframes", []))
    key = (
        item["selector_type"],
        item["selector"],
        item["studio_shot_index"],
        item["frame_in_shot"],
    )
    if any(
        (
            existing["selector_type"],
            existing["selector"],
            existing["studio_shot_index"],
            existing["frame_in_shot"],
        )
        == key
        for existing in keyframes
    ):
        raise ValueError("Duplicate Skin Finish keyframe for this selector and Studio shot frame")
    if len(keyframes) >= MAX_SKIN_FINISH_KEYFRAMES:
        raise ValueError(f"At most {MAX_SKIN_FINISH_KEYFRAMES} Skin Finish keyframes may be chained")
    keyframes.append(item)
    keyframes.sort(
        key=lambda value: (
            value["studio_shot_index"],
            value["selector_type"],
            value["selector"],
            value["frame_in_shot"],
        )
    )
    plan = {
        "schema": SKIN_FINISH_TIMELINE_PLAN_SCHEMA,
        "status": "skin_finish_timeline_plan_ready",
        "timeline_hash": timeline["timeline_hash"],
        "project_id": timeline.get("project_id"),
        "fps": float(timeline["fps"]),
        "total_frames": int(timeline["total_frames"]),
        "keyframes": keyframes,
        "keyframe_count": len(keyframes),
        "interpolation_contract": {
            "scope": "within_each_studio_shot_only",
            "outside_key_range": "hold_nearest_key",
            "preset": "categorical_hold_until_destination_key",
            "continuous_parameters": list(_NUMERIC_PARAMETERS),
            "sam_track_domain_is_independent_from_studio_shot_domain": True,
        },
        "automatic_accept": False,
    }
    plan["sha256"] = _json_hash(plan)
    report = {
        "schema": SKIN_FINISH_TIMELINE_PLAN_REPORT_SCHEMA,
        "status": plan["status"],
        "keyframe_count": len(keyframes),
        "added_keyframe": {
            "selector_type": item["selector_type"],
            "selector": item["selector"],
            "studio_shot_index": item["studio_shot_index"],
            "frame_in_shot": item["frame_in_shot"],
            "absolute_frame": item["absolute_frame"],
        },
        "no_cross_shot_interpolation": True,
        "preset_is_not_numerically_interpolated": True,
        "timeline_plan_sha256": plan["sha256"],
        "automatic_accept": False,
    }
    return plan, canonical_json(report)


def _group_keyframes(plan: dict[str, Any]) -> dict[tuple[str, str, int], list[dict[str, Any]]]:
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for item in plan["keyframes"]:
        groups[(item["selector_type"], item["selector"], item["studio_shot_index"])].append(item)
    return dict(groups)


def _interpolated_parameters(
    keyframes: list[dict[str, Any]], frame_in_shot: int
) -> dict[str, Any]:
    if not keyframes:
        raise ValueError("Cannot evaluate an empty Skin Finish keyframe group")
    frame = int(frame_in_shot)
    if frame <= int(keyframes[0]["frame_in_shot"]):
        return {key: keyframes[0][key] for key in ("preset", *_NUMERIC_PARAMETERS)}
    if frame >= int(keyframes[-1]["frame_in_shot"]):
        return {key: keyframes[-1][key] for key in ("preset", *_NUMERIC_PARAMETERS)}
    for left, right in zip(keyframes, keyframes[1:], strict=True):
        left_frame = int(left["frame_in_shot"])
        right_frame = int(right["frame_in_shot"])
        if frame < right_frame:
            curve = left["interpolation_to_next"]
            if curve == "hold":
                weight = 0.0
            else:
                weight = (frame - left_frame) / (right_frame - left_frame)
                if curve == "smoothstep":
                    weight = weight * weight * (3.0 - 2.0 * weight)
            result: dict[str, Any] = {"preset": left["preset"]}
            for key in _NUMERIC_PARAMETERS:
                result[key] = float(left[key]) + (float(right[key]) - float(left[key])) * weight
            return result
    raise RuntimeError("Skin Finish keyframe interval resolution failed")


def _studio_shot_for_frame(timeline: dict[str, Any], frame_index: int) -> dict[str, Any]:
    for shot in timeline["shots"]:
        if int(shot["start_frame"]) <= frame_index < int(shot["end_frame_exclusive"]):
            return shot
    raise _PerPersonUnavailable(
        "ABSTAIN_STUDIO_TIMELINE_FRAME_UNMAPPED",
        f"frame {frame_index} is not covered by the Studio Timeline",
    )


def _validate_plan_selectors(
    *,
    groups: dict[tuple[str, str, int], list[dict[str, Any]]],
    track_plan: dict[str, Any],
    identity_assignment: dict | None,
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    try:
        identity_labels = _identity_labels(identity_assignment, track_plan)
    except Exception as error:
        raise _PerPersonUnavailable(
            "ABSTAIN_IDENTITY_ASSIGNMENT_INVALID", str(error)
        ) from error
    valid_tracks = {str(key) for shot in track_plan["shots"] for key in shot["track_keys"]}
    exact = {selector for selector_type, selector, _ in groups if selector_type == "shot_track"}
    characters = {
        selector for selector_type, selector, _ in groups if selector_type == "character_id"
    }
    unknown_tracks = sorted(exact - valid_tracks)
    if unknown_tracks:
        raise _PerPersonUnavailable(
            "ABSTAIN_TIMELINE_SELECTOR_UNRESOLVED",
            f"timeline plan references unknown SAM shot:track selectors: {unknown_tracks}",
        )
    if characters and identity_assignment is None:
        raise _PerPersonUnavailable(
            "ABSTAIN_CHARACTER_TIMELINE_REQUIRES_IDENTITY_ASSIGNMENT",
            "character_id keyframes require a source-bound identity_assignment",
        )
    unknown_characters = sorted(characters - set(identity_labels.values()))
    if unknown_characters:
        raise _PerPersonUnavailable(
            "ABSTAIN_TIMELINE_SELECTOR_UNRESOLVED",
            f"timeline plan character selectors did not resolve: {unknown_characters}",
        )
    selectors = sorted({(kind, selector) for kind, selector, _ in groups})
    route_catalog = [
        {
            "route_id": index,
            "selector_type": kind,
            "selector": selector,
            "color_rgb": _PREVIEW_COLORS[index % len(_PREVIEW_COLORS)],
        }
        for index, (kind, selector) in enumerate(selectors)
    ]
    return identity_labels, route_catalog


def run_skin_finish_timeline(
    frames: torch.Tensor,
    studio_timeline: dict,
    timeline_plan: dict,
    track_plan: dict,
    semantic_skin_mask: torch.Tensor,
    semantic_report_json: str,
    execution_mode: str,
    accept_candidate: bool,
    chunk_frames: int,
    proxy_long_side: int,
    preview_count: int,
    identity_assignment: dict | None = None,
    audio: dict | None = None,
) -> tuple:
    started = time.perf_counter()
    frame_count, height, width, channels = _validate_frames(frames)
    if execution_mode not in {"candidate_only", "review_only", "bypass"}:
        raise ValueError(f"Unsupported execution_mode: {execution_mode}")
    if not 1 <= int(chunk_frames) <= 32:
        raise ValueError("chunk_frames must stay within 1..32")
    if not 128 <= int(proxy_long_side) <= 1280:
        raise ValueError("proxy_long_side must stay within 128..1280")
    audio_report = _audio_contract(audio)
    memory_before = _memory_snapshot()
    zero = torch.zeros((frame_count, height, width), dtype=torch.float32)
    used_mask = zero.clone()
    rejected_mask = zero.clone()
    candidate: torch.Tensor = frames
    preview_frames: dict[int, torch.Tensor] = {}
    semantic: torch.Tensor | None = None
    semantic_report: dict[str, Any] = {}
    normalized_plan: dict[str, Any] = {}
    route_catalog: list[dict[str, Any]] = []
    route_stats: dict[int, dict[str, Any]] = {}
    findings: list[str] = []
    detail = ""
    status = "BYPASS_EXACT" if execution_mode == "bypass" else "ABSTAIN_NOT_EXECUTED"
    ambiguous_pixel_count = 0
    unmatched_pixel_count = 0
    semantic_pixel_count = 0
    progress = _progress_bar(frame_count)

    try:
        if execution_mode == "bypass":
            findings.append("execution_mode_bypass")
            _interrupt_and_progress(progress, frame_count, frame_count)
        else:
            timeline = _validate_studio_timeline(studio_timeline)
            if int(timeline["total_frames"]) != frame_count:
                raise _PerPersonUnavailable(
                    "ABSTAIN_STUDIO_TIMELINE_SOURCE_MISMATCH",
                    f"Studio Timeline has {timeline['total_frames']} frames but source has {frame_count}",
                )
            try:
                person_plan = _validate_track_plan(frames, track_plan)
            except Exception as error:
                raise _PerPersonUnavailable(
                    getattr(error, "status", "ABSTAIN_TRACK_PLAN_MISSING_OR_INVALID"),
                    getattr(error, "detail", str(error)),
                ) from error
            track_fps = float(person_plan.get("source", {}).get("fps", 0.0))
            if not math.isfinite(track_fps) or abs(track_fps - float(timeline["fps"])) > 1.0e-9:
                raise _PerPersonUnavailable(
                    "ABSTAIN_STUDIO_TIMELINE_FPS_MISMATCH",
                    "SAM track-plan fps does not match Studio Timeline fps",
                )
            semantic = _normalize_mask(
                semantic_skin_mask,
                frame_count,
                height,
                width,
                name="semantic_skin_mask",
            )
            semantic_report = _parse_semantic_report(
                semantic_report_json,
                plan=person_plan,
                semantic_mask=semantic,
            )
            normalized_plan = _validate_timeline_plan(timeline_plan, timeline=timeline) or {}
            groups = _group_keyframes(normalized_plan)
            identity_labels, route_catalog = _validate_plan_selectors(
                groups=groups,
                track_plan=person_plan,
                identity_assignment=identity_assignment,
            )
            route_ids = {
                (item["selector_type"], item["selector"]): item["route_id"]
                for item in route_catalog
            }
            route_colors = {
                item["route_id"]: tuple(item["color_rgb"]) for item in route_catalog
            }
            route_stats = {
                item["route_id"]: {
                    **item,
                    "track_keys": set(),
                    "studio_shot_indices": set(),
                    "frames_with_owned_skin": 0,
                    "owned_skin_pixels": 0,
                }
                for item in route_catalog
            }
            candidate = torch.empty(tuple(frames.shape), dtype=frames.dtype, device="cpu")
            preview_indices = set(_preview_indices(frame_count, int(preview_count)))
            for start in range(0, frame_count, int(chunk_frames)):
                end = min(frame_count, start + int(chunk_frames))
                _interrupt_and_progress(progress, start, frame_count)
                source_chunk = frames[start:end].detach().to(device="cpu")
                for local_index, frame_index in enumerate(range(start, end)):
                    source_frame = source_chunk[local_index : local_index + 1]
                    candidate_frame = source_frame.clone()
                    studio_shot = _studio_shot_for_frame(timeline, frame_index)
                    studio_shot_index = int(studio_shot["index"])
                    studio_local = frame_index - int(studio_shot["start_frame"])
                    sam_shot = _shot_for_frame(person_plan, frame_index)
                    sam_local = frame_index - int(sam_shot["start_frame"])
                    person_masks = [
                        _mask_at_source(sam_shot, sam_local, track_index, height, width)
                        for track_index in range(int(sam_shot["object_count"]))
                    ]
                    ownership_count = torch.zeros((height, width), dtype=torch.uint8)
                    for person_mask in person_masks:
                        ownership_count.add_(person_mask.to(dtype=torch.uint8))
                    semantic_frame = semantic[frame_index]
                    semantic_active = semantic_frame > 1.0e-5
                    unique_owner = ownership_count == 1
                    ambiguous = semantic_active & (ownership_count > 1)
                    ambiguous_pixel_count += int(ambiguous.sum())
                    semantic_pixel_count += int(semantic_active.sum())
                    any_used = torch.zeros((height, width), dtype=torch.bool)
                    frame_route_masks: dict[int, torch.Tensor] = {}
                    frame_route_parameters: dict[int, dict[str, Any]] = {}
                    for track_index, person_mask in enumerate(person_masks):
                        track_key = str(sam_shot["track_keys"][track_index])
                        character_id = identity_labels.get(track_key)
                        candidates = (
                            ("shot_track", track_key, studio_shot_index),
                            ("character_id", str(character_id), studio_shot_index),
                            ("global", "*", studio_shot_index),
                        )
                        resolved = next(
                            (group_key for group_key in candidates if group_key in groups), None
                        )
                        if resolved is None:
                            continue
                        selector_type, selector, _ = resolved
                        route_id = route_ids[(selector_type, selector)]
                        owned = semantic_frame * (person_mask & unique_owner).float()
                        if not bool((owned > 1.0e-5).any()):
                            continue
                        if route_id in frame_route_masks:
                            frame_route_masks[route_id] = torch.maximum(
                                frame_route_masks[route_id], owned
                            )
                        else:
                            frame_route_masks[route_id] = owned
                        frame_route_parameters[route_id] = _interpolated_parameters(
                            groups[resolved], studio_local
                        )
                        any_used |= owned > 1.0e-5
                        stats = route_stats[route_id]
                        stats["track_keys"].add(track_key)
                        stats["studio_shot_indices"].add(studio_shot_index)
                        stats["frames_with_owned_skin"] += 1
                        stats["owned_skin_pixels"] += int((owned > 1.0e-5).sum())
                    used_frame = torch.zeros((height, width), dtype=torch.float32)
                    for route_mask in frame_route_masks.values():
                        used_frame = torch.maximum(used_frame, route_mask)
                    used_mask[frame_index] = used_frame
                    rejected_frame = torch.where(
                        any_used, torch.zeros_like(semantic_frame), semantic_frame
                    )
                    rejected_mask[frame_index] = rejected_frame
                    unmatched_pixel_count += int(
                        (semantic_active & unique_owner & ~any_used).sum()
                    )
                    for route_id, route_mask in frame_route_masks.items():
                        parameters = frame_route_parameters[route_id]
                        processed = _process_chunk(
                            source_frame,
                            route_mask.unsqueeze(0),
                            preset=parameters["preset"],
                            amount=parameters["amount"],
                            texture_keep=parameters["texture_keep"],
                            shine_control=parameters["shine_control"],
                            tone_adjust=parameters["tone_adjust"],
                            proxy_long_side=int(proxy_long_side),
                        )
                        active = route_mask > 1.0e-5
                        candidate_frame[0, ..., :3][active] = processed[0, ..., :3][active]
                    candidate[frame_index] = candidate_frame[0]
                    if frame_index in preview_indices:
                        preview_frames[frame_index] = _render_ownership_preview(
                            source_frame[0], frame_route_masks, rejected_frame, route_colors
                        )
                _interrupt_and_progress(progress, end, frame_count)
            if int(torch.count_nonzero(used_mask)) == 0:
                status = "ABSTAIN_NO_TIMELINE_PROFILED_SKIN_PIXELS"
                candidate = frames
                findings.append("no_unambiguous_skin_resolved_to_a_timeline_keyframe")
            else:
                status = "CANDIDATE_READY"
            if ambiguous_pixel_count:
                findings.append("overlapping_person_masks_preserved_source")
            if unmatched_pixel_count:
                findings.append("unmatched_timeline_skin_preserved_source")
            if execution_mode == "review_only":
                findings.append("review_only_forces_source_selection")
    except _PerPersonUnavailable as error:
        status = error.status
        detail = error.detail
        findings.append("contract_failed_closed_to_source")
        candidate = frames
        if semantic is not None:
            rejected_mask.copy_(semantic)
    except Exception as error:
        status = "ABSTAIN_SKIN_FINISH_TIMELINE_FAILED"
        detail = f"{type(error).__name__}: {error}"
        findings.append("unexpected_timeline_failure_closed_to_source")
        candidate = frames
    finally:
        gc.collect()

    if not bool(torch.isfinite(candidate).all()):
        raise ValueError("Timeline Skin Finish candidate contains NaN or Inf")
    outside_exact = True
    alpha_preserved = True
    difference_sum = 0.0
    difference_count = 0
    difference_max = 0.0
    for start in range(0, frame_count, max(1, int(chunk_frames))):
        end = min(frame_count, start + max(1, int(chunk_frames)))
        source_chunk = frames[start:end].detach().to(device="cpu")
        candidate_chunk = candidate[start:end].detach().to(device="cpu")
        delta = (candidate_chunk[..., :3].float() - source_chunk[..., :3].float()).abs()
        difference_sum += float(delta.double().sum())
        difference_count += int(delta.numel())
        difference_max = max(difference_max, float(delta.max()))
        outside = used_mask[start:end] <= 0
        if not torch.equal(candidate_chunk[..., :3][outside], source_chunk[..., :3][outside]):
            outside_exact = False
        if channels > 3 and not torch.equal(candidate_chunk[..., 3:], source_chunk[..., 3:]):
            alpha_preserved = False
    if not outside_exact:
        raise RuntimeError("Timeline Skin Finish changed pixels outside the owned skin mask")
    if not alpha_preserved:
        raise RuntimeError("Timeline Skin Finish changed alpha or auxiliary channels")

    accepted = (
        bool(accept_candidate)
        and status == "CANDIDATE_READY"
        and execution_mode != "review_only"
    )
    selected = candidate if accepted else frames
    if not preview_frames:
        preview = frames[:1, ..., :3].detach().to(device="cpu", dtype=torch.float32)
    else:
        preview = torch.stack([preview_frames[index] for index in sorted(preview_frames)])
    state = {
        "schema": SKIN_FINISH_STATE_SCHEMA,
        "status": status,
        "mode": "per_person_studio_timeline_advanced",
        "source_proxy_sha256": _tensor_proxy_sha256(frames),
        "mask_proxy_sha256": _tensor_proxy_sha256(used_mask),
        "frame_count": frame_count,
        "height": height,
        "width": width,
        "preset": "studio_timeline_keyframes",
        "timeline_plan_sha256": str((timeline_plan or {}).get("sha256", "")),
        "studio_timeline_hash": str((studio_timeline or {}).get("timeline_hash", "")),
        "track_plan_sha256": str((track_plan or {}).get("sha256", "")),
        "accepted_candidate": accepted,
        "automatic_accept": False,
    }
    state["sha256"] = _json_hash(state)
    route_summaries = []
    for route in route_catalog:
        stats = dict(route_stats.get(route["route_id"], route))
        stats["track_keys"] = sorted(stats.get("track_keys", []))
        stats["studio_shot_indices"] = sorted(stats.get("studio_shot_indices", []))
        route_summaries.append(stats)
    report = {
        "schema": SKIN_FINISH_REPORT_SCHEMA,
        "status": status,
        "detail": detail,
        "mode": "per_person_studio_timeline_advanced",
        "findings": findings,
        "product_boundary": (
            "Non-generative SDR Skin Finish keyframes only. Studio shots locate time while "
            "SAM shots locate people; they are deliberately independent. Interpolation never "
            "crosses a Studio cut, never changes audio and never auto-accepts a result."
        ),
        "source": {
            "frame_count": frame_count,
            "height": height,
            "width": width,
            "channels": channels,
            "dtype": str(frames.dtype),
            "device": str(frames.device),
            "proxy_sha256": state["source_proxy_sha256"],
        },
        "timeline": {
            "timeline_hash": state["studio_timeline_hash"],
            "timeline_plan_sha256": state["timeline_plan_sha256"],
            "keyframe_count": int(normalized_plan.get("keyframe_count", 0)),
            "no_cross_shot_interpolation": True,
            "preset_policy": "categorical_hold_until_destination_key",
            "sam_track_domain_is_independent_from_studio_shot_domain": True,
        },
        "mask_source": {
            "source": "source_bound_multiface_semantic_mask",
            "semantic_report_schema": semantic_report.get("schema"),
            "semantic_mask_proxy_sha256": semantic_report.get("mask_proxy_sha256"),
            "track_plan_sha256": semantic_report.get("track_plan_sha256"),
        },
        "routing": {
            "precedence": "sam_shot_track_over_character_id_over_global_over_source",
            "routes": route_summaries,
            "overlap_policy": "source_on_any_multi_track_overlap",
            "semantic_active_pixels": semantic_pixel_count,
            "ambiguous_overlap_pixels": ambiguous_pixel_count,
            "unmatched_unique_owner_pixels": unmatched_pixel_count,
            "processed_pixels": int(torch.count_nonzero(used_mask > 1.0e-5)),
            "rejected_pixels": int(torch.count_nonzero(rejected_mask > 1.0e-5)),
        },
        "parameters": {
            "execution_mode": execution_mode,
            "chunk_frames": int(chunk_frames),
            "proxy_long_side": int(proxy_long_side),
            "preview_count": int(preview_count),
        },
        "mechanical_gates": {
            "finite": True,
            "shape_preserved": tuple(candidate.shape) == tuple(frames.shape),
            "outside_mask_bit_exact": outside_exact,
            "alpha_or_aux_channels_preserved": alpha_preserved,
            "source_overwrite_performed": False,
            "candidate_selected": accepted,
            "automatic_accept": False,
            "audio_same_python_object": True,
        },
        "difference": {
            "mean_abs_rgb": difference_sum / max(1, difference_count),
            "max_abs_rgb": difference_max,
        },
        "audio": audio_report,
        "memory": {
            "before": memory_before,
            "after": _memory_snapshot(),
            "cpu_bounded_chunk": True,
            "persistent_model_cache": False,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "skin_finish_state_sha256": state["sha256"],
    }
    return (
        candidate,
        frames,
        selected,
        audio,
        used_mask,
        rejected_mask,
        preview,
        state,
        canonical_json(report),
    )
