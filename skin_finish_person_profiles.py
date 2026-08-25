from __future__ import annotations

import gc
import json
import re
import time
from typing import Any

import torch

from .multiface_refine_advanced import _mask_at_source
from .skin_finish import (
    PRESET_CONFIG,
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
from .skin_finish_multiface_parser import (
    SKIN_FINISH_MULTIFACE_SEMANTIC_SCHEMA,
    _identity_labels,
    _validate_track_plan,
)
from .skin_finish_p1 import _shot_for_frame


SKIN_FINISH_PERSON_PROFILES_SCHEMA = "h3_t8_skin_finish_person_profiles/v1"
SKIN_FINISH_PERSON_PROFILE_REPORT_SCHEMA = "h3_t8_skin_finish_person_profile_report/v1"
MAX_PERSON_PROFILES = 8
_SHOT_TRACK_PATTERN = re.compile(r"^(0|[1-9][0-9]*):(0|[1-9][0-9]*)$")
_PREVIEW_COLORS = (
    (0.10, 0.75, 1.00),
    (1.00, 0.55, 0.10),
    (0.30, 0.90, 0.35),
    (0.85, 0.30, 1.00),
    (1.00, 0.85, 0.20),
    (0.20, 0.95, 0.80),
    (1.00, 0.35, 0.55),
    (0.55, 0.55, 1.00),
)


class _PerPersonUnavailable(RuntimeError):
    def __init__(self, status: str, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail


def _validate_parameter_values(
    *,
    preset: str,
    amount: float,
    texture_keep: float,
    shine_control: float,
    tone_adjust: float,
) -> dict[str, Any]:
    if preset not in PRESET_CONFIG:
        raise ValueError(f"Unsupported Skin Finish preset: {preset}")
    if not 0.0 <= float(amount) <= 1.0:
        raise ValueError("amount must stay within 0..1")
    if not 0.0 <= float(texture_keep) <= 1.0:
        raise ValueError("texture_keep must stay within 0..1")
    if not 0.0 <= float(shine_control) <= 1.0:
        raise ValueError("shine_control must stay within 0..1")
    if not -1.0 <= float(tone_adjust) <= 1.0:
        raise ValueError("tone_adjust must stay within -1..1")
    return {
        "preset": preset,
        "amount": float(amount),
        "texture_keep": float(texture_keep),
        "shine_control": float(shine_control),
        "tone_adjust": float(tone_adjust),
    }


def _validate_selector(selector_type: str, selector: str) -> str:
    value = str(selector).strip()
    if selector_type not in {"character_id", "shot_track"}:
        raise ValueError(f"Unsupported selector_type: {selector_type}")
    if not value or len(value) > 96 or any(ord(character) < 32 for character in value):
        raise ValueError("selector must be a non-empty printable value of at most 96 characters")
    if selector_type == "shot_track" and _SHOT_TRACK_PATTERN.fullmatch(value) is None:
        raise ValueError("shot_track selector must use shot:track syntax, for example 0:1")
    return value


def _profile_stack_hash(payload: dict[str, Any]) -> str:
    return _json_hash(payload)


def _validate_profile_stack(value: dict | None) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict) or value.get("schema") != SKIN_FINISH_PERSON_PROFILES_SCHEMA:
        raise _PerPersonUnavailable(
            "ABSTAIN_PERSON_PROFILES_INVALID",
            f"profiles must use {SKIN_FINISH_PERSON_PROFILES_SCHEMA}",
        )
    expected_keys = {
        "schema",
        "status",
        "profiles",
        "profile_count",
        "automatic_accept",
        "sha256",
    }
    if set(value) != expected_keys:
        raise _PerPersonUnavailable(
            "ABSTAIN_PERSON_PROFILES_INVALID",
            "profiles contains missing or unknown top-level fields",
        )
    if (
        value.get("status") != "skin_finish_person_profiles_ready"
        or value.get("automatic_accept") is not False
    ):
        raise _PerPersonUnavailable(
            "ABSTAIN_PERSON_PROFILES_INVALID",
            "profiles is not in the ready, non-automatic state",
        )
    unsigned = {key: item for key, item in value.items() if key != "sha256"}
    if value.get("sha256") != _profile_stack_hash(unsigned):
        raise _PerPersonUnavailable(
            "ABSTAIN_PERSON_PROFILES_INVALID",
            "profiles hash mismatch; the profile stack may be stale or modified",
        )
    profiles = value.get("profiles")
    if not isinstance(profiles, list) or not 1 <= len(profiles) <= MAX_PERSON_PROFILES:
        raise _PerPersonUnavailable(
            "ABSTAIN_PERSON_PROFILES_INVALID",
            f"profiles must contain 1..{MAX_PERSON_PROFILES} entries",
        )
    if value.get("profile_count") != len(profiles):
        raise _PerPersonUnavailable(
            "ABSTAIN_PERSON_PROFILES_INVALID",
            "profiles profile_count does not match the actual entries",
        )
    seen: set[tuple[str, str]] = set()
    normalized = []
    try:
        for profile in profiles:
            if not isinstance(profile, dict) or set(profile) != {
                "selector_type",
                "selector",
                "preset",
                "amount",
                "texture_keep",
                "shine_control",
                "tone_adjust",
            }:
                raise ValueError("profile contains missing or unknown fields")
            selector_type = str(profile["selector_type"])
            selector = _validate_selector(selector_type, profile["selector"])
            key = (selector_type, selector)
            if key in seen:
                raise ValueError(f"duplicate Skin Finish profile selector: {selector_type}:{selector}")
            seen.add(key)
            parameters = _validate_parameter_values(
                preset=str(profile["preset"]),
                amount=float(profile["amount"]),
                texture_keep=float(profile["texture_keep"]),
                shine_control=float(profile["shine_control"]),
                tone_adjust=float(profile["tone_adjust"]),
            )
            normalized.append(
                {
                    "selector_type": selector_type,
                    "selector": selector,
                    **parameters,
                }
            )
    except (TypeError, ValueError) as error:
        raise _PerPersonUnavailable(
            "ABSTAIN_PERSON_PROFILES_INVALID", str(error)
        ) from error
    return {**value, "profiles": normalized}


def build_skin_finish_person_profile(
    selector_type: str,
    selector: str,
    preset: str,
    amount: float,
    texture_keep: float,
    shine_control: float,
    tone_adjust: float,
    previous_profiles: dict | None = None,
) -> tuple[dict, str]:
    previous = _validate_profile_stack(previous_profiles)
    selector = _validate_selector(selector_type, selector)
    profile = {
        "selector_type": selector_type,
        "selector": selector,
        **_validate_parameter_values(
            preset=preset,
            amount=amount,
            texture_keep=texture_keep,
            shine_control=shine_control,
            tone_adjust=tone_adjust,
        ),
    }
    existing = list((previous or {}).get("profiles", []))
    key = (selector_type, selector)
    if any((item["selector_type"], item["selector"]) == key for item in existing):
        raise ValueError(f"Duplicate Skin Finish selector: {selector_type}:{selector}")
    if len(existing) >= MAX_PERSON_PROFILES:
        raise ValueError(f"At most {MAX_PERSON_PROFILES} Skin Finish profiles may be chained")
    profiles = existing + [profile]
    stack = {
        "schema": SKIN_FINISH_PERSON_PROFILES_SCHEMA,
        "status": "skin_finish_person_profiles_ready",
        "profiles": profiles,
        "profile_count": len(profiles),
        "automatic_accept": False,
    }
    stack["sha256"] = _profile_stack_hash(stack)
    report = {
        "schema": SKIN_FINISH_PERSON_PROFILE_REPORT_SCHEMA,
        "status": stack["status"],
        "profile_count": len(profiles),
        "added_selector": {"type": selector_type, "value": selector},
        "precedence": "shot_track_over_character_id_over_optional_default",
        "tone_adjust_semantics": (
            "bounded midtone exposure adjustment, not automatic skin-tone estimation or identity"
        ),
        "profiles_sha256": stack["sha256"],
        "automatic_accept": False,
    }
    return stack, canonical_json(report)


def _parse_semantic_report(
    value: str,
    *,
    plan: dict,
    semantic_mask: torch.Tensor,
) -> dict[str, Any]:
    try:
        report = json.loads(str(value))
    except (TypeError, json.JSONDecodeError) as error:
        raise _PerPersonUnavailable(
            "ABSTAIN_SEMANTIC_REPORT_INVALID",
            f"semantic_report_json must contain valid JSON: {error}",
        ) from error
    if not isinstance(report, dict) or report.get("schema") != SKIN_FINISH_MULTIFACE_SEMANTIC_SCHEMA:
        raise _PerPersonUnavailable(
            "ABSTAIN_SEMANTIC_REPORT_INVALID",
            f"semantic_report_json must use {SKIN_FINISH_MULTIFACE_SEMANTIC_SCHEMA}",
        )
    if report.get("status") != "READY":
        raise _PerPersonUnavailable(
            "ABSTAIN_SEMANTIC_REPORT_NOT_READY",
            f"semantic parser status is {report.get('status')!r}, not READY",
        )
    if report.get("track_plan_sha256") != plan.get("sha256"):
        raise _PerPersonUnavailable(
            "ABSTAIN_SEMANTIC_REPORT_TRACK_MISMATCH",
            "semantic report belongs to a different SAM3.1 track plan",
        )
    expected_source = plan.get("source", {})
    observed_source = report.get("source", {})
    for key in ("frame_count", "height", "width", "proxy_sha256"):
        if observed_source.get(key) != expected_source.get(key):
            raise _PerPersonUnavailable(
                "ABSTAIN_SEMANTIC_REPORT_SOURCE_MISMATCH",
                "semantic report belongs to different source pixels or geometry",
            )
    mask_hash = _tensor_proxy_sha256(semantic_mask)
    if report.get("mask_proxy_sha256") != mask_hash:
        raise _PerPersonUnavailable(
            "ABSTAIN_SEMANTIC_MASK_REPORT_MISMATCH",
            "semantic_skin_mask does not match the connected parser report",
        )
    return report


def _resolve_routes(
    *,
    plan: dict,
    profile_stack: dict | None,
    identity_assignment: dict | None,
    default_policy: str,
    default_parameters: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, int | None], dict[str, str], dict[str, Any]]:
    if default_policy not in {"source_unmatched", "default_profile"}:
        raise ValueError(f"Unsupported default_policy: {default_policy}")
    try:
        identity_labels = _identity_labels(identity_assignment, plan)
    except Exception as error:
        raise _PerPersonUnavailable(
            "ABSTAIN_IDENTITY_ASSIGNMENT_INVALID", str(error)
        ) from error

    profiles = list((profile_stack or {}).get("profiles", []))
    exact: dict[str, int] = {}
    character: dict[str, int] = {}
    routes: list[dict[str, Any]] = []
    for index, profile in enumerate(profiles):
        route = {
            "route_id": index,
            "source": profile["selector_type"],
            "selector": profile["selector"],
            "parameters": {
                key: profile[key]
                for key in (
                    "preset",
                    "amount",
                    "texture_keep",
                    "shine_control",
                    "tone_adjust",
                )
            },
        }
        routes.append(route)
        target = exact if profile["selector_type"] == "shot_track" else character
        target[profile["selector"]] = index

    valid_track_keys = [
        str(key) for shot in plan["shots"] for key in shot["track_keys"]
    ]
    unknown_tracks = sorted(set(exact) - set(valid_track_keys))
    if unknown_tracks:
        raise _PerPersonUnavailable(
            "ABSTAIN_PROFILE_SELECTOR_UNRESOLVED",
            f"shot_track profiles reference unknown tracks: {unknown_tracks}",
        )
    if character and identity_assignment is None:
        raise _PerPersonUnavailable(
            "ABSTAIN_CHARACTER_PROFILE_REQUIRES_IDENTITY_ASSIGNMENT",
            "character_id profiles require a source-bound identity_assignment",
        )
    unknown_characters = sorted(set(character) - set(identity_labels.values()))
    if unknown_characters:
        raise _PerPersonUnavailable(
            "ABSTAIN_PROFILE_SELECTOR_UNRESOLVED",
            f"character_id profiles did not resolve to any reviewed track: {unknown_characters}",
        )

    default_route_id: int | None = None
    if default_policy == "default_profile":
        default_route_id = len(routes)
        routes.append(
            {
                "route_id": default_route_id,
                "source": "default_profile",
                "selector": "*",
                "parameters": default_parameters,
            }
        )
    if not routes:
        raise _PerPersonUnavailable(
            "ABSTAIN_PERSON_PROFILES_MISSING",
            "Connect at least one profile or choose default_profile for unmatched tracks",
        )

    track_routes: dict[str, int | None] = {}
    resolution = []
    for track_key in valid_track_keys:
        character_id = identity_labels.get(track_key)
        if track_key in exact:
            route_id = exact[track_key]
            source = "shot_track"
        elif character_id in character:
            route_id = character[str(character_id)]
            source = "character_id"
        else:
            route_id = default_route_id
            source = "default_profile" if default_route_id is not None else "source_unmatched"
        track_routes[track_key] = route_id
        resolution.append(
            {
                "track_key": track_key,
                "character_id": character_id,
                "route_id": route_id,
                "resolved_by": source,
            }
        )

    # An exact shot:track override may legitimately shadow the only occurrence of a
    # character profile in a short clip.  The character selector was still resolved
    # against the reviewed identity assignment, so do not misclassify that deliberate
    # precedence as an unresolved profile.
    return routes, track_routes, identity_labels, {
        "precedence": "shot_track_over_character_id_over_optional_default",
        "default_policy": default_policy,
        "tracks": resolution,
        "identity_is_suggestion_not_proof": True,
    }


def _preview_indices(frame_count: int, count: int) -> list[int]:
    count = max(1, min(int(count), frame_count))
    if count == 1:
        return [0]
    return sorted(
        {int(round(index * (frame_count - 1) / (count - 1))) for index in range(count)}
    )


def _render_ownership_preview(
    frame: torch.Tensor,
    route_masks: dict[int, torch.Tensor],
    rejected: torch.Tensor,
    route_colors: dict[int, tuple[float, float, float]],
) -> torch.Tensor:
    output = frame[..., :3].detach().to(device="cpu", dtype=torch.float32).clone()
    for route_id, mask in route_masks.items():
        alpha = mask.detach().to(device="cpu", dtype=torch.float32).clamp(0.0, 1.0) * 0.55
        color = torch.tensor(route_colors[route_id], dtype=torch.float32).view(1, 1, 3)
        output = output * (1.0 - alpha[..., None]) + color * alpha[..., None]
    rejected_alpha = rejected.detach().to(device="cpu", dtype=torch.float32).clamp(0.0, 1.0) * 0.60
    rejected_color = torch.tensor((1.0, 0.05, 0.05), dtype=torch.float32).view(1, 1, 3)
    output = output * (1.0 - rejected_alpha[..., None]) + rejected_color * rejected_alpha[..., None]
    return output.clamp(0.0, 1.0)


def run_skin_finish_per_person(
    frames: torch.Tensor,
    track_plan: dict,
    semantic_skin_mask: torch.Tensor,
    semantic_report_json: str,
    default_policy: str,
    default_preset: str,
    default_amount: float,
    default_texture_keep: float,
    default_shine_control: float,
    default_tone_adjust: float,
    execution_mode: str,
    accept_candidate: bool,
    chunk_frames: int,
    proxy_long_side: int,
    preview_count: int,
    profiles: dict | None = None,
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
    default_parameters = _validate_parameter_values(
        preset=default_preset,
        amount=default_amount,
        texture_keep=default_texture_keep,
        shine_control=default_shine_control,
        tone_adjust=default_tone_adjust,
    )
    audio_report = _audio_contract(audio)
    memory_before = _memory_snapshot()
    zero = torch.zeros((frame_count, height, width), dtype=torch.float32)
    used_mask = zero.clone()
    rejected_mask = zero.clone()
    candidate: torch.Tensor = frames
    preview_frames: dict[int, torch.Tensor] = {}
    semantic: torch.Tensor | None = None
    semantic_report: dict[str, Any] = {}
    routing_report: dict[str, Any] = {}
    route_stats: dict[int, dict[str, Any]] = {}
    routes: list[dict[str, Any]] = []
    detail = ""
    findings: list[str] = []
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
            try:
                plan = _validate_track_plan(frames, track_plan)
            except Exception as error:
                raise _PerPersonUnavailable(
                    getattr(error, "status", "ABSTAIN_TRACK_PLAN_MISSING_OR_INVALID"),
                    getattr(error, "detail", str(error)),
                ) from error
            semantic = _normalize_mask(
                semantic_skin_mask,
                frame_count,
                height,
                width,
                name="semantic_skin_mask",
            )
            semantic_report = _parse_semantic_report(
                semantic_report_json,
                plan=plan,
                semantic_mask=semantic,
            )
            profile_stack = _validate_profile_stack(profiles)
            routes, track_routes, _, routing_report = _resolve_routes(
                plan=plan,
                profile_stack=profile_stack,
                identity_assignment=identity_assignment,
                default_policy=default_policy,
                default_parameters=default_parameters,
            )
            route_colors = {
                route["route_id"]: _PREVIEW_COLORS[route["route_id"] % len(_PREVIEW_COLORS)]
                for route in routes
            }
            route_stats = {
                route["route_id"]: {
                    "route_id": route["route_id"],
                    "selector_source": route["source"],
                    "selector": route["selector"],
                    "track_keys": [],
                    "frames_with_owned_skin": 0,
                    "owned_skin_pixels": 0,
                    "color_rgb": route_colors[route["route_id"]],
                    "_diagnostic_pixel_count": 0,
                    "_source_luma_sum": 0.0,
                    "_candidate_luma_sum": 0.0,
                    "_absolute_rgb_delta_sum": 0.0,
                    "_absolute_rgb_delta_max": 0.0,
                    "_candidate_low_clip_pixels": 0,
                    "_candidate_high_clip_pixels": 0,
                }
                for route in routes
            }
            for item in routing_report["tracks"]:
                route_id = item["route_id"]
                if route_id is not None:
                    route_stats[route_id]["track_keys"].append(item["track_key"])

            candidate = torch.empty(tuple(frames.shape), dtype=frames.dtype, device="cpu")
            preview_indices = set(_preview_indices(frame_count, int(preview_count)))
            for start in range(0, frame_count, int(chunk_frames)):
                end = min(frame_count, start + int(chunk_frames))
                _interrupt_and_progress(progress, start, frame_count)
                source_chunk = frames[start:end].detach().to(device="cpu")
                candidate_chunk = source_chunk.clone()
                route_masks = {
                    route["route_id"]: torch.zeros(
                        (end - start, height, width), dtype=torch.float32
                    )
                    for route in routes
                }
                for local_index, frame_index in enumerate(range(start, end)):
                    shot = _shot_for_frame(plan, frame_index)
                    shot_local = frame_index - int(shot["start_frame"])
                    person_masks = [
                        _mask_at_source(shot, shot_local, track_index, height, width)
                        for track_index in range(int(shot["object_count"]))
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
                    for track_index, person_mask in enumerate(person_masks):
                        track_key = str(shot["track_keys"][track_index])
                        route_id = track_routes[track_key]
                        if route_id is None:
                            continue
                        owned = semantic_frame * (person_mask & unique_owner).float()
                        if bool((owned > 1.0e-5).any()):
                            route_masks[route_id][local_index] = torch.maximum(
                                route_masks[route_id][local_index], owned
                            )
                            any_used |= owned > 1.0e-5
                            route_stats[route_id]["frames_with_owned_skin"] += 1
                            route_stats[route_id]["owned_skin_pixels"] += int(
                                (owned > 1.0e-5).sum()
                            )
                            frame_route_masks[route_id] = route_masks[route_id][local_index]
                    used_frame = torch.zeros((height, width), dtype=torch.float32)
                    for route_mask in route_masks.values():
                        used_frame = torch.maximum(used_frame, route_mask[local_index])
                    used_mask[frame_index] = used_frame
                    rejected_frame = torch.where(
                        any_used,
                        torch.zeros_like(semantic_frame),
                        semantic_frame,
                    )
                    rejected_mask[frame_index] = rejected_frame
                    unmatched_pixel_count += int(
                        (semantic_active & unique_owner & ~any_used).sum()
                    )
                    if frame_index in preview_indices:
                        preview_frames[frame_index] = _render_ownership_preview(
                            source_chunk[local_index],
                            frame_route_masks,
                            rejected_frame,
                            route_colors,
                        )

                for route in routes:
                    route_id = route["route_id"]
                    route_mask = route_masks[route_id]
                    if not bool((route_mask > 1.0e-5).any()):
                        continue
                    parameters = route["parameters"]
                    processed = _process_chunk(
                        source_chunk,
                        route_mask,
                        preset=parameters["preset"],
                        amount=parameters["amount"],
                        texture_keep=parameters["texture_keep"],
                        shine_control=parameters["shine_control"],
                        tone_adjust=parameters["tone_adjust"],
                        proxy_long_side=int(proxy_long_side),
                    )
                    active = route_mask > 1.0e-5
                    source_rgb = source_chunk[..., :3][active].float()
                    processed_rgb = processed[..., :3][active].float()
                    if int(source_rgb.shape[0]) > 0:
                        luma_weights = source_rgb.new_tensor((0.2126, 0.7152, 0.0722))
                        source_luma = (source_rgb * luma_weights).sum(dim=-1)
                        candidate_luma = (processed_rgb * luma_weights).sum(dim=-1)
                        absolute_delta = (processed_rgb - source_rgb).abs()
                        stats = route_stats[route_id]
                        stats["_diagnostic_pixel_count"] += int(source_rgb.shape[0])
                        stats["_source_luma_sum"] += float(source_luma.double().sum())
                        stats["_candidate_luma_sum"] += float(
                            candidate_luma.double().sum()
                        )
                        stats["_absolute_rgb_delta_sum"] += float(
                            absolute_delta.double().sum()
                        )
                        stats["_absolute_rgb_delta_max"] = max(
                            float(stats["_absolute_rgb_delta_max"]),
                            float(absolute_delta.max()),
                        )
                        stats["_candidate_low_clip_pixels"] += int(
                            (processed_rgb <= 1.0e-6).any(dim=-1).sum()
                        )
                        stats["_candidate_high_clip_pixels"] += int(
                            (processed_rgb >= 1.0 - 1.0e-6).any(dim=-1).sum()
                        )
                    candidate_chunk[..., :3][active] = processed[..., :3][active]
                candidate[start:end] = candidate_chunk
                _interrupt_and_progress(progress, end, frame_count)

            if int(torch.count_nonzero(used_mask)) == 0:
                status = "ABSTAIN_NO_PROFILED_SKIN_PIXELS"
                candidate = frames
                findings.append("no_unambiguous_semantic_skin_resolved_to_a_profile")
            else:
                status = "CANDIDATE_READY"
            if ambiguous_pixel_count:
                findings.append("overlapping_person_masks_preserved_source")
            if unmatched_pixel_count:
                findings.append("unmatched_person_skin_preserved_source")
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
        status = "ABSTAIN_PER_PERSON_ROUTING_FAILED"
        detail = f"{type(error).__name__}: {error}"
        findings.append("unexpected_routing_failure_closed_to_source")
        candidate = frames
    finally:
        gc.collect()

    if not bool(torch.isfinite(candidate).all()):
        raise ValueError("Per-person Skin Finish candidate contains NaN or Inf")
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
        if not torch.equal(
            candidate_chunk[..., :3][outside], source_chunk[..., :3][outside]
        ):
            outside_exact = False
        if channels > 3 and not torch.equal(
            candidate_chunk[..., 3:], source_chunk[..., 3:]
        ):
            alpha_preserved = False
    if not outside_exact:
        raise RuntimeError("Per-person Skin Finish changed pixels outside the owned skin mask")
    if not alpha_preserved:
        raise RuntimeError("Per-person Skin Finish changed alpha or auxiliary channels")

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
        "mode": "per_person_per_shot_advanced",
        "source_proxy_sha256": _tensor_proxy_sha256(frames),
        "mask_proxy_sha256": _tensor_proxy_sha256(used_mask),
        "frame_count": frame_count,
        "height": height,
        "width": width,
        "preset": "per_person_profiles",
        "profiles_sha256": str((profiles or {}).get("sha256", "")),
        "track_plan_sha256": str((track_plan or {}).get("sha256", "")),
        "accepted_candidate": accepted,
        "automatic_accept": False,
    }
    state["sha256"] = _json_hash(state)
    route_summaries = []
    for route in routes:
        summary = dict(route_stats.get(route["route_id"], {}))
        diagnostic_pixel_count = int(summary.pop("_diagnostic_pixel_count", 0))
        source_luma_sum = float(summary.pop("_source_luma_sum", 0.0))
        candidate_luma_sum = float(summary.pop("_candidate_luma_sum", 0.0))
        absolute_rgb_delta_sum = float(summary.pop("_absolute_rgb_delta_sum", 0.0))
        absolute_rgb_delta_max = float(summary.pop("_absolute_rgb_delta_max", 0.0))
        candidate_low_clip_pixels = int(summary.pop("_candidate_low_clip_pixels", 0))
        candidate_high_clip_pixels = int(summary.pop("_candidate_high_clip_pixels", 0))
        summary["treatment_diagnostics"] = {
            "metric_space": "display_referred_sdr_rec709_luma_proxy",
            "pixel_count": diagnostic_pixel_count,
            "source_mean_luma_proxy": (
                round(source_luma_sum / diagnostic_pixel_count, 10)
                if diagnostic_pixel_count
                else None
            ),
            "candidate_mean_luma_proxy": (
                round(candidate_luma_sum / diagnostic_pixel_count, 10)
                if diagnostic_pixel_count
                else None
            ),
            "mean_abs_rgb_delta": (
                round(absolute_rgb_delta_sum / (diagnostic_pixel_count * 3), 10)
                if diagnostic_pixel_count
                else None
            ),
            "max_abs_rgb_delta": (
                round(absolute_rgb_delta_max, 10)
                if diagnostic_pixel_count
                else None
            ),
            "candidate_low_clip_fraction": (
                round(candidate_low_clip_pixels / diagnostic_pixel_count, 10)
                if diagnostic_pixel_count
                else None
            ),
            "candidate_high_clip_fraction": (
                round(candidate_high_clip_pixels / diagnostic_pixel_count, 10)
                if diagnostic_pixel_count
                else None
            ),
        }
        summary["parameters"] = route["parameters"]
        route_summaries.append(summary)
    report = {
        "schema": SKIN_FINISH_REPORT_SCHEMA,
        "status": status,
        "detail": detail,
        "mode": "per_person_per_shot_advanced",
        "findings": findings,
        "product_boundary": (
            "Explicit per-character or shot-local non-generative SDR skin finishing only. "
            "Identity assignments remain reviewed routing suggestions; the node never infers "
            "skin tone, reconstructs faces, deblurs, changes lip sync or auto-accepts a result."
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
        "mask_source": {
            "source": "source_bound_multiface_semantic_mask",
            "semantic_report_schema": semantic_report.get("schema"),
            "semantic_mask_proxy_sha256": semantic_report.get("mask_proxy_sha256"),
            "track_plan_sha256": semantic_report.get("track_plan_sha256"),
        },
        "routing": {
            **routing_report,
            "routes": route_summaries,
            "overlap_policy": "source_on_any_multi_track_overlap",
            "semantic_active_pixels": semantic_pixel_count,
            "ambiguous_overlap_pixels": ambiguous_pixel_count,
            "unmatched_unique_owner_pixels": unmatched_pixel_count,
            "processed_pixels": int(torch.count_nonzero(used_mask > 1.0e-5)),
            "rejected_pixels": int(torch.count_nonzero(rejected_mask > 1.0e-5)),
            "diagnostic_contract": {
                "metric_space": "display_referred_sdr_rec709_luma_proxy",
                "automatic_fairness_decision": False,
                "interpretation": (
                    "Per-route luminance, treatment magnitude and clipping are review aids only; "
                    "they do not establish skin-tone fairness, beauty, identity or naturalness."
                ),
            },
        },
        "parameters": {
            "default_policy": default_policy,
            "default_parameters": default_parameters,
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
            "audio_object_passthrough": True,
        },
        "difference": {
            "mean_abs_rgb": round(difference_sum / max(1, difference_count), 10),
            "max_abs_rgb": round(difference_max, 10),
        },
        "audio": audio_report,
        "memory_before": memory_before,
        "memory_after": _memory_snapshot(),
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "state_sha256": state["sha256"],
        "review_required": status == "CANDIDATE_READY",
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
