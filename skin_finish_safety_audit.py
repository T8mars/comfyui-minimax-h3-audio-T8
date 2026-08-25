from __future__ import annotations

import hashlib
import time
from typing import Any

import torch

from .multiface_refine_advanced import _mask_at_source
from .skin_finish import (
    _audio_contract,
    _memory_snapshot,
    _normalize_mask,
    _tensor_proxy_sha256,
    _validate_frames,
    canonical_json,
)
from .skin_finish_multiface_parser import _validate_track_plan
from .skin_finish_p1 import _shot_for_frame


SKIN_FINISH_SAFETY_AUDIT_SCHEMA = "h3_t8_skin_finish_safety_audit/v1"
AUDIT_SCOPES = ("mask_only", "track_union", "unique_track_owner")
TEMPORAL_POLICIES = ("report_only", "hard_gate")


def _json_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _effect_vector(delta_rgb: torch.Tensor, active: torch.Tensor) -> list[float] | None:
    pixel_count = int(active.sum())
    if pixel_count < 1:
        return None
    selected = delta_rgb[active].float()
    mean_rgb = selected.mean(dim=0)
    mean_abs = selected.abs().mean()
    signed_luma = (
        mean_rgb[0] * 0.2126 + mean_rgb[1] * 0.7152 + mean_rgb[2] * 0.0722
    )
    # Signed chroma residuals are used only as change detectors.  They are not a
    # colour-science or skin-tone quality score.
    signed_cb = -0.114572 * mean_rgb[0] - 0.385428 * mean_rgb[1] + 0.5 * mean_rgb[2]
    signed_cr = 0.5 * mean_rgb[0] - 0.454153 * mean_rgb[1] - 0.045847 * mean_rgb[2]
    return [
        float(mean_abs),
        float(signed_luma),
        float(signed_cb),
        float(signed_cr),
    ]


def _effect_jump(current: list[float], previous: list[float]) -> float:
    return max(abs(float(a) - float(b)) for a, b in zip(current, previous, strict=True))


def _audio_match(
    audio_source: dict | None,
    audio_passthrough: dict | None,
) -> tuple[bool, str, dict[str, Any], dict[str, Any]]:
    if audio_source is None and audio_passthrough is None:
        empty = {"provided": False, "passthrough": True}
        return True, "not_provided", empty, empty
    if audio_source is None or audio_passthrough is None:
        source = _audio_contract(audio_source)
        candidate = _audio_contract(audio_passthrough)
        return False, "one_side_missing", source, candidate
    source = _audio_contract(audio_source)
    candidate = _audio_contract(audio_passthrough)
    exact = source == candidate
    return exact, "pcm_exact" if exact else "pcm_mismatch", source, candidate


def audit_skin_finish_candidate(
    source_frames: torch.Tensor,
    candidate_frames: torch.Tensor,
    used_skin_mask: torch.Tensor,
    *,
    audit_scope: str = "mask_only",
    temporal_policy: str = "report_only",
    maximum_mean_abs_change: float = 0.08,
    maximum_peak_abs_change: float = 0.30,
    maximum_temporal_effect_jump: float = 0.04,
    maximum_track_leak_fraction: float = 0.001,
    minimum_temporal_pixels: int = 64,
    scene_cut_reset_threshold: float = 0.20,
    accept_candidate: bool = False,
    track_plan: dict | None = None,
    audio_source: dict | None = None,
    audio_passthrough: dict | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict | None,
    bool,
    int,
    torch.Tensor,
    str,
]:
    """Audit a finished Skin Finish IMAGE candidate without grading aesthetics.

    The gate is intentionally asymmetric: it may reject a candidate for a measurable
    contract failure, but it never accepts one as aesthetically superior.  Explicit user
    acceptance is still required after all hard gates pass.
    """

    started = time.perf_counter()
    frame_count, height, width, channels = _validate_frames(
        source_frames, name="source_frames"
    )
    candidate_shape = _validate_frames(candidate_frames, name="candidate_frames")
    if candidate_shape != (frame_count, height, width, channels):
        raise ValueError("candidate_frames must exactly match source_frames shape")
    if audit_scope not in AUDIT_SCOPES:
        raise ValueError(f"Unsupported audit_scope: {audit_scope}")
    if temporal_policy not in TEMPORAL_POLICIES:
        raise ValueError(f"Unsupported temporal_policy: {temporal_policy}")
    if not 0.0 <= float(maximum_mean_abs_change) <= 1.0:
        raise ValueError("maximum_mean_abs_change must stay within 0..1")
    if not 0.0 <= float(maximum_peak_abs_change) <= 1.0:
        raise ValueError("maximum_peak_abs_change must stay within 0..1")
    if float(maximum_peak_abs_change) < float(maximum_mean_abs_change):
        raise ValueError("maximum_peak_abs_change must be >= maximum_mean_abs_change")
    if not 0.0 <= float(maximum_temporal_effect_jump) <= 1.0:
        raise ValueError("maximum_temporal_effect_jump must stay within 0..1")
    if not 0.0 <= float(maximum_track_leak_fraction) <= 1.0:
        raise ValueError("maximum_track_leak_fraction must stay within 0..1")
    if not 1 <= int(minimum_temporal_pixels) <= height * width:
        raise ValueError("minimum_temporal_pixels is outside the frame area")
    if not 0.0 <= float(scene_cut_reset_threshold) <= 1.0:
        raise ValueError("scene_cut_reset_threshold must stay within 0..1")

    source = source_frames.detach().to(device="cpu")
    candidate = candidate_frames.detach().to(device="cpu")
    mask = _normalize_mask(
        used_skin_mask,
        frame_count,
        height,
        width,
        name="used_skin_mask",
    )
    plan: dict | None = None
    plan_error = ""
    if audit_scope != "mask_only" or track_plan is not None:
        try:
            if track_plan is None:
                raise ValueError(f"{audit_scope} requires a source-bound track_plan")
            plan = _validate_track_plan(source_frames, track_plan)
        except Exception as error:
            plan_error = getattr(error, "detail", str(error))

    audio_exact, audio_status, source_audio, candidate_audio = _audio_match(
        audio_source,
        audio_passthrough,
    )
    memory_before = _memory_snapshot()
    frame_reports: list[dict[str, Any]] = []
    failed_indices: list[int] = []
    previous_effects: dict[str, dict[str, Any]] = {}
    maximum_observed_temporal_jump = 0.0
    total_active_pixels = 0
    total_track_leak_pixels = 0
    total_ambiguous_pixels = 0
    first_failure_mask: torch.Tensor | None = None
    first_failure_index: int | None = None
    previous_source_rgb: torch.Tensor | None = None

    for frame_index in range(frame_count):
        source_rgb = source[frame_index, ..., :3].float()
        candidate_rgb = candidate[frame_index, ..., :3].float()
        delta_rgb = candidate_rgb - source_rgb
        active = mask[frame_index] > 1.0e-8
        active_pixels = int(active.sum())
        total_active_pixels += active_pixels
        reasons: list[str] = []
        failure_mask = torch.zeros((height, width), dtype=torch.bool)

        if active_pixels:
            outside = ~active
            if not torch.equal(
                candidate[frame_index, ..., :3][outside],
                source[frame_index, ..., :3][outside],
            ):
                reasons.append("outside_skin_mask_pixels_changed")
                failure_mask |= outside & (
                    (candidate_rgb - source_rgb).abs().amax(dim=-1) > 0.0
                )
            selected_delta = delta_rgb[active]
            mean_abs_change = float(selected_delta.abs().mean())
            peak_abs_change = float(selected_delta.abs().max())
            if mean_abs_change > float(maximum_mean_abs_change):
                reasons.append("mean_abs_change_limit_failed")
                failure_mask |= active
            if peak_abs_change > float(maximum_peak_abs_change):
                reasons.append("peak_abs_change_limit_failed")
                failure_mask |= active & (
                    delta_rgb.abs().amax(dim=-1) > float(maximum_peak_abs_change)
                )
        else:
            mean_abs_change = 0.0
            peak_abs_change = 0.0

        if channels > 3 and not torch.equal(
            candidate[frame_index, ..., 3:], source[frame_index, ..., 3:]
        ):
            reasons.append("alpha_or_aux_channels_changed")
            failure_mask |= torch.ones_like(active)

        shot = None
        person_masks: list[torch.Tensor] = []
        ownership_count = torch.zeros((height, width), dtype=torch.uint8)
        shot_cut = False
        if plan is not None:
            shot = _shot_for_frame(plan, frame_index)
            shot_local = frame_index - int(shot["start_frame"])
            person_masks = [
                _mask_at_source(shot, shot_local, track_index, height, width)
                for track_index in range(int(shot["object_count"]))
            ]
            for person_mask in person_masks:
                ownership_count.add_(person_mask.to(dtype=torch.uint8))
            union = ownership_count > 0
            leak = active & ~union
            ambiguous = active & (ownership_count > 1)
            leak_pixels = int(leak.sum())
            ambiguous_pixels = int(ambiguous.sum())
            total_track_leak_pixels += leak_pixels
            total_ambiguous_pixels += ambiguous_pixels
            denominator = max(1, active_pixels)
            track_leak_fraction = leak_pixels / denominator
            ambiguous_fraction = ambiguous_pixels / denominator
            if (
                audit_scope in {"track_union", "unique_track_owner"}
                and track_leak_fraction > float(maximum_track_leak_fraction)
            ):
                reasons.append("skin_mask_outside_person_track")
                failure_mask |= leak
            if (
                audit_scope == "unique_track_owner"
                and ambiguous_fraction > float(maximum_track_leak_fraction)
            ):
                reasons.append("skin_mask_on_ambiguous_person_overlap")
                failure_mask |= ambiguous
            shot_cut = frame_index == int(shot["start_frame"])
        else:
            track_leak_fraction = 0.0
            ambiguous_fraction = 0.0
            if previous_source_rgb is not None:
                source_delta = float((source_rgb - previous_source_rgb).abs().mean())
                shot_cut = source_delta >= float(scene_cut_reset_threshold)

        effect_reports: list[dict[str, Any]] = []
        if plan is not None and shot is not None:
            for track_index, person_mask in enumerate(person_masks):
                track_key = str(shot["track_keys"][track_index])
                if audit_scope == "unique_track_owner":
                    local_active = active & person_mask & (ownership_count == 1)
                else:
                    local_active = active & person_mask
                local_pixels = int(local_active.sum())
                vector = (
                    _effect_vector(delta_rgb, local_active)
                    if local_pixels >= int(minimum_temporal_pixels)
                    else None
                )
                jump = None
                previous = previous_effects.get(track_key)
                if (
                    vector is not None
                    and previous is not None
                    and int(previous["frame_index"]) == frame_index - 1
                    and int(previous["shot_id"]) == int(shot["shot_id"])
                    and not shot_cut
                ):
                    jump = _effect_jump(vector, previous["vector"])
                    maximum_observed_temporal_jump = max(
                        maximum_observed_temporal_jump, jump
                    )
                    if (
                        temporal_policy == "hard_gate"
                        and jump > float(maximum_temporal_effect_jump)
                    ):
                        reasons.append(f"temporal_effect_jump:{track_key}")
                        failure_mask |= local_active
                if vector is not None:
                    previous_effects[track_key] = {
                        "frame_index": frame_index,
                        "shot_id": int(shot["shot_id"]),
                        "vector": vector,
                    }
                effect_reports.append(
                    {
                        "track_key": track_key,
                        "active_pixels": local_pixels,
                        "effect_vector": (
                            [round(value, 8) for value in vector]
                            if vector is not None
                            else None
                        ),
                        "temporal_effect_jump": (
                            round(float(jump), 8) if jump is not None else None
                        ),
                    }
                )
        else:
            vector = (
                _effect_vector(delta_rgb, active)
                if active_pixels >= int(minimum_temporal_pixels)
                else None
            )
            jump = None
            previous = previous_effects.get("global")
            if (
                vector is not None
                and previous is not None
                and int(previous["frame_index"]) == frame_index - 1
                and not shot_cut
            ):
                jump = _effect_jump(vector, previous["vector"])
                maximum_observed_temporal_jump = max(maximum_observed_temporal_jump, jump)
                if (
                    temporal_policy == "hard_gate"
                    and jump > float(maximum_temporal_effect_jump)
                ):
                    reasons.append("temporal_effect_jump:global")
                    failure_mask |= active
            if vector is not None:
                previous_effects["global"] = {
                    "frame_index": frame_index,
                    "shot_id": 0,
                    "vector": vector,
                }
            effect_reports.append(
                {
                    "track_key": "global",
                    "active_pixels": active_pixels,
                    "effect_vector": (
                        [round(value, 8) for value in vector]
                        if vector is not None
                        else None
                    ),
                    "temporal_effect_jump": (
                        round(float(jump), 8) if jump is not None else None
                    ),
                }
            )

        if reasons:
            failed_indices.append(frame_index)
            if first_failure_index is None:
                first_failure_index = frame_index
                first_failure_mask = failure_mask.clone()
        frame_reports.append(
            {
                "frame_index": frame_index,
                "status": "FAIL" if reasons else ("SOURCE_ONLY" if not active_pixels else "PASS"),
                "reasons": reasons,
                "active_skin_pixels": active_pixels,
                "mean_abs_change": round(mean_abs_change, 8),
                "peak_abs_change": round(peak_abs_change, 8),
                "track_leak_fraction": round(track_leak_fraction, 8),
                "ambiguous_owner_fraction": round(ambiguous_fraction, 8),
                "temporal_reset": bool(shot_cut),
                "effects": effect_reports,
            }
        )
        previous_source_rgb = source_rgb

    if plan_error:
        failed_indices = sorted(set(failed_indices) | set(range(frame_count)))
    if not audio_exact:
        failed_indices = sorted(set(failed_indices) | set(range(frame_count)))
    if total_active_pixels == 0:
        failed_indices = sorted(set(failed_indices) | set(range(frame_count)))

    hard_gate_pass = (
        not failed_indices
        and not plan_error
        and audio_exact
        and total_active_pixels > 0
    )
    accepted = bool(accept_candidate) and hard_gate_pass
    selected = candidate_frames if accepted else source_frames
    gated_candidate = candidate_frames if hard_gate_pass else source_frames
    selected_audio = (
        audio_passthrough
        if accepted and audio_passthrough is not None
        else audio_source
        if audio_source is not None
        else audio_passthrough
    )

    preview_index = first_failure_index if first_failure_index is not None else 0
    preview = source[preview_index, ..., :3].float().clone()
    preview_mask = (
        first_failure_mask
        if first_failure_mask is not None and bool(first_failure_mask.any())
        else mask[preview_index] > 1.0e-8
    )
    overlay_colour = torch.tensor(
        [1.0, 0.05, 0.05] if not hard_gate_pass else [0.05, 0.95, 0.20],
        dtype=torch.float32,
    )
    alpha = preview_mask.float().unsqueeze(-1) * 0.55
    preview = (preview * (1.0 - alpha) + overlay_colour * alpha).clamp(0.0, 1.0)

    status = (
        "PASS_HARD_GATES"
        if hard_gate_pass
        else "ABSTAIN_TRACK_PLAN_INVALID"
        if plan_error
        else "ABSTAIN_AUDIO_MISMATCH"
        if not audio_exact
        else "ABSTAIN_EMPTY_SKIN_MASK"
        if total_active_pixels == 0
        else "ABSTAIN_HARD_GATE_FAILED"
    )
    report: dict[str, Any] = {
        "schema": SKIN_FINISH_SAFETY_AUDIT_SCHEMA,
        "status": status,
        "product_boundary": (
            "Automatic hard-failure audit only. Exact mask containment, bounded pixel change, "
            "source-relative treatment jumps, track containment and PCM equality can reject a "
            "candidate; none of them can prove beauty, natural skin, identity, correct mouth "
            "semantics or aesthetic superiority. Human review remains mandatory."
        ),
        "source": {
            "frame_count": frame_count,
            "height": height,
            "width": width,
            "channels": channels,
            "proxy_sha256": _tensor_proxy_sha256(source_frames),
        },
        "candidate_proxy_sha256": _tensor_proxy_sha256(candidate_frames),
        "mask_proxy_sha256": _tensor_proxy_sha256(mask),
        "track_plan": {
            "required": audit_scope != "mask_only",
            "connected": track_plan is not None,
            "valid": plan is not None,
            "sha256": str((plan or {}).get("sha256", "")),
            "error": plan_error,
        },
        "parameters": {
            "audit_scope": audit_scope,
            "temporal_policy": temporal_policy,
            "maximum_mean_abs_change": float(maximum_mean_abs_change),
            "maximum_peak_abs_change": float(maximum_peak_abs_change),
            "maximum_temporal_effect_jump": float(maximum_temporal_effect_jump),
            "maximum_track_leak_fraction": float(maximum_track_leak_fraction),
            "minimum_temporal_pixels": int(minimum_temporal_pixels),
            "scene_cut_reset_threshold": float(scene_cut_reset_threshold),
        },
        "summary": {
            "hard_gate_pass": hard_gate_pass,
            "failed_frame_count": len(failed_indices),
            "failed_frame_indices": failed_indices,
            "active_skin_pixels": total_active_pixels,
            "track_leak_pixels": total_track_leak_pixels,
            "ambiguous_owner_pixels": total_ambiguous_pixels,
            "maximum_observed_temporal_effect_jump": round(
                maximum_observed_temporal_jump, 8
            ),
            "audio_status": audio_status,
            "candidate_selected": accepted,
            "automatic_reject": True,
            "automatic_accept": False,
            "human_review_required": hard_gate_pass,
        },
        "audio": {
            "status": audio_status,
            "exact": audio_exact,
            "source": source_audio,
            "candidate": candidate_audio,
        },
        "frame_reports": frame_reports,
        "memory_before": memory_before,
        "memory_after": _memory_snapshot(),
        "elapsed_seconds": round(time.perf_counter() - started, 6),
    }
    report["sha256"] = _json_hash(report)
    return (
        selected,
        gated_candidate,
        source_frames,
        selected_audio,
        hard_gate_pass,
        len(failed_indices),
        preview.unsqueeze(0),
        canonical_json(report),
    )
