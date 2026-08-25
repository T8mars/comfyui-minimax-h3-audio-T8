from __future__ import annotations

import hashlib
import json
import time
from typing import Any

import torch

from .skin_finish import (
    _audio_contract,
    _interrupt_and_progress,
    _memory_snapshot,
    _progress_bar,
    _tensor_proxy_sha256,
    _validate_frames,
    canonical_json,
)
from .skin_finish_frequency import (
    _two_pass_box_lowpass,
    separate_skin_finish_frequencies,
)


SKIN_FINISH_SPECULAR_FREQUENCY_SCHEMA = (
    "h3_t8_skin_finish_specular_frequency_split/v1"
)


def _json_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _smoothstep(value: torch.Tensor, start: float, end: float) -> torch.Tensor:
    normalized = ((value - float(start)) / (float(end) - float(start))).clamp(0.0, 1.0)
    return normalized.square() * (3.0 - 2.0 * normalized)


def separate_skin_finish_specular_frequencies(
    source_frames: torch.Tensor,
    candidate_frames: torch.Tensor,
    used_skin_mask: torch.Tensor,
    *,
    low_frequency_strength: float = 1.0,
    source_detail_gain: float = 1.0,
    separation_radius_percent: float = 3.0,
    maximum_radius_px: int = 32,
    highlight_detail_suppression: float = 0.65,
    highlight_start: float = 0.60,
    highlight_end: float = 0.92,
    positive_detail_threshold: float = 0.004,
    treatment_intent_scale: float = 0.004,
    maximum_specular_delta: float = 0.04,
    minimum_mask_area: float = 0.0001,
    maximum_mask_area: float = 0.50,
    maximum_new_clipped_fraction: float = 0.0005,
    clipping_epsilon: float = 1.0 / 255.0,
    chunk_frames: int = 4,
    accept_candidate: bool = False,
    audio: dict | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict | None,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    str,
]:
    """Preserve source detail while selectively restore the candidate's highlight intent.

    The ordinary Frequency Split remains untouched. This append-only experimental route first
    executes that exact split, then blends back only the part of the original Skin Finish
    candidate that the split lost where all of the following agree: the source is bright, the
    source has positive local detail and the original candidate is darker than the reconstructed
    base. The correction is a bounded interpolation toward the input candidate, never an
    invented subtraction beyond it. Negative detail, ordinary non-highlight texture, mask
    exterior, alpha and audio remain source-bound.
    """

    started = time.perf_counter()
    if not 0.0 <= float(highlight_detail_suppression) <= 1.0:
        raise ValueError("highlight_detail_suppression must stay within 0..1")
    if not 0.0 <= float(highlight_start) < float(highlight_end) <= 1.0:
        raise ValueError("highlight range must satisfy 0 <= start < end <= 1")
    if not 0.0 <= float(positive_detail_threshold) <= 0.05:
        raise ValueError("positive_detail_threshold must stay within 0..0.05")
    if not 0.0001 <= float(treatment_intent_scale) <= 0.10:
        raise ValueError("treatment_intent_scale must stay within 0.0001..0.10")
    if not 0.0 <= float(maximum_specular_delta) <= 0.10:
        raise ValueError("maximum_specular_delta must stay within 0..0.10")

    (
        base_candidate,
        base_source,
        _,
        base_audio,
        base_mask,
        base_rejected,
        base_difference,
        base_report_json,
    ) = separate_skin_finish_frequencies(
        source_frames,
        candidate_frames,
        used_skin_mask,
        low_frequency_strength=low_frequency_strength,
        source_detail_gain=source_detail_gain,
        separation_radius_percent=separation_radius_percent,
        maximum_radius_px=maximum_radius_px,
        minimum_mask_area=minimum_mask_area,
        maximum_mask_area=maximum_mask_area,
        maximum_new_clipped_fraction=maximum_new_clipped_fraction,
        clipping_epsilon=clipping_epsilon,
        chunk_frames=chunk_frames,
        accept_candidate=False,
        audio=audio,
    )
    base_report = json.loads(base_report_json)
    frame_count, height, width, channels = _validate_frames(
        source_frames, name="source_frames"
    )
    if tuple(base_candidate.shape) != tuple(source_frames.shape):
        raise RuntimeError("base Frequency Split changed the source geometry")
    audio_report = _audio_contract(audio)
    memory_before = _memory_snapshot()

    if float(highlight_detail_suppression) == 0.0:
        accepted = bool(accept_candidate) and bool(base_report["accepted_frame_count"])
        selected = base_candidate if accepted else source_frames
        report = {
            "schema": SKIN_FINISH_SPECULAR_FREQUENCY_SCHEMA,
            "status": base_report["status"],
            "method": "ordinary_frequency_split_exact_noop_specular_stage",
            "base_frequency_report": base_report,
            "parameters": {
                "highlight_detail_suppression": 0.0,
                "highlight_start": float(highlight_start),
                "highlight_end": float(highlight_end),
                "positive_detail_threshold": float(positive_detail_threshold),
                "treatment_intent_scale": float(treatment_intent_scale),
                "maximum_specular_delta": float(maximum_specular_delta),
                "chunk_frames": int(chunk_frames),
            },
            "accepted_frame_count": int(base_report["accepted_frame_count"]),
            "rejected_frame_count": int(base_report["rejected_frame_count"]),
            "accepted_frame_indices": list(base_report["accepted_frame_indices"]),
            "rejected_frame_indices": list(base_report["rejected_frame_indices"]),
            "mechanical_gates": {
                "zero_strength_exact_base_candidate": True,
                "outside_effective_mask_bit_exact": True,
                "alpha_or_aux_channels_preserved": True,
                "audio_object_passthrough": True,
                "automatic_accept": False,
                "candidate_selected": accepted,
            },
            "audio": audio_report,
            "memory_before": memory_before,
            "memory_after": _memory_snapshot(),
            "elapsed_seconds": round(time.perf_counter() - started, 6),
            "human_review_required": bool(base_report["accepted_frame_count"]),
        }
        return (
            base_candidate,
            base_source,
            selected,
            base_audio,
            base_mask,
            base_rejected,
            base_difference,
            canonical_json(report),
        )

    requested_radius = max(
        1,
        int(round(min(height, width) * float(separation_radius_percent) / 100.0)),
    )
    actual_radius = min(requested_radius, int(maximum_radius_px))
    chunk_size = int(chunk_frames)
    clip_epsilon = float(clipping_epsilon)
    luma_weights = torch.tensor([0.2126, 0.7152, 0.0722], dtype=torch.float32).view(
        1, 3, 1, 1
    )
    output = torch.empty(tuple(source_frames.shape), dtype=source_frames.dtype, device="cpu")
    effective_mask = torch.zeros_like(base_mask, device="cpu")
    rejected_mask = base_rejected.detach().to(device="cpu").clone()
    difference = torch.zeros(
        (frame_count, height, width, 3), dtype=torch.float16, device="cpu"
    )
    accepted_indices: list[int] = []
    rejected_indices: list[int] = []
    frame_reports: list[dict[str, Any]] = []
    outside_exact = True
    auxiliary_preserved = True
    progress = _progress_bar(frame_count)

    for start in range(0, frame_count, chunk_size):
        end = min(frame_count, start + chunk_size)
        _interrupt_and_progress(progress, start, frame_count)
        source_chunk = source_frames[start:end].detach().to(device="cpu")
        candidate_chunk = candidate_frames[start:end].detach().to(device="cpu")
        base_chunk = base_candidate[start:end].detach().to(device="cpu")
        mask_chunk = base_mask[start:end].detach().to(device="cpu", dtype=torch.float32)
        source_rgb = source_chunk[..., :3].float()
        candidate_rgb = candidate_chunk[..., :3].float()
        base_rgb = base_chunk[..., :3].float()
        source_nchw = source_rgb.movedim(-1, 1)
        candidate_nchw = candidate_rgb.movedim(-1, 1)
        base_nchw = base_rgb.movedim(-1, 1)
        source_luma = (source_nchw * luma_weights).sum(dim=1, keepdim=True)
        candidate_luma = (candidate_nchw * luma_weights).sum(dim=1, keepdim=True)
        base_luma = (base_nchw * luma_weights).sum(dim=1, keepdim=True)
        low_source_luma = _two_pass_box_lowpass(source_luma, actual_radius)
        positive_detail = (
            source_luma - low_source_luma - float(positive_detail_threshold)
        ).clamp_min(0.0)
        highlight_gate = _smoothstep(
            source_luma, float(highlight_start), float(highlight_end)
        )
        lost_candidate_darkening = (base_luma - candidate_luma).clamp_min(0.0)
        treatment_intent = (
            lost_candidate_darkening / float(treatment_intent_scale)
        ).clamp(0.0, 1.0)
        positive_detail_gate = (
            positive_detail / max(float(positive_detail_threshold), 1.0e-6)
        ).clamp(0.0, 1.0)
        blend_strength = (
            positive_detail_gate
            * highlight_gate
            * treatment_intent
            * float(highlight_detail_suppression)
        )
        requested_correction = (candidate_nchw - base_nchw) * blend_strength
        requested_luma_correction = (
            requested_correction * luma_weights
        ).sum(dim=1, keepdim=True)
        requested_darkening = (-requested_luma_correction).clamp_min(0.0)
        cap_scale = (
            float(maximum_specular_delta)
            / requested_darkening.clamp_min(1.0e-8)
        ).clamp(max=1.0)
        applied_correction = requested_correction * cap_scale
        applied_darkening = (
            -(applied_correction * luma_weights).sum(dim=1, keepdim=True)
        ).clamp_min(0.0)
        alpha = mask_chunk.unsqueeze(1).clamp(0.0, 1.0)
        adjusted_nchw = base_nchw + applied_correction * alpha
        # A convex blend toward the input candidate is the core safety boundary.
        lower = torch.minimum(base_nchw, candidate_nchw)
        upper = torch.maximum(base_nchw, candidate_nchw)
        adjusted_nchw = torch.maximum(lower, torch.minimum(upper, adjusted_nchw))
        adjusted_rgb = adjusted_nchw.movedim(1, -1).clamp(0.0, 1.0)
        composed_rgb = torch.where(alpha.movedim(1, -1) > 0.0, adjusted_rgb, source_rgb)

        candidate_bound_valid = (
            (adjusted_nchw >= lower - 1.0e-7)
            & (adjusted_nchw <= upper + 1.0e-7)
        )

        mask_area = (mask_chunk > 1.0e-5).float().mean(dim=(1, 2))
        mask_valid = (mask_area >= float(minimum_mask_area)) & (
            mask_area <= float(maximum_mask_area)
        )
        source_clipped = (
            (source_rgb <= clip_epsilon) | (source_rgb >= 1.0 - clip_epsilon)
        ).any(dim=-1)
        candidate_clipped = (
            (composed_rgb <= clip_epsilon) | (composed_rgb >= 1.0 - clip_epsilon)
        ).any(dim=-1)
        mask_binary = mask_chunk > 0.10
        new_clipped = candidate_clipped & ~source_clipped & mask_binary
        mask_pixels = mask_binary.flatten(1).sum(dim=1).clamp_min(1)
        new_clipped_fraction = new_clipped.flatten(1).sum(dim=1).float() / mask_pixels
        clipping_valid = new_clipped_fraction <= float(maximum_new_clipped_fraction)
        base_valid = mask_binary.flatten(1).any(dim=1)
        frame_valid = mask_valid & clipping_valid & base_valid

        for local_index in range(end - start):
            absolute_index = start + local_index
            reasons: list[str] = []
            if not bool(base_valid[local_index]):
                reasons.append("base_frequency_frame_rejected")
            if not bool(mask_valid[local_index]):
                reasons.append("mask_area_gate_failed")
            if not bool(clipping_valid[local_index]):
                reasons.append("new_clipping_limit_failed")
            active = mask_binary[local_index].unsqueeze(0)
            local_darkening = applied_darkening[local_index][active]
            if bool(frame_valid[local_index]):
                accepted_indices.append(absolute_index)
                frame_rgb = composed_rgb[local_index]
                effective_mask[absolute_index] = mask_chunk[local_index]
            else:
                rejected_indices.append(absolute_index)
                frame_rgb = source_rgb[local_index]
                rejected_mask[absolute_index] = torch.maximum(
                    rejected_mask[absolute_index], mask_chunk[local_index]
                )
            output[absolute_index] = source_chunk[local_index]
            output[absolute_index, ..., :3] = frame_rgb.to(dtype=source_chunk.dtype)
            difference[absolute_index] = (
                frame_rgb - source_rgb[local_index]
            ).abs().to(dtype=torch.float16)
            frame_reports.append(
                {
                    "frame_index": absolute_index,
                    "status": "PASS" if bool(frame_valid[local_index]) else "REJECT",
                    "reasons": reasons,
                    "mask_area_fraction": round(float(mask_area[local_index]), 8),
                    "mean_applied_specular_delta": round(
                        float(local_darkening.mean()) if int(local_darkening.numel()) else 0.0,
                        8,
                    ),
                    "peak_applied_specular_delta": round(
                        float(local_darkening.max()) if int(local_darkening.numel()) else 0.0,
                        8,
                    ),
                    "candidate_interpolation_bound": bool(
                        candidate_bound_valid[local_index].all()
                    ),
                    "new_clipped_fraction": round(
                        float(new_clipped_fraction[local_index]), 8
                    ),
                }
            )
        output_chunk = output[start:end]
        outside = effective_mask[start:end] <= 0.0
        if not torch.equal(
            output_chunk[..., :3][outside], source_chunk[..., :3][outside]
        ):
            outside_exact = False
        if channels > 3 and not torch.equal(output_chunk[..., 3:], source_chunk[..., 3:]):
            auxiliary_preserved = False
        _interrupt_and_progress(progress, end, frame_count)

    if not outside_exact:
        raise RuntimeError("Specular Frequency Split changed pixels outside its effective mask")
    if not auxiliary_preserved:
        raise RuntimeError("Specular Frequency Split changed alpha or auxiliary channels")
    if not bool(torch.isfinite(output).all()):
        raise RuntimeError("Specular Frequency Split produced NaN or Inf")
    accepted = bool(accept_candidate) and bool(accepted_indices)
    selected = output if accepted else source_frames
    state = {
        "schema": SKIN_FINISH_SPECULAR_FREQUENCY_SCHEMA,
        "source_proxy_sha256": _tensor_proxy_sha256(source_frames),
        "input_candidate_proxy_sha256": _tensor_proxy_sha256(candidate_frames),
        "base_frequency_candidate_proxy_sha256": _tensor_proxy_sha256(base_candidate),
        "mask_proxy_sha256": _tensor_proxy_sha256(base_mask),
        "accepted_frame_indices": accepted_indices,
        "rejected_frame_indices": rejected_indices,
        "candidate_selected": accepted,
    }
    state["sha256"] = _json_hash(state)
    status = (
        "PASS_WITH_REJECTED_FRAMES"
        if accepted_indices and rejected_indices
        else "PASS"
        if accepted_indices
        else "ABSTAIN_ALL_FRAMES_REJECTED"
    )
    report = {
        "schema": SKIN_FINISH_SPECULAR_FREQUENCY_SCHEMA,
        "status": status,
        "method": "ordinary_frequency_split_plus_candidate_bounded_highlight_intent_restore",
        "product_boundary": (
            "Non-generative SDR highlight-detail attenuation. It cannot infer physical skin, "
            "create pores, deblur, repair identity or prove a more natural result."
        ),
        "base_frequency_report": base_report,
        "parameters": {
            "low_frequency_strength": float(low_frequency_strength),
            "source_detail_gain": float(source_detail_gain),
            "separation_radius_percent": float(separation_radius_percent),
            "requested_radius_px": requested_radius,
            "actual_radius_px": actual_radius,
            "maximum_radius_px": int(maximum_radius_px),
            "highlight_detail_suppression": float(highlight_detail_suppression),
            "highlight_start": float(highlight_start),
            "highlight_end": float(highlight_end),
            "positive_detail_threshold": float(positive_detail_threshold),
            "treatment_intent_scale": float(treatment_intent_scale),
            "maximum_specular_delta": float(maximum_specular_delta),
            "maximum_new_clipped_fraction": float(maximum_new_clipped_fraction),
            "clipping_epsilon": float(clipping_epsilon),
            "chunk_frames": chunk_size,
        },
        "frame_count": frame_count,
        "accepted_frame_count": len(accepted_indices),
        "rejected_frame_count": len(rejected_indices),
        "accepted_frame_indices": accepted_indices,
        "rejected_frame_indices": rejected_indices,
        "frame_reports": frame_reports,
        "mechanical_gates": {
            "shape_preserved": tuple(output.shape) == tuple(source_frames.shape),
            "finite": True,
            "outside_effective_mask_bit_exact": outside_exact,
            "alpha_or_aux_channels_preserved": auxiliary_preserved,
            "negative_detail_directly_suppressed": False,
            "bounded_between_frequency_base_and_input_candidate": all(
                bool(item["candidate_interpolation_bound"]) for item in frame_reports
            ),
            "source_overwrite_performed": False,
            "audio_object_passthrough": True,
            "automatic_accept": False,
            "candidate_selected": accepted,
        },
        "audio": audio_report,
        "memory_before": memory_before,
        "memory_after": _memory_snapshot(),
        "state_sha256": state["sha256"],
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "human_review_required": bool(accepted_indices),
    }
    return (
        output,
        source_frames,
        selected,
        audio,
        effective_mask,
        rejected_mask,
        difference,
        canonical_json(report),
    )
