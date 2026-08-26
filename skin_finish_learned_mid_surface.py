from __future__ import annotations

import time
from typing import Any

import torch

from .skin_finish import (
    _normalize_mask,
    _tensor_proxy_sha256,
    _validate_frames,
    canonical_json,
)
from .skin_finish_frequency import _masked_rms, _two_pass_box_lowpass
from .skin_finish_surface import _mask_interior_gate


SKIN_FINISH_LEARNED_MID_SURFACE_SCHEMA = (
    "h3_t8_skin_finish_learned_mid_surface_prototype/v1"
)
GFPGAN_REFERENCE = (
    "Wang, Li, Zhang and Shan, Towards Real-World Blind Face Restoration with "
    "Generative Facial Prior, CVPR 2021"
)


def _direction_preserving_cap(
    delta: torch.Tensor, maximum_abs_component: float
) -> tuple[torch.Tensor, torch.Tensor]:
    peak = delta.abs().amax(dim=1, keepdim=True)
    limiter = (
        torch.full_like(peak, float(maximum_abs_component))
        / peak.clamp_min(1.0e-8)
    ).clamp_max(1.0)
    return delta * limiter, limiter


def _frame_weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    numerator = (value * weight).flatten(1).sum(dim=1)
    denominator = weight.flatten(1).sum(dim=1).clamp_min(1.0)
    return numerator / denominator


def _masked_cosine(
    first: torch.Tensor, second: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    numerator = (first * second * weight).flatten(1).sum(dim=1)
    first_norm = torch.sqrt(
        (first.square() * weight).flatten(1).sum(dim=1).clamp_min(1.0e-12)
    )
    second_norm = torch.sqrt(
        (second.square() * weight).flatten(1).sum(dim=1).clamp_min(1.0e-12)
    )
    return numerator / (first_norm * second_norm).clamp_min(1.0e-12)


def fuse_learned_skin_mid_surface(
    source_frames: torch.Tensor,
    learned_proposal_frames: torch.Tensor,
    used_skin_mask: torch.Tensor,
    *,
    broad_amount: float = 0.65,
    mid_amount: float = 0.85,
    fine_split_radius_px: int = 2,
    broad_radius_px: int = 12,
    maximum_broad_component_delta: float = 0.18,
    maximum_mid_component_delta: float = 0.12,
    candidate_rgb_delta_cap: float = 0.10,
    minimum_masked_mean_abs_change: float = 0.012,
    maximum_masked_mean_abs_change: float = 0.060,
    minimum_applied_mid_rms: float = 0.0035,
    minimum_source_fine_cosine: float = 0.95,
    minimum_texture_ratio: float = 0.88,
    maximum_texture_ratio: float = 1.20,
    maximum_new_clipped_fraction: float = 0.002,
    minimum_mask_area: float = 0.0001,
    maximum_mask_area: float = 0.50,
    chunk_frames: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]:
    """Transfer broad and mid-frequency skin surface while retaining source fine detail.

    The learned proposal contributes only two bounded RGB surface bands. The source remains the
    pixel carrier and supplies all detail above the fine split. Semantic feature exclusion and
    identity/temporal checks are external responsibilities. This research primitive is not a
    registered ComfyUI node and contains no model loading or cache.
    """

    started = time.perf_counter()
    frame_count, height, width, channels = _validate_frames(
        source_frames, name="source_frames"
    )
    proposal_shape = _validate_frames(
        learned_proposal_frames, name="learned_proposal_frames"
    )
    if proposal_shape != (frame_count, height, width, channels):
        raise ValueError(
            "learned_proposal_frames must exactly match source_frames shape; "
            f"got {tuple(learned_proposal_frames.shape)} versus "
            f"{tuple(source_frames.shape)}"
        )
    if not 0.0 <= float(broad_amount) <= 1.0:
        raise ValueError("broad_amount must stay within 0..1")
    if not 0.0 <= float(mid_amount) <= 1.0:
        raise ValueError("mid_amount must stay within 0..1")
    if not 1 <= int(fine_split_radius_px) <= 8:
        raise ValueError("fine_split_radius_px must stay within 1..8")
    if not int(fine_split_radius_px) + 2 <= int(broad_radius_px) <= 64:
        raise ValueError(
            "broad_radius_px must exceed fine_split_radius_px by at least 2 and stay <=64"
        )
    if not 0.01 <= float(maximum_broad_component_delta) <= 0.50:
        raise ValueError("maximum_broad_component_delta must stay within 0.01..0.50")
    if not 0.01 <= float(maximum_mid_component_delta) <= 0.30:
        raise ValueError("maximum_mid_component_delta must stay within 0.01..0.30")
    if not 0.005 <= float(candidate_rgb_delta_cap) <= 0.20:
        raise ValueError("candidate_rgb_delta_cap must stay within 0.005..0.20")
    if not 0.0 <= float(minimum_masked_mean_abs_change) < float(
        maximum_masked_mean_abs_change
    ) <= 0.20:
        raise ValueError(
            "masked mean change limits must satisfy 0 <= minimum < maximum <= 0.20"
        )
    if not 0.0 <= float(minimum_applied_mid_rms) <= 0.10:
        raise ValueError("minimum_applied_mid_rms must stay within 0..0.10")
    if not 0.90 <= float(minimum_source_fine_cosine) <= 1.0:
        raise ValueError("minimum_source_fine_cosine must stay within 0.90..1.0")
    if not 0.50 <= float(minimum_texture_ratio) <= 1.0:
        raise ValueError("minimum_texture_ratio must stay within 0.50..1.0")
    if not 1.0 <= float(maximum_texture_ratio) <= 2.0:
        raise ValueError("maximum_texture_ratio must stay within 1.0..2.0")
    if not 0.0 <= float(maximum_new_clipped_fraction) <= 0.05:
        raise ValueError("maximum_new_clipped_fraction must stay within 0..0.05")
    if not 0.0 <= float(minimum_mask_area) < float(maximum_mask_area) <= 1.0:
        raise ValueError("mask area limits must satisfy 0 <= minimum < maximum <= 1")
    if not 1 <= int(chunk_frames) <= 32:
        raise ValueError("chunk_frames must stay within 1..32")

    mask = _normalize_mask(
        used_skin_mask,
        frame_count,
        height,
        width,
        name="used_skin_mask",
    )
    output = source_frames.detach().to(device="cpu").clone()
    effective_mask = torch.zeros_like(mask)
    rejected_mask = torch.zeros_like(mask)
    difference = torch.zeros(
        (frame_count, height, width, 3), dtype=torch.float16, device="cpu"
    )
    luma_weights = torch.tensor(
        [0.2126, 0.7152, 0.0722], dtype=torch.float32
    ).view(1, 3, 1, 1)
    reports: list[dict[str, Any]] = []
    accepted_indices: list[int] = []
    rejected_indices: list[int] = []
    unchanged_indices: list[int] = []
    outside_exact = True
    auxiliary_preserved = True

    for start in range(0, frame_count, int(chunk_frames)):
        end = min(frame_count, start + int(chunk_frames))
        source_chunk = source_frames[start:end].detach().to(device="cpu")
        proposal_chunk = learned_proposal_frames[start:end].detach().to(device="cpu")
        source_rgb = source_chunk[..., :3].float().movedim(-1, 1)
        proposal_rgb = proposal_chunk[..., :3].float().movedim(-1, 1)
        mask_chunk = mask[start:end].float()
        treatment_alpha = _mask_interior_gate(mask_chunk.unsqueeze(1), radius=2)
        mask_area = (mask_chunk > 0.10).float().mean(dim=(1, 2))

        source_fine_base = _two_pass_box_lowpass(
            source_rgb, int(fine_split_radius_px)
        )
        proposal_fine_base = _two_pass_box_lowpass(
            proposal_rgb, int(fine_split_radius_px)
        )
        source_broad = _two_pass_box_lowpass(source_rgb, int(broad_radius_px))
        proposal_broad = _two_pass_box_lowpass(proposal_rgb, int(broad_radius_px))
        source_mid = source_fine_base - source_broad
        proposal_mid = proposal_fine_base - proposal_broad
        broad_delta, broad_limiter = _direction_preserving_cap(
            proposal_broad - source_broad,
            float(maximum_broad_component_delta),
        )
        mid_delta, mid_limiter = _direction_preserving_cap(
            proposal_mid - source_mid,
            float(maximum_mid_component_delta),
        )
        requested_delta = (
            broad_delta * float(broad_amount) + mid_delta * float(mid_amount)
        ) * treatment_alpha
        applied_delta, candidate_limiter = _direction_preserving_cap(
            requested_delta, float(candidate_rgb_delta_cap)
        )
        candidate_rgb = (source_rgb + applied_delta).clamp(0.0, 1.0)
        applied_delta = candidate_rgb - source_rgb

        source_luma = (source_rgb * luma_weights).sum(dim=1, keepdim=True)
        candidate_luma = (candidate_rgb * luma_weights).sum(dim=1, keepdim=True)
        source_fine = source_luma - _two_pass_box_lowpass(
            source_luma, int(fine_split_radius_px)
        )
        candidate_fine = candidate_luma - _two_pass_box_lowpass(
            candidate_luma, int(fine_split_radius_px)
        )
        texture_weight = treatment_alpha
        source_texture = _masked_rms(source_fine, texture_weight)
        candidate_texture = _masked_rms(candidate_fine, texture_weight)
        texture_ratio = candidate_texture / source_texture.clamp_min(1.0e-8)
        fine_cosine = _masked_cosine(
            source_fine, candidate_fine, texture_weight
        )
        applied_mid_rms = _masked_rms(
            mid_delta.mean(dim=1, keepdim=True) * float(mid_amount),
            texture_weight,
        )

        absolute_delta = applied_delta.abs()
        masked_mean_change = _frame_weighted_mean(
            absolute_delta.mean(dim=1, keepdim=True), treatment_alpha
        )
        peak_change = absolute_delta.flatten(1).amax(dim=1)
        source_clipped = ((source_rgb <= 0.0) | (source_rgb >= 1.0)).any(
            dim=1, keepdim=True
        )
        candidate_clipped = ((candidate_rgb <= 0.0) | (candidate_rgb >= 1.0)).any(
            dim=1, keepdim=True
        )
        new_clipped = candidate_clipped & ~source_clipped & (treatment_alpha > 0.0)
        new_clipped_fraction = new_clipped.float().flatten(1).mean(dim=1)

        for local_index in range(end - start):
            frame_index = start + local_index
            reasons = []
            area = float(mask_area[local_index])
            mean_change = float(masked_mean_change[local_index])
            peak = float(peak_change[local_index])
            mid_rms = float(applied_mid_rms[local_index])
            cosine = float(fine_cosine[local_index])
            ratio = float(texture_ratio[local_index])
            clipped_fraction = float(new_clipped_fraction[local_index])
            if not float(minimum_mask_area) <= area <= float(maximum_mask_area):
                reasons.append("mask_area_out_of_range")
            if mean_change > float(maximum_masked_mean_abs_change):
                reasons.append("masked_mean_change_too_large")
            if peak > float(candidate_rgb_delta_cap) + 1.0e-7:
                reasons.append("peak_change_too_large")
            if cosine < float(minimum_source_fine_cosine):
                reasons.append("source_fine_phase_not_preserved")
            if not float(minimum_texture_ratio) <= ratio <= float(maximum_texture_ratio):
                reasons.append("source_texture_not_preserved")
            if clipped_fraction > float(maximum_new_clipped_fraction):
                reasons.append("new_clipping_too_large")

            changed = (
                mean_change >= float(minimum_masked_mean_abs_change)
                and mid_rms >= float(minimum_applied_mid_rms)
            )
            accepted = not reasons and changed
            if accepted:
                candidate_hwc = candidate_rgb[local_index].movedim(0, -1).to(
                    dtype=output.dtype
                )
                alpha_hw = treatment_alpha[local_index, 0]
                changed_support = absolute_delta[local_index].amax(dim=0) > 1.0e-8
                support = alpha_hw * changed_support.float()
                output[frame_index, ..., :3] = torch.where(
                    support.unsqueeze(-1) > 0.0,
                    candidate_hwc,
                    source_chunk[local_index, ..., :3],
                )
                effective_mask[frame_index] = support
                difference[frame_index] = (
                    output[frame_index, ..., :3] - source_chunk[local_index, ..., :3]
                ).abs().to(torch.float16)
                accepted_indices.append(frame_index)
                state = "ACCEPTED_FOR_IDENTITY_AND_HUMAN_REVIEW"
            elif reasons:
                rejected_mask[frame_index] = mask_chunk[local_index]
                rejected_indices.append(frame_index)
                state = "REJECTED"
            else:
                unchanged_indices.append(frame_index)
                state = "ABSTAIN_INSUFFICIENT_MID_SURFACE_EFFECT"

            reports.append(
                {
                    "frame_index": frame_index,
                    "state": state,
                    "reasons": reasons,
                    "mask_area_fraction": round(area, 8),
                    "masked_mean_abs_rgb_change": round(mean_change, 8),
                    "peak_abs_rgb_change": round(peak, 8),
                    "applied_mid_surface_rms": round(mid_rms, 8),
                    "source_fine_cosine": round(cosine, 8),
                    "source_texture_rms": round(float(source_texture[local_index]), 8),
                    "candidate_texture_rms": round(
                        float(candidate_texture[local_index]), 8
                    ),
                    "texture_ratio": round(ratio, 8),
                    "new_clipped_fraction": round(clipped_fraction, 8),
                    "broad_delta_limited_fraction": round(
                        float(
                            (broad_limiter[local_index, 0] < 1.0)
                            .float()
                            .mean()
                        ),
                        8,
                    ),
                    "mid_delta_limited_fraction": round(
                        float(
                            (mid_limiter[local_index, 0] < 1.0)
                            .float()
                            .mean()
                        ),
                        8,
                    ),
                    "candidate_delta_limited_fraction": round(
                        float(
                            (candidate_limiter[local_index, 0] < 1.0)
                            .float()
                            .mean()
                        ),
                        8,
                    ),
                }
            )

        outside = effective_mask[start:end] <= 0.0
        if not torch.equal(
            output[start:end, ..., :3][outside], source_chunk[..., :3][outside]
        ):
            outside_exact = False
        if channels > 3 and not torch.equal(
            output[start:end, ..., 3:], source_chunk[..., 3:]
        ):
            auxiliary_preserved = False

    if not outside_exact:
        raise RuntimeError("learned mid-surface prototype changed pixels outside its mask")
    if not auxiliary_preserved:
        raise RuntimeError("learned mid-surface prototype changed alpha or auxiliary channels")
    if not bool(torch.isfinite(output).all()):
        raise RuntimeError("learned mid-surface prototype produced NaN or Inf")

    status = (
        "PASS_WITH_REJECTED_FRAMES"
        if accepted_indices and rejected_indices
        else "PASS_REQUIRES_IDENTITY_AND_HUMAN_REVIEW"
        if accepted_indices
        else "ABSTAIN_ALL_FRAMES_REJECTED"
        if rejected_indices
        else "ABSTAIN_INSUFFICIENT_MID_SURFACE_EFFECT"
    )
    report = {
        "schema": SKIN_FINISH_LEARNED_MID_SURFACE_SCHEMA,
        "status": status,
        "method": "bounded_learned_broad_plus_mid_skin_surface_with_source_fine_detail",
        "learned_model_reference": GFPGAN_REFERENCE,
        "source_proxy_sha256": _tensor_proxy_sha256(source_frames),
        "proposal_proxy_sha256": _tensor_proxy_sha256(learned_proposal_frames),
        "frame_count": frame_count,
        "accepted_frame_indices": accepted_indices,
        "rejected_frame_indices": rejected_indices,
        "unchanged_frame_indices": unchanged_indices,
        "frame_reports": reports,
        "parameters": {
            "broad_amount": float(broad_amount),
            "mid_amount": float(mid_amount),
            "fine_split_radius_px": int(fine_split_radius_px),
            "broad_radius_px": int(broad_radius_px),
            "maximum_broad_component_delta": float(maximum_broad_component_delta),
            "maximum_mid_component_delta": float(maximum_mid_component_delta),
            "candidate_rgb_delta_cap": float(candidate_rgb_delta_cap),
            "minimum_masked_mean_abs_change": float(
                minimum_masked_mean_abs_change
            ),
            "maximum_masked_mean_abs_change": float(
                maximum_masked_mean_abs_change
            ),
            "minimum_applied_mid_rms": float(minimum_applied_mid_rms),
            "minimum_source_fine_cosine": float(minimum_source_fine_cosine),
            "minimum_texture_ratio": float(minimum_texture_ratio),
            "maximum_texture_ratio": float(maximum_texture_ratio),
            "maximum_new_clipped_fraction": float(maximum_new_clipped_fraction),
            "chunk_frames": int(chunk_frames),
        },
        "mechanical_gates": {
            "outside_effective_mask_bit_exact": outside_exact,
            "alpha_or_aux_channels_preserved": auxiliary_preserved,
            "source_rgb_is_pixel_carrier": True,
            "source_fine_frequency_phase_retained": True,
            "proposal_geometry_or_fine_detail_pasted": False,
            "broad_and_mid_delta_direction_preserving_caps": True,
            "candidate_rgb_delta_direction_preserving_cap": True,
            "identity_check_inside_primitive": False,
            "temporal_check_inside_primitive": False,
            "model_loading_or_cache_inside_primitive": False,
            "automatic_accept": False,
        },
        "product_boundary": (
            "This prototype transfers a generative model's broad and mid-frequency skin-surface "
            "proposal. It excludes semantic facial features and fine generated detail but may "
            "still alter local skin structure. External relative SFace, temporal-effect and human "
            "review gates are mandatory before any product claim or node registration."
        ),
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "human_review_required": bool(accepted_indices),
    }
    return output, effective_mask, rejected_mask, difference, canonical_json(report)
