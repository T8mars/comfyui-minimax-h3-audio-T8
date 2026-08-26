from __future__ import annotations

import time
from typing import Any

import torch
import torch.nn.functional as torch_functional

from .skin_finish import (
    _normalize_mask,
    _tensor_proxy_sha256,
    _validate_frames,
    canonical_json,
)
from .skin_finish_frequency import _masked_rms, _two_pass_box_lowpass
from .skin_finish_surface import _mask_interior_gate, _smoothstep


SKIN_FINISH_LEARNED_SKIN_RECONSTRUCTION_SCHEMA = (
    "h3_t8_skin_finish_learned_skin_reconstruction_prototype/v1"
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


def _sobel(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    kernel_x = value.new_tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
    ).view(1, 1, 3, 3) / 8.0
    kernel_y = kernel_x.transpose(-1, -2).contiguous()
    padded = torch_functional.pad(value, (1, 1, 1, 1), mode="replicate")
    return (
        torch_functional.conv2d(padded, kernel_x),
        torch_functional.conv2d(padded, kernel_y),
    )


def _masked_gradient_cosine(
    first_x: torch.Tensor,
    first_y: torch.Tensor,
    second_x: torch.Tensor,
    second_y: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    numerator = ((first_x * second_x + first_y * second_y) * weight).flatten(1).sum(
        dim=1
    )
    first_norm = torch.sqrt(
        ((first_x.square() + first_y.square()) * weight)
        .flatten(1)
        .sum(dim=1)
        .clamp_min(1.0e-12)
    )
    second_norm = torch.sqrt(
        ((second_x.square() + second_y.square()) * weight)
        .flatten(1)
        .sum(dim=1)
        .clamp_min(1.0e-12)
    )
    return numerator / (first_norm * second_norm).clamp_min(1.0e-12)


def fuse_bounded_semantic_skin_reconstruction(
    source_frames: torch.Tensor,
    learned_proposal_frames: torch.Tensor,
    used_skin_mask: torch.Tensor,
    *,
    amount: float = 0.70,
    proposal_prefilter_radius_px: int = 1,
    maximum_proposal_component_delta: float = 0.25,
    candidate_rgb_delta_cap: float = 0.12,
    flat_edge_low: float = 0.012,
    flat_edge_high: float = 0.035,
    aligned_edge_cosine_low: float = 0.70,
    aligned_edge_cosine_high: float = 0.90,
    minimum_structural_gradient_cosine: float = 0.92,
    minimum_masked_mean_abs_change: float = 0.025,
    maximum_masked_mean_abs_change: float = 0.080,
    minimum_texture_ratio: float = 0.55,
    maximum_texture_ratio: float = 1.30,
    maximum_new_clipped_fraction: float = 0.002,
    minimum_mask_area: float = 0.0001,
    maximum_mask_area: float = 0.50,
    chunk_frames: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]:
    """Blend bounded learned RGB only on semantically approved, structure-safe skin.

    This route intentionally permits a generative proposal to change skin texture. It is more
    capable and riskier than the source-phase prototypes. The caller must supply a semantic skin
    mask that already excludes facial features, and must run external identity, temporal and human
    review gates. This unregistered primitive contains no model loading, network access or cache.
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
    if not 0.0 <= float(amount) <= 1.0:
        raise ValueError("amount must stay within 0..1")
    if not 1 <= int(proposal_prefilter_radius_px) <= 4:
        raise ValueError("proposal_prefilter_radius_px must stay within 1..4")
    if not 0.05 <= float(maximum_proposal_component_delta) <= 0.50:
        raise ValueError("maximum_proposal_component_delta must stay within 0.05..0.50")
    if not 0.01 <= float(candidate_rgb_delta_cap) <= 0.20:
        raise ValueError("candidate_rgb_delta_cap must stay within 0.01..0.20")
    if not 0.0 <= float(flat_edge_low) < float(flat_edge_high) <= 0.20:
        raise ValueError("flat edge limits must satisfy 0 <= low < high <= 0.20")
    if not -1.0 <= float(aligned_edge_cosine_low) < float(
        aligned_edge_cosine_high
    ) <= 1.0:
        raise ValueError("aligned edge cosine limits must satisfy -1 <= low < high <= 1")
    if not 0.50 <= float(minimum_structural_gradient_cosine) <= 1.0:
        raise ValueError("minimum_structural_gradient_cosine must stay within 0.50..1.0")
    if not 0.0 <= float(minimum_masked_mean_abs_change) < float(
        maximum_masked_mean_abs_change
    ) <= 0.20:
        raise ValueError(
            "masked mean change limits must satisfy 0 <= minimum < maximum <= 0.20"
        )
    if not 0.20 <= float(minimum_texture_ratio) <= 1.0:
        raise ValueError("minimum_texture_ratio must stay within 0.20..1.0")
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
        proposal_prepared = _two_pass_box_lowpass(
            proposal_rgb, int(proposal_prefilter_radius_px)
        )
        mask_chunk = mask[start:end].float()
        treatment_alpha = _mask_interior_gate(mask_chunk.unsqueeze(1), radius=2)
        mask_area = (mask_chunk > 0.10).float().mean(dim=(1, 2))

        source_luma = (source_rgb * luma_weights).sum(dim=1, keepdim=True)
        proposal_luma = (proposal_prepared * luma_weights).sum(dim=1, keepdim=True)
        source_gx, source_gy = _sobel(source_luma)
        proposal_gx, proposal_gy = _sobel(proposal_luma)
        source_magnitude = torch.sqrt(source_gx.square() + source_gy.square() + 1.0e-12)
        proposal_magnitude = torch.sqrt(
            proposal_gx.square() + proposal_gy.square() + 1.0e-12
        )
        local_gradient_cosine = (
            source_gx * proposal_gx + source_gy * proposal_gy
        ) / (source_magnitude * proposal_magnitude).clamp_min(1.0e-8)
        source_flat_support = 1.0 - _smoothstep(
            source_magnitude, float(flat_edge_low), float(flat_edge_high)
        )
        proposal_flat_support = 1.0 - _smoothstep(
            proposal_magnitude, float(flat_edge_low), float(flat_edge_high)
        )
        flat_support = source_flat_support * proposal_flat_support
        orientation_support = _smoothstep(
            local_gradient_cosine,
            float(aligned_edge_cosine_low),
            float(aligned_edge_cosine_high),
        )
        magnitude_ratio = proposal_magnitude / source_magnitude.clamp_min(1.0e-6)
        magnitude_support = (
            _smoothstep(magnitude_ratio, 0.35, 0.65)
            * (1.0 - _smoothstep(magnitude_ratio, 1.8, 2.8))
        )
        local_structure_support = torch.maximum(
            flat_support, orientation_support * magnitude_support
        )
        structure_risk = torch_functional.max_pool2d(
            1.0 - local_structure_support,
            kernel_size=5,
            stride=1,
            padding=2,
        )
        structure_support = 1.0 - structure_risk
        alpha = treatment_alpha * structure_support

        raw_delta, proposal_limiter = _direction_preserving_cap(
            proposal_prepared - source_rgb,
            float(maximum_proposal_component_delta),
        )
        requested_delta = raw_delta * float(amount) * alpha
        applied_delta, candidate_limiter = _direction_preserving_cap(
            requested_delta, float(candidate_rgb_delta_cap)
        )
        candidate_rgb = (source_rgb + applied_delta).clamp(0.0, 1.0)
        applied_delta = candidate_rgb - source_rgb

        candidate_luma = (candidate_rgb * luma_weights).sum(dim=1, keepdim=True)
        candidate_gx, candidate_gy = _sobel(candidate_luma)
        structural_weight = treatment_alpha * _smoothstep(
            source_magnitude, float(flat_edge_low), float(flat_edge_high)
        )
        gradient_cosine = _masked_gradient_cosine(
            source_gx,
            source_gy,
            candidate_gx,
            candidate_gy,
            structural_weight,
        )
        source_detail = source_luma - _two_pass_box_lowpass(source_luma, 2)
        candidate_detail = candidate_luma - _two_pass_box_lowpass(candidate_luma, 2)
        source_texture = _masked_rms(source_detail, treatment_alpha)
        candidate_texture = _masked_rms(candidate_detail, treatment_alpha)
        texture_ratio = candidate_texture / source_texture.clamp_min(1.0e-8)

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
            edge_cosine = float(gradient_cosine[local_index])
            ratio = float(texture_ratio[local_index])
            clipped_fraction = float(new_clipped_fraction[local_index])
            if not float(minimum_mask_area) <= area <= float(maximum_mask_area):
                reasons.append("mask_area_out_of_range")
            if mean_change > float(maximum_masked_mean_abs_change):
                reasons.append("masked_mean_change_too_large")
            if peak > float(candidate_rgb_delta_cap) + 1.0e-7:
                reasons.append("peak_change_too_large")
            if edge_cosine < float(minimum_structural_gradient_cosine):
                reasons.append("structural_gradient_cosine_too_low")
            if not float(minimum_texture_ratio) <= ratio <= float(maximum_texture_ratio):
                reasons.append("texture_energy_out_of_range")
            if clipped_fraction > float(maximum_new_clipped_fraction):
                reasons.append("new_clipping_too_large")

            changed = mean_change >= float(minimum_masked_mean_abs_change)
            accepted = not reasons and changed
            if accepted:
                candidate_hwc = candidate_rgb[local_index].movedim(0, -1).to(
                    dtype=output.dtype
                )
                support = alpha[local_index, 0] * (
                    absolute_delta[local_index].amax(dim=0) > 1.0e-8
                ).float()
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
                state = "ABSTAIN_INSUFFICIENT_VISIBLE_CHANGE"

            reports.append(
                {
                    "frame_index": frame_index,
                    "state": state,
                    "reasons": reasons,
                    "mask_area_fraction": round(area, 8),
                    "mean_structure_support": round(
                        float(structure_support[local_index].mean()), 8
                    ),
                    "masked_mean_abs_rgb_change": round(mean_change, 8),
                    "peak_abs_rgb_change": round(peak, 8),
                    "structural_gradient_cosine": round(edge_cosine, 8),
                    "source_texture_rms": round(float(source_texture[local_index]), 8),
                    "candidate_texture_rms": round(
                        float(candidate_texture[local_index]), 8
                    ),
                    "texture_ratio": round(ratio, 8),
                    "new_clipped_fraction": round(clipped_fraction, 8),
                    "proposal_delta_limited_fraction": round(
                        float(
                            (proposal_limiter[local_index, 0] < 1.0)
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
        raise RuntimeError("learned skin reconstruction changed pixels outside its mask")
    if not auxiliary_preserved:
        raise RuntimeError("learned skin reconstruction changed alpha or auxiliary channels")
    if not bool(torch.isfinite(output).all()):
        raise RuntimeError("learned skin reconstruction produced NaN or Inf")

    status = (
        "PASS_WITH_REJECTED_FRAMES"
        if accepted_indices and rejected_indices
        else "PASS_REQUIRES_IDENTITY_AND_HUMAN_REVIEW"
        if accepted_indices
        else "ABSTAIN_ALL_FRAMES_REJECTED"
        if rejected_indices
        else "ABSTAIN_INSUFFICIENT_VISIBLE_CHANGE"
    )
    report = {
        "schema": SKIN_FINISH_LEARNED_SKIN_RECONSTRUCTION_SCHEMA,
        "status": status,
        "method": "bounded_semantic_skin_only_learned_rgb_reconstruction",
        "learned_model_reference": GFPGAN_REFERENCE,
        "source_proxy_sha256": _tensor_proxy_sha256(source_frames),
        "proposal_proxy_sha256": _tensor_proxy_sha256(learned_proposal_frames),
        "frame_count": frame_count,
        "accepted_frame_indices": accepted_indices,
        "rejected_frame_indices": rejected_indices,
        "unchanged_frame_indices": unchanged_indices,
        "frame_reports": reports,
        "parameters": {
            "amount": float(amount),
            "proposal_prefilter_radius_px": int(proposal_prefilter_radius_px),
            "maximum_proposal_component_delta": float(
                maximum_proposal_component_delta
            ),
            "candidate_rgb_delta_cap": float(candidate_rgb_delta_cap),
            "flat_edge_low": float(flat_edge_low),
            "flat_edge_high": float(flat_edge_high),
            "aligned_edge_cosine_low": float(aligned_edge_cosine_low),
            "aligned_edge_cosine_high": float(aligned_edge_cosine_high),
            "minimum_structural_gradient_cosine": float(
                minimum_structural_gradient_cosine
            ),
            "minimum_masked_mean_abs_change": float(
                minimum_masked_mean_abs_change
            ),
            "maximum_masked_mean_abs_change": float(
                maximum_masked_mean_abs_change
            ),
            "minimum_texture_ratio": float(minimum_texture_ratio),
            "maximum_texture_ratio": float(maximum_texture_ratio),
            "maximum_new_clipped_fraction": float(maximum_new_clipped_fraction),
            "chunk_frames": int(chunk_frames),
        },
        "mechanical_gates": {
            "outside_effective_mask_bit_exact": outside_exact,
            "alpha_or_aux_channels_preserved": auxiliary_preserved,
            "semantic_skin_mask_required": True,
            "structural_edge_direction_gate": True,
            "proposal_prefiltered_before_transfer": True,
            "proposal_and_candidate_direction_preserving_caps": True,
            "proposal_generated_skin_rgb_can_be_transferred": True,
            "identity_check_inside_primitive": False,
            "temporal_check_inside_primitive": False,
            "model_loading_or_cache_inside_primitive": False,
            "automatic_accept": False,
        },
        "risk_boundary": (
            "Unlike the source-phase prototypes, this route can transfer generated skin RGB and "
            "texture. Semantic feature exclusion, edge consistency and RGB caps reduce but do "
            "not remove hallucination, identity or temporal-flicker risk. It must remain EXP and "
            "source-selected until external identity, temporal and human review pass."
        ),
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "human_review_required": bool(accepted_indices),
    }
    return output, effective_mask, rejected_mask, difference, canonical_json(report)
