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
from .skin_finish_dichromatic import _linear_to_srgb, _srgb_to_linear
from .skin_finish_frequency import _masked_rms, _two_pass_box_lowpass
from .skin_finish_surface import _mask_interior_gate, _smoothstep


SKIN_FINISH_LEARNED_DETAIL_SCHEMA = "h3_t8_skin_finish_learned_detail_prototype/v1"
GFPGAN_REFERENCE = (
    "Wang, Li, Zhang and Shan, Towards Real-World Blind Face Restoration with "
    "Generative Facial Prior, CVPR 2021"
)


def _frame_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    numerator = (value * weight).flatten(1).sum(dim=1)
    denominator = weight.flatten(1).sum(dim=1).clamp_min(1.0)
    return numerator / denominator


def fuse_proposal_guided_skin_detail(
    source_frames: torch.Tensor,
    learned_proposal_frames: torch.Tensor,
    used_skin_mask: torch.Tensor,
    *,
    amount: float = 0.70,
    surface_amount: float = 0.45,
    surface_radius_px: int = 10,
    maximum_surface_mismatch: float = 0.12,
    maximum_surface_luma_delta: float = 0.035,
    chroma_amount: float = 0.20,
    maximum_chroma_component_delta: float = 0.04,
    candidate_rgb_delta_cap: float = 0.10,
    detail_radius_px: int = 2,
    energy_radius_px: int = 5,
    maximum_detail_gain: float = 1.80,
    maximum_linear_luma_delta: float = 0.025,
    low_frequency_tolerance: float = 0.025,
    minimum_source_detail_rms: float = 0.00075,
    minimum_source_luma: float = 0.005,
    maximum_texture_ratio: float = 1.45,
    maximum_mean_abs_change: float = 0.025,
    maximum_peak_abs_change: float = 0.12,
    minimum_mask_area: float = 0.0001,
    maximum_mask_area: float = 0.50,
    chunk_frames: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]:
    """Use a learned restoration only as a local detail-energy proposal.

    The proposal's RGB, geometry and high-frequency phase are never pasted into the source. Its
    zero-mean, bounded low-frequency luminance trend may shape the skin surface, while local
    luminance-detail energy only determines how much of the *source's own* detail is amplified.
    A small bounded low-frequency chromaticity correction is optional. This remains intentionally
    less powerful than full-face GFPGAN restoration and much less likely to rewrite identity.

    This is a calibration primitive, not a registered ComfyUI node.  It deliberately contains no
    model loading, network access, cache or automatic candidate acceptance.
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
    if not 0.0 <= float(surface_amount) <= 1.0:
        raise ValueError("surface_amount must stay within 0..1")
    if not 2 <= int(surface_radius_px) <= 32:
        raise ValueError("surface_radius_px must stay within 2..32")
    if not 0.02 <= float(maximum_surface_mismatch) <= 0.30:
        raise ValueError("maximum_surface_mismatch must stay within 0.02..0.30")
    if not 0.0 <= float(maximum_surface_luma_delta) <= 0.08:
        raise ValueError("maximum_surface_luma_delta must stay within 0..0.08")
    if not 0.0 <= float(chroma_amount) <= 0.50:
        raise ValueError("chroma_amount must stay within 0..0.50")
    if not 0.0 <= float(maximum_chroma_component_delta) <= 0.10:
        raise ValueError("maximum_chroma_component_delta must stay within 0..0.10")
    if not 0.0 <= float(candidate_rgb_delta_cap) <= 0.20:
        raise ValueError("candidate_rgb_delta_cap must stay within 0..0.20")
    if not 1 <= int(detail_radius_px) <= 8:
        raise ValueError("detail_radius_px must stay within 1..8")
    if not 1 <= int(energy_radius_px) <= 32:
        raise ValueError("energy_radius_px must stay within 1..32")
    if not 1.0 <= float(maximum_detail_gain) <= 3.0:
        raise ValueError("maximum_detail_gain must stay within 1..3")
    if not 0.0 <= float(maximum_linear_luma_delta) <= 0.08:
        raise ValueError("maximum_linear_luma_delta must stay within 0..0.08")
    if not 0.001 <= float(low_frequency_tolerance) <= 0.20:
        raise ValueError("low_frequency_tolerance must stay within 0.001..0.20")
    if not 1.0e-6 <= float(minimum_source_detail_rms) <= 0.05:
        raise ValueError("minimum_source_detail_rms must stay within 1e-6..0.05")
    if not 0.0 <= float(minimum_source_luma) <= 0.05:
        raise ValueError("minimum_source_luma must stay within 0..0.05")
    if not 1.0 <= float(maximum_texture_ratio) <= 2.0:
        raise ValueError("maximum_texture_ratio must stay within 1..2")
    if not 0.0 <= float(maximum_mean_abs_change) <= 0.08:
        raise ValueError("maximum_mean_abs_change must stay within 0..0.08")
    if not 0.0 <= float(maximum_peak_abs_change) <= 0.30:
        raise ValueError("maximum_peak_abs_change must stay within 0..0.30")
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
    outside_exact = True
    auxiliary_preserved = True

    for start in range(0, frame_count, int(chunk_frames)):
        end = min(frame_count, start + int(chunk_frames))
        source_chunk = source_frames[start:end].detach().to(device="cpu")
        proposal_chunk = learned_proposal_frames[start:end].detach().to(device="cpu")
        source_rgb = source_chunk[..., :3].float()
        proposal_rgb = proposal_chunk[..., :3].float()
        mask_chunk = mask[start:end].float()
        treatment_alpha = _mask_interior_gate(mask_chunk.unsqueeze(1), radius=2)
        mask_binary = mask_chunk > 0.10
        mask_area = mask_binary.float().mean(dim=(1, 2))
        mask_valid = (mask_area >= float(minimum_mask_area)) & (
            mask_area <= float(maximum_mask_area)
        )

        source_linear = _srgb_to_linear(source_rgb.movedim(-1, 1))
        proposal_linear = _srgb_to_linear(proposal_rgb.movedim(-1, 1))
        source_luma = (source_linear * luma_weights).sum(dim=1, keepdim=True)
        proposal_luma = (proposal_linear * luma_weights).sum(dim=1, keepdim=True)
        source_low = _two_pass_box_lowpass(source_luma, int(detail_radius_px))
        proposal_low = _two_pass_box_lowpass(proposal_luma, int(detail_radius_px))
        source_detail = source_luma - source_low
        proposal_detail = proposal_luma - proposal_low
        source_energy = torch.sqrt(
            _two_pass_box_lowpass(source_detail.square(), int(energy_radius_px))
            + 1.0e-12
        )
        proposal_energy = torch.sqrt(
            _two_pass_box_lowpass(proposal_detail.square(), int(energy_radius_px))
            + 1.0e-12
        )

        desired_gain = (
            proposal_energy / source_energy.clamp_min(float(minimum_source_detail_rms))
        ).clamp(1.0, float(maximum_detail_gain))
        low_mismatch = (proposal_low - source_low).abs()
        consistency = 1.0 - _smoothstep(
            low_mismatch,
            float(low_frequency_tolerance),
            float(low_frequency_tolerance) * 3.0,
        )
        detail_support = _smoothstep(
            source_energy,
            float(minimum_source_detail_rms),
            float(minimum_source_detail_rms) * 4.0,
        )
        luminance_support = _smoothstep(
            source_luma,
            float(minimum_source_luma),
            max(float(minimum_source_luma) * 4.0, 0.02),
        )
        confidence = treatment_alpha * consistency * detail_support * luminance_support
        gain = 1.0 + (desired_gain - 1.0) * float(amount) * confidence
        detail_luma_delta = (source_detail * (gain - 1.0)).clamp(
            -float(maximum_linear_luma_delta),
            float(maximum_linear_luma_delta),
        )
        source_surface = _two_pass_box_lowpass(source_luma, int(surface_radius_px))
        proposal_surface = _two_pass_box_lowpass(
            proposal_luma, int(surface_radius_px)
        )
        raw_surface_delta = proposal_surface - source_surface
        surface_weight_sum = treatment_alpha.flatten(1).sum(dim=1).clamp_min(1.0)
        surface_mean = (
            (raw_surface_delta * treatment_alpha).flatten(1).sum(dim=1)
            / surface_weight_sum
        ).view(-1, 1, 1, 1)
        centered_surface_delta = raw_surface_delta - surface_mean
        surface_confidence = 1.0 - _smoothstep(
            centered_surface_delta.abs(),
            float(maximum_surface_mismatch) * 0.5,
            float(maximum_surface_mismatch),
        )
        surface_alpha = treatment_alpha * surface_confidence
        surface_luma_delta = (
            centered_surface_delta
            * float(surface_amount)
            * surface_alpha
        ).clamp(
            -float(maximum_surface_luma_delta),
            float(maximum_surface_luma_delta),
        )
        total_luma_delta = surface_luma_delta + detail_luma_delta
        target_luma = (source_luma + total_luma_delta).clamp_min(0.0)

        source_surface_rgb = _two_pass_box_lowpass(
            source_linear, int(surface_radius_px)
        )
        proposal_surface_rgb = _two_pass_box_lowpass(
            proposal_linear, int(surface_radius_px)
        )
        source_surface_chroma = source_surface_rgb / source_surface_rgb.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0e-8)
        proposal_chroma = proposal_surface_rgb / proposal_surface_rgb.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0e-8)
        chroma_delta = (proposal_chroma - source_surface_chroma).clamp(
            -float(maximum_chroma_component_delta),
            float(maximum_chroma_component_delta),
        )
        source_pixel_chroma = source_linear / source_linear.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0e-8)
        raw_target_chroma = (
            source_pixel_chroma
            + chroma_delta * float(chroma_amount) * surface_alpha
        ).clamp_min(1.0e-6)
        raw_target_chroma = raw_target_chroma / raw_target_chroma.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0e-8)
        raw_applied_chroma_delta = raw_target_chroma - source_pixel_chroma
        applied_limit = float(chroma_amount) * float(
            maximum_chroma_component_delta
        )
        chroma_limiter = (
            torch.full_like(raw_applied_chroma_delta[:, :1], applied_limit)
            / raw_applied_chroma_delta.abs().amax(dim=1, keepdim=True).clamp_min(
                1.0e-8
            )
        ).clamp_max(1.0)
        target_chroma = (
            source_pixel_chroma + raw_applied_chroma_delta * chroma_limiter
        )
        target_chroma_luma = (target_chroma * luma_weights).sum(
            dim=1, keepdim=True
        )
        maximum_target_luma = target_chroma_luma / target_chroma.amax(
            dim=1, keepdim=True
        ).clamp_min(1.0e-8)
        target_luma = torch.minimum(target_luma, maximum_target_luma)
        candidate_linear = (
            target_chroma
            * target_luma
            / target_chroma_luma.clamp_min(1.0e-8)
        )
        raw_candidate_rgb = _linear_to_srgb(candidate_linear).movedim(1, -1)
        raw_rgb_delta = raw_candidate_rgb - source_rgb
        raw_pixel_peak = raw_rgb_delta.abs().amax(dim=-1, keepdim=True)
        rgb_delta_limiter = (
            torch.full_like(raw_pixel_peak, float(candidate_rgb_delta_cap))
            / raw_pixel_peak.clamp_min(1.0e-8)
        ).clamp_max(1.0)
        candidate_rgb = source_rgb + raw_rgb_delta * rgb_delta_limiter

        equal_frames = torch.eq(source_chunk, proposal_chunk).flatten(1).all(dim=1)
        if (
            float(amount) == 0.0
            and float(surface_amount) == 0.0
            and float(chroma_amount) == 0.0
        ):
            candidate_rgb = source_rgb
        elif bool(equal_frames.any()):
            candidate_rgb[equal_frames] = source_rgb[equal_frames]
        active_alpha = torch.maximum(confidence, surface_alpha)
        candidate_rgb = torch.where(
            (active_alpha > 0.0).movedim(1, -1), candidate_rgb, source_rgb
        )

        source_output_detail = source_luma - _two_pass_box_lowpass(
            source_luma, int(detail_radius_px)
        )
        candidate_output_luma = (
            _srgb_to_linear(candidate_rgb.movedim(-1, 1)) * luma_weights
        ).sum(dim=1, keepdim=True)
        candidate_output_detail = candidate_output_luma - _two_pass_box_lowpass(
            candidate_output_luma, int(detail_radius_px)
        )
        texture_weight = mask_binary.unsqueeze(1).float()
        source_rms = _masked_rms(source_output_detail, texture_weight)
        proposal_rms = _masked_rms(proposal_detail, texture_weight)
        candidate_rms = _masked_rms(candidate_output_detail, texture_weight)
        texture_ratio = candidate_rms / source_rms.clamp_min(1.0e-8)
        delta = (candidate_rgb - source_rgb).abs()
        mean_change = _frame_mean(delta.mean(dim=-1), mask_binary.float())
        peak_change = delta.masked_fill(~mask_binary.unsqueeze(-1), 0.0).flatten(1).amax(dim=1)
        finite_valid = torch.isfinite(candidate_rgb).flatten(1).all(dim=1)
        texture_valid = texture_ratio <= float(maximum_texture_ratio)
        change_valid = (mean_change <= float(maximum_mean_abs_change)) & (
            peak_change <= float(maximum_peak_abs_change)
        )
        frame_valid = mask_valid & finite_valid & texture_valid & change_valid

        for local_index in range(end - start):
            absolute_index = start + local_index
            reasons: list[str] = []
            if not bool(mask_valid[local_index]):
                reasons.append("mask_area_gate_failed")
            if not bool(finite_valid[local_index]):
                reasons.append("finite_gate_failed")
            if not bool(texture_valid[local_index]):
                reasons.append("texture_ratio_gate_failed")
            if not bool(change_valid[local_index]):
                reasons.append("change_gate_failed")
            if bool(frame_valid[local_index]):
                frame_rgb = candidate_rgb[local_index]
                effective_mask[absolute_index] = active_alpha[local_index, 0]
                accepted_indices.append(absolute_index)
            else:
                frame_rgb = source_rgb[local_index]
                rejected_mask[absolute_index] = mask_chunk[local_index]
                rejected_indices.append(absolute_index)
            output[absolute_index, ..., :3] = frame_rgb.to(source_chunk.dtype)
            difference[absolute_index] = (
                frame_rgb - source_rgb[local_index]
            ).abs().to(torch.float16)
            reports.append(
                {
                    "frame_index": absolute_index,
                    "status": "PASS" if bool(frame_valid[local_index]) else "REJECT",
                    "reasons": reasons,
                    "mask_area_fraction": round(float(mask_area[local_index]), 8),
                    "source_detail_rms": round(float(source_rms[local_index]), 8),
                    "proposal_detail_rms": round(float(proposal_rms[local_index]), 8),
                    "candidate_detail_rms": round(float(candidate_rms[local_index]), 8),
                    "texture_ratio": round(float(texture_ratio[local_index]), 8),
                    "mean_abs_change": round(float(mean_change[local_index]), 8),
                    "peak_abs_change": round(float(peak_change[local_index]), 8),
                    "mean_confidence": round(float(confidence[local_index].mean()), 8),
                    "mean_abs_surface_luma_delta": round(
                        float(surface_luma_delta[local_index].abs().mean()), 8
                    ),
                    "mean_abs_applied_chroma_component_delta": round(
                        float(
                            (
                                target_chroma[local_index]
                                - source_pixel_chroma[local_index]
                            ).abs().mean()
                        ),
                        8,
                    ),
                    "maximum_abs_applied_chroma_component_delta": round(
                        float(
                            (
                                target_chroma[local_index]
                                - source_pixel_chroma[local_index]
                            ).abs().amax()
                        ),
                        8,
                    ),
                    "raw_peak_abs_rgb_change_before_limit": round(
                        float(raw_pixel_peak[local_index].amax()), 8
                    ),
                    "rgb_delta_limited_pixel_fraction": round(
                        float(
                            (rgb_delta_limiter[local_index, ..., 0] < 1.0)
                            .float()
                            .mean()
                        ),
                        8,
                    ),
                    "maximum_requested_gain": round(float(desired_gain[local_index].amax()), 8),
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
        raise RuntimeError("learned-detail prototype changed pixels outside its mask")
    if not auxiliary_preserved:
        raise RuntimeError("learned-detail prototype changed alpha or auxiliary channels")
    if not bool(torch.isfinite(output).all()):
        raise RuntimeError("learned-detail prototype produced NaN or Inf")

    report = {
        "schema": SKIN_FINISH_LEARNED_DETAIL_SCHEMA,
        "status": (
            "PASS_WITH_REJECTED_FRAMES"
            if accepted_indices and rejected_indices
            else "PASS"
            if accepted_indices
            else "ABSTAIN_ALL_FRAMES_REJECTED"
        ),
        "method": "learned_proposal_guided_bounded_skin_surface_and_source_phase_detail",
        "learned_model_reference": GFPGAN_REFERENCE,
        "source_proxy_sha256": _tensor_proxy_sha256(source_frames),
        "proposal_proxy_sha256": _tensor_proxy_sha256(learned_proposal_frames),
        "frame_count": frame_count,
        "accepted_frame_indices": accepted_indices,
        "rejected_frame_indices": rejected_indices,
        "frame_reports": reports,
        "parameters": {
            "amount": float(amount),
            "surface_amount": float(surface_amount),
            "surface_radius_px": int(surface_radius_px),
            "maximum_surface_mismatch": float(maximum_surface_mismatch),
            "maximum_surface_luma_delta": float(maximum_surface_luma_delta),
            "chroma_amount": float(chroma_amount),
            "maximum_chroma_component_delta": float(
                maximum_chroma_component_delta
            ),
            "candidate_rgb_delta_cap": float(candidate_rgb_delta_cap),
            "detail_radius_px": int(detail_radius_px),
            "energy_radius_px": int(energy_radius_px),
            "maximum_detail_gain": float(maximum_detail_gain),
            "maximum_linear_luma_delta": float(maximum_linear_luma_delta),
            "low_frequency_tolerance": float(low_frequency_tolerance),
            "minimum_source_detail_rms": float(minimum_source_detail_rms),
            "minimum_source_luma": float(minimum_source_luma),
            "maximum_texture_ratio": float(maximum_texture_ratio),
            "maximum_mean_abs_change": float(maximum_mean_abs_change),
            "maximum_peak_abs_change": float(maximum_peak_abs_change),
            "chunk_frames": int(chunk_frames),
        },
        "mechanical_gates": {
            "outside_effective_mask_bit_exact": outside_exact,
            "alpha_or_aux_channels_preserved": auxiliary_preserved,
            "source_high_frequency_phase_preserved": True,
            "proposal_low_frequency_chroma_centered_and_bounded": True,
            "proposal_low_frequency_luma_centered_and_bounded": True,
            "proposal_rgb_or_detail_phase_pasted": False,
            "candidate_rgb_delta_direction_preserving_cap": True,
            "frame_independent_no_rgb_temporal_average": True,
            "model_loading_or_cache_inside_primitive": False,
            "automatic_accept": False,
        },
        "product_boundary": (
            "GFPGAN is a generative face-restoration prior and may hallucinate identity details. "
            "This prototype therefore uses only a zero-mean bounded low-frequency luminance "
            "trend, bounded low-frequency chromaticity, and local detail energy to scale the "
            "source's own detail phase. It does not paste restored RGB, geometry or generated "
            "detail phase; it does not prove "
            "pores, identity fidelity, temporal stability or aesthetic benefit."
        ),
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "human_review_required": bool(accepted_indices),
    }
    return output, effective_mask, rejected_mask, difference, canonical_json(report)
