from __future__ import annotations

import hashlib
import time
from typing import Any

import torch
import torch.nn.functional as torch_functional

from .skin_finish import (
    _audio_contract,
    _interrupt_and_progress,
    _memory_snapshot,
    _normalize_mask,
    _progress_bar,
    _tensor_proxy_sha256,
    _validate_frames,
    canonical_json,
)
from .skin_finish_frequency import _masked_rms, _two_pass_box_lowpass


SKIN_FINISH_SURFACE_SCHEMA = "h3_t8_skin_finish_surface/v2"
GUIDED_FILTER_REFERENCE = (
    "He, Sun and Tang, Guided Image Filtering, ECCV 2010, "
    "doi:10.1007/978-3-642-15549-9_1"
)
TONE_MAPPING_REFERENCE = (
    "Reinhard, Stark, Shirley and Ferwerda, Photographic Tone Reproduction for "
    "Digital Images, ACM TOG 2002, doi:10.1145/566654.566575"
)
FACIAL_SPECULAR_REFERENCE = (
    "Li, Lin, Zhou and Ikeuchi, Specular Highlight Removal in Facial Images, "
    "CVPR 2017"
)


def _json_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _box_mean(value: torch.Tensor, radius: int) -> torch.Tensor:
    radius = max(1, int(radius))
    kernel = radius * 2 + 1
    horizontal = torch_functional.avg_pool2d(
        torch_functional.pad(
            value,
            (radius, radius, 0, 0),
            mode="replicate",
        ),
        kernel_size=(1, kernel),
        stride=1,
    )
    return torch_functional.avg_pool2d(
        torch_functional.pad(
            horizontal,
            (0, 0, radius, radius),
            mode="replicate",
        ),
        kernel_size=(kernel, 1),
        stride=1,
    )
def _guided_rgb_base(
    rgb_nchw: torch.Tensor,
    *,
    radius: int,
    epsilon: float,
    luma_weights: torch.Tensor,
) -> torch.Tensor:
    """Edge-aware RGB base from the scalar-luma guided-filter equations.

    This is an independent tensor implementation of the local linear model from He, Sun and
    Tang (ECCV 2010). It does not use or copy CineStyle's Matchbox-derived beauty passes.
    """

    guidance = (rgb_nchw * luma_weights).sum(dim=1, keepdim=True)
    mean_guidance = _box_mean(guidance, radius)
    mean_rgb = _box_mean(rgb_nchw, radius)
    correlation_guidance = _box_mean(guidance * guidance, radius)
    correlation_guidance_rgb = _box_mean(guidance * rgb_nchw, radius)
    variance_guidance = (
        correlation_guidance - mean_guidance.square()
    ).clamp_min(0.0)
    covariance_guidance_rgb = (
        correlation_guidance_rgb - mean_guidance * mean_rgb
    )
    linear_a = covariance_guidance_rgb / (
        variance_guidance + float(epsilon)
    )
    linear_b = mean_rgb - linear_a * mean_guidance
    return _box_mean(linear_a, radius) * guidance + _box_mean(
        linear_b, radius
    )


def _smoothstep(value: torch.Tensor, start: float, end: float) -> torch.Tensor:
    normalized = ((value - float(start)) / (float(end) - float(start))).clamp(
        0.0, 1.0
    )
    return normalized.square() * (3.0 - 2.0 * normalized)


def _mask_interior_gate(mask_nchw: torch.Tensor, radius: int = 2) -> torch.Tensor:
    """Fade treatment to zero at a hard semantic-mask boundary."""

    geometry = (mask_nchw > 1.0e-5).to(dtype=torch.float32)
    support = _box_mean(geometry, radius)
    return mask_nchw * _smoothstep(support, 0.60, 1.0)


def _masked_macro_luma(
    guided_luma: torch.Tensor,
    mask_nchw: torch.Tensor,
    *,
    radius: int,
) -> torch.Tensor:
    """Estimate broad skin illumination without pulling in hair/background colors."""

    support = _box_mean(mask_nchw, radius)
    weighted = _box_mean(guided_luma * mask_nchw, radius)
    local = weighted / support.clamp_min(1.0e-4)
    return torch.where(support >= 0.05, local, guided_luma)


def finish_skin_surface(
    source_frames: torch.Tensor,
    used_skin_mask: torch.Tensor,
    *,
    amount: float = 0.65,
    surface_smoothing: float = 0.70,
    texture_keep: float = 0.85,
    highlight_compression: float = 0.65,
    broad_highlight_compression: float = 0.45,
    broad_highlight_start: float = 0.68,
    broad_highlight_end: float = 0.94,
    blemish_balance: float = 0.35,
    surface_radius_percent: float = 2.0,
    maximum_radius_px: int = 32,
    edge_epsilon: float = 0.0025,
    edge_protection_scale: float = 0.055,
    highlight_threshold: float = 0.006,
    blemish_threshold: float = 0.008,
    maximum_surface_delta: float = 0.08,
    minimum_texture_ratio: float = 0.82,
    minimum_reference_texture: float = 0.003,
    maximum_mean_abs_change: float = 0.035,
    maximum_peak_abs_change: float = 0.18,
    minimum_mask_area: float = 0.0001,
    maximum_mask_area: float = 0.50,
    maximum_new_clipped_fraction: float = 0.0005,
    clipping_epsilon: float = 1.0 / 255.0,
    chunk_frames: int = 2,
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
    """Create a source-safe, non-generative guided surface-finish candidate.

    The filter operates independently on every frame. It neither averages RGB over time nor
    reconstructs missing pores. The source remains selected unless the user explicitly accepts
    the candidate after downstream Texture Guard, Safety Audit and full-video review.
    """

    started = time.perf_counter()
    frame_count, height, width, channels = _validate_frames(
        source_frames, name="source_frames"
    )
    for name, value in {
        "amount": amount,
        "surface_smoothing": surface_smoothing,
        "texture_keep": texture_keep,
        "highlight_compression": highlight_compression,
        "broad_highlight_compression": broad_highlight_compression,
        "blemish_balance": blemish_balance,
    }.items():
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must stay within 0..1")
    if not 0.0 <= float(broad_highlight_start) < float(
        broad_highlight_end
    ) <= 1.0:
        raise ValueError(
            "broad highlight limits must satisfy 0 <= start < end <= 1"
        )
    if not 0.10 <= float(surface_radius_percent) <= 5.0:
        raise ValueError("surface_radius_percent must stay within 0.10..5.0")
    if not 1 <= int(maximum_radius_px) <= 128:
        raise ValueError("maximum_radius_px must stay within 1..128")
    if not 1.0e-6 <= float(edge_epsilon) <= 0.10:
        raise ValueError("edge_epsilon must stay within 0.000001..0.10")
    if not 0.001 <= float(edge_protection_scale) <= 0.50:
        raise ValueError("edge_protection_scale must stay within 0.001..0.50")
    if not 0.0 <= float(highlight_threshold) <= 0.10:
        raise ValueError("highlight_threshold must stay within 0..0.10")
    if not 0.0 <= float(blemish_threshold) <= 0.10:
        raise ValueError("blemish_threshold must stay within 0..0.10")
    if not 0.0 <= float(maximum_surface_delta) <= 0.25:
        raise ValueError("maximum_surface_delta must stay within 0..0.25")
    if not 0.0 <= float(minimum_texture_ratio) <= 1.0:
        raise ValueError("minimum_texture_ratio must stay within 0..1")
    if not 0.0 <= float(minimum_reference_texture) <= 0.10:
        raise ValueError("minimum_reference_texture must stay within 0..0.10")
    if not 0.0 <= float(maximum_mean_abs_change) <= 0.25:
        raise ValueError("maximum_mean_abs_change must stay within 0..0.25")
    if not 0.0 <= float(maximum_peak_abs_change) <= 1.0:
        raise ValueError("maximum_peak_abs_change must stay within 0..1")
    if not 0.0 <= float(minimum_mask_area) < float(maximum_mask_area) <= 1.0:
        raise ValueError("mask area limits must satisfy 0 <= minimum < maximum <= 1")
    if not 0.0 <= float(maximum_new_clipped_fraction) <= 0.25:
        raise ValueError("maximum_new_clipped_fraction must stay within 0..0.25")
    if not 0.0 < float(clipping_epsilon) <= 0.05:
        raise ValueError("clipping_epsilon must stay within 0..0.05")
    if not 1 <= int(chunk_frames) <= 16:
        raise ValueError("chunk_frames must stay within 1..16")

    mask = _normalize_mask(
        used_skin_mask,
        frame_count,
        height,
        width,
        name="used_skin_mask",
    )
    memory_before = _memory_snapshot()
    audio_report = _audio_contract(audio)
    source_proxy_sha = _tensor_proxy_sha256(source_frames)
    mask_proxy_sha = _tensor_proxy_sha256(mask)
    requested_radius = max(
        1,
        int(round(min(height, width) * float(surface_radius_percent) / 100.0)),
    )
    actual_radius = min(requested_radius, int(maximum_radius_px))
    chunk_size = int(chunk_frames)
    luma_weights = torch.tensor(
        [0.2126, 0.7152, 0.0722], dtype=torch.float32
    ).view(1, 3, 1, 1)
    output = torch.empty(
        tuple(source_frames.shape), dtype=source_frames.dtype, device="cpu"
    )
    effective_mask = torch.zeros_like(mask, device="cpu")
    rejected_mask = torch.zeros_like(mask, device="cpu")
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
        source_rgb = source_chunk[..., :3].float()
        source_nchw = source_rgb.movedim(-1, 1)
        mask_chunk = mask[start:end].float().clamp(0.0, 1.0)
        alpha = mask_chunk.unsqueeze(1)
        treatment_alpha = _mask_interior_gate(alpha, radius=2)
        luma = (source_nchw * luma_weights).sum(dim=1, keepdim=True)
        guided_base = _guided_rgb_base(
            source_nchw,
            radius=actual_radius,
            epsilon=float(edge_epsilon),
            luma_weights=luma_weights,
        )
        guided_luma = (guided_base * luma_weights).sum(dim=1, keepdim=True)
        detail = source_nchw - guided_base
        detail_luma = luma - guided_luma
        edge_gate = torch.exp(
            -detail_luma.abs() / float(edge_protection_scale)
        ).clamp(0.0, 1.0)
        retained_surface_fraction = 1.0 - (
            float(surface_smoothing) * (1.0 - float(texture_keep)) * edge_gate
        )
        smooth_correction = -detail * (1.0 - retained_surface_fraction)
        highlight_excess = (
            detail_luma - float(highlight_threshold)
        ).clamp_min(0.0)
        blemish_excess = (
            -detail_luma - float(blemish_threshold)
        ).clamp_min(0.0)
        bright_gate = _smoothstep(luma, 0.45, 0.92)
        midtone_gate = (
            1.0 - (luma - 0.50).abs() * 1.65
        ).clamp(0.10, 1.0)
        luma_correction = (
            blemish_excess * float(blemish_balance) * midtone_gate
            - highlight_excess * float(highlight_compression) * bright_gate
        )
        broad_highlight_gate = _smoothstep(
            guided_luma,
            float(broad_highlight_start),
            float(broad_highlight_end),
        )
        macro_radius = min(
            max(24, actual_radius * 4),
            max(24, min(height, width) // 4),
        )
        macro_luma = _masked_macro_luma(
            guided_luma,
            alpha,
            radius=macro_radius,
        )
        broad_highlight_excess = (guided_luma - macro_luma).clamp_min(0.0)
        local_gate_start = max(0.003, float(highlight_threshold) * 0.5)
        local_gate_end = max(
            local_gate_start + 0.005,
            float(edge_protection_scale),
        )
        local_highlight_gate = _smoothstep(
            broad_highlight_excess,
            local_gate_start,
            local_gate_end,
        )
        broad_luma_correction = -(
            broad_highlight_excess
            * float(broad_highlight_compression)
            * broad_highlight_gate
            * local_highlight_gate
        )
        broad_target_luma = (luma + broad_luma_correction).clamp_min(0.0)
        broad_rgb_scale = broad_target_luma / luma.clamp_min(1.0e-6)
        broad_rgb_correction = source_nchw * (broad_rgb_scale - 1.0)
        requested_correction = (
            smooth_correction
            + luma_correction.expand(-1, 3, -1, -1)
            + broad_rgb_correction
        ) * float(amount)
        requested_peak = requested_correction.abs().amax(dim=1, keepdim=True)
        cap_scale = (
            float(maximum_surface_delta) / requested_peak.clamp_min(1.0e-8)
        ).clamp(max=1.0)
        applied_correction = requested_correction * cap_scale
        candidate_nchw = (
            source_nchw + applied_correction * treatment_alpha
        ).clamp(0.0, 1.0)
        candidate_rgb = candidate_nchw.movedim(1, -1)
        composed_rgb = torch.where(
            alpha.movedim(1, -1) > 0.0,
            candidate_rgb,
            source_rgb,
        )

        treatment_mask = treatment_alpha.squeeze(1)
        mask_binary = treatment_mask > 0.10
        mask_area = mask_binary.float().mean(dim=(1, 2))
        mask_valid = (mask_area >= float(minimum_mask_area)) & (
            mask_area <= float(maximum_mask_area)
        )
        texture_weight = mask_binary.unsqueeze(1).float()
        source_texture = source_nchw - _two_pass_box_lowpass(source_nchw, 1)
        candidate_texture = candidate_nchw - _two_pass_box_lowpass(
            candidate_nchw, 1
        )
        source_texture_rms = _masked_rms(source_texture, texture_weight)
        candidate_texture_rms = _masked_rms(candidate_texture, texture_weight)
        texture_ratio = candidate_texture_rms / source_texture_rms.clamp_min(
            1.0e-8
        )
        texture_valid = (source_texture_rms < float(minimum_reference_texture)) | (
            texture_ratio >= float(minimum_texture_ratio)
        )
        masked_delta = (composed_rgb - source_rgb).abs()
        mask_pixels = mask_binary.flatten(1).sum(dim=1).clamp_min(1)
        mean_change = (
            (masked_delta * mask_binary.unsqueeze(-1)).flatten(1).sum(dim=1)
            / (mask_pixels * 3)
        )
        peak_change = (
            masked_delta.masked_fill(~mask_binary.unsqueeze(-1), 0.0)
            .flatten(1)
            .amax(dim=1)
        )
        change_valid = (mean_change <= float(maximum_mean_abs_change)) & (
            peak_change <= float(maximum_peak_abs_change)
        )
        clip_epsilon_value = float(clipping_epsilon)
        source_clipped = (
            (source_rgb <= clip_epsilon_value)
            | (source_rgb >= 1.0 - clip_epsilon_value)
        ).any(dim=-1)
        candidate_clipped = (
            (composed_rgb <= clip_epsilon_value)
            | (composed_rgb >= 1.0 - clip_epsilon_value)
        ).any(dim=-1)
        newly_clipped = candidate_clipped & ~source_clipped & mask_binary
        new_clipped_fraction = (
            newly_clipped.flatten(1).sum(dim=1).float() / mask_pixels
        )
        clipping_valid = new_clipped_fraction <= float(
            maximum_new_clipped_fraction
        )
        frame_valid = mask_valid & texture_valid & change_valid & clipping_valid

        for local_index in range(end - start):
            absolute_index = start + local_index
            reasons: list[str] = []
            if not bool(mask_valid[local_index]):
                reasons.append("mask_area_gate_failed")
            if not bool(texture_valid[local_index]):
                reasons.append("minimum_texture_ratio_failed")
            if not bool(change_valid[local_index]):
                reasons.append("surface_change_limit_failed")
            if not bool(clipping_valid[local_index]):
                reasons.append("new_clipping_limit_failed")
            if bool(frame_valid[local_index]):
                accepted_indices.append(absolute_index)
                frame_rgb = composed_rgb[local_index]
                effective_mask[absolute_index] = treatment_mask[local_index]
            else:
                rejected_indices.append(absolute_index)
                frame_rgb = source_rgb[local_index]
                rejected_mask[absolute_index] = treatment_mask[local_index]
            output[absolute_index] = source_chunk[local_index]
            output[absolute_index, ..., :3] = frame_rgb.to(
                dtype=source_chunk.dtype
            )
            difference[absolute_index] = (
                frame_rgb - source_rgb[local_index]
            ).abs().to(dtype=torch.float16)
            frame_reports.append(
                {
                    "frame_index": absolute_index,
                    "status": "PASS" if bool(frame_valid[local_index]) else "REJECT",
                    "reasons": reasons,
                    "mask_area_fraction": round(float(mask_area[local_index]), 8),
                    "source_texture_rms": round(
                        float(source_texture_rms[local_index]), 8
                    ),
                    "candidate_texture_rms": round(
                        float(candidate_texture_rms[local_index]), 8
                    ),
                    "texture_ratio": round(float(texture_ratio[local_index]), 8),
                    "mean_abs_change": round(float(mean_change[local_index]), 8),
                    "peak_abs_change": round(float(peak_change[local_index]), 8),
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
        if channels > 3 and not torch.equal(
            output_chunk[..., 3:], source_chunk[..., 3:]
        ):
            auxiliary_preserved = False
        _interrupt_and_progress(progress, end, frame_count)

    if not outside_exact:
        raise RuntimeError("Surface Finish changed pixels outside its effective mask")
    if not auxiliary_preserved:
        raise RuntimeError("Surface Finish changed alpha or auxiliary channels")
    if not bool(torch.isfinite(output).all()):
        raise RuntimeError("Surface Finish produced NaN or Inf")
    accepted = bool(accept_candidate) and bool(accepted_indices)
    selected = output if accepted else source_frames
    state = {
        "schema": SKIN_FINISH_SURFACE_SCHEMA,
        "source_proxy_sha256": source_proxy_sha,
        "mask_proxy_sha256": mask_proxy_sha,
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
        "schema": SKIN_FINISH_SURFACE_SCHEMA,
        "status": status,
        "method": "luma_guided_rgb_surface_decomposition_with_masked_local_highlight_shoulder",
        "reference": GUIDED_FILTER_REFERENCE,
        "tone_mapping_reference": TONE_MAPPING_REFERENCE,
        "facial_specular_reference": FACIAL_SPECULAR_REFERENCE,
        "clean_room_boundary": (
            "Independent guided-filter formulation; no CineStyle Matchbox-derived beauty "
            "passes, constants, weights or dependencies are used."
        ),
        "product_boundary": (
            "Non-generative display-referred SDR surface finishing with a photographic-style "
            "luminance shoulder. It is not the skin/geometry/illumination model required for "
            "physical facial-specular separation; it cannot create pores, deblur, repair "
            "identity, infer reflectance or certify natural skin."
        ),
        "parameters": {
            "amount": float(amount),
            "surface_smoothing": float(surface_smoothing),
            "texture_keep": float(texture_keep),
            "highlight_compression": float(highlight_compression),
            "broad_highlight_compression": float(broad_highlight_compression),
            "broad_highlight_start": float(broad_highlight_start),
            "broad_highlight_end": float(broad_highlight_end),
            "blemish_balance": float(blemish_balance),
            "surface_radius_percent": float(surface_radius_percent),
            "requested_radius_px": requested_radius,
            "actual_radius_px": actual_radius,
            "broad_macro_radius_px": min(
                max(24, actual_radius * 4),
                max(24, min(height, width) // 4),
            ),
            "broad_local_gate_start": max(
                0.003, float(highlight_threshold) * 0.5
            ),
            "broad_local_gate_end": max(
                max(0.003, float(highlight_threshold) * 0.5) + 0.005,
                float(edge_protection_scale),
            ),
            "mask_interior_fade_radius_px": 2,
            "maximum_radius_px": int(maximum_radius_px),
            "edge_epsilon": float(edge_epsilon),
            "edge_protection_scale": float(edge_protection_scale),
            "highlight_threshold": float(highlight_threshold),
            "blemish_threshold": float(blemish_threshold),
            "maximum_surface_delta": float(maximum_surface_delta),
            "minimum_texture_ratio": float(minimum_texture_ratio),
            "minimum_reference_texture": float(minimum_reference_texture),
            "maximum_mean_abs_change": float(maximum_mean_abs_change),
            "maximum_peak_abs_change": float(maximum_peak_abs_change),
            "minimum_mask_area": float(minimum_mask_area),
            "maximum_mask_area": float(maximum_mask_area),
            "maximum_new_clipped_fraction": float(
                maximum_new_clipped_fraction
            ),
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
            "per_frame_independent_no_rgb_temporal_average": True,
            "broad_highlight_is_local_to_masked_skin_illumination": True,
            "hard_mask_boundary_fades_inside_only": True,
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
