from __future__ import annotations

import hashlib
import time
from typing import Any

import torch

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
from .skin_finish_surface import _box_mean, _mask_interior_gate, _smoothstep


SKIN_FINISH_DICHROMATIC_SCHEMA = "h3_t8_skin_finish_dichromatic/v1"
DICHROMATIC_REFERENCE = (
    "Shafer, Using Color to Separate Reflection Components, Color Research and "
    "Application 1985, doi:10.1002/col.5080100409"
)
FACIAL_SPECULAR_REFERENCE = (
    "Li, Lin, Zhou and Ikeuchi, Specular Highlight Removal in Facial Images, "
    "CVPR 2017"
)


def _json_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _srgb_to_linear(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp(0.0, 1.0)
    return torch.where(
        value <= 0.04045,
        value / 12.92,
        ((value + 0.055) / 1.055).pow(2.4),
    )


def _linear_to_srgb(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp(0.0, 1.0)
    return torch.where(
        value <= 0.0031308,
        value * 12.92,
        1.055 * value.pow(1.0 / 2.4) - 0.055,
    )


def _masked_diffuse_chromaticity(
    linear_rgb: torch.Tensor,
    mask_nchw: torch.Tensor,
    *,
    radius: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate local diffuse chromaticity while down-weighting neutral highlights.

    The returned chromaticity sums to one. The confidence is deliberately low when the local
    colour is nearly neutral, because a neutral illuminant cannot be separated reliably from a
    neutral diffuse colour under the two-component model.
    """

    intensity = linear_rgb.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
    chromaticity = linear_rgb / intensity
    neutral = torch.full_like(chromaticity, 1.0 / 3.0)
    chroma_distance = (chromaticity - neutral).square().sum(
        dim=1, keepdim=True
    ).sqrt()
    chroma_weight = 0.10 + 0.90 * _smoothstep(chroma_distance, 0.008, 0.080)
    weight = mask_nchw * chroma_weight
    support = _box_mean(weight, radius)
    local_rgb = _box_mean(linear_rgb * weight, radius) / support.clamp_min(1.0e-5)
    local_rgb = torch.where(support >= 0.02, local_rgb, linear_rgb)
    diffuse = local_rgb / local_rgb.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
    diffuse_distance = (diffuse - neutral).square().sum(
        dim=1, keepdim=True
    ).sqrt()
    return diffuse, diffuse_distance


def _dichromatic_specular_estimate(
    linear_rgb: torch.Tensor,
    diffuse_chromaticity: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Solve p = diffuse_scale * d + specular * [1, 1, 1] per pixel.

    This is a bounded least-squares estimate under a neutral-illuminant dichromatic model. It is
    not a physical inverse-rendering solution; geometry, calibrated illumination and a skin model
    are intentionally absent.
    """

    diffuse_norm2 = diffuse_chromaticity.square().sum(dim=1, keepdim=True)
    determinant = (3.0 * diffuse_norm2 - 1.0).clamp_min(1.0e-5)
    pixel_dot_diffuse = (linear_rgb * diffuse_chromaticity).sum(
        dim=1, keepdim=True
    )
    pixel_sum = linear_rgb.sum(dim=1, keepdim=True)
    specular = (
        pixel_sum * diffuse_norm2 - pixel_dot_diffuse
    ) / determinant
    specular = specular.clamp_min(0.0)
    specular = torch.minimum(specular, linear_rgb.amin(dim=1, keepdim=True))

    neutral = torch.full_like(diffuse_chromaticity, 1.0 / 3.0)
    pixel_chromaticity = linear_rgb / pixel_sum.clamp_min(1.0e-6)
    diffuse_vector = diffuse_chromaticity - neutral
    pixel_vector = pixel_chromaticity - neutral
    diffuse_distance = diffuse_vector.square().sum(dim=1, keepdim=True).sqrt()
    pixel_distance = pixel_vector.square().sum(dim=1, keepdim=True).sqrt()
    chroma_dilution = (diffuse_distance - pixel_distance).clamp_min(0.0)
    direction_cosine = (
        (diffuse_vector * pixel_vector).sum(dim=1, keepdim=True)
        / (diffuse_distance * pixel_distance).clamp_min(1.0e-6)
    ).clamp(-1.0, 1.0)
    return specular, chroma_dilution, direction_cosine


def attenuate_skin_specular_dichromatic(
    source_frames: torch.Tensor,
    used_skin_mask: torch.Tensor,
    *,
    amount: float = 0.80,
    specular_strength: float = 0.80,
    diffuse_radius_percent: float = 2.5,
    maximum_radius_px: int = 48,
    specular_threshold_linear: float = 0.004,
    specular_softness_linear: float = 0.030,
    chroma_dilution_threshold: float = 0.0015,
    chroma_dilution_softness: float = 0.020,
    minimum_diffuse_chroma: float = 0.008,
    diffuse_chroma_softness: float = 0.050,
    minimum_direction_cosine: float = 0.75,
    maximum_surface_delta: float = 0.10,
    minimum_texture_ratio: float = 0.86,
    maximum_texture_ratio: float = 1.10,
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
    """Attenuate likely neutral specular energy inside a reliable semantic skin mask.

    The operation is frame-independent and source-safe. It requires both a positive neutral
    specular estimate and chroma dilution in the same direction as the local diffuse colour.
    Uniform same-chromaticity bright skin therefore remains unchanged. The source stays selected
    until explicit review acceptance.
    """

    started = time.perf_counter()
    frame_count, height, width, channels = _validate_frames(
        source_frames, name="source_frames"
    )
    for name, value in {
        "amount": amount,
        "specular_strength": specular_strength,
    }.items():
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must stay within 0..1")
    if not 0.10 <= float(diffuse_radius_percent) <= 8.0:
        raise ValueError("diffuse_radius_percent must stay within 0.10..8.0")
    if not 1 <= int(maximum_radius_px) <= 192:
        raise ValueError("maximum_radius_px must stay within 1..192")
    for name, value, upper in (
        ("specular_threshold_linear", specular_threshold_linear, 0.25),
        ("specular_softness_linear", specular_softness_linear, 0.50),
        ("chroma_dilution_threshold", chroma_dilution_threshold, 0.25),
        ("chroma_dilution_softness", chroma_dilution_softness, 0.50),
        ("minimum_diffuse_chroma", minimum_diffuse_chroma, 0.25),
        ("diffuse_chroma_softness", diffuse_chroma_softness, 0.50),
    ):
        if not 0.0 <= float(value) <= upper:
            raise ValueError(f"{name} must stay within 0..{upper}")
    if float(specular_softness_linear) <= 0.0:
        raise ValueError("specular_softness_linear must be positive")
    if float(chroma_dilution_softness) <= 0.0:
        raise ValueError("chroma_dilution_softness must be positive")
    if float(diffuse_chroma_softness) <= 0.0:
        raise ValueError("diffuse_chroma_softness must be positive")
    if not -1.0 <= float(minimum_direction_cosine) < 1.0:
        raise ValueError("minimum_direction_cosine must stay within -1..1")
    if not 0.0 <= float(maximum_surface_delta) <= 0.25:
        raise ValueError("maximum_surface_delta must stay within 0..0.25")
    if not 0.0 <= float(minimum_texture_ratio) <= 1.0:
        raise ValueError("minimum_texture_ratio must stay within 0..1")
    if not 1.0 <= float(maximum_texture_ratio) <= 2.0:
        raise ValueError("maximum_texture_ratio must stay within 1..2")
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
    requested_radius = max(
        1,
        int(round(min(height, width) * float(diffuse_radius_percent) / 100.0)),
    )
    actual_radius = min(requested_radius, int(maximum_radius_px))
    macro_radius = min(
        max(24, actual_radius * 4),
        max(24, min(height, width) // 4),
    )
    chunk_size = int(chunk_frames)
    memory_before = _memory_snapshot()
    audio_report = _audio_contract(audio)
    source_proxy_sha = _tensor_proxy_sha256(source_frames)
    mask_proxy_sha = _tensor_proxy_sha256(mask)

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
        linear_rgb = _srgb_to_linear(source_nchw)
        mask_chunk = mask[start:end].float().clamp(0.0, 1.0)
        semantic_alpha = mask_chunk.unsqueeze(1)
        boundary_alpha = _mask_interior_gate(semantic_alpha, radius=2)

        diffuse_chromaticity, diffuse_distance = _masked_diffuse_chromaticity(
            linear_rgb,
            semantic_alpha,
            radius=macro_radius,
        )
        specular, chroma_dilution, direction_cosine = (
            _dichromatic_specular_estimate(linear_rgb, diffuse_chromaticity)
        )
        specular_gate = _smoothstep(
            specular,
            float(specular_threshold_linear),
            float(specular_threshold_linear) + float(specular_softness_linear),
        )
        dilution_gate = _smoothstep(
            chroma_dilution,
            float(chroma_dilution_threshold),
            float(chroma_dilution_threshold) + float(chroma_dilution_softness),
        )
        diffuse_reliability = _smoothstep(
            diffuse_distance,
            float(minimum_diffuse_chroma),
            float(minimum_diffuse_chroma) + float(diffuse_chroma_softness),
        )
        direction_gate = _smoothstep(
            direction_cosine,
            float(minimum_direction_cosine),
            min(0.999, float(minimum_direction_cosine) + 0.20),
        )

        luma_weights = torch.tensor(
            [0.2126, 0.7152, 0.0722], dtype=torch.float32
        ).view(1, 3, 1, 1)
        linear_luma = (linear_rgb * luma_weights).sum(dim=1, keepdim=True)
        support = _box_mean(semantic_alpha, macro_radius)
        macro_luma = _box_mean(
            linear_luma * semantic_alpha, macro_radius
        ) / support.clamp_min(1.0e-5)
        relative_excess = (
            (linear_luma - macro_luma) / macro_luma.clamp_min(0.02)
        ).clamp_min(0.0)
        local_brightness_gate = 0.35 + 0.65 * _smoothstep(
            relative_excess, 0.0, 0.20
        )
        confidence = (
            specular_gate
            * dilution_gate
            * diffuse_reliability
            * direction_gate
            * local_brightness_gate
        ).clamp(0.0, 1.0)
        confidence_support = _box_mean(semantic_alpha, 3)
        local_confidence = _box_mean(
            confidence * semantic_alpha, 3
        ) / confidence_support.clamp_min(1.0e-5)
        confidence = (
            (0.10 * confidence + 0.90 * local_confidence)
            * (confidence > 0.0).to(dtype=torch.float32)
        ).clamp(0.0, 1.0)
        treatment_alpha = boundary_alpha * confidence

        requested_linear_correction = -(
            specular
            * float(specular_strength)
            * float(amount)
            * treatment_alpha
        ).expand(-1, 3, -1, -1)
        candidate_linear = (linear_rgb + requested_linear_correction).clamp(
            0.0, 1.0
        )
        requested_srgb = _linear_to_srgb(candidate_linear)
        requested_delta = requested_srgb - source_nchw
        requested_peak = requested_delta.abs().amax(dim=1, keepdim=True)
        cap_scale = (
            float(maximum_surface_delta) / requested_peak.clamp_min(1.0e-8)
        ).clamp(max=1.0)
        candidate_nchw = (
            source_nchw + requested_delta * cap_scale
        ).clamp(0.0, 1.0)
        candidate_rgb = candidate_nchw.movedim(1, -1)
        composed_rgb = torch.where(
            treatment_alpha.movedim(1, -1) > 0.0,
            candidate_rgb,
            source_rgb,
        )

        treatment_mask = treatment_alpha.squeeze(1)
        semantic_binary = boundary_alpha.squeeze(1) > 0.10
        active_binary = treatment_mask > 1.0e-5
        mask_area = semantic_binary.float().mean(dim=(1, 2))
        active_area = active_binary.float().mean(dim=(1, 2))
        mask_valid = (mask_area >= float(minimum_mask_area)) & (
            mask_area <= float(maximum_mask_area)
        )
        texture_weight = semantic_binary.unsqueeze(1).float()
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
            (texture_ratio >= float(minimum_texture_ratio))
            & (texture_ratio <= float(maximum_texture_ratio))
        )
        masked_delta = (composed_rgb - source_rgb).abs()
        mask_pixels = semantic_binary.flatten(1).sum(dim=1).clamp_min(1)
        mean_change = (
            (masked_delta * semantic_binary.unsqueeze(-1)).flatten(1).sum(dim=1)
            / (mask_pixels * 3)
        )
        peak_change = (
            masked_delta.masked_fill(~semantic_binary.unsqueeze(-1), 0.0)
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
        newly_clipped = candidate_clipped & ~source_clipped & semantic_binary
        new_clipped_fraction = (
            newly_clipped.flatten(1).sum(dim=1).float() / mask_pixels
        )
        clipping_valid = new_clipped_fraction <= float(
            maximum_new_clipped_fraction
        )
        finite_valid = torch.isfinite(candidate_nchw).flatten(1).all(dim=1)
        frame_valid = (
            mask_valid & texture_valid & change_valid & clipping_valid & finite_valid
        )

        for local_index in range(end - start):
            absolute_index = start + local_index
            reasons: list[str] = []
            if not bool(mask_valid[local_index]):
                reasons.append("mask_area_gate_failed")
            if not bool(texture_valid[local_index]):
                reasons.append("texture_ratio_bounds_failed")
            if not bool(change_valid[local_index]):
                reasons.append("surface_change_limit_failed")
            if not bool(clipping_valid[local_index]):
                reasons.append("new_clipping_limit_failed")
            if not bool(finite_valid[local_index]):
                reasons.append("finite_gate_failed")
            if bool(frame_valid[local_index]):
                accepted_indices.append(absolute_index)
                frame_rgb = composed_rgb[local_index]
                effective_mask[absolute_index] = treatment_mask[local_index]
            else:
                rejected_indices.append(absolute_index)
                frame_rgb = source_rgb[local_index]
                rejected_mask[absolute_index] = boundary_alpha[local_index, 0]
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
                    "semantic_mask_area_fraction": round(
                        float(mask_area[local_index]), 8
                    ),
                    "active_specular_area_fraction": round(
                        float(active_area[local_index]), 8
                    ),
                    "mean_specular_linear": round(
                        float(specular[local_index, 0][semantic_binary[local_index]].mean())
                        if bool(semantic_binary[local_index].any())
                        else 0.0,
                        8,
                    ),
                    "mean_confidence": round(
                        float(confidence[local_index, 0][semantic_binary[local_index]].mean())
                        if bool(semantic_binary[local_index].any())
                        else 0.0,
                        8,
                    ),
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
        raise RuntimeError("Dichromatic Skin Finish changed pixels outside its mask")
    if not auxiliary_preserved:
        raise RuntimeError("Dichromatic Skin Finish changed alpha or auxiliary channels")
    if not bool(torch.isfinite(output).all()):
        raise RuntimeError("Dichromatic Skin Finish produced NaN or Inf")

    accepted = bool(accept_candidate) and bool(accepted_indices)
    selected = output if accepted else source_frames
    state = {
        "schema": SKIN_FINISH_DICHROMATIC_SCHEMA,
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
        "schema": SKIN_FINISH_DICHROMATIC_SCHEMA,
        "status": status,
        "method": "neutral_illuminant_dichromatic_chromaticity_specular_attenuation",
        "reference": DICHROMATIC_REFERENCE,
        "facial_specular_reference": FACIAL_SPECULAR_REFERENCE,
        "product_boundary": (
            "Non-generative SDR highlight attenuation under an assumed neutral illuminant. "
            "It does not estimate face geometry, calibrated lighting or physical skin BRDF; "
            "it cannot create pores, deblur, repair identity or certify natural skin. "
            "Near-neutral diffuse skin deliberately receives low confidence."
        ),
        "parameters": {
            "amount": float(amount),
            "specular_strength": float(specular_strength),
            "diffuse_radius_percent": float(diffuse_radius_percent),
            "requested_radius_px": requested_radius,
            "actual_radius_px": actual_radius,
            "macro_radius_px": macro_radius,
            "maximum_radius_px": int(maximum_radius_px),
            "specular_threshold_linear": float(specular_threshold_linear),
            "specular_softness_linear": float(specular_softness_linear),
            "chroma_dilution_threshold": float(chroma_dilution_threshold),
            "chroma_dilution_softness": float(chroma_dilution_softness),
            "minimum_diffuse_chroma": float(minimum_diffuse_chroma),
            "diffuse_chroma_softness": float(diffuse_chroma_softness),
            "minimum_direction_cosine": float(minimum_direction_cosine),
            "maximum_surface_delta": float(maximum_surface_delta),
            "minimum_texture_ratio": float(minimum_texture_ratio),
            "maximum_texture_ratio": float(maximum_texture_ratio),
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
            "mask_interior_fade_radius_px": 2,
            "confidence_smoothing_radius_px": 3,
            "confidence_smoothing_local_weight": 0.90,
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
            "requires_both_neutral_specular_and_chroma_dilution": True,
            "uniform_same_chromaticity_brightness_is_noop": True,
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
