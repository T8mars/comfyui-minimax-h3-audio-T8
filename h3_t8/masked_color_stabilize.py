"""Explicit paired-source colour correction. No sampling, motion warp or frame blending."""

from __future__ import annotations

import json
import math

import torch
import kornia.color

from .lanpaint_av_advanced import _frame_mask


def _images(value, name):
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != 4
        or min(value.shape[:3]) < 1
        or value.shape[-1] < 3
        or not value.is_floating_point()
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError(
            f"{name} must be a finite floating-point [frames,height,width,RGB] IMAGE batch"
        )
    if bool((value[..., :3] < 0).any()) or bool((value[..., :3] > 1).any()):
        raise ValueError(
            f"{name} requires normalized sRGB values in [0,1]; convert HDR explicitly before colour correction"
        )
    return value


def _lab(value):
    return kornia.color.rgb_to_lab(value[..., :3].float().permute(0, 3, 1, 2)).permute(
        0, 2, 3, 1
    )


def _mean(value, weight):
    return (value * weight[..., None]).sum(dim=(0, 1)) / weight.sum()


def _gamut_rgb(lab):
    """Keep hue while reducing chroma toward a representable neutral at fixed L."""
    values = lab.clone()
    luma_clamped = int(((values[:, 0] < 0) | (values[:, 0] > 100)).sum())
    values[:, 0] = values[:, 0].clamp(0, 100)

    def convert(value):
        return kornia.color.lab_to_rgb(value.T[None, :, None, :], clip=False)[
            0, :, 0, :
        ].T

    rgb = convert(values)
    valid = ((rgb >= -1e-6) & (rgb <= 1 + 1e-6)).all(dim=-1)
    count = int((~valid).sum())
    if count:
        affected = values[~valid]
        lower = torch.zeros(count, device=lab.device)
        upper = torch.ones_like(lower)
        for _ in range(10):  # 1/1024 chroma precision; no spatial or time filtering.
            middle = (lower + upper) * 0.5
            attempt = affected.clone()
            attempt[:, 1:] *= middle[:, None]
            candidate = convert(attempt)
            fits = ((candidate >= -1e-6) & (candidate <= 1 + 1e-6)).all(dim=-1)
            lower = torch.where(fits, middle, lower)
            upper = torch.where(fits, upper, middle)
        affected[:, 1:] *= lower[:, None]
        rgb[~valid] = convert(affected)
    return rgb, count, luma_clamped


def stabilize_masked_color(
    repaired_frames,
    source_frames,
    video_mask,
    target_reference,
    source_reference_frame=0,
    target_reference_index=0,
    strength=1.0,
    follow_source_lighting=True,
    stabilize_chroma=True,
    detail_source="generated",
):
    """Match masked Lab means to a paired target, retaining source illumination changes.

    This deliberately changes generated RGB colours and is reported as postprocessing.
    Generated detail is retained by default; source_recolour explicitly uses source
    texture instead. Both are subject to RGB gamut mapping, not model certification.
    Pixels at zero mask stay bit-identical. Audio/latents are not inputs or outputs.
    """
    repaired = _images(repaired_frames, "repaired_frames")
    source = _images(source_frames, "source_frames")
    target = _images(target_reference, "target_reference")
    count, height, width = repaired.shape[:3]
    if source.shape[:3] != repaired.shape[:3] or target.shape[1:3] != (height, width):
        raise ValueError(
            "Source must contain the same paired frames and geometry; target reference must use that geometry"
        )
    for index, limit, label in (
        (source_reference_frame, count, "source_reference_frame"),
        (target_reference_index, target.shape[0], "target_reference_index"),
    ):
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < limit
        ):
            raise ValueError(f"{label} is outside its actual IMAGE batch")
    if not math.isfinite(float(strength)) or not 0 <= float(strength) <= 1:
        raise ValueError("strength must be a finite interpolation weight in [0,1]")
    if detail_source not in ("generated", "source_recolour"):
        raise ValueError("detail_source must be generated or source_recolour")
    mask = _frame_mask(video_mask, count, height, width, repaired.device)
    report = dict(
        schema="t8.masked_colour_stabilize.v1",
        operation="explicit_generated_RGB_postprocess",
        method="paired_source_masked_Lab_mean_reference_offset",
        frames=count,
        source_reference_frame=source_reference_frame,
        target_reference_index=target_reference_index,
        strength=float(strength),
        follow_source_lighting=bool(follow_source_lighting),
        stabilize_chroma=bool(stabilize_chroma),
        detail_source=detail_source,
        source_texture_recolour=detail_source == "source_recolour",
        generated_texture_retained=detail_source == "generated",
        applied=False,
        zero_mask_RGB_bit_exact=True,
        frame_geometry_or_timing_changed=False,
        temporal_frame_blending=False,
        sampling_or_model_quality_pass_inferred=False,
        audio_and_latents_not_modified=True,
        within_frame_detail="Lab residual retained subject to explicit L bounds and hue-preserving chroma gamut compression",
        gamut_method="fixed_L_radial_chroma_compression_10_bisections",
        corrections=[],
    )
    if float(strength) == 0 or not bool(mask.any()):
        report["reason"] = "zero_strength_or_empty_mask"
        return repaired_frames, json.dumps(report, ensure_ascii=False, allow_nan=False)
    weight = mask[source_reference_frame].float()
    if not bool(weight.sum() > 0):
        report["reason"] = "reference_frame_mask_empty_choose_a_visible_source_frame"
        return repaired_frames, json.dumps(report, ensure_ascii=False, allow_nan=False)
    source = source.to(device=repaired.device)
    target = target.to(device=repaired.device)
    reference_source = _mean(
        _lab(source[source_reference_frame : source_reference_frame + 1])[0], weight
    )
    reference_target = _mean(
        _lab(target[target_reference_index : target_reference_index + 1])[0], weight
    )
    reference_offset = reference_target - reference_source
    output = repaired.clone()
    clipped_values = 0
    processed_values = 0
    compressed_pixels = 0
    luma_clamped_pixels = 0
    for index in range(count):
        weight = mask[index].float()
        if not bool(weight.sum() > 0):
            continue
        raw = _lab(repaired[index : index + 1])[0]
        source_lab = _lab(source[index : index + 1])[0]
        source_mean = _mean(source_lab, weight)
        wanted = source_mean + reference_offset
        if not follow_source_lighting:
            wanted[0] = reference_target[0]
        shift = wanted - _mean(raw, weight)
        if not stabilize_chroma:
            shift[1:] = 0
        corrected = raw + shift
        if detail_source == "source_recolour":
            # Explicit colour-only route: generated local shading can flicker even
            # with a stable whole-mask mean. Keep actual paired source texture.
            corrected = source_lab + (wanted - source_mean)
            if not stabilize_chroma:
                corrected[..., 1:] = raw[..., 1:]
        selected = weight > 0
        rgb, compressed, clamped = _gamut_rgb(corrected[selected])
        compressed_pixels += compressed
        luma_clamped_pixels += clamped
        clipped_values += int(((rgb < 0) | (rgb > 1)).sum())
        processed_values += int(rgb.numel())
        alpha = (weight[selected] * float(strength))[:, None]
        base = repaired if detail_source == "generated" else source
        mixed = (
            base[index, ..., :3][selected].float() * (1 - alpha)
            + rgb.clamp(0, 1) * alpha
        )
        output[index, ..., :3][selected] = mixed.to(repaired.dtype)
        applied_offset = shift
        if detail_source == "source_recolour":
            applied_offset = wanted - source_mean
            if not stabilize_chroma:
                applied_offset = applied_offset.clone()
                applied_offset[1:] = 0
        report["corrections"].append(
            dict(frame=index, Lab_offset=applied_offset.detach().cpu().tolist())
        )
    report.update(
        applied=bool(report["corrections"]),
        RGB_gamut_clipped_values=clipped_values,
        processed_RGB_values=processed_values,
        chroma_compressed_pixels=compressed_pixels,
        L_bounded_pixels=luma_clamped_pixels,
    )
    if detail_source == "source_recolour":
        report.update(
            method="paired_original_source_texture_with_target_Lab_colour_offset",
            within_frame_detail="paired original source Lab detail retained subject to explicit gamut mapping; generated texture replaced inside selected pixels",
        )
    return output, json.dumps(report, ensure_ascii=False, allow_nan=False)
