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


SKIN_FINISH_TEXTURE_GUARD_SCHEMA = "h3_t8_skin_finish_texture_guard/v1"


def _smoothstep(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp(0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def _masked_rms(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    numerator = (value.square() * weight).flatten(1).sum(dim=1)
    denominator = weight.flatten(1).sum(dim=1).clamp_min(1.0)
    return torch.sqrt(numerator / denominator)


def _json_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def guard_skin_finish_candidate(
    source_frames: torch.Tensor,
    candidate_frames: torch.Tensor,
    used_skin_mask: torch.Tensor,
    *,
    shadow_protection: float = 0.10,
    highlight_protection: float = 0.94,
    transition_width: float = 0.06,
    minimum_texture_ratio: float = 0.78,
    minimum_reference_texture: float = 0.003,
    maximum_new_clipped_fraction: float = 0.0005,
    clipping_epsilon: float = 1.0 / 255.0,
    texture_radius: int = 1,
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
    """Apply fail-closed SDR exposure/texture gates to an existing Skin Finish candidate.

    This function is deliberately non-generative. It does not decide whether a candidate is
    aesthetically better. It only prevents edits in source extremes and rejects whole frames
    when mechanical clipping or high-frequency-retention floors fail.
    """

    started = time.perf_counter()
    frame_count, height, width, channels = _validate_frames(
        source_frames, name="source_frames"
    )
    candidate_shape = _validate_frames(candidate_frames, name="candidate_frames")
    if candidate_shape != (frame_count, height, width, channels):
        raise ValueError(
            "candidate_frames must exactly match source_frames shape; "
            f"got {tuple(candidate_frames.shape)} versus {tuple(source_frames.shape)}"
        )
    if not 0.0 <= float(shadow_protection) < float(highlight_protection) <= 1.0:
        raise ValueError(
            "exposure protection must satisfy 0 <= shadow < highlight <= 1"
        )
    if not 0.001 <= float(transition_width) <= 0.25:
        raise ValueError("transition_width must stay within 0.001..0.25")
    if not 0.0 <= float(minimum_texture_ratio) <= 1.0:
        raise ValueError("minimum_texture_ratio must stay within 0..1")
    if not 0.0 <= float(minimum_reference_texture) <= 0.25:
        raise ValueError("minimum_reference_texture must stay within 0..0.25")
    if not 0.0 <= float(maximum_new_clipped_fraction) <= 0.25:
        raise ValueError("maximum_new_clipped_fraction must stay within 0..0.25")
    if not 0.0 < float(clipping_epsilon) <= 0.05:
        raise ValueError("clipping_epsilon must stay within 0..0.05")
    if not 1 <= int(texture_radius) <= 8:
        raise ValueError("texture_radius must stay within 1..8")

    source_proxy_sha = _tensor_proxy_sha256(source_frames)
    source = source_frames.detach().to(device="cpu")
    candidate = candidate_frames.detach().to(device="cpu")
    raw_mask = _normalize_mask(
        used_skin_mask,
        frame_count,
        height,
        width,
        name="used_skin_mask",
    )
    memory_before = _memory_snapshot()
    audio_report = _audio_contract(audio)

    guarded = torch.empty_like(source, device="cpu")
    effective_mask = torch.zeros_like(raw_mask, device="cpu")
    rejected_mask = torch.zeros_like(raw_mask, device="cpu")
    difference = torch.zeros(
        (frame_count, height, width, 3), dtype=torch.float16, device="cpu"
    )
    luma_weights = torch.tensor([0.2126, 0.7152, 0.0722], dtype=torch.float32).view(
        1, 1, 1, 3
    )
    transition = float(transition_width)
    clip_epsilon = float(clipping_epsilon)
    kernel = int(texture_radius) * 2 + 1
    frame_reports: list[dict[str, Any]] = []
    accepted_indices: list[int] = []
    rejected_indices: list[int] = []
    chunk_size = max(1, int(chunk_frames))
    progress = _progress_bar(frame_count)

    for start in range(0, frame_count, chunk_size):
        end = min(frame_count, start + chunk_size)
        _interrupt_and_progress(progress, start, frame_count)
        source_chunk = source[start:end].float()
        candidate_chunk = candidate[start:end].float()
        source_rgb = source_chunk[..., :3]
        candidate_rgb = candidate_chunk[..., :3]
        mask_chunk = raw_mask[start:end]
        source_luma = (source_rgb * luma_weights).sum(dim=-1)

        shadow_gate = _smoothstep(
            (source_luma - float(shadow_protection)) / transition
        )
        highlight_gate = _smoothstep(
            (float(highlight_protection) - source_luma) / transition
        )
        exposure_gate = shadow_gate * highlight_gate
        effective_chunk = (mask_chunk * exposure_gate).clamp(0.0, 1.0)
        alpha = effective_chunk.unsqueeze(-1)
        guarded_rgb = source_rgb + (candidate_rgb - source_rgb) * alpha
        guarded_rgb = torch.where(alpha > 0.0, guarded_rgb, source_rgb).clamp(0.0, 1.0)

        source_luma_nchw = source_luma.unsqueeze(1)
        guarded_luma = (guarded_rgb * luma_weights).sum(dim=-1).unsqueeze(1)
        source_local = torch_functional.avg_pool2d(
            source_luma_nchw,
            kernel_size=kernel,
            stride=1,
            padding=int(texture_radius),
        )
        guarded_local = torch_functional.avg_pool2d(
            guarded_luma,
            kernel_size=kernel,
            stride=1,
            padding=int(texture_radius),
        )
        texture_weight = (effective_chunk > 0.10).float().unsqueeze(1)
        source_texture = _masked_rms(source_luma_nchw - source_local, texture_weight)
        candidate_texture = _masked_rms(guarded_luma - guarded_local, texture_weight)
        texture_ratio = torch.where(
            source_texture >= float(minimum_reference_texture),
            candidate_texture / source_texture.clamp_min(1e-8),
            torch.ones_like(source_texture),
        )

        source_clipped = ((source_rgb <= clip_epsilon) | (source_rgb >= 1.0 - clip_epsilon)).any(
            dim=-1
        )
        guarded_clipped = (
            (guarded_rgb <= clip_epsilon) | (guarded_rgb >= 1.0 - clip_epsilon)
        ).any(dim=-1)
        mask_binary = mask_chunk > 0.10
        new_clipped = guarded_clipped & ~source_clipped & mask_binary
        mask_pixels = mask_binary.flatten(1).sum(dim=1).clamp_min(1)
        new_clipped_fraction = new_clipped.flatten(1).sum(dim=1).float() / mask_pixels

        has_effective_mask = (effective_chunk > 0.10).flatten(1).any(dim=1)
        texture_pass = texture_ratio >= float(minimum_texture_ratio)
        clipping_pass = new_clipped_fraction <= float(maximum_new_clipped_fraction)
        frame_pass = has_effective_mask & texture_pass & clipping_pass

        for local_index in range(end - start):
            absolute_index = start + local_index
            reasons: list[str] = []
            if not bool(has_effective_mask[local_index]):
                reasons.append("no_effective_mask_after_exposure_protection")
            if not bool(texture_pass[local_index]):
                reasons.append("texture_floor_failed")
            if not bool(clipping_pass[local_index]):
                reasons.append("new_clipping_limit_failed")
            if bool(frame_pass[local_index]):
                accepted_indices.append(absolute_index)
                output_rgb = guarded_rgb[local_index]
                effective_mask[absolute_index] = effective_chunk[local_index]
                rejected_mask[absolute_index] = (
                    mask_chunk[local_index] - effective_chunk[local_index]
                ).clamp_min(0.0)
            else:
                rejected_indices.append(absolute_index)
                output_rgb = source_rgb[local_index]
                rejected_mask[absolute_index] = mask_chunk[local_index]
            guarded[absolute_index] = source[absolute_index]
            guarded[absolute_index, ..., :3] = output_rgb.to(dtype=source.dtype)
            difference[absolute_index] = (
                output_rgb - source_rgb[local_index]
            ).abs().to(dtype=torch.float16)
            frame_reports.append(
                {
                    "frame_index": absolute_index,
                    "status": "PASS" if bool(frame_pass[local_index]) else "REJECT",
                    "reasons": reasons,
                    "source_texture_rms": round(float(source_texture[local_index]), 8),
                    "candidate_texture_rms": round(float(candidate_texture[local_index]), 8),
                    "texture_ratio": round(float(texture_ratio[local_index]), 8),
                    "new_clipped_fraction": round(
                        float(new_clipped_fraction[local_index]), 8
                    ),
                    "effective_mask_fraction": round(
                        float((effective_chunk[local_index] > 0.10).float().mean()), 8
                    ),
                }
            )
        _interrupt_and_progress(progress, end, frame_count)

    if channels > 3 and not torch.equal(guarded[..., 3:], source[..., 3:]):
        raise RuntimeError("Skin Finish Texture Guard changed alpha or auxiliary channels")
    outside = effective_mask <= 0.0
    if not torch.equal(guarded[..., :3][outside], source[..., :3][outside]):
        raise RuntimeError("Skin Finish Texture Guard changed pixels outside its effective mask")
    if not bool(torch.isfinite(guarded).all()):
        raise RuntimeError("Skin Finish Texture Guard produced NaN or Inf")

    accepted = bool(accept_candidate) and bool(accepted_indices)
    selected = guarded if accepted else source_frames
    state = {
        "schema": SKIN_FINISH_TEXTURE_GUARD_SCHEMA,
        "source_proxy_sha256": source_proxy_sha,
        "candidate_proxy_sha256": _tensor_proxy_sha256(candidate_frames),
        "used_mask_proxy_sha256": _tensor_proxy_sha256(raw_mask),
        "accepted_frame_indices": accepted_indices,
        "rejected_frame_indices": rejected_indices,
        "candidate_selected": accepted,
    }
    state["sha256"] = _json_hash(state)
    report = {
        "schema": SKIN_FINISH_TEXTURE_GUARD_SCHEMA,
        "status": (
            "PASS_WITH_REJECTED_FRAMES"
            if accepted_indices and rejected_indices
            else "PASS"
            if accepted_indices
            else "ABSTAIN_ALL_FRAMES_REJECTED"
        ),
        "product_boundary": (
            "Non-generative SDR mechanical guard. High-pass energy is only a hard retention "
            "floor; it cannot prove natural skin, pores, sharpness, identity or beauty."
        ),
        "colour_contract": (
            "ComfyUI 0..1 SDR IMAGE interpreted as Rec.709/sRGB-style display values; no HDR, "
            "wide-gamut, linear-light or high-bit-depth claim."
        ),
        "parameters": {
            "shadow_protection": float(shadow_protection),
            "highlight_protection": float(highlight_protection),
            "transition_width": float(transition_width),
            "minimum_texture_ratio": float(minimum_texture_ratio),
            "minimum_reference_texture": float(minimum_reference_texture),
            "maximum_new_clipped_fraction": float(maximum_new_clipped_fraction),
            "clipping_epsilon": float(clipping_epsilon),
            "texture_radius": int(texture_radius),
            "chunk_frames": chunk_size,
            "peak_chunk_frames": min(chunk_size, frame_count),
        },
        "frame_count": frame_count,
        "accepted_frame_count": len(accepted_indices),
        "rejected_frame_count": len(rejected_indices),
        "accepted_frame_indices": accepted_indices,
        "rejected_frame_indices": rejected_indices,
        "frame_reports": frame_reports,
        "mechanical_gates": {
            "shape_preserved": tuple(guarded.shape) == tuple(source_frames.shape),
            "finite": True,
            "outside_effective_mask_bit_exact": True,
            "alpha_or_aux_channels_preserved": True,
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
        guarded,
        source_frames,
        selected,
        audio,
        effective_mask,
        rejected_mask,
        difference,
        canonical_json(report),
    )
