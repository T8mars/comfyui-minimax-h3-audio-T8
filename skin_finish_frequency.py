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


SKIN_FINISH_FREQUENCY_SPLIT_SCHEMA = "h3_t8_skin_finish_frequency_split/v1"


def _json_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _two_pass_box_lowpass(value: torch.Tensor, radius: int) -> torch.Tensor:
    """Return a bounded, edge-safe low-pass proxy for an NCHW tensor.

    Two equal box passes approximate a triangular/Gaussian-like low pass while keeping the
    implementation deterministic on CPU. Replicate padding avoids the artificial dark border
    introduced by avg_pool2d's implicit zero padding.
    """

    radius = max(1, int(radius))
    kernel = radius * 2 + 1
    result = value
    for _ in range(2):
        result = torch_functional.pad(
            result,
            (radius, radius, radius, radius),
            mode="replicate",
        )
        result = torch_functional.avg_pool2d(result, kernel_size=kernel, stride=1)
    return result


def _masked_rms(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    numerator = (value.square() * weight).flatten(1).sum(dim=1)
    denominator = weight.flatten(1).sum(dim=1).clamp_min(1.0)
    return torch.sqrt(numerator / denominator)


def separate_skin_finish_frequencies(
    source_frames: torch.Tensor,
    candidate_frames: torch.Tensor,
    used_skin_mask: torch.Tensor,
    *,
    low_frequency_strength: float = 1.0,
    source_detail_gain: float = 1.0,
    separation_radius_percent: float = 1.0,
    maximum_radius_px: int = 32,
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
    """Recombine candidate low frequencies with source high frequencies.

    This is a non-generative SDR frequency-separation operation. It cannot invent missing pores,
    deblur a source or decide that a result is aesthetically better. The source remains selected
    unless the user explicitly accepts the candidate.
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
    if not 0.0 <= float(low_frequency_strength) <= 1.0:
        raise ValueError("low_frequency_strength must stay within 0..1")
    if not 0.0 <= float(source_detail_gain) <= 1.25:
        raise ValueError("source_detail_gain must stay within 0..1.25")
    if not 0.10 <= float(separation_radius_percent) <= 5.0:
        raise ValueError("separation_radius_percent must stay within 0.10..5.0")
    if not 1 <= int(maximum_radius_px) <= 128:
        raise ValueError("maximum_radius_px must stay within 1..128")
    if not 0.0 <= float(minimum_mask_area) < float(maximum_mask_area) <= 1.0:
        raise ValueError("mask area limits must satisfy 0 <= minimum < maximum <= 1")
    if not 0.0 <= float(maximum_new_clipped_fraction) <= 0.25:
        raise ValueError("maximum_new_clipped_fraction must stay within 0..0.25")
    if not 0.0 < float(clipping_epsilon) <= 0.05:
        raise ValueError("clipping_epsilon must stay within 0..0.05")
    if not 1 <= int(chunk_frames) <= 32:
        raise ValueError("chunk_frames must stay within 1..32")
    if not bool(torch.isfinite(source_frames).all()):
        raise ValueError("source_frames contains NaN or Inf")
    if not bool(torch.isfinite(candidate_frames).all()):
        raise ValueError("candidate_frames contains NaN or Inf")

    source_proxy_sha = _tensor_proxy_sha256(source_frames)
    candidate_proxy_sha = _tensor_proxy_sha256(candidate_frames)
    mask = _normalize_mask(
        used_skin_mask,
        frame_count,
        height,
        width,
        name="used_skin_mask",
    )
    memory_before = _memory_snapshot()
    audio_report = _audio_contract(audio)

    requested_radius = max(
        1,
        int(round(min(height, width) * float(separation_radius_percent) / 100.0)),
    )
    actual_radius = min(requested_radius, int(maximum_radius_px))
    radius_was_capped = actual_radius != requested_radius
    chunk_size = int(chunk_frames)
    clip_epsilon = float(clipping_epsilon)
    luma_weights_nchw = torch.tensor(
        [0.2126, 0.7152, 0.0722], dtype=torch.float32
    ).view(1, 3, 1, 1)

    output = torch.empty(tuple(source_frames.shape), dtype=source_frames.dtype, device="cpu")
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
        candidate_chunk = candidate_frames[start:end].detach().to(device="cpu")
        source_rgb = source_chunk[..., :3].float()
        candidate_rgb = candidate_chunk[..., :3].float()
        mask_chunk = mask[start:end].float().clamp(0.0, 1.0)

        source_nchw = source_rgb.movedim(-1, 1)
        candidate_nchw = candidate_rgb.movedim(-1, 1)
        low_source = _two_pass_box_lowpass(source_nchw, actual_radius)
        low_candidate = _two_pass_box_lowpass(candidate_nchw, actual_radius)
        source_detail = source_nchw - low_source
        candidate_detail = candidate_nchw - low_candidate
        low_recombined = low_source + (
            low_candidate - low_source
        ) * float(low_frequency_strength)
        recombined_nchw = low_recombined + source_detail * float(source_detail_gain)
        recombined_rgb = recombined_nchw.movedim(1, -1)

        # Make the two important no-op contracts exact rather than tolerance based.
        if (
            float(low_frequency_strength) == 0.0
            and float(source_detail_gain) == 1.0
        ):
            recombined_rgb = source_rgb
        else:
            equal_frames = torch.eq(source_chunk, candidate_chunk).flatten(1).all(dim=1)
            if float(source_detail_gain) == 1.0 and bool(equal_frames.any()):
                recombined_rgb[equal_frames] = source_rgb[equal_frames]

        mask_area = (mask_chunk > 1.0e-5).float().mean(dim=(1, 2))
        mask_valid = (mask_area >= float(minimum_mask_area)) & (
            mask_area <= float(maximum_mask_area)
        )
        alpha = mask_chunk.unsqueeze(-1)
        composed_rgb = source_rgb + (recombined_rgb - source_rgb) * alpha
        composed_rgb = torch.where(alpha > 0.0, composed_rgb, source_rgb)

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
        frame_valid = mask_valid & clipping_valid

        texture_weight = mask_binary.float().unsqueeze(1)
        source_detail_luma = (source_detail * luma_weights_nchw).sum(dim=1, keepdim=True)
        candidate_detail_luma = (
            candidate_detail * luma_weights_nchw
        ).sum(dim=1, keepdim=True)
        recombined_detail_luma = (
            source_detail * float(source_detail_gain) * luma_weights_nchw
        ).sum(dim=1, keepdim=True)
        low_delta_luma = (
            (low_recombined - low_source) * luma_weights_nchw
        ).sum(dim=1, keepdim=True)
        source_detail_rms = _masked_rms(source_detail_luma, texture_weight)
        candidate_detail_rms = _masked_rms(candidate_detail_luma, texture_weight)
        recombined_detail_rms = _masked_rms(recombined_detail_luma, texture_weight)
        applied_low_delta_rms = _masked_rms(low_delta_luma, texture_weight)

        for local_index in range(end - start):
            absolute_index = start + local_index
            reasons: list[str] = []
            if not bool(mask_valid[local_index]):
                reasons.append("mask_area_gate_failed")
            if not bool(clipping_valid[local_index]):
                reasons.append("new_clipping_limit_failed")
            if bool(frame_valid[local_index]):
                accepted_indices.append(absolute_index)
                frame_rgb = composed_rgb[local_index].clamp(0.0, 1.0)
                effective_mask[absolute_index] = mask_chunk[local_index]
            else:
                rejected_indices.append(absolute_index)
                frame_rgb = source_rgb[local_index]
                rejected_mask[absolute_index] = mask_chunk[local_index]
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
                    "source_detail_rms": round(float(source_detail_rms[local_index]), 8),
                    "input_candidate_detail_rms": round(
                        float(candidate_detail_rms[local_index]), 8
                    ),
                    "recombined_source_detail_rms": round(
                        float(recombined_detail_rms[local_index]), 8
                    ),
                    "applied_low_frequency_delta_rms": round(
                        float(applied_low_delta_rms[local_index]), 8
                    ),
                    "new_clipped_fraction": round(
                        float(new_clipped_fraction[local_index]), 8
                    ),
                }
            )
        output_chunk = output[start:end]
        outside_chunk = effective_mask[start:end] <= 0.0
        if not torch.equal(
            output_chunk[..., :3][outside_chunk],
            source_chunk[..., :3][outside_chunk],
        ):
            outside_exact = False
        if channels > 3 and not torch.equal(
            output_chunk[..., 3:], source_chunk[..., 3:]
        ):
            auxiliary_preserved = False
        _interrupt_and_progress(progress, end, frame_count)

    if not auxiliary_preserved:
        raise RuntimeError("Skin Finish Frequency Split changed alpha or auxiliary channels")
    if not outside_exact:
        raise RuntimeError("Skin Finish Frequency Split changed pixels outside its effective mask")
    if not bool(torch.isfinite(output).all()):
        raise RuntimeError("Skin Finish Frequency Split produced NaN or Inf")

    accepted = bool(accept_candidate) and bool(accepted_indices)
    selected = output if accepted else source_frames
    state = {
        "schema": SKIN_FINISH_FREQUENCY_SPLIT_SCHEMA,
        "source_proxy_sha256": source_proxy_sha,
        "candidate_proxy_sha256": candidate_proxy_sha,
        "mask_proxy_sha256": _tensor_proxy_sha256(mask),
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
        "schema": SKIN_FINISH_FREQUENCY_SPLIT_SCHEMA,
        "status": status,
        "method": "two_pass_box_lowpass_candidate_low_plus_source_high",
        "product_boundary": (
            "Non-generative SDR frequency separation. It preserves only detail already present "
            "in the source and cannot deblur, reconstruct pores, repair a face, prove natural "
            "texture or choose an aesthetically better result."
        ),
        "colour_contract": (
            "ComfyUI 0..1 SDR IMAGE interpreted as Rec.709/sRGB-style display values. The split "
            "runs in display-referred RGB, not linear light; HDR, wide gamut and high bit depth "
            "remain unsupported."
        ),
        "parameters": {
            "low_frequency_strength": float(low_frequency_strength),
            "source_detail_gain": float(source_detail_gain),
            "separation_radius_percent": float(separation_radius_percent),
            "requested_radius_px": requested_radius,
            "actual_radius_px": actual_radius,
            "maximum_radius_px": int(maximum_radius_px),
            "radius_was_capped": radius_was_capped,
            "minimum_mask_area": float(minimum_mask_area),
            "maximum_mask_area": float(maximum_mask_area),
            "maximum_new_clipped_fraction": float(maximum_new_clipped_fraction),
            "clipping_epsilon": float(clipping_epsilon),
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
            "shape_preserved": tuple(output.shape) == tuple(source_frames.shape),
            "finite": True,
            "outside_effective_mask_bit_exact": outside_exact,
            "alpha_or_aux_channels_preserved": auxiliary_preserved,
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
