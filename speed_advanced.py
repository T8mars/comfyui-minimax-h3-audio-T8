from __future__ import annotations

import json
import hashlib
import math
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

import comfy.nested_tensor
from comfy.ldm.minimax.model import MiniMaxH3Model

from .conditioning import build_conditioning, resolve_task_type
from .core import (
    FPS,
    MAX_PIXELS,
    align_frame_count,
    empty_av_latent,
    nested_av_parts,
    resize_image,
    sorted_autogrow_values,
    video_latent_t,
)
from .sampling import native_flow_sigmas, setup_dual_clock_sampling


SPEED_PROFILE_SCHEMA = "minimax_h3_speed_spectrum_profile_t8_v1"
SPEED_SPECTRUM_DATASET_SCHEMA = "minimax_h3_speed_spectrum_dataset_t8_v1"
SPEED_DATASET_PROVENANCE_SCHEMA = "minimax_h3_speed_dataset_provenance_t8_v1"
SPEED_SOURCE_ENTRY_SCHEMA = "minimax_h3_speed_source_entry_t8_v1"
SPEED_PLAN_SCHEMA = "minimax_h3_speed_plan_t8_v1"
SPEED_SOURCE_SCHEMA = "minimax_h3_speed_source_t8_v1"
SPEED_REPORT_SCHEMA = "minimax_h3_speed_execution_t8_v1"
OFFICIAL_SPEED_COMMIT = "ca7801c9bdffe681742e9592345bcf4885959be5"
OFFICIAL_SPEED_PAPER = "arXiv:2605.18736v3"
H3_CORE_AUDIT_COMMIT = "7fe8a6138504f90ff7be82f3babf416da32876b1"
SPEED_SCOPED_HEADROOM_BYTES = int(1.5 * 1024**3)
SPEED_AUDIO_NOISE_SEED_XOR = 0x9E3779B97F4A7C15


def canonical_json(value: Any, *, indent: int | None = 2) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=indent)


@torch.no_grad()
def prepare_speed_calibration_window(
    frames: torch.Tensor,
    *,
    source_fps: float,
    width: int,
    height: int,
    length: int,
    start_seconds: float,
    resize_mode: str,
) -> tuple[torch.Tensor, int, str]:
    """Create one aspect-safe, fixed-grid H3 spectrum-calibration window."""

    if not isinstance(frames, torch.Tensor) or frames.ndim != 4:
        shape = tuple(frames.shape) if isinstance(frames, torch.Tensor) else type(frames).__name__
        raise ValueError(f"frames must be IMAGE [N,H,W,C], got {shape}")
    if frames.shape[0] < 1 or frames.shape[-1] < 3:
        raise ValueError("Calibration frames are empty or do not contain RGB channels")
    source_fps = float(source_fps)
    start_seconds = float(start_seconds)
    width = int(width)
    height = int(height)
    if not math.isfinite(source_fps) or source_fps <= 0.0:
        raise ValueError("source_fps must be finite and positive")
    if not math.isfinite(start_seconds) or start_seconds < 0.0:
        raise ValueError("start_seconds must be finite and nonnegative")
    if width < 32 or height < 32 or width % 32 or height % 32:
        raise ValueError("Calibration width and height must be positive multiples of 32")
    if width * height > MAX_PIXELS:
        raise ValueError(
            f"Calibration canvas {width}x{height} exceeds the {MAX_PIXELS:,}-pixel limit"
        )
    if resize_mode != "center_cover":
        raise ValueError("SPEED calibration only supports aspect-safe center_cover")

    frame_count = align_frame_count(int(length))
    target_times = start_seconds + torch.arange(frame_count, dtype=torch.float64) / FPS
    source_indices = torch.round(target_times * source_fps).to(torch.long)
    if int(source_indices[-1]) >= frames.shape[0]:
        required_end = start_seconds + (frame_count - 1) / FPS
        available_end = (frames.shape[0] - 1) / source_fps
        raise ValueError(
            "Calibration video is too short for a strict H3 window: "
            f"needs frame time {required_end:.6f}s, available through {available_end:.6f}s"
        )
    source_indices = source_indices.to(frames.device)
    selected = frames.index_select(0, source_indices)
    source_height = int(selected.shape[1])
    source_width = int(selected.shape[2])
    source_aspect = source_width / source_height
    target_aspect = width / height
    crop_left_right = 0
    crop_top_bottom = 0
    if source_aspect > target_aspect:
        crop_each = round(
            (source_width - source_width * (target_aspect / source_aspect)) / 2
        )
        crop_left_right = int(crop_each * 2)
    elif source_aspect < target_aspect:
        crop_each = round(
            (source_height - source_height * (source_aspect / target_aspect)) / 2
        )
        crop_top_bottom = int(crop_each * 2)
    selected = resize_image(selected, width, height, "center")
    if tuple(selected.shape[1:3]) != (height, width):
        raise RuntimeError("Aspect-safe calibration resize returned an unexpected canvas")

    report = {
        "schema": "minimax_h3_speed_calibration_window_t8_v1",
        "status": "aspect_safe_center_cover",
        "source": {
            "frames": int(frames.shape[0]),
            "width": source_width,
            "height": source_height,
            "fps": source_fps,
            "aspect_ratio": source_aspect,
        },
        "target": {
            "frames": frame_count,
            "width": width,
            "height": height,
            "fps": FPS,
            "aspect_ratio": target_aspect,
            "start_seconds": start_seconds,
        },
        "sampling": {
            "first_source_index": int(source_indices[0]),
            "last_source_index": int(source_indices[-1]),
            "unique_source_frames": int(torch.unique(source_indices).numel()),
        },
        "resize": {
            "mode": resize_mode,
            "anisotropic_stretch": False,
            "cropped_source_pixels_width": crop_left_right,
            "cropped_source_pixels_height": crop_top_bottom,
            "retained_source_fraction": (
                (source_width - crop_left_right)
                * (source_height - crop_top_bottom)
                / (source_width * source_height)
            ),
        },
        "boundary": (
            "Center-cover preserves geometry but discards source edges. It is appropriate for "
            "a fixed-grid spectrum dataset and is not evidence that the selected clips are "
            "independent, representative, diverse, or authorized."
        ),
    }
    return selected, frame_count, canonical_json(report)


def power_spectrum(omega: float, amplitude: float, beta: float) -> float:
    if not all(math.isfinite(v) for v in (omega, amplitude, beta)):
        raise ValueError("omega, amplitude, and beta must be finite")
    if omega <= 0.0 or amplitude <= 0.0 or beta <= 0.0:
        raise ValueError("omega, amplitude, and beta must be greater than zero")
    return amplitude * abs(omega) ** (-beta)


def activation_time(power: float, delta: float) -> float:
    """Official SPEED Eq. 9."""
    if not math.isfinite(power) or power <= 0.0:
        raise ValueError("power must be finite and greater than zero")
    if not math.isfinite(delta) or not 0.0 < delta < 1.0:
        raise ValueError("delta must be finite and in (0, 1)")
    denominator = power * (1.0 + power - delta)
    if denominator <= 0.0:
        raise ValueError("delta is incompatible with the supplied power spectrum")
    return 1.0 / (1.0 + math.sqrt(delta / denominator))


def kappa(sigma: float, ratio: float) -> float:
    """Official SPEED Eq. 5, evaluated in the actual shifted sampler sigma."""
    if not math.isfinite(sigma) or not 0.0 <= sigma <= 1.0:
        raise ValueError("sigma must be finite and in [0, 1]")
    if not math.isfinite(ratio) or ratio < 1.0:
        raise ValueError("resolution ratio must be finite and at least 1")
    return ratio / (1.0 + (ratio - 1.0) * sigma)


def align_sigma(sigma: float, ratio: float) -> float:
    """Official SPEED Eq. 6."""
    return sigma * kappa(sigma, ratio)


def parse_float_list(value: str, *, name: str) -> list[float]:
    try:
        output = [float(token.strip()) for token in str(value).split(",") if token.strip()]
    except ValueError as exc:
        raise ValueError(f"{name} must be a comma-separated numeric list") from exc
    if not output or any(not math.isfinite(item) for item in output):
        raise ValueError(f"{name} must contain finite numbers")
    return output


def validate_scales(scales: Sequence[float]) -> list[float]:
    values = [float(scale) for scale in scales]
    if not values:
        raise ValueError("SPEED requires at least one spatial scale")
    if any(not math.isfinite(scale) or not 0.0 < scale <= 1.0 for scale in values):
        raise ValueError("Every SPEED spatial scale must be finite and in (0, 1]")
    if not math.isclose(values[-1], 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("The final SPEED scale must be 1.0")
    if any(a >= b for a, b in zip(values, values[1:])):
        raise ValueError("SPEED scales must be strictly increasing")
    return values


def _snap32(value: float) -> int:
    return max(32, int(round(float(value) / 32.0)) * 32)


def _aspect_preserving_stage_shape(width: int, height: int, scale: float) -> tuple[int, int]:
    expected_width = width * scale
    expected_height = height * scale
    width_center = _snap32(expected_width)
    height_center = _snap32(expected_height)
    width_candidates = {
        max(32, width_center + offset * 32) for offset in range(-2, 3)
    }
    height_candidates = {
        max(32, height_center + offset * 32) for offset in range(-2, 3)
    }

    def score(shape: tuple[int, int]) -> tuple[float, int]:
        stage_width, stage_height = shape
        scale_width = stage_width / width
        scale_height = stage_height / height
        effective_scale = math.sqrt(scale_width * scale_height)
        aspect_error = abs(math.log(scale_width / scale_height))
        scale_error = abs(math.log(effective_scale / scale))
        pixel_error = abs(stage_width * stage_height - expected_width * expected_height)
        return 2.0 * aspect_error + scale_error, int(pixel_error)

    return min(
        ((candidate_width, candidate_height) for candidate_width in width_candidates
         for candidate_height in height_candidates),
        key=score,
    )


def resolve_stage_shapes(width: int, height: int, scales: Sequence[float]) -> list[dict[str, Any]]:
    width = int(width)
    height = int(height)
    if width <= 0 or height <= 0 or width % 32 or height % 32:
        raise ValueError("Final H3 width and height must be positive multiples of 32")
    if width * height > MAX_PIXELS:
        raise ValueError(
            f"Final H3 canvas has {width * height:,} pixels and exceeds the 2.0MP cap"
        )
    scales = validate_scales(scales)
    output: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for index, requested in enumerate(scales):
        if index == len(scales) - 1:
            stage_width, stage_height = width, height
        else:
            stage_width, stage_height = _aspect_preserving_stage_shape(
                width, height, requested
            )
        shape = (stage_width, stage_height)
        if shape in seen:
            raise ValueError(
                "Two requested SPEED scales collapse to the same multiple-of-32 canvas; "
                "remove one scale or increase the final resolution"
            )
        seen.add(shape)
        scale_w = stage_width / width
        scale_h = stage_height / height
        anisotropy = abs(scale_w - scale_h) / max(scale_w, scale_h)
        if anisotropy > 0.05:
            raise ValueError(
                "Multiple-of-32 snapping made a SPEED stage more than 5% anisotropic; "
                "choose a less aggressive scale"
            )
        output.append(
            {
                "index": index,
                "requested_scale": requested,
                "width": stage_width,
                "height": stage_height,
                "latent_width": stage_width // 16,
                "latent_height": stage_height // 16,
                "effective_scale_width": scale_w,
                "effective_scale_height": scale_h,
                "effective_scale": math.sqrt(scale_w * scale_h),
                "snap_anisotropy": anisotropy,
            }
        )
    for current, following in zip(output, output[1:]):
        if following["width"] <= current["width"] or following["height"] <= current["height"]:
            raise ValueError("Resolved SPEED stage canvases must grow in both spatial axes")
    return output


def _linear_fit(x: torch.Tensor, y: torch.Tensor) -> tuple[float, float, float]:
    design = torch.stack((torch.ones_like(x), x), dim=1)
    solution = torch.linalg.lstsq(design, y[:, None]).solution[:, 0]
    intercept = float(solution[0])
    slope = float(solution[1])
    predicted = design @ solution
    residual = torch.sum((y - predicted) ** 2)
    total = torch.sum((y - y.mean()) ** 2)
    r_squared = 1.0 if float(total) == 0.0 else 1.0 - float(residual / total)
    return intercept, slope, r_squared


def _spectrum_input_contract(
    video_latent: torch.Tensor, *, max_temporal_samples: int
) -> tuple[dict[str, int], torch.Tensor]:
    if not isinstance(video_latent, torch.Tensor) or video_latent.ndim != 5:
        raise ValueError("video_latent must be [B,C,T,H,W]")
    batch, channels, frames, height, width = map(int, video_latent.shape)
    if batch < 1 or channels < 1 or frames < 1 or min(height, width) < 8:
        raise ValueError("video_latent is too small for a spatial spectrum fit")
    max_temporal_samples = int(max_temporal_samples)
    if max_temporal_samples < 1:
        raise ValueError("max_temporal_samples must be positive")
    temporal_indices = torch.linspace(
        0,
        frames - 1,
        min(frames, max_temporal_samples),
        dtype=torch.float64,
    ).round().long().unique()
    return (
        {
            "batch": batch,
            "channels": channels,
            "frames": frames,
            "height": height,
            "width": width,
            "max_temporal_samples": max_temporal_samples,
            "temporal_samples": int(temporal_indices.numel()),
        },
        temporal_indices,
    )


@torch.no_grad()
def _spatial_power_per_clip(
    video_latent: torch.Tensor, *, max_temporal_samples: int
) -> tuple[torch.Tensor, dict[str, int]]:
    contract, temporal_indices = _spectrum_input_contract(
        video_latent, max_temporal_samples=max_temporal_samples
    )
    source_indices = temporal_indices.to(video_latent.device)
    rows: list[torch.Tensor] = []
    for batch_index in range(contract["batch"]):
        # Stream one clip at a time to CPU. The retained calibration state is one
        # float64 HxW power grid, never the source latent or a CUDA tensor.
        value = video_latent[batch_index : batch_index + 1].index_select(
            2, source_indices
        ).detach()
        value = value.to(device="cpu", dtype=torch.float32)
        value = value - value.mean(dim=(-2, -1), keepdim=True)
        spectrum = torch.fft.fft2(value, norm="ortho")
        rows.append(
            spectrum.abs().square().mean(dim=(0, 1, 2)).to(torch.float64)
        )
        del value, spectrum
    del source_indices
    return torch.stack(rows, dim=0), contract


def _fit_spatial_mean_power(
    mean_power: torch.Tensor,
    *,
    latent_contract: Mapping[str, int],
    minimum_radius: int,
    maximum_radius_fraction: float,
    clip_count: int,
) -> dict[str, Any]:
    if (
        not isinstance(mean_power, torch.Tensor)
        or mean_power.device.type != "cpu"
        or mean_power.ndim != 2
    ):
        raise ValueError("mean_power must be a CPU [H,W] tensor")
    if not bool(torch.isfinite(mean_power).all()) or bool((mean_power < 0).any()):
        raise ValueError("mean_power must contain finite non-negative values")
    height = int(latent_contract["height"])
    width = int(latent_contract["width"])
    if list(mean_power.shape) != [height, width]:
        raise ValueError("mean_power shape does not match the latent contract")
    if not math.isfinite(maximum_radius_fraction) or not (
        0.1 <= maximum_radius_fraction <= 1.0
    ):
        raise ValueError("maximum_radius_fraction must be in [0.1, 1.0]")

    fy = torch.fft.fftfreq(height, d=1.0, dtype=torch.float64) * height
    fx = torch.fft.fftfreq(width, d=1.0, dtype=torch.float64) * width
    radius = torch.sqrt(fy[:, None].square() + fx[None, :].square())
    radial_bin = radius.round().long()
    maximum_radius = max(
        int(minimum_radius) + 2,
        int(math.floor(min(height, width) * maximum_radius_fraction / 2.0)),
    )
    radii: list[float] = []
    powers: list[float] = []
    counts: list[int] = []
    for index in range(max(1, int(minimum_radius)), maximum_radius + 1):
        mask = radial_bin == index
        count = int(mask.sum())
        if count < 2:
            continue
        power = float(mean_power[mask].mean())
        if math.isfinite(power) and power > 0.0:
            radii.append(float(index))
            powers.append(power)
            counts.append(count)
    if len(radii) < 4:
        raise ValueError("The latent did not provide enough finite radial spectrum bins")

    log_radius = torch.tensor(radii, dtype=torch.float64).log()
    log_power = torch.tensor(powers, dtype=torch.float64).log()
    intercept, slope, r_squared = _linear_fit(log_radius, log_power)
    amplitude = math.exp(intercept)
    beta = -slope
    if not (
        math.isfinite(amplitude)
        and amplitude > 0.0
        and math.isfinite(beta)
        and beta > 0.0
    ):
        raise ValueError("The fitted spatial power law is not physically usable")
    return {
        "amplitude": amplitude,
        "beta": beta,
        "r_squared": r_squared,
        "radial_bins": len(radii),
        "spatial_coefficients": int(sum(counts)),
        "latent_shape": [
            int(clip_count),
            int(latent_contract["channels"]),
            int(latent_contract["frames"]),
            height,
            width,
        ],
        "temporal_samples": int(latent_contract["temporal_samples"]),
        "radius_range": [radii[0], radii[-1]],
    }


@torch.no_grad()
def fit_h3_spatial_power_spectrum(
    video_latent: torch.Tensor,
    *,
    max_temporal_samples: int = 32,
    minimum_radius: int = 1,
    maximum_radius_fraction: float = 0.5,
) -> dict[str, Any]:
    """Fit a radial FFT power law to H3 video latents without importing SciPy."""
    per_clip_power, contract = _spatial_power_per_clip(
        video_latent, max_temporal_samples=max_temporal_samples
    )
    return _fit_spatial_mean_power(
        per_clip_power.mean(dim=0),
        latent_contract=contract,
        minimum_radius=minimum_radius,
        maximum_radius_fraction=maximum_radius_fraction,
        clip_count=contract["batch"],
    )


def _power_sha256(value: torch.Tensor) -> str:
    array = value.detach().to(device="cpu", dtype=torch.float64).contiguous().numpy()
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest().upper()


def _usable_provenance(value: Any) -> str:
    normalized = str(value).strip()
    if not normalized or normalized.lower() in {"unrecorded", "unknown", "none"}:
        raise ValueError("Dataset calibration requires a recorded model/VAE fingerprint")
    return normalized


def _spectrum_dataset_public_report(dataset: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in dataset.items()
        if key != "power_sum"
    }


def _sha256_text(value: Any, *, field: str) -> str:
    normalized = str(value).strip().upper()
    if len(normalized) != 64 or any(character not in "0123456789ABCDEF" for character in normalized):
        raise ValueError(f"{field} must be a complete hexadecimal SHA-256")
    return normalized


def _source_set_sha256(entries: Sequence[Mapping[str, Any]]) -> str:
    normalized = [
        {
            "source_file_sha256": _sha256_text(
                entry.get("source_file_sha256"), field="source_file_sha256"
            ),
            "decoded_window_sha256": _sha256_text(
                entry.get("decoded_window_sha256"), field="decoded_window_sha256"
            ),
        }
        for entry in entries
    ]
    normalized.sort(
        key=lambda item: (item["source_file_sha256"], item["decoded_window_sha256"])
    )
    return hashlib.sha256(
        canonical_json(normalized, indent=None).encode("utf-8")
    ).hexdigest().upper()


def _parse_dataset_provenance(value: Any) -> dict[str, Any] | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("dataset_provenance_json is malformed") from exc
    if not isinstance(value, Mapping) or value.get("schema") != SPEED_DATASET_PROVENANCE_SCHEMA:
        raise ValueError("Dataset provenance schema is missing or unsupported")
    required_text = (
        "source_kind",
        "dataset_id",
        "dataset_revision",
        "dataset_license",
        "selection_policy",
    )
    normalized = dict(value)
    for field in required_text:
        text = str(normalized.get(field, "")).strip()
        if not text:
            raise ValueError(f"Dataset provenance is missing {field}")
        normalized[field] = text
    normalized["curation_report_sha256"] = _sha256_text(
        normalized.get("curation_report_sha256"), field="curation_report_sha256"
    )
    normalized["review_report_sha256"] = _sha256_text(
        normalized.get("review_report_sha256"), field="review_report_sha256"
    )
    normalized["selected_source_set_sha256"] = _sha256_text(
        normalized.get("selected_source_set_sha256"),
        field="selected_source_set_sha256",
    )
    selected_count = int(normalized.get("selected_source_count", 0))
    if selected_count < 1:
        raise ValueError("Dataset provenance selected_source_count must be positive")
    normalized["selected_source_count"] = selected_count
    for field in (
        "independence_reviewed",
        "content_diversity_reviewed",
        "raw_media_redistributed",
    ):
        if not isinstance(normalized.get(field), bool):
            raise ValueError(f"Dataset provenance {field} must be boolean")
    shards = normalized.get("source_shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("Dataset provenance requires at least one source_shard")
    normalized_shards = []
    for shard in shards:
        if not isinstance(shard, Mapping):
            raise ValueError("Dataset provenance source_shards entries must be objects")
        shard_name = str(shard.get("shard", "")).strip()
        if not shard_name or Path(shard_name).is_absolute() or ".." in Path(shard_name).parts:
            raise ValueError("Dataset provenance shard names must be safe relative paths")
        normalized_shards.append(
            {
                "shard": shard_name.replace("\\", "/"),
                "lfs_oid": _sha256_text(shard.get("lfs_oid"), field="source_shard.lfs_oid"),
                "fetch_report_sha256": _sha256_text(
                    shard.get("fetch_report_sha256"),
                    field="source_shard.fetch_report_sha256",
                ),
            }
        )
    normalized["source_shards"] = sorted(
        normalized_shards, key=lambda item: (item["shard"], item["lfs_oid"])
    )
    return normalized


def _parse_source_entry(value: Any, *, batch_id: str) -> dict[str, Any] | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("source_entry_json is malformed") from exc
    if not isinstance(value, Mapping) or value.get("schema") != SPEED_SOURCE_ENTRY_SCHEMA:
        raise ValueError("Source entry provenance schema is missing or unsupported")
    entry_batch_id = str(value.get("batch_id", "")).strip()
    if entry_batch_id != batch_id:
        raise ValueError("Source entry batch_id does not match the accumulated batch")
    return {
        "schema": SPEED_SOURCE_ENTRY_SCHEMA,
        "batch_id": entry_batch_id,
        "source_file_sha256": _sha256_text(
            value.get("source_file_sha256"), field="source_file_sha256"
        ),
        "decoded_window_sha256": _sha256_text(
            value.get("decoded_window_sha256"), field="decoded_window_sha256"
        ),
    }


def _validate_spectrum_dataset(dataset: Mapping[str, Any]) -> None:
    if not isinstance(dataset, Mapping) or dataset.get("schema") != SPEED_SPECTRUM_DATASET_SCHEMA:
        raise ValueError("previous_dataset is not an H3 SPEED spectrum dataset")
    power_sum = dataset.get("power_sum")
    contract = dataset.get("latent_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("Spectrum dataset latent_contract is missing")
    if (
        not isinstance(power_sum, torch.Tensor)
        or power_sum.device.type != "cpu"
        or power_sum.dtype != torch.float64
        or power_sum.ndim != 2
    ):
        raise ValueError("Spectrum dataset power_sum must be a CPU float64 [H,W] tensor")
    expected_shape = [int(contract.get("height", -1)), int(contract.get("width", -1))]
    if list(power_sum.shape) != expected_shape:
        raise ValueError("Spectrum dataset power_sum shape does not match its contract")
    if not bool(torch.isfinite(power_sum).all()) or bool((power_sum < 0).any()):
        raise ValueError("Spectrum dataset power_sum is invalid")
    clip_count = int(dataset.get("independent_clip_count", 0))
    batch_count = int(dataset.get("batch_count", 0))
    batch_ids = dataset.get("batch_ids")
    batch_sizes = dataset.get("batch_sizes")
    clip_fingerprints = dataset.get("clip_fingerprints")
    if (
        clip_count < 1
        or batch_count < 1
        or not isinstance(batch_ids, list)
        or not isinstance(batch_sizes, list)
        or not isinstance(clip_fingerprints, list)
    ):
        raise ValueError("Spectrum dataset counts or provenance lists are invalid")
    if len(batch_ids) != batch_count or len(batch_sizes) != batch_count:
        raise ValueError("Spectrum dataset batch counts are inconsistent")
    if any(
        not isinstance(size, int) or isinstance(size, bool) or size < 1
        for size in batch_sizes
    ):
        raise ValueError("Spectrum dataset batch sizes are invalid")
    if sum(batch_sizes) != clip_count:
        raise ValueError("Spectrum dataset batch sizes do not sum to clip count")
    if len(clip_fingerprints) != clip_count or len(set(clip_fingerprints)) != clip_count:
        raise ValueError("Spectrum dataset clip fingerprints are inconsistent")
    if len(set(batch_ids)) != len(batch_ids):
        raise ValueError("Spectrum dataset contains duplicate batch IDs")
    if dataset.get("power_sum_sha256") != _power_sha256(power_sum):
        raise ValueError("Spectrum dataset power_sum hash mismatch")
    provenance = dataset.get("dataset_provenance")
    source_entries = dataset.get("source_entries")
    if provenance is not None:
        normalized_provenance = _parse_dataset_provenance(provenance)
        if normalized_provenance != provenance:
            raise ValueError("Spectrum dataset provenance is not canonical")
    if source_entries is not None:
        if not isinstance(source_entries, list) or len(source_entries) != batch_count:
            raise ValueError("Spectrum dataset source_entries must match batch_count")
        normalized_entries = [
            _parse_source_entry(entry, batch_id=str(batch_id))
            for entry, batch_id in zip(source_entries, batch_ids)
        ]
        if normalized_entries != source_entries:
            raise ValueError("Spectrum dataset source_entries are not canonical")
        source_hashes = [entry["source_file_sha256"] for entry in source_entries]
        decoded_hashes = [entry["decoded_window_sha256"] for entry in source_entries]
        if len(set(source_hashes)) != len(source_hashes):
            raise ValueError("Spectrum dataset repeats a source file SHA-256")
        if len(set(decoded_hashes)) != len(decoded_hashes):
            raise ValueError("Spectrum dataset repeats a decoded calibration window")


@torch.no_grad()
def accumulate_spectrum_dataset(
    video_latent: torch.Tensor,
    *,
    batch_id: str,
    task_family: str,
    checkpoint_fingerprint: str,
    vae_fingerprint: str,
    max_temporal_samples: int,
    previous_dataset: Mapping[str, Any] | None = None,
    dataset_provenance_json: str = "",
    source_entry_json: str = "",
) -> tuple[dict[str, Any], str]:
    batch_name = str(batch_id).strip()
    if not batch_name:
        raise ValueError("batch_id must be non-empty and unique for this dataset")
    task = str(task_family)
    if task not in {"T2VA", "I2VA", "FL2VA", "L2VA", "Ref2VA", "Hybrid"}:
        raise ValueError("Unknown H3 SPEED task family")
    checkpoint_id = _usable_provenance(checkpoint_fingerprint)
    vae_id = _usable_provenance(vae_fingerprint)
    incoming_provenance = _parse_dataset_provenance(dataset_provenance_json)
    incoming_source_entry = _parse_source_entry(source_entry_json, batch_id=batch_name)
    if (incoming_provenance is None) != (incoming_source_entry is None):
        raise ValueError(
            "Formal source binding requires both dataset_provenance_json and source_entry_json"
        )
    per_clip_power, input_contract = _spatial_power_per_clip(
        video_latent, max_temporal_samples=max_temporal_samples
    )
    if input_contract["channels"] != 24:
        raise ValueError("H3 video latent calibration requires exactly 24 channels")
    latent_contract = {
        key: input_contract[key]
        for key in (
            "channels",
            "frames",
            "height",
            "width",
            "max_temporal_samples",
            "temporal_samples",
        )
    }
    clip_fingerprints = [_power_sha256(row) for row in per_clip_power]
    if len(set(clip_fingerprints)) != len(clip_fingerprints):
        raise ValueError("The input batch repeats an identical clip spectrum")

    if previous_dataset is None:
        old_power_sum = torch.zeros(
            (latent_contract["height"], latent_contract["width"]),
            dtype=torch.float64,
            device="cpu",
        )
        old_batch_ids: list[str] = []
        old_clip_fingerprints: list[str] = []
        old_batch_sizes: list[int] = []
        old_source_entries: list[dict[str, Any]] = []
        dataset_provenance = incoming_provenance
    else:
        _validate_spectrum_dataset(previous_dataset)
        for field, expected in (
            ("task_family", task),
            ("checkpoint_fingerprint", checkpoint_id),
            ("vae_fingerprint", vae_id),
        ):
            if previous_dataset.get(field) != expected:
                raise ValueError(f"Spectrum dataset {field} mismatch")
        if dict(previous_dataset["latent_contract"]) != latent_contract:
            raise ValueError("Spectrum dataset latent/settings contract mismatch")
        old_batch_ids = list(previous_dataset["batch_ids"])
        old_clip_fingerprints = list(previous_dataset["clip_fingerprints"])
        old_batch_sizes = list(previous_dataset.get("batch_sizes", []))
        old_power_sum = previous_dataset["power_sum"].clone()
        old_source_entries = list(previous_dataset.get("source_entries", []))
        dataset_provenance = previous_dataset.get("dataset_provenance")
        if dataset_provenance is None and incoming_provenance is not None:
            raise ValueError("Cannot add formal provenance to an already unbound dataset")
        if dataset_provenance is not None:
            if incoming_provenance is None:
                raise ValueError("A provenance-bound dataset requires source metadata on every append")
            if incoming_provenance != dataset_provenance:
                raise ValueError("Spectrum dataset provenance mismatch")
    if batch_name in old_batch_ids:
        raise ValueError(f"Duplicate spectrum dataset batch_id: {batch_name}")
    duplicates = sorted(set(clip_fingerprints).intersection(old_clip_fingerprints))
    if duplicates:
        raise ValueError("The input repeats a clip spectrum already present in the dataset")

    power_sum = old_power_sum.add(per_clip_power.sum(dim=0))
    all_batch_ids = [*old_batch_ids, batch_name]
    all_clip_fingerprints = [*old_clip_fingerprints, *clip_fingerprints]
    all_batch_sizes = [*old_batch_sizes, input_contract["batch"]]
    all_source_entries = (
        [*old_source_entries, incoming_source_entry]
        if incoming_source_entry is not None
        else []
    )
    dataset = {
        "schema": SPEED_SPECTRUM_DATASET_SCHEMA,
        "task_family": task,
        "checkpoint_fingerprint": checkpoint_id,
        "vae_fingerprint": vae_id,
        "latent_contract": latent_contract,
        "independent_clip_count": len(all_clip_fingerprints),
        "batch_count": len(all_batch_ids),
        "batch_ids": all_batch_ids,
        "batch_sizes": all_batch_sizes,
        "clip_fingerprints": all_clip_fingerprints,
        "power_sum": power_sum,
        "power_sum_sha256": _power_sha256(power_sum),
        "resident_cpu_bytes": int(power_sum.numel() * power_sum.element_size()),
        "duplicate_policy": "batch_id_and_exact_per_clip_power_hash_rejected",
        "independence_boundary": (
            "Unique hashes reject exact repeats but cannot prove dataset independence, consent, "
            "content diversity or absence of near-duplicates."
        ),
    }
    if dataset_provenance is not None:
        if input_contract["batch"] != 1:
            raise ValueError("Formal source binding currently requires one clip per batch")
        dataset["dataset_provenance"] = dataset_provenance
        dataset["source_entries"] = all_source_entries
    report = _spectrum_dataset_public_report(dataset)
    report["status"] = "dataset_accumulated"
    return dataset, canonical_json(report)


def finalize_spectrum_dataset(
    dataset: Mapping[str, Any],
    *,
    profile_name: str,
    minimum_r_squared: float,
    minimum_independent_clips: int = 100,
) -> tuple[dict[str, Any], str]:
    _validate_spectrum_dataset(dataset)
    minimum_independent_clips = int(minimum_independent_clips)
    if minimum_independent_clips < 100:
        raise ValueError("minimum_independent_clips cannot be lower than 100")
    minimum_r_squared = float(minimum_r_squared)
    if not math.isfinite(minimum_r_squared) or not 0.0 <= minimum_r_squared <= 1.0:
        raise ValueError("minimum_r_squared must be in [0, 1]")
    clip_count = int(dataset["independent_clip_count"])
    mean_power = dataset["power_sum"] / clip_count
    fit = _fit_spatial_mean_power(
        mean_power,
        latent_contract=dataset["latent_contract"],
        minimum_radius=1,
        maximum_radius_fraction=0.5,
        clip_count=clip_count,
    )
    dataset_fingerprint = hashlib.sha256(
        canonical_json(_spectrum_dataset_public_report(dataset), indent=None).encode(
            "utf-8"
        )
    ).hexdigest().upper()
    enough_clips = clip_count >= minimum_independent_clips
    fit_passed = fit["r_squared"] >= minimum_r_squared
    provenance = dataset.get("dataset_provenance")
    source_entries = dataset.get("source_entries")
    provenance_complete = False
    source_set_matches = False
    if provenance is not None and isinstance(source_entries, list):
        provenance_complete = bool(
            provenance.get("source_kind") == "independent_natural_video_corpus"
            and provenance.get("selection_policy") == "sha256_rank"
            and provenance.get("independence_reviewed") is True
            and provenance.get("content_diversity_reviewed") is True
            and provenance.get("raw_media_redistributed") is False
            and int(provenance.get("selected_source_count", 0)) == clip_count
        )
        source_set_matches = (
            _source_set_sha256(source_entries)
            == provenance.get("selected_source_set_sha256")
        )
    validated = enough_clips and fit_passed and provenance_complete and source_set_matches
    profile = {
        "schema": SPEED_PROFILE_SCHEMA,
        "profile_name": str(profile_name).strip() or "unnamed_h3_dataset_profile",
        "task_family": dataset["task_family"],
        "checkpoint_fingerprint": dataset["checkpoint_fingerprint"],
        "vae_fingerprint": dataset["vae_fingerprint"],
        "independent_clip_count": clip_count,
        "actual_batch_entries": clip_count,
        "declared_evidence_present_in_input": True,
        "provenance_complete": provenance_complete,
        "fit": fit,
        "dataset": {
            "schema": dataset["schema"],
            "fingerprint": dataset_fingerprint,
            "batch_count": int(dataset["batch_count"]),
            "batch_ids": list(dataset["batch_ids"]),
            "latent_contract": dict(dataset["latent_contract"]),
            "power_sum_sha256": dataset["power_sum_sha256"],
            "dataset_provenance": provenance,
            "source_set_sha256": (
                _source_set_sha256(source_entries)
                if isinstance(source_entries, list) and source_entries
                else None
            ),
        },
        "validated_for_delta_optimal": validated,
        "validation_rule": {
            "minimum_independent_clips": minimum_independent_clips,
            "actual_unique_clip_power_hashes_required": True,
            "checkpoint_and_vae_fingerprints_required": True,
            "reviewed_independent_natural_corpus_provenance_required": True,
            "selected_source_set_hash_must_match": True,
            "minimum_r_squared": minimum_r_squared,
        },
        "validation_checks": {
            "enough_unique_clips": enough_clips,
            "fit_r_squared_passed": fit_passed,
            "provenance_complete": provenance_complete,
            "selected_source_set_matches": source_set_matches,
        },
        "status": "dataset_profile" if validated else "research_probe_only",
        "warning": (
            "A validated numerical fit only authorizes delta-optimal schedule research for the "
            "exact task/model/VAE/grid contract. It does not prove quality, audio, reference "
            "adherence, speedup or memory safety. Dataset independence remains a reviewed "
            "provenance assertion rather than something latent hashes can prove."
        ),
    }
    return profile, canonical_json(profile)


def build_spectrum_profile(
    video_latent: torch.Tensor,
    *,
    profile_name: str,
    task_family: str,
    checkpoint_fingerprint: str,
    vae_fingerprint: str,
    independent_clip_count: int,
    minimum_r_squared: float,
    max_temporal_samples: int,
) -> tuple[dict[str, Any], str]:
    fit = fit_h3_spatial_power_spectrum(
        video_latent, max_temporal_samples=max_temporal_samples
    )
    evidence_count = int(independent_clip_count)
    actual_batch_entries = int(fit["latent_shape"][0])
    evidence_is_present = actual_batch_entries >= evidence_count
    checkpoint_id = str(checkpoint_fingerprint).strip()
    vae_id = str(vae_fingerprint).strip()
    provenance_complete = all(
        value and value.lower() not in {"unrecorded", "unknown", "none"}
        for value in (checkpoint_id, vae_id)
    )
    validated = (
        evidence_count >= 100
        and evidence_is_present
        and provenance_complete
        and fit["r_squared"] >= float(minimum_r_squared)
    )
    profile = {
        "schema": SPEED_PROFILE_SCHEMA,
        "profile_name": str(profile_name).strip() or "unnamed_h3_profile",
        "task_family": str(task_family),
        "checkpoint_fingerprint": checkpoint_id,
        "vae_fingerprint": vae_id,
        "independent_clip_count": evidence_count,
        "actual_batch_entries": actual_batch_entries,
        "declared_evidence_present_in_input": evidence_is_present,
        "provenance_complete": provenance_complete,
        "fit": fit,
        "validated_for_delta_optimal": validated,
        "validation_rule": {
            "minimum_independent_clips": 100,
            "declared_count_must_not_exceed_actual_batch_entries": True,
            "checkpoint_and_vae_fingerprints_required": True,
            "minimum_r_squared": float(minimum_r_squared),
        },
        "status": "dataset_profile" if validated else "research_probe_only",
        "warning": (
            "This is not an H3 default until at least 100 actual independent batch entries pass "
            "the dataset rule; declaring a larger evidence count cannot promote one input clip. "
            "WAN/FLUX constants are never substituted."
        ),
    }
    return profile, canonical_json(profile)


def _spectrum_profile_latent_contract(
    profile: Mapping[str, Any],
) -> dict[str, int]:
    dataset = profile.get("dataset")
    dataset_contract = (
        dataset.get("latent_contract") if isinstance(dataset, Mapping) else None
    )
    fit = profile.get("fit")
    latent_shape = fit.get("latent_shape") if isinstance(fit, Mapping) else None

    candidates: list[dict[str, int]] = []
    if isinstance(dataset_contract, Mapping):
        try:
            candidates.append(
                {
                    "channels": int(dataset_contract["channels"]),
                    "frames": int(dataset_contract["frames"]),
                    "height": int(dataset_contract["height"]),
                    "width": int(dataset_contract["width"]),
                }
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("SPEED spectrum dataset latent contract is invalid") from exc
    if isinstance(latent_shape, Sequence) and not isinstance(latent_shape, (str, bytes)):
        if len(latent_shape) != 5:
            raise ValueError("SPEED spectrum fit latent_shape must be [B,C,T,H,W]")
        try:
            candidates.append(
                {
                    "channels": int(latent_shape[1]),
                    "frames": int(latent_shape[2]),
                    "height": int(latent_shape[3]),
                    "width": int(latent_shape[4]),
                }
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("SPEED spectrum fit latent_shape is invalid") from exc
    if not candidates:
        raise ValueError("SPEED spectrum profile is missing its exact latent grid contract")
    if any(value <= 0 for candidate in candidates for value in candidate.values()):
        raise ValueError("SPEED spectrum latent grid values must be positive")
    if any(candidate["channels"] != 24 for candidate in candidates):
        raise ValueError("SPEED spectrum profile is not an H3 24-channel video latent")
    if any(candidate != candidates[0] for candidate in candidates[1:]):
        raise ValueError("SPEED spectrum dataset and fit latent contracts disagree")
    return candidates[0]


def _find_transition_index(sigmas: Sequence[float], threshold: float) -> int:
    for index in range(len(sigmas) - 1):
        if float(sigmas[index]) <= float(threshold):
            return index
    return len(sigmas) - 1


def _requested_ratio(current: Mapping[str, Any], following: Mapping[str, Any]) -> float:
    """Official SPEED uses the requested scale ratio, not the integer-snapped grid ratio."""
    return float(following["requested_scale"]) / float(current["requested_scale"])


def _actual_grid_ratio(current: Mapping[str, Any], following: Mapping[str, Any]) -> float:
    ratio_w = float(following["latent_width"]) / float(current["latent_width"])
    ratio_h = float(following["latent_height"]) / float(current["latent_height"])
    return math.sqrt(ratio_w * ratio_h)


def build_speed_plan(
    *,
    width: int,
    height: int,
    steps: int,
    scales: str,
    transition_mode: str,
    manual_transition_sigmas: str,
    delta: float,
    shift_video: float,
    transform: str,
    profile_policy: str,
    fallback_policy: str,
    spectrum_profile: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], str]:
    if transform != "dct":
        raise ValueError("The H3 SPEED Advanced runtime currently implements official DCT only")
    if int(steps) < 2:
        raise ValueError("SPEED requires at least two denoising steps")
    if not math.isfinite(shift_video) or shift_video <= 0.0:
        raise ValueError("shift_video must be finite and greater than zero")
    if fallback_policy not in {"error", "full_resolution_passthrough"}:
        raise ValueError("Unknown SPEED fallback policy")
    parsed_scales = validate_scales(parse_float_list(scales, name="scales"))
    stages = resolve_stage_shapes(int(width), int(height), parsed_scales)
    full_sigmas = [float(value) for value in native_flow_sigmas(int(steps), float(shift_video))]

    thresholds: list[float] = []
    profile_summary: dict[str, Any] | None = None
    if len(stages) > 1 and transition_mode == "manual_sigmas":
        thresholds = parse_float_list(
            manual_transition_sigmas, name="manual_transition_sigmas"
        )
        if len(thresholds) != len(stages) - 1:
            raise ValueError("manual_transition_sigmas needs one value per resolution transition")
        if any(not 0.0 < value < 1.0 for value in thresholds):
            raise ValueError("Every manual transition sigma must be in (0, 1)")
        if any(a <= b for a, b in zip(thresholds, thresholds[1:])):
            raise ValueError("Manual transition sigmas must be strictly decreasing")
    elif len(stages) > 1 and transition_mode == "delta_optimal":
        if not isinstance(spectrum_profile, Mapping) or spectrum_profile.get("schema") != SPEED_PROFILE_SCHEMA:
            raise ValueError("delta_optimal requires an H3 SPEED Spectrum Profile")
        if (
            profile_policy == "require_validated_profile"
            and not spectrum_profile.get("validated_for_delta_optimal", False)
        ):
            raise ValueError(
                "This spectrum profile is only a research probe; choose manual sigmas or "
                "explicitly allow the research profile"
            )
        fit = spectrum_profile.get("fit", {})
        amplitude = float(fit["amplitude"])
        beta = float(fit["beta"])
        latent_contract = _spectrum_profile_latent_contract(spectrum_profile)
        requested_spatial_grid = {
            "height": int(height) // 16,
            "width": int(width) // 16,
        }
        profile_spatial_grid = {
            "height": latent_contract["height"],
            "width": latent_contract["width"],
        }
        if profile_spatial_grid != requested_spatial_grid:
            raise ValueError(
                "SPEED spectrum latent grid mismatch: "
                f"profile={profile_spatial_grid}, plan={requested_spatial_grid}"
            )
        omega_max = min(stages[-1]["latent_height"], stages[-1]["latent_width"]) / 2.0
        for stage in stages[:-1]:
            omega = float(stage["effective_scale"]) * omega_max
            thresholds.append(activation_time(power_spectrum(omega, amplitude, beta), delta))
        profile_summary = {
            "profile_name": spectrum_profile.get("profile_name"),
            "profile_status": spectrum_profile.get("status"),
            "task_family": spectrum_profile.get("task_family"),
            "amplitude": amplitude,
            "beta": beta,
            "r_squared": fit.get("r_squared"),
            "checkpoint_fingerprint": spectrum_profile.get("checkpoint_fingerprint"),
            "vae_fingerprint": spectrum_profile.get("vae_fingerprint"),
            "latent_contract": latent_contract,
        }
    elif len(stages) > 1:
        raise ValueError("transition_mode must be manual_sigmas or delta_optimal")

    transitions: list[dict[str, Any]] = []
    previous_index = 0
    for index, (threshold, current, following) in enumerate(
        zip(thresholds, stages[:-1], stages[1:])
    ):
        step_index = _find_transition_index(full_sigmas, threshold)
        if step_index <= previous_index or step_index >= int(steps):
            raise ValueError(
                "SPEED transition thresholds collapse onto invalid/non-increasing step indices; "
                "adjust the sigmas, steps, scales, or spectrum profile"
            )
        sigma = full_sigmas[step_index]
        ratio = _requested_ratio(current, following)
        actual_grid_ratio = _actual_grid_ratio(current, following)
        aligned = align_sigma(sigma, ratio)
        if not sigma < aligned <= 1.0:
            raise ValueError("SPEED sigma alignment produced an invalid transition")
        transitions.append(
            {
                "index": index,
                "from_stage": index,
                "to_stage": index + 1,
                "step_index": step_index,
                "threshold": threshold,
                "sigma": sigma,
                "ratio": ratio,
                "actual_grid_ratio": actual_grid_ratio,
                "ratio_source": "requested_scale_ratio_per_official_speed",
                "kappa": kappa(sigma, ratio),
                "aligned_sigma": aligned,
            }
        )
        previous_index = step_index

    stage_segments: list[dict[str, Any]] = []
    segment_start = 0
    for stage_index in range(len(stages)):
        if stage_index < len(transitions):
            segment_end = int(transitions[stage_index]["step_index"])
            segment_sigmas = full_sigmas[segment_start : segment_end + 1]
        else:
            segment_end = int(steps)
            if stage_index == 0:
                segment_sigmas = full_sigmas
            else:
                aligned_start = float(transitions[stage_index - 1]["aligned_sigma"])
                segment_sigmas = [aligned_start, *full_sigmas[segment_start:]]
        if stage_index > 0 and stage_index < len(stages) - 1:
            aligned_start = float(transitions[stage_index - 1]["aligned_sigma"])
            segment_sigmas = [aligned_start, *full_sigmas[segment_start : segment_end + 1]]
        if len(segment_sigmas) < 2:
            raise ValueError("Every SPEED stage must contain at least one model evaluation")
        stage_segments.append(
            {
                "stage_index": stage_index,
                "start_step": segment_start,
                "end_step": segment_end,
                "sigmas": segment_sigmas,
                "nfe": len(segment_sigmas) - 1,
            }
        )
        if stage_index < len(transitions):
            segment_start = int(transitions[stage_index]["step_index"]) + 1

    total_nfe = sum(segment["nfe"] for segment in stage_segments)
    if total_nfe != int(steps):
        raise RuntimeError(f"SPEED stage schedule changed NFE: expected {steps}, got {total_nfe}")
    plan = {
        "schema": SPEED_PLAN_SCHEMA,
        "status": "full_resolution_passthrough" if len(stages) == 1 else "planned",
        "width": int(width),
        "height": int(height),
        "steps": int(steps),
        "shift_video": float(shift_video),
        "transform": transform,
        "transition_mode": transition_mode,
        "delta": float(delta),
        "profile_policy": profile_policy,
        "fallback_policy": fallback_policy,
        "stages": stages,
        "transitions": transitions,
        "segments": stage_segments,
        "full_sigmas": full_sigmas,
        "profile": profile_summary,
        "nfe": total_nfe,
        "official_method": {
            "paper": OFFICIAL_SPEED_PAPER,
            "source_commit": OFFICIAL_SPEED_COMMIT,
            "spatial_only": True,
            "temporal_resolution_unchanged": True,
            "wan_constants_reused": False,
        },
        "warnings": [
            "H3 GPU quality, speed, VRAM, and audio non-inferiority are not yet validated.",
            "Aligned audio transport is an H3-specific joint-flow extension, not a claim from SPEED.",
        ],
    }
    return plan, canonical_json(plan)


def build_speed_source(**kwargs) -> tuple[dict[str, Any], str]:
    ref_images = kwargs.get("ref_images")
    ref_videos = kwargs.get("ref_videos")
    ref_audios = kwargs.get("ref_audios")
    has_refs = bool(
        sorted_autogrow_values(ref_images)
        or sorted_autogrow_values(ref_videos)
        or sorted_autogrow_values(ref_audios)
        or sorted_autogrow_values(kwargs.get("ref_video_audios"))
        or (kwargs.get("drive_audio") is not None and kwargs.get("add_source_as_reference"))
    )
    resolved_task = resolve_task_type(
        kwargs.get("task_type", "auto"),
        kwargs.get("first_frame"),
        kwargs.get("last_frame"),
        has_refs,
    )
    source = {"schema": SPEED_SOURCE_SCHEMA, **kwargs, "resolved_task": resolved_task}
    report = {
        "schema": SPEED_SOURCE_SCHEMA,
        "resolved_task": resolved_task,
        "length": int(kwargs["length"]),
        "audio_mode": kwargs["audio_mode"],
        "checkpoint_fingerprint": kwargs.get("checkpoint_fingerprint", "unrecorded"),
        "vae_fingerprint": kwargs.get("vae_fingerprint", "unrecorded"),
        "has_first_frame": kwargs.get("first_frame") is not None,
        "has_last_frame": kwargs.get("last_frame") is not None,
        "reference_counts": {
            "images": len(sorted_autogrow_values(ref_images)),
            "videos": len(sorted_autogrow_values(ref_videos)),
            "audios": len(sorted_autogrow_values(ref_audios)),
        },
        "note": "Raw inputs are retained so each SPEED stage can rebuild H3 conditioning at its own canvas.",
    }
    return source, canonical_json(report)


_DCT_BASIS_CACHE: OrderedDict[int, torch.Tensor] = OrderedDict()


def clear_speed_math_cache() -> None:
    _DCT_BASIS_CACHE.clear()


def _dct_basis_cpu(size: int) -> torch.Tensor:
    size = int(size)
    cached = _DCT_BASIS_CACHE.get(size)
    if cached is not None:
        _DCT_BASIS_CACHE.move_to_end(size)
        return cached
    n = torch.arange(size, dtype=torch.float64)
    k = torch.arange(size, dtype=torch.float64)[:, None]
    basis = math.sqrt(2.0 / size) * torch.cos(math.pi / size * (n + 0.5) * k)
    basis[0] /= math.sqrt(2.0)
    _DCT_BASIS_CACHE[size] = basis
    while len(_DCT_BASIS_CACHE) > 8:
        _DCT_BASIS_CACHE.popitem(last=False)
    return basis


def _dct2(value: torch.Tensor, basis_h: torch.Tensor, basis_w: torch.Tensor) -> torch.Tensor:
    return torch.matmul(torch.matmul(basis_h, value), basis_w.transpose(-1, -2))


def _idct2(value: torch.Tensor, basis_h: torch.Tensor, basis_w: torch.Tensor) -> torch.Tensor:
    return torch.matmul(torch.matmul(basis_h.transpose(-1, -2), value), basis_w)


@torch.no_grad()
def dct_expand_official(
    value: torch.Tensor,
    target_height: int,
    target_width: int,
    *,
    sigma: float,
    ratio: float,
    seed: int,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, float, dict[str, Any]]:
    """Clean-room torch implementation of official SPEED DCT expansion.

    The last two axes are spatial. Existing orthonormal DCT coefficients occupy
    the upper-left low-frequency block; all new coefficients are N(0, sigma^2),
    followed by the official kappa state rescale.
    """
    if not isinstance(value, torch.Tensor) or value.ndim < 2:
        raise ValueError("DCT expansion requires a tensor with two spatial axes")
    source_height, source_width = map(int, value.shape[-2:])
    target_height = int(target_height)
    target_width = int(target_width)
    if target_height < source_height or target_width < source_width:
        raise ValueError("DCT expansion cannot shrink either spatial axis")
    if int(chunk_size) < 1:
        raise ValueError("chunk_size must be at least 1")
    if target_height == source_height and target_width == source_width:
        return value.clone(), align_sigma(sigma, ratio), {
            "source_hw": [source_height, source_width],
            "target_hw": [target_height, target_width],
            "chunk_size": int(chunk_size),
            "new_coefficients": 0,
        }

    original_dtype = value.dtype
    device = value.device
    flattened = value.reshape(-1, source_height, source_width)
    result = torch.empty(
        (flattened.shape[0], target_height, target_width),
        device=device,
        dtype=torch.float32,
    )
    source_h_basis = _dct_basis_cpu(source_height).to(device=device, dtype=torch.float32)
    source_w_basis = _dct_basis_cpu(source_width).to(device=device, dtype=torch.float32)
    target_h_basis = _dct_basis_cpu(target_height).to(device=device, dtype=torch.float32)
    target_w_basis = _dct_basis_cpu(target_width).to(device=device, dtype=torch.float32)
    generator = torch.Generator(device="cpu").manual_seed(int(seed) & 0xFFFFFFFFFFFFFFFF)
    state_scale = kappa(float(sigma), float(ratio))
    for start in range(0, flattened.shape[0], int(chunk_size)):
        stop = min(flattened.shape[0], start + int(chunk_size))
        source = flattened[start:stop].to(dtype=torch.float32)
        low = _dct2(source, source_h_basis, source_w_basis)
        coefficients = torch.randn(
            (stop - start, target_height, target_width),
            generator=generator,
            dtype=torch.float32,
            device="cpu",
        ).to(device=device)
        coefficients.mul_(float(sigma))
        coefficients[..., :source_height, :source_width] = low
        expanded = _idct2(coefficients, target_h_basis, target_w_basis)
        result[start:stop] = expanded * state_scale
        del source, low, coefficients, expanded
    output = result.reshape(*value.shape[:-2], target_height, target_width).to(original_dtype)
    report = {
        "source_hw": [source_height, source_width],
        "target_hw": [target_height, target_width],
        "chunk_size": int(chunk_size),
        "leading_slices": int(flattened.shape[0]),
        "new_coefficients": int(
            flattened.shape[0]
            * (target_height * target_width - source_height * source_width)
        ),
        "sigma": float(sigma),
        "ratio": float(ratio),
        "kappa": state_scale,
        "aligned_sigma": align_sigma(float(sigma), float(ratio)),
    }
    del result, source_h_basis, source_w_basis, target_h_basis, target_w_basis
    clear_speed_math_cache()
    return output, report["aligned_sigma"], report


def reindex_joint_audio_state(
    state: torch.Tensor,
    target: torch.Tensor,
    *,
    sigma_from: float,
    sigma_to: float,
) -> torch.Tensor:
    """Move audio on the same public H3 flow line without spatial noise expansion."""
    if state.shape != target.shape:
        raise ValueError("Audio state and target must have identical shapes")
    if not 0.0 < sigma_from <= 1.0 or not sigma_from <= sigma_to <= 1.0:
        raise ValueError("Audio reindex requires 0 < sigma_from <= sigma_to <= 1")
    return target + (sigma_to / sigma_from) * (state - target)


def recover_raw_flow_state(
    external_output: Any,
    *,
    sigma: float,
    audio_scale: float,
) -> Any:
    video, audio = tuple(external_output.unbind())
    factor = 1.0 - float(sigma)
    return comfy.nested_tensor.NestedTensor((video * factor, audio * float(audio_scale) * factor))


def solve_segment_noise(
    desired_raw_state: Any,
    target_external: Any,
    *,
    sigma: float,
    audio_scale: float,
    noise_scale: float,
) -> Any:
    desired_video, desired_audio = tuple(desired_raw_state.unbind())
    target_video, target_audio = tuple(target_external.unbind())
    denominator = float(sigma) * float(noise_scale)
    if denominator <= 0.0:
        raise ValueError("Segment start sigma and noise_scale must be greater than zero")
    video_noise = (desired_video - (1.0 - sigma) * target_video) / denominator
    audio_target_scaled = target_audio * float(audio_scale)
    audio_noise = (desired_audio - (1.0 - sigma) * audio_target_scaled) / denominator
    return comfy.nested_tensor.NestedTensor((video_noise, audio_noise))


def modality_stable_h3_noise(input_latent: Mapping[str, Any], seed: int) -> Any:
    """Generate AV noise whose audio stream is invariant to video canvas size."""
    import comfy.sample

    samples = input_latent.get("samples")
    if samples is None or not getattr(samples, "is_nested", False):
        raise ValueError("Modality-stable H3 noise requires a nested AV LATENT")
    video, audio = _nested_parts(samples)
    batch_indices = input_latent.get("batch_index")
    video_seed = int(seed) & 0xFFFFFFFFFFFFFFFF
    audio_seed = video_seed ^ SPEED_AUDIO_NOISE_SEED_XOR
    video_noise = comfy.sample.prepare_noise(video, video_seed, batch_indices)
    audio_noise = comfy.sample.prepare_noise(audio, audio_seed, batch_indices)
    return comfy.nested_tensor.NestedTensor((video_noise, audio_noise))


class H3ModalityStableNoise:
    def __init__(self, seed: int):
        self.seed = int(seed) & 0xFFFFFFFFFFFFFFFF

    def generate_noise(self, input_latent):
        return modality_stable_h3_noise(input_latent, self.seed)


def _release_h3_residency_between_stages(model) -> dict[str, Any]:
    """Target only the active H3 clone family before a larger SPEED canvas loads.

    DynamicVRAM can retain weight pages chosen for the cheap low-resolution stage.
    Those pages leave too little activation headroom when the next stage grows.  The
    public ComfyUI helper unloads the supplied patcher and its clones without
    evicting unrelated CLIP/VAE models; this is deliberately not a global unload.
    """
    from comfy import model_management as comfy_model_management

    unload = getattr(comfy_model_management, "unload_model_and_clones", None)
    if not callable(unload):
        return {
            "performed": False,
            "scope": "selected_h3_model_and_clones",
            "global_unload_called": False,
            "reason": "current ComfyUI has no targeted unload_model_and_clones API",
        }
    device = getattr(model, "load_device", None)
    before = None
    if device is not None:
        try:
            before = int(comfy_model_management.get_free_memory(device))
        except Exception:
            before = None
    unload(
        model,
        unload_additional_models=False,
        all_devices=False,
    )
    comfy_model_management.soft_empty_cache()
    after = None
    if device is not None:
        try:
            after = int(comfy_model_management.get_free_memory(device))
        except Exception:
            after = None
    return {
        "performed": True,
        "scope": "selected_h3_model_and_clones",
        "global_unload_called": False,
        "unload_additional_models": False,
        "all_devices": False,
        "free_memory_before_bytes": before,
        "free_memory_after_bytes": after,
        "free_memory_delta_bytes": (
            after - before if before is not None and after is not None else None
        ),
    }


def _set_comfy_reserved_bytes(model_management, value: int) -> str:
    setter = getattr(model_management, "set_extra_reserved_vram", None)
    if callable(setter):
        setter(int(value) / 1024**3)
        return "set_extra_reserved_vram_gib"
    model_management.EXTRA_RESERVED_VRAM = int(value)
    return "EXTRA_RESERVED_VRAM_bytes"


def _speed_dynamic_headroom_control() -> tuple[Any | None, int, str]:
    """Return the AIMDO setter and the best available previous value."""
    try:
        import comfy.memory_management as memory_management

        if not bool(getattr(memory_management, "aimdo_enabled", False)):
            return None, 0, "dynamic_vram_not_enabled"
        import comfy_aimdo.control as control

        setter = getattr(getattr(control, "lib", None), "set_simple_vram_headroom", None)
        if not callable(setter):
            return None, 0, "dynamic_vram_setter_unavailable"
        try:
            from . import vram_policy

            previous = getattr(vram_policy, "_LAST_SIMPLE_HEADROOM_BYTES", None)
        except Exception:
            previous = None
        if previous is None:
            try:
                from comfy.cli_args import args

                previous = int(float(getattr(args, "vram_headroom", 0.0) or 0.0) * 1024**3)
            except Exception:
                previous = 0
        return setter, int(previous), "direct_lib.set_simple_vram_headroom"
    except Exception as error:
        return None, 0, f"unavailable:{type(error).__name__}"


def _apply_speed_scoped_headroom() -> tuple[dict[str, Any], dict[str, Any]]:
    """Raise headroom only for this SPEED execution and return a restoration token."""
    from comfy import model_management as comfy_model_management

    getter = getattr(comfy_model_management, "extra_reserved_memory", None)
    previous_reserved = int(
        getter() if callable(getter) else comfy_model_management.EXTRA_RESERVED_VRAM
    )
    target = max(previous_reserved, SPEED_SCOPED_HEADROOM_BYTES)
    comfy_route = _set_comfy_reserved_bytes(comfy_model_management, target)
    dynamic_setter, previous_dynamic, dynamic_route = _speed_dynamic_headroom_control()
    dynamic_applied = dynamic_setter is not None
    if dynamic_applied:
        dynamic_setter(target)
    token = {
        "model_management": comfy_model_management,
        "previous_reserved_bytes": previous_reserved,
        "dynamic_setter": dynamic_setter,
        "previous_dynamic_bytes": previous_dynamic,
        "restored": False,
    }
    report = {
        "applied": target > previous_reserved or dynamic_applied,
        "temporary": True,
        "target_bytes": target,
        "previous_comfy_reserved_bytes": previous_reserved,
        "previous_dynamic_headroom_bytes": previous_dynamic if dynamic_applied else None,
        "comfy_route": comfy_route,
        "dynamic_route": dynamic_route,
        "global_model_unload_called": False,
    }
    return token, report


def _restore_speed_scoped_headroom(token: dict[str, Any] | None) -> dict[str, Any]:
    if not token or token.get("restored"):
        return {"restored": bool(token and token.get("restored"))}
    model_management = token["model_management"]
    _set_comfy_reserved_bytes(model_management, token["previous_reserved_bytes"])
    dynamic_setter = token.get("dynamic_setter")
    if callable(dynamic_setter):
        dynamic_setter(token["previous_dynamic_bytes"])
    token["restored"] = True
    return {
        "restored": True,
        "comfy_reserved_bytes": token["previous_reserved_bytes"],
        "dynamic_headroom_bytes": (
            token["previous_dynamic_bytes"] if callable(dynamic_setter) else None
        ),
    }


def _source_conditioning_kwargs(source: Mapping[str, Any], width: int, height: int) -> dict[str, Any]:
    keys = (
        "task_type",
        "audio_mode",
        "audio_denoise_strength",
        "add_source_as_reference",
        "prompt_primary_audio_ordinal",
        "strict_prompt_tags",
        "ref_image_size",
        "reference_video_policy",
        "drive_audio",
        "final_audio",
        "first_frame",
        "last_frame",
        "ref_images",
        "ref_videos",
        "ref_video_audios",
        "ref_audios",
    )
    return {
        "clip": source["clip"],
        "video_vae": source["video_vae"],
        "audio_vae": source["audio_vae"],
        "prompt": source["prompt"],
        "width": int(width),
        "height": int(height),
        "length": int(source["length"]),
        **{key: source.get(key) for key in keys},
    }


def _ensure_native_h3_model(model) -> None:
    diffusion_model = getattr(getattr(model, "model", None), "diffusion_model", None)
    if not isinstance(diffusion_model, MiniMaxH3Model):
        raise ValueError("SPEED Advanced requires a native ComfyUI MiniMax H3 MODEL")
    transformer = model.model_options.get("transformer_options", {})
    if transformer.get("wrappers"):
        raise ValueError("SPEED Advanced refuses Transformer wrappers")
    if transformer.get("callbacks"):
        raise ValueError("SPEED Advanced refuses Transformer callbacks")
    if transformer.get("patches"):
        raise ValueError("SPEED Advanced refuses Transformer patches")
    replacements = transformer.get("patches_replace", {}).get("dit", {})
    if replacements:
        raise ValueError(
            "SPEED Advanced refuses existing DiT block replacements (BlockCache/STG/ActivationChunk); "
            "run an isolated SPEED workflow"
        )
    forbidden = [
        key
        for key in (
            "sampler_post_cfg_function",
            "sampler_pre_cfg_function",
            "model_function_wrapper",
        )
        if model.model_options.get(key)
    ]
    if forbidden:
        raise ValueError("SPEED Advanced refuses conflicting MODEL wrappers: " + ", ".join(forbidden))
    base_model = getattr(model, "model", None)
    extra_conds = getattr(base_model, "extra_conds", None)
    forward = getattr(getattr(base_model, "diffusion_model", None), "forward", None)
    extra_fn = getattr(extra_conds, "__func__", extra_conds)
    forward_fn = getattr(forward, "__func__", forward)
    scoped_markers = {
        "long_video": getattr(extra_fn, "_t8_long_video_patch_version", None),
        "multikeyframe_extra_conds": getattr(
            extra_fn, "_t8_multikeyframe_patch_version", None
        ),
        "multikeyframe_forward": getattr(
            forward_fn, "_t8_multikeyframe_patch_version", None
        ),
    }
    active_markers = sorted(name for name, value in scoped_markers.items() if value is not None)
    if active_markers:
        raise ValueError(
            "SPEED Advanced refuses scoped MODEL patches: " + ", ".join(active_markers)
        )


def _task_support(
    source: Mapping[str, Any],
    execution_scope: str,
    steps: int,
    shift_video: float,
    shift_audio: float,
) -> tuple[bool, str]:
    task = str(source.get("resolved_task", "unknown"))
    audio_mode = str(source.get("audio_mode", "unknown"))
    if execution_scope == "strict_t2va_stock20":
        conditioning_media = bool(
            source.get("first_frame") is not None
            or source.get("last_frame") is not None
            or sorted_autogrow_values(source.get("ref_images"))
            or sorted_autogrow_values(source.get("ref_videos"))
            or sorted_autogrow_values(source.get("ref_video_audios"))
            or sorted_autogrow_values(source.get("ref_audios"))
            or (
                source.get("drive_audio") is not None
                and bool(source.get("add_source_as_reference"))
            )
        )
        supported = (
            task == "t2va"
            and audio_mode == "native"
            and int(steps) == 20
            and math.isclose(float(shift_video), 12.0, rel_tol=0.0, abs_tol=1e-7)
            and math.isclose(float(shift_audio), 3.0, rel_tol=0.0, abs_tol=1e-7)
            and not conditioning_media
        )
        return (
            supported,
            "strict P1 requires media-free T2VA + native audio + exactly 20 steps + shifts 12/3",
        )
    if execution_scope == "turbo8_t2va_research_exp":
        conditioning_media = bool(
            source.get("first_frame") is not None
            or source.get("last_frame") is not None
            or sorted_autogrow_values(source.get("ref_images"))
            or sorted_autogrow_values(source.get("ref_videos"))
            or sorted_autogrow_values(source.get("ref_video_audios"))
            or sorted_autogrow_values(source.get("ref_audios"))
            or source.get("drive_audio") is not None
        )
        supported = (
            task == "t2va"
            and audio_mode == "native"
            and int(steps) == 8
            and math.isclose(float(shift_video), 12.0, rel_tol=0.0, abs_tol=1e-7)
            and math.isclose(float(shift_audio), 3.0, rel_tol=0.0, abs_tol=1e-7)
            and not conditioning_media
        )
        return (
            supported,
            "Turbo8 research requires media-free T2VA + native audio + exactly 8 steps + "
            "shifts 12/3 and a separately verified compatible Turbo LoRA",
        )
    if execution_scope == "multimodal_research_exp":
        return True, "multimodal mechanics implemented; GPU quality/audio validation pending"
    raise ValueError("Unknown SPEED execution_scope")


def _weight_patch_contract(model, execution_scope: str) -> tuple[bool, str, dict[str, Any]]:
    has_weight_patches = bool(getattr(model, "patches", {}))
    report = {
        "has_weight_patches": has_weight_patches,
        "lora_identity_verified_by_runtime": False,
        "scope": execution_scope,
    }
    if execution_scope == "strict_t2va_stock20" and has_weight_patches:
        return (
            False,
            "strict P1 refuses LoRA/weight-patched models; use an unpatched stock H3 model",
            report,
        )
    if execution_scope == "turbo8_t2va_research_exp" and not has_weight_patches:
        return (
            False,
            "Turbo8 research requires a weight-patched MODEL from a compatible Turbo LoRA; "
            "the runtime cannot infer LoRA identity from patch tensors alone",
            report,
        )
    return True, "weight patch presence matches the selected execution scope", report


def _profile_binding(
    plan: Mapping[str, Any], source: Mapping[str, Any]
) -> dict[str, Any]:
    if plan.get("transition_mode") != "delta_optimal":
        return {"required": False, "status": "not_applicable_manual_schedule"}
    profile = plan.get("profile")
    if not isinstance(profile, Mapping):
        raise ValueError("delta_optimal SPEED plan lost its spectrum profile binding")
    expected_task = str(profile.get("task_family", "")).strip().lower()
    actual_task = str(source.get("resolved_task", "")).strip().lower()
    if expected_task != actual_task:
        raise ValueError(
            f"SPEED spectrum task mismatch: profile={expected_task!r}, source={actual_task!r}"
        )
    latent_contract = profile.get("latent_contract")
    if not isinstance(latent_contract, Mapping):
        raise ValueError("delta_optimal SPEED plan lost its latent grid binding")
    try:
        expected_grid = {
            "channels": int(latent_contract["channels"]),
            "frames": int(latent_contract["frames"]),
            "height": int(latent_contract["height"]),
            "width": int(latent_contract["width"]),
        }
        frame_count = align_frame_count(int(source["length"]))
        actual_grid = {
            "channels": 24,
            "frames": video_latent_t(frame_count),
            "height": int(plan["height"]) // 16,
            "width": int(plan["width"]) // 16,
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("SPEED runtime cannot resolve the exact latent grid binding") from exc
    if expected_grid != actual_grid:
        raise ValueError(
            "SPEED spectrum latent grid mismatch: "
            f"profile={expected_grid}, runtime={actual_grid}"
        )
    fields = ("checkpoint_fingerprint", "vae_fingerprint")

    def normalize_fingerprint(value: Any) -> str:
        text = str(value).strip()
        prefix, separator, digest = text.partition(":")
        if (
            separator
            and prefix.lower() == "sha256"
            and len(digest) == 64
            and all(character in "0123456789abcdefABCDEF" for character in digest)
        ):
            return f"sha256:{digest.lower()}"
        return text

    expected = {
        field: normalize_fingerprint(profile.get(field, "")) for field in fields
    }
    actual = {field: normalize_fingerprint(source.get(field, "")) for field in fields}

    def known(value: str) -> bool:
        return bool(value) and value.lower() not in {"unrecorded", "unknown", "none"}

    mismatches = [
        field
        for field in fields
        if known(expected[field]) and known(actual[field]) and expected[field] != actual[field]
    ]
    missing = []
    unequal = []
    if plan.get("profile_policy") == "require_validated_profile":
        missing = [field for field in fields if not known(actual[field])]
        unequal = [field for field in fields if expected[field] != actual[field]]
    return {
        "required": True,
        "status": (
            "matched"
            if all(known(actual[field]) and actual[field] == expected[field] for field in fields)
            else "research_unrecorded_runtime_fingerprint"
        ),
        "task_family": actual_task,
        "latent_grid": actual_grid,
        "expected": expected,
        "actual": actual,
        "fingerprint_mismatches": mismatches,
        "required_profile_missing_fingerprints": missing,
        "required_profile_unequal_fingerprints": unequal,
        "model_identity_policy": "diagnostic_only_not_a_runtime_gate",
    }


@dataclass
class StageConditioning:
    positive: Any
    latent: dict[str, Any]
    mux_audio: Any
    conditioned_prompt: str
    media_map: str
    report: str
    route: str = "full_conditioning_rebuild"


def _build_stage(source: Mapping[str, Any], width: int, height: int) -> StageConditioning:
    result = build_conditioning(**_source_conditioning_kwargs(source, width, height))
    return StageConditioning(*result)


def _build_empty_t2va_stage(
    source: Mapping[str, Any],
    width: int,
    height: int,
    template: StageConditioning,
) -> StageConditioning:
    latent, _ = empty_av_latent(width, height, int(source["length"]))
    return StageConditioning(
        positive=template.positive,
        latent=latent,
        mux_audio=template.mux_audio,
        conditioned_prompt=template.conditioned_prompt,
        media_map=template.media_map,
        report=template.report + f"\nSPEED text conditioning reused at canvas={width}x{height}",
        route="reused_t2va_text_plus_stage_empty_av",
    )


def _nested_parts(value: Any) -> tuple[torch.Tensor, torch.Tensor]:
    parts = tuple(value.unbind())
    if len(parts) != 2:
        raise ValueError("Expected a two-stream H3 nested latent")
    return parts[0], parts[1]


@torch.no_grad()
def execute_speed_sampling(
    model,
    speed_plan: Mapping[str, Any],
    speed_source: Mapping[str, Any],
    *,
    shift_audio: float,
    seed: int,
    execution_scope: str,
    dct_chunk_size: int,
) -> tuple[dict[str, Any], Any, str, str, str]:
    if speed_plan.get("schema") != SPEED_PLAN_SCHEMA:
        raise ValueError("speed_plan is not a T8 H3 SPEED plan")
    if speed_source.get("schema") != SPEED_SOURCE_SCHEMA:
        raise ValueError("speed_source is not a T8 H3 SPEED source")
    _ensure_native_h3_model(model)
    profile_binding = _profile_binding(speed_plan, speed_source)
    steps = int(speed_plan["steps"])
    shift_video = float(speed_plan["shift_video"])
    supported, support_reason = _task_support(
        speed_source,
        execution_scope,
        steps,
        shift_video,
        float(shift_audio),
    )
    patch_supported, patch_reason, weight_patch_report = _weight_patch_contract(
        model, execution_scope
    )
    if not patch_supported:
        supported = False
        support_reason = patch_reason
    fallback = speed_plan["fallback_policy"]
    if not supported and fallback == "error":
        raise ValueError(support_reason)

    from comfy_extras.nodes_custom_sampler import Guider_Basic

    stages = list(speed_plan["stages"])
    segments = list(speed_plan["segments"])
    transitions = list(speed_plan["transitions"])
    if not supported or speed_plan.get("status") == "full_resolution_passthrough":
        stages = [stages[-1]]
        transitions = []
        segments = [
            {
                "stage_index": 0,
                "start_step": 0,
                "end_step": steps,
                "sigmas": speed_plan["full_sigmas"],
                "nfe": steps,
            }
        ]

    headroom_token = None
    headroom_report = None
    output_nested = None
    pending_noise = None
    pending_stage: StageConditioning | None = None
    stage_records: list[dict[str, Any]] = []
    final_stage: StageConditioning | None = None
    initial_residency_release = None
    try:
        headroom_token, headroom_report = _apply_speed_scoped_headroom()
        initial_residency_release = _release_h3_residency_between_stages(model)
        for stage_index, (shape, segment) in enumerate(zip(stages, segments)):
            stage = pending_stage or _build_stage(
                speed_source, shape["width"], shape["height"]
            )
            pending_stage = None
            final_stage = stage
            video, audio = nested_av_parts(stage.latent)
            stage_model, sampler, reference_sigmas = setup_dual_clock_sampling(
                model,
                stage.latent,
                steps,
                shift_video,
                float(shift_audio),
                "euler",
                "native_flow",
            )
            if not torch.allclose(
                reference_sigmas.cpu(),
                torch.tensor(speed_plan["full_sigmas"], dtype=torch.float32),
                atol=1e-7,
                rtol=0.0,
            ):
                raise RuntimeError("Native H3 sigma construction changed since the SPEED plan was built")
            model_sampling = stage_model.get_model_object("model_sampling")
            audio_scale = float(getattr(model_sampling, "audio_scale", 1.0))
            noise_scale = float(getattr(model_sampling, "noise_scale", 1.0))
            guider = Guider_Basic(stage_model)
            guider.set_conds(stage.positive)
            if stage_index == 0:
                noise = modality_stable_h3_noise(stage.latent, int(seed))
            else:
                noise = pending_noise
                pending_noise = None
            if noise is None:
                raise RuntimeError("SPEED did not construct the next stage noise state")
            segment_sigmas = torch.tensor(segment["sigmas"], dtype=torch.float32)
            denoise_mask = stage.latent.get("noise_mask")
            output_nested = guider.sample(
                noise,
                stage.latent["samples"],
                sampler,
                segment_sigmas,
                denoise_mask=denoise_mask,
                callback=None,
                disable_pbar=False,
                seed=int(seed),
            )
            record: dict[str, Any] = {
                "stage_index": stage_index,
                "canvas": [shape["width"], shape["height"]],
                "video_shape": list(video.shape),
                "audio_shape": list(audio.shape),
                "segment_sigmas": [float(value) for value in segment["sigmas"]],
                "nfe": int(segment["nfe"]),
                "conditioning_report": stage.report,
                "conditioning_route": stage.route,
                "audio_scale": audio_scale,
                "noise_scale": noise_scale,
            }
            has_transition = stage_index < len(transitions)
            if has_transition:
                transition = transitions[stage_index]
                sigma_from = float(transition["sigma"])
                desired_raw = recover_raw_flow_state(
                    output_nested, sigma=sigma_from, audio_scale=audio_scale
                )
                raw_video, raw_audio = _nested_parts(desired_raw)
                next_shape = stages[stage_index + 1]
                expanded_video, sigma_to, dct_report = dct_expand_official(
                    raw_video,
                    next_shape["latent_height"],
                    next_shape["latent_width"],
                    sigma=sigma_from,
                    ratio=float(transition["ratio"]),
                    seed=int(seed) + (stage_index + 1) * 10_000,
                    chunk_size=int(dct_chunk_size),
                )
                if not math.isclose(
                    sigma_to, float(transition["aligned_sigma"]), rel_tol=0.0, abs_tol=1e-7
                ):
                    raise RuntimeError("DCT transition sigma disagrees with the frozen SPEED plan")
                if supported and execution_scope in {
                    "strict_t2va_stock20",
                    "turbo8_t2va_research_exp",
                }:
                    next_stage = _build_empty_t2va_stage(
                        speed_source,
                        next_shape["width"],
                        next_shape["height"],
                        stage,
                    )
                else:
                    next_stage = _build_stage(
                        speed_source, next_shape["width"], next_shape["height"]
                    )
                next_video_target, next_audio_target = nested_av_parts(next_stage.latent)
                next_video_target = next_video_target.to(
                    device=raw_video.device, dtype=raw_video.dtype
                )
                next_audio_target = next_audio_target.to(
                    device=raw_audio.device, dtype=raw_audio.dtype
                )
                if next_video_target.shape != expanded_video.shape:
                    raise RuntimeError("Stage-rebuilt H3 video latent does not match DCT target shape")
                next_audio_scaled = next_audio_target * audio_scale
                expanded_audio = reindex_joint_audio_state(
                    raw_audio,
                    next_audio_scaled,
                    sigma_from=sigma_from,
                    sigma_to=sigma_to,
                )
                desired_next = comfy.nested_tensor.NestedTensor(
                    (expanded_video, expanded_audio)
                )
                next_target_external = comfy.nested_tensor.NestedTensor(
                    (next_video_target, next_audio_target)
                )
                pending_noise = solve_segment_noise(
                    desired_next,
                    next_target_external,
                    sigma=sigma_to,
                    audio_scale=audio_scale,
                    noise_scale=noise_scale,
                )
                pending_stage = next_stage
                record["transition"] = {
                    **transition,
                    "dct": dct_report,
                    "audio_transport": "target_anchored_public_flow_reindex_exp",
                    "audio_spatial_noise_expansion": False,
                }
                del desired_raw, raw_video, raw_audio, expanded_video, expanded_audio
                del desired_next, next_target_external, next_video_target, next_audio_target
                del next_audio_scaled
            stage_records.append(record)
            del stage_model, sampler, guider, noise, stage
            if has_transition:
                record["transition"]["stage_residency_release"] = (
                    _release_h3_residency_between_stages(model)
                )
        if output_nested is None or final_stage is None:
            raise RuntimeError("SPEED did not execute any sampling stage")
        runtime_devices = sorted({part.device.type for part in output_nested.unbind()})
        gpu_generated = any(device == "cuda" for device in runtime_devices)

        from comfy import model_management as comfy_model_management

        output_nested = output_nested.to(
            device=comfy_model_management.intermediate_device(),
            dtype=comfy_model_management.intermediate_dtype(),
        )
        output = final_stage.latent.copy()
        output.pop("downscale_ratio_spacial", None)
        output.pop("downscale_ratio_temporal", None)
        output["samples"] = output_nested
        report = {
            "schema": SPEED_REPORT_SCHEMA,
            "status": "completed_unvalidated_quality" if supported else "fallback_completed",
            "execution_scope": execution_scope,
            "support_reason": support_reason,
            "weight_patch_contract": weight_patch_report,
            "resolved_task": speed_source.get("resolved_task"),
            "audio_mode": speed_source.get("audio_mode"),
            "spectrum_profile_binding": profile_binding,
            "runtime_device_types": runtime_devices,
            "steps": steps,
            "nfe": sum(record["nfe"] for record in stage_records),
            "stages": stage_records,
            "official_speed_commit": OFFICIAL_SPEED_COMMIT,
            "paper": OFFICIAL_SPEED_PAPER,
            "h3_core_audit_commit": H3_CORE_AUDIT_COMMIT,
            "noise_contract": {
                "type": "modality_stable_nested_av_v1",
                "video_seed": int(seed) & 0xFFFFFFFFFFFFFFFF,
                "audio_seed": (
                    (int(seed) & 0xFFFFFFFFFFFFFFFF) ^ SPEED_AUDIO_NOISE_SEED_XOR
                ),
                "audio_invariant_to_video_canvas_size": True,
            },
            "initial_residency_release": initial_residency_release,
            "scoped_vram_headroom": headroom_report,
            "claims": {
                "gpu_generated": gpu_generated,
                "quality_validated": False,
                "speedup_validated": False,
                "vram_safe_16gb": False,
                "audio_noninferiority_validated": False,
            },
            "next_validation": "Run controlled ComfyUI GPU baseline vs SPEED with identical H3 inputs.",
        }
        report["scoped_vram_headroom"]["restoration"] = (
            _restore_speed_scoped_headroom(headroom_token)
        )
        return (
            output,
            final_stage.mux_audio,
            final_stage.conditioned_prompt,
            final_stage.media_map,
            canonical_json(report),
        )
    finally:
        _restore_speed_scoped_headroom(headroom_token)
        pending_noise = None
        pending_stage = None
        clear_speed_math_cache()
