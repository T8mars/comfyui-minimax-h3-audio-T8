from __future__ import annotations

import json
import math
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

import comfy.nested_tensor
from comfy.ldm.minimax.model import MiniMaxH3Model

from .conditioning import build_conditioning, resolve_task_type
from .core import MAX_PIXELS, empty_av_latent, nested_av_parts, sorted_autogrow_values
from .sampling import native_flow_sigmas, setup_dual_clock_sampling


SPEED_PROFILE_SCHEMA = "minimax_h3_speed_spectrum_profile_t8_v1"
SPEED_PLAN_SCHEMA = "minimax_h3_speed_plan_t8_v1"
SPEED_SOURCE_SCHEMA = "minimax_h3_speed_source_t8_v1"
SPEED_REPORT_SCHEMA = "minimax_h3_speed_execution_t8_v1"
OFFICIAL_SPEED_COMMIT = "ca7801c9bdffe681742e9592345bcf4885959be5"
OFFICIAL_SPEED_PAPER = "arXiv:2605.18736v3"
H3_CORE_AUDIT_COMMIT = "7fe8a6138504f90ff7be82f3babf416da32876b1"


def canonical_json(value: Any, *, indent: int | None = 2) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=indent)


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


@torch.no_grad()
def fit_h3_spatial_power_spectrum(
    video_latent: torch.Tensor,
    *,
    max_temporal_samples: int = 32,
    minimum_radius: int = 1,
    maximum_radius_fraction: float = 0.5,
) -> dict[str, Any]:
    """Fit a radial FFT power law to H3 video latents without importing SciPy."""
    if not isinstance(video_latent, torch.Tensor) or video_latent.ndim != 5:
        raise ValueError("video_latent must be [B,C,T,H,W]")
    batch, channels, frames, height, width = map(int, video_latent.shape)
    if batch < 1 or channels < 1 or frames < 1 or min(height, width) < 8:
        raise ValueError("video_latent is too small for a spatial spectrum fit")
    if not math.isfinite(maximum_radius_fraction) or not 0.1 <= maximum_radius_fraction <= 1.0:
        raise ValueError("maximum_radius_fraction must be in [0.1, 1.0]")

    temporal_indices = torch.linspace(
        0, frames - 1, min(frames, int(max_temporal_samples)), dtype=torch.float64
    ).round().long().unique()
    source_temporal_indices = temporal_indices.to(video_latent.device)
    mean_power = torch.zeros((height, width), dtype=torch.float64, device="cpu")
    for batch_index in range(batch):
        # Dataset profiles may contain many clips. Stream one batch entry through CPU
        # float32 FFTs instead of materializing the entire dataset as float64.
        x = video_latent[batch_index : batch_index + 1].index_select(
            2, source_temporal_indices
        ).detach()
        x = x.to(device="cpu", dtype=torch.float32)
        x = x - x.mean(dim=(-2, -1), keepdim=True)
        spectrum = torch.fft.fft2(x, norm="ortho")
        mean_power.add_(spectrum.abs().square().mean(dim=(0, 1, 2)).to(torch.float64))
        del x, spectrum
    mean_power.div_(batch)
    del source_temporal_indices

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
        value = float(mean_power[mask].mean())
        if math.isfinite(value) and value > 0.0:
            radii.append(float(index))
            powers.append(value)
            counts.append(count)
    if len(radii) < 4:
        raise ValueError("The latent did not provide enough finite radial spectrum bins")

    log_radius = torch.tensor(radii, dtype=torch.float64).log()
    log_power = torch.tensor(powers, dtype=torch.float64).log()
    intercept, slope, r_squared = _linear_fit(log_radius, log_power)
    amplitude = math.exp(intercept)
    beta = -slope
    if not (math.isfinite(amplitude) and amplitude > 0.0 and math.isfinite(beta) and beta > 0.0):
        raise ValueError("The fitted spatial power law is not physically usable")
    return {
        "amplitude": amplitude,
        "beta": beta,
        "r_squared": r_squared,
        "radial_bins": len(radii),
        "spatial_coefficients": int(sum(counts)),
        "latent_shape": [batch, channels, frames, height, width],
        "temporal_samples": int(temporal_indices.numel()),
        "radius_range": [radii[0], radii[-1]],
    }


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
    if execution_scope == "multimodal_research_exp":
        return True, "multimodal mechanics implemented; GPU quality/audio validation pending"
    raise ValueError("Unknown SPEED execution_scope")


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
    fields = ("checkpoint_fingerprint", "vae_fingerprint")
    expected = {field: str(profile.get(field, "")).strip() for field in fields}
    actual = {field: str(source.get(field, "")).strip() for field in fields}

    def known(value: str) -> bool:
        return bool(value) and value.lower() not in {"unrecorded", "unknown", "none"}

    mismatches = [
        field
        for field in fields
        if known(expected[field]) and known(actual[field]) and expected[field] != actual[field]
    ]
    if mismatches:
        raise ValueError("SPEED spectrum fingerprint mismatch: " + ", ".join(mismatches))
    if plan.get("profile_policy") == "require_validated_profile":
        missing = [field for field in fields if not known(actual[field])]
        if missing:
            raise ValueError(
                "Validated SPEED profile requires runtime source fingerprints: "
                + ", ".join(missing)
            )
        unequal = [field for field in fields if expected[field] != actual[field]]
        if unequal:
            raise ValueError(
                "Validated SPEED profile is not bound to this runtime source: "
                + ", ".join(unequal)
            )
    return {
        "required": True,
        "status": (
            "matched"
            if all(known(actual[field]) and actual[field] == expected[field] for field in fields)
            else "research_unrecorded_runtime_fingerprint"
        ),
        "task_family": actual_task,
        "expected": expected,
        "actual": actual,
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
    if execution_scope == "strict_t2va_stock20" and getattr(model, "patches", {}):
        supported = False
        support_reason = "strict P1 refuses LoRA/weight-patched models; use an unpatched stock H3 model"
    fallback = speed_plan["fallback_policy"]
    if not supported and fallback == "error":
        raise ValueError(support_reason)

    from comfy_extras.nodes_custom_sampler import Guider_Basic, Noise_RandomNoise

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

    output_nested = None
    pending_noise = None
    pending_stage: StageConditioning | None = None
    stage_records: list[dict[str, Any]] = []
    final_stage: StageConditioning | None = None
    try:
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
                noise = Noise_RandomNoise(int(seed)).generate_noise(stage.latent)
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
            if stage_index < len(transitions):
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
                if supported and execution_scope == "strict_t2va_stock20":
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
            "claims": {
                "gpu_generated": gpu_generated,
                "quality_validated": False,
                "speedup_validated": False,
                "vram_safe_16gb": False,
                "audio_noninferiority_validated": False,
            },
            "next_validation": "Run controlled ComfyUI GPU baseline vs SPEED with identical H3 inputs.",
        }
        return (
            output,
            final_stage.mux_audio,
            final_stage.conditioned_prompt,
            final_stage.media_map,
            canonical_json(report),
        )
    finally:
        pending_noise = None
        pending_stage = None
        clear_speed_math_cache()
