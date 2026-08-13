from __future__ import annotations

import inspect
import json
import math
from collections.abc import Mapping
from typing import Any

import torch
from comfy_extras.nodes_audio import vae_decode_audio

from .core import FPS, classify_h3_vae, nested_av_parts
from .vram_policy import runtime_snapshot


SCHEMA = "t8.minimax_h3.av_decode_safety.v1"
MODES = ("preflight_only", "decode_regular", "decode_tiled_exp")
ENFORCEMENT = ("report_only", "block_known_unsafe")
MIB = 1024**2


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str)


def _empty_outputs(video: torch.Tensor, audio: torch.Tensor):
    frames = torch.zeros(
        (0, max(1, int(video.shape[-2]) * 16), max(1, int(video.shape[-1]) * 16), 3),
        dtype=torch.float32,
        device="cpu",
    )
    decoded_audio = {
        "waveform": torch.zeros((1, 2, 0), dtype=torch.float32),
        "sample_rate": 32000,
    }
    return frames, decoded_audio


def _safe_source(value: Any) -> str | None:
    try:
        return inspect.getsource(value)
    except (OSError, TypeError):
        return None


def _h3_tiled_decode_core_contract() -> dict[str, Any]:
    try:
        from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE, create_token_ids
    except Exception as error:
        return {
            "state": "unknown",
            "error": f"{type(error).__name__}: {error}",
            "adaptive_internal_tiling": None,
            "explicit_decode_tiled_alias": None,
        }
    token_source = _safe_source(create_token_ids)
    adaptive_source = _safe_source(MiniMaxH3VideoVAE._adaptive_decode)
    tiled_alias_source = _safe_source(MiniMaxH3VideoVAE.decode_tiled)
    if token_source is None or adaptive_source is None or tiled_alias_source is None:
        state = "unknown"
    else:
        state = (
            "supported"
            if "full_dims" in token_source and "offset" in token_source
            else "unsupported"
        )
    return {
        "state": state,
        "coordinate_contract": "global_full_dims_and_tile_offset",
        "adaptive_internal_tiling": bool(
            adaptive_source and "self.tiled_decode" in adaptive_source
        ),
        "explicit_decode_tiled_alias": bool(
            tiled_alias_source and "return self.decode(z)" in tiled_alias_source
        ),
    }


def _video_decode_route(video_vae: Any, width: int, height: int, mode: str) -> dict[str, Any]:
    first_stage = getattr(video_vae, "first_stage_model", None)
    first_stage_class = None if first_stage is None else type(first_stage).__name__
    is_h3 = first_stage_class == "MiniMaxH3VideoVAE"
    tiling = getattr(first_stage, "tiling", None) if is_h3 else None
    tile_size = getattr(first_stage, "tile_size", None) if is_h3 else None
    split_expected = bool(
        is_h3
        and tiling is True
        and isinstance(tile_size, (int, float))
        and (width > int(tile_size) or height > int(tile_size))
    )
    core = _h3_tiled_decode_core_contract() if is_h3 else {
        "state": "unknown",
        "adaptive_internal_tiling": None,
        "explicit_decode_tiled_alias": None,
    }
    explicit_tiled_alias = core.get("explicit_decode_tiled_alias")
    explicit_tile_controls_effective = (
        False if explicit_tiled_alias is True else True if explicit_tiled_alias is False else None
    )
    return {
        "first_stage_class": first_stage_class,
        "is_h3_video_vae": is_h3,
        "requested_mode": mode,
        "internal_tiling_enabled": tiling,
        "internal_tile_size_pixels": tile_size,
        "internal_spatial_split_expected": split_expected,
        "global_tile_coordinate_contract": core,
        "explicit_tile_controls_effective": explicit_tile_controls_effective,
        "resolved_path": (
            "preflight_only_no_decode"
            if mode == "preflight_only"
            else "h3_internal_spatial_tiled_decode"
            if split_expected
            else "h3_decode_without_spatial_split"
            if is_h3
            else "generic_vae_decode"
        ),
    }


def _split_latents(av_latent: dict, video: torch.Tensor, audio: torch.Tensor):
    shared = {key: value for key, value in av_latent.items() if key not in {"samples", "noise_mask"}}
    video_latent = shared.copy()
    audio_latent = shared.copy()
    video_latent["samples"] = video
    audio_latent["samples"] = audio
    masks = av_latent.get("noise_mask")
    if getattr(masks, "is_nested", False):
        parts = tuple(masks.unbind())
        if len(parts) != 2:
            raise ValueError("Nested AV noise_mask must contain exactly video and audio masks")
        video_latent["noise_mask"], audio_latent["noise_mask"] = parts
    elif masks is not None:
        raise ValueError("AV Decode Safety requires a nested video/audio noise_mask when present")
    return video_latent, audio_latent


def inspect_av_decode(
    av_latent: dict,
    video_vae,
    audio_vae,
    *,
    mode: str,
    minimum_current_headroom_mib: float,
    maximum_estimated_output_mib: float,
    runtime: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if mode not in MODES:
        raise ValueError(f"unsupported AV decode mode: {mode!r}")
    if not math.isfinite(float(minimum_current_headroom_mib)) or minimum_current_headroom_mib < 0:
        raise ValueError("minimum_current_headroom_mib must be finite and non-negative")
    if not math.isfinite(float(maximum_estimated_output_mib)) or maximum_estimated_output_mib <= 0:
        raise ValueError("maximum_estimated_output_mib must be finite and positive")

    video, audio = nested_av_parts(av_latent)
    hard: list[dict[str, Any]] = []
    high_risk: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    unknown: list[dict[str, Any]] = []

    video_kind = classify_h3_vae(video_vae)
    audio_kind = classify_h3_vae(audio_vae)
    if video_kind != "video":
        hard.append({"code": "video_vae_contract_mismatch", "detected": video_kind})
    if audio_kind != "audio":
        hard.append({"code": "audio_vae_contract_mismatch", "detected": audio_kind})
    if not torch.isfinite(video).all().item():
        hard.append({"code": "video_latent_nonfinite"})
    if not torch.isfinite(audio).all().item():
        hard.append({"code": "audio_latent_nonfinite"})

    latent_t = int(video.shape[2])
    inferred_frames = 5 if latent_t <= 2 else 5 + 17 * ((latent_t - 2) // 5)
    latent_time_remainder = max(0, (latent_t - 2) % 5)
    if latent_t > 2 and latent_time_remainder:
        hard.append(
            {
                "code": "video_latent_time_off_h3_grid",
                "latent_t": latent_t,
                "remainder": latent_time_remainder,
            }
        )
    expected_audio_t = round(inferred_frames / FPS * 40.0)
    audio_t = int(audio.shape[-1])
    if audio_t != expected_audio_t:
        warnings.append(
            {
                "code": "audio_video_latent_duration_mismatch",
                "audio_t": audio_t,
                "expected_audio_t": expected_audio_t,
                "difference": audio_t - expected_audio_t,
            }
        )

    height = int(video.shape[-2]) * 16
    width = int(video.shape[-1]) * 16
    estimated_video_output_mib = inferred_frames * height * width * 3 * 4 / MIB
    estimated_audio_output_mib = max(audio_t, 1) * 800 * 2 * 4 / MIB
    estimated_output_mib = estimated_video_output_mib + estimated_audio_output_mib
    decode_route = _video_decode_route(video_vae, width, height, mode)
    coordinate_state = decode_route["global_tile_coordinate_contract"].get("state")
    if decode_route["internal_spatial_split_expected"]:
        if coordinate_state == "unsupported":
            high_risk.append(
                {
                    "code": "h3_spatial_tiling_global_coordinates_missing",
                    "message": (
                        "The H3 VAE will spatially tile this canvas even on regular decode, "
                        "but tile-global decoder coordinates are absent. Grid seams or "
                        "coordinate-dependent artifacts are a known mechanical risk."
                    ),
                    "width": width,
                    "height": height,
                    "internal_tile_size_pixels": decode_route["internal_tile_size_pixels"],
                }
            )
        elif coordinate_state != "supported":
            unknown.append(
                {
                    "code": "h3_spatial_tiling_coordinate_contract_unknown",
                    "width": width,
                    "height": height,
                }
            )
    if mode == "decode_tiled_exp" and decode_route["explicit_tile_controls_effective"] is False:
        warnings.append(
            {
                "code": "h3_explicit_tile_controls_ignored",
                "message": (
                    "Current H3 decode_tiled delegates to regular decode, so the requested "
                    "tile size/overlap controls do not change its internal 256-pixel tiling."
                ),
            }
        )
    elif mode == "decode_tiled_exp" and decode_route["explicit_tile_controls_effective"] is None:
        unknown.append(
            {
                "code": "h3_explicit_tile_controls_contract_unknown",
                "message": "The H3 first-stage decode_tiled delegation contract could not be inspected.",
            }
        )
    if estimated_output_mib > float(maximum_estimated_output_mib):
        high_risk.append(
            {
                "code": "estimated_decode_output_exceeds_gate",
                "estimated_output_mib": estimated_output_mib,
                "maximum_estimated_output_mib": float(maximum_estimated_output_mib),
            }
        )

    current = dict(runtime) if runtime is not None else runtime_snapshot()
    free_mib = current.get("gpu", {}).get("whole_device_free_mib")
    if free_mib is None:
        unknown.append({"code": "current_vram_headroom_unknown"})
    elif float(free_mib) < float(minimum_current_headroom_mib):
        high_risk.append(
            {
                "code": "current_vram_headroom_below_decode_gate",
                "current_mib": float(free_mib),
                "minimum_mib": float(minimum_current_headroom_mib),
            }
        )

    if mode == "decode_tiled_exp":
        if not callable(getattr(video_vae, "decode_tiled", None)):
            hard.append({"code": "video_vae_tiled_decode_unavailable"})
        warnings.append(
            {
                "code": "tiled_decode_is_not_quality_equivalent",
                "message": "Tiled decode can reduce peak activation but may introduce seams or coordinate artifacts.",
            }
        )

    status = (
        "blocked"
        if hard
        else "high_risk"
        if high_risk
        else "unknown"
        if unknown
        else "pass_with_warnings"
        if warnings
        else "pass"
    )
    return {
        "schema": SCHEMA,
        "mode": mode,
        "status": status,
        "no_known_blocker": not hard and not high_risk,
        "memory_safe_claim": False,
        "latent": {
            "video_shape": list(video.shape),
            "audio_shape": list(audio.shape),
            "inferred_frames": inferred_frames,
            "inferred_width": width,
            "inferred_height": height,
            "expected_audio_t": expected_audio_t,
        },
        "estimated_output": {
            "video_float32_mib": estimated_video_output_mib,
            "audio_float32_mib": estimated_audio_output_mib,
            "total_float32_mib": estimated_output_mib,
            "excludes": [
                "VAE weights",
                "VAE intermediate activations",
                "temporary tiles/blends",
                "allocator fragmentation",
                "preview and save-node copies",
            ],
        },
        "vae_contract": {"video": video_kind, "audio": audio_kind},
        "video_decode_route": decode_route,
        "runtime": current,
        "issues": {
            "hard": hard,
            "high_risk": high_risk,
            "warnings": warnings,
            "unknown": unknown,
        },
        "scientific_boundaries": [
            "This preflight is not a VAE peak-memory predictor.",
            "Regular H3 decode may still use internal 256-pixel spatial tiling; it is not an untiled control at larger canvases.",
            "Current H3 decode_tiled can alias regular decode and ignore explicit tile controls; the report records the resolved path.",
            "Neither regular nor explicit tiled decode is assumed visually equivalent until full-frame coordinate and A/B gates pass.",
            "No model, VAE, cache, or other global resource is unloaded automatically.",
        ],
    }


def decode_av_safely(
    av_latent: dict,
    video_vae,
    audio_vae,
    mode: str = "preflight_only",
    minimum_current_headroom_mib: float = 512.0,
    maximum_estimated_output_mib: float = 8192.0,
    enforcement: str = "report_only",
    video_tile_size: int = 32,
    video_tile_overlap: int = 8,
    video_tile_temporal: int = 999,
):
    if enforcement not in ENFORCEMENT:
        raise ValueError(f"unsupported AV decode enforcement: {enforcement!r}")
    report = inspect_av_decode(
        av_latent,
        video_vae,
        audio_vae,
        mode=mode,
        minimum_current_headroom_mib=minimum_current_headroom_mib,
        maximum_estimated_output_mib=maximum_estimated_output_mib,
    )
    video, audio = nested_av_parts(av_latent)
    video_latent, audio_latent = _split_latents(av_latent, video, audio)
    should_block = report["status"] in {"blocked", "high_risk"}
    if enforcement == "block_known_unsafe" and should_block:
        codes = [
            issue["code"]
            for group in ("hard", "high_risk")
            for issue in report["issues"][group]
        ]
        raise ValueError("MiniMax H3 AV decode safety blocked execution: " + ", ".join(codes))

    if mode == "preflight_only":
        frames, decoded_audio = _empty_outputs(video, audio)
        report["decoded"] = False
        report["preflight_only"] = True
        return frames, decoded_audio, video_latent, audio_latent, _json(report)

    if mode == "decode_regular":
        images = video_vae.decode(video)
    elif mode == "decode_tiled_exp":
        if video_tile_size < 2 or video_tile_overlap < 0 or video_tile_overlap >= video_tile_size:
            raise ValueError("video tile overlap must satisfy 0 <= overlap < tile size")
        images = video_vae.decode_tiled(
            video,
            tile_x=int(video_tile_size),
            tile_y=int(video_tile_size),
            overlap=int(video_tile_overlap),
            tile_t=max(2, int(video_tile_temporal)),
            overlap_t=1,
        )
    else:
        raise ValueError(f"unsupported AV decode mode: {mode!r}")

    if images.ndim == 5:
        images = images.reshape(-1, *images.shape[-3:])
    if images.ndim != 4 or not torch.isfinite(images).all().item():
        raise ValueError(f"Video VAE returned invalid IMAGE data: {tuple(images.shape)}")
    decoded_audio = vae_decode_audio(audio_vae, {"samples": audio})
    waveform = decoded_audio.get("waveform") if isinstance(decoded_audio, Mapping) else None
    if not isinstance(waveform, torch.Tensor) or not torch.isfinite(waveform).all().item():
        raise ValueError("Audio VAE returned invalid/non-finite AUDIO data")

    report["decoded"] = True
    report["preflight_only"] = False
    report["actual"] = {
        "frames_shape": list(images.shape),
        "audio_waveform_shape": list(waveform.shape),
        "audio_sample_rate": int(decoded_audio["sample_rate"]),
    }
    return images, decoded_audio, video_latent, audio_latent, _json(report)
