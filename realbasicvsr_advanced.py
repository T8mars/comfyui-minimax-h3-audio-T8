from __future__ import annotations

import gc
import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .realbasicvsr_arch import RealBasicVSRNet


REPORT_SCHEMA = "h3_t8_realbasicvsr_report/v1"
OUTPUT_MODES = ("native_size_restore", "x4_super_resolution")
PRECISIONS = ("auto", "fp16", "fp32")
RELEASE_POLICIES = ("offload_after", "clear_after", "keep_loaded")
CHECKPOINT_BRANCHES = ("prefer_ema", "prefer_generator")

_MODEL_CACHE: dict[tuple[str, str, str, int, int], RealBasicVSRNet] = {}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)


def _validate_frames(frames: torch.Tensor) -> tuple[int, int, int, int]:
    if not isinstance(frames, torch.Tensor) or frames.ndim != 4:
        shape = tuple(frames.shape) if isinstance(frames, torch.Tensor) else type(frames).__name__
        raise ValueError(f"frames must be ComfyUI IMAGE [N,H,W,C], got {shape}")
    count, height, width, channels = map(int, frames.shape)
    if count < 1 or height < 8 or width < 8 or channels < 3:
        raise ValueError(f"frames has unsupported shape {tuple(frames.shape)}")
    if not bool(torch.isfinite(frames).all()):
        raise ValueError("frames contains NaN or Inf")
    return count, height, width, channels


def _device_and_dtype(precision: str) -> tuple[torch.device, torch.dtype]:
    import comfy.model_management as model_management

    device = model_management.get_torch_device()
    if precision == "fp32" or device.type != "cuda":
        return device, torch.float32
    return device, torch.float16


def _model_cache_key(
    model_path: Path,
    dtype: torch.dtype,
    checkpoint_branch: str,
) -> tuple[str, str, str, int, int]:
    resolved = model_path.resolve()
    stat = resolved.stat()
    return (
        str(resolved),
        str(dtype),
        str(checkpoint_branch),
        int(stat.st_size),
        int(stat.st_mtime_ns),
    )


def _unwrap_state_dict(payload: Any, branch: str) -> dict[str, torch.Tensor]:
    if isinstance(payload, Mapping) and isinstance(payload.get("state_dict"), Mapping):
        payload = payload["state_dict"]
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint must contain a tensor state_dict")
    candidates = (
        ("generator_ema.", "generator.")
        if branch == "prefer_ema"
        else ("generator.", "generator_ema.")
    )
    keys = [str(key) for key in payload]
    prefix = next((item for item in candidates if any(key.startswith(item) for key in keys)), "")
    state: dict[str, torch.Tensor] = {}
    for key, value in payload.items():
        key = str(key)
        if prefix and not key.startswith(prefix):
            continue
        if prefix:
            key = key[len(prefix) :]
        for wrapper in ("module.", "model."):
            if key.startswith(wrapper):
                key = key[len(wrapper) :]
        if key == "step_counter":
            continue
        if isinstance(value, torch.Tensor):
            state[key] = value
    if not state:
        raise ValueError("checkpoint contains no usable RealBasicVSR tensor weights")
    return state


def _load_model(
    model_path: Path,
    *,
    precision: str,
    checkpoint_branch: str,
) -> tuple[RealBasicVSRNet, torch.device, torch.dtype, bool]:
    import comfy.utils

    device, dtype = _device_and_dtype(precision)
    cache_key = _model_cache_key(model_path, dtype, checkpoint_branch)
    cache_identity = cache_key[:3]
    for stale_key, stale_model in list(_MODEL_CACHE.items()):
        if stale_key[:3] == cache_identity and stale_key != cache_key:
            stale_model.to(device="cpu")
            _MODEL_CACHE.pop(stale_key, None)
    cached = _MODEL_CACHE.get(cache_key)
    if cached is not None:
        cached.to(device=device, dtype=dtype)
        return cached, device, dtype, True

    payload = comfy.utils.load_torch_file(str(model_path), safe_load=True)
    state = _unwrap_state_dict(payload, checkpoint_branch)
    model = RealBasicVSRNet(sequential_cleaning=True)
    incompatible = model.load_state_dict(state, strict=False)
    missing = [key for key in incompatible.missing_keys if not key.endswith("num_batches_tracked")]
    if missing or incompatible.unexpected_keys:
        raise ValueError(
            "checkpoint architecture does not match RealBasicVSR: "
            f"missing={missing[:8]}, unexpected={incompatible.unexpected_keys[:8]}"
        )
    model.eval().requires_grad_(False).to(device=device, dtype=dtype)
    _MODEL_CACHE[cache_key] = model
    return model, device, dtype, False


def _release_model(
    model: RealBasicVSRNet,
    model_path: Path,
    dtype: torch.dtype,
    release_policy: str,
) -> dict[str, Any]:
    import comfy.model_management as model_management

    if release_policy == "keep_loaded":
        return {
            "policy": release_policy,
            "cache_retained": True,
            "device": str(next(model.parameters()).device),
        }
    model.to(device="cpu")
    if release_policy == "clear_after":
        for cache_key, cached_model in list(_MODEL_CACHE.items()):
            if cached_model is model:
                _MODEL_CACHE.pop(cache_key, None)
    gc.collect()
    model_management.soft_empty_cache()
    return {
        "policy": release_policy,
        "cache_retained": release_policy == "offload_after",
        "device": "cpu",
    }


def _window_ranges(count: int, chunk_frames: int, overlap_frames: int) -> list[tuple[int, int]]:
    if chunk_frames < 2:
        raise ValueError("chunk_frames must be at least 2")
    if overlap_frames < 0 or overlap_frames >= chunk_frames:
        raise ValueError("overlap_frames must stay within 0..chunk_frames-1")
    if count <= chunk_frames:
        return [(0, count)]
    stride = chunk_frames - overlap_frames
    starts = list(range(0, count, stride))
    ranges = []
    for start in starts:
        end = min(count, start + chunk_frames)
        start = max(0, end - chunk_frames)
        item = (start, end)
        if not ranges or ranges[-1] != item:
            ranges.append(item)
        if end == count:
            break
    return ranges


def _blend_weights(length: int, overlap: int, *, first: bool, last: bool) -> torch.Tensor:
    weights = torch.ones(length, dtype=torch.float32)
    fade = min(overlap, length // 2)
    if fade and not first:
        weights[:fade] = torch.linspace(1.0 / (fade + 1), 1.0, fade)
    if fade and not last:
        weights[-fade:] = torch.linspace(1.0, 1.0 / (fade + 1), fade)
    return weights


def restore_realbasicvsr(
    frames: torch.Tensor,
    audio: Any = None,
    *,
    model_path: Path,
    model_name: str,
    output_mode: str = "native_size_restore",
    strength: float = 0.30,
    chunk_frames: int = 8,
    overlap_frames: int = 2,
    precision: str = "auto",
    checkpoint_branch: str = "prefer_ema",
    release_policy: str = "offload_after",
) -> tuple[torch.Tensor, torch.Tensor, Any, str]:
    count, height, width, channels = _validate_frames(frames)
    if output_mode not in OUTPUT_MODES:
        raise ValueError(f"output_mode must be one of {OUTPUT_MODES}")
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}")
    if release_policy not in RELEASE_POLICIES:
        raise ValueError(f"release_policy must be one of {RELEASE_POLICIES}")
    if checkpoint_branch not in CHECKPOINT_BRANCHES:
        raise ValueError(f"checkpoint_branch must be one of {CHECKPOINT_BRANCHES}")
    strength = float(strength)
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must stay within 0..1")
    ranges = _window_ranges(count, int(chunk_frames), int(overlap_frames))
    source = frames.detach().to(device="cpu", dtype=torch.float32).contiguous()
    model = None
    release: dict[str, Any] = {"policy": release_policy, "completed": False}
    started = time.perf_counter()
    cache_hit = False
    try:
        model, device, dtype, cache_hit = _load_model(
            Path(model_path), precision=precision, checkpoint_branch=checkpoint_branch
        )
        out_height = height if output_mode == "native_size_restore" else height * 4
        out_width = width if output_mode == "native_size_restore" else width * 4
        accumulator = torch.zeros(count, out_height, out_width, 3, dtype=torch.float32)
        weights_total = torch.zeros(count, 1, 1, 1, dtype=torch.float32)
        with torch.inference_mode():
            for window_index, (start, end) in enumerate(ranges):
                window = source[start:end, ..., :3]
                original_length = int(window.shape[0])
                if original_length == 1:
                    window = torch.cat((window, window), dim=0)
                input_tensor = window.permute(0, 3, 1, 2).unsqueeze(0).to(
                    device=device, dtype=dtype
                )
                restored = model(input_tensor)[0, :original_length]
                if output_mode == "native_size_restore":
                    restored = F.interpolate(
                        restored, size=(height, width), mode="bicubic", align_corners=False
                    )
                restored_cpu = restored.permute(0, 2, 3, 1).float().cpu().clamp(0.0, 1.0)
                weights = _blend_weights(
                    original_length,
                    int(overlap_frames),
                    first=window_index == 0,
                    last=window_index == len(ranges) - 1,
                ).view(-1, 1, 1, 1)
                accumulator[start:end] += restored_cpu * weights
                weights_total[start:end] += weights
                del input_tensor, restored, restored_cpu
        restored_rgb = accumulator / weights_total.clamp_min(1e-6)
        source_rgb = source[..., :3]
        if output_mode == "x4_super_resolution":
            source_rgb = F.interpolate(
                source_rgb.permute(0, 3, 1, 2),
                size=(out_height, out_width),
                mode="bicubic",
                align_corners=False,
            ).permute(0, 2, 3, 1)
        candidate_rgb = source_rgb.lerp(restored_rgb, strength).clamp(0.0, 1.0)
        if channels > 3:
            extras = source[..., 3:]
            if output_mode == "x4_super_resolution":
                extras = F.interpolate(
                    extras.permute(0, 3, 1, 2),
                    size=(out_height, out_width),
                    mode="bilinear",
                    align_corners=False,
                ).permute(0, 2, 3, 1)
            candidate = torch.cat((candidate_rgb, extras), dim=-1)
        else:
            candidate = candidate_rgb
    finally:
        if model is not None:
            release = _release_model(model, Path(model_path), dtype, release_policy)
            release["completed"] = True
    elapsed = time.perf_counter() - started
    report = {
        "schema": REPORT_SCHEMA,
        "backend": "OpenMMLab RealBasicVSR x4 architecture",
        "model_name": str(model_name),
        "model_path": str(model_path),
        "checkpoint_branch": checkpoint_branch,
        "model_identity_gate": "none; architecture compatibility only",
        "input": {"frames": count, "width": width, "height": height, "channels": channels},
        "output": {
            "mode": output_mode,
            "width": int(candidate.shape[2]),
            "height": int(candidate.shape[1]),
            "strength": strength,
        },
        "temporal_windows": {
            "chunk_frames": int(chunk_frames),
            "overlap_frames": int(overlap_frames),
            "ranges": [list(item) for item in ranges],
            "boundary_blend": "linear_overlap",
        },
        "precision": str(dtype),
        "cache_hit": cache_hit,
        "audio": {
            "provided": audio is not None,
            "exact_object_passthrough": True,
            "resampled": False,
            "modified": False,
        },
        "release": release,
        "elapsed_seconds": round(elapsed, 4),
        "limits": (
            "Restores temporal/detail consistency but cannot reconstruct missing identity, "
            "correct lip sync or guarantee removal of generative artifacts."
        ),
    }
    return candidate.contiguous(), source, audio, _canonical_json(report)
