from __future__ import annotations

import gc
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


REPORT_SCHEMA = "h3_t8_flashvsr_report/v1"
MODEL_TYPE = "H3_T8_FLASHVSR_MODEL"
PLAN_TYPE = "H3_T8_FLASHVSR_PLAN"
PIPELINE_MODES = ("tiny", "tiny_long", "full")
PRECISIONS = ("bf16", "fp16")
QUALITY_PROFILES = ("quality_locked", "balanced_dynamic_exp", "memory_safe")
SPATIAL_STRATEGIES = ("auto", "full_frame", "adaptive_tiles")
MEMORY_POLICIES = ("auto", "resident", "staged")
RELEASE_POLICIES = ("offload_after", "clear_after", "keep_loaded")

_MODEL_CACHE: dict[tuple[Any, ...], "FlashVSRModelHandle"] = {}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)


def _validate_frames(frames: torch.Tensor) -> tuple[int, int, int, int]:
    if not isinstance(frames, torch.Tensor) or frames.ndim != 4:
        shape = tuple(frames.shape) if isinstance(frames, torch.Tensor) else type(frames).__name__
        raise ValueError(f"frames must be ComfyUI IMAGE [N,H,W,C], got {shape}")
    count, height, width, channels = map(int, frames.shape)
    if count < 1 or height < 16 or width < 16 or channels < 3:
        raise ValueError(f"frames has unsupported shape {tuple(frames.shape)}")
    if not bool(torch.isfinite(frames).all()):
        raise ValueError("frames contains NaN or Inf")
    return count, height, width, channels


def next_8n_plus_5(count: int) -> int:
    """Minimum supported FlashVSR input length; padding repeats the final frame."""

    return 21 if count < 21 else ((count - 5 + 7) // 8) * 8 + 5


def largest_8n_plus_1(count: int) -> int:
    return 0 if count < 1 else ((count - 1) // 8) * 8 + 1


def _target_dimensions(width: int, height: int, scale: int) -> tuple[int, int]:
    scaled_width = width * scale
    scaled_height = height * scale
    return (
        max(128, (scaled_width // 128) * 128),
        max(128, (scaled_height // 128) * 128),
    )


def _prepare_input_tensor(
    frames: torch.Tensor,
    *,
    scale: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, int, int, int]:
    count, height, width, _ = _validate_frames(frames)
    target_width, target_height = _target_dimensions(width, height, scale)
    frame_count = largest_8n_plus_1(count + 4)
    if frame_count < 21:
        raise ValueError("FlashVSR requires at least 21 padded input frames")

    source = frames[:frame_count, ..., :3].detach().float().cpu()
    if source.shape[0] < frame_count:
        source = torch.cat(
            (source, source[-1:].repeat(frame_count - source.shape[0], 1, 1, 1)), dim=0
        )
    source = source.permute(0, 3, 1, 2)
    resized = F.interpolate(
        source,
        size=(height * scale, width * scale),
        mode="bicubic",
        align_corners=False,
    )
    y0 = max(0, (resized.shape[-2] - target_height) // 2)
    x0 = max(0, (resized.shape[-1] - target_width) // 2)
    resized = resized[..., y0 : y0 + target_height, x0 : x0 + target_width]
    video = resized.permute(1, 0, 2, 3).unsqueeze(0).to(dtype=dtype)
    return video.mul_(2.0).sub_(1.0).contiguous(), target_height, target_width, frame_count


def _tensor_to_frames(video: torch.Tensor) -> torch.Tensor:
    if video.ndim == 5:
        video = video.squeeze(0)
    if video.ndim != 4:
        raise RuntimeError(f"FlashVSR pipeline returned unsupported shape {tuple(video.shape)}")
    # The public pipelines return [C,F,H,W].
    return video.permute(1, 2, 3, 0).float().add_(1.0).mul_(0.5).clamp_(0.0, 1.0)


def _device_and_dtype(precision: str) -> tuple[torch.device, torch.dtype]:
    import comfy.model_management as model_management

    device = model_management.get_torch_device()
    if device.type != "cuda":
        raise RuntimeError("FlashVSR currently requires an NVIDIA CUDA device")
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        dtype = torch.float16
    return device, dtype


def _required_files(model_dir: Path, mode: str) -> dict[str, Path]:
    files = {
        "dit": model_dir / "diffusion_pytorch_model_streaming_dmd.safetensors",
        "lq": model_dir / "LQ_proj_in.ckpt",
        "prompt": model_dir / "posi_prompt.pth",
    }
    if mode == "full":
        files["vae"] = model_dir / "Wan2.1_VAE.pth"
    else:
        files["decoder"] = model_dir / "TCDecoder.ckpt"
    missing = [path.name for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"FlashVSR model folder '{model_dir}' is missing: {', '.join(missing)}. "
            "Download the official JunhaoZhuang/FlashVSR-v1.1 files into this folder."
        )
    return files


def _model_cache_key(
    model_dir: Path,
    mode: str,
    precision: str,
    files: dict[str, Path],
) -> tuple[Any, ...]:
    fingerprint: list[Any] = [str(model_dir.resolve()), mode, precision]
    for key, path in sorted(files.items()):
        stat = path.stat()
        fingerprint.extend((key, path.name, int(stat.st_mtime_ns)))
    return tuple(fingerprint)


@dataclass
class FlashVSRModelHandle:
    pipe: Any
    model_dir: Path
    model_name: str
    mode: str
    device: torch.device
    dtype: torch.dtype
    precision: str
    cache_key: tuple[Any, ...]
    cache_hit: bool = False


def _initialize_pipeline(
    model_dir: Path,
    *,
    model_name: str,
    mode: str,
    precision: str,
) -> FlashVSRModelHandle:
    if mode not in PIPELINE_MODES:
        raise ValueError(f"mode must be one of {PIPELINE_MODES}")
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}")
    files = _required_files(model_dir, mode)
    cache_key = _model_cache_key(model_dir, mode, precision, files)
    cached = _MODEL_CACHE.get(cache_key)
    if cached is not None:
        cached.cache_hit = True
        return cached

    device, dtype = _device_and_dtype(precision)
    from .flashvsr_vendor import (
        FlashVSRFullPipeline,
        FlashVSRTinyLongPipeline,
        FlashVSRTinyPipeline,
        ModelManager,
    )
    from .flashvsr_vendor.models.TCDecoder import build_tcdecoder
    from .flashvsr_vendor.models.utils import Buffer_LQ4x_Proj, Causal_LQ4x_Proj

    manager = ModelManager(torch_dtype=dtype, device="cpu")
    load_paths = [str(files["dit"])]
    if mode == "full":
        load_paths.append(str(files["vae"]))
    manager.load_models(load_paths)
    if manager.fetch_model("wan_video_dit") is None:
        raise RuntimeError(
            "FlashVSR could not load the DiT checkpoint structure. "
            "Use an official FlashVSR/FlashVSR-v1.1 model, not an H3 or Wan generation model."
        )

    if mode == "full":
        pipe = FlashVSRFullPipeline.from_model_manager(manager, device=device)
        if pipe.vae is None:
            raise RuntimeError("FlashVSR full mode could not load Wan2.1_VAE.pth")
        pipe.vae.model.encoder = None
        pipe.vae.model.conv1 = None
    else:
        pipeline_cls = FlashVSRTinyPipeline if mode == "tiny" else FlashVSRTinyLongPipeline
        pipe = pipeline_cls.from_model_manager(manager, device=device)
        pipe.TCDecoder = build_tcdecoder(
            new_channels=[512, 256, 128, 128],
            device=device,
            dtype=dtype,
            new_latent_channels=784,
        )
        decoder_payload = torch.load(files["decoder"], map_location="cpu", weights_only=True)
        pipe.TCDecoder.load_state_dict(decoder_payload, strict=False)
        pipe.TCDecoder.clean_mem()

    is_v11 = "v1.1" in model_name.lower() or "v1.1" in model_dir.name.lower()
    lq_projection = (
        Causal_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1)
        if is_v11
        else Buffer_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1)
    )
    lq_payload = torch.load(files["lq"], map_location="cpu", weights_only=True)
    lq_projection.load_state_dict(lq_payload, strict=True)
    pipe.denoising_model().LQ_proj_in = lq_projection.to(device=device, dtype=dtype)
    pipe.to(device, dtype=dtype)
    pipe.enable_vram_management(num_persistent_param_in_dit=None)
    pipe.init_cross_kv(prompt_path=str(files["prompt"]))
    pipe.load_models_to_device(["dit", "vae"])
    pipe.offload_model()

    handle = FlashVSRModelHandle(
        pipe=pipe,
        model_dir=model_dir,
        model_name=model_name,
        mode=mode,
        device=device,
        dtype=dtype,
        precision=precision,
        cache_key=cache_key,
    )
    _MODEL_CACHE[cache_key] = handle
    return handle


def load_flashvsr_model(
    *,
    model_dir: Path,
    model_name: str,
    mode: str,
    precision: str,
) -> tuple[FlashVSRModelHandle, str]:
    handle = _initialize_pipeline(
        Path(model_dir), model_name=model_name, mode=mode, precision=precision
    )
    report = {
        "schema": REPORT_SCHEMA,
        "operation": "load_model",
        "model": {"name": model_name, "directory": str(model_dir), "mode": mode},
        "precision": str(handle.dtype),
        "device": str(handle.device),
        "cache_hit": bool(handle.cache_hit),
        "identity_gate": "none; required files and loadable architecture only",
    }
    return handle, _canonical_json(report)


class MotionBudgetController:
    def __init__(self, values: list[tuple[float, float, int]], report: list[dict[str, Any]]):
        self._values = values
        self.report = report

    def values_for(
        self,
        index: int,
        base_topk: float,
        base_kv: float,
        base_local: int,
    ) -> tuple[float, float, int]:
        if not self._values:
            return float(base_topk), float(base_kv), int(base_local)
        return self._values[min(max(int(index), 0), len(self._values) - 1)]


def _motion_scores(frames: torch.Tensor, chunks: int) -> list[float]:
    sample = frames.detach().float().cpu()[..., :3].mean(dim=-1).unsqueeze(1)
    target_h = min(96, int(sample.shape[-2]))
    target_w = min(96, int(sample.shape[-1]))
    sample = F.interpolate(sample, size=(target_h, target_w), mode="area")[:, 0]
    if sample.shape[0] < 2:
        return [0.0] * chunks
    deltas = (sample[1:] - sample[:-1]).abs().mean(dim=(1, 2))
    scores: list[float] = []
    for index in range(chunks):
        start = min(index * 8, max(0, deltas.shape[0] - 1))
        end = min(deltas.shape[0], start + 12)
        scores.append(float(deltas[start:end].mean()) if end > start else 0.0)
    return scores


def build_flashvsr_plan(
    frames: torch.Tensor,
    *,
    quality_profile: str = "quality_locked",
    spatial_strategy: str = "auto",
    memory_policy: str = "auto",
    base_attention_budget: float = 2.0,
    kv_retention: float = 3.0,
    local_radius: int = 11,
    tile_size: int = 256,
    tile_overlap: int = 24,
) -> tuple[dict[str, Any], str]:
    count, height, width, _ = _validate_frames(frames)
    if quality_profile not in QUALITY_PROFILES:
        raise ValueError(f"quality_profile must be one of {QUALITY_PROFILES}")
    if spatial_strategy not in SPATIAL_STRATEGIES:
        raise ValueError(f"spatial_strategy must be one of {SPATIAL_STRATEGIES}")
    if memory_policy not in MEMORY_POLICIES:
        raise ValueError(f"memory_policy must be one of {MEMORY_POLICIES}")
    if tile_overlap < 0 or tile_overlap * 2 >= tile_size:
        raise ValueError("tile_overlap must be non-negative and less than half tile_size")

    padded_count = next_8n_plus_5(count)
    effective = largest_8n_plus_1(padded_count + 4)
    chunks = max(1, (effective - 1) // 8 - 2)
    chosen_spatial = spatial_strategy
    if chosen_spatial == "auto":
        chosen_spatial = "adaptive_tiles" if quality_profile == "memory_safe" else "full_frame"
    chosen_memory = memory_policy
    if chosen_memory == "auto":
        chosen_memory = "staged" if quality_profile == "memory_safe" else "resident"

    base = (float(base_attention_budget), float(kv_retention), int(local_radius))
    motion = _motion_scores(frames, chunks)
    values = [base] * chunks
    chunk_report: list[dict[str, Any]] = []
    if quality_profile == "balanced_dynamic_exp" and chunks > 2:
        # Boundary chunks are always baseline guards, so calibrate the motion
        # tiers only from chunks that are actually eligible for reduction.
        sorted_motion = sorted(motion[1:-1])
        low_cut = sorted_motion[max(0, math.floor((len(sorted_motion) - 1) * 0.35))]
        high_cut = sorted_motion[max(0, math.ceil((len(sorted_motion) - 1) * 0.75))]
        values = []
        for index, score in enumerate(motion):
            if index in (0, chunks - 1) or score >= high_cut:
                value = base
                tier = "baseline_guard"
            elif score <= low_cut:
                value = (max(1.5, base[0] * 0.85), max(1.0, base[1] - 1.0), 9)
                tier = "low_motion_reduced_exp"
            else:
                value = (max(1.5, base[0] * 0.925), max(1.0, base[1] - 0.5), base[2])
                tier = "mid_motion_reduced_exp"
            values.append(value)
            chunk_report.append(
                {
                    "chunk": index,
                    "motion_score": round(score, 8),
                    "tier": tier,
                    "topk": round(value[0], 4),
                    "kv": round(value[1], 4),
                    "local": value[2],
                }
            )
    else:
        chunk_report = [
            {
                "chunk": index,
                "motion_score": round(score, 8),
                "tier": "fixed_quality" if quality_profile != "memory_safe" else "memory_route_fixed_quality",
                "topk": base[0],
                "kv": base[1],
                "local": base[2],
            }
            for index, score in enumerate(motion)
        ]

    plan = {
        "schema": REPORT_SCHEMA,
        "quality_profile": quality_profile,
        "spatial_strategy": chosen_spatial,
        "memory_policy": chosen_memory,
        "base_attention_budget": base[0],
        "kv_retention": base[1],
        "local_radius": base[2],
        "tile_size": int(tile_size),
        "tile_overlap": int(tile_overlap),
        "input": {"frames": count, "width": width, "height": height},
        "padded_frames": padded_count,
        "denoise_chunks": chunks,
        "budget_values": values,
        "chunk_report": chunk_report,
        "quality_contract": (
            "quality_locked keeps the published 2.0/3.0/11 LCSA budget; "
            "balanced_dynamic_exp is opt-in and not claimed bit-exact"
        ),
    }
    return plan, _canonical_json(plan)


def _tile_coordinates(height: int, width: int, tile_size: int, overlap: int):
    tile_h = min(tile_size, height)
    tile_w = min(tile_size, width)
    stride_h = max(1, tile_h - overlap)
    stride_w = max(1, tile_w - overlap)
    rows = max(1, math.ceil(max(0, height - overlap) / stride_h))
    cols = max(1, math.ceil(max(0, width - overlap) / stride_w))
    coords = []
    for row in range(rows):
        for col in range(cols):
            y1 = min(row * stride_h, height - tile_h)
            x1 = min(col * stride_w, width - tile_w)
            item = (x1, y1, x1 + tile_w, y1 + tile_h)
            if item not in coords:
                coords.append(item)
    return coords


def _feather_mask(height: int, width: int, overlap: int) -> torch.Tensor:
    mask = torch.ones(1, height, width, 1, dtype=torch.float32)
    fade = min(overlap, height // 2, width // 2)
    if fade <= 0:
        return mask
    ramp = torch.linspace(1.0 / (fade + 1), 1.0, fade)
    mask[:, :, :fade] *= ramp.view(1, 1, -1, 1)
    mask[:, :, -fade:] *= ramp.flip(0).view(1, 1, -1, 1)
    mask[:, :fade] *= ramp.view(1, -1, 1, 1)
    mask[:, -fade:] *= ramp.flip(0).view(1, -1, 1, 1)
    return mask


def _run_full_frame(
    handle: FlashVSRModelHandle,
    frames: torch.Tensor,
    *,
    scale: int,
    seed: int,
    color_fix: bool,
    controller: MotionBudgetController,
    staged: bool,
) -> torch.Tensor:
    padded_count = next_8n_plus_5(int(frames.shape[0]))
    padded = frames
    if padded_count > frames.shape[0]:
        padded = torch.cat(
            (frames, frames[-1:].repeat(padded_count - frames.shape[0], 1, 1, 1)), dim=0
        )
    low_quality, target_height, target_width, effective_frames = _prepare_input_tensor(
        padded, scale=scale, dtype=handle.dtype
    )
    if handle.mode != "tiny_long":
        low_quality = low_quality.to(handle.device)
    output = handle.pipe(
        prompt="",
        negative_prompt="",
        cfg_scale=1.0,
        num_inference_steps=1,
        seed=int(seed),
        tiled=True,
        LQ_video=low_quality,
        num_frames=effective_frames,
        height=target_height,
        width=target_width,
        is_full_block=False,
        if_buffer=True,
        topk_ratio=float(controller._values[0][0]),
        kv_ratio=float(controller._values[0][1]),
        local_range=int(controller._values[0][2]),
        budget_controller=controller,
        color_fix=bool(color_fix),
        stage_memory=bool(staged),
    )
    result = _tensor_to_frames(output).cpu()
    del output, low_quality
    return result[: frames.shape[0]]


def _run_tiled(
    handle: FlashVSRModelHandle,
    frames: torch.Tensor,
    *,
    scale: int,
    seed: int,
    color_fix: bool,
    controller: MotionBudgetController,
    staged: bool,
    tile_size: int,
    tile_overlap: int,
) -> tuple[torch.Tensor, int]:
    _, height, width, _ = _validate_frames(frames)
    output_height, output_width = height * scale, width * scale
    canvas = torch.zeros(frames.shape[0], output_height, output_width, 3, dtype=torch.float32)
    weights = torch.zeros(frames.shape[0], output_height, output_width, 1, dtype=torch.float32)
    coords = _tile_coordinates(height, width, tile_size, tile_overlap)
    for index, (x1, y1, x2, y2) in enumerate(coords):
        tile = frames[:, y1:y2, x1:x2]
        restored = _run_full_frame(
            handle,
            tile,
            scale=scale,
            # Keep one noise field across every spatial tile.  Changing the seed per
            # tile creates independently sampled overlap regions and can turn the
            # feather blend into a visible temporal seam.
            seed=seed,
            color_fix=color_fix,
            controller=controller,
            staged=staged,
        )
        expected_height = (y2 - y1) * scale
        expected_width = (x2 - x1) * scale
        if restored.shape[1:3] != (expected_height, expected_width):
            restored = F.interpolate(
                restored.permute(0, 3, 1, 2),
                size=(expected_height, expected_width),
                mode="bicubic",
                align_corners=False,
            ).permute(0, 2, 3, 1).clamp(0.0, 1.0)
        mask = _feather_mask(
            expected_height, expected_width, tile_overlap * scale
        )
        ox1, oy1 = x1 * scale, y1 * scale
        ox2, oy2 = ox1 + expected_width, oy1 + expected_height
        canvas[:, oy1:oy2, ox1:ox2] += restored * mask
        weights[:, oy1:oy2, ox1:ox2] += mask
        del restored, tile
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return canvas.div_(weights.clamp_min_(1e-6)).clamp_(0.0, 1.0), len(coords)


def _release_handle(handle: FlashVSRModelHandle, policy: str) -> dict[str, Any]:
    import comfy.model_management as model_management

    if policy == "keep_loaded":
        return {"policy": policy, "cache_retained": True}
    try:
        handle.pipe.offload_model()
    finally:
        if policy == "clear_after":
            _MODEL_CACHE.pop(handle.cache_key, None)
        gc.collect()
        model_management.soft_empty_cache()
    return {"policy": policy, "cache_retained": policy == "offload_after"}


def restore_flashvsr(
    model: FlashVSRModelHandle,
    plan: dict[str, Any],
    frames: torch.Tensor,
    audio: Any = None,
    *,
    scale: int = 2,
    seed: int = 0,
    color_fix: bool = True,
    release_policy: str = "offload_after",
) -> tuple[torch.Tensor, torch.Tensor, Any, str]:
    count, height, width, channels = _validate_frames(frames)
    if scale not in (2, 4):
        raise ValueError("FlashVSR scale must be 2 or 4")
    if release_policy not in RELEASE_POLICIES:
        raise ValueError(f"release_policy must be one of {RELEASE_POLICIES}")
    if not isinstance(plan, dict) or plan.get("schema") != REPORT_SCHEMA:
        raise ValueError("plan must come from MiniMax H3 FlashVSR Execution Plan")

    source = frames.detach().float().cpu().contiguous()
    values = [tuple(item) for item in plan.get("budget_values", [])]
    if not values:
        values = [
            (
                float(plan["base_attention_budget"]),
                float(plan["kv_retention"]),
                int(plan["local_radius"]),
            )
        ]
    controller = MotionBudgetController(values, list(plan.get("chunk_report", [])))
    started = time.perf_counter()
    tile_count = 1
    release: dict[str, Any] = {"policy": release_policy, "completed": False}
    try:
        if plan["spatial_strategy"] == "adaptive_tiles":
            result, tile_count = _run_tiled(
                model,
                source,
                scale=scale,
                seed=seed,
                color_fix=color_fix,
                controller=controller,
                staged=plan["memory_policy"] == "staged",
                tile_size=int(plan["tile_size"]),
                tile_overlap=int(plan["tile_overlap"]),
            )
        else:
            result = _run_full_frame(
                model,
                source,
                scale=scale,
                seed=seed,
                color_fix=color_fix,
                controller=controller,
                staged=plan["memory_policy"] == "staged",
            )
        if channels > 3:
            extras = F.interpolate(
                source[..., 3:].permute(0, 3, 1, 2),
                size=result.shape[1:3],
                mode="bilinear",
                align_corners=False,
            ).permute(0, 2, 3, 1)
            result = torch.cat((result, extras), dim=-1)
    finally:
        release = _release_handle(model, release_policy)
        release["completed"] = True

    report = {
        "schema": REPORT_SCHEMA,
        "operation": "restore",
        "backend": "FlashVSR v1.1 public streaming core + spas_sage_attn LCSA",
        "model": {"name": model.model_name, "mode": model.mode, "directory": str(model.model_dir)},
        "input": {"frames": count, "width": width, "height": height, "channels": channels},
        "output": {
            "frames": int(result.shape[0]),
            "width": int(result.shape[2]),
            "height": int(result.shape[1]),
            "scale_requested": scale,
        },
        "execution": {
            "quality_profile": plan["quality_profile"],
            "spatial_strategy": plan["spatial_strategy"],
            "memory_policy": plan["memory_policy"],
            "tile_count": tile_count,
            "color_fix": bool(color_fix),
            "chunk_budgets": controller.report,
        },
        "audio": {
            "provided": audio is not None,
            "exact_object_passthrough": True,
            "resampled": False,
            "modified": False,
        },
        "release": release,
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "limits": (
            "Post-processing cannot reconstruct missing identity or lip sync. "
            "balanced_dynamic_exp changes sparse-attention budgets and requires visual review."
        ),
    }
    return result.contiguous(), source, audio, _canonical_json(report)
