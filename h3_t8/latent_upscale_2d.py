"""Optional learned 2D visual resize with native H3 AV and mask preservation."""

from __future__ import annotations

import gc
import json
import threading
from pathlib import Path

import torch
import comfy.model_management as mm
import comfy.model_patcher
import comfy.nested_tensor
import comfy.utils
import folder_paths

from . import learned_latent_upscale_advanced as shared
from .latent_upscale import _nested_parts, _resize_mask_tensor
from .vendor.lbh_upscale2d import VideoLatentResizer, _detect_arch


_CACHE = {}
_LOCK = threading.RLock()


def build_resizer(state):
    """Infer the actual structure; PyTorch's strict key/shape load is authoritative."""
    config = _detect_arch(state)
    with torch.device("meta"):
        network = VideoLatentResizer(**config)
    network.load_state_dict(state, strict=True, assign=True)
    network.eval().requires_grad_(False)
    return network, config


def _load(model_name, precision):
    dtype = shared._precision_dtype(precision)
    path = Path(
        folder_paths.get_full_path_or_raise("latent_upscale_models", model_name)
    )
    key = shared._cache_key(path, precision)
    with _LOCK:
        if key in _CACHE:
            return _CACHE[key], True
        state = comfy.utils.load_torch_file(str(path), safe_load=True)
        if isinstance(state, dict) and isinstance(state.get("model"), dict):
            state = state["model"]
        if any(k.startswith("upscaler.") for k in state):
            state = {
                k.removeprefix("upscaler."): v
                for k, v in state.items()
                if k.startswith("upscaler.")
            }
        network, config = build_resizer(state)
        network.to(dtype=dtype)
        patcher = comfy.model_patcher.ModelPatcher(
            network,
            load_device=mm.get_torch_device(),
            offload_device=mm.unet_offload_device(),
        )
        entry = shared._CachedModel(
            patcher,
            path,
            shared._sha256_file(path),
            precision,
            {"architecture": config, "tensor_count": len(state)},
        )
        _CACHE[key] = entry
        return entry, False


def temporal_halo(network):
    """Sum the radii of every temporal call in the actual full-forward traversal."""
    if not network.temporal_blocks:
        return 0
    every = network.temporal_every
    if every <= 0:
        raise ValueError(
            "A network with temporal blocks requires positive temporal_every"
        )
    total = 0
    for side, layers in enumerate(
        (network.resizer.in_blocks, network.resizer.out_blocks)
    ):
        if len(network.temporal_blocks) <= side:
            continue
        block = network.temporal_blocks[side]
        if hasattr(block, "dwconv"):
            radius = int(block.dwconv.kernel_size[0]) // 2
        elif hasattr(block, "branches"):
            radius = max(
                int(branch[0].kernel_size[0]) // 2 for branch in block.branches
            )
        else:
            raise ValueError("Temporal block has no inspectable kernel support")
        total += len(range(0, len(layers), every)) * radius
    return total


def _forget(entry):
    with _LOCK:
        for key, cached in list(_CACHE.items()):
            if cached is entry:
                del _CACHE[key]


def clear_resizer_cache():
    with _LOCK:
        entries = list(_CACHE.values())
        _CACHE.clear()
    for entry in entries:
        mm.unload_model_and_clones(entry.patcher)
    return len(entries)


def upscale_2d(
    latent,
    model_name,
    sharpness=1,
    precision="bf16",
    chunk_frames=8,
    release_policy="offload_after",
    size_mode="scale_by",
    scale_by=2.0,
    target_megapixels=0.6,
    target_width=1024,
    target_height=576,
    aspect_policy="preserve_source",
    max_anisotropy=1.05,
):
    if release_policy not in shared.RELEASE_POLICIES:
        raise ValueError("Unknown upscaler release policy")
    if (
        not isinstance(sharpness, int)
        or isinstance(sharpness, bool)
        or not 1 <= sharpness <= 6
    ):
        raise ValueError("sharpness must be an integer author tier from 1 to 6")
    if (
        not isinstance(chunk_frames, int)
        or isinstance(chunk_frames, bool)
        or chunk_frames < 0
    ):
        raise ValueError(
            "chunk_frames must be a nonnegative integer (0 runs the full video)"
        )
    if not isinstance(latent, dict) or "samples" not in latent:
        raise ValueError("Expected a LATENT dictionary with H3 AV samples")
    video, audio = _nested_parts(latent["samples"], "samples")
    if video.shape[1] != 24:
        raise ValueError(
            f"H3 normalization requires 24 visual channels, got {video.shape[1]}"
        )
    if not torch.isfinite(video).all() or not torch.isfinite(audio).all():
        raise ValueError("Input H3 AV latent contains NaN or Inf")
    geometry = shared.learned_upscale_geometry(
        int(video.shape[-1]),
        int(video.shape[-2]),
        size_mode,
        scale_by,
        target_megapixels,
        target_width,
        target_height,
        aspect_policy,
        max_anisotropy,
        legacy_bounds=False,
    )
    height, width = geometry["output_latent_height"], geometry["output_latent_width"]
    report = {
        "schema": "t8.h3.upscale2d.v1",
        "geometry": geometry,
        "sharpness": sharpness,
        "audio_preserved": True,
    }
    if (height, width) == tuple(video.shape[-2:]):
        report.update(status="noop", model_loaded=False)
        return (
            latent,
            geometry["output_width"],
            geometry["output_height"],
            json.dumps(report),
        )

    # Validate mask before allocating weights or running a partial result.
    output = latent.copy()
    mask_status = "absent"
    if latent.get("noise_mask") is not None:
        video_mask, audio_mask = shared._nested_mask_parts(latent["noise_mask"])
        resized, mask_status = _resize_mask_tensor(
            video_mask, int(video.shape[-1]), int(video.shape[-2]), width, height
        )
        output["noise_mask"] = comfy.nested_tensor.NestedTensor((resized, audio_mask))
    entry = None
    failed = True
    try:
        entry, hit = _load(model_name, precision)
        network = entry.patcher.model
        halo = temporal_halo(network)
        frames = int(video.shape[2])
        chunk = chunk_frames or frames
        dtype = shared._precision_dtype(precision)
        active_frames = min(frames, chunk + 2 * halo)
        estimate = (
            int(video.shape[0])
            * network.resizer.conv_in.out_channels
            * active_frames
            * height
            * width
            * torch.empty((), dtype=dtype).element_size()
            * 6
        )
        mm.load_models_gpu(
            [entry.patcher], memory_required=estimate, force_full_load=True
        )
        device = entry.patcher.load_device
        result = torch.empty(
            (*video.shape[:3], height, width),
            dtype=video.dtype,
            device=mm.intermediate_device(),
        )
        calls = 0
        with torch.inference_mode():
            mean = torch.tensor(shared.LATENTS_MEAN, device=device, dtype=dtype).view(
                1, 24, 1, 1, 1
            )
            std = torch.tensor(shared.LATENTS_STD, device=device, dtype=dtype).view(
                1, 24, 1, 1, 1
            )
            for start in range(0, frames, chunk):
                mm.throw_exception_if_processing_interrupted()
                end = min(frames, start + chunk)
                lo, hi = max(0, start - halo), min(frames, end + halo)
                work = video[:, :, lo:hi].to(device=device, dtype=dtype)
                normalized = (work - mean) / std
                predicted = network._forward_full(
                    normalized, (height, width), tier=sharpness
                )
                if not torch.isfinite(predicted).all():
                    raise RuntimeError("2D visual resizer produced NaN or Inf")
                restored = predicted[:, :, start - lo : end - lo] * std + mean
                result[:, :, start:end] = restored.to(
                    device=result.device, dtype=video.dtype
                )
                calls += 1
                del work, normalized, predicted, restored
            mm.throw_exception_if_processing_interrupted()
        output["samples"] = comfy.nested_tensor.NestedTensor((result, audio))
        report.update(
            status="ok",
            model_loaded=True,
            model_sha256=entry.sha256,
            model_name=model_name,
            model_cache_hit=hit,
            contract=entry.contract,
            noise_mask=mask_status,
            chunk_frames=chunk_frames,
            temporal_halo=halo,
            actual_network_calls=calls,
            precision=precision,
            source_video_shape=list(video.shape),
            output_video_shape=list(result.shape),
            tier_consumed=bool(network.resizer.tier_dim),
            release_policy=release_policy,
        )
        if not network.resizer.tier_dim:
            report["warning"] = (
                "Selected weights have no tier channels; sharpness has no effect"
            )
        failed = False
        return (
            output,
            geometry["output_width"],
            geometry["output_height"],
            json.dumps(report),
        )
    finally:
        if entry is not None and (failed or release_policy != "keep_loaded"):
            mm.unload_model_and_clones(entry.patcher)
        if entry is not None and (failed or release_policy == "clear_after"):
            _forget(entry)
            gc.collect()
            mm.soft_empty_cache()
