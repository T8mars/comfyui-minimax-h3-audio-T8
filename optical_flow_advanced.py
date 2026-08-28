from __future__ import annotations

import gc
import hashlib
import json
import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


FLOW_AUDIT_SCHEMA = "h3_t8_optical_flow_audit/v1"
FLOW_MASK_SCHEMA = "h3_t8_flow_mask_propagation/v1"
MODEL_TYPES = ("raft_small", "raft_large")
PRECISIONS = ("auto", "fp32", "fp16")
RELEASE_POLICIES = ("offload_after", "clear_after", "keep_loaded")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _validate_frames(frames: torch.Tensor) -> tuple[int, int, int]:
    if not isinstance(frames, torch.Tensor) or frames.ndim != 4:
        raise ValueError("frames must be a ComfyUI IMAGE batch [T,H,W,C]")
    if frames.shape[-1] < 3:
        raise ValueError("frames must contain at least three RGB channels")
    count, height, width = (int(frames.shape[0]), int(frames.shape[1]), int(frames.shape[2]))
    if count < 2:
        raise ValueError("optical-flow analysis requires at least two frames")
    if min(height, width) < 16:
        raise ValueError("frames are too small for optical-flow analysis")
    if not torch.isfinite(frames[..., :3]).all():
        raise ValueError("frames contain NaN or Inf")
    return count, height, width


def _analysis_size(height: int, width: int, max_side: int) -> tuple[int, int]:
    if max_side <= 0 or max(height, width) <= max_side:
        return height, width
    scale = float(max_side) / float(max(height, width))
    return max(16, int(round(height * scale))), max(16, int(round(width * scale)))


def _analysis_frames(frames: torch.Tensor, max_side: int) -> torch.Tensor:
    _, height, width = _validate_frames(frames)
    target_h, target_w = _analysis_size(height, width, int(max_side))
    rgb = frames[..., :3].detach().float().cpu().permute(0, 3, 1, 2).contiguous()
    if (target_h, target_w) != (height, width):
        rgb = F.interpolate(rgb, size=(target_h, target_w), mode="bilinear", align_corners=False)
    return rgb.clamp(0.0, 1.0)


def _unwrap_state_dict(value: Any) -> dict[str, torch.Tensor]:
    if isinstance(value, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            nested = value.get(key)
            if isinstance(nested, dict) and nested:
                value = nested
                break
    if not isinstance(value, dict) or not value:
        raise ValueError("RAFT checkpoint does not contain a state dict")
    state = {}
    for key, tensor in value.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        normalized = str(key)
        for prefix in ("module.", "model."):
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix) :]
        state[normalized] = tensor
    if not state:
        raise ValueError("RAFT checkpoint contains no tensor weights")
    return state


def _device() -> torch.device:
    try:
        import comfy.model_management as model_management

        return model_management.get_torch_device()
    except (ImportError, AttributeError):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _dtype_for(precision: str, device: torch.device) -> torch.dtype:
    if precision not in PRECISIONS:
        raise ValueError(f"Unknown precision {precision!r}")
    if precision == "fp16":
        if device.type == "cpu":
            raise ValueError("fp16 RAFT execution requires a CUDA-class device")
        return torch.float16
    # RAFT correlation and recurrent updates are most stable in fp32.  Auto
    # deliberately chooses correctness instead of silently enabling half.
    return torch.float32


@dataclass
class _CachedRAFT:
    model: torch.nn.Module
    path: Path
    model_type: str
    precision: str


_MODEL_CACHE: dict[tuple[str, int, int, str, str], _CachedRAFT] = {}
_MODEL_LOCK = threading.RLock()


def _cache_key(path: Path, model_type: str, precision: str) -> tuple[str, int, int, str, str]:
    stat = path.stat()
    return (
        str(path.resolve()).casefold(),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        model_type,
        precision,
    )


def clear_raft_cache() -> int:
    with _MODEL_LOCK:
        entries = list(_MODEL_CACHE.values())
        _MODEL_CACHE.clear()
    for entry in entries:
        entry.model.to(torch.device("cpu"))
    gc.collect()
    try:
        import comfy.model_management as model_management

        model_management.soft_empty_cache()
    except (ImportError, AttributeError):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return len(entries)


def _load_raft(model_path: Path, model_type: str, precision: str) -> tuple[_CachedRAFT, bool]:
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Unknown model_type {model_type!r}")
    if not model_path.is_file():
        raise ValueError(f"RAFT checkpoint not found: {model_path}")
    key = _cache_key(model_path, model_type, precision)
    with _MODEL_LOCK:
        cached = _MODEL_CACHE.get(key)
        if cached is not None:
            return cached, True

        from torchvision.models.optical_flow import raft_large, raft_small

        builder = raft_small if model_type == "raft_small" else raft_large
        network = builder(weights=None, progress=False)
        try:
            import comfy.utils

            raw = comfy.utils.load_torch_file(str(model_path), safe_load=True)
        except ImportError:
            raw = torch.load(str(model_path), map_location="cpu", weights_only=True)
        state = _unwrap_state_dict(raw)
        # No filename, byte-size or hash gate is used.  The explicit architecture
        # chosen by the user is authoritative; incompatible tensors raise the
        # native PyTorch load error instead of being guessed from a fingerprint.
        network.load_state_dict(state, strict=True)
        network.eval().requires_grad_(False)
        entry = _CachedRAFT(network, model_path, model_type, precision)
        _MODEL_CACHE[key] = entry
        return entry, False


def _release_raft(entry: _CachedRAFT, release_policy: str) -> dict[str, Any]:
    if release_policy not in RELEASE_POLICIES:
        raise ValueError(f"Unknown release_policy {release_policy!r}")
    result = {
        "policy": release_policy,
        "gpu_weights_released": False,
        "cpu_cache_cleared": False,
        "global_unload_called": False,
    }
    if release_policy == "keep_loaded":
        return result
    entry.model.to(torch.device("cpu"))
    result["gpu_weights_released"] = True
    if release_policy == "clear_after":
        with _MODEL_LOCK:
            for key, value in list(_MODEL_CACHE.items()):
                if value is entry:
                    del _MODEL_CACHE[key]
        result["cpu_cache_cleared"] = True
    gc.collect()
    try:
        import comfy.model_management as model_management

        model_management.soft_empty_cache()
    except (ImportError, AttributeError):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return result


def _pad_to_eight(value: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
    height, width = value.shape[-2:]
    # Torchvision RAFT builds a four-level correlation pyramid after an 8x
    # encoder reduction.  Each encoded side therefore needs at least 16 cells,
    # which means the padded input must be at least 128 pixels on both axes.
    padded_h = max(128, int(math.ceil(height / 8.0) * 8))
    padded_w = max(128, int(math.ceil(width / 8.0) * 8))
    pad_h = padded_h - height
    pad_w = padded_w - width
    if pad_h or pad_w:
        value = F.pad(value, (0, pad_w, 0, pad_h), mode="replicate")
    return value, (height, width)


@torch.inference_mode()
def _estimate_batch(
    model: torch.nn.Module,
    image_a: torch.Tensor,
    image_b: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    a, original = _pad_to_eight(image_a)
    b, _ = _pad_to_eight(image_b)
    a = a.to(device=device, dtype=dtype, non_blocking=True).mul(2.0).sub(1.0)
    b = b.to(device=device, dtype=dtype, non_blocking=True).mul(2.0).sub(1.0)
    predictions = model(a, b)
    flow = predictions[-1][..., : original[0], : original[1]].float().cpu()
    del a, b, predictions
    return flow


def _estimate_sequence(
    frames_bchw: torch.Tensor,
    entry: _CachedRAFT,
    *,
    precision: str,
    pair_batch_size: int,
    bidirectional: bool,
) -> tuple[list[torch.Tensor], list[torch.Tensor] | None]:
    device = _device()
    dtype = _dtype_for(precision, device)
    entry.model.to(device=device, dtype=dtype)
    forward: list[torch.Tensor] = []
    backward: list[torch.Tensor] | None = [] if bidirectional else None
    batch = max(1, int(pair_batch_size))
    for start in range(0, int(frames_bchw.shape[0]) - 1, batch):
        end = min(int(frames_bchw.shape[0]) - 1, start + batch)
        a = frames_bchw[start:end]
        b = frames_bchw[start + 1 : end + 1]
        fwd = _estimate_batch(entry.model, a, b, device=device, dtype=dtype)
        forward.extend(t.contiguous() for t in fwd)
        if backward is not None:
            bwd = _estimate_batch(entry.model, b, a, device=device, dtype=dtype)
            backward.extend(t.contiguous() for t in bwd)
    return forward, backward


def _sample_with_flow(source: torch.Tensor, target_to_source: torch.Tensor) -> torch.Tensor:
    if source.ndim == 2:
        source = source[None, None]
    elif source.ndim == 3:
        source = source[None]
    if target_to_source.ndim == 3:
        target_to_source = target_to_source[None]
    _, _, height, width = target_to_source.shape
    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=torch.float32, device=target_to_source.device),
        torch.arange(width, dtype=torch.float32, device=target_to_source.device),
        indexing="ij",
    )
    x = xx[None] + target_to_source[:, 0]
    y = yy[None] + target_to_source[:, 1]
    if width > 1:
        x = x.mul(2.0 / (width - 1)).sub(1.0)
    else:
        x = torch.zeros_like(x)
    if height > 1:
        y = y.mul(2.0 / (height - 1)).sub(1.0)
    else:
        y = torch.zeros_like(y)
    grid = torch.stack((x, y), dim=-1)
    return F.grid_sample(
        source.float(), grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )


def _pair_confidence(
    target_to_source: torch.Tensor,
    source_to_target: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    sampled_reverse = _sample_with_flow(source_to_target, target_to_source)
    error = torch.linalg.vector_norm(target_to_source[None].float() + sampled_reverse, dim=1)
    return torch.exp(-error / max(float(threshold), 1e-4)).clamp(0.0, 1.0)


def _parse_indices(value: str, frame_count: int, mask_count: int) -> list[int]:
    parts = [part.strip() for part in str(value).replace(";", ",").split(",") if part.strip()]
    if not parts:
        raise ValueError("keyframe_indices must contain comma-separated frame indices")
    try:
        indices = [int(part) for part in parts]
    except ValueError as error:
        raise ValueError("keyframe_indices must contain integers only") from error
    if len(indices) != mask_count:
        raise ValueError(
            f"keyframe_indices contains {len(indices)} entries but the MASK batch has {mask_count} masks"
        )
    if indices != sorted(set(indices)):
        raise ValueError("keyframe_indices must be unique and strictly increasing")
    if indices[0] < 0 or indices[-1] >= frame_count:
        raise ValueError(f"keyframe_indices must stay within 0..{frame_count - 1}")
    return indices


def _scene_deltas(frames_bchw: torch.Tensor) -> list[float]:
    small = F.interpolate(frames_bchw, size=(64, 64), mode="area")
    return [float((small[index + 1] - small[index]).abs().mean()) for index in range(len(small) - 1)]


def _hsv_to_rgb(hue: torch.Tensor, saturation: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    h6 = (hue.remainder(1.0) * 6.0)
    sector = torch.floor(h6).to(torch.int64)
    fraction = h6 - sector.float()
    p = value * (1.0 - saturation)
    q = value * (1.0 - saturation * fraction)
    t = value * (1.0 - saturation * (1.0 - fraction))
    choices = (
        torch.stack((value, t, p), dim=-1),
        torch.stack((q, value, p), dim=-1),
        torch.stack((p, value, t), dim=-1),
        torch.stack((p, q, value), dim=-1),
        torch.stack((t, p, value), dim=-1),
        torch.stack((value, p, q), dim=-1),
    )
    output = torch.zeros_like(choices[0])
    for index, choice in enumerate(choices):
        output = torch.where((sector == index)[..., None], choice, output)
    return output


def _flow_preview(flows: list[torch.Tensor], maximum: int = 12) -> torch.Tensor:
    if not flows:
        return torch.zeros((1, 64, 64, 3), dtype=torch.float32)
    indices = torch.linspace(0, len(flows) - 1, steps=min(maximum, len(flows))).round().int().tolist()
    previews = []
    for index in indices:
        flow = flows[index].float()
        magnitude = torch.linalg.vector_norm(flow, dim=0)
        scale = float(torch.quantile(magnitude.flatten(), 0.95)) if magnitude.numel() else 1.0
        hue = (torch.atan2(flow[1], flow[0]) + math.pi) / (2.0 * math.pi)
        saturation = (magnitude / max(scale, 1e-6)).clamp(0.0, 1.0)
        previews.append(_hsv_to_rgb(hue, saturation, torch.ones_like(hue) * 0.92))
    return torch.stack(previews).clamp(0.0, 1.0)


def audit_optical_flow(
    *,
    frames: torch.Tensor,
    model_path: Path,
    model_name: str,
    model_type: str,
    precision: str,
    analysis_max_side: int,
    pair_batch_size: int,
    consistency_check: bool,
    scene_cut_threshold: float,
    release_policy: str,
) -> tuple[torch.Tensor, str, float, float, int]:
    frame_count, height, width = _validate_frames(frames)
    analysis = _analysis_frames(frames, int(analysis_max_side))
    entry, cache_hit = _load_raft(model_path, model_type, precision)
    failed = True
    release = None
    try:
        forward, backward = _estimate_sequence(
            analysis,
            entry,
            precision=precision,
            pair_batch_size=pair_batch_size,
            bidirectional=bool(consistency_check),
        )
        magnitudes = [torch.linalg.vector_norm(flow.float(), dim=0) for flow in forward]
        mean_motion = float(torch.stack([value.mean() for value in magnitudes]).mean())
        p95_motion = float(torch.quantile(torch.cat([value.flatten() for value in magnitudes]), 0.95))
        deltas = _scene_deltas(analysis)
        scene_cuts = [index + 1 for index, value in enumerate(deltas) if value >= scene_cut_threshold]
        consistency_mean = None
        if backward is not None:
            confidences = [
                _pair_confidence(backward[index], forward[index], max(1.0, p95_motion * 0.15))
                for index in range(len(forward))
            ]
            consistency_mean = float(torch.stack([value.mean() for value in confidences]).mean())
        report = {
            "schema": FLOW_AUDIT_SCHEMA,
            "status": "ok",
            "scientific_scope": (
                "read_only_pairwise_optical_flow; diagnoses motion and propagates no pixels; "
                "it is not a face repair, deblur or H3 denoising method"
            ),
            "backend": {
                "family": "RAFT",
                "implementation": "torchvision",
                "model_type": model_type,
                "model_name": model_name,
                "cache_hit": cache_hit,
                "identity_policy": "user_selected_architecture; no filename/hash/size allowlist",
                "sea_raft_note": (
                    "This node uses original RAFT-compatible torchvision inference. SEA-RAFT remains "
                    "a separately labeled backend because its uncertainty head and checkpoint contract differ."
                ),
            },
            "source": {"frames": frame_count, "height": height, "width": width},
            "analysis": {
                "height": int(analysis.shape[-2]),
                "width": int(analysis.shape[-1]),
                "pair_count": len(forward),
                "mean_motion_px": mean_motion,
                "p95_motion_px": p95_motion,
                "forward_backward_confidence_mean": consistency_mean,
                "scene_cut_threshold": float(scene_cut_threshold),
                "scene_cut_frames": scene_cuts,
            },
        }
        failed = False
        preview = _flow_preview(forward)
        return preview, _canonical_json(report), mean_motion, p95_motion, len(scene_cuts)
    finally:
        if failed or release_policy != "keep_loaded":
            release = _release_raft(entry, "clear_after" if failed else release_policy)
        del release


def propagate_masks_with_flows(
    *,
    keyframe_masks: torch.Tensor,
    keyframe_indices: list[int],
    forward: list[torch.Tensor],
    backward: list[torch.Tensor],
    scene_deltas: list[float],
    scene_cut_threshold: float,
    consistency_threshold: float,
    minimum_confidence: float,
    extend_edges: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    frame_count = len(forward) + 1
    if len(backward) != len(forward):
        raise ValueError("forward and backward flow lists must have equal length")
    if len(keyframe_indices) != int(keyframe_masks.shape[0]):
        raise ValueError("keyframe mask count does not match keyframe_indices")
    height, width = forward[0].shape[-2:]
    anchors = F.interpolate(
        keyframe_masks[:, None].float(), size=(height, width), mode="bilinear", align_corners=False
    )[:, 0].clamp(0.0, 1.0)
    masks = torch.zeros((frame_count, height, width), dtype=torch.float32)
    confidence = torch.zeros_like(masks)
    for anchor_index, frame_index in enumerate(keyframe_indices):
        masks[frame_index] = anchors[anchor_index]
        confidence[frame_index] = 1.0

    valid_pair = [float(value) < float(scene_cut_threshold) for value in scene_deltas]

    def step(source_mask, source_conf, target_to_source, source_to_target, pair_ok):
        if not pair_ok:
            return torch.zeros_like(source_mask), torch.zeros_like(source_conf)
        warped_mask = _sample_with_flow(source_mask, target_to_source)[0, 0].cpu()
        warped_conf = _sample_with_flow(source_conf, target_to_source)[0, 0].cpu()
        pair_conf = _pair_confidence(
            target_to_source, source_to_target, consistency_threshold
        )[0].cpu()
        return warped_mask, (warped_conf * pair_conf).clamp(0.0, 1.0)

    for interval in range(len(keyframe_indices) - 1):
        left = keyframe_indices[interval]
        right = keyframe_indices[interval + 1]
        left_masks = {left: masks[left].clone()}
        left_conf = {left: confidence[left].clone()}
        for target in range(left + 1, right + 1):
            left_masks[target], left_conf[target] = step(
                left_masks[target - 1],
                left_conf[target - 1],
                backward[target - 1],
                forward[target - 1],
                valid_pair[target - 1],
            )
        right_masks = {right: masks[right].clone()}
        right_conf = {right: confidence[right].clone()}
        for target in range(right - 1, left - 1, -1):
            right_masks[target], right_conf[target] = step(
                right_masks[target + 1],
                right_conf[target + 1],
                forward[target],
                backward[target],
                valid_pair[target],
            )
        span = max(1, right - left)
        for target in range(left + 1, right):
            alpha = float(target - left) / float(span)
            weight_left = left_conf[target] * (1.0 - alpha)
            weight_right = right_conf[target] * alpha
            total = weight_left + weight_right
            combined = (
                left_masks[target] * weight_left + right_masks[target] * weight_right
            ) / total.clamp_min(1e-6)
            masks[target] = torch.where(total >= minimum_confidence, combined, 0.0)
            confidence[target] = total.clamp(0.0, 1.0)

    if extend_edges:
        first = keyframe_indices[0]
        for target in range(first - 1, -1, -1):
            masks[target], confidence[target] = step(
                masks[target + 1],
                confidence[target + 1],
                forward[target],
                backward[target],
                valid_pair[target],
            )
        last = keyframe_indices[-1]
        for target in range(last + 1, frame_count):
            masks[target], confidence[target] = step(
                masks[target - 1],
                confidence[target - 1],
                backward[target - 1],
                forward[target - 1],
                valid_pair[target - 1],
            )

    masks = torch.where(confidence >= minimum_confidence, masks, 0.0).clamp(0.0, 1.0)
    return masks, confidence.clamp(0.0, 1.0)


def _mask_preview(frames: torch.Tensor, masks: torch.Tensor, maximum: int = 12) -> torch.Tensor:
    count = int(frames.shape[0])
    indices = torch.linspace(0, count - 1, steps=min(maximum, count)).round().int().tolist()
    previews = []
    for index in indices:
        frame = frames[index, ..., :3].detach().float().cpu().clamp(0.0, 1.0)
        mask = masks[index].detach().float().cpu().clamp(0.0, 1.0)[..., None]
        color = torch.tensor((0.05, 0.95, 0.35), dtype=frame.dtype)
        previews.append(frame * (1.0 - mask * 0.48) + color * (mask * 0.48))
    return torch.stack(previews).clamp(0.0, 1.0)


def propagate_keyframe_masks(
    *,
    frames: torch.Tensor,
    keyframe_masks: torch.Tensor,
    keyframe_indices: str,
    model_path: Path,
    model_name: str,
    model_type: str,
    precision: str,
    analysis_max_side: int,
    pair_batch_size: int,
    scene_cut_threshold: float,
    consistency_threshold: float,
    minimum_confidence: float,
    extend_edges: bool,
    release_policy: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
    frame_count, height, width = _validate_frames(frames)
    if not isinstance(keyframe_masks, torch.Tensor) or keyframe_masks.ndim != 3:
        raise ValueError("keyframe_masks must be a ComfyUI MASK batch [K,H,W]")
    indices = _parse_indices(keyframe_indices, frame_count, int(keyframe_masks.shape[0]))
    analysis = _analysis_frames(frames, int(analysis_max_side))
    entry, cache_hit = _load_raft(model_path, model_type, precision)
    failed = True
    try:
        forward, backward_optional = _estimate_sequence(
            analysis,
            entry,
            precision=precision,
            pair_batch_size=pair_batch_size,
            bidirectional=True,
        )
        assert backward_optional is not None
        deltas = _scene_deltas(analysis)
        propagated, confidence = propagate_masks_with_flows(
            keyframe_masks=keyframe_masks,
            keyframe_indices=indices,
            forward=forward,
            backward=backward_optional,
            scene_deltas=deltas,
            scene_cut_threshold=scene_cut_threshold,
            consistency_threshold=consistency_threshold,
            minimum_confidence=minimum_confidence,
            extend_edges=extend_edges,
        )
        output_masks = F.interpolate(
            propagated[:, None], size=(height, width), mode="bilinear", align_corners=False
        )[:, 0].clamp(0.0, 1.0)
        output_confidence = F.interpolate(
            confidence[:, None], size=(height, width), mode="bilinear", align_corners=False
        )[:, 0].clamp(0.0, 1.0)
        cuts = [index + 1 for index, value in enumerate(deltas) if value >= scene_cut_threshold]
        report = {
            "schema": FLOW_MASK_SCHEMA,
            "status": "ok",
            "scientific_scope": (
                "bidirectional RAFT mask transport with forward-backward consistency; "
                "does not infer identity and does not replace SAM detection at scene cuts"
            ),
            "backend": {
                "family": "RAFT",
                "implementation": "torchvision",
                "model_type": model_type,
                "model_name": model_name,
                "cache_hit": cache_hit,
                "identity_policy": "user_selected_architecture; no filename/hash/size allowlist",
            },
            "source": {"frames": frame_count, "height": height, "width": width},
            "analysis": {"height": int(analysis.shape[-2]), "width": int(analysis.shape[-1])},
            "keyframe_indices": indices,
            "extend_edges": bool(extend_edges),
            "scene_cut_frames": cuts,
            "mean_confidence": float(output_confidence.mean()),
            "minimum_confidence": float(minimum_confidence),
            "automatic_identity_assignment": False,
            "recommended_use": (
                "propagate one reviewed person/object mask between detections; run once per person, "
                "and provide a new anchor after every cut or prolonged occlusion"
            ),
        }
        report["plan_sha256"] = _sha256(report)
        failed = False
        return output_masks, output_confidence, _mask_preview(frames, output_masks), _canonical_json(report)
    finally:
        if failed or release_policy != "keep_loaded":
            _release_raft(entry, "clear_after" if failed else release_policy)
