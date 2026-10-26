"""Independent short SolarRenderer protocol adapter; no node registration.

Reference: Enderfga/SolarRenderer 00b89093234a194bb06e12b7df0d2ed354769a8d,
comfyui/nodes.py (Apache-2.0). Model derivatives have their separate Community
License. This module delegates the existing LoRA loader, VAE and Core layout.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch
from safetensors import safe_open
import comfy.model_management
import comfy.utils

from .core import empty_av_latent, video_latent_t
from .h3_lora_compat_advanced import load_minimax_h3_lora_model

SOURCE_COMMIT = "00b89093234a194bb06e12b7df0d2ed354769a8d"
MODEL_REVISION = "c5512be825f297ed8db390cbeda0289fb0c24779"
SILENCE_SHA256 = "e3be621209aee7a7833bdc17316c15522cf3bbd222f9185c44fced9c756ae331"
FPS = 24


def finite_tensor(value, name):
    if (not isinstance(value, torch.Tensor) or value.layout != torch.strided
            or value.device.type == "meta" or not value.is_floating_point()
            or value.numel() == 0 or not bool(torch.isfinite(value).all())):
        raise ValueError(f"{name} requires a nonempty finite floating tensor")
    return value


def file_identity(path, expected_sha256=None):
    path = Path(path).resolve(strict=True)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            comfy.model_management.throw_exception_if_processing_interrupted()
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Solar selected file changed while its SHA was read")
    sha = digest.hexdigest()
    if expected_sha256 is not None and sha != expected_sha256:
        raise ValueError(f"Solar selected file SHA256 differs from its supplied receipt: {path.name}")
    return {"path": str(path), "bytes": after.st_size, "sha256": sha,
            "mtime_ns": after.st_mtime_ns}


def _unchanged(identity):
    stat = Path(identity["path"]).stat()
    if (stat.st_size, stat.st_mtime_ns) != (identity["bytes"], identity["mtime_ns"]):
        raise ValueError("Solar selected file changed during loading")


def load_silence(path, *, key="silence_170f", expected_sha256=None):
    """Read an explicitly selected encoded waveform; never substitute zeros."""
    identity = file_identity(path, expected_sha256)
    with safe_open(identity["path"], framework="pt", device="cpu") as handle:
        if key not in handle.keys():
            raise ValueError(f"Selected silence asset has no tensor {key}")
        value = finite_tensor(handle.get_tensor(key), "encoded silence")
        metadata = handle.metadata() or {}
    _unchanged(identity)
    if value.ndim != 3 or tuple(value.shape[:2]) != (2, 32):
        raise ValueError("Encoded silence must have [stereo=2, channels=32, audio ticks]")
    identity.update(tensor_key=key, source_metadata=metadata,
                    fixed_source_bytes=identity["sha256"] == SILENCE_SHA256)
    return value.permute(1, 0, 2).unsqueeze(0).float().contiguous(), identity


def load_lora_pair(model, fewstep_path, control_path, *, fewstep_strength=1.,
                   control_strength=1., expected_sha256=None):
    """Append the selected pair in source order; keep all earlier user patches."""
    expected = {} if expected_sha256 is None else expected_sha256
    selected, receipts = model, []
    for role, path, strength in (("fewstep", fewstep_path, fewstep_strength),
                                  ("control", control_path, control_strength)):
        if isinstance(strength, bool) or not math.isfinite(float(strength)):
            raise ValueError("Solar LoRA strengths must be finite")
        identity = file_identity(path, expected.get(role))
        selected, raw = load_minimax_h3_lora_model(selected, identity["path"], strength)
        _unchanged(identity)
        receipts.append({"role": role, "file": identity, "loader": json.loads(raw)})
    return selected, {"order": ["fewstep", "control"], "adapters": receipts,
                      "trained_consumer_qualified": False}


def plan_structure(source_frames, length, width, height, *, source_start=0, fps=FPS):
    for name, value, minimum in (("source_frames", source_frames, 1), ("length", length, 1),
                                 ("width", width, 32), ("height", height, 32),
                                 ("source_start", source_start, 0)):
        if type(value) is not int or value < minimum:
            raise ValueError(f"Solar {name} must be an integer >= {minimum}")
    if isinstance(fps, bool) or not math.isfinite(float(fps)) or float(fps) != FPS:
        raise ValueError("Solar's frame-aligned condition clock is 24 fps; conform source frames explicitly first")
    if width % 32 or height % 32:
        raise ValueError("Solar's H3 target canvas must be divisible by 32")
    available = min(length, source_frames - source_start)
    frames = available - (available - 5) % 17
    if frames < 5:
        raise ValueError("Selected structure interval cannot cover a complete 17n+5 H3 frame grid")
    shown = frames if frames <= 362 else 158
    warnings = []
    if frames != length:
        warnings.append(f"Requested {length} frames; selected source/grid delivers {frames} frames")
    if frames != 158:
        warnings.append("The published short pair was trained at 158 frames; this length is unqualified")
    if frames > 362:
        warnings.append("This is full-clip execution; no causal sampler or constant-memory claim")
    return {"source_commit": SOURCE_COMMIT, "width": width, "height": height, "fps": FPS,
            "requested_frames": length, "frames": frames, "source_frames": source_frames,
            "source_range": [source_start, source_start + frames],
            "video_ticks": video_latent_t(frames), "audio_ticks": round(frames * 40 / FPS),
            "qwen_source_indices": list(range(source_start, source_start + shown, 12)),
            "warnings": warnings, "trained_consumer_qualified": False}


def _fit(frames, width, height):
    finite_tensor(frames, "structure/first frame")
    if frames.ndim != 4 or frames.shape[-1] < 3:
        raise ValueError("Solar images require [frames, height, width, RGB channels]")
    value = frames[..., :3].movedim(-1, 1)
    if tuple(value.shape[-2:]) != (height, width):
        value = comfy.utils.common_upscale(value, width, height, "lanczos", "center")
    return value.movedim(1, -1).clamp(0., 1.)


def _shown(frames):
    indices = list(range(0, frames.shape[0], 12))
    return {"type": "video", "data": frames[indices],
            "timestamps": [index / 2. for index in range(len(indices))]}


def encode_structure(clip, vae, structure, prompt, width, height, length, *, first_frame=None,
                     structure_vae=None, source_start=0, first_frame_source_index=None, fps=FPS):
    """Dense target-time keyframes plus an optional aligned appearance frame.

The explicit first-frame index binds its declared source coordinate. It cannot
prove visual identity; no pixel-equality test is imposed on a styled first frame.
"""
    finite_tensor(structure, "structure")
    plan = plan_structure(structure.shape[0], length, width, height, source_start=source_start, fps=fps)
    for owner, methods in ((clip, ("tokenize", "encode_from_tokens_scheduled")), (vae, ("encode",))):
        for method in methods:
            if not callable(getattr(owner, method, None)):
                raise TypeError(f"Solar selected encoder needs callable {method}")
    structure_vae = vae if structure_vae is None else structure_vae
    if not callable(getattr(structure_vae, "encode", None)):
        raise TypeError("Solar structure VAE needs callable encode")
    start, stop = plan["source_range"]
    video = _fit(structure[start:stop], width, height)
    items = [_shown(video if plan["frames"] <= 362 else video[:158])]
    still = None
    if first_frame is not None:
        if first_frame_source_index is not None and (
                type(first_frame_source_index) is not int or first_frame_source_index != start):
            raise ValueError("Appearance first-frame source coordinate must equal the selected structure start")
        still = _fit(first_frame, width, height)
        if still.shape[0] != 1:
            raise ValueError("Solar appearance input must contain exactly one first frame")
        items.append(_shown(still.repeat(13, 1, 1, 1)))
        if first_frame_source_index is None:
            plan["warnings"].append("Appearance source coordinate was not supplied; alignment is caller asserted")
    positive = clip.encode_from_tokens_scheduled(clip.tokenize(prompt, minimax_ref_items=items))
    encoded = finite_tensor(structure_vae.encode(video), "encoded structure")
    shape = (1, 24, plan["video_ticks"], height // 16, width // 16)
    if tuple(encoded.shape) != shape:
        raise ValueError(f"Structure VAE output {tuple(encoded.shape)} must cover target latent shape {shape}")
    keyframes = [{"resolved_frame_index": 0, "latent": encoded}]
    if still is not None:
        anchor = finite_tensor(vae.encode(still), "encoded first frame")
        if tuple(anchor.shape) != (1, 24, 1, height // 16, width // 16):
            raise ValueError("Appearance VAE output must cover exactly target frame 0 on the same canvas")
        keyframes.append({"resolved_frame_index": 0, "latent": anchor})
    if not isinstance(positive, (list, tuple)) or not positive:
        raise ValueError("Selected CLIP returned no conditioning rows")
    result = []
    for row in positive:
        metadata = dict(row[1])
        metadata["minimax_keyframes"] = [*(metadata.get("minimax_keyframes") or []), *keyframes]
        metadata["minimax_frame_count"] = plan["frames"]
        result.append([row[0], metadata])
    latent, delivered = empty_av_latent(width, height, plan["frames"])
    if delivered != plan["frames"]:
        raise ValueError("Target latent frame clock changed during Solar preparation")
    plan.update(first_frame_source_index=first_frame_source_index,
                first_frame_content_verified=False, structural_keyframes=len(keyframes),
                soundtrack="encoded_silence_during_model_evaluation; original_soundtrack_mux_is_not_audio_driving")
    return result, latent, plan
