from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any

import torch


FREE_NOISE_OPTION_KEY = "h3_t8_free_noise"
FREE_NOISE_SCHEMA = "h3_t8_free_noise_model_plan/v1"
MODES = ("paper_permutation", "variance_preserving_blend")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)


def _permutation_seed(base_seed: int, segment_index: int) -> int:
    payload = f"h3-t8-freenoise:{int(base_seed)}:{int(segment_index)}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def free_noise_config(model) -> dict[str, Any] | None:
    options = getattr(model, "model_options", {})
    if not isinstance(options, Mapping):
        return None
    transformer = options.get("transformer_options", {})
    if not isinstance(transformer, Mapping):
        return None
    value = transformer.get(FREE_NOISE_OPTION_KEY)
    if not isinstance(value, Mapping):
        return None
    config = dict(value)
    if config.get("schema") != FREE_NOISE_SCHEMA:
        raise ValueError("MiniMax H3 FreeNoise model plan schema is unsupported")
    return config


def build_free_noise_model(
    model,
    *,
    mode: str,
    base_seed: int,
    reuse_ratio: float,
) -> tuple[Any, str]:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    reuse_ratio = float(reuse_ratio)
    if not math.isfinite(reuse_ratio) or not 0.0 <= reuse_ratio <= 1.0:
        raise ValueError("reuse_ratio must stay within 0..1")
    if mode == "paper_permutation":
        reuse_ratio = 1.0
    config = {
        "schema": FREE_NOISE_SCHEMA,
        "mode": mode,
        "base_seed": int(base_seed),
        "reuse_ratio": reuse_ratio,
        "video_noise": "shared temporal pool with deterministic per-segment permutation",
        "audio_noise": "native independent noise; not rescheduled",
        "attention_fusion": "not implemented; H3 keeps its native trained-window attention",
        "scientific_scope": (
            "FreeNoise noise-rescheduling adaptation for independent H3 continuation windows; "
            "not a faithful reproduction of the paper's long single-latent sliding-attention path"
        ),
    }
    patched = model.clone()
    model_options = dict(getattr(patched, "model_options", {}))
    transformer = dict(model_options.get("transformer_options", {}))
    transformer[FREE_NOISE_OPTION_KEY] = config
    model_options["transformer_options"] = transformer
    patched.model_options = model_options
    return patched, _canonical_json(config)


def reschedule_h3_noise(
    noise,
    *,
    config: Mapping[str, Any],
    segment_index: int,
) -> tuple[Any, dict[str, Any]]:
    """Reschedule only the video stream of a MiniMax H3 NestedTensor noise value."""

    if not getattr(noise, "is_nested", False):
        raise ValueError("MiniMax H3 FreeNoise expects the native joint AV NestedTensor noise")
    streams = list(noise.unbind())
    if len(streams) < 2:
        raise ValueError("MiniMax H3 FreeNoise requires video and audio noise streams")
    video = streams[0]
    if not isinstance(video, torch.Tensor) or video.ndim != 5:
        raise ValueError("MiniMax H3 video noise must be [B,C,T,H,W]")
    base_seed = int(config["base_seed"])
    ratio = float(config["reuse_ratio"])
    generator = torch.Generator(device="cpu").manual_seed(base_seed)
    base = torch.randn(
        video.shape,
        generator=generator,
        device="cpu",
        dtype=torch.float32,
    ).to(dtype=video.dtype)
    temporal_tokens = int(video.shape[2])
    perm_generator = torch.Generator(device="cpu").manual_seed(
        _permutation_seed(base_seed, int(segment_index))
    )
    permutation = torch.randperm(temporal_tokens, generator=perm_generator)
    correlated = base.index_select(2, permutation)
    if ratio >= 1.0:
        rescheduled = correlated
    elif ratio <= 0.0:
        rescheduled = video
    else:
        rescheduled = (
            math.sqrt(ratio) * correlated + math.sqrt(1.0 - ratio) * video
        )
    streams[0] = rescheduled.to(device=video.device, dtype=video.dtype)
    # Native H3 NestedTensor construction is deliberately delayed until execution so
    # importing this planning module never adds a ComfyUI core dependency.
    import comfy.nested_tensor

    output = comfy.nested_tensor.NestedTensor(streams)
    report = {
        "schema": FREE_NOISE_SCHEMA,
        "mode": str(config["mode"]),
        "segment_index": int(segment_index),
        "base_seed": base_seed,
        "permutation_seed": _permutation_seed(base_seed, int(segment_index)),
        "video_temporal_tokens": temporal_tokens,
        "reuse_ratio": ratio,
        "permutation": permutation.tolist(),
        "video_rescheduled": ratio > 0.0,
        "audio_unchanged": streams[1] is noise.unbind()[1],
    }
    return output, report

