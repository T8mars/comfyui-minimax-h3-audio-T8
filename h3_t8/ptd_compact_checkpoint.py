"""Stacked-hop PTD interchange, not an ordinary LoRA or EPFL recipe.

Independent implementation of the schema-1 storage and Taobao stacked-hop
contract described by Nynxz/ComfyUI-PTD-Compact a50419a7 (pdd.py), and the
literal BF16 storage decoder in Jalen-Brunson/ComfyUI-MiniMax-H3-PTD 4f283cc8
(ptd_core.py). Loading does not execute a producer, pickle or conversion tool.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from collections.abc import Mapping

import torch
from safetensors import safe_open

from .ptd_checkpoint import PARTS, map_factor

HEADS = {"proj_out": "final_layer.video_out", "audio_proj_out": "final_layer.audio_out"}
_MODULE = re.compile(
    r"^(transformer_blocks|token_refiner\.refiner_blocks)\.(0|[1-9][0-9]*)\.(.+)$"
)


@dataclass(frozen=True)
class StackedPair:
    source: str
    target: str
    a: torch.Tensor
    b: torch.Tensor
    qkv_index: int | None
    swap_halves: bool


@dataclass(frozen=True)
class CompactCheckpoint:
    pairs: tuple[StackedPair, ...]
    heads: dict[str, tuple[torch.Tensor, torch.Tensor | None]]
    hops: int
    steps: int
    rank: int
    alpha: float
    blocks: int
    refiners: int
    storage: str
    metadata: dict
    warnings: tuple[str, ...]


def recognizes_stacked_ptd(metadata):
    return isinstance(metadata, Mapping) and "ptd_num_hops" in metadata


def is_compact_file(path):
    """Header-only dispatch for the existing PTD entry; no broad error fallback."""
    path = Path(path)
    if path.suffix.lower() not in (".safetensors", ".sft"):
        return False
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return recognizes_stacked_ptd(handle.metadata() or {})


def _positive_int(metadata, name, default=None):
    value = metadata.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError(f"Stacked PTD {name} must be a positive integer")
    if isinstance(value, str) and not re.fullmatch(r"[0-9]+", value):
        raise ValueError(f"Stacked PTD {name} must be a positive integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"Stacked PTD {name} must be positive")
    return result


def _tensor(value, key, *, floating=True):
    if (not isinstance(value, torch.Tensor) or value.layout != torch.strided
            or value.device.type == "meta" or min(value.shape, default=0) < 1):
        raise ValueError(f"Stacked PTD needs a materialized dense tensor: {key}")
    if floating and not value.is_floating_point():
        raise ValueError(f"Stacked PTD needs floating values: {key}")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"Non-finite stacked PTD tensor: {key}")
    return value.detach().cpu()


def decode_rows(value, scale, key):
    """Literal schema1 storage decode; signed/zero finite scales are retained."""
    value = _tensor(value, key, floating=False)
    scale = _tensor(scale, key + "_scale")
    if value.dtype != torch.int8 or tuple(scale.shape) != tuple(value.shape[:-1]):
        raise ValueError(f"Stacked PTD row scale/INT8 dimensions disagree: {key}")
    result = (value.float() * scale.float().unsqueeze(-1)).to(torch.bfloat16)
    if not bool(torch.isfinite(result).all()):
        raise ValueError(f"Stacked PTD storage decode overflow: {key}")
    return result


def parse_compact_checkpoint(tensors, metadata):
    """Validate every key, preserving caller dictionaries/tensors unchanged."""
    if not isinstance(tensors, Mapping) or not recognizes_stacked_ptd(metadata):
        raise ValueError("Stacked PTD requires ptd_num_hops metadata, not an ordinary LoRA")
    hops = _positive_int(metadata, "ptd_num_hops")
    steps = _positive_int(metadata, "pdd_num_steps", 32)
    rank = _positive_int(metadata, "lora_rank", 128)
    if steps % hops:
        raise ValueError("pdd_num_steps must be divisible by ptd_num_hops")
    try:
        alpha = float(metadata.get("lora_alpha", rank))
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Stacked PTD lora_alpha must be finite") from error
    if isinstance(metadata.get("lora_alpha"), bool) or not math.isfinite(alpha):
        raise ValueError("Stacked PTD lora_alpha must be finite")
    compact = metadata.get("ptd_compact")
    if compact is not None:
        try:
            schema = json.loads(compact)
        except (TypeError, ValueError) as error:
            raise ValueError("Invalid ptd_compact storage metadata") from error
        if (not isinstance(schema, dict) or type(schema.get("schema")) is not int
                or schema["schema"] != 1 or schema.get("quant") != "int8 per-row"):
            raise ValueError("Unsupported ptd_compact storage; expected schema1 int8 per-row")

    normalized = {}
    for key, value in tensors.items():
        if not isinstance(key, str):
            raise ValueError("Stacked PTD tensor keys must be strings")
        name = key.replace("._checkpoint_wrapped_module", "")
        if name in normalized:
            raise ValueError(f"Stacked PTD normalized source collision: {name}")
        normalized[name] = value
    consumed, warnings, heads = set(), [], {}
    for prefix, target in HEADS.items():
        key = prefix + ".weight"
        if key not in normalized:
            raise ValueError(f"Missing stacked PTD absolute head bank: {key}")
        weight = _tensor(normalized[key], key)
        if weight.ndim != 3 or weight.shape[0] != steps:
            raise ValueError(f"Stacked PTD head bank must be [pdd_num_steps,out,in]: {key}")
        consumed.add(key)
        bias_key = prefix + ".bias"
        bias = None
        if bias_key in normalized:
            bias = _tensor(normalized[bias_key], bias_key)
            if tuple(bias.shape) != tuple(weight.shape[:2]):
                raise ValueError(f"Stacked PTD head bias dimensions disagree: {bias_key}")
            consumed.add(bias_key)
        heads[target] = (weight.clone(), None if bias is None else bias.clone())

    modules = set()
    for key in normalized:
        if key.endswith((".lora_down", ".lora_up", ".lora_down_scale", ".lora_up_scale")):
            modules.add(key.rsplit(".lora_", 1)[0])
    pairs, indices, dequantized = [], {"transformer_blocks": set(), "token_refiner.refiner_blocks": set()}, 0
    for source in sorted(modules):
        match = _MODULE.fullmatch(source)
        if match is None or match[3] not in PARTS:
            raise ValueError(f"Unknown stacked PTD target projection: {source}")
        indices[match[1]].add(int(match[2]))
        decoded = []
        for suffix in ("lora_down", "lora_up"):
            key, scale_key = source + "." + suffix, source + "." + suffix + "_scale"
            if key not in normalized:
                raise ValueError(f"Incomplete stacked PTD pair: {key}")
            value = _tensor(normalized[key], key, floating=False)
            if value.ndim != 3 or value.shape[0] != hops:
                raise ValueError(f"Stacked PTD factor requires [hops,rows,columns]: {key}")
            consumed.add(key)
            if value.dtype == torch.int8:
                if compact is None or scale_key not in normalized:
                    raise ValueError(f"INT8 stacked PTD needs schema1 and row scale: {key}")
                value = decode_rows(value, normalized[scale_key], key)
                consumed.add(scale_key)
                dequantized += 1
            else:
                if not value.is_floating_point() or scale_key in normalized:
                    raise ValueError(f"Unexpected stacked PTD factor dtype/scale pair: {key}")
                value = value.clone()
            decoded.append(value)
        a, b = decoded
        if a.shape[1] != b.shape[2]:
            raise ValueError(f"Stacked PTD A/B rank dimensions disagree: {source}")
        if a.shape[1] != rank:
            warnings.append(f"{source}: actual rank {a.shape[1]} differs from metadata {rank}; literal alpha/rank retained")
        target, _, _, qkv, swap, _, _ = map_factor("transformer." + source + ".lora_A.0.weight")
        if swap and b.shape[1] % 2:
            raise ValueError(f"Stacked PTD SwiGLU rows must be even: {source}")
        pairs.append(StackedPair(source, target, a, b, qkv, swap))
    for prefix, found in indices.items():
        if found != set(range(len(found))):
            raise ValueError(f"Non-contiguous stacked PTD {prefix} indices")
        for index in found:
            expected = {f"{prefix}.{index}.{part}" for part in PARTS}
            if not expected <= modules:
                raise ValueError(f"Incomplete stacked PTD projection topology: {prefix}.{index}")
    if not indices["transformer_blocks"]:
        raise ValueError("Stacked PTD contains no trunk experts")
    training_only = {key for key in normalized if key.endswith(("curv_weight", "curv_bias"))}
    for key in training_only:
        _tensor(normalized[key], key)
    unknown = set(normalized) - consumed - training_only
    if unknown:
        raise ValueError(f"Unconsumed stacked PTD keys: {sorted(unknown)[:6]}")
    if training_only:
        warnings.append("Training-only curvature keys retained in source, not used for inference")
    return CompactCheckpoint(tuple(pairs), heads, hops, steps, rank, alpha,
        len(indices["transformer_blocks"]), len(indices["token_refiner.refiner_blocks"]),
        "schema1_int8_dequantized_to_bf16" if dequantized else "stacked_floating",
        dict(metadata), tuple(sorted(set(warnings))))


def load_compact_file(path):
    path = Path(path)
    if path.suffix.lower() not in (".safetensors", ".sft"):
        raise ValueError("Stacked PTD accepts safetensors, not DCP/pickle")
    with path.open("rb") as stream:
        before = hashlib.file_digest(stream, "sha256").hexdigest()
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        tensors = {key: handle.get_tensor(key) for key in handle.keys()}
    checkpoint = parse_compact_checkpoint(tensors, metadata)
    with path.open("rb") as stream:
        after = hashlib.file_digest(stream, "sha256").hexdigest()
    if before != after:
        raise ValueError("Stacked PTD file changed while loading")
    return checkpoint, {"path": str(path.resolve()), "sha256": after,
        "storage": checkpoint.storage, "pickle_executed": False,
        "file_read_not_forward_or_quality_evidence": True}
