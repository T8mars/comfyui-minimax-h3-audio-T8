"""Safe PTD interchange, separate from ordinary LoRA and DCP deserialization.

Independently implemented from the published EPFL VITA PTD format contract at
43ae22c3c94f8a24cc5e82ca82e4e3a5f736de85 (Apache-2.0). See H21_PTD_EXP.md.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re

import torch
from safetensors import safe_open

GRIDS = {
    "h3_4step": (0.999, 0.947, 0.857, 0.667, 0.0),
    "h3_6step": (0.999, 0.968, 0.923, 0.857, 0.750, 0.545, 0.0),
}
BASE_MODEL = "MiniMaxAI/MiniMax-H3"
BASE_REVISION = "42ed227ee7df40d41602854ae760620d6eb651fe"
BUFFERS = {
    "transformer.rope.inv_freq": "rope.inv_freq",
    "transformer.proj_out.orig_weight": "final_layer.video_out.weight",
    "transformer.proj_out.orig_bias": "final_layer.video_out.bias",
    "transformer.audio_proj_out.orig_weight": "final_layer.audio_out.weight",
    "transformer.audio_proj_out.orig_bias": "final_layer.audio_out.bias",
}
PARTS = (
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.to_out.0",
    "ff.net.0.proj",
    "ff.net.2",
)
_FACTOR = re.compile(r"^(transformer\..+)\.lora_([AB])\.(0|[1-9][0-9]*)\.weight$")
_BLOCK = re.compile(
    r"^transformer\.(transformer_blocks|token_refiner\.refiner_blocks)\.(0|[1-9][0-9]*)\.(.+)$"
)


@dataclass(frozen=True)
class Pair:
    module: str
    hop: int
    a_key: str
    b_key: str
    qkv_index: int | None = None
    swap_halves: bool = False
    dense_head: bool = False


def recipe_grid(recipe):
    if type(recipe) is not str or recipe not in GRIDS:
        raise ValueError("PTD recipe must be h3_4step or h3_6step")
    return GRIDS[recipe]


def map_factor(key):
    match = _FACTOR.fullmatch(key)
    if match is None:
        raise ValueError(f"Not a PTD hop A/B factor: {key}")
    source, factor, hop = match.groups()
    heads = {
        "transformer.proj_out.main": "final_layer.video_out",
        "transformer.audio_proj_out.main": "final_layer.audio_out",
    }
    if source in heads:
        return heads[source], int(hop), factor, None, False, True, source
    block = _BLOCK.fullmatch(source)
    if block is None or block[3] not in PARTS:
        raise ValueError(f"Unknown PTD projection mapping: {key}")
    prefix = "blocks" if block[1] == "transformer_blocks" else "token_refiner.blocks"
    qkv = {"attn.to_q": 0, "attn.to_k": 1, "attn.to_v": 2}
    part = block[3]
    suffix = (
        "attn.qkv_proj"
        if part in qkv
        else {
            "attn.to_out.0": "attn.out_proj",
            "ff.net.0.proj": "mlp.fc1",
            "ff.net.2": "mlp.fc2",
        }[part]
    )
    return (
        f"{prefix}.{block[2]}.{suffix}",
        int(hop),
        factor,
        qkv.get(part),
        part == "ff.net.0.proj",
        False,
        source,
    )


def expected_sources(blocks, refiners):
    result = {
        f"transformer.transformer_blocks.{i}.{part}"
        for i in range(blocks)
        for part in PARTS
    }
    result.update(
        f"transformer.token_refiner.refiner_blocks.{i}.{part}"
        for i in range(refiners)
        for part in PARTS
    )
    return result | {"transformer.proj_out.main", "transformer.audio_proj_out.main"}


def validate_tensors(tensors, recipe, *, blocks, refiners):
    """Require complete actual topology, not a model name or fixed layer count."""
    hops = len(recipe_grid(recipe)) - 1
    groups, warnings = {}, []
    for key, value in tensors.items():
        if (
            not isinstance(value, torch.Tensor)
            or value.layout != torch.strided
            or not value.is_floating_point()
            or value.device.type == "meta"
        ):
            raise ValueError(
                f"PTD requires a materialized dense floating tensor: {key}"
            )
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"Non-finite PTD tensor: {key}")
        if key in BUFFERS:
            ndim = 2 if key.endswith("orig_weight") else 1
            if value.ndim != ndim or min(value.shape, default=0) < 1:
                raise ValueError(f"Invalid PTD reference buffer dimensions: {key}")
            continue
        mapped = map_factor(key)
        module, hop, factor, qkv, swap, dense, source = mapped
        if hop >= hops or value.ndim != 2 or min(value.shape) < 1:
            raise ValueError(f"PTD factor has invalid hop or matrix shape: {key}")
        groups.setdefault((source, hop), {})[factor] = (key, module, qkv, swap, dense)
    expected = {
        (source, hop)
        for source in expected_sources(blocks, refiners)
        for hop in range(hops)
    }
    if set(groups) != expected:
        raise ValueError(
            f"Incomplete PTD topology: missing={sorted(expected - set(groups))[:6]}, extra={sorted(set(groups) - expected)[:6]}"
        )
    pairs, shapes = [], {}
    for (source, hop), factors in sorted(groups.items()):
        if set(factors) != {"A", "B"}:
            raise ValueError(f"Incomplete PTD A/B pair: {source}, hop={hop}")
        a_key, module, qkv, swap, dense = factors["A"]
        b_key = factors["B"][0]
        a, b = tensors[a_key], tensors[b_key]
        if a.dtype != b.dtype or a.shape[0] != b.shape[1]:
            raise ValueError(f"PTD A/B rank or dtype mismatch: {source}, hop={hop}")
        if swap and b.shape[0] % 2:
            raise ValueError(f"PTD SwiGLU output rows must be even: {source}")
        current = (tuple(a.shape), tuple(b.shape))
        if shapes.setdefault(source, current) != current:
            warnings.append(
                f"Non-uniform PTD factors between hops at {source}; independent selected factors retained, target dimensions checked at MODEL setup"
            )
        if a.shape[0] != (b.shape[0] if dense else 128):
            warnings.append(
                f"Non-published PTD rank at {source}: {a.shape[0]}; retained as selected"
            )
        pairs.append(Pair(module, hop, a_key, b_key, qkv, swap, dense))
    for source, shape in shapes.items():
        if source.endswith("attn.to_q"):
            for suffix in ("to_k", "to_v"):
                if shapes[source.removesuffix("to_q") + suffix] != shape:
                    warnings.append(
                        f"Non-uniform independent PTD Q/K/V factors at {source}; selected factors retained, target dimensions checked at MODEL setup"
                    )
    head_buffers = {key for key in BUFFERS if ".orig_" in key}
    present = head_buffers & set(tensors)
    if present and present != head_buffers:
        raise ValueError(
            f"Incomplete PTD original head buffers: {sorted(head_buffers - present)}"
        )
    return tuple(pairs), sorted(set(warnings))


def metadata_for(recipe):
    return {
        "ptd_format": "comfyui-native-h3",
        "ptd_schema": "1",
        "ptd_recipe": recipe,
        "ptd_hops": str(len(recipe_grid(recipe)) - 1),
        "ptd_mapping": "diffusers-h3-to-native-h3-v1",
        "ptd_lora_scale": "1.0",
        "ptd_base_model": BASE_MODEL,
        "ptd_base_revision": BASE_REVISION,
    }


def validate_metadata(metadata, recipe=None):
    chosen = metadata.get("ptd_recipe")
    expected = metadata_for(chosen)
    if recipe is not None and chosen != recipe:
        raise ValueError(f"PTD requested recipe {recipe} differs from stored {chosen}")
    for key in (
        "ptd_format",
        "ptd_schema",
        "ptd_hops",
        "ptd_mapping",
        "ptd_lora_scale",
    ):
        if metadata.get(key) != expected[key]:
            raise ValueError(
                f"Invalid PTD interchange field {key}: {metadata.get(key)!r}, expected {expected[key]!r}"
            )
    warnings = [
        f"PTD provenance {key} differs from published reference; selected file retained"
        for key in ("ptd_base_model", "ptd_base_revision")
        if metadata.get(key) != expected[key]
    ]
    return chosen, warnings


def load_ptd_file(path, *, recipe=None):
    """No torch.load, DCP reader, pickle, conversion, download or fallback."""
    path = Path(path)
    if path.suffix.lower() not in (".safetensors", ".sft"):
        raise ValueError(
            "PTD runtime accepts only trusted converted safetensors; raw DCP/pickle is not executed"
        )
    with path.open("rb") as stream:
        before = hashlib.file_digest(stream, "sha256").hexdigest()
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        chosen, warnings = validate_metadata(metadata, recipe)
        tensors = {key: handle.get_tensor(key) for key in handle.keys()}
    with path.open("rb") as stream:
        after = hashlib.file_digest(stream, "sha256").hexdigest()
    if before != after:
        raise ValueError("PTD file changed while loading")
    return (
        tensors,
        metadata,
        {
            "path": str(path.resolve()),
            "sha256": after,
            "recipe": chosen,
            "warnings": warnings,
            "pickle_executed": False,
        },
    )
