# SPDX-FileCopyrightText: Copyright (c) 2026 EPFL VITA Lab
# SPDX-License-Identifier: Apache-2.0
"""Lossless key/layout conversion of the original, unpruned H3 transformer."""

from __future__ import annotations

import math
import re


def base_key(key: str) -> str:
    prefixes = {
        "proj_in.": "video_patch_proj.", "audio_proj_in.": "audio_patch_proj.",
        "context_embedder.": "condition_proj.",
        "time_embedder.linear_1.": "time_embedder.proj_in.",
        "time_embedder.linear_2.": "time_embedder.proj_out.",
        "norm_out.norm.": "final_layer.norm.",
        "norm_out.linear.": "final_layer.adaln_proj.linear.",
        "proj_out.": "final_layer.video_out.", "audio_proj_out.": "final_layer.audio_out.",
        "token_refiner.final_norm.": "token_refiner.final_norm.",
    }
    for old, new in prefixes.items():
        if key.startswith(old):
            if key[len(old):] not in ("weight", "bias"):
                raise ValueError(f"Unsupported base-model tensor: {key}")
            return new + key[len(old):]
    if not re.match(r"^(?:transformer_blocks|token_refiner\.refiner_blocks)\.(0|[1-9][0-9]*)\.", key):
        raise ValueError(f"Unsupported base-model tensor: {key}")
    key = re.sub(r"^transformer_blocks\.(\d+)\.", r"blocks.\1.", key)
    key = re.sub(r"^token_refiner\.refiner_blocks\.(\d+)\.", r"token_refiner.blocks.\1.", key)
    if not re.match(r"^(?:blocks|token_refiner\.blocks)\.\d+\.", key):
        raise ValueError(f"Unsupported base-model tensor: {key}")
    for old, new in {
        ".attn.to_out.0.": ".attn.out_proj.",
        ".attn.norm_q.": ".attn.q_norm.", ".attn.norm_k.": ".attn.k_norm.",
        ".ff.net.0.proj.": ".mlp.fc1.", ".ff.net.2.": ".mlp.fc2.",
    }.items():
        key = key.replace(old, new)
    if not re.search(r"\.(?:norm[12]|attn\.(?:to_[qkv]|out_proj|q_norm|k_norm)|mlp\.fc[12]|adaln_proj\.linear)\.(?:weight|bias)$", key):
        raise ValueError(f"Unsupported base-model tensor: {key}")
    return key


def _expected_source_shapes(config):
    """Original H3 parameter contract, independent of either runtime package."""
    defaults = {"hidden_size": 5376, "num_layers": 50, "num_refiner_layers": 2,
                "num_attention_heads": 56, "attention_head_dim": 128, "ffn_dim": 14336,
                "in_channels": 24, "audio_in_channels": 32, "text_dim": 5120, "freq_dim": 256,
                "time_embed_hidden_dim": 5376, "time_embed_dim": 2688, "rope_freq_dim": 16}
    dims = {key: int(config.get(key, value)) for key, value in defaults.items()}
    for key, value in dims.items():
        if value < (0 if key in ("num_layers", "num_refiner_layers") else 1):
            raise ValueError(f"Invalid H3 configuration {key}={value}")
    patch = tuple(int(v) for v in config.get("patch_size", (1, 2, 2)))
    if len(patch) != 3 or min(patch) < 1:
        raise ValueError(f"Invalid patch_size: {patch}")
    if 6 * dims["rope_freq_dim"] > dims["attention_head_dim"]:
        raise ValueError("Rotary embedding does not fit in the attention head")
    theta = float(config.get("rope_theta", 10000.0))
    if not math.isfinite(theta) or theta <= 0:
        raise ValueError("rope_theta must be finite and positive")
    hidden, inner = dims["hidden_size"], dims["num_attention_heads"] * dims["attention_head_dim"]
    ffn, time_dim = dims["ffn_dim"], dims["time_embed_dim"]
    video, audio = dims["in_channels"] * math.prod(patch), dims["audio_in_channels"]
    shapes = {}

    def linear(name, out_features, in_features, bias=True):
        shapes[name + ".weight"] = (out_features, in_features)
        if bias:
            shapes[name + ".bias"] = (out_features,)

    linear("proj_in", hidden, video)
    linear("audio_proj_in", hidden, audio)
    linear("context_embedder", hidden, dims["text_dim"])
    linear("time_embedder.linear_1", dims["time_embed_hidden_dim"], dims["freq_dim"])
    linear("time_embedder.linear_2", time_dim, dims["time_embed_hidden_dim"])
    shapes["token_refiner.final_norm.weight"] = (hidden,)
    shapes["norm_out.norm.weight"] = (hidden,)
    linear("norm_out.linear", 2 * hidden, time_dim)
    linear("proj_out", video, hidden)
    linear("audio_proj_out", audio, hidden)
    for family, count in (("transformer_blocks", dims["num_layers"]),
                          ("token_refiner.refiner_blocks", dims["num_refiner_layers"])):
        for i in range(count):
            prefix = f"{family}.{i}"
            for norm in ("norm1", "norm2"):
                shapes[f"{prefix}.{norm}.weight"] = (hidden,)
            for part in "qkv":
                linear(f"{prefix}.attn.to_{part}", inner, hidden, bias=False)
            for norm in ("norm_q", "norm_k"):
                shapes[f"{prefix}.attn.{norm}.weight"] = (dims["attention_head_dim"],)
            linear(f"{prefix}.attn.to_out.0", hidden, inner, bias=False)
            linear(f"{prefix}.ff.net.0.proj", 2 * ffn, hidden, bias=False)
            linear(f"{prefix}.ff.net.2", hidden, ffn, bias=False)
            if family == "transformer_blocks":
                linear(f"{prefix}.adaln_proj.linear", 18 * hidden, time_dim)
    return shapes


def convert_base_state_dict(state, config):
    """Rename tensors, pack Q/K/V and SwiGLU, and reconstruct the RoPE buffer.

    No quantization or pruning is performed; tensor dtypes are preserved.
    The returned dictionary reuses source storage except for Q/K/V packing and
    SwiGLU's [up, gate] -> [gate, up] row permutation.
    """
    import torch

    expected = _expected_source_shapes(config)
    if set(state) != set(expected):
        missing, unknown = sorted(set(expected) - set(state)), sorted(set(state) - set(expected))
        raise ValueError(f"Incomplete or unsupported H3 state: missing {missing[:8]}, unknown {unknown[:8]}")
    output, qkv = {}, {}
    for key, tensor in state.items():
        if not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided or not tensor.is_floating_point():
            raise ValueError(f"Expected a dense floating-point base tensor: {key}")
        if tuple(tensor.shape) != expected[key]:
            raise ValueError(f"Wrong base tensor shape: {key}: {tuple(tensor.shape)}; expected {expected[key]}")
        dest = base_key(key)
        match = re.match(r"^(.*\.attn)\.to_([qkv])\.(weight|bias)$", dest)
        if match:
            group = match[1] + ".qkv_proj." + match[3]
            qkv.setdefault(group, {})[match[2]] = tensor
        else:
            if dest in output:
                raise ValueError(f"Duplicate mapped tensor: {dest}")
            if ".mlp.fc1." in dest:
                up, gate = tensor.chunk(2, dim=0)
                tensor = torch.cat((gate, up), dim=0)
            output[dest] = tensor
    for dest, pieces in qkv.items():
        if set(pieces) != {"q", "k", "v"}:
            raise ValueError(f"Incomplete Q/K/V group: {dest}")
        if len({(tensor.dtype, tensor.device) for tensor in pieces.values()}) != 1:
            raise ValueError(f"Q/K/V dtypes or devices differ; refusing implicit conversion: {dest}")
        output[dest] = torch.cat([pieces[k] for k in "qkv"], dim=0)
    freq_dim = int(config.get("rope_freq_dim", 16))
    output["rope.inv_freq"] = 1.0 / (float(config.get("rope_theta", 10000.0)) **
        (torch.arange(0, 2 * freq_dim, 2, dtype=torch.float32) / (2 * freq_dim)))
    return output
