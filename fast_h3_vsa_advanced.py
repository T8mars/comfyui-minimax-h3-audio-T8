from __future__ import annotations

import functools
import importlib
import inspect
import math
from typing import Any

import torch

import comfy.patcher_extension
import comfy.ldm.minimax.model as minimax_model

from .h3_lora_compat_advanced import FAST_H3_VSA_GATE_ATTACHMENT_KEY


FAST_H3_VSA_ATTACHMENT_KEY = "t8_fast_h3_vsa_runtime_contract_v1"
FAST_H3_VSA_WRAPPER_KEY = "t8_fast_h3_vsa_layout_v1"
FAST_H3_VSA_SPARSITY = 0.90
FAST_H3_VSA_TOPK_RATIO = 1.0 - FAST_H3_VSA_SPARSITY
FAST_H3_VSA_TILE_SHAPE = (4, 4, 4)
FAST_H3_VSA_TILE_ROWS = math.prod(FAST_H3_VSA_TILE_SHAPE)


class _NotPlainT2VALayout(RuntimeError):
    pass


def _sol_attn_function():
    try:
        kitchen = importlib.import_module("comfy_kitchen")
        function = getattr(kitchen, "sol_attn")
        parameters = inspect.signature(function).parameters
    except (ImportError, AttributeError, TypeError, ValueError):
        return None
    required = {"topk_ratio", "tail", "block_len", "coarse_gate"}
    return function if required.issubset(parameters) else None


def probe_comfy_kitchen_vsa() -> dict[str, Any]:
    function = _sol_attn_function()
    try:
        kitchen = importlib.import_module("comfy_kitchen")
        version = str(getattr(kitchen, "__version__", "unknown"))
    except ImportError:
        version = "not_installed"
    return {
        "comfy_kitchen_version": version,
        "sol_attn_vsa_arguments_available": function is not None,
        "external_vsa_executor_available": function is not None,
        "backend": "comfy_kitchen_sol_attn_vsa" if function else None,
        "required_arguments": ["topk_ratio", "tail", "block_len", "coarse_gate"],
        "policy": (
            "VSA is enabled only when Comfy Kitchen exposes its merged VSA API and "
            "the selected H3 MODEL carries one learned to_gate_compress matrix per "
            "main block. Dense, Sage and legacy Sol-Attn are not relabeled as VSA."
        ),
    }


def _gate_modules(model) -> tuple[list[Any], str | None]:
    try:
        diffusion = model.get_model_object("diffusion_model")
    except (AttributeError, KeyError):
        return [], "MODEL does not expose diffusion_model"
    blocks = list(getattr(diffusion, "blocks", ()))
    if not blocks:
        return [], "MODEL is not a native MiniMaxH3Model"
    gates: list[Any] = []
    for index in range(len(blocks)):
        path = f"diffusion_model.blocks.{index}.attn.to_gate_compress"
        try:
            gate = model.get_model_object(path)
        except (AttributeError, KeyError):
            return [], f"missing learned compression gate at block {index}"
        if not callable(gate) or not isinstance(getattr(gate, "weight", None), torch.Tensor):
            return [], f"invalid learned compression gate at block {index}"
        gates.append(gate)
    return gates, None


def _attention_conflict(model) -> str | None:
    options = getattr(model, "model_options", {}).get("transformer_options", {})
    replacements = options.get("patches_replace", {}).get("dit", {})
    if replacements:
        return "an existing DiT block replacement already owns the H3 main blocks"
    if "optimized_attention_override" in options:
        return "an optimized_attention_override already owns attention"
    patches = options.get("patches", {})
    if patches.get("attn1_patch") or patches.get("attn1_output_patch"):
        return "an existing attention patch already owns attention"
    return None


@functools.lru_cache(maxsize=16)
def _geometry_cpu(
    prefix_lengths: tuple[int, ...],
    video_shape: tuple[int, int, int],
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    tile = FAST_H3_VSA_TILE_ROWS
    source_in_tile_order: list[int] = []
    block_lengths: list[int] = []
    source_offset = 0
    for length in prefix_lengths:
        for start in range(0, length, tile):
            count = min(tile, length - start)
            source_in_tile_order.extend(range(source_offset + start, source_offset + start + count))
            block_lengths.append(count)
        source_offset += length
    prefix_blocks = len(block_lengths)

    frames, height, width = video_shape
    tile_t, tile_h, tile_w = FAST_H3_VSA_TILE_SHAPE
    for t0 in range(0, frames, tile_t):
        for h0 in range(0, height, tile_h):
            for w0 in range(0, width, tile_w):
                tile_sources: list[int] = []
                for t in range(t0, min(t0 + tile_t, frames)):
                    for h in range(h0, min(h0 + tile_h, height)):
                        for w in range(w0, min(w0 + tile_w, width)):
                            tile_sources.append(
                                source_offset + (t * height + h) * width + w
                            )
                source_in_tile_order.extend(tile_sources)
                block_lengths.append(len(tile_sources))

    total_source = source_offset + frames * height * width
    if len(source_in_tile_order) != total_source:
        raise RuntimeError("FastH3 VSA geometry did not cover the packed sequence")
    destination_by_source = torch.empty(total_source, dtype=torch.long)
    cursor = 0
    block_index = 0
    for count in block_lengths:
        sources = source_in_tile_order[cursor : cursor + count]
        destination_by_source[torch.tensor(sources, dtype=torch.long)] = (
            block_index * tile + torch.arange(count, dtype=torch.long)
        )
        cursor += count
        block_index += 1
    return (
        destination_by_source,
        torch.tensor(block_lengths, dtype=torch.int32),
        prefix_blocks,
        block_index * tile,
    )


def _plain_t2va_geometry(layout, sequence: int, device: torch.device):
    segments = list(getattr(layout, "segments", ()))
    kinds = [str(item[2]) for item in segments]
    if kinds != ["text", "audio", "video"]:
        raise _NotPlainT2VALayout(
            "FastH3 VSA Preview v1 supports plain T2VA packing only"
        )
    signature = tuple(getattr(layout, "signature", ()))
    if len(signature) != 5:
        raise _NotPlainT2VALayout("H3 packed layout has no standard signature")
    _text, latent_t, latent_h, latent_w, _audio_t = map(int, signature)
    video_shape = (latent_t, latent_h // 2, latent_w // 2)
    prefix_lengths = tuple(int(stop - start) for start, stop, _kind in segments[:-1])
    expected_video = math.prod(video_shape)
    actual_video = int(segments[-1][1] - segments[-1][0])
    if expected_video != actual_video or int(segments[-1][1]) != sequence:
        raise _NotPlainT2VALayout(
            "H3 packed video geometry does not match the FastH3 VSA tile contract"
        )
    destination, lengths, prefix_blocks, padded_rows = _geometry_cpu(
        prefix_lengths, video_shape
    )
    return (
        destination.to(device=device),
        lengths.to(device=device),
        prefix_blocks,
        padded_rows,
    )


def _pad_rows(value: torch.Tensor, destination: torch.Tensor, padded_rows: int):
    output = value.new_zeros((padded_rows, *value.shape[1:]))
    output[destination] = value
    return output


def _fast_h3_vsa_attention(attention, x, rope_freqs, transformer_options, layout):
    sequence = int(x.shape[0])
    destination, block_len, prefix_blocks, padded_rows = _plain_t2va_geometry(
        layout, sequence, x.device
    )
    x_padded = _pad_rows(x, destination, padded_rows)
    rope_padded = None
    if rope_freqs is not None:
        rope_padded = rope_freqs.new_zeros(
            (rope_freqs.shape[0], padded_rows, *rope_freqs.shape[2:])
        )
        rope_padded[:, destination] = rope_freqs

    q, k, v = attention.qkv_proj(x_padded).split(
        attention.heads * attention.head_dim, dim=-1
    )
    v = v.view(padded_rows, attention.heads, attention.head_dim)
    if rope_padded is not None:
        q = q.view(1, padded_rows, attention.heads, attention.head_dim)
        k = k.view(1, padded_rows, attention.heads, attention.head_dim)
        qw = minimax_model.comfy.model_management.cast_to(
            attention.q_norm.weight, device=x.device
        )
        kw = minimax_model.comfy.model_management.cast_to(
            attention.k_norm.weight, device=x.device
        )
        rot = rope_padded.shape[-3] * 2
        if minimax_model.comfy.model_management.in_training:
            q, k = minimax_model.comfy.quant_ops.ck.rms_rope_split_half(
                q, k, rope_padded, qw, kw,
                epsilon=attention.q_norm.eps, rot_dim=rot,
            )
        else:
            minimax_model.comfy.quant_ops.ck.rms_rope_split_half_(
                q, k, rope_padded, qw, kw,
                epsilon=attention.q_norm.eps, rot_dim=rot,
            )
        q, k = q[0], k[0]
    else:
        q = attention.q_norm(q.view(padded_rows, attention.heads, attention.head_dim))
        k = attention.k_norm(k.view(padded_rows, attention.heads, attention.head_dim))
    v = v.clone()
    coarse_gate = attention.to_gate_compress(x_padded).view(
        1, padded_rows, attention.heads, attention.head_dim
    )
    sol_attn = _sol_attn_function()
    if sol_attn is None:
        raise RuntimeError(
            "FastH3 VSA requires a Comfy Kitchen build with topk_ratio, tail, "
            "block_len and coarse_gate support"
        )
    output = sol_attn(
        q.unsqueeze(0),
        k.unsqueeze(0),
        v.unsqueeze(0),
        topk_ratio=FAST_H3_VSA_TOPK_RATIO,
        tail=False,
        block_len=block_len,
        coarse_gate=coarse_gate,
        sink_blocks=[0, prefix_blocks],
        sink_q=[0, prefix_blocks],
    )
    # Comfy Kitchen returns ``(B, T, H, D)`` while H3's output projection
    # consumes the original fused ``(T, H*D)`` representation.
    return attention.out_proj(output[0, destination].flatten(-2))


def _vsa_block(block, args, original):
    layout = args["transformer_options"].get("t8_fast_h3_vsa_layout")
    if layout is None:
        return original(args)
    try:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            block.adaln_proj(args["t_emb"])
        )
        hidden = minimax_model._mod_scale_shift(
            block.norm1(args["img"]), shift_msa, scale_msa, args["mod_segments"]
        )
        image = minimax_model._mod_gate(
            args["img"],
            gate_msa,
            _fast_h3_vsa_attention(
                block.attn,
                hidden,
                args["rope_freqs"],
                args["transformer_options"],
                layout,
            ),
            args["mod_segments"],
        )
        hidden = minimax_model._mod_scale_shift(
            block.norm2(image), shift_mlp, scale_mlp, args["mod_segments"]
        )
        return {
            "img": minimax_model._mod_gate(
                image, gate_mlp, block.mlp(hidden), args["mod_segments"]
            )
        }
    except _NotPlainT2VALayout:
        return original(args)


def _layout_wrapper(executor, *args, **kwargs):
    options = kwargs.get("transformer_options")
    if not isinstance(options, dict) and len(args) >= 4 and isinstance(args[3], dict):
        options = args[3]
    payload = kwargs.get("minimax_payload")
    if isinstance(options, dict):
        layout = payload.get("layout") if isinstance(payload, dict) else None
        if layout is None:
            options.pop("t8_fast_h3_vsa_layout", None)
        else:
            options["t8_fast_h3_vsa_layout"] = layout
    return executor(*args, **kwargs)


def apply_fast_h3_vsa(model):
    capability = probe_comfy_kitchen_vsa()
    if not capability["external_vsa_executor_available"]:
        return model, None, "Comfy Kitchen VSA API is unavailable"
    gates, gate_error = _gate_modules(model)
    if gate_error is not None:
        return model, None, gate_error
    conflict = _attention_conflict(model)
    if conflict is not None:
        return model, None, conflict

    diffusion = model.get_model_object("diffusion_model")
    blocks = list(diffusion.blocks)
    patched = model.clone()
    for index, block in enumerate(blocks):
        def hook(args, original, _block=block):
            return _vsa_block(_block, args, original["original_block"])

        patched.set_model_patch_replace(hook, "dit", "double_block", index)
    patched.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
        FAST_H3_VSA_WRAPPER_KEY,
        _layout_wrapper,
    )
    receipt = {
        "schema": "t8.minimax_h3.fast_h3_vsa.v1",
        "status": "configured",
        "executor": "comfy_kitchen.sol_attn",
        "main_block_count": len(blocks),
        "learned_gate_count": len(gates),
        "sparsity": FAST_H3_VSA_SPARSITY,
        "topk_ratio": FAST_H3_VSA_TOPK_RATIO,
        "tile_shape": list(FAST_H3_VSA_TILE_SHAPE),
        "tile_rows": FAST_H3_VSA_TILE_ROWS,
        "tail": False,
        "prefix_policy": "text_and_audio_key_sinks_and_dense_queries",
        "scope": "plain_t2va_only_runtime_falls_back_dense_for_other_layouts",
        "gate_attachment": getattr(model, "get_attachment", lambda _key: None)(
            FAST_H3_VSA_GATE_ATTACHMENT_KEY
        ),
        "identity_policy": "structural_capability_only_no_filename_size_or_hash_gate",
    }
    if hasattr(patched, "set_attachments"):
        patched.set_attachments(FAST_H3_VSA_ATTACHMENT_KEY, receipt)
    return patched, receipt, None
