from __future__ import annotations

import inspect
import json
import types
from typing import Any

import torch

import comfy.ldm.minimax.model as minimax_model


ATTENTION_HOOK_ATTACHMENT_KEY = "t8_minimax_h3_attention_hooks_contract"


def _source_has(callable_object, names: tuple[str, ...]) -> bool:
    try:
        source = inspect.getsource(callable_object)
    except (OSError, TypeError):
        return False
    return all(name in source for name in names)


def probe_native_attention_hooks(diffusion) -> dict[str, Any]:
    blocks = list(getattr(diffusion, "blocks", ()))
    attention = getattr(blocks[0], "attn", None) if blocks else None
    attention_forward = getattr(type(attention), "forward", None)
    model_forward = getattr(type(diffusion), "_forward", None)
    attention_hooks = callable(attention_forward) and _source_has(
        attention_forward,
        ("attn1_patch", "attn1_output_patch", "extra_options"),
    )
    block_metadata = callable(model_forward) and _source_has(
        model_forward,
        ("block_index", "block_type", "total_blocks"),
    )
    return {
        "available": bool(attention_hooks and block_metadata),
        "attention_hooks": bool(attention_hooks),
        "block_metadata": bool(block_metadata),
        "main_block_count": len(blocks),
        "policy": "semantic_source_feature_probe_no_version_or_hash_gate",
    }


def _hooked_attention_forward(
    attention,
    x: torch.Tensor,
    rope_freqs,
    transformer_options: dict | None,
    *,
    block_index: int,
    block_type: str,
    total_blocks: int,
):
    options = dict(transformer_options or {})
    options.update(
        {
            "block_index": int(block_index),
            "block_type": str(block_type),
            "total_blocks": int(total_blocks),
        }
    )
    patches = options.get("patches", {})
    sequence = x.shape[0]
    q, k, v = attention.qkv_proj(x).split(
        attention.heads * attention.head_dim, dim=-1
    )
    v = v.view(sequence, attention.heads, attention.head_dim)
    if rope_freqs is not None:
        q = q.view(1, sequence, attention.heads, attention.head_dim)
        k = k.view(1, sequence, attention.heads, attention.head_dim)
        qw = minimax_model.comfy.model_management.cast_to(
            attention.q_norm.weight, device=x.device
        )
        kw = minimax_model.comfy.model_management.cast_to(
            attention.k_norm.weight, device=x.device
        )
        rot = rope_freqs.shape[-3] * 2
        if minimax_model.comfy.model_management.in_training:
            q, k = minimax_model.comfy.quant_ops.ck.rms_rope_split_half(
                q,
                k,
                rope_freqs,
                qw,
                kw,
                epsilon=attention.q_norm.eps,
                rot_dim=rot,
            )
        else:
            minimax_model.comfy.quant_ops.ck.rms_rope_split_half_(
                q,
                k,
                rope_freqs,
                qw,
                kw,
                epsilon=attention.q_norm.eps,
                rot_dim=rot,
            )
        q = q[0]
        k = k[0]
    else:
        q = attention.q_norm(q.view(sequence, attention.heads, attention.head_dim))
        k = attention.k_norm(k.view(sequence, attention.heads, attention.head_dim))
    v = v.clone()

    extra_options = None
    if "attn1_patch" in patches or "attn1_output_patch" in patches:
        extra_options = {
            key: value
            for key, value in options.items()
            if key not in ("patches", "patches_replace")
        }
        extra_options["n_heads"] = attention.heads
        extra_options["dim_head"] = attention.head_dim

    if "attn1_patch" in patches:
        q = q.reshape(1, sequence, -1)
        k = k.reshape(1, sequence, -1)
        v = v.reshape(1, sequence, -1)
        for patch in patches["attn1_patch"]:
            output = patch(q, k, v, extra_options=extra_options)
            if isinstance(output, dict):
                q = output.get("q", q)
                k = output.get("k", k)
                v = output.get("v", v)
            else:
                q, k, v = output
        q = q.view(q.shape[0], q.shape[1], attention.heads, attention.head_dim).transpose(1, 2)
        k = k.view(k.shape[0], k.shape[1], attention.heads, attention.head_dim).transpose(1, 2)
        v = v.view(v.shape[0], v.shape[1], attention.heads, attention.head_dim).transpose(1, 2)
    else:
        q = q.transpose(0, 1).unsqueeze(0)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)

    q = minimax_model.AttentionTensorContainer(q)
    k = minimax_model.AttentionTensorContainer(k)
    v = minimax_model.AttentionTensorContainer(v)
    output = minimax_model.optimized_attention(
        q,
        k,
        v,
        attention.heads,
        mask=None,
        skip_reshape=True,
        transformer_options=options,
    )
    if "attn1_output_patch" in patches:
        for patch in patches["attn1_output_patch"]:
            output = patch(output, extra_options)
    return attention.out_proj(output.squeeze(0))


def _make_attention_forward(attention, block_index, block_type, total_blocks):
    def forward(self, x, rope_freqs=None, transformer_options=None):
        return _hooked_attention_forward(
            self,
            x,
            rope_freqs,
            transformer_options,
            block_index=block_index,
            block_type=block_type,
            total_blocks=total_blocks,
        )

    return types.MethodType(forward, attention)


def build_attention_hook_compatibility(model):
    diffusion = model.get_model_object("diffusion_model")
    if diffusion.__class__.__name__ != "MiniMaxH3Model":
        raise TypeError("MiniMax H3 Attention Hooks requires a native MiniMaxH3Model MODEL.")

    capability = probe_native_attention_hooks(diffusion)
    patched = model.clone()
    patched_paths: list[str] = []
    if not capability["available"]:
        groups = [
            ("blocks", list(getattr(diffusion, "blocks", ())), "double"),
            (
                "token_refiner.blocks",
                list(getattr(getattr(diffusion, "token_refiner", None), "blocks", ())),
                "token_refiner",
            ),
        ]
        existing = set(getattr(patched, "object_patches", {}))
        for prefix, blocks, block_type in groups:
            for index, block in enumerate(blocks):
                attention = getattr(block, "attn", None)
                if attention is None:
                    continue
                path = f"diffusion_model.{prefix}.{index}.attn.forward"
                if path in existing:
                    raise RuntimeError(
                        "MiniMax H3 Attention Hooks will not replace an existing "
                        f"attention forward patch: {path}"
                    )
                patched.add_object_patch(
                    path,
                    _make_attention_forward(
                        attention,
                        index,
                        block_type,
                        len(blocks),
                    ),
                )
                patched_paths.append(path)

    transformer_options = patched.model_options.get("transformer_options", {})
    patches = transformer_options.get("patches", {})
    replacements = transformer_options.get("patches_replace", {})
    report = {
        "schema": "t8_minimax_h3_attention_hooks_v1",
        "status": "native" if capability["available"] else "compatibility_patch_ready",
        "native_core_probe": capability,
        "patched_path_count": len(patched_paths),
        "main_blocks": len(getattr(diffusion, "blocks", ())),
        "token_refiner_blocks": len(
            getattr(getattr(diffusion, "token_refiner", None), "blocks", ())
        ),
        "registered_attn1_patches": len(patches.get("attn1_patch", ())),
        "registered_attn1_output_patches": len(
            patches.get("attn1_output_patch", ())
        ),
        "optimized_attention_override": "optimized_attention_override"
        in transformer_options,
        "dit_block_replacements": len(replacements.get("dit", {})),
        "object_patch_scope": "model_patcher_clone_only",
        "source_pr": "https://github.com/Comfy-Org/ComfyUI/pull/15270",
    }
    if hasattr(patched, "set_attachments"):
        patched.set_attachments(ATTENTION_HOOK_ATTACHMENT_KEY, report)
    return patched, json.dumps(report, ensure_ascii=False, sort_keys=True)
