from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import comfy.lora
import comfy.lora_convert
import comfy.model_base
import comfy.utils
import torch
from torch import nn


LOG = logging.getLogger(__name__)
SCHEMA = "t8.minimax_h3.lora_compat.v1"
FAST_H3_VSA_GATE_ATTACHMENT_KEY = "t8_fast_h3_vsa_gate_contract_v1"


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)


def add_minimax_h3_direct_lora_keys(model, key_map: dict | None = None) -> dict:
    """Add the direct DiffSynth/ModelScope H3 aliases proposed in ComfyUI #15662.

    The alias is purely structural: ``diffusion_model.blocks.0...weight`` maps
    from ``blocks.0...``.  No filename, file size, hash or metadata identity is
    required.  Other model families retain the normal ComfyUI key map.
    """

    result = {} if key_map is None else dict(key_map)
    if not isinstance(model, comfy.model_base.MiniMaxH3):
        return result
    for key in model.state_dict().keys():
        if key.startswith("diffusion_model.") and key.endswith(".weight"):
            result[key[len("diffusion_model.") : -len(".weight")]] = key
    return result


def build_minimax_h3_lora_key_map(model) -> tuple[dict, int]:
    base = comfy.lora.model_lora_keys_unet(model, {})
    mapped = add_minimax_h3_direct_lora_keys(model, base)
    return mapped, len(mapped) - len(base)


def _format_hint(keys: list[str]) -> str:
    if any(key.startswith("transformer_blocks.0.attn.to_q.lora_A") for key in keys):
        return "fastvideo_h3_diffusers_split_qkv"
    if any(".lora_A.default.weight" in key for key in keys):
        return "diffsynth_or_modelscope_direct"
    if any(key.startswith("diffusion_model.") for key in keys):
        return "comfyui_direct"
    if any(key.startswith("lora_unet_") for key in keys):
        return "kohya_or_comfyui_lora_unet"
    return "unknown_or_mixed"


def _fastvideo_h3_layout(state: dict[str, torch.Tensor]) -> bool:
    """Detect FastVideo's published H3 adapter layout structurally.

    This is deliberately not a filename, byte-size, hash, or model identity
    check.  It only selects the converter when the tensor namespace itself is
    the FastVideo/Diffusers H3 split-QKV layout.
    """

    required = {
        "transformer_blocks.0.attn.to_q.lora_A.weight",
        "transformer_blocks.0.attn.to_q.lora_B.weight",
        "transformer_blocks.0.attn.to_k.lora_A.weight",
        "transformer_blocks.0.attn.to_v.lora_A.weight",
    }
    return required.issubset(state)


def _extract_fastvideo_vsa_gate_weights(
    state: dict[str, torch.Tensor],
) -> tuple[dict[int, torch.Tensor], dict[str, torch.Tensor]]:
    """Separate FastVideo's whole-parameter VSA gates from LoRA/diff payloads."""

    suffix = ".attn.to_gate_compress.set_weight"
    gates: dict[int, torch.Tensor] = {}
    remaining: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if not (key.startswith("transformer_blocks.") and key.endswith(suffix)):
            remaining[key] = value
            continue
        block_text = key[len("transformer_blocks.") : -len(suffix)]
        try:
            block_index = int(block_text)
        except ValueError as exc:
            raise ValueError(f"invalid FastVideo H3 VSA gate key: {key}") from exc
        if value.ndim != 2:
            raise ValueError(
                f"FastVideo H3 VSA gate must be a matrix, got {key} {tuple(value.shape)}"
            )
        if block_index in gates:
            raise ValueError(f"duplicate FastVideo H3 VSA gate for block {block_index}")
        gates[block_index] = value
    return gates, remaining


def _attach_fastvideo_vsa_gates(model, gates: dict[int, torch.Tensor], strength: float):
    if not gates:
        return model, {
            "present": False,
            "attached_gate_count": 0,
            "requires_vsa_attention": False,
        }

    diffusion = model.get_model_object("diffusion_model")
    blocks = list(getattr(diffusion, "blocks", ()))
    expected = set(range(len(blocks)))
    actual = set(gates)
    if not blocks or actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            "FastVideo H3 VSA adapter must provide exactly one compression gate "
            f"for every H3 block; blocks={len(blocks)}, gates={len(gates)}, "
            f"missing={missing[:8]}, extra={extra[:8]}"
        )

    patched = model.clone()
    existing_patches = set(getattr(patched, "object_patches", {}))
    shapes: set[tuple[int, int]] = set()
    for index, block in enumerate(blocks):
        weight = gates[index]
        out_features, in_features = map(int, weight.shape)
        hidden = int(getattr(block.attn.qkv_proj, "in_features", in_features))
        inner = int(getattr(block.attn, "heads", 0)) * int(
            getattr(block.attn, "head_dim", 0)
        )
        if in_features != hidden or out_features != inner:
            raise ValueError(
                "FastVideo H3 VSA gate shape does not match the selected base: "
                f"block={index}, gate={tuple(weight.shape)}, expected=({inner}, {hidden})"
            )
        path = f"diffusion_model.blocks.{index}.attn.to_gate_compress"
        if path in existing_patches:
            raise ValueError(f"a model object patch already owns {path}")
        gate = nn.Linear(
            in_features,
            out_features,
            bias=False,
            device=weight.device,
            dtype=weight.dtype,
        )
        gate.weight = nn.Parameter(
            (weight * float(strength)).contiguous(), requires_grad=False
        )
        gate.eval()
        patched.add_object_patch(path, gate)
        shapes.add((out_features, in_features))

    receipt = {
        "present": True,
        "attached_gate_count": len(gates),
        "requires_vsa_attention": True,
        "application": "whole_parameter_replacement_scaled_by_strength",
        "gate_shapes": [list(shape) for shape in sorted(shapes)],
        "identity_policy": "structural_payload_only_no_filename_size_or_hash_gate",
    }
    if hasattr(patched, "set_attachments"):
        patched.set_attachments(FAST_H3_VSA_GATE_ATTACHMENT_KEY, receipt)
    return patched, receipt


def _fastvideo_scope_target(scope: str) -> str:
    if scope.startswith("transformer_blocks."):
        return "blocks." + scope[len("transformer_blocks.") :]
    if scope.startswith("token_refiner.refiner_blocks."):
        return "token_refiner.blocks." + scope[len("token_refiner.refiner_blocks.") :]
    raise ValueError(f"unsupported FastVideo H3 block scope: {scope}")


def _fastvideo_direct_target(module: str) -> str:
    replacements = {
        "audio_proj_in": "audio_patch_proj",
        "audio_proj_out": "final_layer.audio_out",
        "context_embedder": "condition_proj",
        "norm_out.linear": "final_layer.adaln_proj.linear",
        "norm_out.norm": "final_layer.norm",
        "proj_in": "video_patch_proj",
        "proj_out": "final_layer.video_out",
        "time_embedder.linear_1": "time_embedder.proj_in",
        "time_embedder.linear_2": "time_embedder.proj_out",
    }
    if module in replacements:
        return replacements[module]
    if module.startswith("transformer_blocks."):
        scope, rest = module.split(".", 2)[1:]
        return f"blocks.{scope}.{rest}"
    raise ValueError(f"unsupported FastVideo H3 dense module: {module}")


def convert_fastvideo_h3_adapter(
    state: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, int | str]]:
    """Convert a FastVideo H3 adapter to ComfyUI's fused H3 module layout.

    FastVideo publishes Diffusers-style split Q/K/V factors while ComfyUI H3
    owns one fused ``qkv_proj``.  The block-diagonal construction below is an
    exact algebraic representation of the three original low-rank updates.
    Dense ``.diff``/``.diff_b`` payloads are renamed onto ComfyUI's H3 module
    names and remain additive patches.
    """

    if not _fastvideo_h3_layout(state):
        return dict(state), {
            "conversion": "none",
            "fused_qkv_groups": 0,
            "direct_lora_modules": 0,
            "dense_delta_tensors": 0,
            "swiglu_half_swaps": 0,
        }

    output: dict[str, torch.Tensor] = {}
    consumed: set[str] = set()
    qkv_scopes = sorted(
        {
            key[: -len(".attn.to_q.lora_A.weight")]
            for key in state
            if key.endswith(".attn.to_q.lora_A.weight")
            and key.startswith(("transformer_blocks.", "token_refiner.refiner_blocks."))
        }
    )
    fused_count = 0
    for scope in qkv_scopes:
        factors: list[tuple[torch.Tensor, torch.Tensor]] = []
        keys: list[str] = []
        for projection in ("q", "k", "v"):
            a_key = f"{scope}.attn.to_{projection}.lora_A.weight"
            b_key = f"{scope}.attn.to_{projection}.lora_B.weight"
            if a_key not in state or b_key not in state:
                raise ValueError(f"incomplete FastVideo H3 QKV group under {scope}")
            a = state[a_key]
            b = state[b_key]
            if a.ndim != 2 or b.ndim != 2 or b.shape[1] != a.shape[0]:
                raise ValueError(f"invalid FastVideo H3 LoRA pair {a_key}/{b_key}")
            factors.append((a, b))
            keys.extend((a_key, b_key))
        input_widths = {int(a.shape[1]) for a, _ in factors}
        if len(input_widths) != 1:
            raise ValueError(f"FastVideo H3 Q/K/V input widths differ under {scope}")
        ranks = [int(a.shape[0]) for a, _ in factors]
        output_widths = [int(b.shape[0]) for _, b in factors]
        fused_a = torch.cat([a for a, _ in factors], dim=0).contiguous()
        fused_b = torch.zeros(
            (sum(output_widths), sum(ranks)),
            dtype=factors[0][1].dtype,
            device=factors[0][1].device,
        )
        row = 0
        col = 0
        for (_, b), out_width, rank in zip(factors, output_widths, ranks):
            fused_b[row : row + out_width, col : col + rank] = b
            row += out_width
            col += rank
        target = f"{_fastvideo_scope_target(scope)}.attn.qkv_proj"
        output[f"{target}.lora_A.weight"] = fused_a
        output[f"{target}.lora_B.weight"] = fused_b.contiguous()
        consumed.update(keys)
        fused_count += 1

    direct_suffixes = {
        ".attn.to_out.0": ".attn.out_proj",
        ".ff.net.0.proj": ".mlp.fc1",
        ".ff.net.2": ".mlp.fc2",
        ".adaln_proj.linear": ".adaln_proj.linear",
    }
    direct_count = 0
    swiglu_swaps = 0
    for key in sorted(state):
        if key in consumed or not key.endswith(".lora_A.weight"):
            continue
        b_key = key[: -len(".lora_A.weight")] + ".lora_B.weight"
        if b_key not in state:
            raise ValueError(f"FastVideo H3 LoRA A tensor has no B pair: {key}")
        module = key[: -len(".lora_A.weight")]
        if module == "norm_out.linear":
            matched = None
            target = "final_layer.adaln_proj.linear"
        else:
            matched = next((item for item in direct_suffixes if module.endswith(item)), None)
            if matched is None:
                raise ValueError(f"unsupported FastVideo H3 LoRA module: {module}")
            scope = module[: -len(matched)]
            target = _fastvideo_scope_target(scope) + direct_suffixes[matched]
        a = state[key]
        b = state[b_key]
        if a.ndim != 2 or b.ndim != 2 or b.shape[1] != a.shape[0]:
            raise ValueError(f"invalid FastVideo H3 LoRA pair {key}/{b_key}")
        if matched == ".ff.net.0.proj":
            if b.shape[0] % 2:
                raise ValueError(f"FastVideo H3 SwiGLU B rows must be even: {b_key}")
            half = b.shape[0] // 2
            b = torch.cat((b[half:], b[:half]), dim=0).contiguous()
            swiglu_swaps += 1
        output[f"{target}.lora_A.weight"] = a
        output[f"{target}.lora_B.weight"] = b
        consumed.update((key, b_key))
        direct_count += 1

    dense_count = 0
    for key in sorted(state):
        if key in consumed:
            continue
        suffix = next((value for value in (".diff_b", ".diff") if key.endswith(value)), None)
        if suffix is None:
            raise ValueError(f"unsupported FastVideo H3 adapter tensor: {key}")
        module = key[: -len(suffix)]
        target = _fastvideo_direct_target(module)
        output[f"{target}{suffix}"] = state[key]
        consumed.add(key)
        dense_count += 1

    if consumed != set(state):
        raise ValueError("FastVideo H3 conversion did not consume every source tensor")
    return output, {
        "conversion": "fastvideo_h3_diffusers_to_comfyui_fused",
        "fused_qkv_groups": fused_count,
        "direct_lora_modules": direct_count,
        "dense_delta_tensors": dense_count,
        "swiglu_half_swaps": swiglu_swaps,
    }


def load_minimax_h3_lora_model(
    model,
    lora_path: str | Path,
    strength_model: float = 1.0,
) -> tuple[Any, str]:
    path = Path(lora_path)
    strength = float(strength_model)
    if not strength == strength or not -100.0 <= strength <= 100.0:
        raise ValueError("strength_model must be finite and within [-100, 100]")

    raw, metadata = comfy.utils.load_torch_file(
        str(path), safe_load=True, return_metadata=True
    )
    vsa_gates, adapter_state = _extract_fastvideo_vsa_gate_weights(raw)
    converted, conversion = convert_fastvideo_h3_adapter(adapter_state)
    converted = comfy.lora_convert.convert_lora(converted)
    source_keys = sorted(str(key) for key in raw.keys())
    key_map, direct_alias_count = build_minimax_h3_lora_key_map(model.model)
    patches = comfy.lora.load_lora(converted, key_map, log_missing=False)

    patched = model.clone()
    applied = set(patched.add_patches(patches, strength))
    patched, vsa_gate_receipt = _attach_fastvideo_vsa_gates(
        patched, vsa_gates, strength
    )
    if metadata and hasattr(patched, "set_attachments"):
        patched.set_attachments("t8_h3_lora_metadata", dict(metadata))

    patch_targets = sorted(str(key) for key in patches)
    applied_targets = sorted(str(key) for key in applied)
    missed_targets = sorted(set(patch_targets) - applied)
    gate_count = int(vsa_gate_receipt.get("attached_gate_count", 0))
    status = "applied" if applied_targets or gate_count else "no_compatible_patches"
    if not applied_targets and not gate_count:
        LOG.warning(
            "MiniMax H3 LoRA compatibility loader found no applicable patches in %s; "
            "the MODEL is returned unchanged.",
            path.name,
        )

    report = {
        "schema": SCHEMA,
        "status": status,
        "file": {
            "name": path.name,
            "size_bytes": path.stat().st_size,
            "identity_policy": "display_only_not_a_load_gate_no_hash_scan",
        },
        "format_hint": _format_hint(source_keys),
        "selected_strength": strength,
        "input_tensor_count": len(raw),
        "converted_tensor_count": len(converted),
        "structural_conversion": conversion,
        "key_map_count": len(key_map),
        "added_h3_direct_alias_count": direct_alias_count,
        "patch_target_count": len(patch_targets),
        "applied_patch_count": len(applied_targets),
        "missed_patch_target_count": len(missed_targets),
        "applied_patch_targets_preview": applied_targets[:16],
        "missed_patch_targets_preview": missed_targets[:16],
        "vsa_compression_gates": vsa_gate_receipt,
        "model_class": model.model.__class__.__name__,
        "contract": (
            "Uses ComfyUI's native LoRA adapter parser plus the direct MiniMaxH3 "
            "module aliases from ComfyUI PR #15662. No model or adapter fingerprint "
            "is used as an execution gate."
        ),
    }
    return patched, _json(report)
