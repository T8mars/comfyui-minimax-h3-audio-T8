from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import comfy.lora
import comfy.lora_convert
import comfy.model_base
import comfy.utils
import torch
from torch import nn

from .h3_weight_diagnostics import describe_h3_payload, lora_mapping_diagnostics


LOG = logging.getLogger(__name__)
SCHEMA = "t8.minimax_h3.lora_compat.v1"
FAST_H3_VSA_GATE_ATTACHMENT_KEY = "t8_fast_h3_vsa_gate_contract_v1"
FILE_ALPHA_POLICIES = ("legacy_tensor_only", "metadata_alpha_per_rank")
PREFIX_POLICIES = ("legacy", "unique_double_prefix")
LINEAR_LORA_FORMATS = ((".lora_down.weight", ".lora_up.weight"),
                       ("_lora.down.weight", "_lora.up.weight"),
                       (".lora_A.weight", ".lora_B.weight"),
                       (".lora.down.weight", ".lora.up.weight"),
                       (".lora_A", ".lora_B"),
                       (".lora_linear_layer.down.weight", ".lora_linear_layer.up.weight"),
                       (".lora_A.default.weight", ".lora_B.default.weight"))


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)


def add_minimax_h3_direct_lora_keys(model, key_map: dict | None = None) -> dict:
    """Add the direct DiffSynth/ModelScope H3 aliases proposed in ComfyUI #15662.

    The alias is purely structural: ``diffusion_model.blocks.0...weight`` maps
    from ``blocks.0...``.  No filename, file size, hash or metadata identity is
    required. An alternate MODEL class exposing the same real parameter paths
    receives the same aliases; existing Core mappings always keep priority.
    """

    result = {} if key_map is None else dict(key_map)
    state_dict = getattr(model, "state_dict", None)
    if not callable(state_dict):
        return result  # optional aliases must not invalidate the Core key map
    for key in state_dict().keys():
        if key.startswith("diffusion_model.") and key.endswith(".weight"):
            result.setdefault(key[len("diffusion_model.") : -len(".weight")], key)
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


def _fastvideo_qkv_scopes(state: dict[str, torch.Tensor]) -> list[str]:
    scopes: dict[str, set[str]] = {}
    for key in state:
        if not key.startswith(("transformer_blocks.", "token_refiner.refiner_blocks.")):
            continue
        for projection in ("q", "k", "v"):
            for factor in ("A", "B"):
                suffix = f".attn.to_{projection}.lora_{factor}.weight"
                if key.endswith(suffix):
                    scopes.setdefault(key[:-len(suffix)], set()).add(projection)
    return sorted(scope for scope, projections in scopes.items() if projections == {"q", "k", "v"})


def _fastvideo_h3_layout(state: dict[str, torch.Tensor]) -> bool:
    """Detect FastVideo's published H3 adapter layout structurally.

    This is deliberately not a filename, byte-size, hash, or model identity
    check.  It only selects the converter when the tensor namespace itself is
    the FastVideo/Diffusers H3 split-QKV layout.
    """

    # Recognition must not require a valid pair: otherwise a missing Q.B makes
    # a clearly known, damaged Q/K/V group silently bypass its validation.
    return bool(_fastvideo_qkv_scopes(state))


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
            from .patch_stack_policy import warn_patch_stack
            warn_patch_stack(f"FastVideo adapter replaces the gate at {path} using the explicitly selected newer payload")
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
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Convert a FastVideo H3 adapter to ComfyUI's fused H3 module layout.

    FastVideo publishes Diffusers-style split Q/K/V factors while ComfyUI H3
    owns one fused ``qkv_proj``.  The block-diagonal construction below is an
    exact algebraic representation of the three original low-rank updates.
    Dense ``.diff``/``.diff_b`` payloads are renamed onto ComfyUI's H3 module
    names and remain additive patches. Unknown extensions retain their original
    keys and objects for the Core parser and its unused-key diagnostics; a
    recognized QKV group does not make the rest of the file a closed schema.
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

    def scalar_alpha(key: str) -> float:
        value = state[key]
        if not isinstance(value, torch.Tensor) or value.numel() != 1:
            raise ValueError(f"FastVideo H3 alpha must be a finite scalar: {key}")
        result = float(value.item())
        if not math.isfinite(result):
            raise ValueError(f"FastVideo H3 alpha must be a finite scalar: {key}")
        return result

    qkv_scopes = _fastvideo_qkv_scopes(state)
    fused_count = 0
    for scope in qkv_scopes:
        factors: list[tuple[torch.Tensor, torch.Tensor]] = []
        scales: list[float] = []
        has_alpha = False
        keys: list[str] = []
        for projection in ("q", "k", "v"):
            a_key = f"{scope}.attn.to_{projection}.lora_A.weight"
            b_key = f"{scope}.attn.to_{projection}.lora_B.weight"
            if a_key not in state or b_key not in state:
                raise ValueError(f"incomplete FastVideo H3 QKV group under {scope}")
            a = state[a_key]
            b = state[b_key]
            if a.ndim != 2 or b.ndim != 2 or a.shape[0] < 1 or b.shape[1] != a.shape[0]:
                raise ValueError(f"invalid FastVideo H3 LoRA pair {a_key}/{b_key}")
            factors.append((a, b))
            keys.extend((a_key, b_key))
            alpha_key = f"{scope}.attn.to_{projection}.alpha"
            if alpha_key in state:
                scales.append(scalar_alpha(alpha_key) / int(a.shape[0]))
                keys.append(alpha_key)
                has_alpha = True
            else:
                scales.append(1.0)
        input_widths = {int(a.shape[1]) for a, _ in factors}
        if len(input_widths) != 1:
            raise ValueError(f"FastVideo H3 Q/K/V input widths differ under {scope}")
        ranks = [int(a.shape[0]) for a, _ in factors]
        output_widths = [int(b.shape[0]) for _, b in factors]
        unequal_scales = len(set(scales)) > 1
        promoted_dtype = factors[0][1].dtype
        for _, b in factors[1:]:
            promoted_dtype = torch.promote_types(promoted_dtype, b.dtype)
        fused_a = torch.cat([a for a, _ in factors], dim=0).contiguous()
        fused_b = torch.zeros(
            (sum(output_widths), sum(ranks)),
            # Unequal source alpha/rank scales cannot be represented by one
            # fused alpha. Keep scaled B in FP32 rather than rounding it back
            # to low-precision storage. Alpha-free legacy conversion is intact.
            dtype=torch.promote_types(promoted_dtype, torch.float32) if unequal_scales else promoted_dtype,
            device=factors[0][1].device,
        )
        row = 0
        col = 0
        for (_, b), out_width, rank, scale in zip(factors, output_widths, ranks, scales):
            fused_b[row : row + out_width, col : col + rank] = (
                b.to(fused_b.dtype) * scale if unequal_scales else b
            )
            row += out_width
            col += rank
        target = f"{_fastvideo_scope_target(scope)}.attn.qkv_proj"
        output[f"{target}.lora_A.weight"] = fused_a
        output[f"{target}.lora_B.weight"] = fused_b.contiguous()
        if has_alpha and not unequal_scales:
            # Core divides alpha by the new, summed rank, not an old Q/K/V rank.
            output[f"{target}.alpha"] = torch.tensor(
                scales[0] * sum(ranks), dtype=torch.float64, device=fused_a.device
            )
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
            if matched is None or not module.startswith((
                "transformer_blocks.", "token_refiner.refiner_blocks."
            )):
                continue
            scope = module[: -len(matched)]
            target = _fastvideo_scope_target(scope) + direct_suffixes[matched]
        a = state[key]
        b = state[b_key]
        if a.ndim != 2 or b.ndim != 2 or a.shape[0] < 1 or b.shape[1] != a.shape[0]:
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
        alpha_key = module + ".alpha"
        if alpha_key in state:
            scalar_alpha(alpha_key)
            output[f"{target}.alpha"] = state[alpha_key]
            consumed.add(alpha_key)
        direct_count += 1

    dense_count = 0
    for key in sorted(state):
        if key in consumed:
            continue
        suffix = next((value for value in (".diff_b", ".diff") if key.endswith(value)), None)
        if suffix is None:
            continue
        module = key[: -len(suffix)]
        try:
            target = _fastvideo_direct_target(module)
        except ValueError:
            # An unrecognized namespace is the selected Core parser's concern,
            # not a reason to reject otherwise usable user-selected patches.
            continue
        output[f"{target}{suffix}"] = state[key]
        consumed.add(key)
        dense_count += 1

    preserved = sorted(set(state) - consumed)
    for key in preserved:
        if key in output:
            raise ValueError(f"ambiguous FastVideo H3 conversion target collision: {key}")
        output[key] = state[key]
    report = {
        "conversion": "fastvideo_h3_diffusers_to_comfyui_fused",
        "fused_qkv_groups": fused_count,
        "direct_lora_modules": direct_count,
        "dense_delta_tensors": dense_count,
        "swiglu_half_swaps": swiglu_swaps,
    }
    if preserved:
        report["preserved_source_keys"] = preserved
        report["unknown_payload_policy"] = "preserve_for_core_parser_and_unused_diagnostics"
    return output, report


def explicit_prefix_aliases(state, key_map, model, policy):
    """Add a unique whole-pair alias without rewriting or splicing source keys."""
    if policy not in PREFIX_POLICIES:
        raise ValueError(f"Unknown prefix_policy: {policy!r}")
    report = {"requested_policy": policy, "added_aliases": [], "not_applied": [],
              "source_keys_rewritten": False, "execution_verified": False}
    result = dict(key_map)
    if policy == "legacy":
        return result, report
    complete = {}
    for key in state:
        for down, up in LINEAR_LORA_FORMATS:
            if key.endswith(down):
                module = key[:-len(down)]
                if module + up in state:
                    complete.setdefault(module, []).append((down, up))
    # Check the existing *converted* consumer graph: split-QKV and Core's other
    # format conversions can occupy a target absent from the raw key namespace.
    # Release this temporary conversion before building the selected final one.
    baseline, _ = convert_fastvideo_h3_adapter(state)
    baseline = comfy.lora_convert.convert_lora(baseline)
    occupied = set(comfy.lora.load_lora(baseline, key_map, log_missing=False))
    del baseline
    weights = model.state_dict()
    for module in sorted(complete):
        if module in key_map or not module.startswith("diffusion_model.diffusion_model."):
            continue
        canonical = module[len("diffusion_model."):]
        target = key_map.get(canonical)
        if not isinstance(target, str):
            report["not_applied"].append({"module": module, "reason": "no_unique_existing_target"})
            continue
        if target in occupied or len(complete[module]) != 1:
            report["not_applied"].append({"module": module, "reason": "ambiguous_target_or_pair_formats"})
            continue
        weight = weights.get(target)
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
            report["not_applied"].append({"module": module, "reason": "not_a_known_linear_target"})
            continue
        down, up = complete[module][0]
        a, b = state[module + down], state[module + up]
        if (not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor)
                or a.ndim != 2 or b.ndim != 2 or a.shape[0] < 1
                or b.shape[1] != a.shape[0] or (b.shape[0], a.shape[1]) != tuple(weight.shape)):
            raise ValueError(f"Repeated-prefix linear pair does not match actual target {target}: {module}")
        result[module] = target
        occupied.add(target)
        report["added_aliases"].append({"source_module": module, "target": target})
    return result, report


def _file_alpha_target(module, state, key_map, weights):
    """Resolve existing linear mappings; no invented alias or model-name gate."""
    target = key_map.get(module)
    split = False
    if target is None and _fastvideo_h3_layout(state):
        for scope in _fastvideo_qkv_scopes(state):
            if module in {f"{scope}.attn.to_{part}" for part in ("q", "k", "v")}:
                target = key_map.get(_fastvideo_scope_target(scope) + ".attn.qkv_proj")
                split = True
                break
    if target is None and _fastvideo_h3_layout(state):
        for suffix, replacement in ((".attn.to_out.0", ".attn.out_proj"),
                                    (".ff.net.0.proj", ".mlp.fc1"),
                                    (".ff.net.2", ".mlp.fc2"),
                                    (".adaln_proj.linear", ".adaln_proj.linear")):
            if (module.endswith(suffix)
                    and module.startswith(("transformer_blocks.", "token_refiner.refiner_blocks."))):
                target = key_map.get(_fastvideo_scope_target(module[:-len(suffix)]) + replacement)
                break
        if module == "norm_out.linear":
            target = key_map.get("final_layer.adaln_proj.linear")
    weight = weights.get(target) if isinstance(target, str) else None
    return (target, weight, split) if isinstance(weight, torch.Tensor) and weight.ndim == 2 else None


def interpret_file_alpha(state, metadata, policy, model, key_map):
    """Explicit literal alpha/rank fallback, before any QKV rank fusion.

    Metadata can describe a configured rather than effective training alpha.
    Never auto-enable this interpretation, rescale factors, or alter strength.
    Only complete mapped 2D linear pairs without their own alpha participate.
    """
    if policy not in FILE_ALPHA_POLICIES:
        raise ValueError(f"Unknown file_alpha_policy: {policy!r}")
    report = {"requested_policy": policy, "injected_module_count": 0,
              "injected_modules": [], "per_key_alpha_modules": [],
              "skipped_modules": [], "training_scale_verified": False,
              "meaning": "explicit literal file alpha / actual tensor rank; not producer provenance"}
    result = dict(state)
    if policy == "legacy_tensor_only" or not metadata or metadata.get("ss_network_alpha") is None:
        report["status"] = "legacy_tensor_only" if policy == "legacy_tensor_only" else "no_file_alpha"
        return result, report
    weights = model.state_dict()
    formats = LINEAR_LORA_FORMATS
    native_patches = comfy.lora.load_lora(state, key_map, log_missing=False)
    modules = {key[:-len(down)] for key in state for down, _ in formats if key.endswith(down)}
    resolved = {module: _file_alpha_target(module, state, key_map, weights) for module in modules}
    native_sources = {}
    for module, target in resolved.items():
        if target is not None and not target[2] and any(
                module + down in state and module + up in state for down, up in formats):
            native_sources.setdefault(target[0], set()).add(module)
    candidates = []
    for module in sorted(modules):
        target = resolved[module]
        if target is not None and not target[2] and len(native_sources.get(target[0], ())) > 1:
            report["skipped_modules"].append({"module": module, "reason": "ambiguous_target_aliases_not_interpreted"})
            continue
        if module + ".alpha" in state:
            report["per_key_alpha_modules"].append(module)
            continue
        # Core has a format priority. Do not validate an unused second format
        # or combine factors from different namespaces. Ambiguity leaves Core's
        # original payload/priority unchanged rather than guessing an alpha.
        complete = [(down, up) for down, up in formats
                    if module + down in state and module + up in state]
        up_markers = (".lora_up.weight", "_lora.up.weight", ".lora_B.weight",
                      ".lora.up.weight", ".lora_B", ".lora_linear_layer.up.weight",
                      ".lora_B.default.weight")
        if (len(complete) != 1 or sum(module + suffix in state for suffix in up_markers) != 1
                or sum(module + down in state for down, _ in formats) != 1):
            report["skipped_modules"].append({"module": module, "reason": "ambiguous_or_incomplete_pair_not_interpreted"})
            continue
        a, b = (state[module + suffix] for suffix in complete[0])
        special = any(module + suffix in state for suffix in (
            ".dora_scale", ".lora_magnitude_vector", ".lora_mid.weight", ".reshape_weight",
            ".diff", ".w_norm", ".set_weight"))
        if (target is None or special or not isinstance(a, torch.Tensor) or a.ndim != 2
                or not isinstance(b, torch.Tensor) or b.ndim != 2):
            report["skipped_modules"].append({"module": module, "reason": "not_a_complete_mapped_plain_linear_pair"})
            continue
        if module in key_map:
            adapter = native_patches.get(target[0])
            pair_keys = {module + suffix for suffix in complete[0]}
            if (type(adapter) is not comfy.lora.weight_adapter.LoRAAdapter
                    or not pair_keys.issubset(adapter.loaded_keys)):
                report["skipped_modules"].append({"module": module, "reason": "not_the_actual_native_linear_consumer"})
                continue
        _, weight, split = target
        if (a.shape[0] < 1 or b.shape[1] != a.shape[0] or a.shape[1] != weight.shape[1]
                or (not split and b.shape[0] != weight.shape[0])):
            raise ValueError(f"File alpha metadata cannot interpret mismatched linear pair: {module}")
        candidates.append((module, a))
    if candidates:
        try:
            alpha = float(metadata["ss_network_alpha"])
        except (TypeError, ValueError) as error:
            raise ValueError("File metadata alpha must be a finite number for the selected interpretation") from error
        if not math.isfinite(alpha):
            raise ValueError("File metadata alpha must be finite for the selected interpretation")
        declared_rank = metadata.get("ss_network_dim")
        if declared_rank is not None:
            try:
                rank = float(declared_rank)
            except (TypeError, ValueError) as error:
                raise ValueError("File metadata rank must be a positive integer") from error
            if not math.isfinite(rank) or not rank.is_integer() or rank <= 0:
                raise ValueError("File metadata rank must be a positive integer")
            if any(int(a.shape[0]) != int(rank) for _, a in candidates):
                raise ValueError("File metadata rank disagrees with actual alpha-free linear tensor rank; cannot infer its training scale")
        for module, a in candidates:
            result[module + ".alpha"] = torch.tensor(alpha, dtype=torch.float64, device=a.device)
            report["injected_modules"].append({"module": module, "actual_rank": int(a.shape[0]),
                                                "alpha": alpha, "scale": alpha / int(a.shape[0])})
        report.update(file_alpha=alpha, declared_rank=declared_rank)
    report["injected_module_count"] = len(candidates)
    report["status"] = "applied_explicit_interpretation" if candidates else "no_eligible_alpha_free_pair"
    return result, report


def load_minimax_h3_lora_model(
    model,
    lora_path: str | Path,
    strength_model: float = 1.0,
    *,
    file_alpha_policy: str = "legacy_tensor_only",
    prefix_policy: str = "legacy",
) -> tuple[Any, str]:
    path = Path(lora_path)
    strength = float(strength_model)
    if not math.isfinite(strength):
        raise ValueError("strength_model must be finite")

    raw, metadata = comfy.utils.load_torch_file(
        str(path), safe_load=True, return_metadata=True
    )
    vsa_gates, adapter_state = _extract_fastvideo_vsa_gate_weights(raw)
    key_map, direct_alias_count = build_minimax_h3_lora_key_map(model.model)
    key_map, prefix_interpretation = explicit_prefix_aliases(adapter_state, key_map, model.model, prefix_policy)
    adapter_state, alpha_interpretation = interpret_file_alpha(
        adapter_state, metadata, file_alpha_policy, model.model, key_map)
    converted, conversion = convert_fastvideo_h3_adapter(adapter_state)
    converted = comfy.lora_convert.convert_lora(converted)
    source_keys = sorted(str(key) for key in raw.keys())
    payload_diagnostics = describe_h3_payload(source_keys, metadata)
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
    mapping = lora_mapping_diagnostics(converted, key_map, patches, applied)
    warnings = []
    if alpha_interpretation["skipped_modules"]:
        warnings.append("File alpha interpretation was not applied to ambiguous, unmapped or non-plain-linear payloads; the original payload is retained for Core. See alpha_interpretation.skipped_modules.")
    if prefix_interpretation["not_applied"]:
        warnings.append("Repeated-prefix interpretation was not applied to ambiguous or unknown targets; original payload and Core mappings were retained. See prefix_interpretation.not_applied.")
    for requirement in payload_diagnostics["special_runtime_requirements"]:
        warnings.append(f"{requirement['kind']} requires {requirement['runtime']}; ordinary weight patches do not establish the full algorithm.")
    if mapping["converted_unmapped_tensor_count"]:
        warnings.append(f"{mapping['converted_unmapped_tensor_count']} converted tensor keys were not reported as consumed by the native parser.")
    if missed_targets:
        warnings.append(f"{len(missed_targets)} mapped targets were not registered by this MODEL.")
    if strength == 0:
        warnings.append("Strength is zero: registered patches do not imply an effective weight change.")
    for warning in warnings:
        LOG.warning("MiniMax H3 LoRA %s: %s", path.name, warning)
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
        "alpha_interpretation": alpha_interpretation,
        "prefix_interpretation": prefix_interpretation,
        "payload_diagnostics": payload_diagnostics,
        "mapping_diagnostics": mapping,
        "warnings": warnings,
        "full_algorithm_verified": False,
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
    if hasattr(patched, "set_attachments"):
        patched.set_attachments("t8_h3_lora_compat_report", report)
    return patched, _json(report)
