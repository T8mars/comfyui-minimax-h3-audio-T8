"""Kijai's native relative 32-head PDD format, not T8's absolute-bank format.

Keep every source backbone/AdaLN/bias patch. Only the four shape-changing
head adapters become algebraically identical padded relative differences.
Do not rename, rewrite, lift or substitute the user's original safetensors.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch
import comfy.lora
import comfy.sd
import comfy.utils
from comfy.weight_adapter.lora import LoRAAdapter

from . import pdd_advanced as pdd
from .long_video_dual_identity import content_identity
from .patch_stack_policy import warn_patch_stack

MODE = "comfyui_native_kijai_pdd_relative_heads_plus_backbone"
HEAD_KEYS = tuple(f"diffusion_model.final_layer.{stream}_out.{field}"
                  for stream in ("video", "audio") for field in ("weight", "bias"))


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _consumed(state, key_map, patches):
    used = set().union(*(getattr(patch, "loaded_keys", set()) for patch in patches.values()))
    for alias, target in key_map.items():
        if target in patches:
            used.update(alias + suffix for suffix in (".alpha", ".dora_scale", ".w_norm", ".diff", ".set_weight")
                        if alias + suffix in state)
        bias = str(target).removesuffix(".weight") + ".bias"
        if bias in patches:
            used.update(alias + suffix for suffix in (".b_norm", ".diff_b") if alias + suffix in state)
    return used & set(state)


def relative_head_diff(adapter, expected_shape, key):
    """Exact Core LoRA update before strength/base padding, including bias heads.

    A proven identity B avoids a huge redundant GEMM. This is equality of every
    entry, not a diagonal sample or approximate identity test. Other finite B
    matrices still use the original B@A operation, never silently discard B.
    """
    if type(adapter) is not LoRAAdapter:
        raise ValueError("Kijai PDD head must be a native LoRAAdapter: " + key)
    up, down, alpha, mid, dora, reshape = adapter.weights
    if (mid is not None or dora is not None or reshape is None
            or tuple(reshape) != tuple(expected_shape)
            or up.ndim != 2 or down.ndim != 2 or up.shape[1] != down.shape[0]
            or up.numel() == 0 or down.numel() == 0
            or up.shape[0] * down.shape[1] != math.prod(expected_shape)):
        raise ValueError("Kijai PDD head reshape/adapter geometry is invalid: " + key)
    if not bool(torch.isfinite(up).all()) or not bool(torch.isfinite(down).all()):
        raise ValueError("Kijai PDD head contains nonfinite weights: " + key)
    scale = 1.0 if alpha is None else float(alpha) / down.shape[0]
    if not math.isfinite(scale):
        raise ValueError("Kijai PDD head alpha is not finite: " + key)
    if up.shape[0] == up.shape[1] and torch.equal(up, torch.eye(up.shape[0], device=up.device, dtype=up.dtype)):
        difference = down.to(dtype=torch.float32).clone()
    else:
        difference = up.to(dtype=torch.float32) @ down.to(dtype=torch.float32)
    difference = (difference.reshape(expected_shape) * scale).contiguous()
    if not bool(torch.isfinite(difference).all()):
        raise ValueError("Kijai PDD head update overflows: " + key)
    return "diff", (difference, {"pad_weight": True})


def compile_native_patches(model, state):
    """Parse every source key and validate against the actually selected MODEL."""
    if not state or any(not isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError("Kijai PDD requires a tensor-only original adapter")
    if any(not bool(torch.isfinite(value).all()) for value in state.values()):
        raise ValueError("Kijai PDD adapter contains nonfinite source tensors")
    for key in state:
        for suffix, counterpart in ((".lora_A.weight", ".lora_B.weight"),
                                    (".lora_B.weight", ".lora_A.weight"),
                                    (".lora_down.weight", ".lora_up.weight"),
                                    (".lora_up.weight", ".lora_down.weight")):
            if key.endswith(suffix) and key.removesuffix(suffix) + counterpart not in state:
                raise ValueError("Kijai PDD adapter has an incomplete A/B pair: " + key)
    key_map = comfy.lora.model_lora_keys_unet(model.model, {})
    loaded = comfy.lora.load_lora(state, key_map, log_missing=False)
    unused = set(state) - _consumed(state, key_map, loaded)
    if unused:
        raise ValueError("Kijai PDD adapter has unmapped keys: " + ", ".join(sorted(unused)[:8]))
    if not set(HEAD_KEYS).issubset(loaded):
        raise ValueError("Kijai PDD needs all four relative 32-head weight/bias adapters")
    heads, backbone = {}, {}
    final = model.get_model_object("diffusion_model.final_layer")
    for key, patch in loaded.items():
        value = comfy.utils.get_attr(model.model, key)
        if key in HEAD_KEYS:
            stream, field = key.rsplit(".", 2)[1:]
            head = getattr(final, stream)
            shape = ((pdd.PDD_NUM_STEPS * head.out_features, head.in_features)
                     if field == "weight" else (pdd.PDD_NUM_STEPS * head.out_features,))
            expected_base = (head.out_features,) if field == "bias" else (head.out_features, head.in_features)
            if tuple(value.shape) != expected_base:
                # Core clones share a network and original-weight backups. A
                # completed LOW remains loaded even after cleanup clears its
                # current_patcher. Read its authenticated original head, never
                # unload a user MODEL or accept an expanded checkpoint itself.
                owners = [item.model for item in comfy.model_management.current_loaded_models
                          if item.model is not None and item.model.model is model.model
                          and item.model.backup is model.backup
                          and item.model.patches_uuid == model.model.current_weight_patches_uuid]
                backup = model.backup.get(key)
                if (len(owners) != 1 or type(backup).__module__ != "comfy.model_patcher"
                        or getattr(backup, "_fields", None) != ("weight", "inplace_update")
                        or type(backup.inplace_update) is not bool
                        or not isinstance(backup.weight, torch.Tensor)
                        or tuple(backup.weight.shape) != expected_base
                        or not bool(torch.isfinite(backup.weight).all())):
                    raise ValueError("Kijai PDD expanded base lacks its loaded Core original-head owner: " + key)
                from .modular_sampling.pdd_stages import _head_contract
                _head_contract(owners[0])  # Bind the actual loaded PDD head receipt.
                value = backup.weight
            if tuple(value.shape) != expected_base:
                raise ValueError("Kijai PDD base already has expanded or invalid heads: " + key)
            heads[key] = relative_head_diff(patch, shape, key)
        elif type(patch) is LoRAAdapter:
            up, down, alpha, mid, dora, reshape = patch.weights
            if (mid is not None or dora is not None or reshape is not None
                    or up.ndim != 2 or down.ndim != 2 or up.shape[1] != down.shape[0]
                    or tuple(value.shape) != (up.shape[0], down.shape[1])):
                raise ValueError("Kijai PDD backbone/AdaLN shape differs from the selected base: " + key)
            if alpha is not None and not math.isfinite(float(alpha)):
                raise ValueError("Kijai PDD backbone alpha is not finite: " + key)
            backbone[key] = patch
        else:
            if (not isinstance(patch, tuple) or patch[0] not in ("diff", "set")
                    or not isinstance(patch[1][0], torch.Tensor)
                    or tuple(patch[1][0].shape) != tuple(value.shape)):
                raise ValueError("Kijai PDD regular/bias patch shape differs from the selected base: " + key)
            backbone[key] = patch
    return backbone, heads, len(loaded)


def apply_native_state(model, state, strength=1.0):
    strength = float(strength)
    if not math.isfinite(strength) or not 0.0 <= strength <= 1.0:
        raise ValueError("Kijai PDD strength must be finite and in [0,1]")
    backbone, heads, count = compile_native_patches(model, state)
    # Core's generic bypass helper overwrites a pre-existing bypass_lora owner.
    # Never do that to a user's stack: in that case use native weight patches
    # for this additional backbone and retain every existing injection verbatim.
    if model.get_injections("bypass_lora"):
        warn_patch_stack("Existing bypass LoRA retained; Kijai PDD backbone uses ordinary Core patches")
        patched = model.clone()
        if set(patched.add_patches(backbone, strength)) != set(backbone):
            raise ValueError("Kijai PDD did not apply every backbone/bias patch")
        backbone_mode = "native_weight_patches_existing_bypass_preserved"
    else:
        # Core adapters may share the parser's global loaded_keys set. Do not
        # use one head's set to subtract the entire backbone by accident.
        key_map = comfy.lora.model_lora_keys_unet(model.model, {})
        head_keys = {alias + suffix for alias, target in key_map.items() if target in HEAD_KEYS
                     for suffix in (".lora_A.weight", ".lora_B.weight", ".lora_up.weight", ".lora_down.weight",
                                    ".reshape_weight", ".alpha")
                     if alias + suffix in state}
        if len([key for key in head_keys if key.endswith((".lora_A.weight", ".lora_down.weight"))]) != 4:
            raise ValueError("Kijai PDD heads must provide four native converted adapter pairs")
        source_backbone = {key: value for key, value in state.items() if key not in head_keys}
        patched, _ = comfy.sd.load_bypass_lora_for_models(model, None, source_backbone, strength, 0.0)
        backbone_mode = "native_model_only_bypass_plus_regular_bias"
    if set(patched.add_patches(heads, strength_patch=strength, strength_model=1.0)) != set(heads):
        raise ValueError("Kijai PDD did not apply all four relative head banks")
    return patched, {"application_mode": MODE, "backbone_mode": backbone_mode,
                     "mapped_targets": count, "backbone_targets": len(backbone), "native_head_patch_targets": 4,
                     "strength": strength, "head_difference_identity": content_identity(heads),
                     "native_head_differences": {key: content_identity(value[1][0]) for key, value in heads.items()},
                     "head_semantics": "pad original one-head base with zeros, then add strength times every original head update"}


def build_kijai_pdd_8step_setup(model, av_latent, path, *, base_variant="FL2VA", strength=1.0):
    if base_variant not in pdd.PDD_VARIANTS:
        raise ValueError("Select FL2VA or Ref2VA")
    diffusion = model.get_model_object("diffusion_model")
    if diffusion.__class__.__name__ != "MiniMaxH3Model":
        raise TypeError("Kijai PDD requires native MiniMaxH3Model")
    capability = pdd.probe_native_pdd_core(diffusion)
    if not capability["available"]:
        raise RuntimeError("Kijai relative PDD needs native shape-changing weight/bias lifecycle and FinalLayer schedule support: "
                           + json.dumps(capability, sort_keys=True))
    pdd._assert_clean_lora_stack(model)
    path = Path(path)
    before = _sha(path)
    state, metadata = comfy.utils.load_torch_file(str(path), safe_load=True, return_metadata=True)
    patched, lora = apply_native_state(model, state, strength)
    if _sha(path) != before:
        raise ValueError("Kijai PDD source changed during setup")
    prepared, sampler, sigmas = pdd.setup_dual_clock_sampling(patched, av_latent, 8, 12.0, 3.0, "euler", "simple")
    report = {"schema": "t8_minimax_h3_pdd_8step_setup_v2", "status": "validated_kijai_relative_setup_contract",
              "lora": lora, "adapter": {"filename": path.name, "sha256": before, "tensor_count": len(state)},
              "base": {"variant_declared_by_user": base_variant, "pruned_curve": hasattr(diffusion, "adaln_t_table"),
                       "variant_identity_limit": "Shape compatibility is not proof of FL/Ref training basis; select the matching source/base."},
              "sampling": pdd.validate_pdd_sigmas(sigmas), "native_core_probe": capability,
              "qualification": {"render_performed_by_setup": False, "quality_accepted": False},
              "boundary": "Original Kijai full/pruned relative heads, no disk conversion. Absolute LOW0:4/HIGH4:8 via existing PDD Stage Setup. "
                          "External EAV/Relay and user patches remain explicit; not arbitrary combination or teacher-equivalence certification."}
    prepared.set_attachments(pdd.PDD_ATTACHMENT_KEY, report)
    prepared.set_attachments("t8_minimax_h3_pdd_lora_metadata", dict(metadata or {}))
    return prepared, sampler, sigmas, json.dumps(report, ensure_ascii=False, sort_keys=True)
