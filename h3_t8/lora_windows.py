"""Per-LoRA diffusion-progress windows, consumed by native Core weight hooks."""

from __future__ import annotations

import json
import math
from pathlib import Path

import comfy.hooks
import comfy.lora
import comfy.lora_convert
import comfy.ops
import comfy.model_patcher
import comfy.patcher_extension
import comfy.utils
import folder_paths

from .h3_lora_compat_advanced import (
    build_minimax_h3_lora_key_map,
    convert_fastvideo_h3_adapter,
    explicit_prefix_aliases,
    interpret_file_alpha,
    _extract_fastvideo_vsa_gate_weights,
)
from .h3_weight_diagnostics import lora_mapping_diagnostics


def quantized_hook_scope(executor, *args, **kwargs):
    """Delegate Core hooks without feeding virtual serialization fields to getattr.

    MixedPrecision Linear serializes scales inside QuantizedTensor as extra
    state_dict keys, not Python module attributes. Only those known virtual
    fields are hidden from Core's original-weight inventory; actual tensors and
    their internal scales are untouched. All changes are instance-local and
    restored even when a foreign getter or sampling fails.
    """
    patcher = executor.class_obj.model_patcher
    getter, state_dict = patcher.get_key_patches, patcher.model_state_dict
    absent = object()
    prior_getter = patcher.__dict__.get("get_key_patches", absent)
    prior_state = patcher.__dict__.get("model_state_dict", absent)
    quantized_type = getattr(comfy.ops, "QuantizedTensor", ())
    setters = {}
    prior_methods = {}

    def unpin_before_replacement(key):
        unpin = getattr(patcher, "unpin_weight", None)
        if (
            key in getattr(patcher, "pinned", ())
            and getattr(unpin, "__func__", None)
            is comfy.model_patcher.ModelPatcher.unpin_weight
        ):
            unpin(key)
            return True
        return False

    def scope_core_method(name, wrapper):
        original = getattr(patcher, name)
        function = getattr(original, "__func__", original)
        if getattr(function, "__module__", None) != comfy.model_patcher.__name__:
            return  # A foreign implementation keeps its own contract.
        prior_methods[name] = patcher.__dict__.get(name, absent)
        setattr(patcher, name, wrapper(original))

    def unpatch_wrapper(original):
        def unpatch(whitelist_keys_set=None):
            restored = {
                key: value
                for key, value in patcher.hook_backup.items()
                if isinstance(value[0], quantized_type)
                and (not whitelist_keys_set or key in whitelist_keys_set)
            }
            repin = [key for key in restored if unpin_before_replacement(key)]
            # Core's .data.copy_ creates a detached QuantizedTensor whose layout
            # parameters are separate. Replace from the complete Core backup,
            # retaining both storage and scale rather than requantizing again.
            try:
                result = original(whitelist_keys_set)
                for key, (weight, device) in restored.items():
                    comfy.utils.set_attr_param(patcher.model, key, weight.to(device))
                return result
            finally:
                for key in repin:
                    patcher.pin_weight_to_device(key)

        return unpatch

    def patch_wrapper(original):
        def patch(*patch_args, **patch_kwargs):
            hooks = patch_kwargs.get("hooks", patch_args[0] if patch_args else None)
            key = patch_kwargs.get(
                "key", patch_args[2] if len(patch_args) > 2 else None
            )
            repin = unpin_before_replacement(key)
            try:
                result = original(*patch_args, **patch_kwargs)
            finally:
                if repin:
                    patcher.pin_weight_to_device(key)
            cache = patcher.cached_hook_patches.get(hooks, {})
            if key in cache:
                weight = comfy.utils.get_attr(patcher.model, key)
                if isinstance(weight, quantized_type):
                    stored, device = cache[key]
                    cache[key] = (weight.to(device=stored.device, copy=True), device)
            return result

        return patch

    def cached_wrapper(original):
        def patch(cached_weights, key, memory_counter):
            weight = comfy.utils.get_attr(patcher.model, key)
            cached, device = cached_weights[key]
            if not isinstance(weight, quantized_type) or not isinstance(
                cached, quantized_type
            ):
                return original(cached_weights, key, memory_counter)
            if key not in patcher.hook_backup:
                target = patcher.offload_device
                if (
                    patcher.hook_mode == comfy.hooks.EnumHookMode.MaxSpeed
                    and memory_counter.use(weight)
                ):
                    target = weight.device
                patcher.hook_backup[key] = (
                    weight.to(device=target, copy=True),
                    weight.device,
                )
            repin = unpin_before_replacement(key)
            try:
                comfy.utils.set_attr_param(
                    patcher.model, key, cached.to(device=device, copy=True)
                )
            finally:
                if repin:
                    patcher.pin_weight_to_device(key)

        return patch

    def prepare_setter(module):
        if id(module) in setters:
            return
        original = getattr(module, "set_weight", None)
        function = getattr(original, "__func__", original)
        if (
            not callable(original)
            or getattr(function, "__module__", None) != comfy.ops.__name__
        ):
            return  # Unknown providers retain their own setter contract.
        previous = module.__dict__.get("set_weight", absent)

        def set_weight(weight, *setter_args, **setter_kwargs):
            # Reuse Core's normal requantizer, including scale calculation and
            # caller seed; MixedPrecisionOp requires out-of-place replacement.
            if setter_kwargs.get("inplace_update") is True:
                setter_kwargs = {**setter_kwargs, "inplace_update": False}
            return original(weight, *setter_args, **setter_kwargs)

        setters[id(module)] = (module, previous)
        module.set_weight = set_weight

    def inventory(*inventory_args, **inventory_kwargs):
        values = state_dict(*inventory_args, **inventory_kwargs)
        skipped = []
        for key in values:
            parent, _, attribute = key.rpartition(".")
            if attribute not in (
                "weight",
                "weight_scale",
                "weight_scale_2",
                "comfy_quant",
            ):
                continue
            try:
                module = comfy.utils.get_attr(patcher.model, parent)
            except AttributeError:
                continue  # Keep actual missing-module errors for the delegate.
            if attribute == "weight":
                prepare_setter(module)
            if isinstance(getattr(module, "weight", None), quantized_type):
                prepare_setter(module)
                if not hasattr(module, attribute):
                    skipped.append(key)
        if not skipped:
            return values
        filtered = values.copy()
        for key in skipped:
            del filtered[key]
        return filtered

    def get_keys(*getter_args, **getter_kwargs):
        previous = patcher.__dict__.get("model_state_dict", absent)
        patcher.model_state_dict = inventory
        try:
            return getter(*getter_args, **getter_kwargs)
        finally:
            if previous is absent:
                del patcher.model_state_dict
            else:
                patcher.model_state_dict = previous

    patcher.get_key_patches = get_keys
    scope_core_method("unpatch_hooks", unpatch_wrapper)
    scope_core_method("patch_hook_weight_to_device", patch_wrapper)
    scope_core_method("patch_cached_hook_weights", cached_wrapper)
    try:
        return executor(*args, **kwargs)
    finally:
        for module, previous in setters.values():
            if previous is absent:
                module.__dict__.pop("set_weight", None)
            else:
                module.set_weight = previous
        for name, previous in (
            ("get_key_patches", prior_getter),
            ("model_state_dict", prior_state),
            *prior_methods.items(),
        ):
            if previous is absent:
                patcher.__dict__.pop(name, None)
            else:
                setattr(patcher, name, previous)


class WindowWeightHook(comfy.hooks.WeightHook):
    def add_hook_patches(self, model, model_options, target_dict, registered):
        added = super().add_hook_patches(model, model_options, target_dict, registered)
        if added and model_options is not None:
            kind = comfy.patcher_extension.WrappersMP.OUTER_SAMPLE
            key = "t8_lora_window_quant_state_v1"
            existing = comfy.patcher_extension.get_wrappers_with_key(
                kind, key, model_options, is_model_options=True
            )
            if quantized_hook_scope not in existing:
                comfy.patcher_extension.add_wrapper_with_key(
                    kind,
                    key,
                    quantized_hook_scope,
                    model_options,
                    is_model_options=True,
                )
        return added


def window_keyframes(start, end):
    if not all(math.isfinite(x) for x in (start, end)) or not 0 <= start < end <= 1:
        raise ValueError(
            "LoRA window must satisfy 0 <= start_percent < end_percent <= 1"
        )
    frames = comfy.hooks.HookKeyframeGroup()
    if start > 0:
        frames.add(comfy.hooks.HookKeyframe(0.0, 0.0, guarantee_steps=0))
    frames.add(comfy.hooks.HookKeyframe(1.0, start, guarantee_steps=0))
    if end < 1:
        frames.add(comfy.hooks.HookKeyframe(0.0, end, guarantee_steps=0))
    return frames


def create_window_hook(
    model,
    path,
    strength=1.0,
    start_percent=0.0,
    end_percent=1.0,
    previous_hooks=None,
    file_alpha_policy="legacy_tensor_only",
    prefix_policy="legacy",
):
    if not math.isfinite(strength):
        raise ValueError("LoRA strength must be finite")
    frames = window_keyframes(start_percent, end_percent)
    raw, metadata = comfy.utils.load_torch_file(
        str(path), safe_load=True, return_metadata=True
    )
    gates, state = _extract_fastvideo_vsa_gate_weights(raw)
    key_map, _ = build_minimax_h3_lora_key_map(model.model)
    key_map, prefix_report = explicit_prefix_aliases(
        state, key_map, model.model, prefix_policy
    )
    state, alpha_report = interpret_file_alpha(
        state, metadata, file_alpha_policy, model.model, key_map
    )
    state, conversion = convert_fastvideo_h3_adapter(state)
    state = comfy.lora_convert.convert_lora(state)
    patches = comfy.lora.load_lora(state, key_map, log_missing=False)
    if not patches:
        raise ValueError(
            "No LoRA weight targets mapped to the selected MODEL. "
            "Check tensor names and shapes, or use the selected algorithm's own runtime; "
            "an empty weight hook would not apply this file."
        )
    hook = WindowWeightHook(strength_model=strength, strength_clip=0.0)
    hook.weights = patches
    hook.need_weight_init = False
    hook.hook_keyframe = frames
    current = comfy.hooks.HookGroup()
    current.add(hook)
    combined = (
        previous_hooks.clone_and_combine(current)
        if previous_hooks is not None
        else current
    )
    report = {
        "schema": "t8.h3.lora_window.v1",
        "file": Path(path).name,
        "strength": strength,
        "start_percent": start_percent,
        "end_percent": end_percent,
        "clock": "native_model_sampling_percent_to_sigma",
        "end_exclusive": end_percent < 1,
        "mapped_targets": len(patches),
        "registration_is_execution": False,
        "mapping": lora_mapping_diagnostics(state, key_map, patches, set()),
        "prefix_interpretation": prefix_report,
        "alpha_interpretation": alpha_report,
        "conversion": conversion,
        "warning": "VSA gate tensors are not native weight hooks; gate algorithm is not enabled here"
        if gates
        else None,
    }
    return combined, report


def attach_windows(model, positive, rows):
    if not isinstance(rows, list):
        raise ValueError("LoRA windows must be an ordered list")
    hooks = None
    reports = []
    for row in rows:
        if not row.get("enabled", True):
            continue
        path = folder_paths.get_full_path_or_raise("loras", row["name"])
        hooks, report = create_window_hook(
            model,
            path,
            row.get("strength", 1.0),
            row.get("start_percent", 0.0),
            row.get("end_percent", 1.0),
            hooks,
            row.get("file_alpha_policy", "legacy_tensor_only"),
            row.get("prefix_policy", "legacy"),
        )
        reports.append(report)
    output = []
    for tensor, metadata in positive:
        updated = dict(metadata)
        existing = updated.get("hooks")
        if hooks is not None:
            updated["hooks"] = (
                existing.clone_and_combine(hooks) if existing is not None else hooks
            )
        output.append([tensor, updated])
    return output, json.dumps(
        {
            "schema": "t8.h3.lora_windows.v1",
            "rows": reports,
            "existing_hooks_preserved": True,
            "runtime_verified": False,
        }
    )
