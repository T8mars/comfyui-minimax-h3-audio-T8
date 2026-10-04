"""Read-only identity of stock Core inference LoRA bypass lifecycle.

Only a private module-tree view is projected. Never inject/eject, modify live
forwards, move source tensors, or erase an unknown adapter/delegate.
"""
from copy import copy
from functools import lru_cache
import hashlib
import math
from pathlib import Path
from types import CodeType, FunctionType, MethodType

import torch
from comfy.patcher_extension import PatcherInjection
from comfy.weight_adapter.bypass import BypassInjectionManager, BypassForwardHook, get_module_type_info
from comfy.weight_adapter.lora import LoRAAdapter
from comfy.weight_adapter import base as core_base, bypass as core_bypass, lora as core_lora

from ..long_video_dual_identity import content_identity, _implementation, _sha256_json
from ..patch_stack_policy import UnverifiedModelStack
from ..source_code_identity import executable_code_equal
from ..vdn_attention_compat import _factory_closure

KEY = "bypass_lora"
_SOURCE_PINS = {
    "comfy.weight_adapter.bypass": "b60abb835679840be070b1f3ec7d45f8c26fd53e0349c2d084f92a4b971f12fd",
    "comfy.weight_adapter.base": "a5083daaae27a0c0146bdbbaa1b5d06376b01b582fa38cb52eba4c8b10214d66",
    "comfy.weight_adapter.lora": "5e30c8b8a22be6459883cb5d758fa76f725cc4688d4048200145b885567c4cec",
}


def _require(condition, message):
    if not condition:
        raise UnverifiedModelStack("Core bypass LoRA " + message)


@lru_cache(maxsize=3)
def _compiled_codes(source, filename):
    codes = {}
    def visit(code):
        codes[code.co_qualname] = code
        for item in code.co_consts:
            if type(item) is CodeType:
                visit(item)
    visit(compile(source, filename, "exec", dont_inherit=True))
    return codes


def _authenticate_core():
    """Authenticate live methods too, not just their class names/source files.

    An unrecognized Core/plugin override remains a nonportable delegate through
    the callers' existing advisory path. It is never executed by this inspector.
    """
    methods = (
        (core_bypass, "get_module_type_info", get_module_type_info, None),
        (core_bypass, "BypassInjectionManager.create_injections", BypassInjectionManager.create_injections, None),
        (core_bypass, "BypassInjectionManager._get_module_by_key", BypassInjectionManager._get_module_by_key, None),
        (core_bypass, "BypassForwardHook.__init__", BypassForwardHook.__init__, (1.0,)),
        (core_bypass, "BypassForwardHook.inject", BypassForwardHook.inject, None),
        (core_bypass, "BypassForwardHook.eject", BypassForwardHook.eject, None),
        (core_bypass, "BypassForwardHook._move_adapter_weights_to_device", BypassForwardHook._move_adapter_weights_to_device, (None,)),
        (core_bypass, "BypassForwardHook._bypass_forward", BypassForwardHook._bypass_forward, None),
        (core_lora, "LoRAAdapter.h", LoRAAdapter.h, None),
        (core_base, "WeightAdapterBase.g", LoRAAdapter.g, None),
        (core_base, "WeightAdapterBase.bypass_forward", LoRAAdapter.bypass_forward, None),
    )
    _require(core_bypass.get_module_type_info is get_module_type_info
             and core_bypass.BypassInjectionManager is BypassInjectionManager
             and core_bypass.BypassForwardHook is BypassForwardHook
             and core_lora.LoRAAdapter is LoRAAdapter
             and LoRAAdapter.g is core_base.WeightAdapterBase.g
             and LoRAAdapter.bypass_forward is core_base.WeightAdapterBase.bypass_forward,
             "live class/function binding changed")
    compiled = {}
    for module, name, function, defaults in methods:
        _require(type(function) is FunctionType and function.__globals__ is vars(module)
                 and function.__closure__ is None and function.__defaults__ == defaults
                 and function.__kwdefaults__ is None, "live method ownership/defaults changed")
        path = Path(module.__file__).resolve()
        _require(Path(function.__code__.co_filename).resolve() == path, "live method source owner changed")
        if module.__name__ not in compiled:
            source = path.read_bytes()
            _require(hashlib.sha256(source).hexdigest() == _SOURCE_PINS[module.__name__],
                     "source is outside the audited native lifecycle")
            compiled[module.__name__] = _compiled_codes(source, function.__code__.co_filename)
        _require(executable_code_equal(function.__code__, compiled[module.__name__].get(name)),
                 "live method differs from the audited executable source")


def _inspect(model, *, allow_shared_owner=True):
    _authenticate_core()
    values = model.get_injections(KEY)
    _require(type(values) is list and len(values) == 1, "needs the original single manager injection")
    injection = values[0]
    _require(type(injection) is PatcherInjection and set(vars(injection)) == {"inject", "eject"},
             "has unknown injection state")
    states = []
    for function, name in ((injection.inject, "inject_all"), (injection.eject, "eject_all")):
        state = _factory_closure(function, BypassInjectionManager.create_injections, name)
        _require(state is not None and set(state) == {"self"} and function.__defaults__ is None
                 and function.__kwdefaults__ is None, "inject/eject source or defaults changed")
        states.append(state)
    manager = states[0]["self"]
    _require(states[1]["self"] is manager and type(manager) is BypassInjectionManager
             and set(vars(manager)) == {"adapters", "hooks"}
             and type(manager.adapters) is dict and type(manager.hooks) is list
             and len(manager.adapters) == len(manager.hooks) > 0, "manager ownership/targets changed")
    records, restores = [], {}
    shared_original_forwards = None
    for (path, pair), hook in zip(manager.adapters.items(), manager.hooks, strict=True):
        _require(type(path) is str and type(pair) is tuple and len(pair) == 2, "target descriptor changed")
        adapter, strength = pair
        _require(type(strength) in (int, float) and math.isfinite(strength), "strength is not finite")
        _require(type(hook) is BypassForwardHook and set(vars(hook)) == {
            "module", "adapter", "multiplier", "original_forward"}, "hook has foreign executable state")
        _require(type(adapter) is LoRAAdapter and hook.adapter is adapter and hook.multiplier == strength
                 and set(vars(adapter)) == {"loaded_keys", "weights", "multiplier", "is_conv", "conv_dim",
                    "kernel_size", "in_channels", "out_channels", "kw_dict"},
                 "adapter owner or executable state changed")
        module = manager._get_module_by_key(model.model, path)
        _require(module is not None and module is hook.module, "hook belongs to another module")
        info = get_module_type_info(module)
        expected = {key: info[key] for key in ("is_conv", "conv_dim", "kernel_size", "in_channels", "out_channels")}
        expected.update(multiplier=strength, kw_dict={key: info[key] for key in
            ("stride", "padding", "dilation", "groups")} if info["is_conv"] else {})
        _require(all(getattr(adapter, key) == value for key, value in expected.items()), "layer geometry changed")
        _require(type(adapter.loaded_keys) is set and all(type(key) is str for key in adapter.loaded_keys)
                 and type(adapter.weights) is tuple and len(adapter.weights) == 6, "weight descriptor changed")
        b, a, _, mid, dora, reshape = adapter.weights
        _require(isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)
                 and a.ndim == b.ndim == 2 and b.shape[1] == a.shape[0]
                 and tuple(module.weight.shape) == (b.shape[0], a.shape[1])
                 and mid is None and dora is None and reshape is None,
                 "shape or adapter family is not the audited linear H3 LoRA")
        dtype = module.weight.dtype
        # This is exactly Core's derived inject cast, on private operands only.
        weights = tuple(value.to(dtype=dtype) if isinstance(value, torch.Tensor)
                        and dtype in (torch.float32, torch.float16, torch.bfloat16) else value
                        for value in adapter.weights)
        records.append({"path": path, "loaded_keys": sorted(adapter.loaded_keys),
                        "weights": content_identity(weights), "geometry": content_identity(expected)})
        previous = hook.original_forward
        if previous is None:
            previous = module.forward
            _require(not (isinstance(previous, MethodType) and previous.__self__ is hook),
                     "live hook lacks its original forward")
            if (allow_shared_owner and isinstance(previous, MethodType)
                    and type(previous.__self__) is BypassForwardHook
                    and previous.__func__ is BypassForwardHook._bypass_forward):
                # A fresh clone's hook has not been injected yet. Core will
                # detach its loaded sibling before switching clones. Project
                # only that authenticated outgoing owner on a private view;
                # do not eject it or erase an unknown preexisting delegate.
                if shared_original_forwards is None:
                    from comfy.model_management import current_loaded_models
                    outgoing = [item.model for item in current_loaded_models
                                if item.model is not None and item.model is not model
                                and getattr(item.model.model, "diffusion_model", None) is model.model.diffusion_model
                                and item.model.backup is model.backup
                                and item.model.is_injected
                                and item.model.patches_uuid == model.model.current_weight_patches_uuid]
                    _require(len(outgoing) == 1, "shared live hook has no unique loaded Core owner")
                    _, shared_original_forwards = _inspect(outgoing[0], allow_shared_owner=False)
                _require(path in shared_original_forwards and shared_original_forwards[path][0] is module
                         and previous.__self__.module is module,
                         "shared live hook target differs from its loaded owner")
                previous = shared_original_forwards[path][1]
        else:
            _require(isinstance(module.forward, MethodType) and module.forward.__self__ is hook
                     and module.forward.__func__ is BypassForwardHook._bypass_forward,
                     "live forward is outside the original injection lifecycle")
        restores[path] = (module, previous)
    # Bind every ordered target, strength, geometry and complete tensor digest,
    # but do not recursively expand hundreds of records into Relay/Stage JSON.
    # Hashing is lossless identity binding, not dropping weights or checking a
    # subset; the full inspection above still runs before and after sampling.
    return {"schema": "t8.core-inference-bypass-lora.v2", "adapter_count": len(records),
            "adapters_sha256": _sha256_json(records),
            "target_paths_sha256": _sha256_json([record["path"] for record in records]),
            "implementation": {name: _implementation(value) for name, value in (
                ("manager", BypassInjectionManager), ("hook", BypassForwardHook), ("adapter", LoRAAdapter),
                ("source_code_comparison", executable_code_equal))}}, restores


def project(model):
    contract, restores = _inspect(model)
    view = model.clone()
    view.model = copy(model.model)
    view.model._modules = dict(model.model._modules)
    cloned = {"": view.model}
    for path, (original, forward) in restores.items():
        parent = ""
        for part in path.split("."):
            current = (parent + "." + part).strip(".")
            if current not in cloned:
                child = copy(cloned[parent]._modules[part])
                child._modules = dict(child._modules)
                cloned[parent]._modules[part] = child
                cloned[current] = child
            parent = current
        target = cloned[path]
        if isinstance(forward, MethodType) and forward.__self__ is original and forward.__func__ is type(original).forward:
            target.__dict__.pop("forward", None)
        else:
            # Unknown preexisting forwards remain visible to the base inspector.
            target.forward = forward
    view.injections = dict(view.injections)
    view.injections.pop(KEY)
    return view, contract
