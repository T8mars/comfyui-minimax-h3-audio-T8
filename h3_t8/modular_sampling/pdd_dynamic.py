"""Read-only identity of the original PDD bypass/head lifecycle.

Authenticate actual source closures, adapters and module ownership. Project
only a private module-tree view: never eject, unload or edit the live MODEL.
"""
from copy import copy
from types import MethodType

import torch
from torch import nn
from comfy.patcher_extension import PatcherInjection
from comfy.weight_adapter.bypass import BypassInjectionManager, BypassForwardHook, get_module_type_info
from comfy.weight_adapter.lora import LoRAAdapter

from .. import pdd_advanced as pdd
from ..long_video_dual_identity import content_identity
from ..patch_stack_policy import UnverifiedModelStack
from ..vdn_attention_compat import _factory_closure


def require(condition, message):
    if not condition:
        raise UnverifiedModelStack("PDD dynamic " + message)


def closure(function, factory, name, keys=None):
    state = _factory_closure(function, factory, name)
    require(state is not None and function.__defaults__ is None and function.__kwdefaults__ is None,
            "closure source/defaults changed: " + name)
    if keys is not None:
        require(set(state) == set(keys), "closure has unknown state: " + name)
    return state


def injection(value, factory, inject_name="inject", eject_name="eject", keys=None):
    require(type(value) is PatcherInjection and set(vars(value)) == {"inject", "eject"},
            "injection has another executable owner")
    a = closure(value.inject, factory, inject_name, keys)
    b = closure(value.eject, factory, eject_name, keys)
    for key in set(a) & set(b):
        require(a[key] is b[key], "inject/eject owners differ: " + key)
    return a, b


def _inspect(model):
    values = model.get_injections(pdd.PDD_INJECTION_KEY)
    require(isinstance(values, list) and len(values) == 1, "needs its original single transactional injection")
    runtime, _ = injection(values[0], pdd._create_pdd_runtime_injection,
                           keys={"backbone_injection", "final_injection"})
    final_state, final_eject = injection(runtime["final_injection"], pdd._create_pdd_final_layer_injection)
    require(set(final_state) == {"base_final", "pdd_final", "owner_attribute", "owner_token", "state", "offload"}
            and set(final_eject) == {"base_final", "owner_attribute", "owner_token", "state", "offload"},
            "final-head closure shape changed")
    head = model.model_options.get("transformer_options", {}).get("minimax_h3_pdd_final")
    require(type(head) is pdd.PDDHeadFinalLayer and final_state["pdd_final"] is head,
            "head module differs from injection")
    require(set(vars(head)) == set(vars(nn.Module())) | {"strength", "variant", "_block_index", "_selection_count"}
            and set(head._modules) == {"base"}
            and set(head._buffers) == {"video_weight", "video_bias", "audio_weight", "audio_bias"}
            and not head._parameters and not head._forward_hooks and not head._forward_pre_hooks,
            "head module has unknown executable state")
    require(type(head._block_index) is int and 0 <= head._block_index < 8
            and type(head._selection_count) is int and head._selection_count >= 0,
            "head telemetry is corrupt")
    final = model.get_model_object("diffusion_model.final_layer")
    require(final is head.base and final_state["base_final"] is final, "head is bound to another MODEL")
    offload = closure(final_state["offload"], pdd._create_pdd_final_layer_injection, "offload", {"pdd_final"})
    require(offload["pdd_final"] is head, "head offload owner differs")
    require(final_state["owner_attribute"] == "_t8_pdd_final_layer_injection_owner"
            and type(final_state["owner_token"]) is object, "final owner marker changed")
    lifecycle = final_state["state"]
    require(type(lifecycle) is dict and set(lifecycle) == {"original_forward"}, "final lifecycle state changed")
    marker = getattr(final, final_state["owner_attribute"], None)
    original = lifecycle["original_forward"]
    if marker is None:
        require(original is None, "final owner marker disappeared")
        original = final.forward
    else:
        require(marker is final_state["owner_token"] and original is not None
                and isinstance(final.forward, MethodType) and final.forward.__self__ is head
                and final.forward.__func__ is pdd.PDDHeadFinalLayer.forward,
                "live final forward is outside its injection lifecycle")
    restores = {"diffusion_model.final_layer": (final, original)}

    backbone, _ = injection(runtime["backbone_injection"], pdd._create_offloading_bypass_injections,
                            keys={"native_injection", "offload"})
    native, _ = injection(backbone["native_injection"], BypassInjectionManager.create_injections,
                          "inject_all", "eject_all", {"self"})
    manager = native["self"]
    require(type(manager) is BypassInjectionManager and set(vars(manager)) == {"adapters", "hooks"}
            and type(manager.adapters) is dict and type(manager.hooks) is list,
            "backbone manager has unknown state")
    offload = closure(backbone["offload"], pdd._create_offloading_bypass_injections, "offload", {"adapters"})
    require(type(offload["adapters"]) is tuple and len(offload["adapters"]) == len(manager.adapters)
            and len(manager.hooks) == len(manager.adapters), "backbone offload/hook count differs")
    records = []
    for index, ((path, pair), hook) in enumerate(zip(manager.adapters.items(), manager.hooks)):
        require(type(path) is str and type(pair) is tuple and len(pair) == 2, "backbone target descriptor changed")
        adapter, strength = pair
        require(type(hook) is BypassForwardHook and set(vars(hook)) == {
            "module", "adapter", "multiplier", "original_forward"}, "backbone hook has unknown state")
        require(type(adapter) is LoRAAdapter and offload["adapters"][index] is adapter
                and hook.adapter is adapter and hook.multiplier == strength, "backbone adapter order/owner differs")
        require(set(vars(adapter)) == {"loaded_keys", "weights", "multiplier", "is_conv", "conv_dim",
            "kernel_size", "in_channels", "out_channels", "kw_dict"}, "LoRA has unknown executable members")
        module = manager._get_module_by_key(model.model, path)
        require(module is hook.module and module is not None, "backbone hook belongs to another module")
        info = get_module_type_info(module)
        expected = {key: info[key] for key in ("is_conv", "conv_dim", "kernel_size", "in_channels", "out_channels")}
        expected.update(multiplier=strength, kw_dict={key: info[key] for key in
            ("stride", "padding", "dilation", "groups")} if info["is_conv"] else {})
        require(all(getattr(adapter, key) == value for key, value in expected.items()),
                "LoRA actual layer geometry/multiplier changed")
        require(type(adapter.loaded_keys) is set and all(type(key) is str for key in adapter.loaded_keys)
                and type(adapter.weights) is tuple and len(adapter.weights) == 6, "LoRA weights descriptor changed")
        # Core inject casts inference adapters to the module's floating storage
        # dtype. Hash effective operands before/after this derived cast; do not
        # mutate tensors or dequantize any model weight for the inspection.
        dtype = getattr(module.weight, "dtype", None)
        weights = tuple(value.to(dtype=dtype) if isinstance(value, torch.Tensor)
                        and dtype in (torch.float32, torch.float16, torch.bfloat16) else value for value in adapter.weights)
        records.append({"path": path, "loaded_keys": sorted(adapter.loaded_keys), "weights": content_identity(weights),
                        "geometry": content_identity(expected)})
        previous = hook.original_forward
        if previous is None:
            previous = module.forward
        else:
            require(isinstance(module.forward, MethodType) and module.forward.__self__ is hook
                    and module.forward.__func__ is BypassForwardHook._bypass_forward,
                    "live backbone forward changed during injection")
        restores[path] = (module, previous)
    wrappers = model.get_wrappers("diffusion_model", pdd.PDD_WRAPPER_KEY)
    require(wrappers == [pdd.pdd_forward_wrapper], "absolute head selector source/order changed")
    contract = {"schema": "t8.modular-sampling.pdd-dynamic.v1", "backbone": records,
                "head": content_identity({name: getattr(head, name) for name in
                    ("video_weight", "video_bias", "audio_weight", "audio_bias", "strength", "variant")})}
    return contract, restores


def describe(model):
    return _inspect(model)[0]


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
            # Foreign preexisting forward stays visible and unqualified; do not
            # replace it with native math to manufacture a portable identity.
            target.forward = forward
        target.__dict__.pop("_t8_pdd_final_layer_injection_owner", None)
    view.injections = dict(view.injections)
    view.injections.pop(pdd.PDD_INJECTION_KEY)
    view.remove_wrappers_with_key("diffusion_model", pdd.PDD_WRAPPER_KEY)
    view.model_options["transformer_options"].pop("minimax_h3_pdd_final")
    return view, contract
