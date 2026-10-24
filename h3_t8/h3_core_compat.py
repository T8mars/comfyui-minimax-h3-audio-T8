"""Small, capability-based adapters for the native H3 Core contracts."""

from __future__ import annotations

import functools
import inspect
import types

from comfy.ldm.modules import attention as core_attention
from comfy.model_patcher import ModelPatcher


def _probe_attention_override(function):
    import torch
    sentinel = object()
    def probe(*args, **kwargs):
        return sentinel
    q = torch.zeros(1, 1, 8)
    result = function(q, q, q, 1, transformer_options={"optimized_attention_override": probe})
    return result is sentinel


@functools.lru_cache(maxsize=16)
def _cached_attention_override(function):
    return _probe_attention_override(function)


def _supports_attention_override(function):
    # Callable owners may deliberately be unhashable or expose a live signature.
    # Only immutable function identities participate in our optional cache.
    if isinstance(function, (types.FunctionType, types.BuiltinFunctionType)):
        return _cached_attention_override(function)
    return _probe_attention_override(function)


def _set_legacy_attention_backend(model, optimized_attention):
    def optimized_attention_override(_, *args, **kwargs):
        return optimized_attention(*args, **kwargs)
    container = getattr(optimized_attention, "container_function", None)
    if container is not None:
        optimized_attention_override.container_function = container
    options = model.model_options.get("transformer_options", {}).copy()
    options["optimized_attention_override"] = optimized_attention_override
    model.model_options["transformer_options"] = options


def set_h3_attention_backend(model, attention):
    """Use native setter or the older, executable-tested override protocol."""
    if not callable(attention):
        raise TypeError("attention backend must be callable")
    setter = getattr(model, "set_model_optimized_attention", None)
    if callable(setter):
        setter(attention)
    else:
        if not _supports_attention_override(core_attention.attention_pytorch):
            raise RuntimeError("This Core has no executable attention-override protocol")
        _set_legacy_attention_backend(model, attention)


def plain_attention_backend(override) -> str | None:
    """Recognize Core's unmodified backend selector, not arbitrary attention patches.

    Compare the live closure code and target identity. A copied function name or
    module string is not sufficient to exempt an algorithm-changing patch.
    """
    setters = (getattr(ModelPatcher, "set_model_optimized_attention", None), _set_legacy_attention_backend)
    expected_codes = tuple(item for setter in setters for item in getattr(getattr(setter, "__code__", None), "co_consts", ())
                           if isinstance(item, types.CodeType) and item.co_name == "optimized_attention_override")
    if not any(getattr(override, "__code__", None) is item for item in expected_codes):
        return None
    closure = dict(zip(override.__code__.co_freevars, override.__closure__ or ()))
    cell = closure.get("optimized_attention")
    if cell is None:
        return None
    try:
        target = cell.cell_contents
    except ValueError:
        return None
    if not callable(target):
        return None
    for name, attribute in (
        ("pytorch", "attention_pytorch"), ("sage", "attention_sage"),
        ("sage3", "attention3_sage"), ("flash", "attention_flash"),
        ("xformers", "attention_xformers"), ("comfy_kitchen_int8", "attention_comfy_kitchen_int8"),
        ("basic", "attention_basic"), ("split", "attention_split"),
        ("sub_quad", "attention_sub_quad"),
    ):
        if target is getattr(core_attention, attribute, None):
            # A changed container route is also an attention replacement.
            container = getattr(override, "container_function", None)
            if container is not None and container is not getattr(target, "container_function", None):
                return None
            return name
    return None


def _final_layer_schedule_parameters(forward):
    parameters = _parameters(forward)
    names = {"sigma", "sample_sigmas", "shifts"}
    if not names.intersection(parameters):
        # HyperFlow's own remap closure accepts *args/**kwargs to support both
        # Core FinalLayer ABIs. Inspect its proven delegate's interface only;
        # the actual call must still enter the wrapper and perform the remap.
        # User wrappers with similar marker names are not this source closure.
        from . import hyperflow_runtime_advanced as hyperflow
        from .vdn_attention_compat import _factory_closure

        state = _factory_closure(forward, hyperflow._install_two_time, "final_forward")
        if (state is not None and set(state) == {"owner", "final_original", "remap"}
                and set(vars(forward)) == {"_t8_hyperflow_owner", "_t8_hyperflow_inner"}
                and forward._t8_hyperflow_owner == state["owner"]
                and forward._t8_hyperflow_inner is state["final_original"]):
            parameters = _parameters(state["final_original"])
    present = names.intersection(parameters)
    if present and present != names:
        raise RuntimeError("Unsupported partial H3 FinalLayer schedule interface")
    return bool(present)


def call_h3_final_layer(layer, x, t_emb, video_seg, audio_seg, *, sigma, sample_sigmas, shifts):
    forward = getattr(layer, "forward", layer)
    function = getattr(forward, "__func__", forward)
    if _final_layer_schedule_parameters(function):
        return layer(x, t_emb, video_seg, audio_seg,
                     sigma=sigma, sample_sigmas=sample_sigmas, shifts=shifts)
    return layer(x, t_emb, video_seg, audio_seg)


@functools.lru_cache(maxsize=64)
def _cached_function_signature(function):
    return inspect.signature(function)


def _signature(function):
    if isinstance(function, (types.FunctionType, types.BuiltinFunctionType)):
        return _cached_function_signature(function)
    return inspect.signature(function)


def _parameters(function):
    return frozenset(_signature(function).parameters)


def cleanup_h3_prefetched_modules(module, comfy_modules):
    """Bind a known one/two-argument Core ABI before executing it exactly once."""
    import comfy.model_prefetch

    function = comfy.model_prefetch.cleanup_prefetched_modules
    signature = _signature(function)
    try:
        signature.bind(module, comfy_modules)
    except TypeError:
        signature.bind(comfy_modules)
        return function(comfy_modules)
    return function(module, comfy_modules)


def call_h3_block(block, args):
    kwargs = {"transformer_options": args["transformer_options"]}
    replacement = args.get("attention")
    if replacement is not None:
        forward = getattr(block.forward, "__func__", block.forward)
        if "attention" not in _parameters(forward):
            raise RuntimeError("This H3 Core does not support replacement attention in DiTBlock")
        kwargs["attention"] = replacement
    return block(args["img"], args["t_emb"], args["mod_segments"], args["rope_freqs"], **kwargs)


def prefetch_h3_block(queue, device, block):
    import comfy.model_prefetch

    function = comfy.model_prefetch.prefetch_queue_pop
    if "malloc_scope" in _parameters(function):
        return function(queue, device, block, malloc_scope="block")
    if queue is not None:
        return function(queue, device, block)
