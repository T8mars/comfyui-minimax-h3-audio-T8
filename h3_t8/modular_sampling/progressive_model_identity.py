"""Native identity plus a narrow execution-local Core cleanup normalization.

Core unpatch_model may restore a native bound forward as an instance attribute
where the module previously used its class descriptor. Those dispatches are
identical. No unknown callable, hook, class implementation or weight is omitted,
and this normalization never grants portable cache reuse.
"""
import inspect
from types import MethodType

from ..progressive_checkpoint import native_model_identity as original_identity
from ..patch_stack_policy import UnverifiedModelStack, _execution_selection


def native_model_identity(model, sampler):
    from .progressive_effect_identity import project, function_identity
    from .continuation_identity import project as project_continuation
    try:
        inspection, effect = project(model)
        inspection, motion = project_continuation(inspection)
        if motion is not None:
            effect = {"effects": effect, "continuation": motion}
        if effect is not None:
            dispatch, methods = {}, {}
            for name, module in model.model.named_modules():
                method = inspect.getattr_static(type(module), "forward")
                if method not in methods:
                    methods[method] = function_identity(method)
                dispatch[name] = methods[method]
            effect["model_class_forward"] = dispatch
    except UnverifiedModelStack:
        inspection, effect = model, None
    value = original_identity(inspection, sampler)
    if effect is not None:
        value = {"source": value, "progressive_effects": effect}

    def normalize(item):
        if not isinstance(item, dict):
            return
        selection = item.get("execution_selection")
        if item.get("portable_cache_reuse") is False and isinstance(selection, dict):
            network = selection.get("network_execution", {})
            dispatch = {}
            for name, module in model.model.named_modules():
                native = inspect.getattr_static(type(module), "forward")
                # Bind the actual class descriptor even when no instance alias
                # exists. A monkeypatched class method must still invalidate.
                dispatch[name] = _execution_selection(native)
                current = vars(module).get("forward")
                if (type(current) is MethodType and current.__self__ is module
                        and current.__func__ is native):
                    pre, post = module._forward_pre_hooks, module._forward_hooks
                    if pre or post:
                        network[name] = _execution_selection({"forward": None, "pre_hooks": pre, "hooks": post})
                    else:
                        network.pop(name, None)
            selection["native_class_forward_dispatch"] = dispatch
        for child in item.values():
            if isinstance(child, dict):
                normalize(child)
    normalize(value)
    return value
