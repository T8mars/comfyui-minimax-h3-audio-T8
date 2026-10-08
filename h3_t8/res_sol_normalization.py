"""RES-only, read-only projection of authenticated original Sol composition.

The executing MODEL, class constructor, hooks, delegates and caches are never
unpatched. This projection binds actual raw weights/ordered native LoRA through
the existing strict adapter; it is not a CUDA or portable recovery certificate.
"""
from copy import copy
import hashlib
import json
from pathlib import Path
from types import MethodType

from .long_video_dual_identity import _audited_stage_model_identity
from .patch_stack_policy import UnverifiedModelStack
from .res_memory_identity import mixed_memory_model_identity
from .res_sol_identity import inspect_original_sol
from .vdn_attention_compat import _factory_closure


def _container(module):
    view = copy(module)
    view._modules = dict(module._modules)
    # A native instance-bound method must follow its inspection owner. Other
    # methods are left intact for the strict identity validator to reject.
    method = vars(module).get("forward")
    if (type(method) is MethodType and method.__self__ is module
            and method.__func__ is type(module).forward):
        view.forward = MethodType(method.__func__, view)
    return view


def _project(model, contract, delegates):
    """Called only after actual source/closure/delegate authentication."""
    from . import sol_attn_minimax_v2 as sol

    if not contract["unwrapped_attention_delegates_authenticated"]:
        raise UnverifiedModelStack("RES Sol raw calculation delegates are not authenticated")
    network = model.model.diffusion_model
    expected = {f"diffusion_model.blocks.{index}.attn.forward" for index in range(len(network.blocks))}
    if set(delegates) != expected:
        raise UnverifiedModelStack("RES Sol projection requires every actual attention delegate")
    view = model.clone()
    from .res_memory_effects import KEY as memory_effect_key
    # inspect_original_sol authenticated this execution layer and bound its
    # content before returning the actual original KJ delegates.
    if "RES_memory_effects" in contract:
        view.remove_attachments(memory_effect_key)
    view.model = _container(model.model)
    apply = vars(model.model).get("apply_model")
    if (type(apply) is MethodType and apply.__self__ is model.model
            and apply.__func__ is type(model.model).apply_model):
        view.model.apply_model = MethodType(apply.__func__, view.model)
    diffusion = _container(network)
    block_list = _container(network.blocks)
    for index, block in enumerate(network.blocks):
        block_view = _container(block)
        # inspect_original_sol verified each actual hook and its ordered role.
        # Reject stray invocation flags instead of silently clearing them.
        if (set(block._forward_pre_hooks_with_kwargs) - set(block._forward_pre_hooks)
                or block._forward_hooks_with_kwargs or block._forward_hooks_always_called):
            raise UnverifiedModelStack("RES Sol hook invocation flags have an extra owner")
        for name in ("_forward_pre_hooks", "_forward_pre_hooks_with_kwargs", "_forward_hooks",
                     "_forward_hooks_with_kwargs", "_forward_hooks_always_called"):
            private = copy(getattr(block, name))
            private.clear()
            setattr(block_view, name, private)
        path = f"diffusion_model.blocks.{index}.attn.forward"
        attention = _container(block.attn)
        delegate = delegates[path]
        rebound = MethodType(delegate.__func__, attention)
        view.object_patches[path] = rebound
        current = vars(block.attn).get("forward")
        if current is not None:
            # The original Sol inspector already checked that current is either
            # its exact composer/delegate or the actual native class method.
            attention.forward = (MethodType(type(attention).forward, attention)
                                 if type(current) is MethodType
                                 and current.__func__ is type(attention).forward else rebound)
        block_view._modules["attn"] = attention
        # Keep MLP owners shared until the native FFN receipt is independently
        # authenticated by mixed_memory_model_identity. Never alter that receipt.
        block_list._modules[str(index)] = block_view
    diffusion._modules["blocks"] = block_list
    for name in ("_forward", "rope_freqs"):
        cells = _factory_closure(vars(network)[name], sol.install_h3_morton, name)
        original = cells["original_forward" if name == "_forward" else "original_rope_freqs"]
        setattr(diffusion, name, MethodType(original.__func__, diffusion))
    view.model._modules["diffusion_model"] = diffusion
    options = view.model_options["transformer_options"]
    cells = _factory_closure(options["optimized_attention_override"], sol.make_override, "override")
    previous = cells["previous"]
    for name in ("sol_compose", "sol_morton", "sol_morton_curve", "optimized_attention_override"):
        options.pop(name, None)
    if previous is not None:
        options["optimized_attention_override"] = previous
    return view


def normalized_original_sol_identity(model, *, observed_artifacts=None):
    """Return (raw weight identity, Sol content, normalization report), or None.

Unknown stacks still get their ordinary full-run path. A recognized structure
with an opaque computation is diagnostic-only, and is not projected at all.
"""
    inspected = inspect_original_sol(model, observed_artifacts=observed_artifacts)
    if inspected is None:
        return None
    contract, delegates = inspected
    report = {"schema": "t8.minimax_h3.RES_Sol_readonly_projection.v1",
              "provider_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "executing_MODEL_mutated": False, "global_constructor_mutated": False,
              "kernel_launched": False, "readonly_normalization_verified": False}
    try:
        view = _project(model, contract, delegates)
        base = mixed_memory_model_identity(view)
        if base is None:
            base = _audited_stage_model_identity(view, _omit_dormant_sampling=True)
    except UnverifiedModelStack as error:
        return None, contract, {**report, "reason": str(error)}
    report.update(readonly_normalization_verified=True, attention_paths=sorted(delegates),
                  original_Sol_content_sha256=contract["sha256"],
                  raw_tensors_shared_not_dequantized=True)
    digest = hashlib.sha256(json.dumps({"base": base, "projection": report},
                                      sort_keys=True, allow_nan=False).encode()).hexdigest()
    # The Sol computation contract is retained separately by loaded_model_identity;
    # its portable=false prevents recovery even though raw weights are now bound.
    return {**base, "schema": "t8.minimax_h3.RES_Sol_raw_weights.v1", "sha256": digest,
            "readonly_Sol_projection": report}, contract, report
