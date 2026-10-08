"""Inspect the original, unchanged Sol node's actual RES execution owners.

This is deliberately not a portable kernel certificate. It separates verified
Python composition from the still-required comfy-kitchen dispatch/binary audit.
Inspection never installs, invokes, removes or replaces an execution owner.
"""
import ast
import hashlib
import inspect
import json
from pathlib import Path
from types import FunctionType, MethodType

import torch
from comfy.ldm.minimax import model as core_h3

from .h3_core_compat import plain_attention_backend
from .long_video_dual_identity import content_identity
from .modular_sampling.progressive_effect_identity import _code_identity
from .patch_stack_policy import UnverifiedModelStack
from .relay_kj_backend import _codes, _live_code_matches
from .vdn_attention_compat import _factory_closure


_SOL_SOURCE_SHA256 = "931c3602d7433a1dad313aebbbbe13067fdf6c1c607cdcb1b52ac173ae21e5e6"


def _require(condition, message):
    if not condition:
        raise UnverifiedModelStack(message)


def _execution(function, module, compiled):
    _require(_live_code_matches(function, compiled, module),
             "RES Sol live executable differs from its actual source")
    payload = json.dumps(_code_identity(function.__code__), sort_keys=True, allow_nan=False)
    return {"code_sha256": hashlib.sha256(payload.encode()).hexdigest(),
            "defaults": _data(function.__defaults__),
            "kwdefaults": _data(function.__kwdefaults__)}


def _data(value):
    # Sol's native dense_blocks default is an immutable set of integer indices.
    # This local adapter does not relax legacy MODEL or callable serialization.
    if type(value) is frozenset:
        _require(all(type(item) is int for item in value), "RES Sol default has noninteger depth gates")
        return {"type": "frozenset", "items": sorted(value)}
    if type(value) in (tuple, list):
        return {"type": type(value).__name__, "items": [_data(item) for item in value]}
    if type(value) is dict and all(type(key) is str for key in value):
        return {key: _data(item) for key, item in sorted(value.items())}
    return content_identity(value)


def _closure(function, factory, name, expected_keys):
    state = _factory_closure(function, factory, name)
    _require(state is not None and set(state) == set(expected_keys),
             "RES Sol callable is not its actual factory closure")
    return state


def _native_method(method, owner, name, compiled):
    _require(isinstance(method, MethodType) and method.__self__ is owner
             and method.__func__ is getattr(type(owner), name),
             "RES Sol delegated H3 method has a foreign owner")
    return _execution(method.__func__, core_h3, compiled)


def _composer(function, owner, sol):
    state = _closure(function, sol._compose_module_patch, "forward",
                     ("module", "patched_forward", "stock"))
    _require(state["module"] is owner and state["stock"] is type(owner).forward
             and vars(function) == {"_sol_composed": True},
             "RES Sol attention composer owner/stock/attributes changed")
    delegate = state["patched_forward"]
    _require(isinstance(delegate, MethodType) and delegate.__self__ is owner,
             "RES Sol attention composer requires its actual bound delegate")
    return delegate


def inspect_original_sol(model, *, observed_artifacts=None):
    """Return address-independent structural evidence, never a recovery permit.

    The raw bound delegates are returned separately for a subsequent KJ identity
    adapter. Their Python names or this structural inspection do not authenticate
    their computation. Unknown closures/hooks remain selected and unverified.
    """
    from . import sol_attn_minimax_v2 as sol

    options = model.model_options.get("transformer_options", {})
    override = options.get("optimized_attention_override")
    # No eager import/installation of another plugin, no calling a selector.
    if _factory_closure(override, sol.make_override, "override") is None:
        return None
    source_path = Path(sol.__file__).resolve()
    expected_path = Path(__file__).resolve().parents[1] / "sol_attn_minimax_v2.py"
    source = source_path.read_bytes()
    _require(source_path == expected_path and hashlib.sha256(source).hexdigest() == _SOL_SOURCE_SHA256,
             "RES original Sol source needs a fresh composition audit")
    compiled = tuple(_codes(compile(source, str(source_path), "exec", dont_inherit=True)))
    _require(sol.torch is torch and sol.inspect is inspect and sol.HEAD_DIM == 128 and sol.BLOCK_SIZE == 64,
             "RES original Sol imported execution dependencies/constants changed")
    # Authenticate every top-level local helper, not only the selected factory.
    helper_names = [node.name for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)]
    helpers = {}
    for name in helper_names:
        value = getattr(sol, name, None)
        _require(type(value) is FunctionType and value.__globals__ is vars(sol),
                 "RES Sol helper was replaced or detached: " + name)
        actual_code = [code for code in compiled if code.co_qualname == name][-1]
        helpers[name] = _execution(value, sol, (actual_code,))
    state = _closure(override, sol.make_override, "override", (
        "centroid_tail", "dense_blocks", "min_tokens", "previous", "reuse_qkv_memory",
        "routed_cap_percent", "sigma_end", "sigma_start", "sink_conditioning",
        "tau", "tau_profile", "verbose"))
    previous = state.pop("previous")
    previous_backend = None if previous is None else plain_attention_backend(previous)
    _require(previous is None or previous_backend is not None,
             "RES Sol previous selector needs its own execution identity")
    _require(type(state["dense_blocks"]) is frozenset and type(state["tau_profile"]) is dict,
             "RES Sol depth gates are not their native data containers")
    _require(all(type(key) is int and type(value) in (int, float)
                 for key, value in state["tau_profile"].items()), "RES Sol tau profile is not native data")
    settings = {**state, "dense_blocks": sorted(state["dense_blocks"]),
                "tau_profile": [[key, value] for key, value in sorted(state["tau_profile"].items())]}
    gate = {key: state[key] for key in ("sigma_start", "sigma_end", "min_tokens")}
    _require(options.get("sol_compose") == gate and type(options.get("sol_compose")) is dict,
             "RES Sol selector and forward-level gate disagree")
    network = model.model.diffusion_model
    _require(type(network) is core_h3.MiniMaxH3Model and len(network.blocks) > 0,
             "RES Sol structural identity requires the actual native H3 block tree")
    core_path = Path(core_h3.__file__).resolve()
    core_source = core_path.read_bytes()
    core_compiled = tuple(_codes(compile(core_source, str(core_path), "exec", dont_inherit=True)))
    native = {}
    for name in ("_forward", "rope_freqs"):
        closure_keys = ("model", "original_forward") if name == "_forward" else ("model", "original_rope_freqs")
        closure = _closure(vars(network).get(name), sol.install_h3_morton, name, closure_keys)
        _require(closure["model"] is network, "RES Sol H3 closure is bound to another model")
        native[name] = _native_method(closure[closure_keys[1]], network, name, core_compiled)
        _execution(vars(network)[name], sol, compiled)
    layout_init = core_h3.PackedLayout.__init__
    layout = _closure(layout_init, sol._patch_packed_layout, "__init__", ("original_init",))
    _require(layout["original_init"].__qualname__ == "PackedLayout.__init__",
             "RES Sol PackedLayout delegates to a different constructor")
    native["PackedLayout.__init__"] = _execution(layout["original_init"], core_h3, core_compiled)
    _execution(layout_init, sol, compiled)
    _require(id(network) in sol._INSTALLED and id(network) in sol._COMPOSE_HOOKED
             and id(core_h3.PackedLayout) in sol._PATCHED_LAYOUTS,
             "RES Sol installation registries do not cover its actual owners")
    for key, expected in (("_sol_morton_active", False), ("_sol_morton_span", None),
                          ("_sol_morton_state", None), ("_sol_transformer_options", None)):
        _require(vars(network).get(key, expected) is expected,
                 "RES Sol cannot bind a checkpoint during an active transient forward")
    indexed = bool(state["dense_blocks"] or state["tau_profile"])
    _require(not indexed or id(network) in sol._BLOCK_INDEX_HOOKED,
             "RES Sol depth gates lack their actual block-index hooks")
    hook_roles, delegates = [], {}
    for index, block in enumerate(network.blocks):
        roles = []
        pre_hooks = tuple(block._forward_pre_hooks.items())
        expected_roles = ["morton"] + (["index"] if indexed else []) + ["compose"]
        _require(len(pre_hooks) == len(expected_roles), "RES Sol block has extra or missing pre-hooks")
        for (hook_id, hook), role in zip(pre_hooks, expected_roles, strict=True):
            if role == "morton":
                closure = _closure(hook, sol.install_h3_morton, "pre_hook", ("first", "model"))
                _require(closure["first"] is network.blocks[0] and closure["model"] is network,
                         "RES Sol Morton hook is bound to different blocks")
            elif role == "index":
                # make_hook is nested in _install_block_index, not a public factory.
                _execution(hook, sol, compiled)
                closure = inspect.getclosurevars(hook).nonlocals
                _require(closure == {"index": index}, "RES Sol block-index hook changed")
            else:
                closure = _closure(hook, sol._install_compose_hooks, "pre_hook", ("attn_attr",))
                _require(closure == {"attn_attr": "attn"}, "RES Sol hook selects a different module")
            _require((hook_id in block._forward_pre_hooks_with_kwargs) == (role == "index"),
                     "RES Sol pre-hook invocation signature changed")
            roles.append({"role": role, "execution": _execution(hook, sol, compiled)})
        posts = tuple(block._forward_hooks.items())
        _require(len(posts) == (1 if index == len(network.blocks) - 1 else 0),
                 "RES Sol block has extra or missing post-hooks")
        for hook_id, hook in posts:
            closure = _closure(hook, sol.install_h3_morton, "post_hook", ("model",))
            _require(closure["model"] is network and hook_id not in block._forward_hooks_with_kwargs
                     and hook_id not in block._forward_hooks_always_called,
                     "RES Sol post-hook owner/invocation changed")
            roles.append({"role": "restore", "execution": _execution(hook, sol, compiled)})
        hook_roles.append(roles)
        path = f"diffusion_model.blocks.{index}.attn.forward"
        selected = model.object_patches.get(path)
        if type(selected) is FunctionType:
            delegates[path] = _composer(selected, block.attn, sol)
            _execution(selected, sol, compiled)
        current = vars(block.attn).get("forward")
        if current is not None:
            if type(current) is FunctionType:
                actual = _composer(current, block.attn, sol)
                _require(path in delegates and actual is delegates[path],
                         "RES Sol live attention delegate differs from the selected patch")
                _execution(current, sol, compiled)
            else:
                _require(isinstance(current, MethodType) and current.__self__ is block.attn
                         and (current is delegates.get(path) or current is selected
                              or current.__func__ is type(block.attn).forward),
                         "RES Sol live attention owner is foreign")
    block_ids = {id(block) for block in network.blocks}
    _require(all(id(module) in block_ids or not (module._forward_pre_hooks or module._forward_hooks)
                 for module in model.model.modules()), "RES Sol model has another shared hook owner")
    from .res_kitchen_identity import inspect_sol_kitchen
    from .res_sol_cache_identity import inspect_sol_caches
    from .res_kj_sage_identity import inspect_raw_kj_sage
    kitchen = inspect_sol_kitchen(sol._ck)
    caches = inspect_sol_caches(sol)
    # The ordinary Sol composition remains available even if the raw delegate
    # cannot receive a calculation certificate. Never execute or strip it.
    raw_functions, raw_paths, inspected = {}, {}, {}
    memory_effect = None
    from .res_memory_effects import raw_delegate, KEY as memory_effect_key
    if model.get_attachment(memory_effect_key) is not None:
        for path, delegate in list(delegates.items()):
            delegates[path], memory_effect = raw_delegate(model, path, delegate)
    for path, delegate in delegates.items():
        if delegate.__func__ not in inspected:
            try:
                evidence = inspect_raw_kj_sage(delegate, delegate.__self__, observed_artifacts=observed_artifacts)
            except UnverifiedModelStack as error:
                evidence = {"calculation_content_verified": False, "reason": str(error),
                            "CUDA_execution_qualified": False, "portable_cache_reuse": False}
            digest = hashlib.sha256(json.dumps(evidence, sort_keys=True, allow_nan=False).encode()).hexdigest()
            inspected[delegate.__func__] = digest
            raw_functions[digest] = evidence
        raw_paths[path] = inspected[delegate.__func__]
    raw_verified = bool(raw_paths) and all(item["calculation_content_verified"] for item in raw_functions.values())
    contract = {"schema": "t8.minimax_h3.RES_original_Sol_structure.v1",
                "source_sha256": _SOL_SOURCE_SHA256,
                "core_source_sha256": hashlib.sha256(core_source).hexdigest(),
                "provider_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "helpers": helpers, "selector": _execution(override, sol, compiled),
                "settings": settings, "previous_backend": previous_backend,
                "native_delegates": native, "ordered_block_hooks": hook_roles,
                "composed_attention_paths": sorted(delegates),
                "morton": content_identity({key: options.get(key) for key in ("sol_morton", "sol_morton_curve")}),
                "kitchen_execution_chain": kitchen, "actual_cache_integrity": caches,
                "raw_KJ_calculation_functions": raw_functions, "raw_KJ_calculation_paths": raw_paths,
                "python_composition_verified": True,
                "unwrapped_attention_delegates_authenticated": raw_verified,
                "kernel_dispatch_binary_and_layout_cache_verified": False,
                "portable_cache_reuse": False}
    contract["sha256"] = hashlib.sha256(json.dumps(contract, sort_keys=True, allow_nan=False).encode()).hexdigest()
    if memory_effect is not None:
        contract["RES_memory_effects"] = memory_effect
        contract.pop("sha256", None)
        contract["sha256"] = hashlib.sha256(json.dumps(contract, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return contract, delegates
