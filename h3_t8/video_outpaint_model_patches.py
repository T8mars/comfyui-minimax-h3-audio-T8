"""Identify the two pinned KJ memory patches without installing or copying them.

This is an execution/cache compatibility adapter, not an acceleration or visual
acceptance claim. Unknown compositions still need a separately tested adapter.
"""
from __future__ import annotations

import hashlib
import ast
import inspect
from pathlib import Path
import sys
from types import CodeType, FunctionType, MethodType


from .video_outpaint_kj_contract import KJ_SOURCE_SHA256 as KJ_SOURCE_SHA256, kj_source_contract
from .source_code_identity import executable_code_equal


def _codes(code):
    yield code
    for item in code.co_consts:
        if isinstance(item, CodeType):
            yield from _codes(item)


def _matches_default(value, node):
    # The authenticated revisions use only None and empty dict defaults here.
    # Do not call equality on arbitrary live objects or execute source defaults.
    if isinstance(node, ast.Constant) and node.value is None:
        return value is None
    if isinstance(node, ast.Dict) and not node.keys:
        return type(value) is dict and not value
    raise ValueError("KJ memory source default requires a separate compatibility audit")


def _check_defaults(fn, definition):
    positional = definition.args.defaults
    live = fn.__defaults__
    valid = (live is None if not positional else
             type(live) is tuple and len(live) == len(positional)
             and all(_matches_default(value, node) for value, node in zip(live, positional)))
    keyword = {arg.arg: value for arg, value in zip(definition.args.kwonlyargs, definition.args.kw_defaults)
               if value is not None}
    live_keyword = fn.__kwdefaults__
    valid_keyword = (live_keyword is None if not keyword else
                     type(live_keyword) is dict and set(live_keyword) == set(keyword)
                     and all(_matches_default(live_keyword[name], node) for name, node in keyword.items()))
    if not valid or not valid_keyword:
        raise ValueError(f"KJ memory live defaults differ from pinned source: {definition.name}")


def _verified_module(function):
    if not isinstance(function, FunctionType):
        raise ValueError("KJ memory adapter expected an actual Python function")
    module = sys.modules.get(function.__module__)
    if module is None or function.__globals__ is not vars(module):
        raise ValueError("KJ memory function is detached from its loaded module")
    source = inspect.getsourcefile(function)
    if source is None:
        raise ValueError("KJ memory implementation source is unavailable")
    payload = Path(source).read_bytes()
    source_contract = kj_source_contract(payload)
    # Compile only; never execute code obtained from a model argument. Comparing
    # code objects also rejects a live monkey patch with a misleading filename.
    compiled = list(_codes(compile(payload, source, "exec", dont_inherit=True)))
    definitions = [node for node in ast.walk(ast.parse(payload, filename=source))
                   if isinstance(node, ast.FunctionDef)]

    def check(fn, name):
        matches = [code for code in compiled if code.co_name == name]
        if (not isinstance(fn, FunctionType) or fn.__globals__ is not vars(module)
                or len(matches) != 1 or not executable_code_equal(fn.__code__, matches[0])):
            raise ValueError(f"KJ memory live function differs from pinned source: {name}")
        declarations = [node for node in definitions if node.name == name]
        if len(declarations) != 1:
            raise ValueError(f"KJ memory source default declaration is ambiguous: {name}")
        _check_defaults(fn, declarations[0])

    for name in ("minimax_attn_lowmem_forward", "minimax_block_lowmem_forward", "minimax_mlp_chunked_forward"):
        check(getattr(module, name, None), name)
    from comfy.ldm.minimax import model as native
    from comfy.ldm.modules import attention
    if (module.optimized_attention is not attention.optimized_attention
            or module._mod_gate is not native._mod_gate or module._mod_scale_shift is not native._mod_scale_shift):
        raise ValueError("KJ memory backend globals were replaced")
    return module, check, source_contract


def inspect_outpaint_model_patches_advisory(model, *, regional_contract=None):
    try:
        return inspect_outpaint_model_patches(model, regional_contract=regional_contract)
    except (ValueError, TypeError, RuntimeError) as error:
        from .patch_stack_policy import warn_patch_stack
        warn_patch_stack(f"Outpaint retains unverified MODEL patch stack: {error}")
        return {"kind": "user_selected_unverified", "portable_cache_reuse": False}, {}


def inspect_outpaint_model_patches(model, *, regional_contract=None):
    """Return stable composition metadata and allowed instance-forward bindings."""
    patches = getattr(model, "object_patches", {})
    options = getattr(model, "model_options", None)
    regional_keys = {"optimized_attention_override"} if regional_contract is not None else set()
    if not patches:
        allowed_options = ({}, {"transformer_options": {}}) if not regional_keys else ()
        if not regional_keys and options not in allowed_options:
            raise ValueError("stock outpaint identity requires unmodified model_options")
        if regional_keys and (not isinstance(options, dict) or set(options) != {"transformer_options"}
                              or not isinstance(options["transformer_options"], dict)
                              or set(options["transformer_options"]) != regional_keys):
            raise ValueError("regional outpaint identity requires its exact attention option")
        return {"kind": "bare_native"}, {}
    blocks = getattr(model.model.diffusion_model, "blocks", None)
    if not blocks or not isinstance(options, dict) or set(options) != {"transformer_options"}:
        raise ValueError("outpaint memory adapter requires native blocks and only transformer_options")
    transformer = options["transformer_options"]
    allowed_transformer = {"minimax_head_chunks", "sol_take_forward"} | regional_keys
    if not isinstance(transformer, dict) or set(transformer) - allowed_transformer:
        raise ValueError("outpaint memory adapter does not cover these transformer options")
    if regional_keys and not regional_keys <= set(transformer):
        raise ValueError("regional outpaint attention option is missing")
    attention_enabled = "diffusion_model.blocks.0.attn.forward" in patches
    ffn_enabled = "diffusion_model.blocks.0.mlp.forward" in patches
    expected = {}
    for index, block in enumerate(blocks):
        prefix = f"diffusion_model.blocks.{index}"
        if attention_enabled:
            expected[prefix+".forward"] = (block, "minimax_block_lowmem_forward")
            expected[prefix+".attn.forward"] = (block.attn, "minimax_attn_lowmem_forward")
        if ffn_enabled:
            expected[prefix+".mlp.forward"] = (block.mlp, "wrapped_forward")
    if not expected or set(patches) != set(expected):
        raise ValueError("outpaint memory adapter requires complete KJ patch sets, without extra object patches")
    first = next(iter(patches.values()))
    if not isinstance(first, MethodType):
        raise ValueError("KJ memory patch must be bound to its native model module")
    module, check, source_contract = _verified_module(first.__func__)
    ffn_config = None
    allowed = {}
    for path, (owner, name) in expected.items():
        method = patches[path]
        if not isinstance(method, MethodType) or method.__self__ is not owner:
            raise ValueError("KJ memory patch is bound to a different model module")
        check(method.__func__, name)
        if name == "wrapped_forward":
            closure = inspect.getclosurevars(method.__func__)
            patch = closure.nonlocals.get("self")
            if (set(closure.nonlocals) != {"self"} or type(patch) is not module.MiniMaxFFNChunkPatch
                    or set(vars(patch)) != {"num_chunks", "seq_threshold"}):
                raise ValueError("KJ FFN closure is not the audited chunk configuration")
            config = (patch.num_chunks, patch.seq_threshold)
            if (any(type(v) is not int for v in config) or not 2 <= config[0] <= 64
                    or not 256 <= config[1] <= 262144 or config[1] % 256):
                raise ValueError("KJ FFN chunk configuration is outside the audited node range")
            if ffn_config is not None and config != ffn_config:
                raise ValueError("KJ FFN chunk settings differ between model blocks")
            ffn_config = config
        allowed[path[:-len(".forward")]] = method
    heads = transformer.get("minimax_head_chunks", 1)
    if attention_enabled:
        if (type(heads) is not int or not 1 <= heads <= 56
                or transformer.get("sol_take_forward") is not module.minimax_attn_lowmem_forward):
            raise ValueError("KJ low-memory attention options do not match its verified forward")
    elif set(transformer) - regional_keys:
        raise ValueError("KJ FFN-only model has unexpected attention options")
    return {"kind": "pinned_kj_memory", "source_sha256": source_contract["source_sha256"],
            "attention_head_chunks": heads if attention_enabled else None,
            "ffn_chunks_threshold": list(ffn_config) if ffn_config else None,
            "patch_count": len(expected),
            "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, allowed


def verify_instance_forward(module, name, allowed):
    """Allow known applied patches and Comfy's restored bound native forward."""
    if module._forward_hooks or module._forward_pre_hooks:
        raise ValueError("stock outpaint identity does not cover runtime forward replacement/hooks")
    if "forward" not in vars(module):
        return
    current = vars(module)["forward"]
    expected = allowed.get(name)
    if expected is not None and isinstance(current, MethodType) and current.__self__ is module:
        if current.__func__ is expected.__func__ or current.__func__ is type(module).forward:
            return
    raise ValueError("stock outpaint identity does not cover runtime forward replacement/hooks")
