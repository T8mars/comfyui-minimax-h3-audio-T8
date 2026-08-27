from __future__ import annotations

import ast
import copy
import functools
import inspect
import json
import textwrap
import types
from typing import Any

import torch

import comfy.ldm.minimax.vae as minimax_vae


TILED_VAE_ATTACHMENT_ATTR = "_t8_minimax_h3_tiled_coordinate_contract"
REAL_VALIDATION_WARNING = (
    "Current fp16 H3 VAE real validation produced stronger grid/stripe artifacts "
    "with the upstream global-coordinate candidate. Keep report_only unless you "
    "are deliberately reproducing the experiment."
)


def create_token_ids_global(
    patch_dims,
    device,
    dtype,
    full_dims=None,
    offset=None,
):
    if full_dims is None:
        full_dims = patch_dims
    if offset is None:
        offset = (0,) * len(patch_dims)
    if not (len(patch_dims) == len(full_dims) == len(offset)):
        raise ValueError("patch_dims, full_dims, and offset must have the same rank")
    coords_list = []
    for dim_size, full_size, axis_offset in zip(patch_dims, full_dims, offset):
        if int(dim_size) <= 0 or int(full_size) <= 0:
            raise ValueError("token-grid dimensions must be positive")
        if int(axis_offset) < 0 or int(axis_offset) + int(dim_size) > int(full_size):
            raise ValueError("tile offset must stay inside the full latent grid")
        coords = (
            torch.arange(0.5, dim_size, dtype=dtype, device=device) + axis_offset
        )
        coords = coords / full_size
        coords = 2.0 * coords - 1.0
        coords_list.append(coords)
    coords = torch.stack(torch.meshgrid(*coords_list, indexing="ij"), dim=-1)
    return coords.flatten(0, len(patch_dims) - 1).unsqueeze(0)


def _source_of(callable_object) -> str:
    try:
        return textwrap.dedent(inspect.getsource(callable_object))
    except (OSError, TypeError):
        return ""


def _has_parameters(callable_object, names: tuple[str, ...]) -> bool:
    try:
        parameters = inspect.signature(callable_object).parameters
    except (TypeError, ValueError):
        return False
    return all(name in parameters for name in names)


def probe_native_tiled_coordinates(video_vae) -> dict[str, Any]:
    first_stage = getattr(video_vae, "first_stage_model", None)
    decoder = getattr(first_stage, "decoder", None)
    compat_marker = getattr(video_vae, TILED_VAE_ATTACHMENT_ATTR, None)
    if isinstance(compat_marker, dict):
        return dict(compat_marker)
    token_ids = _has_parameters(
        getattr(minimax_vae, "create_token_ids", None), ("full_dims", "offset")
    )
    decoder_forward = _has_parameters(
        getattr(type(decoder), "forward", None), ("full_dims", "offset")
    )
    decode_pixels = _has_parameters(
        getattr(type(first_stage), "_decode_pixels", None), ("full_dims", "offset")
    )
    tiled_source = _source_of(getattr(type(first_stage), "tiled_decode", None))
    tiled_offsets = "full_dims=full_dims" in tiled_source and "offset=(0, zi, zj)" in tiled_source
    return {
        "available": bool(
            token_ids and decoder_forward and decode_pixels and tiled_offsets
        ),
        "token_ids_global_coordinates": bool(token_ids),
        "decoder_accepts_full_grid": bool(decoder_forward),
        "decode_pixels_forwards_full_grid": bool(decode_pixels),
        "tiled_decode_supplies_offsets": bool(tiled_offsets),
        "policy": "semantic_signature_and_source_probe_no_version_or_hash_gate",
    }


def _compile_function(
    tree: ast.Module,
    original,
    helper_globals=None,
    *,
    extra_defaults: tuple[Any, ...] = (),
):
    ast.fix_missing_locations(tree)
    source = ast.unparse(tree)
    namespace = dict(getattr(original, "__globals__", {}))
    namespace.update(helper_globals or {})
    module_code = compile(source, "<t8_minimax_h3_tiled_coordinates>", "exec")
    candidates = [
        value
        for value in module_code.co_consts
        if isinstance(value, types.CodeType) and value.co_name == original.__name__
    ]
    if len(candidates) != 1 or candidates[0].co_freevars:
        raise RuntimeError(
            "MiniMax H3 rewritten VAE function could not be isolated safely; "
            "no patch was applied."
        )
    defaults = tuple(getattr(original, "__defaults__", None) or ()) + tuple(
        extra_defaults
    )
    replacement = types.FunctionType(
        candidates[0],
        namespace,
        original.__name__,
        defaults or None,
    )
    replacement.__kwdefaults__ = getattr(original, "__kwdefaults__", None)
    functools.update_wrapper(replacement, original)
    replacement.__t8_global_coordinates__ = True
    return replacement, source


def _compile_decoder_forward(callable_object):
    original = callable_object.__func__ if inspect.ismethod(callable_object) else callable_object
    source = _source_of(original)
    if not source:
        raise RuntimeError("MiniMax H3 decoder source is unavailable; no patch was applied.")
    tree = ast.parse(source)
    function = tree.body[0]
    if not isinstance(function, ast.FunctionDef):
        raise RuntimeError("MiniMax H3 decoder forward is not a normal Python function.")
    if any(arg.arg in ("full_dims", "offset") for arg in function.args.args):
        return original, source
    function.args.args.extend([ast.arg(arg="full_dims"), ast.arg(arg="offset")])
    function.args.defaults.extend([ast.Constant(None), ast.Constant(None)])

    call_count = 0
    for node in ast.walk(function):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "create_token_ids":
            node.func.id = "_t8_create_token_ids_global"
            node.keywords.extend(
                [
                    ast.keyword(arg="full_dims", value=ast.Name("full_dims", ast.Load())),
                    ast.keyword(arg="offset", value=ast.Name("offset", ast.Load())),
                ]
            )
            call_count += 1
    if call_count != 1:
        raise RuntimeError(
            "MiniMax H3 decoder token-coordinate structure is unknown; no patch was applied."
        )
    tree.body[0] = function
    return _compile_function(
        tree,
        original,
        {"_t8_create_token_ids_global": create_token_ids_global},
        extra_defaults=(None, None),
    )


def _compile_tiled_decode(callable_object):
    original = callable_object.__func__ if inspect.ismethod(callable_object) else callable_object
    source = _source_of(original)
    if not source:
        raise RuntimeError("MiniMax H3 tiled-decode source is unavailable; no patch was applied.")
    tree = ast.parse(source)
    function = tree.body[0]
    if not isinstance(function, ast.FunctionDef):
        raise RuntimeError("MiniMax H3 tiled decode is not a normal Python function.")

    rewritten_body: list[ast.stmt] = []
    full_dims_count = 0
    for statement in function.body:
        rewritten_body.append(statement)
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Tuple)
            and [item.id for item in statement.targets[0].elts if isinstance(item, ast.Name)]
            == ["height", "width"]
        ):
            rewritten_body.append(
                ast.parse("full_dims = (z.shape[-3], z.shape[-2], z.shape[-1])").body[0]
            )
            full_dims_count += 1
    function.body = rewritten_body

    decode_call_count = 0
    for node in ast.walk(function):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_decode_pixels"
        ):
            if not any(keyword.arg == "full_dims" for keyword in node.keywords):
                node.keywords.extend(
                    [
                        ast.keyword(arg="full_dims", value=ast.Name("full_dims", ast.Load())),
                        ast.keyword(
                            arg="offset",
                            value=ast.Tuple(
                                elts=[
                                    ast.Constant(0),
                                    ast.Name("zi", ast.Load()),
                                    ast.Name("zj", ast.Load()),
                                ],
                                ctx=ast.Load(),
                            ),
                        ),
                    ]
                )
                decode_call_count += 1
    if full_dims_count != 1 or decode_call_count != 1:
        raise RuntimeError(
            "MiniMax H3 tiled-decode geometry is unknown; no patch was applied."
        )
    tree.body[0] = function
    return _compile_function(tree, original)


def _decode_pixels_with_global_coordinates(self, z, full_dims=None, offset=None):
    return self.decoder(
        self.post_quant_conv(z),
        full_dims=full_dims,
        offset=offset,
    )


_decode_pixels_with_global_coordinates.__t8_global_coordinates__ = True


def _shallow_clone_module(module):
    cloned = copy.copy(module)
    for name in ("_parameters", "_buffers", "_modules"):
        value = getattr(module, name, None)
        if isinstance(value, dict):
            setattr(cloned, name, value.copy())
    non_persistent = getattr(module, "_non_persistent_buffers_set", None)
    if isinstance(non_persistent, set):
        cloned._non_persistent_buffers_set = non_persistent.copy()
    return cloned


def build_tiled_vae_coordinate_compatibility(video_vae, mode="report_only"):
    if mode not in ("report_only", "apply_global_coordinates_exp"):
        raise ValueError("mode must be report_only or apply_global_coordinates_exp")
    first_stage = getattr(video_vae, "first_stage_model", None)
    if first_stage is None or first_stage.__class__.__name__ != "MiniMaxH3VideoVAE":
        raise TypeError(
            "MiniMax H3 Global-Coordinate Tiled VAE requires the native H3 video VAE."
        )
    decoder = getattr(first_stage, "decoder", None)
    if decoder is None:
        raise RuntimeError("MiniMax H3 video VAE decoder is unavailable.")

    capability = probe_native_tiled_coordinates(video_vae)
    if capability["available"]:
        report = {
            "schema": "t8_minimax_h3_tiled_coordinates_v1",
            "status": "native",
            "native_core_probe": capability,
            "clone_scope": "none",
            "requested_mode": mode,
            "warning": REAL_VALIDATION_WARNING,
            "source_pr": "https://github.com/Comfy-Org/ComfyUI/pull/15561",
        }
        return video_vae, json.dumps(report, ensure_ascii=False, sort_keys=True)

    if mode == "report_only":
        report = {
            "schema": "t8_minimax_h3_tiled_coordinates_v1",
            "status": "report_only_abstain_real_validation_failed",
            "native_core_probe": capability,
            "requested_mode": mode,
            "candidate_applied": False,
            "source_vae_unchanged": True,
            "warning": REAL_VALIDATION_WARNING,
            "source_pr": "https://github.com/Comfy-Org/ComfyUI/pull/15561",
        }
        return video_vae, json.dumps(report, ensure_ascii=False, sort_keys=True)

    decoder_forward, _ = _compile_decoder_forward(decoder.forward)
    tiled_decode, _ = _compile_tiled_decode(first_stage.tiled_decode)

    cloned_vae = copy.copy(video_vae)
    cloned_first_stage = _shallow_clone_module(first_stage)
    cloned_decoder = _shallow_clone_module(decoder)
    cloned_decoder.forward = types.MethodType(decoder_forward, cloned_decoder)
    cloned_first_stage.decoder = cloned_decoder
    cloned_first_stage._decode_pixels = types.MethodType(
        _decode_pixels_with_global_coordinates,
        cloned_first_stage,
    )
    cloned_first_stage.tiled_decode = types.MethodType(
        tiled_decode,
        cloned_first_stage,
    )
    cloned_vae.first_stage_model = cloned_first_stage

    compat_capability = {
        "available": True,
        "token_ids_global_coordinates": True,
        "decoder_accepts_full_grid": True,
        "decode_pixels_forwards_full_grid": True,
        "tiled_decode_supplies_offsets": True,
        "policy": "isolated_shallow_clone_compatibility_no_global_monkeypatch",
    }
    setattr(cloned_vae, TILED_VAE_ATTACHMENT_ATTR, compat_capability)
    report = {
        "schema": "t8_minimax_h3_tiled_coordinates_v1",
        "status": "compatibility_clone_ready",
        "native_core_probe": capability,
        "compatibility_probe": compat_capability,
        "clone_scope": "vae_wrapper_first_stage_decoder_methods_only_weights_shared",
        "source_vae_unchanged": True,
        "requested_mode": mode,
        "candidate_applied": True,
        "warning": REAL_VALIDATION_WARNING,
        "source_pr": "https://github.com/Comfy-Org/ComfyUI/pull/15561",
    }
    return cloned_vae, json.dumps(report, ensure_ascii=False, sort_keys=True)
