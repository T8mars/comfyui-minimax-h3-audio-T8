from __future__ import annotations

import ast
import functools
import inspect
import json
import textwrap
import types
from typing import Any


FORWARD_SYNC_ATTACHMENT_KEY = "t8_minimax_h3_forward_sync_contract"
FORWARD_PATCH_PATH = "diffusion_model._forward"


def _source_of(callable_object) -> str:
    try:
        return textwrap.dedent(inspect.getsource(callable_object))
    except (OSError, TypeError):
        return ""


def probe_native_forward_sync(diffusion_or_callable) -> dict[str, Any]:
    callable_object = diffusion_or_callable
    if not callable(diffusion_or_callable):
        callable_object = getattr(type(diffusion_or_callable), "_forward", None)
    marker_owner = (
        callable_object.__func__ if inspect.ismethod(callable_object) else callable_object
    )
    marker = getattr(marker_owner, "__t8_forward_sync_capability__", None)
    if isinstance(marker, dict):
        return dict(marker)
    source = _source_of(callable_object)
    sigma_single_sync = (
        "sigma_v = float(" in source or "sigma_v_scalar = float(sigma_v)" in source
    ) and "float(1.0 - time_shift_sigma(" not in source
    tag_list_cache = (
        "_text_token_tags_list" in source
        and "payload[\"_text_token_tags_list\"]" in source
    )
    return {
        "available": bool(sigma_single_sync and tag_list_cache),
        "sigma_single_sync": bool(sigma_single_sync),
        "text_tag_list_cache": bool(tag_list_cache),
        "source_available": bool(source),
        "policy": "semantic_source_feature_probe_no_version_or_hash_gate",
    }


def _target_name(node: ast.Assign, name: str) -> bool:
    return len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == name


def _strip_float_call(value: ast.expr) -> ast.expr | None:
    if (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id == "float"
        and len(value.args) == 1
        and not value.keywords
    ):
        return value.args[0]
    return None


class _ReplaceName(ast.NodeTransformer):
    def __init__(self, old: str, new: str):
        self.old = old
        self.new = new

    def visit_Name(self, node: ast.Name):
        if node.id == self.old:
            return ast.copy_location(ast.Name(id=self.new, ctx=node.ctx), node)
        return node


def _is_text_tag_list_conversion(value: ast.expr) -> bool:
    # text_tags.view(-1).tolist()
    return (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Attribute)
        and value.func.attr == "tolist"
        and isinstance(value.func.value, ast.Call)
        and isinstance(value.func.value.func, ast.Attribute)
        and value.func.value.func.attr == "view"
        and isinstance(value.func.value.func.value, ast.Name)
        and value.func.value.func.value.id == "text_tags"
    )


class _ForwardSyncTransformer(ast.NodeTransformer):
    def __init__(self, *, optimize_sigma: bool, optimize_tags: bool):
        self.optimize_sigma = optimize_sigma
        self.optimize_tags = optimize_tags
        self.sigma_assignment_count = 0
        self.t_video_count = 0
        self.t_audio_count = 0
        self.text_tags_assignment_count = 0
        self.text_condition_count = 0
        self.text_conversion_removed_count = 0

    def visit_Assign(self, node: ast.Assign):
        node = self.generic_visit(node)
        if self.optimize_sigma and _target_name(node, "sigma_v"):
            self.sigma_assignment_count += 1
        if self.optimize_sigma and _target_name(node, "t_v"):
            inner = _strip_float_call(node.value)
            if inner is not None:
                node.value = _ReplaceName("sigma_v", "sigma_v_scalar").visit(inner)
                self.t_video_count += 1
        if self.optimize_sigma and _target_name(node, "t_a"):
            inner = _strip_float_call(node.value)
            if inner is not None:
                node.value = _ReplaceName("sigma_v", "sigma_v_scalar").visit(inner)
                self.t_audio_count += 1
        if self.optimize_tags and _target_name(node, "text_tags"):
            self.text_tags_assignment_count += 1
        if self.optimize_tags and _target_name(node, "tags") and _is_text_tag_list_conversion(node.value):
            self.text_conversion_removed_count += 1
            return None
        return node

    def visit_If(self, node: ast.If):
        node = self.generic_visit(node)
        if not self.optimize_tags:
            return node
        test_text = ast.unparse(node.test)
        if "kind == 'text'" in test_text and "text_tags is not None" in test_text:
            node.test = _ReplaceName("text_tags", "tags").visit(node.test)
            self.text_condition_count += 1
        return node


def _rewrite_forward_source(callable_object) -> tuple[str, dict[str, Any]]:
    source = _source_of(callable_object)
    if not source:
        raise RuntimeError("MiniMax H3 forward source is unavailable; no patch was applied.")
    capability = probe_native_forward_sync(callable_object)
    optimize_sigma = not capability["sigma_single_sync"]
    optimize_tags = not capability["text_tag_list_cache"]
    tree = ast.parse(source)
    if not tree.body or not isinstance(tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef)):
        raise RuntimeError("MiniMax H3 forward source is not a normal Python function.")
    function = tree.body[0]
    transformer = _ForwardSyncTransformer(
        optimize_sigma=optimize_sigma,
        optimize_tags=optimize_tags,
    )
    function = transformer.visit(function)

    rewritten_body: list[ast.stmt] = []
    inserted_sigma = 0
    inserted_tags = 0
    sigma_cache = ast.parse("sigma_v_scalar = float(sigma_v)").body[0]
    tag_cache = ast.parse(
        """
tags = payload.get("_text_token_tags_list")
if text_tags is not None and tags is None:
    tags = text_tags.view(-1).tolist()
    payload["_text_token_tags_list"] = tags
"""
    ).body
    for statement in function.body:
        rewritten_body.append(statement)
        if optimize_sigma and isinstance(statement, ast.Assign) and _target_name(statement, "sigma_v"):
            rewritten_body.append(sigma_cache)
            inserted_sigma += 1
        if optimize_tags and isinstance(statement, ast.Assign) and _target_name(statement, "text_tags"):
            rewritten_body.extend(tag_cache)
            inserted_tags += 1
    function.body = rewritten_body
    tree.body[0] = function
    ast.fix_missing_locations(tree)

    if optimize_sigma and (
        transformer.sigma_assignment_count != 1
        or transformer.t_video_count != 1
        or transformer.t_audio_count != 1
        or inserted_sigma != 1
    ):
        raise RuntimeError(
            "MiniMax H3 sigma synchronization structure is unknown; no patch was applied."
        )
    if optimize_tags and (
        transformer.text_tags_assignment_count != 1
        or transformer.text_condition_count != 1
        or transformer.text_conversion_removed_count != 1
        or inserted_tags != 1
    ):
        raise RuntimeError(
            "MiniMax H3 text-tag structure is unknown; no patch was applied."
        )

    rewritten = ast.unparse(tree)
    report = {
        "sigma_rewritten": bool(optimize_sigma),
        "text_tags_rewritten": bool(optimize_tags),
        "source_before": capability,
    }
    return rewritten, report


def _compile_rewritten_forward(callable_object):
    rewritten, rewrite_report = _rewrite_forward_source(callable_object)
    original = callable_object.__func__ if inspect.ismethod(callable_object) else callable_object
    namespace = dict(getattr(original, "__globals__", {}))
    filename = "<t8_minimax_h3_forward_sync_compatibility>"
    module_code = compile(rewritten, filename, "exec")
    candidates = [
        value
        for value in module_code.co_consts
        if isinstance(value, types.CodeType) and value.co_name == original.__name__
    ]
    if len(candidates) != 1 or candidates[0].co_freevars:
        raise RuntimeError(
            "MiniMax H3 rewritten forward could not be isolated safely; "
            "no patch was applied."
        )
    replacement = types.FunctionType(
        candidates[0],
        namespace,
        original.__name__,
        getattr(original, "__defaults__", None),
    )
    replacement.__kwdefaults__ = getattr(original, "__kwdefaults__", None)
    functools.update_wrapper(replacement, original)
    replacement.__t8_forward_sync_capability__ = {
        "available": True,
        "sigma_single_sync": True,
        "text_tag_list_cache": True,
        "source_available": True,
        "policy": "semantic_source_feature_probe_no_version_or_hash_gate",
    }
    return replacement, rewrite_report


def build_forward_sync_optimization(model):
    diffusion = model.get_model_object("diffusion_model")
    if diffusion.__class__.__name__ != "MiniMaxH3Model":
        raise TypeError("MiniMax H3 Forward Sync Optimization requires MiniMaxH3Model.")

    existing_patches = getattr(model, "object_patches", {})
    existing = existing_patches.get(FORWARD_PATCH_PATH)
    effective_callable = existing if callable(existing) else getattr(type(diffusion), "_forward", None)
    if effective_callable is None:
        raise RuntimeError("MiniMax H3 _forward callable is unavailable; no patch was applied.")

    capability = probe_native_forward_sync(effective_callable)
    patched = model.clone()
    rewrite_report: dict[str, Any] = {
        "sigma_rewritten": False,
        "text_tags_rewritten": False,
        "source_before": capability,
    }
    status = "native"
    if not capability["available"]:
        replacement, rewrite_report = _compile_rewritten_forward(effective_callable)
        patched.add_object_patch(
            FORWARD_PATCH_PATH,
            types.MethodType(replacement, diffusion),
        )
        status = "compatibility_patch_ready"

    report = {
        "schema": "t8_minimax_h3_forward_sync_v1",
        "status": status,
        "native_core_probe": capability,
        "rewrite": rewrite_report,
        "composed_existing_forward_patch": bool(existing is not None),
        "object_patch_scope": "model_patcher_clone_only",
        "expected_host_syncs_per_step": 1,
        "text_tag_host_copy": "once_per_payload",
        "source_pr": "https://github.com/Comfy-Org/ComfyUI/pull/15560",
    }
    if hasattr(patched, "set_attachments"):
        patched.set_attachments(FORWARD_SYNC_ATTACHMENT_KEY, report)
    return patched, json.dumps(report, ensure_ascii=False, sort_keys=True)
