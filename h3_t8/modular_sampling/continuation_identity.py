"""Inspection-only projection of the actual native long-video payload owner.

Authenticate factory code, closure and bound native delegate. Never strip an
unknown extra_conds or mutate the live MODEL to grant persistent cache reuse.
"""
import hashlib
import json
from pathlib import Path
from types import MethodType

import comfy.model_base

from .. import long_video
from ..vdn_attention_compat import _factory_closure
from .effect_identity import _require
from .progressive_effect_identity import function_identity, _code_identity


def project(model):
    patch = model.object_patches.get("extra_conds")
    if patch is None:
        return model, None
    function = getattr(patch, "__func__", None)
    _require(type(patch) is MethodType and patch.__self__ is model.model,
             "Continuation extra_conds needs its native bound owner")
    closure = _factory_closure(function, long_video.patch_long_video_model, "_patched_extra_conds")
    _require(closure is not None and set(closure) == {"original"},
             "Continuation extra_conds is not the actual long-video factory")
    _require(set(vars(function)) == {"_t8_long_video_patch_version", "_t8_long_video_original_extra_conds"}
             and function._t8_long_video_patch_version == long_video.LONG_VIDEO_PATCH_VERSION
             and function.__defaults__ is None and function.__kwdefaults__ is None,
             "Continuation payload factory has unknown executable state")
    original = closure["original"]
    _require(function._t8_long_video_original_extra_conds is original,
             "Continuation restore descriptor differs from its actual delegate")
    _require(type(model.model) is comfy.model_base.MiniMaxH3 and type(original) is MethodType
             and original.__self__ is model.model and original.__func__ is comfy.model_base.MiniMaxH3.extra_conds,
             "Continuation native delegate has another user owner; persistent adapter required")
    clone = model.clone()
    clone.object_patches.pop("extra_conds")
    return clone, {"schema": "t8.modular-sampling.continuation-model.v1",
        "version": long_video.LONG_VIDEO_PATCH_VERSION,
        "factory_code_sha256": hashlib.sha256(json.dumps(_code_identity(function.__code__),
            sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "factory": function_identity(long_video.patch_long_video_model),
        "native_delegate": function_identity(original.__func__),
        "payload_repair": function_identity(long_video.repair_long_video_payload),
        "layout_repair": function_identity(long_video.repair_long_video_layout),
        "implementation": {Path(module.__file__).name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                           for module in (long_video, comfy.model_base)}}
