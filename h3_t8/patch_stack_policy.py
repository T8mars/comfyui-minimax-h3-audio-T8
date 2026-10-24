"""User-selected MODEL stacks are advisory, not an admission allowlist.

Do not catch sampler/kernel errors here. Unknown executable state cannot claim
portable cache identity: use a fresh execution nonce rather than reject its use.
"""
from __future__ import annotations

from functools import wraps
from contextvars import ContextVar
import hashlib
import json
import logging
import uuid

_audit_advisories = ContextVar("t8_patch_stack_audit_advisories", default=None)


class UnverifiedModelStack(ValueError):
    """Classifier cannot give a portable identity; not a sampling prohibition."""


def warn_patch_stack(message):
    advisories = _audit_advisories.get()
    if advisories is not None and message not in advisories:
        advisories.append(message)
    logging.warning(
        "[MiniMax H3 compatibility advisory / 组合风险自负] %s; continuing. "
        "This is not a compatibility or quality guarantee. Existing patches may "
        "bypass this node; real execution errors still propagate.", message,
    )


def advisory_audit(function):
    """Keep an incomplete composition audit from falsely claiming verification.

    No exception is caught: kernel failures, invalid clocks/inputs and damaged
    receipts still propagate normally.
    """
    @wraps(function)
    def audit(*args, **kwargs):
        advisories = []
        token = _audit_advisories.set(advisories)
        try:
            latent, serialized = function(*args, **kwargs)
            if advisories:
                report = json.loads(serialized)
                report.update(status="executed_user_stack_unverified",
                              compatibility_advisories=advisories,
                              composition_verified=False)
                serialized = json.dumps(report, ensure_ascii=False, allow_nan=False)
            return latent, serialized
        finally:
            _audit_advisories.reset(token)
    return audit


def advisory_inspection(function):
    """Only for optional composition classifiers, never receipt/hash validators."""
    @wraps(function)
    def inspect(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except (ValueError, TypeError, RuntimeError) as error:
            warn_patch_stack(f"{function.__name__}: unverified composition: {error}")
            return None
    return inspect


def nonportable_model_identity(model, reason, *, schema):
    """Allow sampling without unsafe cross-run reuse of unaudited live state."""
    warn_patch_stack(f"{reason}; portable cache reuse disabled for this execution")
    state = model.model_state_dict() if hasattr(model, "model_state_dict") else model.model.state_dict()
    if not state:
        raise ValueError("MODEL has no loaded tensor state")
    # Validate actual tensor bytes too; opaque hooks are not a way around NaN,
    # missing tensors or malformed weight storage checks.
    from .long_video_dual_identity import (
        content_identity, _original_state, _original_buffers, UnknownBufferBackupSchema,
    )
    unknown_buffers = False
    known_digest = None
    try:
        original = _original_state(model, state)
    except UnknownBufferBackupSchema as error:
        # An unsupported future/foreign backup record is not a ban on normal
        # execution, and not a reason to forge an original-weight SHA. Validate
        # all known live/original tensors and preserve only process-local scope.
        warn_patch_stack(str(error))
        content_identity(state)
        original = _original_state(model, state, allow_unknown_buffers=True)
        known_digest = hashlib.sha256(repr(content_identity({
            "known_state": original,
            "known_original_buffers": _original_buffers(model, allow_unknown=True),
        })).encode("utf-8")).hexdigest()
        unknown_buffers = True
    state_digest = None if unknown_buffers else hashlib.sha256(
        repr(content_identity(original)).encode("utf-8")
    ).hexdigest()
    selected = {name: _execution_selection(getattr(model, name, None)) for name in (
        "patches", "object_patches", "wrappers", "callbacks", "injections",
        "weight_wrapper_patches", "hook_patches", "forced_hooks", "current_hooks",
        "model_options", "additional_models",
    )}
    network = getattr(model, "model", None)
    if callable(getattr(network, "named_modules", None)):
        selected["network_execution"] = {
            name: _execution_selection({
                "forward": vars(module).get("forward"),
                "pre_hooks": getattr(module, "_forward_pre_hooks", {}),
                "hooks": getattr(module, "_forward_hooks", {}),
            }) for name, module in network.named_modules()
            if "forward" in vars(module) or getattr(module, "_forward_pre_hooks", {})
            or getattr(module, "_forward_hooks", {})
        }
    result = {
        "schema": schema, "sha256": hashlib.sha256(uuid.uuid4().bytes).hexdigest(),
        "backend": {"kind": "user_selected_unverified"},
        "memory": None, "composition": {"kind": "user_selected_unverified"},
        "portable_cache_reuse": False, "model_filename_trusted": False,
        "execution_weight_sha256": state_digest,
        "execution_selection": selected,
        "opaque_internal_state_verified": False,
        "tensor_count": len(state), "lora_target_count": len(getattr(model, "patches", {})),
    }
    if unknown_buffers:
        result["execution_weight_content_verified"] = False
        result["execution_known_tensor_sha256"] = known_digest
        selected["unverified_buffer_backups"] = _execution_selection(getattr(model, "backup_buffers", None))
    return result


def nonportable_component_identity(component, reason, *, schema):
    patcher = getattr(component, "patcher", None)
    inspectable = callable(getattr(patcher, "model_state_dict", None)) or (
        not hasattr(patcher, "model_state_dict")
        and callable(getattr(getattr(patcher, "model", None), "state_dict", None))
    )
    if not inspectable:
        # Executable encoders need not expose Core's optional inspection API.
        # A process-local opaque selection is not a weight/content proof, but
        # lack of that proof must not turn ordinary encoding into AttributeError.
        warn_patch_stack(f"{reason}; no native CLIP/VAE patcher to inspect, portable cache reuse disabled")
        result = {
            "schema": schema, "sha256": hashlib.sha256(uuid.uuid4().bytes).hexdigest(),
            "portable_cache_reuse": False, "model_filename_trusted": False,
            "execution_weight_sha256": None, "execution_weight_content_verified": False,
            "opaque_internal_state_verified": False, "tensor_count": None,
            "lora_target_count": None,
            "identity_unavailable_reason": "No native CLIP/VAE weight-inspection API is available for portable content identity",
            "execution_selection": {"component_object": _execution_selection(component),
                                    "opaque_patcher": _execution_selection(patcher)},
        }
    else:
        result = nonportable_model_identity(patcher, reason, schema=schema)
    result["execution_selection"]["component"] = _execution_selection({
        "tokenizer": getattr(component, "tokenizer", None),
        "tokenizer_options": getattr(component, "tokenizer_options", None),
        "use_clip_schedule": getattr(component, "use_clip_schedule", None),
        "apply_hooks_to_conds": getattr(component, "apply_hooks_to_conds", None),
        "methods": {name: getattr(component, name, None) for name in (
            "tokenize", "encode_from_tokens", "encode_from_tokens_scheduled",
            "encode", "decode", "encode_tiled", "decode_tiled")},
    })
    return result


def _execution_selection(value):
    """Process-local selection snapshot, never a portable callable identity.

    Opaque callable internals are deliberately not certified. Detect replacement
    and weight/LoRA mutation without running hooks, repr or dequantization.
    """
    from .long_video_dual_identity import content_identity
    if isinstance(value, dict):
        return [( _execution_selection(key), _execution_selection(item))
                for key, item in value.items()]
    if isinstance(value, (list, tuple)):
        return [_execution_selection(item) for item in value]
    try:
        return content_identity(value)
    except UnverifiedModelStack:
        function = getattr(value, "__func__", value)
        owner = getattr(value, "__self__", None)
        result = {"process_object": id(function), "bound_owner": id(owner)}
        try:
            from comfy.weight_adapter.lora import LoRAAdapter
        except ImportError:
            LoRAAdapter = None
        if LoRAAdapter is not None and isinstance(value, LoRAAdapter):
            result["adapter_weights"] = _execution_selection(value.weights)
        return result


def model_identity_matches(expected, current):
    """Within-run check only: nonce remains in persisted cache keys."""
    if isinstance(expected, dict) and isinstance(current, dict):
        if expected.keys() != current.keys():
            return False
        ignored = {"sha256"} if (expected.get("portable_cache_reuse") is False
                                 and current.get("portable_cache_reuse") is False) else set()
        return all(model_identity_matches(expected[key], current[key])
                   for key in expected.keys() - ignored)
    if isinstance(expected, (list, tuple)) and isinstance(current, (list, tuple)):
        return len(expected) == len(current) and all(
            model_identity_matches(a, b) for a, b in zip(expected, current))
    return expected == current


def compose_dit_hook(previous, current, owner):
    """Preserve the foreign owner, inserting our route only when it delegates."""
    if previous is None:
        return current
    if not callable(previous):
        raise TypeError(f"{owner}: existing DiT hook must be callable")
    warn_patch_stack(f"{owner}: retaining foreign DiT owner; a non-delegating hook may bypass this node")

    def composed(args, extra):
        return previous(args, {**extra, "original_block": lambda local: current(local, extra)})
    return composed


def slice_attention_mask(mask, start, end):
    if mask is None or mask.ndim < 2 or mask.shape[-2] == 1:
        return mask
    return mask[..., start:end, :]


def merge_attention_bias(bias, mask):
    if mask is None:
        return bias
    import torch
    return bias.masked_fill(~mask, float('-inf')) if mask.dtype == torch.bool else bias + mask
