"""Explicit, paired and request-local approximate DiT reference-input cache.

Unknown execution owners run unchanged. No persistent/portable cache receipt is
issued; this is not an exact diffusion memoizer or Qwen visual-prefix cache.
"""
from __future__ import annotations

from contextvars import ContextVar
from copy import deepcopy
from dataclasses import asdict, dataclass, field
import hashlib
from pathlib import Path
import threading
from types import FunctionType
import uuid

import comfy.conds
from comfy.patcher_extension import CallbacksMP, WrappersMP

from .patch_stack_policy import _execution_selection, warn_patch_stack
from .reference_temporal_pack import condition_identity, digest, tensor_record
from . import reference_step_cache as primitive


KEY = "t8_reference_step_cache"
PAIR_KEY = KEY + "_binding"
SCHEMA = "t8.h3.reference_step_cache.v1"
_CALL = ContextVar(KEY + "_call", default=None)
_PAIR = ContextVar(KEY + "_pair", default=None)


def _helpers():
    functions = {"primitive." + name: getattr(primitive, name) for name in (
        "parameter_versions", "module_contract", "weight_digest", "native_network", "native_block", "native_attention_backend", "logical_step",
        "row_indices", "select_segments", "condition_key", "cached_block")}
    functions.update({"primitive." + name: getattr(primitive, name) for name in (
        "source_current", "module_call_native", "attention_source_current")})
    functions["primitive.ReferenceAttention.__call__"] = primitive.ReferenceAttention.__call__
    functions.update({name: globals()[name] for name in ("condition_identity", "digest", "tensor_record", "_execution_selection")})
    return {key: (value, value.__code__ if type(value) is FunctionType else None) for key, value in functions.items()}


def _selection(model):
    return _execution_selection({name: getattr(model, name, None) for name in (
        "patches", "object_patches", "model_options", "wrappers", "callbacks",
        "hook_patches", "forced_hooks", "injections", "weight_wrapper_patches")})


@dataclass
class _Runtime:
    plan: primitive.ReferenceStepPlan
    phase: str
    positive: object
    positive_identity: dict
    seal: str
    network: object
    previous: object
    implementation: dict
    config_binding: str
    helpers: dict
    wrapper: object = None
    diffusion_wrapper: object = None
    blocks: dict = field(default_factory=dict)
    lock: object = field(default_factory=threading.RLock)
    active: bool = False
    cleanup_observed: bool = False
    ambiguous: bool = False
    request: int = 0
    completed: int = 0
    failed: int = 0
    full: int = 0
    hits: int = 0
    skipped_rows: int = 0
    peak_bytes: int = 0
    evaluations: list = field(default_factory=list)
    reasons: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    buckets: dict = field(default_factory=dict)
    patcher: object = None
    selection: object = None
    versions: object = None
    modules: object = None
    weights: str | None = None
    prepare_callback: object = None
    cleanup_callback: object = None

    def reason(self, value):
        if value not in self.reasons:
            self.reasons.append(value)

    def verify(self):
        if digest({"plan": asdict(self.plan), "phase": self.phase}) != self.config_binding:
            raise ValueError("Reference step cache explicit plan/phase binding changed")
        current = _helpers()
        if any(current[key][0] is not value[0] or current[key][1] is not value[1]
               for key, value in self.helpers.items()):
            raise ValueError("Reference step cache selected execution helper changed")
        if condition_identity(self.positive) != self.positive_identity:
            raise ValueError("Reference step cache selected conditioning content SHA changed")
        for path, expected in self.implementation.items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
                raise ValueError("Reference step cache implementation source SHA changed")
        if self.active and _selection(self.patcher) != self.selection:
            raise ValueError("Reference step cache selected MODEL/ordered patch ownership changed")

    def prepare(self, patcher, *_):
        with self.lock:
            self.verify()
            overlapping = self.active
            self.request += 1
            self.active, self.cleanup_observed, self.ambiguous = True, False, overlapping
            self.completed = self.failed = self.full = self.hits = self.skipped_rows = self.peak_bytes = 0
            self.evaluations, self.reasons, self.errors, self.buckets = [], [], [], {}
            self.patcher, self.selection = patcher, _selection(patcher)
            self.versions, self.weights, self.modules = None, None, None
            if overlapping:
                self.reason("overlapping_request_original_delegated")

    def cleanup(self, *_):
        with self.lock:
            self.buckets.clear()
            self.active, self.cleanup_observed = False, True

    def begin(self, args):
        self.verify()
        c = args["c"]
        if c.get(PAIR_KEY) != self.seal:
            raise ValueError("Reference step cache MODEL/positive pair binding differs")
        options = c.get("transformer_options", {})
        step = primitive.logical_step(options)
        capable = primitive.native_network(self.network)
        if args.get("unpaired_negative"):
            capable = False
            self.reason("unpaired_negative_full_recompute")
        from comfy import model_management
        if model_management.in_training:
            capable = False
            self.reason("training_functional_attention_original_delegated")
        if not primitive.native_attention_backend():
            capable = False
            self.reason("unknown_global_attention_owner_original_delegated")
        override = options.get("optimized_attention_override")
        if override is not None:
            from .h3_core_compat import plain_attention_backend
            if plain_attention_backend(override) is None:
                capable = False
                self.reason("unknown_attention_owner_original_delegated")
        if options.get("patches"):
            capable = False
            self.reason("unknown_attention_patches_original_delegated")
        if self.previous is not None:
            # It may change context or silently omit the actual MODEL call.
            capable = False
            self.reason("previous_model_wrapper_original_delegated")
        if not primitive.native_network(self.network):
            self.reason("unknown_network_forward_original_delegated")
        if step is None:
            self.reason("unidentified_logical_sigma_original_delegated")
        if not self.active or self.ambiguous:
            capable = False
            self.reason("unowned_lifecycle_original_delegated")
        if capable and step is not None and self.weights is None:
            self.weights = primitive.weight_digest(self.network)
            self.versions = primitive.parameter_versions(self.network)
            self.modules = primitive.module_contract(self.network)
            if self.weights is None:
                self.reason("opaque_weight_storage_original_delegated")
        if self.versions is not None and primitive.parameter_versions(self.network) != self.versions:
            raise ValueError("Reference step cache actual selected weight version changed")
        if self.modules is not None and primitive.module_contract(self.network) != self.modules:
            raise ValueError("Reference step cache selected executable module ownership/structure changed")
        return {"capable": capable and self.weights is not None and step is not None,
                "step": step, "context": c.get("c_crossattn"), "payload": c.get("minimax_payload"),
                "buckets": {}, "events": [],
                "condition_scope": "unpaired_negative" if args.get("unpaired_negative") else "paired"}

    def finish(self, call):
        self.verify()
        if self.versions is not None and primitive.parameter_versions(self.network) != self.versions:
            raise ValueError("Reference step cache actual selected weight version changed")
        if self.modules is not None and primitive.module_contract(self.network) != self.modules:
            raise ValueError("Reference step cache selected executable module ownership/structure changed")
        if self.weights is not None and call["step"] is not None and call["step"][1] == len(call["step"][0]) - 2:
            if primitive.weight_digest(self.network) != self.weights:
                raise ValueError("Reference step cache actual selected weight content SHA changed")
        for bucket in call["buckets"].values():
            bucket["last"] = call["step"][1]
        self.evaluations.append({"sigma": call["step"][0][call["step"][1]] if call["step"] else None,
                                 "logical_step": call["step"][1] if call["step"] else None,
                                 "condition_scope": call["condition_scope"],
                                 "blocks": call["events"]})
        self.completed += 1

    def block(self, index, block, previous, args, extra):
        scoped = _CALL.get()
        if scoped is None or scoped[0] is not self:
            return previous(args, extra) if previous is not None else extra["original_block"](args)
        call = scoped[1]
        if previous is not None:
            self.reason("existing_block_owner_original_delegated")
            return previous(args, extra)
        if not call["capable"] or not primitive.native_block(block) or args.get("attention") is not None:
            if call["capable"]:
                self.reason("unknown_block_or_module_hook_original_delegated")
            self.full += 1
            return extra["original_block"](args)
        rows = primitive.row_indices(args["layout"], self.plan.include_text)
        if rows is None or not rows[0].numel() or not rows[1].numel():
            self.reason("no_reference_rows_original_delegated")
            self.full += 1
            return extra["original_block"](args)
        cached, active = rows
        grid, current = call["step"]
        key = (grid, primitive.condition_key(call["context"], call["payload"], args["layout"]))
        bucket = self.buckets.setdefault(key, {"last": None, "blocks": {}})
        call["buckets"][key] = bucket
        last = bucket["last"]
        same_step = last == current
        consecutive = last is not None and current == last + 1
        refresh = (self.plan.refresh_every == 1 or current % self.plan.refresh_every == 0
                   or not consecutive or index not in bucket["blocks"])
        event = {"index": index, "packed_rows": args["img"].shape[0],
                 "reference_rows": cached.numel(), "active_rows": active.numel(),
                 "decision": "same_sigma_full" if same_step else "full_refresh" if refresh else "approximate_hit"}
        call["events"].append(event)
        if refresh:
            stored = args["img"].index_select(0, cached.to(args["img"].device)).detach().to("cpu", copy=True)
            record = tensor_record(stored)
            result = extra["original_block"](args)
            if not same_step:
                bucket["blocks"][index] = (stored, record)
            self.full += 1
        else:
            stored, record = bucket["blocks"][index]
            if tensor_record(stored) != record:
                raise ValueError("Reference step cache stored reference input SHA changed")
            result = primitive.cached_block(block, args, extra, stored, cached, active)
            self.hits += 1
            self.skipped_rows += cached.numel()
        self.peak_bytes = max(self.peak_bytes, sum(value[0].numel() * value[0].element_size()
            for item in self.buckets.values() for value in item["blocks"].values()))
        return result


def configure_reference_step_cache(model, positive, mode="off", refresh_every=2, include_text=False, *, phase="SINGLE"):
    plan = primitive.ReferenceStepPlan(refresh_every, include_text)
    if mode not in ("off", "approximate"):
        raise ValueError("Reference step cache mode must be off or approximate")
    if not isinstance(phase, str) or not phase:
        raise ValueError("Reference cache phase must be a nonempty explicit label")
    if mode == "off":
        return model, positive, {"schema": SCHEMA, "status": "disabled_identity", "portable_cache_identity": False}
    required = ("clone", "get_attachment", "set_attachments", "add_callback_with_key",
                "set_model_unet_function_wrapper", "set_model_patch_replace", "get_model_object",
                "add_wrapper_with_key", "get_wrappers")
    if not all(callable(getattr(model, name, None)) for name in required):
        warn_patch_stack("Reference step cache cannot observe this MODEL; original MODEL/positive retained")
        return model, positive, {"schema": SCHEMA, "status": "unknown_original_delegated", "portable_cache_identity": False}
    existing = model.get_attachment(KEY)
    if existing is not None and type(existing) is not _Runtime:
        warn_patch_stack("Reference step cache retains an unknown attachment owner and original MODEL/positive")
        return model, positive, {"schema": SCHEMA, "status": "unknown_owner_original_delegated", "portable_cache_identity": False}
    if existing is not None:
        existing.verify()
        if model.model_options.get("model_function_wrapper") is not existing.wrapper:
            warn_patch_stack("Reference step cache retains a later MODEL wrapper; reselection not applied")
            return model, positive, audit_reference_step_cache(model)
        # Strip only our exact old scalar binding on the new conditioning copy.
        clean = []
        for embedding, metadata in positive:
            conditions = dict(metadata.get("model_conds", {}))
            old = conditions.get(PAIR_KEY)
            if old is not None:
                if type(old) is not comfy.conds.CONDConstant or old.cond != existing.seal:
                    raise ValueError("Reference step cache previous MODEL/positive pair binding differs")
                conditions.pop(PAIR_KEY)
            clean.append([embedding, {**metadata, "model_conds": conditions}])
        positive = clean
    # Validate/hash actual chosen contents. Unknown metadata remains executable;
    # it cannot turn known neighboring tensors into unchecked state.
    original_identity = condition_identity(positive)
    selected = model.clone()
    network = selected.get_model_object("diffusion_model")
    blocks = getattr(network, "blocks", ())
    if not blocks:
        warn_patch_stack("Reference step cache found no observable blocks; original MODEL retained")
        return model, positive, {"schema": SCHEMA, "status": "unknown_original_delegated", "portable_cache_identity": False}
    previous = existing.previous if existing is not None else selected.model_options.get("model_function_wrapper")
    if previous is not None and not callable(previous):
        raise TypeError("Existing MODEL function wrapper is not callable")
    seal = digest({"plan": asdict(plan), "phase": phase, "positive": original_identity,
                   "request_owner": uuid.uuid4().hex})
    output = []
    for embedding, metadata in positive:
        conditions = dict(metadata.get("model_conds", {}))
        if PAIR_KEY in conditions:
            raise ValueError("Reference step cache cannot overwrite an existing MODEL/positive binding")
        conditions[PAIR_KEY] = comfy.conds.CONDConstant(seal)
        output.append([embedding, {**metadata, "model_conds": conditions}])
    from comfy import ops
    from comfy.ldm.modules import attention
    from torch.nn.modules import module as module_owner, normalization, linear
    from torch.nn import functional
    implementation = {str(Path(path).resolve()): hashlib.sha256(Path(path).read_bytes()).hexdigest()
                      for path in (__file__, primitive.__file__, primitive.core.__file__, ops.__file__,
                                   attention.__file__, module_owner.__file__, normalization.__file__, linear.__file__,
                                   functional.__file__)}
    runtime = _Runtime(plan, phase, output, condition_identity(output), seal, network, previous, implementation,
                       digest({"plan": asdict(plan), "phase": phase}), _helpers())

    def wrapper(apply_model, args):
        with runtime.lock:
            token, failures, completed = None, runtime.failed, runtime.completed
            try:
                runtime.verify()
                binding = args["c"].get(PAIR_KEY)
                roles = args.get("cond_or_uncond", ())
                negative = binding is None and bool(roles) and all(type(role) is int and role == 1 for role in roles)
                if not negative and binding != seal:
                    raise ValueError("Reference step cache MODEL/positive pair binding differs")
                token = _PAIR.set((runtime, not negative))
                c = dict(args["c"])
                c.pop(PAIR_KEY, None)
                result = (apply_model(args["input"], args["timestep"], **c) if previous is None
                          else previous(apply_model, {**args, "c": c}))
                runtime.verify()
                if runtime.completed == completed:
                    runtime.reason("diffusion_wrapper_bypassed_no_consumption")
                return result
            except BaseException as error:
                if runtime.failed == failures:
                    runtime.failed += 1
                    runtime.errors.append(type(error).__name__ + ": " + str(error))
                runtime.buckets.clear()
                raise
            finally:
                if token is not None:
                    _PAIR.reset(token)

    def diffusion_wrapper(executor, *args, **kwargs):
        paired_scope = _PAIR.get()
        if paired_scope is None or paired_scope[0] is not runtime:
            runtime.reason("unpaired_diffusion_original_delegated")
            return executor(*args, **kwargs)
        if len(args) < 4:
            runtime.reason("unknown_diffusion_call_signature_original_delegated")
            return executor(*args, **kwargs)
        with runtime.lock:
            token = None
            try:
                call = runtime.begin({"unpaired_negative": not paired_scope[1], "c": {PAIR_KEY: seal, "c_crossattn": args[2],
                    "transformer_options": args[3], "minimax_payload": kwargs.get("minimax_payload")}})
                token = _CALL.set((runtime, call))
                result = executor(*args, **kwargs)
                runtime.finish(call)
                return result
            except BaseException as error:
                runtime.failed += 1
                runtime.buckets.clear()
                runtime.errors.append(type(error).__name__ + ": " + str(error))
                raise
            finally:
                if token is not None:
                    _CALL.reset(token)

    runtime.wrapper = wrapper
    runtime.diffusion_wrapper = diffusion_wrapper
    values = selected.wrappers.get(WrappersMP.DIFFUSION_MODEL, {}).get(KEY, [])
    old = existing.diffusion_wrapper if existing is not None else None
    if old is not None and any(value is old for value in values):
        selected.wrappers[WrappersMP.DIFFUSION_MODEL][KEY] = [
            diffusion_wrapper if value is old else value for value in values]
    else:
        selected.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, KEY, diffusion_wrapper)
    current = dict(selected.model_options.get("transformer_options", {}).get("patches_replace", {}).get("dit", {}))
    for index, block in enumerate(blocks):
        previous_block = current.get(("double_block", index))
        if existing is not None and previous_block is existing.blocks.get(index):
            # Our old slot had no foreign predecessor, otherwise its original
            # predecessor is retained inside that old hook. Delegate it whole.
            previous_block = getattr(previous_block, "_reference_step_previous", previous_block)
        if previous_block is not None and not callable(previous_block):
            raise TypeError("Existing DiT block owner is not callable")
        def patch(args, extra, index=index, block=block, previous_block=previous_block):
            return runtime.block(index, block, previous_block, args, extra)
        patch._reference_step_previous = previous_block
        runtime.blocks[index] = patch
        selected.set_model_patch_replace(patch, "dit", "double_block", index)
    selected.set_model_unet_function_wrapper(wrapper)
    selected.set_attachments(KEY, runtime)
    runtime.prepare_callback, runtime.cleanup_callback = runtime.prepare, runtime.cleanup
    for kind, callback, attr in ((CallbacksMP.ON_PRE_RUN, runtime.prepare_callback, "prepare_callback"),
                                 (CallbacksMP.ON_CLEANUP, runtime.cleanup_callback, "cleanup_callback")):
        old = getattr(existing, attr, None)
        values = selected.callbacks.get(kind, {}).get(KEY, [])
        if old is not None and any(value is old for value in values):
            selected.callbacks[kind][KEY] = [callback if value is old else value for value in values]
        else:
            selected.add_callback_with_key(kind, KEY, callback)
    return selected, output, audit_reference_step_cache(selected)


def audit_reference_step_cache(model):
    getter = getattr(model, "get_attachment", None)
    runtime = getter(KEY) if callable(getter) else None
    if type(runtime) is not _Runtime:
        return {"schema": SCHEMA, "status": "not_selected", "portable_cache_identity": False}
    runtime.verify()
    present = model.model_options.get("model_function_wrapper") is runtime.wrapper
    diffusion_present = any(value is runtime.diffusion_wrapper for value in
        model.get_wrappers(WrappersMP.DIFFUSION_MODEL, KEY))
    hooks = model.model_options.get("transformer_options", {}).get("patches_replace", {}).get("dit", {})
    hooks_present = all(hooks.get(("double_block", index)) is owner for index, owner in runtime.blocks.items())
    partial = not present or not diffusion_present or not hooks_present or bool(runtime.reasons) or runtime.ambiguous
    return deepcopy({"schema": SCHEMA, "status": "failed" if runtime.failed else
        "observed_partial" if runtime.completed and partial else "observed_approximate" if runtime.hits else
        "observed_full_recompute" if runtime.completed else "selected_not_executed",
        "partial": partial, "approximate_algorithm": True, "bit_exact_claimed": False,
        "portable_cache_identity": False, "GPU_performance_qualified": False, "quality_accepted": False,
        "plan": asdict(runtime.plan), "phase": runtime.phase, "binding_sha256": runtime.seal,
        "selected_condition_identity": runtime.positive_identity,
        "selected_wrapper_present": present, "selected_block_owners_present": hooks_present,
        "selected_diffusion_wrapper_present": diffusion_present,
        "request_number": runtime.request, "request_active": runtime.active,
        "cleanup_observed": runtime.cleanup_observed, "cached_bytes_after_request": sum(
            value[0].numel() * value[0].element_size() for bucket in runtime.buckets.values() for value in bucket["blocks"].values()),
        "peak_CPU_cache_bytes": runtime.peak_bytes, "completed_model_evaluations": runtime.completed,
        "completed_paired_model_evaluations": sum(event["condition_scope"] == "paired" for event in runtime.evaluations),
        "completed_unpaired_negative_evaluations": sum(event["condition_scope"] == "unpaired_negative" for event in runtime.evaluations),
        "failed_model_evaluations": runtime.failed, "full_block_evaluations": runtime.full,
        "approximate_block_hits": runtime.hits, "reference_query_outproj_MLP_rows_skipped": runtime.skipped_rows,
        "evaluations": runtime.evaluations, "reasons": runtime.reasons, "errors": runtime.errors,
        "selected_live_weight_sha256": runtime.weights,
        "opaque_internal_state_verified": False, "implementation": runtime.implementation})
