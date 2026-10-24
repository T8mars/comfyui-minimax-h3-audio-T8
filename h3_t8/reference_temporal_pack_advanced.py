"""Paired MODEL/positive reference packing without shared-network mutation.

The legacy model-function wrapper boundary precedes DIFFUSION_MODEL wrappers,
so an authenticated Relay can bind the actual physical packed layout too.
Unknown callable owners remain executable and never receive portable identity.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import inspect
import hashlib
import math
from pathlib import Path
import threading
from types import CodeType, FunctionType

import torch
import comfy.conds
from comfy.ldm.minimax import model as native
from comfy.patcher_extension import CallbacksMP, WrappersMP

from .patch_stack_policy import warn_patch_stack
from .reference_temporal_pack import (
    KEY, PAIR_KEY, SCHEMA, condition_identity, digest, pack_conditioning, repair_payload, verify_ref,
)
from .source_code_identity import executable_code_equal


_SOURCE_CODES = {}
_SOURCE_LOCK = threading.RLock()


def _codes(code):
    yield code
    for value in code.co_consts:
        if type(value) is CodeType:
            yield from _codes(value)


def _native_source_current(function, qualname):
    function = getattr(function, "__func__", function)
    if type(function) is not FunctionType or function.__globals__ is not vars(native):
        return False
    path = inspect.getsourcefile(function)
    if path is None or Path(path).resolve() != Path(native.__file__).resolve():
        return False
    source = Path(path).read_bytes()
    source_key = (path, hashlib.sha256(source).hexdigest())
    with _SOURCE_LOCK:
        codes = _SOURCE_CODES.get(source_key)
        if codes is None:
            codes = {code.co_qualname: code for code in _codes(compile(source, path, "exec", dont_inherit=True))}
            _SOURCE_CODES[source_key] = codes
    expected = codes.get(qualname)
    return executable_code_equal(function.__code__, expected)


def _capable(model):
    getter = getattr(model, "get_model_object", None)
    if not callable(getter):
        return False
    try:
        network = getter("diffusion_model")
    except (AttributeError, KeyError):
        return False
    helpers = ("_frame_grid", "_axis_from_sqrt_area", "_ref_t_span", "_audio_grid",
               "_video_grid", "_video_t_grid", "_video_t_spans")
    return (type(native.FRAME_PER_TOKEN) is tuple
            and all(type(value) is int for value in native.FRAME_PER_TOKEN)
            and native.FRAME_PER_TOKEN == (1, 4, 4, 4, 4)
            and type(native.FRAME_RESCALE) is float and native.FRAME_RESCALE == 5. / 3.
            and native.torch is torch and native.math is math
            and _native_source_current(getattr(network, "_forward", None), "MiniMaxH3Model._forward")
            and _native_source_current(native.PackedLayout.__init__, "PackedLayout.__init__")
            and all(_native_source_current(getattr(native, name, None), name) for name in helpers))


@dataclass
class _Runtime:
    contract: dict
    seal: str
    previous: object
    lock: object = field(default_factory=threading.RLock)
    wrapper: object = None
    request: int = 0
    active: bool = False
    completed: int = 0
    failed: int = 0
    last_layout: dict | None = None
    cleanup_observed: bool = False
    errors: list = field(default_factory=list)
    positive: object = None
    positive_identity: dict | None = None
    implementation: dict = field(default_factory=dict)
    lifecycle_ambiguous: bool = False

    def verify(self):
        if digest(self.contract) != self.seal:
            raise ValueError("Reference temporal-pack MODEL binding SHA changed")
        if any(hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected
               for path, expected in self.implementation.items()):
            raise ValueError("Reference temporal-pack execution source SHA changed")
        if self.positive_identity is not None:
            if condition_identity(self.positive) != self.positive_identity:
                raise ValueError("Reference temporal-pack selected conditioning content SHA changed")

    def prepare(self, *_):
        with self.lock:
            self.verify()
            overlapping = self.active
            self.request += 1
            self.active, self.cleanup_observed = True, False
            self.completed = self.failed = 0
            self.last_layout, self.errors = None, []
            self.lifecycle_ambiguous = overlapping

    def cleanup(self, *_):
        with self.lock:
            self.active, self.cleanup_observed = False, True


def _relay_rebind(model, original, packed):
    from . import prompt_relay_advanced as relay
    bindings = [meta.get(relay.PROMPT_RELAY_BINDING_KEY) for _emb, meta in original]
    if not any(value is not None for value in bindings):
        return model, packed, False
    contract = relay.prompt_relay_model_contract(model)
    if not contract["attention_owner_verified"]:
        return None, None, False
    bound = contract["binding"]
    if any(value is None or value.get("binding_hash") != bound["binding_hash"] for value in bindings):
        raise ValueError("Reference pack MODEL/positive do not have the same authentic Relay binding")
    signature = tuple(bound["layout_contract"]["signature"])
    meta = original[0][1]
    original_layout = native.PackedLayout(*signature, keyframes=meta.get("minimax_keyframes"),
                                         refs=meta.get("minimax_refs"))
    if relay._layout_contract(original_layout) != bound["layout_contract"]:
        # An unrecognized positional composer remains usable unchanged. It must
        # not get guessed new event offsets or a falsely authenticated binding.
        return None, None, False
    pmeta = packed[0][1]
    payload = {"keyframes": pmeta.get("minimax_keyframes"), "refs": pmeta.get("minimax_refs")}
    repaired, _ = repair_payload(payload, signature)
    replacement = relay._bind_layout_contract(bound, repaired["layout"], resolved_task=bound["task"],
                                               keyframes=payload["keyframes"], refs=payload["refs"])
    clone = model.clone()
    candidates = clone.get_wrappers(WrappersMP.DIFFUSION_MODEL, relay.PROMPT_RELAY_WRAPPER_KEY)
    from .vdn_attention_compat import _factory_closure
    own = [value for value in candidates if _factory_closure(value, relay._install_prompt_relay_model,
                                                             "_diffusion_wrapper") is not None]
    if len(own) != 1:
        return None, None, False
    clone.wrappers[WrappersMP.DIFFUSION_MODEL][relay.PROMPT_RELAY_WRAPPER_KEY] = [
        value for value in candidates if value is not own[0]]
    transformer = clone.model_options["transformer_options"]
    source = contract.get("source_plain_override")
    if source is None:
        transformer.pop("optimized_attention_override", None)
    else:
        transformer["optimized_attention_override"] = source
    clone, _ = relay._install_prompt_relay_model(
        clone, replacement, contract["query_chunk_rows"], contract["core_hashes"],
        attention_backend=contract.get("attention_backend"))
    installed = clone.get_wrappers(WrappersMP.DIFFUSION_MODEL, relay.PROMPT_RELAY_WRAPPER_KEY)
    new_owners = [value for value in installed if not any(value is old for old in candidates)]
    if len(new_owners) != 1 or _factory_closure(new_owners[0], relay._install_prompt_relay_model,
                                               "_diffusion_wrapper") is None:
        raise ValueError("Reference pack could not bind the newly installed Relay owner")
    # The public factory appends. Restore the exact previous slot on this new
    # clone so a same-key user wrapper is not silently moved across Relay's
    # runtime boundary. Do not edit the old callable, its closure or any foreign
    # owner; the factory's changed-chain advisory remains conservative.
    clone.wrappers[WrappersMP.DIFFUSION_MODEL][relay.PROMPT_RELAY_WRAPPER_KEY] = [
        new_owners[0] if value is own[0] else value for value in candidates]
    result = []
    for embedding, metadata in packed:
        models = dict(metadata.get("model_conds", {}))
        models[relay.PROMPT_RELAY_PAYLOAD_KEY] = comfy.conds.CONDConstant(replacement["binding_hash"])
        result.append([embedding, {**metadata, relay.PROMPT_RELAY_BINDING_KEY: replacement, "model_conds": models}])
    return clone, result, True


def configure_reference_temporal_pack(model, positive, ratio="0/5", *, phase="SINGLE"):
    if ratio == "0/5":
        packed, contract = pack_conditioning(positive, ratio, phase=phase)
        return model, packed, contract
    from .reference_temporal_pack import retained_indices
    retained_indices(1, ratio)  # Validate the explicit option, not an unknown MODEL's tensors.
    if not _capable(model):
        warn_patch_stack("Reference temporal pack left unknown MODEL/layout computation unchanged; compression not applied")
        return model, positive, {"schema": SCHEMA, "status": "unknown_original_delegated",
                                 "portable_cache_identity": False, "compression_applied": False}
    packed, contract = pack_conditioning(positive, ratio, phase=phase)
    if packed is positive:
        return model, positive, contract
    if not all(callable(getattr(model, name, None)) for name in (
            "clone", "get_attachment", "set_attachments", "add_callback_with_key",
            "set_model_unet_function_wrapper")):
        raise TypeError("Reference packing MODEL requires clone/attachment/lifecycle/model-wrapper APIs")
    existing = model.get_attachment(KEY)
    if existing is not None:
        warn_patch_stack("Reference temporal pack already has an owner; original MODEL/positive retained")
        return model, positive, {"schema": SCHEMA, "status": "existing_owner_original_retained",
                                 "portable_cache_identity": False, "compression_applied": False}
    selected, packed, relay = _relay_rebind(model, positive, packed)
    if selected is None:
        warn_patch_stack("Reference temporal pack retained unknown Relay/position owner unchanged; compression not applied")
        return model, positive, {"schema": SCHEMA, "status": "unknown_relay_original_delegated",
                                 "portable_cache_identity": False, "compression_applied": False}
    selected = selected.clone()
    contract = deepcopy(contract)
    contract["source_pack_contract_sha256"] = contract.pop("sha256")
    contract.update({"phase": phase, "relay_rebound": relay,
                "runtime_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    )
    seal = digest(contract)
    previous = selected.model_options.get("model_function_wrapper")
    if previous is not None and not callable(previous):
        raise TypeError("Existing MODEL function wrapper is not callable")
    runtime = _Runtime(contract, seal, previous)
    from . import reference_temporal_pack as primitive
    runtime.implementation = {str(Path(path).resolve()): hashlib.sha256(Path(path).read_bytes()).hexdigest()
                              for path in (__file__, primitive.__file__, native.__file__)}

    def wrapper(apply_model, args):
        runtime.verify()
        if not _capable(selected):
            raise ValueError("Selected reference-pack native execution code changed during the request")
        c = dict(args["c"])
        if c.pop(PAIR_KEY, None) != seal:
            raise ValueError("Reference temporal-pack MODEL/positive pair binding differs")
        payload = c.get("minimax_payload")
        if type(payload) is not dict:
            raise ValueError("Reference temporal pack requires actual native minimax_payload")
        for ref in payload.get("refs") or []:
            if KEY in ref:
                verify_ref(ref)
        shapes = c.get("latent_shapes")
        if not isinstance(shapes, (list, tuple)) or len(shapes) != 2 or len(shapes[0]) != 5:
            raise ValueError("Reference temporal pack requires actual packed AV latent_shapes")
        video, audio = shapes
        context = c.get("c_crossattn")
        if type(context) is not torch.Tensor or context.ndim != 3:
            raise ValueError("Reference temporal pack requires actual scheduled text context")
        signature = (context.shape[1], video[2], (video[3] + 1) // 2 * 2,
                     (video[4] + 1) // 2 * 2, audio[-1])
        local_payload, report = repair_payload(payload, signature)
        report["boundary"] = "Physical layout prepared before MODEL delegation; no standalone GPU execution receipt"
        epoch = runtime.request
        c["minimax_payload"] = local_payload
        forwarded = {**args, "c": c}
        try:
            if previous is None:
                answer = apply_model(forwarded["input"], forwarded["timestep"], **c)
            else:
                answer = previous(apply_model, forwarded)
            runtime.verify()
            if not _capable(selected):
                raise ValueError("Selected reference-pack native execution code changed during the request")
            for ref in payload.get("refs") or []:
                if KEY in ref:
                    verify_ref(ref)
            with runtime.lock:
                runtime.lifecycle_ambiguous |= epoch != runtime.request
                runtime.completed += 1
                runtime.last_layout = report
            return answer
        except BaseException as error:
            with runtime.lock:
                runtime.lifecycle_ambiguous |= epoch != runtime.request
                runtime.failed += 1
                runtime.errors.append(type(error).__name__ + ": " + str(error))
            raise

    runtime.wrapper = wrapper
    selected.set_model_unet_function_wrapper(wrapper)
    selected.add_callback_with_key(CallbacksMP.ON_PRE_RUN, KEY, runtime.prepare)
    selected.add_callback_with_key(CallbacksMP.ON_CLEANUP, KEY, runtime.cleanup)
    selected.set_attachments(KEY, runtime)
    output = []
    for embedding, metadata in packed:
        models = dict(metadata.get("model_conds", {}))
        if PAIR_KEY in models:
            raise ValueError("Reference temporal pack cannot overwrite an existing pair binding")
        models[PAIR_KEY] = comfy.conds.CONDConstant(seal)
        output.append([embedding, {**metadata, "model_conds": models}])
    # Actual existing provider/Stage digesters also see all these values. An
    # opaque predecessor disables portable reuse; a descriptive producer label
    # never substitutes for actual selected tensors or execution ownership.
    runtime.positive = output
    runtime.positive_identity = condition_identity(output)
    return selected, output, audit_reference_temporal_pack(selected)


def audit_reference_temporal_pack(model):
    getter = getattr(model, "get_attachment", None)
    runtime = getter(KEY) if callable(getter) else None
    if type(runtime) is not _Runtime:
        return {"schema": SCHEMA, "status": "not_selected", "portable_cache_identity": False}
    runtime.verify()
    with runtime.lock:
        present = model.model_options.get("model_function_wrapper") is runtime.wrapper
        upstream_present = runtime.previous is not None
        partial = not present or runtime.lifecycle_ambiguous or upstream_present
        status = ("failed" if runtime.failed else "observed_partial" if runtime.completed and partial
                  else "observed" if runtime.completed else "selected_but_wrapper_replaced" if not present
                  else "selected_not_executed")
        return deepcopy({"schema": SCHEMA, "status": status, "partial": partial,
                         "binding_sha256": runtime.seal, "contract": runtime.contract,
                         "completed_model_calls": runtime.completed, "failed_model_calls": runtime.failed,
                         "request_number": runtime.request, "request_active": runtime.active,
                         "lifecycle_ambiguous": runtime.lifecycle_ambiguous,
                         "cleanup_observed": runtime.cleanup_observed, "errors": runtime.errors,
                         "last_layout": runtime.last_layout, "portable_cache_identity": False,
                         "packed_condition_identity": runtime.positive_identity,
                         "content_hashing_performed": True,
                         "selected_wrapper_present": present,
                         "upstream_model_wrapper_present": upstream_present,
                         "compression_consumption_certified": False,
                         "quality_accepted": False, "GPU_performance_qualified": False,
                         "shared_network_or_layout_modified": False})
