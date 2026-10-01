"""Authenticate Progressive effect owners on an inspection-only MODEL clone.

Counters are not numerical inputs. Actual factory closures, paired condition
bindings, masks and selected delegates are; no live MODEL is unpatched here.
"""
import hashlib
import inspect
import json
from pathlib import Path
from types import CodeType, FunctionType

import comfy.model_patcher

from .. import enhance_a_video_advanced as eav
from .. import progressive_eav, progressive_relay
from .. import prompt_relay_advanced as relay
from ..relay_kj_memory import bind_memory_runtime
from ..h3_core_compat import _set_legacy_attention_backend, plain_attention_backend
from ..long_video_dual_identity import content_identity
from ..progressive_attention import _PlainDelegate
from ..vdn_attention_compat import _factory_closure
from .effect_identity import _require, _mask_identity


def _code_identity(code):
    def constant(value):
        if type(value) is CodeType:
            return {"code": _code_identity(value)}
        if type(value) in (tuple, frozenset):
            items = [constant(item) for item in value]
            if type(value) is frozenset:
                items.sort(key=lambda item: json.dumps(item, sort_keys=True))
            return {"type": type(value).__name__, "items": items}
        _require(type(value) in (str, bytes, int, float, complex, bool, type(None), type(Ellipsis)),
                 "Progressive method contains an unknown code constant")
        return {"type": type(value).__name__, "value": repr(value)}
    # marshal also records object-sharing/interning flags; those can change
    # after an ordinary first call without a code change. Bind immutable code
    # values instead, including nested code and the exception table.
    return {"bytecode": code.co_code.hex(), "exceptions": getattr(code, "co_exceptiontable", b"").hex(),
            "args": [code.co_argcount, code.co_posonlyargcount, code.co_kwonlyargcount, code.co_flags],
            "names": code.co_names, "locals": code.co_varnames,
            "free": code.co_freevars, "cells": code.co_cellvars,
            "constants": [constant(value) for value in code.co_consts]}


def function_identity(function):
    _require(type(function) is FunctionType, "Progressive class method is not a source function")
    closure = {}
    for name, cell in zip(function.__code__.co_freevars, function.__closure__ or ()):
        value = cell.cell_contents
        # Core operation forwards use Python's zero-argument super() cell.
        # This is the real declaring class, not an arbitrary captured callable.
        _require(name == "__class__" and isinstance(value, type)
                 and any(item is function for item in vars(value).values()),
                 "Progressive class method has an unknown executable closure")
        source = inspect.getsourcefile(value)
        _require(source is not None, "Progressive declaring class has no source identity")
        closure[name] = {"class": value.__module__ + "." + value.__qualname__,
                         "source_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest()}
    code = json.dumps(_code_identity(function.__code__), sort_keys=True, separators=(",", ":"))
    return {"code_sha256": hashlib.sha256(code.encode()).hexdigest(),
            "closure": closure,
            "defaults": content_identity(function.__defaults__),
            "kwdefaults": content_identity(function.__kwdefaults__)}


def _class_identity(kind):
    return {name: function_identity(value) for name, value in vars(kind).items() if type(value) is FunctionType}


def _selected(selector):
    _require(getattr(selector, "container_function", None) is None,
             "Progressive effects selector has another container owner")
    for factory in (getattr(comfy.model_patcher.ModelPatcher, "set_model_optimized_attention", None),
                    _set_legacy_attention_backend):
        state = _factory_closure(selector, factory, "optimized_attention_override")
        if state is not None:
            _require(set(state) == {"optimized_attention"}, "Unknown Core selector execution state")
            return state["optimized_attention"]
    _require(False, "Progressive effects selector is not its actual Core factory")


def _delegate(backend, source=None):
    if backend is None:
        _require(source is None or plain_attention_backend(source) is not None,
                 "Progressive Relay source selector lacks its identity adapter")
        return source
    _require(type(backend) is _PlainDelegate and set(vars(backend)) == {
        "override", "name", "calls", "masked_delegate"},
        "Progressive effect delegate needs a persistent identity adapter")
    _require(backend.name != "sage",
             "Progressive lazy biased Sage delegate needs its persistent identity adapter")
    _require(backend.masked_delegate is None,
             "Progressive biased Sage delegate needs its persistent identity adapter")
    _require(backend.name == plain_attention_backend(backend.override) and backend.name is not None,
             "Progressive plain delegate differs from its selected backend")
    _require(source is None or source is backend.override, "Progressive Relay source/delegate differ")
    return backend.override


def _binding(binding, report, expected_hash):
    _require(type(binding) is dict, "Progressive Relay binding is not a content descriptor")
    unsigned = dict(binding)
    claimed = unsigned.pop("binding_hash", None)
    _require(claimed == expected_hash == relay._sha256_json(unsigned), "Progressive Relay binding changed")
    for report_key, binding_key in (("binding_hash", "binding_hash"), ("plan_hash", "plan_hash"),
                                   ("layout", "layout_contract"), ("query_route", "query_route"), ("events", "events")):
        _require(report[report_key] == binding[binding_key], "Progressive Relay report and execution binding differ")
    return content_identity(binding)


def _observer(function, host, *, combined):
    factory = progressive_eav.prepare_progressive_eav if combined else progressive_relay.install_relay_stage
    state = _factory_closure(function, factory, "observe")
    expected = "relay_report" if combined else "counts"
    value = host.relay_report if combined else host.relay_report["completed_calls"]
    _require(state is not None and set(state) == {expected} and state[expected] is value,
             "Progressive effect observer is not paired with its exact stage report")


def _standalone_relay(model, host, selector):
    key = relay.PROMPT_RELAY_WRAPPER_KEY
    wrappers = model.get_wrappers("diffusion_model", key)
    _require(len(wrappers) == 1, "Progressive Relay wrapper count changed")
    state = _factory_closure(wrappers[0], relay._install_prompt_relay_model, "_diffusion_wrapper")
    route = _factory_closure(_selected(selector), relay._install_prompt_relay_model, "_attention_router")
    _require(state is not None and route is not None, "Progressive Relay factory owner changed")
    contract = relay.prompt_relay_model_contract(model)
    _require(contract["attention_owner_verified"]
             and state["installed_override"] is selector, "Progressive Relay owner changed")
    for name in ("relay_backend", "execution_observer", "execution_counts"):
        _require(state[name] is route[name], "Progressive Relay wrapper/router ownership differs")
    _observer(state["execution_observer"], host, combined=False)
    _require(route["query_chunk_rows"] == contract["query_chunk_rows"], "Progressive Relay query chunk changed")
    _require(state["binding"] == contract["binding"], "Progressive Relay MODEL binding differs")
    _require(state["bind_memory_runtime"] is bind_memory_runtime, "Progressive Relay memory binder changed")
    source = _delegate(state["relay_backend"], state["source_plain_override"])
    binding = _binding(state["binding"], host.relay_report, state["expected_hash"])
    # Only diagnostic counters and wrapper-chain provenance are omitted. The
    # clone keeps every other wrapper so unknown user additions remain opaque.
    ignored = {"relay_backend", "execution_observer", "execution_counts", "installed_override",
               "expected_diffusion_wrappers", "source_plain_override", "bind_memory_runtime"}
    return key, source, {"binding": binding, "query_chunk_rows": route["query_chunk_rows"],
                        "execution": content_identity({k: v for k, v in state.items() if k not in ignored})}


def _eav(model, host, selector):
    combined = host.relay_report is not None
    key = eav.EAV_PROMPT_RELAY_WRAPPER_KEY if combined else eav.EAV_WRAPPER_KEY
    factory = eav.build_eav_prompt_relay_model if combined else eav.build_eav_model
    wrappers = model.get_wrappers("diffusion_model", key)
    _require(len(wrappers) == 1, "Progressive EAV wrapper count changed")
    state = _factory_closure(wrappers[0], factory, "_combined_wrapper" if combined else "_diffusion_wrapper")
    _require(state is not None and state["runtime"] is host.eav_runtime
             and state["progressive_mask_contract"] is host.mask,
             "Progressive EAV wrapper is not paired with its runtime and mask")
    runtime = host.eav_runtime
    _require(type(runtime) is eav.EAVRuntime and set(vars(runtime)) == {
        "config", "_lock", "_run_index", "_consumed", "_aborted", "_forwards"},
        "Progressive EAV runtime has unknown executable state")
    target = _selected(selector)
    if combined:
        route = _factory_closure(target, factory, "_combined_attention")
        _require(route is not None and state["installed"] is selector
                 and route["relay"] is state["relay"]
                 and route["execution_observer"] is state["execution_observer"],
                 "Progressive Relay/EAV router differs from its wrapper")
        _observer(state["execution_observer"], host, combined=True)
        _require(state["_prompt_relay_runtime_route"] is relay._runtime_route,
                 "Progressive combined Relay route changed")
        contract = state["relay"]
        _require(contract["attention_owner_verified"] and contract["binding"] == state["binding"],
                 "Progressive combined Relay binding differs")
        source = _delegate(contract["attention_backend"], contract["source_plain_override"])
        relay_identity = {"binding": _binding(state["binding"], host.relay_report, state["expected_hash"]),
                          "query_chunk_rows": contract["query_chunk_rows"]}
    else:
        _require(state["bind_memory_runtime"] is bind_memory_runtime, "Progressive EAV memory binder changed")
        _require(state["expected_override"] is selector and state["config"] is runtime.config,
                 "Progressive EAV configuration/selector ownership differs")
        backend = state["composed_backend"]
        if backend is None:
            _require(target is eav.route_eav_attention, "Progressive EAV route changed")
        else:
            route = _factory_closure(target, factory, "composed_attention")
            _require(route is not None and route.get("composed_backend") is backend,
                     "Progressive EAV delegate is not paired with its wrapper")
        source = _delegate(backend)
        relay_identity = None
    ignored = {"runtime", "progressive_mask_contract", "installed", "expected_override", "config",
               "composed_backend", "relay", "execution_observer", "bind_memory_runtime", "_prompt_relay_runtime_route"}
    accepted = host.continuation.contexts.source if host.continuation is not None else None
    return key, source, {"mask": _mask_identity(host.mask, accepted_source=accepted), "relay": relay_identity,
                        "execution": content_identity({k: v for k, v in state.items() if k not in ignored})}


def project(model):
    from .progressive_effects import KEY, StageEffects
    host = model.get_attachment(KEY)
    if host is None:
        return model, None
    _require(type(host) is StageEffects and set(vars(host)) == {
        "plan", "phase", "relay_report", "eav_runtime", "mask", "continuation", "lock", "last_report", "binding"},
        "Progressive stage effects owner has unknown executable state")
    host.verify()
    selector = model.model_options.get("transformer_options", {}).get("optimized_attention_override")
    if host.eav_runtime is not None:
        key, source, execution = _eav(model, host, selector)
    else:
        _require(host.relay_report is not None and host.mask is None, "Empty Progressive effects owner")
        key, source, execution = _standalone_relay(model, host, selector)
    clone = model.clone()
    clone.remove_wrappers_with_key("diffusion_model", key)
    if not clone.wrappers.get("diffusion_model"):
        clone.wrappers.pop("diffusion_model", None)
    clone.remove_attachments(key)
    clone.remove_attachments(KEY)
    if host.continuation is not None and host.relay_report is not None:
        from ..prompt_relay_long_video_advanced import PROMPT_RELAY_LONG_VIDEO_ATTACHMENT_KEY
        from .continuation_effects import relay_attachment
        expected = relay_attachment(host.continuation, host.relay_report)
        _require(model.get_attachment(PROMPT_RELAY_LONG_VIDEO_ATTACHMENT_KEY) == expected,
                 "Continuation Relay attachment differs from actual stage execution")
        clone.remove_attachments(PROMPT_RELAY_LONG_VIDEO_ATTACHMENT_KEY)
    options = clone.model_options["transformer_options"]
    if source is None:
        options.pop("optimized_attention_override", None)
    else:
        options["optimized_attention_override"] = source
    return clone, {"schema": "t8.modular-sampling.progressive-effect-identity.v1",
                   "stage": content_identity(host.descriptor()), "execution": execution,
                   "class_execution": {"stage": _class_identity(StageEffects), "eav": _class_identity(eav.EAVRuntime),
                       "mask": _class_identity(type(host.mask)) if host.mask is not None else None,
                       "plain_delegate": _class_identity(_PlainDelegate)}}
