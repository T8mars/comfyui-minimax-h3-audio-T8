"""Opt-in MODEL selection for local INT8 scheduling, not a cache receipt."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import json
import threading

import torch
from comfy.patcher_extension import CallbacksMP, WrappersMP

from .int8_row_chunk import Int8RowPlan, int8_row_scope
from .patch_stack_policy import warn_patch_stack

KEY = "t8_h3_int8_row_chunk_v1"
SCHEMA = "t8.h3.int8_row_model.v1"


def _binding(plan, phase):
    return json.dumps(
        {"plan": asdict(plan), "phase": phase},
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
    )


@dataclass
class _Observation:
    lock: object = field(default_factory=threading.RLock)
    epoch: int = 0
    active: bool = False
    active_calls: int = 0
    explicit_pre_run: bool = False
    cleanup_seen: bool = False
    ambiguous: bool = False
    completed: int = 0
    failed: int = 0
    calls: dict = field(default_factory=dict)
    delegated: set = field(default_factory=set)
    errors: list = field(default_factory=list)
    last: dict | None = None

    def reset(self, explicit):
        self.epoch += 1
        self.active = True
        self.explicit_pre_run = explicit
        self.cleanup_seen = False
        self.ambiguous = bool(self.active_calls)
        self.completed = self.failed = 0
        self.calls, self.delegated, self.errors, self.last = {}, set(), [], None

    def pre_run(self, *_):
        with self.lock:
            self.reset(True)

    def enter(self):
        with self.lock:
            if not self.active:
                self.reset(False)
            self.active_calls += 1
            self.ambiguous |= self.active_calls > 1
            return self.epoch

    def finish(self, epoch, runtime, error):
        with self.lock:
            self.active_calls -= 1
            self.ambiguous |= epoch != self.epoch
            if error is None:
                self.completed += 1
            else:
                self.failed += 1
                self.errors.append(type(error).__name__ + ": " + str(error))
            if runtime is None:
                self.delegated.add(
                    "network_api_unrecognized_original_executor_retained"
                )
                return
            value = runtime.report()
            self.last = deepcopy(value)
            self.delegated.update(value["delegated"])
            for path, item in value["calls"].items():
                result = self.calls.setdefault(
                    path, {"calls": 0, "rows": 0, "max_rows": 0, "chunks": 0}
                )
                for key in ("calls", "rows", "chunks"):
                    result[key] += item[key]
                result["max_rows"] = max(result["max_rows"], item["max_rows"])

    def cleanup(self, *_):
        with self.lock:
            self.active = False
            self.cleanup_seen = True


@dataclass(frozen=True)
class _Selection:
    plan: Int8RowPlan
    phase: str
    binding: str
    observation: _Observation
    wrapper: object
    prepare: object
    cleanup: object


def _selection(model):
    getter = getattr(model, "get_attachment", None)
    return getter(KEY) if callable(getter) else None


def _validate(selection):
    if (
        type(selection.plan) is not Int8RowPlan
        or _binding(selection.plan, selection.phase) != selection.binding
    ):
        raise ValueError(
            "INT8 row configuration binding changed after explicit selection"
        )


def configure_int8_row_chunks(model, plan=None, *, phase="SINGLE", enabled=True):
    if type(enabled) is not bool:
        raise ValueError("INT8 row enabled must be boolean")
    if not enabled:
        return model, {
            "schema": SCHEMA,
            "status": "disabled",
            "portable_cache_identity": False,
        }
    plan = Int8RowPlan() if plan is None else plan
    if type(plan) is not Int8RowPlan:
        raise TypeError("INT8 row configuration needs Int8RowPlan")
    if not plan.rows:
        return model, {
            "schema": SCHEMA,
            "status": "disabled",
            "portable_cache_identity": False,
        }
    if type(phase) is not str or not phase.strip():
        raise ValueError("INT8 row phase must be a nonempty label")
    if not all(
        callable(getattr(model, name, None))
        for name in (
            "clone",
            "get_attachment",
            "set_attachments",
            "add_wrapper_with_key",
            "add_callback_with_key",
        )
    ):
        raise TypeError(
            "INT8 row MODEL needs clone, attachment, diffusion-wrapper and lifecycle APIs"
        )
    previous = _selection(model)
    if previous is not None and type(previous) is not _Selection:
        warn_patch_stack(
            "Existing INT8 row attachment has a different owner; preserving the original MODEL"
        )
        return model, {
            "schema": SCHEMA,
            "status": "not_installed_existing_owner_preserved",
            "partial": True,
            "portable_cache_identity": False,
        }
    selected = model.clone()
    if previous is not None:
        _validate(previous)
    observation = _Observation()
    sealed = _binding(plan, phase)

    def wrapper(executor, *args, **kwargs):
        if _binding(plan, phase) != sealed:
            raise ValueError("INT8 row binding changed before diffusion execution")
        epoch, runtime, error = observation.enter(), None, None
        network = getattr(executor, "class_obj", None)
        try:
            if isinstance(network, torch.nn.Module):
                with int8_row_scope(network, plan, phase=phase) as runtime:
                    result = executor(*args, **kwargs)
            else:
                result = executor(*args, **kwargs)
            if _binding(plan, phase) != sealed:
                raise ValueError("INT8 row binding changed during diffusion execution")
            return result
        except BaseException as caught:
            error = caught
            raise
        finally:
            observation.finish(epoch, runtime, error)

    prepare, cleanup = observation.pre_run, observation.cleanup
    selection = _Selection(plan, phase, sealed, observation, wrapper, prepare, cleanup)
    for group, kind, replacement, attribute, add in (
        (selected.wrappers, WrappersMP.DIFFUSION_MODEL, wrapper, "wrapper", selected.add_wrapper_with_key),
        (selected.callbacks, CallbacksMP.ON_PRE_RUN, prepare, "prepare", selected.add_callback_with_key),
        (selected.callbacks, CallbacksMP.ON_CLEANUP, cleanup, "cleanup", selected.add_callback_with_key),
    ):
        old = getattr(previous, attribute, None)
        values = group.get(kind, {}).get(KEY, [])
        if old is not None and any(value is old for value in values):
            group[kind][KEY] = [replacement if value is old else value for value in values]
        else:
            add(kind, KEY, replacement)
    selected.set_attachments(KEY, selection)
    return selected, audit_int8_row_chunks(selected)


def audit_int8_row_chunks(model):
    selection = _selection(model)
    if type(selection) is not _Selection:
        return {
            "schema": SCHEMA,
            "status": "not_selected" if selection is None else "unverified_attachment",
            "portable_cache_identity": False,
            "quality_accepted": False,
        }
    _validate(selection)
    state = selection.observation
    with state.lock:
        present = any(
            value is selection.wrapper
            for value in model.get_wrappers(WrappersMP.DIFFUSION_MODEL, KEY)
        )
        partial = bool(state.delegated) or state.ambiguous or not present
        status = (
            "failed"
            if state.failed
            else "observed_partial"
            if state.completed and partial
            else "observed"
            if state.completed and state.calls
            else "observed_no_eligible_calls"
            if state.completed
            else "selected_not_executed"
        )
        return deepcopy(
            {
                "schema": SCHEMA,
                "status": status,
                "phase": selection.phase,
                "plan": asdict(selection.plan),
                "request_number": state.epoch,
                "request_active": state.active,
                "explicit_pre_run_observed": state.explicit_pre_run,
                "cleanup_observed": state.cleanup_seen,
                "completed_diffusion_calls": state.completed,
                "failed_diffusion_calls": state.failed,
                "calls": state.calls,
                "delegated": sorted(state.delegated),
                "errors": state.errors,
                "partial": partial,
                "selected_wrapper_present": present,
                "lifecycle_ambiguous": state.ambiguous,
                "last_forward": state.last,
                "portable_cache_identity": False,
                "quality_accepted": False,
                "allocator_peak_measured": False,
                "performance_qualified": False,
                "count_boundary": "Actual wrapper/operator invocation counts, not NFE or sampler completion",
                "memory_boundary": "Full outputs and full K/V retained; reported row/byte counts are not allocator, weight or host RAM peaks",
            }
        )
