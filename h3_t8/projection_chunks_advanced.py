"""Clone-local MODEL selection for request-scoped projection working sets.

No shared module is modified by Apply. The actual diffusion executor supplies
the network at call time; its scoped primitives restore methods before return.
Reports describe observations, not an NFE, allocator, cache or quality receipt.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import json
import threading

import torch
from comfy.patcher_extension import CallbacksMP, WrappersMP

from .h3_projection_chunks import ProjectionPlan, projection_scope
from .patch_stack_policy import warn_patch_stack


KEY = "t8_h3_projection_chunks_v1"
SCHEMA = "t8.h3.projection_model.v1"


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _binding(plan, phase):
    return canonical({"plan": asdict(plan), "phase": phase})


@dataclass
class _Observation:
    phase: str
    lock: object = field(default_factory=threading.RLock)
    request_number: int = 0
    request_active: bool = False
    cleanup_observed: bool = False
    explicit_pre_run: bool = False
    completed: int = 0
    failed: int = 0
    calls: dict = field(default_factory=dict)
    delegated: set = field(default_factory=set)
    errors: list = field(default_factory=list)
    last_forward: dict | None = None
    lifecycle_ambiguous: bool = False

    def pre_run(self, *_):
        with self.lock:
            self._reset(explicit=True)

    def _reset(self, *, explicit):
        self.request_number += 1
        self.request_active = True
        self.cleanup_observed = False
        self.explicit_pre_run = explicit
        self.completed = self.failed = 0
        self.calls = {}
        self.delegated = set()
        self.errors = []
        self.last_forward = None
        self.lifecycle_ambiguous = False

    def enter(self):
        with self.lock:
            if not self.request_active:
                self._reset(explicit=False)
            return self.request_number

    def finish(self, epoch, report, error):
        with self.lock:
            if epoch != self.request_number:
                self.lifecycle_ambiguous = True
                self.delegated.add("overlapping_request_lifecycle_not_qualified")
            if error is None:
                self.completed += 1
            else:
                self.failed += 1
                self.errors.append(type(error).__name__ + ": " + str(error))
            if report is None:
                self.delegated.add("network_api_unrecognized_original_executor_retained")
                return
            self.last_forward = deepcopy(report)
            self.delegated.update(report["delegated"])
            for path, row in report["calls"].items():
                target = self.calls.setdefault(path, {"calls": 0, "rows": 0, "max_rows": 0, "chunks": 0})
                for name in ("calls", "rows", "chunks"):
                    target[name] += row[name]
                target["max_rows"] = max(target["max_rows"], row["max_rows"])

    def cleanup(self, *_):
        with self.lock:
            self.request_active = False
            self.cleanup_observed = True


@dataclass(frozen=True)
class _Selection:
    plan: ProjectionPlan
    phase: str
    binding: str
    observation: _Observation
    wrapper: object
    prepare: object
    cleanup: object


def _validate(owner):
    if type(owner.plan) is not ProjectionPlan or _binding(owner.plan, owner.phase) != owner.binding:
        raise ValueError("Projection chunk binding changed after explicit selection")


def _selection(model):
    getter = getattr(model, "get_attachment", None)
    return getter(KEY) if callable(getter) else None


def configure_projection_chunks(model, plan=None, *, phase="SINGLE", enabled=True):
    """Select only this clone; unknown executable predecessors are not erased."""
    if type(enabled) is not bool:
        raise ValueError("Projection enabled must be boolean")
    if not enabled:
        return model, {"schema": SCHEMA, "status": "disabled", "portable_cache_identity": False}
    plan = ProjectionPlan() if plan is None else plan
    if type(plan) is not ProjectionPlan:
        raise TypeError("Projection configuration requires an explicit ProjectionPlan")
    if not isinstance(phase, str) or not phase.strip():
        raise ValueError("Projection phase must be a nonempty label")
    if not all(callable(getattr(model, name, None)) for name in (
            "clone", "get_attachment", "set_attachments", "add_wrapper_with_key", "add_callback_with_key")):
        raise TypeError("Projection MODEL needs actual clone/attachment/wrapper/callback APIs")
    previous = _selection(model)
    if previous is not None and type(previous) is not _Selection:
        warn_patch_stack("Projection attachment has another owner; retaining it and original MODEL without installing this optimization")
        return model, {"schema": SCHEMA, "status": "not_installed_existing_owner_preserved",
                       "partial": True, "portable_cache_identity": False}
    selected = model.clone()
    if previous is not None:
        _validate(previous)
    observation = _Observation(phase)
    binding = _binding(plan, phase)

    def wrapper(executor, *args, **kwargs):
        if _binding(plan, phase) != binding:
            raise ValueError("Projection chunk binding changed before diffusion execution")
        epoch = observation.enter()
        runtime, error = None, None
        network = getattr(executor, "class_obj", None)
        try:
            if not isinstance(network, torch.nn.Module):
                result = executor(*args, **kwargs)
            else:
                with projection_scope(network, plan, phase=phase) as runtime:
                    result = executor(*args, **kwargs)
            # Check after the scope restores the live network, including on
            # the final diffusion call. There may be no next forward or Audit
            # node to notice mutation of this explicitly selected plan.
            if _binding(plan, phase) != binding:
                raise ValueError("Projection chunk binding changed during diffusion execution")
            return result
        except BaseException as caught:
            error = caught
            raise  # No retry, sampler fallback or swallowed kernel failure.
        finally:
            observation.finish(epoch, None if runtime is None else runtime.report(), error)

    prepare, cleanup = observation.pre_run, observation.cleanup
    owner = _Selection(plan, phase, binding, observation, wrapper, prepare, cleanup)
    for group, kind, replacement, attribute, add in (
        (selected.wrappers, WrappersMP.DIFFUSION_MODEL, wrapper, "wrapper", selected.add_wrapper_with_key),
        (selected.callbacks, CallbacksMP.ON_PRE_RUN, prepare, "prepare", selected.add_callback_with_key),
        (selected.callbacks, CallbacksMP.ON_CLEANUP, cleanup, "cleanup", selected.add_callback_with_key),
    ):
        old = getattr(previous, attribute, None)
        values = group.get(kind, {}).get(KEY, [])
        if old is not None and any(entry is old for entry in values):
            # Replace only exact own slots. Appending after removal would move
            # later user wrappers outside this scope and reorder callbacks.
            group[kind][KEY] = [replacement if entry is old else entry for entry in values]
        else:
            add(kind, KEY, replacement)
    selected.set_attachments(KEY, owner)
    return selected, audit_projection_chunks(selected)


def audit_projection_chunks(model):
    """Detached observations; never authorizes portable or completed Stage reuse."""
    owner = _selection(model)
    if type(owner) is not _Selection:
        return {"schema": SCHEMA, "status": "not_selected" if owner is None else "unverified_attachment",
                "portable_cache_identity": False, "quality_accepted": False}
    _validate(owner)
    state = owner.observation
    with state.lock:
        # User callables need not have hash/equality semantics. Inspection
        # must neither run __eq__ nor confuse value equality with ownership.
        active = any(value is owner.wrapper for value in model.get_wrappers(WrappersMP.DIFFUSION_MODEL, KEY))
        partial = bool(state.delegated) or not active or state.lifecycle_ambiguous
        status = ("failed" if state.failed else "observed_partial" if state.completed and partial
                  else "observed" if state.completed and state.calls
                  else "observed_no_eligible_calls" if state.completed else "selected_not_executed")
        return deepcopy({
            "schema": SCHEMA, "status": status, "phase": owner.phase, "plan": asdict(owner.plan),
            "request_number": state.request_number, "request_active": state.request_active,
            "explicit_pre_run_observed": state.explicit_pre_run, "cleanup_observed": state.cleanup_observed,
            "completed_diffusion_calls": state.completed, "failed_diffusion_calls": state.failed,
            "calls": state.calls, "delegated": sorted(state.delegated), "partial": partial,
            "selected_wrapper_present": active, "lifecycle_ambiguous": state.lifecycle_ambiguous,
            "errors": state.errors, "last_forward": state.last_forward,
            "portable_cache_identity": False, "quality_accepted": False,
            "allocator_peak_measured": False, "performance_qualified": False,
            "count_boundary": "Actual diffusion-wrapper calls, not declared sampler steps, NFE or GPU kernel count",
            "memory_boundary": "Primitive byte/row observations are not CUDA allocator or process/host peak measurements",
        })
