"""Explicit same-canvas PTD HEAD/TAIL, not legacy 4+4 or portable cold state.

Core keeps its normal loading, guider, ordered patches and latent conversion.
The typed handoff captures the integrator's actual internal FP32 noisy state,
before Core's inverse-noise/latent-output transforms. Original PTD stays intact.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
import threading
import weakref

import torch
import comfy.model_management as mm
import comfy.sample
from comfy.samplers import KSAMPLER
from comfy_extras.nodes_custom_sampler import SamplerCustomAdvanced

from .ptd_advanced import get_controller
from . import ptd_sampling as clocks

_LOCK = threading.RLock()
_ISSUED = {}


class PTDHead:
    """Only an observed HEAD run can issue a registered in-process instance."""
    __slots__ = ("payload", "__weakref__")


class PTDCompletion:
    __slots__ = ("payload", "__weakref__")


def _clone(value, warnings):
    if isinstance(value, torch.Tensor) or getattr(value, "is_nested", False):
        if getattr(value, "is_nested", False):
            return type(value)(tuple(_clone(part, warnings) for part in value.unbind()))
        return value.detach().to(device="cpu").clone()
    if isinstance(value, Mapping):
        # Core accepts dictionary providers, not just type(dict). Snapshot
        # their actual LATENT fields into owned data; never retain the mutable
        # source mapping and later rebind an old packed state to a new canvas.
        if type(value) is not dict:
            warnings.add("mapping_provider_values_snapshotted_to_owned_LATENT_data")
        return {key: _clone(item, warnings) for key, item in value.items()}
    if type(value) in (list, tuple):
        return type(value)(_clone(item, warnings) for item in value)
    if value is None or type(value) in (str, int, float, bool):
        return value
    warnings.add("opaque_auxiliary_value_process_local_only")
    return value


def _digest(value):
    digest = hashlib.sha256()

    def label(item):
        raw = json.dumps(item, allow_nan=False, separators=(",", ":")).encode()
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)

    def visit(item):
        mm.throw_exception_if_processing_interrupted()
        if isinstance(item, torch.Tensor) or getattr(item, "is_nested", False):
            if getattr(item, "is_nested", False):
                label("nested")
                for part in item.unbind():
                    visit(part)
                return
            if item.layout != torch.strided or item.device.type == "meta":
                raise ValueError("PTD handoff needs materialized dense state")
            data = item.detach().cpu().contiguous()
            if data.is_floating_point() and not bool(torch.isfinite(data).all()):
                raise ValueError("Non-finite PTD handoff tensor")
            label(("tensor", str(data.dtype), list(data.shape)))
            raw = memoryview(data.reshape(-1).view(torch.uint8).numpy()).cast("B")
            for start in range(0, len(raw), 8 * 1024 * 1024):
                mm.throw_exception_if_processing_interrupted()
                digest.update(raw[start:start + 8 * 1024 * 1024])
            return
        if isinstance(item, Mapping):
            label(("mapping", len(item)))
            for key, part in item.items():
                visit(key)
                visit(part)
        elif type(item) in (tuple, list):
            label((type(item).__name__, len(item)))
            for part in item:
                visit(part)
        elif item is None or type(item) in (str, int, float, bool):
            label((type(item).__name__, item))
        else:
            # No user equality/hash/repr callbacks; opaque auxiliaries cannot
            # become a portable identity. The critical state/grid stay typed.
            label(("opaque_process_local", id(item), id(type(item))))

    visit(value)
    return digest.hexdigest()


def _issue(kind, payload):
    value = kind()
    value.payload = payload
    identity = id(value)
    seal = _digest(payload)

    def release(reference):
        with _LOCK:
            row = _ISSUED.get(identity)
            if row is not None and row[0] is reference:
                _ISSUED.pop(identity)

    reference = weakref.ref(value, release)
    with _LOCK:
        _ISSUED[identity] = (reference, seal)
    return value


def _verify(value, kind):
    if type(value) is not kind:
        raise ValueError("Connect the actual typed PTD stage output")
    with _LOCK:
        row = _ISSUED.get(id(value))
    if row is None or row[0]() is not value:
        raise ValueError("PTD stage was not issued by a completed actual execution")
    if _digest(value.payload) != row[1]:
        raise ValueError("PTD stage contents changed after actual completion")
    return value.payload


def inspect_stage(value):
    kind = type(value)
    if kind is PTDHead:
        payload = _verify(value, PTDHead)
    elif kind is PTDCompletion:
        payload = _verify(value, PTDCompletion)
        _verify(payload["head"], PTDHead)
    else:
        raise ValueError("Expected an actual typed PTD HEAD or completion")
    return deepcopy(payload["report"])


def _require_state(value):
    parts = clocks.streams(value)
    if any(not isinstance(part, torch.Tensor) or part.dtype is not torch.float32
           or part.ndim < 1 or part.shape[0] < 1 or not bool(torch.isfinite(part).all())
           for part in parts):
        raise ValueError("PTD continuous state must contain finite nonempty FP32 streams")
    if any(part.device != parts[0].device or part.shape[0] != parts[0].shape[0] for part in parts):
        raise ValueError("PTD state streams must share a batch and device")


def _report(controller, start, stop, *, warnings=()):
    raw = controller.report()
    last = raw["last_request"]
    hops = list(range(start, stop))
    if last is None or last["hops"] != hops:
        raise ValueError("Actual PTD range did not complete its ordered global hops")
    ignored = tuple(f"selected_expert_not_observed:{hop}:" for hop in range(len(controller.grid) - 1)
                    if hop not in hops)
    delegated = [item for item in last["delegated"] if not item.startswith(ignored)]
    all_warnings = sorted(set(raw["warnings"]) | set(warnings))
    return {"schema": "t8.h3.ptd.continuous.v1", "recipe": controller.recipe,
            "full_grid": list(controller.grid), "global_hops": hops,
            "range_completed": True, "whole_recipe_completed": False,
            "integrator_evaluations": len(hops), "observed_network_forwards": deepcopy(last["forwards"]),
            "actual_expert_calls": deepcopy(last["expert_calls"]),
            "delegated": delegated, "warnings": all_warnings,
            "partial": bool(delegated or all_warnings),
            "expert_content_sha256": raw["expert_content_sha256"],
            "fp32_internal_noisy_state": True, "portable_cache_identity": False,
            "cold_saved_or_restored": False, "trained_GPU_qualified": False,
            "quality_accepted": False}


@torch.no_grad()
def _range(model, x, sigmas, extra_args=None, callback=None, disable=None, *, controller,
           start, stop, capture, previous=None):
    clocks.validate_grid(sigmas, controller.recipe)
    extra = {} if extra_args is None else dict(extra_args)
    if extra.get("denoise_mask") is not None:
        raise ValueError("PTD continuous range has no partial-denoise mask contract")
    original = getattr(model, "latent_image", None)
    if original is not None and any(bool(torch.count_nonzero(part)) for part in clocks.streams(original)):
        raise ValueError("PTD HEAD/TAIL uses the original empty generated latent template")
    if previous is None:
        raw = getattr(model, "noise", None)
        state = clocks.initial_state(raw) if raw is not None else clocks.map_streams(x, lambda part: part.float())
    else:
        payload = _verify(previous, PTDHead)
        if payload["recipe"] != controller.recipe or payload["next_hop"] != start:
            raise ValueError("PTD TAIL recipe/global index disagrees with its actual HEAD")
        state = clocks.map_streams(payload["state"], lambda part: part.to(device=clocks.streams(x)[0].device).clone())
    _require_state(state)
    clocks.pair_streams(state, x, lambda left, right: left)
    first = clocks.streams(state)[0]
    grid = clocks.ptd_sigmas(controller.recipe, first.device)
    controller.begin_request()
    try:
        for hop in range(start, stop):
            mm.throw_exception_if_processing_interrupted()
            time = controller.grid[hop]
            with controller.activate(hop, time):
                predicted = model(state, grid[hop].expand(first.shape[0]), **extra)
            velocity = clocks.pair_streams(state, predicted, lambda a, b: (a.float() - b.float()) / time)
            if callback is not None:
                callback({"x": state, "i": hop, "sigma": grid[hop], "sigma_hat": grid[hop], "denoised": predicted})
            dt = (grid[hop + 1] - grid[hop]).float()
            state = clocks.pair_streams(state, velocity, lambda a, b: a.float() + dt * b.float())
            _require_state(state)
            controller.completed_hop(hop)
        if previous is not None:
            _verify(previous, PTDHead)
    except BaseException:
        controller.end_request(False)
        raise
    controller.end_request(True)
    capture["report"] = _report(controller, start, stop)
    capture["state"] = clocks.map_streams(state, lambda part: part.detach().cpu().clone())
    capture["seed"] = extra.get("seed")
    capture["ran"] = True
    return state


def _controller(guider):
    model = getattr(guider, "model_patcher", None)
    if model is None:
        raise TypeError("PTD stage guider must expose its selected model_patcher")
    return get_controller(model)


def _sampler(controller, start, stop, capture, previous=None):
    # The mutable observation record is not a public sampler extra_option.
    def execute(model, x, sigmas, extra_args=None, callback=None, disable=None):
        capture["calls"] = capture.get("calls", 0) + 1
        return _range(model, x, sigmas, extra_args, callback, disable,
                      controller=controller, start=start, stop=stop,
                      capture=capture, previous=previous)

    return KSAMPLER(execute)


def run_head(noise, guider, av_latent):
    controller = _controller(guider)
    middle = (len(controller.grid) - 1) // 2
    capture, warnings = {}, set()
    template = _clone(av_latent, warnings)
    template_seal = _digest(av_latent)
    sampler = _sampler(controller, 0, middle, capture)
    SamplerCustomAdvanced.execute(noise, guider, sampler, clocks.ptd_sigmas(controller.recipe), av_latent)
    if not capture.get("ran") or capture.get("calls") != 1:
        raise ValueError("The selected guider did not execute a PTD HEAD integrator")
    controller._verify_bank()
    if _digest(av_latent) != template_seal:
        raise ValueError("PTD HEAD source template changed during execution")
    report = deepcopy(capture["report"])
    report["phase"] = "HEAD"
    report["warnings"] = sorted(set(report["warnings"]) | warnings)
    report["partial"] = bool(report["delegated"] or report["warnings"])
    report["next_global_hop"] = middle
    payload = {"recipe": controller.recipe, "full_grid": tuple(controller.grid),
               "next_hop": middle, "state": capture["state"], "template": template,
               "seed": capture["seed"], "report": report,
               "process_local_MODEL_selection": (id(guider.model_patcher), id(guider.model_patcher.model))}
    head = _issue(PTDHead, payload)
    return head, inspect_stage(head)


class _EmptyNoise:
    def __init__(self, seed):
        self.seed = seed

    def generate_noise(self, latent):
        return comfy.sample.prepare_empty_noise(latent["samples"])


def run_tail(guider, head):
    payload = _verify(head, PTDHead)
    controller = _controller(guider)
    if controller.recipe != payload["recipe"] or tuple(controller.grid) != payload["full_grid"]:
        raise ValueError("PTD TAIL must use the same physical recipe/grid as its HEAD")
    capture, warnings = {}, set()
    template = _clone(payload["template"], warnings)
    sampler = _sampler(controller, payload["next_hop"], len(controller.grid) - 1, capture, head)
    out = SamplerCustomAdvanced.execute(_EmptyNoise(payload["seed"]), guider, sampler,
                                        clocks.ptd_sigmas(controller.recipe), template).result[0]
    if not capture.get("ran") or capture.get("calls") != 1:
        raise ValueError("The selected guider did not execute a PTD TAIL integrator")
    controller._verify_bank()
    _verify(head, PTDHead)
    report = deepcopy(capture["report"])
    selection = (id(guider.model_patcher), id(guider.model_patcher.model))
    if selection != payload["process_local_MODEL_selection"]:
        warnings.add("different_TAIL_MODEL_branch_retained_unverified_process_local_selection")
    if report["expert_content_sha256"] != payload["report"]["expert_content_sha256"]:
        warnings.add("TAIL_selected_expert_bank_differs_from_HEAD_retained")
    report.update(phase="TAIL", whole_recipe_completed=True,
                  head_global_hops=list(payload["report"]["global_hops"]),
                  all_global_hops=payload["report"]["global_hops"] + report["global_hops"],
                  head_report=deepcopy(payload["report"]))
    report["warnings"] = sorted(set(report["warnings"]) | warnings)
    report["partial"] = bool(report["partial"] or payload["report"]["partial"] or warnings)
    result = _issue(PTDCompletion, {"head": head, "output": out, "report": report})
    return out, result, inspect_stage(result)
