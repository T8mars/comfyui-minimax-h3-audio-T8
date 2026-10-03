"""Independent fitted-curve HEAD/TAIL; never calls the full split executor.

Local EXP: exact captured model-space x_sigma, separate Core scaffold, actual
native51 coverage and explicit producer/input identities. A fitted curve is an
approximation, not a full-HyperFlow equivalence or human quality certificate.
"""
from dataclasses import dataclass
import hashlib
import inspect
import json
import os
from pathlib import Path
from types import MethodType

import comfy.patcher_extension
import comfy.sample
import comfy.utils
import torch

from .. import hyperflow_curve_runtime_exp as runtime
from .. import hyperflow_curve_fit_exp as fitting
from .. import hyperflow_sampling_advanced as integrator
from .. import progressive_sampling_runtime as cleanup
from ..hyperflow_weights_advanced import DEFAULT_RAW_GRID
from ..patch_stack_policy import (UnverifiedModelStack, nonportable_model_identity,
                                 model_identity_matches, _execution_selection)
from .hyperflow import _parameters, _finite_av
from .progressive import snapshot, portable
from .results import canonical, sha

RECIPE = "hyperflow8_pruned_curve_approx_exp_v1"
HEAD = "t8.hyperflow.curve_continuous_head.v1"
TAIL = "t8.hyperflow.curve_continuous_tail.v1"
RESULT = "t8.hyperflow.curve_completed_av.v1"


def implementation():
    from .. import hyperflow_curve_loader_exp
    from . import hyperflow_curve_identity, hyperflow_curve_effects, eav, effect_identity, hyperflow, progressive, results
    from .. import prompt_relay_advanced, enhance_a_video_advanced
    import comfy.samplers
    import comfy.model_patcher
    from .. import sampling
    modules = (runtime, fitting, integrator, runtime.full_runtime, cleanup, sampling,
               comfy.sample, comfy.samplers, comfy.model_patcher, hyperflow_curve_identity,
               hyperflow_curve_effects, eav, effect_identity, prompt_relay_advanced, enhance_a_video_advanced,
               hyperflow, progressive, results, comfy.utils, hyperflow_curve_loader_exp)
    paths = {Path(module.__file__).resolve() for module in modules} | {Path(__file__).resolve()}
    return {path.as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}


def model_selection(model):
    from .hyperflow_curve_identity import model_identity
    try:
        return model_identity(model)
    except UnverifiedModelStack as error:
        value = nonportable_model_identity(model, str(error), schema="t8.hyperflow.curve_live_model.v1")
    selected = value["execution_selection"]
    network = selected["network_execution"]
    dispatch = {}
    for name, module in model.model.named_modules():
        descriptor = inspect.getattr_static(type(module), "forward")
        dispatch[name] = _execution_selection(descriptor)
        current = vars(module).get("forward")
        # An explicit selected object patch replaces dormant shared storage.
        # This normalization is strictly execution-local, never portable proof.
        selected_method = name + ".forward" in model.object_patches
        native_alias = (type(current) is MethodType and current.__self__ is module
                        and current.__func__ is descriptor)
        if selected_method or native_alias:
            if module._forward_pre_hooks or module._forward_hooks:
                network[name] = _execution_selection({"forward": None,
                    "pre_hooks": module._forward_pre_hooks, "hooks": module._forward_hooks})
            else:
                network.pop(name, None)
    selected["class_dispatch"] = dispatch
    selected["base_protocol"] = _execution_selection({name: getattr(model.model, name, None) for name in
        ("audio_scale", "_scale_audio_slice", "process_latent_in", "process_latent_out",
         "extra_conds", "apply_model", "_apply_model")})
    latent = model.model.latent_format
    selected["latent_coordinates"] = _execution_selection({"state": vars(latent), "scale": latent.scale_factor,
        "in": latent.process_in, "out": latent.process_out})
    value["execution_selection"] = {"selection_sha256": sha(selected)}
    return value


def _source(source):
    video, audio = _finite_av(source)
    if (video.ndim != 5 or video.shape[:2] != (1, 24) or audio.ndim != 4 or audio.shape[:3] != (1, 32, 2)
            or source.get("noise_mask") is not None):
        raise ValueError("Curve continuous stages need native joint AV without a restart mask")
    return video, audio


def _plan(value, start, stop):
    expected = {"schema", "recipe", "source_sha256", "raw_sigmas", "video_sigmas", "audio_sigmas",
                "absolute_interval", "nfe", "note"}
    if (type(value) is not dict or set(value) != expected or value["recipe"] != RECIPE
            or value["schema"] != "t8.minimax_h3.hyperflow.plan.v1"
            or value["absolute_interval"] != [start, stop] or value["nfe"] != stop - start
            or type(start) is not int or type(stop) is not int or not 0 <= start < stop <= 8):
        raise ValueError("Curve receipt has another absolute interval/recipe")
    raw = value["raw_sigmas"]
    if type(raw) is not list or len(raw) != 9 or any(abs(a - b) > 1e-6 for a, b in zip(raw, DEFAULT_RAW_GRID)):
        raise ValueError("Curve receipt changed its trained raw grid")
    for name, shift in (("video_sigmas", 12.), ("audio_sigmas", 3.)):
        if value[name] != integrator.shifted_grid(tuple(raw), shift).tolist():
            raise ValueError("Curve receipt changed its actual AV clock")


def _portable_execution(request, execution):
    start, stop = request["plan"]["absolute_interval"]
    return bool(portable(request["model"]) and portable(request["inputs"])
                and execution.get("actual_51_coverage") is True
                and set(execution["actual_apply_intervals"]) == set(range(start, stop))
                and (execution.get("effects") is None or execution["effects"].get("composition_verified") is True))


def _execution(request, execution):
    start, stop = request["plan"]["absolute_interval"]
    if (execution["callbacks"] != list(range(stop - start))
            or execution["portable_identity"] != _portable_execution(request, execution)):
        raise ValueError("Incomplete or changed curve stage execution receipt")


@dataclass(frozen=True)
class CurveBoundary:
    x_sigma: torch.Tensor
    scaffold: dict
    receipt_json: str

    def verify(self):
        value = json.loads(self.receipt_json)
        if type(value) is not dict or set(value) != {"schema", "request", "outputs", "execution", "receipt_sha256"}:
            raise ValueError("Unknown fitted curve HEAD fields")
        unsigned = dict(value)
        digest = unsigned.pop("receipt_sha256")
        if value["schema"] != HEAD or sha(unsigned) != digest:
            raise ValueError("Curve HEAD receipt changed")
        stop = value["request"]["plan"]["absolute_interval"][1]
        if type(stop) is not int or not 1 <= stop <= 7:
            raise ValueError("Curve HEAD requires an interior boundary")
        _plan(value["request"]["plan"], 0, stop)
        _source(self.scaffold)
        packed, _ = comfy.utils.pack_latents(self.scaffold["samples"].unbind())
        if (type(self.x_sigma) is not torch.Tensor or self.x_sigma.dtype != torch.float32
                or self.x_sigma.shape != packed.shape or not bool(torch.isfinite(self.x_sigma).all())
                or value["outputs"] != snapshot({"x_sigma": self.x_sigma, "scaffold": self.scaffold})):
            raise ValueError("Curve captured raw state/scaffold changed")
        _execution(value["request"], value["execution"])
        if value["execution"]["endpoint_captures"] != 1:
            raise ValueError("Curve HEAD has no unique actual endpoint capture")
        return value


@dataclass(frozen=True)
class CurveResult:
    output: dict
    receipt_json: str

    def verify(self):
        value = json.loads(self.receipt_json)
        if type(value) is not dict or set(value) != {"schema", "sampling", "output", "receipt_sha256"}:
            raise ValueError("Unknown fitted curve completed fields")
        unsigned = dict(value)
        digest = unsigned.pop("receipt_sha256")
        if value["schema"] != RESULT or sha(unsigned) != digest or value["sampling"]["schema"] != TAIL:
            raise ValueError("Curve completed receipt changed")
        report = value["sampling"]
        start, stop = report["request"]["plan"]["absolute_interval"]
        if not 1 <= start < stop == 8:
            raise ValueError("Curve TAIL did not complete the trained trajectory")
        _plan(report["request"]["plan"], start, stop)
        _execution(report["request"], report["execution"])
        if report["head_executed"] is not False or report["fresh_noise"] is not False:
            raise ValueError("Curve continuation cannot certify a fresh restart")
        _source(self.output)
        if value["output"] != snapshot(self.output):
            raise ValueError("Curve completed AV/metadata changed")
        return value


def _run(model, plan, source, positive, negative, noise, *, seed, cfg, reserve_vram_mib,
         callback=None, raw=None, capture=False):
    _parameters(seed, cfg, reserve_vram_mib)
    video, audio = _source(source)
    parts = _finite_av({"samples": noise})
    if [tuple(x.shape) for x in parts] != [tuple(video.shape), tuple(audio.shape)]:
        raise ValueError("Curve noise and source native geometries differ")
    request = {"plan": plan.as_report(), "model": model_selection(model),
        "execution_base": {"pid": os.getpid(), "object": id(model.model)},
        "seed": seed, "cfg": cfg, "reserve_vram_mib": reserve_vram_mib,
        "inputs": snapshot({"source": source, "positive": positive, "negative": negative, "noise": noise, "raw": raw}),
        "implementation": implementation()}
    execution = {"callbacks": [], "actual_apply_intervals": [], "endpoint_captures": 0,
        "resource_snapshots": [], "actual_51_coverage": False, "portable_identity": False}
    captured = {}
    def endpoint(state):
        if captured:
            raise RuntimeError("Curve endpoint captured twice")
        captured["raw"] = state
        execution["endpoint_captures"] += 1
    branch, sampler, sigmas = runtime.setup_sampler(model, source, plan, continuation=True,
        x_sigma=(lambda: raw) if raw is not None else None, capture=endpoint if capture else None)
    kind, key = comfy.patcher_extension.WrappersMP.APPLY_MODEL, "t8_curve_stage_observer_v1"
    def observe(executor, *args, **kwargs):
        step = runtime.full_runtime.active_step()
        result = executor(*args, **kwargs)
        if step is not None and step.owner == plan.owner:
            execution["actual_apply_intervals"].append(step.absolute_index)
        return result
    def resources():
        execution["resource_snapshots"].append(cleanup._resource_snapshot(torch.device(branch.load_device), reserve_vram_mib * 1024**2))
    def progress(index, prediction, state, total):
        if index != len(execution["callbacks"]) or total != plan.nfe:
            raise RuntimeError("Curve callback no longer matches the typed interval")
        execution["callbacks"].append(index)
        resources()
        if callback is not None:
            callback(plan.start_interval + index, prediction, state, 8)
    audit = model.get_attachment(runtime.DETAILS_KEY).audit
    before = len(audit["forwards"])
    branch.add_wrapper_with_key(kind, key, observe)
    try:
        from .hyperflow_curve_effects import execution as effect_execution
        with effect_execution(model, plan, source, positive, negative, cfg) as effects:
            resources()
            result = cleanup._native_stage(branch, sampler, sigmas, source["samples"], noise,
                positive, negative, cfg, seed, progress,
                preview_phase="low" if plan.start_interval == 0 else "high",
                preview_offset=plan.start_interval, preview_total=8)
        if effects is not None:
            execution["effects"] = effects
    finally:
        branch.remove_wrappers_with_key(kind, key)
    if not model_identity_matches(request["model"], model_selection(model)):
        raise ValueError("Curve MODEL changed during stage execution")
    if request["inputs"] != snapshot({"source": source, "positive": positive, "negative": negative, "noise": noise, "raw": raw}):
        raise ValueError("Curve stage inputs changed during execution")
    if request["implementation"] != implementation():
        raise ValueError("Curve stage implementation changed during execution")
    rows = audit["forwards"][before:]
    execution["actual_51_coverage"] = bool(rows) and all(row["complete_51"] is True for row in rows)
    execution["actual_51_coverage"] &= set(row["interval"] for row in rows) == set(range(plan.start_interval, plan.stop_interval))
    execution["portable_identity"] = _portable_execution(request, execution)
    _execution(request, execution)
    output = {**source, "samples": result}
    _source(output)
    return output, captured, request, execution


@torch.inference_mode()
def sample_head(model, source, positive, negative, noise, *, seed, split=4, cfg=1., callback=None, reserve_vram_mib=1024):
    if type(split) is not int or not 1 <= split <= 7:
        raise ValueError("Curve HEAD requires an explicit split in [1,7]")
    plan = runtime.build_plan(model, 0, split)
    output, captured, request, execution = _run(model, plan, source, positive, negative, noise,
        seed=seed, cfg=cfg, callback=callback, reserve_vram_mib=reserve_vram_mib, capture=True)
    if set(captured) != {"raw"}:
        raise RuntimeError("Curve HEAD returned without its actual x_sigma")
    value = {"schema": HEAD, "request": request, "outputs": snapshot({"x_sigma": captured["raw"], "scaffold": output}), "execution": execution}
    value["receipt_sha256"] = sha(value)
    result = CurveBoundary(captured["raw"], output, canonical(value))
    result.verify()
    return result


@torch.inference_mode()
def sample_tail(boundary, model, positive, negative, *, seed, cfg=1., callback=None, reserve_vram_mib=1024):
    if type(boundary) is not CurveBoundary:
        raise ValueError("Curve TAIL requires its dedicated HEAD, not x0 or full-HyperFlow state")
    head = boundary.verify()
    if head["request"]["implementation"] != implementation():
        raise ValueError("Curve HEAD producer implementation changed; no unaudited cross-version continuation")
    old, current = head["request"]["model"], model_selection(model)
    if old.get("portable_cache_reuse") is True and current.get("portable_cache_reuse") is True:
        same = old["common_base_sha256"] == current["common_base_sha256"] and old["adapter_sha256"] == current["adapter_sha256"]
    else:
        same = head["request"]["execution_base"] == {"pid": os.getpid(), "object": id(model.model)}
    start = head["request"]["plan"]["absolute_interval"][1]
    plan = runtime.build_plan(model, start, 8)
    if not same or any(plan.as_report()[key] != head["request"]["plan"][key] for key in
            ("source_sha256", "raw_sigmas", "video_sigmas", "audio_sigmas")):
        raise ValueError("Curve TAIL must retain HEAD's common base, exact fit and AV clocks")
    output, _, request, execution = _run(model, plan, boundary.scaffold, positive, negative,
        comfy.sample.prepare_empty_noise(boundary.scaffold["samples"]), seed=seed, cfg=cfg,
        callback=callback, reserve_vram_mib=reserve_vram_mib, raw=boundary.x_sigma)
    boundary.verify()
    report = {"schema": TAIL, "request": request, "execution": execution,
        "head_receipt_sha256": head["receipt_sha256"], "source_portable_identity": bool(head["execution"]["portable_identity"] and portable(head["outputs"])),
        "head_executed": False, "fresh_noise": False, "learned_upscale": False, "quality_accepted": False,
        "handoff": "exact_captured_x_sigma_no_new_noise_or_audio_rebase"}
    value = {"schema": RESULT, "sampling": report, "output": snapshot(output)}
    value["receipt_sha256"] = sha(value)
    result = CurveResult(output, canonical(value))
    result.verify()
    return result, canonical(report)
