"""Independent HyperFlow continuous HEAD/TAIL with the exact native x_sigma.

This is the continuous S13 route, not either fresh-noise upscale or P7. No
whole-split forwarding, learned lift or second descent in a stage operation.
Persistence/public effect owners are separate adapters, not claimed here.
"""
from dataclasses import asdict, dataclass
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
from types import MethodType

import torch
import comfy.patcher_extension
import comfy.sample
import comfy.utils

from .. import hyperflow_sampling_advanced as native
from .. import hyperflow_runtime_advanced as runtime
from .. import hyperflow_two_pass_advanced as legacy
from .. import progressive_sampling_runtime as cleanup
from ..patch_stack_policy import nonportable_model_identity, model_identity_matches, _execution_selection, UnverifiedModelStack
from .progressive import snapshot, portable
from .results import canonical, sha

SCHEMA = "t8.modular-sampling.hyperflow-continuous-boundary.v1"


def implementation():
    from .. import sampling
    from . import hyperflow_identity, hyperflow_effects, eav, effect_identity
    from .. import prompt_relay_advanced, enhance_a_video_advanced
    import comfy.samplers
    import comfy.model_patcher
    modules = (native, runtime, legacy, sampling, cleanup, comfy.sample, comfy.samplers, comfy.model_patcher,
               hyperflow_identity, hyperflow_effects, eav, effect_identity, prompt_relay_advanced, enhance_a_video_advanced)
    paths = {Path(module.__file__).resolve() for module in modules} | {Path(__file__).resolve()}
    return {path.as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}


def model_selection(model):
    """Authenticate known two-time owners; preserve unknown user stacks locally."""
    from .hyperflow_identity import model_identity
    try:
        return model_identity(model)
    except UnverifiedModelStack as error:
        reason = str(error)
    from . import hyperflow_fresh
    if model.get_attachment(hyperflow_fresh.KEY) is not None:
        return hyperflow_fresh.live_identity(model, reason)
    identity = nonportable_model_identity(model, reason,
        schema="t8.modular-sampling.hyperflow-live-model.v1")
    selected = identity["execution_selection"]
    network = selected["network_execution"]
    classes = {}
    for name, module in model.model.named_modules():
        native_forward = inspect.getattr_static(type(module), "forward")
        classes[name] = _execution_selection(native_forward)
        current = vars(module).get("forward")
        if type(current) is MethodType and current.__self__ is module and current.__func__ is native_forward:
            if module._forward_pre_hooks or module._forward_hooks:
                network[name] = _execution_selection({"forward": None,
                    "pre_hooks": module._forward_pre_hooks, "hooks": module._forward_hooks})
            else:
                network.pop(name, None)
    selected["class_forward_dispatch"] = classes
    selected["base_protocol"] = _execution_selection({name: getattr(model.model, name, None) for name in
        ("audio_scale", "_scale_audio_slice", "process_latent_in", "process_latent_out", "extra_conds", "apply_model", "_apply_model")})
    selected["latent_coordinates"] = _execution_selection({
        "attributes": vars(model.model.latent_format), "scale_factor": model.model.latent_format.scale_factor,
        "in": model.model.latent_format.process_in, "out": model.model.latent_format.process_out})
    # Keep the complete within-run selection check without repeating hundreds
    # of LoRA descriptors in every JSON receipt. This is still process-local,
    # not an opaque callable serialization or a portable identity promotion.
    identity["execution_selection"] = {"selection_sha256": sha(selected)}
    identity["binding"] = asdict(model.get_attachment(runtime.ATTACHMENT_KEY))
    return identity


def _parameters(seed, cfg, reserve_vram_mib):
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("HyperFlow stage seed must be unsigned 64-bit")
    if type(cfg) not in (int, float) or not math.isfinite(cfg) or not 0 <= cfg <= 100:
        raise ValueError("HyperFlow stage CFG must be finite in [0,100]")
    if type(reserve_vram_mib) is not int or reserve_vram_mib < 512:
        raise ValueError("HyperFlow stage requires reserve of at least512 MiB")


def _finite_av(latent):
    video, audio = legacy.nested_av_parts(latent)
    if not all(part.dtype.is_floating_point and bool(torch.isfinite(part).all()) for part in (video, audio)):
        raise ValueError("HyperFlow AV must contain finite floating-point video/audio")
    return video, audio


@dataclass(frozen=True)
class ContinuousBoundary:
    x_sigma: torch.Tensor
    scaffold: dict
    receipt_json: str

    def verify(self):
        receipt = json.loads(self.receipt_json)
        if set(receipt) != {"schema", "request", "request_sha256", "outputs", "execution", "receipt_sha256"}:
            raise ValueError("Unknown HyperFlow continuous boundary fields")
        expected = dict(receipt)
        digest = expected.pop("receipt_sha256")
        if receipt["schema"] != SCHEMA or sha(expected) != digest or sha(receipt["request"]) != receipt["request_sha256"]:
            raise ValueError("HyperFlow boundary receipt changed")
        plan = receipt["request"]["plan"]
        start, stop = plan["absolute_interval"]
        if start != 0 or type(stop) is not int or not 1 <= stop <= 7:
            raise ValueError("HyperFlow HEAD must stop on an interior absolute boundary")
        _finite_av(self.scaffold)
        packed, _ = comfy.utils.pack_latents(self.scaffold["samples"].unbind())
        if (not isinstance(self.x_sigma, torch.Tensor) or self.x_sigma.shape != packed.shape
                or self.x_sigma.dtype != torch.float32 or not bool(torch.isfinite(self.x_sigma).all())
                or self.scaffold.get("noise_mask") is not None):
            raise ValueError("HyperFlow continuous raw state/scaffold changed")
        if receipt["outputs"] != snapshot({"x_sigma": self.x_sigma, "scaffold": self.scaffold}):
            raise ValueError("HyperFlow boundary tensor or metadata contents changed")
        execution = receipt["execution"]
        if execution["callbacks"] != list(range(stop)) or execution["endpoint_captures"] != 1:
            raise ValueError("An incomplete HyperFlow HEAD is not a continuation boundary")
        if execution["portable_identity"] != _portable_execution(receipt["request"], execution):
            raise ValueError("HyperFlow boundary portable identity differs from its actual execution")
        return receipt


def _portable_execution(request, execution):
    start, stop = request["plan"]["absolute_interval"]
    return bool(portable(request["model"]) and portable(request["inputs"])
                and execution.get("actual_forward_coverage") is True
                and set(execution["actual_apply_intervals"]) == set(range(start, stop))
                and (execution.get("effects") is None or execution["effects"].get("composition_verified") is True))


def _run(model, plan, source, positive, negative, noise, *, seed, cfg, reserve_vram_mib,
         callback=None, x_sigma=None, capture_endpoint=False):
    _parameters(seed, cfg, reserve_vram_mib)
    _finite_av(source)
    if [tuple(x.shape) for x in _finite_av({"samples": noise})] != [tuple(x.shape) for x in source["samples"].unbind()]:
        raise ValueError("HyperFlow stage noise geometry changed")
    request = {"plan": plan.as_report(), "model": model_selection(model),
        "execution_base": {"pid": os.getpid(), "object": id(model.model)},
        "seed": seed, "cfg": cfg, "reserve_vram_mib": reserve_vram_mib,
        "inputs": snapshot({"source": source, "positive": positive, "negative": negative, "noise": noise, "x_sigma": x_sigma}),
        "implementation": implementation()}
    captured = {}
    execution = {"callbacks": [], "actual_apply_intervals": [], "endpoint_captures": 0,
        "resource_snapshots": [], "portable_identity": False}

    def capture(state):
        if captured:
            raise RuntimeError("HyperFlow HEAD endpoint captured twice")
        captured["x_sigma"] = state
        execution["endpoint_captures"] += 1

    branch, sampler, sigmas = native.setup_hyperflow_sampler(model, source, plan, internal_continuation=True,
        endpoint_capture=capture if capture_endpoint else None,
        x_sigma_override=(lambda: x_sigma) if x_sigma is not None else None)
    kind = comfy.patcher_extension.WrappersMP.APPLY_MODEL
    key = "t8_modular_hyperflow_stage_observer_v1"

    def observe(executor, *args, **kwargs):
        step = runtime.active_step()
        result = executor(*args, **kwargs)
        if step is not None and step.owner == plan.owner:
            execution["actual_apply_intervals"].append(step.absolute_index)
        return result

    def resources():
        execution["resource_snapshots"].append(cleanup._resource_snapshot(
            torch.device(branch.load_device), reserve_vram_mib * 1024**2))

    def progress(index, prediction, state, total):
        if index != len(execution["callbacks"]) or total != plan.nfe:
            raise RuntimeError("HyperFlow stage callbacks do not match its absolute interval")
        execution["callbacks"].append(index)
        resources()
        if callback is not None:
            callback(plan.start_interval + index, prediction, state, 8)

    branch.add_wrapper_with_key(kind, key, observe)
    try:
        from .hyperflow_effects import execution as effect_execution
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
    if execution["callbacks"] != list(range(plan.nfe)):
        raise RuntimeError("HyperFlow stage did not complete all intervals")
    if not model_identity_matches(request["model"], model_selection(model)):
        raise ValueError("HyperFlow MODEL changed during this stage")
    if request["inputs"] != snapshot({"source": source, "positive": positive, "negative": negative, "noise": noise, "x_sigma": x_sigma}):
        raise ValueError("HyperFlow stage inputs changed during execution")
    if request["implementation"] != implementation():
        raise ValueError("HyperFlow stage implementation changed during execution")
    output = {**source, "samples": result}
    _finite_av(output)
    execution["actual_forward_coverage"] = set(execution["actual_apply_intervals"]) == set(range(plan.start_interval, plan.stop_interval))
    execution["portable_identity"] = _portable_execution(request, execution)
    return output, captured, request, execution


@torch.inference_mode()
def sample_head(model, source, positive, negative, noise, *, seed, split_interval=4, cfg=1., callback=None,
                reserve_vram_mib=1024):
    """Only absolute0:split. No HIGH MODEL, conditions, RNG, upscale or hidden tail."""
    legacy._validate_pair(model, model, source, split_interval)
    plan = native.build_hyperflow_plan(model, 0, split_interval)
    output, captured, request, execution = _run(model, plan, source, positive, negative, noise,
        seed=seed, cfg=cfg, callback=callback, reserve_vram_mib=reserve_vram_mib, capture_endpoint=True)
    if "x_sigma" not in captured:
        raise RuntimeError("HyperFlow HEAD returned without its exact x_sigma")
    receipt = {"schema": SCHEMA, "request": request, "request_sha256": sha(request),
        "outputs": snapshot({"x_sigma": captured["x_sigma"], "scaffold": output}), "execution": execution}
    receipt["receipt_sha256"] = sha(receipt)
    boundary = ContinuousBoundary(captured["x_sigma"], output, canonical(receipt))
    boundary.verify()
    return boundary


@torch.inference_mode()
def sample_tail(boundary, model, positive, negative, *, seed, cfg=1., callback=None, reserve_vram_mib=1024):
    """Only absolutesplit:8, injecting exact captured state with no fresh noise."""
    if type(boundary) is not ContinuousBoundary:
        raise ValueError("Connect HyperFlow continuous HEAD, not x0/Progressive/LATENT")
    low = boundary.verify()
    start = low["request"]["plan"]["absolute_interval"][1]
    legacy._validate_pair(model, model, boundary.scaffold, start)
    binding = model.get_attachment(runtime.ATTACHMENT_KEY)
    old_model = low["request"]["model"]
    old = old_model["binding"]
    current = model_selection(model)
    if old_model.get("portable_cache_reuse") is True and current.get("portable_cache_reuse") is True:
        same_base = (old_model["common_base_sha256"] == current["common_base_sha256"]
                     and old_model["adapter_sha256"] == current["adapter_sha256"])
    else:
        # Unknown stacks still execute on the original live common base. This
        # branch cannot turn a saved opaque owner into cross-process identity.
        live = low["request"].get("execution_base")
        same_base = (live == {"pid": os.getpid(), "object": id(model.model)} if live is not None
                     else binding.model_identity == old.get("model_identity"))
    if (not same_base or binding.sha256 != old["sha256"]
            or list(binding.raw_sigmas) != list(old["raw_sigmas"])):
        raise ValueError("HyperFlow continuous TAIL requires the same common base and original adapter as HEAD")
    plan = native.build_hyperflow_plan(model, start, 8)
    for field in ("video_sigmas", "audio_sigmas"):
        if list(getattr(plan, field)) != low["request"]["plan"][field]:
            raise ValueError("HyperFlow continuous AV clocks changed")
    empty_noise = comfy.sample.prepare_empty_noise(boundary.scaffold["samples"])
    output, _, request, execution = _run(model, plan, boundary.scaffold, positive, negative, empty_noise,
        seed=seed, cfg=cfg, callback=callback, reserve_vram_mib=reserve_vram_mib, x_sigma=boundary.x_sigma)
    boundary.verify()
    report = {"schema": "t8.modular-sampling.hyperflow-continuous-tail.v1", "request": request,
        "head_receipt_sha256": low["receipt_sha256"], "execution": execution,
        "source_portable_identity": bool(low["execution"]["portable_identity"] and portable(low["outputs"])),
        "head_executed": False, "fresh_noise": False, "learned_upscale": False,
        "handoff": "direct_captured_x_sigma_no_new_noise_no_audio_rebase", "quality_accepted": False}
    return output, canonical(report)


@dataclass(frozen=True)
class ContinuousResult:
    output: dict
    receipt_json: str

    def verify(self):
        receipt = json.loads(self.receipt_json)
        if set(receipt) != {"schema", "sampling", "output", "receipt_sha256"}:
            raise ValueError("Unknown HyperFlow completed result fields")
        expected = dict(receipt)
        digest = expected.pop("receipt_sha256")
        if receipt["schema"] != "t8.modular-sampling.hyperflow-completed-av.v1" or sha(expected) != digest:
            raise ValueError("HyperFlow completed result receipt changed")
        report = receipt["sampling"]
        if report["schema"] != "t8.modular-sampling.hyperflow-continuous-tail.v1":
            raise ValueError("Completed HyperFlow result has another route")
        start, stop = report["request"]["plan"]["absolute_interval"]
        if not 1 <= start < stop == 8 or report["execution"]["callbacks"] != list(range(stop - start)):
            raise ValueError("Incomplete HyperFlow tail cannot be delivered")
        if report["execution"]["portable_identity"] != _portable_execution(report["request"], report["execution"]):
            raise ValueError("HyperFlow completed identity differs from its actual execution")
        _finite_av(self.output)
        if receipt["output"] != snapshot(self.output):
            raise ValueError("Completed HyperFlow AV or metadata changed")
        return receipt


def sample_tail_result(boundary, model, positive, negative, **options):
    output, report = sample_tail(boundary, model, positive, negative, **options)
    receipt = {"schema": "t8.modular-sampling.hyperflow-completed-av.v1",
        "sampling": json.loads(report), "output": snapshot(output)}
    receipt["receipt_sha256"] = sha(receipt)
    result = ContinuousResult(output, canonical(receipt))
    result.verify()
    return result, report
