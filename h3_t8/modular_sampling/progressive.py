"""Separate native Progressive LOW, typed boundary, handoff and HIGH execution.

No call to the legacy two-stage executor. The learned lift is an external graph
operation. This module retains the legacy model-space video/audio mathematics.
"""
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path

import torch
import comfy.model_management as mm
import comfy.nested_tensor
import comfy.patcher_extension
import comfy.sample
import comfy.samplers

from .. import progressive_sampling_runtime as legacy
from .. import progressive_sampling_contract as contract
from .. import progressive_masking as masks
from ..progressive_stage_models import _architecture
from .progressive_model_identity import native_model_identity
from ..progressive_continuation_runtime import _input_identity
from ..patch_stack_policy import UnverifiedModelStack, _execution_selection, model_identity_matches
from .results import canonical, sha, _conditions

SCHEMA = "t8.modular-sampling.progressive-boundary.v1"
FIELDS = {"clean_video", "audio_next", "anchor_audio_noise"}


def snapshot(value):
    try:
        return {"portable": True, "value": _conditions(value)}
    except UnverifiedModelStack:
        return {"portable": False, "value": _execution_selection(value)}


def portable(value):
    if isinstance(value, dict):
        return value.get("portable", True) and value.get("portable_cache_reuse", True) and all(portable(x) for x in value.values())
    return all(portable(x) for x in value) if isinstance(value, (list, tuple)) else True


def implementation():
    import comfy.model_base
    import comfy.model_sampling
    import comfy.latent_formats
    import comfy.k_diffusion.sampling
    from .. import progressive_checkpoint, progressive_stage_models
    from . import progressive_effects, progressive_model_identity, progressive_effect_identity
    from .. import progressive_relay, progressive_eav, progressive_eav_masks, progressive_attention
    from .. import enhance_a_video_advanced, prompt_relay_advanced
    from .. import h3_core_compat, relay_kj_memory
    from . import effect_identity, results, continuation_identity
    modules = (legacy, contract, masks, progressive_checkpoint, progressive_stage_models,
        comfy.model_base, comfy.model_sampling, comfy.latent_formats, comfy.k_diffusion.sampling,
        comfy.samplers, comfy.sample, comfy.model_patcher, progressive_relay, progressive_eav,
        progressive_eav_masks, progressive_attention, enhance_a_video_advanced, prompt_relay_advanced,
        h3_core_compat, relay_kj_memory, effect_identity, results, comfy.conds, continuation_identity)
    paths = {Path(module.__file__).resolve() for module in modules} | {
        Path(__file__).resolve(), Path(progressive_effects.__file__).resolve(),
        Path(progressive_model_identity.__file__).resolve(), Path(progressive_effect_identity.__file__).resolve()}
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}


def validate_plan(plan):
    if type(plan) is not contract.ProgressivePlan or plan.schema != contract.SCHEMA:
        raise ValueError("Expected original ProgressivePlan")
    if (type(plan.low_evaluations) is not int or type(plan.high_evaluations) is not int
            or min(plan.low_evaluations, plan.high_evaluations) < 1 or plan.total_evaluations > 1000
            or len(plan.sigmas) != plan.total_evaluations + 1
            or not all(type(v) in (int, float) and math.isfinite(v) for v in plan.sigmas)
            or not 0 < plan.sigmas[0] <= 1 or plan.sigmas[-1] != 0
            or any(a <= b for a, b in zip(plan.sigmas, plan.sigmas[1:]))):
        raise ValueError("Invalid progressive stage interval")
    if (len(plan.video_shape) != 5 or plan.video_shape[:2] != (1, 24)
            or len(plan.audio_shape) != 4 or plan.audio_shape[:3] != (1, 32, 2)
            or any(type(n) is not int or n < 1 for n in (*plan.video_shape, *plan.audio_shape))
            or plan.video_shape[2] < 2 or (plan.video_shape[2] - 2) % 5
            or plan.audio_shape[-1] != round((5 + (plan.video_shape[2] - 2) // 5 * 17) * 40 / 24)
            or (plan.target_height, plan.target_width) != (plan.video_shape[-2] * 16, plan.video_shape[-1] * 16)
            or any(type(n) is not int or n < 32 or n % 32 for n in
                   (plan.target_width, plan.target_height, plan.low_width, plan.low_height))
            or not 0 < plan.low_width < plan.target_width or not 0 < plan.low_height < plan.target_height
            or plan.sigma_dtype not in ("torch.float32", "torch.float64", "torch.float16", "torch.bfloat16")):
        raise ValueError("Invalid progressive AV geometry/dtype")
    scale = plan.requested_low_scale
    if (plan.task not in ("t2va", "i2va") or type(scale) not in (int, float)
            or not math.isfinite(scale) or not .25 <= scale < 1
            or plan.low_width != max(32, round(plan.target_width * scale / 32) * 32)
            or plan.low_height != max(32, round(plan.target_height * scale / 32) * 32)):
        raise ValueError("Invalid progressive task/LOW scale")
    return plan


def plan_from_dict(value):
    if type(value) is not dict or set(value) != set(contract.ProgressivePlan.__dataclass_fields__):
        raise ValueError("Unknown progressive plan fields")
    data = dict(value)
    for key in ("sigmas", "video_shape", "audio_shape"):
        data[key] = tuple(data[key])
    return validate_plan(contract.ProgressivePlan(**data))


def build_plan(av_latent, sigmas, low_evaluations=4, low_scale=.5, task="t2va", input_mode="empty"):
    if input_mode not in ("empty", "initialized_av_exp"):
        raise ValueError("Unknown progressive source mode")
    video, audio = masks._av_parts(av_latent["samples"], "progressive source")
    kwargs = dict(low_evaluations=low_evaluations, low_scale=low_scale, task=task)
    if input_mode == "empty":
        plan = contract.plan_progressive_first_sample(video, audio, sigmas, noise_mask=av_latent.get("noise_mask"), **kwargs)
    else:
        plan = contract.plan_progressive_initialized_sample(video, audio, sigmas, **kwargs)
    return validate_plan(plan)


def prepare_low_source(av_latent, plan, input_mode="empty", *, explicit_low=None):
    """Return only LOW's template; a prepared continuation may supply its own LOW."""
    validate_plan(plan)
    video, audio = masks._av_parts(av_latent["samples"], "target source")
    if tuple(video.shape) != plan.video_shape or tuple(audio.shape) != plan.audio_shape:
        raise ValueError("Target source no longer matches the plan")
    if explicit_low is not None:
        if input_mode != "initialized_av_exp":
            raise ValueError("Explicit LOW requires initialized mode")
        low_video, low_audio = masks._av_parts(explicit_low["samples"], "explicit LOW")
        if not torch.equal(low_audio, audio):
            raise ValueError("LOW and HIGH source audio differ")
        low = dict(explicit_low)
    elif input_mode == "initialized_av_exp":
        low_video = masks.resize_video_source(video, plan.low_height // 16, plan.low_width // 16)
        low = {**av_latent, "samples": comfy.nested_tensor.NestedTensor((low_video, audio))}
    elif input_mode == "empty":
        if av_latent.get("noise_mask") is not None or any(bool(torch.count_nonzero(x)) for x in (video, audio)):
            raise ValueError("Empty LOW requires zero AV without a mask")
        low_video = torch.zeros((*video.shape[:-2], plan.low_height // 16, plan.low_width // 16), dtype=video.dtype, device=video.device)
        low = {**av_latent, "samples": comfy.nested_tensor.NestedTensor((low_video, audio))}
    else:
        raise ValueError("Unknown progressive source mode")
    _geometry(low, plan, "low")
    normalized = masks.normalize_av_masks(low.get("noise_mask"), *low["samples"].unbind())
    low.pop("noise_mask", None)
    if normalized is not None:
        low["noise_mask"] = normalized
    return low


def _geometry(latent, plan, phase):
    if phase not in ("low", "high"):
        raise ValueError("Unknown progressive stage")
    video, audio = masks._av_parts(latent["samples"], phase)
    wanted = ((*plan.video_shape[:-2], plan.low_height // 16, plan.low_width // 16)
              if phase == "low" else plan.video_shape)
    if tuple(video.shape) != wanted or tuple(audio.shape) != plan.audio_shape:
        raise ValueError("Progressive " + phase + " geometry differs from the plan")
    return video, audio


def coordinates(model, sampler):
    sampling = legacy.validate_native_model(model, sampler)
    fmt = model.get_model_object("latent_format")
    return {"architecture": _architecture(model), "sampling": {key: getattr(sampling, key, 1. if key == "noise_scale" else None)
        for key in ("shift", "audio_shift", "audio_scale", "multiplier", "noise_scale")},
        "latent_format": {key: getattr(fmt, key, None) for key in
            ("scale_factor", "shift_factor", "latent_channels", "spacial_downscale_ratio", "temporal_downscale_ratio")}}


def sampler_identity(sampler):
    import comfy.k_diffusion.sampling
    known = (type(sampler) is comfy.samplers.KSAMPLER
             and set(vars(sampler)) == {"sampler_function", "extra_options", "inpaint_options"}
             and sampler.sampler_function is comfy.k_diffusion.sampling.sample_euler)
    return {"portable": known, "selection": (snapshot({"native_euler": True,
        "extra_options": sampler.extra_options, "inpaint_options": sampler.inpaint_options}) if known
        else {"portable": False, "value": _execution_selection(sampler)})}


@dataclass(frozen=True)
class ProgressiveBoundary:
    tensors: dict
    receipt_json: str

    def verify(self):
        receipt = json.loads(self.receipt_json)
        if type(receipt) is not dict or set(receipt) != {
            "schema", "request", "request_sha256", "tensors", "execution", "portable_identity", "receipt_sha256"}:
            raise ValueError("Unknown progressive boundary receipt")
        expected = dict(receipt)
        digest = expected.pop("receipt_sha256")
        if receipt["schema"] != SCHEMA or sha(expected) != digest or sha(receipt["request"]) != receipt["request_sha256"]:
            raise ValueError("Progressive boundary receipt changed")
        plan = plan_from_dict(receipt["request"]["plan"])
        if type(self.tensors) is not dict or set(self.tensors) != FIELDS:
            raise ValueError("Progressive boundary needs clean_video, audio_next and anchor_audio_noise")
        for key, value in self.tensors.items():
            wanted = (*plan.video_shape[:-2], plan.low_height // 16, plan.low_width // 16) if key == "clean_video" else plan.audio_shape
            masks._finite(value, key)
            if tuple(value.shape) != wanted:
                raise ValueError("Progressive boundary tensor geometry changed: " + key)
        if _input_identity(self.tensors) != receipt["tensors"]:
            raise ValueError("Progressive boundary tensor contents changed")
        execution = receipt["execution"]
        if (type(execution) is not dict or execution.get("callbacks") != list(range(plan.low_evaluations))
                or execution.get("phase") != "low" or type(execution.get("actual_apply_calls")) is not int
                or execution["actual_apply_calls"] < 0):
            raise ValueError("An incomplete LOW cannot be a progressive boundary")
        portable_expected = bool(portable(receipt["request"]) and execution["actual_apply_calls"] >= plan.low_evaluations)
        if type(receipt["portable_identity"]) is not bool or receipt["portable_identity"] != portable_expected:
            raise ValueError("Progressive boundary portable identity is inconsistent")
        return receipt


def _run(model, sampler, plan, phase, latent, noise, positive, negative, cfg, seed, *,
         callback=None, capture=None, reserve_vram_mib=1024):
    """Exactly one legacy native stage, with scoped observation and cleanup."""
    validate_plan(plan)
    if type(seed) is not int or not 0 <= seed < 2**64 or type(cfg) not in (int, float) or not math.isfinite(cfg) or not 0 <= cfg <= 100:
        raise ValueError("Invalid native stage seed/cfg")
    if type(reserve_vram_mib) is not int or reserve_vram_mib < 512:
        raise ValueError("GPU reserve must be an integer of at least 512 MiB")
    _geometry(latent, plan, phase)
    if [tuple(x.shape) for x in masks._av_parts(noise, "native noise")] != [tuple(x.shape) for x in latent["samples"].unbind()]:
        raise ValueError("Noise geometry differs from stage")
    start, end = (0, plan.low_evaluations) if phase == "low" else (plan.low_evaluations, plan.total_evaluations)
    schedule = torch.tensor(plan.sigmas, dtype=getattr(torch, plan.sigma_dtype.split(".")[-1]))
    branch = model.clone()
    execution = {"phase": phase, "callbacks": [], "actual_apply_calls": 0, "resource_snapshots": []}
    key = "t8_modular_progressive_stage_observer_v1"
    kind = comfy.patcher_extension.WrappersMP.APPLY_MODEL

    def resources():
        execution["resource_snapshots"].append(legacy._resource_snapshot(
            torch.device(branch.load_device), reserve_vram_mib * 1024**2))

    def observe(executor, *args, **kwargs):
        result = executor(*args, **kwargs)
        execution["actual_apply_calls"] += 1
        return result

    def progress(step, prediction, state, total):
        if step != len(execution["callbacks"]) or total != end - start:
            raise RuntimeError("Native stage callbacks differ from its exact interval")
        execution["callbacks"].append(step)
        resources()
        if capture is not None and step + 1 == end - start:
            capture(prediction, state)
        if callback is not None:
            callback(start + step, prediction, state, plan.total_evaluations)

    branch.add_wrapper_with_key(kind, key, observe)
    try:
        from .progressive_effects import execution as effects_execution
        with effects_execution(branch, plan, phase, latent, cfg) as effects_report:
            resources()
            result = legacy._native_stage(branch, sampler, schedule[start:end+1], latent["samples"], noise,
                positive, negative, cfg, seed, progress, denoise_mask=latent.get("noise_mask"),
                preview_phase=phase, preview_offset=start, preview_total=plan.total_evaluations)
        if effects_report is not None:
            execution["effects"] = effects_report
    finally:
        branch.remove_wrappers_with_key(kind, key)
    if execution["callbacks"] != list(range(end - start)):
        raise RuntimeError("Native stage ended without its complete interval")
    _geometry({"samples": result}, plan, phase)
    return result, execution


@torch.inference_mode()
def sample_low(model, sampler, plan, low_latent, positive, negative, noise, *, seed, cfg=1.,
               callback=None, reserve_vram_mib=1024):
    """Execute LOW only; no HIGH model/conditions/lifter in this request."""
    coords = coordinates(model, sampler)
    request = {"plan": asdict(validate_plan(plan)), "coordinates": coords, "seed": seed, "cfg": cfg,
        "sampler": sampler_identity(sampler), "reserve_vram_mib": reserve_vram_mib,
        "inputs": snapshot({"latent": low_latent, "positive": positive, "negative": negative, "noise": noise}),
        "model": native_model_identity(model, sampler), "implementation": implementation()}
    captured = {}

    def capture(prediction, state):
        video, audio_prediction = prediction.unbind()
        audio_state = state.unbind()[1]
        captured["clean_video"] = video.detach().to(device="cpu", dtype=torch.float32, copy=True)
        captured["audio_next"] = contract.euler_sampler_space_step(audio_state, audio_prediction,
            plan.prediction_sigma, plan.resume_sigma).detach().to(device="cpu", copy=True)

    _, execution = _run(model, sampler, plan, "low", low_latent, noise, positive, negative, cfg, seed,
                        callback=callback, capture=capture, reserve_vram_mib=reserve_vram_mib)
    if not model_identity_matches(request["model"], native_model_identity(model, sampler)):
        raise ValueError("LOW MODEL changed during sampling")
    if request["inputs"] != snapshot({"latent": low_latent, "positive": positive, "negative": negative, "noise": noise}):
        raise ValueError("LOW inputs changed during sampling")
    if request["implementation"] != implementation():
        raise ValueError("LOW implementation changed during sampling")
    if request["sampler"] != sampler_identity(sampler):
        raise ValueError("LOW sampler changed during sampling")
    captured["anchor_audio_noise"] = noise.unbind()[1].detach().to(device="cpu", copy=True)
    receipt = {"schema": SCHEMA, "request": request, "request_sha256": sha(request), "tensors": _input_identity(captured),
        "execution": execution, "portable_identity": bool(portable(request) and execution["actual_apply_calls"] >= plan.low_evaluations)}
    receipt["receipt_sha256"] = sha(receipt)
    result = ProgressiveBoundary(captured, canonical(receipt))
    result.verify()
    return result


def lift_input(boundary, model, sampler):
    receipt = boundary.verify()
    if coordinates(model, sampler) != receipt["request"]["coordinates"]:
        raise ValueError("Progressive handoff MODEL coordinates differ from LOW")
    return {"samples": comfy.nested_tensor.NestedTensor((
        model.get_model_object("latent_format").process_out(boundary.tensors["clean_video"]).clone(),
        torch.zeros_like(boundary.tensors["audio_next"])))}


@dataclass(frozen=True)
class ProgressiveRestart:
    plan: contract.ProgressivePlan
    tensors: dict
    contract_json: str
    metadata: dict

    def verify(self):
        data = json.loads(self.contract_json)
        validate_plan(self.plan)
        if type(data) is not dict or set(data) != {"schema", "plan", "coordinates", "low_receipt_sha256",
                "tensors", "metadata", "contract_sha256"} or data["schema"] != "t8.modular-sampling.progressive-restart.v1":
            raise ValueError("Unknown progressive HIGH restart contract")
        expected = dict(data)
        digest = expected.pop("contract_sha256")
        if sha(expected) != digest:
            raise ValueError("Progressive HIGH restart contract changed")
        if type(self.tensors) is not dict or set(self.tensors) != {"restart", "anchor", "anchor_noise", "mask"}:
            raise ValueError("Progressive HIGH restart inventory differs")
        for name in ("restart", "anchor", "anchor_noise"):
            _geometry({"samples": self.tensors[name]}, self.plan, "high")
        video, audio = self.tensors["anchor"].unbind()
        normalized = masks.normalize_av_masks(self.tensors["mask"], video, audio)
        if _input_identity(normalized) != _input_identity(self.tensors["mask"]):
            raise ValueError("Progressive HIGH masks must already be normalized")
        if (type(self.metadata) is not dict or {"samples", "noise_mask"} & set(self.metadata)
                or data["tensors"] != _input_identity(self.tensors)
                or data["metadata"] != snapshot(self.metadata)
                or data["plan"] != json.loads(canonical(asdict(self.plan)))):
            raise ValueError("Progressive HIGH restart changed")
        return data


@torch.inference_mode()
def prepare_high(boundary, model, sampler, lifted_av, high_source, high_video_noise, *, high_sigmas=None):
    """No sampling/lift: original video renoise and audio_next coordinate restore."""
    receipt = boundary.verify()
    plan = plan_from_dict(receipt["request"]["plan"])
    if high_sigmas is not None:
        if (not isinstance(high_sigmas, torch.Tensor) or high_sigmas.ndim != 1
                or str(high_sigmas.dtype) != plan.sigma_dtype or not 2 <= high_sigmas.numel() <= 1000):
            raise ValueError("HIGH sigmas must retain the LOW boundary dtype and be a 1D interval")
        tail = tuple(high_sigmas.detach().to(device="cpu", dtype=torch.float64).tolist())
        if tail[0] != plan.resume_sigma:
            raise ValueError("HIGH sigmas must start at the frozen LOW resume sigma")
        plan = validate_plan(replace(plan, sigmas=plan.sigmas[:plan.low_evaluations] + tail,
                                     high_evaluations=len(tail) - 1))
    coords = coordinates(model, sampler)
    if coords != receipt["request"]["coordinates"]:
        raise ValueError("Progressive HIGH MODEL coordinates/architecture differ from LOW")
    video, audio = _geometry(high_source, plan, "high")
    lifted, _ = _geometry(lifted_av, plan, "high")
    masks._finite(high_video_noise, "HIGH video noise")
    if high_video_noise.shape != lifted.shape:
        raise ValueError("HIGH noise must match lifted video")
    fmt = model.get_model_object("latent_format")
    sampling = model.get_model_object("model_sampling")
    clean = fmt.process_in(lifted.float())
    noise = high_video_noise.to(clean)
    sigma = torch.tensor(plan.sigmas, dtype=getattr(torch, plan.sigma_dtype.split(".")[-1]))[plan.low_evaluations].to(clean)
    high_state = sampling.noise_scaling(sigma, noise, clean)
    restart_video = fmt.process_out(sampling.inverse_noise_scaling(sigma, high_state))
    audio_next = boundary.tensors["audio_next"]
    restart_audio = sampling.inverse_noise_scaling(sigma.to(audio_next), audio_next) / float(sampling.audio_scale)
    intermediate = mm.intermediate_device()
    nested = comfy.nested_tensor.NestedTensor
    values = {"restart": nested((restart_video.to(intermediate), restart_audio.to(intermediate))),
        "anchor": nested((video.detach().clone(), audio.detach().clone())),
        "anchor_noise": nested((noise.detach().clone().to(intermediate), boundary.tensors["anchor_audio_noise"].clone().to(intermediate))),
        "mask": masks.normalize_av_masks(high_source.get("noise_mask"), video, audio)}
    metadata = {key: value for key, value in high_source.items() if key not in ("samples", "noise_mask")}
    data = {"schema": "t8.modular-sampling.progressive-restart.v1", "plan": asdict(plan), "coordinates": coords,
        "low_receipt_sha256": receipt["receipt_sha256"], "tensors": _input_identity(values), "metadata": snapshot(metadata)}
    data["contract_sha256"] = sha(data)
    result = ProgressiveRestart(plan, values, canonical(data), metadata)
    result.verify()
    return result


@torch.inference_mode()
def sample_high(restart, model, sampler, positive, negative, *, seed, cfg=1., callback=None, reserve_vram_mib=1024):
    """Execute HIGH only using explicit zero-noise reconstruction; LOW is absent."""
    data = restart.verify()
    if coordinates(model, sampler) != data["coordinates"]:
        raise ValueError("HIGH MODEL coordinates changed after handoff")
    before = native_model_identity(model, sampler)
    inputs = snapshot((positive, negative, seed, cfg))
    code = implementation()
    original_sampler = sampler_identity(sampler)
    values = restart.tensors
    selected = (masks.sampler_with_clean_anchor(sampler, values["anchor"], values["anchor_noise"])
                if values["mask"] is not None else sampler)
    latent = {"samples": values["restart"]}
    if values["mask"] is not None:
        latent["noise_mask"] = values["mask"]
    noise = comfy.nested_tensor.NestedTensor([torch.zeros_like(part) for part in values["restart"].unbind()])
    result, execution = _run(model, selected, restart.plan, "high", latent, noise, positive, negative, cfg, seed,
                             callback=callback, reserve_vram_mib=reserve_vram_mib)
    restart.verify()
    if not model_identity_matches(before, native_model_identity(model, sampler)) or code != implementation():
        raise ValueError("HIGH MODEL or implementation changed during sampling")
    if inputs != snapshot((positive, negative, seed, cfg)):
        raise ValueError("HIGH conditions changed during sampling")
    if original_sampler != sampler_identity(sampler):
        raise ValueError("HIGH sampler changed during sampling")
    report = {"schema": "t8.modular-sampling.progressive-high.v1", "execution": execution,
        "low_receipt_sha256": data["low_receipt_sha256"], "low_executed": False,
        "model": before, "inputs": inputs, "implementation": code, "restart": data,
        "sampler": original_sampler, "reserve_vram_mib": reserve_vram_mib,
        "portable_identity": bool(portable(before) and portable(inputs) and portable(data["metadata"])
            and portable(original_sampler) and execution["actual_apply_calls"] >= restart.plan.high_evaluations),
        "boundary": "One original native HIGH interval; no learned lift or LOW execution, quality unverified."}
    return {**restart.metadata, "samples": result.to(mm.intermediate_device())}, canonical(report)
