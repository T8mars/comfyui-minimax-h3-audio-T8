"""Opt-in H3 clock adapter for an independently installed LanPaint dependency.

No upstream sampler is shipped or rewritten. The reviewed installed module is
loaded into a separate namespace; only that namespace receives the Core sigma
mapping and its analytic derivative. Ordinary LanPaint and Core globals stay
unchanged. This does not certify editing quality or arbitrary patch stacks.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import threading
from types import CodeType, FunctionType, ModuleType

import torch

SCHEMA = "t8.h3.lanpaint_clock_adapter.v1"
UPSTREAM_REVISION = "2d7912f9a5efe5ece8de334c7ca18317b8288c39"
SOURCE_PINS = {
    "nodes.py": "2c4973d5c4edf0df6b9625f4d2237e6c7513bc56804f03a2218b7c5b38f4ef88",
    "lanpaint.py": "d1a4932855d451bed5bc0aefea241742d05b876385b86ccd743602559403fe7b",
    "earlystop.py": "5bb0a033a509967d147924cb29a2c6fc14f61ab4e66f4228a746ab4ebc91e933",
    "types.py": "3a9a3b542c00a6e676225a5a61565b3a0c24ca31520e55f4b839c9e71f6e1c5c",
}
CORE_H3_PIN = "5e243986ad23469962a36291eb6b4e4427571b91f7a5930225e67c8322e6d122"
_LOCK = threading.RLock()


def time_shift_slope(sigma, from_shift, to_shift):
    """d sigma_audio / d sigma_video for Core's two-shift rational map.

    Derived directly: map = to*sigma/(from + sigma*(to-from)). Retain
    the tensor dtype/device and autograd; never clamp or change the schedule.
    """
    if any(not math.isfinite(float(value)) or float(value) <= 0
           for value in (from_shift, to_shift)):
        raise ValueError("H3 sigma shifts must be finite and strictly positive")
    return (to_shift * from_shift) / (from_shift + sigma * (to_shift - from_shift)) ** 2


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _reviewed_backend():
    import nodes
    from comfy.ldm.minimax import model as core_h3

    kind = nodes.NODE_CLASS_MAPPINGS.get("LanPaint_SamplerCustomAdvanced")
    if kind is None:
        raise RuntimeError("Install the external GPL-3.0 scraed/LanPaint pack and restart ComfyUI")
    original = sys.modules.get(kind.__module__)
    if not isinstance(original, ModuleType) or original.__dict__.get("LanPaint_SamplerCustomAdvanced") is not kind:
        raise RuntimeError("Cannot locate the installed LanPaint sampler module")
    source = Path(original.__file__).resolve()
    if source.name != "nodes.py" or not original.__package__:
        raise RuntimeError("Unsupported LanPaint source layout; ordinary LanPaint is unchanged")
    pins = {name: _sha(source.parent / name) for name in SOURCE_PINS}
    if pins != SOURCE_PINS:
        raise RuntimeError("This opt-in H3 clock adapter needs the reviewed LanPaint source; "
                           "use ordinary LanPaint for another version, not an automatic downgrade")
    core_source = Path(core_h3.__file__).resolve()
    sigma_fn = core_h3.__dict__.get("time_shift_sigma")
    if (_sha(core_source) != CORE_H3_PIN or not isinstance(sigma_fn, FunctionType)
            or sigma_fn.__globals__ is not core_h3.__dict__
            or Path(sigma_fn.__code__.co_filename).resolve() != core_source):
        raise RuntimeError("The selected Core H3 time mapping is not the reviewed API")
    # Source bytes alone cannot authenticate a replaced live function. Compile
    # the already pinned file without executing it and compare this tiny pure
    # function's real instructions/constants/arguments, not debug line tables.
    compiled = compile(core_source.read_bytes(), str(core_source), "exec", dont_inherit=True)
    expected = next(item for item in compiled.co_consts
                    if isinstance(item, CodeType) and item.co_name == "time_shift_sigma")
    actual = sigma_fn.__code__
    if (sigma_fn.__defaults__ is not None or sigma_fn.__kwdefaults__ is not None or sigma_fn.__closure__ is not None
            or any(type(item) not in (type(None), int, float, str) for item in actual.co_consts)
            or any(getattr(actual, key) != getattr(expected, key) for key in
                   ("co_code", "co_argcount", "co_kwonlyargcount", "co_posonlyargcount", "co_flags",
                    "co_consts", "co_names", "co_varnames", "co_freevars", "co_cellvars", "co_exceptiontable"))):
        raise RuntimeError("The live Core H3 time function differs from its pinned source")
    name = original.__package__ + "._t8_h3_clock_v1"
    backend = sys.modules.get(name)
    if backend is None:
        spec = importlib.util.spec_from_file_location(name, source)
        if spec is None or spec.loader is None:
            raise RuntimeError("Cannot load the installed LanPaint dependency")
        backend = importlib.util.module_from_spec(spec)
        sys.modules[name] = backend
        try:
            spec.loader.exec_module(backend)
            backend.time_shift_sigma = sigma_fn
            backend.time_shift_slope = time_shift_slope
        except BaseException:
            sys.modules.pop(name, None)
            raise
    if (not isinstance(backend, ModuleType)
            or backend.time_shift_sigma is not sigma_fn
            or backend.time_shift_slope is not time_shift_slope):
        raise RuntimeError("The isolated LanPaint clock namespace changed")
    if {key: _sha(source.parent / key) for key in SOURCE_PINS} != pins:
        raise RuntimeError("LanPaint source changed while preparing the isolated adapter")
    return backend, original, dict(upstream_revision_reviewed=UPSTREAM_REVISION,
        dependency_source_sha256=pins, Core_H3_source_sha256=CORE_H3_PIN,
        Core_time_shift_slope_present=callable(core_h3.__dict__.get("time_shift_slope")),
        upstream_clock_import_available=callable(original.time_shift_sigma)
            and callable(original.time_shift_slope),
        isolated_clock_namespace=True, upstream_sampler_bundled=False)


def sample_lanpaint_clock(noise, guider, sampler, sigmas, av_latent, inner_steps=5,
                         guidance_lambda=5.0, step_size=0.2, prompt_mode="Image First"):
    """Run the actual external sampler; do not replace unknown MODEL patches."""
    from comfy.model_patcher import create_model_options_clone

    samples = av_latent.get("samples") if isinstance(av_latent, dict) else None
    mask = av_latent.get("noise_mask") if isinstance(av_latent, dict) else None
    if samples is None or not getattr(samples, "is_nested", False):
        raise ValueError("Connect the nested H3 AV latent from LanPaint AV Prepare")
    streams = samples.unbind()
    if (len(streams) != 2 or streams[0].ndim != 5 or streams[0].shape[1] != 24
            or streams[1].ndim != 4 or streams[1].shape[1:3] != (32, 2)
            or streams[1].shape[0] != streams[0].shape[0]):
        raise ValueError("Expected H3 video [B,24,T,H,W] and audio [B,32,2,T]")
    if mask is None or not getattr(mask, "is_nested", False):
        raise ValueError("Explicit nested video/audio noise masks are required")
    masks = mask.unbind()
    if len(masks) != 2 or any(a.shape != b.shape for a, b in zip(streams, masks, strict=True)):
        raise ValueError("Each noise mask must match its actual H3 stream")
    if (any(not bool(torch.isfinite(value).all()) for value in (*streams, *masks))
            or any(bool((value < 0).any()) or bool((value > 1).any()) for value in masks)):
        raise ValueError("H3 streams/masks must be finite; noise masks must be in [0,1]")
    if (not isinstance(sigmas, torch.Tensor) or sigmas.ndim != 1 or sigmas.numel() < 2
            or not bool(torch.isfinite(sigmas).all()) or bool((sigmas < 0).any())
            or bool((sigmas >= 1).any()) or bool((sigmas[1:] > sigmas[:-1]).any())):
        raise ValueError("LanPaint requires a finite descending flow schedule in [0,1); no clamping is performed")
    if isinstance(inner_steps, bool) or not isinstance(inner_steps, int) or not 0 <= inner_steps <= 100:
        raise ValueError("inner_steps must be an integer from 0 to 100")
    if (not math.isfinite(float(guidance_lambda)) or not 0.1 <= guidance_lambda <= 50
            or not math.isfinite(float(step_size)) or not 0.0001 <= step_size <= 1):
        raise ValueError("Invalid explicit LanPaint dynamics parameters")
    if prompt_mode not in ("Image First", "Prompt First"):
        raise ValueError("Unknown LanPaint prompt_mode")

    # The native clone keeps LoRA ordering, hooks and callable delegates; it
    # shares model weights, not LanPaint configuration on the caller's patcher.
    selected = copy.copy(guider)
    selected.model_patcher = guider.model_patcher.clone()
    selected.model_options = create_model_options_clone(guider.model_options)
    selected.model_patcher.model_options = selected.model_options
    selected.original_conds = {key: list(value) for key, value in guider.original_conds.items()}
    selected.model_options["video_inpainting"] = True
    diff = selected.model_patcher.model.diffusion_model
    shifts = selected.model_options.get("transformer_options", {})
    video_shift = shifts.get("minimax_h3_sigma_shift_video", getattr(diff, "sigma_shift_video", None))
    audio_shift = shifts.get("minimax_h3_sigma_shift_audio", getattr(diff, "sigma_shift_audio", None))
    if video_shift is None or audio_shift is None:
        raise ValueError("The selected MODEL has no H3 video/audio sigma shifts")
    time_shift_slope(sigmas[0], video_shift, audio_shift)

    with _LOCK:
        backend, original, report = _reviewed_backend()
        if original._override_active or backend._override_active:
            raise RuntimeError("Another LanPaint context is active; run this sampler serially")
        sigma_fn, slope_fn = backend.time_shift_sigma, backend.time_shift_slope
        calls = {"clock_map_calls": 0, "clock_slope_calls": 0}

        def observe_sigma(*args):
            calls["clock_map_calls"] += 1
            return sigma_fn(*args)

        def observe_slope(*args):
            calls["clock_slope_calls"] += 1
            return slope_fn(*args)

        backend.time_shift_sigma, backend.time_shift_slope = observe_sigma, observe_slope
        try:
            result = backend.LanPaint_SamplerCustomAdvanced().sample(
                noise, selected, sampler, sigmas, av_latent, inner_steps, guidance_lambda,
                step_size, prompt_mode)
        finally:
            backend.time_shift_sigma, backend.time_shift_slope = sigma_fn, slope_fn
        for latent in result:
            actual_streams = latent["samples"].unbind()
            if (len(actual_streams) != 2
                    or any(a.shape != b.shape or not bool(torch.isfinite(b).all())
                           for a, b in zip(streams, actual_streams, strict=True))):
                raise RuntimeError("LanPaint returned invalid H3 AV streams")
        report.update(calls)
        report.update(schema=SCHEMA, status="external_sampler_returned",
            audio_clock_observed=calls["clock_map_calls"] > 0 and calls["clock_slope_calls"] > 0,
            video_shift=float(video_shift), audio_shift=float(audio_shift),
            outer_schedule_steps=sigmas.numel() - 1, inner_steps=inner_steps,
            guidance_lambda=float(guidance_lambda), step_size=float(step_size), prompt_mode=prompt_mode,
            caller_patcher_configuration_preserved=True, Core_or_upstream_files_modified=False,
            ordinary_LanPaint_namespace_modified=False, serial_sampler_dependency=True,
            generated_audio_PCM_exact=False, generic_patch_stack_qualified=False,
            human_edit_quality_accepted=False)
        if not report["audio_clock_observed"]:
            raise RuntimeError("The actual external sampler did not use the H3 audio clock; no success qualification")
        return (*result, json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
