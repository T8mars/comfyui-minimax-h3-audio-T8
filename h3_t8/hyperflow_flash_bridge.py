"""Interop with the separately installed official Flash checkpoint consumer.

The external plugin owns the factorized layers, quantization and two-time
implementation. This bridge never treats Flash as an ordinary LoRA.
"""
import json

import nodes as core_nodes
import torch


def load_flash(model_name, weights_in="auto"):
    provider = core_nodes.NODE_CLASS_MAPPINGS.get("HyperFlowLoader")
    if provider is None:
        raise ImportError(
            "HyperFlow Flash needs the official ComfyUI-HyperFlow-Flash HyperFlowLoader consumer. "
            "Install https://github.com/Video-Rebirth/ComfyUI-HyperFlow-Flash; "
            "INT8 safetensors use Core kernels, GGUF additionally needs ComfyUI-GGUF.")
    consumer = provider()
    method = getattr(consumer, getattr(provider, "FUNCTION", "load"), None)
    if not callable(method):
        raise TypeError("Registered Flash provider has no callable loading interface")
    result = method(unet_name=model_name, weights_in=weights_in)
    if hasattr(result, "result"):
        result = result.result
    if not result:
        raise ValueError("Flash provider returned no MODEL")
    model = result[0]
    diffusion = getattr(getattr(model, "model", None), "diffusion_model", None)
    schedule = getattr(diffusion, "hyperflow_sigmas", None)
    report = {"schema": "t8.h3.flash_bridge.v1", "file": model_name,
              "provider": f"{provider.__module__}.{provider.__name__}",
              "weights_in": weights_in, "provider_MODEL_preserved": True,
              "has_checkpoint_schedule": isinstance(schedule, torch.Tensor),
              "checkpoint_schedule": schedule.detach().cpu().tolist() if isinstance(schedule, torch.Tensor) else None,
              "quality_validated": False,
              "note": "Loading is delegated; no ordinary LoRA conversion or extra time shift is applied"}
    return model, json.dumps(report)


def checkpoint_schedule(model):
    diffusion = getattr(getattr(model, "model", None), "diffusion_model", None)
    schedule = getattr(diffusion, "hyperflow_sigmas", None)
    if not isinstance(schedule, torch.Tensor):
        raise ValueError("The connected MODEL has no checkpoint hyperflow_sigmas tensor; use your explicit SIGMAS input")
    if schedule.ndim != 1 or schedule.numel() < 2 or not torch.isfinite(schedule).all():
        raise ValueError("Checkpoint hyperflow_sigmas must be a finite one-dimensional schedule")
    if (schedule < 0).any() or (schedule[1:] > schedule[:-1]).any() or schedule[-1] != 0:
        raise ValueError("Checkpoint hyperflow_sigmas must be nonnegative, nonincreasing and end at zero")
    return schedule.detach().clone().cpu()


def flash_schedule(model, schedule="checkpoint"):
    if schedule == "checkpoint":
        return checkpoint_schedule(model)
    provider = core_nodes.NODE_CLASS_MAPPINGS.get("HyperFlowSigmas")
    if provider is None:
        raise ImportError("Flash schedule presets need the official HyperFlowSigmas provider; or use explicit SIGMAS")
    consumer = provider()
    method = getattr(consumer, getattr(provider, "FUNCTION", "sigmas"), None)
    if not callable(method):
        raise TypeError("Registered Flash schedule provider has no callable interface")
    # Preset options describe the official recipe. Delegate other explicit API
    # step values too, so a newer provider can support them without a T8 gate.
    if isinstance(schedule, bool):
        raise ValueError("Flash step count must be an integer")
    steps = int(schedule)
    if isinstance(schedule, float) and schedule != steps:
        raise ValueError("Flash step count must be an integer")
    result = method(model=model, steps=steps)
    if hasattr(result, "result"):
        result = result.result
    return result[0]
