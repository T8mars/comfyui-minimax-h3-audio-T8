from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

import folder_paths

from .vram_policy import build_vram_policy


RESIDENCY_SCHEMA = "t8.minimax_h3.residency_strategy.v1"
RESIDENCY_PRESETS = {
    "report_only": {
        "policy_mode": "report_only",
        "reserve_gib": 4.0,
        "intent": "observe current residency without changing runtime settings",
    },
    "minimum_memory": {
        "policy_mode": "fixed_total_reserved_exp",
        "reserve_gib": 6.0,
        "intent": "prefer offload/headroom over weight residency and speed",
    },
    "balanced": {
        "policy_mode": "fixed_total_reserved_exp",
        "reserve_gib": 4.0,
        "intent": "bounded 16GB starting point; re-measure each workflow",
    },
    "faster": {
        "policy_mode": "fixed_total_reserved_exp",
        "reserve_gib": 2.0,
        "intent": "allow more residency; highest activation/OOM risk of these presets",
    },
}


def _safe_call(value, default=None):
    try:
        return value() if callable(value) else value
    except Exception:
        return default


def _device(value) -> str | None:
    return None if value is None else str(value)


def _model_snapshot(model) -> dict[str, Any]:
    if model is None:
        return {"connected": False}
    result: dict[str, Any] = {"connected": True, "class": type(model).__name__}
    result["load_device"] = _device(getattr(model, "load_device", None))
    result["offload_device"] = _device(getattr(model, "offload_device", None))
    size = _safe_call(getattr(model, "model_size", None))
    loaded = _safe_call(getattr(model, "loaded_size", None))
    result["model_size_mib"] = None if size is None else int(size) / 1024**2
    result["loaded_size_mib"] = None if loaded is None else int(loaded) / 1024**2
    result["lowvram_patch_counter"] = int(getattr(model, "lowvram_patch_counter", 0) or 0)
    result["partially_loaded"] = bool(
        result["model_size_mib"] is not None
        and result["loaded_size_mib"] is not None
        and result["loaded_size_mib"] + 1e-6 < result["model_size_mib"]
    )
    options = getattr(model, "model_options", {})
    transformer = options.get("transformer_options", {}) if isinstance(options, Mapping) else {}
    keys = sorted(map(str, transformer)) if isinstance(transformer, Mapping) else []
    owner_markers = (
        "patches_replace",
        "sol_morton",
        "t8_h3_lightx2v_sla_runtime_v1",
        "t8_h3_eav_runtime",
        "minimax_h3_block_cache_t8",
        "t8_h3_prompt_relay",
    )
    result["transformer_option_keys"] = keys
    result["attention_or_forward_owners"] = [
        marker for marker in owner_markers if marker in keys
    ]
    diffusion = getattr(getattr(model, "model", None), "diffusion_model", None)
    result["diffusion_model_class"] = None if diffusion is None else type(diffusion).__name__
    result["quantization_evidence"] = {
        "class_name": result["diffusion_model_class"],
        "has_manual_cast_dtype": hasattr(diffusion, "manual_cast_dtype") if diffusion else False,
        "has_weight_adapter": bool(getattr(model, "weight_wrapper", None)),
        "v_cache_policy": "not_observable_without_executing_the_attention_backend",
    }
    return result


def _clip_snapshot(clip) -> dict[str, Any]:
    if clip is None:
        return {"connected": False}
    patcher = getattr(clip, "patcher", None)
    return {
        "connected": True,
        "class": type(clip).__name__,
        "patcher_class": None if patcher is None else type(patcher).__name__,
        "load_device": _device(getattr(patcher, "load_device", None)),
        "offload_device": _device(getattr(patcher, "offload_device", None)),
        "host_memory_mode": (
            "offload-capable" if getattr(patcher, "offload_device", None) is not None else "unknown"
        ),
    }


def _external_plugin_snapshot() -> dict[str, Any]:
    custom_nodes = Path(getattr(folder_paths, "base_path", ".")) / "custom_nodes"
    names: list[str] = []
    try:
        names = sorted(path.name for path in custom_nodes.iterdir() if path.is_dir())
    except OSError as error:
        return {"root": str(custom_nodes), "inspection_error": f"{type(error).__name__}: {error}"}
    lowered = {name.lower(): name for name in names}
    candidates = {
        "comfyui-kjnodes": any("kjnodes" in item for item in lowered),
        "comfyui-sol-attn": any("sol-attn" in item or "sol_attn" in item for item in lowered),
        "h3-optimizations-like": any("h3" in item and "optim" in item for item in lowered),
        "block-cache-like": any("block" in item and "cache" in item for item in lowered),
    }
    return {
        "root": str(custom_nodes),
        "detected": candidates,
        "combination_contract": (
            "directory detection is advisory only; unknown forward/attention owners must not be double-wrapped"
        ),
    }


def build_h3_residency_strategy(
    strategy: str,
    policy_epoch: int = 0,
    model=None,
    clip=None,
) -> tuple[dict[str, Any], float, bool, str]:
    if strategy not in RESIDENCY_PRESETS:
        raise ValueError(f"unknown residency strategy: {strategy}")
    preset = RESIDENCY_PRESETS[strategy]
    policy, planner = build_vram_policy(
        mode=preset["policy_mode"],
        fixed_total_reserved_gib=preset["reserve_gib"],
        external_margin_gib=1.0,
        maximum_reserved_gib=8.0,
        clean_before_load=False,
        require_dynamic_vram=True,
        minimum_current_headroom_mib=512.0,
        minimum_commit_headroom_gib=16.0,
        block_when_commit_below_gate=True,
        policy_epoch=int(policy_epoch),
    )
    model_report = _model_snapshot(model)
    clip_report = _clip_snapshot(clip)
    owners = model_report.get("attention_or_forward_owners", [])
    conflicts: list[str] = []
    if len(owners) > 1:
        conflicts.append(
            "multiple forward/attention owner markers are present; inspect the exact composition before applying another wrapper"
        )
    report = {
        "schema": RESIDENCY_SCHEMA,
        "strategy": strategy,
        "intent": preset["intent"],
        "policy": policy,
        "policy_applied": False,
        "side_effects": False,
        "unload_all_models_called": False,
        "model": model_report,
        "clip": clip_report,
        "external_plugins": _external_plugin_snapshot(),
        "conflicts": conflicts,
        "runtime": planner.get("runtime", {}),
        "current_gate_pass": bool(planner.get("current_gate_pass")),
        "memory_safe_claim": False,
        "usage": (
            "connect the returned policy to a compatible T8 loader to opt in; leaving it unconnected is report-only"
        ),
        "limits": [
            "reserve values are starting points, not universal safe values",
            "activation, VAE, attention workspace, other CUDA users and host commit can still OOM",
            "this node owns no model reference and never unloads unrelated ComfyUI models",
        ],
    }
    return (
        policy,
        float(preset["reserve_gib"]),
        bool(report["current_gate_pass"]),
        json.dumps(report, ensure_ascii=False, indent=2),
    )
