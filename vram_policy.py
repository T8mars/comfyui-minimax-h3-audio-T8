from __future__ import annotations

import ctypes
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import threading
import time
from typing import Any

import torch


VRAM_POLICY_TYPE = "H3_T8_VRAM_POLICY"
VRAM_POLICY_SCHEMA = "t8.minimax_h3.vram_policy.v1"
VRAM_POLICY_MODES = (
    "report_only",
    "fixed_total_reserved_exp",
    "external_usage_plus_margin_exp",
)
GIB = 1024**3
MIB = 1024**2
_POLICY_LOCK = threading.RLock()
_LAST_SIMPLE_HEADROOM_BYTES: int | None = None


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _round_gib(value: float) -> float:
    return round(float(value), 6)


def host_memory_snapshot() -> dict[str, Any]:
    """Report host RAM and commit headroom without adding a runtime dependency."""
    result: dict[str, Any] = {"platform": os.name}
    if os.name == "nt":
        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatusEx()
        status.dwLength = ctypes.sizeof(status)
        try:
            ok = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        except (AttributeError, OSError) as error:
            result["inspection_error"] = f"{type(error).__name__}: {error}"
            return result
        if not ok:
            result["inspection_error"] = "GlobalMemoryStatusEx returned false"
            return result
        result.update(
            {
                "source": "GlobalMemoryStatusEx",
                "ram_total_gib": _round_gib(status.ullTotalPhys / GIB),
                "ram_available_gib": _round_gib(status.ullAvailPhys / GIB),
                "commit_limit_gib": _round_gib(status.ullTotalPageFile / GIB),
                "commit_headroom_gib": _round_gib(status.ullAvailPageFile / GIB),
                "commit_used_gib": _round_gib(
                    (status.ullTotalPageFile - status.ullAvailPageFile) / GIB
                ),
            }
        )
        return result

    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        values: dict[str, int] = {}
        try:
            for line in meminfo.read_text(encoding="utf-8").splitlines():
                key, raw = line.split(":", 1)
                token = raw.strip().split()[0]
                values[key] = int(token) * 1024
        except (OSError, ValueError, IndexError) as error:
            result["inspection_error"] = f"{type(error).__name__}: {error}"
            return result
        result.update(
            {
                "source": "/proc/meminfo",
                "ram_total_gib": _round_gib(values.get("MemTotal", 0) / GIB),
                "ram_available_gib": _round_gib(
                    values.get("MemAvailable", values.get("MemFree", 0)) / GIB
                ),
            }
        )
        if "CommitLimit" in values and "Committed_AS" in values:
            result.update(
                {
                    "commit_limit_gib": _round_gib(values["CommitLimit"] / GIB),
                    "commit_used_gib": _round_gib(values["Committed_AS"] / GIB),
                    "commit_headroom_gib": _round_gib(
                        max(0, values["CommitLimit"] - values["Committed_AS"]) / GIB
                    ),
                }
            )
        return result

    result["inspection_error"] = "host commit telemetry is unavailable"
    return result


def _runtime_modules():
    import comfy.memory_management as memory_management
    import comfy.model_management as model_management

    return model_management, memory_management


def _aimdo_control():
    import comfy_aimdo.control as control

    return control


def _extra_reserved_bytes(model_management) -> int | None:
    try:
        getter = getattr(model_management, "extra_reserved_memory", None)
        value = getter() if callable(getter) else model_management.EXTRA_RESERVED_VRAM
        return int(value)
    except (AttributeError, TypeError, ValueError):
        return None


def runtime_snapshot() -> dict[str, Any]:
    result: dict[str, Any] = {
        "captured_unix": time.time(),
        "gpu": {},
        "comfy": {},
        "aimdo": {},
        "host": host_memory_snapshot(),
    }
    try:
        model_management, memory_management = _runtime_modules()
        device = model_management.get_torch_device()
        extra_reserved = _extra_reserved_bytes(model_management)
        result["comfy"].update(
            {
                "dynamic_vram_enabled": bool(
                    getattr(memory_management, "aimdo_enabled", False)
                ),
                "extra_reserved_vram_gib": (
                    None if extra_reserved is None else _round_gib(extra_reserved / GIB)
                ),
            }
        )
        try:
            from comfy.cli_args import args

            result["comfy"].update(
                {
                    "startup_reserve_vram_gib": getattr(args, "reserve_vram", None),
                    "startup_vram_headroom_gib": float(
                        getattr(args, "vram_headroom", 0.0) or 0.0
                    ),
                }
            )
        except (ImportError, AttributeError, TypeError, ValueError) as error:
            result["comfy"]["cli_inspection_error"] = (
                f"{type(error).__name__}: {error}"
            )
        result["gpu"]["device"] = str(device)
        if getattr(device, "type", None) == "cuda" and torch.cuda.is_available():
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
            result["gpu"].update(
                {
                    "whole_device_free_mib": free_bytes / MIB,
                    "whole_device_total_mib": total_bytes / MIB,
                    "whole_device_used_mib": (total_bytes - free_bytes) / MIB,
                    "torch_allocated_mib": torch.cuda.memory_allocated(device) / MIB,
                    "torch_reserved_mib": torch.cuda.memory_reserved(device) / MIB,
                }
            )
    except Exception as error:
        result["comfy"]["inspection_error"] = f"{type(error).__name__}: {error}"

    try:
        control = _aimdo_control()
        result["aimdo"].update(
            {
                "package_version": importlib.metadata.version("comfy-aimdo"),
                "library_loaded": getattr(control, "lib", None) is not None,
                "initialized_device_count": len(getattr(control, "devctxs", []) or []),
                "simple_headroom_readback_available": False,
                "last_t8_simple_headroom_gib": (
                    None
                    if _LAST_SIMPLE_HEADROOM_BYTES is None
                    else _round_gib(_LAST_SIMPLE_HEADROOM_BYTES / GIB)
                ),
            }
        )
        usage = getattr(control, "get_total_vram_usage", None)
        if callable(usage) and getattr(control, "lib", None) is not None:
            result["aimdo"]["managed_vram_usage_mib"] = int(usage()) / MIB
    except Exception as error:
        result["aimdo"]["inspection_error"] = f"{type(error).__name__}: {error}"
    return result


def _policy_payload(policy: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "schema",
        "mode",
        "fixed_total_reserved_gib",
        "external_margin_gib",
        "maximum_reserved_gib",
        "clean_before_load",
        "require_dynamic_vram",
        "minimum_current_headroom_mib",
        "minimum_commit_headroom_gib",
        "block_when_commit_below_gate",
        "policy_epoch",
    )
    return {key: policy[key] for key in keys}


def _validate_policy_values(values: dict[str, Any]) -> None:
    mode = values["mode"]
    if mode not in VRAM_POLICY_MODES:
        raise ValueError(f"unsupported VRAM policy mode: {mode!r}")
    fixed = float(values["fixed_total_reserved_gib"])
    margin = float(values["external_margin_gib"])
    maximum = float(values["maximum_reserved_gib"])
    current_gate = float(values["minimum_current_headroom_mib"])
    commit_gate = float(values["minimum_commit_headroom_gib"])
    numeric = (fixed, margin, maximum, current_gate, commit_gate)
    if not all(math.isfinite(value) for value in numeric):
        raise ValueError("VRAM policy numeric inputs must be finite")
    if not 0.0 <= fixed <= 64.0:
        raise ValueError("fixed_total_reserved_gib must be between 0 and 64")
    if not 0.0 <= margin <= 64.0:
        raise ValueError("external_margin_gib must be between 0 and 64")
    if not 0.25 <= maximum <= 64.0:
        raise ValueError("maximum_reserved_gib must be between 0.25 and 64")
    if current_gate < 0.0 or commit_gate < 0.0:
        raise ValueError("headroom gates must be non-negative")
    epoch = int(values["policy_epoch"])
    if epoch < 0 or epoch > 0x7FFFFFFF:
        raise ValueError("policy_epoch must be between 0 and 2147483647")
    if mode == "external_usage_plus_margin_exp" and not bool(
        values["clean_before_load"]
    ):
        raise ValueError(
            "external_usage_plus_margin_exp requires clean_before_load=true so cached "
            "ComfyUI models are not misclassified as external GPU usage"
        )


def build_vram_policy(
    mode: str,
    fixed_total_reserved_gib: float,
    external_margin_gib: float,
    maximum_reserved_gib: float,
    clean_before_load: bool,
    require_dynamic_vram: bool,
    minimum_current_headroom_mib: float,
    minimum_commit_headroom_gib: float,
    block_when_commit_below_gate: bool,
    policy_epoch: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    fixed = float(fixed_total_reserved_gib)
    margin = float(external_margin_gib)
    maximum = float(maximum_reserved_gib)
    current_gate = float(minimum_current_headroom_mib)
    commit_gate = float(minimum_commit_headroom_gib)
    policy = {
        "schema": VRAM_POLICY_SCHEMA,
        "mode": mode,
        "fixed_total_reserved_gib": fixed,
        "external_margin_gib": margin,
        "maximum_reserved_gib": maximum,
        "clean_before_load": bool(clean_before_load),
        "require_dynamic_vram": bool(require_dynamic_vram),
        "minimum_current_headroom_mib": current_gate,
        "minimum_commit_headroom_gib": commit_gate,
        "block_when_commit_below_gate": bool(block_when_commit_below_gate),
        "policy_epoch": int(policy_epoch),
    }
    _validate_policy_values(policy)
    policy["fingerprint"] = _sha256_json(_policy_payload(policy))
    snapshot = runtime_snapshot()
    free_mib = snapshot.get("gpu", {}).get("whole_device_free_mib")
    commit_free = snapshot.get("host", {}).get("commit_headroom_gib")
    warnings = [
        "Planner only: no global memory setting changes until this policy is connected to a compatible loader.",
        "Reserved VRAM and VBAR reduce weight residency pressure but cannot guarantee that activations, attention workspaces, VAEs, CLIP, CUDA users, host pins, or system commit will not OOM.",
    ]
    if mode == "report_only" and clean_before_load:
        warnings.append("clean_before_load is ignored in report_only mode.")
    report = {
        "schema": "t8.minimax_h3.vram_policy_planner_report.v1",
        "policy": policy,
        "runtime": snapshot,
        "current_gate_pass": (
            False if free_mib is None else float(free_mib) >= current_gate
        ),
        "commit_gate_pass": (
            None if commit_free is None else float(commit_free) >= commit_gate
        ),
        "memory_safe_claim": False,
        "warnings": warnings,
    }
    return policy, report


def validate_vram_policy(policy: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(policy, dict) or policy.get("schema") != VRAM_POLICY_SCHEMA:
        raise ValueError("vram_policy is not an H3 T8 VRAM policy descriptor")
    payload = _policy_payload(policy)
    expected = _sha256_json(payload)
    if policy.get("fingerprint") != expected:
        raise ValueError("vram_policy fingerprint mismatch")
    _validate_policy_values(payload)
    return dict(policy)


def policy_descriptor_fingerprint(policy: dict[str, Any]) -> str:
    return str(validate_vram_policy(policy)["fingerprint"])


def policy_input_fingerprint(**kwargs: Any) -> str:
    payload = {"schema": VRAM_POLICY_SCHEMA, **kwargs}
    return _sha256_json(payload)


def _set_model_management_reserved(model_management, target_bytes: int) -> str:
    setter = getattr(model_management, "set_extra_reserved_vram", None)
    if callable(setter):
        setter(target_bytes / GIB)
        return "set_extra_reserved_vram_gib"
    model_management.EXTRA_RESERVED_VRAM = int(target_bytes)
    return "EXTRA_RESERVED_VRAM_bytes"


def _target_reserved_gib(policy: dict[str, Any], snapshot: dict[str, Any]) -> tuple[float, float, bool]:
    if policy["mode"] == "fixed_total_reserved_exp":
        raw = float(policy["fixed_total_reserved_gib"])
    else:
        gpu = snapshot.get("gpu", {})
        total_mib = gpu.get("whole_device_total_mib")
        free_mib = gpu.get("whole_device_free_mib")
        if total_mib is None or free_mib is None:
            raise RuntimeError("whole-device CUDA memory telemetry is required for auto policy")
        raw = (float(total_mib) - float(free_mib)) / 1024.0
        raw += float(policy["external_margin_gib"])
    target = min(max(0.0, raw), float(policy["maximum_reserved_gib"]))
    return raw, target, target < raw


def apply_vram_policy(policy: dict[str, Any]) -> dict[str, Any]:
    """Apply a validated process-global policy before a diffusion model is loaded."""
    global _LAST_SIMPLE_HEADROOM_BYTES

    validated = validate_vram_policy(policy)
    before = runtime_snapshot()
    if validated["mode"] == "report_only":
        return {
            "schema": "t8.minimax_h3.vram_policy_apply_report.v1",
            "policy_fingerprint": validated["fingerprint"],
            "mode": validated["mode"],
            "applied": False,
            "cleanup_performed": False,
            "before": before,
            "after": before,
            "memory_safe_claim": False,
            "warnings": ["report_only mode made no global memory-policy changes."],
        }

    with _POLICY_LOCK:
        model_management, memory_management = _runtime_modules()
        dynamic_enabled = bool(getattr(memory_management, "aimdo_enabled", False))
        control = None
        simple_setter = None
        if dynamic_enabled:
            try:
                control = _aimdo_control()
                simple_setter = getattr(
                    getattr(control, "lib", None),
                    "set_simple_vram_headroom",
                    None,
                )
            except Exception:
                simple_setter = None
        if validated["require_dynamic_vram"] and not dynamic_enabled:
            raise RuntimeError(
                "VRAM policy requires DynamicVRAM/VBAR, but comfy.memory_management.aimdo_enabled is false"
            )
        if validated["require_dynamic_vram"] and not callable(simple_setter):
            raise RuntimeError(
                "DynamicVRAM is enabled but the direct simple-headroom setter is unavailable"
            )

        commit_headroom = before.get("host", {}).get("commit_headroom_gib")
        if (
            validated["block_when_commit_below_gate"]
            and commit_headroom is not None
            and float(commit_headroom) < validated["minimum_commit_headroom_gib"]
        ):
            raise RuntimeError(
                "host commit headroom is below the configured gate: "
                f"{float(commit_headroom):.3f} GiB < "
                f"{validated['minimum_commit_headroom_gib']:.3f} GiB"
            )

        cleanup_performed = False
        if validated["clean_before_load"]:
            model_management.unload_all_models()
            model_management.soft_empty_cache()
            gc.collect()
            cleanup_performed = True
        basis = runtime_snapshot()
        raw_target, target_gib, capped = _target_reserved_gib(validated, basis)
        total_mib = basis.get("gpu", {}).get("whole_device_total_mib")
        if total_mib is not None and target_gib >= float(total_mib) / 1024.0:
            raise RuntimeError("reserved VRAM target must remain below total device VRAM")
        target_bytes = int(round(target_gib * GIB))
        previous_bytes = _extra_reserved_bytes(model_management)
        route = _set_model_management_reserved(model_management, target_bytes)
        dynamic_route = "not_enabled"
        try:
            if callable(simple_setter):
                simple_setter(target_bytes)
                _LAST_SIMPLE_HEADROOM_BYTES = target_bytes
                dynamic_route = "direct_lib.set_simple_vram_headroom"
            elif dynamic_enabled:
                dynamic_route = "setter_unavailable"
        except Exception:
            if previous_bytes is not None:
                _set_model_management_reserved(model_management, previous_bytes)
            raise
        after = runtime_snapshot()
        free_mib = basis.get("gpu", {}).get("whole_device_free_mib")
        warnings = [
            "This is a process-global policy and persists until another policy changes it or ComfyUI restarts.",
            "The startup per-device --vram-headroom value is unchanged; this node adjusts ComfyUI reserve plus AIMDO simple headroom only.",
            "VBAR can still report OOM, and non-weight CUDA allocations or exhausted host commit can still fail.",
        ]
        if cleanup_performed:
            warnings.append(
                "clean_before_load globally unloaded all ComfyUI models; it was not H3-only."
            )
        if capped:
            warnings.append(
                "The raw auto target exceeded maximum_reserved_gib and was capped; external-use isolation is reduced."
            )
        if dynamic_enabled and not callable(simple_setter):
            warnings.append(
                "DynamicVRAM was enabled but simple headroom could not be synchronized."
            )
        return {
            "schema": "t8.minimax_h3.vram_policy_apply_report.v1",
            "policy_fingerprint": validated["fingerprint"],
            "mode": validated["mode"],
            "applied": True,
            "policy_scope": "global_process_persistent",
            "cleanup_performed": cleanup_performed,
            "raw_target_reserved_gib": raw_target,
            "target_reserved_gib": target_gib,
            "target_capped": capped,
            "model_management_route": route,
            "dynamic_vram_route": dynamic_route,
            "before": before,
            "calculation_basis": basis,
            "after": after,
            "current_gate_pass": (
                False
                if free_mib is None
                else float(free_mib) >= validated["minimum_current_headroom_mib"]
            ),
            "commit_gate_pass": (
                None
                if commit_headroom is None
                else float(commit_headroom)
                >= validated["minimum_commit_headroom_gib"]
            ),
            "memory_safe_claim": False,
            "warnings": warnings,
        }
