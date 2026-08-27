from __future__ import annotations

import ctypes
import importlib
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "minimax_h3_raven_streaming_t8_v1"
REVIEWED_PLUGIN_VERSION = (0, 1, 0)
PUBLISHED_PROFILE = {
    "steps": 4,
    "video_shift": 12.0,
    "audio_shift": 3.0,
    "sink": 2,
    "window": 2,
    "kv_cache_storage": "cpu_pinned",
}
REVIEWED_MIN_GPU_GIB = 23.5
REVIEWED_MIN_TOTAL_RAM_GIB = 192.0
REVIEWED_MIN_AVAILABLE_RAM_GIB = 160.0


def _pretty(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def _version_tuple(value: Any) -> tuple[int, ...]:
    numbers = re.findall(r"\d+", str(value or ""))
    return tuple(int(item) for item in numbers[:3])


def build_raven_streaming_profile(
    preset: str,
    manual_steps: int,
    manual_video_shift: float,
    manual_audio_shift: float,
    manual_sink: int,
    manual_window: int,
    manual_kv_cache_storage: str,
):
    if preset == "published_preview_4nfe":
        values = dict(PUBLISHED_PROFILE)
        source = "published_preview_4nfe"
    elif preset == "manual_experimental":
        values = {
            "steps": int(manual_steps),
            "video_shift": float(manual_video_shift),
            "audio_shift": float(manual_audio_shift),
            "sink": int(manual_sink),
            "window": int(manual_window),
            "kv_cache_storage": str(manual_kv_cache_storage),
        }
        source = "manual_experimental"
    else:
        raise ValueError(f"unknown RAVEN profile preset: {preset!r}")

    if values["steps"] < 1:
        raise ValueError("RAVEN steps must be >= 1")
    if values["video_shift"] <= 0 or values["audio_shift"] <= 0:
        raise ValueError("RAVEN video/audio shifts must be > 0")
    if values["sink"] < 0 or values["window"] < 1:
        raise ValueError("RAVEN sink must be >= 0 and window must be >= 1")
    if values["kv_cache_storage"] not in {"cpu_pinned", "cpu", "gpu"}:
        raise ValueError("unsupported RAVEN KV cache storage")

    exact_published = values == PUBLISHED_PROFILE
    report = {
        "schema": SCHEMA_VERSION,
        "kind": "streaming_profile",
        "source": source,
        "exact_published_profile": exact_published,
        "values": values,
        "boundary": (
            "Only the exact published 4-NFE profile is treated as reviewed. Manual values are "
            "forwarded exactly but remain experimental until separately validated."
        ),
    }
    return (
        values["steps"],
        values["video_shift"],
        values["audio_shift"],
        values["sink"],
        values["window"],
        values["kv_cache_storage"],
        _pretty(report),
    )


def _load_runtime() -> dict[str, Any]:
    package = importlib.import_module("raven_streaming")
    return {
        "package": package,
        "nodes": importlib.import_module("raven_streaming.nodes"),
        "compat": importlib.import_module("raven_streaming.compat"),
        "contracts": importlib.import_module("raven_streaming.contracts"),
        "loader": importlib.import_module("raven_streaming.loader"),
    }


def _module_root(runtime: Mapping[str, Any]) -> str:
    package = runtime.get("package")
    path = getattr(package, "__file__", None)
    return str(Path(path).resolve().parent.parent) if path else "unknown"


def _discover_installations(runtime: Mapping[str, Any]) -> list[str]:
    roots: set[str] = set()
    module_root = _module_root(runtime)
    if module_root != "unknown":
        roots.add(module_root)
        custom_nodes = Path(module_root).parent
        if custom_nodes.name.lower() == "custom_nodes" and custom_nodes.is_dir():
            for child in custom_nodes.iterdir():
                if (
                    child.is_dir()
                    and (child / "raven_streaming" / "__init__.py").is_file()
                ):
                    roots.add(str(child.resolve()))
    return sorted(roots, key=str.casefold)


def _system_memory_gib() -> tuple[float | None, float | None]:
    if os.name == "nt":

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("memory_load", ctypes.c_ulong),
                ("total_phys", ctypes.c_ulonglong),
                ("avail_phys", ctypes.c_ulonglong),
                ("total_page_file", ctypes.c_ulonglong),
                ("avail_page_file", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("avail_virtual", ctypes.c_ulonglong),
                ("avail_extended_virtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            gib = float(1024**3)
            return status.total_phys / gib, status.avail_phys / gib
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        values = {}
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            key, _, raw = line.partition(":")
            if raw:
                values[key] = int(raw.strip().split()[0]) * 1024
        gib = float(1024**3)
        return values.get("MemTotal", 0) / gib, values.get("MemAvailable", 0) / gib
    return None, None


def raven_hardware_snapshot() -> dict[str, Any]:
    result: dict[str, Any] = {
        "cuda_available": False,
        "bf16_supported": False,
        "gpu_name": None,
        "gpu_total_gib": None,
        "system_total_ram_gib": None,
        "system_available_ram_gib": None,
    }
    total_ram, available_ram = _system_memory_gib()
    result["system_total_ram_gib"] = total_ram
    result["system_available_ram_gib"] = available_ram
    try:
        import torch

        result["cuda_available"] = bool(torch.cuda.is_available())
        if result["cuda_available"]:
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            result["gpu_name"] = str(props.name)
            result["gpu_total_gib"] = int(props.total_memory) / float(1024**3)
            supported = getattr(torch.cuda, "is_bf16_supported", None)
            result["bf16_supported"] = (
                bool(supported()) if callable(supported) else False
            )
    except Exception as exc:  # noqa: BLE001 - reported, then fail-closed by policy
        result["probe_error"] = f"{type(exc).__name__}: {exc}"
    return result


def _resolve_selected_file(folder: str, name: str) -> tuple[str | None, int | None]:
    if not name or name == "<missing>":
        return None, None
    try:
        folder_paths = importlib.import_module("folder_paths")
        getter = getattr(folder_paths, "get_full_path_or_raise", None)
        if callable(getter):
            path = Path(getter(folder, name)).resolve()
        else:
            path_value = folder_paths.get_full_path(folder, name)
            path = Path(path_value).resolve() if path_value else None
        if path is None or not path.is_file():
            return None, None
        return str(path), int(path.stat().st_size)
    except Exception:  # noqa: BLE001 - upstream loader will give the detailed path error
        return None, None


def _feature_report(runtime: Mapping[str, Any]) -> tuple[bool, Any]:
    report = runtime["compat"].check_features()
    payload = report.to_dict() if hasattr(report, "to_dict") else {"text": str(report)}
    return bool(getattr(report, "ok", False)), payload


def _finding(code: str, message: str) -> dict[str, str]:
    return {"code": code, "message": message}


def _preflight_loader(
    unet_name: str,
    lora_name: str,
    weight_dtype: str,
    *,
    runtime: Mapping[str, Any] | None = None,
    hardware: Mapping[str, Any] | None = None,
    installations: list[str] | None = None,
    resolved_files: Mapping[str, tuple[str | None, int | None]] | None = None,
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    mechanical: list[dict[str, str]] = []
    reviewed: list[dict[str, str]] = []
    model_diagnostics: list[dict[str, str]] = []
    runtime_error = None
    if runtime is None:
        try:
            runtime = _load_runtime()
        except Exception as exc:  # noqa: BLE001
            runtime_error = f"{type(exc).__name__}: {exc}"
            runtime = {}

    if runtime_error:
        mechanical.append(
            _finding(
                "RAVEN_PLUGIN_MISSING_OR_BROKEN",
                "Install the external MiniMax H3 RAVEN Streaming plugin separately: "
                + runtime_error,
            )
        )

    package = runtime.get("package") if runtime else None
    version = str(getattr(package, "__version__", "unknown"))
    if runtime and _version_tuple(version) < REVIEWED_PLUGIN_VERSION:
        mechanical.append(
            _finding(
                "RAVEN_PLUGIN_VERSION", f"RAVEN plugin {version} is older than 0.1.0"
            )
        )

    installs = (
        installations
        if installations is not None
        else (_discover_installations(runtime) if runtime else [])
    )
    if runtime and len(installs) != 1:
        mechanical.append(
            _finding(
                "RAVEN_INSTALLATION_COUNT",
                f"Expected exactly one external RAVEN plugin installation, found {len(installs)}",
            )
        )

    features: Any = None
    if runtime:
        try:
            feature_ok, features = _feature_report(runtime)
            if not feature_ok:
                mechanical.append(
                    _finding(
                        "COMFY_FEATURE_PROBE",
                        "RAVEN required ComfyUI features are missing",
                    )
                )
        except Exception as exc:  # noqa: BLE001
            mechanical.append(
                _finding(
                    "COMFY_FEATURE_PROBE",
                    f"feature probe failed: {type(exc).__name__}: {exc}",
                )
            )

        loader_class = getattr(runtime.get("nodes"), "RAVENModelLoader", None)
        if loader_class is None:
            mechanical.append(
                _finding(
                    "RAVEN_LOADER_CLASS", "external RAVENModelLoader is unavailable"
                )
            )

    if weight_dtype not in {"default", "bf16", "fp32"}:
        mechanical.append(
            _finding("WEIGHT_DTYPE", f"unsupported weight dtype {weight_dtype!r}")
        )
    if weight_dtype == "fp32":
        reviewed.append(
            _finding(
                "FP32_OUTSIDE_REVIEW",
                "FP32 doubles the already large base-weight footprint",
            )
        )

    lower_name = unet_name.casefold()
    quantized_tokens = ("int8", "fp8", "nvfp", "quant", "convrot", "pruned")
    if any(token in lower_name for token in quantized_tokens):
        model_diagnostics.append(
            _finding(
                "QUANTIZED_OR_PRUNED_BASE",
                "RAVEN v0.1 requires the full non-pruned, non-quantized BF16 H3 base model",
            )
        )

    if resolved_files is None:
        resolved_files = {
            "unet": _resolve_selected_file("diffusion_models", unet_name),
            "lora": _resolve_selected_file("loras", lora_name),
        }
    file_report = {}
    for key in ("unet", "lora"):
        path, size = resolved_files.get(key, (None, None))
        file_report[key] = {"path": path, "bytes": size}
        if not path:
            model_diagnostics.append(
                _finding(
                    f"{key.upper()}_FILE",
                    f"selected RAVEN {key} file cannot be resolved",
                )
            )

    hw = dict(hardware or raven_hardware_snapshot())
    if not hw.get("cuda_available"):
        model_diagnostics.append(_finding("CUDA_REQUIRED", "RAVEN streaming requires CUDA"))
    if not hw.get("bf16_supported"):
        model_diagnostics.append(
            _finding("BF16_REQUIRED", "the selected GPU must support BF16")
        )

    gpu_gib = hw.get("gpu_total_gib")
    total_ram = hw.get("system_total_ram_gib")
    available_ram = hw.get("system_available_ram_gib")
    if gpu_gib is None or float(gpu_gib) < REVIEWED_MIN_GPU_GIB:
        reviewed.append(
            _finding(
                "GPU_OUTSIDE_REVIEWED_ENVELOPE",
                f"reviewed minimum is approximately 24 GiB VRAM; detected {gpu_gib!r}",
            )
        )
    if total_ram is None or float(total_ram) < REVIEWED_MIN_TOTAL_RAM_GIB:
        reviewed.append(
            _finding(
                "RAM_TOTAL_OUTSIDE_REVIEWED_ENVELOPE",
                f"reviewed host-memory envelope is 192 GiB total; detected {total_ram!r}",
            )
        )
    if available_ram is None or float(available_ram) < REVIEWED_MIN_AVAILABLE_RAM_GIB:
        reviewed.append(
            _finding(
                "RAM_AVAILABLE_OUTSIDE_REVIEWED_ENVELOPE",
                f"reviewed pre-load availability is 160 GiB; detected {available_ram!r}",
            )
        )

    report = {
        "schema": SCHEMA_VERSION,
        "kind": "guarded_loader_preflight",
        "plugin": {
            "version": version,
            "root": _module_root(runtime) if runtime else None,
            "installations": installs,
            "source_revision_reviewed": "bcfa38138ddf1a5041af9880760815874138d4e1",
        },
        "selection": {
            "unet_name": unet_name,
            "lora_name": lora_name,
            "weight_dtype": weight_dtype,
        },
        "files": file_report,
        "hardware": hw,
        "comfy_feature_report": features,
        "mechanical_findings": mechanical,
        "model_and_runtime_diagnostics": model_diagnostics,
        "reviewed_envelope_findings": reviewed,
        "mechanically_compatible": not mechanical,
        "inside_reviewed_envelope": not reviewed,
        "scientific_boundary": (
            "The external v0.1 runtime is T2VA-only and was validated on an H200 constrained to "
            "a 24.1-GiB free-memory window, not on a physical 24-GiB consumer card. Its recorded "
            "host RSS is above 129 GiB. Passing this preflight is not a quality or OOM guarantee."
        ),
    }
    return report, runtime


def load_raven_model_guarded(
    unet_name: str,
    lora_name: str,
    weight_dtype: str,
    enforcement: str,
    *,
    runtime: Mapping[str, Any] | None = None,
    hardware: Mapping[str, Any] | None = None,
    installations: list[str] | None = None,
    resolved_files: Mapping[str, tuple[str | None, int | None]] | None = None,
):
    report, runtime = _preflight_loader(
        unet_name,
        lora_name,
        weight_dtype,
        runtime=runtime,
        hardware=hardware,
        installations=installations,
        resolved_files=resolved_files,
    )
    mechanical = report["mechanical_findings"]
    reviewed = report["reviewed_envelope_findings"]
    blocked = bool(mechanical) or (
        enforcement == "block_outside_reviewed_envelope" and bool(reviewed)
    )
    if enforcement not in {
        "report_only",
        "block_mechanical_conflicts",
        "block_outside_reviewed_envelope",
    }:
        raise ValueError(f"unknown RAVEN loader enforcement: {enforcement!r}")
    # User-selected model identity, file selection and hardware suitability are
    # diagnostic only. The delegated runtime is authoritative and may raise its
    # native error when those selections cannot execute.
    if blocked:
        codes = [item["code"] for item in mechanical + reviewed]
        raise RuntimeError(
            "RAVEN guarded loader blocked before model load: " + ", ".join(codes)
        )

    loader_class = getattr(runtime["nodes"], "RAVENModelLoader")
    loaded = loader_class().load_model(unet_name, lora_name, weight_dtype)
    model = loaded[0] if isinstance(loaded, (tuple, list)) else loaded
    report["decision"] = "LOADED_OUTSIDE_REVIEW" if reviewed else "LOADED"
    report["delegate"] = "external raven_streaming.nodes.RAVENModelLoader"
    return model, _pretty(report)


def _attachment_report(attachment: Any) -> dict[str, Any]:
    if attachment is None:
        return {"present": False}
    try:
        count = len(attachment)
    except Exception:  # noqa: BLE001
        count = None
    return {
        "present": True,
        "modules": count,
        "rank": getattr(attachment, "rank", None),
        "alpha": getattr(attachment, "alpha", None),
        "strength": getattr(attachment, "strength", None),
        "detached": getattr(attachment, "detached", None),
    }


def audit_raven_streaming_request(
    model: Any,
    positive: Any,
    latent: Any,
    steps: int,
    video_shift: float,
    audio_shift: float,
    sink: int,
    window: int,
    kv_cache_storage: str,
    allow_experimental_over_192: bool,
    enforcement: str,
    *,
    runtime: Mapping[str, Any] | None = None,
):
    mechanical: list[dict[str, str]] = []
    reviewed: list[dict[str, str]] = []
    runtime = runtime or _load_runtime()

    try:
        feature_ok, features = _feature_report(runtime)
        if not feature_ok:
            mechanical.append(
                _finding("COMFY_FEATURE_PROBE", "required features are missing")
            )
    except Exception as exc:  # noqa: BLE001
        features = {"error": f"{type(exc).__name__}: {exc}"}
        mechanical.append(_finding("COMFY_FEATURE_PROBE", features["error"]))

    conditioning_info = None
    latent_info = None
    model_info = None
    attachment_info = {"present": False}
    try:
        parsed_conditioning = runtime["contracts"].parse_conditioning(positive)
        conditioning_info = {"tokens": int(parsed_conditioning.cross_attn.shape[1])}
    except Exception as exc:  # noqa: BLE001
        mechanical.append(
            _finding("CONDITIONING_CONTRACT", f"{type(exc).__name__}: {exc}")
        )
    try:
        request = runtime["contracts"].parse_latent(latent)
        latent_info = {
            "frames": int(request.frames),
            "width": int(request.width),
            "height": int(request.height),
            "video_latent_t": int(request.latent_t),
            "audio_latent_t": int(request.audio_t),
        }
        if request.frames > 192 and not allow_experimental_over_192:
            reviewed.append(
                _finding(
                    "LONG_REQUEST_ACK_REQUIRED",
                    f"{request.frames} frames exceeds the reviewed <=192-frame band",
                )
            )
    except Exception as exc:  # noqa: BLE001
        mechanical.append(_finding("LATENT_CONTRACT", f"{type(exc).__name__}: {exc}"))

    try:
        resolved_model = runtime["contracts"].resolve_model(model)
        model_info = {
            "patcher_class": type(resolved_model.patcher).__name__,
            "diffusion_model_class": type(resolved_model.diffusion_model).__name__,
            "num_layers": int(resolved_model.num_layers),
        }
        object_patches = getattr(model, "object_patches", None)
        if object_patches:
            mechanical.append(
                _finding(
                    "OBJECT_PATCH_CONFLICT",
                    "RAVEN MODEL carries object patches; attention/sampler/model replacements "
                    "have not been proven compatible with the causal KV-cache lane",
                )
            )
        attachment = runtime["loader"].get_raven_attachment(model)
        attachment_info = _attachment_report(attachment)
        expected = {"modules": 266, "rank": 128, "alpha": 128.0, "strength": 1.0}
        for key, value in expected.items():
            actual = attachment_info.get(key)
            if actual is None or float(actual) != float(value):
                mechanical.append(
                    _finding(
                        "RAVEN_ATTACHMENT",
                        f"mandatory adapter {key} is {actual!r}, expected {value!r}",
                    )
                )
    except Exception as exc:  # noqa: BLE001
        mechanical.append(_finding("MODEL_CONTRACT", f"{type(exc).__name__}: {exc}"))

    values = {
        "steps": int(steps),
        "video_shift": float(video_shift),
        "audio_shift": float(audio_shift),
        "sink": int(sink),
        "window": int(window),
        "kv_cache_storage": str(kv_cache_storage),
    }
    for key, expected in PUBLISHED_PROFILE.items():
        if values[key] != expected:
            reviewed.append(
                _finding(
                    "PROFILE_DEVIATION",
                    f"{key}={values[key]!r} differs from published {expected!r}",
                )
            )
    if kv_cache_storage == "gpu":
        reviewed.append(
            _finding(
                "GPU_KV_OUTSIDE_24GIB",
                "GPU KV storage alone is documented near 28 GiB for the published request",
            )
        )

    if enforcement not in {
        "report_only",
        "block_mechanical_conflicts",
        "block_outside_reviewed_envelope",
    }:
        raise ValueError(f"unknown RAVEN request enforcement: {enforcement!r}")
    blocked = bool(mechanical) or (
        enforcement == "block_outside_reviewed_envelope" and bool(reviewed)
    )
    compatible = not mechanical and not (
        enforcement == "block_outside_reviewed_envelope" and reviewed
    )
    decision = "PASS" if not mechanical and not reviewed else "ABSTAIN"
    if blocked and enforcement != "report_only":
        codes = [item["code"] for item in mechanical + reviewed]
        raise ValueError("RAVEN request audit blocked: " + ", ".join(codes))

    report = {
        "schema": SCHEMA_VERSION,
        "kind": "request_audit",
        "decision": decision,
        "compatible": compatible,
        "mechanical_findings": mechanical,
        "reviewed_envelope_findings": reviewed,
        "profile": values,
        "conditioning": conditioning_info,
        "latent": latent_info,
        "model": model_info,
        "attachment": attachment_info,
        "comfy_feature_report": features,
        "scope": (
            "T2VA only. Keyframes, reference rows, sampled latents, masks, CFG, negative "
            "conditioning, stock samplers and external model/attention patches are not supported."
        ),
    }
    return model, positive, latent, compatible, decision, _pretty(report)
