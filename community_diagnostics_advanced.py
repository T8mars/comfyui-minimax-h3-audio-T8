from __future__ import annotations

from collections.abc import Mapping
import importlib
import inspect
import json
import math
from typing import Any


GENERIC_LOOP_SCHEMA = "t8.minimax_h3.generic_loop_capability.v1"
OFFICIAL_DIAGNOSTIC_SCHEMA = "t8.minimax_h3.official_issue_diagnostic.v1"


def _json_object(value: str, name: str) -> dict[str, Any]:
    text = str(value or "").strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"{name} is invalid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be a JSON object")
    return payload


def _safe_signature(value: object) -> str | None:
    try:
        return str(inspect.signature(value))
    except (TypeError, ValueError):
        return None


def probe_generic_loop_capability() -> tuple[bool, str, str]:
    """Report the draft core loop contract without importing or enabling it."""

    modules: dict[str, Any] = {}
    import_errors: dict[str, str] = {}
    for module_name in (
        "comfy_execution.graph",
        "comfy_execution.graph_utils",
        "comfy_api.latest",
        "comfy_extras.nodes_loop",
        "comfy_extras.nodes_loops",
    ):
        try:
            modules[module_name] = importlib.import_module(module_name)
        except Exception as error:
            import_errors[module_name] = f"{type(error).__name__}: {error}"

    symbol_candidates = {
        "Loop": [],
        "CloseLoop": [],
        "LoopVariable": [],
        "ExecutionList": [],
        "EXECUTION_LIST": [],
    }
    for module_name, module in modules.items():
        for symbol in symbol_candidates:
            value = getattr(module, symbol, None)
            if value is not None:
                symbol_candidates[symbol].append(
                    {
                        "module": module_name,
                        "type": type(value).__name__,
                        "signature": _safe_signature(value),
                    }
                )

    graph = modules.get("comfy_execution.graph")
    execution_list = getattr(graph, "ExecutionList", None) if graph else None
    methods = {
        name: _safe_signature(getattr(execution_list, name))
        for name in (
            "add_projection",
            "release_projection",
            "get_descendants",
            "defer_staged_node",
        )
        if execution_list is not None and callable(getattr(execution_list, name, None))
    }
    complete = all(symbol_candidates[name] for name in ("Loop", "CloseLoop", "LoopVariable"))
    scheduler_projection = bool(methods)
    available = bool(complete and scheduler_projection)
    status = "AVAILABLE_DRAFT_CONTRACT_DETECTED" if available else "UNAVAILABLE_CURRENT_CORE"
    report = {
        "schema": GENERIC_LOOP_SCHEMA,
        "status": status,
        "available": available,
        "source_pr": "https://github.com/Comfy-Org/ComfyUI/pull/15923",
        "source_pr_state_at_2026_08_28": "draft",
        "symbols": symbol_candidates,
        "execution_list_projection_methods": methods,
        "import_errors": import_errors,
        "side_effects": False,
        "backend_imported_for_execution": False,
        "workflow_switched": False,
        "integration_decision": (
            "do not switch T8 long-video execution while the upstream scheduler contract is draft; "
            "use the released T8 in-node loop and probe again after the core contract lands"
        ),
        "why_it_matters": [
            "downstream-to-leaf iteration can preview and save each segment",
            "loop-carried last-frame/context values can feed the next H3 segment",
            "incremental save can avoid retaining an entire long video in RAM",
        ],
        "boundary": (
            "symbol presence is only a capability observation. It does not validate nested loops, "
            "cancellation, cache invalidation, crash recovery, audio continuity, or H3 quality"
        ),
    }
    return available, status, json.dumps(report, ensure_ascii=False, indent=2)


def _finite_number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _dark_flash_periodicity(report: Mapping[str, Any]) -> dict[str, Any]:
    indices = report.get("dark_frame_indices", [])
    if not isinstance(indices, list):
        return {"state": "unknown", "evidence": "dark_frame_indices is not a list"}
    normalized = sorted({int(value) for value in indices if isinstance(value, int) and value >= 0})
    if len(normalized) < 3:
        return {
            "state": "unknown",
            "evidence": "fewer than three observed dark-frame indices",
            "indices": normalized,
        }
    deltas = [b - a for a, b in zip(normalized, normalized[1:])]
    hits = sum(delta == 17 for delta in deltas)
    ratio = hits / len(deltas)
    return {
        "state": "suspected" if ratio >= 0.6 else "not_observed",
        "indices": normalized,
        "delta_17_ratio": ratio,
        "evidence": "diagnostic periodicity only; cuts, flashes and source lighting can be confounders",
    }


def diagnose_official_h3_risks(
    width: int,
    height: int,
    length: int,
    reference_media_count: int,
    speaker_count: int,
    isolated_voice_reference_count: int,
    attention_backend: str,
    runtime_report_json: str = "",
    audio_report_json: str = "",
    frame_report_json: str = "",
) -> tuple[str, int, str]:
    runtime = _json_object(runtime_report_json, "runtime_report_json")
    audio = _json_object(audio_report_json, "audio_report_json")
    frames = _json_object(frame_report_json, "frame_report_json")
    width, height, length = int(width), int(height), int(length)
    if width <= 0 or height <= 0 or length <= 0:
        raise ValueError("width, height and length must be positive")
    if reference_media_count < 0 or speaker_count < 0 or isolated_voice_reference_count < 0:
        raise ValueError("counts must be non-negative")

    risks: list[dict[str, Any]] = []
    unknown: list[str] = []
    pixel_frames = width * height * length

    free_mib = _finite_number(runtime.get("minimum_free_vram_mib"))
    peak_mib = _finite_number(runtime.get("peak_used_vram_mib"))
    if free_mib is None and peak_mib is None:
        unknown.append("VRAM telemetry was not supplied; V-copy/activation/VAE OOM cannot be diagnosed")
    elif free_mib is not None and free_mib < 512.0:
        risks.append(
            {
                "code": "LOW_RESIDUAL_VRAM",
                "severity": "high",
                "evidence": {"minimum_free_vram_mib": free_mib, "project_gate_mib": 512.0},
                "action": "lower canvas/frame/reference load or use an explicit offload policy",
            }
        )

    v_copy = _finite_number(runtime.get("v_copy_peak_mib"))
    if v_copy is not None and v_copy > 512.0:
        risks.append(
            {
                "code": "V_COPY_SPIKE_OBSERVED",
                "severity": "medium",
                "evidence": {"v_copy_peak_mib": v_copy},
                "action": "capture the owning allocation stack; do not infer that model weights alone caused OOM",
            }
        )
    elif v_copy is None:
        unknown.append("no v_copy_peak_mib field; the reported V-copy regression is unmeasured")

    if reference_media_count > 9:
        risks.append(
            {
                "code": "HIGH_REFERENCE_COUNT",
                "severity": "medium",
                "evidence": {"reference_media_count": reference_media_count},
                "action": "verify task-specific media limits and measure packed-token/VRAM growth",
            }
        )
    if pixel_frames > 1920 * 1088 * 124:
        risks.append(
            {
                "code": "ABOVE_REFERENCE_PIXEL_FRAME_LOAD",
                "severity": "medium",
                "evidence": {"pixel_frames": pixel_frames, "reference": 1920 * 1088 * 124},
                "action": "run one serial probe; this is a risk reference, never a hard canvas ban",
            }
        )

    if speaker_count > 1 and isolated_voice_reference_count < speaker_count:
        risks.append(
            {
                "code": "MULTISPEAKER_REFERENCE_UNDERSPECIFIED",
                "severity": "high",
                "evidence": {
                    "speaker_count": speaker_count,
                    "isolated_voice_reference_count": isolated_voice_reference_count,
                },
                "action": "bind one isolated final <Audio N> reference per speaker and review cross-talk",
            }
        )

    peak = _finite_number(audio.get("peak_abs"))
    nonfinite = int(audio.get("nonfinite_sample_count", 0) or 0)
    clipped = int(audio.get("clipped_sample_count", 0) or 0)
    if nonfinite > 0 or clipped > 0 or (peak is not None and peak >= 0.999):
        risks.append(
            {
                "code": "AUDIO_NUMERICAL_INTEGRITY_RISK",
                "severity": "high" if nonfinite else "medium",
                "evidence": {"peak_abs": peak, "nonfinite": nonfinite, "clipped": clipped},
                "action": "decode PCM and listen; numerical checks cannot judge hiss, distance shifts or voice quality",
            }
        )
    elif not audio:
        unknown.append("audio report absent; static, clipping and perceptual drift are unmeasured")

    if attention_backend == "sage_fp8":
        capability = None
        try:
            import torch

            capability = torch.cuda.get_device_capability() if torch.cuda.is_available() else None
        except Exception:
            capability = None
        if capability is None:
            unknown.append("Sage FP8 selected but CUDA capability is unavailable")
        elif capability[0] >= 12:
            risks.append(
                {
                    "code": "SAGE_FP8_SM120_REQUIRES_REAL_NOISE_CHECK",
                    "severity": "medium",
                    "evidence": {"cuda_capability": list(capability)},
                    "action": "compare one strict decoded clip against stock attention; do not trust import success alone",
                }
            )

    flash = _dark_flash_periodicity(frames)
    if flash["state"] == "suspected":
        risks.append(
            {
                "code": "PERIODIC_17_FRAME_DARK_FLASH_SUSPECTED",
                "severity": "high",
                "evidence": flash,
                "action": "inspect exact frames and latent/VAE boundaries before changing sampler math",
            }
        )
    elif flash["state"] == "unknown":
        unknown.append("17-frame dark-flash periodicity is unmeasured")

    status = "RISKS_OBSERVED" if risks else ("INSUFFICIENT_EVIDENCE" if unknown else "NO_OBSERVED_RISK")
    report = {
        "schema": OFFICIAL_DIAGNOSTIC_SCHEMA,
        "status": status,
        "workload": {
            "width": width,
            "height": height,
            "length": length,
            "reference_media_count": int(reference_media_count),
            "speaker_count": int(speaker_count),
            "isolated_voice_reference_count": int(isolated_voice_reference_count),
            "attention_backend": attention_backend,
        },
        "risks": risks,
        "unknowns": unknown,
        "dark_flash_periodicity": flash,
        "side_effects": False,
        "hard_gates": False,
        "model_fingerprint_checked": False,
        "boundary": (
            "This read-only node classifies supplied observations. It never proves universal safety, "
            "never blocks a model by name/hash/size, and never changes sampling, attention or memory policy."
        ),
    }
    return status, len(risks), json.dumps(report, ensure_ascii=False, indent=2)
