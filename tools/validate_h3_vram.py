from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
import time
from typing import Any
from urllib.parse import urlparse, urlunparse
import uuid


SCHEMA_VERSION = 1
TOOL_VERSION = "1.2.0"
MIB = 1024**2
TERMINAL_EVENTS = {"execution_success", "execution_error", "execution_interrupted"}
SAMPLER_TYPES = {
    "MiniMaxH3DualClockSamplerT8",
    "MiniMaxH3MultiRateSamplerEXPT8",
    "KSampler",
    "KSamplerAdvanced",
    "SamplerCustomAdvanced",
}
SCHEDULER_TYPES = {
    "BasicScheduler",
    "KarrasScheduler",
    "ExponentialScheduler",
    "PolyexponentialScheduler",
    "SDTurboScheduler",
    "MiniMaxH3SigmaShift",
    "KSamplerSelect",
}
VRAM_POLICY_NODE_TYPE = "MiniMaxH3VRAMPolicyT8Advanced"
ACTIVATION_CHUNK_NODE_TYPE = "MiniMaxH3ActivationChunkT8Advanced"
DISABLE_DYNAMIC_VRAM_FLAGS = {
    "--disable-dynamic-vram",
    "--gpu-only",
    "--highvram",
    "--novram",
    "--cpu",
}

# These outputs are literal projections of the orchestrator's same-named inputs. Resolve only
# values whose node contract guarantees identity; computed outputs such as length, prompt and seed
# must remain links because their runtime values depend on the accepted manifest and segment plan.
LITERAL_OUTPUT_INPUTS = {
    "MiniMaxH3LongVideoOrchestratorT8": {
        16: "steps",
        17: "shift_video",
        18: "shift_audio",
        19: "sampler_name",
        20: "scheduler",
    },
}


class ValidationError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_link(value: Any, node_ids: set[str]) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and str(value[0]) in node_ids
        and isinstance(value[1], int)
    )


def _direct_input(
    node: dict[str, Any],
    name: str,
    node_ids: set[str],
    prompt: dict[str, dict[str, Any]] | None = None,
) -> Any:
    value = node.get("inputs", {}).get(name)
    if not _is_link(value, node_ids):
        return value
    if prompt is None:
        return None

    source_id, source_slot = str(value[0]), value[1]
    source = prompt[source_id]
    source_map = LITERAL_OUTPUT_INPUTS.get(source["class_type"], {})
    source_input_name = source_map.get(source_slot)
    if source_input_name is None:
        return None
    source_value = source.get("inputs", {}).get(source_input_name)
    return None if _is_link(source_value, node_ids) else source_value


def load_api_prompt(path: Path) -> dict[str, dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"Cannot read workflow JSON: {exc}") from exc

    if isinstance(payload, dict) and isinstance(payload.get("prompt"), dict):
        payload = payload["prompt"]
    if isinstance(payload, dict) and isinstance(payload.get("nodes"), list):
        raise ValidationError(
            "This is a frontend workflow. Export it with ComfyUI's 'Save (API Format)' first."
        )
    if not isinstance(payload, dict) or not payload:
        raise ValidationError("Expected a non-empty ComfyUI API prompt object.")

    prompt: dict[str, dict[str, Any]] = {}
    for node_id, node in payload.items():
        if not isinstance(node, dict) or not isinstance(node.get("class_type"), str):
            raise ValidationError(f"Node {node_id!r} is not a valid API prompt node.")
        if not isinstance(node.get("inputs", {}), dict):
            raise ValidationError(f"Node {node_id!r} has invalid inputs.")
        prompt[str(node_id)] = node
    return prompt


def analyze_prompt(prompt: dict[str, dict[str, Any]]) -> dict[str, Any]:
    node_ids = set(prompt)
    nodes = [
        {"id": node_id, "class_type": node["class_type"]}
        for node_id, node in sorted(prompt.items())
    ]
    by_type: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for node_id, node in prompt.items():
        by_type[node["class_type"]].append((node_id, node))

    unets = []
    for class_type in ("UNETLoader", "CheckpointLoaderSimple"):
        for node_id, node in by_type.get(class_type, []):
            unets.append({
                "node_id": node_id,
                "class_type": class_type,
                "name": _direct_input(node, "unet_name", node_ids, prompt)
                or _direct_input(node, "ckpt_name", node_ids, prompt),
                "weight_dtype": _direct_input(node, "weight_dtype", node_ids, prompt),
            })

    loras = []
    for class_type, entries in by_type.items():
        if "lora" not in class_type.lower():
            continue
        for node_id, node in entries:
            loras.append({
                "node_id": node_id,
                "class_type": class_type,
                "name": _direct_input(node, "lora_name", node_ids, prompt),
                "strength_model": _direct_input(node, "strength_model", node_ids, prompt),
            })

    conditioning = []
    for class_type in ("MiniMaxH3AudioConditioningT8", "MiniMaxH3StillConditioningT8"):
        for node_id, node in by_type.get(class_type, []):
            width = _direct_input(node, "width", node_ids, prompt)
            height = _direct_input(node, "height", node_ids, prompt)
            length = _direct_input(node, "length", node_ids, prompt)
            entry = {
                "node_id": node_id,
                "class_type": class_type,
                "width": width,
                "height": height,
                "length": length,
                "task_type": _direct_input(node, "task_type", node_ids, prompt),
                "audio_mode": _direct_input(node, "audio_mode", node_ids, prompt),
                "target_mode": _direct_input(node, "target_mode", node_ids, prompt),
            }
            if isinstance(width, int) and isinstance(height, int):
                entry["pixel_area"] = width * height
            conditioning.append(entry)

    sampling = []
    for node_id, node in prompt.items():
        class_type = node["class_type"]
        if class_type not in SAMPLER_TYPES and class_type not in SCHEDULER_TYPES:
            continue
        sampling.append({
            "node_id": node_id,
            "class_type": class_type,
            "steps": _direct_input(node, "steps", node_ids, prompt),
            "video_steps": _direct_input(node, "video_steps", node_ids, prompt),
            "audio_steps": _direct_input(node, "audio_steps", node_ids, prompt),
            "shift_video": _direct_input(node, "shift_video", node_ids, prompt),
            "shift_audio": _direct_input(node, "shift_audio", node_ids, prompt),
            "scheduler": _direct_input(node, "scheduler", node_ids, prompt),
            "sampler_name": _direct_input(node, "sampler_name", node_ids, prompt),
        })

    vram_policies = []
    for node_id, node in by_type.get(VRAM_POLICY_NODE_TYPE, []):
        vram_policies.append({
            "node_id": node_id,
            "class_type": VRAM_POLICY_NODE_TYPE,
            "mode": _direct_input(node, "mode", node_ids, prompt),
            "fixed_total_reserved_gib": _direct_input(
                node, "fixed_total_reserved_gib", node_ids, prompt
            ),
            "external_margin_gib": _direct_input(
                node, "external_margin_gib", node_ids, prompt
            ),
            "maximum_reserved_gib": _direct_input(
                node, "maximum_reserved_gib", node_ids, prompt
            ),
            "clean_before_load": _direct_input(
                node, "clean_before_load", node_ids, prompt
            ),
            "require_dynamic_vram": _direct_input(
                node, "require_dynamic_vram", node_ids, prompt
            ),
        })

    activation_chunks = []
    for node_id, node in by_type.get(ACTIVATION_CHUNK_NODE_TYPE, []):
        activation_chunks.append({
            "node_id": node_id,
            "class_type": ACTIVATION_CHUNK_NODE_TYPE,
            "mode": _direct_input(node, "mode", node_ids, prompt),
            "chunk_rows": _direct_input(node, "chunk_rows", node_ids, prompt),
            "block_start": _direct_input(node, "block_start", node_ids, prompt),
            "block_end": _direct_input(node, "block_end", node_ids, prompt),
            "preserve_short_path": _direct_input(
                node, "preserve_short_path", node_ids, prompt
            ),
        })

    seeds = []
    for node_id, node in prompt.items():
        for key in ("noise_seed", "seed"):
            value = _direct_input(node, key, node_ids, prompt)
            if isinstance(value, int):
                seeds.append({"node_id": node_id, "input": key, "value": value})

    risks = []
    bypass_loras = [item for item in loras if item["class_type"] == "LoraLoaderBypassModelOnly"]
    if bypass_loras:
        risks.append({
            "code": "bypass_lora_gpu_residency",
            "severity": "info",
            "message": (
                "Bypass LoRA is present. Its adapter weights and extra forward path need "
                "separate VRAM attribution from the VBAR-managed base model."
            ),
        })

    dual_nodes = [item for item in sampling if item["class_type"] == "MiniMaxH3DualClockSamplerT8"]
    for item in dual_nodes:
        if item["steps"] != 4:
            risks.append({
                "code": "dual_clock_non_turbo_step_count",
                "severity": "warning",
                "node_id": item["node_id"],
                "message": f"Stable Turbo comparison expects 4 steps, found {item['steps']!r}.",
            })

    external_schedulers = [
        item for item in sampling
        if item["class_type"] in SCHEDULER_TYPES
    ]
    if dual_nodes and external_schedulers:
        risks.append({
            "code": "mixed_sampling_setup",
            "severity": "warning",
            "message": "Dual-clock and external scheduler nodes coexist in the API prompt.",
        })

    treatment_node_ids = {
        node_id
        for node_id, node in prompt.items()
        if node["class_type"] in SAMPLER_TYPES | SCHEDULER_TYPES
        or node["class_type"] in {
            VRAM_POLICY_NODE_TYPE,
            ACTIVATION_CHUNK_NODE_TYPE,
        }
    }
    non_sampling_literals = []
    non_sampling_links = []
    for node_id, node in sorted(prompt.items()):
        if node_id in treatment_node_ids:
            continue
        literals = {}
        for name, value in sorted(node.get("inputs", {}).items()):
            if _is_link(value, node_ids):
                source_id = str(value[0])
                if source_id not in treatment_node_ids:
                    non_sampling_links.append({
                        "source_id": source_id,
                        "source_slot": value[1],
                        "target_id": node_id,
                        "target_input": name,
                    })
            else:
                literals[name] = value
        non_sampling_literals.append({
            "node_id": node_id,
            "class_type": node["class_type"],
            "inputs": literals,
        })

    controls = {
        "unets": sorted(unets, key=lambda item: (item["class_type"], str(item["name"]))),
        "loras": sorted(loras, key=lambda item: (item["class_type"], str(item["name"]))),
        "conditioning": sorted(conditioning, key=lambda item: item["node_id"]),
        "seeds": sorted(seeds, key=lambda item: (item["node_id"], item["input"])),
        "non_sampling_literals": non_sampling_literals,
        "non_sampling_links": sorted(
            non_sampling_links,
            key=lambda item: (
                item["source_id"],
                item["source_slot"],
                item["target_id"],
                item["target_input"],
            ),
        ),
    }
    treatment = {
        "sampling": sorted(sampling, key=lambda item: item["node_id"]),
        "vram_policy": sorted(vram_policies, key=lambda item: item["node_id"]),
        "activation_chunk": sorted(
            activation_chunks, key=lambda item: item["node_id"]
        ),
    }
    return {
        "node_count": len(prompt),
        "nodes": nodes,
        "controls": controls,
        "treatment": treatment,
        "risks": risks,
    }


def make_ab_prompts(
    prompt: dict[str, dict[str, Any]],
    *,
    steps: int = 4,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    if steps < 1:
        raise ValidationError("A/B step count must be at least 1.")
    matches = [
        (node_id, node)
        for node_id, node in prompt.items()
        if node["class_type"] == "MiniMaxH3DualClockSamplerT8"
    ]
    if len(matches) != 1:
        raise ValidationError(
            "A/B generation requires exactly one MiniMaxH3DualClockSamplerT8 node."
        )
    dual_id, source_dual = matches[0]
    required = {"model", "av_latent", "shift_video", "shift_audio"}
    missing = required - set(source_dual.get("inputs", {}))
    if missing:
        raise ValidationError(
            "Dual-clock node is missing required inputs: " + ", ".join(sorted(missing))
        )

    dual = copy.deepcopy(prompt)
    dual[dual_id]["inputs"]["steps"] = steps
    dual[dual_id].setdefault("_meta", {})["title"] = (
        f"VRAM A/B treatment: dual-clock Euler ({steps} steps)"
    )

    stock = copy.deepcopy(prompt)
    stock_dual = stock.pop(dual_id)
    numeric_ids = [int(node_id) for node_id in stock if node_id.isdigit()]
    next_id = max(numeric_ids, default=0) + 1
    while str(next_id) in stock:
        next_id += 1
    shift_id = str(next_id)
    sampler_id = str(next_id + 1)
    scheduler_id = str(next_id + 2)

    stock[shift_id] = {
        "class_type": "MiniMaxH3SigmaShift",
        "inputs": {
            "model": copy.deepcopy(stock_dual["inputs"]["model"]),
            "shift_video": copy.deepcopy(stock_dual["inputs"]["shift_video"]),
            "shift_audio": copy.deepcopy(stock_dual["inputs"]["shift_audio"]),
        },
        "_meta": {"title": "VRAM A/B control: native H3 sigma shift"},
    }
    stock[sampler_id] = {
        "class_type": "KSamplerSelect",
        "inputs": {"sampler_name": "euler"},
        "_meta": {"title": "VRAM A/B control: stock Euler"},
    }
    stock[scheduler_id] = {
        "class_type": "BasicScheduler",
        "inputs": {
            "model": [shift_id, 0],
            "scheduler": "simple",
            "steps": steps,
            "denoise": 1.0,
        },
        "_meta": {"title": f"VRAM A/B control: simple scheduler ({steps} steps)"},
    }

    replacements = {
        0: [shift_id, 0],
        1: [sampler_id, 0],
        2: [scheduler_id, 0],
    }
    replaced_slots: set[int] = set()
    for node in stock.values():
        for name, value in node.get("inputs", {}).items():
            if (
                isinstance(value, list)
                and len(value) == 2
                and str(value[0]) == dual_id
                and value[1] in replacements
            ):
                replaced_slots.add(value[1])
                node["inputs"][name] = copy.deepcopy(replacements[value[1]])
    if replaced_slots != {0, 1, 2}:
        raise ValidationError(
            "The dual-clock MODEL/SAMPLER/SIGMAS outputs are not all connected; "
            "cannot build a trustworthy stock control workflow."
        )
    return stock, dual


def make_vram_policy_prompts(
    prompt: dict[str, dict[str, Any]],
    *,
    mode: str = "fixed_total_reserved_exp",
    fixed_total_reserved_gib: float = 2.0,
    external_margin_gib: float = 1.0,
    maximum_reserved_gib: float = 8.0,
    clean_before_load: bool = False,
    require_dynamic_vram: bool = True,
    minimum_current_headroom_mib: float = 512.0,
    minimum_commit_headroom_gib: float = 16.0,
    block_when_commit_below_gate: bool = True,
    policy_epoch: int = 0,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Build a strict single-variable baseline/policy pair for the Advanced loader."""
    modes = {
        "report_only",
        "fixed_total_reserved_exp",
        "external_usage_plus_margin_exp",
    }
    if mode not in modes:
        raise ValidationError(f"Unsupported VRAM policy mode: {mode!r}")
    if mode == "external_usage_plus_margin_exp" and not clean_before_load:
        raise ValidationError(
            "external_usage_plus_margin_exp requires clean_before_load=true"
        )
    if any(node["class_type"] == VRAM_POLICY_NODE_TYPE for node in prompt.values()):
        raise ValidationError("The source workflow already contains a T8 VRAM Policy node.")
    loaders = [
        (node_id, node)
        for node_id, node in prompt.items()
        if node["class_type"] == "MiniMaxH3HybridModelLoaderT8Advanced"
    ]
    if len(loaders) != 1:
        raise ValidationError(
            "VRAM Policy A/B generation requires exactly one "
            "MiniMaxH3HybridModelLoaderT8Advanced node."
        )
    loader_id, source_loader = loaders[0]
    if "vram_policy" in source_loader.get("inputs", {}):
        raise ValidationError("The source Hybrid Loader already has a vram_policy input.")

    baseline = copy.deepcopy(prompt)
    treatment = copy.deepcopy(prompt)
    numeric_ids = [int(node_id) for node_id in treatment if node_id.isdigit()]
    next_id = max(numeric_ids, default=0) + 1
    while str(next_id) in treatment:
        next_id += 1
    policy_id = str(next_id)
    treatment[policy_id] = {
        "class_type": VRAM_POLICY_NODE_TYPE,
        "inputs": {
            "mode": mode,
            "fixed_total_reserved_gib": float(fixed_total_reserved_gib),
            "external_margin_gib": float(external_margin_gib),
            "maximum_reserved_gib": float(maximum_reserved_gib),
            "clean_before_load": bool(clean_before_load),
            "require_dynamic_vram": bool(require_dynamic_vram),
            "minimum_current_headroom_mib": float(minimum_current_headroom_mib),
            "minimum_commit_headroom_gib": float(minimum_commit_headroom_gib),
            "block_when_commit_below_gate": bool(block_when_commit_below_gate),
            "policy_epoch": int(policy_epoch),
        },
        "_meta": {
            "title": "VRAM Policy A/B treatment (Advanced; process-global EXP)"
        },
    }
    treatment[loader_id]["inputs"]["vram_policy"] = [policy_id, 0]

    baseline_analysis = analyze_prompt(baseline)
    treatment_analysis = analyze_prompt(treatment)
    if baseline_analysis["controls"] != treatment_analysis["controls"]:
        raise ValidationError(
            "Generated VRAM Policy pair changed non-treatment workflow controls."
        )
    if baseline_analysis["treatment"] == treatment_analysis["treatment"]:
        raise ValidationError("Generated VRAM Policy treatment was not detected.")
    return baseline, treatment


def make_activation_chunk_prompts(
    prompt: dict[str, dict[str, Any]],
    *,
    chunk_rows: int = 256,
    block_start: int = 0,
    block_end: int = 49,
    preserve_short_path: bool = True,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Build a single-variable report-only/apply pair for MLP activation chunking."""
    if chunk_rows < 16 or chunk_rows > 65536:
        raise ValidationError("Activation chunk_rows must be between 16 and 65536.")
    if block_start < 0 or block_end < block_start or block_end > 49:
        raise ValidationError("Activation block range must be within 0..49.")
    matches = [
        (node_id, node)
        for node_id, node in prompt.items()
        if node["class_type"] == ACTIVATION_CHUNK_NODE_TYPE
    ]
    if len(matches) != 1:
        raise ValidationError(
            "Activation Chunk A/B generation requires exactly one "
            f"{ACTIVATION_CHUNK_NODE_TYPE} node."
        )
    node_id, source = matches[0]
    if "model" not in source.get("inputs", {}):
        raise ValidationError("Activation Chunk node is missing its MODEL input.")
    prompt_ids = set(prompt)
    consumers = [
        (target_id, name)
        for target_id, node in prompt.items()
        for name, value in node.get("inputs", {}).items()
        if _is_link(value, prompt_ids)
        and str(value[0]) == node_id
        and value[1] == 0
    ]
    if not consumers:
        raise ValidationError(
            "Activation Chunk MODEL output is unused; cannot build a meaningful A/B pair."
        )

    baseline = copy.deepcopy(prompt)
    treatment = copy.deepcopy(prompt)
    shared = {
        "chunk_rows": int(chunk_rows),
        "block_start": int(block_start),
        "block_end": int(block_end),
        "preserve_short_path": bool(preserve_short_path),
    }
    baseline[node_id]["inputs"].update(shared)
    treatment[node_id]["inputs"].update(shared)
    baseline[node_id]["inputs"]["mode"] = "report_only"
    treatment[node_id]["inputs"]["mode"] = "apply_exp"
    baseline[node_id].setdefault("_meta", {})["title"] = (
        "Activation Chunk A/B control: report only"
    )
    treatment[node_id].setdefault("_meta", {})["title"] = (
        f"Activation Chunk A/B treatment: {chunk_rows} rows"
    )

    baseline_analysis = analyze_prompt(baseline)
    treatment_analysis = analyze_prompt(treatment)
    if baseline_analysis["controls"] != treatment_analysis["controls"]:
        raise ValidationError(
            "Generated Activation Chunk pair changed non-treatment workflow controls."
        )
    if baseline_analysis["treatment"] == treatment_analysis["treatment"]:
        raise ValidationError("Generated Activation Chunk treatment was not detected.")
    return baseline, treatment


def dynamic_vram_evidence(
    system_stats: dict[str, Any] | None,
    log_text: str | None = None,
) -> dict[str, Any]:
    system = (system_stats or {}).get("system", {})
    devices = (system_stats or {}).get("devices", [])
    argv = [str(item) for item in system.get("argv", [])]
    packages = {
        item.get("name"): item.get("installed")
        for item in system.get("comfy_package_versions", [])
        if isinstance(item, dict)
    }
    disabling_flags = sorted(set(argv) & DISABLE_DYNAMIC_VRAM_FLAGS)
    has_nvidia = any("nvidia" in str(item.get("name", "")).lower() for item in devices)
    enabled_marker = "DynamicVRAM support detected and enabled"
    fallback_markers = (
        "DynamicVRAM support unavailable",
        "Falling back to legacy ModelPatcher",
        "DynamicVRAM disabled",
    )

    status = "unknown"
    source = "none"
    if log_text:
        enabled_at = log_text.rfind(enabled_marker)
        fallback_at = max(log_text.rfind(marker) for marker in fallback_markers)
        if enabled_at >= 0 and enabled_at > fallback_at:
            status = "enabled"
            source = "log"
        elif fallback_at >= 0:
            status = "disabled_or_fallback"
            source = "log"
    if source == "none" and disabling_flags:
        status = "disabled_by_cli"
        source = "system_stats.argv"
    elif source == "none" and packages.get("comfy-aimdo") and has_nvidia:
        status = "available_not_proven"
        source = "system_stats"

    return {
        "status": status,
        "source": source,
        "comfy_aimdo_version": packages.get("comfy-aimdo"),
        "disabling_flags": disabling_flags,
        "devices": [item.get("name") for item in devices if isinstance(item, dict)],
        "note": (
            "Only an explicit startup-log marker proves that DynamicVRAM was enabled. "
            "Package presence alone proves availability, not activation."
        ),
    }


def _read_log(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _default_log_path() -> Path | None:
    candidate = Path(__file__).resolve().parents[3] / "user" / "comfyui.log"
    return candidate if candidate.is_file() else None


def _normalize_server(server: str) -> str:
    parsed = urlparse(server if "://" in server else f"http://{server}")
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValidationError(f"Invalid ComfyUI server URL: {server!r}")
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", "", ""))


def _ws_url(server: str, client_id: str) -> str:
    parsed = urlparse(server)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    path = f"{parsed.path.rstrip('/')}/ws"
    return urlunparse((scheme, parsed.netloc, path, "", f"clientId={client_id}", ""))


async def _json_request(session, method: str, url: str, **kwargs) -> dict[str, Any]:
    async with session.request(method, url, **kwargs) as response:
        text = await response.text()
        if response.status >= 400:
            raise ValidationError(f"{method} {url} failed ({response.status}): {text[:2000]}")
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValidationError(f"{method} {url} returned non-JSON content.") from exc


def _device_sample(stats: dict[str, Any], device_index: int) -> dict[str, Any]:
    devices = stats.get("devices", [])
    if not isinstance(devices, list) or not devices:
        raise ValidationError("ComfyUI /system_stats returned no devices.")
    device = next(
        (item for item in devices if item.get("index") == device_index),
        devices[0] if device_index == 0 else None,
    )
    if device is None:
        raise ValidationError(f"Device index {device_index} is not exposed by ComfyUI.")

    vram_total = int(device.get("vram_total", 0))
    vram_free = int(device.get("vram_free", 0))
    torch_total = int(device.get("torch_vram_total", 0))
    torch_free = int(device.get("torch_vram_free", 0))
    return {
        "device_name": device.get("name"),
        "vram_total_bytes": vram_total,
        "vram_free_bytes": vram_free,
        "vram_used_bytes": max(0, vram_total - vram_free),
        "torch_pool_total_bytes": torch_total,
        "torch_pool_free_bytes": torch_free,
        "torch_pool_used_bytes": max(0, torch_total - torch_free),
        "ram_free_bytes": int(stats.get("system", {}).get("ram_free", 0)),
    }


def _resolve_local_server_pid(server: str) -> int | None:
    parsed = urlparse(server)
    if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        return None
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        import psutil

        matches = {
            int(connection.pid)
            for connection in psutil.net_connections(kind="tcp")
            if connection.pid is not None
            and connection.status == psutil.CONN_LISTEN
            and connection.laddr
            and int(connection.laddr.port) == port
        }
    except Exception:
        return None
    return next(iter(matches)) if len(matches) == 1 else None


def _process_sample(pid: int | None) -> dict[str, Any]:
    if pid is None:
        return {}
    try:
        import psutil

        process = psutil.Process(pid)
        memory = process.memory_info()
        io = process.io_counters()
        return {
            "server_pid": pid,
            "process_rss_bytes": int(getattr(memory, "rss", 0)),
            "process_private_bytes": getattr(memory, "private", None),
            "process_pagefile_bytes": getattr(memory, "pagefile", None),
            "process_page_faults": getattr(memory, "num_page_faults", None),
            "process_read_bytes": getattr(io, "read_bytes", None),
            "process_write_bytes": getattr(io, "write_bytes", None),
            "process_thread_count": process.num_threads(),
        }
    except Exception as error:
        return {"process_telemetry_error": f"{type(error).__name__}: {error}"}


def _nvml_sample(device_index: int) -> dict[str, Any]:
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            throttle = int(pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle))
            thermal_mask = int(
                getattr(pynvml, "nvmlClocksThrottleReasonHwThermalSlowdown", 0)
            ) | int(getattr(pynvml, "nvmlClocksThrottleReasonSwThermalSlowdown", 0))
            return {
                "gpu_temperature_c": pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                ),
                "gpu_power_mw": pynvml.nvmlDeviceGetPowerUsage(handle),
                "gpu_sm_clock_mhz": pynvml.nvmlDeviceGetClockInfo(
                    handle, pynvml.NVML_CLOCK_SM
                ),
                "gpu_memory_clock_mhz": pynvml.nvmlDeviceGetClockInfo(
                    handle, pynvml.NVML_CLOCK_MEM
                ),
                "gpu_throttle_reasons_raw": throttle,
                "gpu_thermal_throttling": bool(throttle & thermal_mask),
            }
        finally:
            pynvml.nvmlShutdown()
    except Exception as error:
        return {"gpu_health_telemetry_error": f"{type(error).__name__}: {error}"}


def _optional_int_values(
    samples: list[dict[str, Any]], key: str
) -> list[int]:
    return [
        int(item[key])
        for item in samples
        if isinstance(item.get(key), (int, float))
    ]


def _counter_delta(samples: list[dict[str, Any]], key: str) -> int | None:
    values = _optional_int_values(samples, key)
    if len(values) < 2:
        return None
    return max(0, values[-1] - values[0])


async def _poll_stats(
    session,
    server: str,
    device_index: int,
    interval: float,
    start_time: float,
    state: dict[str, Any],
    samples: list[dict[str, Any]],
    stop: asyncio.Event,
    server_pid: int | None,
) -> None:
    while not stop.is_set():
        try:
            stats = await _json_request(session, "GET", f"{server}/system_stats")
            sample = _device_sample(stats, device_index)
            sample.update({
                "elapsed_seconds": round(time.monotonic() - start_time, 6),
                "timestamp": utc_now(),
                "phase": state.get("phase"),
                "node_id": state.get("node_id"),
                "node_type": state.get("node_type"),
                "progress_value": state.get("progress_value"),
                "progress_max": state.get("progress_max"),
            })
            sample.update(_process_sample(server_pid))
            sample.update(_nvml_sample(device_index))
            samples.append(sample)
        except Exception as exc:  # Keep the model run alive if one telemetry poll fails.
            samples.append({
                "elapsed_seconds": round(time.monotonic() - start_time, 6),
                "timestamp": utc_now(),
                "phase": state.get("phase"),
                "node_id": state.get("node_id"),
                "node_type": state.get("node_type"),
                "telemetry_error": str(exc),
            })
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
        except asyncio.TimeoutError:
            pass


def summarize_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [item for item in samples if "vram_used_bytes" in item]
    if not valid:
        return {"sample_count": 0, "error": "No valid VRAM samples were collected."}

    baseline = [item for item in valid if item.get("phase") == "baseline"]
    baseline_used = int(statistics.median(item["vram_used_bytes"] for item in baseline)) \
        if baseline else valid[0]["vram_used_bytes"]
    baseline_torch = int(statistics.median(item["torch_pool_used_bytes"] for item in baseline)) \
        if baseline else valid[0]["torch_pool_used_bytes"]
    peak_global = max(valid, key=lambda item: item["vram_used_bytes"])
    peak_torch = max(valid, key=lambda item: item["torch_pool_used_bytes"])
    read_delta = _counter_delta(valid, "process_read_bytes")
    fault_delta = _counter_delta(valid, "process_page_faults")
    elapsed = max(float(item.get("elapsed_seconds", 0.0)) for item in valid)

    grouped: dict[tuple[str | None, str | None], list[dict[str, Any]]] = defaultdict(list)
    for item in valid:
        if item.get("phase") == "running":
            grouped[(item.get("node_id"), item.get("node_type"))].append(item)
    per_node = []
    for (node_id, node_type), group in grouped.items():
        peak = max(group, key=lambda item: item["vram_used_bytes"])
        torch_peak = max(item["torch_pool_used_bytes"] for item in group)
        per_node.append({
            "node_id": node_id,
            "node_type": node_type,
            "sample_count": len(group),
            "peak_vram_used_bytes": peak["vram_used_bytes"],
            "peak_torch_pool_used_bytes": torch_peak,
            "peak_progress_value": peak.get("progress_value"),
            "peak_progress_max": peak.get("progress_max"),
        })
    per_node.sort(key=lambda item: item["peak_vram_used_bytes"], reverse=True)

    private_values = _optional_int_values(valid, "process_private_bytes")
    rss_values = _optional_int_values(valid, "process_rss_bytes")
    ram_free_values = _optional_int_values(valid, "ram_free_bytes")
    temperatures = _optional_int_values(valid, "gpu_temperature_c")
    powers = _optional_int_values(valid, "gpu_power_mw")
    sm_clocks = _optional_int_values(valid, "gpu_sm_clock_mhz")
    thermal_samples = sum(item.get("gpu_thermal_throttling") is True for item in valid)
    high_io_threshold = 64 * 1024**3
    low_ram_threshold = 8 * 1024**3
    if thermal_samples or (ram_free_values and min(ram_free_values) < low_ram_threshold):
        resource_behavior = "unsafe"
    elif read_delta is not None and read_delta >= high_io_threshold:
        resource_behavior = "fits_with_thrashing"
    elif read_delta is None or fault_delta is None:
        resource_behavior = "unknown"
    else:
        resource_behavior = "fits"

    return {
        "sample_count": len(valid),
        "telemetry_error_count": len(samples) - len(valid),
        "baseline_sample_count": len(baseline),
        "baseline_vram_used_bytes": baseline_used,
        "baseline_torch_pool_used_bytes": baseline_torch,
        "peak_vram_used_bytes": peak_global["vram_used_bytes"],
        "peak_vram_delta_from_baseline_bytes": peak_global["vram_used_bytes"] - baseline_used,
        "peak_vram_node_id": peak_global.get("node_id"),
        "peak_vram_node_type": peak_global.get("node_type"),
        "peak_vram_progress_value": peak_global.get("progress_value"),
        "peak_vram_progress_max": peak_global.get("progress_max"),
        "peak_torch_pool_used_bytes": peak_torch["torch_pool_used_bytes"],
        "peak_torch_delta_from_baseline_bytes": (
            peak_torch["torch_pool_used_bytes"] - baseline_torch
        ),
        "peak_torch_node_id": peak_torch.get("node_id"),
        "peak_torch_node_type": peak_torch.get("node_type"),
        "process_read_delta_bytes": read_delta,
        "process_write_delta_bytes": _counter_delta(valid, "process_write_bytes"),
        "process_page_fault_delta": fault_delta,
        "process_peak_rss_bytes": max(rss_values) if rss_values else None,
        "process_peak_private_bytes": max(private_values) if private_values else None,
        "minimum_ram_free_bytes": min(ram_free_values) if ram_free_values else None,
        "average_process_read_bytes_per_second": (
            None if read_delta is None or elapsed <= 0 else read_delta / elapsed
        ),
        "maximum_gpu_temperature_c": max(temperatures) if temperatures else None,
        "maximum_gpu_power_w": max(powers) / 1000.0 if powers else None,
        "minimum_gpu_sm_clock_mhz": min(sm_clocks) if sm_clocks else None,
        "maximum_gpu_sm_clock_mhz": max(sm_clocks) if sm_clocks else None,
        "thermal_throttle_sample_count": thermal_samples,
        "resource_behavior": resource_behavior,
        "resource_behavior_thresholds": {
            "high_process_read_bytes": high_io_threshold,
            "low_ram_free_bytes": low_ram_threshold,
        },
        "resource_behavior_note": "Process I/O counters include the ComfyUI process as a whole and may include cached I/O. The classification is a conservative screening heuristic, not a storage benchmark or memory-safe guarantee.",
        "per_node": per_node,
    }


def _event_record(event_type: str, data: Any, elapsed: float) -> dict[str, Any]:
    if event_type == "executed" and isinstance(data, dict):
        output = data.get("output")
        data = {
            "node": data.get("node"),
            "display_node": data.get("display_node"),
            "prompt_id": data.get("prompt_id"),
            "output_keys": sorted(output) if isinstance(output, dict) else [],
        }
    return {
        "elapsed_seconds": round(elapsed, 6),
        "timestamp": utc_now(),
        "type": event_type,
        "data": data,
    }


async def collect_run(
    prompt: dict[str, dict[str, Any]],
    *,
    server: str,
    device_index: int,
    poll_interval: float,
    baseline_seconds: float,
    timeout_seconds: float,
    preview_method: str,
) -> dict[str, Any]:
    try:
        import aiohttp
    except ImportError as exc:
        raise ValidationError(
            "aiohttp is required for run mode; use the Python environment that starts ComfyUI."
        ) from exc

    server = _normalize_server(server)
    analysis = analyze_prompt(prompt)
    node_types = {item["id"]: item["class_type"] for item in analysis["nodes"]}
    client_id = uuid.uuid4().hex
    prompt_id = str(uuid.uuid4())
    start_time = time.monotonic()
    state: dict[str, Any] = {
        "phase": "baseline",
        "node_id": None,
        "node_type": None,
        "progress_value": None,
        "progress_max": None,
    }
    samples: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    stop = asyncio.Event()
    terminal: dict[str, Any] | None = None
    server_snapshot: dict[str, Any] | None = None
    server_pid = _resolve_local_server_pid(server)

    timeout = aiohttp.ClientTimeout(total=None, connect=15, sock_connect=15, sock_read=None)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        server_snapshot = await _json_request(session, "GET", f"{server}/system_stats")
        async with session.ws_connect(_ws_url(server, client_id), heartbeat=30) as ws:
            poll_task = asyncio.create_task(_poll_stats(
                session,
                server,
                device_index,
                poll_interval,
                start_time,
                state,
                samples,
                stop,
                server_pid,
            ))
            try:
                await asyncio.sleep(baseline_seconds)
                state["phase"] = "queued"
                payload: dict[str, Any] = {
                    "prompt": prompt,
                    "client_id": client_id,
                    "prompt_id": prompt_id,
                }
                if preview_method != "server":
                    payload["extra_data"] = {"preview_method": preview_method}
                queued = await _json_request(
                    session, "POST", f"{server}/prompt", json=payload
                )
                if queued.get("prompt_id") != prompt_id:
                    raise ValidationError("ComfyUI returned an unexpected prompt id.")

                deadline = time.monotonic() + timeout_seconds
                while terminal is None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise ValidationError(
                            f"Timed out after {timeout_seconds:.1f}s waiting for prompt {prompt_id}."
                        )
                    try:
                        message = await asyncio.wait_for(ws.receive(), timeout=min(1.0, remaining))
                    except asyncio.TimeoutError:
                        continue
                    if message.type == aiohttp.WSMsgType.BINARY:
                        continue
                    if message.type in {
                        aiohttp.WSMsgType.CLOSE,
                        aiohttp.WSMsgType.CLOSED,
                        aiohttp.WSMsgType.ERROR,
                    }:
                        raise ValidationError("ComfyUI WebSocket closed before execution completed.")
                    if message.type != aiohttp.WSMsgType.TEXT:
                        continue
                    try:
                        packet = json.loads(message.data)
                    except json.JSONDecodeError:
                        continue
                    event_type = packet.get("type")
                    data = packet.get("data", {})
                    event_prompt = data.get("prompt_id") if isinstance(data, dict) else None
                    if event_prompt not in {None, prompt_id}:
                        continue
                    if event_type in {
                        "execution_start",
                        "execution_cached",
                        "executing",
                        "executed",
                        "progress_state",
                        *TERMINAL_EVENTS,
                    }:
                        events.append(_event_record(
                            event_type, data, time.monotonic() - start_time
                        ))
                    if event_type == "execution_start":
                        state["phase"] = "running"
                    elif event_type == "executing" and isinstance(data, dict):
                        node_id = data.get("node")
                        state["node_id"] = node_id
                        state["node_type"] = node_types.get(str(node_id))
                        state["progress_value"] = None
                        state["progress_max"] = None
                    elif event_type == "progress_state" and isinstance(data, dict):
                        active = data.get("nodes", {})
                        if isinstance(active, dict):
                            running = [
                                value for value in active.values()
                                if isinstance(value, dict) and value.get("state") == "running"
                            ]
                            if running:
                                progress = running[-1]
                                node_id = str(progress.get("node_id"))
                                state["node_id"] = node_id
                                state["node_type"] = node_types.get(node_id)
                                state["progress_value"] = progress.get("value")
                                state["progress_max"] = progress.get("max")
                    if event_type in TERMINAL_EVENTS and event_prompt == prompt_id:
                        terminal = {"type": event_type, "data": data}
            finally:
                state["phase"] = "finished"
                stop.set()
                await poll_task

    status_by_event = {
        "execution_success": "success",
        "execution_error": "error",
        "execution_interrupted": "interrupted",
    }
    return {
        "prompt_id": prompt_id,
        "status": status_by_event.get((terminal or {}).get("type"), "unknown"),
        "terminal_event": terminal,
        "duration_seconds": round(time.monotonic() - start_time, 6),
        "server_snapshot": server_snapshot,
        "events": events,
        "samples": samples,
        "summary": summarize_samples(samples),
    }


def build_report(
    *,
    label: str,
    workflow_path: Path,
    analysis: dict[str, Any],
    runtime: dict[str, Any] | None,
    system_stats: dict[str, Any] | None,
    log_path: Path | None,
) -> dict[str, Any]:
    evidence = dynamic_vram_evidence(system_stats, _read_log(log_path))
    return {
        "schema_version": SCHEMA_VERSION,
        "tool_version": TOOL_VERSION,
        "created_at": utc_now(),
        "label": label,
        "workflow": {
            "path": str(workflow_path.resolve()),
            "sha256": sha256_file(workflow_path),
            "analysis": analysis,
        },
        "environment": {
            "dynamic_vram": evidence,
            "system_stats": system_stats,
            "log_path": str(log_path.resolve()) if log_path else None,
        },
        "runtime": runtime,
    }


def compare_reports(
    first: dict[str, Any],
    second: dict[str, Any],
    *,
    material_bytes: int = 256 * MIB,
) -> dict[str, Any]:
    first_analysis = first.get("workflow", {}).get("analysis", {})
    second_analysis = second.get("workflow", {}).get("analysis", {})
    control_differences = []
    first_controls = first_analysis.get("controls", {})
    second_controls = second_analysis.get("controls", {})
    for key in sorted(set(first_controls) | set(second_controls)):
        if first_controls.get(key) != second_controls.get(key):
            control_differences.append({
                "field": key,
                "first": first_controls.get(key),
                "second": second_controls.get(key),
            })

    first_treatment = first_analysis.get("treatment")
    second_treatment = second_analysis.get("treatment")
    treatment_changed = first_treatment != second_treatment

    first_runtime = first.get("runtime") or {}
    second_runtime = second.get("runtime") or {}
    first_summary = first_runtime.get("summary") or {}
    second_summary = second_runtime.get("summary") or {}
    first_peak = first_summary.get("peak_vram_delta_from_baseline_bytes")
    second_peak = second_summary.get("peak_vram_delta_from_baseline_bytes")
    delta = None
    if isinstance(first_peak, int) and isinstance(second_peak, int):
        delta = second_peak - first_peak

    if control_differences:
        verdict = "not_comparable_control_inputs_changed"
    elif first_runtime.get("status") != "success" or second_runtime.get("status") != "success":
        verdict = "not_comparable_incomplete_run"
    elif delta is None:
        verdict = "not_comparable_missing_telemetry"
    elif abs(delta) < material_bytes:
        verdict = "no_material_peak_difference"
    elif delta > 0:
        verdict = "second_run_has_higher_peak"
    else:
        verdict = "second_run_has_lower_peak"

    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "first_label": first.get("label"),
        "second_label": second.get("label"),
        "comparable": not control_differences,
        "control_differences": control_differences,
        "treatment_changed": treatment_changed,
        "first_treatment": first_treatment,
        "second_treatment": second_treatment,
        "first_status": first_runtime.get("status"),
        "second_status": second_runtime.get("status"),
        "first_peak_delta_bytes": first_peak,
        "second_peak_delta_bytes": second_peak,
        "second_minus_first_peak_bytes": delta,
        "material_threshold_bytes": material_bytes,
        "verdict": verdict,
    }


def _safe_label(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-.")
    return cleaned or "h3-vram"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _mib(value: Any) -> str:
    return "n/a" if not isinstance(value, (int, float)) else f"{value / MIB:.1f} MiB"


def print_analysis(analysis: dict[str, Any]) -> None:
    print(f"Nodes: {analysis['node_count']}")
    controls = analysis["controls"]
    for item in controls["unets"]:
        print(f"UNet: {item['class_type']} / {item['name']}")
    for item in controls["loras"]:
        print(
            f"LoRA: {item['class_type']} / {item['name']} / strength={item['strength_model']}"
        )
    for item in controls["conditioning"]:
        area = item.get("pixel_area")
        area_text = f" / {area / 1_000_000:.3f} MP" if isinstance(area, int) else ""
        print(
            f"Conditioning: {item['class_type']} / {item['width']}x{item['height']} "
            f"/ frames={item['length']}{area_text}"
        )
    for item in analysis["treatment"]["sampling"]:
        print(
            f"Sampling: {item['class_type']} / steps={item['steps']} "
            f"video_steps={item['video_steps']} audio_steps={item['audio_steps']}"
        )
    for item in analysis["treatment"].get("vram_policy", []):
        print(
            f"VRAM Policy: mode={item['mode']} / "
            f"fixed={item['fixed_total_reserved_gib']} GiB / "
            f"clean_before_load={item['clean_before_load']}"
        )
    for risk in analysis["risks"]:
        print(f"[{risk['severity'].upper()}] {risk['code']}: {risk['message']}")


def print_runtime(runtime: dict[str, Any]) -> None:
    summary = runtime.get("summary", {})
    print(f"Status: {runtime.get('status')}")
    print(f"Duration: {runtime.get('duration_seconds')} s")
    print(f"Resource behavior: {summary.get('resource_behavior')}")
    print(f"Baseline VRAM used: {_mib(summary.get('baseline_vram_used_bytes'))}")
    print(f"Peak VRAM used: {_mib(summary.get('peak_vram_used_bytes'))}")
    print(
        "Peak delta from baseline: "
        f"{_mib(summary.get('peak_vram_delta_from_baseline_bytes'))}"
    )
    print(
        "Peak location: "
        f"node={summary.get('peak_vram_node_id')} "
        f"type={summary.get('peak_vram_node_type')} "
        f"progress={summary.get('peak_vram_progress_value')}/"
        f"{summary.get('peak_vram_progress_max')}"
    )
    print(
        "Process I/O: "
        f"read={_mib(summary.get('process_read_delta_bytes'))} "
        f"faults={summary.get('process_page_fault_delta')} "
        f"peak_private={_mib(summary.get('process_peak_private_bytes'))}"
    )
    print(
        "GPU health: "
        f"max_temp={summary.get('maximum_gpu_temperature_c')}C "
        f"thermal_samples={summary.get('thermal_throttle_sample_count')}"
    )
    terminal = runtime.get("terminal_event") or {}
    if terminal.get("type") == "execution_error":
        data = terminal.get("data", {})
        print(f"OOM/error node: {data.get('node_id')} ({data.get('node_type')})")
        print(f"Exception: {data.get('exception_type')}: {data.get('exception_message')}")


async def inspect_server(server: str) -> dict[str, Any]:
    try:
        import aiohttp
    except ImportError as exc:
        raise ValidationError(
            "aiohttp is required for server inspection; use the ComfyUI Python environment."
        ) from exc
    server = _normalize_server(server)
    timeout = aiohttp.ClientTimeout(total=15)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        return await _json_request(session, "GET", f"{server}/system_stats")


def _report_output_path(output_dir: Path, label: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return output_dir / f"{stamp}-{_safe_label(label)}.json"


def _resolve_log(value: str | None) -> Path | None:
    if value is None or value.lower() == "none":
        return None
    if value.lower() == "auto":
        return _default_log_path()
    return Path(value)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect and measure MiniMax H3 ComfyUI workflows without changing sampler math."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser(
        "inspect", help="Statically inspect an API workflow and optionally inspect a live server."
    )
    inspect_parser.add_argument("workflow", type=Path)
    inspect_parser.add_argument("--server")
    inspect_parser.add_argument("--log", default="auto")
    inspect_parser.add_argument("--output", type=Path)
    inspect_parser.add_argument("--label")

    run_parser = subparsers.add_parser(
        "run", help="Queue one API workflow and collect node-aware VRAM telemetry."
    )
    run_parser.add_argument("workflow", type=Path)
    run_parser.add_argument("--server", default="http://127.0.0.1:8188")
    run_parser.add_argument("--label")
    run_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts" / "vram-validation",
    )
    run_parser.add_argument("--poll-interval", type=float, default=0.25)
    run_parser.add_argument("--baseline-seconds", type=float, default=2.0)
    run_parser.add_argument("--timeout", type=float, default=3600.0)
    run_parser.add_argument("--device-index", type=int, default=0)
    run_parser.add_argument(
        "--preview-method",
        choices=["server", "none", "auto", "latent2rgb", "taesd"],
        default="none",
    )
    run_parser.add_argument("--log", default="auto")

    compare_parser = subparsers.add_parser(
        "compare", help="Compare two probe reports and enforce controlled-input equality."
    )
    compare_parser.add_argument("first", type=Path)
    compare_parser.add_argument("second", type=Path)
    compare_parser.add_argument("--material-mib", type=float, default=256.0)
    compare_parser.add_argument("--output", type=Path)

    pair_parser = subparsers.add_parser(
        "make-pair",
        help=(
            "Create controlled stock-Euler and dual-clock API prompts from one dual-clock prompt."
        ),
    )
    pair_parser.add_argument("workflow", type=Path)
    pair_parser.add_argument("--steps", type=int, default=4)
    pair_parser.add_argument("--output-dir", type=Path, required=True)
    pair_parser.add_argument("--prefix")

    policy_pair_parser = subparsers.add_parser(
        "make-policy-pair",
        help=(
            "Create a controlled baseline and Advanced VRAM Policy API pair from one "
            "Hybrid Advanced prompt."
        ),
    )
    policy_pair_parser.add_argument("workflow", type=Path)
    policy_pair_parser.add_argument("--output-dir", type=Path, required=True)
    policy_pair_parser.add_argument("--prefix")
    policy_pair_parser.add_argument(
        "--mode",
        choices=[
            "report_only",
            "fixed_total_reserved_exp",
            "external_usage_plus_margin_exp",
        ],
        default="fixed_total_reserved_exp",
    )
    policy_pair_parser.add_argument("--fixed-total-reserved-gib", type=float, default=2.0)
    policy_pair_parser.add_argument("--external-margin-gib", type=float, default=1.0)
    policy_pair_parser.add_argument("--maximum-reserved-gib", type=float, default=8.0)
    policy_pair_parser.add_argument(
        "--clean-before-load",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    policy_pair_parser.add_argument(
        "--require-dynamic-vram",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    policy_pair_parser.add_argument(
        "--minimum-current-headroom-mib", type=float, default=512.0
    )
    policy_pair_parser.add_argument(
        "--minimum-commit-headroom-gib", type=float, default=16.0
    )
    policy_pair_parser.add_argument(
        "--block-when-commit-below-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    policy_pair_parser.add_argument("--policy-epoch", type=int, default=0)

    activation_pair_parser = subparsers.add_parser(
        "make-activation-pair",
        help=(
            "Create controlled report-only and apply-exp prompts from one "
            "Activation Chunk workflow."
        ),
    )
    activation_pair_parser.add_argument("workflow", type=Path)
    activation_pair_parser.add_argument("--output-dir", type=Path, required=True)
    activation_pair_parser.add_argument("--prefix")
    activation_pair_parser.add_argument("--chunk-rows", type=int, default=256)
    activation_pair_parser.add_argument("--block-start", type=int, default=0)
    activation_pair_parser.add_argument("--block-end", type=int, default=49)
    activation_pair_parser.add_argument(
        "--preserve-short-path",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    try:
        if args.command == "make-activation-pair":
            prompt = load_api_prompt(args.workflow)
            baseline, treatment = make_activation_chunk_prompts(
                prompt,
                chunk_rows=args.chunk_rows,
                block_start=args.block_start,
                block_end=args.block_end,
                preserve_short_path=args.preserve_short_path,
            )
            prefix = _safe_label(args.prefix or args.workflow.stem)
            baseline_path = args.output_dir / f"{prefix}-activation-baseline.json"
            treatment_path = args.output_dir / f"{prefix}-activation-chunk.json"
            write_json(baseline_path, baseline)
            write_json(treatment_path, treatment)
            print(f"Activation baseline: {baseline_path.resolve()}")
            print(f"Activation treatment: {treatment_path.resolve()}")
            print(
                "Both prompts retain identical model, conditioning, sampling, seed and "
                "outputs; only Activation Chunk mode changes."
            )
            return 0

        if args.command == "make-pair":
            prompt = load_api_prompt(args.workflow)
            stock, dual = make_ab_prompts(prompt, steps=args.steps)
            prefix = _safe_label(args.prefix or args.workflow.stem)
            stock_path = args.output_dir / f"{prefix}-stock-euler-{args.steps}step.json"
            dual_path = args.output_dir / f"{prefix}-dual-clock-{args.steps}step.json"
            write_json(stock_path, stock)
            write_json(dual_path, dual)
            print(f"Stock control: {stock_path.resolve()}")
            print(f"Dual treatment: {dual_path.resolve()}")
            print(
                "Both prompts retain the same model, LoRA, conditioning, seed, latent and outputs."
            )
            return 0

        if args.command == "make-policy-pair":
            prompt = load_api_prompt(args.workflow)
            baseline, policy = make_vram_policy_prompts(
                prompt,
                mode=args.mode,
                fixed_total_reserved_gib=args.fixed_total_reserved_gib,
                external_margin_gib=args.external_margin_gib,
                maximum_reserved_gib=args.maximum_reserved_gib,
                clean_before_load=args.clean_before_load,
                require_dynamic_vram=args.require_dynamic_vram,
                minimum_current_headroom_mib=args.minimum_current_headroom_mib,
                minimum_commit_headroom_gib=args.minimum_commit_headroom_gib,
                block_when_commit_below_gate=args.block_when_commit_below_gate,
                policy_epoch=args.policy_epoch,
            )
            prefix = _safe_label(args.prefix or args.workflow.stem)
            baseline_path = args.output_dir / f"{prefix}-baseline.json"
            policy_path = args.output_dir / f"{prefix}-{_safe_label(args.mode)}.json"
            write_json(baseline_path, baseline)
            write_json(policy_path, policy)
            print(f"Baseline: {baseline_path.resolve()}")
            print(f"Policy treatment: {policy_path.resolve()}")
            print(
                "Both prompts retain identical model, conditioning, sampling, seed, latent "
                "and output controls; only the typed Loader policy dependency differs."
            )
            return 0

        if args.command == "compare":
            first = json.loads(args.first.read_text(encoding="utf-8"))
            second = json.loads(args.second.read_text(encoding="utf-8"))
            comparison = compare_reports(
                first,
                second,
                material_bytes=math.ceil(args.material_mib * MIB),
            )
            print(f"Verdict: {comparison['verdict']}")
            print(f"Controlled inputs equal: {comparison['comparable']}")
            print(
                "Second minus first peak: "
                f"{_mib(comparison['second_minus_first_peak_bytes'])}"
            )
            if comparison["control_differences"]:
                print("Changed controls: " + ", ".join(
                    item["field"] for item in comparison["control_differences"]
                ))
            if args.output:
                write_json(args.output, comparison)
                print(f"Report: {args.output.resolve()}")
            return 0 if comparison["comparable"] else 2

        prompt = load_api_prompt(args.workflow)
        analysis = analyze_prompt(prompt)
        print_analysis(analysis)
        log_path = _resolve_log(args.log)
        label = args.label or args.workflow.stem

        if args.command == "inspect":
            stats = asyncio.run(inspect_server(args.server)) if args.server else None
            report = build_report(
                label=label,
                workflow_path=args.workflow,
                analysis=analysis,
                runtime=None,
                system_stats=stats,
                log_path=log_path,
            )
            evidence = report["environment"]["dynamic_vram"]
            print(
                f"DynamicVRAM: {evidence['status']} "
                f"(source={evidence['source']}, aimdo={evidence['comfy_aimdo_version']})"
            )
            if args.output:
                write_json(args.output, report)
                print(f"Report: {args.output.resolve()}")
            return 0

        if args.poll_interval < 0.05:
            raise ValidationError("--poll-interval must be at least 0.05 seconds.")
        if args.baseline_seconds < 0:
            raise ValidationError("--baseline-seconds cannot be negative.")
        if args.timeout <= 0:
            raise ValidationError("--timeout must be positive.")
        runtime = asyncio.run(collect_run(
            prompt,
            server=args.server,
            device_index=args.device_index,
            poll_interval=args.poll_interval,
            baseline_seconds=args.baseline_seconds,
            timeout_seconds=args.timeout,
            preview_method=args.preview_method,
        ))
        print_runtime(runtime)
        report = build_report(
            label=label,
            workflow_path=args.workflow,
            analysis=analysis,
            runtime=runtime,
            system_stats=runtime.get("server_snapshot"),
            log_path=log_path,
        )
        output = _report_output_path(args.output_dir, label)
        write_json(output, report)
        print(f"Report: {output.resolve()}")
        return 0 if runtime.get("status") == "success" else 1
    except (ValidationError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
