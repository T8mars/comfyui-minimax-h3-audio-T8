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
TOOL_VERSION = "1.0.0"
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
DISABLE_DYNAMIC_VRAM_FLAGS = {
    "--disable-dynamic-vram",
    "--gpu-only",
    "--highvram",
    "--novram",
    "--cpu",
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


def _direct_input(node: dict[str, Any], name: str, node_ids: set[str]) -> Any:
    value = node.get("inputs", {}).get(name)
    return None if _is_link(value, node_ids) else value


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
                "name": _direct_input(node, "unet_name", node_ids)
                or _direct_input(node, "ckpt_name", node_ids),
                "weight_dtype": _direct_input(node, "weight_dtype", node_ids),
            })

    loras = []
    for class_type, entries in by_type.items():
        if "lora" not in class_type.lower():
            continue
        for node_id, node in entries:
            loras.append({
                "node_id": node_id,
                "class_type": class_type,
                "name": _direct_input(node, "lora_name", node_ids),
                "strength_model": _direct_input(node, "strength_model", node_ids),
            })

    conditioning = []
    for class_type in ("MiniMaxH3AudioConditioningT8", "MiniMaxH3StillConditioningT8"):
        for node_id, node in by_type.get(class_type, []):
            width = _direct_input(node, "width", node_ids)
            height = _direct_input(node, "height", node_ids)
            length = _direct_input(node, "length", node_ids)
            entry = {
                "node_id": node_id,
                "class_type": class_type,
                "width": width,
                "height": height,
                "length": length,
                "task_type": _direct_input(node, "task_type", node_ids),
                "audio_mode": _direct_input(node, "audio_mode", node_ids),
                "target_mode": _direct_input(node, "target_mode", node_ids),
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
            "steps": _direct_input(node, "steps", node_ids),
            "video_steps": _direct_input(node, "video_steps", node_ids),
            "audio_steps": _direct_input(node, "audio_steps", node_ids),
            "shift_video": _direct_input(node, "shift_video", node_ids),
            "shift_audio": _direct_input(node, "shift_audio", node_ids),
            "scheduler": _direct_input(node, "scheduler", node_ids),
            "sampler_name": _direct_input(node, "sampler_name", node_ids),
        })

    seeds = []
    for node_id, node in prompt.items():
        for key in ("noise_seed", "seed"):
            value = _direct_input(node, key, node_ids)
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

    sampling_node_ids = {
        node_id
        for node_id, node in prompt.items()
        if node["class_type"] in SAMPLER_TYPES | SCHEDULER_TYPES
    }
    non_sampling_literals = []
    non_sampling_links = []
    for node_id, node in sorted(prompt.items()):
        if node_id in sampling_node_ids:
            continue
        literals = {}
        for name, value in sorted(node.get("inputs", {}).items()):
            if _is_link(value, node_ids):
                source_id = str(value[0])
                if source_id not in sampling_node_ids:
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


async def _poll_stats(
    session,
    server: str,
    device_index: int,
    interval: float,
    start_time: float,
    state: dict[str, Any],
    samples: list[dict[str, Any]],
    stop: asyncio.Event,
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
    for risk in analysis["risks"]:
        print(f"[{risk['severity'].upper()}] {risk['code']}: {risk['message']}")


def print_runtime(runtime: dict[str, Any]) -> None:
    summary = runtime.get("summary", {})
    print(f"Status: {runtime.get('status')}")
    print(f"Duration: {runtime.get('duration_seconds')} s")
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    try:
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
