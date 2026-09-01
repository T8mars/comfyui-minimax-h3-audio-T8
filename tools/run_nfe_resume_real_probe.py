#!/usr/bin/env python3
"""Run one low-load, cross-process MiniMax H3 NFE resume proof.

The default invocation is preflight-only.  A real GPU run requires ``--confirm-run`` and is
refused while the selected GPU does not have the configured free-VRAM headroom.  The tool owns
only the isolated ComfyUI processes that it starts; it never interrupts, unloads, or stops the
user's normal ComfyUI service.

The real proof is deliberately small (256x256, 22 frames, four dual-clock Euler NFEs):

1. complete a control render and save its native AV latent;
2. start the same render, interrupt after completed NFE 2, and verify the atomic NFE checkpoint;
3. stop that ComfyUI process, start a new process, resume the remaining two NFEs;
4. compare control/resume native tensors exactly and strictly decode/hash both media streams.

This is a correctness probe, not a quality benchmark, load test, crash-fuzz test, or proof that
larger resolutions fit a particular GPU.
"""

from __future__ import annotations

import argparse
import asyncio
from contextlib import suppress
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from typing import Any, Mapping
import uuid


SCHEMA = "t8.minimax_h3.nfe_resume_real_probe.v1"
NFE_METADATA_KEY = "t8_minimax_h3_nfe_resume_json"
NATIVE_METADATA_KEY = "t8_native_latent_checkpoint_json"
TERMINAL_EVENTS = {"execution_success", "execution_error", "execution_interrupted"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest().upper()


def gpu_memory_mib() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        values = [int(float(item.strip())) for item in completed.stdout.splitlines()[0].split(",")]
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return {"available": False}
    return {
        "available": True,
        "total_mib": values[0],
        "used_mib": values[1],
        "free_mib": values[2],
        "utilization_percent": values[3],
        "temperature_c": values[4],
    }


def port_is_listening(host: str, port: int, timeout: float = 0.25) -> bool:
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except OSError:
        return False


def _file_contract(
    path: Path,
    *,
    include_sha_sidecar: bool = False,
    verified_sha256: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {"name": path.name, "bytes": path.stat().st_size}
    if verified_sha256 is not None:
        token = str(verified_sha256).strip().upper()
        if len(token) != 64 or any(char not in "0123456789ABCDEF" for char in token):
            raise ValueError(f"invalid verified SHA-256 for {path.name}")
        result["sha256"] = token
        return result
    if include_sha_sidecar:
        candidates = (
            path.with_suffix(path.suffix + ".sha256"),
            path.with_suffix(".sha256"),
            path.parent / f"{path.name}.sha256.txt",
        )
        for candidate in candidates:
            if not candidate.is_file():
                continue
            token = candidate.read_text(encoding="utf-8", errors="replace").strip().split()[0]
            if len(token) == 64 and all(char in "0123456789abcdefABCDEF" for char in token):
                result["sha256"] = token.upper()
                break
    return result


def model_contract_json(
    paths: Mapping[str, Path],
    *,
    verified_hashes: Mapping[str, str] | None = None,
) -> str:
    """Create the explicit model contract used by checkpoint and resume phases."""

    if verified_hashes is not None and set(verified_hashes) != set(paths):
        raise ValueError("verified model hashes must cover every declared model role exactly")

    def contract(role: str, *, allow_sidecar: bool = False) -> dict[str, Any]:
        verified = verified_hashes.get(role) if verified_hashes is not None else None
        return _file_contract(
            paths[role],
            include_sha_sidecar=allow_sidecar and verified is None,
            verified_sha256=verified,
        )

    payload = {
        "schema": "t8.h3.nfe_probe.model_contract.v1",
        "base": contract("base", allow_sidecar=True),
        "lora": {**contract("lora"), "strength": 1.0},
        "clip": contract("clip"),
        "projection": contract("projection"),
        "video_vae": contract("video_vae"),
        "audio_vae": contract("audio_vae"),
        "weight_hash_scope": (
            "full_sha256_all_declared_files" if verified_hashes is not None else "sidecar_or_identity_only"
        ),
        "attention": "stock",
        "sampler": "dual_clock_euler/native_flow",
        "steps": 4,
        "shift_video": 12.0,
        "shift_audio": 3.0,
    }
    encoded = _json(payload)
    if len(encoded.encode("utf-8")) > 4096:
        raise ValueError("model contract unexpectedly exceeds 4096 bytes")
    return encoded


def hash_model_files(paths: Mapping[str, Path]) -> dict[str, str]:
    """Hash each declared weight exactly; called only after explicit real-run consent."""

    return {role: _sha256_file(path) for role, path in sorted(paths.items())}


def build_prompt(
    *,
    mode: str,
    run_id: str,
    checkpoint_path: str,
    model_contract_id: str,
) -> dict[str, Any]:
    if mode not in {"disabled", "checkpoint_each_step", "resume"}:
        raise ValueError(f"unsupported probe mode: {mode}")
    prompt: dict[str, Any] = {
        "1": {
            "inputs": {"vae_name": "minimax_h3_video_vae_fp16.safetensors"},
            "class_type": "VAELoader",
        },
        "2": {
            "inputs": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"},
            "class_type": "VAELoader",
        },
        "3": {
            "inputs": {
                "clip_name": "qwen3vl_8b_fp8_scaled.safetensors",
                "type": "boogu",
                "device": "default",
            },
            "class_type": "CLIPLoader",
        },
        "4": {
            "inputs": {"clip": ["3", 0], "projection": "mmh3-8b-ClipProj-v3.1.safetensors"},
            "class_type": "ClipProjApply",
        },
        "5": {
            "inputs": {
                "clip": ["4", 0],
                "encoder_family": "8B",
                "encoder_architecture": "qwen3_vl",
                "encoder_quantization": "fp8",
                "load_mode": "stock_pageable",
                "projection_path": "mmh3-8b-ClipProj-v3.1.safetensors",
                "has_reference_images": False,
                "has_reference_videos": False,
                "enforcement": "block_hard_conflicts",
            },
            "class_type": "MiniMaxH3ClipProjCompatibilityAuditT8Advanced",
        },
        "6": {
            "inputs": {
                "unet_name": "minimax_h3_fl2va_int8_convrot.safetensors",
                "weight_dtype": "default",
            },
            "class_type": "UNETLoader",
        },
        "7": {
            "inputs": {
                "lora_name": "minimax_h3_turbo_4步加速ema_comfyui.safetensors",
                "strength_model": 1.0,
                "model": ["6", 0],
            },
            "class_type": "LoraLoaderModelOnly",
        },
        "8": {
            "inputs": {
                "prompt": (
                    "A cinematic close-up of rainfall hitting a metal roof at night. Clear "
                    "synchronized rain impacts and natural ambience, no music."
                ),
                "width": 256,
                "height": 256,
                "length": 22,
                "task_type": "T2VA",
                "audio_mode": "native",
                "audio_denoise_strength": 1.0,
                "add_source_as_reference": False,
                "prompt_primary_audio_ordinal": 0,
                "strict_prompt_tags": True,
                "ref_image_size": "match",
                "reference_video_policy": "official_2_to_15s",
                "clip": ["5", 0],
                "video_vae": ["1", 0],
                "audio_vae": ["2", 0],
            },
            "class_type": "MiniMaxH3AudioConditioningT8",
        },
        "9": {
            "inputs": {
                "steps": 4,
                "shift_video": 12.0,
                "shift_audio": 3.0,
                "mode": mode,
                "checkpoint_path": checkpoint_path,
                "model_contract_id": model_contract_id if mode != "disabled" else "",
                "run_contract_json": ["16", 0] if mode != "disabled" else "{}",
                "confirm_checkpoint_write": mode == "checkpoint_each_step",
                "allow_replace_existing": False,
                "hash_chunk_megabytes": 4,
                "model": ["7", 0],
                "av_latent": ["8", 1],
            },
            "class_type": "MiniMaxH3NFEResumeSamplerT8Advanced",
        },
        "10": {"inputs": {"noise_seed": 2608228001}, "class_type": "RandomNoise"},
        "11": {
            "inputs": {"model": ["9", 0], "conditioning": ["8", 0]},
            "class_type": "BasicGuider",
        },
        "12": {
            "inputs": {
                "noise": ["10", 0],
                "guider": ["11", 0],
                "sampler": ["9", 1],
                "sigmas": ["9", 2],
                "latent_image": ["8", 1],
            },
            "class_type": "SamplerCustomAdvanced",
        },
        "13": {
            "inputs": {
                "av_latent": ["12", 0],
                "video_vae": ["1", 0],
                "audio_vae": ["2", 0],
            },
            "class_type": "MiniMaxH3AVDecodeT8",
        },
        "14": {
            "inputs": {
                "frame_rate": 24,
                "loop_count": 0,
                "filename_prefix": f"MiniMaxH3_NFE_Resume_Probe/{run_id}_{mode}",
                "format": "video/h264-mp4",
                "pix_fmt": "yuv420p",
                "crf": 19,
                "save_metadata": False,
                "trim_to_audio": False,
                "pingpong": False,
                "save_output": True,
                "images": ["13", 0],
                "audio": ["13", 1],
            },
            "class_type": "VHS_VideoCombine",
        },
        "15": {
            "inputs": {
                "av_latent": ["12", 0],
                "filename_prefix": f"{run_id}/{mode}_final",
                "checkpoint_id": f"{run_id}:{mode}:final",
                "confirm_save": mode in {"disabled", "resume"},
                "verify_after_write": True,
                "hash_chunk_megabytes": 4,
            },
            "class_type": "MiniMaxH3NativeLatentCheckpointSaveT8Advanced",
        },
        "16": {
            "inputs": {
                "positive": ["8", 0],
                "conditioned_prompt": ["8", 3],
                "media_map_json": ["8", 4],
                "conditioning_report": ["8", 5],
                "hash_chunk_megabytes": 4,
            },
            "class_type": "MiniMaxH3NFERunContractT8Advanced",
        },
    }
    return prompt


def _model_paths(comfy_root: Path) -> dict[str, Path]:
    models = comfy_root / "models"
    return {
        "base": models / "diffusion_models" / "minimax_h3_fl2va_int8_convrot.safetensors",
        "lora": models / "loras" / "minimax_h3_turbo_4步加速ema_comfyui.safetensors",
        "clip": models / "text_encoders" / "qwen3vl_8b_fp8_scaled.safetensors",
        "projection": models / "clip_projections" / "mmh3-8b-ClipProj-v3.1.safetensors",
        "video_vae": models / "vae" / "minimax_h3_video_vae_fp16.safetensors",
        "audio_vae": models / "vae" / "minimax_h3_audio_vae_fp32.safetensors",
    }


def isolated_start_gate(args: argparse.Namespace) -> dict[str, Any]:
    gpu = gpu_memory_mib()
    target_port_busy = port_is_listening(args.host, args.port)
    checks = {
        "target_port_free": not target_port_busy,
        "gpu_query_available": bool(gpu.get("available")),
        "free_vram_gate": bool(
            gpu.get("available")
            and int(gpu.get("free_mib", 0)) >= args.min_free_vram_mib
        ),
    }
    return {
        "ready": all(checks.values()),
        "checks": checks,
        "gpu": gpu,
        "target": {
            "host": args.host,
            "port": args.port,
            "already_listening": target_port_busy,
        },
        "minimum_free_vram_mib": args.min_free_vram_mib,
    }


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    comfy_root = args.comfy_root.resolve()
    paths = _model_paths(comfy_root)
    required = {
        "comfy_main": comfy_root / "main.py",
        "python": args.python.resolve(),
        "t8_nodes": comfy_root / "custom_nodes" / "minimax-h3-audio-T8",
        "clipproj_nodes": comfy_root / "custom_nodes" / "ComfyUI-ClipProj",
        "vhs_nodes": comfy_root / "custom_nodes" / "ComfyUI-VideoHelperSuite",
        **paths,
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    ffmpeg = shutil.which(args.ffmpeg) or (args.ffmpeg if Path(args.ffmpeg).is_file() else None)
    ffprobe = shutil.which(args.ffprobe) or (args.ffprobe if Path(args.ffprobe).is_file() else None)
    start_gate = isolated_start_gate(args)
    gpu = start_gate["gpu"]
    target_port_busy = start_gate["target"]["already_listening"]
    user_service_observed = port_is_listening(args.host, 8188)
    checks = {
        "required_paths_present": not missing,
        "ffmpeg_present": bool(ffmpeg),
        "ffprobe_present": bool(ffprobe),
        **start_gate["checks"],
    }
    ready = all(checks.values())
    if ready:
        status = "READY"
    elif not checks["required_paths_present"] or not checks["ffmpeg_present"] or not checks["ffprobe_present"]:
        status = "ABSTAIN_MISSING_DEPENDENCY"
    elif not checks["target_port_free"]:
        status = "ABSTAIN_TARGET_PORT_BUSY"
    elif not checks["gpu_query_available"]:
        status = "ABSTAIN_GPU_STATE_UNKNOWN"
    else:
        status = "ABSTAIN_RESOURCE_BUSY"
    return {
        "schema": f"{SCHEMA}.preflight",
        "created_at": _utc_now(),
        "ready_for_real_run": ready,
        "status": status,
        "checks": checks,
        "missing_paths": missing,
        "gpu": gpu,
        "minimum_free_vram_mib": args.min_free_vram_mib,
        "target": {"host": args.host, "port": args.port, "already_listening": target_port_busy},
        "user_service_8188_observed_only": user_service_observed,
        "ffmpeg": ffmpeg,
        "ffprobe": ffprobe,
        "boundary": (
            "The 8188 service is observation-only. This tool never interrupts, unloads, or "
            "terminates it. A failed resource gate starts no isolated ComfyUI process."
        ),
    }


def _server_command(args: argparse.Namespace, run_root: Path) -> list[str]:
    whitelist = [
        "minimax-h3-audio-T8",
        "ComfyUI-ClipProj",
        "ComfyUI-VideoHelperSuite",
        *list(getattr(args, "extra_whitelist_custom_nodes", ())),
    ]
    command = [
        str(args.python.resolve()),
        "main.py",
        "--listen",
        args.host,
        "--port",
        str(args.port),
        "--disable-auto-launch",
        "--preview-method",
        "none",
        "--cache-none",
        "--reserve-vram",
        str(getattr(args, "reserve_vram_gib", 1.0)),
        "--disable-all-custom-nodes",
        "--whitelist-custom-nodes",
        *whitelist,
        "--input-directory",
        str((args.comfy_root / "input").resolve()),
        "--output-directory",
        str((run_root / "output").resolve()),
        "--temp-directory",
        str((run_root / "temp").resolve()),
        "--user-directory",
        str((run_root / "user").resolve()),
        "--database-url",
        "sqlite:///:memory:",
    ]
    if getattr(args, "lowvram", False):
        command.append("--lowvram")
    return command


class IsolatedServer:
    def __init__(self, args: argparse.Namespace, run_root: Path, label: str):
        self.args = args
        self.run_root = run_root
        self.label = label
        self.process: subprocess.Popen[str] | None = None
        self.stdout_handle = None
        self.stderr_handle = None

    def start(self) -> int:
        if port_is_listening(self.args.host, self.args.port):
            raise RuntimeError(f"refusing to start: target port {self.args.port} is already in use")
        for name in ("output", "temp", "user", "logs"):
            (self.run_root / name).mkdir(parents=True, exist_ok=True)
        self.stdout_handle = (self.run_root / "logs" / f"{self.label}.stdout.log").open(
            "w", encoding="utf-8"
        )
        self.stderr_handle = (self.run_root / "logs" / f"{self.label}.stderr.log").open(
            "w", encoding="utf-8"
        )
        creationflags = 0
        if os.name == "nt":
            creationflags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
        self.process = subprocess.Popen(
            _server_command(self.args, self.run_root),
            cwd=self.args.comfy_root,
            stdout=self.stdout_handle,
            stderr=self.stderr_handle,
            text=True,
            creationflags=creationflags,
        )
        deadline = time.monotonic() + self.args.server_start_timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"isolated ComfyUI {self.label} exited with {self.process.returncode}; "
                    f"inspect {self.run_root / 'logs'}"
                )
            if port_is_listening(self.args.host, self.args.port):
                return int(self.process.pid)
            time.sleep(0.5)
        raise TimeoutError(f"isolated ComfyUI {self.label} did not listen in time")

    def stop(self) -> None:
        process = self.process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=15)
        deadline = time.monotonic() + 30
        while port_is_listening(self.args.host, self.args.port) and time.monotonic() < deadline:
            time.sleep(0.25)
        for handle in (self.stdout_handle, self.stderr_handle):
            if handle is not None:
                handle.close()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, _exc_type, _exc, _tb):
        self.stop()


async def _json_request(session, method: str, url: str, **kwargs) -> dict[str, Any]:
    async with session.request(method, url, **kwargs) as response:
        text = await response.text()
        with suppress(json.JSONDecodeError):
            payload = json.loads(text) if text else {}
            if response.status >= 400:
                raise RuntimeError(f"{method} {url} returned HTTP {response.status}: {payload}")
            if isinstance(payload, dict):
                return payload
        raise RuntimeError(f"{method} {url} returned invalid object JSON: {text[:500]}")


async def submit_prompt(
    *,
    server: str,
    prompt: Mapping[str, Any],
    timeout_seconds: float,
    interrupt_node: str | None = None,
    interrupt_after_step: int = 2,
) -> dict[str, Any]:
    try:
        import aiohttp
    except ImportError as error:
        raise RuntimeError("aiohttp is required in the ComfyUI Python environment") from error

    client_id = uuid.uuid4().hex
    requested_prompt_id = str(uuid.uuid4())
    terminal = None
    interrupt_response = None
    progress_at_interrupt = None
    events: list[dict[str, Any]] = []
    started = time.monotonic()
    timeout = aiohttp.ClientTimeout(total=None, connect=20, sock_connect=20, sock_read=None)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        await _json_request(session, "GET", f"{server}/system_stats")
        ws_url = server.replace("http://", "ws://").replace("https://", "wss://")
        async with session.ws_connect(f"{ws_url}/ws?clientId={client_id}", heartbeat=30) as ws:
            submitted = await _json_request(
                session,
                "POST",
                f"{server}/prompt",
                json={
                    "prompt": dict(prompt),
                    "client_id": client_id,
                    "prompt_id": requested_prompt_id,
                },
            )
            prompt_id = str(submitted.get("prompt_id") or "")
            if prompt_id != requested_prompt_id:
                raise RuntimeError(f"ComfyUI returned unexpected prompt_id {prompt_id!r}")
            deadline = time.monotonic() + timeout_seconds
            while terminal is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"timed out waiting for prompt {prompt_id}")
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
                    raise RuntimeError("ComfyUI WebSocket closed before a terminal event")
                if message.type != aiohttp.WSMsgType.TEXT:
                    continue
                with suppress(json.JSONDecodeError):
                    packet = json.loads(message.data)
                    event_type = packet.get("type")
                    data = packet.get("data") if isinstance(packet.get("data"), dict) else {}
                    if data.get("prompt_id") not in {None, prompt_id}:
                        continue
                    if event_type in {"executing", "executed", "progress", "progress_state", *TERMINAL_EVENTS}:
                        events.append(
                            {
                                "elapsed_seconds": round(time.monotonic() - started, 4),
                                "type": event_type,
                                "node": data.get("node"),
                            }
                        )
                    if interrupt_node and interrupt_response is None:
                        value = None
                        maximum = None
                        if event_type == "progress" and str(data.get("node")) == interrupt_node:
                            value = int(data.get("value") or 0)
                            maximum = int(data.get("max") or 0)
                        elif event_type == "progress_state":
                            nodes = data.get("nodes")
                            progress = nodes.get(interrupt_node) if isinstance(nodes, dict) else None
                            if isinstance(progress, dict) and progress.get("state") == "running":
                                value = int(progress.get("value") or 0)
                                maximum = int(progress.get("max") or 0)
                        if value is not None and value >= interrupt_after_step:
                            progress_at_interrupt = {"value": value, "max": maximum}
                            interrupt_response = await _json_request(
                                session,
                                "POST",
                                f"{server}/interrupt",
                                json={"prompt_id": prompt_id},
                            )
                    if event_type in TERMINAL_EVENTS and data.get("prompt_id") == prompt_id:
                        terminal = {"type": event_type, "data": data}
        history = await _json_request(session, "GET", f"{server}/history/{prompt_id}")
    return {
        "prompt_id": prompt_id,
        "terminal": terminal,
        "progress_at_interrupt": progress_at_interrupt,
        "interrupt_response": interrupt_response,
        "history": history.get(prompt_id),
        "events": events,
        "elapsed_seconds": round(time.monotonic() - started, 4),
    }


def _read_safetensors(path: Path, metadata_key: str) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        from safetensors import safe_open
    except ImportError as error:
        raise RuntimeError("safetensors is required in the ComfyUI Python environment") from error
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        raw = (handle.metadata() or {}).get(metadata_key)
        if not raw:
            raise ValueError(f"{path.name} has no {metadata_key} metadata")
        payload = json.loads(raw)
        tensors = {key: handle.get_tensor(key).clone() for key in handle.keys()}
    return payload, tensors


def _tensor_digest(tensor) -> dict[str, Any]:
    cpu = tensor.detach().to("cpu").contiguous()
    raw = cpu.view(dtype=__import__("torch").uint8).numpy().tobytes()
    return {
        "shape": list(cpu.shape),
        "dtype": str(cpu.dtype),
        "sha256": hashlib.sha256(raw).hexdigest().upper(),
        "finite": bool(__import__("torch").isfinite(cpu.float()).all().item()),
    }


def latent_file_report(path: Path) -> dict[str, Any]:
    payload, tensors = _read_safetensors(path, NATIVE_METADATA_KEY)
    return {
        "path": str(path.resolve()),
        "file_sha256": _sha256_file(path),
        "checkpoint_id": payload.get("checkpoint_id"),
        "tensors": {key: _tensor_digest(value) for key, value in sorted(tensors.items())},
    }


def compare_latent_reports(control: Mapping[str, Any], resumed: Mapping[str, Any]) -> dict[str, Any]:
    control_tensors = control.get("tensors") if isinstance(control.get("tensors"), dict) else {}
    resumed_tensors = resumed.get("tensors") if isinstance(resumed.get("tensors"), dict) else {}
    keys_equal = set(control_tensors) == set(resumed_tensors)
    per_tensor = {
        key: control_tensors.get(key) == resumed_tensors.get(key)
        for key in sorted(set(control_tensors) | set(resumed_tensors))
    }
    return {
        "tensor_keys_equal": keys_equal,
        "per_tensor_exact": per_tensor,
        "all_tensors_exact": keys_equal and bool(per_tensor) and all(per_tensor.values()),
    }


def _run_checked(command: list[str], *, binary: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=not binary,
        timeout=180,
    )


def media_report(path: Path, *, ffmpeg: str, ffprobe: str) -> dict[str, Any]:
    strict_commands = {
        "video": [ffmpeg, "-v", "error", "-xerror", "-err_detect", "explode", "-threads", "1", "-i", str(path), "-map", "0:v:0", "-f", "null", "-"],
        "audio": [ffmpeg, "-v", "error", "-xerror", "-err_detect", "explode", "-threads", "1", "-i", str(path), "-map", "0:a:0", "-f", "null", "-"],
        "combined": [ffmpeg, "-v", "error", "-xerror", "-err_detect", "explode", "-threads", "1", "-i", str(path), "-f", "null", "-"],
    }
    strict = {}
    for name, command in strict_commands.items():
        try:
            completed = _run_checked(command)
            strict[name] = {"passed": True, "diagnostic": completed.stderr[-1000:]}
        except subprocess.CalledProcessError as error:
            strict[name] = {"passed": False, "diagnostic": str(error.stderr)[-2000:]}

    probe = json.loads(
        _run_checked(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "stream=index,codec_type,codec_name,width,height,r_frame_rate,avg_frame_rate,nb_frames,sample_rate,channels,duration:format=duration",
                "-of",
                "json",
                str(path),
            ]
        ).stdout
    )
    video_raw = _run_checked(
        [ffmpeg, "-v", "error", "-threads", "1", "-i", str(path), "-map", "0:v:0", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        binary=True,
    ).stdout
    audio_raw = _run_checked(
        [ffmpeg, "-v", "error", "-threads", "1", "-i", str(path), "-map", "0:a:0", "-f", "f32le", "-ac", "2", "-ar", "32000", "-"],
        binary=True,
    ).stdout
    return {
        "path": str(path.resolve()),
        "file_sha256": _sha256_file(path),
        "strict_decode": strict,
        "strict_decode_passed": all(item["passed"] for item in strict.values()),
        "probe": probe,
        "decoded_video": {
            "bytes": len(video_raw),
            "sha256": hashlib.sha256(video_raw).hexdigest().upper(),
        },
        "decoded_audio": {
            "bytes": len(audio_raw),
            "sha256": hashlib.sha256(audio_raw).hexdigest().upper(),
        },
    }


def compare_media_reports(control: Mapping[str, Any], resumed: Mapping[str, Any]) -> dict[str, Any]:
    control_streams = control.get("probe", {}).get("streams", [])
    resumed_streams = resumed.get("probe", {}).get("streams", [])
    return {
        "both_strict_decode": bool(control.get("strict_decode_passed") and resumed.get("strict_decode_passed")),
        "stream_structure_equal": control_streams == resumed_streams,
        "decoded_video_exact": control.get("decoded_video") == resumed.get("decoded_video"),
        "decoded_audio_exact": control.get("decoded_audio") == resumed.get("decoded_audio"),
    }


def _latest_file(root: Path, pattern: str) -> Path:
    candidates = [path for path in root.glob(pattern) if path.is_file()]
    if not candidates:
        raise FileNotFoundError(f"no file matched {pattern!r} under {root}")
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def _checkpoint_report(path: Path) -> dict[str, Any]:
    payload, tensors = _read_safetensors(path, NFE_METADATA_KEY)
    return {
        "path": str(path.resolve()),
        "file_sha256": _sha256_file(path),
        "payload": payload,
        "tensors": {key: _tensor_digest(value) for key, value in sorted(tensors.items())},
    }


def _wait_gpu_return(max_used_mib: int, timeout: float = 60.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    sample = gpu_memory_mib()
    while sample.get("available") and int(sample.get("used_mib", 0)) > max_used_mib:
        if time.monotonic() >= deadline:
            break
        time.sleep(1.0)
        sample = gpu_memory_mib()
    return sample


def run_real_probe(args: argparse.Namespace, preflight_report: Mapping[str, Any]) -> dict[str, Any]:
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    run_root = (args.artifact_root / run_id).resolve()
    run_root.mkdir(parents=True, exist_ok=False)
    (run_root / "prompts").mkdir()
    baseline_gpu = gpu_memory_mib()
    paths = _model_paths(args.comfy_root.resolve())
    verified_model_hashes = hash_model_files(paths)
    model_hash_manifest = {
        "schema": f"{SCHEMA}.model_hash_manifest",
        "created_at": _utc_now(),
        "scope": "full_sha256_all_declared_files",
        "files": {
            role: {
                "path": str(paths[role].resolve()),
                "bytes": paths[role].stat().st_size,
                "sha256": verified_model_hashes[role],
            }
            for role in sorted(paths)
        },
    }
    (run_root / "model_hash_manifest.json").write_text(
        json.dumps(model_hash_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    contract = model_contract_json(paths, verified_hashes=verified_model_hashes)
    post_hash_start_gate = isolated_start_gate(args)
    if not post_hash_start_gate["ready"]:
        report = {
            "schema": SCHEMA,
            "created_at": _utc_now(),
            "run_id": run_id,
            "run_root": str(run_root),
            "status": "ABSTAIN_RESOURCE_CHANGED_AFTER_MODEL_HASH",
            "preflight": dict(preflight_report),
            "model_contract_json": contract,
            "model_hash_manifest": model_hash_manifest,
            "post_hash_start_gate": post_hash_start_gate,
            "process_ids": [],
            "checks": {"no_isolated_server_started": True},
            "passed": False,
            "boundary": (
                "Model files were fully hashed after explicit consent, but the target port or "
                "GPU resource state changed before launch. No isolated ComfyUI process started."
            ),
        }
        report_path = run_root / "validation_report.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return report
    baseline_gpu = post_hash_start_gate["gpu"]
    checkpoint_relative = f"{run_id}/state.h3nfe.safetensors"
    prompts = {
        mode: build_prompt(
            mode=mode,
            run_id=run_id,
            checkpoint_path=checkpoint_relative,
            model_contract_id=contract,
        )
        for mode in ("disabled", "checkpoint_each_step", "resume")
    }
    for mode, prompt in prompts.items():
        (run_root / "prompts" / f"{mode}.json").write_text(
            json.dumps(prompt, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    server_url = f"http://{args.host}:{args.port}"
    phases: dict[str, Any] = {}
    process_ids: list[int] = []
    server_a = IsolatedServer(args, run_root, "control_and_checkpoint")
    try:
        process_ids.append(server_a.start())
        phases["control"] = asyncio.run(
            submit_prompt(server=server_url, prompt=prompts["disabled"], timeout_seconds=args.timeout_seconds)
        )
        if phases["control"].get("terminal", {}).get("type") != "execution_success":
            raise RuntimeError("control render did not complete successfully")
        phases["checkpoint"] = asyncio.run(
            submit_prompt(
                server=server_url,
                prompt=prompts["checkpoint_each_step"],
                timeout_seconds=args.timeout_seconds,
                interrupt_node="12",
                interrupt_after_step=args.interrupt_after_step,
            )
        )
        if phases["checkpoint"].get("terminal", {}).get("type") != "execution_interrupted":
            raise RuntimeError("checkpoint render was not interrupted at the requested boundary")
    finally:
        server_a.stop()

    checkpoint = run_root / "output" / "MiniMaxH3" / "nfe_checkpoints" / checkpoint_relative
    checkpoint_info = _checkpoint_report(checkpoint)
    completed_steps = checkpoint_info.get("payload", {}).get("completed_steps")
    if completed_steps != args.interrupt_after_step:
        raise RuntimeError(
            f"checkpoint completed_steps={completed_steps!r}, expected {args.interrupt_after_step}"
        )
    after_first_process_gpu = _wait_gpu_return(int(baseline_gpu.get("used_mib", 0)) + 512)

    server_b = IsolatedServer(args, run_root, "fresh_process_resume")
    try:
        process_ids.append(server_b.start())
        phases["resume"] = asyncio.run(
            submit_prompt(server=server_url, prompt=prompts["resume"], timeout_seconds=args.timeout_seconds)
        )
        if phases["resume"].get("terminal", {}).get("type") != "execution_success":
            raise RuntimeError("resume render did not complete successfully")
    finally:
        server_b.stop()
    final_gpu = _wait_gpu_return(int(baseline_gpu.get("used_mib", 0)) + 512)

    latent_root = run_root / "output" / "MiniMaxH3" / "latent_checkpoints" / run_id
    control_latent_path = _latest_file(latent_root, "disabled_final*.h3latent.safetensors")
    resume_latent_path = _latest_file(latent_root, "resume_final*.h3latent.safetensors")
    latent = {
        "control": latent_file_report(control_latent_path),
        "resume": latent_file_report(resume_latent_path),
    }
    latent_compare = compare_latent_reports(latent["control"], latent["resume"])

    media_root = run_root / "output" / "MiniMaxH3_NFE_Resume_Probe"
    control_media_path = _latest_file(media_root, f"{run_id}_disabled*.mp4")
    resume_media_path = _latest_file(media_root, f"{run_id}_resume*.mp4")
    ffmpeg = str(preflight_report["ffmpeg"])
    ffprobe = str(preflight_report["ffprobe"])
    media = {
        "control": media_report(control_media_path, ffmpeg=ffmpeg, ffprobe=ffprobe),
        "resume": media_report(resume_media_path, ffmpeg=ffmpeg, ffprobe=ffprobe),
    }
    media_compare = compare_media_reports(media["control"], media["resume"])
    checks = {
        "two_distinct_server_processes": len(process_ids) == 2 and len(set(process_ids)) == 2,
        "control_success": phases["control"].get("terminal", {}).get("type") == "execution_success",
        "checkpoint_interrupted": phases["checkpoint"].get("terminal", {}).get("type") == "execution_interrupted",
        "checkpoint_exact_completed_step": completed_steps == args.interrupt_after_step,
        "checkpoint_total_steps_four": checkpoint_info.get("payload", {}).get("total_steps") == 4,
        "resume_success": phases["resume"].get("terminal", {}).get("type") == "execution_success",
        "native_latents_exact": latent_compare["all_tensors_exact"],
        "strict_media_decode": media_compare["both_strict_decode"],
        "media_stream_structure_equal": media_compare["stream_structure_equal"],
        "decoded_video_exact": media_compare["decoded_video_exact"],
        "decoded_audio_exact": media_compare["decoded_audio_exact"],
    }
    report = {
        "schema": SCHEMA,
        "created_at": _utc_now(),
        "run_id": run_id,
        "run_root": str(run_root),
        "preflight": dict(preflight_report),
        "model_contract_json": contract,
        "model_hash_manifest": model_hash_manifest,
        "post_hash_start_gate": post_hash_start_gate,
        "process_ids": process_ids,
        "phases": phases,
        "checkpoint": checkpoint_info,
        "latent": latent,
        "latent_comparison": latent_compare,
        "media": media,
        "media_comparison": media_compare,
        "gpu": {
            "baseline": baseline_gpu,
            "after_first_process": after_first_process_gpu,
            "final": final_gpu,
        },
        "checks": checks,
        "passed": all(checks.values()),
        "boundary": (
            "One 256x256x22 T2VA 4-NFE control/interruption/fresh-process-resume proof on "
            "one machine. Exact native latent and decoded-stream parity proves this declared "
            "contract only; it does not prove arbitrary model wrappers, samplers, modalities, "
            "mid-forward crash recovery, quality preference, repeated-run stability, or larger "
            "resolution memory safety."
        ),
    }
    report_path = run_root / "validation_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comfy-root",
        type=Path,
        default=Path(r"F:\AI-T8-video-onekey\ComfyUI"),
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(r"F:\AI-T8-video-onekey\python\python.exe"),
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts" / "nfe-resume-real-runtime-v1",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8197)
    parser.add_argument("--min-free-vram-mib", type=int, default=12000)
    parser.add_argument("--interrupt-after-step", type=int, default=2, choices=(1, 2, 3))
    parser.add_argument("--server-start-timeout", type=float, default=180.0)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument(
        "--confirm-run",
        action="store_true",
        help="Run the two isolated GPU processes after all preflight checks pass.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    preflight_report = preflight(args)
    preflight_path = args.artifact_root / "latest_preflight.json"
    preflight_path.write_text(
        json.dumps(preflight_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not args.confirm_run:
        print(
            json.dumps(
                {
                    "status": preflight_report["status"],
                    "ready_for_real_run": preflight_report["ready_for_real_run"],
                    "preflight": str(preflight_path.resolve()),
                    "real_run_started": False,
                },
                ensure_ascii=False,
            )
        )
        return 0
    if not preflight_report["ready_for_real_run"]:
        print(
            json.dumps(
                {
                    "status": preflight_report["status"],
                    "preflight": str(preflight_path.resolve()),
                    "real_run_started": False,
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 3
    report = run_real_probe(args, preflight_report)
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "run_root": report["run_root"],
                "report": str(Path(report["run_root"]) / "validation_report.json"),
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
