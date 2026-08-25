#!/usr/bin/env python3
"""Run one guarded Turbo4 -> Audio Refine4 MiniMax H3 smoke.

The default invocation only writes a preflight report. ``--confirm-run`` permits exactly one
prompt on a private loopback ComfyUI process. The tool refuses an active user service on 8188,
a busy private port, missing local assets, less than the configured free VRAM, or less than
16 GiB host commit headroom. It never retries and never performs a stress or quality matrix.
"""

from __future__ import annotations

import argparse
import asyncio
import ctypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import threading
import time
from typing import Any, Mapping
import uuid

import run_nfe_resume_real_probe as shared


SCHEMA = "t8.minimax_h3.audio_refine.smoke.v1"
MAX_PROMPTS_PER_INVOCATION = 1
BASE_NAME = "minimax_h3_fl2va_int8_convrot.safetensors"
LORA_NAME = "minimax_h3_turbo_4步加速ema_comfyui.safetensors"
CLIP_NAME = "qwen3vl_8b_fp8_scaled.safetensors"
PROJECTION_NAME = "mmh3-8b-ClipProj-v3.1.safetensors"
VIDEO_VAE_NAME = "minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE_NAME = "minimax_h3_audio_vae_fp32.safetensors"
PROMPT = (
    "一位女性面对镜头自然地说：‘你在干嘛呢，我在这里呀，看看效果如何。’ "
    "安静的室内环境，声音清晰，无背景音乐。"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _model_paths(comfy_root: Path) -> dict[str, Path]:
    models = comfy_root / "models"
    return {
        "base": models / "diffusion_models" / BASE_NAME,
        "lora": models / "loras" / LORA_NAME,
        "clip": models / "text_encoders" / CLIP_NAME,
        "projection": models / "clip_projections" / PROJECTION_NAME,
        "video_vae": models / "vae" / VIDEO_VAE_NAME,
        "audio_vae": models / "vae" / AUDIO_VAE_NAME,
    }


def build_prompt(
    *,
    run_id: str,
    seed: int = 2608260404,
    width: int = 256,
    height: int = 256,
    frames: int = 22,
) -> dict[str, Any]:
    """Build the single fixed mechanical graph."""

    return {
        "1": {"inputs": {"vae_name": VIDEO_VAE_NAME}, "class_type": "VAELoader"},
        "2": {"inputs": {"vae_name": AUDIO_VAE_NAME}, "class_type": "VAELoader"},
        "3": {
            "inputs": {"clip_name": CLIP_NAME, "type": "boogu", "device": "default"},
            "class_type": "CLIPLoader",
        },
        "4": {
            "inputs": {"clip": ["3", 0], "projection": PROJECTION_NAME},
            "class_type": "ClipProjApply",
        },
        "5": {
            "inputs": {
                "clip": ["4", 0],
                "encoder_family": "8B",
                "encoder_architecture": "qwen3_vl",
                "encoder_quantization": "fp8",
                "load_mode": "stock_pageable",
                "projection_path": PROJECTION_NAME,
                "has_reference_images": False,
                "has_reference_videos": False,
                "enforcement": "block_hard_conflicts",
            },
            "class_type": "MiniMaxH3ClipProjCompatibilityAuditT8Advanced",
        },
        "6": {
            "inputs": {"unet_name": BASE_NAME, "weight_dtype": "default"},
            "class_type": "UNETLoader",
        },
        "7": {
            "inputs": {
                "model": ["6", 0],
                "lora_name": LORA_NAME,
                "strength_model": 1.0,
            },
            "class_type": "LoraLoaderModelOnly",
        },
        "8": {
            "inputs": {
                "prompt": PROMPT,
                "width": int(width),
                "height": int(height),
                "length": int(frames),
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
                "model": ["7", 0],
                "av_latent": ["8", 1],
                "steps": 4,
                "shift_video": 12.0,
                "shift_audio": 3.0,
                "sampler_name": "dual_clock_euler",
                "scheduler": "native_flow",
            },
            "class_type": "MiniMaxH3DualClockSamplerT8",
        },
        "10": {"inputs": {"noise_seed": seed}, "class_type": "RandomNoise"},
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
            "inputs": {"images": ["13", 0], "fps": 24.0, "audio": ["13", 1]},
            "class_type": "CreateVideo",
        },
        "15": {
            "inputs": {
                "model": ["9", 0],
                "positive": ["8", 0],
                "av_latent": ["12", 0],
                "conditioned_prompt": ["8", 3],
                "media_map_json": ["8", 4],
                "conditioning_report": ["8", 5],
                "minimum_free_vram_mib": 512,
                "minimum_commit_headroom_gib": 16.0,
                "hash_chunk_megabytes": 8,
            },
            "class_type": "MiniMaxH3AudioRefineAuditT8Advanced",
        },
        "16": {
            "inputs": {
                "audit": ["15", 0],
                "refine_steps": 4,
                "audio_denoise": 0.5,
                "refine_seed": seed,
                "model_strategy": "connected_model_explicit",
            },
            "class_type": "MiniMaxH3AudioRefinePlanT8Advanced",
        },
        "17": {
            "inputs": {
                "plan": ["16", 0],
                "model": ["9", 0],
                "positive": ["8", 0],
                "av_latent": ["12", 0],
            },
            "class_type": "MiniMaxH3AudioRefineDualClockSetupT8Advanced",
        },
        "18": {
            "inputs": {
                "noise": ["17", 1],
                "guider": ["17", 2],
                "sampler": ["17", 3],
                "sigmas": ["17", 4],
                "latent_image": ["17", 5],
            },
            "class_type": "SamplerCustomAdvanced",
        },
        "19": {
            "inputs": {
                "second_pass_input": ["17", 5],
                "second_pass_output": ["18", 0],
                "expected_audio_strength": 1.0,
                "fail_on_locked_mismatch": True,
                "locked_atol": 0.0,
            },
            "class_type": "MiniMaxH3TwoPassAudioAuditT8Advanced",
        },
        "20": {
            "inputs": {
                "av_latent": ["19", 0],
                "video_vae": ["1", 0],
                "audio_vae": ["2", 0],
            },
            "class_type": "MiniMaxH3AVDecodeT8",
        },
        "21": {
            "inputs": {"images": ["20", 0], "fps": 24.0, "audio": ["20", 1]},
            "class_type": "CreateVideo",
        },
        "22": {
            "inputs": {
                "video": ["14", 0],
                "filename_prefix": f"MiniMaxH3_AudioRefine_Smoke/{run_id}_original",
                "format": "mp4",
                "codec": "h264",
            },
            "class_type": "SaveVideo",
        },
        "23": {
            "inputs": {
                "video": ["21", 0],
                "filename_prefix": f"MiniMaxH3_AudioRefine_Smoke/{run_id}_refined",
                "format": "mp4",
                "codec": "h264",
            },
            "class_type": "SaveVideo",
        },
        "24": {
            "inputs": {
                "audio": ["20", 1],
                "video_frame_count": int(frames),
                "fps": 24.0,
                "opening_window_ms": 40.0,
                "comparison_window_ms": 250.0,
                "pop_jump_threshold": 0.15,
                "dc_jump_threshold": 0.02,
                "wrap_correlation_threshold": 0.985,
                "clipping_ratio_threshold": 0.001,
                "max_av_delta_ms": 50.0,
            },
            "class_type": "MiniMaxH3AudioIntegrityAuditT8Advanced",
        },
        "25": {
            "inputs": {
                "reference_audio": ["13", 1],
                "candidate_audio": ["20", 1],
                "analysis_window_ms": 500.0,
                "hop_ms": 100.0,
                "active_rms_floor_dbfs": -50.0,
                "spectral_drift_threshold": 0.30,
                "level_delta_threshold_db": 4.0,
                "persistent_window_count": 3,
                "max_duration_delta_ms": 50.0,
            },
            "class_type": "MiniMaxH3AudioPerceptualDriftAuditT8Advanced",
        },
        "26": {
            "inputs": {
                "original_av_latent": ["12", 0],
                "candidate_av_latent": ["19", 0],
                "original_audio": ["13", 1],
                "candidate_audio": ["20", 1],
                "accept_candidate": False,
                "video_frame_count": int(frames),
                "fps": 24.0,
                "maximum_duration_delta_ms": 50.0,
                "spectral_drift_threshold": 0.30,
                "level_delta_threshold_db": 4.0,
                "persistent_window_count": 3,
            },
            "class_type": "MiniMaxH3AudioRefineQualityGateT8Advanced",
        },
        "27": {
            "inputs": {
                "av_latent": ["26", 0],
                "video_vae": ["1", 0],
                "audio_vae": ["2", 0],
            },
            "class_type": "MiniMaxH3AVDecodeT8",
        },
        "28": {
            "inputs": {"images": ["27", 0], "fps": 24.0, "audio": ["27", 1]},
            "class_type": "CreateVideo",
        },
        "29": {
            "inputs": {
                "video": ["28", 0],
                "filename_prefix": f"MiniMaxH3_AudioRefine_Smoke/{run_id}_selected_default_original",
                "format": "mp4",
                "codec": "h264",
            },
            "class_type": "SaveVideo",
        },
    }


def host_memory_snapshot() -> dict[str, Any]:
    """Read physical and commit headroom without adding a dependency."""

    if os.name != "nt":
        return {"available": False, "reason": "only Windows is validated by this runner"}

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
        success = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    except (AttributeError, OSError) as error:
        return {"available": False, "error": f"{type(error).__name__}: {error}"}
    if not success:
        return {"available": False, "error": "GlobalMemoryStatusEx returned false"}
    gib = 1024**3
    return {
        "available": True,
        "ram_available_gib": round(status.ullAvailPhys / gib, 6),
        "commit_headroom_gib": round(status.ullAvailPageFile / gib, 6),
        "commit_limit_gib": round(status.ullTotalPageFile / gib, 6),
    }


def _identity_manifest(paths: Mapping[str, Path]) -> dict[str, Any]:
    return {
        role: {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for role, path in sorted(paths.items())
        if path.is_file()
    }


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    comfy_root = args.comfy_root.resolve()
    plugin_root = args.plugin_root.resolve()
    paths = _model_paths(comfy_root)
    required = {
        "comfy_main": comfy_root / "main.py",
        "python": args.python.resolve(),
        "plugin_init": plugin_root / "__init__.py",
        "audio_refine_node": plugin_root / "nodes_audio_refine_advanced.py",
        "clipproj_nodes": comfy_root / "custom_nodes" / "ComfyUI-ClipProj",
        **paths,
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    ffmpeg = shutil.which(args.ffmpeg) or (
        args.ffmpeg if Path(args.ffmpeg).is_file() else None
    )
    ffprobe = shutil.which(args.ffprobe) or (
        args.ffprobe if Path(args.ffprobe).is_file() else None
    )
    gpu = shared.gpu_memory_mib()
    host = host_memory_snapshot()
    user_active = shared.port_is_listening(args.host, 8188)
    target_busy = shared.port_is_listening(args.host, args.port)
    minimum_free_vram_mib = max(512, int(args.min_free_vram_mib))
    checks = {
        "required_paths_present": not missing,
        "ffmpeg_present": bool(ffmpeg),
        "ffprobe_present": bool(ffprobe),
        "user_service_8188_inactive": not user_active,
        "target_port_free": not target_busy,
        "gpu_query_available": bool(gpu.get("available")),
        "free_vram_gate": bool(
            gpu.get("available")
            and int(gpu.get("free_mib", 0)) >= minimum_free_vram_mib
        ),
        "commit_telemetry_available": isinstance(
            host.get("commit_headroom_gib"), (int, float)
        ),
        "commit_headroom_at_least_16_gib": bool(
            isinstance(host.get("commit_headroom_gib"), (int, float))
            and float(host["commit_headroom_gib"]) >= 16.0
        ),
    }
    ready = all(checks.values())
    if ready:
        status = "READY"
    elif missing or not ffmpeg or not ffprobe:
        status = "ABSTAIN_MISSING_DEPENDENCY"
    elif user_active:
        status = "ABSTAIN_USER_SERVICE_ACTIVE"
    elif target_busy:
        status = "ABSTAIN_TARGET_PORT_BUSY"
    elif not gpu.get("available"):
        status = "ABSTAIN_GPU_STATE_UNKNOWN"
    elif int(gpu.get("free_mib", 0)) < minimum_free_vram_mib:
        status = "ABSTAIN_INSUFFICIENT_FREE_VRAM"
    elif not isinstance(host.get("commit_headroom_gib"), (int, float)):
        status = "ABSTAIN_HOST_COMMIT_UNKNOWN"
    else:
        status = "ABSTAIN_INSUFFICIENT_HOST_COMMIT"
    return {
        "schema": f"{SCHEMA}.preflight",
        "created_at": _utc_now(),
        "status": status,
        "ready_for_real_run": ready,
        "checks": checks,
        "missing_paths": missing,
        "gpu": gpu,
        "host": host,
        "minimum_free_vram_mib": minimum_free_vram_mib,
        "minimum_commit_headroom_gib": 16.0,
        "target": {"host": args.host, "port": args.port, "busy": target_busy},
        "user_service_8188_active": user_active,
        "model_identity": _identity_manifest(paths),
        "ffmpeg": ffmpeg,
        "ffprobe": ffprobe,
        "maximum_prompts": MAX_PROMPTS_PER_INVOCATION,
        "boundary": (
            "Read-only preflight. It performs no model load, queues no prompt, and never "
            "interrupts or unloads the user's service."
        ),
    }


def _server_command(args: argparse.Namespace, run_root: Path) -> list[str]:
    run_root.mkdir(parents=True, exist_ok=True)
    extra_paths = run_root / "extra_model_paths.yaml"
    parent = args.plugin_root.resolve().parent.as_posix().replace("'", "''")
    extra_paths.write_text(
        f"audio_refine_smoke:\n  base_path: '{parent}'\n  custom_nodes: .\n",
        encoding="utf-8",
    )
    return [
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
        "1.0",
        "--extra-model-paths-config",
        str(extra_paths),
        "--disable-all-custom-nodes",
        "--whitelist-custom-nodes",
        args.plugin_root.resolve().name,
        "ComfyUI-ClipProj",
        "--input-directory",
        str((run_root / "input").resolve()),
        "--output-directory",
        str((run_root / "output").resolve()),
        "--temp-directory",
        str((run_root / "temp").resolve()),
        "--user-directory",
        str((run_root / "user").resolve()),
        "--database-url",
        "sqlite:///:memory:",
    ]


class GpuPeakMonitor:
    def __init__(self, interval_seconds: float = 0.25):
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples = 0
        self.peak_used_mib: int | None = None
        self.minimum_free_mib: int | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            sample = shared.gpu_memory_mib()
            if sample.get("available"):
                used = int(sample["used_mib"])
                free = int(sample["free_mib"])
                self.peak_used_mib = (
                    used if self.peak_used_mib is None else max(self.peak_used_mib, used)
                )
                self.minimum_free_mib = (
                    free if self.minimum_free_mib is None else min(self.minimum_free_mib, free)
                )
                self.samples += 1
            self._stop.wait(self.interval_seconds)

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        return {
            "samples": self.samples,
            "interval_seconds": self.interval_seconds,
            "peak_used_mib": self.peak_used_mib,
            "minimum_free_mib": self.minimum_free_mib,
        }


def _history_text(phase: Mapping[str, Any], node_id: str) -> str | None:
    history = phase.get("history")
    outputs = history.get("outputs") if isinstance(history, Mapping) else None
    node_output = outputs.get(node_id) if isinstance(outputs, Mapping) else None
    if not isinstance(node_output, Mapping):
        return None
    for key in ("text", "report_json"):
        values = node_output.get(key)
        if isinstance(values, list) and values and isinstance(values[0], str):
            return values[0]
    return None


def _media_checks(
    report: Mapping[str, Any], *, width: int, height: int, frames: int
) -> dict[str, bool]:
    streams = report.get("probe", {}).get("streams", [])
    video = [stream for stream in streams if stream.get("codec_type") == "video"]
    audio = [stream for stream in streams if stream.get("codec_type") == "audio"]
    return {
        "strict_decode": bool(report.get("strict_decode_passed")),
        "video_dimensions": len(video) == 1
        and int(video[0].get("width") or 0) == int(width)
        and int(video[0].get("height") or 0) == int(height),
        "video_frame_count": int(report.get("decoded_video", {}).get("bytes") or 0)
        == int(frames) * int(width) * int(height) * 3,
        "audio_32khz_stereo": len(audio) == 1
        and int(audio[0].get("sample_rate") or 0) == 32000
        and int(audio[0].get("channels") or 0) == 2,
        "audio_nonempty": int(report.get("decoded_audio", {}).get("bytes") or 0) > 0,
    }


def _mechanical_checks(
    phase: Mapping[str, Any],
    original_checks: Mapping[str, bool],
    refined_checks: Mapping[str, bool],
    selected_checks: Mapping[str, bool],
) -> dict[str, bool]:
    events = phase.get("events", [])
    executed_nodes = {
        str(event.get("node"))
        for event in events
        if isinstance(event, Mapping)
        and event.get("type") == "executing"
        and event.get("node") is not None
    }
    execution_success = bool(
        phase.get("terminal", {}).get("type") == "execution_success"
    )
    return {
        "one_prompt_submitted": bool(phase.get("prompt_id")),
        "execution_success": execution_success,
        "latent_audit_completed": execution_success and "19" in executed_nodes,
        "audio_integrity_audit_completed": execution_success and "24" in executed_nodes,
        "perceptual_drift_audit_completed": execution_success and "25" in executed_nodes,
        "quality_gate_completed": execution_success and "26" in executed_nodes,
        **{f"original_{key}": bool(value) for key, value in original_checks.items()},
        **{f"refined_{key}": bool(value) for key, value in refined_checks.items()},
        **{f"selected_{key}": bool(value) for key, value in selected_checks.items()},
    }


def _write_report(run_root: Path, report: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(report)
    (run_root / "validation_report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def run_real_probe(
    args: argparse.Namespace, preflight_report: Mapping[str, Any]
) -> dict[str, Any]:
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    run_root = (args.artifact_root / run_id).resolve()
    run_root.mkdir(parents=True, exist_ok=False)
    prompt = build_prompt(
        run_id=run_id,
        seed=args.seed,
        width=args.width,
        height=args.height,
        frames=args.frames,
    )
    (run_root / "prompt.json").write_text(
        json.dumps(prompt, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    post_gate = preflight(args)
    if not post_gate["ready_for_real_run"]:
        return _write_report(
            run_root,
            {
                "schema": SCHEMA,
                "status": "ABSTAIN_RESOURCE_CHANGED_BEFORE_START",
                "passed": False,
                "run_root": str(run_root),
                "preflight": dict(preflight_report),
                "post_gate": post_gate,
                "process_ids": [],
                "prompts_submitted": 0,
            },
        )

    original_command = shared._server_command
    shared._server_command = _server_command
    server = shared.IsolatedServer(args, run_root, "audio_refine_smoke")
    monitor = GpuPeakMonitor()
    process_ids: list[int] = []
    phase: dict[str, Any] | None = None
    runtime_error: dict[str, str] | None = None
    started = time.monotonic()
    monitor.start()
    try:
        process_ids.append(server.start())
        phase = asyncio.run(
            shared.submit_prompt(
                server=f"http://{args.host}:{args.port}",
                prompt=prompt,
                timeout_seconds=args.timeout_seconds,
            )
        )
    except Exception as error:
        runtime_error = {"type": type(error).__name__, "message": str(error)}
    finally:
        server.stop()
        shared._server_command = original_command
        gpu_monitor = monitor.stop()

    execution_success = bool(
        phase and phase.get("terminal", {}).get("type") == "execution_success"
    )
    original_media = None
    refined_media = None
    selected_media = None
    original_checks: dict[str, bool] = {}
    refined_checks: dict[str, bool] = {}
    selected_checks: dict[str, bool] = {}
    if execution_success:
        try:
            output_root = run_root / "output" / "MiniMaxH3_AudioRefine_Smoke"
            original_path = shared._latest_file(output_root, f"{run_id}_original*.mp4")
            refined_path = shared._latest_file(output_root, f"{run_id}_refined*.mp4")
            selected_path = shared._latest_file(
                output_root, f"{run_id}_selected_default_original*.mp4"
            )
            original_media = shared.media_report(
                original_path,
                ffmpeg=str(preflight_report["ffmpeg"]),
                ffprobe=str(preflight_report["ffprobe"]),
            )
            refined_media = shared.media_report(
                refined_path,
                ffmpeg=str(preflight_report["ffmpeg"]),
                ffprobe=str(preflight_report["ffprobe"]),
            )
            selected_media = shared.media_report(
                selected_path,
                ffmpeg=str(preflight_report["ffmpeg"]),
                ffprobe=str(preflight_report["ffprobe"]),
            )
            dimensions = {
                "width": args.width,
                "height": args.height,
                "frames": args.frames,
            }
            original_checks = _media_checks(original_media, **dimensions)
            refined_checks = _media_checks(refined_media, **dimensions)
            selected_checks = _media_checks(selected_media, **dimensions)
        except Exception as error:
            runtime_error = {"type": type(error).__name__, "message": str(error)}

    latent_audit_text = _history_text(phase or {}, "19")
    integrity_text = _history_text(phase or {}, "24")
    drift_text = _history_text(phase or {}, "25")
    latent_audit = json.loads(latent_audit_text) if latent_audit_text else None
    integrity = json.loads(integrity_text) if integrity_text else None
    drift = json.loads(drift_text) if drift_text else None
    checks = _mechanical_checks(
        phase or {}, original_checks, refined_checks, selected_checks
    )
    checks["one_isolated_process"] = len(process_ids) == 1
    decoded_video_identical = bool(
        original_media
        and refined_media
        and original_media.get("decoded_video", {}).get("sha256")
        == refined_media.get("decoded_video", {}).get("sha256")
    )
    selected_fallback_exact = bool(
        original_media
        and selected_media
        and original_media.get("decoded_video", {}).get("sha256")
        == selected_media.get("decoded_video", {}).get("sha256")
        and original_media.get("decoded_audio", {}).get("sha256")
        == selected_media.get("decoded_audio", {}).get("sha256")
    )
    checks["quality_gate_default_fallback_exact"] = selected_fallback_exact
    passed = bool(checks) and all(checks.values()) and runtime_error is None
    return _write_report(
        run_root,
        {
            "schema": SCHEMA,
            "created_at": _utc_now(),
            "status": "PASS" if passed else "FAIL_RUNTIME_OR_MEDIA_CONTRACT",
            "passed": passed,
            "run_id": run_id,
            "run_root": str(run_root),
            "preflight": dict(preflight_report),
            "post_gate": post_gate,
            "process_ids": process_ids,
            "prompts_submitted": int(bool(phase and phase.get("prompt_id"))),
            "phase": phase,
            "runtime_error": runtime_error,
            "generation_contract": {
                "prompt": PROMPT,
                "seed": args.seed,
                "width": args.width,
                "height": args.height,
                "frames": args.frames,
                "first_pass_nfe": 4,
                "refine_nfe": 4,
                "audio_denoise": 0.5,
                "video_shift": 12.0,
                "audio_shift": 3.0,
                "cfg": 1.0,
            },
            "latent_audit": latent_audit,
            "audio_integrity": integrity,
            "audio_perceptual_drift": drift,
            "candidate_video_decode_identical": decoded_video_identical,
            "candidate_video_relock_required": not decoded_video_identical,
            "original_media": original_media,
            "refined_media": refined_media,
            "selected_default_media": selected_media,
            "gpu": {"monitor": gpu_monitor, "final": shared.gpu_memory_mib()},
            "elapsed_seconds": round(time.monotonic() - started, 4),
            "checks": checks,
            "boundary": (
                "One bounded prompt only. PASS does not establish perceptual "
                "improvement, transcript or speaker preservation, lip-sync preservation, "
                "repeat stability, or universal 16GiB safety. Candidate video differences "
                "must be removed by the Audio Refine Quality Gate before delivery."
            ),
        },
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comfy-root", type=Path, default=Path(r"F:\AI-T8-video-onekey\ComfyUI")
    )
    parser.add_argument(
        "--python", type=Path, default=Path(r"F:\AI-T8-video-onekey\python\python.exe")
    )
    parser.add_argument("--plugin-root", type=Path, default=project_root)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=project_root / "artifacts" / "audio-refine-smoke-20260826",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8197)
    parser.add_argument("--min-free-vram-mib", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=2608260404)
    parser.add_argument("--server-start-timeout", type=float, default=180.0)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument(
        "--quality-pair",
        action="store_true",
        help="Use the single fixed 1056x608x124 review profile instead of 256x256x22.",
    )
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args(argv)
    if args.quality_pair:
        args.width, args.height, args.frames = 1056, 608, 124
    else:
        args.width, args.height, args.frames = 256, 256, 22
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    report = preflight(args)
    preflight_path = args.artifact_root / "latest_preflight.json"
    preflight_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not args.confirm_run:
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "ready_for_real_run": report["ready_for_real_run"],
                    "real_run_started": False,
                    "preflight": str(preflight_path.resolve()),
                },
                ensure_ascii=False,
            )
        )
        return 0
    if not report["ready_for_real_run"]:
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "real_run_started": False,
                    "preflight": str(preflight_path.resolve()),
                },
                ensure_ascii=False,
            )
        )
        return 3
    result = run_real_probe(args, report)
    print(
        json.dumps(
            {
                "status": result["status"],
                "passed": result["passed"],
                "run_root": result["run_root"],
                "report": str(Path(result["run_root"]) / "validation_report.json"),
            },
            ensure_ascii=False,
        )
    )
    if str(result["status"]).startswith("ABSTAIN"):
        return 3
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
