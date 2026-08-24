#!/usr/bin/env python3
"""Run one guarded MiniMax H3 ClipProj 4B T2VA mechanical probe.

The default invocation is preflight-only. A real GPU run requires ``--confirm-run`` and is
refused unless the isolated port is free, every declared dependency is present, the reviewed 4B
asset sizes match, and the selected GPU has the requested free-VRAM headroom. The official 4B
encoder and projection are fully SHA-256 checked only after explicit run consent, then the GPU and
port gates are checked again before an isolated ComfyUI process is started.

The tool owns only the isolated ComfyUI process that it starts. It never interrupts, unloads, or
terminates the user's normal service on port 8188. The fixed 256x256x22, four-NFE output is a
mechanical runtime proof, not a quality, speed, memory-saving, or universal 16GB claim.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import time
from typing import Any, Mapping
import uuid

import run_nfe_resume_real_probe as shared


SCHEMA = "t8.minimax_h3.clipproj_4b_real_probe.v1"
CLIP_NAME = "qwen3vl_4b_fp8_scaled.safetensors"
PROJECTION_NAME = "mmh3-4b-ClipProj-v3.1.safetensors"
BASE_NAME = "minimax_h3_fl2va_int8_convrot.safetensors"
LORA_NAME = "minimax_h3_turbo_4步加速ema_comfyui.safetensors"
VIDEO_VAE_NAME = "minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE_NAME = "minimax_h3_audio_vae_fp32.safetensors"
EXPECTED_ASSETS = {
    "clip": {
        "bytes": 5_242_467_968,
        "sha256": "54BD5144DF0BBC25DD6CCADFCB826B521445A1B06AE5A42570BDD2974CA87094",
        "revision": "e5ea8b4dd7f38f348b138eb0fe29f92c0e367e96",
    },
    "projection": {
        "bytes": 26_256_128,
        "sha256": "0184E5C8D666A131962506D21949C2D8A8C6F33445B7B5E347E9A7E0A5BAA819",
        "revision": "2ebdbcdc27a29a9607efdb221a9afcb9a0cdd808",
    },
}


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


def build_prompt(*, run_id: str, seed: int = 123456789) -> dict[str, Any]:
    """Build the fixed low-load 4B graph used by the real probe."""

    return {
        "1": {"inputs": {"vae_name": VIDEO_VAE_NAME}, "class_type": "VAELoader"},
        "2": {"inputs": {"vae_name": AUDIO_VAE_NAME}, "class_type": "VAELoader"},
        "3": {
            "inputs": {"clip_name": CLIP_NAME, "type": "krea2", "device": "default"},
            "class_type": "CLIPLoader",
        },
        "4": {
            "inputs": {"clip": ["3", 0], "projection": PROJECTION_NAME},
            "class_type": "ClipProjApply",
        },
        "5": {
            "inputs": {
                "clip": ["4", 0],
                "encoder_family": "4B",
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
                "model": ["7", 0],
                "av_latent": ["8", 1],
                "steps": 4,
                "shift_video": 12.0,
                "shift_audio": 3.0,
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
            "inputs": {
                "frame_rate": 24,
                "loop_count": 0,
                "filename_prefix": f"MiniMaxH3_ClipProj_4B_Probe/{run_id}",
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
    }


def _asset_size_checks(paths: Mapping[str, Path]) -> dict[str, bool]:
    return {
        role: paths[role].is_file()
        and paths[role].stat().st_size == int(EXPECTED_ASSETS[role]["bytes"])
        for role in EXPECTED_ASSETS
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
    ffmpeg = shared.shutil.which(args.ffmpeg) or (
        args.ffmpeg if Path(args.ffmpeg).is_file() else None
    )
    ffprobe = shared.shutil.which(args.ffprobe) or (
        args.ffprobe if Path(args.ffprobe).is_file() else None
    )
    start_gate = shared.isolated_start_gate(args)
    sizes = _asset_size_checks(paths)
    checks = {
        "required_paths_present": not missing,
        "ffmpeg_present": bool(ffmpeg),
        "ffprobe_present": bool(ffprobe),
        "reviewed_asset_sizes_match": all(sizes.values()),
        **start_gate["checks"],
    }
    ready = all(checks.values())
    if ready:
        status = "READY"
    elif missing or not ffmpeg or not ffprobe:
        status = "ABSTAIN_MISSING_DEPENDENCY"
    elif not all(sizes.values()):
        status = "ABSTAIN_ASSET_SIZE_MISMATCH"
    elif not checks["target_port_free"]:
        status = "ABSTAIN_TARGET_PORT_BUSY"
    elif not checks["gpu_query_available"]:
        status = "ABSTAIN_GPU_STATE_UNKNOWN"
    else:
        status = "ABSTAIN_INSUFFICIENT_FREE_VRAM"
    return {
        "schema": f"{SCHEMA}.preflight",
        "created_at": _utc_now(),
        "status": status,
        "ready_for_real_run": ready,
        "checks": checks,
        "missing_paths": missing,
        "reviewed_asset_size_checks": sizes,
        "model_identity": _identity_manifest(paths),
        "gpu": start_gate["gpu"],
        "minimum_free_vram_mib": args.min_free_vram_mib,
        "target": start_gate["target"],
        "user_service_8188_observed_only": shared.port_is_listening(args.host, 8188),
        "ffmpeg": ffmpeg,
        "ffprobe": ffprobe,
        "boundary": (
            "Preflight performs no model load and no full-file hash. Port 8188 is observation-only; "
            "the tool never interrupts, unloads, or terminates it."
        ),
    }


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


def media_contract_checks(report: Mapping[str, Any]) -> dict[str, bool]:
    streams = report.get("probe", {}).get("streams", [])
    video = [stream for stream in streams if stream.get("codec_type") == "video"]
    audio = [stream for stream in streams if stream.get("codec_type") == "audio"]
    decoded_video = report.get("decoded_video", {})
    decoded_audio = report.get("decoded_audio", {})
    expected_video_bytes = 22 * 256 * 256 * 3
    return {
        "strict_video_audio_combined_decode": bool(report.get("strict_decode_passed")),
        "one_video_stream": len(video) == 1,
        "video_h264_256x256": len(video) == 1
        and video[0].get("codec_name") == "h264"
        and int(video[0].get("width") or 0) == 256
        and int(video[0].get("height") or 0) == 256,
        "decoded_video_exactly_22_frames": int(decoded_video.get("bytes") or 0)
        == expected_video_bytes,
        "one_audio_stream": len(audio) == 1,
        "audio_aac_32khz_stereo": len(audio) == 1
        and audio[0].get("codec_name") == "aac"
        and int(audio[0].get("sample_rate") or 0) == 32_000
        and int(audio[0].get("channels") or 0) == 2,
        "decoded_audio_nonempty": int(decoded_audio.get("bytes") or 0) > 0,
    }


def _write_report(run_root: Path, report: Mapping[str, Any]) -> dict[str, Any]:
    path = run_root / "validation_report.json"
    path.write_text(json.dumps(dict(report), ensure_ascii=False, indent=2), encoding="utf-8")
    return dict(report)


def run_real_probe(args: argparse.Namespace, preflight_report: Mapping[str, Any]) -> dict[str, Any]:
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    run_root = (args.artifact_root / run_id).resolve()
    run_root.mkdir(parents=True, exist_ok=False)
    paths = _model_paths(args.comfy_root.resolve())

    verified_assets = {
        role: shared._sha256_file(paths[role]) for role in sorted(EXPECTED_ASSETS)
    }
    asset_checks = {
        role: verified_assets[role] == str(EXPECTED_ASSETS[role]["sha256"])
        for role in EXPECTED_ASSETS
    }
    asset_manifest = {
        "hash_scope": "full_sha256_clip_and_projection_only",
        "assets": {
            role: {
                **EXPECTED_ASSETS[role],
                "path": str(paths[role].resolve()),
                "actual_sha256": verified_assets[role],
                "matched": asset_checks[role],
            }
            for role in sorted(EXPECTED_ASSETS)
        },
        "other_weights": _identity_manifest(
            {role: path for role, path in paths.items() if role not in EXPECTED_ASSETS}
        ),
    }
    (run_root / "asset_manifest.json").write_text(
        json.dumps(asset_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not all(asset_checks.values()):
        return _write_report(
            run_root,
            {
                "schema": SCHEMA,
                "created_at": _utc_now(),
                "run_id": run_id,
                "run_root": str(run_root),
                "status": "ABSTAIN_ASSET_HASH_MISMATCH",
                "passed": False,
                "preflight": dict(preflight_report),
                "asset_manifest": asset_manifest,
                "process_ids": [],
                "checks": {"no_isolated_server_started": True, **asset_checks},
            },
        )

    post_hash_gate = shared.isolated_start_gate(args)
    if not post_hash_gate["ready"]:
        return _write_report(
            run_root,
            {
                "schema": SCHEMA,
                "created_at": _utc_now(),
                "run_id": run_id,
                "run_root": str(run_root),
                "status": "ABSTAIN_RESOURCE_CHANGED_AFTER_ASSET_HASH",
                "passed": False,
                "preflight": dict(preflight_report),
                "asset_manifest": asset_manifest,
                "post_hash_start_gate": post_hash_gate,
                "process_ids": [],
                "checks": {"no_isolated_server_started": True},
            },
        )

    prompt = build_prompt(run_id=run_id, seed=args.seed)
    (run_root / "prompt.json").write_text(
        json.dumps(prompt, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    baseline_gpu = post_hash_gate["gpu"]
    monitor = GpuPeakMonitor()
    server = shared.IsolatedServer(args, run_root, "clipproj_4b")
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
    except Exception as error:  # report the isolated failure without hiding its type
        runtime_error = {"type": type(error).__name__, "message": str(error)}
    finally:
        server.stop()
        gpu_monitor = monitor.stop()

    final_gpu = shared._wait_gpu_return(int(baseline_gpu.get("used_mib", 0)) + 512)
    execution_success = bool(
        phase and phase.get("terminal", {}).get("type") == "execution_success"
    )
    media: dict[str, Any] | None = None
    media_checks: dict[str, bool] = {}
    if execution_success:
        try:
            media_root = run_root / "output" / "MiniMaxH3_ClipProj_4B_Probe"
            media_path = shared._latest_file(media_root, f"{run_id}*.mp4")
            media = shared.media_report(
                media_path,
                ffmpeg=str(preflight_report["ffmpeg"]),
                ffprobe=str(preflight_report["ffprobe"]),
            )
            media_checks = media_contract_checks(media)
        except Exception as error:  # preserve the generation result and report media failure
            runtime_error = {"type": type(error).__name__, "message": str(error)}

    checks = {
        "official_4b_encoder_sha256": asset_checks["clip"],
        "official_4b_projection_sha256": asset_checks["projection"],
        "one_isolated_process": len(process_ids) == 1,
        "execution_success": execution_success,
        **media_checks,
    }
    passed = bool(media_checks) and all(checks.values()) and runtime_error is None
    return _write_report(
        run_root,
        {
            "schema": SCHEMA,
            "created_at": _utc_now(),
            "run_id": run_id,
            "run_root": str(run_root),
            "status": "PASS" if passed else "FAIL_RUNTIME_OR_MEDIA_CONTRACT",
            "passed": passed,
            "preflight": dict(preflight_report),
            "asset_manifest": asset_manifest,
            "post_hash_start_gate": post_hash_gate,
            "process_ids": process_ids,
            "phase": phase,
            "runtime_error": runtime_error,
            "generation_contract": {
                "prompt": prompt["8"]["inputs"]["prompt"],
                "seed": args.seed,
                "width": 256,
                "height": 256,
                "frame_count": 22,
                "steps": 4,
                "shift_video": 12.0,
                "shift_audio": 3.0,
            },
            "media": media,
            "gpu": {
                "baseline": baseline_gpu,
                "monitor": gpu_monitor,
                "final": final_gpu,
            },
            "elapsed_seconds": round(time.monotonic() - started, 4),
            "checks": checks,
            "boundary": (
                "One fixed 256x256x22 T2VA, four-NFE ClipProj 4B mechanical run. A PASS proves "
                "only this asset and runtime chain. It does not prove 32B quality equivalence, "
                "prompt fidelity, speedup, VRAM savings, repeated stability, or universal 16GB safety."
            ),
        },
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comfy-root", type=Path, default=Path(r"F:\AI-T8-video-onekey\ComfyUI")
    )
    parser.add_argument(
        "--python", type=Path, default=Path(r"F:\AI-T8-video-onekey\python\python.exe")
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "artifacts"
        / "clipproj-4b-real-runtime-v1",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8197)
    parser.add_argument("--min-free-vram-mib", type=int, default=12000)
    parser.add_argument("--server-start-timeout", type=float, default=180.0)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument(
        "--seed",
        type=int,
        default=123456789,
        help=(
            "Noise seed for the single guarded run. Keep the default for the original probe; "
            "set it explicitly only when building a registered same-input comparison."
        ),
    )
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument(
        "--confirm-run",
        action="store_true",
        help="Hash the reviewed 4B assets and run one isolated GPU task after all gates pass.",
    )
    return parser.parse_args(argv)


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
                    "preflight": str(preflight_path.resolve()),
                    "real_run_started": False,
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
                    "preflight": str(preflight_path.resolve()),
                    "real_run_started": False,
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
