#!/usr/bin/env python3
"""Run one guarded clear-material H3 T2VA arm for a 4B/8B/native-32B triplet.

The historical comparison used a dark 256x256 rain scene and four NFE, which the reviewer could
not assess reliably. This follow-up keeps the already proven low-load 256x256x22 geometry but uses
one bright, high-contrast, frame-filling subject and eight NFE for every arm. Each invocation runs
exactly one explicitly selected arm in an isolated ComfyUI process. It refuses an active user
service on 8188, a busy private port, missing assets, or insufficient free VRAM.

This tool never changes the stable sampler, never touches the user's 8188/11434 processes and does
not call unload APIs on any process it did not start. A passing arm proves only mechanical media
generation for the fixed contract; perceptual conclusions require the keyed blind package.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import threading
import time
from typing import Any, Mapping
import uuid

import run_nfe_resume_real_probe as shared


SCHEMA = "t8.minimax_h3.clear_clipproj_triplet_probe.v1"
ARMS = ("clipproj_4b", "clipproj_8b", "native_32b")
CLEAR_PROMPT = (
    "Bright daylight studio macro close-up of a glossy red mechanical metronome filling most "
    "of the frame against a clean white background. Its black pendulum swings steadily left "
    "and right; engraved scale marks, sharp metal edges and reflections remain crisp. Clear "
    "synchronized wooden ticking, quiet studio room tone, no music, no text overlay."
)
SEED = 2608241001
WIDTH = 256
HEIGHT = 256
FRAME_COUNT = 22
STEPS = 8
SHIFT_VIDEO = 12.0
SHIFT_AUDIO = 3.0

BASE_NAME = "minimax_h3_fl2va_int8_convrot.safetensors"
LORA_NAME = "minimax_h3_turbo_4步加速ema_comfyui.safetensors"
VIDEO_VAE_NAME = "minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE_NAME = "minimax_h3_audio_vae_fp32.safetensors"
ARM_ASSETS = {
    "clipproj_4b": {
        "clip_name": "qwen3vl_4b_fp8_scaled.safetensors",
        "clip_type": "krea2",
        "projection_name": "mmh3-4b-ClipProj-v3.1.safetensors",
        "encoder_family": "4B",
        "clip_bytes": 5_242_467_968,
        "clip_sha256": "54BD5144DF0BBC25DD6CCADFCB826B521445A1B06AE5A42570BDD2974CA87094",
        "projection_bytes": 26_256_128,
        "projection_sha256": "0184E5C8D666A131962506D21949C2D8A8C6F33445B7B5E347E9A7E0A5BAA819",
        "default_min_free_vram_mib": 12_500,
    },
    "clipproj_8b": {
        "clip_name": "qwen3vl_8b_fp8_scaled.safetensors",
        "clip_type": "boogu",
        "projection_name": "mmh3-8b-ClipProj-v3.1.safetensors",
        "encoder_family": "8B",
        "clip_bytes": 10_588_637_512,
        "clip_sha256": "4BA424CF62E51392E4D1A39933E803706F4E823C1065F36AAF149C6453F66BCD",
        "projection_bytes": 41_990_896,
        "projection_sha256": "DF0661849D0FD51DB66B0C9AA76F2C1C3EABD81B9A4745EDD2A4617AB24C87F7",
        "default_min_free_vram_mib": 13_000,
    },
    "native_32b": {
        "clip_name": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        "clip_type": "minimax",
        "projection_name": None,
        "encoder_family": "32B",
        "clip_bytes": 15_687_142_551,
        "clip_sha256": None,
        "projection_bytes": None,
        "projection_sha256": None,
        # Earlier fixed native runs needed 13,971.225 MiB above their baseline.
        # Adding the established 512 MiB minimum headroom and rounding up gives
        # the 14,500 MiB floor. Do not gamble on allocator or desktop eviction.
        "default_min_free_vram_mib": 14_500,
        "free_vram_gate_basis": {
            "evidence": "artifacts/clipproj-4b-vs-8b-32b-review-v1/objective_analysis.json",
            "observed_baseline_used_mib": 1_156.5,
            "observed_peak_used_mib": 15_127.724876403809,
            "observed_incremental_used_mib": 13_971.224876403809,
            "required_remaining_headroom_mib": 512,
            "unrounded_required_free_mib": 14_483.224876403809,
            "enforced_rounded_floor_mib": 14_500,
        },
    },
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _model_paths(comfy_root: Path, arm: str) -> dict[str, Path]:
    if arm not in ARM_ASSETS:
        raise ValueError(f"arm must be one of {ARMS}")
    models = comfy_root / "models"
    spec = ARM_ASSETS[arm]
    paths = {
        "base": models / "diffusion_models" / BASE_NAME,
        "lora": models / "loras" / LORA_NAME,
        "clip": models / "text_encoders" / str(spec["clip_name"]),
        "video_vae": models / "vae" / VIDEO_VAE_NAME,
        "audio_vae": models / "vae" / AUDIO_VAE_NAME,
    }
    if spec["projection_name"]:
        paths["projection"] = (
            models / "clip_projections" / str(spec["projection_name"])
        )
    return paths


def build_prompt(*, arm: str, run_id: str) -> dict[str, Any]:
    """Build the fixed same-input graph while changing only the text-encoder arm."""

    if arm not in ARM_ASSETS:
        raise ValueError(f"arm must be one of {ARMS}")
    spec = ARM_ASSETS[arm]
    clip_nodes: dict[str, Any]
    conditioning_clip: list[Any]
    if arm == "native_32b":
        clip_nodes = {
            "3": {
                "inputs": {
                    "clip_name": spec["clip_name"],
                    "type": spec["clip_type"],
                    "device": "default",
                },
                "class_type": "CLIPLoader",
            }
        }
        conditioning_clip = ["3", 0]
    else:
        clip_nodes = {
            "3": {
                "inputs": {
                    "clip_name": spec["clip_name"],
                    "type": spec["clip_type"],
                    "device": "default",
                },
                "class_type": "CLIPLoader",
            },
            "4": {
                "inputs": {
                    "clip": ["3", 0],
                    "projection": spec["projection_name"],
                },
                "class_type": "ClipProjApply",
            },
            "5": {
                "inputs": {
                    "clip": ["4", 0],
                    "encoder_family": spec["encoder_family"],
                    "encoder_architecture": "qwen3_vl",
                    "encoder_quantization": "fp8",
                    "load_mode": "stock_pageable",
                    "projection_path": spec["projection_name"],
                    "has_reference_images": False,
                    "has_reference_videos": False,
                    "enforcement": "block_hard_conflicts",
                },
                "class_type": "MiniMaxH3ClipProjCompatibilityAuditT8Advanced",
            },
        }
        conditioning_clip = ["5", 0]

    prompt = {
        "1": {"inputs": {"vae_name": VIDEO_VAE_NAME}, "class_type": "VAELoader"},
        "2": {"inputs": {"vae_name": AUDIO_VAE_NAME}, "class_type": "VAELoader"},
        **clip_nodes,
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
                "prompt": CLEAR_PROMPT,
                "width": WIDTH,
                "height": HEIGHT,
                "length": FRAME_COUNT,
                "task_type": "T2VA",
                "audio_mode": "native",
                "audio_denoise_strength": 1.0,
                "add_source_as_reference": False,
                "prompt_primary_audio_ordinal": 0,
                "strict_prompt_tags": True,
                "ref_image_size": "match",
                "reference_video_policy": "official_2_to_15s",
                "clip": conditioning_clip,
                "video_vae": ["1", 0],
                "audio_vae": ["2", 0],
            },
            "class_type": "MiniMaxH3AudioConditioningT8",
        },
        "9": {
            "inputs": {
                "model": ["7", 0],
                "av_latent": ["8", 1],
                "steps": STEPS,
                "shift_video": SHIFT_VIDEO,
                "shift_audio": SHIFT_AUDIO,
            },
            "class_type": "MiniMaxH3DualClockSamplerT8",
        },
        "10": {"inputs": {"noise_seed": SEED}, "class_type": "RandomNoise"},
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
                "filename_prefix": f"MiniMaxH3_ClearTriplet/{run_id}_{arm}",
                "format": "video/h264-mp4",
                "pix_fmt": "yuv420p",
                "crf": 18,
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
    return prompt


def _asset_size_checks(paths: Mapping[str, Path], arm: str) -> dict[str, bool]:
    spec = ARM_ASSETS[arm]
    checks = {
        "clip": paths["clip"].is_file()
        and paths["clip"].stat().st_size == int(spec["clip_bytes"])
    }
    if "projection" in paths:
        checks["projection"] = paths["projection"].is_file() and (
            paths["projection"].stat().st_size == int(spec["projection_bytes"])
        )
    return checks


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    comfy_root = args.comfy_root.resolve()
    paths = _model_paths(comfy_root, args.arm)
    required = {
        "comfy_main": comfy_root / "main.py",
        "python": args.python.resolve(),
        "t8_nodes": comfy_root / "custom_nodes" / "minimax-h3-audio-T8",
        "vhs_nodes": comfy_root / "custom_nodes" / "ComfyUI-VideoHelperSuite",
        **paths,
    }
    if args.arm != "native_32b":
        required["clipproj_nodes"] = comfy_root / "custom_nodes" / "ComfyUI-ClipProj"
    missing = [str(path) for path in required.values() if not path.exists()]
    size_checks = _asset_size_checks(paths, args.arm)
    ffmpeg = shutil.which(args.ffmpeg) or (
        args.ffmpeg if Path(args.ffmpeg).is_file() else None
    )
    ffprobe = shutil.which(args.ffprobe) or (
        args.ffprobe if Path(args.ffprobe).is_file() else None
    )
    gpu = shared.gpu_memory_mib()
    user_service = shared.port_is_listening(args.host, 8188)
    target_busy = shared.port_is_listening(args.host, args.port)
    checks = {
        "required_paths_present": not missing,
        "reviewed_asset_sizes_match": all(size_checks.values()),
        "ffmpeg_present": bool(ffmpeg),
        "ffprobe_present": bool(ffprobe),
        "user_service_8188_inactive": not user_service,
        "target_port_free": not target_busy,
        "gpu_query_available": bool(gpu.get("available")),
        "free_vram_gate": bool(
            gpu.get("available")
            and int(gpu.get("free_mib", 0)) >= args.min_free_vram_mib
        ),
    }
    ready = all(checks.values())
    if ready:
        status = "READY"
    elif user_service:
        status = "ABSTAIN_USER_SERVICE_8188_ACTIVE"
    elif missing:
        status = "ABSTAIN_MISSING_DEPENDENCY"
    elif not all(size_checks.values()):
        status = "ABSTAIN_ASSET_SIZE_MISMATCH"
    elif target_busy:
        status = "ABSTAIN_TARGET_PORT_BUSY"
    elif not gpu.get("available"):
        status = "ABSTAIN_GPU_STATE_UNKNOWN"
    else:
        status = "ABSTAIN_INSUFFICIENT_FREE_VRAM"
    return {
        "schema": f"{SCHEMA}.preflight",
        "created_at": _utc_now(),
        "arm": args.arm,
        "status": status,
        "ready_for_real_run": ready,
        "checks": checks,
        "missing_paths": missing,
        "reviewed_asset_size_checks": size_checks,
        "gpu": gpu,
        "minimum_free_vram_mib": args.min_free_vram_mib,
        "free_vram_gate_basis": ARM_ASSETS[args.arm].get(
            "free_vram_gate_basis",
            {
                "enforced_rounded_floor_mib": int(
                    ARM_ASSETS[args.arm]["default_min_free_vram_mib"]
                ),
                "evidence": "arm-specific conservative floor",
            },
        ),
        "target": {"host": args.host, "port": args.port, "already_listening": target_busy},
        "user_service_8188_observed_only": user_service,
        "ffmpeg": ffmpeg,
        "ffprobe": ffprobe,
        "contract": {
            "prompt": CLEAR_PROMPT,
            "seed": SEED,
            "width": WIDTH,
            "height": HEIGHT,
            "frame_count": FRAME_COUNT,
            "steps": STEPS,
            "shift_video": SHIFT_VIDEO,
            "shift_audio": SHIFT_AUDIO,
        },
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


def _media_checks(report: Mapping[str, Any]) -> dict[str, bool]:
    streams = report.get("probe", {}).get("streams", [])
    video = [value for value in streams if value.get("codec_type") == "video"]
    audio = [value for value in streams if value.get("codec_type") == "audio"]
    return {
        "strict_decode": bool(report.get("strict_decode_passed")),
        "video_h264_256x256": len(video) == 1
        and video[0].get("codec_name") == "h264"
        and int(video[0].get("width") or 0) == WIDTH
        and int(video[0].get("height") or 0) == HEIGHT,
        "decoded_video_exact_frames": int(
            report.get("decoded_video", {}).get("bytes") or 0
        )
        == FRAME_COUNT * WIDTH * HEIGHT * 3,
        "audio_aac_32khz_stereo": len(audio) == 1
        and audio[0].get("codec_name") == "aac"
        and int(audio[0].get("sample_rate") or 0) == 32_000
        and int(audio[0].get("channels") or 0) == 2,
        "decoded_audio_nonempty": int(
            report.get("decoded_audio", {}).get("bytes") or 0
        )
        > 0,
    }


def _write_report(run_root: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    (run_root / "validation_report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def run_real_probe(
    args: argparse.Namespace, preflight_report: Mapping[str, Any]
) -> dict[str, Any]:
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    run_root = (args.artifact_root / args.arm / run_id).resolve()
    run_root.mkdir(parents=True, exist_ok=False)
    paths = _model_paths(args.comfy_root.resolve(), args.arm)
    spec = ARM_ASSETS[args.arm]

    hashes = {"clip": shared._sha256_file(paths["clip"])}
    if "projection" in paths:
        hashes["projection"] = shared._sha256_file(paths["projection"])
    hash_checks = {
        "clip": spec["clip_sha256"] is None
        or hashes["clip"] == str(spec["clip_sha256"])
    }
    if "projection" in hashes:
        hash_checks["projection"] = hashes["projection"] == str(
            spec["projection_sha256"]
        )
    asset_manifest = {
        "arm": args.arm,
        "full_sha256": hashes,
        "expected_sha256": {
            "clip": spec["clip_sha256"],
            "projection": spec["projection_sha256"],
        },
        "matched": hash_checks,
        "shared_weight_identity": {
            role: {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            }
            for role, path in paths.items()
            if role not in {"clip", "projection"}
        },
    }
    (run_root / "asset_manifest.json").write_text(
        json.dumps(asset_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not all(hash_checks.values()):
        return _write_report(
            run_root,
            {
                "schema": SCHEMA,
                "status": "ABSTAIN_ASSET_HASH_MISMATCH",
                "passed": False,
                "run_root": str(run_root),
                "preflight": dict(preflight_report),
                "asset_manifest": asset_manifest,
            },
        )

    post_hash_gate = preflight(args)
    if not post_hash_gate["ready_for_real_run"]:
        return _write_report(
            run_root,
            {
                "schema": SCHEMA,
                "status": "ABSTAIN_RESOURCE_CHANGED_AFTER_ASSET_HASH",
                "passed": False,
                "run_root": str(run_root),
                "preflight": dict(preflight_report),
                "post_hash_preflight": post_hash_gate,
                "asset_manifest": asset_manifest,
            },
        )

    prompt = build_prompt(arm=args.arm, run_id=run_id)
    (run_root / "prompt.json").write_text(
        json.dumps(prompt, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    baseline_gpu = dict(post_hash_gate["gpu"])
    server = shared.IsolatedServer(args, run_root, f"clear_triplet_{args.arm}")
    monitor = GpuPeakMonitor()
    process_ids: list[int] = []
    phase = None
    runtime_error = None
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
        gpu_monitor = monitor.stop()
    final_gpu = shared._wait_gpu_return(int(baseline_gpu.get("used_mib", 0)) + 512)

    success = bool(phase and phase.get("terminal", {}).get("type") == "execution_success")
    media = None
    media_checks: dict[str, bool] = {}
    if success:
        try:
            media_path = shared._latest_file(
                run_root / "output" / "MiniMaxH3_ClearTriplet",
                f"{run_id}_{args.arm}*.mp4",
            )
            media = shared.media_report(
                media_path,
                ffmpeg=str(post_hash_gate["ffmpeg"]),
                ffprobe=str(post_hash_gate["ffprobe"]),
            )
            media_checks = _media_checks(media)
        except Exception as error:
            runtime_error = {"type": type(error).__name__, "message": str(error)}
    checks = {
        "asset_hash_contract": all(hash_checks.values()),
        "one_isolated_process": len(process_ids) == 1,
        "execution_success": success,
        **media_checks,
    }
    passed = bool(media_checks) and all(checks.values()) and runtime_error is None
    return _write_report(
        run_root,
        {
            "schema": SCHEMA,
            "created_at": _utc_now(),
            "arm": args.arm,
            "status": "PASS" if passed else "FAIL_RUNTIME_OR_MEDIA_CONTRACT",
            "passed": passed,
            "run_root": str(run_root),
            "contract": post_hash_gate["contract"],
            "asset_manifest": asset_manifest,
            "preflight": dict(preflight_report),
            "post_hash_preflight": post_hash_gate,
            "process_ids": process_ids,
            "prompt_to_terminal_seconds": round(time.monotonic() - started, 3),
            "phase": phase,
            "gpu_monitor": gpu_monitor,
            "baseline_gpu": baseline_gpu,
            "final_gpu": final_gpu,
            "media": media,
            "checks": checks,
            "runtime_error": runtime_error,
            "boundary": (
                "One fixed bright 256x256x22 eight-NFE T2VA arm. PASS does not establish "
                "quality equivalence, preference, speed, memory superiority or general 16GB safety."
            ),
        },
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
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
        / "clipproj-clear-triplet-runtime-v1",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8197)
    parser.add_argument("--min-free-vram-mib", type=int)
    parser.add_argument("--server-start-timeout", type=float, default=180.0)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args(argv)
    arm_floor = int(ARM_ASSETS[args.arm]["default_min_free_vram_mib"])
    if args.min_free_vram_mib is None:
        args.min_free_vram_mib = arm_floor
    if args.min_free_vram_mib < arm_floor:
        parser.error(
            f"--min-free-vram-mib cannot be lower than the reviewed {args.arm} "
            f"floor ({arm_floor} MiB)"
        )
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    report = preflight(args)
    preflight_path = args.artifact_root / f"latest_preflight_{args.arm}.json"
    preflight_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not args.confirm_run:
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "arm": args.arm,
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
                    "arm": args.arm,
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
                "arm": args.arm,
                "passed": result["passed"],
                "run_root": result["run_root"],
            },
            ensure_ascii=False,
        )
    )
    if str(result["status"]).startswith("ABSTAIN"):
        return 3
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
