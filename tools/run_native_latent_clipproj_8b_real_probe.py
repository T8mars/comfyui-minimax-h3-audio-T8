#!/usr/bin/env python3
"""Run one guarded 8B ClipProj native-latent two-segment H3 probe.

The default invocation is preflight-only. A real run requires ``--confirm-run`` and is refused
while the user's normal ComfyUI service on port 8188 is active, while the private port is busy, or
when the selected GPU has less than the requested free-VRAM gate. The reviewed 8B encoder and
projection are fully SHA-256 checked before the private ComfyUI process starts, followed by a
second port/GPU gate.

The graph preserves the previously reviewed 256x256, 22+22 -> 39-frame, eight-NFE, 12/3 dual-clock
AV contract. It changes only the text path from native 32B to the separately audited 8B ClipProj
bridge and decodes the combined latent exactly once. A PASS is limited to this fixed local route;
it is not a quality, seamless-continuation, generic 16GB, or long-video claim.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import threading
import time
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import run_nfe_resume_real_probe as shared

SCHEMA = "t8.minimax_h3.native_latent_clipproj_8b_real_probe.v1"
CLIP_NAME = "qwen3vl_8b_fp8_scaled.safetensors"
PROJECTION_NAME = "mmh3-8b-ClipProj-v3.1.safetensors"
BASE_NAME = "minimax_h3_fl2va_int8_convrot.safetensors"
LORA_NAME = "minimax_h3_turbo_4步加速ema_comfyui.safetensors"
VIDEO_VAE_NAME = "minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE_NAME = "minimax_h3_audio_vae_fp32.safetensors"
EXPECTED_ASSETS = {
    "clip": {
        "bytes": 10_588_637_512,
        "sha256": "4BA424CF62E51392E4D1A39933E803706F4E823C1065F36AAF149C6453F66BCD",
        "identity": "reviewed local Qwen3-VL 8B FP8 asset",
    },
    "projection": {
        "bytes": 41_990_896,
        "sha256": "DF0661849D0FD51DB66B0C9AA76F2C1C3EABD81B9A4745EDD2A4617AB24C87F7",
        "identity": "NicoLab28 MiniMax H3 8B ClipProj v3.1",
    },
}
SOURCE_32B_GRAPH_SHA256 = "F827EAE666D3590F86C1A5A36FCBBE704C760FF7DF3F052EE5EE23C3283CA7B4"
EXPECTED_FRAMES = 39
EXPECTED_LOSSLESS_AUDIO_SAMPLES = 52_000
EXPECTED_AAC_DECODED_SAMPLES = 51_200
SOURCE_32B_COMBINED_MP4_SHA256 = (
    "55BB1F7FE41B7F1AF892C47E15B14F019AE7735F34CCE9906987113F25019110"
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


def build_prompt(*, run_id: str) -> dict[str, Any]:
    """Build the fixed 8B two-segment graph without changing the old 32B proof."""

    common_conditioning = {
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
    }
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
                **common_conditioning,
                "prompt": (
                    "A cinematic close-up of steady rain striking a dark metal roof at night. "
                    "Natural synchronized rain impacts and distant thunder, no music."
                ),
            },
            "class_type": "MiniMaxH3AudioConditioningT8",
        },
        "9": {
            "inputs": {
                "steps": 8,
                "shift_video": 12.0,
                "shift_audio": 3.0,
                "model": ["7", 0],
                "av_latent": ["8", 1],
            },
            "class_type": "MiniMaxH3DualClockSamplerT8",
        },
        "10": {"inputs": {"noise_seed": 2608229101}, "class_type": "RandomNoise"},
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
        "18": {
            "inputs": {
                **common_conditioning,
                "prompt": (
                    "The same dark metal roof moments later under continuous night rain. Water "
                    "streams over the ridges while natural rain impacts and distant thunder "
                    "continue, no music."
                ),
            },
            "class_type": "MiniMaxH3AudioConditioningT8",
        },
        "19": {
            "inputs": {
                "steps": 8,
                "shift_video": 12.0,
                "shift_audio": 3.0,
                "model": ["7", 0],
                "av_latent": ["18", 1],
            },
            "class_type": "MiniMaxH3DualClockSamplerT8",
        },
        "20": {"inputs": {"noise_seed": 2608229102}, "class_type": "RandomNoise"},
        "21": {
            "inputs": {"model": ["19", 0], "conditioning": ["18", 0]},
            "class_type": "BasicGuider",
        },
        "22": {
            "inputs": {
                "noise": ["20", 0],
                "guider": ["21", 0],
                "sampler": ["19", 1],
                "sigmas": ["19", 2],
                "latent_image": ["18", 1],
            },
            "class_type": "SamplerCustomAdvanced",
        },
        "23": {
            "inputs": {
                "first_segment": ["12", 0],
                "second_segment": ["22", 0],
                "output_device": "cpu",
                "require_identical_metadata": False,
            },
            "class_type": "MiniMaxH3NativeLatentTimelineConcatT8Advanced",
        },
        "24": {
            "inputs": {
                "av_latent": ["23", 0],
                "video_vae": ["1", 0],
                "audio_vae": ["2", 0],
            },
            "class_type": "MiniMaxH3AVDecodeT8",
        },
        "25": {
            "inputs": {
                "frame_rate": 24,
                "loop_count": 0,
                "filename_prefix": f"MiniMaxH3_NativeLatent_ClipProj8B_Probe/{run_id}",
                "format": "video/h264-mp4",
                "pix_fmt": "yuv420p",
                "crf": 19,
                "save_metadata": False,
                "trim_to_audio": False,
                "pingpong": False,
                "save_output": True,
                "images": ["24", 0],
                "audio": ["24", 1],
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


def _start_gate(args: argparse.Namespace) -> dict[str, Any]:
    gate = shared.isolated_start_gate(args)
    user_service_active = shared.port_is_listening(args.host, 8188)
    checks = {
        **gate["checks"],
        "user_service_8188_inactive": not user_service_active,
    }
    return {
        **gate,
        "checks": checks,
        "ready": all(checks.values()),
        "user_service_8188_observed_only": user_service_active,
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
    start_gate = _start_gate(args)
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
    elif start_gate["user_service_8188_observed_only"]:
        status = "ABSTAIN_USER_SERVICE_8188_ACTIVE"
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
        "safety_headroom_mib": args.safety_headroom_mib,
        "target": start_gate["target"],
        "user_service_8188_observed_only": start_gate["user_service_8188_observed_only"],
        "ffmpeg": ffmpeg,
        "ffprobe": ffprobe,
        "boundary": (
            "Preflight loads no model and hashes no large file. Port 8188 is observation-only; "
            "an active user service causes ABSTAIN and is never interrupted or unloaded."
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
    return {
        "strict_video_audio_combined_decode": bool(report.get("strict_decode_passed")),
        "one_video_stream": len(video) == 1,
        "video_h264_256x256": len(video) == 1
        and video[0].get("codec_name") == "h264"
        and int(video[0].get("width") or 0) == 256
        and int(video[0].get("height") or 0) == 256,
        "decoded_video_exactly_39_frames": int(decoded_video.get("bytes") or 0)
        == EXPECTED_FRAMES * 256 * 256 * 3,
        "one_audio_stream": len(audio) == 1,
        "audio_aac_32khz_stereo": len(audio) == 1
        and audio[0].get("codec_name") == "aac"
        and int(audio[0].get("sample_rate") or 0) == 32_000
        and int(audio[0].get("channels") or 0) == 2,
        "decoded_aac_audio_matches_32b_reference_51200_samples": int(
            decoded_audio.get("bytes") or 0
        )
        == EXPECTED_AAC_DECODED_SAMPLES * 2 * 4,
    }


def _write_report(run_root: Path, report: Mapping[str, Any]) -> dict[str, Any]:
    path = run_root / "validation_report.json"
    path.write_text(json.dumps(dict(report), ensure_ascii=False, indent=2), encoding="utf-8")
    return dict(report)


def run_real_probe(args: argparse.Namespace, preflight_report: Mapping[str, Any]) -> dict[str, Any]:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
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
        "hash_scope": "full_sha256_reviewed_8b_clip_and_projection",
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

    post_hash_gate = _start_gate(args)
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

    prompt = build_prompt(run_id=run_id)
    (run_root / "prompt.json").write_text(
        json.dumps(prompt, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    baseline_gpu = post_hash_gate["gpu"]
    monitor = GpuPeakMonitor()
    server = shared.IsolatedServer(args, run_root, "native_latent_clipproj_8b")
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
    except Exception as error:  # noqa: BLE001 - preserve isolated runtime failure type in report
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
            media_root = run_root / "output" / "MiniMaxH3_NativeLatent_ClipProj8B_Probe"
            media_path = shared._latest_file(media_root, f"{run_id}*.mp4")
            media = shared.media_report(
                media_path,
                ffmpeg=str(preflight_report["ffmpeg"]),
                ffprobe=str(preflight_report["ffprobe"]),
            )
            media_checks = media_contract_checks(media)
        except Exception as error:  # noqa: BLE001 - keep generated result and report media failure
            runtime_error = {"type": type(error).__name__, "message": str(error)}

    minimum_free = gpu_monitor.get("minimum_free_mib")
    mechanical_checks = {
        "reviewed_8b_encoder_sha256": asset_checks["clip"],
        "reviewed_8b_projection_sha256": asset_checks["projection"],
        "one_isolated_process": len(process_ids) == 1,
        "execution_success": execution_success,
        **media_checks,
    }
    mechanical_passed = bool(media_checks) and all(mechanical_checks.values()) and runtime_error is None
    headroom_passed = bool(
        minimum_free is not None and int(minimum_free) >= int(args.safety_headroom_mib)
    )
    final_return_passed = bool(
        final_gpu.get("available")
        and int(final_gpu.get("used_mib", 0)) <= int(baseline_gpu.get("used_mib", 0)) + 512
    )
    passed = mechanical_passed and headroom_passed and final_return_passed
    if passed:
        status = "PASS"
    elif mechanical_passed and not headroom_passed:
        status = "FAIL_MEMORY_HEADROOM_GATE"
    elif mechanical_passed and not final_return_passed:
        status = "FAIL_POST_RUN_GPU_RETURN_GATE"
    else:
        status = "FAIL_RUNTIME_OR_MEDIA_CONTRACT"
    return _write_report(
        run_root,
        {
            "schema": SCHEMA,
            "created_at": _utc_now(),
            "run_id": run_id,
            "run_root": str(run_root),
            "status": status,
            "passed": passed,
            "mechanical_passed": mechanical_passed,
            "memory_headroom_gate_passed": headroom_passed,
            "post_run_gpu_return_gate_passed": final_return_passed,
            "preflight": dict(preflight_report),
            "asset_manifest": asset_manifest,
            "post_hash_start_gate": post_hash_gate,
            "process_ids": process_ids,
            "phase": phase,
            "runtime_error": runtime_error,
            "generation_contract": {
                "source_32b_graph_sha256": SOURCE_32B_GRAPH_SHA256,
                "changed_variable": "native_32b_text_path -> reviewed_8b_clipproj_text_path",
                "prompts": [prompt["8"]["inputs"]["prompt"], prompt["18"]["inputs"]["prompt"]],
                "seeds": [2608229101, 2608229102],
                "width": 256,
                "height": 256,
                "segment_frame_counts": [22, 22],
                "combined_frame_count": EXPECTED_FRAMES,
                "lossless_audio_samples_expected_but_not_measured": (
                    EXPECTED_LOSSLESS_AUDIO_SAMPLES
                ),
                "aac_decoded_audio_samples": EXPECTED_AAC_DECODED_SAMPLES,
                "source_32b_combined_mp4_sha256": SOURCE_32B_COMBINED_MP4_SHA256,
                "steps_per_segment": 8,
                "shift_video": 12.0,
                "shift_audio": 3.0,
                "concat_output_device": "cpu",
                "decode_count": 1,
            },
            "media": media,
            "gpu": {
                "baseline": baseline_gpu,
                "monitor": gpu_monitor,
                "final": final_gpu,
                "safety_headroom_mib": args.safety_headroom_mib,
            },
            "elapsed_seconds": round(time.monotonic() - started, 4),
            "checks": {
                **mechanical_checks,
                "minimum_free_vram_at_least_safety_headroom": headroom_passed,
                "post_run_gpu_return_within_512mib_of_baseline": final_return_passed,
            },
            "boundary": (
                "One fixed 256x256 22+22->39-frame T2VA route, eight NFE per segment, 12/3 "
                "dual clock, CPU concat and one final AV decode. PASS closes only this local "
                "mechanical and sampled-headroom gate. Its 51,200 decoded AAC samples exactly "
                "match the prior 32B container; the 52,000-sample lossless tensor/FLAC contract "
                "was not separately saved in this probe. It does not prove seamless continuation, "
                "8B/32B quality equivalence, repeated stability, arbitrary modalities, or general "
                "16GB long-video safety."
            ),
        },
    )


def reanalyze_existing_run(args: argparse.Namespace, run_root: Path) -> dict[str, Any]:
    """Re-evaluate an existing completed run without starting ComfyUI or loading a model."""

    run_root = run_root.resolve()
    report_path = run_root / "validation_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema") != SCHEMA:
        raise ValueError(f"unexpected report schema: {report.get('schema')!r}")
    media_path = Path(str(report.get("media", {}).get("path") or ""))
    if not media_path.is_file():
        raise FileNotFoundError(f"existing media is missing: {media_path}")

    media = shared.media_report(
        media_path,
        ffmpeg=str(report["preflight"]["ffmpeg"]),
        ffprobe=str(report["preflight"]["ffprobe"]),
    )
    media_checks = media_contract_checks(media)
    asset_manifest = report.get("asset_manifest", {})
    assets = asset_manifest.get("assets", {}) if isinstance(asset_manifest, dict) else {}
    process_ids = report.get("process_ids", [])
    phase = report.get("phase") or {}
    runtime_error = report.get("runtime_error")
    mechanical_checks = {
        "reviewed_8b_encoder_sha256": bool(assets.get("clip", {}).get("matched")),
        "reviewed_8b_projection_sha256": bool(
            assets.get("projection", {}).get("matched")
        ),
        "one_isolated_process": isinstance(process_ids, list) and len(process_ids) == 1,
        "execution_success": phase.get("terminal", {}).get("type") == "execution_success",
        **media_checks,
    }
    gpu = report.get("gpu") or {}
    monitor = gpu.get("monitor") or {}
    baseline = gpu.get("baseline") or {}
    final = gpu.get("final") or {}
    minimum_free = monitor.get("minimum_free_mib")
    safety_headroom = int(gpu.get("safety_headroom_mib") or args.safety_headroom_mib)
    mechanical_passed = all(mechanical_checks.values()) and runtime_error is None
    headroom_passed = bool(
        minimum_free is not None and int(minimum_free) >= safety_headroom
    )
    final_return_passed = bool(
        final.get("available")
        and int(final.get("used_mib", 0)) <= int(baseline.get("used_mib", 0)) + 512
    )
    passed = mechanical_passed and headroom_passed and final_return_passed
    original_status = report.get("status")
    if passed:
        status = "PASS"
    elif mechanical_passed and not headroom_passed:
        status = "FAIL_MEMORY_HEADROOM_GATE"
    elif mechanical_passed and not final_return_passed:
        status = "FAIL_POST_RUN_GPU_RETURN_GATE"
    else:
        status = "FAIL_RUNTIME_OR_MEDIA_CONTRACT"

    generation_contract = dict(report.get("generation_contract") or {})
    generation_contract.pop("combined_audio_samples", None)
    generation_contract.update(
        {
            "lossless_audio_samples_expected_but_not_measured": (
                EXPECTED_LOSSLESS_AUDIO_SAMPLES
            ),
            "aac_decoded_audio_samples": EXPECTED_AAC_DECODED_SAMPLES,
            "source_32b_combined_mp4_sha256": SOURCE_32B_COMBINED_MP4_SHA256,
        }
    )
    report.update(
        {
            "status": status,
            "passed": passed,
            "mechanical_passed": mechanical_passed,
            "memory_headroom_gate_passed": headroom_passed,
            "post_run_gpu_return_gate_passed": final_return_passed,
            "generation_contract": generation_contract,
            "media": media,
            "checks": {
                **mechanical_checks,
                "minimum_free_vram_at_least_safety_headroom": headroom_passed,
                "post_run_gpu_return_within_512mib_of_baseline": final_return_passed,
            },
            "reanalysis": {
                "reanalyzed_at": _utc_now(),
                "original_status": original_status,
                "model_or_comfyui_started": False,
                "reason": (
                    "The original validator incorrectly required the 52,000-sample lossless "
                    "tensor/FLAC length from an AAC container. The prior 32B reference MP4 and "
                    "this 8B MP4 both decode to exactly 51,200 stereo samples; the old 32B "
                    "lossless FLAC remains 52,000 samples."
                ),
            },
            "boundary": (
                "One fixed 256x256 22+22->39-frame T2VA route, eight NFE per segment, 12/3 "
                "dual clock, CPU concat and one final AV decode. PASS closes only this local "
                "mechanical and 0.25-second sampled-headroom gate. Its 51,200 decoded AAC "
                "samples exactly match the prior 32B container; this run did not separately "
                "save the 52,000-sample lossless waveform. It does not prove seamless "
                "continuation, 8B/32B quality equivalence, repeated stability, arbitrary "
                "modalities, or general 16GB long-video safety."
            ),
        }
    )
    temporary = report_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(report_path)
    return report


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
        / "native-latent-clipproj-8b-real-runtime-v1",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8197)
    parser.add_argument("--min-free-vram-mib", type=int, default=12000)
    parser.add_argument("--safety-headroom-mib", type=int, default=512)
    parser.add_argument("--server-start-timeout", type=float, default=180.0)
    parser.add_argument("--timeout-seconds", type=float, default=1200.0)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument(
        "--confirm-run",
        action="store_true",
        help="Hash the reviewed 8B assets and run one private two-segment task after all gates pass.",
    )
    parser.add_argument(
        "--reanalyze-run",
        type=Path,
        help="Re-evaluate one existing run directory without starting ComfyUI or loading a model.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    if args.reanalyze_run is not None:
        if args.confirm_run:
            raise ValueError("--reanalyze-run and --confirm-run are mutually exclusive")
        result = reanalyze_existing_run(args, args.reanalyze_run)
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "passed": result["passed"],
                    "run_root": result["run_root"],
                    "report": str(Path(result["run_root"]) / "validation_report.json"),
                    "real_run_started": False,
                },
                ensure_ascii=False,
            )
        )
        return 0 if result["passed"] else 2
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
