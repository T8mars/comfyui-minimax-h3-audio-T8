#!/usr/bin/env python3
"""Run one serial, low-load real H3 validation for the 2026-08-29 nodes.

Each invocation performs exactly one model render.  The isolated ComfyUI process
is stopped before the script exits so callers can run the modes serially without
turning this into a pressure test.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping

import psutil


TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import run_clear_clipproj_triplet_probe as clipprobe  # noqa: E402
import run_nfe_resume_real_probe as shared  # noqa: E402
import run_pdd_real_validation as pdd  # noqa: E402


SCHEMA = "t8.minimax_h3.community_update_real_validation.v1"
FPS = 24
SEED = 2608291001
VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
CLIP = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
FL_BASE = "minimax_h3_fl2va_int8_convrot.safetensors"
REF_BASE = "minimax_h3_ref2va_int8_convrot.safetensors"
FAST_LORA = r"dense-datafree\adapter_model.safetensors"
PDD_FL = "MiniMax-H3-FL2VA-Acc-8Step_comfyui_pdd.safetensors"
PDD_REF = "MiniMax-H3-Ref2VA-Acc-8Step_comfyui_pdd.safetensors"
UPSCALER = "minimax_h3_latent_upscaler_3d_fp16.safetensors"
REFERENCE_IMAGE = "codex_prompt_relay_fl2va_first.png"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _conditioning(
    *,
    task: str,
    clip_node: str,
    width: int,
    height: int,
    frames: int,
    prompt: str,
    reference_node: str | None = None,
) -> dict[str, Any]:
    inputs: dict[str, Any] = {
        "clip": [clip_node, 0],
        "video_vae": ["1", 0],
        "audio_vae": ["2", 0],
        "prompt": prompt,
        "width": width,
        "height": height,
        "length": frames,
        "task_type": task,
        "audio_mode": "native",
        "audio_denoise_strength": 1.0,
        "add_source_as_reference": False,
        "prompt_primary_audio_ordinal": 0,
        "strict_prompt_tags": True,
        "ref_image_size": "match",
        "reference_video_policy": "official_2_to_15s",
    }
    if task == "FL2VA":
        inputs["first_frame"] = [reference_node or "5", 0]
        inputs["last_frame"] = [reference_node or "5", 0]
    elif task == "Ref2VA":
        inputs["ref_images.ref_image_0"] = [reference_node or "5", 0]
    return inputs


def _loaders(base: str) -> dict[str, Any]:
    return {
        "1": {"class_type": "VAELoader", "inputs": {"vae_name": VIDEO_VAE}},
        "2": {"class_type": "VAELoader", "inputs": {"vae_name": AUDIO_VAE}},
        "3": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": CLIP, "type": "minimax", "device": "default"},
        },
        "4": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": base, "weight_dtype": "default"},
        },
        "5": {"class_type": "LoadImage", "inputs": {"image": REFERENCE_IMAGE}},
    }


def _save_nodes(
    *,
    sampled: list[Any],
    run_id: str,
    mode: str,
    decode_id: str,
    save_id: str,
) -> dict[str, Any]:
    return {
        decode_id: {
            "class_type": "MiniMaxH3AVDecodeT8",
            "inputs": {
                "av_latent": sampled,
                "video_vae": ["1", 0],
                "audio_vae": ["2", 0],
            },
        },
        save_id: {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": [decode_id, 0],
                "audio": [decode_id, 1],
                "frame_rate": FPS,
                "loop_count": 0,
                "filename_prefix": f"MiniMaxH3_Community_Real/{run_id}_{mode}",
                "format": "video/h264-mp4",
                "pix_fmt": "yuv420p",
                "crf": 19,
                "save_metadata": True,
                "trim_to_audio": False,
                "pingpong": False,
                "save_output": True,
            },
        },
    }


def _fast_prompt(args: argparse.Namespace, run_id: str) -> tuple[dict[str, Any], dict[str, str]]:
    prompt = _loaders(FL_BASE)
    prompt.update(
        {
            "6": {
                "class_type": "MiniMaxH3LoRACompatibilityLoaderT8Advanced",
                "inputs": {
                    "model": ["4", 0],
                    "lora_name": FAST_LORA,
                    "strength_model": 1.0,
                },
            },
            "7": {
                "class_type": "MiniMaxH3AudioConditioningT8",
                "inputs": _conditioning(
                    task="T2VA",
                    clip_node="3",
                    width=args.width,
                    height=args.height,
                    frames=args.frame_count,
                    prompt=(
                        "A photorealistic adult woman with a natural symmetrical face, shoulder-length brown "
                        "hair and a red coat stands on a softly lit night street. A steady medium close-up: "
                        "she remains facing the "
                        "camera with only a very small natural head movement and one gentle blink. Her lips "
                        "stay closed and relaxed; no speech and no mouth movement. Stable facial geometry, "
                        "stable eyes and anatomy, soft night city ambience, one continuous shot, no subtitles."
                    ),
                ),
            },
            "8": {
                "class_type": "MiniMaxH3FastH34StepSetupT8Advanced",
                "inputs": {
                    "model": ["6", 0],
                    "av_latent": ["7", 1],
                    "task_family": "t2va_only",
                    "attention_profile": "dense_comfyui",
                },
            },
            "9": {"class_type": "RandomNoise", "inputs": {"noise_seed": SEED}},
            "10": {
                "class_type": "BasicGuider",
                "inputs": {"model": ["8", 0], "conditioning": ["7", 0]},
            },
            "11": {
                "class_type": "SamplerCustomAdvanced",
                "inputs": {
                    "noise": ["9", 0],
                    "guider": ["10", 0],
                    "sampler": ["8", 1],
                    "sigmas": ["8", 2],
                    "latent_image": ["7", 1],
                },
            },
            "12": {"class_type": "PreviewAny", "inputs": {"source": ["6", 1]}},
            "13": {"class_type": "PreviewAny", "inputs": {"source": ["8", 3]}},
        }
    )
    prompt.update(
        _save_nodes(
            sampled=["11", 0], run_id=run_id, mode="fast_h3", decode_id="14", save_id="15"
        )
    )
    return prompt, {"lora": "12", "fast": "13"}


def _timed_prompt(args: argparse.Namespace, run_id: str) -> tuple[dict[str, Any], dict[str, str]]:
    prompt = _loaders(REF_BASE)
    prompt.update(
        {
            "6": {
                "class_type": "VHS_LoadVideoPath",
                "inputs": {
                    "video": str(args.timed_source_video),
                    "force_rate": 24,
                    "custom_width": 320,
                    "custom_height": 192,
                    "frame_load_cap": 24,
                    "skip_first_frames": 0,
                    "select_every_nth": 1,
                },
            },
            "7": {
                "class_type": "MiniMaxH3TimedImageReferenceT8Advanced",
                "inputs": {
                    "clip": ["3", 0],
                    "image": ["5", 0],
                    "prompt_tag": "lighting",
                    "time_seconds": 0.25,
                    "image_size": "256",
                },
            },
            "8": {
                "class_type": "MiniMaxH3TimedVideoReferenceT8Advanced",
                "inputs": {
                    "clip": ["7", 0],
                    "video_frames": ["6", 0],
                    "prompt_tag": "motion",
                    "target_start_seconds": 0.0,
                    "source_fps": 24.0,
                    "analysis_fps": 2.0,
                    "video_size": "256",
                },
            },
            "9": {
                "class_type": "MiniMaxH3AudioConditioningT8",
                "inputs": _conditioning(
                    task="Ref2VA",
                    clip_node="8",
                    width=args.width,
                    height=args.height,
                    frames=args.frame_count,
                    reference_node="5",
                    prompt=(
                        "Use <Picture 1> for the adult woman's identity. At #lighting, keep "
                        "the soft night illumination. Following #motion, use only a subtle, slow "
                        "head turn while keeping her identity and facial geometry unchanged. A steady "
                        "medium close-up with relaxed closed lips, no speech and no mouth movement, "
                        "stable eyes and anatomy, one continuous shot, no subtitles or cuts."
                    ),
                ),
            },
            "10": {
                "class_type": "MiniMaxH3PDD8StepSetupT8Advanced",
                "inputs": {
                    "model": ["4", 0],
                    "av_latent": ["9", 1],
                    "pdd_lora_name": PDD_REF,
                    "base_variant": "Ref2VA",
                    "strength": 1.0,
                },
            },
            "11": {"class_type": "RandomNoise", "inputs": {"noise_seed": SEED + 1}},
            "12": {
                "class_type": "BasicGuider",
                "inputs": {"model": ["10", 0], "conditioning": ["9", 0]},
            },
            "13": {
                "class_type": "SamplerCustomAdvanced",
                "inputs": {
                    "noise": ["11", 0],
                    "guider": ["12", 0],
                    "sampler": ["10", 1],
                    "sigmas": ["10", 2],
                    "latent_image": ["9", 1],
                },
            },
            "14": {"class_type": "PreviewAny", "inputs": {"source": ["9", 3]}},
            "15": {"class_type": "PreviewAny", "inputs": {"source": ["9", 5]}},
        }
    )
    prompt.update(
        _save_nodes(
            sampled=["13", 0], run_id=run_id, mode="timed_reference", decode_id="16", save_id="17"
        )
    )
    return prompt, {"conditioned_prompt": "14", "conditioning": "15"}


def _chunked_prompt(args: argparse.Namespace, run_id: str) -> tuple[dict[str, Any], dict[str, str]]:
    prompt = _loaders(FL_BASE)
    prompt["6"] = {"class_type": "LoadImage", "inputs": {"image": REFERENCE_IMAGE}}
    prompt.update(
        {
            "7": {
                "class_type": "MiniMaxH3AudioConditioningT8",
                "inputs": _conditioning(
                    task="FL2VA",
                    clip_node="3",
                    width=args.width,
                    height=args.height,
                    frames=args.frame_count,
                    reference_node="5",
                    prompt=(
                        "One continuous medium close-up of exactly the same adult woman from the first "
                        "and last frames. She remains nearly still, makes one very small natural head "
                        "movement and blinks once. Her lips stay closed and relaxed; no speech and no "
                        "mouth movement. Preserve facial geometry, eyes, skin and anatomy across every "
                        "frame, quiet room ambience, no cuts or subtitles."
                    ),
                ),
            },
            "8": {
                "class_type": "MiniMaxH3PDD8StepSetupT8Advanced",
                "inputs": {
                    "model": ["4", 0],
                    "av_latent": ["7", 1],
                    "pdd_lora_name": PDD_FL,
                    "base_variant": "FL2VA",
                    "strength": 1.0,
                },
            },
            "9": {"class_type": "SplitSigmas", "inputs": {"sigmas": ["8", 2], "step": 4}},
            "10": {
                "class_type": "BasicGuider",
                "inputs": {"model": ["8", 0], "conditioning": ["7", 0]},
            },
            "11": {"class_type": "RandomNoise", "inputs": {"noise_seed": SEED + 2}},
            "12": {
                "class_type": "SamplerCustomAdvanced",
                "inputs": {
                    "noise": ["11", 0],
                    "guider": ["10", 0],
                    "sampler": ["8", 1],
                    "sigmas": ["9", 0],
                    "latent_image": ["7", 1],
                },
            },
            "13": {
                "class_type": "MiniMaxH3AudioConditioningT8",
                "inputs": _conditioning(
                    task="FL2VA",
                    clip_node="3",
                    width=args.target_width,
                    height=args.target_height,
                    frames=args.frame_count,
                    reference_node="5",
                    prompt=(
                        "One continuous medium close-up of exactly the same adult woman from the first "
                        "and last frames. She remains nearly still, makes one very small natural head "
                        "movement and blinks once. Her lips stay closed and relaxed; no speech and no "
                        "mouth movement. Preserve facial geometry, eyes, skin and anatomy across every "
                        "frame, quiet room ambience, no cuts or subtitles."
                    ),
                ),
            },
            "14": {
                "class_type": "MiniMaxH3ChunkedTwoPassPlanT8Advanced",
                "inputs": {
                    "model_name": UPSCALER,
                    "target_width": args.target_width,
                    "target_height": args.target_height,
                    "temporal_chunk_frames": args.temporal_chunk_frames,
                    "temporal_overlap_frames": args.temporal_overlap_frames,
                    "anchor_strength": 0.999,
                    "tile_width": args.target_width,
                    "tile_height": args.target_height,
                    "spatial_overlap": 32,
                    "spatial_fade": 32,
                    "minimum_tile_size": 128,
                    "overlap_blend": "smoothstep",
                    "precision": "fp16",
                    "release_policy": "offload_after",
                    "spatial_strategy": "full_frame_safe",
                },
            },
            "15": {
                "class_type": "MiniMaxH3ChunkedTwoPassUpscaleT8Advanced",
                "inputs": {
                    "model": ["8", 0],
                    "conditioning": ["13", 0],
                    "latent": ["12", 1],
                    "noise": ["11", 0],
                    "sampler": ["8", 1],
                    "sigmas": ["9", 1],
                    "plan": ["14", 0],
                    "cfg": 1.0,
                },
            },
            "16": {"class_type": "PreviewAny", "inputs": {"source": ["14", 1]}},
            "17": {"class_type": "PreviewAny", "inputs": {"source": ["15", 1]}},
        }
    )
    prompt.update(
        _save_nodes(
            sampled=["15", 0], run_id=run_id, mode="chunked_two_pass", decode_id="18", save_id="19"
        )
    )
    return prompt, {"plan": "16", "execution": "17"}


def _phase_text(phase: Mapping[str, Any], node_id: str) -> str:
    return pdd._phase_text(phase, node_id)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("fast_h3", "timed_reference", "chunked_two_pass"), required=True)
    parser.add_argument("--width", type=int, default=1088)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--frame-count", type=int, default=124)
    parser.add_argument("--target-width", type=int, default=1152)
    parser.add_argument("--target-height", type=int, default=640)
    parser.add_argument("--temporal-chunk-frames", type=int, default=68)
    parser.add_argument("--temporal-overlap-frames", type=int, default=17)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8197)
    parser.add_argument("--server-start-timeout", type=float, default=240.0)
    parser.add_argument("--timeout-seconds", type=float, default=2400.0)
    parser.add_argument("--min-free-vram-mib", type=int, default=11000)
    parser.add_argument("--min-free-ram-gib", type=float, default=50.0)
    parser.add_argument("--reserve-vram-gib", type=float, default=2.0)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--comfy-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts/community-update-real-validation-20260829"),
    )
    parser.add_argument(
        "--timed-source-video",
        type=Path,
        default=Path(
            "artifacts/pdd-ref2va-0p7mp-validation-20260827/"
            "20260827-023133-ref2va/output/MiniMaxH3_PDD_Validation/"
            "20260827-023133_ref2va_1152x640_124f_00001-audio.mp4"
        ),
    )
    return parser


def _required(args: argparse.Namespace) -> dict[str, Path]:
    models = args.comfy_root / "models"
    paths = {
        "python": args.python,
        "comfy_main": args.comfy_root / "main.py",
        "project": args.project_root,
        "vhs": args.comfy_root / "custom_nodes" / "ComfyUI-VideoHelperSuite",
        "clip": models / "text_encoders" / CLIP,
        "video_vae": models / "vae" / VIDEO_VAE,
        "audio_vae": models / "vae" / AUDIO_VAE,
        "reference": args.comfy_root / "input" / REFERENCE_IMAGE,
    }
    if args.mode == "fast_h3":
        paths.update(
            {
                "base": models / "diffusion_models" / FL_BASE,
                "fast_lora": models / "loras" / FAST_LORA,
            }
        )
    elif args.mode == "timed_reference":
        paths.update(
            {
                "base": models / "diffusion_models" / REF_BASE,
                "pdd": models / "loras" / PDD_REF,
                "source_video": args.timed_source_video,
            }
        )
    else:
        paths.update(
            {
                "base": models / "diffusion_models" / FL_BASE,
                "pdd": models / "loras" / PDD_FL,
                "upscaler": models / "latent_upscale_models" / UPSCALER,
            }
        )
    return paths


def _wait_for_mux_finalize(
    output_dir: Path,
    pattern: str,
    ffmpeg: str,
    *,
    timeout_seconds: float = 90.0,
) -> Path:
    """Keep the isolated server alive until VHS finishes the final AV mux.

    VideoHelperSuite can emit ComfyUI's ``execution_success`` before its final
    ``*-audio.mp4`` child process has flushed every packet.  Stopping the
    isolated server at that point leaves a valid video-only MP4 beside a
    truncated AV MP4.  A strict full decode is the completion signal here; file
    existence or a stable size alone is insufficient.
    """

    deadline = time.monotonic() + timeout_seconds
    last_error = "output did not appear"
    while time.monotonic() < deadline:
        candidates = sorted(output_dir.glob(pattern), key=lambda item: item.stat().st_mtime)
        if candidates:
            candidate = candidates[-1]
            completed = subprocess.run(
                [
                    ffmpeg,
                    "-v",
                    "error",
                    "-i",
                    str(candidate),
                    "-map",
                    "0:v:0",
                    "-map",
                    "0:a:0",
                    "-f",
                    "null",
                    "-",
                ],
                capture_output=True,
                text=True,
                timeout=45,
                check=False,
            )
            if completed.returncode == 0:
                return candidate
            last_error = (completed.stderr or completed.stdout or "decode failed").strip()
        time.sleep(1.0)
    raise RuntimeError(
        "final AV mux did not become strictly decodable before isolated-server shutdown: "
        + last_error[-1000:]
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    args.project_root = args.project_root.resolve()
    args.comfy_root = args.comfy_root.resolve()
    args.python = args.python.resolve()
    args.artifact_root = (
        args.artifact_root if args.artifact_root.is_absolute() else args.project_root / args.artifact_root
    ).resolve()
    args.timed_source_video = (
        args.timed_source_video
        if args.timed_source_video.is_absolute()
        else args.project_root / args.timed_source_video
    ).resolve()
    args.lowvram = True
    paths = _required(args)
    gpu = shared.gpu_memory_mib()
    free_ram_gib = psutil.virtual_memory().available / 1024**3
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    checks = {
        "required_paths_present": all(path.exists() for path in paths.values()),
        "ffmpeg_present": bool(ffmpeg),
        "ffprobe_present": bool(ffprobe),
        "user_port_8188_free": not shared.port_is_listening(args.host, 8188),
        "isolated_port_free": not shared.port_is_listening(args.host, args.port),
        "gpu_query_available": bool(gpu.get("available")),
        "free_vram_gate": bool(gpu.get("available") and int(gpu.get("free_mib") or 0) >= args.min_free_vram_mib),
        "free_ram_gate": free_ram_gib >= args.min_free_ram_gib,
    }
    preflight = {
        "schema": f"{SCHEMA}.preflight",
        "created_at": _utc_now(),
        "mode": args.mode,
        "checks": checks,
        "missing_paths": [str(path) for path in paths.values() if not path.exists()],
        "gpu": gpu,
        "free_ram_gib": round(free_ram_gib, 3),
        "ready": all(checks.values()),
        "policy": "one real render, serial, isolated ComfyUI, lowvram, no pressure loop",
    }
    print(json.dumps(preflight, ensure_ascii=False, sort_keys=True), flush=True)
    if not preflight["ready"]:
        return 2

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_root = args.artifact_root / f"{run_id}-{args.mode}"
    run_root.mkdir(parents=True, exist_ok=False)
    builders = {
        "fast_h3": _fast_prompt,
        "timed_reference": _timed_prompt,
        "chunked_two_pass": _chunked_prompt,
    }
    prompt, report_nodes = builders[args.mode](args, run_id)
    (run_root / "prompt.json").write_text(
        json.dumps(prompt, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "created_at": _utc_now(),
        "mode": args.mode,
        "run_id": run_id,
        "preflight": preflight,
        "contract": {
            "real_model_inference": True,
            "serial_single_render": True,
            "lowvram": True,
            "width": args.target_width if args.mode == "chunked_two_pass" else args.width,
            "height": args.target_height if args.mode == "chunked_two_pass" else args.height,
            "frame_count": args.frame_count,
        },
    }
    phase = None
    video: Path | None = None
    monitor = clipprobe.GpuPeakMonitor(interval_seconds=0.25)
    try:
        with shared.IsolatedServer(args, run_root, f"community_{args.mode}"):
            monitor.start()
            phase = asyncio.run(
                pdd._submit_prompt_capture(
                    server=f"http://{args.host}:{args.port}",
                    prompt=prompt,
                    timeout_seconds=args.timeout_seconds,
                )
            )
            if phase and phase.get("terminal", {}).get("type") == "execution_success":
                output_dir = run_root / "output" / "MiniMaxH3_Community_Real"
                video = _wait_for_mux_finalize(
                    output_dir,
                    f"{run_id}_{args.mode}*audio.mp4",
                    str(ffmpeg),
                )
    finally:
        report["gpu_monitor"] = monitor.stop()
    report["phase"] = phase
    (run_root / "phase.json").write_text(
        json.dumps(phase, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not phase or phase.get("terminal", {}).get("type") != "execution_success":
        report["status"] = "FAIL_EXECUTION"
        (run_root / "validation_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
        return 1

    node_reports: dict[str, Any] = {}
    for name, node_id in report_nodes.items():
        text = _phase_text(phase, node_id)
        try:
            node_reports[name] = json.loads(text)
        except json.JSONDecodeError:
            node_reports[name] = text
    output_dir = run_root / "output" / "MiniMaxH3_Community_Real"
    if video is None:
        video = shared._latest_file(output_dir, f"{run_id}_{args.mode}*audio.mp4")
    media = shared.media_report(video, ffmpeg=str(ffmpeg), ffprobe=str(ffprobe))
    audio = pdd._audio_numeric(video, str(ffmpeg))
    expected_width = args.target_width if args.mode == "chunked_two_pass" else args.width
    expected_height = args.target_height if args.mode == "chunked_two_pass" else args.height
    expected_frames = args.frame_count
    media_checks = pdd._media_checks(
        media,
        audio,
        width=expected_width,
        height=expected_height,
        frame_count=expected_frames,
    )
    if args.mode == "fast_h3":
        lora = node_reports["lora"]
        fast = node_reports["fast"]
        feature_checks = {
            "lora_applied": lora.get("status") == "applied" and int(lora.get("applied_patch_count") or 0) > 0,
            "direct_h3_aliases_used": int(lora.get("added_h3_direct_alias_count") or 0) > 0,
            "no_hash_gate": "sha256" not in json.dumps(lora).lower()
            and lora.get("file", {}).get("identity_policy")
            == "display_only_not_a_load_gate_no_hash_scan",
            "fast_h3_four_nfe": fast.get("trained_contract", {}).get("steps_nfe") == 4,
            "fast_h3_dense": fast.get("attention_profile_effective") == "dense_comfyui",
        }
    elif args.mode == "timed_reference":
        activity = [
            event
            for event in phase.get("events", [])
            if event.get("type") in {"executing", "executed"} and event.get("node") is not None
        ]
        visited = {str(event.get("node")) for event in activity}
        first_position: dict[str, int] = {}
        for index, event in enumerate(activity):
            first_position.setdefault(str(event.get("node")), index)
        feature_checks = {
            "timed_image_executed": "7" in visited,
            "timed_video_executed": "8" in visited,
            "conditioning_received_timed_clip": "9" in visited,
            "timed_chain_order": all(node in first_position for node in ("7", "8", "9"))
            and first_position["7"] < first_position["8"] < first_position["9"],
            "conditioned_prompt_retained": all(
                token in str(node_reports["conditioned_prompt"])
                for token in ("#lighting", "#motion", "stable eyes")
            ),
        }
    else:
        execution = node_reports["execution"]
        feature_checks = {
            "multiple_temporal_segments": int(execution.get("segment_count") or 0) >= 2,
            "full_frame_spatial_context": execution.get("spatial_strategy")
            == "full_frame_safe"
            and all(
                len(item.get("rows", [])) == 1 and len(item.get("cols", [])) == 1
                for item in execution.get("tiles", [])
            ),
            "audio_exact_tensor_passthrough": bool(execution.get("audio_preserved_by_identity")),
            "no_project_pixel_ceiling": execution.get("pixel_limit_policy") == "no_project_pixel_area_limit",
        }
    passed = all(media_checks.values()) and all(feature_checks.values())
    contact = run_root / "contact_0s_to_end.png"
    pdd._contact_sheet(
        video,
        contact,
        str(ffmpeg),
        width=expected_width,
        height=expected_height,
        frame_count=expected_frames,
    )
    report.update(
        {
            "node_reports": node_reports,
            "feature_checks": feature_checks,
            "media": media,
            "audio_numeric": audio,
            "media_checks": media_checks,
            "output_video": str(video.resolve()),
            "contact_sheet": str(contact.resolve()),
            "status": (
                "MECHANICAL_RENDER_PASS_REVIEW_REQUIRED"
                if passed
                else "FAIL_REAL_VALIDATION"
            ),
            "quality_claim": (
                "Mechanical execution and strict media validation only. Face quality is not passed "
                "until the full video and face crops receive explicit human review."
            ),
            "perceptual_review": {
                "status": "PENDING_HUMAN_REVIEW" if passed else "NOT_ELIGIBLE",
                "required_checks": [
                    "face_geometry_stable_all_frames",
                    "eyes_and_mouth_not_deformed",
                    "identity_consistent",
                    "no_temporal_face_flicker",
                ],
            },
        }
    )
    (run_root / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
