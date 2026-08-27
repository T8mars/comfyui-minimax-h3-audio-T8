#!/usr/bin/env python3
"""Run one guarded, low-load PDD learned-latent 4+4 validation.

This tool intentionally performs one render only.  It splits the official PDD
sigma trajectory after block 3, learned-upscales the pass-1 denoised latent,
then completes blocks 4..7 at the new geometry.  It never loops or compares
variants and it starts an isolated ComfyUI process in low-VRAM mode.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any

import psutil


TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import run_clear_clipproj_triplet_probe as clipprobe  # noqa: E402
import run_nfe_resume_real_probe as shared  # noqa: E402
import run_pdd_real_validation as pdd_validation  # noqa: E402


SCHEMA = "t8.minimax_h3.pdd_learned_two_pass_real_validation.v1"
FPS = 24
SEED = 2608274404
BASE_NAME = {
    "FL2VA": "minimax_h3_fl2va_int8_convrot.safetensors",
    "Ref2VA": "minimax_h3_ref2va_int8_convrot.safetensors",
}
PDD_NAME = {
    "FL2VA": "MiniMax-H3-FL2VA-Acc-8Step_comfyui_pdd.safetensors",
    "Ref2VA": "MiniMax-H3-Ref2VA-Acc-8Step_comfyui_pdd.safetensors",
}


def _aligned_output_geometry(width: int, height: int, scale_by: float) -> tuple[int, int]:
    """Mirror the learned-upscaler's 32-aligned preserve-aspect selection."""

    ideal_width = width * scale_by
    ideal_height = height * scale_by
    source_aspect = width / height
    width_floor = max(32, math.floor(ideal_width / 32) * 32)
    width_ceil = max(32, math.ceil(ideal_width / 32) * 32)
    height_floor = max(32, math.floor(ideal_height / 32) * 32)
    height_ceil = max(32, math.ceil(ideal_height / 32) * 32)

    def score(size: tuple[int, int]) -> tuple[float, float]:
        candidate_width, candidate_height = size
        aspect_error = abs(math.log((candidate_width / candidate_height) / source_aspect))
        size_error = math.hypot(
            (candidate_width - ideal_width) / max(ideal_width, 1.0),
            (candidate_height - ideal_height) / max(ideal_height, 1.0),
        )
        return aspect_error, size_error

    return min(
        (
            (candidate_width, candidate_height)
            for candidate_width in {width_floor, width_ceil}
            for candidate_height in {height_floor, height_ceil}
        ),
        key=score,
    )


def _scale_label(scale_by: float) -> str:
    return f"{scale_by:g}".replace(".", "p")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _paths(args: argparse.Namespace) -> dict[str, Path]:
    models = args.comfy_root / "models"
    return {
        "comfy_main": args.comfy_root / "main.py",
        "python": args.python,
        "project": args.project_root,
        "vhs": args.comfy_root / "custom_nodes" / "ComfyUI-VideoHelperSuite",
        "base": models / "diffusion_models" / BASE_NAME[args.variant],
        "clip": models
        / "text_encoders"
        / "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        "video_vae": models / "vae" / "minimax_h3_video_vae_fp16.safetensors",
        "audio_vae": models / "vae" / "minimax_h3_audio_vae_fp32.safetensors",
        "pdd_lora": models / "loras" / PDD_NAME[args.variant],
        "upscaler": models
        / "latent_upscale_models"
        / "minimax_h3_latent_upscaler_3d_fp16.safetensors",
        "reference": args.comfy_root
        / "input"
        / "codex_prompt_relay_fl2va_first.png",
    }


def _conditioning(
    *, args: argparse.Namespace, high: bool
) -> dict[str, Any]:
    width: int | list[Any] = ["13", 1] if high else args.width
    height: int | list[Any] = ["13", 2] if high else args.height
    common: dict[str, Any] = {
        "prompt": (
            "Use <Picture 1> as the visual identity and appearance reference. "
            "In one continuous natural cinematic portrait, the same adult woman "
            "turns gently toward the camera and blinks once. She clearly says in "
            "Mandarin exactly once: <d>你在干嘛呢，我在这里呀，看看效果如何。</d> "
            "Synchronized speech and quiet room ambience, no extra words, music, "
            "subtitles, cuts or additional people."
            if args.variant == "Ref2VA"
            else "One continuous locked cinematic portrait of the same adult woman. "
            "She turns gently, blinks once and clearly says in Mandarin exactly once: "
            "<d>你在干嘛呢，我在这里呀，看看效果如何。</d> Synchronized speech, "
            "natural ambience, no extra words, music, subtitles, cuts or scale changes."
        ),
        "width": width,
        "height": height,
        "length": args.frame_count,
        "task_type": args.variant,
        "audio_mode": "native",
        "audio_denoise_strength": 1.0,
        "add_source_as_reference": False,
        "prompt_primary_audio_ordinal": 0,
        "strict_prompt_tags": True,
        "ref_image_size": "match",
        "reference_video_policy": "official_2_to_15s",
        "clip": ["3", 0],
        "video_vae": ["1", 0],
        "audio_vae": ["2", 0],
    }
    if high:
        common["allow_above_reference_area"] = True
    if args.variant == "Ref2VA":
        common["ref_images.ref_image_0"] = ["5", 0]
    else:
        common["first_frame"] = ["5", 0]
        common["last_frame"] = ["6", 0]
    return common


def _prompt(args: argparse.Namespace, run_id: str) -> dict[str, Any]:
    prompt: dict[str, Any] = {
        "1": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "minimax_h3_video_vae_fp16.safetensors"},
        },
        "2": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"},
        },
        "3": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
                "type": "minimax",
                "device": "default",
            },
        },
        "4": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": BASE_NAME[args.variant],
                "weight_dtype": "default",
            },
        },
        "5": {
            "class_type": "LoadImage",
            "inputs": {"image": "codex_prompt_relay_fl2va_first.png"},
        },
        "7": {
            "class_type": "MiniMaxH3AudioConditioningT8",
            "inputs": _conditioning(args=args, high=False),
        },
        "8": {
            "class_type": "MiniMaxH3PDD8StepSetupT8Advanced",
            "inputs": {
                "model": ["4", 0],
                "av_latent": ["7", 1],
                "pdd_lora_name": PDD_NAME[args.variant],
                "base_variant": args.variant,
                "strength": 1.0,
            },
        },
        "9": {
            "class_type": "SplitSigmas",
            "inputs": {"sigmas": ["8", 2], "step": 4},
        },
        "10": {
            "class_type": "BasicGuider",
            "inputs": {"model": ["8", 0], "conditioning": ["7", 0]},
        },
        "11": {"class_type": "RandomNoise", "inputs": {"noise_seed": SEED}},
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
            "class_type": "MiniMaxH3LearnedLatentUpscaleT8Advanced",
            "inputs": {
                "av_latent": ["12", 1],
                "model_name": "minimax_h3_latent_upscaler_3d_fp16.safetensors",
                "size_mode": "scale_by",
                "scale_by": args.scale_by,
                "target_megapixels": 1.0,
                "target_width": round(args.width * args.scale_by),
                "target_height": round(args.height * args.scale_by),
                "aspect_policy": "preserve_source",
                "max_anisotropy": 1.05,
                "precision": "fp16",
                "release_policy": "offload_after",
            },
        },
        "14": {
            "class_type": "MiniMaxH3AudioConditioningT8",
            "inputs": _conditioning(args=args, high=True),
        },
        "15": {
            "class_type": "MiniMaxH3TwoPassLatentReconcileT8Advanced",
            "inputs": {
                "learned_latent": ["13", 0],
                "highres_template": ["14", 1],
                "positive": ["14", 0],
                "audio_policy": "auto",
                "second_pass_audio_source": "legacy_policy",
                "second_pass_audio_strength": 0.0,
            },
        },
        "16": {
            "class_type": "MiniMaxH3DualClockSamplerT8",
            "inputs": {
                "model": ["8", 0],
                "av_latent": ["15", 0],
                "steps": 8,
                "shift_video": 12.0,
                "shift_audio": 3.0,
                "sampler_name": "dual_clock_euler",
                "scheduler": "native_flow",
            },
        },
        "17": {
            "class_type": "BasicGuider",
            "inputs": {"model": ["16", 0], "conditioning": ["15", 1]},
        },
        "18": {"class_type": "RandomNoise", "inputs": {"noise_seed": SEED}},
        "19": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["18", 0],
                "guider": ["17", 0],
                "sampler": ["16", 1],
                "sigmas": ["9", 1],
                "latent_image": ["15", 0],
            },
        },
        "20": {
            "class_type": "MiniMaxH3AVDecodeT8",
            "inputs": {
                "av_latent": ["19", 0],
                "video_vae": ["1", 0],
                "audio_vae": ["2", 0],
            },
        },
        "21": {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": ["20", 0],
                "audio": ["20", 1],
                "frame_rate": FPS,
                "loop_count": 0,
                "filename_prefix": (
                    f"MiniMaxH3_PDD_TwoPass_Validation/{run_id}_{args.variant.lower()}_"
                    f"{args.width}x{args.height}_scale_{_scale_label(args.scale_by)}_"
                    f"{args.frame_count}f"
                ),
                "format": "video/h264-mp4",
                "pix_fmt": "yuv420p",
                "crf": 19,
                "save_metadata": True,
                "trim_to_audio": False,
                "pingpong": False,
                "save_output": True,
            },
        },
        "22": {
            "class_type": "PreviewAny",
            "inputs": {"source": ["8", 3]},
        },
        "23": {
            "class_type": "PreviewAny",
            "inputs": {"source": ["13", 3]},
        },
        "24": {
            "class_type": "PreviewAny",
            "inputs": {"source": ["15", 2]},
        },
    }
    if args.variant == "FL2VA":
        prompt["6"] = {
            "class_type": "LoadImage",
            "inputs": {"image": "codex_prompt_relay_fl2va_first.png"},
        }
    return prompt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("FL2VA", "Ref2VA"), default="Ref2VA")
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--frame-count", type=int, default=22)
    parser.add_argument("--scale-by", type=float, default=2.0)
    parser.add_argument("--min-free-vram-mib", type=int, default=11000)
    parser.add_argument("--min-free-ram-gib", type=float, default=60.0)
    parser.add_argument("--reserve-vram-gib", type=float, default=2.0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8197)
    parser.add_argument("--server-start-timeout", type=float, default=240.0)
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--comfy-root", type=Path, default=Path(__file__).resolve().parents[3]
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts/pdd-learned-two-pass-real-validation-20260827"),
    )
    parser.add_argument(
        "--recover-run-root",
        type=Path,
        help="Inspect an already completed output without starting ComfyUI again.",
    )
    return parser


def _validate_contract(args: argparse.Namespace) -> None:
    if args.width <= 0 or args.height <= 0 or args.frame_count <= 0:
        raise ValueError("width, height and frame-count must be positive")
    if args.width % 32 or args.height % 32:
        raise ValueError("LOW width and height must be divisible by 32")
    if not math.isfinite(args.scale_by) or not 1.0 <= args.scale_by <= 4.0:
        raise ValueError("scale-by must be finite and within [1.0, 4.0]")


def _recover_completed_output(args: argparse.Namespace, run_root: Path) -> int:
    """Validate an output left behind after post-processing, without inference."""

    prompt = json.loads((run_root / "prompt.json").read_text(encoding="utf-8"))
    run_id = run_root.name.rsplit("-", 1)[0]
    output_dir = run_root / "output" / "MiniMaxH3_PDD_TwoPass_Validation"
    video = shared._latest_file(output_dir, f"{run_id}_{args.variant.lower()}_*audio.mp4")
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise RuntimeError("ffmpeg and ffprobe are required for recovery")
    media = shared.media_report(video, ffmpeg=ffmpeg, ffprobe=ffprobe)
    audio = pdd_validation._audio_numeric(video, ffmpeg)
    high_width, high_height = _aligned_output_geometry(
        args.width, args.height, args.scale_by
    )
    media_checks = pdd_validation._media_checks(
        media,
        audio,
        width=high_width,
        height=high_height,
        frame_count=args.frame_count,
    )
    graph_checks = {
        "one_pdd_setup": sum(
            node.get("class_type") == "MiniMaxH3PDD8StepSetupT8Advanced"
            for node in prompt.values()
        )
        == 1,
        "split_at_four": prompt.get("9", {}).get("inputs", {}).get("step") == 4,
        "pass1_uses_upper_sigmas": prompt.get("12", {}).get("inputs", {}).get("sigmas")
        == ["9", 0],
        "pass2_uses_lower_sigmas": prompt.get("19", {}).get("inputs", {}).get("sigmas")
        == ["9", 1],
        "upscale_consumes_pass1_denoised": prompt.get("13", {}).get("inputs", {}).get(
            "av_latent"
        )
        == ["12", 1],
        "pass2_uses_reconciled_high_latent": prompt.get("19", {}).get("inputs", {}).get(
            "latent_image"
        )
        == ["15", 0],
    }
    stderr_path = run_root / "logs" / "pdd_learned_twopass.stderr.log"
    stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")
    execution_checks = {
        "two_four_step_progressions_logged": stderr_text.count("0/4") >= 2
        and stderr_text.count("4/4") >= 2,
        "combined_output_written": video.is_file() and video.stat().st_size > 0,
    }
    contact = run_root / "contact_0s_to_end.png"
    pdd_validation._contact_sheet(
        video,
        contact,
        ffmpeg,
        width=high_width,
        height=high_height,
        frame_count=args.frame_count,
    )
    passed = (
        all(graph_checks.values())
        and all(execution_checks.values())
        and all(media_checks.values())
    )
    report = {
        "schema": SCHEMA,
        "created_at": _utc_now(),
        "run_id": run_id,
        "variant": args.variant,
        "recovered_without_model_rerun": True,
        "contract": {
            "low_geometry": [args.width, args.height, args.frame_count],
            "high_geometry": [high_width, high_height, args.frame_count],
            "pdd_blocks_low": [0, 1, 2, 3],
            "pdd_blocks_high": [4, 5, 6, 7],
            "total_nfe": 8,
        },
        "graph_checks": graph_checks,
        "execution_checks": execution_checks,
        "media": media,
        "audio_numeric": audio,
        "media_checks": media_checks,
        "output_video": str(video.resolve()),
        "contact_sheet": str(contact.resolve()),
        "status": "MECHANICAL_PASS_RECOVERED" if passed else "FAIL_MECHANICAL",
        "quality_claim": "Mechanical validation only; no human quality claim.",
        "recovery_reason": (
            "The one real render completed; the first validator revision failed only "
            "while reading the learned-upscaler report after output publication."
        ),
    }
    (run_root / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if passed else 1


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _validate_contract(args)
    args.project_root = args.project_root.resolve()
    args.comfy_root = args.comfy_root.resolve()
    args.python = args.python.resolve()
    args.artifact_root = (
        args.artifact_root
        if args.artifact_root.is_absolute()
        else args.project_root / args.artifact_root
    ).resolve()
    args.lowvram = True
    if args.recover_run_root is not None:
        return _recover_completed_output(args, args.recover_run_root.resolve())
    paths = _paths(args)
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
        "free_vram_gate": bool(
            gpu.get("available")
            and int(gpu.get("free_mib") or 0) >= args.min_free_vram_mib
        ),
        "free_ram_gate": free_ram_gib >= args.min_free_ram_gib,
    }
    preflight = {
        "schema": f"{SCHEMA}.preflight",
        "created_at": _utc_now(),
        "checks": checks,
        "gpu": gpu,
        "free_ram_gib": round(free_ram_gib, 3),
        "ready": all(checks.values()),
        "policy": "one render, serial, lowvram, no hash/filename identity gate",
    }
    print(json.dumps(preflight, ensure_ascii=False, sort_keys=True), flush=True)
    if not preflight["ready"]:
        return 2

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_root = args.artifact_root / f"{run_id}-{args.variant.lower()}"
    run_root.mkdir(parents=True, exist_ok=False)
    prompt = _prompt(args, run_id)
    (run_root / "prompt.json").write_text(
        json.dumps(prompt, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "created_at": _utc_now(),
        "run_id": run_id,
        "variant": args.variant,
        "preflight": preflight,
        "contract": {
            "low_geometry": [args.width, args.height, args.frame_count],
            "high_geometry": [
                *_aligned_output_geometry(args.width, args.height, args.scale_by),
                args.frame_count,
            ],
            "pdd_blocks_low": [0, 1, 2, 3],
            "pdd_blocks_high": [4, 5, 6, 7],
            "total_nfe": 8,
            "single_render_only": True,
            "lowvram": True,
            "reserve_vram_gib": args.reserve_vram_gib,
        },
        "models": {
            "base": str(paths["base"]),
            "pdd_lora": str(paths["pdd_lora"]),
            "pdd_lora_bytes": paths["pdd_lora"].stat().st_size,
            "latent_upscaler": str(paths["upscaler"]),
            "identity_policy": "reported only; runtime loaders are authoritative",
        },
    }
    phase = None
    monitor = clipprobe.GpuPeakMonitor(interval_seconds=0.25)
    try:
        with shared.IsolatedServer(args, run_root, "pdd_learned_twopass"):
            monitor.start()
            phase = asyncio.run(
                pdd_validation._submit_prompt_capture(
                    server=f"http://{args.host}:{args.port}",
                    prompt=prompt,
                    timeout_seconds=args.timeout_seconds,
                )
            )
    finally:
        report["gpu_monitor"] = monitor.stop()
    report["phase"] = phase
    (run_root / "phase.json").write_text(
        json.dumps(phase, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_root / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not phase or phase.get("terminal", {}).get("type") != "execution_success":
        report["status"] = "FAIL_EXECUTION"
        (run_root / "validation_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
        return 1

    setup = json.loads(pdd_validation._phase_text(phase, "22"))
    upscale = json.loads(pdd_validation._phase_text(phase, "23"))
    reconcile = json.loads(pdd_validation._phase_text(phase, "24"))
    output_dir = run_root / "output" / "MiniMaxH3_PDD_TwoPass_Validation"
    video = shared._latest_file(
        output_dir,
        (
            f"{run_id}_{args.variant.lower()}_{args.width}x{args.height}_scale_"
            f"{_scale_label(args.scale_by)}_{args.frame_count}f*audio.mp4"
        ),
    )
    high_width = int(upscale["geometry"]["output_width"])
    high_height = int(upscale["geometry"]["output_height"])
    media = shared.media_report(video, ffmpeg=str(ffmpeg), ffprobe=str(ffprobe))
    audio = pdd_validation._audio_numeric(video, str(ffmpeg))
    media_checks = pdd_validation._media_checks(
        media,
        audio,
        width=high_width,
        height=high_height,
        frame_count=args.frame_count,
    )
    setup_checks = {
        "pdd_nfe_8": int(setup.get("sampling", {}).get("nfe") or 0) == 8,
        "pdd_blocks_0_to_7": setup.get("sampling", {}).get("block_indices")
        == list(range(8)),
        "upscale_matches_aligned_requested_scale": (high_width, high_height)
        == _aligned_output_geometry(args.width, args.height, args.scale_by),
        "reconcile_reports_native_audio_continuation": (
            reconcile.get("second_pass_audio_source") == "legacy_policy"
            or reconcile.get("audio", {}).get("second_pass_audio_source")
            == "legacy_policy"
        ),
    }
    contact = run_root / "contact_0s_to_end.png"
    pdd_validation._contact_sheet(
        video,
        contact,
        str(ffmpeg),
        width=high_width,
        height=high_height,
        frame_count=args.frame_count,
    )
    passed = all(media_checks.values()) and all(setup_checks.values())
    report.update(
        {
            "setup_report": setup,
            "upscale_report": upscale,
            "reconcile_report": reconcile,
            "setup_checks": setup_checks,
            "media": media,
            "audio_numeric": audio,
            "media_checks": media_checks,
            "output_video": str(video.resolve()),
            "contact_sheet": str(contact.resolve()),
            "status": "MECHANICAL_PASS" if passed else "FAIL_MECHANICAL",
            "quality_claim": "Mechanical validation only; no human quality claim.",
        }
    )
    (run_root / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
