#!/usr/bin/env python3
"""Run one isolated real OpenVDN-H3 DMD8 T2VA render and verify its AV output."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any


TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import run_clear_clipproj_triplet_probe as clipprobe  # noqa: E402
import run_nfe_resume_real_probe as shared  # noqa: E402
import run_pdd_real_validation as pdd  # noqa: E402


SCHEMA = "t8.minimax_h3.openvdn.real_validation.v1"
FPS = 24
SEED = 2609032101
BASE_MODEL = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
CLIP_MODEL = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
VDN_ROOT = "OpenVDN/vdn-minimax-h3"

PROMPT = (
    "One continuous locked-off cinematic medium close-up of one adult East Asian woman "
    "facing the camera in a quiet, softly lit concert room. She remains the only visible "
    "and audible person. She says exactly once, clearly and naturally in Mandarin Chinese: "
    "<d>[Chinese] 你在哪里</d>. Her mouth movement precisely matches this one short sentence, "
    "then her lips relax. Clean classical chamber music plays continuously underneath: warm "
    "solo cello and soft acoustic piano performing a slow adagio, with the voice clear in the "
    "foreground and music at a lower level. Stable face, eyes and anatomy, realistic skin, one "
    "continuous shot. No additional words, repeated speech, singing, subtitles, cuts, camera "
    "movement, hiss, static, crackle, clipping, distortion, duplicate person or reflection."
)


def stable_media_report(
    path: Path, *, ffmpeg: str, ffprobe: str, attempts: int = 3
) -> dict[str, Any]:
    """Repeat a failed strict read to distinguish a transient post-write decoder race."""

    history: list[dict[str, Any]] = []
    report: dict[str, Any] = {}
    for attempt in range(1, max(1, attempts) + 1):
        report = shared.media_report(path, ffmpeg=ffmpeg, ffprobe=ffprobe)
        history.append(
            {
                "attempt": attempt,
                "strict_decode_passed": report.get("strict_decode_passed") is True,
                "strict_decode": report.get("strict_decode"),
            }
        )
        if report.get("strict_decode_passed") is True:
            break
        if attempt < attempts:
            time.sleep(0.5)
    report["strict_decode_attempts"] = history
    report["strict_decode_transient_recovered"] = bool(
        len(history) > 1 and history[-1]["strict_decode_passed"]
    )
    return report


def build_prompt(args: argparse.Namespace, run_id: str) -> dict[str, Any]:
    return {
        "1": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": VIDEO_VAE},
        },
        "2": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": AUDIO_VAE},
        },
        "3": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": CLIP_MODEL,
                "type": "minimax",
                "device": "default",
            },
        },
        "4": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": BASE_MODEL, "weight_dtype": "default"},
        },
        "5": {
            "class_type": "MiniMaxH3VDNModelComposerT8Advanced",
            "inputs": {
                "model": ["4", 0],
                "vdn_root": VDN_ROOT,
                "stage": "stage_dmd_8nfe",
                "verify_hashes": False,
                "allow_structural_base": True,
            },
        },
        "6": {
            "class_type": "MiniMaxH3AudioConditioningT8",
            "inputs": {
                "clip": ["3", 0],
                "video_vae": ["1", 0],
                "audio_vae": ["2", 0],
                "prompt": PROMPT,
                "width": args.width,
                "height": args.height,
                "length": args.frame_count,
                "task_type": "T2VA",
                "audio_mode": "native",
                "audio_denoise_strength": 1.0,
                "add_source_as_reference": False,
                "prompt_primary_audio_ordinal": 0,
                "strict_prompt_tags": True,
                "ref_image_size": "match",
                "reference_video_policy": "official_2_to_15s",
            },
        },
        "7": {
            "class_type": "MiniMaxH3VDNExecutionPlanT8Advanced",
            "inputs": {"model": ["5", 0], "av_latent": ["6", 1]},
        },
        "8": {"class_type": "RandomNoise", "inputs": {"noise_seed": args.seed}},
        "9": {
            "class_type": "BasicGuider",
            "inputs": {"model": ["7", 0], "conditioning": ["6", 0]},
        },
        "10": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["8", 0],
                "guider": ["9", 0],
                "sampler": ["7", 1],
                "sigmas": ["7", 2],
                "latent_image": ["6", 1],
            },
        },
        "11": {
            "class_type": "MiniMaxH3AVDecodeT8",
            "inputs": {
                "av_latent": ["10", 0],
                "video_vae": ["1", 0],
                "audio_vae": ["2", 0],
            },
        },
        "12": {
            "class_type": "MiniMaxH3OutputTrimT8",
            "inputs": {
                "frames": ["11", 0],
                "audio": ["11", 1],
                "start_seconds": 0.0,
                "duration_seconds": args.frame_count / FPS,
                "fps": float(FPS),
            },
        },
        "13": {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": ["12", 0],
                "audio": ["12", 1],
                "frame_rate": FPS,
                "loop_count": 0,
                "filename_prefix": (
                    f"MiniMaxH3_OpenVDN/{run_id}_{args.width}x{args.height}_"
                    f"{args.frame_count}f_dmd8"
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
        "14": {
            "class_type": "PreviewAny",
            "inputs": {"source": ["5", 1]},
        },
        "15": {
            "class_type": "PreviewAny",
            "inputs": {"source": ["7", 3]},
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--comfy-root", type=Path, default=Path(__file__).resolve().parents[3]
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8205)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--frame-count", type=int, default=73)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--server-start-timeout", type=float, default=180.0)
    parser.add_argument("--timeout-seconds", type=float, default=2400.0)
    parser.add_argument("--min-free-vram-mib", type=int, default=12_000)
    parser.add_argument("--reserve-vram-gib", type=float, default=0.5)
    parser.add_argument(
        "--lowvram", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts/openvdn-h3-real-validation-20260903"),
    )
    parser.add_argument(
        "--recheck-report",
        type=Path,
        help=(
            "Re-run strict media checks for an existing validation_report.json "
            "without starting ComfyUI or sampling again."
        ),
    )
    return parser


def _asset_paths(args: argparse.Namespace) -> list[Path]:
    models = args.comfy_root / "models"
    vdn = models / "diffusion_models" / VDN_ROOT / "stage-dmd-step-250"
    return [
        args.comfy_root / "main.py",
        models / "diffusion_models" / BASE_MODEL,
        models / "text_encoders" / CLIP_MODEL,
        models / "vae" / VIDEO_VAE,
        models / "vae" / AUDIO_VAE,
        vdn / "linear_branch" / "model.safetensors",
        vdn / "adapters" / "default" / "adapter_model.safetensors",
        vdn / "adapters" / "turbo" / "adapter_model.safetensors",
    ]


def recheck_validation_report(path: Path) -> int:
    path = path.resolve()
    report = json.loads(path.read_text(encoding="utf-8"))
    video = Path(str(report["output_video"])).resolve()
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        raise RuntimeError("ffmpeg and ffprobe must be available on PATH")
    contract = report["contract"]
    media = stable_media_report(video, ffmpeg=ffmpeg, ffprobe=ffprobe)
    audio = pdd._audio_numeric(video, ffmpeg)
    media_checks = pdd._media_checks(
        media,
        audio,
        width=int(contract["width"]),
        height=int(contract["height"]),
        frame_count=int(contract["frame_count"]),
    )
    passed = (
        all(media_checks.values())
        and all(report.get("composition_checks", {}).values())
        and all(report.get("resource_checks", {}).values())
    )
    report.setdefault("media_initial", report.get("media"))
    report["media"] = media
    report["audio_numeric"] = audio
    report["media_checks"] = media_checks
    report["rechecked_at"] = datetime.now(timezone.utc).isoformat()
    report["status"] = (
        "MECHANICAL_PASS_HUMAN_REVIEW_PENDING" if passed else "FAIL_MECHANICAL"
    )
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if passed else 1


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    args.project_root = args.project_root.resolve()
    args.comfy_root = args.comfy_root.resolve()
    args.python = args.python.resolve()
    args.artifact_root = (
        args.artifact_root
        if args.artifact_root.is_absolute()
        else args.project_root / args.artifact_root
    ).resolve()
    if args.recheck_report is not None:
        report_path = (
            args.recheck_report
            if args.recheck_report.is_absolute()
            else args.project_root / args.recheck_report
        )
        return recheck_validation_report(report_path)
    args.extra_whitelist_custom_nodes = ()
    if args.width % 32 or args.height % 32:
        raise ValueError("width and height must be positive multiples of 32")
    if args.frame_count < 5 or (args.frame_count - 5) % 17:
        raise ValueError("frame-count must follow the H3 17n+5 grid")

    missing = [str(path) for path in _asset_paths(args) if not path.is_file()]
    gpu = shared.gpu_memory_mib()
    preflight = {
        "required_assets_present": not missing,
        "missing": missing,
        "port_free": not shared.port_is_listening(args.host, args.port),
        "gpu": gpu,
        "free_vram_gate": bool(
            gpu.get("available")
            and int(gpu.get("free_mib") or 0) >= args.min_free_vram_mib
        ),
        "ffmpeg": shutil.which("ffmpeg"),
        "ffprobe": shutil.which("ffprobe"),
    }
    if not all(
        (
            preflight["required_assets_present"],
            preflight["port_free"],
            preflight["free_vram_gate"],
            preflight["ffmpeg"],
            preflight["ffprobe"],
        )
    ):
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return 2

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_root = args.artifact_root / run_id
    run_root.mkdir(parents=True, exist_ok=False)
    prompt = build_prompt(args, run_id)
    (run_root / "prompt.json").write_text(
        json.dumps(prompt, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "run_root": str(run_root),
        "contract": {
            "task": "T2VA",
            "stage": "stage_dmd_8nfe",
            "width": args.width,
            "height": args.height,
            "frame_count": args.frame_count,
            "seed": args.seed,
            "steps": 8,
            "video_shift": 12.0,
            "audio_shift": 3.0,
            "sampler": "euler",
            "scheduler": "native_flow",
            "base": BASE_MODEL,
            "external_ema_b_stacked": False,
        },
        "preflight": preflight,
    }
    phase = None
    monitor = clipprobe.GpuPeakMonitor(interval_seconds=0.25)
    try:
        with shared.IsolatedServer(args, run_root, "openvdn_h3"):
            monitor.start()
            phase = asyncio.run(
                pdd._submit_prompt_capture(
                    server=f"http://{args.host}:{args.port}",
                    prompt=prompt,
                    timeout_seconds=args.timeout_seconds,
                )
            )
    finally:
        report["gpu_monitor"] = monitor.stop()
    report["phase"] = phase
    if not phase or phase.get("terminal", {}).get("type") != "execution_success":
        report["status"] = "FAIL_EXECUTION"
        (run_root / "validation_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1

    composition = json.loads(pdd._phase_text(phase, "14"))
    execution = json.loads(pdd._phase_text(phase, "15"))
    output_dir = run_root / "output" / "MiniMaxH3_OpenVDN"
    video = shared._latest_file(output_dir, f"{run_id}_*audio.mp4")
    media = stable_media_report(
        video,
        ffmpeg=str(preflight["ffmpeg"]),
        ffprobe=str(preflight["ffprobe"]),
    )
    audio = pdd._audio_numeric(video, str(preflight["ffmpeg"]))
    contact = run_root / "contact_sheet.png"
    pdd._contact_sheet(
        video,
        contact,
        str(preflight["ffmpeg"]),
        width=args.width,
        height=args.height,
        frame_count=args.frame_count,
    )
    media_checks = pdd._media_checks(
        media,
        audio,
        width=args.width,
        height=args.height,
        frame_count=args.frame_count,
    )
    composition_checks = {
        "configured": composition.get("status") == "configured",
        "stage": composition.get("stage") == "stage_dmd_8nfe",
        "branch_800": composition.get("branch", {}).get("tensor_count") == 800,
        "adapter_targets": [
            item.get("applied_targets") for item in composition.get("adapters", [])
        ]
        == [104, 259],
        "blocks_50": composition.get("main_block_count") == 50,
        "no_external_ema_b": report["contract"]["external_ema_b_stacked"] is False,
        "execution_nfe_8": execution.get("nfe") == 8,
        "execution_shifts": execution.get("video_shift") == 12.0
        and execution.get("audio_shift") == 3.0,
    }
    minimum_free = report["gpu_monitor"].get("minimum_free_mib")
    resource_checks = {
        "telemetry_available": minimum_free is not None,
        "minimum_free_vram_at_least_512_mib": minimum_free is not None
        and int(minimum_free) >= 512,
    }
    passed = (
        all(media_checks.values())
        and all(composition_checks.values())
        and all(resource_checks.values())
    )
    report.update(
        {
            "composition": composition,
            "execution": execution,
            "composition_checks": composition_checks,
            "media": media,
            "audio_numeric": audio,
            "media_checks": media_checks,
            "resource_checks": resource_checks,
            "output_video": str(video.resolve()),
            "contact_sheet": str(contact.resolve()),
            "status": (
                "MECHANICAL_PASS_HUMAN_REVIEW_PENDING" if passed else "FAIL_MECHANICAL"
            ),
            "quality_claim": "No visual, audio, or lip-sync quality claim before human review.",
        }
    )
    (run_root / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
