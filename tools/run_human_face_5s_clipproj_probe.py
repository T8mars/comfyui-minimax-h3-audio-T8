#!/usr/bin/env python3
"""Run one guarded 5.17-second human-face H3 I2VA comparison arm.

The earlier 22-frame metronome material was mechanically valid but human-unassessable. This tool
uses one SHA-locked, front-facing close portrait, 512x256, 124 frames, eight NFE, a visible head
motion and one Mandarin utterance. Each invocation runs exactly one 4B, 8B or native-32B arm in an
isolated ComfyUI process. It refuses an active user service, a busy private port, asset drift and the
reviewed per-arm free-VRAM floor. The stable sampler and existing workflows are never modified.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Mapping
import uuid

from PIL import Image

import run_clear_clipproj_triplet_probe as base


SCHEMA = "t8.minimax_h3.human_face_5s_clipproj_probe.v1"
ARMS = base.ARMS
INPUT_IMAGE = "10A.jpg"
INPUT_IMAGE_BYTES = 1_289_954
INPUT_IMAGE_SHA256 = "34E67512265DA29076075030B62BA93EC304210A09171FF68E1F44894D15A36C"
INPUT_IMAGE_WIDTH = 3_027
INPUT_IMAGE_HEIGHT = 1_531
PROMPT = (
    "Bright clean studio close-up of the same young woman, framed from the chest up with her "
    "face occupying most of the image. She looks directly into the camera, blinks naturally, "
    "slowly turns her head slightly left and back, then gives a gentle smile while speaking in "
    "clear Mandarin: <d>你在干嘛呢？我在这里呀，看看效果如何。</d> Stable facial identity, "
    "natural skin texture, crisp eyes and lips, dry close voice, quiet room tone, no music, no "
    "subtitles, no text overlay."
)
SEED = 2608245001
WIDTH = 512
HEIGHT = 256
FRAME_COUNT = 124
FPS = 24
STEPS = 8
SHIFT_VIDEO = 12.0
SHIFT_AUDIO = 3.0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_prompt(*, arm: str, run_id: str) -> dict[str, Any]:
    """Keep the reviewed text arm graph and replace only the fixed media contract."""

    prompt = base.build_prompt(arm=arm, run_id=run_id)
    prompt["8"]["inputs"].update(
        {
            "prompt": PROMPT,
            "width": WIDTH,
            "height": HEIGHT,
            "length": FRAME_COUNT,
            "task_type": "I2VA",
            "first_frame": ["15", 0],
        }
    )
    if arm != "native_32b":
        prompt["5"]["inputs"]["has_reference_images"] = True
    prompt["10"]["inputs"]["noise_seed"] = SEED
    prompt["14"]["inputs"].update(
        {
            "frame_rate": FPS,
            "filename_prefix": f"MiniMaxH3_HumanFace5s/{run_id}_{arm}",
        }
    )
    prompt["15"] = {
        "inputs": {"image": INPUT_IMAGE},
        "class_type": "LoadImage",
    }
    return prompt


def _input_path(args: argparse.Namespace) -> Path:
    return (args.comfy_root / "input" / INPUT_IMAGE).resolve()


def _input_checks(path: Path) -> dict[str, bool]:
    dimensions = (0, 0)
    if path.is_file():
        try:
            with Image.open(path) as image:
                dimensions = image.size
        except Exception:
            dimensions = (0, 0)
    return {
        "input_image_present": path.is_file(),
        "input_image_size": path.is_file() and path.stat().st_size == INPUT_IMAGE_BYTES,
        "input_image_dimensions": dimensions == (INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT),
    }


def _contract() -> dict[str, Any]:
    return {
        "prompt": PROMPT,
        "seed": SEED,
        "input_image": INPUT_IMAGE,
        "input_image_sha256": INPUT_IMAGE_SHA256,
        "width": WIDTH,
        "height": HEIGHT,
        "frame_count": FRAME_COUNT,
        "fps": FPS,
        "duration_seconds": FRAME_COUNT / FPS,
        "steps": STEPS,
        "shift_video": SHIFT_VIDEO,
        "shift_audio": SHIFT_AUDIO,
        "task_type": "I2VA",
    }


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    report = base.preflight(args)
    input_path = _input_path(args)
    input_checks = _input_checks(input_path)
    report["schema"] = f"{SCHEMA}.preflight"
    report["checks"].update(input_checks)
    report["input_image"] = {
        "path": str(input_path),
        "bytes": input_path.stat().st_size if input_path.is_file() else None,
        "expected_bytes": INPUT_IMAGE_BYTES,
        "expected_sha256": INPUT_IMAGE_SHA256,
        "expected_dimensions": [INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT],
    }
    report["contract"] = _contract()
    report["ready_for_real_run"] = all(report["checks"].values())
    if not all(input_checks.values()):
        report["status"] = "ABSTAIN_INPUT_IMAGE_CONTRACT_MISMATCH"
    elif report["ready_for_real_run"]:
        report["status"] = "READY"
    return report


def _media_checks(report: Mapping[str, Any]) -> dict[str, bool]:
    streams = report.get("probe", {}).get("streams", [])
    video = [value for value in streams if value.get("codec_type") == "video"]
    audio = [value for value in streams if value.get("codec_type") == "audio"]
    return {
        "strict_decode": bool(report.get("strict_decode_passed")),
        "video_h264_512x256": len(video) == 1
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
    run_root = (args.artifact_root / args.arm / run_id).resolve()
    run_root.mkdir(parents=True, exist_ok=False)
    paths = base._model_paths(args.comfy_root.resolve(), args.arm)
    spec = base.ARM_ASSETS[args.arm]
    input_path = _input_path(args)

    hashes = {
        "clip": base.shared._sha256_file(paths["clip"]),
        "input_image": base.shared._sha256_file(input_path),
    }
    if "projection" in paths:
        hashes["projection"] = base.shared._sha256_file(paths["projection"])
    hash_checks = {
        "clip": spec["clip_sha256"] is None
        or hashes["clip"] == str(spec["clip_sha256"]),
        "input_image": hashes["input_image"] == INPUT_IMAGE_SHA256,
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
            "input_image": INPUT_IMAGE_SHA256,
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
    server = base.shared.IsolatedServer(args, run_root, f"human_face_5s_{args.arm}")
    monitor = base.GpuPeakMonitor()
    process_ids: list[int] = []
    phase = None
    runtime_error = None
    started = time.monotonic()
    monitor.start()
    try:
        process_ids.append(server.start())
        phase = asyncio.run(
            base.shared.submit_prompt(
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
    final_gpu = base.shared._wait_gpu_return(
        int(baseline_gpu.get("used_mib", 0)) + 512
    )
    success = bool(
        phase and phase.get("terminal", {}).get("type") == "execution_success"
    )
    media = None
    media_checks: dict[str, bool] = {}
    if success:
        try:
            media_path = base.shared._latest_file(
                run_root / "output" / "MiniMaxH3_HumanFace5s",
                f"{run_id}_{args.arm}*.mp4",
            )
            media = base.shared.media_report(
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
            "contract": _contract(),
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
                "One fixed SHA-locked front-facing portrait, 512x256x124, eight-NFE I2VA arm. "
                "PASS establishes mechanical media generation only; facial quality, spoken-text "
                "accuracy and route preference require the keyed human blind review."
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
        / "human-face-5s-clipproj-runtime-v1",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8197)
    parser.add_argument("--min-free-vram-mib", type=int)
    parser.add_argument("--server-start-timeout", type=float, default=180.0)
    parser.add_argument("--timeout-seconds", type=float, default=1_200.0)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args(argv)
    floor = int(base.ARM_ASSETS[args.arm]["default_min_free_vram_mib"])
    if args.min_free_vram_mib is None:
        args.min_free_vram_mib = floor
    if args.min_free_vram_mib < floor:
        parser.error(
            f"--min-free-vram-mib cannot be lower than the reviewed {args.arm} "
            f"floor ({floor} MiB)"
        )
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = preflight(args)
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    latest = args.artifact_root / f"latest_preflight_{args.arm}.json"
    latest.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.confirm_run:
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "arm": args.arm,
                    "ready_for_real_run": report["ready_for_real_run"],
                    "real_run_started": False,
                    "preflight": str(latest.resolve()),
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
                    "ready_for_real_run": False,
                    "real_run_started": False,
                    "preflight": str(latest.resolve()),
                },
                ensure_ascii=False,
            )
        )
        return 2
    result = run_real_probe(args, report)
    print(
        json.dumps(
            {
                "status": result["status"],
                "arm": args.arm,
                "passed": result.get("passed", False),
                "run_root": result.get("run_root"),
            },
            ensure_ascii=False,
        )
    )
    return 0 if result.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
