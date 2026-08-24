#!/usr/bin/env python3
"""Run one guarded 0.6MP human-face H3 I2VA ClipProj comparison arm.

This is the high-resolution replacement for the earlier 512x256 review material. It keeps the
same SHA-locked image, prompt, seed, 124 frames, eight NFE and dual-clock shifts, changing only the
canvas to 1088x544 (591872 pixels). Each invocation runs exactly one arm. The start gate is 14500
MiB free VRAM and a generated arm is accepted only when observed minimum free VRAM remains at
least 512 MiB. The original 5-second probe and all user workflows remain unchanged.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
from typing import Any, Iterator, Mapping

import run_human_face_5s_clipproj_probe as legacy


SCHEMA = "t8.minimax_h3.human_face_0p6mp_clipproj_probe.v1"
ARMS = ("clipproj_4b", "clipproj_8b", "native_32b")
WIDTH = 1088
HEIGHT = 544
PIXELS = WIDTH * HEIGHT
TARGET_MEGAPIXELS = PIXELS / 1_000_000
MIN_FREE_VRAM_MIB = 14_500
MIN_OBSERVED_HEADROOM_MIB = 512


def _contract() -> dict[str, Any]:
    source_aspect = legacy.INPUT_IMAGE_WIDTH / legacy.INPUT_IMAGE_HEIGHT
    canvas_aspect = WIDTH / HEIGHT
    return {
        "prompt": legacy.PROMPT,
        "seed": legacy.SEED,
        "input_image": legacy.INPUT_IMAGE,
        "input_image_sha256": legacy.INPUT_IMAGE_SHA256,
        "width": WIDTH,
        "height": HEIGHT,
        "pixels": PIXELS,
        "megapixels_decimal": TARGET_MEGAPIXELS,
        "multiple_of_32": WIDTH % 32 == 0 and HEIGHT % 32 == 0,
        "source_aspect_ratio": source_aspect,
        "canvas_aspect_ratio": canvas_aspect,
        "relative_aspect_error": abs(canvas_aspect / source_aspect - 1.0),
        "frame_count": legacy.FRAME_COUNT,
        "fps": legacy.FPS,
        "duration_seconds": legacy.FRAME_COUNT / legacy.FPS,
        "steps": legacy.STEPS,
        "shift_video": legacy.SHIFT_VIDEO,
        "shift_audio": legacy.SHIFT_AUDIO,
        "task_type": "I2VA",
        "single_variable_vs_previous_review": "canvas_512x256_to_1088x544",
    }


def _media_checks(report: Mapping[str, Any]) -> dict[str, bool]:
    streams = report.get("probe", {}).get("streams", [])
    video = [value for value in streams if value.get("codec_type") == "video"]
    audio = [value for value in streams if value.get("codec_type") == "audio"]
    return {
        "strict_decode": bool(report.get("strict_decode_passed")),
        "video_h264_exact_1088x544": len(video) == 1
        and video[0].get("codec_name") == "h264"
        and int(video[0].get("width") or 0) == WIDTH
        and int(video[0].get("height") or 0) == HEIGHT,
        "decoded_video_exact_frames": int(
            report.get("decoded_video", {}).get("bytes") or 0
        )
        == legacy.FRAME_COUNT * WIDTH * HEIGHT * 3,
        "audio_aac_32khz_stereo": len(audio) == 1
        and audio[0].get("codec_name") == "aac"
        and int(audio[0].get("sample_rate") or 0) == 32_000
        and int(audio[0].get("channels") or 0) == 2,
        "decoded_audio_nonempty": int(
            report.get("decoded_audio", {}).get("bytes") or 0
        )
        > 0,
    }


@contextmanager
def _configured_legacy() -> Iterator[None]:
    replacements = {
        "SCHEMA": SCHEMA,
        "WIDTH": WIDTH,
        "HEIGHT": HEIGHT,
        "_contract": _contract,
        "_media_checks": _media_checks,
    }
    previous = {name: getattr(legacy, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(legacy, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(legacy, name, value)


def build_prompt(*, arm: str, run_id: str) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError(f"unsupported 0.6MP arm: {arm}")
    with _configured_legacy():
        return legacy.build_prompt(arm=arm, run_id=run_id)


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    with _configured_legacy():
        report = legacy.preflight(args)
    report["resolution_acceptance"] = {
        "requested_megapixels": 0.6,
        "actual_megapixels_decimal": TARGET_MEGAPIXELS,
        "canvas": [WIDTH, HEIGHT],
        "multiple_of_32": True,
        "relative_aspect_error": _contract()["relative_aspect_error"],
    }
    return report


def _finalize_headroom(result: dict[str, Any]) -> dict[str, Any]:
    minimum = int(result.get("gpu_monitor", {}).get("minimum_free_mib", -1))
    accepted = minimum >= MIN_OBSERVED_HEADROOM_MIB
    result.setdefault("checks", {})[
        "observed_minimum_free_vram_at_least_512_mib"
    ] = accepted
    result["headroom_acceptance"] = {
        "minimum_required_mib": MIN_OBSERVED_HEADROOM_MIB,
        "observed_minimum_free_mib": minimum,
        "passed": accepted,
    }
    result["contract"] = _contract()
    result["boundary"] = (
        "One fixed SHA-locked portrait at 1088x544x124 and eight NFE. PASS establishes exact "
        "media geometry and at least 512 MiB observed headroom for this one run only; it does not "
        "establish universal 16GB safety or perceptual preference."
    )
    if result.get("passed") and not accepted:
        result["passed"] = False
        result["status"] = "FAIL_OBSERVED_MEMORY_HEADROOM_GATE"
    run_root = Path(str(result["run_root"]))
    (run_root / "validation_report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def run_real_probe(
    args: argparse.Namespace, preflight_report: Mapping[str, Any]
) -> dict[str, Any]:
    with _configured_legacy():
        result = legacy.run_real_probe(args, preflight_report)
    return _finalize_headroom(result)


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
        / "human-face-0p6mp-clipproj-runtime-v1",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8197)
    parser.add_argument(
        "--min-free-vram-mib", type=int, default=MIN_FREE_VRAM_MIB
    )
    parser.add_argument("--server-start-timeout", type=float, default=180.0)
    parser.add_argument("--timeout-seconds", type=float, default=1_200.0)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args(argv)
    if args.min_free_vram_mib < MIN_FREE_VRAM_MIB:
        parser.error(
            "--min-free-vram-mib cannot be lower than the reviewed 0.6MP floor "
            f"({MIN_FREE_VRAM_MIB} MiB)"
        )
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = preflight(args)
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    latest = args.artifact_root / f"latest_preflight_{args.arm}.json"
    latest.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.confirm_run or not report["ready_for_real_run"]:
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
        return 0 if not args.confirm_run else 2
    result = run_real_probe(args, report)
    print(
        json.dumps(
            {
                "status": result["status"],
                "arm": args.arm,
                "passed": result.get("passed", False),
                "run_root": result.get("run_root"),
                "minimum_free_mib": result.get("gpu_monitor", {}).get(
                    "minimum_free_mib"
                ),
            },
            ensure_ascii=False,
        )
    )
    return 0 if result.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
