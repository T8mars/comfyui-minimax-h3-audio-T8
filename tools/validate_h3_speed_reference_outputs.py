from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .validate_h3_speed_multimodal_outputs import (
        ValidationError,
        _decode_audio,
        _frame_metrics,
        _read_video,
        _resize_like_conditioning,
        _strict_decode,
        _telemetry_summary,
    )
except ImportError:
    from validate_h3_speed_multimodal_outputs import (
        ValidationError,
        _decode_audio,
        _frame_metrics,
        _read_video,
        _resize_like_conditioning,
        _strict_decode,
        _telemetry_summary,
    )


CASES: tuple[dict[str, Any], ...] = (
    {
        "name": "ref_image_native",
        "task": "ref2va",
        "source_counts": {"images": 1, "videos": 0, "audios": 0},
        "conditioning_counts": "pictures=1, videos=0, audios=0",
        "telemetry_label": "RefImage_native",
        "first_anchor": False,
    },
    {
        "name": "ref_video_audio_native",
        "task": "ref2va",
        "source_counts": {"images": 0, "videos": 1, "audios": 0},
        "conditioning_counts": "pictures=0, videos=1, audios=1",
        "telemetry_label": "RefVideoAudio_native",
        "first_anchor": False,
    },
    {
        "name": "hybrid_first_image_audio",
        "task": "hybrid",
        "source_counts": {"images": 1, "videos": 0, "audios": 1},
        "conditioning_counts": "pictures=2, videos=0, audios=1",
        "telemetry_label": "Hybrid_first_image_audio",
        "first_anchor": True,
    },
)


def _conditioning_contract(report: dict[str, Any], case: dict[str, Any]) -> bool:
    stages = report.get("stages", [])
    return bool(stages) and all(
        stage.get("conditioning_route") == "full_conditioning_rebuild"
        and f"task={case['task']}" in str(stage.get("conditioning_report", ""))
        and case["conditioning_counts"] in str(stage.get("conditioning_report", ""))
        for stage in stages
    )


def validate_reference_outputs(
    *,
    source_video: Path,
    output_dir: Path,
    telemetry_dir: Path,
    width: int = 1024,
    height: int = 576,
    length: int = 124,
    fps: float = 24.0,
    ffmpeg: str = "ffmpeg",
) -> dict[str, Any]:
    source_frames, _source_fps = _read_video(source_video)
    if len(source_frames) < length:
        raise ValidationError(
            f"Source has {len(source_frames)} frames but validation needs {length}"
        )
    results: dict[str, Any] = {}
    for case in CASES:
        name = str(case["name"])
        output_matches = sorted(output_dir.glob(f"{name}_*.mp4"))
        speed_matches = sorted(output_dir.glob(f"{name}_speed_report_*.json"))
        source_matches = sorted(output_dir.glob(f"{name}_source_report_*.json"))
        telemetry_matches = sorted(
            telemetry_dir.glob(f"*-SPEED_P3_{case['telemetry_label']}.json")
        )
        if not output_matches or not speed_matches or not source_matches or not telemetry_matches:
            raise ValidationError(f"Missing output/report/telemetry for {name}")
        output_path = output_matches[-1]
        frames, measured_fps = _read_video(output_path)
        audio = _decode_audio(output_path, ffmpeg)
        speed_report = json.loads(speed_matches[-1].read_text(encoding="utf-8"))
        source_report = json.loads(source_matches[-1].read_text(encoding="utf-8"))
        strict_decode = _strict_decode(output_path, ffmpeg)
        duration_delta = audio.shape[0] / 32000.0 - len(frames) / fps
        first_anchor = None
        if case["first_anchor"]:
            reference = _resize_like_conditioning(
                source_frames[0], width, height, "disabled"
            )
            first_anchor = _frame_metrics(reference, frames[0])
        finite_audio = bool(np.isfinite(audio).all())
        checks = {
            "strict_decode_3_of_3": strict_decode["passed"],
            "frame_count_exact": len(frames) == length,
            "canvas_exact": frames[0].shape[:2] == (height, width),
            "fps_exact": abs(measured_fps - fps) < 1e-6,
            "audio_finite": finite_audio,
            "audio_stereo_32khz_decode": audio.ndim == 2 and audio.shape[1] == 2,
            "av_duration_within_one_frame": abs(duration_delta) <= 1.0 / fps,
            "source_task_exact": source_report.get("resolved_task") == case["task"],
            "source_reference_counts_exact": source_report.get("reference_counts")
            == case["source_counts"],
            "execution_task_exact": speed_report.get("resolved_task") == case["task"],
            "conditioning_rebuilt_with_expected_refs": _conditioning_contract(
                speed_report, case
            ),
        }
        results[name] = {
            "output": str(output_path.resolve()),
            "strict_decode": strict_decode,
            "video": {
                "frame_count": len(frames),
                "width": frames[0].shape[1],
                "height": frames[0].shape[0],
                "fps": measured_fps,
                "first_anchor": first_anchor,
            },
            "audio": {
                "sample_rate": 32000,
                "channels": 2,
                "sample_count": int(audio.shape[0]),
                "rms": float(np.sqrt(np.mean(np.square(audio)))),
                "peak": float(np.max(np.abs(audio))),
                "finite": finite_audio,
            },
            "av_duration": {
                "audio_minus_video_seconds": duration_delta,
                "within_one_video_frame": abs(duration_delta) <= 1.0 / fps,
            },
            "telemetry": _telemetry_summary(telemetry_matches[-1]),
            "mechanical_checks": checks,
            "mechanical_pass": all(checks.values()),
            "reference_quality_validated": False,
            "audio_noninferiority_validated": False,
        }
    minimum_headroom = min(
        item["telemetry"]["minimum_headroom_bytes"] for item in results.values()
    )
    return {
        "schema": "minimax_h3_speed_p3_reference_output_validation_v1",
        "source_video": str(source_video.resolve()),
        "controlled": {"width": width, "height": height, "length": length, "fps": fps},
        "cases": results,
        "summary": {
            "all_mechanical_pass": all(item["mechanical_pass"] for item in results.values()),
            "minimum_vram_headroom_bytes": minimum_headroom,
            "passes_512_mib_headroom_gate": minimum_headroom >= 512 * 1024 * 1024,
            "reference_quality_validated": False,
            "audio_noninferiority_validated": False,
            "memory_safe_16gb": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mechanically validate three H3 SPEED P3 reference GPU outputs."
    )
    parser.add_argument("source_video", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("telemetry_dir", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=576)
    parser.add_argument("--length", type=int, default=124)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    result = validate_reference_outputs(
        source_video=args.source_video,
        output_dir=args.output_dir,
        telemetry_dir=args.telemetry_dir,
        width=args.width,
        height=args.height,
        length=args.length,
        fps=args.fps,
        ffmpeg=args.ffmpeg,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
