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
        _read_video,
        _strict_decode,
        _telemetry_summary,
    )
except ImportError:
    from validate_h3_speed_multimodal_outputs import (
        ValidationError,
        _decode_audio,
        _read_video,
        _strict_decode,
        _telemetry_summary,
    )


def _turbo8_report_contract(report: dict[str, Any]) -> dict[str, bool]:
    stages = report.get("stages", [])
    return {
        "scope_exact": report.get("execution_scope") == "turbo8_t2va_research_exp",
        "task_exact": report.get("resolved_task") == "t2va",
        "steps_exact": report.get("steps") == 8,
        "nfe_exact": report.get("nfe") == 8,
        "stage_nfe_exact": [stage.get("nfe") for stage in stages] == [6, 2],
        "weight_patch_present": report.get("weight_patch_contract", {}).get(
            "has_weight_patches"
        )
        is True,
        "lora_identity_not_overclaimed": report.get("weight_patch_contract", {}).get(
            "lora_identity_verified_by_runtime"
        )
        is False,
    }


def validate_turbo8_output(
    *,
    output_dir: Path,
    telemetry_dir: Path,
    width: int = 1024,
    height: int = 576,
    length: int = 124,
    fps: float = 24.0,
    ffmpeg: str = "ffmpeg",
) -> dict[str, Any]:
    output_matches = sorted(output_dir.glob("t2va_turbo8_*.mp4"))
    report_matches = sorted(output_dir.glob("t2va_turbo8_speed_report_*.json"))
    telemetry_matches = sorted(telemetry_dir.glob("*-SPEED_Turbo8_T2VA.json"))
    if not output_matches or not report_matches or not telemetry_matches:
        raise ValidationError("Missing Turbo8 output, execution report or telemetry")
    output_path = output_matches[-1]
    frames, measured_fps = _read_video(output_path)
    audio = _decode_audio(output_path, ffmpeg)
    speed_report = json.loads(report_matches[-1].read_text(encoding="utf-8"))
    strict_decode = _strict_decode(output_path, ffmpeg)
    duration_delta = audio.shape[0] / 32000.0 - len(frames) / fps
    checks = {
        "strict_decode_3_of_3": strict_decode["passed"],
        "frame_count_exact": len(frames) == length,
        "canvas_exact": frames[0].shape[:2] == (height, width),
        "fps_exact": abs(measured_fps - fps) < 1e-6,
        "audio_finite": bool(np.isfinite(audio).all()),
        "audio_stereo_32khz_decode": audio.ndim == 2 and audio.shape[1] == 2,
        "av_duration_within_one_frame": abs(duration_delta) <= 1.0 / fps,
        **_turbo8_report_contract(speed_report),
    }
    telemetry = _telemetry_summary(telemetry_matches[-1])
    return {
        "schema": "minimax_h3_speed_turbo8_output_validation_v1",
        "output": str(output_path.resolve()),
        "video": {
            "frame_count": len(frames),
            "width": frames[0].shape[1],
            "height": frames[0].shape[0],
            "fps": measured_fps,
        },
        "audio": {
            "sample_rate": 32000,
            "channels": 2,
            "sample_count": int(audio.shape[0]),
            "rms": float(np.sqrt(np.mean(np.square(audio)))),
            "peak": float(np.max(np.abs(audio))),
            "finite": bool(np.isfinite(audio).all()),
        },
        "telemetry": telemetry,
        "mechanical_checks": checks,
        "mechanical_pass": all(checks.values()),
        "quality_validated": False,
        "speedup_validated": False,
        "audio_noninferiority_validated": False,
        "memory_safe_16gb": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate one H3 SPEED Turbo8 GPU output.")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("telemetry_dir", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    result = validate_turbo8_output(
        output_dir=args.output_dir,
        telemetry_dir=args.telemetry_dir,
        ffmpeg=args.ffmpeg,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "mechanical_pass": result["mechanical_pass"],
                "minimum_headroom_bytes": result["telemetry"]["minimum_headroom_bytes"],
                "passes_512_mib_headroom_gate": result["telemetry"][
                    "passes_512_mib_headroom_gate"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
