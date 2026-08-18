from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np


CASES: tuple[dict[str, Any], ...] = (
    {
        "name": "i2va_lock_source",
        "task": "i2va",
        "audio_mode": "lock_source",
        "anchors": ("first",),
    },
    {
        "name": "fl2va_remix_source",
        "task": "fl2va",
        "audio_mode": "remix_source",
        "anchors": ("first", "last"),
    },
    {
        "name": "l2va_native",
        "task": "l2va",
        "audio_mode": "native",
        "anchors": ("last",),
    },
)


class ValidationError(RuntimeError):
    pass


def _run(command: list[str]) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(command, capture_output=True, check=False)


def _strict_decode(path: Path, ffmpeg: str, attempts: int = 3) -> dict[str, Any]:
    results = []
    for attempt in range(1, attempts + 1):
        completed = _run([ffmpeg, "-v", "error", "-i", str(path), "-f", "null", "-"])
        results.append(
            {
                "attempt": attempt,
                "returncode": completed.returncode,
                "stderr": completed.stderr.decode("utf-8", errors="replace")[-1000:],
            }
        )
    return {
        "attempts": results,
        "passed": all(item["returncode"] == 0 for item in results),
    }


def _read_video(path: Path) -> tuple[list[np.ndarray], float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValidationError(f"Could not open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise ValidationError(f"Video contains no decodable frames: {path}")
    return frames, fps


def _decode_audio(path: Path, ffmpeg: str) -> np.ndarray:
    completed = _run(
        [
            ffmpeg,
            "-v",
            "error",
            "-i",
            str(path),
            "-vn",
            "-ac",
            "2",
            "-ar",
            "32000",
            "-f",
            "f32le",
            "pipe:1",
        ]
    )
    if completed.returncode:
        raise ValidationError(
            "ffmpeg audio decode failed: "
            + completed.stderr.decode("utf-8", errors="replace")[-1000:]
        )
    samples = np.frombuffer(completed.stdout, dtype=np.float32).copy()
    if samples.size == 0 or samples.size % 2:
        raise ValidationError(f"Decoded audio is empty or malformed: {path}")
    return samples.reshape(-1, 2)


def _resize_like_conditioning(
    frame: np.ndarray, width: int, height: int, crop: str
) -> np.ndarray:
    if crop == "center":
        old_height, old_width = frame.shape[:2]
        old_aspect = old_width / old_height
        new_aspect = width / height
        x = y = 0
        if old_aspect > new_aspect:
            x = round((old_width - old_width * (new_aspect / old_aspect)) / 2)
        elif old_aspect < new_aspect:
            y = round((old_height - old_height * (old_aspect / new_aspect)) / 2)
        frame = frame[y : old_height - y, x : old_width - x]
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LANCZOS4)


def _frame_metrics(reference: np.ndarray, generated: np.ndarray) -> dict[str, Any]:
    left = reference.astype(np.float64) / 255.0
    right = generated.astype(np.float64) / 255.0
    error = left - right
    mse = float(np.mean(np.square(error)))
    left_centered = left.reshape(-1) - float(left.mean())
    right_centered = right.reshape(-1) - float(right.mean())
    denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    correlation = (
        float(np.dot(left_centered, right_centered) / denominator)
        if denominator > 0.0
        else 0.0
    )
    return {
        "normalized_mae": float(np.mean(np.abs(error))),
        "psnr_db": None if mse == 0.0 else float(10.0 * math.log10(1.0 / mse)),
        "zero_mean_pixel_correlation": correlation,
        "scope": "decoded-frame proxy; not a semantic or perceptual anchor-quality verdict",
    }


def _audio_pair_metrics(reference: np.ndarray, generated: np.ndarray) -> dict[str, Any]:
    count = min(reference.shape[0], generated.shape[0])
    left = reference[:count].astype(np.float64).reshape(-1)
    right = generated[:count].astype(np.float64).reshape(-1)
    left_centered = left - float(left.mean())
    right_centered = right - float(right.mean())
    denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    correlation = (
        float(np.dot(left_centered, right_centered) / denominator)
        if denominator > 0.0
        else None
    )
    signal_power = float(np.mean(np.square(left)))
    error_power = float(np.mean(np.square(left - right)))
    return {
        "reference_sample_count": int(reference.shape[0]),
        "generated_sample_count": int(generated.shape[0]),
        "compared_sample_count": int(count),
        "sample_count_equal": reference.shape[0] == generated.shape[0],
        "zero_lag_correlation": correlation,
        "reference_rms": float(np.sqrt(signal_power)),
        "generated_rms": float(np.sqrt(np.mean(np.square(right)))),
        "reference_snr_db": (
            float(10.0 * np.log10(signal_power / error_power))
            if signal_power > 0.0 and error_power > 0.0
            else None
        ),
        "scope": "waveform/codec proxy; remix quality still requires listening",
    }


def _telemetry_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    runtime = payload["runtime"]
    summary = runtime["summary"]
    samples = runtime["samples"]
    total = next(
        int(sample["vram_total_bytes"])
        for sample in samples
        if sample.get("vram_total_bytes")
    )
    peak = int(summary["peak_vram_used_bytes"])
    return {
        "status": runtime["status"],
        "duration_seconds": runtime["duration_seconds"],
        "resource_behavior": summary["resource_behavior"],
        "peak_vram_used_bytes": peak,
        "vram_total_bytes": total,
        "minimum_headroom_bytes": total - peak,
        "passes_512_mib_headroom_gate": total - peak >= 512 * 1024 * 1024,
    }


def validate_multimodal_outputs(
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
    source_frames, source_fps = _read_video(source_video)
    source_audio = _decode_audio(source_video, ffmpeg)
    if len(source_frames) < length:
        raise ValidationError(
            f"Source has {len(source_frames)} frames but validation needs {length}"
        )
    results: dict[str, Any] = {}
    for case in CASES:
        name = str(case["name"])
        output_matches = sorted(output_dir.glob(f"{name}_*.mp4"))
        speed_report_matches = sorted(output_dir.glob(f"{name}_speed_report_*.json"))
        telemetry_matches = sorted(telemetry_dir.glob(f"*-SPEED_P2_{case['task'].upper()}*.json"))
        if not output_matches or not speed_report_matches or not telemetry_matches:
            raise ValidationError(f"Missing output/report/telemetry for {name}")
        output_path = output_matches[-1]
        frames, measured_fps = _read_video(output_path)
        generated_audio = _decode_audio(output_path, ffmpeg)
        finite_audio = bool(np.isfinite(generated_audio).all())
        duration_delta = generated_audio.shape[0] / 32000.0 - len(frames) / fps
        anchors = {}
        for anchor in case["anchors"]:
            source_index = 0 if anchor == "first" else length - 1
            output_index = 0 if anchor == "first" else len(frames) - 1
            crop = "disabled" if anchor == "first" else "center"
            reference = _resize_like_conditioning(
                source_frames[source_index], width, height, crop
            )
            anchors[anchor] = _frame_metrics(reference, frames[output_index])
        pair = (
            _audio_pair_metrics(source_audio[: generated_audio.shape[0]], generated_audio)
            if case["audio_mode"] != "native"
            else None
        )
        report = json.loads(speed_report_matches[-1].read_text(encoding="utf-8"))
        strict_decode = _strict_decode(output_path, ffmpeg)
        mechanical = {
            "strict_decode_3_of_3": strict_decode["passed"],
            "frame_count_exact": len(frames) == length,
            "canvas_exact": frames[0].shape[:2] == (height, width),
            "fps_exact": abs(measured_fps - fps) < 1e-6,
            "audio_finite": finite_audio,
            "audio_stereo_32khz_decode": generated_audio.ndim == 2
            and generated_audio.shape[1] == 2,
            "av_duration_within_one_frame": abs(duration_delta) <= 1.0 / fps,
            "task_report_exact": report.get("resolved_task") == case["task"],
            "audio_mode_report_exact": report.get("audio_mode") == case["audio_mode"],
            "stage_conditioning_rebuilt": all(
                stage.get("conditioning_route") == "full_conditioning_rebuild"
                for stage in report.get("stages", [])
            ),
        }
        if case["audio_mode"] == "lock_source":
            mechanical["locked_audio_codec_correlation_ge_0p98"] = bool(
                pair is not None
                and pair["zero_lag_correlation"] is not None
                and pair["zero_lag_correlation"] >= 0.98
            )
        results[name] = {
            "output": str(output_path.resolve()),
            "speed_report": str(speed_report_matches[-1].resolve()),
            "strict_decode": strict_decode,
            "video": {
                "frame_count": len(frames),
                "width": frames[0].shape[1],
                "height": frames[0].shape[0],
                "fps": measured_fps,
                "anchors": anchors,
            },
            "audio": {
                "sample_rate": 32000,
                "channels": 2,
                "sample_count": int(generated_audio.shape[0]),
                "rms": float(np.sqrt(np.mean(np.square(generated_audio)))),
                "peak": float(np.max(np.abs(generated_audio))),
                "finite": finite_audio,
                "source_pair": pair,
            },
            "av_duration": {
                "audio_minus_video_seconds": duration_delta,
                "within_one_video_frame": abs(duration_delta) <= 1.0 / fps,
            },
            "telemetry": _telemetry_summary(telemetry_matches[-1]),
            "mechanical_checks": mechanical,
            "mechanical_pass": all(mechanical.values()),
            "quality_validated": False,
        }
    minimum_headroom = min(
        item["telemetry"]["minimum_headroom_bytes"] for item in results.values()
    )
    return {
        "schema": "minimax_h3_speed_p2_multimodal_output_validation_v1",
        "source_video": str(source_video.resolve()),
        "source_fps": source_fps,
        "controlled": {"width": width, "height": height, "length": length, "fps": fps},
        "cases": results,
        "summary": {
            "all_mechanical_pass": all(item["mechanical_pass"] for item in results.values()),
            "minimum_vram_headroom_bytes": minimum_headroom,
            "passes_512_mib_headroom_gate": minimum_headroom >= 512 * 1024 * 1024,
            "quality_validated": False,
            "audio_noninferiority_validated": False,
            "memory_safe_16gb": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mechanically validate three H3 SPEED P2 multimodal GPU outputs."
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
    result = validate_multimodal_outputs(
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
