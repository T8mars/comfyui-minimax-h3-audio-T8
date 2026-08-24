#!/usr/bin/env python3
"""Objectively analyze the long human-face Creator and ClipProj replacement media.

This report-only tool performs no model loading. It strictly decodes the four fixed review arms,
checks duration and finite audio, measures pairwise route differences, and inspects the Creator
video/audio join. The metrics can reject broken or badly discontinuous material but cannot select
the perceptually better arm; keyed human blind review remains authoritative.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from analyze_h3_speed_quality_pairs import (
    _audio_stats,
    _decode_audio,
    _decode_video,
    _pair_metrics,
    _strict_decode,
    _video_stats,
)


SCHEMA = "t8.minimax_h3.human_face_replacement_objective_analysis.v1"
AUDIO_RATE = 32_000


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _safe_ratio(value: float, baseline: float) -> float | None:
    return value / baseline if abs(baseline) > 1e-12 else None


def _db_ratio(first: float, second: float) -> float | None:
    if first <= 0.0 or second <= 0.0:
        return None
    return 20.0 * math.log10(second / first)


def _creator_join_metrics(
    frames: np.ndarray,
    audio: np.ndarray,
    *,
    join_frame: int,
    join_sample: int,
) -> dict[str, Any]:
    if not 1 <= join_frame < len(frames):
        raise ValueError("join_frame must be inside the decoded frame batch")
    if not 1 <= join_sample < len(audio):
        raise ValueError("join_sample must be inside the decoded audio batch")

    gray = np.stack(
        [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in frames]
    ).astype(np.float32)
    temporal_mad = np.abs(np.diff(gray, axis=0)).mean(axis=(1, 2))
    join_video_mad = float(temporal_mad[join_frame - 1])
    median_video_mad = float(np.median(temporal_mad))

    sample_jumps = np.max(np.abs(np.diff(audio, axis=0)), axis=1)
    join_audio_jump = float(np.max(np.abs(audio[join_sample] - audio[join_sample - 1])))
    median_audio_jump = float(np.median(sample_jumps))
    window = min(AUDIO_RATE // 4, join_sample, len(audio) - join_sample)
    before = audio[join_sample - window : join_sample]
    after = audio[join_sample : join_sample + window]
    before_rms = float(np.sqrt(np.mean(np.square(before))))
    after_rms = float(np.sqrt(np.mean(np.square(after))))
    return {
        "join_frame": join_frame,
        "join_sample": join_sample,
        "video_boundary_absdiff_mean": join_video_mad,
        "video_boundary_over_median_temporal_ratio": _safe_ratio(
            join_video_mad, median_video_mad
        ),
        "audio_single_sample_jump_max_abs": join_audio_jump,
        "audio_jump_over_median_adjacent_ratio": _safe_ratio(
            join_audio_jump, median_audio_jump
        ),
        "audio_250ms_before_rms": before_rms,
        "audio_250ms_after_rms": after_rms,
        "audio_250ms_after_vs_before_db": _db_ratio(before_rms, after_rms),
        "interpretation": (
            "Boundary ratios are discontinuity signals, not perceptual quality scores. A large "
            "value requires human inspection/listening at the join."
        ),
    }


def _frame_health(frames: np.ndarray) -> dict[str, Any]:
    gray = np.stack(
        [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in frames]
    ).astype(np.float32)
    per_frame_luma = gray.mean(axis=(1, 2))
    temporal = (
        np.abs(np.diff(gray, axis=0)).mean(axis=(1, 2))
        if len(frames) > 1
        else np.zeros(0, dtype=np.float32)
    )
    return {
        "minimum_frame_luma_mean": float(per_frame_luma.min()),
        "maximum_frame_luma_mean": float(per_frame_luma.max()),
        "near_black_frame_count_luma_below_5": int(np.sum(per_frame_luma < 5.0)),
        "near_white_frame_count_luma_above_250": int(np.sum(per_frame_luma > 250.0)),
        "near_frozen_transition_count_mad_below_0p01": int(np.sum(temporal < 0.01)),
    }


def _decode_arm(
    path: Path, *, ffmpeg: str, expected_frames: int, expected_fps: float
) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    frames, fps = _decode_video(path)
    audio = _decode_audio(path, ffmpeg)
    video_seconds = len(frames) / fps
    audio_seconds = len(audio) / AUDIO_RATE
    checks = {
        "strict_decode": bool(_strict_decode(path, ffmpeg, attempts=1)["passed"]),
        "frame_count_exact": len(frames) == expected_frames,
        "fps_exact": abs(fps - expected_fps) <= 1e-6,
        "audio_finite": bool(np.isfinite(audio).all()),
        "av_duration_within_one_frame": abs(video_seconds - audio_seconds)
        <= 1.0 / expected_fps,
    }
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "frames": frames,
        "audio": audio,
        "fps": fps,
        "video_seconds": video_seconds,
        "audio_seconds": audio_seconds,
        "video_stats": _video_stats(frames),
        "frame_health": _frame_health(frames),
        "audio_stats": _audio_stats(audio),
        "checks": checks,
    }


def _public_arm(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: item
        for key, item in value.items()
        if key not in {"frames", "audio"}
    }


def analyze(
    *,
    creator_control: Path,
    creator_candidate: Path,
    clipproj_4b: Path,
    clipproj_8b: Path,
    ffmpeg: str,
) -> dict[str, Any]:
    creator = {
        "control": _decode_arm(
            creator_control, ffmpeg=ffmpeg, expected_frames=243, expected_fps=24.0
        ),
        "candidate": _decode_arm(
            creator_candidate, ffmpeg=ffmpeg, expected_frames=243, expected_fps=24.0
        ),
    }
    clipproj = {
        "clipproj_4b": _decode_arm(
            clipproj_4b, ffmpeg=ffmpeg, expected_frames=124, expected_fps=24.0
        ),
        "clipproj_8b": _decode_arm(
            clipproj_8b, ffmpeg=ffmpeg, expected_frames=124, expected_fps=24.0
        ),
    }
    mechanics = {
        f"creator_{arm}_{name}": passed
        for arm, value in creator.items()
        for name, passed in value["checks"].items()
    }
    mechanics.update(
        {
            f"{arm}_{name}": passed
            for arm, value in clipproj.items()
            for name, passed in value["checks"].items()
        }
    )
    return {
        "schema": SCHEMA,
        "mechanical_checks": mechanics,
        "mechanical_pass": all(mechanics.values()),
        "creator": {
            "arms": {arm: _public_arm(value) for arm, value in creator.items()},
            "pairwise": _pair_metrics(
                creator["control"]["frames"],
                creator["candidate"]["frames"],
                creator["control"]["audio"],
                creator["candidate"]["audio"],
            ),
            "join": {
                arm: _creator_join_metrics(
                    value["frames"],
                    value["audio"],
                    join_frame=124,
                    join_sample=165_600,
                )
                for arm, value in creator.items()
            },
        },
        "clipproj_4b_vs_8b": {
            "arms": {arm: _public_arm(value) for arm, value in clipproj.items()},
            "pairwise": _pair_metrics(
                clipproj["clipproj_4b"]["frames"],
                clipproj["clipproj_8b"]["frames"],
                clipproj["clipproj_4b"]["audio"],
                clipproj["clipproj_8b"]["audio"],
            ),
        },
        "claims": {
            "human_preference": False,
            "creator_seam_equivalence": False,
            "clipproj_quality_equivalence": False,
            "spoken_text_accuracy": False,
            "audio_noninferiority": False,
            "general_16gb_safety": False,
        },
        "limitations": [
            "Objective similarity and boundary metrics identify route differences or gross failures; they do not identify the better-looking or better-sounding arm.",
            "The Creator pair reuses one exact joint AV latent; the ClipProj pair uses one exact portrait, prompt and seed.",
            "Human full-video and full-audio keyed blind review remains required.",
            "Native 32B is absent because its evidence-derived resource gate was not met.",
        ],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--creator-control", type=Path, required=True)
    parser.add_argument("--creator-candidate", type=Path, required=True)
    parser.add_argument("--clipproj-4b", type=Path, required=True)
    parser.add_argument("--clipproj-8b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = analyze(
        creator_control=args.creator_control,
        creator_candidate=args.creator_candidate,
        clipproj_4b=args.clipproj_4b,
        clipproj_8b=args.clipproj_8b,
        ffmpeg=args.ffmpeg,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "PASS" if report["mechanical_pass"] else "FAIL",
                "output": str(args.output.resolve()),
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["mechanical_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
