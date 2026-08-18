from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
from skimage.metrics import structural_similarity


PAIR_ORDER = ("t2va", "fl2va", "ref2va")
PAIR_TITLES = {
    "t2va": "T2VA 文生音视频",
    "fl2va": "FL2VA 首尾帧 + 源音重混",
    "ref2va": "Ref2VA 单图参考",
}


def _strict_decode(path: Path, ffmpeg: str, attempts: int = 3) -> dict[str, Any]:
    rows = []
    for _ in range(attempts):
        completed = subprocess.run(
            [
                ffmpeg,
                "-v",
                "error",
                "-xerror",
                "-err_detect",
                "explode",
                "-i",
                str(path),
                "-f",
                "null",
                "-",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=False,
        )
        rows.append(
            {
                "returncode": completed.returncode,
                "stderr": completed.stderr.decode("utf-8", errors="replace")[-1000:],
            }
        )
    return {"attempts": rows, "passed": all(row["returncode"] == 0 for row in rows)}


def _decode_video(path: Path) -> tuple[np.ndarray, float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if not frames:
        raise ValueError(f"Video contains no decodable frames: {path}")
    return np.stack(frames), fps


def _decode_audio(path: Path, ffmpeg: str) -> np.ndarray:
    completed = subprocess.run(
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
            "-",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode:
        raise ValueError(
            "ffmpeg audio decode failed: "
            + completed.stderr.decode("utf-8", errors="replace")[-1000:]
        )
    values = np.frombuffer(completed.stdout, dtype="<f4")
    if values.size == 0 or values.size % 2:
        raise ValueError(f"Expected non-empty stereo audio: {path}")
    return values.reshape(-1, 2)


def _video_stats(frames: np.ndarray) -> dict[str, Any]:
    gray = np.stack(
        [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in frames]
    ).astype(np.float32)
    sharpness = np.asarray(
        [cv2.Laplacian(frame, cv2.CV_32F).var() for frame in gray], dtype=np.float64
    )
    temporal = (
        np.abs(np.diff(gray, axis=0)).mean(axis=(1, 2))
        if len(frames) > 1
        else np.zeros(1, dtype=np.float32)
    )
    hsv = np.stack(
        [cv2.cvtColor(frame, cv2.COLOR_RGB2HSV) for frame in frames]
    )
    return {
        "frame_count": int(len(frames)),
        "width": int(frames.shape[2]),
        "height": int(frames.shape[1]),
        "luma_mean": float(gray.mean()),
        "saturation_mean": float(hsv[..., 1].mean()),
        "laplacian_variance_median": float(np.median(sharpness)),
        "temporal_absdiff_mean": float(temporal.mean()),
    }


def _audio_stats(audio: np.ndarray) -> dict[str, Any]:
    return {
        "sample_rate": 32000,
        "channels": 2,
        "sample_count": int(audio.shape[0]),
        "finite": bool(np.isfinite(audio).all()),
        "rms": float(np.sqrt(np.mean(np.square(audio)))),
        "peak": float(np.max(np.abs(audio))),
        "clipping_fraction": float(np.mean(np.abs(audio) >= 0.999)),
    }


def _pair_metrics(
    baseline_frames: np.ndarray,
    speed_frames: np.ndarray,
    baseline_audio: np.ndarray,
    speed_audio: np.ndarray,
) -> dict[str, Any]:
    frame_count = min(len(baseline_frames), len(speed_frames))
    if baseline_frames.shape[1:] != speed_frames.shape[1:]:
        raise ValueError("Baseline and SPEED frame canvases differ")
    base = baseline_frames[:frame_count]
    treatment = speed_frames[:frame_count]
    ssim = np.asarray(
        [
            structural_similarity(a, b, channel_axis=2, data_range=255)
            for a, b in zip(base, treatment)
        ],
        dtype=np.float64,
    )
    mae = np.abs(base.astype(np.float32) - treatment.astype(np.float32)).mean(
        axis=(1, 2, 3)
    )
    sample_count = min(len(baseline_audio), len(speed_audio))
    audio_a = baseline_audio[:sample_count].reshape(-1).astype(np.float64)
    audio_b = speed_audio[:sample_count].reshape(-1).astype(np.float64)
    denom = float(np.linalg.norm(audio_a) * np.linalg.norm(audio_b))
    correlation = float(np.dot(audio_a, audio_b) / denom) if denom > 0 else None
    return {
        "video": {
            "paired_frames": frame_count,
            "ssim_mean": float(ssim.mean()),
            "ssim_min": float(ssim.min()),
            "uint8_mae_mean": float(mae.mean()),
        },
        "audio": {
            "paired_samples": sample_count,
            "zero_lag_cosine": correlation,
            "rms_ratio_speed_over_baseline": (
                float(np.sqrt(np.mean(np.square(audio_b))))
                / max(float(np.sqrt(np.mean(np.square(audio_a)))), 1e-12)
            ),
        },
        "interpretation": (
            "Similarity values measure difference between two stochastic generation routes; "
            "they do not identify which route has better perceptual quality, motion, audio or "
            "reference adherence. Human blind review remains required."
        ),
    }


def analyze_pair(
    baseline: Path,
    speed: Path,
    *,
    ffmpeg: str,
    expected_frames: int,
    expected_fps: float,
) -> dict[str, Any]:
    baseline_frames, baseline_fps = _decode_video(baseline)
    speed_frames, speed_fps = _decode_video(speed)
    baseline_audio = _decode_audio(baseline, ffmpeg)
    speed_audio = _decode_audio(speed, ffmpeg)
    baseline_video_seconds = len(baseline_frames) / baseline_fps
    speed_video_seconds = len(speed_frames) / speed_fps
    baseline_audio_seconds = len(baseline_audio) / 32000.0
    speed_audio_seconds = len(speed_audio) / 32000.0
    mechanics = {
        "baseline_strict_decode_3_of_3": _strict_decode(baseline, ffmpeg)["passed"],
        "speed_strict_decode_3_of_3": _strict_decode(speed, ffmpeg)["passed"],
        "baseline_frame_count_exact": len(baseline_frames) == expected_frames,
        "speed_frame_count_exact": len(speed_frames) == expected_frames,
        "baseline_fps_exact": math.isclose(
            baseline_fps, expected_fps, rel_tol=0.0, abs_tol=1e-6
        ),
        "speed_fps_exact": math.isclose(
            speed_fps, expected_fps, rel_tol=0.0, abs_tol=1e-6
        ),
        "baseline_av_within_one_frame": abs(
            baseline_audio_seconds - baseline_video_seconds
        )
        <= 1.0 / expected_fps,
        "speed_av_within_one_frame": abs(speed_audio_seconds - speed_video_seconds)
        <= 1.0 / expected_fps,
        "baseline_audio_finite": bool(np.isfinite(baseline_audio).all()),
        "speed_audio_finite": bool(np.isfinite(speed_audio).all()),
    }
    return {
        "baseline": {
            "path": str(baseline.resolve()),
            "fps": baseline_fps,
            "video": _video_stats(baseline_frames),
            "audio": _audio_stats(baseline_audio),
            "audio_minus_video_seconds": baseline_audio_seconds - baseline_video_seconds,
        },
        "speed": {
            "path": str(speed.resolve()),
            "fps": speed_fps,
            "video": _video_stats(speed_frames),
            "audio": _audio_stats(speed_audio),
            "audio_minus_video_seconds": speed_audio_seconds - speed_video_seconds,
        },
        "pair": _pair_metrics(
            baseline_frames, speed_frames, baseline_audio, speed_audio
        ),
        "mechanical_checks": mechanics,
        "mechanical_pass": all(mechanics.values()),
    }


def build_blind_review(
    pairs: Mapping[str, Mapping[str, Path]], *, output_dir: Path, seed: int
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    rows = []
    reveal: dict[str, Any] = {
        "schema": "minimax_h3_speed_blind_reveal_v1",
        "seed": seed,
        "pairs": {},
    }
    for index, name in enumerate(PAIR_ORDER, start=1):
        source = pairs[name]
        baseline_is_a = bool(rng.getrandbits(1))
        mapping = {
            "A": "baseline" if baseline_is_a else "speed",
            "B": "speed" if baseline_is_a else "baseline",
        }
        files = {}
        for side, treatment in mapping.items():
            destination = output_dir / f"{index:02d}_{name}_{side}.mp4"
            shutil.copy2(source[treatment], destination)
            files[side] = destination.name
        reveal["pairs"][name] = mapping
        rows.append(
            {
                "name": name,
                "title": PAIR_TITLES[name],
                "A": files["A"],
                "B": files["B"],
            }
        )
    (output_dir / "reveal.json").write_text(
        json.dumps(reveal, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    rows_json = json.dumps(rows, ensure_ascii=False)
    html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>H3 SPEED 三组盲测</title>
<style>body{{font-family:system-ui;background:#101216;color:#eee;margin:24px}}h1{{margin-bottom:8px}}.hint{{color:#bbb}}.pair{{border:1px solid #444;border-radius:12px;padding:16px;margin:20px 0}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}video{{width:100%;background:#000}}label{{margin-right:16px}}button{{padding:10px 18px}}@media(max-width:900px){{.grid{{grid-template-columns:1fr}}}}</style></head>
<body><h1>MiniMax H3 SPEED：全分辨率 vs SPEED（三组匿名）</h1>
<p class="hint">同模型、同媒体、同提示词、同seed、同20 NFE。请完整观看并听声音；漏填按“平”处理。</p><main id="root"></main><button id="export">导出评分 JSON</button>
<script>const rows={rows_json}; const root=document.getElementById('root');
for(const row of rows){{const el=document.createElement('section');el.className='pair';el.innerHTML=`<h2>${{row.title}}</h2><div class="grid"><div><h3>A</h3><video controls preload="metadata" src="${{row.A}}"></video></div><div><h3>B</h3><video controls preload="metadata" src="${{row.B}}"></video></div></div><div class="votes"></div>`; const votes=el.querySelector('.votes'); for(const metric of ['overall','motion_detail','audio','reference_adherence']){{if(metric==='reference_adherence'&&row.name!=='ref2va')continue; const p=document.createElement('p');p.innerHTML=`<b>${{metric}}</b> `+['A','B','tie'].map(v=>`<label><input type="radio" name="${{row.name}}_${{metric}}" value="${{v}}">${{v==='tie'?'平':v}}</label>`).join('');votes.appendChild(p)}}root.appendChild(el)}}
document.getElementById('export').onclick=()=>{{const reviews=rows.map(row=>{{const out={{name:row.name}};for(const metric of ['overall','motion_detail','audio','reference_adherence']){{const x=document.querySelector(`input[name="${{row.name}}_${{metric}}"]:checked`);if(x)out[metric]=x.value;else if(metric!=='reference_adherence'||row.name==='ref2va')out[metric]='tie'}}return out}});const blob=new Blob([JSON.stringify({{schema:'minimax_h3_speed_blind_review_v1',reviews}},null,2)],{{type:'application/json'}});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='h3_speed_blind_review.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)}};</script></body></html>"""
    (output_dir / "blind_review.html").write_text(html, encoding="utf-8")
    return reveal


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze and blind-package three full-resolution versus H3 SPEED pairs."
    )
    parser.add_argument("output_dir", type=Path)
    for name in PAIR_ORDER:
        parser.add_argument(f"--{name}-baseline", type=Path, required=True)
        parser.add_argument(f"--{name}-speed", type=Path, required=True)
    parser.add_argument("--expected-frames", type=int, default=124)
    parser.add_argument("--expected-fps", type=float, default=24.0)
    parser.add_argument("--blind-seed", type=int, default=2608199001)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    pairs = {
        name: {
            "baseline": getattr(args, f"{name}_baseline"),
            "speed": getattr(args, f"{name}_speed"),
        }
        for name in PAIR_ORDER
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    analysis = {
        "schema": "minimax_h3_speed_quality_analysis_v1",
        "pairs": {
            name: analyze_pair(
                pair["baseline"],
                pair["speed"],
                ffmpeg=args.ffmpeg,
                expected_frames=args.expected_frames,
                expected_fps=args.expected_fps,
            )
            for name, pair in pairs.items()
        },
        "claims": {
            "quality_validated": False,
            "audio_noninferiority_validated": False,
            "reference_noninferiority_validated": False,
        },
    }
    analysis["all_mechanical_pass"] = all(
        pair["mechanical_pass"] for pair in analysis["pairs"].values()
    )
    (args.output_dir / "analysis.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    build_blind_review(pairs, output_dir=args.output_dir / "blind", seed=args.blind_seed)
    print(
        json.dumps(
            {
                "all_mechanical_pass": analysis["all_mechanical_pass"],
                "blind_review": str((args.output_dir / "blind" / "blind_review.html").resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
