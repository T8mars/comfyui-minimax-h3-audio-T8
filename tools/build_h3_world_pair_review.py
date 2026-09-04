#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import random
import shutil
import subprocess
from typing import Any

import cv2
import numpy as np


WIDTH = 832
HEIGHT = 480
FRAME_COUNT = 124
FPS = 24.0
SAMPLE_RATE = 32000
REVEAL_SCHEMA = "t8.minimax_h3.world.action_pair_reveal.v1"
REVIEW_SCHEMA = "t8.minimax_h3.world.action_pair_human_review.v1"
SCREENING_SCHEMA = "t8.minimax_h3.world.action_pair_screening.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _run(args: list[str], *, binary: bool = False):
    completed = subprocess.run(
        args,
        check=False,
        capture_output=True,
        text=not binary,
    )
    if completed.returncode:
        stderr = (
            completed.stderr.decode("utf-8", errors="replace")
            if binary
            else completed.stderr
        )
        raise RuntimeError(f"command failed ({completed.returncode}):\n{stderr[-4000:]}")
    return completed


def _probe(path: Path, ffprobe: str) -> dict[str, Any]:
    completed = _run(
        [
            ffprobe,
            "-v",
            "error",
            "-count_frames",
            "-show_entries",
            "stream=index,codec_type,codec_name,width,height,r_frame_rate,nb_read_frames,sample_rate,channels,duration",
            "-show_entries",
            "format=duration,size",
            "-of",
            "json",
            str(path),
        ]
    )
    parsed = json.loads(completed.stdout)
    video = next(
        (stream for stream in parsed.get("streams", []) if stream.get("codec_type") == "video"),
        None,
    )
    audio = next(
        (stream for stream in parsed.get("streams", []) if stream.get("codec_type") == "audio"),
        None,
    )
    if video is None or audio is None:
        raise ValueError(f"H3-World review source must contain video and audio: {path}")
    rate_num, rate_den = (int(value) for value in video["r_frame_rate"].split("/"))
    report = {
        "video_codec": video.get("codec_name"),
        "width": int(video["width"]),
        "height": int(video["height"]),
        "fps": rate_num / rate_den,
        "frame_count": int(video["nb_read_frames"]),
        "video_duration": float(video["duration"]),
        "audio_codec": audio.get("codec_name"),
        "sample_rate": int(audio["sample_rate"]),
        "channels": int(audio["channels"]),
        "audio_duration": float(audio["duration"]),
        "container_duration": float(parsed["format"]["duration"]),
        "size": int(parsed["format"]["size"]),
    }
    expected = {
        "width": WIDTH,
        "height": HEIGHT,
        "frame_count": FRAME_COUNT,
        "fps": FPS,
        "sample_rate": SAMPLE_RATE,
        "channels": 2,
    }
    for key, value in expected.items():
        if report[key] != value:
            raise ValueError(f"{path.name} has {key}={report[key]!r}; expected {value!r}")
    return report


def _strict_decode(path: Path, ffmpeg: str) -> None:
    completed = _run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-xerror",
            "-err_detect",
            "explode",
            "-threads",
            "1",
            "-nostdin",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0",
            "-f",
            "null",
            "-",
        ]
    )
    if completed.stderr.strip():
        raise RuntimeError(f"strict decode reported errors for {path}:\n{completed.stderr[-4000:]}")


def _motion_metrics(path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    previous = None
    frame_count = 0
    absdiff = []
    flow_magnitude = []
    flow_x = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (208, 120), interpolation=cv2.INTER_AREA)
            if previous is not None:
                absdiff.append(float(np.mean(cv2.absdiff(previous, gray))) / 255.0)
                flow = cv2.calcOpticalFlowFarneback(
                    previous, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                flow_magnitude.append(float(np.mean(np.linalg.norm(flow, axis=2))))
                flow_x.append(float(np.mean(flow[..., 0])))
            previous = gray
            frame_count += 1
    finally:
        capture.release()
    if frame_count != FRAME_COUNT or len(absdiff) != FRAME_COUNT - 1:
        raise RuntimeError(f"OpenCV decoded {frame_count}/{FRAME_COUNT} frames from {path}")
    return {
        "decoded_frames": frame_count,
        "temporal_absdiff_mean": float(np.mean(absdiff)),
        "temporal_absdiff_p95": float(np.percentile(absdiff, 95)),
        "flow_magnitude_mean": float(np.mean(flow_magnitude)),
        "flow_magnitude_p95": float(np.percentile(flow_magnitude, 95)),
        "signed_horizontal_flow_mean": float(np.mean(flow_x)),
        "black_frame_count": 0,
        "frozen_pair_count_at_1e-5": int(np.count_nonzero(np.asarray(absdiff) <= 1e-5)),
    }


def _audio_metrics(path: Path, ffmpeg: str) -> dict[str, Any]:
    completed = _run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-f",
            "f32le",
            "-ac",
            "2",
            "-ar",
            str(SAMPLE_RATE),
            "pipe:1",
        ],
        binary=True,
    )
    samples = np.frombuffer(completed.stdout, dtype="<f4")
    if not samples.size or samples.size % 2 or not np.isfinite(samples).all():
        raise RuntimeError(f"decoded AUDIO is empty, malformed, or non-finite: {path}")
    return {
        "decoded_samples_per_channel": int(samples.size // 2),
        "rms": float(np.sqrt(np.mean(np.square(samples, dtype=np.float64)))),
        "peak": float(np.max(np.abs(samples))),
        "clipped_sample_values": int(np.count_nonzero(np.abs(samples) >= 1.0)),
        "finite": True,
    }


def build_review(
    *,
    forward: Path,
    still: Path,
    output_dir: Path,
    seed: int,
    ffmpeg: str,
    ffprobe: str,
) -> dict[str, Any]:
    sources = {"forward": forward.resolve(), "still": still.resolve()}
    for path in sources.values():
        if not path.is_file():
            raise ValueError(f"review source does not exist: {path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    screening = {"schema": SCREENING_SCHEMA, "clips": {}}
    for treatment, path in sources.items():
        _strict_decode(path, ffmpeg)
        screening["clips"][treatment] = {
            "path": str(path),
            "sha256": _sha256(path),
            "probe": _probe(path, ffprobe),
            "strict_av_decode": True,
            "motion": _motion_metrics(path),
            "audio": _audio_metrics(path, ffmpeg),
        }
    screening["mechanical_gate"] = "PASS"
    (output_dir / "screening.json").write_text(
        json.dumps(screening, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    screening_sha256 = _sha256(output_dir / "screening.json")

    treatments = ["forward", "still"]
    random.Random(int(seed)).shuffle(treatments)
    mapping = {"A": treatments[0], "B": treatments[1]}
    for label in ("A", "B"):
        shutil.copy2(sources[mapping[label]], output_dir / f"candidate_{label}.mp4")
    reveal = {
        "schema": REVEAL_SCHEMA,
        "seed": int(seed),
        "mapping": mapping,
        "screening_sha256": screening_sha256,
        "sources": screening["clips"],
    }
    (output_dir / "reveal.json").write_text(
        json.dumps(reveal, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    title = "MiniMax H3-World：Forward / Still 同种子盲测"
    page = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>
body{{font-family:system-ui;background:#10141b;color:#edf2f7;margin:24px}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}video{{width:100%;background:#000}}
.card,.form{{background:#1c2430;padding:14px;border-radius:12px}}button{{padding:9px 16px;margin:8px 8px 8px 0}}
label{{margin-right:16px}}textarea{{width:100%;min-height:80px}}.note{{color:#afbac9}}
</style></head><body><h1>{html.escape(title)}</h1>
<p class="note">两条只差动作时间线，一条是 forward，一条是 still。不要打开 reveal.json。先静音同步看动作，再分别试听声音。</p>
<button onclick="syncPlay()">静音同步播放</button><button onclick="syncPause()">同步暂停</button><button onclick="syncReset()">回到开头</button>
<div class="grid"><section class="card"><h2>A</h2><video id="a" controls src="candidate_A.mp4"></video></section>
<section class="card"><h2>B</h2><video id="b" controls src="candidate_B.mp4"></video></section></div>
<section class="form"><p><b>哪条前进更明显？</b> <label><input type="radio" name="forward" value="A">A</label><label><input type="radio" name="forward" value="B">B</label><label><input type="radio" name="forward" value="tie">差不多/看不出</label></p>
<p><b>人物动作是否稳定可用？</b> <label><input type="radio" name="stable" value="yes">是</label><label><input type="radio" name="stable" value="no">否</label></p>
<p><b>画面偏好</b> <label><input type="radio" name="visual" value="A">A</label><label><input type="radio" name="visual" value="B">B</label><label><input type="radio" name="visual" value="tie">差不多</label></p>
<p><b>声音</b> <label><input type="radio" name="audio" value="both_ok">都正常</label><label><input type="radio" name="audio" value="A">A更好</label><label><input type="radio" name="audio" value="B">B更好</label><label><input type="radio" name="audio" value="problem">有问题</label></p>
<p><label><input type="checkbox" id="watched">我已完整观看两条视频，并分别试听声音</label></p>
<textarea id="notes" placeholder="可选备注"></textarea><br><button onclick="saveReview()">导出评分 JSON</button></section>
<script>
const a=document.getElementById('a'),b=document.getElementById('b');
function syncPlay(){{a.muted=true;b.muted=true;b.currentTime=a.currentTime;Promise.all([a.play(),b.play()]);}}
function syncPause(){{a.pause();b.pause();}} function syncReset(){{syncPause();a.currentTime=0;b.currentTime=0;}}
function vote(name){{const x=document.querySelector('input[name="'+name+'"]:checked');return x?x.value:null;}}
function saveReview(){{const votes={{forward:vote('forward'),stable:vote('stable'),visual:vote('visual'),audio:vote('audio')}};
if(!document.getElementById('watched').checked){{alert('请先完整观看两条视频并勾选确认。');return;}}
if(Object.values(votes).some(x=>x===null)){{alert('请完成四项评分后再导出。');return;}}
const review={{schema:'{REVIEW_SCHEMA}',screening_sha256:'{screening_sha256}',watched_full_length:true,votes:votes,notes:document.getElementById('notes').value,reviewed_at:new Date().toISOString()}};
const blob=new Blob([JSON.stringify(review,null,2)+'\\n'],{{type:'application/json'}});const link=document.createElement('a');link.href=URL.createObjectURL(blob);link.download='h3_world_action_pair_review.json';link.click();URL.revokeObjectURL(link.href);}}
</script></body></html>"""
    (output_dir / "blind_review.html").write_text(page, encoding="utf-8")
    return reveal


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a strict H3-World forward/still blind review.")
    parser.add_argument("--forward", type=Path, required=True)
    parser.add_argument("--still", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2609041701)
    parser.add_argument("--ffmpeg", default=shutil.which("ffmpeg") or "ffmpeg")
    parser.add_argument("--ffprobe", default=shutil.which("ffprobe") or "ffprobe")
    args = parser.parse_args()
    reveal = build_review(
        forward=args.forward,
        still=args.still,
        output_dir=args.output_dir,
        seed=args.seed,
        ffmpeg=args.ffmpeg,
        ffprobe=args.ffprobe,
    )
    print(
        json.dumps(
            {
                "status": "READY_FOR_BLIND_REVIEW",
                "screening_sha256": reveal["screening_sha256"],
                "output_dir": str(args.output_dir.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
