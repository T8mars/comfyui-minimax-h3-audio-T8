#!/usr/bin/env python3
"""Build a bounded, reproducible SLA Precision V2 versus dense AV review pack."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Iterator

import cv2
import numpy as np
from PIL import Image, ImageDraw


SCHEMA = "t8.minimax_h3.sla_precision_v2.pair_review.v1"
EXPECTED_TEXT = "你在干嘛呢我在这里呀看看效果如何"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _probe(path: Path, ffprobe: str) -> dict[str, Any]:
    result = _run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ]
    )
    payload = json.loads(result.stdout)
    streams = payload.get("streams") or []
    video = next(stream for stream in streams if stream.get("codec_type") == "video")
    audio = next(stream for stream in streams if stream.get("codec_type") == "audio")
    return {
        "width": int(video["width"]),
        "height": int(video["height"]),
        "frame_count": int(video.get("nb_frames") or 0),
        "video_codec": video.get("codec_name"),
        "video_duration_seconds": float(video.get("duration") or 0.0),
        "audio_codec": audio.get("codec_name"),
        "audio_sample_rate": int(audio["sample_rate"]),
        "audio_channels": int(audio["channels"]),
        "audio_duration_seconds": float(audio.get("duration") or 0.0),
    }


def _strict_decode(path: Path, ffmpeg: str) -> dict[str, bool]:
    commands = {
        "video": [ffmpeg, "-v", "error", "-xerror", "-threads", "1", "-i", str(path), "-map", "0:v:0", "-f", "null", "-"],
        "audio": [ffmpeg, "-v", "error", "-xerror", "-i", str(path), "-map", "0:a:0", "-f", "null", "-"],
        "combined": [ffmpeg, "-v", "error", "-xerror", "-threads", "1", "-i", str(path), "-f", "null", "-"],
    }
    return {
        name: subprocess.run(command, capture_output=True).returncode == 0
        for name, command in commands.items()
    }


def _normalize_text(value: str) -> str:
    return "".join(char for char in value if "\u4e00" <= char <= "\u9fff")


def _asr(path: Path, model_path: Path) -> dict[str, Any]:
    from faster_whisper import WhisperModel

    model = WhisperModel(
        str(model_path), device="cpu", compute_type="int8", local_files_only=True
    )
    segments, info = model.transcribe(
        str(path), language="zh", beam_size=5, word_timestamps=True, vad_filter=False
    )
    result = []
    for segment in segments:
        words = [
            {
                "word": word.word.strip(),
                "start": round(float(word.start), 3),
                "end": round(float(word.end), 3),
                "probability": round(float(word.probability), 6),
            }
            for word in (segment.words or [])
        ]
        result.append(
            {
                "start": round(float(segment.start), 3),
                "end": round(float(segment.end), 3),
                "text": segment.text.strip(),
                "words": words,
            }
        )
    text = "".join(item["text"] for item in result)
    normalized = _normalize_text(text)
    expected = _normalize_text(EXPECTED_TEXT)
    return {
        "language": info.language,
        "language_probability": round(float(info.language_probability), 6),
        "text": text,
        "normalized_text": normalized,
        "expected_text": expected,
        "exact_normalized_match": normalized == expected,
        "segments": result,
        "boundary": "ASR checks audible speech content, not phoneme-level lip correctness.",
    }


def _audio_stats(path: Path, ffmpeg: str) -> dict[str, Any]:
    result = subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-ac",
            "1",
            "-ar",
            "32000",
            "-f",
            "f32le",
            "-",
        ],
        check=True,
        capture_output=True,
    )
    samples = np.frombuffer(result.stdout, dtype="<f4")
    abs_samples = np.abs(samples)
    return {
        "sample_count_mono_32khz": int(samples.size),
        "peak_abs": round(float(abs_samples.max(initial=0.0)), 8),
        "rms": round(float(np.sqrt(np.mean(np.square(samples), dtype=np.float64))), 8),
        "clipped_sample_count_ge_0_999": int(np.count_nonzero(abs_samples >= 0.999)),
        "finite": bool(np.isfinite(samples).all()),
    }


def _video_diagnostics(path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    sharpness = []
    center_sharpness = []
    motion = []
    previous = None
    frame_count = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        center = gray[
            int(height * 0.10) : int(height * 0.95),
            int(width * 0.20) : int(width * 0.80),
        ]
        sharpness.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        center_sharpness.append(float(cv2.Laplacian(center, cv2.CV_64F).var()))
        if previous is not None:
            motion.append(float(np.mean(cv2.absdiff(gray, previous))))
        previous = gray
    capture.release()
    return {
        "decoded_frames": frame_count,
        "global_laplacian_variance": {
            "minimum": round(float(np.min(sharpness)), 6),
            "median": round(float(np.median(sharpness)), 6),
            "maximum": round(float(np.max(sharpness)), 6),
        },
        "central_subject_roi_laplacian_variance": {
            "minimum": round(float(np.min(center_sharpness)), 6),
            "median": round(float(np.median(center_sharpness)), 6),
            "maximum": round(float(np.max(center_sharpness)), 6),
        },
        "consecutive_frame_mean_abs_difference": {
            "minimum": round(float(np.min(motion)), 6),
            "median": round(float(np.median(motion)), 6),
            "maximum": round(float(np.max(motion)), 6),
        },
        "boundary": "Sharpness and frame-difference statistics are descriptive; they do not judge identity, anatomy, or aesthetics.",
    }


def _make_contact_sheet(path: Path, output: Path, *, columns: int = 8, rows: int = 4) -> None:
    capture = cv2.VideoCapture(str(path))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, max(0, frame_count - 1), columns * rows).round().astype(int)
    cells: list[Image.Image] = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"Could not read frame {index} from {path}")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb).resize((368, 208), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 96, 20), fill=(0, 0, 0))
        draw.text((4, 3), f"frame {index}", fill=(255, 255, 255))
        cells.append(image)
    capture.release()
    canvas = Image.new("RGB", (columns * 368, rows * 208), (20, 20, 20))
    for item, image in enumerate(cells):
        canvas.paste(image, ((item % columns) * 368, (item // columns) * 208))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=94)


def _prepare_syncnet_input(
    source: Path, output: Path, ffmpeg: str, *, delay_video_seconds: float = 0.0
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    base_filter = (
        "crop=416:416:(iw-416)/2:(ih-416)/2,"
        "scale=224:224:flags=lanczos,fps=25"
    )
    if delay_video_seconds > 0:
        duration = _probe(source, shutil.which("ffprobe") or "ffprobe")[
            "video_duration_seconds"
        ]
        base_filter += (
            f",tpad=start_mode=clone:start_duration={delay_video_seconds:.3f},"
            f"trim=duration={duration:.6f},setpts=PTS-STARTPTS"
        )
    _run(
        [
            ffmpeg,
            "-y",
            "-v",
            "error",
            "-i",
            str(source),
            "-vf",
            base_filter,
            "-map",
            "0:v:0",
            "-map",
            "0:a:0",
            "-c:v",
            "ffv1",
            "-c:a",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(output),
        ]
    )


@contextmanager
def _syncnet_import(syncnet_root: Path) -> Iterator[type]:
    sys.path.insert(0, str(syncnet_root))
    try:
        from SyncNetInstance import SyncNetInstance

        yield SyncNetInstance
    finally:
        sys.path.remove(str(syncnet_root))


def _syncnet_eval(
    prepared: Path, syncnet_root: Path, model_path: Path, work_root: Path, reference: str
) -> dict[str, Any]:
    class Options:
        batch_size = 20
        vshift = 15

    options = Options()
    options.reference = reference
    options.tmp_dir = str(work_root / "syncnet_tmp")
    Path(options.tmp_dir).mkdir(parents=True, exist_ok=True)
    with _syncnet_import(syncnet_root) as instance_type:
        instance = instance_type(device="cuda")
        instance.loadParameters(str(model_path))
        offset, confidence, distances = instance.evaluate(options, str(prepared))
    mean_distances = np.asarray(distances).mean(axis=0)
    return {
        "av_offset_frames_at_25fps": int(offset),
        "av_offset_milliseconds": int(offset) * 40,
        "min_distance": round(float(np.min(mean_distances)), 6),
        "confidence": round(float(confidence), 6),
    }


def _blind_html(output: Path) -> None:
    output.write_text(
        """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>SLA Precision V2 匿名 A/B</title>
<style>body{font-family:system-ui;background:#151515;color:#eee;margin:24px} .grid{display:grid;grid-template-columns:1fr 1fr;gap:20px} video{width:100%;background:#000} .note{color:#bbb;max-width:980px}</style></head>
<body><h1>SLA Precision V2 匿名 A/B</h1><p class="note">请分别正常速度、开声音播放 A 与 B。重点看人脸/轮廓是否在 1 秒后崩坏、动作是否重影、对白是否清晰、嘴型与声音是否同步。两条使用同输入、同 Seed、同 LoRA、同 8 NFE 与 12/3 shift；只改变注意力路由。</p>
<div class="grid"><section><h2>A</h2><video controls preload="metadata" src="A.mp4"></video></section><section><h2>B</h2><video controls preload="metadata" src="B.mp4"></video></section></div>
</body></html>""",
        encoding="utf-8",
    )


def _parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description=__doc__, parents=[])


def main(argv: list[str] | None = None) -> int:
    project = Path(__file__).resolve().parents[1]
    parser = _parser()
    parser.add_argument("--precision-video", type=Path, required=True)
    parser.add_argument("--dense-video", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=project / "artifacts" / "sla-precision-v2-pair-review-20260902",
    )
    parser.add_argument(
        "--whisper-model",
        type=Path,
        default=Path(r"F:\AI-T8-video-onekey\ComfyUI\models\TTS\faster-whisper-small-multilingual-536b0662"),
    )
    parser.add_argument(
        "--syncnet-root",
        type=Path,
        default=project / "artifacts" / "external-tools" / "syncnet_python",
    )
    args = parser.parse_args(argv)
    precision_video = args.precision_video.resolve()
    dense_video = args.dense_video.resolve()
    output_root = args.output_root.resolve()
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    model_path = args.syncnet_root / "data" / "syncnet_v2.model"
    required = [
        precision_video,
        dense_video,
        args.whisper_model,
        args.syncnet_root / "SyncNetInstance.py",
        model_path,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing or not ffmpeg or not ffprobe:
        print(json.dumps({"missing": missing, "ffmpeg": ffmpeg, "ffprobe": ffprobe}))
        return 2
    output_root.mkdir(parents=True, exist_ok=True)

    inputs = {"precision_v2": precision_video, "dense_control": dense_video}
    hashes = {name: _sha256(path) for name, path in inputs.items()}
    digest = hashlib.sha256((hashes["precision_v2"] + hashes["dense_control"]).encode()).digest()
    order = ["precision_v2", "dense_control"] if digest[0] % 2 == 0 else ["dense_control", "precision_v2"]
    mapping = {"A": order[0], "B": order[1]}
    for label, route in mapping.items():
        shutil.copy2(inputs[route], output_root / f"{label}.mp4")
    _blind_html(output_root / "blind_review.html")

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "created_at": _utc_now(),
        "status": "RUNNING",
        "contract": {
            "same_input_prompt_seed_base_lora_nfe_shifts": True,
            "only_intended_difference": "Precision V2 attention routing versus dense xformers control",
            "expected_dialogue": EXPECTED_TEXT,
        },
        "private_mapping": mapping,
        "routes": {},
    }
    for route, path in inputs.items():
        contact = output_root / f"{route}_contact_32.jpg"
        _make_contact_sheet(path, contact)
        prepared = output_root / f"{route}_syncnet_224x224_25fps.avi"
        shifted = output_root / f"{route}_syncnet_shifted_400ms.avi"
        _prepare_syncnet_input(path, prepared, ffmpeg)
        _prepare_syncnet_input(path, shifted, ffmpeg, delay_video_seconds=0.4)
        syncnet = _syncnet_eval(prepared, args.syncnet_root, model_path, output_root, route)
        negative = _syncnet_eval(
            shifted, args.syncnet_root, model_path, output_root, f"{route}_shifted"
        )
        report["routes"][route] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": hashes[route],
            "media": _probe(path, ffprobe),
            "strict_decode": _strict_decode(path, ffmpeg),
            "asr": _asr(path, args.whisper_model),
            "audio": _audio_stats(path, ffmpeg),
            "video_diagnostics": _video_diagnostics(path),
            "syncnet": syncnet,
            "syncnet_negative_control_400ms_video_delay": negative,
            "contact_sheet": str(contact),
        }

    checks: dict[str, bool] = {}
    for route, data in report["routes"].items():
        checks[f"{route}_strict_decode"] = all(data["strict_decode"].values())
        checks[f"{route}_124_frames"] = data["media"]["frame_count"] == 124
        checks[f"{route}_finite_audio"] = data["audio"]["finite"]
        checks[f"{route}_speech_recognized"] = bool(data["asr"]["normalized_text"])
        checks[f"{route}_syncnet_within_one_frame"] = abs(
            data["syncnet"]["av_offset_frames_at_25fps"]
        ) <= 1
        measured = data["syncnet_negative_control_400ms_video_delay"][
            "av_offset_frames_at_25fps"
        ]
        baseline = data["syncnet"]["av_offset_frames_at_25fps"]
        checks[f"{route}_syncnet_detects_400ms_control"] = measured - baseline >= 8
    report["checks"] = checks
    report["status"] = (
        "MECHANICAL_AV_PAIR_PASS_HUMAN_BLIND_REVIEW_PENDING"
        if all(checks.values())
        else "FAIL_MECHANICAL_AV_PAIR"
    )
    report["claim_boundary"] = (
        "Strict decode, ASR and calibrated SyncNet test bounded media integrity, audible "
        "dialogue and temporal AV offset. Full-speed human review is still required for "
        "phoneme plausibility, identity, anatomy, motion quality and preference."
    )
    (output_root / "private_mapping.json").write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_root / "pair_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
