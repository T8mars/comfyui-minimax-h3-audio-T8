#!/usr/bin/env python3
"""Build strict seam diagnostics and a blind review pack for the native-mask real A/B."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping

import cv2
import numpy as np
from PIL import Image, ImageDraw


SCHEMA = "t8.minimax_h3.native_masked_context.pair_review.v1"
ROUTES = ("soft_context", "hard_mask_plan_b")
FPS = 24
AUDIO_RATE = 32_000
WIDTH = 736
HEIGHT = 416
SEGMENT_ZERO_FRAMES = 124
CONTINUATION_FRAMES = 102


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command: list[str], *, input_bytes: bytes | None = None) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            command,
            input=input_bytes,
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as error:
        diagnostic = (error.stderr or b"").decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"command failed with exit {error.returncode}: {diagnostic}"
        ) from error


def _read_frames(path: Path) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise ValueError(f"no video frames decoded from {path}")
    return frames


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
    return json.loads(result.stdout.decode("utf-8"))


def _strict_decode(path: Path, ffmpeg: str) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    for label, maps in {
        "video": ["-map", "0:v:0"],
        "audio": ["-map", "0:a:0"],
        "combined": ["-map", "0:v:0", "-map", "0:a:0"],
    }.items():
        result = subprocess.run(
            [
                ffmpeg,
                "-v",
                "error",
                "-xerror",
                "-i",
                str(path),
                *maps,
                "-f",
                "null",
                "NUL",
            ],
            check=False,
            capture_output=True,
        )
        checks[label] = result.returncode == 0
    return checks


def _gray(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)


def _ssim(first: np.ndarray, second: np.ndarray) -> float:
    first_gray = _gray(first)
    second_gray = _gray(second)
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    mu_first = cv2.GaussianBlur(first_gray, (11, 11), 1.5)
    mu_second = cv2.GaussianBlur(second_gray, (11, 11), 1.5)
    sigma_first = cv2.GaussianBlur(first_gray * first_gray, (11, 11), 1.5) - (
        mu_first * mu_first
    )
    sigma_second = cv2.GaussianBlur(second_gray * second_gray, (11, 11), 1.5) - (
        mu_second * mu_second
    )
    sigma_cross = cv2.GaussianBlur(first_gray * second_gray, (11, 11), 1.5) - (
        mu_first * mu_second
    )
    numerator = (2.0 * mu_first * mu_second + c1) * (2.0 * sigma_cross + c2)
    denominator = (mu_first * mu_first + mu_second * mu_second + c1) * (
        sigma_first + sigma_second + c2
    )
    return float(np.mean(numerator / np.maximum(denominator, 1e-12)))


def _flow(first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    flow = cv2.calcOpticalFlowFarneback(
        _gray(first),
        _gray(second),
        None,
        0.5,
        3,
        21,
        5,
        7,
        1.5,
        0,
    )
    border_y = max(1, flow.shape[0] // 10)
    border_x = max(1, flow.shape[1] // 10)
    core = flow[border_y:-border_y, border_x:-border_x]
    magnitude = np.linalg.norm(core, axis=2)
    return {
        "mean_dx": float(np.mean(core[..., 0])),
        "mean_dy": float(np.mean(core[..., 1])),
        "median_magnitude": float(np.median(magnitude)),
        "mean_magnitude": float(np.mean(magnitude)),
        "p90_magnitude": float(np.percentile(magnitude, 90)),
    }


def _vector_distance(first: Mapping[str, float], second: Mapping[str, float]) -> float:
    return math.hypot(
        float(first["mean_dx"]) - float(second["mean_dx"]),
        float(first["mean_dy"]) - float(second["mean_dy"]),
    )


def seam_video_metrics(
    segment_zero_frames: list[np.ndarray], continuation_frames: list[np.ndarray]
) -> dict[str, Any]:
    if len(segment_zero_frames) < 3 or len(continuation_frames) < 3:
        raise ValueError("seam metrics require at least three frames on both sides")
    last = segment_zero_frames[-1]
    first = continuation_frames[0]
    before = _flow(segment_zero_frames[-2], last)
    across = _flow(last, first)
    after = _flow(first, continuation_frames[1])
    difference = last.astype(np.float32) - first.astype(np.float32)
    return {
        "boundary_frame_mae_0_to_1": float(np.mean(np.abs(difference)) / 255.0),
        "boundary_frame_rmse_0_to_1": float(np.sqrt(np.mean(difference * difference)) / 255.0),
        "boundary_frame_ssim": _ssim(last, first),
        "flow_before": before,
        "flow_across_boundary": across,
        "flow_after": after,
        "flow_vector_jump_from_before": _vector_distance(before, across),
        "flow_vector_jump_to_after": _vector_distance(across, after),
        "flow_magnitude_ratio_to_before": float(
            across["mean_magnitude"] / max(before["mean_magnitude"], 1e-9)
        ),
        "flow_magnitude_ratio_to_after": float(
            across["mean_magnitude"] / max(after["mean_magnitude"], 1e-9)
        ),
        "interpretation_boundary": (
            "These are descriptive seam discontinuity proxies. Lower pixel or flow deltas can "
            "also mean frozen motion, so they do not select a winner without full-speed review."
        ),
    }


def _decode_audio(path: Path, ffmpeg: str) -> np.ndarray:
    result = _run(
        [
            ffmpeg,
            "-v",
            "error",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-ar",
            str(AUDIO_RATE),
            "-ac",
            "2",
            "pipe:1",
        ]
    )
    audio = np.frombuffer(result.stdout, dtype="<f4")
    if audio.size == 0 or audio.size % 2:
        raise ValueError(f"invalid decoded stereo audio from {path}")
    return audio.reshape(-1, 2)


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(values.astype(np.float64) ** 2)))


def _spectral_cosine(first: np.ndarray, second: np.ndarray) -> float:
    length = min(len(first), len(second))
    if length < 16:
        return 0.0
    window = np.hanning(length).astype(np.float32)[:, None]
    first_spectrum = np.abs(np.fft.rfft(first[-length:] * window, axis=0)).mean(axis=1)
    second_spectrum = np.abs(np.fft.rfft(second[:length] * window, axis=0)).mean(axis=1)
    denominator = float(np.linalg.norm(first_spectrum) * np.linalg.norm(second_spectrum))
    return float(np.dot(first_spectrum, second_spectrum) / max(denominator, 1e-12))


def audio_quality_metrics(audio: np.ndarray) -> dict[str, Any]:
    """Return descriptive clipping, DC, high-band and noise-likeness proxies."""
    if audio.ndim != 2 or audio.shape[1] != 2 or len(audio) < 2048:
        raise ValueError("audio quality metrics require at least 2048 stereo samples")
    mono = audio.astype(np.float64).mean(axis=1)
    dc_offset = float(np.mean(mono))
    centered = mono - dc_offset
    rms = _rms(centered)
    peak = float(np.max(np.abs(audio)))
    frame_size = 2048
    hop = 1024
    window = np.hanning(frame_size)
    spectra = []
    for start in range(0, len(centered) - frame_size + 1, hop):
        frame = centered[start : start + frame_size] * window
        spectra.append(np.abs(np.fft.rfft(frame)) ** 2)
    power = np.stack(spectra)
    frequencies = np.fft.rfftfreq(frame_size, d=1.0 / AUDIO_RATE)
    high_band = frequencies >= 10_000.0
    high_band_ratio = float(
        np.sum(power[:, high_band]) / max(float(np.sum(power)), 1e-24)
    )
    positive_power = power[:, 1:] + 1e-24
    flatness = np.exp(np.mean(np.log(positive_power), axis=1)) / np.mean(
        positive_power, axis=1
    )
    return {
        "peak_abs": peak,
        "rms": rms,
        "dc_offset": dc_offset,
        "clipping_sample_fraction_at_0p999": float(np.mean(np.abs(audio) >= 0.999)),
        "crest_factor_db": float(20.0 * math.log10(max(peak, 1e-12) / max(rms, 1e-12))),
        "high_band_10k_to_16k_energy_ratio": high_band_ratio,
        "mean_spectral_flatness": float(np.mean(flatness)),
        "finite": bool(np.isfinite(audio).all()),
        "interpretation_boundary": (
            "Clipping and DC are direct signal checks. High-band energy and spectral flatness "
            "are only descriptive hiss/noise-likeness proxies; percussion and synth textures "
            "can also raise them, so human listening remains required."
        ),
    }


def audio_seam_metrics(segment_zero: np.ndarray, continuation: np.ndarray) -> dict[str, Any]:
    if len(segment_zero) < AUDIO_RATE // 5 or len(continuation) < AUDIO_RATE // 5:
        raise ValueError("audio seam metrics require at least 200ms on both sides")
    short = round(AUDIO_RATE * 0.05)
    long = round(AUDIO_RATE * 0.20)
    tail_rms = _rms(segment_zero[-short:])
    head_rms = _rms(continuation[:short])
    step = continuation[0] - segment_zero[-1]
    return {
        "boundary_sample_step_peak": float(np.max(np.abs(step))),
        "boundary_sample_step_rms": _rms(step),
        "tail_50ms_rms": tail_rms,
        "head_50ms_rms": head_rms,
        "head_to_tail_rms_ratio": float(head_rms / max(tail_rms, 1e-12)),
        "head_minus_tail_db": float(20.0 * math.log10(max(head_rms, 1e-12) / max(tail_rms, 1e-12))),
        "spectral_cosine_200ms": _spectral_cosine(
            segment_zero[-long:], continuation[:long]
        ),
        "finite": bool(np.isfinite(segment_zero).all() and np.isfinite(continuation).all()),
        "interpretation_boundary": (
            "Waveform and spectrum continuity are descriptive only; listen at normal volume "
            "because ambience changes can be valid and AAC is lossy."
        ),
    }


def _detect_faces(detector, frame: np.ndarray) -> list[np.ndarray]:
    detector.setInputSize((frame.shape[1], frame.shape[0]))
    _, faces = detector.detect(frame)
    if faces is None or len(faces) == 0:
        return []
    return [np.asarray(face, dtype=np.float32) for face in faces]


def _face_box_iou(first: np.ndarray, second: np.ndarray) -> float:
    left = max(float(first[0]), float(second[0]))
    top = max(float(first[1]), float(second[1]))
    right = min(float(first[0] + first[2]), float(second[0] + second[2]))
    bottom = min(float(first[1] + first[3]), float(second[1] + second[3]))
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, float(first[2])) * max(0.0, float(first[3]))
    second_area = max(0.0, float(second[2])) * max(0.0, float(second[3]))
    return intersection / max(first_area + second_area - intersection, 1e-9)


def _select_face_near_reference(
    faces: list[np.ndarray], reference: np.ndarray | None
) -> np.ndarray | None:
    if not faces:
        return None
    if reference is None:
        return max(faces, key=lambda face: float(face[2] * face[3]))
    reference_center = np.array(
        [reference[0] + reference[2] / 2.0, reference[1] + reference[3] / 2.0],
        dtype=np.float32,
    )
    reference_scale = max(float(np.hypot(reference[2], reference[3])), 1.0)
    reference_area = max(float(reference[2] * reference[3]), 1.0)

    def rank(face: np.ndarray) -> tuple[float, float, float, float]:
        center = np.array(
            [face[0] + face[2] / 2.0, face[1] + face[3] / 2.0], dtype=np.float32
        )
        center_distance = float(np.linalg.norm(center - reference_center)) / reference_scale
        area_ratio_error = abs(math.log(max(float(face[2] * face[3]), 1.0) / reference_area))
        confidence = float(face[-1]) if len(face) > 14 else 0.0
        return (
            _face_box_iou(reference, face),
            -center_distance,
            -area_ratio_error,
            confidence,
        )

    return max(faces, key=rank)


def _track_faces(
    detector, frames: list[np.ndarray], reference: np.ndarray | None = None
) -> tuple[list[np.ndarray | None], list[int]]:
    selected: list[np.ndarray | None] = []
    candidate_counts: list[int] = []
    current = reference
    for frame in frames:
        faces = _detect_faces(detector, frame)
        candidate_counts.append(len(faces))
        face = _select_face_near_reference(faces, current)
        selected.append(face)
        if face is not None:
            current = face
    return selected, candidate_counts


def _face_boundary(
    segment_zero_frames: list[np.ndarray],
    continuation_frames: list[np.ndarray],
    yunet_model: Path,
    sface_model: Path,
) -> dict[str, Any]:
    if not yunet_model.is_file() or not sface_model.is_file():
        return {
            "available": False,
            "reason": "pinned YuNet or SFace model is missing",
        }
    detector = cv2.FaceDetectorYN.create(
        str(yunet_model), "", (WIDTH, HEIGHT), 0.35, 0.3, 5000
    )
    recognizer = cv2.FaceRecognizerSF.create(str(sface_model), "")
    tail_frames = segment_zero_frames[-5:]
    head_frames = continuation_frames[:5]
    tail_faces, tail_candidate_counts = _track_faces(detector, tail_frames)
    head_faces, head_candidate_counts = _track_faces(
        detector, head_frames, reference=tail_faces[-1]
    )
    left_face = tail_faces[-1]
    right_face = head_faces[0]
    similarity = None
    if left_face is not None and right_face is not None:
        left_feature = recognizer.feature(
            recognizer.alignCrop(tail_frames[-1], left_face)
        )
        right_feature = recognizer.feature(
            recognizer.alignCrop(head_frames[0], right_face)
        )
        similarity = float(
            recognizer.match(left_feature, right_feature, cv2.FaceRecognizerSF_FR_COSINE)
        )
    return {
        "available": True,
        "tail_detected_frames": sum(face is not None for face in tail_faces),
        "head_detected_frames": sum(face is not None for face in head_faces),
        "tail_candidate_counts": tail_candidate_counts,
        "head_candidate_counts": head_candidate_counts,
        "selection": "continuity_tracked_from_segment_tail",
        "boundary_faces_both_detected": left_face is not None and right_face is not None,
        "boundary_sface_cosine": similarity,
        "interpretation_boundary": (
            "YuNet detection and SFace cosine are descriptive and do not verify human identity, "
            "anatomy, or visual quality."
        ),
    }


def _contact_sheet(
    segment_zero_frames: list[np.ndarray],
    route_frames: Mapping[str, list[np.ndarray]],
    route_order: list[str],
    output: Path,
) -> None:
    selected_left = list(enumerate(segment_zero_frames[-4:], start=-4))
    cell_width, cell_height = 276, 156
    labels = [f"S0 {index}" for index, _ in selected_left] + [
        f"CONT +{index}" for index in range(5)
    ]
    canvas = Image.new(
        "RGB", (cell_width * len(labels), cell_height * len(route_order) + 32), (18, 18, 18)
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), "Same segment-0 tail | continuation begins after red line", fill="white")
    for row, route in enumerate(route_order):
        frames = [frame for _, frame in selected_left] + route_frames[route][:5]
        for column, (label, frame) in enumerate(zip(labels, frames, strict=True)):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb).resize(
                (cell_width, cell_height), Image.Resampling.LANCZOS
            )
            x = column * cell_width
            y = 32 + row * cell_height
            canvas.paste(image, (x, y))
            draw.rectangle((x, y, x + 92, y + 18), fill=(0, 0, 0))
            draw.text((x + 3, y + 2), label, fill="white")
            if column == len(selected_left):
                draw.line((x, y, x, y + cell_height), fill=(255, 50, 50), width=5)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=95)


def _concat_review(segment_zero: Path, continuation: Path, output: Path, ffmpeg: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            ffmpeg,
            "-y",
            "-v",
            "error",
            "-i",
            str(segment_zero),
            "-i",
            str(continuation),
            "-filter_complex",
            (
                "[0:v]fps=24,format=yuv420p,setpts=PTS-STARTPTS[v0];"
                "[0:a]aresample=32000,asetpts=PTS-STARTPTS[a0];"
                "[1:v]fps=24,format=yuv420p,setpts=PTS-STARTPTS[v1];"
                "[1:a]aresample=32000,asetpts=PTS-STARTPTS[a1];"
                "[v0][a0][v1][a1]concat=n=2:v=1:a=1[v][a]"
            ),
            "-filter_threads",
            "1",
            "-filter_complex_threads",
            "1",
            "-map",
            "[v]",
            "-map",
            "[a]",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-threads:v",
            "1",
            "-crf",
            "16",
            "-pix_fmt",
            "yuv420p",
            "-x264-params",
            "keyint=1:min-keyint=1:scenecut=0",
            "-bf",
            "0",
            "-coder",
            "0",
            "-c:a",
            "aac",
            "-threads:a",
            "1",
            "-b:a",
            "192k",
            "-ar",
            str(AUDIO_RATE),
            "-ac",
            "2",
            "-movflags",
            "+faststart",
            str(output),
        ]
    )


def _video_stream_summary(path: Path, ffmpeg: str, ffprobe: str) -> dict[str, Any]:
    probe = _probe(path, ffprobe)
    stream = next(
        (item for item in probe.get("streams", []) if item.get("codec_type") == "video"),
        None,
    )
    if stream is None:
        raise ValueError(f"review transport has no video stream: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
        "frames": int(stream.get("nb_frames") or 0),
        "fps": stream.get("avg_frame_rate"),
        "strict_video_decode": subprocess.run(
            [
                ffmpeg,
                "-v",
                "error",
                "-xerror",
                "-i",
                str(path),
                "-map",
                "0:v:0",
                "-f",
                "null",
                "NUL",
            ],
            check=False,
            capture_output=True,
        ).returncode
        == 0,
    }


def _build_review_transports(
    output_root: Path,
    ffmpeg: str,
    ffprobe: str,
    *,
    source_width: int,
    source_height: int,
) -> dict[str, Any]:
    first = output_root / "A.mp4"
    second = output_root / "B.mp4"
    full = output_root / "AB_side_by_side.mp4"
    focus = output_root / "AB_seam_focus_2s.mp4"
    face = output_root / "AB_seam_face_zoom_2s.mp4"
    common = [
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-threads:v",
        "1",
        "-crf",
        "16",
        "-pix_fmt",
        "yuv420p",
        "-x264-params",
        "keyint=1:min-keyint=1:scenecut=0",
        "-bf",
        "0",
        "-coder",
        "0",
        "-movflags",
        "+faststart",
    ]
    _run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(first),
            "-i",
            str(second),
            "-filter_complex",
            "[0:v]setpts=PTS-STARTPTS[a];[1:v]setpts=PTS-STARTPTS[b];[a][b]hstack=inputs=2[v]",
            "-map",
            "[v]",
            "-frames:v",
            str(SEGMENT_ZERO_FRAMES + CONTINUATION_FRAMES),
            *common,
            str(full),
        ]
    )
    seam_first = SEGMENT_ZERO_FRAMES - 16
    seam_stop = seam_first + 48
    trim = f"trim=start_frame={seam_first}:end_frame={seam_stop},setpts=PTS-STARTPTS"
    face_size = min(480, source_width, source_height)
    face_size -= face_size % 2
    face_x = (source_width - face_size) // 2
    face_y = (source_height - face_size) // 2
    _run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(first),
            "-i",
            str(second),
            "-filter_complex",
            f"[0:v]{trim}[a];[1:v]{trim}[b];[a][b]hstack=inputs=2[v]",
            "-map",
            "[v]",
            "-frames:v",
            "48",
            *common,
            str(focus),
        ]
    )
    _run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(first),
            "-i",
            str(second),
            "-filter_complex",
            (
                f"[0:v]{trim},crop={face_size}:{face_size}:{face_x}:{face_y}[a];"
                f"[1:v]{trim},crop={face_size}:{face_size}:{face_x}:{face_y}[b];"
                "[a][b]hstack=inputs=2[v]"
            ),
            "-map",
            "[v]",
            "-frames:v",
            "48",
            *common,
            str(face),
        ]
    )
    report = {
        "schema": "t8.minimax_h3.native_masked_context.review_transport.v2",
        "created_at": _utc_now(),
        "layout": {"left": "A", "right": "B", "mapping_disclosed": False},
        "source_frame_window": [seam_first, seam_stop - 1],
        "boundary_source_frames": [SEGMENT_ZERO_FRAMES - 1, SEGMENT_ZERO_FRAMES],
        "boundary_local_frames": [15, 16],
        "full": _video_stream_summary(full, ffmpeg, ffprobe),
        "focus": _video_stream_summary(focus, ffmpeg, ffprobe),
        "face": _video_stream_summary(face, ffmpeg, ffprobe),
        "face_crop_per_route": {
            "x": face_x,
            "y": face_y,
            "width": face_size,
            "height": face_size,
        },
        "resampling_or_ai_processing": False,
        "claim_boundary": (
            "These anonymous transports preserve source order and exact frame windows for review; "
            "H.264 transport encoding and crops do not rank the routes or prove subjective quality."
        ),
    }
    report["checks"] = {
        "full_exact_frames_and_dimensions": report["full"]["frames"] == 226
        and report["full"]["width"] == source_width * 2
        and report["full"]["height"] == source_height,
        "focus_exact_frames_and_dimensions": report["focus"]["frames"] == 48
        and report["focus"]["width"] == source_width * 2
        and report["focus"]["height"] == source_height,
        "face_exact_frames_and_dimensions": report["face"]["frames"] == 48
        and report["face"]["width"] == face_size * 2
        and report["face"]["height"] == face_size,
        "all_strict_video_decode": all(
            report[name]["strict_video_decode"] for name in ("full", "focus", "face")
        ),
    }
    (output_root / "review_transport_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def blind_mapping(soft_sha: str, hard_sha: str) -> dict[str, str]:
    digest = hashlib.sha256((soft_sha + hard_sha).encode("ascii")).digest()
    order = list(ROUTES) if digest[0] % 2 == 0 else list(reversed(ROUTES))
    return {"A": order[0], "B": order[1]}


def _resource_claim(resource_pass: bool) -> str:
    if resource_pass:
        return (
            "Both continuation routes kept at least the project's 512 MiB free-VRAM "
            "margin in this exact local run. This bounded pass does not establish "
            "general 16GB safety across other resolutions, models, workflows, or hosts."
        )
    return (
        "This exact local run completed, but at least one continuation route fell below "
        "the project's 512 MiB free-VRAM margin; do not claim general 16GB safety."
    )


def _blind_html(
    output: Path,
    boundary_frame: int,
    *,
    audio_profile: str,
    color_match_enabled: bool = False,
) -> None:
    if audio_profile == "instrumental_music":
        soundtrack_note = (
            "本轮声音由 H3 原生生成纯器乐合成器背景音乐；没有后期降噪、直流修正、外部音乐覆盖或音频接缝淡化。"
        )
        scene_checks = "人物和环境是否跳变"
    elif audio_profile == "classical_mandarin_speech":
        soundtrack_note = (
            "本轮声音由 H3 原生生成；第0段要求人物只说一次“你在哪里”，续段要求人物静默、只延续同一段"
            "大提琴与钢琴古典音乐。没有后期降噪、外部音轨覆盖或音频接缝淡化。"
        )
        scene_checks = "人物面部、肤色、灯光和背景是否跳变"
    else:
        soundtrack_note = "本轮声音由 H3 原生生成；没有后期降噪、外部音轨覆盖或音频接缝淡化。"
        scene_checks = "人物和环境是否跳变"
    color_note = (
        "本轮 A/B 两条都启用同一个默认 Color Match：比较上一段末尾与续段开头各最多5帧，"
        "先统一全局 Lab 色彩/对比度，再用8x5局部分区补偿左、中、右不同区域的色差；"
        "每像素通道总改变量最大0.02，并在24帧内渐隐。请重点判断原来的颜色跳变是否消失，"
        "以及是否新增不自然偏色或缓慢回色。Color Match不改音频或原生latent。"
        if color_match_enabled
        else "本轮没有启用新的接缝 Color Match。"
    )
    output.write_text(
        f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>H3 原生 Mask Plan B 匿名接缝 A/B</title>
<style>body{{font-family:system-ui;background:#151515;color:#eee;margin:24px}} .grid{{display:grid;grid-template-columns:1fr 1fr;gap:20px}} video{{width:100%;background:#000}} .wide{{display:block;max-width:1440px}} .zoom{{display:block;max-width:960px}} img{{max-width:100%}} .note{{color:#bbb;max-width:1200px}} code{{color:#ffd166}} .frame-controls,.verdict-controls{{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:10px 0 18px}} button{{border:1px solid #666;border-radius:6px;background:#292929;color:#fff;padding:8px 12px;cursor:pointer}} button:hover,button:focus,button[aria-pressed="true"]{{border-color:#ffd166}} .frame-status,.verdict-status{{color:#ffd166;min-width:220px}} .verdict-panel{{max-width:920px;margin:28px 0;padding:16px 20px;border:1px solid #555;border-radius:8px;background:#1d1d1d}}</style></head>
<body><h1>H3 原生 Mask Plan B 匿名接缝 A/B</h1>
<p class="note">{soundtrack_note}</p>
<p class="note">{color_note}</p>
<p class="note">A 与 B 共用完全相同的第 0 段、续段 prompt、Seed、模型、LoRA、4 NFE 和 12/3 shift；唯一差异是续段是否经过 Plan B 原生画面硬 mask。接缝位于完整视频第 <code>{boundary_frame}</code> 帧之后（约 5.167 秒）。请打开声音，以正常速度和逐帧两种方式观察：{scene_checks}，运动方向和镜头速度是否顿挫，是否出现冻结、重影或溶解，以及声音边界是否有爆音或氛围突变。允许“两条差不多”。</p>
<h2>脸部接缝 2 秒局部循环</h2><p class="note">左边 A、右边 B；各自直接裁出480×480区域，无AI重绘、插帧或锐化。</p>
<video id="faceZoom" class="zoom" controls autoplay muted loop playsinline preload="auto" src="AB_seam_face_zoom_2s.mp4"></video>
<div class="frame-controls"><button type="button" data-step="-1">上一帧</button><button type="button" data-frame="15">接缝前123</button><button type="button" data-frame="16">接缝后124</button><button type="button" data-step="1">下一帧</button><button type="button" data-rate="0.5">0.5倍</button><button type="button" data-rate="1">1倍</button><span id="frameStatus" class="frame-status">正在1倍循环播放</span></div>
<h2>全画面接缝 2 秒同屏循环</h2><video class="wide" controls muted loop playsinline preload="auto" src="AB_seam_focus_2s.mp4"></video>
<h2>完整同屏对照</h2><video class="wide" controls muted playsinline preload="metadata" src="AB_side_by_side.mp4"></video>
<div class="grid"><section><h2>A</h2><video controls preload="metadata" src="A.mp4"></video></section><section><h2>B</h2><video controls preload="metadata" src="B.mp4"></video></section></div>
<section class="verdict-panel"><h2>你的最终匿名裁决</h2><p class="note">看完后点一次；这里只记录匿名选择，不显示真实路线。</p><div class="verdict-controls"><button type="button" data-verdict="A" aria-pressed="false">A更好</button><button type="button" data-verdict="B" aria-pressed="false">B更好</button><button type="button" data-verdict="tie" aria-pressed="false">两条差不多</button><span id="verdictStatus" class="verdict-status">尚未选择</span></div></section>
<h2>匿名接缝帧条</h2><p class="note">上、下两行对应 A/B；红线右侧是续段第一帧。路线映射保存在私有 JSON，不在本页显示。</p><img src="blind_seam_contact.jpg" alt="blind seam contact sheet">
<script>(()=>{{const v=document.getElementById('faceZoom'),c=document.querySelector('.frame-controls'),s=document.getElementById('frameStatus'),fps=24;c.addEventListener('click',e=>{{const b=e.target.closest('button');if(!b)return;if(b.dataset.frame!==undefined){{v.pause();v.currentTime=Number(b.dataset.frame)/fps+0.0005;v.dataset.frame=b.dataset.frame;s.textContent=`已暂停：源第${{108+Number(b.dataset.frame)}}帧`;return}}if(b.dataset.step!==undefined){{const f=v.dataset.frame===''||v.dataset.frame===undefined?Math.floor(v.currentTime*fps):Number(v.dataset.frame);const n=Math.max(0,Math.min(47,f+Number(b.dataset.step)));v.pause();v.currentTime=n/fps+0.0005;v.dataset.frame=String(n);s.textContent=`已暂停：源第${{108+n}}帧`;return}}if(b.dataset.rate!==undefined){{v.dataset.frame='';v.playbackRate=Number(b.dataset.rate);v.play();s.textContent=`正在${{b.dataset.rate}}倍循环播放`}}}})}})();</script>
<script>(()=>{{const labels={{A:'A更好',B:'B更好',tie:'两条差不多'}},c=document.querySelector('.verdict-controls'),s=document.getElementById('verdictStatus');const record=(x,u=true)=>{{if(!(x in labels))return;c.querySelectorAll('button').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.verdict===x)));s.textContent=`已记录匿名选择：${{labels[x]}}。现在可以直接回到对话。`;document.title=`盲测裁决：${{labels[x]}}`;if(u)history.replaceState(null,'',`#verdict=${{encodeURIComponent(x)}}`)}};c.addEventListener('click',e=>{{const b=e.target.closest('button[data-verdict]');if(b)record(b.dataset.verdict)}});const x=new URLSearchParams(location.hash.slice(1)).get('verdict');if(x)record(x,false)}})();</script>
</body></html>""",
        encoding="utf-8",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--comfy-root", type=Path, default=Path(r"F:\AI-T8-video-onekey\ComfyUI")
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    run_root = args.run_root.resolve()
    output_root = (
        args.output_root.resolve()
        if args.output_root
        else run_root.parent.parent / "native-masked-context-pair-review-20260902"
    )
    comfy_root = args.comfy_root.resolve()
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise FileNotFoundError("ffmpeg and ffprobe are required")
    run_report_path = run_root / "run_report.json"
    run_report = json.loads(run_report_path.read_text(encoding="utf-8"))
    if run_report.get("status") != "REAL_PAIR_COMPLETE_ANALYSIS_PENDING":
        raise ValueError("run_report is not a completed real pair")
    segment_zero = Path(run_report["phases"]["segment0"]["artifact_video"])
    continuation_paths = {
        route: Path(run_report["phases"][route]["artifact_video"]) for route in ROUTES
    }
    required = [segment_zero, *continuation_paths.values()]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing real pair media: {missing}")
    output_root.mkdir(parents=True, exist_ok=True)

    segment_frames = _read_frames(segment_zero)
    route_frames = {route: _read_frames(path) for route, path in continuation_paths.items()}
    segment_audio = _decode_audio(segment_zero, ffmpeg)
    route_audio = {route: _decode_audio(path, ffmpeg) for route, path in continuation_paths.items()}
    yunet = comfy_root / "models" / "face_detection" / "face_detection_yunet_2023mar.onnx"
    sface = comfy_root / "models" / "face_detection" / "face_recognition_sface_2021dec.onnx"

    full_paths: dict[str, Path] = {}
    routes: dict[str, Any] = {}
    for route in ROUTES:
        full = output_root / f"{route}_full.mp4"
        _concat_review(segment_zero, continuation_paths[route], full, ffmpeg)
        full_paths[route] = full
        full_probe = _probe(full, ffprobe)
        full_strict_decode = _strict_decode(full, ffmpeg)
        routes[route] = {
            "continuation": str(continuation_paths[route]),
            "continuation_sha256": _sha256(continuation_paths[route]),
            "continuation_frames": len(route_frames[route]),
            "video_seam": seam_video_metrics(segment_frames, route_frames[route]),
            "audio_seam": audio_seam_metrics(segment_audio, route_audio[route]),
            "audio_quality": audio_quality_metrics(route_audio[route]),
            "face_boundary": _face_boundary(segment_frames, route_frames[route], yunet, sface),
            "color_match": run_report["phases"][route].get("color_match"),
            "full_review_video": str(full),
            "full_review_sha256": _sha256(full),
            "full_review_probe": full_probe,
            "full_review_strict_decode": full_strict_decode,
        }

    soft_frames = route_frames["soft_context"]
    hard_frames = route_frames["hard_mask_plan_b"]
    pair_mae = [
        float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))) / 255.0)
        for a, b in zip(soft_frames, hard_frames, strict=True)
    ]
    pair_ssim = [_ssim(a, b) for a, b in zip(soft_frames, hard_frames, strict=True)]
    soft_audio = route_audio["soft_context"]
    hard_audio = route_audio["hard_mask_plan_b"]
    audio_length = min(len(soft_audio), len(hard_audio))
    pair_difference = {
        "continuation_framewise_mae_mean": float(np.mean(pair_mae)),
        "continuation_framewise_mae_max": float(np.max(pair_mae)),
        "continuation_framewise_ssim_mean": float(np.mean(pair_ssim)),
        "continuation_framewise_ssim_min": float(np.min(pair_ssim)),
        "continuation_audio_pcm_rmse": _rms(
            soft_audio[:audio_length] - hard_audio[:audio_length]
        ),
        "identical_decoded_video": routes["soft_context"]["continuation_sha256"]
        == routes["hard_mask_plan_b"]["continuation_sha256"],
        "interpretation_boundary": (
            "This proves the routes produced different outputs; it does not rank their quality."
        ),
    }

    mapping = blind_mapping(
        routes["soft_context"]["full_review_sha256"],
        routes["hard_mask_plan_b"]["full_review_sha256"],
    )
    for label, route in mapping.items():
        shutil.copy2(full_paths[route], output_root / f"{label}.mp4")
    blind_route_order = [mapping[label] for label in ("A", "B")]
    ordered_frames = {route: route_frames[route] for route in blind_route_order}
    temporary_contact = output_root / "route_seam_contact.jpg"
    _contact_sheet(segment_frames, ordered_frames, blind_route_order, temporary_contact)
    blind_contact = output_root / "blind_seam_contact.jpg"
    image = Image.open(temporary_contact).convert("RGB")
    draw = ImageDraw.Draw(image)
    row_height = 156
    draw.rectangle((0, 32, 54, 32 + row_height), fill=(0, 0, 0))
    draw.text((8, 42), "A", fill=(255, 220, 80))
    draw.rectangle((0, 32 + row_height, 54, 32 + 2 * row_height), fill=(0, 0, 0))
    draw.text((8, 42 + row_height), "B", fill=(255, 220, 80))
    image.save(blind_contact, quality=95)
    temporary_contact.unlink()
    review_transport = _build_review_transports(
        output_root,
        ffmpeg,
        ffprobe,
        source_width=int(run_report["contract"]["width"]),
        source_height=int(run_report["contract"]["height"]),
    )
    color_match_enabled = bool(
        run_report.get("contract", {}).get("color_match", {}).get("enabled")
    )
    _blind_html(
        output_root / "blind_review.html",
        SEGMENT_ZERO_FRAMES - 1,
        audio_profile=str(run_report.get("contract", {}).get("audio_profile", "unknown")),
        color_match_enabled=color_match_enabled,
    )

    checks = {
        "run_pair_complete": run_report.get("status") == "REAL_PAIR_COMPLETE_ANALYSIS_PENDING",
        "shared_context_unchanged": bool(run_report["context"].get("unchanged_during_pair")),
        "shared_color_reference_unchanged": bool(
            run_report.get("color_reference", {}).get("unchanged_during_pair")
        ),
        "segment_zero_exact_124_frames": len(segment_frames) == SEGMENT_ZERO_FRAMES,
        "both_continuations_exact_102_frames": all(
            len(route_frames[route]) == CONTINUATION_FRAMES for route in ROUTES
        ),
        "all_source_media_strict_decode": all(
            run_report["phases"][route]["media"]["strict_decode_passed"]
            for route in ("segment0", *ROUTES)
        ),
        "both_audio_streams_finite": all(
            routes[route]["audio_seam"]["finite"] for route in ROUTES
        ),
        "both_full_reviews_strict_decode": all(
            all(routes[route]["full_review_strict_decode"].values()) for route in ROUTES
        ),
        "both_full_reviews_exact_226_frames": all(
            any(
                stream.get("codec_type") == "video"
                and int(stream.get("nb_frames") or 0)
                == SEGMENT_ZERO_FRAMES + CONTINUATION_FRAMES
                for stream in routes[route]["full_review_probe"].get("streams", [])
            )
            for route in ROUTES
        ),
        "plan_b_node_executed": any(
            event.get("node") == "18"
            for event in json.loads(
                (run_root / "hard_mask_plan_b" / "phase.json").read_text(encoding="utf-8")
            ).get("events", [])
        ),
        "soft_route_has_no_plan_b_node": "18"
        not in json.loads(
            (run_root / "soft_context" / "prompt.json").read_text(encoding="utf-8")
        ),
        "color_match_default_enabled_for_pair": color_match_enabled,
        "color_match_executed_both_routes": all(
            run_report["phases"][route]
            .get("color_match", {})
            .get("report", {})
            .get("status")
            == "COLOR_MATCH_APPLIED"
            for route in ROUTES
        ),
        "color_match_reduced_rgb_jump_both_routes": all(
            float(
                run_report["phases"][route]["color_match"]["report"][
                    "maximum_rgb_jump_after"
                ]
            )
            < float(
                run_report["phases"][route]["color_match"]["report"][
                    "maximum_rgb_jump_before"
                ]
            )
            for route in ROUTES
        ),
        "color_match_reduced_spatial_rgb_jump_both_routes": all(
            float(
                run_report["phases"][route]["color_match"]["report"][
                    "maximum_spatial_rgb_jump_after"
                ]
            )
            < float(
                run_report["phases"][route]["color_match"]["report"][
                    "maximum_spatial_rgb_jump_before"
                ]
            )
            for route in ROUTES
        ),
        "color_match_used_v2_lab_spatial_method_both_routes": all(
            run_report["phases"][route]["color_match"]["report"]["method"]
            == "bounded_uniform_reinhard_lab_spatial_rgb_with_fade"
            and run_report["phases"][route]["color_match"]["report"]["spatial_grid"]
            == [5, 8]
            and run_report["phases"][route]["color_match"]["report"][
                "lab_scale_bounds"
            ]
            == [0.85, 1.18]
            and float(
                run_report["phases"][route]["color_match"]["report"][
                    "maximum_total_rgb_delta"
                ]
            )
            <= 0.020001
            for route in ROUTES
        ),
        "color_match_kept_audio_and_latent_untouched": all(
            run_report["phases"][route]["color_match"]["report"]["audio_touched"]
            is False
            and run_report["phases"][route]["color_match"]["report"]["latent_touched"]
            is False
            for route in ("segment0", *ROUTES)
        ),
        "review_transports_strict_and_exact": all(review_transport["checks"].values()),
        "review_pack_created": all(
            (output_root / name).is_file()
            for name in ("A.mp4", "B.mp4", "blind_review.html", "blind_seam_contact.jpg")
        ),
    }
    minimum_free = {
        route: int(run_report["phases"][route]["gpu_monitor"]["minimum_free_mib"])
        for route in ROUTES
    }
    resource_checks = {
        "soft_minimum_free_at_least_512_mib": minimum_free["soft_context"] >= 512,
        "plan_b_minimum_free_at_least_512_mib": minimum_free["hard_mask_plan_b"] >= 512,
    }
    mechanical_pass = all(checks.values())
    resource_pass = all(resource_checks.values())
    if not mechanical_pass:
        status = "FAIL_MECHANICAL_PAIR"
    elif not resource_pass:
        status = "MECHANICAL_PAIR_PASS_RESOURCE_MARGIN_BELOW_512_HUMAN_REVIEW_PENDING"
    else:
        status = "MECHANICAL_AND_RESOURCE_PAIR_PASS_HUMAN_REVIEW_PENDING"
    report = {
        "schema": SCHEMA,
        "created_at": _utc_now(),
        "status": status,
        "run_root": str(run_root),
        "output_root": str(output_root),
        "contract": run_report["contract"],
        "shared_context": run_report["context"],
        "shared_color_reference": run_report.get("color_reference"),
        "segment_zero": {
            "path": str(segment_zero),
            "sha256": _sha256(segment_zero),
            "frames": len(segment_frames),
            "audio_quality": audio_quality_metrics(segment_audio),
        },
        "routes": routes,
        "pair_difference": pair_difference,
        "review_transport": review_transport,
        "checks": checks,
        "resource": {
            "minimum_free_mib": minimum_free,
            "checks": resource_checks,
            "claim": _resource_claim(resource_pass),
        },
        "private_mapping": mapping,
        "blind_review": str((output_root / "blind_review.html").resolve()),
        "human_review": {
            "status": "PENDING",
            "winner": None,
            "seam_quality_validated": False,
            "identity_quality_validated": False,
            "audio_noninferiority_validated": False,
        },
        "claim_boundary": (
            "Strict decode, exact frame counts, context immutability, runtime topology and "
            "descriptive seam proxies passed. Human full-speed and frame-step review is still "
            "required; automated proxies do not choose a winner."
        ),
    }
    (output_root / "private_mapping.json").write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_root / "pair_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if mechanical_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
