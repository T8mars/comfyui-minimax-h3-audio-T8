from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
from contextlib import contextmanager
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
from typing import Any

try:
    from .validate_h3_vram import (
        ValidationError,
        analyze_prompt,
        build_report,
        collect_run,
        load_api_prompt,
        sha256_file,
    )
except ImportError:  # Direct script execution from tools/.
    from validate_h3_vram import (  # type: ignore[no-redef]
        ValidationError,
        analyze_prompt,
        build_report,
        collect_run,
        load_api_prompt,
        sha256_file,
    )


SCHEMA = "t8.minimax_h3.hybrid_matrix.v1"
TOOL_VERSION = "1.1.0"
BASE_CONTROL = "fl2va_base_control"
REF_CONTROL = "ref2va_stock_control"
KNOWN_PROFILES = (
    "blocks_25_49_video_audio_exp",
    "blocks_25_49_all_modalities_exp",
    "blocks_25_49_video_exp",
    "blocks_25_49_audio_exp",
    "blocks_0_49_video_audio_exp",
    "blocks_0_49_all_modalities_exp",
)
DEFAULT_PROFILES = (
    "blocks_25_49_video_audio_exp",
    "blocks_25_49_all_modalities_exp",
    "blocks_0_49_video_audio_exp",
)
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
REFERENCE_PREFIXES = (
    "ref_images.",
    "ref_videos.",
    "ref_video_audios.",
    "ref_audios.",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def safe_label(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-._")
    return cleaned[:120] or "hybrid-matrix"


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        with temp.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass


@contextmanager
def matrix_lock(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / "matrix.lock"
    payload = canonical_json({"pid": os.getpid(), "created_at": utc_now()})
    try:
        descriptor = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise ValidationError(
            f"Matrix directory is already owned by another run, or has a stale lock: {lock_path}"
        ) from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        yield
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def _single_node(prompt: dict[str, dict[str, Any]], class_type: str) -> tuple[str, dict[str, Any]]:
    matches = [
        (node_id, node)
        for node_id, node in prompt.items()
        if node.get("class_type") == class_type
    ]
    if len(matches) != 1:
        raise ValidationError(
            f"Hybrid matrix template requires exactly one {class_type}, found {len(matches)}."
        )
    return matches[0]


def validate_hybrid_template(prompt: dict[str, dict[str, Any]]) -> dict[str, Any]:
    inspector_id, inspector = _single_node(
        prompt, "MiniMaxH3HybridPairInspectorT8Advanced"
    )
    builder_id, builder = _single_node(
        prompt, "MiniMaxH3HybridArtifactBuilderT8Advanced"
    )
    loader_id, loader = _single_node(prompt, "MiniMaxH3HybridModelLoaderT8Advanced")
    conditioning_id, conditioning = _single_node(prompt, "MiniMaxH3AudioConditioningT8")
    sampler_id, sampler = _single_node(prompt, "MiniMaxH3DualClockSamplerT8")
    seed_id, seed_node = _single_node(prompt, "RandomNoise")

    if any("lora" in node.get("class_type", "").lower() for node in prompt.values()):
        raise ValidationError(
            "The quality matrix template must not contain a LoRA; evaluate Hybrid profiles on "
            "Stock20 before adding another treatment variable."
        )
    if any(node.get("class_type") == "UNETLoader" for node in prompt.values()):
        raise ValidationError("The Hybrid template must not contain a second UNETLoader.")
    if builder.get("inputs", {}).get("hybrid_plan") != [inspector_id, 0]:
        raise ValidationError("Artifact Builder must consume this template's Pair Inspector plan.")
    if loader.get("inputs", {}).get("hybrid_artifact") != [builder_id, 0]:
        raise ValidationError("Hybrid Loader must consume this template's Artifact Builder output.")
    if sampler.get("inputs", {}).get("model") != [loader_id, 0]:
        raise ValidationError("Dual-clock sampler must consume this template's Hybrid Loader MODEL.")
    if int(sampler.get("inputs", {}).get("steps", 0)) != 20:
        raise ValidationError("The first Hybrid quality matrix requires exactly Stock20 sampling.")
    if seed_node.get("inputs", {}).get("noise_seed") is None:
        raise ValidationError("RandomNoise must expose a literal noise_seed.")

    conditioning_inputs = conditioning.get("inputs", {})
    connected_references = sorted(
        name
        for name, value in conditioning_inputs.items()
        if name.startswith(REFERENCE_PREFIXES)
        and isinstance(value, list)
        and len(value) == 2
    )
    if not connected_references:
        raise ValidationError(
            "Hybrid quality matrix requires at least one connected extra image/video/audio reference."
        )

    sink_matches = [
        (node_id, node)
        for node_id, node in prompt.items()
        if node.get("class_type") in {"VHS_VideoCombine", "SaveVideo"}
    ]
    if len(sink_matches) != 1:
        raise ValidationError(
            "Hybrid matrix template requires exactly one VHS_VideoCombine or SaveVideo sink."
        )
    sink_id, sink = sink_matches[0]
    return {
        "inspector_id": inspector_id,
        "builder_id": builder_id,
        "loader_id": loader_id,
        "conditioning_id": conditioning_id,
        "sampler_id": sampler_id,
        "seed_id": seed_id,
        "sink_id": sink_id,
        "sink_type": sink["class_type"],
        "quality_base": inspector.get("inputs", {}).get("quality_base"),
        "reference_overlay": inspector.get("inputs", {}).get("reference_overlay"),
        "weight_dtype": loader.get("inputs", {}).get("weight_dtype", "default"),
        "connected_references": connected_references,
    }


def _normalize_controls(
    prompt: dict[str, dict[str, Any]], contract: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    normalized = copy.deepcopy(prompt)
    for node_id in (
        contract["inspector_id"],
        contract["builder_id"],
        contract["loader_id"],
    ):
        normalized.pop(node_id, None)
    sink = normalized[contract["sink_id"]]["inputs"]
    sink["filename_prefix"] = "<treatment-output>"
    normalized[contract["seed_id"]]["inputs"]["noise_seed"] = "<matrix-seed>"
    return normalized


def control_fingerprint(
    prompt: dict[str, dict[str, Any]], contract: dict[str, Any]
) -> str:
    return sha256_value(_normalize_controls(prompt, contract))


def treatment_specs(profiles: list[str] | tuple[str, ...]) -> list[dict[str, str]]:
    seen: set[str] = set()
    result = [
        {"id": BASE_CONTROL, "kind": "base_control"},
        {"id": REF_CONTROL, "kind": "ref_control"},
    ]
    for profile in profiles:
        if profile not in KNOWN_PROFILES:
            raise ValidationError(f"Unknown Hybrid profile in matrix: {profile!r}")
        if profile in seen:
            raise ValidationError(f"Duplicate Hybrid profile in matrix: {profile!r}")
        seen.add(profile)
        result.append({"id": profile, "kind": "hybrid", "profile": profile})
    return result


def make_treatment_prompt(
    template: dict[str, dict[str, Any]],
    treatment: dict[str, str],
    *,
    seed: int,
    filename_prefix: str,
) -> dict[str, dict[str, Any]]:
    prompt = copy.deepcopy(template)
    contract = validate_hybrid_template(prompt)
    inspector = prompt[contract["inspector_id"]]
    loader_id = contract["loader_id"]
    loader = prompt[loader_id]

    kind = treatment["kind"]
    if kind == "base_control":
        loader["inputs"] = {
            "quality_base": contract["quality_base"],
            "mode": "base_only",
            "weight_dtype": contract["weight_dtype"],
        }
        loader.setdefault("_meta", {})["title"] = "Matrix control: stock FL2VA base"
    elif kind == "ref_control":
        prompt[loader_id] = {
            "inputs": {
                "unet_name": contract["reference_overlay"],
                "weight_dtype": contract["weight_dtype"],
            },
            "class_type": "UNETLoader",
            "_meta": {"title": "Matrix control: stock Ref2VA"},
        }
    elif kind == "hybrid":
        profile = treatment["profile"]
        inspector["inputs"]["profile"] = profile
        loader["inputs"] = {
            "quality_base": contract["quality_base"],
            "mode": "apply_artifact_exp",
            "weight_dtype": contract["weight_dtype"],
            "hybrid_artifact": [contract["builder_id"], 0],
        }
        loader.setdefault("_meta", {})["title"] = f"Matrix treatment: {profile}"
    else:
        raise ValidationError(f"Unsupported matrix treatment kind: {kind!r}")

    prompt[contract["seed_id"]]["inputs"]["noise_seed"] = int(seed)
    prompt[contract["sink_id"]]["inputs"]["filename_prefix"] = filename_prefix
    return prompt


def build_prompt_matrix(
    template: dict[str, dict[str, Any]],
    profiles: list[str] | tuple[str, ...],
    seeds: list[int] | tuple[int, ...],
    *,
    output_prefix: str,
) -> list[dict[str, Any]]:
    if not seeds:
        raise ValidationError("Hybrid matrix requires at least one seed.")
    if len(set(seeds)) != len(seeds):
        raise ValidationError("Hybrid matrix seeds must be unique.")
    contract = validate_hybrid_template(template)
    records = []
    for seed in seeds:
        for treatment in treatment_specs(profiles):
            run_id = safe_label(f"seed-{seed}-{treatment['id']}")
            prompt = make_treatment_prompt(
                template,
                treatment,
                seed=seed,
                filename_prefix=f"{output_prefix}/{run_id}",
            )
            if control_fingerprint(prompt, contract) != control_fingerprint(template, contract):
                raise ValidationError(
                    f"Treatment {treatment['id']} changed a non-treatment control input."
                )
            records.append(
                {
                    "run_id": run_id,
                    "seed": seed,
                    "treatment": treatment,
                    "prompt": prompt,
                    "control_fingerprint": control_fingerprint(prompt, contract),
                }
            )
    return records


async def _json_request(session, method: str, url: str, **kwargs) -> Any:
    async with session.request(method, url, **kwargs) as response:
        text = await response.text()
        if response.status >= 400:
            raise ValidationError(f"{method} {url} failed ({response.status}): {text[:2000]}")
        if not text.strip():
            return {}
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValidationError(f"{method} {url} returned non-JSON content") from exc


def _queue_count(payload: Any) -> int:
    if not isinstance(payload, dict):
        return 0
    return sum(
        len(payload.get(name, []))
        for name in ("queue_running", "queue_pending")
        if isinstance(payload.get(name, []), list)
    )


async def require_idle_server(server: str) -> None:
    import aiohttp

    timeout = aiohttp.ClientTimeout(total=15)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        queue = await _json_request(session, "GET", f"{server.rstrip('/')}/queue")
    if _queue_count(queue):
        raise ValidationError(
            "Hybrid matrix requires a dedicated idle ComfyUI server; existing queue work was detected."
        )


async def release_server_models(server: str, settle_seconds: float) -> dict[str, Any]:
    import aiohttp

    if settle_seconds < 0:
        raise ValidationError("settle_seconds cannot be negative")
    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        before = await _json_request(session, "GET", f"{server.rstrip('/')}/system_stats")
        await _json_request(
            session,
            "POST",
            f"{server.rstrip('/')}/free",
            json={"unload_models": True, "free_memory": True},
        )
        if settle_seconds:
            await asyncio.sleep(settle_seconds)
        after = await _json_request(session, "GET", f"{server.rstrip('/')}/system_stats")
    return {"requested_at": utc_now(), "before": before, "after": after}


async def fetch_history(server: str, prompt_id: str) -> dict[str, Any]:
    import aiohttp

    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        payload = await _json_request(
            session, "GET", f"{server.rstrip('/')}/history/{prompt_id}"
        )
    if not isinstance(payload, dict):
        raise ValidationError("ComfyUI history response is not a JSON object")
    return payload


def output_descriptors(history: dict[str, Any], prompt_id: str) -> list[dict[str, str]]:
    record = history.get(prompt_id)
    if not isinstance(record, dict):
        return []
    outputs = record.get("outputs", {})
    result = []
    if not isinstance(outputs, dict):
        return result
    for node_id, node_output in outputs.items():
        if not isinstance(node_output, dict):
            continue
        for output_kind, values in node_output.items():
            if not isinstance(values, list):
                continue
            for value in values:
                if not isinstance(value, dict) or not isinstance(value.get("filename"), str):
                    continue
                result.append(
                    {
                        "node_id": str(node_id),
                        "output_kind": str(output_kind),
                        "filename": value["filename"],
                        "subfolder": str(value.get("subfolder", "")),
                        "type": str(value.get("type", "output")),
                    }
                )
    return result


def resolve_output_files(
    descriptors: list[dict[str, str]], comfy_root: Path
) -> list[dict[str, Any]]:
    roots = {
        "output": (comfy_root / "output").resolve(),
        "temp": (comfy_root / "temp").resolve(),
        "input": (comfy_root / "input").resolve(),
    }
    result = []
    for descriptor in descriptors:
        root = roots.get(descriptor["type"])
        if root is None:
            raise ValidationError(f"Unknown ComfyUI output type: {descriptor['type']!r}")
        path = (root / descriptor["subfolder"] / descriptor["filename"]).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValidationError(f"ComfyUI output path escapes its declared root: {path}") from exc
        if not path.is_file():
            raise ValidationError(f"ComfyUI reported an output file that does not exist: {path}")
        result.append(
            {
                **descriptor,
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return result


def _decode_video_frames(path: Path, maximum_width: int = 512):
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValidationError(f"OpenCV cannot open generated video: {path}")
    frames = []
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame.shape[1] > maximum_width:
            scale = maximum_width / frame.shape[1]
            frame = cv2.resize(
                frame,
                (maximum_width, max(1, round(frame.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )
        frames.append(frame)
    capture.release()
    if not frames:
        raise ValidationError(f"Generated video contains no decodable frames: {path}")
    return frames, fps


def video_metrics(path: Path) -> dict[str, Any]:
    import cv2
    import numpy as np

    frames, fps = _decode_video_frames(path)
    highpass = []
    laplacian = []
    saturation = []
    temporal = []
    previous = None
    decoded_hasher = hashlib.sha256()
    for frame in frames:
        decoded_hasher.update(frame.shape[0].to_bytes(4, "little"))
        decoded_hasher.update(frame.shape[1].to_bytes(4, "little"))
        decoded_hasher.update(frame.tobytes(order="C"))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
        highpass.append(float(np.mean(np.abs(gray - blur))))
        laplacian.append(float(cv2.Laplacian(gray, cv2.CV_32F).var()))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        saturation.append(float(np.mean(hsv[:, :, 1]) / 255.0))
        if previous is not None:
            temporal.append(float(np.mean(np.abs(rgb - previous))))
        previous = rgb
    return {
        "frame_count": len(frames),
        "fps": fps,
        "duration_seconds": len(frames) / fps if fps > 0 else None,
        "highpass_mean": float(np.mean(highpass)),
        "laplacian_variance_median": float(np.median(laplacian)),
        "saturation_mean": float(np.mean(saturation)),
        "temporal_mad_median": float(np.median(temporal)) if temporal else 0.0,
        "decoded_bgr_sha256": decoded_hasher.hexdigest(),
        "scope": "proxy metrics only; not identity, reference adherence, oiliness, or quality",
    }


def audio_metrics(path: Path, ffmpeg: str = "ffmpeg") -> dict[str, Any]:
    import numpy as np

    command = [
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
    completed = subprocess.run(command, capture_output=True, check=False)
    if completed.returncode != 0:
        raise ValidationError(
            f"ffmpeg audio decode failed for {path}: "
            + completed.stderr.decode("utf-8", errors="replace")[-1000:]
        )
    samples = np.frombuffer(completed.stdout, dtype=np.float32)
    if not samples.size or samples.size % 2:
        raise ValidationError(f"Decoded audio is empty or malformed: {path}")
    stereo = samples.reshape(-1, 2)
    mono = stereo.mean(axis=1)
    rms = float(np.sqrt(np.mean(np.square(stereo, dtype=np.float64))))
    peak = float(np.max(np.abs(stereo)))
    clipping_fraction = float(np.mean(np.abs(stereo) >= 0.999))
    spectrum = np.abs(np.fft.rfft(mono.astype(np.float64))) ** 2
    frequencies = np.fft.rfftfreq(mono.size, 1.0 / 32000.0)
    total = float(spectrum.sum())
    high = float(spectrum[frequencies >= 8000.0].sum())
    high_ratio_db = 10.0 * math.log10(max(high, 1.0e-30) / max(total, 1.0e-30))
    return {
        "sample_rate": 32000,
        "channels": 2,
        "sample_count": int(stereo.shape[0]),
        "duration_seconds": stereo.shape[0] / 32000.0,
        "rms": rms,
        "peak": peak,
        "clipping_fraction": clipping_fraction,
        "high_frequency_ratio_db_8khz": high_ratio_db,
        "decoded_pcm_f32le_sha256": hashlib.sha256(completed.stdout).hexdigest(),
        "scope": "signal proxies only; not ASR, speaker identity, SFX adherence, or listening quality",
    }


def media_metrics(outputs: list[dict[str, Any]], ffmpeg: str = "ffmpeg") -> dict[str, Any]:
    video = next(
        (Path(item["path"]) for item in outputs if Path(item["path"]).suffix.lower() in VIDEO_SUFFIXES),
        None,
    )
    if video is None:
        raise ValidationError("Successful Hybrid matrix run did not produce a video file")
    return {"video": video_metrics(video), "audio": audio_metrics(video, ffmpeg)}


def refresh_media_metrics(manifest: dict[str, Any], ffmpeg: str = "ffmpeg") -> None:
    for record in manifest["runs"].values():
        if completed_record_is_valid(record):
            refreshed = media_metrics(record["outputs"], ffmpeg)
            preserved = record.get("metrics", {})
            for optional_key in ("asr", "face_identity", "speaker_identity"):
                if optional_key in preserved:
                    refreshed[optional_key] = preserved[optional_key]
            record["metrics"] = refreshed


def add_asr_metrics(
    manifest: dict[str, Any],
    model_directory: Path,
    *,
    language: str,
    beam_size: int,
) -> None:
    if not model_directory.is_dir():
        raise ValidationError(f"Local faster-whisper model directory does not exist: {model_directory}")
    if beam_size < 1:
        raise ValidationError("ASR beam size must be at least 1")
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise ValidationError("faster-whisper is required only when --asr-model is used") from exc

    model = WhisperModel(
        str(model_directory),
        device="cpu",
        compute_type="int8",
        cpu_threads=max(1, min(os.cpu_count() or 1, 8)),
    )
    try:
        for record in manifest["runs"].values():
            if not completed_record_is_valid(record):
                continue
            media = _video_output(record)
            segments, info = model.transcribe(
                str(media),
                language=None if language == "auto" else language,
                beam_size=beam_size,
                vad_filter=True,
            )
            rows = [
                {
                    "start_seconds": round(float(segment.start), 3),
                    "end_seconds": round(float(segment.end), 3),
                    "text": segment.text.strip(),
                }
                for segment in segments
            ]
            record.setdefault("metrics", {})["asr"] = {
                "model_directory": str(model_directory.resolve()),
                "requested_language": language,
                "detected_language": getattr(info, "language", None),
                "language_probability": getattr(info, "language_probability", None),
                "segments": rows,
                "nonempty_segment_count": len(rows),
                "scope": "ASR research signal only; silence/empty transcription is not listening quality",
            }
    finally:
        del model


def add_face_identity_metrics(
    manifest: dict[str, Any],
    reference_image: Path,
    *,
    model_root: Path,
    model_name: str,
    sample_count: int,
    detector_threshold: float,
) -> None:
    if not reference_image.is_file():
        raise ValidationError(f"Face reference image does not exist: {reference_image}")
    model_directory = model_root / "models" / model_name
    if not model_directory.is_dir():
        raise ValidationError(
            f"Local InsightFace model directory does not exist; no download will be attempted: {model_directory}"
        )
    if sample_count < 1:
        raise ValidationError("Face sample count must be at least 1")
    if not 0.0 <= detector_threshold <= 1.0:
        raise ValidationError("Face detector threshold must be between 0 and 1")
    try:
        import cv2
        from insightface.app import FaceAnalysis
        import numpy as np
    except ImportError as exc:
        raise ValidationError("InsightFace/OpenCV is required only when --face-reference is used") from exc

    app = FaceAnalysis(
        name=model_name,
        root=str(model_root),
        providers=["CPUExecutionProvider"],
    )
    app.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=detector_threshold)
    reference_pixels = cv2.imread(str(reference_image))
    if reference_pixels is None:
        raise ValidationError(f"OpenCV cannot read face reference image: {reference_image}")
    reference_faces = app.get(reference_pixels, max_num=1)
    if not reference_faces:
        raise ValidationError("InsightFace found no face in the explicit reference image")
    reference_embedding = reference_faces[0].normed_embedding.astype(np.float64)

    for record in manifest["runs"].values():
        if not completed_record_is_valid(record):
            continue
        media = _video_output(record)
        frames, _fps = _decode_video_frames(media, maximum_width=736)
        indices = sorted(
            {
                round(index * (len(frames) - 1) / max(1, sample_count - 1))
                for index in range(sample_count)
            }
        )
        similarities = []
        face_area_fractions = []
        for frame_index in indices:
            faces = app.get(frames[frame_index], max_num=1)
            if not faces:
                continue
            face = faces[0]
            similarities.append(
                float(np.dot(reference_embedding, face.normed_embedding.astype(np.float64)))
            )
            left, top, right, bottom = [float(value) for value in face.bbox]
            face_area_fractions.append(
                max(0.0, right - left)
                * max(0.0, bottom - top)
                / (frames[frame_index].shape[0] * frames[frame_index].shape[1])
            )
        record.setdefault("metrics", {})["face_identity"] = {
            "model_name": model_name,
            "model_root": str(model_root.resolve()),
            "reference_image": str(reference_image.resolve()),
            "requested_frames": len(indices),
            "detected_frames": len(similarities),
            "detection_coverage": len(similarities) / len(indices),
            "cosine_median": float(np.median(similarities)) if similarities else None,
            "cosine_min": min(similarities) if similarities else None,
            "cosine_max": max(similarities) if similarities else None,
            "face_area_fraction_median": (
                float(np.median(face_area_fractions)) if face_area_fractions else None
            ),
            "detector_threshold": detector_threshold,
            "scope": (
                "single-reference research proxy; valid only with reported detection coverage "
                "and not a universal identity threshold"
            ),
        }


def _decode_audio_mono_16k(path: Path, ffmpeg: str) -> tuple[Any, dict[str, Any]]:
    import numpy as np

    command = [
        ffmpeg,
        "-v",
        "error",
        "-i",
        str(path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "f32le",
        "pipe:1",
    ]
    completed = subprocess.run(command, capture_output=True, check=False)
    if completed.returncode != 0:
        raise ValidationError(
            f"ffmpeg speaker-audio decode failed for {path}: "
            + completed.stderr.decode("utf-8", errors="replace")[-1000:]
        )
    samples = np.frombuffer(completed.stdout, dtype=np.float32).copy()
    if not samples.size or not np.isfinite(samples).all():
        raise ValidationError(f"Decoded speaker audio is empty or non-finite: {path}")
    return samples, {
        "path": str(path.resolve()),
        "sample_rate": 16000,
        "sample_count": int(samples.size),
        "duration_seconds": samples.size / 16000.0,
        "decoded_pcm_f32le_sha256": hashlib.sha256(completed.stdout).hexdigest(),
    }


def _wavlm_embedding(extractor, model, samples):
    import torch

    inputs = extractor(
        samples,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
    )
    with torch.inference_mode():
        embedding = model(**inputs).embeddings[0].to(dtype=torch.float32)
    if not bool(torch.isfinite(embedding).all()) or not bool(embedding.numel()):
        raise ValidationError("WavLM produced an empty or non-finite speaker embedding")
    return torch.nn.functional.normalize(embedding, dim=0).cpu()


def add_speaker_identity_metrics(
    manifest: dict[str, Any],
    reference_audio: Path,
    *,
    model_directory: Path,
    ffmpeg: str,
) -> None:
    if not reference_audio.is_file():
        raise ValidationError(f"Speaker reference audio does not exist: {reference_audio}")
    required = (model_directory / "config.json", model_directory / "preprocessor_config.json")
    weights = (model_directory / "model.safetensors", model_directory / "pytorch_model.bin")
    if not model_directory.is_dir() or any(not path.is_file() for path in required) or not any(
        path.is_file() for path in weights
    ):
        raise ValidationError(
            "Local WavLM X-Vector directory is incomplete; config.json, "
            "preprocessor_config.json and model.safetensors or pytorch_model.bin are required: "
            f"{model_directory}"
        )
    try:
        import torch
        from transformers import AutoFeatureExtractor, WavLMForXVector
    except ImportError as exc:
        raise ValidationError(
            "transformers with WavLMForXVector is required only when --speaker-reference is used"
        ) from exc

    reference_samples, reference_decode = _decode_audio_mono_16k(reference_audio, ffmpeg)
    extractor = AutoFeatureExtractor.from_pretrained(
        str(model_directory), local_files_only=True
    )
    model = WavLMForXVector.from_pretrained(
        str(model_directory), local_files_only=True
    )
    model.eval()
    try:
        reference_embedding = _wavlm_embedding(extractor, model, reference_samples)
        for record in manifest["runs"].values():
            if not completed_record_is_valid(record):
                continue
            media = _video_output(record)
            generated_samples, generated_decode = _decode_audio_mono_16k(media, ffmpeg)
            generated_embedding = _wavlm_embedding(extractor, model, generated_samples)
            cosine = float(torch.dot(reference_embedding, generated_embedding))
            record.setdefault("metrics", {})["speaker_identity"] = {
                "model_directory": str(model_directory.resolve()),
                "engine": "transformers.WavLMForXVector",
                "device": "cpu",
                "reference": reference_decode,
                "generated": generated_decode,
                "cosine_similarity": cosine,
                "threshold": None,
                "scope": (
                    "single-reference research signal only; no universal threshold is applied, "
                    "and one cosine score cannot prove high-fidelity cloning"
                ),
            }
    finally:
        del model
        del extractor
        if "torch" in locals():
            import gc

            gc.collect()


def completed_record_is_valid(record: dict[str, Any]) -> bool:
    if record.get("status") != "success":
        return False
    outputs = record.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        return False
    for output in outputs:
        try:
            path = Path(output["path"])
            if not path.is_file() or sha256_file(path) != output["sha256"]:
                return False
        except (KeyError, OSError, TypeError):
            return False
    return True


def _video_output(record: dict[str, Any]) -> Path:
    for item in record.get("outputs", []):
        path = Path(item["path"])
        if path.suffix.lower() in VIDEO_SUFFIXES:
            return path
    raise ValidationError(f"Run {record.get('run_id')} has no video output")


def build_blind_package(manifest: dict[str, Any], output_dir: Path, blind_seed: int) -> None:
    import cv2
    import numpy as np

    successful = [
        value
        for value in manifest["runs"].values()
        if completed_record_is_valid(value)
    ]
    if len(successful) != len(manifest["runs"]):
        return

    blind_root = output_dir / "blind"
    media_root = blind_root / "media"
    media_root.mkdir(parents=True, exist_ok=True)
    key = {"schema": SCHEMA, "created_at": utc_now(), "seeds": {}}
    review_rows = []
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in successful:
        grouped[int(record["seed"])].append(record)

    for seed, records in sorted(grouped.items()):
        records.sort(key=lambda item: item["treatment"]["id"])
        rng = random.Random(blind_seed ^ seed)
        rng.shuffle(records)
        rows = []
        seed_key = {}
        for index, record in enumerate(records):
            code = chr(ord("A") + index)
            source = _video_output(record)
            blind_name = f"seed-{seed}-{code}{source.suffix.lower()}"
            blind_path = media_root / blind_name
            if not blind_path.exists():
                try:
                    os.link(source, blind_path)
                except OSError:
                    shutil.copy2(source, blind_path)
            frames, _fps = _decode_video_frames(source, maximum_width=320)
            indices = [round(value) for value in [
                0,
                (len(frames) - 1) * 0.25,
                (len(frames) - 1) * 0.5,
                (len(frames) - 1) * 0.75,
                len(frames) - 1,
            ]]
            selected = [frames[min(max(index_value, 0), len(frames) - 1)] for index_value in indices]
            strip = np.concatenate(selected, axis=1)
            cv2.putText(
                strip,
                code,
                (12, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1,
                (255, 255, 255),
                3,
                cv2.LINE_AA,
            )
            rows.append(strip)
            seed_key[code] = {
                "treatment": record["treatment"],
                "source_path": str(source),
                "source_sha256": sha256_file(source),
                "blind_media": str(blind_path),
            }
            review_rows.append(
                {
                    "seed": seed,
                    "blind_code": code,
                    "blind_media": str(blind_path),
                    "oiliness_1_low_5_high": "",
                    "reference_adherence_1_low_5_high": "",
                    "speaker_identity_1_low_5_high": "",
                    "spoken_text_accuracy_1_low_5_high": "",
                    "extra_unrequested_speech_yes_no": "",
                    "motion_naturalness_1_low_5_high": "",
                    "audio_quality_1_low_5_high": "",
                    "overall_preference_1_low_5_high": "",
                    "notes": "",
                }
            )
        contact = np.concatenate(rows, axis=0)
        contact_path = blind_root / f"seed-{seed}-contact.png"
        if not cv2.imwrite(str(contact_path), contact):
            raise ValidationError(f"Failed to write blind contact sheet: {contact_path}")
        key["seeds"][str(seed)] = {"contact_sheet": str(contact_path), "codes": seed_key}

    write_json_atomic(blind_root / "blind_key.json", key)
    review_path = blind_root / "blind_review.csv"
    with review_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(review_rows[0]))
        writer.writeheader()
        writer.writerows(review_rows)


def _device_total_bytes(record: dict[str, Any]) -> int | None:
    devices = (
        record.get("release_before", {})
        .get("after", {})
        .get("devices", [])
    )
    if not devices or not isinstance(devices[0], dict):
        return None
    value = devices[0].get("vram_total")
    return int(value) if isinstance(value, (int, float)) and value > 0 else None


def build_matrix_summary(manifest: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    rows = []
    for record in sorted(
        manifest["runs"].values(),
        key=lambda item: (int(item["seed"]), item["treatment"]["id"]),
    ):
        metrics = record.get("metrics", {})
        video = metrics.get("video", {})
        audio = metrics.get("audio", {})
        face = metrics.get("face_identity", {})
        speaker = metrics.get("speaker_identity", {})
        asr = metrics.get("asr", {})
        summary = record.get("runtime_summary", {})
        peak_bytes = summary.get("peak_vram_used_bytes")
        total_bytes = _device_total_bytes(record)
        headroom_bytes = (
            total_bytes - int(peak_bytes)
            if total_bytes is not None and isinstance(peak_bytes, (int, float))
            else None
        )
        asr_text = " ".join(
            str(segment.get("text", "")).strip()
            for segment in asr.get("segments", [])
            if str(segment.get("text", "")).strip()
        )
        rows.append(
            {
                "seed": int(record["seed"]),
                "treatment": record["treatment"]["id"],
                "kind": record["treatment"]["kind"],
                "status": record.get("status"),
                "peak_vram_mib": (
                    round(float(peak_bytes) / 2**20, 3) if peak_bytes is not None else None
                ),
                "headroom_mib": (
                    round(headroom_bytes / 2**20, 3) if headroom_bytes is not None else None
                ),
                "headroom_512mib_gate": (
                    headroom_bytes >= 512 * 2**20 if headroom_bytes is not None else None
                ),
                "video_frames": video.get("frame_count"),
                "video_duration_seconds": video.get("duration_seconds"),
                "video_highpass_mean": video.get("highpass_mean"),
                "video_laplacian_variance_median": video.get(
                    "laplacian_variance_median"
                ),
                "video_temporal_mad_median": video.get("temporal_mad_median"),
                "video_saturation_mean": video.get("saturation_mean"),
                "face_detection_coverage": face.get("detection_coverage"),
                "face_cosine_median": face.get("cosine_median"),
                "audio_duration_seconds": audio.get("duration_seconds"),
                "audio_rms": audio.get("rms"),
                "audio_peak": audio.get("peak"),
                "audio_clipping_fraction": audio.get("clipping_fraction"),
                "audio_high_frequency_ratio_db_8khz": audio.get(
                    "high_frequency_ratio_db_8khz"
                ),
                "asr_transcript": asr_text,
                "speaker_cosine": speaker.get("cosine_similarity"),
            }
        )

    measured_headroom = [
        row["headroom_mib"] for row in rows if row["headroom_mib"] is not None
    ]
    report = {
        "schema": SCHEMA,
        "created_at": utc_now(),
        "rows": rows,
        "resource_gate": {
            "required_minimum_headroom_mib": 512.0,
            "minimum_measured_headroom_mib": min(measured_headroom) if measured_headroom else None,
            "all_measured_runs_pass": (
                all(row["headroom_512mib_gate"] for row in rows)
                if measured_headroom and len(measured_headroom) == len(rows)
                else False
            ),
        },
        "quality_decision": "not_ranked_requires_blind_review_and_broader_matrix",
        "limitations": [
            "Speaker and face cosine values are model-specific research signals, not universal thresholds.",
            "Signal proxies and one seed cannot establish oiliness, identity, reference adherence, or quality.",
            "A profile must not be promoted from this summary without blind review and multi-material evidence.",
        ],
    }
    write_json_atomic(output_dir / "matrix_summary.json", report)
    csv_path = output_dir / "matrix_summary.csv"
    if rows:
        with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return report


def build_manifest(
    template_path: Path,
    template: dict[str, dict[str, Any]],
    records: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    workflow_dir = output_dir / "workflows"
    workflow_dir.mkdir(parents=True, exist_ok=True)
    contract = validate_hybrid_template(template)
    runs = {}
    for record in records:
        workflow_path = workflow_dir / f"{record['run_id']}.json"
        write_json_atomic(workflow_path, record["prompt"])
        runs[record["run_id"]] = {
            "run_id": record["run_id"],
            "seed": record["seed"],
            "treatment": record["treatment"],
            "control_fingerprint": record["control_fingerprint"],
            "workflow_path": str(workflow_path.resolve()),
            "workflow_sha256": sha256_file(workflow_path),
            "status": "pending",
            "outputs": [],
        }
    return {
        "schema": SCHEMA,
        "tool_version": TOOL_VERSION,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "template": {
            "path": str(template_path.resolve()),
            "sha256": sha256_file(template_path),
            "control_fingerprint": control_fingerprint(template, contract),
            "connected_references": contract["connected_references"],
        },
        "runs": runs,
        "scientific_limits": [
            "Proxy metrics cannot prove oiliness, identity, reference adherence, or quality.",
            "No profile is promoted without blind review and a broader multi-material matrix.",
            "The tool globally frees models between treatments and requires a dedicated idle server.",
        ],
    }


def load_or_create_manifest(
    template_path: Path,
    template: dict[str, dict[str, Any]],
    records: list[dict[str, Any]],
    output_dir: Path,
    *,
    resume: bool,
) -> dict[str, Any]:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        manifest = build_manifest(template_path, template, records, output_dir)
        write_json_atomic(manifest_path, manifest)
        return manifest
    if not resume:
        raise ValidationError(
            f"Matrix output already contains a manifest; pass --resume or use a new directory: {manifest_path}"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"Cannot resume damaged matrix manifest: {exc}") from exc
    expected = build_manifest(template_path, template, records, output_dir)
    if manifest.get("schema") != SCHEMA:
        raise ValidationError("Matrix manifest schema is unsupported")
    for key in ("sha256", "control_fingerprint"):
        if manifest.get("template", {}).get(key) != expected["template"][key]:
            raise ValidationError(f"Resume template {key} differs from the existing matrix")
    if set(manifest.get("runs", {})) != set(expected["runs"]):
        raise ValidationError("Resume treatments/seeds differ from the existing matrix")
    for run_id, expected_record in expected["runs"].items():
        current = manifest["runs"][run_id]
        for key in ("seed", "treatment", "control_fingerprint", "workflow_sha256"):
            if current.get(key) != expected_record[key]:
                raise ValidationError(f"Resume run {run_id} changed {key}")
    return manifest


async def execute_matrix(
    manifest: dict[str, Any],
    *,
    output_dir: Path,
    server: str,
    comfy_root: Path,
    poll_interval: float,
    baseline_seconds: float,
    settle_seconds: float,
    timeout_seconds: float,
    device_index: int,
    ffmpeg: str,
) -> None:
    manifest_path = output_dir / "manifest.json"
    report_dir = output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    await require_idle_server(server)
    for run_id, record in manifest["runs"].items():
        if completed_record_is_valid(record):
            print(f"SKIP {run_id}: verified completed output")
            continue
        await require_idle_server(server)
        print(f"RUN {run_id}: globally freeing model/cache state")
        record["release_before"] = await release_server_models(server, settle_seconds)
        record["status"] = "running"
        record["started_at"] = utc_now()
        manifest["updated_at"] = utc_now()
        write_json_atomic(manifest_path, manifest)
        workflow_path = Path(record["workflow_path"])
        prompt = load_api_prompt(workflow_path)
        runtime = await collect_run(
            prompt,
            server=server,
            device_index=device_index,
            poll_interval=poll_interval,
            baseline_seconds=baseline_seconds,
            timeout_seconds=timeout_seconds,
            preview_method="none",
        )
        report = build_report(
            label=run_id,
            workflow_path=workflow_path,
            analysis=analyze_prompt(prompt),
            runtime=runtime,
            system_stats=runtime.get("server_snapshot"),
            log_path=None,
        )
        report_path = report_dir / f"{run_id}.json"
        write_json_atomic(report_path, report)
        record["report_path"] = str(report_path.resolve())
        record["prompt_id"] = runtime.get("prompt_id")
        record["finished_at"] = utc_now()
        record["runtime_status"] = runtime.get("status")
        record["runtime_summary"] = runtime.get("summary")
        if runtime.get("status") == "success":
            history = await fetch_history(server, str(runtime["prompt_id"]))
            descriptors = output_descriptors(history, str(runtime["prompt_id"]))
            record["outputs"] = resolve_output_files(descriptors, comfy_root)
            record["metrics"] = media_metrics(record["outputs"], ffmpeg)
            record["status"] = "success"
        else:
            record["status"] = runtime.get("status", "error")
        manifest["updated_at"] = utc_now()
        write_json_atomic(manifest_path, manifest)
        if record["status"] != "success":
            raise ValidationError(f"Matrix treatment {run_id} failed; see {report_path}")
    manifest["release_after"] = await release_server_models(server, settle_seconds)
    manifest["status"] = "complete"
    manifest["completed_at"] = utc_now()
    manifest["updated_at"] = utc_now()
    write_json_atomic(manifest_path, manifest)


def parse_csv_values(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_seeds(value: str) -> list[int]:
    try:
        seeds = [int(item) for item in parse_csv_values(value)]
    except ValueError as exc:
        raise ValidationError("--seeds must be comma-separated integers") from exc
    if not seeds:
        raise ValidationError("--seeds must contain at least one integer")
    return seeds


def default_output_dir(template: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return (
        Path(__file__).resolve().parents[1]
        / "artifacts"
        / "hybrid-model-matrix"
        / f"{stamp}-{safe_label(template.stem)}"
    )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate and optionally run a resumable, sequential, controlled MiniMax H3 "
            "FL2VA/Ref2VA/Hybrid profile matrix."
        )
    )
    parser.add_argument("template", type=Path, help="Hybrid Advanced API-format template")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--server", default="http://127.0.0.1:8188")
    parser.add_argument("--comfy-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--profiles", default=",".join(DEFAULT_PROFILES))
    parser.add_argument("--seeds", default="2608125201")
    parser.add_argument("--output-prefix", default="MiniMaxH3_T8/Hybrid_Matrix")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--poll-interval", type=float, default=0.25)
    parser.add_argument("--baseline-seconds", type=float, default=2.0)
    parser.add_argument("--settle-seconds", type=float, default=3.0)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--blind-seed", type=int, default=2608125999)
    parser.add_argument(
        "--asr-model",
        type=Path,
        help="Optional existing local faster-whisper directory; never downloaded automatically",
    )
    parser.add_argument("--asr-language", default="auto")
    parser.add_argument("--asr-beam-size", type=int, default=5)
    parser.add_argument(
        "--face-reference",
        type=Path,
        help="Optional explicit identity reference image for CPU InsightFace proxy reporting",
    )
    parser.add_argument(
        "--face-model-root",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "models" / "insightface",
    )
    parser.add_argument("--face-model-name", default="buffalo_l")
    parser.add_argument("--face-sample-count", type=int, default=12)
    parser.add_argument("--face-detector-threshold", type=float, default=0.15)
    parser.add_argument(
        "--speaker-reference",
        type=Path,
        help="Optional explicit reference audio for local CPU WavLM X-Vector cosine reporting",
    )
    parser.add_argument(
        "--speaker-model",
        type=Path,
        help="Existing local WavLM X-Vector model directory; never downloaded automatically",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    try:
        if bool(args.speaker_reference) != bool(args.speaker_model):
            raise ValidationError(
                "--speaker-reference and --speaker-model must be provided together"
            )
        if args.speaker_reference and not args.speaker_reference.is_file():
            raise ValidationError(
                f"Speaker reference audio does not exist: {args.speaker_reference}"
            )
        if args.speaker_model and not args.speaker_model.is_dir():
            raise ValidationError(
                f"Local WavLM X-Vector directory does not exist: {args.speaker_model}"
            )
        template_path = args.template.resolve()
        template = load_api_prompt(template_path)
        profiles = parse_csv_values(args.profiles)
        seeds = parse_seeds(args.seeds)
        records = build_prompt_matrix(
            template,
            profiles,
            seeds,
            output_prefix=args.output_prefix,
        )
        output_dir = (args.output_dir or default_output_dir(template_path)).resolve()
        with matrix_lock(output_dir):
            manifest = load_or_create_manifest(
                template_path,
                template,
                records,
                output_dir,
                resume=args.resume,
            )
            print(f"Matrix: {output_dir}")
            print(
                f"Runs: {len(records)} ({len(seeds)} seed(s) x "
                f"{len(treatment_specs(profiles))} treatments)"
            )
            if args.dry_run:
                print("Dry run complete; no ComfyUI prompt was queued.")
                return 0
            if args.poll_interval < 0.05:
                raise ValidationError("--poll-interval must be at least 0.05 seconds")
            if args.timeout <= 0 or args.baseline_seconds < 0 or args.settle_seconds < 0:
                raise ValidationError("timeout must be positive and baseline/settle cannot be negative")
            asyncio.run(
                execute_matrix(
                    manifest,
                    output_dir=output_dir,
                    server=args.server,
                    comfy_root=args.comfy_root.resolve(),
                    poll_interval=args.poll_interval,
                    baseline_seconds=args.baseline_seconds,
                    settle_seconds=args.settle_seconds,
                    timeout_seconds=args.timeout,
                    device_index=args.device_index,
                    ffmpeg=args.ffmpeg,
                )
            )
            refresh_media_metrics(manifest, args.ffmpeg)
            if args.asr_model:
                add_asr_metrics(
                    manifest,
                    args.asr_model.resolve(),
                    language=args.asr_language,
                    beam_size=args.asr_beam_size,
                )
            if args.face_reference:
                add_face_identity_metrics(
                    manifest,
                    args.face_reference.resolve(),
                    model_root=args.face_model_root.resolve(),
                    model_name=args.face_model_name,
                    sample_count=args.face_sample_count,
                    detector_threshold=args.face_detector_threshold,
                )
            if args.speaker_reference:
                add_speaker_identity_metrics(
                    manifest,
                    args.speaker_reference.resolve(),
                    model_directory=args.speaker_model.resolve(),
                    ffmpeg=args.ffmpeg,
                )
            manifest["matrix_summary"] = build_matrix_summary(manifest, output_dir)
            build_blind_package(manifest, output_dir, args.blind_seed)
            write_json_atomic(output_dir / "manifest.json", manifest)
            print(f"Complete: {output_dir / 'manifest.json'}")
        return 0
    except (ValidationError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
