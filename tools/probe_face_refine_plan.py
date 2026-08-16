#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
from typing import Any

import cv2
import numpy as np
import psutil
import torch


SCHEMA = "t8.face_refine_plan_probe.v1"
PLUGIN_PACKAGE = "h3_audio_t8_pkg"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _load_face_refine_module(repo_root: Path, comfy_root: Path):
    sys.path.insert(0, str(comfy_root))
    if PLUGIN_PACKAGE not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            PLUGIN_PACKAGE,
            repo_root / "__init__.py",
            submodule_search_locations=[str(repo_root)],
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot import plugin package from {repo_root}")
        package = importlib.util.module_from_spec(spec)
        sys.modules[PLUGIN_PACKAGE] = package
        spec.loader.exec_module(package)
    return __import__(f"{PLUGIN_PACKAGE}.face_refine_advanced", fromlist=["*"])


def _decode_video(path: Path) -> tuple[torch.Tensor, dict[str, Any]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    declared_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if declared_frames <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError("Video metadata does not provide a positive frame count and size")
    if declared_frames > 362:
        capture.release()
        raise ValueError("The Face Refine tensor route is capped at 362 frames")

    decoded = np.empty((declared_frames, height, width, 3), dtype=np.uint8)
    decoded_frames = 0
    try:
        while decoded_frames < declared_frames:
            ok, bgr = capture.read()
            if not ok:
                break
            decoded[decoded_frames] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            decoded_frames += 1
    finally:
        capture.release()
    if decoded_frames != declared_frames:
        raise RuntimeError(
            f"Decoded {decoded_frames} frames but the container declares {declared_frames}"
        )
    frames = torch.from_numpy(decoded).float().div_(255.0)
    return frames, {
        "frame_count": declared_frames,
        "width": width,
        "height": height,
        "fps": fps,
    }


def _track_metrics(plan: dict[str, Any]) -> dict[str, Any]:
    width = float(plan["source"]["width"])
    height = float(plan["source"]["height"])
    diagonal = math.hypot(width, height)
    prior: tuple[float, float, float] | None = None
    center_jumps = []
    scale_jumps = []
    for record in plan["frames"]:
        if record["state"] == "lost":
            prior = None
            continue
        left, top, right, bottom = record["source_face_box_xyxy"]
        face_width = max(1e-9, right - left)
        face_height = max(1e-9, bottom - top)
        current = ((left + right) * 0.5, (top + bottom) * 0.5, face_width * face_height)
        if prior is not None:
            center_jumps.append(
                math.hypot(current[0] - prior[0], current[1] - prior[1]) / diagonal
            )
            scale_jumps.append(abs(math.log(current[2] / prior[2])))
        prior = current
    return {
        "max_adjacent_center_jump_fraction_of_source_diagonal": max(
            center_jumps, default=0.0
        ),
        "p95_adjacent_center_jump_fraction_of_source_diagonal": float(
            np.percentile(center_jumps, 95)
        )
        if center_jumps
        else 0.0,
        "max_adjacent_log_area_change": max(scale_jumps, default=0.0),
        "p95_adjacent_log_area_change": float(np.percentile(scale_jumps, 95))
        if scale_jumps
        else 0.0,
    }


def _save_preview(path: Path, preview: torch.Tensor, indices: list[int]) -> None:
    images = preview.detach().clamp(0.0, 1.0).mul(255.0).byte().cpu().numpy()
    rendered = []
    for image, frame_index in zip(images, indices, strict=True):
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.putText(
            bgr,
            f"frame {frame_index}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        rendered.append(bgr)
    if not rendered:
        raise RuntimeError("Face Refine Plan returned an empty preview")
    columns = min(4, len(rendered))
    rows = []
    for offset in range(0, len(rendered), columns):
        chunk = rendered[offset : offset + columns]
        while len(chunk) < columns:
            chunk.append(np.zeros_like(rendered[0]))
        rows.append(np.hstack(chunk))
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), np.vstack(rows)):
        raise RuntimeError(f"Cannot write preview image: {path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the exact Face Refine Advanced Plan path and record detector, trajectory "
            "and host-memory evidence without loading H3 weights."
        )
    )
    parser.add_argument("video", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview", type=Path)
    parser.add_argument("--canvas", choices=("384", "512", "640", "768"), default="384")
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--scene-cut-threshold", type=float, default=0.28)
    parser.add_argument("--max-track-jump", type=float, default=0.18)
    parser.add_argument("--max-gap-frames", type=int, default=4)
    parser.add_argument("--smoothing-radius", type=int, default=2)
    parser.add_argument("--crop-context-scale", type=float, default=3.0)
    parser.add_argument("--analysis-chunk-frames", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    video = args.video.resolve()
    output = args.output.resolve()
    if not video.is_file():
        raise FileNotFoundError(video)

    repo_root = Path(__file__).resolve().parents[1]
    comfy_root = repo_root.parent.parent
    module = _load_face_refine_module(repo_root, comfy_root)
    process = psutil.Process()
    samples: list[dict[str, int | None]] = []
    stopping = threading.Event()

    def monitor() -> None:
        while not stopping.is_set():
            memory = process.memory_info()
            samples.append(
                {
                    "rss": int(memory.rss),
                    "private": int(getattr(memory, "private", 0)) or None,
                    "vms": int(memory.vms),
                }
            )
            stopping.wait(0.05)

    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    started = time.perf_counter()
    frames = crops = preview = plan = None
    try:
        frames, source = _decode_video(video)
        if abs(float(source["fps"]) - 24.0) > 0.01:
            raise ValueError(f"Face Refine requires exact 24fps; got {source['fps']:.6g}")
        plan, crops, preview, _, canvas_width, canvas_height, frame_count = (
            module.build_face_refine_plan(
                frames=frames,
                fps=float(source["fps"]),
                detector_mode="local_opencv_yunet",
                detector_model=module.YUNET_2023MAR_RELATIVE,
                detector_device="cpu",
                confidence=float(args.confidence),
                manual_roi_x=0.30,
                manual_roi_y=0.10,
                manual_roi_width=0.40,
                manual_roi_height=0.55,
                scene_cut_threshold=float(args.scene_cut_threshold),
                max_track_jump=float(args.max_track_jump),
                max_gap_frames=int(args.max_gap_frames),
                smoothing_radius=int(args.smoothing_radius),
                crop_context_scale=float(args.crop_context_scale),
                canvas_size=str(args.canvas),
                require_h3_grid=True,
                analysis_chunk_frames=int(args.analysis_chunk_frames),
            )
        )
        if args.preview:
            _save_preview(
                args.preview.resolve(), preview, list(plan["preview_frame_indices"])
            )
        result = {
            "schema": SCHEMA,
            "status": "mechanical_plan_completed_quality_not_inferred",
            "source": {
                "filename": video.name,
                "sha256": _sha256_file(video),
                **source,
            },
            "settings": {
                "canvas": str(args.canvas),
                "confidence": float(args.confidence),
                "scene_cut_threshold": float(args.scene_cut_threshold),
                "max_track_jump": float(args.max_track_jump),
                "max_gap_frames": int(args.max_gap_frames),
                "smoothing_radius": int(args.smoothing_radius),
                "crop_context_scale": float(args.crop_context_scale),
            },
            "output": {
                "frame_count": int(frame_count),
                "canvas_width": int(canvas_width),
                "canvas_height": int(canvas_height),
                "plan_sha256": plan["plan_sha256"],
                "preview": str(args.preview.resolve()) if args.preview else None,
            },
            "detector": plan["detector"],
            "shots": plan["shots"],
            "metrics": plan["metrics"],
            "trajectory": _track_metrics(plan),
            "state_counts": {
                state: sum(record["state"] == state for record in plan["frames"])
                for state in sorted({record["state"] for record in plan["frames"]})
            },
            "tensor_shapes": {
                "source_frames": list(frames.shape),
                "crops": list(crops.shape),
                "preview": list(preview.shape),
            },
            "elapsed_seconds": time.perf_counter() - started,
        }
    finally:
        del plan, crops, preview, frames
        gc.collect()
        time.sleep(2.0)
        stopping.set()
        monitor_thread.join(timeout=2.0)

    current = process.memory_info()
    result["process_memory"] = {
        "baseline_rss_mib": samples[0]["rss"] / 2**20 if samples else None,
        "peak_rss_mib": max(item["rss"] for item in samples) / 2**20 if samples else None,
        "post_gc_rss_mib": current.rss / 2**20,
        "peak_private_mib": max(item["private"] or 0 for item in samples) / 2**20
        if samples
        else None,
        "sample_count": len(samples),
    }
    _write_json_atomic(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
