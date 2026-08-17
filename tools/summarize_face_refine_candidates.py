#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import cv2
import numpy as np
from skimage.metrics import structural_similarity

try:
    from .probe_face_refine_plan import _decode_video, _load_face_refine_module
except ImportError:  # Direct script execution adds tools/ rather than the package root.
    from probe_face_refine_plan import _decode_video, _load_face_refine_module


SCHEMA = "t8.face_refine_candidate_proxy_summary.v1"


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


def metric_summary(values: list[float]) -> dict[str, Any]:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    if finite.size == 0:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": int(finite.size),
        "mean": float(finite.mean()),
        "median": float(np.median(finite)),
        "min": float(finite.min()),
        "max": float(finite.max()),
        "p05": float(np.percentile(finite, 5)),
        "p95": float(np.percentile(finite, 95)),
    }


def expanded_face_roi(
    box: list[float], width: int, height: int, scale: float = 1.5
) -> tuple[int, int, int, int]:
    center_x = (box[0] + box[2]) * 0.5
    center_y = (box[1] + box[3]) * 0.5
    box_width = max(1.0, box[2] - box[0]) * scale
    box_height = max(1.0, box[3] - box[1]) * scale
    left = max(0, int(math.floor(center_x - box_width * 0.5)))
    top = max(0, int(math.floor(center_y - box_height * 0.5)))
    right = min(width, int(math.ceil(center_x + box_width * 0.5)))
    bottom = min(height, int(math.ceil(center_y + box_height * 0.5)))
    if right - left < 7 or bottom - top < 7:
        raise ValueError(f"Face ROI is too small for SSIM: {(left, top, right, bottom)}")
    return left, top, right, bottom


def _gray(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0


def _normalized_crop(gray: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    left, top, right, bottom = roi
    return cv2.resize(gray[top:bottom, left:right], (128, 128), interpolation=cv2.INTER_AREA)


def _pearson(first: np.ndarray, second: np.ndarray) -> float:
    first = first.reshape(-1).astype(np.float64)
    second = second.reshape(-1).astype(np.float64)
    first -= first.mean()
    second -= second.mean()
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    return float(np.dot(first, second) / denominator) if denominator > 1e-12 else 1.0


def analyze_candidate(
    source_frames: np.ndarray,
    source_gray: list[np.ndarray],
    plan: dict[str, Any],
    candidate_path: Path,
    source_contract: dict[str, Any],
) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(candidate_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open candidate: {candidate_path}")
    contract = {
        "frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
    }
    for key in ("frame_count", "width", "height"):
        if contract[key] != source_contract[key]:
            capture.release()
            raise ValueError(
                f"Candidate {candidate_path.name} differs from source at {key}: "
                f"{contract[key]} != {source_contract[key]}"
            )
    if abs(contract["fps"] - source_contract["fps"]) > 0.01:
        capture.release()
        raise ValueError(
            f"Candidate {candidate_path.name} differs from source fps: "
            f"{contract['fps']} != {source_contract['fps']}"
        )

    metrics: dict[str, list[float]] = {
        "full_gray_ssim": [],
        "full_rgb_mae": [],
        "face_gray_ssim": [],
        "face_rgb_mae": [],
        "face_laplacian_source": [],
        "face_laplacian_candidate": [],
        "face_laplacian_ratio_candidate_over_source": [],
        "face_motion_correlation": [],
        "face_residual_temporal_jitter_mae": [],
    }
    previous_source_crop = None
    previous_candidate_crop = None
    previous_residual = None
    decoded = 0
    try:
        while decoded < source_contract["frame_count"]:
            ok, bgr = capture.read()
            if not ok:
                break
            candidate_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            candidate_gray = _gray(candidate_rgb)
            original_rgb = source_frames[decoded]
            original_gray = source_gray[decoded]
            metrics["full_gray_ssim"].append(
                float(structural_similarity(original_gray, candidate_gray, data_range=1.0))
            )
            metrics["full_rgb_mae"].append(
                float(
                    np.mean(
                        np.abs(
                            original_rgb.astype(np.float32)
                            - candidate_rgb.astype(np.float32)
                        )
                    )
                    / 255.0
                )
            )
            record = plan["frames"][decoded]
            if record["state"] not in {"lost", "reacquired_unverified"}:
                roi = expanded_face_roi(
                    record["source_face_box_xyxy"],
                    source_contract["width"],
                    source_contract["height"],
                )
                left, top, right, bottom = roi
                original_face_gray = original_gray[top:bottom, left:right]
                candidate_face_gray = candidate_gray[top:bottom, left:right]
                metrics["face_gray_ssim"].append(
                    float(
                        structural_similarity(
                            original_face_gray, candidate_face_gray, data_range=1.0
                        )
                    )
                )
                metrics["face_rgb_mae"].append(
                    float(
                        np.mean(
                            np.abs(
                                original_rgb[top:bottom, left:right].astype(np.float32)
                                - candidate_rgb[top:bottom, left:right].astype(np.float32)
                            )
                        )
                        / 255.0
                    )
                )
                source_laplacian = float(
                    cv2.Laplacian(original_face_gray, cv2.CV_32F).var()
                )
                candidate_laplacian = float(
                    cv2.Laplacian(candidate_face_gray, cv2.CV_32F).var()
                )
                metrics["face_laplacian_source"].append(source_laplacian)
                metrics["face_laplacian_candidate"].append(candidate_laplacian)
                metrics["face_laplacian_ratio_candidate_over_source"].append(
                    candidate_laplacian / max(source_laplacian, 1e-9)
                )
                source_crop = _normalized_crop(original_gray, roi)
                candidate_crop = _normalized_crop(candidate_gray, roi)
                residual = candidate_crop - source_crop
                if previous_source_crop is not None and previous_candidate_crop is not None:
                    metrics["face_motion_correlation"].append(
                        _pearson(
                            source_crop - previous_source_crop,
                            candidate_crop - previous_candidate_crop,
                        )
                    )
                if previous_residual is not None:
                    metrics["face_residual_temporal_jitter_mae"].append(
                        float(np.mean(np.abs(residual - previous_residual)))
                    )
                previous_source_crop = source_crop
                previous_candidate_crop = candidate_crop
                previous_residual = residual
            else:
                previous_source_crop = None
                previous_candidate_crop = None
                previous_residual = None
            decoded += 1
    finally:
        capture.release()
    if decoded != source_contract["frame_count"]:
        raise RuntimeError(
            f"Candidate {candidate_path.name} decoded {decoded} frames, expected "
            f"{source_contract['frame_count']}"
        )
    return {
        "file": candidate_path.name,
        "sha256": _sha256_file(candidate_path),
        "contract": contract,
        "metrics": {key: metric_summary(values) for key, values in metrics.items()},
    }


def summarize(
    source_path: Path,
    candidates: list[Path],
    repo_root: Path,
    comfy_root: Path,
    canvas: str,
) -> dict[str, Any]:
    frames_tensor, source_contract = _decode_video(source_path)
    if abs(source_contract["fps"] - 24.0) > 0.01:
        raise ValueError(f"Face Refine requires exact 24fps; got {source_contract['fps']}")
    module = _load_face_refine_module(repo_root, comfy_root)
    plan = crops = preview = None
    try:
        plan, crops, preview, _, _, _, _ = module.build_face_refine_plan(
            frames=frames_tensor,
            fps=24.0,
            detector_mode="local_opencv_yunet",
            detector_model=module.YUNET_2023MAR_RELATIVE,
            detector_device="cpu",
            confidence=0.35,
            manual_roi_x=0.30,
            manual_roi_y=0.10,
            manual_roi_width=0.40,
            manual_roi_height=0.55,
            scene_cut_threshold=0.28,
            max_track_jump=0.18,
            max_gap_frames=4,
            smoothing_radius=2,
            crop_context_scale=3.0,
            canvas_size=canvas,
            require_h3_grid=True,
            analysis_chunk_frames=8,
        )
        source_frames = (
            frames_tensor.detach().clamp(0.0, 1.0).mul(255.0).byte().cpu().numpy()
        )
        source_gray = [_gray(frame) for frame in source_frames]
        candidate_results = [
            analyze_candidate(
                source_frames, source_gray, plan, path.resolve(), source_contract
            )
            for path in candidates
        ]
        return {
            "schema": SCHEMA,
            "status": "automatic_proxy_signals_only_human_blind_review_required",
            "source": {
                "file": source_path.name,
                "sha256": _sha256_file(source_path),
                **source_contract,
            },
            "plan": {
                "sha256": plan["plan_sha256"],
                "detector": plan["detector"],
                "metrics": plan["metrics"],
                "canvas": canvas,
            },
            "candidate_count": len(candidate_results),
            "candidates": candidate_results,
            "limitations": [
                "SSIM, MAE, Laplacian and temporal correlations are source-similarity proxies, not proof of improved quality.",
                "Laplacian variance can reward noise or ringing and cannot establish facial detail restoration.",
                "No face-recognition model is used; identity preservation is not measured.",
                "Metrics are computed after video codec round trips and cannot prove tensor-level outside-mask equality.",
                "The randomized human review remains the authority for identity, naturalness, expression and preference.",
            ],
            "quality_promotion": False,
            "identity_validated": False,
        }
    finally:
        del plan, crops, preview, frames_tensor
        gc.collect()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize source-similarity and temporal proxy signals for Face Refine "
            "candidates. This does not replace blind review."
        )
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--canvas", choices=("384", "512", "640", "768"), default="512")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    candidates = [path.resolve() for path in args.candidate]
    if not source.is_file() or any(not path.is_file() for path in candidates):
        raise FileNotFoundError("Source and every candidate must exist")
    repo_root = Path(__file__).resolve().parents[1]
    result = summarize(source, candidates, repo_root, repo_root.parent.parent, args.canvas)
    _write_json_atomic(args.output.resolve(), result)
    print(
        json.dumps(
            {
                "source": result["source"]["file"],
                "candidates": result["candidate_count"],
                "output": str(args.output.resolve()),
                "quality_promotion": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
