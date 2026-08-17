#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any


SCHEMA = "t8.face_refine_yunet_wider_eval.v1"
WIDER_VAL_SHA256 = "f9efbd09f28c5d2d884be8c0eaef3967158c866a593fc36ab0413e4b2a58a17a"
WIDER_SPLIT_SHA256 = "c7561e4f5e7a118c249e0a5c5c902b0de90bbf120d7da9fa28d99041f68a8a5c"
YUNET_2023MAR_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"


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


def parse_wider_annotations(path: Path) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    records = []
    cursor = 0
    while cursor < len(lines):
        relative = lines[cursor].strip()
        cursor += 1
        if not relative:
            continue
        if not relative.lower().endswith(".jpg"):
            raise ValueError(f"Expected an image path at line {cursor}: {relative!r}")
        if cursor >= len(lines):
            raise ValueError(f"Missing face count after {relative}")
        count = int(lines[cursor].strip())
        cursor += 1
        boxes = []
        rows_to_read = max(1, count)
        for row_index in range(rows_to_read):
            if cursor >= len(lines):
                raise ValueError(f"Truncated annotations after {relative}")
            values = [int(value) for value in lines[cursor].split()]
            cursor += 1
            if len(values) != 10:
                raise ValueError(
                    f"Expected ten values for {relative} row {row_index}; got {values}"
                )
            if count == 0:
                continue
            x, y, width, height, blur, expression, illumination, invalid, occlusion, pose = (
                values
            )
            boxes.append(
                {
                    "box": [float(x), float(y), float(x + width), float(y + height)],
                    "width": int(width),
                    "height": int(height),
                    "blur": int(blur),
                    "expression": int(expression),
                    "illumination": int(illumination),
                    "invalid": bool(invalid),
                    "occlusion": int(occlusion),
                    "pose": int(pose),
                }
            )
        records.append({"relative_path": relative, "boxes": boxes})
    return records


def _iou(first: list[float], second: list[float]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    return intersection / max(first_area + second_area - intersection, 1e-9)


def match_detections(
    detections: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    iou_threshold: float,
) -> dict[str, Any]:
    valid = [item for item in annotations if not item["invalid"]]
    ignored = [item for item in annotations if item["invalid"]]
    matched_valid: set[int] = set()
    matches = []
    false_positives = 0
    ignored_detections = 0
    for detection in sorted(
        detections, key=lambda item: float(item["confidence"]), reverse=True
    ):
        candidates = [
            (_iou(detection["box"], annotation["box"]), index)
            for index, annotation in enumerate(valid)
            if index not in matched_valid
        ]
        best_iou, best_index = max(candidates, default=(0.0, -1))
        if best_iou >= iou_threshold:
            matched_valid.add(best_index)
            matches.append(
                {
                    "annotation_index": best_index,
                    "iou": best_iou,
                    "confidence": float(detection["confidence"]),
                }
            )
            continue
        if any(_iou(detection["box"], item["box"]) >= iou_threshold for item in ignored):
            ignored_detections += 1
        else:
            false_positives += 1
    return {
        "true_positives": len(matches),
        "false_positives": false_positives,
        "false_negatives": len(valid) - len(matches),
        "ignored_detections": ignored_detections,
        "matched_annotation_indices": sorted(matched_valid),
        "matches": matches,
    }


def _size_bin(width: int, height: int) -> str:
    minimum = min(width, height)
    if minimum < 16:
        return "tiny_lt16px"
    if minimum < 32:
        return "small_16_31px"
    if minimum < 64:
        return "medium_32_63px"
    return "large_ge64px"


def _ratio_bin(width: int, height: int, image_width: int, image_height: int) -> str:
    ratio = min(width, height) / max(1, min(image_width, image_height))
    if ratio < 0.02:
        return "tiny_lt2pct"
    if ratio < 0.05:
        return "small_2_to_5pct"
    return "large_ge5pct"


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _finalize_counts(counts: dict[str, int]) -> dict[str, Any]:
    true_positives = counts["true_positives"]
    false_positives = counts["false_positives"]
    false_negatives = counts["false_negatives"]
    precision = _safe_ratio(true_positives, true_positives + false_positives)
    recall = _safe_ratio(true_positives, true_positives + false_negatives)
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall > 0.0
        else None
    )
    return {**counts, "precision": precision, "recall": recall, "f1": f1}


def _update_recall_bin(
    table: dict[str, dict[str, int]], key: str, matched: bool
) -> None:
    row = table.setdefault(key, {"ground_truth": 0, "matched": 0})
    row["ground_truth"] += 1
    row["matched"] += int(matched)


def evaluate(
    image_root: Path,
    annotation_path: Path,
    model_path: Path,
    thresholds: list[float],
    iou_threshold: float,
    max_images: int | None = None,
) -> dict[str, Any]:
    try:
        import cv2
    except Exception as error:
        raise RuntimeError("This evaluation requires OpenCV FaceDetectorYN") from error
    if not hasattr(cv2, "FaceDetectorYN"):
        raise RuntimeError("This OpenCV build does not provide FaceDetectorYN")

    records = parse_wider_annotations(annotation_path)
    if max_images is not None:
        if max_images <= 0:
            raise ValueError("max_images must be positive")
        records = records[:max_images]
    detectors = {
        threshold: cv2.FaceDetectorYN.create(
            str(model_path), "", (320, 320), float(threshold), 0.30, 5000
        )
        for threshold in thresholds
    }
    summaries = {
        str(threshold): {
            "counts": {
                "images": 0,
                "images_with_valid_faces": 0,
                "images_with_detections": 0,
                "ground_truth_valid": 0,
                "ground_truth_invalid": 0,
                "detections": 0,
                "true_positives": 0,
                "false_positives": 0,
                "false_negatives": 0,
                "ignored_detections": 0,
            },
            "pixel_size_recall": {},
            "relative_size_recall": {},
        }
        for threshold in thresholds
    }
    started = time.perf_counter()
    for record_index, record in enumerate(records, 1):
        image_path = image_root / record["relative_path"]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Cannot decode {image_path}")
        image_height, image_width = image.shape[:2]
        valid_annotations = [item for item in record["boxes"] if not item["invalid"]]
        for threshold, detector in detectors.items():
            detector.setInputSize((image_width, image_height))
            _, raw = detector.detect(image)
            detections = []
            if raw is not None:
                for row in raw:
                    values = [float(value) for value in row.tolist()]
                    x, y, width, height = values[:4]
                    detections.append(
                        {
                            "box": [x, y, x + width, y + height],
                            "confidence": values[14] if len(values) >= 15 else values[-1],
                        }
                    )
            matched = match_detections(detections, record["boxes"], iou_threshold)
            summary = summaries[str(threshold)]
            counts = summary["counts"]
            counts["images"] += 1
            counts["images_with_valid_faces"] += int(bool(valid_annotations))
            counts["images_with_detections"] += int(bool(detections))
            counts["ground_truth_valid"] += len(valid_annotations)
            counts["ground_truth_invalid"] += sum(
                item["invalid"] for item in record["boxes"]
            )
            counts["detections"] += len(detections)
            for key in (
                "true_positives",
                "false_positives",
                "false_negatives",
                "ignored_detections",
            ):
                counts[key] += int(matched[key])
            matched_indices = set(matched["matched_annotation_indices"])
            for annotation_index, annotation in enumerate(valid_annotations):
                is_matched = annotation_index in matched_indices
                _update_recall_bin(
                    summary["pixel_size_recall"],
                    _size_bin(annotation["width"], annotation["height"]),
                    is_matched,
                )
                _update_recall_bin(
                    summary["relative_size_recall"],
                    _ratio_bin(
                        annotation["width"],
                        annotation["height"],
                        image_width,
                        image_height,
                    ),
                    is_matched,
                )
        if record_index % 100 == 0 or record_index == len(records):
            elapsed = time.perf_counter() - started
            rate = record_index / max(elapsed, 1e-9)
            print(
                f"evaluated {record_index}/{len(records)} images "
                f"({rate:.2f} images/s)",
                flush=True,
            )

    for summary in summaries.values():
        summary["metrics"] = _finalize_counts(
            {
                key: int(summary["counts"][key])
                for key in (
                    "true_positives",
                    "false_positives",
                    "false_negatives",
                    "ignored_detections",
                )
            }
        )
        for table_name in ("pixel_size_recall", "relative_size_recall"):
            for row in summary[table_name].values():
                row["recall"] = _safe_ratio(row["matched"], row["ground_truth"])

    return {
        "schema": SCHEMA,
        "status": "fixed_threshold_integration_metrics_not_official_wider_ap",
        "scope": {
            "images": len(records),
            "thresholds": thresholds,
            "iou_threshold": iou_threshold,
            "invalid_ground_truth_policy": (
                "invalid boxes are excluded from recall; detections overlapping only an "
                "invalid box at the IoU threshold are ignored rather than counted false-positive"
            ),
            "limitations": [
                "These are fixed-threshold integration metrics, not the official WIDER easy/medium/hard AP protocol.",
                "Detection accuracy does not measure temporal identity tracking or H3 repair quality.",
                "WIDER FACE is CC BY-NC-ND 4.0 and is used locally for non-commercial evaluation only.",
            ],
        },
        "dataset": {
            "name": "WIDER FACE validation",
            "images_root": str(image_root.resolve()),
            "annotation_file": str(annotation_path.resolve()),
            "source": "https://huggingface.co/datasets/CUHK-CSE/wider_face",
            "license": "CC-BY-NC-ND-4.0",
            "expected_validation_zip_sha256": WIDER_VAL_SHA256,
            "expected_annotation_zip_sha256": WIDER_SPLIT_SHA256,
        },
        "model": {
            "path": str(model_path.resolve()),
            "sha256": _sha256_file(model_path),
            "expected_yunet_2023mar_sha256": YUNET_2023MAR_SHA256,
            "official_model_match": _sha256_file(model_path) == YUNET_2023MAR_SHA256,
            "backend": "OpenCV FaceDetectorYN CPU",
            "nms_threshold": 0.30,
            "top_k": 5000,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "threshold_results": summaries,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the pinned Face Refine YuNet integration on local WIDER FACE "
            "validation images. This tool never downloads data or models."
        )
    )
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--threshold", type=float, action="append", required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    thresholds = list(dict.fromkeys(float(value) for value in args.threshold))
    if any(not math.isfinite(value) or not 0.0 < value <= 1.0 for value in thresholds):
        raise ValueError("Every threshold must be finite and in (0, 1]")
    if not 0.0 < args.iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in (0, 1]")
    for path in (args.image_root, args.annotations, args.model):
        if not path.exists():
            raise FileNotFoundError(path)
    result = evaluate(
        args.image_root.resolve(),
        args.annotations.resolve(),
        args.model.resolve(),
        thresholds,
        float(args.iou_threshold),
        args.max_images,
    )
    _write_json_atomic(args.output.resolve(), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
