#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any

try:
    from .probe_face_refine_plan import _load_face_refine_module
except ImportError:  # Direct script execution adds tools/ rather than the package root.
    from probe_face_refine_plan import _load_face_refine_module


SCHEMA = "t8.face_refine_tracker_crossing_audit.v1"


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


def _box(center_x: float, center_y: float, size: float) -> list[float]:
    half = size * 0.5
    return [center_x - half, center_y - half, center_x + half, center_y + half]


def build_crossing_scenario(
    name: str, frame_count: int = 61
) -> tuple[list[list[dict[str, Any]]], list[dict[str, list[float]]]]:
    if frame_count < 9 or frame_count % 2 == 0:
        raise ValueError("frame_count must be an odd integer of at least 9")
    midpoint = frame_count // 2
    detections = []
    truth = []
    for index in range(frame_count):
        mix = index / (frame_count - 1)
        a_y = 90.0
        b_y = 90.0
        if name == "vertical_separation_control":
            a_y = 75.0
            b_y = 105.0
        a_box = _box(35.0 + 150.0 * mix, a_y, 24.0)
        b_box = _box(185.0 - 150.0 * mix, b_y, 24.0)
        truth.append({"A": a_box, "B": b_box})
        a = {"box": a_box, "confidence": 0.92, "synthetic_identity": "A"}
        b = {"box": b_box, "confidence": 0.91, "synthetic_identity": "B"}

        if name == "stable_a_first" or name == "vertical_separation_control":
            frame_detections = [a, b]
        elif name == "confidence_order_flips_after_overlap":
            if index >= midpoint:
                b["confidence"] = 0.96
                frame_detections = [b, a]
            else:
                frame_detections = [a, b]
        elif name == "alternating_order_near_overlap":
            if midpoint - 4 <= index <= midpoint + 4 and index % 2:
                frame_detections = [b, a]
            else:
                frame_detections = [a, b]
        elif name == "target_a_occluded_three_frames":
            frame_detections = [b] if abs(index - midpoint) <= 1 else [a, b]
        else:
            raise ValueError(f"Unknown scenario: {name}")
        detections.append(frame_detections)
    return detections, truth


def _center(box: list[float]) -> tuple[float, float]:
    return ((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5)


def _identity_for_box(
    selected: list[float] | None, truth: dict[str, list[float]]
) -> str:
    if selected is None:
        return "lost"
    selected_center = _center(selected)
    distances = {
        identity: (
            (selected_center[0] - _center(box)[0]) ** 2
            + (selected_center[1] - _center(box)[1]) ** 2
        )
        ** 0.5
        for identity, box in truth.items()
    }
    if abs(distances["A"] - distances["B"]) <= 1e-6:
        return "ambiguous_overlap"
    return min(distances, key=distances.get)


def summarize_crossing(
    selected_boxes: list[list[float] | None],
    states: list[str],
    truth: list[dict[str, list[float]]],
) -> dict[str, Any]:
    identities = [
        _identity_for_box(selected, frame_truth)
        for selected, frame_truth in zip(selected_boxes, truth, strict=True)
    ]
    unambiguous = [value for value in identities if value in {"A", "B"}]
    transitions = sum(
        current != previous
        for previous, current in zip(unambiguous, unambiguous[1:], strict=False)
    )
    first_identity = unambiguous[0] if unambiguous else None
    final_identity = unambiguous[-1] if unambiguous else None
    return {
        "target_identity": "A",
        "first_identity": first_identity,
        "final_identity": final_identity,
        "identity_transition_count": transitions,
        "target_a_frames": identities.count("A"),
        "wrong_b_frames": identities.count("B"),
        "ambiguous_overlap_frames": identities.count("ambiguous_overlap"),
        "lost_frames": identities.count("lost"),
        "state_counts": {state: states.count(state) for state in sorted(set(states))},
        "swap_detected": first_identity == "A"
        and (final_identity == "B" or identities.count("B") > 0),
        "identity_sequence_rle": _run_length_encode(identities),
    }


def _run_length_encode(values: list[str]) -> list[dict[str, Any]]:
    output = []
    for index, value in enumerate(values):
        if not output or output[-1]["value"] != value:
            output.append({"value": value, "start_frame": index, "end_frame": index})
        else:
            output[-1]["end_frame"] = index
    return output


def audit(repo_root: Path, comfy_root: Path) -> dict[str, Any]:
    module = _load_face_refine_module(repo_root, comfy_root)
    scenarios = [
        "stable_a_first",
        "confidence_order_flips_after_overlap",
        "alternating_order_near_overlap",
        "target_a_occluded_three_frames",
        "vertical_separation_control",
    ]
    results = {}
    for name in scenarios:
        detections, truth = build_crossing_scenario(name)
        boxes, states, _, multi_face_frames = module._select_track(
            detections,
            [(0, len(detections) - 1)],
            220,
            180,
            0.18,
            4,
        )
        summary = summarize_crossing(boxes, states, truth)
        summary["multi_face_frames"] = multi_face_frames
        results[name] = summary
    swaps = [name for name, result in results.items() if result["swap_detected"]]
    return {
        "schema": SCHEMA,
        "status": "identity_exchange_risk_demonstrated" if swaps else "no_exchange_observed",
        "contract": {
            "frame_count": 61,
            "canvas": [220, 180],
            "target_identity": "A",
            "face_size_px": 24,
            "max_track_jump": 0.18,
            "max_gap_frames": 4,
            "appearance_features": False,
            "identity_verification": False,
            "purpose": (
                "deterministic geometry-only tracker stress; this is not detector precision "
                "or generated-video quality evidence"
            ),
        },
        "swap_scenarios": swaps,
        "results": results,
        "decision": (
            "A detection-order or short-occlusion swap proves the current tracker cannot be "
            "advertised as identity-safe. Multi-face plans require preview review and cannot "
            "be automatically accepted."
            if swaps
            else "No controlled swap was observed; real identity validation is still required."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit Face Refine geometry-only tracking under controlled face crossings."
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    result = audit(repo_root, repo_root.parent.parent)
    _write_json_atomic(args.output.resolve(), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
