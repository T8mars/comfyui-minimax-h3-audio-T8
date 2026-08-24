#!/usr/bin/env python3
"""Analyze one same-input MiniMax H3 4B/8B/native-32B T2VA triplet.

This tool is deliberately report-only. It verifies the shared generation contract, strictly
decodes every AV file, reports per-output signals and computes all three pairwise differences.
Similarity is never interpreted as quality, and one observed runtime is never promoted to a
general speed or memory claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

try:
    from .analyze_h3_speed_quality_pairs import (
        _audio_stats,
        _decode_audio,
        _decode_video,
        _pair_metrics,
        _strict_decode,
        _video_stats,
    )
except ImportError:  # pragma: no cover - direct script execution
    from analyze_h3_speed_quality_pairs import (
        _audio_stats,
        _decode_audio,
        _decode_video,
        _pair_metrics,
        _strict_decode,
        _video_stats,
    )


SCHEMA = "t8.minimax_h3.clipproj_same_input_three_way.v1"
ARM_ORDER = ("clipproj_4b", "clipproj_8b", "native_32b")
PAIR_ORDER = (
    ("clipproj_4b", "clipproj_8b"),
    ("clipproj_4b", "native_32b"),
    ("clipproj_8b", "native_32b"),
)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _one_node(graph: Mapping[str, Any], class_type: str) -> Mapping[str, Any]:
    rows = [
        value
        for value in graph.values()
        if isinstance(value, Mapping) and value.get("class_type") == class_type
    ]
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one {class_type} node, found {len(rows)}")
    return rows[0]


def extract_generation_contract(graph: Mapping[str, Any]) -> dict[str, Any]:
    conditioning = _one_node(graph, "MiniMaxH3AudioConditioningT8").get("inputs", {})
    sampler = _one_node(graph, "MiniMaxH3DualClockSamplerT8").get("inputs", {})
    noise = _one_node(graph, "RandomNoise").get("inputs", {})
    contract = {
        "prompt": str(conditioning.get("prompt", "")),
        "seed": int(noise["noise_seed"]),
        "width": int(conditioning["width"]),
        "height": int(conditioning["height"]),
        "frame_count": int(conditioning["length"]),
        "task_type": str(conditioning["task_type"]),
        "audio_mode": str(conditioning["audio_mode"]),
        "audio_denoise_strength": float(conditioning["audio_denoise_strength"]),
        "steps": int(sampler["steps"]),
        "shift_video": float(sampler["shift_video"]),
        "shift_audio": float(sampler["shift_audio"]),
    }
    if not contract["prompt"].strip():
        raise ValueError("Generation prompt must be non-empty")
    return contract


def require_equal_contracts(contracts: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    missing = set(ARM_ORDER) - set(contracts)
    if missing:
        raise ValueError(f"Missing arm contracts: {sorted(missing)}")
    reference = dict(contracts[ARM_ORDER[0]])
    mismatches: dict[str, dict[str, Any]] = {}
    for arm in ARM_ORDER[1:]:
        current = dict(contracts[arm])
        if current != reference:
            fields = sorted(set(reference) | set(current))
            mismatches[arm] = {
                field: {"reference": reference.get(field), "actual": current.get(field)}
                for field in fields
                if reference.get(field) != current.get(field)
            }
    if mismatches:
        raise ValueError(
            "Generation contracts are not identical: "
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )
    encoded = json.dumps(
        reference, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {**reference, "contract_sha256": hashlib.sha256(encoded).hexdigest().upper()}


def runtime_observation(
    report: Mapping[str, Any], *, source_kind: str
) -> dict[str, Any]:
    if source_kind == "guarded_probe":
        monitor = report.get("gpu", {}).get("monitor", {})
        prompt_seconds = report.get("phase", {}).get("elapsed_seconds")
        return {
            "source_kind": source_kind,
            "prompt_to_terminal_seconds": float(prompt_seconds),
            "whole_device_peak_mib": float(monitor["peak_used_mib"]),
            "minimum_free_mib": float(monitor["minimum_free_mib"]),
            "baseline_used_mib": float(report.get("gpu", {}).get("baseline", {})["used_mib"]),
            "terminal_success": report.get("status") == "PASS" and bool(report.get("passed")),
        }
    if source_kind == "runtime_trace":
        runtime = report["runtime"]
        summary = runtime["summary"]
        total_bytes = int(runtime["server_snapshot"]["devices"][0]["vram_total"])
        peak_bytes = int(summary["peak_vram_used_bytes"])
        return {
            "source_kind": source_kind,
            "prompt_to_terminal_seconds": float(runtime["duration_seconds"]),
            "whole_device_peak_mib": peak_bytes / (1024**2),
            "minimum_free_mib": (total_bytes - peak_bytes) / (1024**2),
            "baseline_used_mib": int(summary["baseline_vram_used_bytes"]) / (1024**2),
            "terminal_success": runtime.get("status") == "success",
        }
    raise ValueError(f"Unsupported runtime source kind: {source_kind}")


def _spectral_magnitude_cosine(first: np.ndarray, second: np.ndarray) -> float | None:
    sample_count = min(len(first), len(second))
    if sample_count < 2:
        return None
    a = first[:sample_count].mean(axis=1).astype(np.float64)
    b = second[:sample_count].mean(axis=1).astype(np.float64)
    window = np.hanning(sample_count)
    spectrum_a = np.abs(np.fft.rfft(a * window))
    spectrum_b = np.abs(np.fft.rfft(b * window))
    denominator = float(np.linalg.norm(spectrum_a) * np.linalg.norm(spectrum_b))
    return float(np.dot(spectrum_a, spectrum_b) / denominator) if denominator else None


def analyze(
    *,
    graphs: Mapping[str, Path],
    runtimes: Mapping[str, tuple[Path, str]],
    media: Mapping[str, Path],
    ffmpeg: str,
) -> dict[str, Any]:
    contracts = {
        arm: extract_generation_contract(_load_json(graphs[arm])) for arm in ARM_ORDER
    }
    common_contract = require_equal_contracts(contracts)
    expected_frames = int(common_contract["frame_count"])
    decoded: dict[str, dict[str, Any]] = {}
    observations: dict[str, dict[str, Any]] = {}
    mechanical_checks: dict[str, bool] = {}
    for arm in ARM_ORDER:
        media_path = media[arm].resolve()
        if not media_path.is_file():
            raise FileNotFoundError(media_path)
        frames, fps = _decode_video(media_path)
        audio = _decode_audio(media_path, ffmpeg)
        strict = _strict_decode(media_path, ffmpeg, attempts=3)
        decoded[arm] = {"frames": frames, "audio": audio}
        observations[arm] = {
            "media_path": str(media_path),
            "media_sha256": _sha256_file(media_path),
            "fps": fps,
            "video": _video_stats(frames),
            "audio": _audio_stats(audio),
            "runtime": runtime_observation(
                _load_json(runtimes[arm][0]), source_kind=runtimes[arm][1]
            ),
        }
        mechanical_checks[f"{arm}_strict_decode_3_of_3"] = bool(strict["passed"])
        mechanical_checks[f"{arm}_frame_count_exact"] = len(frames) == expected_frames
        mechanical_checks[f"{arm}_fps_24"] = abs(fps - 24.0) <= 1e-6
        mechanical_checks[f"{arm}_audio_finite"] = bool(np.isfinite(audio).all())
        mechanical_checks[f"{arm}_runtime_success"] = bool(
            observations[arm]["runtime"]["terminal_success"]
        )
    pairwise: dict[str, Any] = {}
    for first, second in PAIR_ORDER:
        metrics = _pair_metrics(
            decoded[first]["frames"],
            decoded[second]["frames"],
            decoded[first]["audio"],
            decoded[second]["audio"],
        )
        metrics["audio"]["spectral_magnitude_cosine"] = _spectral_magnitude_cosine(
            decoded[first]["audio"], decoded[second]["audio"]
        )
        pairwise[f"{first}_vs_{second}"] = metrics
    return {
        "schema": SCHEMA,
        "common_generation_contract": common_contract,
        "arms": observations,
        "pairwise": pairwise,
        "mechanical_checks": mechanical_checks,
        "mechanical_pass": all(mechanical_checks.values()),
        "claims": {
            "perceptual_quality_equivalence": False,
            "audio_noninferiority": False,
            "speed_superiority": False,
            "memory_superiority": False,
            "general_16gb_safety": False,
        },
        "limitations": [
            "SSIM, MAE, sharpness, temporal difference and audio similarity measure route difference, not which output is better.",
            "Runtime traces were collected in separate isolated executions and are single observations, not a randomized benchmark.",
            "This evidence covers one T2VA rain prompt, one seed, 256x256x22 and four NFE only.",
            "Human full-video and full-audio blind review remains required.",
        ],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for arm in ARM_ORDER:
        parser.add_argument(f"--{arm.replace('_', '-')}-graph", type=Path, required=True)
        parser.add_argument(f"--{arm.replace('_', '-')}-runtime", type=Path, required=True)
        parser.add_argument(f"--{arm.replace('_', '-')}-media", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    graphs = {
        arm: getattr(args, f"{arm}_graph") for arm in ARM_ORDER
    }
    media = {
        arm: getattr(args, f"{arm}_media") for arm in ARM_ORDER
    }
    runtimes = {
        "clipproj_4b": (args.clipproj_4b_runtime, "guarded_probe"),
        "clipproj_8b": (args.clipproj_8b_runtime, "runtime_trace"),
        "native_32b": (args.native_32b_runtime, "runtime_trace"),
    }
    report = analyze(
        graphs=graphs,
        runtimes=runtimes,
        media=media,
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
                "contract_sha256": report["common_generation_contract"]["contract_sha256"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["mechanical_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
