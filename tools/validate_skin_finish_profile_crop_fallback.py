#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

from PIL import Image, ImageDraw, ImageFont
import torch

import validate_skin_finish_multiface_semantic_representative as base


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "h3_audio_t8_skin_profile_crop_validation"
SOURCE_INDICES = [0, 32, 43, 48, 51, 68]
DEFAULT_OUTPUT = (
    ROOT / "artifacts" / "skin-finish-profile-crop-fallback-6frame-20260825"
)


def _font(size: int) -> ImageFont.ImageFont:
    for path in (
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\msyh.ttc"),
    ):
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _contact_sheet(
    frames: torch.Tensor,
    strict_preview: torch.Tensor,
    profile_preview: torch.Tensor,
    output: Path,
) -> None:
    tile_width, tile_height, header = 480, 352, 38
    rows = (
        ("SOURCE", frames),
        ("STRICT FIVE-POINT", strict_preview),
        ("PROFILE-CROP FALLBACK", profile_preview),
    )
    canvas = Image.new(
        "RGB",
        (tile_width * len(SOURCE_INDICES), (tile_height + header) * len(rows)),
        (25, 27, 31),
    )
    draw = ImageDraw.Draw(canvas)
    font = _font(22)
    for row_index, (label, batch) in enumerate(rows):
        for column, source_index in enumerate(SOURCE_INDICES):
            x = column * tile_width
            y = row_index * (tile_height + header)
            image = base._to_image(batch[column]).resize(
                (tile_width, tile_height),
                Image.Resampling.LANCZOS,
            )
            canvas.paste(image, (x, y + header))
            draw.text(
                (x + 8, y + 7),
                f"{label}  F{source_index}",
                font=font,
                fill=(245, 245, 245),
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=True)


def _accepted_counts(report: dict) -> list[int]:
    return [int(item["accepted_track_count"]) for item in report["frames"]]


def _skin_areas(mask: torch.Tensor) -> list[float]:
    return [
        round(float((mask[index] > 0.05).float().mean()), 8)
        for index in range(int(mask.shape[0]))
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args()

    preflight = {
        "source": str(base.SOURCE),
        "source_exists": base.SOURCE.is_file(),
        "selected_frames": SOURCE_INDICES,
        "frame_count": len(SOURCE_INDICES),
        "model_run": "two bounded CPU ParseNet passes; no SAM or H3",
        "confirmed": bool(args.confirm_run),
    }
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "preflight.json").write_text(
        json.dumps(preflight, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if not preflight["source_exists"] or not preflight["confirmed"]:
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return 2

    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    torch.set_num_threads(2)
    base.PACKAGE_NAME = PACKAGE_NAME
    base.SOURCE_INDICES = SOURCE_INDICES
    started = time.perf_counter()
    base._load_package()
    frames = base._decode_selected(base.SOURCE)
    source_snapshot = frames.clone()
    plan = base._track_plan(frames, PACKAGE_NAME)
    module = sys.modules[f"{PACKAGE_NAME}.skin_finish_multiface_parser"]
    common = {
        "frames": frames,
        "track_plan": plan,
        "parser_model": module.PARSENET_MODEL_NAME,
        "detection_threshold": 0.45,
        "minimum_face_height_px": 32.0,
        "minimum_detail": 0.005,
        "minimum_person_overlap": 0.20,
        "minimum_track_quality": 0.08,
        "minimum_class_probability": 0.55,
        "feature_protection_px": 3,
        "include_neck": False,
        "minimum_skin_area_per_face": 0.00005,
        "maximum_skin_area_per_frame": 0.35,
        "maximum_alignment_rms": 0.08,
        "minimum_ready_frame_fraction": 1.0,
        "preview_count": len(SOURCE_INDICES),
    }
    strict_mask, strict_preview, strict_json = (
        module.run_multiface_semantic_skin_mask(
            **common,
            alignment_policy="five_point_strict",
            profile_crop_expansion=1.45,
        )
    )
    profile_mask, profile_preview, profile_json = (
        module.run_multiface_semantic_skin_mask(
            **common,
            alignment_policy="five_point_then_profile_crop",
            profile_crop_expansion=1.45,
        )
    )
    strict = json.loads(strict_json)
    profile = json.loads(profile_json)
    strict_counts = _accepted_counts(strict)
    profile_counts = _accepted_counts(profile)
    strict_total = sum(strict_counts)
    profile_total = sum(profile_counts)
    fallback_total = sum(
        int(value)
        for value in profile["selection"]["profile_crop_ready_counts"].values()
    )
    profile_areas = _skin_areas(profile_mask)
    checks = {
        "source_unchanged": torch.equal(frames, source_snapshot),
        "strict_ready": strict["status"] == "READY",
        "profile_ready": profile["status"] == "READY",
        "strict_exposes_rejections": strict_total < len(SOURCE_INDICES) * 2,
        "profile_improves_track_coverage": profile_total > strict_total,
        "profile_accepts_both_tracks_each_frame": profile_counts
        == [2] * len(SOURCE_INDICES),
        "fallback_was_actually_used": fallback_total > 0,
        "profile_mask_finite": bool(torch.isfinite(profile_mask).all()),
        "profile_mask_in_unit_interval": not bool((profile_mask < 0).any())
        and not bool((profile_mask > 1).any()),
        "profile_skin_area_bounded": all(0.0 < area < 0.10 for area in profile_areas),
        "preview_shapes_preserved": tuple(strict_preview.shape) == tuple(frames.shape)
        and tuple(profile_preview.shape) == tuple(frames.shape),
    }
    contact = output / "source_strict_profile_crop_6frames.png"
    _contact_sheet(frames, strict_preview, profile_preview, contact)
    report = {
        "schema": "h3_t8_skin_finish_profile_crop_fallback_validation/v1",
        "created_at_unix": time.time(),
        "source": {
            "path": str(base.SOURCE),
            "sha256": base._sha256(base.SOURCE),
            "selected_frame_indices": SOURCE_INDICES,
            "decoded_shape": list(frames.shape),
        },
        "method": {
            "strict": "YuNet five-point to fixed FFHQ-512 similarity; RMS <= 0.08",
            "fallback": (
                "Only after five-point ValueError: 1.45x square source face crop, ParseNet, "
                "inverse resize, then exact shot-local person-mask intersection"
            ),
            "threshold_was_not_loosened": True,
            "sam_or_h3_loaded": False,
        },
        "strict": {
            "accepted_tracks_per_frame": strict_counts,
            "accepted_track_total": strict_total,
            "report": strict,
        },
        "profile_crop": {
            "accepted_tracks_per_frame": profile_counts,
            "accepted_track_total": profile_total,
            "fallback_track_total": fallback_total,
            "skin_area_fractions": profile_areas,
            "report": profile,
        },
        "checks": checks,
        "contact_sheet": {
            "path": str(contact),
            "sha256": base._sha256(contact),
        },
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "boundary": (
            "Six selected clear two-person profile frames and deterministic source-bound "
            "left/right person regions. Real pinned YuNet and CPU ParseNet run; native SAM, H3, "
            "crossing people, different skin tones, long video and human aesthetic preference "
            "are not tested."
        ),
    }
    report["passed"] = all(checks.values())
    (output / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "strict_counts": strict_counts,
                "profile_counts": profile_counts,
                "fallback_total": fallback_total,
                "profile_skin_area_fractions": profile_areas,
                "elapsed_seconds": report["elapsed_seconds"],
                "report": str(output / "validation_report.json"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
