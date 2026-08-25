from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
DEFAULT_REVIEW = (
    ROOT
    / "artifacts"
    / "skin-finish-surface-stream-20260825-v2"
    / "blind-review"
)
EXPECTED_REVIEW_ID = "b3aad4e0d57b"
EXPECTED_PUBLIC_SHA256 = (
    "E13D6C760326D32E9DBB7B409CA6BA39CDFE8E3698FB40631840CE5915C22A80"
)

if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import validate_skin_finish_quality_stream_representative as common  # noqa: E402


def _canonical_public_hash(manifest: dict) -> str:
    payload = {key: value for key, value in manifest.items() if key != "sha256"}
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().upper()


def _frame_metrics(left_path: Path, right_path: Path) -> list[dict]:
    import av
    import numpy as np

    left_container = av.open(str(left_path), mode="r")
    right_container = av.open(str(right_path), mode="r")
    left_frames = left_container.decode(video=0)
    right_frames = right_container.decode(video=0)
    luma_weights = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
    rows: list[dict] = []
    previous_roi_delta = None
    try:
        for index, pair in enumerate(zip(left_frames, right_frames, strict=True)):
            left, right = pair
            left_rgb = left.to_ndarray(format="rgb24").astype(np.float32) / 255.0
            right_rgb = right.to_ndarray(format="rgb24").astype(np.float32) / 255.0
            if left_rgb.shape != right_rgb.shape:
                raise RuntimeError("blind A/B decoded frame shapes differ")
            height, width = left_rgb.shape[:2]
            y1, y2 = int(height * 0.04), int(height * 0.96)
            x1, x2 = int(width * 0.24), int(width * 0.76)
            delta = right_rgb - left_rgb
            roi_delta = delta[y1:y2, x1:x2]
            roi_luma_delta = float((roi_delta * luma_weights).sum(axis=-1).mean())
            temporal_jump = (
                0.0
                if previous_roi_delta is None
                else abs(roi_luma_delta - previous_roi_delta)
            )
            previous_roi_delta = roi_luma_delta
            delta_luma = (delta * luma_weights).sum(axis=-1)
            gradient_y, gradient_x = np.gradient(delta_luma)
            rows.append(
                {
                    "frame_index": index,
                    "full_rgb_mae": float(np.abs(delta).mean()),
                    "roi_luma_delta_b_minus_a": roi_luma_delta,
                    "roi_temporal_effect_jump": temporal_jump,
                    "difference_edge_p99": float(
                        np.quantile(np.hypot(gradient_x, gradient_y), 0.99)
                    ),
                }
            )
    finally:
        left_container.close()
        right_container.close()
    return rows


def _read_selected(path: Path, selected: set[int]) -> dict[int, object]:
    import av

    result = {}
    container = av.open(str(path), mode="r")
    try:
        for index, frame in enumerate(container.decode(video=0)):
            if index in selected:
                result[index] = frame.to_image().convert("RGB")
            if len(result) == len(selected):
                break
    finally:
        container.close()
    if set(result) != selected:
        raise RuntimeError("could not decode every selected blind-review frame")
    return result


def _write_contact_sheet(
    path: Path,
    left_path: Path,
    right_path: Path,
    selected_indices: list[int],
) -> None:
    import numpy as np
    from PIL import Image, ImageDraw

    selected = set(selected_indices)
    left = _read_selected(left_path, selected)
    right = _read_selected(right_path, selected)
    tile_width, tile_height = 384, 218
    label_width = 150
    sheet = Image.new(
        "RGB",
        (label_width + tile_width * len(selected_indices), tile_height * 3 + 42),
        (23, 27, 33),
    )
    draw = ImageDraw.Draw(sheet)
    for column, frame_index in enumerate(selected_indices):
        x = label_width + column * tile_width
        draw.text((x + 6, 8), f"frame {frame_index}", fill=(235, 240, 245))
        left_tile = left[frame_index].resize((tile_width, tile_height))
        right_tile = right[frame_index].resize((tile_width, tile_height))
        left_array = np.asarray(left_tile, dtype=np.int16)
        right_array = np.asarray(right_tile, dtype=np.int16)
        difference = np.clip(np.abs(right_array - left_array) * 8, 0, 255).astype(
            np.uint8
        )
        sheet.paste(left_tile, (x, 42))
        sheet.paste(right_tile, (x, 42 + tile_height))
        sheet.paste(Image.fromarray(difference), (x, 42 + tile_height * 2))
    for row, label in enumerate(("A", "B", "|B-A| x8")):
        draw.text((12, 42 + row * tile_height + 96), label, fill=(235, 240, 245))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, format="PNG", optimize=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit public blind A/B temporal differences without reading the private mapping."
        )
    )
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--expected-review-id")
    parser.add_argument("--expected-public-sha256")
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args()
    review = args.review.resolve()
    is_default_review = review == DEFAULT_REVIEW.resolve()
    expected_review_id = (
        str(args.expected_review_id)
        if args.expected_review_id
        else EXPECTED_REVIEW_ID
        if is_default_review
        else ""
    )
    expected_public_sha256 = (
        str(args.expected_public_sha256).upper()
        if args.expected_public_sha256
        else EXPECTED_PUBLIC_SHA256
        if is_default_review
        else ""
    )
    if not expected_review_id or not expected_public_sha256:
        raise ValueError(
            "custom review paths require --expected-review-id and "
            "--expected-public-sha256"
        )
    output = review / "temporal-audit-v1"
    plan = {
        "review": str(review),
        "output": str(output),
        "private_key_accessed": False,
        "model_loaded": False,
        "stress_or_repeat": False,
    }
    if not args.confirm_run:
        print(json.dumps({"status": "PLAN_ONLY", **plan}, ensure_ascii=False, indent=2))
        return 0
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite blind temporal audit: {output}")
    manifest_path = review / "public_manifest.json"
    left_path = review / "media" / "A.mp4"
    right_path = review / "media" / "B.mp4"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("review_id") != expected_review_id:
        raise RuntimeError("unexpected blind review id")
    if _canonical_public_hash(manifest) != expected_public_sha256:
        raise RuntimeError("public blind manifest hash changed")
    for path in (left_path, right_path):
        if not path.is_file():
            raise FileNotFoundError(path)
        common._strict_decode(path)

    started = time.perf_counter()
    rows = _frame_metrics(left_path, right_path)
    if len(rows) != int(manifest["contract"]["frame_count"]):
        raise RuntimeError("blind temporal audit frame count changed")
    ranked = sorted(
        rows[1:], key=lambda item: item["roi_temporal_effect_jump"], reverse=True
    )
    selected_indices = sorted({0, len(rows) - 1, *[row["frame_index"] for row in ranked[:4]]})
    contact_sheet = output / "blind_ab_temporal_risk_contact_sheet.png"
    _write_contact_sheet(contact_sheet, left_path, right_path, selected_indices)
    left_pcm = common._pcm_sha256(left_path)
    right_pcm = common._pcm_sha256(right_path)
    maximum_jump = max(row["roi_temporal_effect_jump"] for row in rows)
    report = {
        "schema": "h3_t8_skin_finish_blind_temporal_audit/v1",
        "status": (
            "PASS_NO_GROSS_TEMPORAL_DELTA_DETECTED"
            if maximum_jump <= 0.01
            else "REVIEW_TEMPORAL_DELTA_EXCEEDS_DIAGNOSTIC_LIMIT"
        ),
        "review_id": expected_review_id,
        "public_manifest_sha256": expected_public_sha256,
        "private_key_accessed": False,
        "automatic_selection": False,
        "contract": manifest["contract"],
        "summary": {
            "frame_count": len(rows),
            "mean_full_rgb_mae": sum(row["full_rgb_mae"] for row in rows)
            / len(rows),
            "maximum_full_rgb_mae": max(row["full_rgb_mae"] for row in rows),
            "maximum_roi_temporal_effect_jump": maximum_jump,
            "maximum_difference_edge_p99": max(
                row["difference_edge_p99"] for row in rows
            ),
            "selected_risk_frame_indices": selected_indices,
            "decoded_pcm_exact": left_pcm == right_pcm,
            "decoded_pcm_sha256": left_pcm if left_pcm == right_pcm else None,
        },
        "per_frame": rows,
        "outputs": {
            "contact_sheet": str(contact_sheet),
            "contact_sheet_sha256": common._sha256(contact_sheet),
        },
        "runtime": {
            "elapsed_seconds": round(time.perf_counter() - started, 6),
            "model_loaded": False,
            "stress_or_repeat": False,
        },
        "claim_boundary": (
            "This mapping-blind decoded-media diagnostic can flag gross A/B temporal-delta spikes "
            "and audio mismatch. It cannot identify the candidate, judge skin naturalness, prove "
            "absence of subtle flicker or replace the human review. Encoding differences are part "
            "of the measured A/B delta."
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "temporal_audit_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "report": str(report_path),
                "contact_sheet": str(contact_sheet),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
