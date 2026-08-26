#!/usr/bin/env python3
"""Build one hash-bound PNG-only input bundle for the VRetouchEr validator.

The default invocation is inspection-only and writes nothing. ``--write-bundle`` copies one to six
explicit source PNGs, the reviewed semantic-skin mask and optional reviewed person mask into a new,
non-overwriting directory. It then records every SHA-256, one reviewed face box per source frame and
the fixed newest/current-frame scope in ``manifest.json``.

This tool imports neither Torch nor ComfyUI, detects no face or identity, loads no model, and performs
no inference. Face boxes and masks must already have been reviewed by the caller.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any


SCHEMA = "t8.minimax_h3.skin_finish_vretoucher_bundle_builder.v1"
MANIFEST_SCHEMA = "t8.minimax_h3.skin_finish_vretoucher_input_manifest.v1"
MAX_SOURCE_FRAMES = 6
MAX_PIXELS_PER_FRAME = 2_100_000


class BundleUnavailable(RuntimeError):
    def __init__(self, status: str, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest().upper()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _inspect_png(path: Path, *, role: str) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise BundleUnavailable("ABSTAIN_INPUT_FILE_MISSING", f"missing {role}: {resolved}")
    try:
        from PIL import Image

        with Image.open(resolved) as image:
            image_format = str(image.format or "").upper()
            frame_count = int(getattr(image, "n_frames", 1))
            width, height = [int(item) for item in image.size]
            mode = str(image.mode)
    except (OSError, ValueError) as error:
        raise BundleUnavailable(
            "ABSTAIN_INPUT_IMAGE_INVALID", f"cannot inspect {role}: {error}"
        ) from error
    if image_format != "PNG" or frame_count != 1:
        raise BundleUnavailable(
            "ABSTAIN_INPUT_FORMAT_UNSAFE", f"{role} must be one static PNG"
        )
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
        "width": width,
        "height": height,
        "mode": mode,
        "format": image_format,
    }


def _parse_face_box(value: str, *, index: int) -> list[float]:
    try:
        items = [float(item.strip()) for item in str(value).split(",")]
    except ValueError as error:
        raise BundleUnavailable(
            "ABSTAIN_FACE_BOX_INVALID",
            f"face box {index} must be left,top,right,bottom",
        ) from error
    if len(items) != 4 or not all(math.isfinite(item) for item in items):
        raise BundleUnavailable(
            "ABSTAIN_FACE_BOX_INVALID",
            f"face box {index} must contain four finite coordinates",
        )
    left, top, right, bottom = items
    if right - left < 4.0 or bottom - top < 4.0:
        raise BundleUnavailable(
            "ABSTAIN_FACE_BOX_INVALID", f"face box {index} is inverted or too small"
        )
    return items


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    base = {
        "schema": SCHEMA,
        "created_at": _utc_now(),
        "mode": "inspection_only",
        "files_written": False,
        "torch_imported": False,
        "comfyui_imported": False,
        "model_loaded": False,
        "inference_executed": False,
        "identity_detected": False,
        "face_boxes_detected": False,
        "reviewed_inputs_required": True,
    }
    errors: list[dict[str, str]] = []
    frames: list[dict[str, Any]] = []
    boxes: list[list[float]] = []
    frame_paths = [Path(item).expanduser().resolve() for item in args.frame]
    if not 1 <= len(frame_paths) <= MAX_SOURCE_FRAMES:
        errors.append(
            {
                "status": "ABSTAIN_FRAME_COUNT_INVALID",
                "detail": f"one bundle requires 1..{MAX_SOURCE_FRAMES} source frames",
            }
        )
    if len(set(frame_paths)) != len(frame_paths):
        errors.append(
            {
                "status": "ABSTAIN_DUPLICATE_SOURCE_FRAME",
                "detail": "repeat a shorter causal window through the validator, not duplicate files",
            }
        )
    if len(args.face_box) != len(frame_paths):
        errors.append(
            {
                "status": "ABSTAIN_FACE_BOX_COUNT_MISMATCH",
                "detail": "provide exactly one reviewed --face-box for every --frame",
            }
        )
    else:
        for index, value in enumerate(args.face_box):
            try:
                boxes.append(_parse_face_box(value, index=index))
            except BundleUnavailable as error:
                errors.append({"status": error.status, "detail": error.detail})
    track_key = str(args.track_key or "").strip()
    if not track_key:
        errors.append(
            {"status": "ABSTAIN_TRACK_KEY_MISSING", "detail": "--track-key is required"}
        )
    if not 1.0 <= float(args.context_factor) <= 3.0:
        errors.append(
            {
                "status": "ABSTAIN_PARAMETER_INVALID",
                "detail": "context-factor must stay within 1.0..3.0",
            }
        )
    if not 0.0 <= float(args.amount) <= 1.0 or not 0 <= int(args.feather_px) <= 64:
        errors.append(
            {
                "status": "ABSTAIN_PARAMETER_INVALID",
                "detail": "amount must be 0..1 and feather-px must be 0..64",
            }
        )
    if not errors:
        try:
            frames = [
                _inspect_png(path, role=f"source frame {index}")
                for index, path in enumerate(frame_paths)
            ]
            geometry = (frames[0]["width"], frames[0]["height"])
            mode = frames[0]["mode"]
            if mode not in {"RGB", "RGBA"}:
                raise BundleUnavailable(
                    "ABSTAIN_SOURCE_MODE_UNSAFE", "source frames must be RGB or RGBA PNG"
                )
            for index, frame in enumerate(frames):
                if (frame["width"], frame["height"]) != geometry or frame["mode"] != mode:
                    raise BundleUnavailable(
                        "ABSTAIN_SOURCE_GEOMETRY_OR_MODE_MISMATCH",
                        "all source frames require identical geometry and channel mode",
                    )
                if frame["width"] * frame["height"] > MAX_PIXELS_PER_FRAME:
                    raise BundleUnavailable(
                        "ABSTAIN_VALIDATOR_INPUT_TOO_LARGE",
                        f"source frame {index} exceeds {MAX_PIXELS_PER_FRAME} pixels",
                    )
                if boxes:
                    left, top, right, bottom = boxes[index]
                    if right <= 0 or bottom <= 0 or left >= geometry[0] or top >= geometry[1]:
                        raise BundleUnavailable(
                            "ABSTAIN_FACE_BOX_OUTSIDE_FRAME",
                            f"face box {index} does not intersect its source frame",
                        )
        except BundleUnavailable as error:
            errors.append({"status": error.status, "detail": error.detail})
    masks: dict[str, dict[str, Any] | None] = {
        "semantic_skin_mask": None,
        "person_mask": None,
    }
    if frames and not errors:
        geometry = (frames[0]["width"], frames[0]["height"])
        try:
            semantic = _inspect_png(args.semantic_mask, role="semantic skin mask")
            if semantic["mode"] != "L" or (
                semantic["width"], semantic["height"]
            ) != geometry:
                raise BundleUnavailable(
                    "ABSTAIN_MASK_CONTRACT_MISMATCH",
                    "semantic skin mask must be L-mode PNG matching source geometry",
                )
            masks["semantic_skin_mask"] = semantic
            if args.person_mask is not None:
                person = _inspect_png(args.person_mask, role="reviewed person mask")
                if person["mode"] != "L" or (
                    person["width"], person["height"]
                ) != geometry:
                    raise BundleUnavailable(
                        "ABSTAIN_MASK_CONTRACT_MISMATCH",
                        "person mask must be L-mode PNG matching source geometry",
                    )
                masks["person_mask"] = person
        except BundleUnavailable as error:
            errors.append({"status": error.status, "detail": error.detail})
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        errors.append(
            {
                "status": "ABSTAIN_OUTPUT_DIRECTORY_EXISTS",
                "detail": f"refusing to overwrite existing bundle: {output_dir}",
            }
        )
    manifest_preview: dict[str, Any] | None = None
    if not errors:
        manifest_preview = {
            "schema": MANIFEST_SCHEMA,
            "frames": [
                {
                    "path": f"frames/{index:03d}.png",
                    "sha256": frame["sha256"],
                }
                for index, frame in enumerate(frames)
            ],
            "current_frame": len(frames) - 1,
            "shot_start": 0,
            "shot_end": len(frames) - 1,
            "track_key": track_key,
            "frame_track_keys": [track_key] * len(frames),
            "face_boxes": boxes,
            "semantic_skin_mask": {
                "path": "masks/semantic_skin.png",
                "sha256": masks["semantic_skin_mask"]["sha256"],
            },
            "person_mask": (
                {
                    "path": "masks/person.png",
                    "sha256": masks["person_mask"]["sha256"],
                }
                if masks["person_mask"] is not None
                else None
            ),
            "output_current_frame_only": True,
            "context_factor": float(args.context_factor),
            "amount": float(args.amount),
            "feather_px": int(args.feather_px),
        }
    status = errors[0]["status"] if errors else "READY_TO_WRITE_HASH_BOUND_BUNDLE"
    return {
        **base,
        "status": status,
        "ready_to_write": not errors,
        "errors": errors,
        "output_directory": str(output_dir),
        "source_frames": frames,
        "masks": masks,
        "face_boxes": boxes,
        "manifest_preview": manifest_preview,
        "boundary": (
            "READY proves only reviewed PNG geometry, paths, boxes and hashes. It does not detect "
            "identity, validate mask meaning, load VRetouchEr or establish any quality/safety claim."
        ),
    }


def _copy_verified(source: Path, destination: Path, expected_sha256: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    if temporary.exists() or destination.exists():
        raise BundleUnavailable(
            "ABSTAIN_BUNDLE_STAGING_CONFLICT", f"staging target already exists: {destination}"
        )
    try:
        with source.open("rb") as reader, temporary.open("xb") as writer:
            while chunk := reader.read(8 * 1024 * 1024):
                writer.write(chunk)
            writer.flush()
            os.fsync(writer.fileno())
        if _sha256_file(temporary) != expected_sha256:
            raise BundleUnavailable(
                "ABSTAIN_SOURCE_CHANGED_DURING_COPY",
                f"source changed while copying: {source}",
            )
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_bundle(args: argparse.Namespace, report: dict[str, Any]) -> dict[str, Any]:
    if not report.get("ready_to_write"):
        raise BundleUnavailable(
            "ABSTAIN_PREFLIGHT_NOT_READY", "bundle preflight must pass before writing"
        )
    output_dir = Path(report["output_directory"])
    if output_dir.exists():
        raise BundleUnavailable(
            "ABSTAIN_OUTPUT_DIRECTORY_EXISTS", f"refusing to overwrite: {output_dir}"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    manifest = report["manifest_preview"]
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-staging-",
        dir=output_dir.parent,
        ignore_cleanup_errors=True,
    ) as temporary_name:
        staging = Path(temporary_name)
        for index, source_contract in enumerate(report["source_frames"]):
            _copy_verified(
                Path(source_contract["path"]),
                staging / "frames" / f"{index:03d}.png",
                source_contract["sha256"],
            )
        semantic = report["masks"]["semantic_skin_mask"]
        _copy_verified(
            Path(semantic["path"]),
            staging / "masks" / "semantic_skin.png",
            semantic["sha256"],
        )
        person = report["masks"]["person_mask"]
        if person is not None:
            _copy_verified(
                Path(person["path"]),
                staging / "masks" / "person.png",
                person["sha256"],
            )
        manifest_bytes = _json_bytes(manifest)
        manifest_path = staging / "manifest.json"
        with manifest_path.open("xb") as handle:
            handle.write(manifest_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        build_report = {
            "schema": SCHEMA,
            "created_at": _utc_now(),
            "status": "HASH_BOUND_SINGLE_WINDOW_BUNDLE_WRITTEN_NOT_EXECUTED",
            "files_written": True,
            "manifest": {
                "path": "manifest.json",
                "sha256": _sha256_bytes(manifest_bytes),
            },
            "source_preflight": report,
            "torch_imported": False,
            "comfyui_imported": False,
            "model_loaded": False,
            "inference_executed": False,
            "automatic_accept": False,
            "candidate_selected": False,
            "boundary": (
                "This is only a reproducible input bundle. The formal validator and all model, "
                "identity, temporal, memory and human-review gates remain separate."
            ),
        }
        build_bytes = _json_bytes(build_report)
        with (staging / "build_report.json").open("xb") as handle:
            handle.write(build_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        if output_dir.exists():
            raise BundleUnavailable(
                "ABSTAIN_OUTPUT_DIRECTORY_APPEARED_DURING_BUILD",
                f"refusing to replace newly created path: {output_dir}",
            )
        staging.replace(output_dir)
    return {
        "schema": SCHEMA,
        "status": "HASH_BOUND_SINGLE_WINDOW_BUNDLE_WRITTEN_NOT_EXECUTED",
        "output_directory": str(output_dir),
        "manifest": str(output_dir / "manifest.json"),
        "manifest_sha256": _sha256_file(output_dir / "manifest.json"),
        "build_report": str(output_dir / "build_report.json"),
        "files_written": True,
        "model_loaded": False,
        "inference_executed": False,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame", action="append", type=Path, default=[])
    parser.add_argument("--semantic-mask", type=Path, required=True)
    parser.add_argument("--person-mask", type=Path)
    parser.add_argument(
        "--face-box",
        action="append",
        default=[],
        metavar="LEFT,TOP,RIGHT,BOTTOM",
    )
    parser.add_argument("--track-key", default="0:0")
    parser.add_argument("--context-factor", type=float, default=1.45)
    parser.add_argument("--amount", type=float, default=1.0)
    parser.add_argument("--feather-px", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--write-bundle",
        action="store_true",
        help="Write a new non-overwriting bundle only after preflight passes.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = preflight(args)
    if not args.write_bundle:
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "ready_to_write": report["ready_to_write"],
                    "output_directory": report["output_directory"],
                    "files_written": False,
                    "model_loaded": False,
                    "inference_executed": False,
                },
                ensure_ascii=False,
            )
        )
        return 0
    if not report["ready_to_write"]:
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "errors": report["errors"],
                    "files_written": False,
                },
                ensure_ascii=False,
            )
        )
        return 3
    result = write_bundle(args, report)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
