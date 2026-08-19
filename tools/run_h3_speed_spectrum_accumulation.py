#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
from typing import Any, Mapping
import urllib.request
import uuid

from safetensors import safe_open


MANIFEST_SCHEMA = "minimax_h3_speed_spectrum_accumulation_manifest_v1"
RUN_SCHEMA = "minimax_h3_speed_spectrum_accumulation_run_v1"
STORAGE_SCHEMA = "minimax_h3_speed_spectrum_dataset_file_t8_v1"
PREPARED_SCHEMA = "minimax_h3_speed_prepared_window_v1"
_SAFE_BATCH_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
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


def build_accumulation_prompt(
    *,
    input_file: str,
    batch_id: str,
    task_family: str,
    dataset_name: str,
    video_vae_name: str,
    checkpoint_fingerprint: str,
    vae_fingerprint: str,
    append_existing: bool,
    width: int = 736,
    height: int = 416,
    length: int = 124,
    max_temporal_samples: int = 32,
    dataset_provenance: Mapping[str, Any] | None = None,
    source_entry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if task_family not in {"T2VA", "I2VA", "FL2VA", "L2VA", "Ref2VA", "Hybrid"}:
        raise ValueError(f"Unsupported task_family: {task_family}")
    if not input_file or not batch_id or not dataset_name:
        raise ValueError("input_file, batch_id and dataset_name must be non-empty")
    prompt: dict[str, Any] = {
        "1": {"class_type": "LoadVideo", "inputs": {"file": input_file}},
        "2": {"class_type": "GetVideoComponents", "inputs": {"video": ["1", 0]}},
        "3": {
            "class_type": "MiniMaxH3SPEEDCalibrationWindowT8Advanced",
            "inputs": {
                "frames": ["2", 0],
                "source_fps": ["2", 2],
                "width": int(width),
                "height": int(height),
                "length": int(length),
                "start_seconds": 0.0,
                "resize_mode": "center_cover",
            },
        },
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": video_vae_name}},
        "5": {
            "class_type": "VAEEncode",
            "inputs": {"pixels": ["3", 0], "vae": ["4", 0]},
        },
        "7": {
            "class_type": "MiniMaxH3SPEEDSpectrumDatasetAccumulateT8Advanced",
            "inputs": {
                "video_latent": ["5", 0],
                "batch_id": batch_id,
                "task_family": task_family,
                "checkpoint_fingerprint": checkpoint_fingerprint,
                "vae_fingerprint": vae_fingerprint,
                "max_temporal_samples": int(max_temporal_samples),
                "dataset_provenance_json": (
                    json.dumps(dataset_provenance, ensure_ascii=False, sort_keys=True)
                    if dataset_provenance is not None
                    else ""
                ),
                "source_entry_json": (
                    json.dumps(source_entry, ensure_ascii=False, sort_keys=True)
                    if source_entry is not None
                    else ""
                ),
            },
        },
        "8": {
            "class_type": "MiniMaxH3SPEEDSpectrumDatasetFileT8Advanced",
            "inputs": {
                "mode": "save",
                "dataset_name": dataset_name,
                "overwrite": bool(append_existing),
                "confirm_write": True,
                "spectrum_dataset": ["7", 0],
            },
        },
    }
    if append_existing:
        prompt["6"] = {
            "class_type": "MiniMaxH3SPEEDSpectrumDatasetFileT8Advanced",
            "inputs": {
                "mode": "load",
                "dataset_name": dataset_name,
                "overwrite": False,
                "confirm_write": False,
            },
        }
        prompt["7"]["inputs"]["previous_dataset"] = ["6", 0]
    return prompt


def _json_request(url: str, *, payload: Mapping[str, Any] | None = None) -> Any:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def queue_and_wait(
    *, server: str, prompt: Mapping[str, Any], timeout_seconds: float
) -> dict[str, Any]:
    queued = _json_request(
        f"{server.rstrip('/')}/prompt",
        payload={"prompt": prompt, "client_id": str(uuid.uuid4())},
    )
    prompt_id = str(queued["prompt_id"])
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        history = _json_request(f"{server.rstrip('/')}/history/{prompt_id}")
        if prompt_id in history:
            record = history[prompt_id]
            status = record.get("status", {})
            if status.get("status_str") != "success":
                raise RuntimeError(
                    f"ComfyUI prompt {prompt_id} failed: "
                    + json.dumps(status, ensure_ascii=False)
                )
            return {
                "prompt_id": prompt_id,
                "status": status,
                "dataset_file_output": record.get("outputs", {}).get("8", {}),
            }
        time.sleep(0.5)
    raise TimeoutError(f"ComfyUI prompt {prompt_id} exceeded {timeout_seconds}s")


def validate_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    if value.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"Manifest schema must be {MANIFEST_SCHEMA}")
    entries = value.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Manifest entries must be a non-empty list")
    batch_ids = []
    provenance = value.get("dataset_provenance")
    formal = bool(value.get("provenance", {}).get("formal_dataset_authorized"))
    if formal and not isinstance(provenance, Mapping):
        raise ValueError("A formal manifest requires dataset_provenance")
    for entry in entries:
        if not isinstance(entry, dict) or not entry.get("file") or not entry.get("batch_id"):
            raise ValueError("Every entry must contain file and batch_id")
        batch_ids.append(str(entry["batch_id"]))
        source_entry = entry.get("source_entry")
        if provenance is not None:
            if not isinstance(source_entry, Mapping):
                raise ValueError("Every provenance-bound manifest entry needs source_entry")
            if str(source_entry.get("batch_id", "")) != str(entry["batch_id"]):
                raise ValueError("Manifest source_entry batch_id mismatch")
            for field in ("source_file_sha256", "decoded_window_sha256"):
                value_hash = str(source_entry.get(field, "")).strip().upper()
                if len(value_hash) != 64:
                    raise ValueError(f"Manifest source_entry is missing {field}")
    if len(batch_ids) != len(set(batch_ids)):
        raise ValueError("Manifest batch_id values must be unique")
    required = (
        "dataset_name",
        "task_family",
        "video_vae_name",
        "checkpoint_fingerprint",
        "vae_fingerprint",
    )
    for key in required:
        if not value.get(key):
            raise ValueError(f"Manifest is missing {key}")
    return dict(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def existing_dataset_batch_ids(dataset_path: Path) -> set[str]:
    if not dataset_path.is_file() or dataset_path.is_symlink():
        raise ValueError(f"Existing dataset must be a regular file: {dataset_path}")
    with safe_open(dataset_path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    if metadata.get("storage_schema") != STORAGE_SCHEMA:
        raise ValueError("Existing dataset storage schema mismatch")
    try:
        dataset = json.loads(metadata["dataset_json"])
    except (KeyError, json.JSONDecodeError) as exc:
        raise ValueError("Existing dataset metadata is missing or malformed") from exc
    batch_ids = dataset.get("batch_ids")
    if not isinstance(batch_ids, list) or not all(isinstance(v, str) for v in batch_ids):
        raise ValueError("Existing dataset batch_ids are missing or malformed")
    if len(batch_ids) != len(set(batch_ids)):
        raise ValueError("Existing dataset contains duplicate batch_ids")
    return set(batch_ids)


def build_preprocess_command(
    *,
    ffmpeg: str,
    source: Path,
    temporary: Path,
    width: int,
    height: int,
    length: int,
) -> list[str]:
    if width <= 0 or height <= 0 or width % 32 or height % 32:
        raise ValueError("Prepared calibration width and height must be positive multiples of 32")
    if length < 5 or (length - 5) % 17:
        raise ValueError("Prepared calibration length must follow the H3 17n+5 grid")
    video_filter = (
        f"fps=24,scale={width}:{height}:force_original_aspect_ratio=increase:"
        f"flags=lanczos,crop={width}:{height},trim=start_frame=0:end_frame={length},"
        "setpts=PTS-STARTPTS"
    )
    return [
        ffmpeg,
        "-nostdin",
        "-v",
        "error",
        "-xerror",
        "-err_detect",
        "explode",
        "-i",
        str(source),
        "-an",
        "-vf",
        video_filter,
        "-frames:v",
        str(length),
        "-c:v",
        "ffv1",
        "-level",
        "3",
        "-pix_fmt",
        "yuv420p",
        "-y",
        str(temporary),
    ]


def prepare_calibration_window_file(
    *,
    source: Path,
    source_sha256: str,
    prepared_root: Path,
    batch_id: str,
    width: int,
    height: int,
    length: int,
    ffmpeg: str,
    ffprobe: str,
) -> tuple[Path, dict[str, Any]]:
    if not _SAFE_BATCH_ID.fullmatch(batch_id):
        raise ValueError(f"Unsafe batch_id for prepared window: {batch_id!r}")
    prepared_root.mkdir(parents=True, exist_ok=True)
    destination = prepared_root / f"{batch_id}_{width}x{height}x{length}.mkv"
    sidecar = destination.with_suffix(destination.suffix + ".json")
    expected_recipe = {
        "schema": PREPARED_SCHEMA,
        "batch_id": batch_id,
        "source_file_sha256": source_sha256,
        "width": int(width),
        "height": int(height),
        "length": int(length),
        "fps": 24,
        "resize": "center_cover_lanczos",
        "codec": "ffv1_level3_yuv420p",
    }
    if destination.is_file() and sidecar.is_file():
        cached = json.loads(sidecar.read_text(encoding="utf-8"))
        prepared_hash = _sha256(destination)
        if (
            all(cached.get(k) == v for k, v in expected_recipe.items())
            and cached.get("prepared_file_sha256") == prepared_hash
        ):
            return destination, cached

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.", suffix=".tmp.mkv", dir=prepared_root
    )
    os.close(descriptor)
    os.unlink(temporary_name)
    temporary = Path(temporary_name)
    try:
        command = build_preprocess_command(
            ffmpeg=ffmpeg,
            source=source,
            temporary=temporary,
            width=width,
            height=height,
            length=length,
        )
        completed = subprocess.run(
            command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=False
        )
        if completed.returncode:
            raise RuntimeError(
                "FFmpeg calibration-window preparation failed: "
                + completed.stderr.decode("utf-8", errors="replace")[-2000:]
            )
        probe = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-count_frames",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,avg_frame_rate,nb_read_frames",
                "-of",
                "json",
                str(temporary),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if probe.returncode:
            raise RuntimeError(
                "FFprobe prepared-window validation failed: "
                + probe.stderr.decode("utf-8", errors="replace")[-2000:]
            )
        metadata = json.loads(probe.stdout.decode("utf-8"))
        streams = metadata.get("streams", [])
        if len(streams) != 1:
            raise ValueError("Prepared window must contain exactly one video stream")
        stream = streams[0]
        if (
            int(stream.get("width", 0)) != width
            or int(stream.get("height", 0)) != height
            or str(stream.get("avg_frame_rate")) != "24/1"
            or int(stream.get("nb_read_frames", 0)) != length
        ):
            raise ValueError(f"Prepared window contract mismatch: {stream}")
        os.replace(temporary, destination)
        report = {
            **expected_recipe,
            "prepared_file_sha256": _sha256(destination),
            "source_path": str(source),
        }
        _write_json_atomic(sidecar, report)
        return destination, report
    finally:
        try:
            temporary.unlink()
        except OSError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sequentially accumulate pre-curated H3 clips into one SPEED spectrum dataset."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-record", type=Path, required=True)
    parser.add_argument("--server", default="http://127.0.0.1:8197")
    parser.add_argument("--comfy-output", type=Path, required=True)
    parser.add_argument("--comfy-input", type=Path)
    parser.add_argument("--append-existing", action="store_true")
    parser.add_argument("--skip-existing-batches", action="store_true")
    parser.add_argument(
        "--prepared-window-dir",
        help=(
            "Optional directory under ComfyUI input. FFmpeg first creates an exact bounded "
            "24fps/center-cover window there so LoadVideo never decodes an unbounded source."
        ),
    )
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    args = parser.parse_args()

    manifest = validate_manifest(
        json.loads(args.manifest.read_text(encoding="utf-8"))
    )
    dataset_path = (
        args.comfy_output
        / "h3_speed_spectrum_datasets"
        / f"{manifest['dataset_name']}.safetensors"
    )
    if dataset_path.exists() and not args.append_existing:
        raise FileExistsError(
            f"Dataset already exists; use --append-existing only after verifying it: {dataset_path}"
        )
    if not dataset_path.exists() and args.append_existing:
        raise FileNotFoundError(f"Dataset does not exist for append: {dataset_path}")

    run = {
        "schema": RUN_SCHEMA,
        "manifest": str(args.manifest.resolve()),
        "server": args.server,
        "dataset_path": str(dataset_path.resolve()),
        "initial_mode": "append" if args.append_existing else "create",
        "entries": [],
        "status": "running",
        "skipped_existing_batches": [],
    }
    _write_json_atomic(args.run_record, run)
    append = args.append_existing
    comfy_input = (args.comfy_input or (args.comfy_output.parent / "input")).resolve()
    existing_batch_ids = (
        existing_dataset_batch_ids(dataset_path)
        if args.append_existing and args.skip_existing_batches
        else set()
    )
    prepared_root: Path | None = None
    if args.prepared_window_dir:
        prepared_root = (comfy_input / args.prepared_window_dir).resolve()
        try:
            prepared_root.relative_to(comfy_input)
        except ValueError as exc:
            raise ValueError("prepared-window-dir escaped ComfyUI input") from exc
    for entry in manifest["entries"]:
        batch_id = str(entry["batch_id"])
        if batch_id in existing_batch_ids:
            run["skipped_existing_batches"].append(batch_id)
            _write_json_atomic(args.run_record, run)
            continue
        source_path = (comfy_input / str(entry["file"])).resolve()
        try:
            source_path.relative_to(comfy_input)
        except ValueError as exc:
            raise ValueError(f"Manifest source escaped ComfyUI input: {source_path}") from exc
        if not source_path.is_file() or source_path.is_symlink():
            raise FileNotFoundError(f"Manifest source is missing or unsafe: {source_path}")
        expected_source_hash = str(entry.get("source_file_sha256", "")).upper()
        actual_source_hash = _sha256(source_path)
        if expected_source_hash and actual_source_hash != expected_source_hash:
            raise ValueError(f"Manifest source hash changed before accumulation: {source_path}")
        prompt_input_file = str(entry["file"])
        prepared_report: dict[str, Any] | None = None
        if prepared_root is not None:
            prepared_path, prepared_report = prepare_calibration_window_file(
                source=source_path,
                source_sha256=actual_source_hash,
                prepared_root=prepared_root,
                batch_id=batch_id,
                width=int(manifest.get("width", 736)),
                height=int(manifest.get("height", 416)),
                length=int(manifest.get("length", 124)),
                ffmpeg=args.ffmpeg,
                ffprobe=args.ffprobe,
            )
            prompt_input_file = prepared_path.relative_to(comfy_input).as_posix()
        prompt = build_accumulation_prompt(
            input_file=prompt_input_file,
            batch_id=batch_id,
            task_family=str(manifest["task_family"]),
            dataset_name=str(manifest["dataset_name"]),
            video_vae_name=str(manifest["video_vae_name"]),
            checkpoint_fingerprint=str(manifest["checkpoint_fingerprint"]),
            vae_fingerprint=str(manifest["vae_fingerprint"]),
            append_existing=append,
            width=int(manifest.get("width", 736)),
            height=int(manifest.get("height", 416)),
            length=int(manifest.get("length", 124)),
            max_temporal_samples=int(manifest.get("max_temporal_samples", 32)),
            dataset_provenance=manifest.get("dataset_provenance"),
            source_entry=entry.get("source_entry"),
        )
        try:
            result = queue_and_wait(
                server=args.server,
                prompt=prompt,
                timeout_seconds=args.timeout_seconds,
            )
        except BaseException as exc:
            run["status"] = "failed"
            run["failure"] = {
                "batch_id": str(entry["batch_id"]),
                "type": type(exc).__name__,
                "message": str(exc),
            }
            _write_json_atomic(args.run_record, run)
            raise
        run["entries"].append(
            {
                "file": str(entry["file"]),
                "batch_id": batch_id,
                "prepared_window": prepared_report,
                **result,
            }
        )
        _write_json_atomic(args.run_record, run)
        append = True
    run["status"] = "complete"
    run["completed_entries"] = len(run["entries"])
    run["total_dataset_entries"] = len(run["entries"]) + len(existing_batch_ids)
    _write_json_atomic(args.run_record, run)
    print(json.dumps({"status": run["status"], "entries": len(run["entries"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
