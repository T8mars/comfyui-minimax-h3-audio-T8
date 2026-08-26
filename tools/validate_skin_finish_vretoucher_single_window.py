#!/usr/bin/env python3
"""Preflight, then optionally run, one audited VRetouchEr six-frame window.

The default invocation is preflight-only. A real run requires the exact confirmation token,
a hash-bound PNG-only input manifest, a complete trusted checkpoint SHA-256, the pinned bundled
source, a quiet user port 8188, and a provisional free-VRAM floor. The floor is a conservative
start gate, not a measured peak or a 16GiB-safety claim.

Only the newest/current frame is written. The result remains an unselected candidate requiring
identity, temporal and human review. This tool never queues ComfyUI, touches audio, unloads global
ComfyUI models, performs repeated/stress runs, or registers a node/workflow.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import socket
import subprocess
import sys
import tempfile
import traceback
from types import ModuleType
from typing import Any, Callable, Mapping


SCHEMA = "t8.minimax_h3.skin_finish_vretoucher_single_window.v1"
MANIFEST_SCHEMA = "t8.minimax_h3.skin_finish_vretoucher_input_manifest.v1"
CONFIRMATION_TOKEN = "I_ACCEPT_ONE_VRETOUCHER_WINDOW"
OFFICIAL_CHECKPOINT_BYTES = 630_172_363
PROVISIONAL_MINIMUM_FREE_VRAM_MIB = 12_000
MAX_SOURCE_FRAMES = 6
MAX_PIXELS_PER_FRAME = 2_100_000
_REAL_WINDOW_STARTED = False
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT.parents[1]
    / "models"
    / "facerestore_models"
    / "VRetouchEr"
    / "gen_best.pth"
)


class ValidationUnavailable(RuntimeError):
    def __init__(self, status: str, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().upper()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _write_atomic(path: Path, value: Any) -> None:
    _write_bytes_atomic(path, _json_bytes(value))


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _safe_relative(value: Any, context: str) -> PurePosixPath:
    normalized = str(value or "").replace("\\", "/").strip()
    relative = PurePosixPath(normalized)
    if (
        not normalized
        or relative.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValidationUnavailable(
            "ABSTAIN_MANIFEST_PATH_UNSAFE", f"{context} must be a safe relative path"
        )
    return relative


def _path_below(root: Path, value: Any, context: str) -> Path:
    base = root.resolve()
    relative = _safe_relative(value, context)
    path = base.joinpath(*relative.parts).resolve()
    try:
        path.relative_to(base)
    except ValueError as error:
        raise ValidationUnavailable(
            "ABSTAIN_MANIFEST_PATH_UNSAFE", f"{context} escapes the manifest directory"
        ) from error
    return path


def _trusted_sha256(value: Any, context: str) -> str:
    token = str(value or "").strip().upper()
    if len(token) != 64 or any(char not in "0123456789ABCDEF" for char in token):
        raise ValidationUnavailable(
            "ABSTAIN_TRUSTED_SHA256_REQUIRED", f"{context} requires a complete SHA-256"
        )
    return token


def _inspect_png(path: Path, *, expected_sha256: str, role: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValidationUnavailable("ABSTAIN_INPUT_FILE_MISSING", f"missing {role}: {path}")
    actual_sha256 = _sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValidationUnavailable(
            "ABSTAIN_INPUT_SHA256_MISMATCH", f"SHA-256 mismatch for {role}: {path.name}"
        )
    try:
        from PIL import Image

        with Image.open(path) as image:
            image_format = str(image.format or "").upper()
            frame_count = int(getattr(image, "n_frames", 1))
            width, height = [int(item) for item in image.size]
            mode = str(image.mode)
    except (OSError, ValueError) as error:
        raise ValidationUnavailable(
            "ABSTAIN_INPUT_IMAGE_INVALID", f"cannot inspect {role}: {error}"
        ) from error
    if image_format != "PNG" or frame_count != 1:
        raise ValidationUnavailable(
            "ABSTAIN_INPUT_FORMAT_UNSAFE", f"{role} must be one static PNG"
        )
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": actual_sha256,
        "width": width,
        "height": height,
        "mode": mode,
        "format": image_format,
    }


def load_and_verify_manifest(path: Path) -> dict[str, Any]:
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise ValidationUnavailable(
            "ABSTAIN_MANIFEST_MISSING", f"missing input manifest: {manifest_path}"
        )
    try:
        raw = manifest_path.read_bytes()
        manifest = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValidationUnavailable(
            "ABSTAIN_MANIFEST_INVALID", f"cannot read immutable JSON manifest: {error}"
        ) from error
    if not isinstance(manifest, dict) or manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValidationUnavailable(
            "ABSTAIN_MANIFEST_INVALID", f"manifest schema must be {MANIFEST_SCHEMA}"
        )
    frames = manifest.get("frames")
    if not isinstance(frames, list) or not 1 <= len(frames) <= MAX_SOURCE_FRAMES:
        raise ValidationUnavailable(
            "ABSTAIN_MANIFEST_FRAME_COUNT",
            f"single-window manifest requires 1..{MAX_SOURCE_FRAMES} source frames",
        )
    current_frame = int(manifest.get("current_frame", -1))
    shot_start = int(manifest.get("shot_start", -1))
    shot_end = int(manifest.get("shot_end", -1))
    if shot_start != 0 or shot_end != len(frames) - 1 or current_frame != shot_end:
        raise ValidationUnavailable(
            "ABSTAIN_MANIFEST_WINDOW_NOT_BOUNDED",
            "validation bundle must be one shot starting at zero with the newest frame current",
        )
    if manifest.get("output_current_frame_only") is not True:
        raise ValidationUnavailable(
            "ABSTAIN_MANIFEST_OUTPUT_SCOPE",
            "output_current_frame_only must be explicitly true",
        )
    track_key = str(manifest.get("track_key") or "").strip()
    track_keys = manifest.get("frame_track_keys")
    face_boxes = manifest.get("face_boxes")
    if not track_key or not isinstance(track_keys, list) or len(track_keys) != len(frames):
        raise ValidationUnavailable(
            "ABSTAIN_MANIFEST_TRACK_INVALID", "one reviewed track key is required per frame"
        )
    if not isinstance(face_boxes, list) or len(face_boxes) != len(frames):
        raise ValidationUnavailable(
            "ABSTAIN_MANIFEST_FACE_BOX_INVALID", "one reviewed face box is required per frame"
        )
    if any(value != track_key for value in track_keys):
        raise ValidationUnavailable(
            "ABSTAIN_MANIFEST_TRACK_DISCONTINUITY",
            "all validation frames must belong to the same reviewed shot-local track",
        )
    root = manifest_path.parent
    frame_contracts: list[dict[str, Any]] = []
    geometry: tuple[int, int] | None = None
    source_mode: str | None = None
    for index, entry in enumerate(frames):
        if not isinstance(entry, dict):
            raise ValidationUnavailable(
                "ABSTAIN_MANIFEST_INVALID", f"frames[{index}] must be an object"
            )
        asset = _path_below(root, entry.get("path"), f"frames[{index}].path")
        contract = _inspect_png(
            asset,
            expected_sha256=_trusted_sha256(
                entry.get("sha256"), f"frames[{index}].sha256"
            ),
            role=f"frame {index}",
        )
        if contract["mode"] not in {"RGB", "RGBA"}:
            raise ValidationUnavailable(
                "ABSTAIN_INPUT_MODE_UNSAFE", "source PNG frames must be RGB or RGBA"
            )
        if source_mode is None:
            source_mode = str(contract["mode"])
        elif str(contract["mode"]) != source_mode:
            raise ValidationUnavailable(
                "ABSTAIN_INPUT_MODE_MISMATCH", "all source frames require the same channel mode"
            )
        candidate_geometry = (contract["width"], contract["height"])
        if geometry is None:
            geometry = candidate_geometry
        elif candidate_geometry != geometry:
            raise ValidationUnavailable(
                "ABSTAIN_INPUT_GEOMETRY_MISMATCH", "all source frames require identical geometry"
            )
        if contract["width"] * contract["height"] > MAX_PIXELS_PER_FRAME:
            raise ValidationUnavailable(
                "ABSTAIN_VALIDATOR_INPUT_TOO_LARGE",
                f"single-window validator caps each source frame at {MAX_PIXELS_PER_FRAME} pixels",
            )
        frame_contracts.append(contract)
    assert geometry is not None
    for index, box in enumerate(face_boxes):
        if not isinstance(box, list) or len(box) != 4:
            raise ValidationUnavailable(
                "ABSTAIN_MANIFEST_FACE_BOX_INVALID",
                f"face_boxes[{index}] must be [left,top,right,bottom]",
            )
        try:
            left, top, right, bottom = [float(item) for item in box]
        except (TypeError, ValueError) as error:
            raise ValidationUnavailable(
                "ABSTAIN_MANIFEST_FACE_BOX_INVALID",
                f"face_boxes[{index}] contains a non-number",
            ) from error
        if (
            not all(math.isfinite(item) for item in (left, top, right, bottom))
            or right - left < 4.0
            or bottom - top < 4.0
        ):
            raise ValidationUnavailable(
                "ABSTAIN_MANIFEST_FACE_BOX_INVALID",
                f"face_boxes[{index}] is non-finite, inverted or too small",
            )
    mask_contracts: dict[str, dict[str, Any] | None] = {}
    for key, required in (("semantic_skin_mask", True), ("person_mask", False)):
        entry = manifest.get(key)
        if entry is None and not required:
            mask_contracts[key] = None
            continue
        if not isinstance(entry, dict):
            raise ValidationUnavailable(
                "ABSTAIN_MANIFEST_INVALID", f"{key} must be an immutable PNG asset object"
            )
        asset = _path_below(root, entry.get("path"), f"{key}.path")
        contract = _inspect_png(
            asset,
            expected_sha256=_trusted_sha256(entry.get("sha256"), f"{key}.sha256"),
            role=key,
        )
        if contract["mode"] != "L" or (contract["width"], contract["height"]) != geometry:
            raise ValidationUnavailable(
                "ABSTAIN_MASK_CONTRACT_MISMATCH",
                f"{key} must be an L-mode PNG matching the source geometry",
            )
        mask_contracts[key] = contract
    context_factor = float(manifest.get("context_factor", 1.45))
    amount = float(manifest.get("amount", 1.0))
    feather_px = int(manifest.get("feather_px", 8))
    if not 1.0 <= context_factor <= 3.0:
        raise ValidationUnavailable(
            "ABSTAIN_MANIFEST_PARAMETER_INVALID", "context_factor must stay within 1.0..3.0"
        )
    if not 0.0 <= amount <= 1.0 or not 0 <= feather_px <= 64:
        raise ValidationUnavailable(
            "ABSTAIN_MANIFEST_PARAMETER_INVALID",
            "amount must be 0..1 and feather_px must be 0..64",
        )
    normalized = {
        "schema": MANIFEST_SCHEMA,
        "manifest_path": str(manifest_path),
        "manifest_file_sha256": hashlib.sha256(raw).hexdigest().upper(),
        "frames": frame_contracts,
        "current_frame": current_frame,
        "shot_start": shot_start,
        "shot_end": shot_end,
        "track_key": track_key,
        "frame_track_keys": list(track_keys),
        "face_boxes": face_boxes,
        "semantic_skin_mask": mask_contracts["semantic_skin_mask"],
        "person_mask": mask_contracts["person_mask"],
        "output_current_frame_only": True,
        "context_factor": context_factor,
        "amount": amount,
        "feather_px": feather_px,
        "geometry": {"width": geometry[0], "height": geometry[1]},
    }
    normalized["normalized_manifest_sha256"] = _canonical_sha256(normalized)
    return normalized


def _port_is_listening(port: int = 8188) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", int(port)), timeout=0.25):
            return True
    except OSError:
        return False


def _gpu_memory_mib() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--id=0",
        "--query-gpu=memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        row = completed.stdout.splitlines()[0]
        values = [int(float(item.strip())) for item in row.split(",")]
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return {"available": False}
    return {
        "available": True,
        "gpu_index": 0,
        "total_mib": values[0],
        "used_mib": values[1],
        "free_mib": values[2],
        "utilization_percent": values[3],
        "temperature_c": values[4],
    }


def _ensure_research_package() -> None:
    package_name = "h3_audio_t8_pkg"
    if package_name in sys.modules:
        return
    package = ModuleType(package_name)
    package.__path__ = [str(PROJECT_ROOT)]
    package.__package__ = package_name
    sys.modules[package_name] = package


def _research_modules():
    _ensure_research_package()
    adapter = importlib.import_module("h3_audio_t8_pkg.skin_finish_vretoucher_adapter")
    pipeline = importlib.import_module("h3_audio_t8_pkg.skin_finish_vretoucher_pipeline")
    runtime = importlib.import_module("h3_audio_t8_pkg.skin_finish_vretoucher_runtime")
    return adapter, pipeline, runtime


def _verify_bundled_source() -> dict[str, Any]:
    _, _, runtime = _research_modules()
    return runtime.verify_vretoucher_source()


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    base = {
        "schema": SCHEMA,
        "created_at": _utc_now(),
        "mode": "preflight_only",
        "real_model_loaded": False,
        "checkpoint_deserialized": False,
        "inference_executed": False,
        "automatic_accept": False,
        "candidate_selected": False,
        "audio_touched": False,
        "registered_node": False,
        "workflow_added": False,
    }
    errors: list[dict[str, str]] = []
    manifest: dict[str, Any] | None = None
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    expected_hash = ""
    try:
        manifest = load_and_verify_manifest(args.manifest)
    except (OSError, ValueError, ValidationUnavailable) as error:
        errors.append(
            {
                "status": getattr(error, "status", "ABSTAIN_MANIFEST_INVALID"),
                "detail": str(error),
            }
        )
    try:
        expected_hash = _trusted_sha256(
            args.checkpoint_sha256, "--checkpoint-sha256"
        )
    except ValidationUnavailable as error:
        errors.append({"status": error.status, "detail": error.detail})
    checkpoint_contract: dict[str, Any] = {
        "path": str(checkpoint),
        "official_reference_bytes": OFFICIAL_CHECKPOINT_BYTES,
        "exists": checkpoint.is_file(),
        "stream_hash_checked": False,
        "structure_check": "deferred_to_confirmed_safe_weights_only_runtime_gate",
    }
    if checkpoint.is_file():
        checkpoint_contract["bytes"] = checkpoint.stat().st_size
        if checkpoint.stat().st_size != OFFICIAL_CHECKPOINT_BYTES:
            errors.append(
                {
                    "status": "ABSTAIN_CHECKPOINT_SIZE_MISMATCH",
                    "detail": f"expected {OFFICIAL_CHECKPOINT_BYTES} bytes",
                }
            )
        elif expected_hash:
            actual_hash = _sha256_file(checkpoint)
            checkpoint_contract.update(
                {"sha256": actual_hash, "stream_hash_checked": True}
            )
            if actual_hash != expected_hash:
                errors.append(
                    {
                        "status": "ABSTAIN_CHECKPOINT_SHA256_MISMATCH",
                        "detail": "checkpoint SHA-256 differs from the trusted value",
                    }
                )
    else:
        errors.append(
            {
                "status": "ABSTAIN_CHECKPOINT_MISSING",
                "detail": f"missing official checkpoint: {checkpoint}",
            }
        )
    port_8188_listening = _port_is_listening(8188)
    if port_8188_listening:
        errors.append(
            {
                "status": "ABSTAIN_USER_COMFYUI_8188_ACTIVE",
                "detail": "refusing a research model load while the user's port 8188 is listening",
            }
        )
    gpu = _gpu_memory_mib()
    if not gpu.get("available"):
        errors.append(
            {
                "status": "ABSTAIN_GPU_STATE_UNKNOWN",
                "detail": "nvidia-smi GPU state is unavailable",
            }
        )
    elif int(gpu.get("free_mib", 0)) < int(args.minimum_free_vram_mib):
        errors.append(
            {
                "status": "ABSTAIN_PROVISIONAL_FREE_VRAM_FLOOR",
                "detail": (
                    f"free VRAM {gpu.get('free_mib')}MiB is below provisional "
                    f"{args.minimum_free_vram_mib}MiB start floor"
                ),
            }
        )
    source: dict[str, Any] = {"status": "NOT_CHECKED_DUE_TO_EARLIER_PREFLIGHT_FAILURE"}
    if not errors:
        try:
            source = _verify_bundled_source()
        except Exception as error:
            errors.append(
                {
                    "status": getattr(error, "status", "ABSTAIN_PINNED_SOURCE_INVALID"),
                    "detail": str(error),
                }
            )
    output_root = Path(args.output_root).expanduser().resolve()
    manifest_hash = str((manifest or {}).get("normalized_manifest_sha256") or "")
    result_dir = output_root / f"run-{manifest_hash[:12].lower()}" if manifest_hash else None
    if result_dir is not None and result_dir.exists():
        errors.append(
            {
                "status": "ABSTAIN_SINGLE_WINDOW_ALREADY_EXECUTED",
                "detail": f"result directory already exists: {result_dir}",
            }
        )
    status = errors[0]["status"] if errors else "READY_ONE_WINDOW_ONLY"
    return {
        **base,
        "status": status,
        "ready_for_real_run": not errors,
        "errors": errors,
        "manifest": manifest,
        "checkpoint": checkpoint_contract,
        "source": source,
        "port_8188": {"listening": port_8188_listening, "touched": False},
        "gpu": gpu,
        "device": "cuda:0",
        "precision": args.precision,
        "minimum_free_vram_mib": int(args.minimum_free_vram_mib),
        "free_vram_gate_status": "PROVISIONAL_START_FLOOR_NOT_VALIDATED_PEAK",
        "result_directory": str(result_dir) if result_dir is not None else None,
        "boundary": (
            "READY means only that immutable inputs, pinned source, checkpoint identity, quiet port "
            "and a provisional start floor passed. It does not establish numerical correctness, "
            "quality, identity, temporal stability, unload completeness or 16GiB safety."
        ),
    }


def _load_png_inputs(manifest: Mapping[str, Any]):
    import numpy as np
    from PIL import Image
    import torch

    frame_tensors = []
    for contract in manifest["frames"]:
        with Image.open(contract["path"]) as image:
            mode = str(contract["mode"])
            array = np.array(image.convert(mode), copy=True)
        frame_tensors.append(torch.from_numpy(array).float().div_(255.0))
    frames = torch.stack(frame_tensors)

    def load_mask(contract: Mapping[str, Any] | None):
        if contract is None:
            return None
        with Image.open(contract["path"]) as image:
            array = np.array(image.convert("L"), copy=True)
        return torch.from_numpy(array).float().div_(255.0)

    return (
        frames,
        load_mask(manifest["semantic_skin_mask"]),
        load_mask(manifest["person_mask"]),
    )


def _verify_normalized_assets_unchanged(manifest: Mapping[str, Any]) -> None:
    contracts = list(manifest["frames"]) + [manifest["semantic_skin_mask"]]
    if manifest.get("person_mask") is not None:
        contracts.append(manifest["person_mask"])
    for contract in contracts:
        path = Path(contract["path"])
        if not path.is_file() or _sha256_file(path) != str(contract["sha256"]):
            raise ValidationUnavailable(
                "ABSTAIN_INPUT_CHANGED_AFTER_PREFLIGHT",
                f"input changed after preflight: {path}",
            )


def execute_one_window(
    manifest: Mapping[str, Any],
    session: Any,
    *,
    processor_factory: Callable[[Any], Any] | None = None,
) -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
    """Execute one already-authorized window; injectable session keeps tests weight-free."""
    try:
        prepared = prepare_one_window(manifest)
    except Exception:
        if hasattr(session, "close"):
            session.close()
        raise
    return execute_prepared_window(
        prepared,
        session,
        processor_factory=processor_factory,
    )


def prepare_one_window(
    manifest: Mapping[str, Any],
    *,
    adapter: Any | None = None,
) -> tuple[Any, Any, Any, dict[str, Any], Mapping[str, Any]]:
    """Decode and validate all bounded inputs before any real model is loaded."""
    if adapter is None:
        adapter, _, _ = _research_modules()
    _verify_normalized_assets_unchanged(manifest)
    frames, semantic_mask, person_mask = _load_png_inputs(manifest)
    plan = adapter.build_vretoucher_context_plan(
        frames,
        current_frame=int(manifest["current_frame"]),
        shot_start=int(manifest["shot_start"]),
        shot_end=int(manifest["shot_end"]),
        track_key=str(manifest["track_key"]),
        frame_track_keys=list(manifest["frame_track_keys"]),
        face_boxes=list(manifest["face_boxes"]),
        context_factor=float(manifest["context_factor"]),
    )
    context = adapter.extract_vretoucher_context(frames, plan)
    if tuple(context.shape) != (6, 3, 512, 512):
        raise ValidationUnavailable(
            "ABSTAIN_PREMODEL_CONTEXT_SHAPE", "premodel context is not six 512-square crops"
        )
    del context
    return frames, semantic_mask, person_mask, plan, manifest


def execute_prepared_window(
    prepared: tuple[Any, Any, Any, dict[str, Any], Mapping[str, Any]],
    session: Any,
    *,
    processor_factory: Callable[[Any], Any] | None = None,
) -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
    """Execute an already decoded and premodel-validated window exactly once."""
    frames, semantic_mask, person_mask, plan, manifest = prepared
    _, pipeline, _ = _research_modules()
    factory = processor_factory or pipeline.VRetoucherWindowProcessor
    processor = None
    try:
        processor = factory(session)
        with processor:
            candidate, effective_mask, report_json = processor.process(
                frames,
                current_frame=int(plan["current_frame"]),
                shot_start=int(plan["shot"]["start_frame"]),
                shot_end=int(plan["shot"]["end_frame"]),
                track_key=str(plan["track_key"]),
                frame_track_keys=list(manifest["frame_track_keys"]),
                face_boxes=list(manifest["face_boxes"]),
                semantic_skin_mask=semantic_mask,
                person_mask=person_mask,
                context_factor=float(manifest["context_factor"]),
                amount=float(manifest["amount"]),
                feather_px=int(manifest["feather_px"]),
            )
    finally:
        if processor is not None and not processor.closed:
            processor.close()
        elif processor is None and hasattr(session, "close"):
            session.close()
    report = json.loads(report_json)
    if report.get("automatic_accept") is not False or report.get("candidate_selected") is not False:
        raise ValidationUnavailable(
            "ABSTAIN_PIPELINE_SELECTION_CONTRACT",
            "single-window pipeline attempted to accept or select its candidate",
        )
    current_source = frames[int(plan["current_frame"])].clone()
    return current_source, candidate, effective_mask, {
        "plan": plan,
        "pipeline": report,
        "release": processor.close_report,
    }


def _png_bytes(tensor: Any, *, mask: bool = False) -> bytes:
    from io import BytesIO

    import numpy as np
    from PIL import Image

    array = tensor.detach().cpu().clamp(0.0, 1.0).mul(255.0).round().byte()
    array = array.numpy().astype(np.uint8, copy=False)
    if mask:
        image = Image.fromarray(array, mode="L")
    else:
        if array.shape[-1] not in {3, 4}:
            array = array[..., :3]
        image = Image.fromarray(array, mode="RGBA" if array.shape[-1] == 4 else "RGB")
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=False)
    return buffer.getvalue()


def run_confirmed(args: argparse.Namespace, preflight_report: Mapping[str, Any]) -> dict[str, Any]:
    global _REAL_WINDOW_STARTED
    if args.confirm_run != CONFIRMATION_TOKEN:
        return {
            "schema": SCHEMA,
            "created_at": _utc_now(),
            "status": "ABSTAIN_EXACT_CONFIRMATION_TOKEN_REQUIRED",
            "real_model_loaded": False,
            "inference_executed": False,
            "automatic_accept": False,
            "candidate_selected": False,
        }
    if _REAL_WINDOW_STARTED:
        return {
            "schema": SCHEMA,
            "created_at": _utc_now(),
            "status": "ABSTAIN_ONE_REAL_WINDOW_PER_PROCESS",
            "real_model_loaded": False,
            "inference_executed": False,
            "automatic_accept": False,
            "candidate_selected": False,
        }
    recheck = preflight(args)
    if not recheck["ready_for_real_run"]:
        return {
            "schema": SCHEMA,
            "created_at": _utc_now(),
            "status": "ABSTAIN_PREFLIGHT_CHANGED_BEFORE_MODEL_LOAD",
            "preflight": recheck,
            "real_model_loaded": False,
            "inference_executed": False,
            "automatic_accept": False,
            "candidate_selected": False,
        }
    if recheck.get("manifest", {}).get("normalized_manifest_sha256") != preflight_report.get(
        "manifest", {}
    ).get("normalized_manifest_sha256"):
        return {
            "schema": SCHEMA,
            "created_at": _utc_now(),
            "status": "ABSTAIN_MANIFEST_CHANGED_AFTER_PREFLIGHT",
            "real_model_loaded": False,
            "inference_executed": False,
            "automatic_accept": False,
            "candidate_selected": False,
        }
    _, _, runtime = _research_modules()
    result_dir = Path(recheck["result_directory"])
    result_dir.mkdir(parents=True, exist_ok=False)
    report_path = result_dir / "validation_report.json"
    try:
        prepared = prepare_one_window(recheck["manifest"])
        resource_recheck = _gpu_memory_mib()
        if _port_is_listening(8188):
            raise ValidationUnavailable(
                "ABSTAIN_USER_COMFYUI_8188_BECAME_ACTIVE",
                "port 8188 became active before model load",
            )
        if (
            not resource_recheck.get("available")
            or int(resource_recheck.get("free_mib", 0)) < int(args.minimum_free_vram_mib)
        ):
            raise ValidationUnavailable(
                "ABSTAIN_RESOURCE_CHANGED_BEFORE_MODEL_LOAD",
                "GPU state changed below the provisional floor before model load",
            )
        _REAL_WINDOW_STARTED = True
        session = runtime.load_vretoucher_session(
            None,
            Path(args.checkpoint),
            expected_checkpoint_sha256=str(args.checkpoint_sha256),
            device="cuda:0",
            precision=str(args.precision),
        )
        source, candidate, effective_mask, execution = execute_prepared_window(
            prepared, session
        )
        source_path = result_dir / "source_current.png"
        candidate_path = result_dir / "candidate_current.png"
        mask_path = result_dir / "effective_mask.png"
        _write_bytes_atomic(source_path, _png_bytes(source))
        _write_bytes_atomic(candidate_path, _png_bytes(candidate))
        _write_bytes_atomic(mask_path, _png_bytes(effective_mask, mask=True))
        report = {
            "schema": SCHEMA,
            "created_at": _utc_now(),
            "status": "CANDIDATE_WRITTEN_REQUIRES_IDENTITY_TEMPORAL_AND_HUMAN_REVIEW",
            "passed_mechanical_single_window": True,
            "real_model_loaded": True,
            "checkpoint_deserialized": True,
            "inference_executed": True,
            "current_frame_only": True,
            "automatic_accept": False,
            "candidate_selected": False,
            "quality_validated": False,
            "identity_validated": False,
            "temporal_stability_validated": False,
            "sixteen_gib_safety_validated": False,
            "audio_touched": False,
            "global_comfy_models_unloaded": False,
            "resource_recheck_before_model_load": resource_recheck,
            "preflight": recheck,
            "execution": execution,
            "outputs": {
                "source_current": {
                    "path": str(source_path),
                    "sha256": _sha256_file(source_path),
                },
                "candidate_current": {
                    "path": str(candidate_path),
                    "sha256": _sha256_file(candidate_path),
                },
                "effective_mask": {
                    "path": str(mask_path),
                    "sha256": _sha256_file(mask_path),
                },
            },
            "boundary": (
                "This is one current-frame candidate, not a node release, quality pass, identity "
                "pass, temporal pass, multi-person pass, unload-completeness proof or 16GiB claim."
            ),
        }
    except Exception as error:
        error_status = getattr(error, "status", "FAIL_SINGLE_WINDOW_EXECUTION")
        model_forward_completed = bool(
            getattr(error, "model_forward_completed", False)
            or error_status == "ABSTAIN_PROPOSAL_VALUES_INVALID"
        )
        error_type = type(error).__name__
        error_message = str(error)
        error_traceback = traceback.format_exc()
        if error.__traceback__ is not None:
            error.__traceback__ = None
        del error
        gc.collect()
        release = (
            session.close()
            if "session" in locals() and hasattr(session, "close")
            else None
        )
        report = {
            "schema": SCHEMA,
            "created_at": _utc_now(),
            "status": error_status,
            "passed_mechanical_single_window": False,
            "real_model_loaded": "session" in locals(),
            "inference_executed": model_forward_completed,
            "model_forward_completed": model_forward_completed,
            "automatic_accept": False,
            "candidate_selected": False,
            "error": {
                "type": error_type,
                "message": error_message,
                "traceback": error_traceback,
            },
            "global_comfy_models_unloaded": False,
            "release": release,
        }
    _write_atomic(report_path, report)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--checkpoint-sha256", default="")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "skin-finish-vretoucher-single-window",
    )
    parser.add_argument("--precision", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument(
        "--minimum-free-vram-mib",
        type=int,
        default=PROVISIONAL_MINIMUM_FREE_VRAM_MIB,
        help="Provisional start floor only; values below 12000 are refused.",
    )
    parser.add_argument(
        "--confirm-run",
        default="",
        metavar="TOKEN",
        help=f"Exact token required for one real window: {CONFIRMATION_TOKEN}",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if int(args.minimum_free_vram_mib) < PROVISIONAL_MINIMUM_FREE_VRAM_MIB:
        print(
            json.dumps(
                {
                    "status": "ABSTAIN_PROVISIONAL_VRAM_FLOOR_CANNOT_BE_LOWERED",
                    "minimum_allowed_mib": PROVISIONAL_MINIMUM_FREE_VRAM_MIB,
                },
                ensure_ascii=False,
            )
        )
        return 3
    report = preflight(args)
    preflight_path = Path(args.output_root).expanduser().resolve() / "latest_preflight.json"
    _write_atomic(preflight_path, report)
    if not args.confirm_run:
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "ready_for_real_run": report["ready_for_real_run"],
                    "preflight": str(preflight_path),
                    "real_model_loaded": False,
                    "inference_executed": False,
                },
                ensure_ascii=False,
            )
        )
        return 0
    if args.confirm_run != CONFIRMATION_TOKEN:
        print(
            json.dumps(
                {
                    "status": "ABSTAIN_EXACT_CONFIRMATION_TOKEN_REQUIRED",
                    "preflight": str(preflight_path),
                    "real_model_loaded": False,
                },
                ensure_ascii=False,
            )
        )
        return 3
    if not report["ready_for_real_run"]:
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "preflight": str(preflight_path),
                    "real_model_loaded": False,
                },
                ensure_ascii=False,
            )
        )
        return 3
    result = run_confirmed(args, report)
    print(
        json.dumps(
            {
                "status": result["status"],
                "passed_mechanical_single_window": result.get(
                    "passed_mechanical_single_window", False
                ),
                "automatic_accept": False,
                "candidate_selected": False,
            },
            ensure_ascii=False,
        )
    )
    return 0 if result.get("passed_mechanical_single_window") else 2


if __name__ == "__main__":
    raise SystemExit(main())
