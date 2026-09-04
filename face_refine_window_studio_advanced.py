from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import time
import uuid

import numpy as np
import torch

from .face_refine_advanced import canonical_json, source_proxy_sha256
from .face_refine_window_advanced import (
    WINDOW_MAPPING_SCHEMA,
    WINDOW_PLAN_SCHEMA,
    _validate_signed,
    apply_face_refine_manual_review,
)
from .long_video import sanitize_chain_id
from .long_video_background import register_background_progress_provider
from .long_video_delivery import (
    _atomic_write_bytes,
    _manifest_lock,
    _sha256_file,
    long_video_chain_root,
)


STUDIO_SCHEMA = "h3_t8_face_refine_window_studio/v1"
STUDIO_FORMAT = "minimax_h3_t8_face_refine_window_studio"
STUDIO_MANIFEST_NAME = "face_refine_window_studio.json"
STUDIO_CHAIN_PREFIX = "face_refine_window__"
RESOLVED_WINDOW_STATES = {"accepted", "rejected"}


def _digest(value: Mapping) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def studio_chain_id(studio_id: str) -> str:
    safe = sanitize_chain_id(str(studio_id))
    return sanitize_chain_id(f"{STUDIO_CHAIN_PREFIX}{safe}")


def _studio_root_from_chain(chain_id: str) -> Path:
    safe = sanitize_chain_id(chain_id)
    if not safe.startswith(STUDIO_CHAIN_PREFIX):
        raise ValueError("Face Refine Studio chain_id uses an invalid namespace")
    return long_video_chain_root(safe)


def _manifest_path(chain_id: str) -> Path:
    return _studio_root_from_chain(chain_id) / STUDIO_MANIFEST_NAME


def _signed_manifest(payload: Mapping) -> dict:
    result = dict(payload)
    result.pop("manifest_sha256", None)
    result["manifest_sha256"] = _digest(result)
    return result


def _validate_manifest(payload: Mapping, *, chain_id: str | None = None) -> dict:
    if not isinstance(payload, Mapping):
        raise ValueError("Face Refine Studio manifest root must be an object")
    manifest = dict(payload)
    if manifest.get("schema") != STUDIO_SCHEMA or manifest.get("format") != STUDIO_FORMAT:
        raise ValueError("Unsupported Face Refine Studio manifest schema or format")
    expected = str(manifest.get("manifest_sha256", ""))
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256", None)
    if expected != _digest(unsigned):
        raise ValueError("Face Refine Studio manifest hash mismatch")
    safe_chain = sanitize_chain_id(str(manifest.get("chain_id", "")))
    if chain_id is not None and safe_chain != sanitize_chain_id(chain_id):
        raise ValueError("Face Refine Studio manifest belongs to a different chain")
    windows = manifest.get("windows")
    if not isinstance(windows, list) or len(windows) != int(manifest.get("window_count", -1)):
        raise ValueError("Face Refine Studio manifest window table is invalid")
    for index, item in enumerate(windows):
        if not isinstance(item, Mapping) or int(item.get("window_index", -1)) != index:
            raise ValueError("Face Refine Studio window order is invalid")
        if item.get("state") not in {"pending", *RESOLVED_WINDOW_STATES}:
            raise ValueError("Face Refine Studio window state is invalid")
        if item.get("state") == "accepted":
            overlay_path = str(item.get("overlay_path", ""))
            overlay_hash = str(item.get("overlay_sha256", ""))
            if not overlay_path or len(overlay_hash) != 64:
                raise ValueError("Accepted Face Refine Studio window lacks its overlay identity")
    return manifest


def load_face_refine_window_studio(chain_id: str) -> dict | None:
    path = _manifest_path(chain_id)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Face Refine Studio manifest is unreadable: {error}") from error
    return _validate_manifest(payload, chain_id=chain_id)


def _write_manifest(chain_id: str, payload: Mapping) -> dict:
    manifest = _signed_manifest(payload)
    _validate_manifest(manifest, chain_id=chain_id)
    _atomic_write_bytes(
        _manifest_path(chain_id),
        (canonical_json(manifest) + "\n").encode("utf-8"),
    )
    return manifest


def _resolved_prefix(manifest: Mapping) -> int:
    count = 0
    for item in manifest["windows"]:
        if item["state"] not in RESOLVED_WINDOW_STATES:
            break
        count += 1
    if any(
        item["state"] in RESOLVED_WINDOW_STATES
        for item in manifest["windows"][count:]
    ):
        raise ValueError("Face Refine Studio manifest contains a resolved window after a gap")
    return count


def face_refine_window_studio_position(chain_id: str) -> tuple[int, bool]:
    manifest = load_face_refine_window_studio(chain_id)
    if manifest is None:
        return 0, False
    count = _resolved_prefix(manifest)
    return count, count == int(manifest["window_count"])


register_background_progress_provider(
    STUDIO_CHAIN_PREFIX, face_refine_window_studio_position
)


def prepare_face_refine_window_studio(window_plan: Mapping, studio_id: str):
    plan = _validate_signed(
        dict(window_plan), WINDOW_PLAN_SCHEMA, "plan_sha256", "window_plan"
    )
    if int(plan.get("window_count", 0)) < 1:
        raise ValueError("Face Refine Window Studio requires at least one planned window")
    chain_id = studio_chain_id(studio_id)
    root = _studio_root_from_chain(chain_id)
    root.mkdir(parents=True, exist_ok=True)
    path = _manifest_path(chain_id)
    with _manifest_lock(root):
        existing = load_face_refine_window_studio(chain_id)
        if existing is None:
            now = time.time()
            manifest = _write_manifest(
                chain_id,
                {
                    "schema": STUDIO_SCHEMA,
                    "format": STUDIO_FORMAT,
                    "chain_id": chain_id,
                    "studio_id": sanitize_chain_id(str(studio_id)),
                    "source": dict(plan["source"]),
                    "window_plan_sha256": plan["plan_sha256"],
                    "window_count": int(plan["window_count"]),
                    "windows": [
                        {
                            "window_index": index,
                            "state": "pending",
                            "mapping_sha256": "",
                            "overlay_path": "",
                            "overlay_sha256": "",
                            "decision_report": {},
                        }
                        for index in range(int(plan["window_count"]))
                    ],
                    "revision": 0,
                    "created_unix": now,
                    "updated_unix": now,
                    "automatic_accept": False,
                    "source_overwrite_allowed": False,
                },
            )
        else:
            manifest = existing
            if (
                manifest["window_plan_sha256"] != plan["plan_sha256"]
                or manifest["source"] != plan["source"]
                or int(manifest["window_count"]) != int(plan["window_count"])
            ):
                raise ValueError(
                    "This studio_id is already bound to a different source or window plan"
                )
        resolved = _resolved_prefix(manifest)
    complete = resolved == int(manifest["window_count"])
    current_index = max(0, min(resolved, int(manifest["window_count"]) - 1))
    report = {
        "schema": STUDIO_SCHEMA,
        "status": "complete" if complete else "ready",
        "chain_id": chain_id,
        "manifest_path": str(path),
        "manifest_revision": int(manifest["revision"]),
        "window_plan_sha256": plan["plan_sha256"],
        "current_window_index": current_index,
        "resolved_window_count": resolved,
        "window_count": int(manifest["window_count"]),
        "complete": complete,
        "automatic_accept": False,
        "queue_contract": "At most one next prompt may be queued after an explicit decision.",
    }
    return chain_id, current_index, str(path), complete, manifest, canonical_json(report)


def _tensor_numpy_exact(value: torch.Tensor) -> tuple[np.ndarray, str]:
    tensor = value.detach().cpu().contiguous()
    dtype_name = str(tensor.dtype)
    if tensor.dtype == torch.bfloat16:
        return tensor.view(torch.uint16).numpy(), dtype_name
    return tensor.numpy(), dtype_name


def _numpy_tensor_exact(value: np.ndarray, dtype_name: str) -> torch.Tensor:
    tensor = torch.from_numpy(np.array(value, copy=True))
    if dtype_name == "torch.bfloat16":
        return tensor.view(torch.bfloat16)
    expected = getattr(torch, dtype_name.removeprefix("torch."), None)
    if not isinstance(expected, torch.dtype) or tensor.dtype != expected:
        raise ValueError("Stored Face Refine overlay dtype is invalid")
    return tensor


def _atomic_write_overlay(
    path: Path,
    *,
    result_crop: torch.Tensor,
    mask_crop: torch.Tensor,
    bbox: tuple[int, int, int, int, int, int],
    mapping_sha256: str,
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    result_array, dtype_name = _tensor_numpy_exact(result_crop)
    mask_array, mask_dtype_name = _tensor_numpy_exact(mask_crop)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            np.savez_compressed(
                handle,
                result=result_array,
                mask=mask_array,
                bbox=np.asarray(bbox, dtype=np.int64),
                result_dtype=np.asarray(dtype_name),
                mask_dtype=np.asarray(mask_dtype_name),
                mapping_sha256=np.asarray(mapping_sha256),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return _sha256_file(path)


def _overlay_relative_path(index: int, mapping_sha256: str) -> str:
    return f"face_refine_overlays/window_{index:04d}_{mapping_sha256[:16]}.npz"


def _load_overlay(root: Path, item: Mapping):
    relative = str(item["overlay_path"])
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("Face Refine overlay path escapes its studio directory") from error
    if not path.is_file() or _sha256_file(path) != item["overlay_sha256"]:
        raise ValueError("Accepted Face Refine overlay failed its file/hash check")
    with np.load(path, allow_pickle=False) as payload:
        result = _numpy_tensor_exact(payload["result"], str(payload["result_dtype"].item()))
        mask = _numpy_tensor_exact(payload["mask"], str(payload["mask_dtype"].item()))
        bbox = tuple(int(value) for value in payload["bbox"].tolist())
        mapping_sha256 = str(payload["mapping_sha256"].item())
    if len(bbox) != 6 or mapping_sha256 != item["mapping_sha256"]:
        raise ValueError("Accepted Face Refine overlay metadata is stale")
    return result, mask, bbox


def _validate_overlay_bounds(
    crop: torch.Tensor,
    mask: torch.Tensor,
    bbox: tuple[int, ...],
    base_frames: torch.Tensor,
) -> tuple[int, int, int, int, int, int]:
    if len(bbox) != 6:
        raise ValueError("Accepted Face Refine overlay bbox must have six coordinates")
    t0, t1, y0, y1, x0, x1 = (int(value) for value in bbox)
    frame_count, height, width = map(int, base_frames.shape[:3])
    if not (
        0 <= t0 < t1 <= frame_count
        and 0 <= y0 < y1 <= height
        and 0 <= x0 < x1 <= width
    ):
        raise ValueError("Accepted Face Refine overlay bbox is outside the source timeline")
    expected_shape = (t1 - t0, y1 - y0, x1 - x0)
    if tuple(mask.shape) != expected_shape or tuple(crop.shape[:3]) != expected_shape:
        raise ValueError("Accepted Face Refine overlay shape does not match its bbox")
    if crop.ndim != 4 or int(crop.shape[-1]) != int(base_frames.shape[-1]):
        raise ValueError("Accepted Face Refine overlay channel count changed")
    if not torch.isfinite(crop).all() or not torch.isfinite(mask).all():
        raise ValueError("Accepted Face Refine overlay contains NaN or Inf")
    if bool((mask < 0).any()) or bool((mask > 1).any()):
        raise ValueError("Accepted Face Refine overlay mask must stay within 0..1")
    return t0, t1, y0, y1, x0, x1


def commit_face_refine_window_studio(
    base_frames: torch.Tensor,
    candidate_window_frames: torch.Tensor,
    changed_mask: torch.Tensor,
    window_mapping: Mapping,
    window_plan: Mapping,
    studio_id: str,
    decision: str,
    accepted_subranges: str,
    confirm_accept: bool,
    edge_fade_frames: int,
    *,
    fault_inject_after_overlay: bool = False,
):
    chain_id, expected_index, manifest_path, complete, _manifest, _ = (
        prepare_face_refine_window_studio(window_plan, studio_id)
    )
    if complete:
        raise ValueError("Face Refine Window Studio is already complete; no window may be redone")
    mapping = _validate_signed(
        dict(window_mapping), WINDOW_MAPPING_SCHEMA, "mapping_sha256", "window_mapping"
    )
    if int(mapping["window"]["window_index"]) != expected_index:
        raise ValueError(
            f"Studio expects window {expected_index}, got {mapping['window']['window_index']}"
        )
    if mapping["window_plan_sha256"] != dict(window_plan).get("plan_sha256"):
        raise ValueError("Window mapping belongs to a different Studio plan")

    review = apply_face_refine_manual_review(
        base_frames,
        candidate_window_frames,
        changed_mask,
        dict(mapping),
        decision,
        accepted_subranges,
        confirm_accept,
        edge_fade_frames,
    )
    review_frames, result_frames, accepted_mask, rejected_mask = review[:4]
    decision_report = json.loads(review[-1])
    committed = False
    resolved_state = ""
    if decision == "preview_only":
        pass
    elif decision == "reject":
        if decision_report["status"] not in {"reject", "rejected_contract"}:
            raise ValueError("Face Refine rejection did not preserve the source contract")
        resolved_state = "rejected"
    elif decision == "accept_selected":
        if decision_report["status"] != "accept_selected":
            raise ValueError(
                "Face Refine candidate was not accepted: "
                + str(decision_report.get("reason") or decision_report["status"])
            )
        if int(review[4]) < 1:
            raise ValueError("Accepted Face Refine candidate changed no source frame")
        resolved_state = "accepted"
    else:
        raise ValueError(f"Unsupported Studio decision: {decision}")

    if resolved_state:
        root = _studio_root_from_chain(chain_id)
        with _manifest_lock(root):
            manifest = load_face_refine_window_studio(chain_id)
            if manifest is None:
                raise ValueError("Face Refine Studio manifest disappeared before commit")
            current = _resolved_prefix(manifest)
            if current != expected_index:
                raise ValueError("Face Refine Studio advanced in another process; rerun status")
            item = dict(manifest["windows"][current])
            if item["state"] != "pending":
                raise ValueError("Resolved Face Refine window cannot be rolled back or redone")
            overlay_path = ""
            overlay_hash = ""
            bbox_list: list[int] = []
            if resolved_state == "accepted":
                active = accepted_mask > 0
                coordinates = active.nonzero(as_tuple=False)
                if coordinates.numel() == 0:
                    raise ValueError("Accepted Face Refine overlay has an empty mask")
                t0, y0, x0 = coordinates.amin(dim=0).tolist()
                t1, y1, x1 = (coordinates.amax(dim=0) + 1).tolist()
                bbox = (int(t0), int(t1), int(y0), int(y1), int(x0), int(x1))
                relative = _overlay_relative_path(current, mapping["mapping_sha256"])
                overlay = root / relative
                overlay_hash = _atomic_write_overlay(
                    overlay,
                    result_crop=result_frames[t0:t1, y0:y1, x0:x1],
                    mask_crop=accepted_mask[t0:t1, y0:y1, x0:x1],
                    bbox=bbox,
                    mapping_sha256=mapping["mapping_sha256"],
                )
                overlay_path = relative
                bbox_list = list(bbox)
                if fault_inject_after_overlay:
                    raise RuntimeError("fault injection after durable overlay")
            item.update(
                {
                    "state": resolved_state,
                    "mapping_sha256": mapping["mapping_sha256"],
                    "overlay_path": overlay_path,
                    "overlay_sha256": overlay_hash,
                    "overlay_bbox": bbox_list,
                    "decision_report": decision_report,
                    "resolved_unix": time.time(),
                }
            )
            updated = dict(manifest)
            updated_windows = list(manifest["windows"])
            updated_windows[current] = item
            updated.update(
                {
                    "windows": updated_windows,
                    "revision": int(manifest["revision"]) + 1,
                    "updated_unix": time.time(),
                }
            )
            manifest = _write_manifest(chain_id, updated)
            committed = True
    else:
        manifest = load_face_refine_window_studio(chain_id)
        assert manifest is not None

    resolved_count = _resolved_prefix(manifest)
    complete = resolved_count == int(manifest["window_count"])
    report = {
        "schema": STUDIO_SCHEMA,
        "status": "complete" if complete else (resolved_state or "preview_only"),
        "chain_id": chain_id,
        "manifest_path": manifest_path,
        "manifest_revision": int(manifest["revision"]),
        "window_index": expected_index,
        "window_state": resolved_state or "pending",
        "committed": committed,
        "resolved_window_count": resolved_count,
        "window_count": int(manifest["window_count"]),
        "complete": complete,
        "source_overwritten": False,
        "automatic_accept": False,
        "manual_review": decision_report,
    }
    return (
        review_frames,
        result_frames,
        accepted_mask,
        rejected_mask,
        committed,
        manifest_path,
        resolved_count,
        complete,
        canonical_json(report),
    )


def compose_face_refine_window_studio(
    base_frames: torch.Tensor,
    window_plan: Mapping,
    studio_id: str,
):
    plan = _validate_signed(
        dict(window_plan), WINDOW_PLAN_SCHEMA, "plan_sha256", "window_plan"
    )
    chain_id = studio_chain_id(studio_id)
    manifest = load_face_refine_window_studio(chain_id)
    if manifest is None:
        raise ValueError("Face Refine Studio manifest does not exist")
    if (
        manifest["window_plan_sha256"] != plan["plan_sha256"]
        or manifest["source"] != plan["source"]
        or source_proxy_sha256(base_frames) != plan["source"]["proxy_sha256"]
    ):
        raise ValueError("Base frames or plan do not match the source-bound Studio manifest")
    result = base_frames.clone()
    combined_mask = base_frames.new_zeros(base_frames.shape[:3])
    root = _studio_root_from_chain(chain_id)
    accepted = 0
    rejected = 0
    for item in manifest["windows"]:
        if item["state"] == "rejected":
            rejected += 1
            continue
        if item["state"] != "accepted":
            continue
        crop, mask, bbox = _load_overlay(root, item)
        t0, t1, y0, y1, x0, x1 = _validate_overlay_bounds(
            crop, mask, bbox, base_frames
        )
        mask = mask.to(device=result.device, dtype=result.dtype)
        crop = crop.to(device=result.device, dtype=result.dtype)
        target = result[t0:t1, y0:y1, x0:x1]
        active = mask > 0
        expanded = active.unsqueeze(-1).expand_as(target)
        target[expanded] = crop[expanded]
        combined = combined_mask[t0:t1, y0:y1, x0:x1]
        combined.copy_(torch.maximum(combined, mask))
        accepted += 1
    outside = combined_mask <= 0
    if not torch.equal(
        result[outside.unsqueeze(-1).expand_as(result)],
        base_frames[outside.unsqueeze(-1).expand_as(base_frames)],
    ):
        raise AssertionError("Studio composition changed pixels outside accepted overlays")
    resolved = _resolved_prefix(manifest)
    complete = resolved == int(manifest["window_count"])
    report = {
        "schema": STUDIO_SCHEMA,
        "status": "complete" if complete else "partial",
        "chain_id": chain_id,
        "manifest_path": str(_manifest_path(chain_id)),
        "manifest_revision": int(manifest["revision"]),
        "accepted_window_count": accepted,
        "rejected_window_count": rejected,
        "resolved_window_count": resolved,
        "window_count": int(manifest["window_count"]),
        "complete": complete,
        "source_preserved_outside_accepted_mask_bit_exact": True,
        "source_overwritten": False,
        "final_full_source_audio_required": True,
    }
    return result, combined_mask, complete, canonical_json(report)
