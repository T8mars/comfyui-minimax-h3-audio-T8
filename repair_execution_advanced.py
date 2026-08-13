from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from .long_video_delivery import (
    _atomic_write_bytes,
    _chain_root,
    _copy_atomic,
    _load_candidate,
    _manifest_lock,
    _relative_path,
    _resolve_inside,
    _safe_token,
    _sha256_file,
    load_delivery_manifest,
)
from .studio_advanced import REPAIR_PLAN_SCHEMA, select_repair_segment


REPAIR_EXECUTION_SCHEMA = "t8.minimax_h3.selective_repair_execution.v1"
REPAIR_MANIFEST_FORMAT = "minimax_h3_t8_selective_repair_overlay"
REPAIR_MANIFEST_SCHEMA = 1
REPAIR_MANIFEST_NAME = "manifest.json"
REPAIR_MANIFEST_BACKUP_NAME = "manifest.json.bak"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)


def _digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(plan, Mapping) or plan.get("schema") != REPAIR_PLAN_SCHEMA:
        raise ValueError("repair_plan must be a MiniMax H3 Selective Repair object")
    plan_hash = str(plan.get("repair_plan_hash", ""))
    if len(plan_hash) != 64 or any(
        char not in "0123456789abcdef" for char in plan_hash
    ):
        raise ValueError("repair_plan_hash is invalid")
    return dict(plan)


def _base_snapshot(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "revision": int(manifest["revision"]),
        "sha256": _digest(manifest),
        "segment_count": len(manifest["segments"]),
    }


def _segment_snapshot(segment: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "index",
        "candidate_id",
        "parent_candidate_id",
        "video_path",
        "video_sha256",
        "context_path",
        "context_sha256",
        "frame_count",
        "fps",
        "width",
        "height",
        "sample_rate",
        "audio_samples",
        "audio_start_sample",
        "audio_end_sample",
        "timeline_start_frame",
        "timeline_end_frame",
        "is_final_segment",
        "model_id",
        "sampling_summary",
        "prompt",
        "seed",
    )
    return {key: deepcopy(segment.get(key)) for key in fields}


def bind_repair_execution(
    repair_plan: Mapping[str, Any],
    chain_id: str,
    repair_index: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    plan = _validate_plan(repair_plan)
    repair, selection = select_repair_segment(plan, int(repair_index))
    manifest, manifest_source = load_delivery_manifest(chain_id)
    index = int(repair["shot_index"])
    if index >= len(manifest["segments"]):
        raise ValueError(
            f"repair shot_index {index} is not present in accepted chain "
            f"{chain_id!r} ({len(manifest['segments'])} segment(s))"
        )
    source = manifest["segments"][index]
    if int(repair["frame_count"]) != int(source["frame_count"]):
        raise ValueError(
            "Repair Timeline frame_count does not match the accepted segment. "
            "Rebuild the Studio Timeline from this accepted chain before repair."
        )
    parent = manifest["segments"][index - 1] if index > 0 else None
    following = (
        manifest["segments"][index + 1]
        if index + 1 < len(manifest["segments"])
        else None
    )
    root = _chain_root(chain_id)
    source_video = _resolve_inside(root, source["video_path"])
    if (
        not source_video.is_file()
        or _sha256_file(source_video) != source["video_sha256"]
    ):
        raise ValueError("Accepted source segment failed its file/hash check")
    execution = {
        "schema": REPAIR_EXECUTION_SCHEMA,
        "status": "bound",
        "chain_id": manifest["chain_id"],
        "repair_plan_hash": plan["repair_plan_hash"],
        "repair_index": int(repair_index),
        "repair_plan_selection": selection,
        "repair": repair,
        "base_manifest": _base_snapshot(manifest),
        "base_manifest_source": manifest_source,
        "source_segment": _segment_snapshot(source),
        "parent_candidate_id": parent["candidate_id"] if parent else "",
        "following_candidate_id": following["candidate_id"] if following else "",
        "timeline_start_seconds": int(source["timeline_start_frame"])
        / int(source["fps"]),
        "non_destructive_overlay": True,
        "base_manifest_mutated": False,
        "automatic_accept": False,
        "context_note": (
            "The candidate may use the accepted previous segment as context. "
            "Adjacent repair takes are not implicitly chained."
        ),
    }
    execution["execution_hash"] = _digest(execution)
    report = {
        "schema": REPAIR_EXECUTION_SCHEMA,
        "status": "bound",
        "chain_id": manifest["chain_id"],
        "repair_index": int(repair_index),
        "shot_index": index,
        "source_candidate_id": source["candidate_id"],
        "base_manifest_revision": manifest["revision"],
        "base_manifest_sha256": execution["base_manifest"]["sha256"],
        "source_video_sha256": source["video_sha256"],
        "frame_count": source["frame_count"],
        "audio_samples": int(source["audio_end_sample"])
        - int(source["audio_start_sample"]),
        "parent_candidate_id": execution["parent_candidate_id"],
        "following_candidate_id": execution["following_candidate_id"],
        "base_manifest_mutated": False,
        "next_action": "Generate a same-length candidate, then stage it for explicit review.",
    }
    return execution, report


def _validate_execution(value: Mapping[str, Any], statuses: set[str]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or value.get("schema") != REPAIR_EXECUTION_SCHEMA:
        raise ValueError(
            "repair_execution must be a MiniMax H3 repair execution object"
        )
    if value.get("status") not in statuses:
        raise ValueError(
            f"repair_execution status must be one of {sorted(statuses)}, got {value.get('status')!r}"
        )
    expected = str(value.get("execution_hash", ""))
    payload = dict(value)
    payload.pop("execution_hash", None)
    if expected != _digest(payload):
        raise ValueError("repair_execution hash check failed")
    return dict(value)


def _assert_base_unchanged(execution: Mapping[str, Any]) -> tuple[dict[str, Any], Path]:
    manifest, _source = load_delivery_manifest(execution["chain_id"])
    snapshot = _base_snapshot(manifest)
    expected = execution["base_manifest"]
    if (
        snapshot["revision"] != int(expected["revision"])
        or snapshot["sha256"] != expected["sha256"]
        or snapshot["segment_count"] != int(expected["segment_count"])
    ):
        raise ValueError(
            "Accepted base manifest changed after repair binding; bind again before staging or accepting."
        )
    return manifest, _chain_root(execution["chain_id"])


def _candidate_contract_mismatches(
    source: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> list[str]:
    fields = (
        "index",
        "frame_count",
        "fps",
        "width",
        "height",
        "sample_rate",
        "audio_samples",
        "audio_start_sample",
        "audio_end_sample",
        "timeline_start_frame",
        "timeline_end_frame",
        "is_final_segment",
        "model_id",
        "sampling_summary",
    )
    return [field for field in fields if candidate.get(field) != source.get(field)]


def stage_repair_candidate(
    repair_execution: Mapping[str, Any],
    candidate_json_path: str,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    execution = _validate_execution(repair_execution, {"bound"})
    manifest, root = _assert_base_unchanged(execution)
    candidate, candidate_root, candidate_video = _load_candidate(candidate_json_path)
    if candidate_root.resolve() != root.resolve():
        raise ValueError(
            "Repair candidate belongs to a different accepted chain directory"
        )
    if candidate["chain_id"] != execution["chain_id"]:
        raise ValueError("Repair candidate chain_id does not match the bound repair")
    source = manifest["segments"][int(execution["source_segment"]["index"])]
    mismatches = _candidate_contract_mismatches(source, candidate)
    if mismatches:
        raise ValueError(
            "Repair candidate must preserve exact accepted timeline/media identity fields: "
            + ", ".join(mismatches)
        )
    if candidate["candidate_id"] == source["candidate_id"]:
        raise ValueError(
            "Repair candidate_id must differ from the accepted source candidate"
        )
    if int(candidate.get("parent_manifest_revision", -1)) != int(manifest["revision"]):
        raise ValueError(
            "Repair candidate was generated from a stale manifest revision"
        )
    expected_parent = execution["parent_candidate_id"]
    if str(candidate.get("parent_candidate_id", "")) != expected_parent:
        raise ValueError(
            "Repair candidate parent_candidate_id does not match the accepted parent"
        )
    staged = dict(execution)
    staged.update(
        {
            "status": "staged",
            "candidate_json_path": str(Path(candidate_json_path).resolve()),
            "candidate_video_path": str(candidate_video.resolve()),
            "candidate": deepcopy(candidate),
            "staged_unix": time.time(),
        }
    )
    staged.pop("execution_hash", None)
    staged["execution_hash"] = _digest(staged)
    report = {
        "schema": REPAIR_EXECUTION_SCHEMA,
        "status": "staged",
        "chain_id": execution["chain_id"],
        "shot_index": int(source["index"]),
        "source_candidate_id": source["candidate_id"],
        "repair_candidate_id": candidate["candidate_id"],
        "candidate_video_path": str(candidate_video),
        "candidate_video_sha256": candidate["video_sha256"],
        "exact_frame_and_sample_boundaries": True,
        "base_manifest_mutated": False,
        "automatic_accept": False,
        "next_action": "Preview the candidate, then explicitly accept it into the repair overlay.",
    }
    return staged, str(candidate_video), report


def _repair_paths(root: Path, plan_hash: str) -> tuple[Path, Path]:
    repair_root = _resolve_inside(root, Path("repairs") / plan_hash)
    return repair_root, repair_root / REPAIR_MANIFEST_NAME


def _cleanup_atomic_temporaries(
    target: Path,
    *,
    temporary_prefix: str | None = None,
    temporary_suffix: str = ".tmp",
) -> list[str]:
    """Remove only abandoned atomic-write files for one exact destination.

    The caller must hold the operation's OS lock. The fixed basename prefix keeps
    cleanup inside this node's own destination namespace and avoids broad temp scans.
    """

    prefix = temporary_prefix or f".{target.name}."
    if not prefix.startswith(".") or any(char in prefix for char in ("/", "\\")):
        raise ValueError("Repair temporary cleanup prefix is invalid")
    if not temporary_suffix.endswith(".tmp") or any(
        char in temporary_suffix for char in ("/", "\\")
    ):
        raise ValueError("Repair temporary cleanup suffix is invalid")
    parent = target.parent.resolve()
    removed = []
    for candidate in sorted(target.parent.glob(f"{prefix}*{temporary_suffix}")):
        if candidate.parent.resolve() != parent:
            raise ValueError("Repair temporary cleanup escaped its destination directory")
        if candidate.is_symlink() or not candidate.is_file():
            raise ValueError(
                f"Repair temporary cleanup refused a non-regular file: {candidate}"
            )
        candidate.unlink()
        removed.append(str(candidate))
    return removed


def _validate_repair_manifest(
    payload: Any, execution: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Repair overlay manifest root must be an object")
    if payload.get("schema") != REPAIR_MANIFEST_SCHEMA:
        raise ValueError("Unsupported repair overlay manifest schema")
    if payload.get("format") != REPAIR_MANIFEST_FORMAT:
        raise ValueError("Repair overlay manifest format marker is invalid")
    if payload.get("chain_id") != execution["chain_id"]:
        raise ValueError("Repair overlay chain_id does not match")
    if payload.get("repair_plan_hash") != execution["repair_plan_hash"]:
        raise ValueError("Repair overlay plan hash does not match")
    if payload.get("base_manifest") != execution["base_manifest"]:
        raise ValueError("Repair overlay was created from a different base manifest")
    if not isinstance(payload.get("revision"), int) or payload["revision"] < 0:
        raise ValueError("Repair overlay revision is invalid")
    replacements = payload.get("replacements")
    if not isinstance(replacements, dict):
        raise ValueError("Repair overlay replacements must be an object")
    for key, entry in replacements.items():
        if str(int(key)) != key or not isinstance(entry, dict):
            raise ValueError("Repair overlay replacement entry is invalid")
    if not isinstance(payload.get("invalidated", []), list):
        raise ValueError("Repair overlay invalidated history must be a list")
    return payload


def _new_repair_manifest(execution: Mapping[str, Any]) -> dict[str, Any]:
    now = time.time()
    return {
        "schema": REPAIR_MANIFEST_SCHEMA,
        "format": REPAIR_MANIFEST_FORMAT,
        "chain_id": execution["chain_id"],
        "repair_plan_hash": execution["repair_plan_hash"],
        "base_manifest": deepcopy(execution["base_manifest"]),
        "revision": 0,
        "replacements": {},
        "invalidated": [],
        "created_unix": now,
        "updated_unix": now,
        "base_manifest_mutated": False,
    }


def _load_repair_manifest(
    manifest_path: Path,
    execution: Mapping[str, Any],
    *,
    allow_new: bool,
) -> tuple[dict[str, Any], str]:
    backup = manifest_path.with_name(REPAIR_MANIFEST_BACKUP_NAME)
    errors = []
    for source, path in (("primary", manifest_path), ("backup", backup)):
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return _validate_repair_manifest(payload, execution), source
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            errors.append(f"{source}: {error}")
    if errors:
        raise ValueError("Repair overlay manifest is corrupt: " + "; ".join(errors))
    if allow_new:
        return _new_repair_manifest(execution), "new"
    raise FileNotFoundError(f"No repair overlay manifest exists at {manifest_path}")


def _write_repair_manifest(
    path: Path,
    payload: Mapping[str, Any],
    execution: Mapping[str, Any],
) -> None:
    _validate_repair_manifest(dict(payload), execution)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        current = path.read_bytes()
        parsed = json.loads(current.decode("utf-8"))
        _validate_repair_manifest(parsed, execution)
        _atomic_write_bytes(path.with_name(REPAIR_MANIFEST_BACKUP_NAME), current)
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode(
        "utf-8"
    )
    _atomic_write_bytes(path, encoded)


def accept_staged_repair(
    staged_repair: Mapping[str, Any],
    accept_repair: bool,
    replace_existing: bool = False,
) -> tuple[str, bool, dict[str, Any]]:
    staged = _validate_execution(staged_repair, {"staged"})
    candidate, candidate_root, candidate_video = _load_candidate(
        staged["candidate_json_path"]
    )
    root = _chain_root(staged["chain_id"])
    if candidate_root.resolve() != root.resolve():
        raise ValueError("Staged candidate chain root changed")
    if not accept_repair:
        return (
            "",
            False,
            {
                "schema": REPAIR_EXECUTION_SCHEMA,
                "status": "preview_only",
                "candidate_video_path": str(candidate_video),
                "base_manifest_mutated": False,
                "repair_overlay_mutated": False,
            },
        )

    with _manifest_lock(root):
        manifest, _ = _assert_base_unchanged(staged)
        source = manifest["segments"][int(staged["source_segment"]["index"])]
        mismatches = _candidate_contract_mismatches(source, candidate)
        if mismatches:
            raise ValueError(
                "Repair candidate changed after staging: " + ", ".join(mismatches)
            )
        repair_root, repair_manifest_path = _repair_paths(
            root, staged["repair_plan_hash"]
        )
        overlay, overlay_source = _load_repair_manifest(
            repair_manifest_path,
            staged,
            allow_new=True,
        )
        safe_candidate = _safe_token(
            candidate["candidate_id"], fallback_prefix="repair"
        )
        accepted_dir = repair_root / "accepted"
        accepted_video = accepted_dir / (
            f"segment_{int(source['index']):05d}_{safe_candidate}.mp4"
        )
        accepted_context = None
        if candidate.get("context_path"):
            accepted_context = accepted_dir / (
                f"segment_{int(source['index']):05d}_{safe_candidate}.context.safetensors"
            )
        removed_temporaries = [
            *_cleanup_atomic_temporaries(accepted_video),
            *_cleanup_atomic_temporaries(repair_manifest_path),
            *_cleanup_atomic_temporaries(
                repair_manifest_path.with_name(REPAIR_MANIFEST_BACKUP_NAME)
            ),
        ]
        if accepted_context is not None:
            removed_temporaries.extend(
                _cleanup_atomic_temporaries(accepted_context)
            )
        key = str(int(source["index"]))
        existing = overlay["replacements"].get(key)
        if existing is not None:
            same = (
                existing.get("candidate_id") == candidate["candidate_id"]
                and existing.get("video_sha256") == candidate["video_sha256"]
                and existing.get("context_sha256", "")
                == candidate.get("context_sha256", "")
            )
            if same:
                return (
                    str(repair_manifest_path),
                    True,
                    {
                        "schema": REPAIR_EXECUTION_SCHEMA,
                        "status": "accepted",
                        "idempotent": True,
                        "overlay_revision": overlay["revision"],
                        "repair_manifest_path": str(repair_manifest_path),
                        "base_manifest_mutated": False,
                        "orphan_temporary_files_removed_before_accept": removed_temporaries,
                    },
                )
            if not replace_existing:
                raise ValueError(
                    "This repair slot already has an accepted overlay candidate. "
                    "Enable replace_existing only after reviewing the new take."
                )

        if (
            accepted_video.exists()
            and _sha256_file(accepted_video) != candidate["video_sha256"]
        ):
            raise ValueError("Repair overlay destination collision")
        video_hash = _copy_atomic(candidate_video, accepted_video)
        context_hash = str(candidate.get("context_sha256", ""))
        if candidate.get("context_path"):
            source_context = _resolve_inside(root, candidate["context_path"])
            assert accepted_context is not None
            if (
                accepted_context.exists()
                and _sha256_file(accepted_context) != context_hash
            ):
                raise ValueError("Repair overlay context destination collision")
            context_hash = _copy_atomic(source_context, accepted_context)

        replacement = deepcopy(source)
        for field in (
            "candidate_id",
            "parent_candidate_id",
            "model_id",
            "sampling_summary",
            "prompt",
            "seed",
            "bit_depth",
            "crf",
            "created_unix",
        ):
            replacement[field] = deepcopy(candidate.get(field))
        replacement.update(
            {
                "video_path": _relative_path(accepted_video, root),
                "video_sha256": video_hash,
                "context_path": (
                    _relative_path(accepted_context, root)
                    if accepted_context is not None
                    else ""
                ),
                "context_sha256": context_hash,
                "repair_source_candidate_id": source["candidate_id"],
                "repair_accepted_unix": time.time(),
            }
        )
        replacements = dict(overlay["replacements"])
        invalidated = list(overlay.get("invalidated", []))
        if existing is not None:
            archived = dict(existing)
            archived["invalidated_unix"] = time.time()
            archived["invalidated_reason"] = (
                f"replaced by repair candidate {candidate['candidate_id']}"
            )
            invalidated.append(archived)
        replacements[key] = replacement
        updated = dict(overlay)
        updated.update(
            {
                "revision": int(overlay["revision"]) + 1,
                "replacements": replacements,
                "invalidated": invalidated,
                "updated_unix": time.time(),
                "base_manifest_mutated": False,
            }
        )
        _write_repair_manifest(repair_manifest_path, updated, staged)

    return (
        str(repair_manifest_path),
        True,
        {
            "schema": REPAIR_EXECUTION_SCHEMA,
            "status": "accepted",
            "idempotent": False,
            "overlay_revision": updated["revision"],
            "overlay_source_before_write": overlay_source,
            "repair_manifest_path": str(repair_manifest_path),
            "accepted_video_path": str(accepted_video),
            "source_candidate_id": source["candidate_id"],
            "repair_candidate_id": candidate["candidate_id"],
            "base_manifest_mutated": False,
            "candidate_files_retained": True,
            "orphan_temporary_files_removed_before_accept": removed_temporaries,
            "rollback": "Compose in base_rollback mode; the accepted base manifest was never changed.",
        },
    )


def load_repair_overlay(
    chain_id: str,
    repair_manifest_path: str,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    root = _chain_root(chain_id)
    resolved = _resolve_inside(root, repair_manifest_path)
    if (
        resolved.name != REPAIR_MANIFEST_NAME
        or resolved.parent.parent.name != "repairs"
    ):
        raise ValueError(
            "repair_manifest_path is not a canonical repair overlay manifest"
        )
    raw = json.loads(resolved.read_text(encoding="utf-8"))
    execution = {
        "schema": REPAIR_EXECUTION_SCHEMA,
        "chain_id": raw.get("chain_id"),
        "repair_plan_hash": raw.get("repair_plan_hash"),
        "base_manifest": raw.get("base_manifest"),
    }
    overlay = _validate_repair_manifest(raw, execution)
    base, _source = load_delivery_manifest(chain_id)
    if _base_snapshot(base) != overlay["base_manifest"]:
        raise ValueError(
            "Accepted base manifest changed after the repair overlay was accepted"
        )
    return overlay, base, root
