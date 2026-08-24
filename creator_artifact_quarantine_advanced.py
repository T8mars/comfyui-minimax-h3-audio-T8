from __future__ import annotations

from collections.abc import Mapping
from contextlib import suppress
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .creator_runtime_advanced import CREATOR_RETENTION_PLAN_SCHEMA
from .creator_workspace_advanced import _hash, canonical_json


CREATOR_QUARANTINE_MANIFEST_SCHEMA = "t8.minimax_h3.creator_quarantine_manifest.v1"
CREATOR_QUARANTINE_RECEIPT_SCHEMA = "t8.minimax_h3.creator_quarantine_receipt.v1"
QUARANTINE_ACTIONS = (
    "prepare_only",
    "quarantine",
    "restore",
    "recover_to_source",
)
MAX_MANIFEST_BYTES = 1024 * 1024
MAX_ENTRY_COUNT = 1024


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path, chunk_bytes: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _json_object(value: str, name: str) -> dict[str, Any]:
    text = str(value or "").strip()
    if not text:
        return {}
    if len(text.encode("utf-8")) > MAX_MANIFEST_BYTES:
        raise ValueError(f"{name} exceeds the 1MiB safety limit")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"{name} is invalid JSON: {error}") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return deepcopy(dict(payload))


def _without_hash(value: Mapping, key: str) -> dict[str, Any]:
    return {name: item for name, item in value.items() if name != key}


def _validate_hashed_object(value: Mapping, *, key: str, name: str) -> None:
    observed = str(value.get(key) or "")
    expected = _hash(_without_hash(value, key))
    if observed != expected:
        raise ValueError(f"{name} {key} does not match its content")


def _is_link_or_junction(path: Path) -> bool:
    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    return bool(is_junction and is_junction())


def _lexical_output_path(output_root: Path, raw_path: str) -> tuple[Path, Path]:
    text = str(raw_path or "").strip()
    if not text or "\x00" in text:
        raise ValueError("artifact path is empty or contains NUL")
    root = output_root.resolve(strict=True)
    candidate_raw = Path(text)
    if candidate_raw.is_absolute():
        lexical = Path(os.path.abspath(os.path.normpath(str(candidate_raw))))
    else:
        parts = list(candidate_raw.parts)
        if parts and parts[0].lower() == "output":
            parts = parts[1:]
        if not parts:
            raise ValueError("artifact path resolves to the output root")
        lexical = Path(os.path.abspath(os.path.normpath(str(root.joinpath(*parts)))))
    try:
        relative_lexical = lexical.relative_to(root)
    except ValueError as error:
        raise ValueError("artifact path escapes the ComfyUI output directory") from error
    if not relative_lexical.parts:
        raise ValueError("artifact path resolves to the output root")

    current = root
    if _is_link_or_junction(current):
        raise ValueError("ComfyUI output root cannot be a symlink or junction")
    for part in relative_lexical.parts:
        current = current / part
        if current.exists() and _is_link_or_junction(current):
            raise ValueError("artifact path traverses a symlink or junction")
    return lexical, relative_lexical


def _validate_retention_plan(retention_plan: Mapping) -> dict[str, Any]:
    if not isinstance(retention_plan, Mapping):
        raise ValueError("retention_plan must be a Creator Retention Plan object")
    plan = deepcopy(dict(retention_plan))
    if plan.get("schema") != CREATOR_RETENTION_PLAN_SCHEMA:
        raise ValueError(f"retention_plan must use schema {CREATOR_RETENTION_PLAN_SCHEMA}")
    _validate_hashed_object(plan, key="plan_hash", name="retention_plan")
    if plan.get("status") != "READY_FOR_EXTERNAL_EXECUTOR":
        raise ValueError("retention_plan is not READY_FOR_EXTERNAL_EXECUTOR")
    if plan.get("external_execution_ready") is not True:
        raise ValueError("retention_plan external_execution_ready must be true")
    if plan.get("artifact_paths_reviewed_by_user") is not True:
        raise ValueError("retention_plan artifact paths were not explicitly reviewed")
    if plan.get("files_mutated") is not False or plan.get("files_deleted") is not False:
        raise ValueError("retention_plan must be an unexecuted report-only plan")
    if plan.get("findings"):
        raise ValueError("retention_plan contains unresolved findings")
    records = plan.get("proposed_delete_manifests")
    if not isinstance(records, list) or not records:
        raise ValueError("retention_plan has no proposed-delete manifests")
    if len(records) > MAX_ENTRY_COUNT:
        raise ValueError(f"retention_plan exceeds {MAX_ENTRY_COUNT} artifact manifests")
    return plan


def _entry_path_hints(plan: Mapping) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    seen: set[str] = set()
    for record in plan["proposed_delete_manifests"]:
        if not isinstance(record, Mapping):
            raise ValueError("proposed-delete manifest entry is invalid")
        artifact_hash = str(record.get("artifact_manifest_hash") or "")
        if len(artifact_hash) != 64:
            raise ValueError("proposed-delete artifact manifest hash is invalid")
        hints = record.get("path_hints")
        if not isinstance(hints, list) or not hints:
            raise ValueError("proposed-delete artifact manifest has no explicit path hint")
        for hint in hints:
            if not isinstance(hint, Mapping):
                raise ValueError("artifact path hint must be an object")
            pointer = str(hint.get("pointer") or "")
            path = str(hint.get("path") or "").strip()
            identity = os.path.normcase(os.path.normpath(path))
            if identity in seen:
                raise ValueError("the same artifact path appears more than once in the plan")
            seen.add(identity)
            entries.append(
                {
                    "artifact_manifest_hash": artifact_hash,
                    "pointer": pointer,
                    "path": path,
                }
            )
    if len(entries) > MAX_ENTRY_COUNT:
        raise ValueError(f"retention plan exceeds {MAX_ENTRY_COUNT} file paths")
    return entries


def _quarantine_base(output_root: Path) -> Path:
    return output_root.resolve() / "MiniMaxH3" / "creator_quarantine"


def prepare_creator_quarantine_manifest(
    retention_plan: Mapping,
    output_root: str | Path,
    hash_chunk_megabytes: int = 8,
) -> dict[str, Any]:
    if not 1 <= int(hash_chunk_megabytes) <= 64:
        raise ValueError("hash_chunk_megabytes must be between 1 and 64")
    root = Path(output_root).resolve(strict=True)
    plan = _validate_retention_plan(retention_plan)
    quarantine_base = _quarantine_base(root)
    chunk_bytes = int(hash_chunk_megabytes) * 1024 * 1024
    entries = []
    resolved_sources: set[str] = set()
    for hint in _entry_path_hints(plan):
        source, relative = _lexical_output_path(root, hint["path"])
        source_identity = os.path.normcase(str(source))
        if source_identity in resolved_sources:
            raise ValueError(
                "the same resolved artifact appears more than once in the retention plan"
            )
        resolved_sources.add(source_identity)
        try:
            source.relative_to(quarantine_base)
        except ValueError:
            pass
        else:
            raise ValueError("retention plan cannot quarantine an existing quarantine artifact")
        if not source.exists():
            raise FileNotFoundError(f"artifact does not exist: {source}")
        if not source.is_file():
            raise ValueError(f"artifact is not a regular file: {source}")
        stat = source.stat()
        entries.append(
            {
                "artifact_manifest_hash": hint["artifact_manifest_hash"],
                "pointer": hint["pointer"],
                "source_relative": relative.as_posix(),
                "bytes": int(stat.st_size),
                "sha256": _sha256_file(source, chunk_bytes),
            }
        )
    manifest = {
        "schema": CREATOR_QUARANTINE_MANIFEST_SCHEMA,
        "plan_hash": plan["plan_hash"],
        "workspace_hash": plan["workspace_hash"],
        "ledger_hash": plan["ledger_hash"],
        "output_root_identity": str(root),
        "entry_count": len(entries),
        "total_bytes": sum(item["bytes"] for item in entries),
        "entries": entries,
        "prepared_at": _utc_now(),
        "files_mutated": False,
        "files_deleted": False,
        "permanent_delete_supported": False,
    }
    manifest["manifest_hash"] = _hash(manifest)
    if len(canonical_json(manifest).encode("utf-8")) > MAX_MANIFEST_BYTES:
        raise ValueError("prepared quarantine manifest exceeds the 1MiB safety limit")
    return manifest


def _validate_manifest(
    manifest: Mapping,
    retention_plan: Mapping,
    output_root: Path,
) -> dict[str, Any]:
    if manifest.get("schema") != CREATOR_QUARANTINE_MANIFEST_SCHEMA:
        raise ValueError(
            f"execution manifest must use schema {CREATOR_QUARANTINE_MANIFEST_SCHEMA}"
        )
    value = deepcopy(dict(manifest))
    _validate_hashed_object(value, key="manifest_hash", name="execution manifest")
    if value.get("plan_hash") != retention_plan.get("plan_hash"):
        raise ValueError("execution manifest plan_hash does not match retention_plan")
    if value.get("workspace_hash") != retention_plan.get("workspace_hash"):
        raise ValueError("execution manifest workspace_hash does not match retention_plan")
    if value.get("ledger_hash") != retention_plan.get("ledger_hash"):
        raise ValueError("execution manifest ledger_hash does not match retention_plan")
    if value.get("output_root_identity") != str(output_root.resolve()):
        raise ValueError("execution manifest belongs to a different ComfyUI output root")
    entries = value.get("entries")
    if not isinstance(entries, list) or not entries or len(entries) > MAX_ENTRY_COUNT:
        raise ValueError("execution manifest entry list is invalid")
    if value.get("entry_count") != len(entries):
        raise ValueError("execution manifest entry_count is invalid")
    if value.get("files_mutated") is not False or value.get("files_deleted") is not False:
        raise ValueError("execution manifest must describe an unmodified prepared state")
    return value


def _receipt_manifest(payload: Mapping) -> dict[str, Any]:
    if payload.get("schema") != CREATOR_QUARANTINE_RECEIPT_SCHEMA:
        return deepcopy(dict(payload))
    receipt = deepcopy(dict(payload))
    _validate_hashed_object(receipt, key="receipt_hash", name="quarantine receipt")
    manifest = receipt.get("execution_manifest")
    if not isinstance(manifest, Mapping):
        raise ValueError("quarantine receipt has no execution_manifest")
    return deepcopy(dict(manifest))


def _entry_paths(
    output_root: Path,
    plan_hash: str,
    epoch: int,
    entry: Mapping,
) -> tuple[Path, Path]:
    source, relative = _lexical_output_path(output_root, str(entry["source_relative"]))
    target = (
        _quarantine_base(output_root)
        / plan_hash
        / f"epoch-{int(epoch):010d}"
        / "files"
        / relative
    )
    target_parent = target.parent
    try:
        target_parent.resolve(strict=False).relative_to(_quarantine_base(output_root))
    except ValueError as error:
        raise ValueError("quarantine target escaped its bounded root") from error
    return source, target


def _verify_file(path: Path, entry: Mapping, chunk_bytes: int, label: str) -> None:
    if not path.exists() or not path.is_file() or _is_link_or_junction(path):
        raise ValueError(f"{label} is missing, non-regular, or linked: {path}")
    if path.stat().st_size != int(entry["bytes"]):
        raise ValueError(f"{label} byte size changed: {path}")
    if _sha256_file(path, chunk_bytes) != str(entry["sha256"]):
        raise ValueError(f"{label} SHA-256 changed: {path}")


def _atomic_json(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(canonical_json(payload), encoding="utf-8")
    os.replace(temporary, path)


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _acquire_lock(path: Path, *, allow_stale_recovery: bool = False) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(2):
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as error:
            if not allow_stale_recovery or attempt:
                raise ValueError(
                    "another quarantine operation holds this plan/epoch lock"
                ) from error
            try:
                payload = _json_object(path.read_text(encoding="utf-8"), "operation lock")
                owner_pid = int(payload.get("pid", 0))
            except (OSError, ValueError, TypeError):
                owner_pid = 0
            if owner_pid and _pid_is_running(owner_pid):
                raise ValueError(
                    "another quarantine operation holds this plan/epoch lock"
                ) from error
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            continue
        payload = canonical_json({"pid": os.getpid(), "created_at": _utc_now()}).encode(
            "utf-8"
        )
        os.write(descriptor, payload)
        os.fsync(descriptor)
        return descriptor
    raise RuntimeError("failed to acquire quarantine operation lock")


def _validate_operation_root(output_root: Path, operation_root: Path) -> None:
    quarantine_base = _quarantine_base(output_root)
    try:
        operation_root.resolve(strict=False).relative_to(
            quarantine_base.resolve(strict=False)
        )
    except ValueError as error:
        raise ValueError("quarantine operation root escaped its bounded directory") from error
    current = output_root.resolve()
    for part in operation_root.relative_to(current).parts:
        current = current / part
        if current.exists() and _is_link_or_junction(current):
            raise ValueError("quarantine operation root traverses a symlink or junction")


def _operation_receipt(
    *,
    operation: str,
    status: str,
    epoch: int,
    manifest: Mapping,
    journal_relative: str,
    entries: list[dict[str, Any]],
) -> dict[str, Any]:
    receipt = {
        "schema": CREATOR_QUARANTINE_RECEIPT_SCHEMA,
        "operation": operation,
        "status": status,
        "execution_epoch": int(epoch),
        "plan_hash": manifest["plan_hash"],
        "manifest_hash": manifest["manifest_hash"],
        "execution_manifest": deepcopy(dict(manifest)),
        "journal_relative": journal_relative,
        "entry_count": len(entries),
        "entries": entries,
        "files_deleted": False,
        "permanent_delete_supported": False,
        "completed_at": _utc_now(),
    }
    receipt["receipt_hash"] = _hash(receipt)
    return receipt


def execute_creator_artifact_quarantine(
    retention_plan: Mapping,
    output_root: str | Path,
    action: str = "prepare_only",
    execution_manifest_json: str = "",
    expected_plan_hash: str = "",
    execution_epoch: int = 0,
    confirm_action: bool = False,
    hash_chunk_megabytes: int = 8,
) -> tuple[dict, str, str, int, int, str, str]:
    root = Path(output_root).resolve(strict=True)
    plan = _validate_retention_plan(retention_plan)
    normalized_action = str(action or "").strip()
    if normalized_action not in QUARANTINE_ACTIONS:
        raise ValueError("unsupported creator quarantine action")
    if not 1 <= int(hash_chunk_megabytes) <= 64:
        raise ValueError("hash_chunk_megabytes must be between 1 and 64")
    chunk_bytes = int(hash_chunk_megabytes) * 1024 * 1024

    supplied = _json_object(execution_manifest_json, "execution_manifest_json")
    if normalized_action == "prepare_only":
        if supplied:
            raise ValueError("prepare_only requires an empty execution_manifest_json")
        manifest = prepare_creator_quarantine_manifest(
            plan,
            root,
            hash_chunk_megabytes=hash_chunk_megabytes,
        )
        status = "PREPARED_REVIEW_REQUIRED"
        report = {
            "schema": CREATOR_QUARANTINE_MANIFEST_SCHEMA,
            "status": status,
            "plan_hash": plan["plan_hash"],
            "manifest_hash": manifest["manifest_hash"],
            "entry_count": manifest["entry_count"],
            "total_bytes": manifest["total_bytes"],
            "files_mutated": False,
            "files_deleted": False,
            "next_step": (
                "Review and retain manifest_json, then use quarantine with the exact plan hash, "
                "a new positive epoch and confirm_action=true."
            ),
        }
        return (
            manifest,
            status,
            canonical_json(manifest),
            int(manifest["entry_count"]),
            int(manifest["total_bytes"]),
            "",
            canonical_json(report),
        )

    if not supplied:
        raise ValueError(f"{normalized_action} requires execution_manifest_json")
    manifest = _receipt_manifest(supplied)
    manifest = _validate_manifest(manifest, plan, root)
    if str(expected_plan_hash or "").strip() != str(plan["plan_hash"]):
        raise ValueError("expected_plan_hash must exactly match the reviewed retention plan")
    if int(execution_epoch) <= 0:
        raise ValueError("execution_epoch must be a new positive integer")
    if not confirm_action:
        raise ValueError(f"confirm_action must be true before {normalized_action}")

    operation_root = (
        _quarantine_base(root)
        / plan["plan_hash"]
        / f"epoch-{int(execution_epoch):010d}"
    )
    journal_path = operation_root / "journal.json"
    lock_path = operation_root / ".operation.lock"
    _validate_operation_root(root, operation_root)
    lock_fd = _acquire_lock(
        lock_path,
        allow_stale_recovery=normalized_action == "recover_to_source",
    )
    operation_entries: list[dict[str, Any]] = []
    moved: list[tuple[Path, Path, dict[str, Any]]] = []
    try:
        if normalized_action == "quarantine":
            if journal_path.exists():
                raise ValueError("execution_epoch already has a quarantine journal")
            for entry in manifest["entries"]:
                source, target = _entry_paths(
                    root, plan["plan_hash"], int(execution_epoch), entry
                )
                _verify_file(source, entry, chunk_bytes, "source artifact")
                if target.exists():
                    raise ValueError(f"quarantine target already exists: {target}")
            journal = {
                "schema": CREATOR_QUARANTINE_RECEIPT_SCHEMA,
                "operation": "quarantine",
                "state": "moving",
                "execution_epoch": int(execution_epoch),
                "execution_manifest": manifest,
                "moved_source_relatives": [],
                "files_deleted": False,
            }
            _atomic_json(journal_path, journal)
            for entry in manifest["entries"]:
                source, target = _entry_paths(
                    root, plan["plan_hash"], int(execution_epoch), entry
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(source, target)
                moved.append((source, target, entry))
                journal["moved_source_relatives"].append(entry["source_relative"])
                _atomic_json(journal_path, journal)
                operation_entries.append(
                    {
                        "source_relative": entry["source_relative"],
                        "quarantine_relative": target.relative_to(root).as_posix(),
                        "bytes": entry["bytes"],
                        "sha256": entry["sha256"],
                    }
                )
            journal["state"] = "quarantined"
            _atomic_json(journal_path, journal)
            status = "QUARANTINED_RECOVERABLE"
        else:
            if not journal_path.is_file():
                raise FileNotFoundError("matching quarantine journal does not exist")
            journal = _json_object(
                journal_path.read_text(encoding="utf-8"), "quarantine journal"
            )
            if normalized_action == "restore" and journal.get("state") != "quarantined":
                raise ValueError(
                    "restore requires a completed quarantined journal; use recover_to_source "
                    "for an interrupted operation"
                )
            journal_manifest = journal.get("execution_manifest")
            if not isinstance(journal_manifest, Mapping):
                raise ValueError("quarantine journal has no execution manifest")
            validated_journal_manifest = _validate_manifest(journal_manifest, plan, root)
            if validated_journal_manifest["manifest_hash"] != manifest["manifest_hash"]:
                raise ValueError("supplied manifest does not match the quarantine journal")
            states = []
            for entry in manifest["entries"]:
                source, target = _entry_paths(
                    root, plan["plan_hash"], int(execution_epoch), entry
                )
                source_exists = source.exists()
                target_exists = target.exists()
                if source_exists and target_exists:
                    raise ValueError("both source and quarantine target exist; refusing overwrite")
                if not source_exists and not target_exists:
                    raise FileNotFoundError("both source and quarantine target are missing")
                if source_exists:
                    _verify_file(source, entry, chunk_bytes, "restored source artifact")
                    states.append("source")
                else:
                    _verify_file(target, entry, chunk_bytes, "quarantined artifact")
                    states.append("quarantine")
            for entry, state in zip(manifest["entries"], states, strict=True):
                source, target = _entry_paths(
                    root, plan["plan_hash"], int(execution_epoch), entry
                )
                if state == "quarantine":
                    source.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(target, source)
                operation_entries.append(
                    {
                        "source_relative": entry["source_relative"],
                        "bytes": entry["bytes"],
                        "sha256": entry["sha256"],
                        "state": "source",
                    }
                )
            journal["state"] = "restored"
            journal["restored_at"] = _utc_now()
            _atomic_json(journal_path, journal)
            status = (
                "RESTORED"
                if normalized_action == "restore"
                else "RECOVERED_TO_SOURCE"
            )
    except Exception:
        if normalized_action == "quarantine" and moved:
            rollback_failed = []
            for source, target, entry in reversed(moved):
                try:
                    if target.exists() and not source.exists():
                        source.parent.mkdir(parents=True, exist_ok=True)
                        os.replace(target, source)
                except OSError:
                    rollback_failed.append(entry["source_relative"])
            with suppress(OSError):
                journal = {
                    "schema": CREATOR_QUARANTINE_RECEIPT_SCHEMA,
                    "operation": "quarantine",
                    "state": "recovery_required" if rollback_failed else "rolled_back",
                    "execution_epoch": int(execution_epoch),
                    "execution_manifest": manifest,
                    "rollback_failed": rollback_failed,
                    "files_deleted": False,
                }
                _atomic_json(journal_path, journal)
        raise
    finally:
        os.close(lock_fd)
        with suppress(FileNotFoundError):
            lock_path.unlink()

    receipt = _operation_receipt(
        operation=normalized_action,
        status=status,
        epoch=int(execution_epoch),
        manifest=manifest,
        journal_relative=journal_path.relative_to(root).as_posix(),
        entries=operation_entries,
    )
    report = {
        "schema": CREATOR_QUARANTINE_RECEIPT_SCHEMA,
        "status": status,
        "operation": normalized_action,
        "plan_hash": plan["plan_hash"],
        "manifest_hash": manifest["manifest_hash"],
        "receipt_hash": receipt["receipt_hash"],
        "entry_count": len(operation_entries),
        "total_bytes": int(manifest["total_bytes"]),
        "files_mutated": True,
        "files_deleted": False,
        "recoverable": True,
        "journal_relative": receipt["journal_relative"],
        "boundary": (
            "Files are moved only inside the same ComfyUI output root. No permanent-delete "
            "operation exists; restore/recover requires the same reviewed plan, manifest and epoch."
        ),
    }
    return (
        manifest,
        status,
        canonical_json(manifest),
        int(manifest["entry_count"]),
        int(manifest["total_bytes"]),
        canonical_json(receipt),
        canonical_json(report),
    )
