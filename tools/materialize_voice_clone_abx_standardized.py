#!/usr/bin/env python3
"""Safely materialize fair A/B/X media for the formal voice-clone identity panel.

The default invocation is preflight-only.  ``--confirm-run`` processes at most ten unique files
serially (hard maximum 25), converting them to 32kHz mono FLAC without loudness normalization.
Every input is SHA-bound, every output is atomically promoted, and resumable state is written after
each file.  The final manifest is emitted only when every registered output passes the identical
codec/sample-rate/channel/container contract and every A/B/X case has distinct content.
"""

from __future__ import annotations

import argparse
from contextlib import suppress
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tempfile
from typing import Any, Mapping
import uuid

try:
    from . import build_voice_clone_abx_review as abx
    from . import build_voice_clone_identity_formal_matrix as identity
except ImportError:  # pragma: no cover - direct script execution
    import build_voice_clone_abx_review as abx
    import build_voice_clone_identity_formal_matrix as identity


PREFLIGHT_SCHEMA = "minimax_h3_t8_voice_clone_abx_standardization_preflight_v1"
STATE_SCHEMA = "minimax_h3_t8_voice_clone_abx_standardization_state_v1"
RESULT_SCHEMA = "minimax_h3_t8_voice_clone_abx_standardization_result_v1"
MAX_FILES_PER_INVOCATION = 25
DEFAULT_FILES_PER_INVOCATION = 10
ROLES = ("target_reference", "impostor_reference", "candidate")
EXPECTED_CONTRACT = {
    "sample_rate": 32000,
    "channels": 1,
    "codec": "flac",
    "container": "flac",
    "loudness_normalization": False,
    "duration_policy": "preserve_each_source",
}


class JobsContractError(ValueError):
    pass


class InputIdentityError(ValueError):
    pass


class StateContractError(ValueError):
    pass


class OutputConflictError(ValueError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _fixed_sha256(value: Any, context: str) -> str:
    token = str(value or "").strip().upper()
    if len(token) != 64 or any(char not in "0123456789ABCDEF" for char in token):
        raise JobsContractError(f"{context} must be a 64-character SHA-256")
    return token


def _write_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _required_text(row: Mapping[str, Any], field: str, context: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise JobsContractError(f"{context}.{field} must be non-empty text")
    return value.strip()


def _safe_relative(value: Any, context: str) -> PurePosixPath:
    if not isinstance(value, str) or not value.strip():
        raise JobsContractError(f"{context} must be a relative path")
    normalized = value.replace("\\", "/").strip()
    path = PurePosixPath(normalized)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise JobsContractError(f"{context} must be a safe relative path")
    if path.suffix.casefold() != ".flac":
        raise JobsContractError(f"{context} must end in .flac")
    return path


def _below(root: Path, relative: PurePosixPath, context: str) -> Path:
    base = root.resolve()
    target = base.joinpath(*relative.parts).resolve()
    try:
        target.relative_to(base)
    except ValueError as error:
        raise JobsContractError(f"{context} escapes output root") from error
    return target


def _resolve_input(jobs_dir: Path, value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise JobsContractError(f"{context} must be a path/SHA object")
    raw = Path(_required_text(value, "path", context))
    unresolved = raw if raw.is_absolute() else jobs_dir / raw
    if unresolved.is_symlink():
        raise InputIdentityError(f"{context} must not be a symlink")
    try:
        path = unresolved.resolve(strict=True)
    except FileNotFoundError as error:
        raise InputIdentityError(f"{context} input is missing: {path}") from error
    if not path.is_file() or path.is_symlink():
        raise InputIdentityError(f"{context} must resolve to a regular non-symlink file")
    expected = _fixed_sha256(value.get("sha256"), f"{context}.sha256")
    actual = _sha256_file(path)
    if actual != expected:
        raise InputIdentityError(f"{context} SHA-256 drift: {path}")
    return {"path": str(path), "sha256": expected}


def _resolve_executable(value: str) -> Path | None:
    candidate = Path(value)
    if candidate.is_file():
        return candidate.resolve()
    resolved = shutil.which(value)
    return Path(resolved).resolve() if resolved else None


def _probe_audio(path: Path, ffprobe: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            str(ffprobe),
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,codec_name,sample_rate,channels:format=format_name,duration",
            "-of",
            "json",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        shell=False,
    )
    if completed.returncode != 0:
        raise ValueError(f"ffprobe failed for {path.name}: {completed.stderr.strip()}")
    payload = json.loads(completed.stdout)
    audio_streams = [
        row for row in payload.get("streams", []) if row.get("codec_type") == "audio"
    ]
    if len(audio_streams) != 1:
        raise ValueError(f"{path.name} must contain exactly one audio stream")
    stream = audio_streams[0]
    format_row = payload.get("format", {})
    try:
        sample_rate = int(stream["sample_rate"])
        channels = int(stream["channels"])
        duration = float(format_row["duration"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{path.name} has incomplete audio metadata") from error
    if sample_rate <= 0 or channels <= 0 or not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"{path.name} has invalid audio metadata")
    return {
        "codec_name": str(stream.get("codec_name") or ""),
        "sample_rate": sample_rate,
        "channels": channels,
        "format_name": str(format_row.get("format_name") or ""),
        "duration_seconds": duration,
        "bytes": path.stat().st_size,
    }


def _assert_output_contract(contract: Mapping[str, Any], context: str) -> None:
    if (
        contract.get("codec_name") != "flac"
        or contract.get("sample_rate") != 32000
        or contract.get("channels") != 1
        or "flac" not in str(contract.get("format_name") or "").split(",")
    ):
        raise ValueError(f"{context} is not 32kHz mono FLAC")


def normalize_jobs(
    payload: Any,
    *,
    jobs_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("schema") != identity.STANDARDIZATION_JOBS_SCHEMA:
        raise JobsContractError(
            f"jobs schema must be {identity.STANDARDIZATION_JOBS_SCHEMA}"
        )
    if payload.get("contract") != EXPECTED_CONTRACT:
        raise JobsContractError("jobs standardization contract differs from the reviewed v1 contract")
    if payload.get("execution_started") is not False or payload.get("abx_manifest_written") is not False:
        raise JobsContractError("authoritative jobs must remain an unexecuted preregistration")
    if payload.get("high_fidelity_clone_claim") != "NOT_ESTABLISHED":
        raise JobsContractError("jobs must not claim high-fidelity cloning")
    plan_id = _required_text(payload, "plan_id", "jobs")
    review_id = _required_text(payload, "review_id", "jobs")
    generation_plan_sha256 = _fixed_sha256(
        payload.get("generation_plan_sha256"), "jobs.generation_plan_sha256"
    )
    identity_design_sha256 = _fixed_sha256(
        payload.get("identity_design_sha256"), "jobs.identity_design_sha256"
    )
    rows = payload.get("jobs")
    if not isinstance(rows, list) or not rows or payload.get("job_count") != len(rows):
        raise JobsContractError("jobs.job_count must match a non-empty jobs list")

    jobs_dir = jobs_path.resolve().parent
    normalized_rows = []
    files: dict[str, dict[str, Any]] = {}
    seen_case_ids: set[str] = set()
    seen_candidate_hashes: set[str] = set()
    seen_candidate_outputs: set[str] = set()
    for index, raw in enumerate(rows):
        context = f"jobs[{index}]"
        if not isinstance(raw, dict):
            raise JobsContractError(f"{context} must be an object")
        case_id = _required_text(raw, "case_id", context)
        if case_id in seen_case_ids:
            raise JobsContractError(f"duplicate case_id: {case_id}")
        seen_case_ids.add(case_id)
        target = _required_text(raw, "target_speaker_id", context)
        impostor = _required_text(raw, "impostor_speaker_id", context)
        condition = _required_text(raw, "condition_id", context)
        if target == impostor or condition != target:
            raise JobsContractError(f"{case_id} has invalid target/impostor/condition identity")
        seed = raw.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise JobsContractError(f"{case_id}.seed must be an integer")
        if raw.get("seed_known") is not True:
            raise JobsContractError(f"{case_id}.seed_known must be true")
        inputs = raw.get("inputs")
        outputs = raw.get("outputs")
        if not isinstance(inputs, dict) or not isinstance(outputs, dict):
            raise JobsContractError(f"{case_id} requires inputs and outputs objects")

        normalized_inputs = {}
        normalized_outputs = {}
        for role in ROLES:
            input_contract = _resolve_input(jobs_dir, inputs.get(role), f"{case_id}.{role}")
            relative = _safe_relative(outputs.get(role), f"{case_id}.{role} output")
            output_path = _below(output_root, relative, f"{case_id}.{role} output")
            relative_text = relative.as_posix()
            existing = files.get(relative_text)
            identity_contract = {
                "relative_path": relative_text,
                "path": str(output_path),
                "input_path": input_contract["path"],
                "input_sha256": input_contract["sha256"],
            }
            if existing is not None and (
                existing["input_path"] != identity_contract["input_path"]
                or existing["input_sha256"] != identity_contract["input_sha256"]
            ):
                raise JobsContractError(f"output path maps to conflicting inputs: {relative_text}")
            files[relative_text] = existing or identity_contract
            normalized_inputs[role] = input_contract
            normalized_outputs[role] = relative_text
        candidate_hash = normalized_inputs["candidate"]["sha256"]
        candidate_output = normalized_outputs["candidate"]
        if candidate_hash in seen_candidate_hashes:
            raise JobsContractError("candidate input content must be unique across ABX cases")
        if candidate_output in seen_candidate_outputs:
            raise JobsContractError("candidate output path must be unique across ABX cases")
        seen_candidate_hashes.add(candidate_hash)
        seen_candidate_outputs.add(candidate_output)
        normalized_rows.append(
            {
                "case_id": case_id,
                "target_speaker_id": target,
                "impostor_speaker_id": impostor,
                "condition_id": condition,
                "utterance_id": _required_text(raw, "utterance_id", context),
                "language_code": _required_text(raw, "language_code", context),
                "seed": seed,
                "seed_known": True,
                "inputs": normalized_inputs,
                "outputs": normalized_outputs,
            }
        )
    return {
        "schema": identity.STANDARDIZATION_JOBS_SCHEMA,
        "plan_id": plan_id,
        "review_id": review_id,
        "generation_plan_sha256": generation_plan_sha256,
        "identity_design_sha256": identity_design_sha256,
        "contract": EXPECTED_CONTRACT,
        "job_count": len(normalized_rows),
        "jobs": normalized_rows,
        "files": files,
        "jobs_sha256": _sha256_file(jobs_path.resolve()),
    }


def _state_path(output_root: Path) -> Path:
    return output_root.resolve() / "materialization_state.json"


def _state_sha256(value: Any, context: str) -> str:
    try:
        return _fixed_sha256(value, context)
    except JobsContractError as error:
        raise StateContractError(str(error)) from error


def _load_state(output_root: Path, jobs_sha256: str) -> dict[str, Any] | None:
    path = _state_path(output_root)
    if not path.exists():
        return None
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("schema") != STATE_SCHEMA:
        raise StateContractError(f"state schema must be {STATE_SCHEMA}")
    if str(state.get("jobs_sha256") or "").upper() != jobs_sha256:
        raise StateContractError("state belongs to a different jobs file")
    if not isinstance(state.get("files"), dict):
        raise StateContractError("state.files must be an object")
    return state


def _new_state(normalized: Mapping[str, Any]) -> dict[str, Any]:
    now = _utc_now()
    return {
        "schema": STATE_SCHEMA,
        "plan_id": normalized["plan_id"],
        "review_id": normalized["review_id"],
        "jobs_sha256": normalized["jobs_sha256"],
        "created_at": now,
        "updated_at": now,
        "files": {},
        "manifest": None,
        "high_fidelity_clone_claim": "NOT_ESTABLISHED",
    }


def _save_state(output_root: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = _utc_now()
    _write_atomic(_state_path(output_root), state)


def _audit_outputs(
    normalized: Mapping[str, Any],
    state: Mapping[str, Any] | None,
    ffprobe: Path,
) -> tuple[list[str], list[str]]:
    completed = state.get("files", {}) if state is not None else {}
    pending = []
    conflicts = []
    for relative, contract in normalized["files"].items():
        target = Path(contract["path"])
        record = completed.get(relative)
        if record is None:
            if target.exists():
                conflicts.append(f"untracked output exists: {target}")
            else:
                pending.append(relative)
            continue
        if not isinstance(record, dict):
            raise StateContractError(f"invalid state record for {relative}")
        if record.get("input_sha256") != contract["input_sha256"]:
            raise StateContractError(f"state input identity differs for {relative}")
        expected_output_sha = _state_sha256(
            record.get("output_sha256"), f"state.files[{relative}].output_sha256"
        )
        if not target.is_file() or target.is_symlink():
            conflicts.append(f"completed output missing or not regular: {target}")
            continue
        if _sha256_file(target) != expected_output_sha:
            conflicts.append(f"completed output SHA-256 drift: {target}")
            continue
        try:
            media = _probe_audio(target, ffprobe)
            _assert_output_contract(media, relative)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            conflicts.append(str(error))
    return pending, conflicts


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    base = {
        "schema": PREFLIGHT_SCHEMA,
        "created_at": _utc_now(),
        "ready_for_materialization": False,
        "materialization_started": False,
        "high_fidelity_clone_claim": "NOT_ESTABLISHED",
        "boundary": (
            "Preflight hashes inputs and audits tracked outputs only. It never invokes FFmpeg or "
            "creates standardized media."
        ),
    }
    if (
        isinstance(args.max_files, bool)
        or not isinstance(args.max_files, int)
        or not 1 <= args.max_files <= MAX_FILES_PER_INVOCATION
    ):
        return {**base, "status": "ABSTAIN_INVALID_CONFIGURATION"}
    ffmpeg = _resolve_executable(args.ffmpeg)
    ffprobe = _resolve_executable(args.ffprobe)
    if ffmpeg is None or ffprobe is None:
        return {
            **base,
            "status": "ABSTAIN_MISSING_FFMPEG",
            "ffmpeg": str(ffmpeg) if ffmpeg else None,
            "ffprobe": str(ffprobe) if ffprobe else None,
        }
    try:
        jobs_path = args.jobs.resolve(strict=True)
        payload = json.loads(jobs_path.read_text(encoding="utf-8"))
        normalized = normalize_jobs(
            payload, jobs_path=jobs_path, output_root=args.output_root.resolve()
        )
        state = _load_state(args.output_root, normalized["jobs_sha256"])
        pending, conflicts = _audit_outputs(normalized, state, ffprobe)
    except InputIdentityError as error:
        return {**base, "status": "ABSTAIN_INPUT_IDENTITY_DRIFT", "error": str(error)}
    except (JobsContractError, FileNotFoundError, json.JSONDecodeError) as error:
        return {**base, "status": "ABSTAIN_JOBS_INVALID", "error": str(error)}
    except StateContractError as error:
        return {**base, "status": "ABSTAIN_STATE_INVALID", "error": str(error)}
    lock_path = args.output_root.resolve() / "standardization.lock"
    manifest_path = args.output_root.resolve() / "abx_manifest.json"
    manifest_record = state.get("manifest") if state is not None else None
    manifest_complete = False
    if not pending and not conflicts and isinstance(manifest_record, dict):
        expected = str(manifest_record.get("sha256") or "").upper()
        manifest_complete = (
            len(expected) == 64
            and manifest_path.is_file()
            and _sha256_file(manifest_path) == expected
        )
    if manifest_path.exists() and not manifest_complete:
        conflicts.append(f"untracked or stale ABX manifest exists: {manifest_path}")
    if conflicts:
        status = "ABSTAIN_OUTPUT_CONFLICT"
    elif lock_path.exists():
        status = "ABSTAIN_EXECUTION_LOCK_PRESENT"
    elif manifest_complete:
        status = "COMPLETE_ALREADY"
    elif not pending:
        status = "READY_FINALIZE"
    else:
        status = "READY"
    selected = pending[: args.max_files]
    return {
        **base,
        "status": status,
        "ready_for_materialization": status in {"READY", "READY_FINALIZE"},
        "jobs_path": str(jobs_path),
        "jobs_sha256": normalized["jobs_sha256"],
        "plan_id": normalized["plan_id"],
        "review_id": normalized["review_id"],
        "registered_abx_cases": normalized["job_count"],
        "registered_unique_files": len(normalized["files"]),
        "completed_unique_files": len(normalized["files"]) - len(pending),
        "pending_unique_files": len(pending),
        "selected_files": selected,
        "output_conflicts": conflicts,
        "ffmpeg": str(ffmpeg),
        "ffprobe": str(ffprobe),
        "output_root": str(args.output_root.resolve()),
    }


class ExecutionLock:
    def __init__(self, path: Path):
        self.path = path
        self.acquired = False

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as error:
            raise RuntimeError(f"standardization lock already exists: {self.path}") from error
        try:
            os.write(
                descriptor,
                _json_bytes({"pid": os.getpid(), "created_at": _utc_now()}),
            )
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        self.acquired = True
        return self

    def __exit__(self, _exc_type, _exc, _tb):
        if self.acquired:
            with suppress(FileNotFoundError):
                self.path.unlink()


def _encode_one(
    contract: Mapping[str, Any],
    *,
    target: Path,
    ffmpeg: Path,
    ffprobe: Path,
) -> tuple[str, dict[str, Any]]:
    source = Path(str(contract["input_path"]))
    if _sha256_file(source) != contract["input_sha256"]:
        raise InputIdentityError(f"input changed immediately before encode: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.stem}.tmp-{uuid.uuid4().hex}.flac")
    try:
        completed = subprocess.run(
            [
                str(ffmpeg),
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(source),
                "-map",
                "0:a:0",
                "-vn",
                "-sn",
                "-dn",
                "-ar",
                "32000",
                "-ac",
                "1",
                "-c:a",
                "flac",
                "-sample_fmt",
                "s16",
                str(temporary),
            ],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            shell=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"FFmpeg failed for {source.name}: {completed.stderr.strip()}"
            )
        media = _probe_audio(temporary, ffprobe)
        _assert_output_contract(media, temporary.name)
        output_sha256 = _sha256_file(temporary)
        temporary.replace(target)
        if _sha256_file(target) != output_sha256:
            raise RuntimeError(f"atomic output hash mismatch: {target}")
        return output_sha256, media
    finally:
        if temporary.exists():
            temporary.unlink()


def _build_manifest(
    normalized: Mapping[str, Any],
    state: Mapping[str, Any],
    ffprobe: Path,
) -> dict[str, Any]:
    completed = state["files"]
    candidate_hashes: set[str] = set()
    cases = []
    for row in normalized["jobs"]:
        role_hashes = []
        for role in ROLES:
            relative = row["outputs"][role]
            target = Path(normalized["files"][relative]["path"])
            record = completed.get(relative)
            if not isinstance(record, dict) or not target.is_file():
                raise RuntimeError(f"standardized output is incomplete: {relative}")
            output_sha = _state_sha256(
                record.get("output_sha256"), f"state.files[{relative}].output_sha256"
            )
            if _sha256_file(target) != output_sha:
                raise OutputConflictError(f"standardized output hash drift: {target}")
            media = _probe_audio(target, ffprobe)
            _assert_output_contract(media, relative)
            role_hashes.append(output_sha)
        if len(set(role_hashes)) != 3:
            raise OutputConflictError(f"{row['case_id']} A/B/X content is not distinct")
        if role_hashes[2] in candidate_hashes:
            raise OutputConflictError("standardized candidate content is reused across ABX cases")
        candidate_hashes.add(role_hashes[2])
        cases.append(
            {
                "case_id": row["case_id"],
                "target_speaker_id": row["target_speaker_id"],
                "impostor_speaker_id": row["impostor_speaker_id"],
                "condition_id": row["condition_id"],
                "utterance_id": row["utterance_id"],
                "language_code": row["language_code"],
                "seed": row["seed"],
                "seed_known": True,
                "target_reference": row["outputs"]["target_reference"],
                "impostor_reference": row["outputs"]["impostor_reference"],
                "candidate": row["outputs"]["candidate"],
            }
        )
    return {
        "schema": abx.MANIFEST_SCHEMA,
        "review_id": normalized["review_id"],
        "target_position_policy": "balanced_by_target_and_global",
        "source_jobs_sha256": normalized["jobs_sha256"],
        "generation_plan_sha256": normalized["generation_plan_sha256"],
        "identity_design_sha256": normalized["identity_design_sha256"],
        "standardization_contract": EXPECTED_CONTRACT,
        "cases": cases,
        "high_fidelity_clone_claim": "NOT_ESTABLISHED",
        "scientific_boundary": (
            "This manifest proves only input identity and fair media standardization. The blind "
            "package, independent reviews and analysis have not run."
        ),
    }


def run_materialization(args: argparse.Namespace, report: Mapping[str, Any]) -> dict[str, Any]:
    jobs_path = Path(str(report["jobs_path"]))
    output_root = args.output_root.resolve()
    ffmpeg = Path(str(report["ffmpeg"]))
    ffprobe = Path(str(report["ffprobe"]))
    lock_path = output_root / "standardization.lock"
    with ExecutionLock(lock_path):
        payload = json.loads(jobs_path.read_text(encoding="utf-8"))
        normalized = normalize_jobs(payload, jobs_path=jobs_path, output_root=output_root)
        if normalized["jobs_sha256"] != report["jobs_sha256"]:
            raise JobsContractError("jobs changed after preflight")
        state = _load_state(output_root, normalized["jobs_sha256"])
        if state is None:
            state = _new_state(normalized)
        pending, conflicts = _audit_outputs(normalized, state, ffprobe)
        if conflicts:
            raise OutputConflictError("; ".join(conflicts))
        selected = pending[: args.max_files]
        processed = []
        for relative in selected:
            contract = normalized["files"][relative]
            target = Path(contract["path"])
            output_sha256, media = _encode_one(
                contract, target=target, ffmpeg=ffmpeg, ffprobe=ffprobe
            )
            state["files"][relative] = {
                "input_path": contract["input_path"],
                "input_sha256": contract["input_sha256"],
                "output_path": str(target),
                "output_sha256": output_sha256,
                "media_contract": media,
                "completed_at": _utc_now(),
            }
            _save_state(output_root, state)
            processed.append(relative)

        remaining, conflicts = _audit_outputs(normalized, state, ffprobe)
        if conflicts:
            raise OutputConflictError("; ".join(conflicts))
        manifest_path = output_root / "abx_manifest.json"
        if not remaining:
            manifest = _build_manifest(normalized, state, ffprobe)
            _write_atomic(manifest_path, manifest)
            manifest_sha256 = _sha256_file(manifest_path)
            state["manifest"] = {
                "path": str(manifest_path),
                "sha256": manifest_sha256,
                "case_count": len(manifest["cases"]),
                "written_at": _utc_now(),
            }
            _save_state(output_root, state)
            status = "COMPLETE"
        else:
            if manifest_path.exists():
                raise OutputConflictError("partial state must not retain an ABX manifest")
            manifest_sha256 = None
            status = "PARTIAL_PROGRESS"
    return {
        "schema": RESULT_SCHEMA,
        "status": status,
        "materialization_started": True,
        "processed_files": processed,
        "processed_file_count": len(processed),
        "remaining_file_count": len(remaining),
        "manifest_path": str(manifest_path) if not remaining else None,
        "manifest_sha256": manifest_sha256,
        "abx_case_count": normalized["job_count"] if not remaining else 0,
        "high_fidelity_clone_claim": "NOT_ESTABLISHED",
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jobs", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-files", type=int, default=DEFAULT_FILES_PER_INVOCATION)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument(
        "--confirm-run",
        action="store_true",
        help="Explicitly authorize the bounded serial standardization run.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_root.mkdir(parents=True, exist_ok=True)
    report = preflight(args)
    _write_atomic(args.output_root.resolve() / "latest_standardization_preflight.json", report)
    if not args.confirm_run:
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "ready_for_materialization": report["ready_for_materialization"],
                    "selected_files": report.get("selected_files", []),
                    "materialization_started": False,
                },
                ensure_ascii=False,
            )
        )
        return 0
    if report["status"] == "COMPLETE_ALREADY":
        print(json.dumps({"status": "COMPLETE_ALREADY", "materialization_started": False}))
        return 0
    if not report["ready_for_materialization"]:
        print(
            json.dumps(
                {"status": report["status"], "materialization_started": False},
                ensure_ascii=False,
            )
        )
        return 3
    try:
        result = run_materialization(args, report)
    except (
        InputIdentityError,
        JobsContractError,
        OutputConflictError,
        StateContractError,
        FileNotFoundError,
        OSError,
        RuntimeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        result = {
            "schema": RESULT_SCHEMA,
            "status": "ABSTAIN_RUNTIME_FAILURE",
            "materialization_started": True,
            "error": {"type": type(error).__name__, "message": str(error)},
            "high_fidelity_clone_claim": "NOT_ESTABLISHED",
        }
        _write_atomic(args.output_root.resolve() / "latest_standardization_result.json", result)
        print(json.dumps(result, ensure_ascii=False))
        return 2
    _write_atomic(args.output_root.resolve() / "latest_standardization_result.json", result)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
