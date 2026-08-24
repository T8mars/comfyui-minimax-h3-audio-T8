#!/usr/bin/env python3
"""Run a bounded, resumable subset of a pre-registered formal speech matrix.

The default invocation is preflight-only. A real run requires ``--confirm-run``, a loopback-only
private port other than 8188, the reviewed immutable plan and prompt hashes, all referenced models
and input audio, a free target port, and the requested free-VRAM headroom. At most six cases may be
requested per invocation and the default is one. Prompts run strictly serially and already collected
cases are skipped.

The tool starts and owns an isolated ComfyUI process with private user/temp/database directories. It
never queues to, interrupts, unloads, or terminates the user's normal service on port 8188. Each
reviewed prompt already requests ``unload_all_models`` after generation. Execution state is written
atomically after every attempt. The companion collector emits a plan-kind-specific generation
manifest only after every planned audio output is uniquely present and decodable; identity plans
still require their separate A/B/X standardization and blind-package steps.
"""

from __future__ import annotations

import argparse
import asyncio
from contextlib import suppress
from datetime import datetime, timezone
import hashlib
import ipaddress
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import tempfile
import time
from typing import Any, Mapping
import uuid

import build_speech_multilingual_formal_matrix as matrix
import run_nfe_resume_real_probe as shared


SCHEMA = "t8.minimax_h3.speech_multilingual_formal_batch.v1"
STATE_SCHEMA = "t8.minimax_h3.speech_multilingual_formal_execution_state.v1"
MAX_CASES_PER_INVOCATION = 6
MODEL_INPUTS = {
    "UNETLoader": ("unet_name", "diffusion_models"),
    "CLIPLoader": ("clip_name", "text_encoders"),
    "VAELoader": ("vae_name", "vae"),
}


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


def _safe_relative(value: Any, context: str) -> PurePosixPath:
    normalized = str(value or "").replace("\\", "/").strip()
    path = PurePosixPath(normalized)
    if (
        not normalized
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{context} must be a safe relative path")
    return path


def _path_below(base: Path, relative: PurePosixPath, context: str) -> Path:
    root = base.resolve()
    path = root.joinpath(*relative.parts).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{context} escapes its root") from error
    return path


def _loopback_host(value: str) -> bool:
    token = str(value).strip().casefold()
    if token == "localhost":
        return True
    with suppress(ValueError):
        return ipaddress.ip_address(token).is_loopback
    return False


def _find_one(prompt: Mapping[str, Any], class_type: str) -> Mapping[str, Any]:
    nodes = [
        node
        for node in prompt.values()
        if isinstance(node, dict) and node.get("class_type") == class_type
    ]
    if len(nodes) != 1:
        raise ValueError(f"prompt requires exactly one {class_type}; found {len(nodes)}")
    return nodes[0]


def load_plan(plan_root: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]], str]:
    root = plan_root.resolve()
    plan_path = root / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("schema") != matrix.SCHEMA:
        raise ValueError(f"plan schema must be {matrix.SCHEMA}")
    cases = plan.get("cases")
    if not isinstance(cases, list) or not cases or plan.get("case_count") != len(cases):
        raise ValueError("plan case_count must match a non-empty cases list")
    plan_sha256 = _sha256_file(plan_path)
    prompts: dict[str, dict[str, Any]] = {}
    seen_case_ids: set[str] = set()
    seen_prefixes: set[str] = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("every plan case must be an object")
        case_id = str(case.get("case_id") or "")
        if not case_id or case_id.casefold() in seen_case_ids:
            raise ValueError("plan case IDs must be non-empty and unique")
        seen_case_ids.add(case_id.casefold())
        prompt_relative = _safe_relative(case.get("prompt_path"), f"{case_id} prompt_path")
        prompt_path = _path_below(root, prompt_relative, f"{case_id} prompt_path")
        prompt = json.loads(prompt_path.read_text(encoding="utf-8"))
        prompt_sha256 = hashlib.sha256(matrix._json_bytes(prompt)).hexdigest().upper()
        if prompt_sha256 != str(case.get("prompt_sha256") or "").upper():
            raise ValueError(f"prompt SHA-256 drift for {case_id}")
        output_prefix = str(_safe_relative(case.get("output_prefix"), f"{case_id} output_prefix"))
        if output_prefix.casefold() in seen_prefixes:
            raise ValueError(f"duplicate output_prefix for {case_id}")
        seen_prefixes.add(output_prefix.casefold())
        save = _find_one(prompt, "SaveAudio")
        if str(save.get("inputs", {}).get("filename_prefix") or "") != output_prefix:
            raise ValueError(f"SaveAudio prefix differs from plan for {case_id}")
        studio = _find_one(prompt, "MiniMaxH3SpeechStudioT8")
        if studio.get("inputs", {}).get("release_policy") != "unload_all_models":
            raise ValueError(f"{case_id} must request unload_all_models")
        prompts[case_id] = prompt

    source_files = plan.get("source_files")
    if not isinstance(source_files, dict) or not source_files:
        raise ValueError("plan requires source_files identity")
    for name, contract in source_files.items():
        if not isinstance(contract, dict):
            raise ValueError(f"invalid source contract: {name}")
        path = Path(str(contract.get("path") or "")).resolve()
        if not path.is_file() or _sha256_file(path) != str(contract.get("sha256") or "").upper():
            raise ValueError(f"source file identity drift: {name}")
    return plan, prompts, plan_sha256


def _asset_paths(
    *,
    comfy_root: Path,
    prompts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Path]:
    result: dict[str, Path] = {}
    models_root = comfy_root.resolve() / "models"
    input_root = comfy_root.resolve() / "input"
    for prompt in prompts.values():
        for node in prompt.values():
            if not isinstance(node, dict):
                continue
            class_type = str(node.get("class_type") or "")
            inputs = node.get("inputs") if isinstance(node.get("inputs"), dict) else {}
            if class_type in MODEL_INPUTS:
                input_key, folder = MODEL_INPUTS[class_type]
                relative = _safe_relative(inputs.get(input_key), f"{class_type}.{input_key}")
                path = _path_below(models_root / folder, relative, f"{class_type}.{input_key}")
                result[f"model:{folder}/{relative}"] = path
            elif class_type == "LoadAudio":
                relative = _safe_relative(inputs.get("audio"), "LoadAudio.audio")
                path = _path_below(input_root, relative, "LoadAudio.audio")
                result[f"input:{relative}"] = path
    return result


def _verify_reference_hashes(plan: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    checked: dict[str, str] = {}
    for case in plan["cases"]:
        reference = case.get("reference_audio")
        if not isinstance(reference, dict):
            continue
        path = Path(str(reference.get("path") or "")).resolve()
        expected = str(reference.get("sha256") or "").upper()
        key = str(path).casefold()
        if key in checked:
            if checked[key] != expected:
                errors.append(f"conflicting reference hash in plan: {path}")
            continue
        checked[key] = expected
        if not path.is_file() or _sha256_file(path) != expected:
            errors.append(f"reference audio identity drift: {path}")
    return errors


def _state_path(plan_root: Path) -> Path:
    return plan_root.resolve() / "execution_state.json"


def _load_state(plan_root: Path, plan_sha256: str) -> dict[str, Any]:
    path = _state_path(plan_root)
    if not path.exists():
        now = _utc_now()
        return {
            "schema": STATE_SCHEMA,
            "plan_sha256": plan_sha256,
            "created_at": now,
            "updated_at": now,
            "attempts": {},
            "sessions": [],
        }
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("schema") != STATE_SCHEMA:
        raise ValueError(f"execution state schema must be {STATE_SCHEMA}")
    if str(state.get("plan_sha256") or "").upper() != plan_sha256:
        raise ValueError("execution state belongs to a different plan")
    if not isinstance(state.get("attempts"), dict) or not isinstance(state.get("sessions"), list):
        raise ValueError("execution state attempts/sessions are invalid")
    return state


def _save_state(plan_root: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = _utc_now()
    _write_atomic(_state_path(plan_root), state)


def sync_collection(
    *,
    plan_root: Path,
    plan: Mapping[str, Any],
    output_root: Path,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    report, manifest = matrix.collect_outputs(plan, output_root.resolve())
    _write_atomic(plan_root.resolve() / "collection_report.json", report)
    manifest_path = _collection_manifest_path(plan_root, plan)
    if manifest is None:
        if manifest_path.exists():
            manifest_path.unlink()
    else:
        _write_atomic(manifest_path, manifest)
    return report, manifest


def _collection_manifest_path(plan_root: Path, plan: Mapping[str, Any]) -> Path:
    name = (
        "identity_generation_manifest.raw.json"
        if plan.get("plan_kind") == "voice_clone_identity_formal"
        else "multilingual_manifest.json"
    )
    return plan_root.resolve() / name


def _row_map(collection: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(row.get("case_id")): row
        for row in collection.get("rows", [])
        if isinstance(row, dict)
    }


def _filter_pending(
    *,
    plan: Mapping[str, Any],
    collection: Mapping[str, Any],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    rows = _row_map(collection)
    pending = []
    for case in plan["cases"]:
        row = rows.get(str(case["case_id"]), {})
        if row.get("status") != "PENDING_MISSING_OUTPUT":
            continue
        if args.case_id and str(case["case_id"]) != args.case_id:
            continue
        if args.language and str(case.get("language_code")) != args.language:
            continue
        if args.mode and str(case.get("generation_mode")) != args.mode:
            continue
        pending.append(case)
    return pending[: args.max_cases]


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    created_at = _utc_now()
    base: dict[str, Any] = {
        "schema": f"{SCHEMA}.preflight",
        "created_at": created_at,
        "ready_for_real_run": False,
        "real_run_started": False,
        "user_service_8188_observed_only": shared.port_is_listening("127.0.0.1", 8188),
        "boundary": (
            "Preflight performs no model load and never contacts port 8188. A confirmed run starts "
            "only a private loopback ComfyUI process owned by this tool."
        ),
    }
    if (
        not _loopback_host(args.host)
        or args.port == 8188
        or not 1 <= int(args.port) <= 65535
        or not 1 <= int(args.max_cases) <= MAX_CASES_PER_INVOCATION
        or args.min_free_vram_mib < 1
        or args.timeout_seconds <= 0
    ):
        return {**base, "status": "ABSTAIN_INVALID_CONFIGURATION"}
    try:
        plan, prompts, plan_sha256 = load_plan(args.plan_root)
        state = _load_state(args.plan_root, plan_sha256)
        assets = _asset_paths(comfy_root=args.comfy_root, prompts=prompts)
        reference_errors = _verify_reference_hashes(plan)
        output_root = args.plan_root.resolve() / "comfy_output"
        collection, manifest = sync_collection(
            plan_root=args.plan_root,
            plan=plan,
            output_root=output_root,
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        return {
            **base,
            "status": "ABSTAIN_PLAN_OR_STATE_INVALID",
            "error": {"type": type(error).__name__, "message": str(error)},
        }
    output_conflicts = {
        key: value
        for key, value in collection.get("status_counts", {}).items()
        if str(key).startswith("ABSTAIN") and int(value) > 0
    }
    required = {
        "comfy_main": args.comfy_root.resolve() / "main.py",
        "python": args.python.resolve(),
        "t8_nodes": args.comfy_root.resolve() / "custom_nodes" / "minimax-h3-audio-T8",
        **assets,
    }
    missing = [str(path) for path in required.values() if not path.is_file() and not path.is_dir()]
    target_busy = shared.port_is_listening(args.host, args.port)
    gpu = shared.gpu_memory_mib()
    pending = _filter_pending(plan=plan, collection=collection, args=args)
    lock_path = args.plan_root.resolve() / "execution.lock"
    checks = {
        "plan_and_prompt_hashes_match": True,
        "execution_state_matches_plan": state["plan_sha256"] == plan_sha256,
        "required_files_present": not missing,
        "reference_audio_hashes_match": not reference_errors,
        "target_port_free": not target_busy,
        "execution_lock_free": not lock_path.exists(),
        "gpu_query_available": bool(gpu.get("available")),
        "free_vram_gate": bool(
            gpu.get("available")
            and int(gpu.get("free_mib", 0)) >= args.min_free_vram_mib
        ),
        "no_output_conflicts": not output_conflicts,
        "pending_cases_selected": bool(pending),
    }
    if manifest is not None:
        status = "COMPLETE_ALREADY"
    elif output_conflicts:
        status = "ABSTAIN_OUTPUT_CONFLICT"
    elif missing:
        status = "ABSTAIN_MISSING_DEPENDENCY"
    elif reference_errors:
        status = "ABSTAIN_REFERENCE_IDENTITY_DRIFT"
    elif target_busy:
        status = "ABSTAIN_TARGET_PORT_BUSY"
    elif lock_path.exists():
        status = "ABSTAIN_EXECUTION_LOCK_PRESENT"
    elif not gpu.get("available"):
        status = "ABSTAIN_GPU_STATE_UNKNOWN"
    elif int(gpu.get("free_mib", 0)) < args.min_free_vram_mib:
        status = "ABSTAIN_INSUFFICIENT_FREE_VRAM"
    elif not pending:
        status = "ABSTAIN_NO_MATCHING_PENDING_CASES"
    else:
        status = "READY"
    return {
        **base,
        "status": status,
        "ready_for_real_run": status == "READY" and all(checks.values()),
        "plan_sha256": plan_sha256,
        "planned_case_count": int(plan["case_count"]),
        "collected_unique_case_count": int(collection["collected_unique_case_count"]),
        "selected_case_ids": [str(case["case_id"]) for case in pending],
        "checks": checks,
        "missing_paths": missing,
        "reference_errors": reference_errors,
        "output_conflicts": output_conflicts,
        "gpu": gpu,
        "minimum_free_vram_mib": args.min_free_vram_mib,
        "target": {"host": args.host, "port": args.port, "already_listening": target_busy},
        "output_root": str(output_root),
    }


class ExecutionLock:
    def __init__(self, path: Path, session_id: str):
        self.path = path
        self.session_id = session_id
        self.acquired = False

    def __enter__(self):
        payload = _json_bytes(
            {
                "schema": f"{SCHEMA}.lock",
                "pid": os.getpid(),
                "session_id": self.session_id,
                "created_at": _utc_now(),
            }
        )
        try:
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as error:
            raise RuntimeError(f"execution lock already exists: {self.path}") from error
        try:
            os.write(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        self.acquired = True
        return self

    def __exit__(self, _exc_type, _exc, _tb):
        if self.acquired:
            with suppress(FileNotFoundError):
                self.path.unlink()


def _server_command(args: argparse.Namespace, session_root: Path, output_root: Path) -> list[str]:
    return [
        str(args.python.resolve()),
        "main.py",
        "--listen",
        args.host,
        "--port",
        str(args.port),
        "--disable-auto-launch",
        "--preview-method",
        "none",
        "--cache-none",
        "--reserve-vram",
        "1.0",
        "--disable-all-custom-nodes",
        "--whitelist-custom-nodes",
        "minimax-h3-audio-T8",
        "--input-directory",
        str((args.comfy_root.resolve() / "input")),
        "--output-directory",
        str(output_root.resolve()),
        "--temp-directory",
        str((session_root / "temp").resolve()),
        "--user-directory",
        str((session_root / "user").resolve()),
        "--database-url",
        "sqlite:///:memory:",
    ]


class IsolatedSpeechServer:
    def __init__(self, args: argparse.Namespace, session_root: Path, output_root: Path):
        self.args = args
        self.session_root = session_root
        self.output_root = output_root
        self.process: subprocess.Popen[str] | None = None
        self.stdout_handle = None
        self.stderr_handle = None

    def start(self) -> int:
        if shared.port_is_listening(self.args.host, self.args.port):
            raise RuntimeError(f"refusing to start: target port {self.args.port} is in use")
        for name in ("temp", "user", "logs"):
            (self.session_root / name).mkdir(parents=True, exist_ok=True)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.stdout_handle = (self.session_root / "logs" / "server.stdout.log").open(
            "w", encoding="utf-8"
        )
        self.stderr_handle = (self.session_root / "logs" / "server.stderr.log").open(
            "w", encoding="utf-8"
        )
        creationflags = 0
        if os.name == "nt":
            creationflags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
        self.process = subprocess.Popen(
            _server_command(self.args, self.session_root, self.output_root),
            cwd=self.args.comfy_root.resolve(),
            stdout=self.stdout_handle,
            stderr=self.stderr_handle,
            text=True,
            creationflags=creationflags,
        )
        deadline = time.monotonic() + self.args.server_start_timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"isolated speech ComfyUI exited with {self.process.returncode}; "
                    f"inspect {self.session_root / 'logs'}"
                )
            if shared.port_is_listening(self.args.host, self.args.port):
                return int(self.process.pid)
            time.sleep(0.5)
        raise TimeoutError("isolated speech ComfyUI did not listen in time")

    def stop(self) -> None:
        process = self.process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=15)
        deadline = time.monotonic() + 30
        while shared.port_is_listening(self.args.host, self.args.port) and time.monotonic() < deadline:
            time.sleep(0.25)
        for handle in (self.stdout_handle, self.stderr_handle):
            if handle is not None:
                handle.close()


def _attempt_entry(state: dict[str, Any], case_id: str, session_id: str) -> dict[str, Any]:
    attempts = state["attempts"].setdefault(case_id, [])
    entry = {
        "attempt": len(attempts) + 1,
        "session_id": session_id,
        "started_at": _utc_now(),
        "finished_at": None,
        "status": "SUBMITTING",
        "prompt_id": None,
        "terminal": None,
        "elapsed_seconds": None,
        "error": None,
    }
    attempts.append(entry)
    return entry


def run_batch(args: argparse.Namespace, preflight_report: Mapping[str, Any]) -> dict[str, Any]:
    plan, prompts, plan_sha256 = load_plan(args.plan_root)
    state = _load_state(args.plan_root, plan_sha256)
    output_root = args.plan_root.resolve() / "comfy_output"
    collection, _ = sync_collection(plan_root=args.plan_root, plan=plan, output_root=output_root)
    selected = _filter_pending(plan=plan, collection=collection, args=args)
    selected_ids = [str(case["case_id"]) for case in selected]
    if selected_ids != list(preflight_report.get("selected_case_ids", [])):
        return {
            "schema": SCHEMA,
            "created_at": _utc_now(),
            "status": "ABSTAIN_SELECTION_CHANGED_AFTER_PREFLIGHT",
            "passed": False,
            "real_run_started": False,
            "selected_case_ids": selected_ids,
            "process_ids": [],
        }
    session_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    session_root = args.plan_root.resolve() / "executor_runtime" / session_id
    session_root.mkdir(parents=True, exist_ok=False)
    report_path = session_root / "execution_report.json"
    lock_path = args.plan_root.resolve() / "execution.lock"
    process_ids: list[int] = []
    results: list[dict[str, Any]] = []
    runtime_error: dict[str, str] | None = None
    started = time.monotonic()
    baseline_gpu = shared.gpu_memory_mib()
    server = IsolatedSpeechServer(args, session_root, output_root)
    try:
        with ExecutionLock(lock_path, session_id):
            resource_now = shared.gpu_memory_mib()
            if (
                shared.port_is_listening(args.host, args.port)
                or not resource_now.get("available")
                or int(resource_now.get("free_mib", 0)) < args.min_free_vram_mib
            ):
                report = {
                    "schema": SCHEMA,
                    "created_at": _utc_now(),
                    "session_id": session_id,
                    "status": "ABSTAIN_RESOURCE_CHANGED_BEFORE_START",
                    "passed": False,
                    "real_run_started": False,
                    "selected_case_ids": selected_ids,
                    "process_ids": [],
                    "resource_recheck": resource_now,
                }
                _write_atomic(report_path, report)
                return report
            process_ids.append(server.start())
            for case in selected:
                case_id = str(case["case_id"])
                entry = _attempt_entry(state, case_id, session_id)
                _save_state(args.plan_root, state)
                case_result: dict[str, Any] = {"case_id": case_id}
                try:
                    phase = asyncio.run(
                        shared.submit_prompt(
                            server=f"http://{args.host}:{args.port}",
                            prompt=prompts[case_id],
                            timeout_seconds=args.timeout_seconds,
                        )
                    )
                    terminal = str((phase.get("terminal") or {}).get("type") or "")
                    entry.update(
                        {
                            "finished_at": _utc_now(),
                            "status": "EXECUTION_SUCCESS" if terminal == "execution_success" else "EXECUTION_FAILED",
                            "prompt_id": phase.get("prompt_id"),
                            "terminal": terminal,
                            "elapsed_seconds": phase.get("elapsed_seconds"),
                        }
                    )
                    collection, _ = sync_collection(
                        plan_root=args.plan_root,
                        plan=plan,
                        output_root=output_root,
                    )
                    row = _row_map(collection).get(case_id, {})
                    case_result.update(
                        {
                            "prompt_id": phase.get("prompt_id"),
                            "terminal": terminal,
                            "elapsed_seconds": phase.get("elapsed_seconds"),
                            "collection_status": row.get("status"),
                        }
                    )
                    if terminal != "execution_success" or row.get("status") != "COLLECTED_UNEVALUATED":
                        entry["status"] = "FAILED_TERMINAL_OR_OUTPUT_CONTRACT"
                        results.append(case_result)
                        _save_state(args.plan_root, state)
                        break
                    entry["status"] = "COLLECTED_UNEVALUATED"
                    results.append(case_result)
                    _save_state(args.plan_root, state)
                except Exception as error:
                    runtime_error = {"type": type(error).__name__, "message": str(error)}
                    entry.update(
                        {
                            "finished_at": _utc_now(),
                            "status": "EXECUTION_EXCEPTION",
                            "error": runtime_error,
                        }
                    )
                    case_result["error"] = runtime_error
                    results.append(case_result)
                    _save_state(args.plan_root, state)
                    break
    except Exception as error:
        runtime_error = {"type": type(error).__name__, "message": str(error)}
    finally:
        server.stop()

    final_collection, manifest = sync_collection(
        plan_root=args.plan_root,
        plan=plan,
        output_root=output_root,
    )
    successful = sum(
        result.get("collection_status") == "COLLECTED_UNEVALUATED"
        and result.get("terminal") == "execution_success"
        for result in results
    )
    selected_all_passed = successful == len(selected_ids) and runtime_error is None
    if not selected_all_passed:
        status = "FAIL_EXECUTION_OR_OUTPUT_CONTRACT"
    elif manifest is not None:
        status = "PASS_COMPLETE_COLLECTION_PENDING_EVALUATION"
    else:
        status = "PASS_PARTIAL_COLLECTION_PENDING_MORE_CASES"
    session_summary = {
        "session_id": session_id,
        "created_at": _utc_now(),
        "status": status,
        "selected_case_ids": selected_ids,
        "successful_case_count": successful,
        "report": str(report_path),
    }
    state["sessions"].append(session_summary)
    _save_state(args.plan_root, state)
    report = {
        "schema": SCHEMA,
        "created_at": _utc_now(),
        "session_id": session_id,
        "status": status,
        "passed": selected_all_passed,
        "real_run_started": bool(process_ids),
        "preflight": dict(preflight_report),
        "plan_sha256": plan_sha256,
        "selected_case_ids": selected_ids,
        "results": results,
        "runtime_error": runtime_error,
        "process_ids": process_ids,
        "collection": {
            "planned_case_count": final_collection["planned_case_count"],
            "collected_unique_case_count": final_collection["collected_unique_case_count"],
            "status_counts": final_collection["status_counts"],
            "manifest_written": manifest is not None,
            "evaluation_executed": False,
            "stable_multilingual_gate_pass": False,
        },
        "gpu": {"baseline": baseline_gpu, "final": shared.gpu_memory_mib()},
        "elapsed_seconds": round(time.monotonic() - started, 4),
        "boundary": (
            "A passing batch proves only bounded serial generation and unique decodable audio "
            "collection for the selected cases. ASR, human transcript review, identity, naturalness, "
            "acting, clone fidelity and the stable multilingual gate remain separate and false."
        ),
    }
    _write_atomic(report_path, report)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan-root",
        type=Path,
        default=project_root / "artifacts" / "speech-multilingual-formal-en-zh-v1",
    )
    parser.add_argument(
        "--comfy-root", type=Path, default=Path(r"F:\AI-T8-video-onekey\ComfyUI")
    )
    parser.add_argument(
        "--python", type=Path, default=Path(r"F:\AI-T8-video-onekey\python\python.exe")
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8197)
    parser.add_argument("--min-free-vram-mib", type=int, default=12000)
    parser.add_argument("--max-cases", type=int, default=1)
    parser.add_argument("--case-id", default="")
    parser.add_argument("--language", choices=("", "en", "zh"), default="")
    parser.add_argument("--mode", choices=("", "described", "clone"), default="")
    parser.add_argument("--server-start-timeout", type=float, default=180.0)
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    parser.add_argument(
        "--confirm-run",
        action="store_true",
        help="Run the selected bounded cases only after every preflight gate passes.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.plan_root.mkdir(parents=True, exist_ok=True)
    report = preflight(args)
    preflight_path = args.plan_root.resolve() / "latest_execution_preflight.json"
    _write_atomic(preflight_path, report)
    if not args.confirm_run:
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "ready_for_real_run": report["ready_for_real_run"],
                    "selected_case_ids": report.get("selected_case_ids", []),
                    "preflight": str(preflight_path),
                    "real_run_started": False,
                },
                ensure_ascii=False,
            )
        )
        return 0
    if report["status"] == "COMPLETE_ALREADY":
        print(
            json.dumps(
                {"status": "COMPLETE_ALREADY", "real_run_started": False}, ensure_ascii=False
            )
        )
        return 0
    if not report["ready_for_real_run"]:
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "preflight": str(preflight_path),
                    "real_run_started": False,
                },
                ensure_ascii=False,
            )
        )
        return 3
    result = run_batch(args, report)
    print(
        json.dumps(
            {
                "status": result["status"],
                "passed": result["passed"],
                "real_run_started": result["real_run_started"],
                "selected_case_ids": result.get("selected_case_ids", []),
                "collected_unique_case_count": result.get("collection", {}).get(
                    "collected_unique_case_count"
                ),
            },
            ensure_ascii=False,
        )
    )
    if str(result["status"]).startswith("ABSTAIN"):
        return 3
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
