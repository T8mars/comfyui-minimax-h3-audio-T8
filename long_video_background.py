from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import asyncio
import hashlib
import json
import os
import threading
import time
from typing import Callable, Protocol
import uuid

from .long_video import sanitize_chain_id
from .long_video_delivery import (
    _open_advisory_lock,
    _release_advisory_lock,
    _try_advisory_lock,
    atomic_write_long_video_json,
    load_delivery_manifest,
    long_video_chain_root,
)


BACKGROUND_SCHEMA = 2
LEGACY_BACKGROUND_SCHEMA = 1
BACKGROUND_STATE_FORMAT = "minimax_h3_t8_background_job"
BACKGROUND_STATE_NAME = "background_job.json"
BACKGROUND_LEASE_NAME = "background_job.lock.v2"
BACKGROUND_LEASE_KIND = "t8_background_process_lease_v2"
EXECUTION_MODES = ("review_only", "auto_accept_and_continue")
RELEASE_POLICIES = ("keep_loaded", "clear_execution_cache", "unload_all_models")
ACTIVE_STATES = {"running", "pausing", "cancelling", "scheduling", "retry_wait"}

# Background orchestration is shared by long-video delivery and source-bound Studio jobs.
# Providers only replace the durable-progress lookup; queueing, retry, cancellation, process
# leasing, and atomic auxiliary state remain one implementation.
_BACKGROUND_PROGRESS_PROVIDERS: list[
    tuple[str, Callable[[str], tuple[int, bool]]]
] = []


def register_background_progress_provider(
    chain_prefix: str,
    provider: Callable[[str], tuple[int, bool]],
) -> None:
    prefix = str(chain_prefix)
    if not prefix or not callable(provider):
        raise ValueError("A non-empty background chain prefix and callable provider are required")
    for existing_prefix, existing_provider in _BACKGROUND_PROGRESS_PROVIDERS:
        if existing_prefix == prefix:
            if existing_provider is not provider:
                raise ValueError(f"Background progress prefix {prefix!r} is already registered")
            return
    _BACKGROUND_PROGRESS_PROVIDERS.append((prefix, provider))


class BackgroundJobError(RuntimeError):
    pass


class BackgroundStateUnreadableError(BackgroundJobError):
    pass


class UnsupportedBackgroundSchemaError(BackgroundJobError):
    pass


def _normalize_binding_metadata(value: Mapping | None) -> dict:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise BackgroundJobError("binding_metadata must be a mapping")
    try:
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        normalized = json.loads(encoded)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise BackgroundJobError("binding_metadata must contain JSON-safe metadata") from error
    if not isinstance(normalized, dict):
        raise BackgroundJobError("binding_metadata must be a JSON object")
    if len(encoded.encode("utf-8")) > 4096:
        raise BackgroundJobError("binding_metadata exceeds the 4KiB limit")
    return normalized


class _BackgroundProcessLease:
    def __init__(self, chain_id: str):
        self.chain_id = sanitize_chain_id(chain_id)
        self.path = long_video_chain_root(self.chain_id) / BACKGROUND_LEASE_NAME
        self.handle = None
        self.token = uuid.uuid4().hex

    def acquire(self) -> None:
        handle = _open_advisory_lock(self.path)
        if not _try_advisory_lock(handle):
            handle.close()
            raise BackgroundJobError(
                f"Chain '{self.chain_id}' is owned by another ComfyUI process. "
                "Wait for it to finish, cancel it in that process, or stop that process before "
                "reattaching the workflow."
            )
        self.handle = handle
        self.bind("")

    def bind(self, job_id: str) -> None:
        if self.handle is None:
            raise RuntimeError("Background process lease is not acquired")
        payload = json.dumps(
            {
                "lock_kind": BACKGROUND_LEASE_KIND,
                "chain_id": self.chain_id,
                "job_id": str(job_id),
                "pid": os.getpid(),
                "token": self.token,
                "acquired_unix": time.time(),
            },
            ensure_ascii=False,
        ).encode("utf-8")
        self.handle.seek(0)
        self.handle.write(payload)
        self.handle.truncate(len(payload))
        self.handle.flush()
        os.fsync(self.handle.fileno())

    def release(self) -> None:
        handle, self.handle = self.handle, None
        if handle is None:
            return
        try:
            _release_advisory_lock(handle)
        finally:
            handle.close()


class QueueRuntime(Protocol):
    def current_prompt_id(self) -> str: ...

    def current_client_id(self) -> str | None: ...

    def queue_prompt(self, prompt: Mapping, client_id: str | None) -> str: ...

    def prompt_location(self, prompt_id: str) -> str: ...

    def history_record(self, prompt_id: str) -> dict | None: ...

    def cancel_prompt(self, prompt_id: str) -> dict: ...

    def request_release(self, release_policy: str) -> None: ...


class ComfyQueueRuntime:
    """Late-bound adapter around the current ComfyUI queue contract."""

    @staticmethod
    def _server():
        from server import PromptServer

        instance = getattr(PromptServer, "instance", None)
        if instance is None:
            raise BackgroundJobError("ComfyUI PromptServer is not ready")
        return instance

    def current_prompt_id(self) -> str:
        from comfy_execution.utils import get_executing_context

        context = get_executing_context()
        if context is None or not context.prompt_id:
            raise BackgroundJobError(
                "The background controller must execute inside a ComfyUI prompt"
            )
        return str(context.prompt_id)

    def current_client_id(self) -> str | None:
        value = getattr(self._server(), "client_id", None)
        return str(value) if value else None

    def queue_prompt(self, prompt: Mapping, client_id: str | None) -> str:
        server = self._server()
        original_prompt = _clean_prompt_snapshot(prompt)

        async def enqueue() -> str:
            import execution

            prompt_id = str(uuid.uuid4())
            json_data = {"prompt": original_prompt}
            if client_id:
                json_data["client_id"] = client_id
            json_data = server.trigger_on_prompt(json_data)
            prompt_copy = deepcopy(dict(json_data["prompt"]))
            server.node_replace_manager.apply_replacements(prompt_copy)
            valid = await execution.validate_prompt(prompt_id, prompt_copy, None)
            if not valid[0]:
                details = json.dumps(
                    {"error": valid[1], "node_errors": valid[3]},
                    ensure_ascii=False,
                    default=str,
                )
                raise BackgroundJobError(
                    "The next long-video segment prompt failed validation: " + details
                )
            number = server.number
            server.number += 1
            extra_data = {"create_time": int(time.time() * 1000)}
            if client_id:
                extra_data["client_id"] = client_id
            server.prompt_queue.put(
                (number, prompt_id, prompt_copy, extra_data, valid[2], {})
            )
            return prompt_id

        future = asyncio.run_coroutine_threadsafe(enqueue(), server.loop)
        try:
            return future.result(timeout=60.0)
        except TimeoutError as error:
            future.cancel()
            raise BackgroundJobError(
                "Timed out while validating the next long-video segment prompt"
            ) from error

    def prompt_location(self, prompt_id: str) -> str:
        queue = self._server().prompt_queue
        running, queued = queue.get_current_queue()
        if any(str(item[1]) == prompt_id for item in running):
            return "running"
        if any(str(item[1]) == prompt_id for item in queued):
            return "queued"
        history = queue.get_history(prompt_id=prompt_id)
        if prompt_id in history:
            status = history[prompt_id].get("status") or {}
            return str(status.get("status_str") or "history")
        return "missing"

    def history_record(self, prompt_id: str) -> dict | None:
        value = self._server().prompt_queue.get_history(prompt_id=prompt_id)
        record = value.get(prompt_id)
        return dict(record) if isinstance(record, dict) else None

    def cancel_prompt(self, prompt_id: str) -> dict:
        queue = self._server().prompt_queue
        deleted = queue.delete_queue_item(lambda item: str(item[1]) == prompt_id)
        interrupted = False if deleted else queue.interrupt_if_running(prompt_id)
        return {
            "prompt_id": prompt_id,
            "deleted_from_queue": bool(deleted),
            "interrupt_signalled": bool(interrupted),
        }

    def request_release(self, release_policy: str) -> None:
        if release_policy not in RELEASE_POLICIES:
            raise BackgroundJobError(f"Unknown release policy: {release_policy}")
        if release_policy == "keep_loaded":
            return
        queue = self._server().prompt_queue
        # Set both flags explicitly. ComfyUI otherwise treats free_memory as an
        # implicit unload_models request when the latter key is absent.
        queue.set_flag("unload_models", release_policy == "unload_all_models")
        queue.set_flag("free_memory", True)


def _prompt_sha256(prompt: Mapping) -> str:
    encoded = json.dumps(
        prompt, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _clean_prompt_snapshot(prompt: Mapping) -> dict:
    """Remove execution-time cache annotations before a prompt is reused.

    Current ComfyUI mutates prompt nodes with an ``is_changed`` field while building cache
    signatures. Requeueing that mutated object freezes the old fingerprints, so keep_loaded can
    incorrectly cache the entire next segment. Only the API-authored class_type/inputs metadata
    belongs in a reusable snapshot.
    """
    cleaned = deepcopy(dict(prompt))
    for node in cleaned.values():
        if isinstance(node, dict):
            node.pop("is_changed", None)
    return cleaned


def _state_path(chain_id: str):
    return long_video_chain_root(chain_id) / BACKGROUND_STATE_NAME


def load_background_job_state(chain_id: str) -> dict | None:
    safe_chain = sanitize_chain_id(chain_id)
    path = _state_path(safe_chain)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BackgroundStateUnreadableError(
            f"Background job state is unreadable: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise BackgroundStateUnreadableError("Background job state root must be an object")
    try:
        state_schema = int(payload.get("schema", -1))
    except (TypeError, ValueError) as error:
        raise BackgroundStateUnreadableError("Background job state schema is invalid") from error
    if state_schema == LEGACY_BACKGROUND_SCHEMA:
        payload = dict(payload)
        payload.update(
            {
                "schema": BACKGROUND_SCHEMA,
                "format": BACKGROUND_STATE_FORMAT,
                "migrated_from_schema": LEGACY_BACKGROUND_SCHEMA,
            }
        )
    elif state_schema != BACKGROUND_SCHEMA:
        raise UnsupportedBackgroundSchemaError(
            f"Background job state schema {state_schema} is unsupported; expected "
            f"{BACKGROUND_SCHEMA} or migratable legacy schema {LEGACY_BACKGROUND_SCHEMA}. "
            "Use the plugin version that wrote this state."
        )
    if payload.get("format") != BACKGROUND_STATE_FORMAT:
        raise BackgroundStateUnreadableError("Background job state format marker is invalid")
    if payload.get("chain_id") != safe_chain:
        raise BackgroundStateUnreadableError(
            "Background job state chain_id does not match its folder"
        )
    return payload


def _quarantine_background_job_state(chain_id: str) -> str:
    source = _state_path(chain_id)
    target = source.with_name(
        f"background_job.corrupt.{time.time_ns()}.{uuid.uuid4().hex[:8]}.json"
    )
    try:
        os.replace(source, target)
    except FileNotFoundError:
        return ""
    return str(target)


def _history_error(record: Mapping) -> str:
    status = record.get("status") if isinstance(record, Mapping) else None
    messages = status.get("messages", []) if isinstance(status, Mapping) else []
    error_payloads = []
    for message in messages:
        if isinstance(message, (list, tuple)) and len(message) == 2:
            event, payload = message
            if event in {"execution_error", "execution_interrupted"}:
                error_payloads.append(payload)
    value = error_payloads[-1] if error_payloads else None
    if isinstance(value, Mapping):
        # ComfyUI execution_error includes current_inputs, which may contain complete prompts,
        # media tensors, tokens, or third-party objects. Persist only an audit-safe allowlist.
        safe = {
            key: value[key]
            for key in (
                "prompt_id",
                "node_id",
                "node_type",
                "exception_type",
                "exception_message",
            )
            if key in value
        }
        traceback = value.get("traceback")
        if isinstance(traceback, list):
            safe["traceback"] = [str(item)[:2000] for item in traceback[-12:]]
        value = safe
    elif value is None:
        value = {
            "status_str": status.get("status_str") if isinstance(status, Mapping) else "error",
            "message_count": len(messages),
        }
    text = json.dumps(value, ensure_ascii=False, default=str)
    return text[:12000]


class BackgroundJobManager:
    def __init__(
        self,
        runtime: QueueRuntime | None = None,
        *,
        start_monitors: bool = True,
        monitor_interval_seconds: float = 0.5,
    ):
        self.runtime = runtime or ComfyQueueRuntime()
        self.start_monitors = bool(start_monitors)
        self.monitor_interval_seconds = float(monitor_interval_seconds)
        self._lock = threading.RLock()
        self._snapshots: dict[str, tuple[dict, str | None]] = {}
        self._job_chains: dict[str, str] = {}
        self._monitors: set[tuple[str, str]] = set()
        self._chain_leases: dict[str, tuple[str, _BackgroundProcessLease]] = {}

    def close(self) -> None:
        with self._lock:
            leases = list(self._chain_leases.values())
            self._chain_leases.clear()
        for _job_id, lease in leases:
            lease.release()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _acquire_chain_lease(self, chain_id: str) -> bool:
        existing = self._chain_leases.get(chain_id)
        if existing is not None:
            return False
        lease = _BackgroundProcessLease(chain_id)
        lease.acquire()
        self._chain_leases[chain_id] = ("", lease)
        return True

    def _bind_chain_lease(self, chain_id: str, job_id: str) -> None:
        current = self._chain_leases.get(chain_id)
        if current is None:
            raise BackgroundJobError(f"Background process lease for '{chain_id}' was lost")
        bound_job_id, lease = current
        if bound_job_id and bound_job_id != job_id:
            raise BackgroundJobError(
                f"Background process lease for '{chain_id}' belongs to job {bound_job_id}"
            )
        lease.bind(job_id)
        self._chain_leases[chain_id] = (job_id, lease)

    def _release_chain_lease(self, chain_id: str, job_id: str = "") -> None:
        current = self._chain_leases.get(chain_id)
        if current is None:
            return
        bound_job_id, lease = current
        if job_id and bound_job_id and bound_job_id != job_id:
            return
        self._chain_leases.pop(chain_id, None)
        lease.release()

    def _write(self, state: dict) -> dict:
        state = dict(state)
        state["schema"] = BACKGROUND_SCHEMA
        state["format"] = BACKGROUND_STATE_FORMAT
        state["updated_unix"] = time.time()
        atomic_write_long_video_json(_state_path(state["chain_id"]), state)
        # Failed jobs are terminal too. Keeping their process lease/snapshot alive blocks the
        # documented one-requeue recovery path in the same ComfyUI process: attach_prompt creates
        # a fresh job, then _bind_chain_lease rejects it because the old failed job still owns the
        # chain. Release only after the failed state is durably written; the accepted manifest and
        # previous_job_id remain the recovery authority.
        if state.get("state") in {"completed", "failed", "cancelled", "detached"}:
            job_id = str(state.get("job_id") or "")
            self._release_chain_lease(str(state["chain_id"]), job_id)
            self._snapshots.pop(job_id, None)
        return state

    def _request_terminal_release(self, state: dict, reason: str) -> dict:
        """Best-effort release for terminal/control paths.

        Cancellation and failure must remain truthful even if ComfyUI rejects a release flag.
        Record that secondary failure instead of replacing the primary terminal state.
        """
        state = dict(state)
        release_policy = str(state.get("release_policy") or "clear_execution_cache")
        state["release_reason"] = str(reason)
        try:
            self.runtime.request_release(release_policy)
        except Exception as error:
            state["last_release_error"] = str(error)[:4000]
        else:
            state["last_release_policy"] = release_policy
            state["release_requested_unix"] = time.time()
            state.pop("last_release_error", None)
        return state

    @staticmethod
    def _manifest_position(chain_id: str) -> tuple[int, bool]:
        for prefix, provider in _BACKGROUND_PROGRESS_PROVIDERS:
            if str(chain_id).startswith(prefix):
                accepted_count, complete = provider(str(chain_id))
                return int(accepted_count), bool(complete)
        try:
            manifest, _ = load_delivery_manifest(chain_id)
        except FileNotFoundError:
            return 0, False
        segments = manifest["segments"]
        complete = bool(segments and segments[-1].get("is_final_segment"))
        return len(segments), complete

    def status(self, chain_id: str) -> dict:
        safe_chain = sanitize_chain_id(chain_id)
        with self._lock:
            accepted_count, manifest_complete = self._manifest_position(safe_chain)
            try:
                state = load_background_job_state(safe_chain)
            except BackgroundStateUnreadableError as error:
                return {
                    "schema": BACKGROUND_SCHEMA,
                    "chain_id": safe_chain,
                    "state": "detached",
                    "accepted_count": accepted_count,
                    "manifest_complete": manifest_complete,
                    "runtime_location": "none",
                    "resumable_in_memory": False,
                    "recovery_required": True,
                    "recovery_action": (
                        "compose_accepted" if manifest_complete else "queue_workflow_once"
                    ),
                    "state_file_unreadable": True,
                    "last_error": (
                        f"{error}. Queue the workflow once to quarantine the unreadable auxiliary "
                        "state and recover from the accepted manifest."
                    ),
                }
            if state is None:
                return {
                    "schema": BACKGROUND_SCHEMA,
                    "chain_id": safe_chain,
                    "state": "idle",
                    "accepted_count": accepted_count,
                    "manifest_complete": manifest_complete,
                    "resumable_in_memory": False,
                    "recovery_required": False,
                }
            job_id = str(state.get("job_id") or "")
            if job_id:
                self._job_chains[job_id] = safe_chain
            active_prompt_id = str(state.get("active_prompt_id") or "")
            runtime_location = (
                self.runtime.prompt_location(active_prompt_id) if active_prompt_id else "none"
            )
            resumable_in_memory = job_id in self._snapshots

            # Queue/history identifiers do not survive every ComfyUI process restart, and the
            # reusable prompt is intentionally memory-only. Reconcile a stale active record into
            # an explicit detached state instead of continuing to report a phantom running job.
            # The accepted manifest is authoritative: it may be ahead of background_job.json if
            # the process was killed just after the durable acceptance boundary.
            if (
                state.get("state") in ACTIVE_STATES
                and not resumable_in_memory
                and runtime_location not in {"queued", "running"}
            ):
                recovery_action = (
                    "compose_accepted" if manifest_complete else "queue_workflow_once"
                )
                recovery_message = (
                    "The final accepted manifest survived the process restart. Run Compose "
                    "Accepted if the final video is missing; do not generate another segment."
                    if manifest_complete
                    else (
                        "The previous background prompt is no longer present after process "
                        "restart. Queue the background workflow once to reattach it; generation "
                        "will resume from the accepted manifest."
                    )
                )
                state.update(
                    {
                        "state": "detached",
                        "active_prompt_id": "",
                        "accepted_count": accepted_count,
                        "current_segment_index": (
                            max(accepted_count - 1, 0)
                            if manifest_complete
                            else accepted_count
                        ),
                        "orphaned_prompt_id": active_prompt_id,
                        "orphaned_runtime_location": runtime_location,
                        "recovery_action": recovery_action,
                        "recovery_detected_unix": time.time(),
                        "last_error": recovery_message,
                    }
                )
                state = self._write(state)
                runtime_location = "none"

            result = dict(state)
            result["accepted_count"] = accepted_count
            result["manifest_complete"] = manifest_complete
            result["runtime_location"] = runtime_location
            result["resumable_in_memory"] = resumable_in_memory
            result["recovery_required"] = (
                result.get("state") == "detached" and not resumable_in_memory
            )
            if result["recovery_required"] and not result.get("recovery_action"):
                result["recovery_action"] = (
                    "compose_accepted" if manifest_complete else "queue_workflow_once"
                )
            return result

    def attach_prompt(
        self,
        chain_id: str,
        prompt: Mapping,
        controller_node_id: str,
        max_retries: int,
        retry_delay_seconds: float,
        release_policy: str,
        *,
        prompt_id: str | None = None,
        client_id: str | None = None,
        binding_metadata: Mapping | None = None,
    ) -> dict:
        safe_chain = sanitize_chain_id(chain_id)
        if not isinstance(prompt, Mapping) or not prompt:
            raise BackgroundJobError("ComfyUI did not provide a reusable prompt snapshot")
        max_retries = int(max_retries)
        retry_delay_seconds = float(retry_delay_seconds)
        if not 0 <= max_retries <= 10:
            raise BackgroundJobError("max_retries must be between 0 and 10")
        if not 0.0 <= retry_delay_seconds <= 300.0:
            raise BackgroundJobError("retry_delay_seconds must be between 0 and 300")
        if release_policy not in RELEASE_POLICIES:
            raise BackgroundJobError(f"Unknown release policy: {release_policy}")
        prompt_id = str(prompt_id or self.runtime.current_prompt_id())
        client_id = client_id if client_id is not None else self.runtime.current_client_id()
        prompt_copy = _clean_prompt_snapshot(prompt)
        binding_metadata = _normalize_binding_metadata(binding_metadata)

        with self._lock:
            newly_acquired_lease = self._acquire_chain_lease(safe_chain)
            try:
                quarantined_state_path = ""
                recovery_note = ""
                try:
                    existing = load_background_job_state(safe_chain)
                except BackgroundStateUnreadableError as error:
                    quarantined_state_path = _quarantine_background_job_state(safe_chain)
                    recovery_note = (
                        f"Recovered from unreadable auxiliary background state: {error}. "
                        "The accepted manifest remained authoritative."
                    )
                    existing = None
                if existing is not None:
                    existing_binding = _normalize_binding_metadata(
                        existing.get("binding_metadata")
                    )
                    if existing_binding != binding_metadata:
                        raise BackgroundJobError(
                            f"Chain '{safe_chain}' belongs to different binding metadata; "
                            "use a new chain_id instead of reusing another workspace job"
                        )
                if (
                    existing is not None
                    and str(existing.get("active_prompt_id") or "") == prompt_id
                ):
                    if existing.get("state") == "cancelled":
                        raise BackgroundJobError("This background job was cancelled")
                    job_id = str(existing["job_id"])
                    state = dict(existing)
                    state.update(
                        {
                            "state": "pausing" if state.get("pause_requested") else "running",
                            "controller_node_id": str(controller_node_id),
                            "max_retries": max_retries,
                            "retry_delay_seconds": retry_delay_seconds,
                            "release_policy": release_policy,
                            "prompt_sha256": _prompt_sha256(prompt_copy),
                            "binding_metadata": binding_metadata,
                        }
                    )
                else:
                    if existing is not None and existing.get("state") in ACTIVE_STATES:
                        active_id = str(existing.get("active_prompt_id") or "")
                        location = (
                            self.runtime.prompt_location(active_id) if active_id else "missing"
                        )
                        if location in {"queued", "running"}:
                            raise BackgroundJobError(
                                f"Chain '{safe_chain}' already has an active background prompt "
                                f"({active_id}, {location})"
                            )
                    accepted_count, manifest_complete = self._manifest_position(safe_chain)
                    if manifest_complete:
                        raise BackgroundJobError(
                            f"Chain '{safe_chain}' already has a final accepted segment"
                        )
                    job_id = str(uuid.uuid4())
                    now = time.time()
                    state = {
                        "schema": BACKGROUND_SCHEMA,
                        "chain_id": safe_chain,
                        "job_id": job_id,
                        "previous_job_id": (
                            str(existing.get("job_id") or "") if existing is not None else ""
                        ),
                        "previous_state_schema": (
                            int(
                                existing.get(
                                    "migrated_from_schema", existing.get("schema", BACKGROUND_SCHEMA)
                                )
                            )
                            if existing is not None
                            else 0
                        ),
                        "state": "running",
                        "controller_node_id": str(controller_node_id),
                        "active_prompt_id": prompt_id,
                        "accepted_count": accepted_count,
                        "current_segment_index": accepted_count,
                        "max_retries": max_retries,
                        "retry_count": 0,
                        "retry_delay_seconds": retry_delay_seconds,
                        "release_policy": release_policy,
                        "pause_requested": False,
                        "cancel_requested": False,
                        "last_error": "",
                        "last_candidate_json_path": "",
                        "last_manifest_path": "",
                        "final_video_path": "",
                        "prompt_sha256": _prompt_sha256(prompt_copy),
                        "binding_metadata": binding_metadata,
                        "created_unix": now,
                    }
                    if quarantined_state_path:
                        state["quarantined_state_path"] = quarantined_state_path
                        state["recovery_notice"] = recovery_note
                state["active_prompt_id"] = prompt_id
                state = self._write(state)
                self._bind_chain_lease(safe_chain, job_id)
                self._snapshots[job_id] = (prompt_copy, client_id)
                self._job_chains[job_id] = safe_chain
            except Exception:
                if newly_acquired_lease:
                    self._release_chain_lease(safe_chain)
                raise
        self._start_monitor(safe_chain, job_id, prompt_id)
        return dict(state)

    def assert_accept_allowed(self, job_id: str) -> dict:
        with self._lock:
            state = self._find_job(job_id)
            if state.get("cancel_requested") or state.get("state") == "cancelled":
                raise BackgroundJobError("Background job was cancelled before acceptance")
            if state.get("state") not in {"running", "pausing"}:
                raise BackgroundJobError(
                    f"Background job cannot accept a candidate while state={state.get('state')}"
                )
            return dict(state)

    def segment_accepted(
        self,
        job_id: str,
        *,
        candidate_index: int,
        candidate_json_path: str,
        manifest_path: str,
        accepted_count: int,
        is_final_segment: bool,
        final_video_path: str = "",
        post_accept_error: str = "",
    ) -> dict:
        monitor = None
        with self._lock:
            state = self._find_job(job_id)
            accepted_prompt_id = str(state.get("active_prompt_id") or "")
            state.update(
                {
                    "accepted_count": int(accepted_count),
                    "current_segment_index": int(candidate_index),
                    "last_candidate_json_path": str(candidate_json_path),
                    "last_manifest_path": str(manifest_path),
                    "final_video_path": str(final_video_path or ""),
                    "last_accepted_prompt_id": accepted_prompt_id,
                }
            )
            if post_accept_error:
                state.update(
                    {
                        "state": "failed",
                        "active_prompt_id": "",
                        "last_error": str(post_accept_error),
                    }
                )
                state = self._request_terminal_release(state, "post_accept_failure")
                return dict(self._write(state))
            if state.get("cancel_requested"):
                state.update({"state": "cancelled", "active_prompt_id": ""})
                state = self._request_terminal_release(state, "cancel_after_acceptance")
                return dict(self._write(state))
            release_policy = str(state["release_policy"])
            try:
                # Apply the selected policy after every durable acceptance, including a pause
                # boundary and the final segment. Previously it was only requested when another
                # prompt was queued, leaving final/paused jobs holding model VRAM unexpectedly.
                self.runtime.request_release(release_policy)
            except Exception as error:
                state.update(
                    {
                        "state": "failed",
                        "active_prompt_id": "",
                        "last_release_error": str(error)[:4000],
                        "last_error": (
                            "Accepted the segment but failed to apply release policy "
                            f"'{release_policy}': {error}"
                        ),
                    }
                )
                return dict(self._write(state))
            state.update(
                {
                    "last_release_policy": release_policy,
                    "release_requested_unix": time.time(),
                }
            )
            if bool(is_final_segment):
                state.update(
                    {
                        "state": "completed",
                        "active_prompt_id": "",
                        "current_segment_index": int(candidate_index),
                        "last_error": "",
                    }
                )
                return dict(self._write(state))
            if state.get("pause_requested"):
                state.update(
                    {
                        "state": "paused",
                        "active_prompt_id": "",
                        "current_segment_index": int(candidate_index) + 1,
                    }
                )
                return dict(self._write(state))

            snapshot = self._snapshots.get(job_id)
            if snapshot is None:
                state.update(
                    {
                        "state": "detached",
                        "active_prompt_id": "",
                        "last_error": (
                            "Prompt snapshot is unavailable after process restart; queue the "
                            "background workflow once to reattach it."
                        ),
                    }
                )
                return dict(self._write(state))
            state.update(
                {
                    "state": "scheduling",
                    "current_segment_index": int(candidate_index) + 1,
                    "retry_count": 0,
                    "last_error": "",
                }
            )
            self._write(state)
            try:
                next_prompt_id = self.runtime.queue_prompt(snapshot[0], snapshot[1])
            except Exception as error:
                state.update(
                    {
                        "state": "failed",
                        "active_prompt_id": "",
                        "last_error": f"Failed to queue the next segment: {error}",
                    }
                )
                return dict(self._write(state))
            state.update({"state": "running", "active_prompt_id": next_prompt_id})
            state = self._write(state)
            monitor = (state["chain_id"], job_id, next_prompt_id)
        if monitor is not None:
            self._start_monitor(*monitor)
        return dict(state)

    def fail_job(self, job_id: str, message: str) -> dict:
        with self._lock:
            state = self._find_job(job_id)
            state.update(
                {
                    "state": "failed",
                    "last_failed_prompt_id": str(state.get("active_prompt_id") or ""),
                    "active_prompt_id": "",
                    "last_error": str(message),
                }
            )
            state = self._request_terminal_release(state, "explicit_failure")
            return dict(self._write(state))

    def pause(self, chain_id: str) -> dict:
        safe_chain = sanitize_chain_id(chain_id)
        with self._lock:
            state = self._require_state(safe_chain)
            if state.get("state") in {"completed", "cancelled", "failed", "detached"}:
                raise BackgroundJobError(
                    f"Cannot pause a background job while state={state.get('state')}"
                )
            active_id = str(state.get("active_prompt_id") or "")
            state.update({"pause_requested": True, "state": "pausing"})
            self._write(state)
            location = self.runtime.prompt_location(active_id) if active_id else "missing"
            if location == "queued":
                result = self.runtime.cancel_prompt(active_id)
                state.update(
                    {
                        "state": "paused",
                        "active_prompt_id": "",
                        "last_control_result": result,
                    }
                )
                state = self._request_terminal_release(state, "pause_queued_prompt")
            elif location not in {"running"}:
                state.update({"state": "paused", "active_prompt_id": ""})
                state = self._request_terminal_release(state, "pause_inactive_prompt")
            return dict(self._write(state))

    def cancel(self, chain_id: str) -> dict:
        safe_chain = sanitize_chain_id(chain_id)
        with self._lock:
            state = self._require_state(safe_chain)
            active_id = str(state.get("active_prompt_id") or "")
            state.update(
                {
                    "cancel_requested": True,
                    "pause_requested": False,
                    "state": "cancelling",
                    "last_error": "",
                }
            )
            self._write(state)
            result = self.runtime.cancel_prompt(active_id) if active_id else {
                "prompt_id": "",
                "deleted_from_queue": False,
                "interrupt_signalled": False,
            }
            state.update(
                {
                    "state": "cancelled",
                    "last_cancelled_prompt_id": active_id,
                    "active_prompt_id": "",
                    "last_control_result": result,
                }
            )
            state = self._request_terminal_release(state, "cancel_prompt")
            return dict(self._write(state))

    def resume(self, chain_id: str) -> dict:
        safe_chain = sanitize_chain_id(chain_id)
        monitor = None
        with self._lock:
            state = self._require_state(safe_chain)
            if state.get("state") not in {"paused", "failed", "detached"}:
                raise BackgroundJobError(
                    f"Cannot resume a background job while state={state.get('state')}"
                )
            _accepted_count, manifest_complete = self._manifest_position(safe_chain)
            if manifest_complete:
                raise BackgroundJobError(
                    "The accepted manifest is already complete. If final composition failed, "
                    "run Compose Accepted instead of generating another segment."
                )
            job_id = str(state["job_id"])
            snapshot = self._snapshots.get(job_id)
            if snapshot is None:
                raise BackgroundJobError(
                    "The prompt snapshot was lost after process restart. Queue the background "
                    "workflow once to reattach and resume from the accepted manifest."
                )
            state.update(
                {
                    "state": "scheduling",
                    "pause_requested": False,
                    "cancel_requested": False,
                    "retry_count": 0,
                    "last_error": "",
                }
            )
            self._write(state)
            try:
                self.runtime.request_release(str(state["release_policy"]))
                prompt_id = self.runtime.queue_prompt(snapshot[0], snapshot[1])
            except Exception as error:
                state.update(
                    {
                        "state": "failed",
                        "active_prompt_id": "",
                        "last_error": f"Failed to resume background prompt: {error}",
                    }
                )
                self._write(state)
                raise BackgroundJobError(state["last_error"]) from error
            state.update({"state": "running", "active_prompt_id": prompt_id})
            state = self._write(state)
            monitor = (safe_chain, job_id, prompt_id)
        self._start_monitor(*monitor)
        return dict(state)

    def _find_job(self, job_id: str) -> dict:
        job_id = str(job_id)
        known_chain = self._job_chains.get(job_id)
        if known_chain:
            state = load_background_job_state(known_chain)
            if state is not None and str(state.get("job_id")) == job_id:
                return dict(state)
        state_root = long_video_chain_root("_background_index").parent
        if not state_root.is_dir():
            raise BackgroundJobError(f"Unknown or stale background job id: {job_id}")
        for chain_dir in state_root.iterdir():
            if not chain_dir.is_dir():
                continue
            try:
                state = load_background_job_state(chain_dir.name)
            except (BackgroundJobError, ValueError):
                continue
            if state is not None and str(state.get("job_id")) == job_id:
                self._job_chains[job_id] = str(state["chain_id"])
                return dict(state)
        raise BackgroundJobError(f"Unknown or stale background job id: {job_id}")

    @staticmethod
    def _require_state(chain_id: str) -> dict:
        state = load_background_job_state(chain_id)
        if state is None:
            raise BackgroundJobError(f"No background job exists for chain '{chain_id}'")
        return dict(state)

    def _start_monitor(self, chain_id: str, job_id: str, prompt_id: str) -> None:
        if not self.start_monitors:
            return
        key = (job_id, prompt_id)
        with self._lock:
            if key in self._monitors:
                return
            self._monitors.add(key)
        thread = threading.Thread(
            target=self._monitor_prompt,
            args=(chain_id, job_id, prompt_id),
            name=f"h3-t8-bg-{prompt_id[:8]}",
            daemon=True,
        )
        thread.start()

    def _monitor_prompt(self, chain_id: str, job_id: str, prompt_id: str) -> None:
        try:
            while True:
                with self._lock:
                    state = load_background_job_state(chain_id)
                    if state is None or str(state.get("job_id")) != job_id:
                        return
                    if str(state.get("active_prompt_id") or "") != prompt_id:
                        return
                record = self.runtime.history_record(prompt_id)
                if record is not None:
                    self._handle_prompt_history(chain_id, job_id, prompt_id, record)
                    return
                time.sleep(self.monitor_interval_seconds)
        finally:
            with self._lock:
                self._monitors.discard((job_id, prompt_id))

    def _handle_prompt_history(
        self, chain_id: str, job_id: str, prompt_id: str, record: Mapping
    ) -> None:
        retry = None
        with self._lock:
            state = load_background_job_state(chain_id)
            if (
                state is None
                or str(state.get("job_id")) != job_id
                or str(state.get("active_prompt_id") or "") != prompt_id
            ):
                return
            status = record.get("status") if isinstance(record, Mapping) else None
            status_str = status.get("status_str") if isinstance(status, Mapping) else None
            if status_str == "success":
                state.update(
                    {
                        "state": "failed",
                        "last_failed_prompt_id": prompt_id,
                        "active_prompt_id": "",
                        "last_error": (
                            "The prompt completed without the background terminal node advancing "
                            "the accepted manifest. Verify that Auto Accept & Continue is an output."
                        ),
                    }
                )
                state = self._request_terminal_release(state, "missing_terminal_node")
                self._write(state)
                return
            if state.get("cancel_requested") or state.get("state") == "cancelled":
                return
            if state.get("pause_requested"):
                state.update({"state": "paused", "active_prompt_id": ""})
                state = self._request_terminal_release(state, "pause_after_prompt_history")
                self._write(state)
                return
            retry_count = int(state.get("retry_count", 0))
            max_retries = int(state.get("max_retries", 0))
            error_text = _history_error(record)
            if retry_count >= max_retries:
                state.update(
                    {
                        "state": "failed",
                        "last_failed_prompt_id": prompt_id,
                        "active_prompt_id": "",
                        "last_error": error_text,
                    }
                )
                state = self._request_terminal_release(state, "retry_exhausted")
                self._write(state)
                return
            state.update(
                {
                    "state": "retry_wait",
                    "last_failed_prompt_id": prompt_id,
                    "retry_count": retry_count + 1,
                    "last_error": error_text,
                }
            )
            state = self._write(state)
            retry = (
                chain_id,
                job_id,
                prompt_id,
                float(state.get("retry_delay_seconds", 0.0)),
            )
        if retry is not None:
            self._retry_after_delay(*retry)

    def _retry_after_delay(
        self, chain_id: str, job_id: str, failed_prompt_id: str, delay_seconds: float
    ) -> None:
        if delay_seconds:
            time.sleep(delay_seconds)
        monitor = None
        with self._lock:
            state = load_background_job_state(chain_id)
            if (
                state is None
                or str(state.get("job_id")) != job_id
                or state.get("state") != "retry_wait"
                or str(state.get("active_prompt_id") or "") != failed_prompt_id
            ):
                return
            snapshot = self._snapshots.get(job_id)
            if snapshot is None:
                state.update(
                    {
                        "state": "detached",
                        "active_prompt_id": "",
                        "last_error": "Retry prompt snapshot is unavailable after process restart.",
                    }
                )
                state = self._request_terminal_release(state, "retry_snapshot_unavailable")
                self._write(state)
                return
            try:
                self.runtime.request_release(str(state["release_policy"]))
                prompt_id = self.runtime.queue_prompt(snapshot[0], snapshot[1])
            except Exception as error:
                state.update(
                    {
                        "state": "failed",
                        "active_prompt_id": "",
                        "last_error": f"Failed to queue retry: {error}",
                    }
                )
                self._write(state)
                return
            state.update({"state": "running", "active_prompt_id": prompt_id})
            self._write(state)
            monitor = (chain_id, job_id, prompt_id)
        if monitor is not None:
            self._start_monitor(*monitor)


BACKGROUND_JOBS = BackgroundJobManager()
