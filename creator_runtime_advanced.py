from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import json
import re

from .creator_workspace_advanced import (
    _hash,
    canonical_json,
    select_creator_workspace_shot,
    validate_creator_workspace,
)


CREATOR_RUN_LEDGER_SCHEMA = "t8.minimax_h3.creator_run_ledger.v1"
CREATOR_RESUME_PLAN_SCHEMA = "t8.minimax_h3.creator_resume_plan.v1"
CREATOR_RETENTION_PLAN_SCHEMA = "t8.minimax_h3.creator_retention_plan.v1"
OUTCOMES = ("completed", "accepted", "rejected", "cancelled", "failed")
CACHE_OBSERVATIONS = ("unknown", "executed", "cache_hit", "partial_cache_reuse")
_PROMPT_ID = re.compile(r"^[A-Za-z0-9._:-]{0,128}$")
CREATOR_BACKGROUND_BINDING_KIND = "t8.creator_workspace.long_video_background.v1"
BACKGROUND_STATE_FORMAT = "minimax_h3_t8_background_job"
BACKGROUND_ACTIVE_STATES = {"running", "pausing"}


def _json_object(value: str, name: str) -> dict:
    text = str(value or "").strip()
    if not text:
        return {}
    if len(text.encode("utf-8")) > 65536:
        raise ValueError(f"{name} exceeds the 64KiB metadata limit")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"{name} is invalid JSON: {error}") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return deepcopy(dict(payload))


def _run_descriptor(workspace: Mapping, run_position: int, variant_index: int) -> dict:
    position = int(run_position)
    if not 0 <= position < len(workspace["shots"]):
        raise ValueError(f"run_position must be between 0 and {len(workspace['shots']) - 1}")
    shot = workspace["shots"][position]
    variant = int(variant_index)
    if not 0 <= variant < len(shot["variant_seeds"]):
        raise ValueError(
            f"variant_index must be between 0 and {len(shot['variant_seeds']) - 1} "
            f"for run position {position}"
        )
    descriptor = {
        "workspace_hash": workspace["workspace_hash"],
        "run_position": position,
        "shot_index": int(shot["shot_index"]),
        "shot_id": str(shot["shot_id"]),
        "variant_index": variant,
        "selected_seed": int(shot["variant_seeds"][variant]),
        "frame_count": int(shot["frame_count"]),
        "source_packet_hash": shot.get("source_packet_hash"),
        "override_hash": shot.get("override_hash"),
    }
    descriptor["run_key"] = _hash(descriptor)
    return descriptor


def _empty_ledger(workspace: Mapping) -> dict:
    ledger = {
        "schema": CREATOR_RUN_LEDGER_SCHEMA,
        "workspace_hash": workspace["workspace_hash"],
        "source_timeline_hash": workspace["source_timeline_hash"],
        "events": [],
        "event_count": 0,
        "compiler_only": True,
        "runtime_authority": False,
        "queue_mutated": False,
        "files_mutated": False,
        "file_written": False,
    }
    ledger["ledger_hash"] = _hash(ledger)
    return ledger


def validate_creator_run_ledger(ledger: Mapping, workspace: Mapping) -> dict:
    workspace = validate_creator_workspace(workspace)
    if not isinstance(ledger, Mapping) or ledger.get("schema") != CREATOR_RUN_LEDGER_SCHEMA:
        raise ValueError("previous_ledger must be a Creator Run Ledger")
    if ledger.get("workspace_hash") != workspace["workspace_hash"]:
        raise ValueError("previous_ledger belongs to a different Creator Workspace")
    events = ledger.get("events")
    if not isinstance(events, list):
        raise ValueError("Creator Run Ledger events must be a list")
    if int(ledger.get("event_count", -1)) != len(events):
        raise ValueError("Creator Run Ledger event_count mismatch")
    expected = _hash({key: value for key, value in ledger.items() if key != "ledger_hash"})
    if ledger.get("ledger_hash") != expected:
        raise ValueError("Creator Run Ledger hash mismatch")
    validated_events_by_run: dict[str, list[dict]] = {}
    for sequence, event in enumerate(events):
        if not isinstance(event, Mapping) or int(event.get("sequence", -1)) != sequence:
            raise ValueError("Creator Run Ledger event sequence is invalid")
        event_payload = {key: value for key, value in event.items() if key != "event_hash"}
        if event.get("event_hash") != _hash(event_payload):
            raise ValueError("Creator Run Ledger event hash mismatch")
        try:
            descriptor = _run_descriptor(
                workspace,
                int(event.get("run_position", -1)),
                int(event.get("variant_index", -1)),
            )
        except (TypeError, ValueError) as error:
            raise ValueError("Creator Run Ledger event targets an invalid workspace run") from error
        if any(event.get(key) != value for key, value in descriptor.items()):
            raise ValueError("Creator Run Ledger event identity does not match the workspace")
        outcome = str(event.get("outcome") or "")
        cache_observation = str(event.get("cache_observation") or "")
        if outcome not in OUTCOMES or cache_observation not in CACHE_OBSERVATIONS:
            raise ValueError("Creator Run Ledger event enum is invalid")
        run_events = validated_events_by_run.setdefault(descriptor["run_key"], [])
        _validate_transition(run_events, int(event.get("attempt_number", 0)), outcome)
        run_events.append(deepcopy(dict(event)))
    return deepcopy(dict(ledger))


def _events_for_run(ledger: Mapping, run_key: str) -> list[dict]:
    return [deepcopy(dict(event)) for event in ledger["events"] if event["run_key"] == run_key]


def _validate_transition(events: list[dict], attempt_number: int, outcome: str) -> dict | None:
    if attempt_number < 1:
        raise ValueError("attempt_number must be at least 1")
    if not events:
        if attempt_number != 1:
            raise ValueError("the first receipt for a run must use attempt_number 1")
        return None
    latest = events[-1]
    latest_attempt = int(latest["attempt_number"])
    latest_outcome = str(latest["outcome"])
    if latest_outcome == "accepted":
        raise ValueError("an accepted run is terminal and cannot receive another receipt")
    if latest_outcome == "completed":
        if attempt_number != latest_attempt or outcome not in {"accepted", "rejected"}:
            raise ValueError(
                "a completed run must be accepted or rejected on the same attempt before retry"
            )
        return latest
    if attempt_number != latest_attempt + 1:
        raise ValueError(
            f"the next receipt must use attempt_number {latest_attempt + 1}"
        )
    return latest


def record_creator_run_receipt(
    workspace: Mapping,
    run_position: int,
    variant_index: int,
    attempt_number: int,
    outcome: str,
    prompt_id: str,
    cache_observation: str,
    artifact_manifest_json: str,
    notes: str,
    previous_ledger=None,
) -> tuple[dict, str, str, str]:
    workspace = validate_creator_workspace(workspace)
    if outcome not in OUTCOMES:
        raise ValueError(f"outcome must be one of {OUTCOMES}")
    if cache_observation not in CACHE_OBSERVATIONS:
        raise ValueError(f"cache_observation must be one of {CACHE_OBSERVATIONS}")
    prompt_id = str(prompt_id or "").strip()
    if not _PROMPT_ID.fullmatch(prompt_id):
        raise ValueError("prompt_id must contain only letters, digits, dot, underscore, colon or dash")
    notes = str(notes or "").strip()
    if len(notes) > 4096:
        raise ValueError("notes exceeds 4096 characters")
    artifacts = _json_object(artifact_manifest_json, "artifact_manifest_json")
    descriptor = _run_descriptor(workspace, run_position, variant_index)
    ledger = (
        _empty_ledger(workspace)
        if previous_ledger is None
        else validate_creator_run_ledger(previous_ledger, workspace)
    )
    run_events = _events_for_run(ledger, descriptor["run_key"])
    previous = _validate_transition(run_events, int(attempt_number), outcome)
    if outcome in {"completed", "accepted"} and not artifacts:
        if previous and previous.get("artifact_manifest"):
            artifacts = deepcopy(previous["artifact_manifest"])
        else:
            raise ValueError(
                "completed or accepted receipts require artifact_manifest_json metadata"
            )
    event = {
        "sequence": len(ledger["events"]),
        **descriptor,
        "attempt_number": int(attempt_number),
        "outcome": outcome,
        "prompt_id": prompt_id or None,
        "cache_observation": cache_observation,
        "cache_observation_source": "user_or_external_runtime_declared",
        "artifact_manifest": artifacts,
        "artifact_paths_verified": False,
        "notes": notes,
    }
    event["event_hash"] = _hash(event)
    ledger["events"].append(event)
    ledger["event_count"] = len(ledger["events"])
    ledger["ledger_hash"] = _hash(
        {key: value for key, value in ledger.items() if key != "ledger_hash"}
    )
    report = {
        "schema": CREATOR_RUN_LEDGER_SCHEMA,
        "workspace_hash": workspace["workspace_hash"],
        "ledger_hash": ledger["ledger_hash"],
        "event_hash": event["event_hash"],
        "run_key": descriptor["run_key"],
        "outcome": outcome,
        "cache_observation": cache_observation,
        "cache_observation_verified": False,
        "artifact_paths_verified": False,
        "queue_mutated": False,
        "files_mutated": False,
        "interpretation": (
            "This is an explicit receipt, not an automatic ComfyUI queue/cache probe. "
            "Record only the outcome actually observed in history or output metadata."
        ),
    }
    return ledger, canonical_json(event), canonical_json(ledger), canonical_json(report)


def compile_creator_resume_plan(
    workspace: Mapping,
    ledger=None,
) -> tuple[dict, str, int, int, int, str, str]:
    workspace = validate_creator_workspace(workspace)
    ledger = (
        _empty_ledger(workspace)
        if ledger is None
        else validate_creator_run_ledger(ledger, workspace)
    )
    accepted_positions: list[int] = []
    next_descriptor = None
    action = "complete"
    attempt_number = 0
    reason = "every workspace shot has one explicitly accepted candidate"

    for position, shot in enumerate(workspace["shots"]):
        variant_states = []
        accepted = False
        completed = None
        untried = None
        retryable = None
        for variant in range(len(shot["variant_seeds"])):
            descriptor = _run_descriptor(workspace, position, variant)
            events = _events_for_run(ledger, descriptor["run_key"])
            latest = events[-1] if events else None
            variant_states.append(
                {
                    "variant_index": variant,
                    "run_key": descriptor["run_key"],
                    "latest_outcome": latest.get("outcome") if latest else "untried",
                    "latest_attempt_number": int(latest["attempt_number"]) if latest else 0,
                }
            )
            if latest and latest["outcome"] == "accepted":
                accepted = True
            elif latest and latest["outcome"] == "completed" and completed is None:
                completed = (descriptor, int(latest["attempt_number"]))
            elif latest is None and untried is None:
                untried = (descriptor, 1)
            elif latest and latest["outcome"] in {"rejected", "cancelled", "failed"}:
                if retryable is None:
                    retryable = (descriptor, int(latest["attempt_number"]) + 1)
        if accepted:
            accepted_positions.append(position)
            continue
        if completed is not None:
            next_descriptor, attempt_number = completed
            action = "review"
            reason = "a completed candidate must be accepted or rejected before another attempt"
        elif untried is not None:
            next_descriptor, attempt_number = untried
            action = "render"
            reason = "the earliest unaccepted shot has an untried deterministic variant"
        elif retryable is not None:
            next_descriptor, attempt_number = retryable
            action = "retry"
            reason = "all deterministic variants were tried without acceptance"
        else:
            action = "blocked"
            reason = "the earliest unaccepted shot has no valid next state"
        break

    if next_descriptor is None:
        run_position = -1
        variant_index = -1
    else:
        run_position = int(next_descriptor["run_position"])
        variant_index = int(next_descriptor["variant_index"])
    plan = {
        "schema": CREATOR_RESUME_PLAN_SCHEMA,
        "workspace_hash": workspace["workspace_hash"],
        "ledger_hash": ledger["ledger_hash"],
        "action": action,
        "run_position": run_position,
        "variant_index": variant_index,
        "attempt_number": int(attempt_number),
        "accepted_run_positions": accepted_positions,
        "reason": reason,
        "next_run": next_descriptor,
        "queue_mutated": False,
        "files_mutated": False,
        "automatic_cache_claim": False,
    }
    plan["plan_hash"] = _hash(plan)
    summary = (
        f"{action}: workspace complete"
        if action == "complete"
        else f"{action}: run_position={run_position}, variant={variant_index}, "
        f"attempt={attempt_number}"
    )
    report = {
        "schema": CREATOR_RESUME_PLAN_SCHEMA,
        "action": action,
        "reason": reason,
        "workspace_hash": workspace["workspace_hash"],
        "ledger_hash": ledger["ledger_hash"],
        "plan_hash": plan["plan_hash"],
        "execution_authority": False,
        "queue_mutated": False,
        "files_mutated": False,
        "interpretation": (
            "Use the returned run_position/variant_index with the existing explicit Shot Select. "
            "This planner never queues, cancels, deletes or infers cache hits."
        ),
    }
    return plan, action, run_position, variant_index, int(attempt_number), summary, canonical_json(report)


def _artifact_path_hints(
    value,
    pointer: str = "$",
    depth: int = 0,
) -> list[dict[str, str]]:
    if depth > 32:
        raise ValueError("artifact manifest nesting exceeds 32 levels")
    hints: list[dict[str, str]] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_pointer = f"{pointer}.{key}"
            if str(key).lower() == "path" and isinstance(child, str) and child.strip():
                hints.append({"pointer": child_pointer, "path": child.strip()})
            else:
                hints.extend(_artifact_path_hints(child, child_pointer, depth + 1))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            hints.extend(_artifact_path_hints(child, f"{pointer}[{index}]", depth + 1))
    return hints


def compile_creator_retention_plan(
    workspace: Mapping,
    ledger=None,
    confirm_artifact_paths_reviewed: bool = False,
) -> tuple[dict, str, str, str, str, str]:
    workspace = validate_creator_workspace(workspace)
    ledger = (
        _empty_ledger(workspace)
        if ledger is None
        else validate_creator_run_ledger(ledger, workspace)
    )
    records: dict[str, dict] = {}
    event_manifest_keys: dict[int, str] = {}
    for event in ledger["events"]:
        manifest = deepcopy(dict(event.get("artifact_manifest") or {}))
        if not manifest:
            continue
        manifest_hash = _hash(manifest)
        event_manifest_keys[int(event["sequence"])] = manifest_hash
        record = records.setdefault(
            manifest_hash,
            {
                "artifact_manifest_hash": manifest_hash,
                "artifact_manifest": manifest,
                "path_hints": _artifact_path_hints(manifest),
                "sources": [],
            },
        )
        record["sources"].append(
            {
                "event_sequence": int(event["sequence"]),
                "run_position": int(event["run_position"]),
                "shot_id": str(event["shot_id"]),
                "variant_index": int(event["variant_index"]),
                "attempt_number": int(event["attempt_number"]),
                "outcome": str(event["outcome"]),
                "run_key": str(event["run_key"]),
            }
        )

    keep_keys: set[str] = set()
    delete_keys: set[str] = set()
    waiting_positions: list[int] = []
    findings: list[str] = []
    shot_reports: list[dict] = []
    for shot in workspace["shots"]:
        position = int(shot["run_position"])
        shot_events = [
            event for event in ledger["events"] if int(event["run_position"]) == position
        ]
        shot_keys = {
            event_manifest_keys[int(event["sequence"])]
            for event in shot_events
            if int(event["sequence"]) in event_manifest_keys
        }
        accepted_events = [event for event in shot_events if event["outcome"] == "accepted"]
        accepted_run_keys = {str(event["run_key"]) for event in accepted_events}
        if len(accepted_run_keys) > 1:
            findings.append(
                f"run_position {position} has more than one accepted candidate"
            )
        accepted_keys = {
            event_manifest_keys[int(event["sequence"])]
            for event in accepted_events
            if int(event["sequence"]) in event_manifest_keys
        }
        pending_keys = {
            event_manifest_keys[int(event["sequence"])]
            for event in shot_events
            if event["outcome"] == "completed"
            and int(event["sequence"]) in event_manifest_keys
            and not any(
                int(later["sequence"]) > int(event["sequence"])
                and later["run_key"] == event["run_key"]
                and int(later["attempt_number"]) == int(event["attempt_number"])
                and later["outcome"] in {"accepted", "rejected"}
                for later in shot_events
            )
        }
        policy = str(shot["retention_policy"])
        if not accepted_events:
            waiting_positions.append(position)
            keep_keys.update(shot_keys)
        elif policy == "keep_all":
            keep_keys.update(shot_keys)
        elif policy == "metadata_only":
            delete_keys.update(shot_keys)
        else:
            keep_keys.update(accepted_keys)
            delete_keys.update(shot_keys - accepted_keys)
        # A completed-but-unreviewed artifact is never proposed for deletion even
        # if another malformed ledger entry would otherwise classify it that way.
        keep_keys.update(pending_keys)
        shot_reports.append(
            {
                "run_position": position,
                "shot_id": str(shot["shot_id"]),
                "retention_policy": policy,
                "accepted_candidate_count": len(accepted_run_keys),
                "artifact_manifest_count": len(shot_keys),
                "waiting_for_acceptance": not bool(accepted_events),
            }
        )

    delete_keys.difference_update(keep_keys)
    keep_records = [records[key] for key in sorted(keep_keys)]
    delete_records = [records[key] for key in sorted(delete_keys)]
    keep_paths = {
        hint["path"] for record in keep_records for hint in record["path_hints"]
    }
    delete_paths = {
        hint["path"] for record in delete_records for hint in record["path_hints"]
    }
    overlapping_paths = sorted(keep_paths & delete_paths)
    if overlapping_paths:
        findings.append(
            "artifact path appears in both keep and proposed-delete manifests: "
            + ", ".join(overlapping_paths)
        )
    missing_delete_paths = [
        record["artifact_manifest_hash"]
        for record in delete_records
        if not record["path_hints"]
    ]
    if missing_delete_paths:
        findings.append(
            "proposed-delete artifact manifests contain no explicit path field: "
            + ", ".join(missing_delete_paths)
        )
    if findings:
        status = "ABSTAIN"
    elif waiting_positions:
        status = "AWAITING_ACCEPTANCE"
    elif delete_records and not bool(confirm_artifact_paths_reviewed):
        status = "REVIEW_REQUIRED"
    elif delete_records:
        status = "READY_FOR_EXTERNAL_EXECUTOR"
    else:
        status = "COMPLETE_KEEP_ONLY"
    external_execution_ready = status == "READY_FOR_EXTERNAL_EXECUTOR"
    plan = {
        "schema": CREATOR_RETENTION_PLAN_SCHEMA,
        "workspace_hash": workspace["workspace_hash"],
        "ledger_hash": ledger["ledger_hash"],
        "status": status,
        "shot_count": len(workspace["shots"]),
        "artifact_manifest_count": len(records),
        "keep_manifest_count": len(keep_records),
        "proposed_delete_manifest_count": len(delete_records),
        "waiting_run_positions": waiting_positions,
        "findings": findings,
        "shots": shot_reports,
        "keep_manifests": keep_records,
        "proposed_delete_manifests": delete_records,
        "artifact_paths_reviewed_by_user": bool(confirm_artifact_paths_reviewed),
        "external_execution_ready": external_execution_ready,
        "files_mutated": False,
        "files_deleted": False,
        "destructive_executor_included": False,
    }
    plan["plan_hash"] = _hash(plan)
    summary = (
        f"status={status}; keep={len(keep_records)}; "
        f"proposed_delete={len(delete_records)}; "
        f"waiting={len(waiting_positions)}; files_deleted=0"
    )
    report = {
        "schema": CREATOR_RETENTION_PLAN_SCHEMA,
        "workspace_hash": workspace["workspace_hash"],
        "ledger_hash": ledger["ledger_hash"],
        "plan_hash": plan["plan_hash"],
        "status": status,
        "files_mutated": False,
        "files_deleted": False,
        "external_execution_ready": external_execution_ready,
        "interpretation": (
            "This node compiles reviewable keep/delete candidates only. It never resolves, "
            "opens, moves or deletes a path. A separate explicitly authorized executor would "
            "still need to revalidate roots, hashes and current ledger state."
        ),
    }
    return (
        plan,
        status,
        canonical_json({"manifests": keep_records}),
        canonical_json({"manifests": delete_records}),
        summary,
        canonical_json(report),
    )


def creator_background_binding(workspace: Mapping) -> dict:
    workspace = validate_creator_workspace(workspace)
    return {
        "kind": CREATOR_BACKGROUND_BINDING_KIND,
        "workspace_hash": workspace["workspace_hash"],
        "run_count": len(workspace["shots"]),
    }


def compile_creator_background_selection(
    workspace: Mapping,
    background_state_json: str,
    variant_policy: str,
) -> tuple[dict, tuple, str]:
    workspace = validate_creator_workspace(workspace)
    state = _json_object(background_state_json, "background_state_json")
    if int(state.get("schema", -1)) != 2 or state.get("format") != BACKGROUND_STATE_FORMAT:
        raise ValueError("background_state_json must be a current MiniMax H3 background state")
    expected_binding = creator_background_binding(workspace)
    if state.get("binding_metadata") != expected_binding:
        raise ValueError("background state is not bound to this Creator Workspace")
    run_count = len(workspace["shots"])
    try:
        accepted_count = int(state.get("accepted_count", 0))
        retry_count = int(state.get("retry_count", 0))
        max_retries = int(state.get("max_retries", 0))
    except (TypeError, ValueError) as error:
        raise ValueError("background progress counters are invalid") from error
    if not 0 <= accepted_count <= run_count:
        raise ValueError("background accepted_count exceeds the Creator Workspace run count")
    if retry_count < 0 or max_retries < 0 or retry_count > max_retries:
        raise ValueError("background retry counters are inconsistent")
    state_name = str(state.get("state") or "unknown")
    manifest_complete = bool(state.get("manifest_complete", False))
    if accepted_count == run_count and not (
        manifest_complete or state_name == "completed"
    ):
        raise ValueError("all Creator shots are accepted but the background manifest is incomplete")
    if manifest_complete and accepted_count != run_count:
        raise ValueError("background manifest completed before all Creator shots were accepted")
    if variant_policy not in {"retry_as_variant_clamped", "fixed_first"}:
        raise ValueError(
            "variant_policy must be retry_as_variant_clamped or fixed_first"
        )

    ready = state_name in BACKGROUND_ACTIVE_STATES and accepted_count < run_count
    action_map = {
        "review_only": "enable_auto_accept_explicitly",
        "idle": "start_background_job",
        "paused": "resume_from_background_control",
        "failed": "inspect_error_then_resume_or_cancel",
        "cancelled": "cancelled_terminal",
        "detached": "reattach_from_accepted_manifest",
        "completed": "complete",
        "cancelling": "cancel_pending",
        "retry_wait": "retry_pending",
        "scheduling": "next_prompt_pending",
    }
    action = "render" if ready else action_map.get(state_name, "wait_for_valid_background_state")
    if accepted_count == run_count:
        action = "complete"

    # A blocked/complete NodeOutput still needs schema-compatible values. Select the final valid
    # shot as an inert placeholder; block_execution prevents it reaching Conditioning/Sampler.
    run_position = min(accepted_count, run_count - 1)
    shot = workspace["shots"][run_position]
    variant_count = len(shot["variant_seeds"])
    variant_index = (
        0
        if variant_policy == "fixed_first"
        else min(retry_count, variant_count - 1)
    )
    attempt_number = retry_count + 1
    selected = select_creator_workspace_shot(workspace, run_position, variant_index)
    plan = {
        "schema": "t8.minimax_h3.creator_background_selection.v1",
        "workspace_hash": workspace["workspace_hash"],
        "background_job_id": str(state.get("job_id") or ""),
        "chain_id": str(state.get("chain_id") or ""),
        "background_state": state_name,
        "accepted_count": accepted_count,
        "run_count": run_count,
        "retry_count": retry_count,
        "max_retries": max_retries,
        "run_position": run_position,
        "variant_index": variant_index,
        "variant_count": variant_count,
        "attempt_number": attempt_number,
        "variant_policy": variant_policy,
        "variant_clamped": (
            variant_policy == "retry_as_variant_clamped" and retry_count >= variant_count
        ),
        "ready": ready,
        "action": action,
        "queue_mutated": False,
        "files_mutated": False,
        "progress_source": "durable_accepted_manifest_and_background_state",
    }
    plan["plan_hash"] = _hash(plan)
    report = {
        **plan,
        "interpretation": (
            "The existing Long Video background controller owns queue, cancellation, retry and "
            "accepted-manifest progress. This adapter only selects the bound Creator shot/seed."
        ),
    }
    return plan, selected, canonical_json(report)
