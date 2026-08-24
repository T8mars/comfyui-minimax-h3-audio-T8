from __future__ import annotations

from copy import deepcopy
import json

import pytest

from h3_audio_t8_pkg.creator_runtime_advanced import (
    BACKGROUND_STATE_FORMAT,
    CREATOR_RETENTION_PLAN_SCHEMA,
    CREATOR_RESUME_PLAN_SCHEMA,
    CREATOR_RUN_LEDGER_SCHEMA,
    compile_creator_background_selection,
    compile_creator_retention_plan,
    compile_creator_resume_plan,
    creator_background_binding,
    record_creator_run_receipt,
    validate_creator_run_ledger,
)
from h3_audio_t8_pkg.creator_workspace_advanced import (
    _hash,
    add_creator_shot_override,
    compile_creator_workspace,
)
from h3_audio_t8_pkg.studio_advanced import build_studio_timeline


def _workspace():
    timeline = build_studio_timeline(
        "creator_runtime",
        json.dumps(
            [
                {"id": "opening", "prompt": "Opening", "duration_seconds": 3.0},
                {"id": "action", "prompt": "Action", "duration_seconds": 4.0},
            ]
        ),
        "minimax_h3",
        5.0,
        "16:9",
        100,
        "increment",
        True,
        True,
    )
    edits = add_creator_shot_override(
        timeline,
        0,
        True,
        "",
        False,
        0,
        2,
        17,
        "{}",
        "keep_winner_and_metadata",
        "none",
        0,
        None,
    )[0]
    return compile_creator_workspace(timeline, 0, -1, False, "", edits)[0]


def _receipt(workspace, previous=None, **values):
    args = {
        "workspace": workspace,
        "run_position": 0,
        "variant_index": 0,
        "attempt_number": 1,
        "outcome": "completed",
        "prompt_id": "prompt-1",
        "cache_observation": "executed",
        "artifact_manifest_json": '{"video":{"path":"candidate.mp4"}}',
        "notes": "",
        "previous_ledger": previous,
    }
    args.update(values)
    return record_creator_run_receipt(**args)


def test_receipt_records_deterministic_identity_without_mutating_runtime():
    workspace = _workspace()
    ledger, event_json, ledger_json, report_json = _receipt(workspace)
    event = json.loads(event_json)
    report = json.loads(report_json)
    assert ledger["schema"] == CREATOR_RUN_LEDGER_SCHEMA
    assert json.loads(ledger_json)["ledger_hash"] == ledger["ledger_hash"]
    assert event["selected_seed"] == 100
    assert event["cache_observation"] == "executed"
    assert event["artifact_paths_verified"] is False
    assert report["queue_mutated"] is False
    assert report["files_mutated"] is False
    assert report["cache_observation_verified"] is False
    assert validate_creator_run_ledger(ledger, workspace) == ledger


def test_completed_candidate_must_be_reviewed_then_next_shot_is_selected():
    workspace = _workspace()
    completed = _receipt(workspace)[0]
    review_plan = compile_creator_resume_plan(workspace, completed)[0]
    assert review_plan["schema"] == CREATOR_RESUME_PLAN_SCHEMA
    assert review_plan["action"] == "review"
    assert review_plan["run_position"] == 0
    assert review_plan["attempt_number"] == 1

    accepted = _receipt(
        workspace,
        completed,
        outcome="accepted",
        artifact_manifest_json="{}",
        cache_observation="unknown",
    )[0]
    next_plan, action, position, variant, attempt, _summary, report = (
        compile_creator_resume_plan(workspace, accepted)
    )
    assert action == "render"
    assert (position, variant, attempt) == (1, 0, 1)
    assert next_plan["accepted_run_positions"] == [0]
    assert json.loads(report)["queue_mutated"] is False


def test_untried_variant_precedes_retry_and_attempt_numbers_are_strict():
    workspace = _workspace()
    failed = _receipt(
        workspace,
        outcome="failed",
        artifact_manifest_json="{}",
        notes="OOM",
    )[0]
    plan = compile_creator_resume_plan(workspace, failed)[0]
    assert plan["action"] == "render"
    assert (plan["run_position"], plan["variant_index"], plan["attempt_number"]) == (
        0,
        1,
        1,
    )

    variant_one_failed = _receipt(
        workspace,
        failed,
        variant_index=1,
        outcome="cancelled",
        artifact_manifest_json="{}",
    )[0]
    retry = compile_creator_resume_plan(workspace, variant_one_failed)[0]
    assert retry["action"] == "retry"
    assert (retry["variant_index"], retry["attempt_number"]) == (0, 2)
    with pytest.raises(ValueError, match="attempt_number 2"):
        _receipt(
            workspace,
            variant_one_failed,
            attempt_number=3,
            outcome="failed",
            artifact_manifest_json="{}",
        )


def test_receipt_fail_closed_on_tampering_or_invalid_transition():
    workspace = _workspace()
    completed = _receipt(workspace)[0]
    with pytest.raises(ValueError, match="accepted or rejected"):
        _receipt(
            workspace,
            completed,
            attempt_number=2,
            outcome="failed",
            artifact_manifest_json="{}",
        )
    with pytest.raises(ValueError, match="require artifact_manifest_json"):
        _receipt(workspace, outcome="accepted", artifact_manifest_json="{}")

    tampered = deepcopy(completed)
    tampered["events"][0]["selected_seed"] = 999
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_creator_run_ledger(tampered, workspace)

    rehashed = deepcopy(completed)
    rehashed["events"][0]["run_position"] = 99
    rehashed["events"][0]["event_hash"] = _hash(
        {
            key: value
            for key, value in rehashed["events"][0].items()
            if key != "event_hash"
        }
    )
    rehashed["ledger_hash"] = _hash(
        {key: value for key, value in rehashed.items() if key != "ledger_hash"}
    )
    with pytest.raises(ValueError, match="invalid workspace run"):
        validate_creator_run_ledger(rehashed, workspace)


def test_retention_plan_waits_for_acceptance_and_never_mutates_files():
    workspace = _workspace()
    completed = _receipt(workspace)[0]
    plan, status, keep_json, delete_json, summary, report_json = (
        compile_creator_retention_plan(workspace, completed, True)
    )
    assert plan["schema"] == CREATOR_RETENTION_PLAN_SCHEMA
    assert status == "AWAITING_ACCEPTANCE"
    assert plan["waiting_run_positions"] == [0, 1]
    assert len(json.loads(keep_json)["manifests"]) == 1
    assert json.loads(delete_json)["manifests"] == []
    assert "files_deleted=0" in summary
    report = json.loads(report_json)
    assert report["files_mutated"] is False
    assert report["files_deleted"] is False


def test_retention_plan_requires_review_then_separates_winner_and_rejected_artifact():
    workspace = _workspace()
    completed = _receipt(workspace)[0]
    rejected = _receipt(
        workspace,
        completed,
        outcome="rejected",
        artifact_manifest_json="{}",
    )[0]
    second_completed = _receipt(
        workspace,
        rejected,
        variant_index=1,
        artifact_manifest_json='{"video":{"path":"output/winner.mp4"}}',
    )[0]
    accepted_first = _receipt(
        workspace,
        second_completed,
        variant_index=1,
        outcome="accepted",
        artifact_manifest_json="{}",
    )[0]
    shot_one_completed = _receipt(
        workspace,
        accepted_first,
        run_position=1,
        artifact_manifest_json='{"video":{"path":"output/shot1.mp4"}}',
    )[0]
    complete_ledger = _receipt(
        workspace,
        shot_one_completed,
        run_position=1,
        outcome="accepted",
        artifact_manifest_json="{}",
    )[0]

    review_plan = compile_creator_retention_plan(workspace, complete_ledger, False)[0]
    assert review_plan["status"] == "REVIEW_REQUIRED"
    assert review_plan["files_deleted"] is False
    ready_plan, status, keep_json, delete_json, _summary, _report = (
        compile_creator_retention_plan(workspace, complete_ledger, True)
    )
    assert status == "READY_FOR_EXTERNAL_EXECUTOR"
    assert ready_plan["external_execution_ready"] is True
    keep_paths = {
        hint["path"]
        for item in json.loads(keep_json)["manifests"]
        for hint in item["path_hints"]
    }
    delete_paths = {
        hint["path"]
        for item in json.loads(delete_json)["manifests"]
        for hint in item["path_hints"]
    }
    assert keep_paths == {"output/winner.mp4", "output/shot1.mp4"}
    assert delete_paths == {"candidate.mp4"}
    assert ready_plan["files_deleted"] is False


def test_retention_plan_metadata_only_proposes_accepted_media_but_never_executes():
    workspace = _workspace()
    workspace["shots"][0]["retention_policy"] = "metadata_only"
    workspace["workspace_hash"] = _hash(
        {key: value for key, value in workspace.items() if key != "workspace_hash"}
    )
    completed = _receipt(workspace)[0]
    accepted = _receipt(
        workspace,
        completed,
        outcome="accepted",
        artifact_manifest_json="{}",
    )[0]
    plan = compile_creator_retention_plan(workspace, accepted, True)[0]
    assert plan["status"] == "AWAITING_ACCEPTANCE"
    assert plan["files_deleted"] is False
    assert plan["proposed_delete_manifest_count"] == 1


def test_retention_plan_abstains_when_keep_and_delete_resolve_to_the_same_path():
    workspace = _workspace()
    completed = _receipt(workspace)[0]
    rejected = _receipt(
        workspace,
        completed,
        outcome="rejected",
        artifact_manifest_json="{}",
    )[0]
    winner_completed = _receipt(
        workspace,
        rejected,
        variant_index=1,
        artifact_manifest_json=(
            '{"video":{"path":"candidate.mp4"},"label":"winner-metadata"}'
        ),
    )[0]
    accepted = _receipt(
        workspace,
        winner_completed,
        variant_index=1,
        outcome="accepted",
        artifact_manifest_json="{}",
    )[0]
    plan = compile_creator_retention_plan(workspace, accepted, True)[0]
    assert plan["status"] == "ABSTAIN"
    assert "both keep and proposed-delete" in plan["findings"][0]
    assert plan["external_execution_ready"] is False


def test_empty_ledger_starts_at_first_variant_without_runtime_side_effects():
    workspace = _workspace()
    plan, action, position, variant, attempt, summary, report_json = (
        compile_creator_resume_plan(workspace)
    )
    assert action == "render"
    assert (position, variant, attempt) == (0, 0, 1)
    assert "run_position=0" in summary
    assert plan["automatic_cache_claim"] is False
    report = json.loads(report_json)
    assert report["execution_authority"] is False
    assert report["files_mutated"] is False


def _background_state(workspace, **updates):
    state = {
        "schema": 2,
        "format": BACKGROUND_STATE_FORMAT,
        "chain_id": "creator-bound-chain",
        "job_id": "creator-job-1",
        "state": "running",
        "accepted_count": 0,
        "retry_count": 0,
        "max_retries": 2,
        "manifest_complete": False,
        "binding_metadata": creator_background_binding(workspace),
    }
    state.update(updates)
    return json.dumps(state)


def test_background_selection_uses_durable_progress_and_retry_variant():
    workspace = _workspace()
    plan, selected, report_json = compile_creator_background_selection(
        workspace,
        _background_state(workspace, retry_count=1),
        "retry_as_variant_clamped",
    )
    assert plan["ready"] is True
    assert plan["action"] == "render"
    assert (plan["run_position"], plan["variant_index"], plan["attempt_number"]) == (
        0,
        1,
        2,
    )
    assert selected[4] == 117
    report = json.loads(report_json)
    assert report["progress_source"] == "durable_accepted_manifest_and_background_state"
    assert report["queue_mutated"] is False


def test_background_selection_advances_shot_and_blocks_terminal_states():
    workspace = _workspace()
    plan, selected, _report = compile_creator_background_selection(
        workspace,
        _background_state(workspace, accepted_count=1, state="paused"),
        "fixed_first",
    )
    assert plan["ready"] is False
    assert plan["action"] == "resume_from_background_control"
    assert plan["run_position"] == 1
    assert selected[4] == 101

    complete, _selected, _report = compile_creator_background_selection(
        workspace,
        _background_state(
            workspace,
            accepted_count=2,
            state="completed",
            manifest_complete=True,
        ),
        "fixed_first",
    )
    assert complete["ready"] is False
    assert complete["action"] == "complete"


def test_background_selection_rejects_cross_workspace_or_impossible_progress():
    workspace = _workspace()
    mismatched = json.loads(_background_state(workspace))
    mismatched["binding_metadata"]["workspace_hash"] = "0" * 64
    with pytest.raises(ValueError, match="not bound"):
        compile_creator_background_selection(
            workspace, json.dumps(mismatched), "retry_as_variant_clamped"
        )
    with pytest.raises(ValueError, match="exceeds"):
        compile_creator_background_selection(
            workspace,
            _background_state(workspace, accepted_count=3),
            "retry_as_variant_clamped",
        )
    with pytest.raises(ValueError, match="manifest completed before"):
        compile_creator_background_selection(
            workspace,
            _background_state(workspace, accepted_count=1, manifest_complete=True),
            "retry_as_variant_clamped",
        )
