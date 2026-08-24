from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from h3_audio_t8_pkg.creator_artifact_quarantine_advanced import (
    CREATOR_QUARANTINE_MANIFEST_SCHEMA,
    CREATOR_QUARANTINE_RECEIPT_SCHEMA,
    execute_creator_artifact_quarantine,
    prepare_creator_quarantine_manifest,
)
from h3_audio_t8_pkg.creator_runtime_advanced import CREATOR_RETENTION_PLAN_SCHEMA
from h3_audio_t8_pkg.creator_workspace_advanced import _hash, canonical_json
from h3_audio_t8_pkg.nodes_creator_artifact_quarantine_advanced import (
    MiniMaxH3CreatorArtifactQuarantineT8Advanced,
)


def _retention_plan(*paths: str) -> dict:
    records = []
    for index, path in enumerate(paths):
        manifest = {"video": {"path": path}, "candidate": index}
        records.append(
            {
                "artifact_manifest_hash": _hash(manifest),
                "artifact_manifest": manifest,
                "path_hints": [{"pointer": "$.video.path", "path": path}],
                "sources": [
                    {
                        "event_sequence": index,
                        "run_position": 0,
                        "shot_id": "shot_0",
                        "variant_index": index,
                        "attempt_number": 1,
                        "outcome": "rejected",
                        "run_key": f"run_{index}",
                    }
                ],
            }
        )
    plan = {
        "schema": CREATOR_RETENTION_PLAN_SCHEMA,
        "workspace_hash": "workspace-hash",
        "ledger_hash": "ledger-hash",
        "status": "READY_FOR_EXTERNAL_EXECUTOR",
        "shot_count": 1,
        "artifact_manifest_count": len(records),
        "keep_manifest_count": 0,
        "proposed_delete_manifest_count": len(records),
        "waiting_run_positions": [],
        "findings": [],
        "shots": [],
        "keep_manifests": [],
        "proposed_delete_manifests": records,
        "artifact_paths_reviewed_by_user": True,
        "external_execution_ready": True,
        "files_mutated": False,
        "files_deleted": False,
        "destructive_executor_included": False,
    }
    plan["plan_hash"] = _hash(plan)
    return plan


def test_prepare_quarantine_and_restore_are_hash_locked_and_recoverable(tmp_path):
    output = tmp_path / "output"
    source = output / "MiniMaxH3" / "candidate.mp4"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"candidate-media-v1")
    plan = _retention_plan("output/MiniMaxH3/candidate.mp4")

    prepared = execute_creator_artifact_quarantine(plan, output)
    manifest, status, manifest_json, count, total_bytes, receipt_json, report_json = prepared
    assert status == "PREPARED_REVIEW_REQUIRED"
    assert manifest["schema"] == CREATOR_QUARANTINE_MANIFEST_SCHEMA
    assert count == 1 and total_bytes == len(b"candidate-media-v1")
    assert receipt_json == ""
    assert json.loads(report_json)["files_mutated"] is False
    assert source.read_bytes() == b"candidate-media-v1"

    quarantined = execute_creator_artifact_quarantine(
        plan,
        output,
        action="quarantine",
        execution_manifest_json=manifest_json,
        expected_plan_hash=plan["plan_hash"],
        execution_epoch=1,
        confirm_action=True,
    )
    assert quarantined[1] == "QUARANTINED_RECOVERABLE"
    receipt = json.loads(quarantined[5])
    assert receipt["schema"] == CREATOR_QUARANTINE_RECEIPT_SCHEMA
    assert receipt["files_deleted"] is False
    assert not source.exists()
    quarantine_path = output / receipt["entries"][0]["quarantine_relative"]
    assert quarantine_path.read_bytes() == b"candidate-media-v1"
    assert (output / receipt["journal_relative"]).is_file()

    restored = execute_creator_artifact_quarantine(
        plan,
        output,
        action="restore",
        execution_manifest_json=quarantined[5],
        expected_plan_hash=plan["plan_hash"],
        execution_epoch=1,
        confirm_action=True,
    )
    assert restored[1] == "RESTORED"
    assert source.read_bytes() == b"candidate-media-v1"
    assert not quarantine_path.exists()
    assert json.loads(restored[-1])["files_deleted"] is False


def test_prepare_rejects_paths_outside_output_and_directories(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"outside")
    with pytest.raises(ValueError, match="escapes"):
        prepare_creator_quarantine_manifest(_retention_plan(str(outside)), output)

    directory = output / "directory"
    directory.mkdir()
    with pytest.raises(ValueError, match="not a regular file"):
        prepare_creator_quarantine_manifest(_retention_plan("directory"), output)


def test_prepare_rejects_duplicate_paths_and_tampered_plan_hash(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    (output / "candidate.mp4").write_bytes(b"candidate")
    duplicate = _retention_plan("candidate.mp4", "candidate.mp4")
    with pytest.raises(ValueError, match="more than once"):
        prepare_creator_quarantine_manifest(duplicate, output)

    tampered = _retention_plan("candidate.mp4")
    tampered["status"] = "COMPLETE_KEEP_ONLY"
    with pytest.raises(ValueError, match="plan_hash"):
        prepare_creator_quarantine_manifest(tampered, output)

    alias_plan = _retention_plan("candidate.mp4", str(output / "candidate.mp4"))
    with pytest.raises(ValueError, match="resolved artifact"):
        prepare_creator_quarantine_manifest(alias_plan, output)


def test_quarantine_requires_confirmation_exact_plan_and_unchanged_content(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    source = output / "candidate.mp4"
    source.write_bytes(b"before")
    plan = _retention_plan("candidate.mp4")
    manifest = prepare_creator_quarantine_manifest(plan, output)
    manifest_json = canonical_json(manifest)

    with pytest.raises(ValueError, match="confirm_action"):
        execute_creator_artifact_quarantine(
            plan,
            output,
            action="quarantine",
            execution_manifest_json=manifest_json,
            expected_plan_hash=plan["plan_hash"],
            execution_epoch=2,
        )
    assert source.read_bytes() == b"before"

    with pytest.raises(ValueError, match="expected_plan_hash"):
        execute_creator_artifact_quarantine(
            plan,
            output,
            action="quarantine",
            execution_manifest_json=manifest_json,
            expected_plan_hash="wrong",
            execution_epoch=2,
            confirm_action=True,
        )
    source.write_bytes(b"after")
    with pytest.raises(ValueError, match="byte size changed|SHA-256 changed"):
        execute_creator_artifact_quarantine(
            plan,
            output,
            action="quarantine",
            execution_manifest_json=manifest_json,
            expected_plan_hash=plan["plan_hash"],
            execution_epoch=2,
            confirm_action=True,
        )
    assert source.read_bytes() == b"after"


def test_partial_move_failure_rolls_already_moved_files_back(monkeypatch, tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    first = output / "first.mp4"
    second = output / "second.mp4"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    plan = _retention_plan("first.mp4", "second.mp4")
    manifest = prepare_creator_quarantine_manifest(plan, output)
    real_replace = os.replace

    def fail_second_source_move(source, target):
        source_path = Path(source)
        target_path = Path(target)
        if source_path.name == "second.mp4" and "creator_quarantine" in str(target_path):
            raise OSError("simulated second move failure")
        return real_replace(source, target)

    monkeypatch.setattr(os, "replace", fail_second_source_move)
    with pytest.raises(OSError, match="simulated"):
        execute_creator_artifact_quarantine(
            plan,
            output,
            action="quarantine",
            execution_manifest_json=canonical_json(manifest),
            expected_plan_hash=plan["plan_hash"],
            execution_epoch=3,
            confirm_action=True,
        )
    assert first.read_bytes() == b"first"
    assert second.read_bytes() == b"second"
    journal = (
        output
        / "MiniMaxH3"
        / "creator_quarantine"
        / plan["plan_hash"]
        / "epoch-0000000003"
        / "journal.json"
    )
    assert json.loads(journal.read_text(encoding="utf-8"))["state"] == "rolled_back"


def test_prepare_rejects_symlink_when_platform_supports_it(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    target = output / "target.mp4"
    target.write_bytes(b"target")
    link = output / "linked.mp4"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("this Windows account cannot create symlinks")
    with pytest.raises(ValueError, match="symlink|junction"):
        prepare_creator_quarantine_manifest(_retention_plan("linked.mp4"), output)


def test_recover_to_source_can_replace_a_dead_process_lock(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    first = output / "first.mp4"
    second = output / "second.mp4"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    plan = _retention_plan("first.mp4", "second.mp4")
    manifest = prepare_creator_quarantine_manifest(plan, output)
    epoch = 77
    operation_root = (
        output
        / "MiniMaxH3"
        / "creator_quarantine"
        / plan["plan_hash"]
        / f"epoch-{epoch:010d}"
    )
    target = operation_root / "files" / "first.mp4"
    target.parent.mkdir(parents=True)
    os.replace(first, target)
    journal = {
        "schema": CREATOR_QUARANTINE_RECEIPT_SCHEMA,
        "operation": "quarantine",
        "state": "moving",
        "execution_epoch": epoch,
        "execution_manifest": manifest,
        "moved_source_relatives": ["first.mp4"],
        "files_deleted": False,
    }
    (operation_root / "journal.json").write_text(
        canonical_json(journal), encoding="utf-8"
    )
    (operation_root / ".operation.lock").write_text(
        canonical_json({"pid": 2_147_483_647, "created_at": "crashed"}),
        encoding="utf-8",
    )

    result = execute_creator_artifact_quarantine(
        plan,
        output,
        action="recover_to_source",
        execution_manifest_json=canonical_json(manifest),
        expected_plan_hash=plan["plan_hash"],
        execution_epoch=epoch,
        confirm_action=True,
    )

    assert result[1] == "RECOVERED_TO_SOURCE"
    assert first.read_bytes() == b"first"
    assert second.read_bytes() == b"second"
    assert not target.exists()
    assert not (operation_root / ".operation.lock").exists()
    recovered_journal = json.loads(
        (operation_root / "journal.json").read_text(encoding="utf-8")
    )
    assert recovered_journal["state"] == "restored"


def test_node_schema_is_experimental_and_non_mutating_by_default():
    schema = MiniMaxH3CreatorArtifactQuarantineT8Advanced.define_schema()
    assert schema.node_id == "MiniMaxH3CreatorArtifactQuarantineT8Advanced"
    assert schema.is_experimental is True
    assert schema.is_output_node is True
    assert schema.inputs[1].default == "prepare_only"
    assert schema.inputs[4].default == 0
    assert schema.inputs[5].default is False
