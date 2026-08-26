from __future__ import annotations

from pathlib import Path
import sys

import torch


TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import audit_skin_finish_vretoucher as audit  # noqa: E402


def test_missing_checkpoint_is_reported_without_model_construction(tmp_path: Path):
    report = audit.audit_checkpoint(tmp_path / "gen_best.pth")
    assert report["status"] == "MISSING_CHECKPOINT"
    assert report["model_constructed"] is False
    assert report["inference_run"] is False


def test_safe_tensor_state_dict_requires_a_trusted_hash(tmp_path: Path):
    checkpoint = tmp_path / "gen_best.pth"
    torch.save({"conv.weight": torch.ones((2, 3, 1, 1))}, checkpoint)
    report = audit.audit_checkpoint(
        checkpoint,
        expected_size_bytes=checkpoint.stat().st_size,
        expected_sha256=None,
    )
    assert report["status"] == "UNVERIFIED_WEIGHT_HASH_REQUIRED"
    assert report["tensor_count"] == 1
    assert report["parameter_numel"] == 6


def test_trusted_hash_passes_structure_only_and_tamper_is_rejected(tmp_path: Path):
    checkpoint = tmp_path / "gen_best.pth"
    torch.save({"conv.weight": torch.ones((2, 3, 1, 1))}, checkpoint)
    digest = audit._sha256(checkpoint)
    report = audit.audit_checkpoint(
        checkpoint,
        expected_size_bytes=checkpoint.stat().st_size,
        expected_sha256=digest,
    )
    assert report["status"] == "STRUCTURE_AND_TRUSTED_HASH_PASS_MODEL_NOT_LOADED"
    assert report["model_constructed"] is False
    assert report["inference_run"] is False

    rejected = audit.audit_checkpoint(
        checkpoint,
        expected_size_bytes=checkpoint.stat().st_size,
        expected_sha256="0" * 64,
    )
    assert rejected["status"] == "REJECTED_SHA256_MISMATCH"


def test_non_tensor_state_dict_entry_is_rejected_by_safe_loader(tmp_path: Path):
    checkpoint = tmp_path / "gen_best.pth"
    torch.save({"conv.weight": torch.ones(1), "metadata": "unsafe-shape"}, checkpoint)
    report = audit.audit_checkpoint(
        checkpoint,
        expected_size_bytes=checkpoint.stat().st_size,
    )
    assert report["status"] == "REJECTED_NON_TENSOR_STATE_DICT_ENTRY"


def test_exact_meta_structure_is_required_before_model_load(tmp_path: Path):
    checkpoint = tmp_path / "gen_best.pth"
    state = {"conv.weight": torch.ones((2, 3, 1, 1))}
    torch.save(state, checkpoint)
    observed = audit._state_structure(state)
    expected = {
        "schema": audit.STRUCTURE_REPORT_SCHEMA,
        "status": "META_STRUCTURE_PASS_CHECKPOINT_NOT_VALIDATED",
        **observed,
    }
    digest = audit._sha256(checkpoint)
    report = audit.audit_checkpoint(
        checkpoint,
        expected_size_bytes=checkpoint.stat().st_size,
        expected_sha256=digest,
        expected_structure=expected,
    )
    assert report["status"] == "EXACT_STRUCTURE_AND_TRUSTED_HASH_PASS_MODEL_NOT_LOADED"
    assert report["exact_state_structure_match"] is True

    expected["entries"][0]["shape"] = [99]
    rejected = audit.audit_checkpoint(
        checkpoint,
        expected_size_bytes=checkpoint.stat().st_size,
        expected_sha256=digest,
        expected_structure=expected,
    )
    assert rejected["status"] == "REJECTED_STATE_STRUCTURE_MISMATCH"
    assert rejected["model_constructed"] is False
    assert rejected["inference_run"] is False
