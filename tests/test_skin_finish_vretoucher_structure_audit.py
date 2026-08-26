from __future__ import annotations

from pathlib import Path
import sys

import torch
from torch import nn


TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import audit_skin_finish_vretoucher_structure as audit  # noqa: E402


def test_missing_pinned_source_fails_before_import_or_construction(tmp_path: Path):
    report = audit.audit_meta_structure(tmp_path)
    assert report["status"] == "REJECTED_PINNED_SOURCE_MISSING"
    assert report["checkpoint_loaded"] is False
    assert report["forward_run"] is False
    assert report["real_parameter_storage_allocated"] is False


def test_pinned_source_hash_mismatch_is_explicit(tmp_path: Path):
    for relative in audit.PINNED_FILES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("not upstream", encoding="utf-8")
    report = audit.verify_pinned_source(tmp_path)
    assert report["status"] == "REJECTED_PINNED_SOURCE_HASH_MISMATCH"
    assert report["missing"] == []
    assert sorted(report["mismatched"]) == sorted(audit.PINNED_FILES)


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 2)
        self.register_buffer("scale", torch.ones(1))


def test_structure_manifest_is_shape_bound_and_stable():
    first = audit.state_structure_manifest(_Tiny())
    second = audit.state_structure_manifest(_Tiny())
    assert first["state_tensor_count"] == 3
    assert first["parameter_numel"] == 8
    assert first["buffer_numel"] == 1
    assert first["estimated_parameter_storage_bytes"] == {
        "fp32": 32,
        "fp16_or_bf16": 16,
    }
    assert first["module_class_counts"] == {"Linear": 1, "_Tiny": 1}
    assert first["top_level_state_tensor_counts"] == {"linear": 2, "scale": 1}
    assert [item["key"] for item in first["entries"]] == [
        "scale",
        "linear.weight",
        "linear.bias",
    ]
    assert first["state_structure_sha256"] == second["state_structure_sha256"]


def test_structure_manifest_changes_with_parameter_shape():
    first = audit.state_structure_manifest(nn.Linear(3, 2))
    second = audit.state_structure_manifest(nn.Linear(4, 2))
    assert first["state_structure_sha256"] != second["state_structure_sha256"]


def test_meta_upfirdn_shape_matches_upstream_formula_without_storage():
    value = torch.empty((2, 4, 8, 10), device="meta")
    kernel = torch.empty((4, 4), device="meta")
    output = audit._meta_upfirdn2d(value, kernel, up=2, down=1, pad=(2, 1))
    assert output.device.type == "meta"
    assert output.shape == (2, 4, 16, 20)
