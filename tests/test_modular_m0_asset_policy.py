"""A verified asset addition is a separate M0 decision, never an exact pass."""

from copy import deepcopy

import folder_paths

from test_modular_m0_hermetic_menu import _snapshots
from tools.audit_modular_m0_asset_policy import decide


def test_m0_policy_reports_actual_exact_pass_without_replay(tmp_path, monkeypatch):
    old, _ = _snapshots()
    monkeypatch.setattr("tools.audit_modular_m0_asset_policy.replay",
                        lambda *args, **kwargs: None)
    result = decide(old, deepcopy(old), lora_root=tmp_path, graph_root=tmp_path)
    assert result["status"] == "pass_exact"
    assert result["exact_status"] == "pass" and result["exact_violations"] == []
    assert result["baseline_sha256"] == result["live_sha256"]


def test_m0_policy_accepts_only_verified_installed_menu_addition(tmp_path, monkeypatch):
    old, new = _snapshots()
    (tmp_path / "new.safetensors").write_bytes(b"installed")
    monkeypatch.setattr(folder_paths, "get_filename_list",
                        lambda category: ["old.safetensors", "new.safetensors"]
                        if category == "loras" else [])
    def frozen_capture():
        assert folder_paths.get_filename_list("loras") == ["old.safetensors"]
        return old

    monkeypatch.setattr("tools.audit_modular_m0_hermetic_menu.capture", frozen_capture)
    result = decide(old, new, lora_root=tmp_path, graph_root=tmp_path)
    assert result["status"] == "pass_verified_asset_additions", result["reason"]
    assert result["exact_status"] == "fail"
    assert result["hermetic"]["hermetic_strict"]["status"] == "pass"
    assert result["hermetic"]["masked_old_node_count"] == 1


def test_m0_policy_does_not_excuse_schema_or_asset_drift(tmp_path, monkeypatch):
    old, new = _snapshots()
    (tmp_path / "new.safetensors").write_bytes(b"installed")
    new["nodes"][0]["info"]["output_name"] = ["changed"]
    def rejected_replay(*args, **kwargs):
        raise ValueError("Old output changed")

    monkeypatch.setattr("tools.audit_modular_m0_asset_policy.replay",
                        rejected_replay)
    result = decide(old, new, lora_root=tmp_path, graph_root=tmp_path)
    assert result["status"] == "fail" and result["exact_status"] == "fail"
    assert "Old output changed" in result["reason"]


def test_m0_policy_requires_cpu_live_capture(tmp_path):
    old, _ = _snapshots()
    live = deepcopy(old)
    live["cuda_initialized"] = True
    result = decide(old, live, lora_root=tmp_path, graph_root=tmp_path)
    assert result["status"] == "fail"
