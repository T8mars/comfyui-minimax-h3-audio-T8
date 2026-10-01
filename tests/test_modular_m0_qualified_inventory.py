"""Qualified M0 menu replay must remain fail-closed and restore live menus."""
from copy import deepcopy

import folder_paths
import pytest

from tools.audit_modular_m0_qualified_inventory import replay
from tools.audit_modular_sampling_compat import SCHEMA


def _fixture(tmp_path):
    old = {"schema": SCHEMA, "cuda_initialized": False, "files": {},
           "nodes": [{"id": "old", "source": "old.py", "source_sha256": "old-sha",
                      "info": {"input_order": {"required": ["lora_name"]},
                               "input": {"required": {"lora_name": ["COMBO", {
                                   "options": ["None", "old.safetensors"],
                                   "default": "None"}]}},
                               "output": ["MODEL"], "output_name": ["model"]}}]}
    new = deepcopy(old)
    new["nodes"][0]["source_sha256"] = "new-sha"
    new["nodes"][0]["info"]["input"]["required"]["lora_name"][1]["options"].append(
        "new.safetensors")
    loras = tmp_path / "loras"
    loras.mkdir()
    (loras / "new.safetensors").write_bytes(b"asset")
    roots = {"preimage_root": tmp_path, "lora_root": loras,
             "controlnet_root": tmp_path / "controlnet",
             "model_patches_root": tmp_path / "model_patches",
             "vae_root": tmp_path / "vae", "taehv_root": tmp_path / "taehv",
             "graph_root": tmp_path}
    return old, new, roots


def _source(_old, _new, _root):
    return {"affected_old_nodes": ["old"],
            "status": "pass_textual_additions_only_not_runtime_parity"}


def test_only_installed_menu_is_temporarily_masked(tmp_path, monkeypatch):
    old, new, roots = _fixture(tmp_path)

    def installed(category):
        return ["old.safetensors", "new.safetensors"] if category == "loras" else []

    monkeypatch.setattr(folder_paths, "get_filename_list", installed)
    monkeypatch.setattr("tools.audit_modular_m0_qualified_inventory.source_audit", _source)

    def frozen_capture():
        assert folder_paths.get_filename_list("loras") == ["old.safetensors"]
        return new | {"nodes": [{**new["nodes"][0], "info": old["nodes"][0]["info"]}]}

    monkeypatch.setattr("tools.audit_modular_m0_qualified_inventory.capture", frozen_capture)
    report = replay(old, new, **roots)
    assert report["menu_fields"] == 1
    assert report["same_inventory_exact_status"] == "fail"
    assert report["same_inventory_remaining_violations"] == [
        {"kind": "old_source_changed_requires_numerical_review", "id": "old"}]
    assert report["removed_filenames"]["loras"] == ["new.safetensors"]
    assert folder_paths.get_filename_list is installed


@pytest.mark.parametrize("change", ["default", "source_id", "missing_asset"])
def test_unrelated_live_change_or_missing_asset_fails_before_capture(tmp_path, monkeypatch, change):
    old, new, roots = _fixture(tmp_path)
    if change == "default":
        new["nodes"][0]["info"]["input"]["required"]["lora_name"][1]["default"] = "new.safetensors"
    elif change == "source_id":
        monkeypatch.setattr("tools.audit_modular_m0_qualified_inventory.source_audit",
                            lambda *_: {"affected_old_nodes": ["different"]})
    else:
        (roots["lora_root"] / "new.safetensors").unlink()
    if change != "source_id":
        monkeypatch.setattr("tools.audit_modular_m0_qualified_inventory.source_audit", _source)
    monkeypatch.setattr("tools.audit_modular_m0_qualified_inventory.capture",
                        lambda: pytest.fail("Invalid live drift must not reach capture"))
    with pytest.raises(ValueError):
        replay(old, new, **roots)


def test_asset_change_during_capture_fails_and_restores_menu(tmp_path, monkeypatch):
    old, new, roots = _fixture(tmp_path)

    def installed(category):
        return ["old.safetensors", "new.safetensors"] if category == "loras" else []

    monkeypatch.setattr(folder_paths, "get_filename_list", installed)
    monkeypatch.setattr("tools.audit_modular_m0_qualified_inventory.source_audit", _source)

    def changed_capture():
        assert folder_paths.get_filename_list("loras") == ["old.safetensors"]
        (roots["lora_root"] / "new.safetensors").write_bytes(b"changed-size")
        return new

    monkeypatch.setattr("tools.audit_modular_m0_qualified_inventory.capture", changed_capture)
    with pytest.raises(ValueError, match="changed during CPU replay"):
        replay(old, new, **roots)
    assert folder_paths.get_filename_list is installed


def test_live_cuda_snapshot_is_rejected_before_capture(tmp_path, monkeypatch):
    old, new, roots = _fixture(tmp_path)
    new["cuda_initialized"] = True
    monkeypatch.setattr("tools.audit_modular_m0_qualified_inventory.capture",
                        lambda: pytest.fail("CUDA live snapshot must not reach capture"))
    with pytest.raises(ValueError, match="CPU-only mode"):
        replay(old, new, **roots)


def test_capture_cuda_flag_is_rejected_and_restores_menu(tmp_path, monkeypatch):
    old, new, roots = _fixture(tmp_path)

    def installed(category):
        return ["old.safetensors", "new.safetensors"] if category == "loras" else []

    monkeypatch.setattr(folder_paths, "get_filename_list", installed)
    monkeypatch.setattr("tools.audit_modular_m0_qualified_inventory.source_audit", _source)

    def unsafe_capture():
        assert folder_paths.get_filename_list("loras") == ["old.safetensors"]
        return {**new, "cuda_initialized": True}

    monkeypatch.setattr("tools.audit_modular_m0_qualified_inventory.capture", unsafe_capture)
    with pytest.raises(ValueError, match="initialized CUDA"):
        replay(old, new, **roots)
    assert folder_paths.get_filename_list is installed


def test_exact_old_sources_need_no_addition_exception(tmp_path, monkeypatch):
    old, new, roots = _fixture(tmp_path)
    new["nodes"][0]["source_sha256"] = "old-sha"
    roots["preimage_root"] = None
    monkeypatch.setattr("tools.audit_modular_m0_qualified_inventory.source_audit",
                        lambda *_: pytest.fail("Exact sources must not use an addition waiver"))

    def installed(category):
        return ["old.safetensors", "new.safetensors"] if category == "loras" else []

    monkeypatch.setattr(folder_paths, "get_filename_list", installed)

    def frozen_capture():
        assert folder_paths.get_filename_list("loras") == ["old.safetensors"]
        return deepcopy(old)

    monkeypatch.setattr("tools.audit_modular_m0_qualified_inventory.capture", frozen_capture)
    report = replay(old, new, **roots)
    assert report["live_exact_status"] == "fail"
    assert report["same_inventory_exact_status"] == "pass"
    assert report["same_inventory_remaining_violations"] == []
    assert report["source_bound_old_node_ids"] == []
    assert report["direct_old_node_sources_exact"] is True
    assert report["status"] == "pass_old_schema_and_direct_source_same_inventory"
    assert folder_paths.get_filename_list is installed


@pytest.mark.parametrize("drift", ["source", "schema", "file", "registry", "cuda"])
def test_restored_sources_cannot_hide_second_capture_drift(tmp_path, monkeypatch, drift):
    old, new, roots = _fixture(tmp_path)
    new["nodes"][0]["source_sha256"] = "old-sha"
    roots["preimage_root"] = None

    def installed(category):
        return ["old.safetensors", "new.safetensors"] if category == "loras" else []

    monkeypatch.setattr(folder_paths, "get_filename_list", installed)
    if drift == "file":
        old["files"]["old.json"] = new["files"]["old.json"] = "frozen-sha"
        monkeypatch.setattr("tools.audit_modular_m0_qualified_inventory.audit_saved_menu_selections",
                            lambda *_: {"regression_status": "pass", "counts": {}})

    def changed_capture():
        assert folder_paths.get_filename_list("loras") == ["old.safetensors"]
        result = deepcopy(old)
        if drift == "source":
            result["nodes"][0]["source_sha256"] = "unexpected-sha"
        elif drift == "schema":
            result["nodes"][0]["info"]["output"] = ["STRING"]
        elif drift == "file":
            result["files"]["old.json"] = "changed-sha"
        elif drift == "registry":
            result["nodes"][0]["id"] = "unexpected-node"
        else:
            result["cuda_initialized"] = True
        return result

    monkeypatch.setattr("tools.audit_modular_m0_qualified_inventory.capture", changed_capture)
    with pytest.raises(ValueError, match="unexpected old schema/file drift|initialized CUDA"):
        replay(old, new, **roots)
    assert folder_paths.get_filename_list is installed


def test_changed_sources_without_preimage_cannot_claim_restoration(tmp_path, monkeypatch):
    old, new, roots = _fixture(tmp_path)
    roots["preimage_root"] = None
    monkeypatch.setattr("tools.audit_modular_m0_qualified_inventory.capture",
                        lambda: pytest.fail("Unknown source must not reach capture"))
    with pytest.raises(ValueError, match="require an exact frozen preimage"):
        replay(old, new, **roots)
