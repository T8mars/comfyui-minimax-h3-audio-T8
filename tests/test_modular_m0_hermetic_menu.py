"""A same-inventory M0 replay may explain drift but cannot excuse regressions."""

from copy import deepcopy

import folder_paths
import pytest

from tools.audit_modular_m0_hermetic_menu import replay
from tools.audit_modular_sampling_compat import SCHEMA


def _snapshots():
    old = {"schema": SCHEMA, "cuda_initialized": False,
           "nodes": [{"id": "old", "source": "old.py",
            "source_sha256": "same", "info": {
                "input_order": {"required": ["lora_name"]},
                "input": {"required": {"lora_name": ["COMBO", {
                    "options": ["None", "old.safetensors"], "default": "None"}]}},
                "output": ["MODEL"], "output_name": ["model"]}}], "files": {}}
    new = deepcopy(old)
    new["nodes"][0]["info"]["input"]["required"]["lora_name"][1]["options"].append(
        "new.safetensors")
    return old, new


def test_m0_hermetic_replay_removes_only_verified_new_inventory(tmp_path, monkeypatch):
    old, new = _snapshots()
    (tmp_path / "new.safetensors").write_bytes(b"new")
    observed = []

    def installed(category):
        assert category == "loras"
        return ["old.safetensors", "new.safetensors"]

    def capture_with_mask():
        observed.extend(folder_paths.get_filename_list("loras"))
        return old

    monkeypatch.setattr(folder_paths, "get_filename_list", installed)
    monkeypatch.setattr("tools.audit_modular_m0_hermetic_menu.capture", capture_with_mask)
    report = replay(old, new, lora_root=tmp_path, graph_root=tmp_path)
    assert report["status"] == "pass_same_inventory_only"
    assert report["live_exact_status"] == "fail"
    assert report["masked_field_count"] == 1
    assert observed == ["old.safetensors"]
    assert folder_paths.get_filename_list is installed


def test_m0_replay_verifies_added_pdd_and_controlnet_without_changing_old_menus(tmp_path, monkeypatch):
    old, _ = _snapshots()
    for node_id, field in (("pdd", "pdd_lora_name"), ("control", "control_net_name")):
        old["nodes"].append({"id": node_id, "source": node_id + ".py", "source_sha256": "same",
                             "info": {"input_order": {"required": [field]},
                                      "input": {"required": {field: ["COMBO", {
                                          "options": ["old.safetensors"]}]}},
                                      "output": ["MODEL"], "output_name": ["model"]}})
    new = deepcopy(old)
    new["nodes"][0]["info"]["input"]["required"]["lora_name"][1]["options"].append("new.safetensors")
    new["nodes"][1]["info"]["input"]["required"]["pdd_lora_name"][1]["options"].append("pdd.safetensors")
    new["nodes"][2]["info"]["input"]["required"]["control_net_name"][1]["options"].insert(
        0, "control.safetensors")
    loras, controls = tmp_path / "loras", tmp_path / "controlnet"
    loras.mkdir()
    controls.mkdir()
    for path in (loras / "new.safetensors", loras / "pdd.safetensors",
                 controls / "control.safetensors"):
        path.write_bytes(b"installed")

    def installed(category):
        if category == "loras":
            return ["old.safetensors", "new.safetensors", "pdd.safetensors"]
        if category == "controlnet":
            return ["control.safetensors", "old.safetensors"]
        raise AssertionError(category)

    def frozen_capture():
        assert folder_paths.get_filename_list("loras") == ["old.safetensors"]
        assert folder_paths.get_filename_list("controlnet") == ["old.safetensors"]
        return old

    monkeypatch.setattr(folder_paths, "get_filename_list", installed)
    monkeypatch.setattr("tools.audit_modular_m0_hermetic_menu.capture", frozen_capture)
    result = replay(old, new, lora_root=loras, controlnet_root=controls, graph_root=tmp_path)
    assert result["status"] == "pass_same_inventory_only"
    assert result["masked_field_count"] == 3
    assert result["masked_old_node_count"] == 3
    assert set(result["masked_installed_additions"]) == {
        "new.safetensors", "pdd.safetensors", "controlnet:control.safetensors"}
    assert result["frozen_inventory_category_calls"] == {"loras": 1, "controlnet": 1}


def test_m0_hermetic_replay_rejects_asset_changed_during_capture(tmp_path, monkeypatch):
    old, new = _snapshots()
    asset = tmp_path / "new.safetensors"
    asset.write_bytes(b"before")
    def original(_category):
        return ["old.safetensors", "new.safetensors"]
    monkeypatch.setattr(folder_paths, "get_filename_list", original)

    def changed_capture():
        assert folder_paths.get_filename_list("loras") == ["old.safetensors"]
        asset.write_bytes(b"after")
        return old

    monkeypatch.setattr("tools.audit_modular_m0_hermetic_menu.capture", changed_capture)
    with pytest.raises(ValueError, match="changed during frozen-inventory replay"):
        replay(old, new, lora_root=tmp_path, graph_root=tmp_path)
    assert folder_paths.get_filename_list is original


@pytest.mark.parametrize("node_id,field,category,directory", [
    ("MiniMaxH3HybridPairInspectorT8Advanced", "quality_base", "diffusion_models", "diffusion_models"),
    ("MiniMaxH3HybridPairInspectorT8Advanced", "reference_overlay", "diffusion_models", "unet"),
    ("MiniMaxH3HybridModelLoaderT8Advanced", "quality_base", "diffusion_models", "diffusion_models"),
    ("MiniMaxH3SPEEDModelVAEFingerprintT8Advanced", "checkpoint_name", "diffusion_models", "diffusion_models"),
    ("MiniMaxH3RavenGuardedLoaderT8Advanced", "unet_name", "diffusion_models", "diffusion_models"),
    ("MiniMaxH3SolEngineTAEHVLoaderT8Advanced", "model_name", "taehv", "vae"),
    ("video", "video_vae_name", "vae", "vae"),
    ("native", "native_video_vae", "vae", "vae"),
    ("control", "control_net_name", "controlnet", "model_patches"),
])
def test_exact_known_categories_and_alternative_installed_roots(tmp_path, monkeypatch, node_id, field, category, directory):
    old, new = _snapshots()
    for snapshot in (old, new):
        node = snapshot["nodes"][0]
        node["id"] = node_id
        node["info"]["input_order"]["required"] = [field]
        spec = node["info"]["input"]["required"].pop("lora_name")
        node["info"]["input"]["required"][field] = spec
    loras = tmp_path / "loras"
    loras.mkdir()
    installed_root = tmp_path / directory
    installed_root.mkdir(exist_ok=True)
    (installed_root / "new.safetensors").write_bytes(b"actual-installed-test-bytes")
    def installed(selected):
        assert selected == category
        return ["old.safetensors", "new.safetensors"]
    def frozen_capture():
        assert folder_paths.get_filename_list(category) == ["old.safetensors"]
        return old
    monkeypatch.setattr(folder_paths, "get_filename_list", installed)
    monkeypatch.setattr("tools.audit_modular_m0_hermetic_menu.capture", frozen_capture)
    report = replay(old, new, lora_root=loras, graph_root=tmp_path)
    assert report["status"] == "pass_same_inventory_only" and report["live_exact_status"] == "fail"
    asset = report["masked_installed_additions"][category + ":new.safetensors"]
    assert asset["path"] == str((installed_root / "new.safetensors").resolve())
    assert folder_paths.get_filename_list is installed


@pytest.mark.parametrize("node_id,field", [("UnknownModel", "model_name"), ("UnknownHybrid", "quality_base"),
                                        ("MiniMaxH3HybridModelLoaderT8Advanced", "reference_overlay")])
def test_generic_or_wrong_node_fields_are_not_normalized(tmp_path, monkeypatch, node_id, field):
    old, new = _snapshots()
    for snapshot in (old, new):
        node = snapshot["nodes"][0]
        node["id"] = node_id
        node["info"]["input_order"]["required"] = [field]
        spec = node["info"]["input"]["required"].pop("lora_name")
        node["info"]["input"]["required"][field] = spec
    (tmp_path / "new.safetensors").write_bytes(b"not-authorized")
    monkeypatch.setattr("tools.audit_modular_m0_hermetic_menu.capture", lambda: pytest.fail("Must not capture"))
    with pytest.raises(ValueError, match="not purely additive"):
        replay(old, new, lora_root=tmp_path, graph_root=tmp_path)


def test_original_folder_menu_restored_if_capture_raises(tmp_path, monkeypatch):
    old, new = _snapshots()
    (tmp_path / "new.safetensors").write_bytes(b"installed")
    def installed(_category):
        return ["old.safetensors", "new.safetensors"]
    def failed_capture():
        assert folder_paths.get_filename_list("loras") == ["old.safetensors"]
        raise RuntimeError("actual capture error")
    monkeypatch.setattr(folder_paths, "get_filename_list", installed)
    monkeypatch.setattr("tools.audit_modular_m0_hermetic_menu.capture", failed_capture)
    with pytest.raises(RuntimeError, match="actual capture error"):
        replay(old, new, lora_root=tmp_path, graph_root=tmp_path)
    assert folder_paths.get_filename_list is installed


@pytest.mark.parametrize("change", ["removed_old", "reordered_old", "changed_default",
                                    "changed_source", "missing_asset"])
def test_m0_hermetic_replay_does_not_hide_other_drift(tmp_path, monkeypatch, change):
    old, new = _snapshots()
    if change != "missing_asset":
        (tmp_path / "new.safetensors").write_bytes(b"new")
    spec = new["nodes"][0]["info"]["input"]["required"]["lora_name"][1]
    if change == "removed_old":
        spec["options"].remove("old.safetensors")
    elif change == "reordered_old":
        spec["options"].reverse()
    elif change == "changed_default":
        spec["default"] = "new.safetensors"
    elif change == "changed_source":
        new["nodes"][0]["source_sha256"] = "other"
    monkeypatch.setattr("tools.audit_modular_m0_hermetic_menu.capture",
                        lambda: pytest.fail("Invalid live drift must be rejected before replay"))
    with pytest.raises(ValueError):
        replay(old, new, lora_root=tmp_path, graph_root=tmp_path)
