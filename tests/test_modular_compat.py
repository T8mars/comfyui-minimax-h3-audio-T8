"""Do not relax the legacy guard merely because new nodes are appended."""
from copy import deepcopy
import hashlib
import json
import sys
import types

import pytest

from tools.audit_modular_sampling_compat import (
    SCHEMA, compare, snapshot_runtime_sources, write_new,
)
from tools.diagnose_modular_menu_drift import audit_saved_menu_selections, diagnose


def snapshot():
    return {"schema": SCHEMA, "nodes": [{"id": "old", "source": "old.py",
            "source_sha256": "abc", "info": {"input_order": {"required": ["model", "steps"]},
            "input": {"required": {"model": ["MODEL"], "steps": ["INT", {"default": 8}]}},
            "output": ["LATENT"], "output_name": ["av_latent"]}}],
            "files": {"examples/old.json": "abc", "subgraphs/user.json": "def"}}


def test_new_nodes_and_new_graphs_do_not_require_migrating_old_ones():
    baseline = snapshot()
    current = deepcopy(baseline)
    current["nodes"].append({**deepcopy(current["nodes"][0]), "id": "new"})
    current["files"]["examples/new.json"] = "new"
    result = compare(baseline, current)
    assert result["status"] == "pass"
    assert result["added_nodes"] == ["new"]
    assert result["added_json"] == ["examples/new.json"]
    assert result["runtime_source_coverage"] == "unbaselined"
    assert baseline == snapshot()


def test_runtime_source_closure_includes_indirect_helper(monkeypatch, tmp_path):
    source = tmp_path / "h3_t8" / "native_latent_timeline_advanced.py"
    source.parent.mkdir()
    source.write_text("# synthetic imported helper\n", encoding="utf-8")
    package = "_t8_modular_compat_capture"
    helper = types.ModuleType(f"{package}.native_latent_timeline_advanced")
    helper.__file__ = str(source)
    monkeypatch.setitem(sys.modules, helper.__name__, helper)
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    assert snapshot_runtime_sources(tmp_path) == {
        "h3_t8/native_latent_timeline_advanced.py": expected,
    }


def test_rolling_runtime_source_baseline_rejects_changed_or_missing_helper():
    baseline = snapshot()
    baseline["runtime_sources"] = {"h3_t8/native_latent_timeline_advanced.py": "old"}
    current = deepcopy(baseline)
    current["runtime_sources"]["h3_t8/native_latent_timeline_advanced.py"] = "new"
    report = compare(baseline, current)
    assert report["status"] == "fail"
    assert report["runtime_source_coverage"] == "compared"
    assert report["runtime_source_changes_requiring_review"] == [
        "h3_t8/native_latent_timeline_advanced.py"]
    assert {item["kind"] for item in report["violations"]} == {
        "runtime_source_changed_requires_review"}

    del current["runtime_sources"]["h3_t8/native_latent_timeline_advanced.py"]
    assert compare(baseline, current)["status"] == "fail"
    del current["runtime_sources"]
    assert {item["kind"] for item in compare(baseline, current)["violations"]} == {
        "runtime_source_capture_missing"}


def test_rolling_runtime_source_baseline_allows_new_source_without_rewriting_old():
    baseline = snapshot()
    baseline["runtime_sources"] = {"h3_t8/old_helper.py": "same"}
    current = deepcopy(baseline)
    current["runtime_sources"]["h3_t8/new_helper.py"] = "new"
    report = compare(baseline, current)
    assert report["status"] == "pass"
    assert report["added_runtime_sources"] == ["h3_t8/new_helper.py"]
    assert report["runtime_source_changes_requiring_review"] == []


@pytest.mark.parametrize("mutation", ["default", "type", "input_order", "output_slot", "remove_node",
                                     "reorder_nodes", "duplicate", "rewrite_graph", "delete_graph"])
def test_legacy_breakages_are_never_silently_accepted(mutation):
    baseline, current = snapshot(), snapshot()
    info = current["nodes"][0]["info"]
    if mutation == "default":
        info["input"]["required"]["steps"][1]["default"] = 4
    elif mutation == "type":
        info["input"]["required"]["model"] = ["STRING"]
    elif mutation == "input_order":
        info["input_order"]["required"].reverse()
    elif mutation == "output_slot":
        info["output"].insert(0, "MODEL")
    elif mutation == "remove_node":
        current["nodes"].clear()
    elif mutation == "reorder_nodes":
        current["nodes"].insert(0, {**deepcopy(current["nodes"][0]), "id": "new"})
    elif mutation == "duplicate":
        current["nodes"].append(deepcopy(current["nodes"][0]))
    elif mutation == "rewrite_graph":
        current["files"]["examples/old.json"] = "changed"
    elif mutation == "delete_graph":
        del current["files"]["subgraphs/user.json"]
    assert compare(baseline, current)["status"] == "fail"


@pytest.mark.parametrize("field, value", [("source_sha256", "changed"), ("source", "relocated.py")])
def test_source_change_is_visible_without_claiming_numerical_equivalence(field, value):
    baseline, current = snapshot(), snapshot()
    current["nodes"][0][field] = value
    report = compare(baseline, current)
    assert report["status"] == "fail"
    assert report["source_changes_requiring_numerical_review"] == ["old"]
    assert {item["kind"] for item in report["violations"]} == {
        "old_source_changed_requires_numerical_review"}
    assert "runtime" in report["qualification"]


def test_evidence_is_append_only(tmp_path):
    target = tmp_path / "baseline.json"
    write_new(target, snapshot())
    original = target.read_bytes()
    with pytest.raises(FileExistsError):
        write_new(target, {})
    assert target.read_bytes() == original


def _menu_snapshot():
    value = snapshot()
    value["nodes"][0]["info"]["input"]["required"]["lora_name"] = [
        "COMBO", {"options": ["None", "old.safetensors"],
                  "default": "None", "multiselect": False}]
    return value


def test_dynamic_installed_lora_menu_is_diagnosed_without_passing_exact_gate(tmp_path):
    baseline, current = _menu_snapshot(), _menu_snapshot()
    current["nodes"][0]["info"]["input"]["required"]["lora_name"][1]["options"].append(
        "new.safetensors")
    (tmp_path / "new.safetensors").write_bytes(b"fixture")
    assert compare(baseline, current)["status"] == "fail"
    report = diagnose(baseline, current, lora_root=tmp_path)
    assert report["exact_compatibility_status"] == "fail"
    assert report["structural_status"] == "pass"
    assert report["menu_additions"] == [{"id": "old", "field": "lora_name",
                                         "category": "loras",
                                         "old_count": 2, "new_count": 3,
                                         "added": ["new.safetensors"]}]
    assert baseline == _menu_snapshot()


def test_installed_hypervae_expands_only_specific_legacy_vae_menus(tmp_path):
    baseline = snapshot()
    baseline["nodes"] = []
    cases = (
        ("MiniMaxH3SPEEDModelVAEFingerprintT8Advanced", "video_vae_name"),
        ("MiniMaxH3TRTVAEDecoderEXPT8", "native_video_vae"),
        ("MiniMaxH3SolEngineTAEHVLoaderT8Advanced", "model_name"),
    )
    for node_id, field in cases:
        baseline["nodes"].append({"id": node_id, "source": "old.py",
                                  "source_sha256": "same", "info": {
                                      "input_order": {"required": [field]},
                                      "input": {"required": {field: ["COMBO", {
                                          "options": ["old.safetensors"],
                                          "default": "old.safetensors"}]}},
                                      "output": ["MODEL"], "output_name": ["model"]}})
    current = deepcopy(baseline)
    fields = dict(cases)
    for node in current["nodes"]:
        node["info"]["input"]["required"][fields[node["id"]]][1][
            "options"].append("hyper.safetensors")
    vae = tmp_path / "vae"
    vae.mkdir()
    (vae / "hyper.safetensors").write_bytes(b"installed")
    report = diagnose(baseline, current, lora_root=tmp_path / "loras")
    assert report["exact_compatibility_status"] == "fail"
    assert report["structural_status"] == "pass"
    assert {(item["id"], item["field"], item["category"])
            for item in report["menu_additions"]} == {
                (cases[0][0], cases[0][1], "vae"),
                (cases[1][0], cases[1][1], "vae"),
                (cases[2][0], cases[2][1], "taehv"),
            }
    (vae / "hyper.safetensors").unlink()
    assert diagnose(baseline, current, lora_root=tmp_path / "loras")[
        "structural_status"] == "fail"


def test_unrelated_model_name_drift_is_not_excused_as_taehv(tmp_path):
    baseline = snapshot()
    baseline["nodes"][0]["info"]["input"]["required"]["model_name"] = [
        "COMBO", {"options": ["old.safetensors"]}]
    current = deepcopy(baseline)
    current["nodes"][0]["info"]["input"]["required"]["model_name"][1][
        "options"].append("new.safetensors")
    vae = tmp_path / "vae"
    vae.mkdir()
    (vae / "new.safetensors").write_bytes(b"installed")
    assert diagnose(baseline, current, lora_root=tmp_path / "loras")[
        "structural_status"] == "fail"


@pytest.mark.parametrize("mutation", ["missing_file", "removed_old", "reordered_old",
                                      "default_changed", "unrelated_schema", "old_json_changed",
                                      "old_source_changed"])
def test_menu_diagnosis_never_excuses_real_compatibility_breakage(tmp_path, mutation):
    baseline, current = _menu_snapshot(), _menu_snapshot()
    options = current["nodes"][0]["info"]["input"]["required"]["lora_name"][1]["options"]
    options.append("new.safetensors")
    if mutation != "missing_file":
        (tmp_path / "new.safetensors").write_bytes(b"fixture")
    if mutation == "removed_old":
        options.remove("old.safetensors")
    elif mutation == "reordered_old":
        options[:] = list(reversed(options))
    elif mutation == "default_changed":
        current["nodes"][0]["info"]["input"]["required"]["lora_name"][1]["default"] = "new.safetensors"
    elif mutation == "unrelated_schema":
        current["nodes"][0]["info"]["output_name"] = ["changed"]
    elif mutation == "old_json_changed":
        current["files"]["examples/old.json"] = "changed"
    elif mutation == "old_source_changed":
        current["nodes"][0]["source_sha256"] = "changed"
    report = diagnose(baseline, current, lora_root=tmp_path)
    assert report["exact_compatibility_status"] == "fail"
    assert report["structural_status"] == "fail"
    assert report["issues"]


def test_saved_old_menu_choices_are_checked_in_frontend_and_api_graphs(tmp_path):
    baseline = _menu_snapshot()
    info = baseline["nodes"][0]["info"]
    info["input_order"]["required"] = ["lora_name"]
    info["input"]["required"] = {"lora_name": info["input"]["required"]["lora_name"]}
    current = deepcopy(baseline)
    current["nodes"][0]["info"]["input"]["required"]["lora_name"][1]["options"].append(
        "new.safetensors")
    graphs = {
        "frontend.json": {"nodes": [{"id": 1, "type": "old",
                                     "widgets_values": ["old.safetensors"]}], "links": []},
        "api.json": {"1": {"class_type": "old", "inputs": {"lora_name": "old.safetensors"}}},
    }
    baseline["files"] = {}
    for name, graph in graphs.items():
        path = tmp_path / name
        path.write_text(json.dumps(graph), encoding="utf8")
        baseline["files"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    findings = [{"id": "old", "field": "lora_name"}]
    report = audit_saved_menu_selections(tmp_path, baseline, current, findings)
    assert report["status"] == report["regression_status"] == "pass"
    assert report["counts"]["checked_values"] == 2
    assert report["counts"]["frontend_instances"] == report["counts"]["api_instances"] == 1

    graphs["frontend.json"]["nodes"][0]["widgets_values"] = ["requires_external.safetensors"]
    path = tmp_path / "frontend.json"
    path.write_text(json.dumps(graphs["frontend.json"]), encoding="utf8")
    baseline["files"]["frontend.json"] = hashlib.sha256(path.read_bytes()).hexdigest()
    report = audit_saved_menu_selections(tmp_path, baseline, current, findings)
    assert report["status"] == "fail"
    assert report["regression_status"] == "pass"
    assert report["counts"]["preexisting_unavailable_values"] == 1
    assert report["issues"][0]["kind"] == "saved_choice_absent_from_frozen_menu"

    baseline["nodes"][0]["info"]["input"]["required"]["lora_name"][1]["options"].append(
        "requires_external.safetensors")
    report = audit_saved_menu_selections(tmp_path, baseline, current, findings)
    assert report["regression_status"] == "fail"
    assert report["counts"]["current_menu_regressions"] == 1
    assert report["issues"][0]["kind"] == "old_saved_choice_lost_in_current_menu"

    path.write_text("{}", encoding="utf8")
    report = audit_saved_menu_selections(tmp_path, baseline, current, findings)
    assert report["regression_status"] == "fail"
    assert any(issue["kind"] == "frozen_file_sha_changed" for issue in report["issues"])
