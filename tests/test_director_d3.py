from __future__ import annotations

import pytest

from h3_audio_t8_pkg.director_d3 import (
    export_d3_package,
    handoff_d3_route,
    inspect_d3_routes,
)
from h3_audio_t8_pkg.director_capabilities import CAPABILITY_SPECS
from h3_audio_t8_pkg.director_project import new_project


def test_d3_preflight_reports_ready_native_handoff_with_dependencies():
    nodes = {node for spec in CAPABILITY_SPECS for node in spec["entry_nodes"]}
    result = inspect_d3_routes(
        node_ids=nodes,
        model_inventory={
            "semantic_bridge": ["t8_compat/semantic_bridge.safetensors"],
            "diffusion_models": ["FastH3-v2.safetensors"],
            "loras": [],
            "vae": [],
            "meridian": ["meridian_dmd_int8_convrot_comfy.safetensors"],
        },
    )
    assert result["schema"] == "t8.minimax_h3.director_d3_preflight.v1"
    rows = {item["id"]: item for item in result["capabilities"]}
    assert rows["semantic_bridge"]["state"] == "ready_for_native_workflow"
    assert rows["fast_h3_v2"]["state"] == "ready_for_native_workflow"
    assert rows["meridian"]["state"] == "ready_for_native_workflow"
    assert rows["semantic_bridge"]["workflows"]
    assert rows["low_vram"]["dependencies"][0]["ok"] is True


def test_d3_preflight_fails_closed_when_entry_or_model_is_missing():
    result = inspect_d3_routes(
        node_ids={"MiniMaxH3TopazEnvironmentEXPT8"},
        model_inventory={"semantic_bridge": [], "diffusion_models": [], "loras": [], "vae": [], "meridian": []},
        capability="semantic_bridge",
    )
    row = result["capabilities"][0]
    assert row["state"] == "blocked_missing_entry_or_dependency"
    assert "MiniMaxH3SemanticBridgeConfigT8" in row["missing_entry_nodes"]
    assert row["dependencies"][0]["ok"] is False


def test_d3_handoff_returns_exact_allowlisted_workflow_without_queueing():
    result = handoff_d3_route("fast_h3_v2")
    assert result["schema"] == "t8.minimax_h3.director_d3_handoff.v1"
    assert result["kind"] == "workflow"
    assert result["selected"].endswith("FastH3_V2_Trained_VSA_73f_h1c1_EXP.json")
    assert result["content"]["nodes"]
    assert result["text"].lstrip().startswith("{")
    assert all(item["exists"] for item in result["files"] if item["kind"] == "workflow")
    assert '/34-fasth3-v2/' in result['selected']
    legacy = handoff_d3_route('fast_h3_v2', result['selected'].replace('/34-fasth3-v2/', '/10-speed/'))
    assert legacy['sha256'] == result['sha256'] and legacy['selected'] == result['selected']


def test_d3_exposes_new_split_suite_without_changing_old_default():
    default = handoff_d3_route("fast_h3_v2")
    assert default["selected"].endswith("FastH3_V2_Trained_VSA_73f_h1c1_EXP.json")
    split = next(item["path"] for item in default["files"]
                 if item["name"] == "FastH3_V2_Split_01_First_LOW_HIGH_Relay_EAV_EXP.json")
    selected = handoff_d3_route("fast_h3_v2", split)
    assert selected["content"]["extra"]["t8_split_example"]["kind"] == "first"
    assert selected["sha256"] != default["sha256"]
    assert sum("FastH3_V2_Split_" in item["name"] for item in default["files"]) == 6


def test_d3_handoff_rejects_path_injection_and_unknown_route():
    with pytest.raises(ValueError, match="未知 D3"):
        handoff_d3_route("../../etc")
    with pytest.raises(ValueError, match="不属于"):
        handoff_d3_route("fast_h3_v2", "README.md")


def test_d3_route_package_preserves_project_and_native_files():
    project = new_project()
    project["title"] = "路线包回归"
    result = export_d3_package("topaz", project, project["current"])
    assert result["schema"] == "t8.minimax_h3.director_d3_route_package.v1"
    assert result["project"]["id"] == project["id"]
    assert result["selected_shot"] == project["current"]
    assert result["contract"]["mode"] == "native_only"
    assert any(item["kind"] == "workflow" for item in result["files"])
    assert all(item["text"] for item in result["files"])


def test_d3_route_package_rejects_unknown_route():
    with pytest.raises(ValueError, match="未知 D3"):
        export_d3_package("not-a-route")


def test_every_native_route_has_actionable_contract_and_package_files():
    project = new_project()
    for capability in (item["id"] for item in CAPABILITY_SPECS):
        result = export_d3_package(capability, project, project["current"])
        contract = result["contract"]
        assert contract["mode"] == "native_only"
        assert contract["steps"] and contract["requires"]
        assert result["files"]
