from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOL = PROJECT_ROOT / "tools" / "build_audio_refine_phase2_workflows.py"
WORKFLOW_ROOT = PROJECT_ROOT / "examples" / "workflows" / "18-audio-refine"
EXPECTED = {
    "same_turbo_stack": (
        "2026-08-26_H3_Audio_Refine_Phase2_Same_Turbo4_Advanced_EXP.json"
    ),
    "base_without_turbo": (
        "2026-08-26_H3_Audio_Refine_Phase2_Base_Refine4_Advanced_EXP.json"
    ),
    "base_ordinary8": (
        "2026-08-26_H3_Audio_Refine_Phase2_Base_Ordinary8_Control_Advanced_EXP.json"
    ),
}


def _load_tool():
    tools_root = str(TOOL.parent)
    if tools_root not in sys.path:
        sys.path.insert(0, tools_root)
    spec = importlib.util.spec_from_file_location(
        "build_audio_refine_phase2_workflows", TOOL
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_phase2_api_sources_are_three_independent_low_load_routes():
    prompts = _load_tool().build_prompts(seed=2608260404)

    assert set(prompts) == set(EXPECTED)
    same = prompts["same_turbo_stack"]
    base = prompts["base_without_turbo"]
    ordinary = prompts["base_ordinary8"]
    assert sum(
        node["class_type"] == "SamplerCustomAdvanced" for node in same.values()
    ) == 2
    assert sum(
        node["class_type"] == "SamplerCustomAdvanced" for node in base.values()
    ) == 2
    assert sum(
        node["class_type"] == "SamplerCustomAdvanced" for node in ordinary.values()
    ) == 1
    same_route = next(
        node
        for node in same.values()
        if node["class_type"] == "MiniMaxH3AudioRefineModelRouteT8Advanced"
    )
    base_route = next(
        node
        for node in base.values()
        if node["class_type"] == "MiniMaxH3AudioRefineModelRouteT8Advanced"
    )
    assert same_route["inputs"]["route_strategy"] == "same_turbo_stack"
    assert base_route["inputs"]["route_strategy"] == "base_without_turbo"
    ordinary_sampler = next(
        node
        for node in ordinary.values()
        if node["class_type"] == "MiniMaxH3DualClockSamplerT8"
    )
    assert ordinary_sampler["inputs"]["steps"] == 8
    assert ordinary_sampler["inputs"]["model"] == ["6", 0]


def test_phase2_frontend_workflows_are_importable_and_explain_the_four_arm_set():
    for strategy, filename in EXPECTED.items():
        workflow = json.loads((WORKFLOW_ROOT / filename).read_text(encoding="utf-8"))
        assert workflow["version"] == 0.4
        assert isinstance(workflow["nodes"], list)
        assert isinstance(workflow["links"], list)
        assert len(
            [node for node in workflow["nodes"] if node["type"] == "MarkdownNote"]
        ) >= 5
        assert not any("NaN" in json.dumps(node) for node in workflow["nodes"])
        ids = {node["id"] for node in workflow["nodes"]}
        assert workflow["last_node_id"] == max(ids)
        assert all(link[1] in ids and link[3] in ids for link in workflow["links"])
        note_text = "\n".join(
            str(node.get("widgets_values", [""])[0])
            for node in workflow["nodes"]
            if node["type"] == "MarkdownNote"
        )
        assert "四臂" in note_text
        assert "训练分布" in note_text
        if strategy != "base_ordinary8":
            gate = next(
                node
                for node in workflow["nodes"]
                if node["type"] == "MiniMaxH3AudioRefineQualityGateT8Advanced"
            )
            assert gate["widgets_values"][0] is False

