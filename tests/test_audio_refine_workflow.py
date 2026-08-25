from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOL = PROJECT_ROOT / "tools" / "build_audio_refine_workflow.py"
WORKFLOW = (
    PROJECT_ROOT
    / "examples"
    / "workflows"
    / "18-audio-refine"
    / "2026-08-26_H3_Audio_Refine_Turbo4_Plus_Refine4_Advanced_EXP.json"
)


def _load_tool():
    tools_root = str(TOOL.parent)
    if tools_root not in sys.path:
        sys.path.insert(0, tools_root)
    spec = importlib.util.spec_from_file_location("build_audio_refine_workflow", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_audio_refine_api_source_uses_one_native_four_plus_four_route():
    prompt = _load_tool().build_prompt()

    samplers = [
        node for node in prompt.values() if node["class_type"] == "SamplerCustomAdvanced"
    ]
    assert len(samplers) == 2
    assert prompt["6"]["inputs"]["width"] == 1056
    assert prompt["6"]["inputs"]["height"] == 608
    assert prompt["6"]["inputs"]["length"] == 124
    assert prompt["7"]["inputs"]["steps"] == 4
    assert prompt["15"]["inputs"]["refine_steps"] == 4
    assert prompt["15"]["inputs"]["audio_denoise"] == 0.5
    assert prompt["24"]["inputs"]["accept_candidate"] is False


def test_audio_refine_workflow_is_frontend_importable_and_documented():
    workflow = json.loads(WORKFLOW.read_text(encoding="utf-8"))

    assert workflow["version"] == 0.4
    assert isinstance(workflow["nodes"], list)
    assert isinstance(workflow["links"], list)
    assert len([node for node in workflow["nodes"] if node["type"] == "MarkdownNote"]) >= 6
    assert not any("NaN" in json.dumps(node) for node in workflow["nodes"])
    node_ids = {node["id"] for node in workflow["nodes"]}
    link_ids = {link[0] for link in workflow["links"]}
    assert workflow["last_node_id"] == max(node_ids)
    assert workflow["last_link_id"] == max(link_ids)
    assert all(link[1] in node_ids and link[3] in node_ids for link in workflow["links"])
    gate = next(
        node
        for node in workflow["nodes"]
        if node["type"] == "MiniMaxH3AudioRefineQualityGateT8Advanced"
    )
    assert gate["widgets_values"][0] is False
    assert sum(node["type"] == "SaveVideo" for node in workflow["nodes"]) == 3
