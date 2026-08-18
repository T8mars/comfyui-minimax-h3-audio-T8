from __future__ import annotations

import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / "examples" / "workflows"
FILES = (
    "2026-08-19_H3_SPEED_I2VA_Lock_Stock20_Advanced_EXP.json",
    "2026-08-19_H3_SPEED_FL2VA_Remix_Stock20_Advanced_EXP.json",
    "2026-08-19_H3_SPEED_L2VA_Native_Stock20_Advanced_EXP.json",
    "2026-08-19_H3_SPEED_RefVideoAudio_Stock20_Advanced_EXP.json",
    "2026-08-19_H3_SPEED_Hybrid_FirstImageAudio_Stock20_Advanced_EXP.json",
    "2026-08-19_H3_SPEED_T2VA_Turbo8_Advanced_EXP.json",
)


@pytest.mark.parametrize("filename", FILES)
def test_extended_speed_workflow_is_frontend_importable_and_has_multiple_notes(filename):
    workflow = json.loads((WORKFLOWS / filename).read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in workflow["nodes"]}
    assert workflow["version"] == 0.4
    assert len(nodes) == len(workflow["nodes"])
    assert workflow["last_node_id"] == max(nodes)
    assert sum(node["type"] == "MarkdownNote" for node in nodes.values()) >= 3
    assert any(node["type"] == "MiniMaxH3SPEEDPlanT8Advanced" for node in nodes.values())
    assert any(node["type"] == "MiniMaxH3SPEEDSamplerT8Advanced" for node in nodes.values())
    links = {link[0]: link for link in workflow["links"]}
    assert len(links) == len(workflow["links"])
    for link_id, origin, origin_slot, target, target_slot, link_type in workflow["links"]:
        assert origin in nodes and target in nodes
        assert link_id in (nodes[origin]["outputs"][origin_slot].get("links") or [])
        assert nodes[target]["inputs"][target_slot]["link"] == link_id
        assert link_type


def test_turbo8_example_keeps_old_scope_options_and_selects_new_appended_scope():
    workflow = json.loads((WORKFLOWS / FILES[-1]).read_text(encoding="utf-8"))
    sampler = next(
        node
        for node in workflow["nodes"]
        if node["type"] == "MiniMaxH3SPEEDSamplerT8Advanced"
    )
    assert sampler["widgets_values"][3] == "turbo8_t2va_research_exp"
