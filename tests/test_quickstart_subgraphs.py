from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tools.build_quickstart_subgraphs import SPECS, build_subgraph


ROOT = Path(__file__).resolve().parents[1]
SUBGRAPH_ROOT = ROOT / "subgraphs"


SOURCE_SHA256 = {
    "examples/workflows/01-basic-generation/2026-08-06_H3_Turbo_Stable_4V4A.json": (
        "f5608f8da04e6f300202c1cfb778ad801a6d1b2ec050131c4c90c1181af3d105"
    ),
    "examples/workflows/02-audio-control/2026-08-06_H3_Audio_Lock_Source_Stable_4V4A.json": (
        "c1f950c1ea96701a964181d2cb260953d8ed81992d7a3e0c9e0f6bee4d33b61f"
    ),
    "examples/workflows/04-long-video/2026-08-09_H3_Long_Video_Auto_Resume_22F_EXP.json": (
        "ad84a52ccabdbA657298503ffd4e837a1b2c2424cdadc6ed84ffba3f12bc5a3f".lower()
    ),
    "examples/workflows/06-face-refine/2026-08-09_H3_Face_Refine_Parity_Advanced_EXP.json": (
        "35ffffee9d1c311b9abe8764909669dc7141f7ea8bc250599f07923b71c78e90"
    ),
}


def _generated_path(spec: dict) -> Path:
    return SUBGRAPH_ROOT / f"{spec['name'].replace('/', '-')}.json"


def _validate_definition(payload: dict) -> None:
    assert payload["version"] == 0.4
    assert len(payload["nodes"]) == 1
    assert len(payload["definitions"]["subgraphs"]) == 1
    top = payload["nodes"][0]
    definition = payload["definitions"]["subgraphs"][0]
    assert top["type"] == definition["id"]
    assert top["properties"]["cnr_id"] == "minimax-h3-audio-t8"

    nodes = {int(node["id"]): node for node in definition["nodes"]}
    links = {int(link["id"]): link for link in definition["links"]}
    assert len(links) == len(definition["links"])

    definition_inputs = definition["inputs"]
    definition_outputs = definition["outputs"]
    for link_id, link in links.items():
        origin_id = int(link["origin_id"])
        origin_slot = int(link["origin_slot"])
        target_id = int(link["target_id"])
        target_slot = int(link["target_slot"])
        if origin_id == -10:
            assert link_id in definition_inputs[origin_slot]["linkIds"]
        else:
            assert link_id in (nodes[origin_id]["outputs"][origin_slot].get("links") or [])
        if target_id == -20:
            assert link_id in definition_outputs[target_slot]["linkIds"]
        else:
            assert nodes[target_id]["inputs"][target_slot]["link"] == link_id

    proxy_widgets = top["properties"]["proxyWidgets"]
    assert len(proxy_widgets) == sum("widget" in item for item in top["inputs"])
    for node_id, input_name in proxy_widgets:
        node = nodes[int(node_id)]
        target = next(item for item in node["inputs"] if item["name"] == input_name)
        assert "widget" in target


def test_quickstart_sources_remain_byte_identical():
    for relative_path, expected in SOURCE_SHA256.items():
        digest = hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest()
        assert digest == expected


def test_quickstart_subgraphs_are_deterministic_and_structurally_valid():
    paths = sorted(SUBGRAPH_ROOT.glob("*.json"))
    assert len(paths) == len(SPECS) == 6
    assert {path for path in paths} == {_generated_path(spec) for spec in SPECS}
    for spec in SPECS:
        checked_in = json.loads(_generated_path(spec).read_text(encoding="utf-8"))
        rebuilt = build_subgraph(spec)
        assert checked_in == rebuilt
        _validate_definition(checked_in)


def test_quickstart_subgraphs_keep_existing_node_types_only():
    for spec in SPECS:
        source = json.loads((ROOT / spec["source"]).read_text(encoding="utf-8"))
        payload = json.loads(_generated_path(spec).read_text(encoding="utf-8"))
        definition = payload["definitions"]["subgraphs"][0]
        assert [node["type"] for node in definition["nodes"]] == [
            node["type"] for node in source["nodes"]
        ]
