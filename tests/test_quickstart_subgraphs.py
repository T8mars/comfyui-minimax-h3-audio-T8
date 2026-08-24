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
        "dcbcab9809b5fd86c8eae34002b3e4b2eb92d114939ecae9dbb01f6f61063f7c"
    ),
    "examples/workflows/04-long-video/2026-08-09_H3_Long_Video_Auto_Resume_22F_EXP.json": (
        "ad84a52ccabdbA657298503ffd4e837a1b2c2424cdadc6ed84ffba3f12bc5a3f".lower()
    ),
    "examples/workflows/06-face-refine/2026-08-09_H3_Face_Refine_Parity_Advanced_EXP.json": (
        "35ffffee9d1c311b9abe8764909669dc7141f7ea8bc250599f07923b71c78e90"
    ),
    "examples/workflows/11-studio-production/2026-08-22_H3_Creator_Synchronized_AV_AB_Advanced.json": (
        "0ccccbd13dea056a8e49155dadbbaf1ecead0ec15800e6885195d725e01f0521"
    ),
}


LEGACY_SUBGRAPH_SHA256 = {
    "2026-08-22_H3_Quick_Audio_Drive.json": (
        "ebc781549032339232c8a3efdae891037250cee3a804911aae3c99e1bc3db482"
    ),
    "2026-08-22_H3_Quick_Face_Repair.json": (
        "b8922ca1f3d5a2caa5dfddcab39a148937aa4c9956c83394a51977e6c326bbab"
    ),
    "2026-08-22_H3_Quick_I2VA_FL2VA.json": (
        "d65e080a700758497684b110ac0b5e082e8e214925311d82aa46aa2e16cd79d9"
    ),
    "2026-08-22_H3_Quick_Long_Video.json": (
        "948f40bf034e5a9bcb0a0c87f092638fd0958328b5145a8303ae0ad82431b996"
    ),
    "2026-08-22_H3_Quick_Ref2VA.json": (
        "14cd364be8c95d457a62697523358b1f53d03de8f51a68d1dd9115344d9ba93a"
    ),
    "2026-08-22_H3_Quick_T2VA.json": (
        "75067562111ed0bd096257d19f345e9aa763975ef1c836e9bffaaa9278eed5f5"
    ),
}


def _generated_path(spec: dict) -> Path:
    return SUBGRAPH_ROOT / spec["filename"]


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
    assert len(paths) == len(SPECS) == 7
    assert {path for path in paths} == {_generated_path(spec) for spec in SPECS}
    assert all(path.name.isascii() for path in paths)
    assert all(path.name.startswith("2026-08-") and "_H3_Quick_" in path.name for path in paths)
    for spec in SPECS:
        checked_in = json.loads(_generated_path(spec).read_text(encoding="utf-8"))
        rebuilt = build_subgraph(spec)
        assert checked_in == rebuilt
        _validate_definition(checked_in)


def test_existing_six_quickstart_subgraphs_remain_byte_identical():
    for filename, expected in LEGACY_SUBGRAPH_SHA256.items():
        digest = hashlib.sha256((SUBGRAPH_ROOT / filename).read_bytes()).hexdigest()
        assert digest == expected


def test_quickstart_subgraphs_keep_existing_node_types_only():
    for spec in SPECS:
        source = json.loads((ROOT / spec["source"]).read_text(encoding="utf-8"))
        payload = json.loads(_generated_path(spec).read_text(encoding="utf-8"))
        definition = payload["definitions"]["subgraphs"][0]
        assert [node["type"] for node in definition["nodes"]] == [
            node["type"] for node in source["nodes"]
        ]


def test_quick_audio_drive_distinguishes_soundtrack_lock_from_exact_lip_sync():
    spec = next(item for item in SPECS if item["id"] == "quick_audio_drive")
    payload = json.loads(_generated_path(spec).read_text(encoding="utf-8"))
    definition = payload["definitions"]["subgraphs"][0]
    conditioning = next(
        node
        for node in definition["nodes"]
        if node["type"] == "MiniMaxH3AudioConditioningT8"
    )
    notes = [
        str(node.get("widgets_values", [""])[0])
        for node in definition["nodes"]
        if node["type"] == "MarkdownNote"
    ]

    assert conditioning["widgets_values"][5] == "lock_source"
    assert conditioning["widgets_values"][7] is True
    assert "<Audio 1>" in conditioning["widgets_values"][0]
    assert "precise synchronization" not in conditioning["widgets_values"][0]
    assert any("mux_audio" in note and "逐音素" in note for note in notes)


def test_quick_creator_av_review_keeps_audio_separate_and_human_reviewed():
    spec = next(item for item in SPECS if item["id"] == "quick_creator_av_review")
    payload = json.loads(_generated_path(spec).read_text(encoding="utf-8"))
    top = payload["nodes"][0]
    definition = payload["definitions"]["subgraphs"][0]
    by_type = {}
    for node in definition["nodes"]:
        by_type.setdefault(node["type"], []).append(node)

    assert top["properties"]["ver"] == "1.45.0"
    assert [item["name"] for item in top["inputs"]] == [
        "baseline_video",
        "candidate_video",
        "label_a",
        "label_b",
        "seed_a",
        "seed_b",
        "winner_after_review",
        "reviewer_notes",
        "require_equal_geometry",
        "output_prefix",
    ]
    assert [item["name"] for item in top["outputs"]] == [
        "comparison_frames",
        "audio_a",
        "audio_b",
        "silent_comparison_video",
        "winner",
        "selected_seed",
        "visual_review_json",
        "audio_drift_decision",
        "audio_drift_report",
    ]
    assert len(by_type["LoadVideo"]) == 2
    assert len(by_type["GetVideoComponents"]) == 2
    assert len(by_type["PreviewAudio"]) == 2
    compare = by_type["MiniMaxH3CreatorSynchronizedCompareT8Advanced"][0]
    assert compare["widgets_values"][-3] == "ABSTAIN"
    create_video = by_type["CreateVideo"][0]
    assert next(item for item in create_video["inputs"] if item["name"] == "audio")[
        "link"
    ] is None
