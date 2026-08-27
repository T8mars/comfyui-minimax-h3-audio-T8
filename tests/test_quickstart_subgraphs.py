from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tools.build_quickstart_subgraphs import SPECS, build_subgraph


ROOT = Path(__file__).resolve().parents[1]
SUBGRAPH_ROOT = ROOT / "subgraphs"


SOURCE_SHA256 = {
    "examples/workflows/01-basic-generation/2026-08-06_H3_Turbo_Stable_4V4A.json": (
        "cb130612fd3972dcf7abfdd77e3b387bee755d1263f597c2b6aa70be03a8ccdc"
    ),
    "examples/workflows/02-audio-control/2026-08-06_H3_Audio_Lock_Source_Stable_4V4A.json": (
        "e712c840ea82fbe84b4607daf1079bbf03abf5de1bd7f0fe9f63fefff59b0132"
    ),
    "examples/workflows/04-long-video/2026-08-09_H3_Long_Video_Auto_Resume_22F_EXP.json": (
        "9003f1d0527a3de1f5b41d8877d5239417a56e07b0c5b91ea40ec2182f5a6469"
    ),
    "examples/workflows/06-face-refine/2026-08-09_H3_Face_Refine_Parity_Advanced_EXP.json": (
        "2f506663b24cb96e87eab3cc875c1e10a87d2c1b65336618f7bc239aad923283"
    ),
    "examples/workflows/11-studio-production/2026-08-22_H3_Creator_Synchronized_AV_AB_Advanced.json": (
        "237d574f8347c9cdbe600e55e3b9f2bb5d23c52747352cc401161d1e579e17ef"
    ),
}


LEGACY_SUBGRAPH_SHA256 = {
    "2026-08-22_H3_Quick_Audio_Drive.json": (
        "2cf6ab05861c062f4c1c68754123a47fedb2256380ab6c953c7bc0ed9515c4a7"
    ),
    "2026-08-22_H3_Quick_Face_Repair.json": (
        "cb85a8bd51ba8278f7e132a45b59152585cadedfc351e04cc96f3829d09e1858"
    ),
    "2026-08-22_H3_Quick_I2VA_FL2VA.json": (
        "08af10b84bbbd050a9be12b1d1077a9e7a8dd685dddccf6b400f725c139e2152"
    ),
    "2026-08-22_H3_Quick_Long_Video.json": (
        "5a135700e9e63b91f3e3ca4875af0670ad2626b9b405620d9d5e64feb7088d1c"
    ),
    "2026-08-22_H3_Quick_Ref2VA.json": (
        "fcfc1a3ddb2927b2c64f8ff3a04768e22aef6a35116830206322ee5091733c6c"
    ),
    "2026-08-22_H3_Quick_T2VA.json": (
        "a75b35b48396b16fdbec2581f50b27f1bab1a556d21546b070532dbc2442a5c3"
    ),
}


def _canonical_json_sha256(path: Path) -> str:
    payload = json.dumps(
        json.loads(path.read_text(encoding="utf-8")),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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


def test_quickstart_sources_remain_semantically_identical():
    for relative_path, expected in SOURCE_SHA256.items():
        digest = _canonical_json_sha256(ROOT / relative_path)
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


def test_existing_six_quickstart_subgraphs_remain_semantically_identical():
    for filename, expected in LEGACY_SUBGRAPH_SHA256.items():
        digest = _canonical_json_sha256(SUBGRAPH_ROOT / filename)
        assert digest == expected


def test_quickstart_subgraphs_keep_existing_node_types_only():
    for spec in SPECS:
        source = json.loads((ROOT / spec["source"]).read_text(encoding="utf-8"))
        payload = json.loads(_generated_path(spec).read_text(encoding="utf-8"))
        definition = payload["definitions"]["subgraphs"][0]
        assert [node["type"] for node in definition["nodes"]] == [
            node["type"] for node in source["nodes"]
        ]


def test_quick_face_repair_model_widgets_do_not_embed_one_machine_inventory():
    spec = next(item for item in SPECS if item["id"] == "quick_repair")
    payload = json.loads(_generated_path(spec).read_text(encoding="utf-8"))
    definition = payload["definitions"]["subgraphs"][0]
    public = {item["name"]: item for item in definition["inputs"]}
    for name in ("model", "text_encoder", "video_vae", "audio_vae"):
        assert public[name]["type"] == "COMBO"


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
