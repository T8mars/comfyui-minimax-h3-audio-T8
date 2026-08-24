from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from h3_audio_t8_pkg.creator_workspace_advanced import (
    CREATOR_EDIT_SCHEMA,
    CREATOR_REVIEW_SCHEMA,
    CREATOR_WORKSPACE_SCHEMA,
    add_creator_shot_override,
    build_synchronized_comparison,
    compile_creator_workspace,
    select_creator_workspace_shot,
)
from h3_audio_t8_pkg.studio_advanced import build_studio_timeline


def _timeline():
    return build_studio_timeline(
        "creator_demo",
        json.dumps(
            [
                {"id": "opening", "prompt": "A calm opening", "duration_seconds": 3.0},
                {"id": "action", "prompt": "A fast action shot", "duration_seconds": 5.0},
                {"id": "ending", "prompt": "A quiet ending", "duration_seconds": 4.0},
            ]
        ),
        "minimax_h3",
        5.0,
        "16:9",
        100,
        "increment",
        True,
        True,
    )


def _override(timeline, **values):
    args = {
        "timeline": timeline,
        "shot_index": 1,
        "enabled": True,
        "compiled_prompt_override": "A reviewed faster action shot",
        "use_seed_override": True,
        "seed_override": 500,
        "variant_count": 3,
        "variant_seed_stride": 17,
        "media_roles_json": '{"picture_1":"identity","audio_1":"voice_reference"}',
        "retention_policy": "keep_winner_and_metadata",
        "hold_policy": "hold_last",
        "hold_frames": 5,
        "previous_edits": None,
    }
    args.update(values)
    return add_creator_shot_override(**args)[0]


def test_shot_override_is_non_destructive_and_has_deterministic_variants():
    timeline = _timeline()
    before = json.dumps(timeline, sort_keys=True)
    plan = _override(timeline)
    assert plan["schema"] == CREATOR_EDIT_SCHEMA
    assert plan["edits"][0]["variant_seeds"] == [500, 517, 534]
    assert plan["edits"][0]["hold_policy"] == "hold_last"
    assert json.dumps(timeline, sort_keys=True) == before
    with pytest.raises(ValueError, match="already has an override"):
        _override(timeline, previous_edits=plan)


def test_workspace_run_window_applies_overlay_and_emits_reproducible_sidecar():
    timeline = _timeline()
    edit = _override(timeline)
    workspace, summary, sidecar_json, workspace_json = compile_creator_workspace(
        timeline, 1, 2, False, "review action through ending", edit
    )
    sidecar = json.loads(sidecar_json)
    assert workspace["schema"] == CREATOR_WORKSPACE_SCHEMA
    assert workspace["run_count"] == 2
    assert workspace["shots"][0]["effective_compiled_prompt"] == "A reviewed faster action shot"
    assert workspace["shots"][0]["variant_seeds"] == [500, 517, 534]
    assert workspace["hold_map"]["action"] == {"policy": "hold_last", "frames": 5}
    assert sidecar["file_written"] is False
    assert sidecar["workspace_hash"] == workspace["workspace_hash"]
    assert "action" in summary and json.loads(workspace_json)["run_count"] == 2


def test_workspace_select_exposes_existing_prompt_seed_and_length_contract():
    timeline = _timeline()
    workspace = compile_creator_workspace(timeline, 0, -1, False, "", _override(timeline))[0]
    packet, prompt, negative, length, seed, shot_json, report_json = (
        select_creator_workspace_shot(workspace, 1, 2)
    )
    assert packet["schema"] == "t8.video_prompt_packet.v1"
    assert prompt == "A reviewed faster action shot"
    assert negative == ""
    assert length == timeline["shots"][1]["frame_count"]
    assert seed == 534
    assert json.loads(shot_json)["shot_id"] == "action"
    assert json.loads(report_json)["variant_index"] == 2


def test_synchronized_comparison_labels_and_preserves_pixels_without_resize():
    frames_a = torch.zeros((3, 32, 48, 3), dtype=torch.float32)
    frames_b = torch.ones((3, 32, 48, 3), dtype=torch.float32)
    output, winner, seed, report_json = build_synchronized_comparison(
        frames_a, frames_b, "base", "variant", 10, 20, "B", "B is cleaner", True
    )
    report = json.loads(report_json)
    assert output.shape == (3, 64, 104, 3)
    assert torch.equal(output[:, 32:, :48], frames_a)
    assert torch.equal(output[:, 32:, 56:], frames_b)
    assert winner == "B" and seed == 20
    assert report["schema"] == CREATOR_REVIEW_SCHEMA
    assert report["spatial_alignment"] == "center_pad_without_resize"
    assert report["audio_compared"] is False


def test_geometry_mismatch_forces_abstain_but_still_produces_review_preview():
    frames_a = torch.zeros((4, 24, 40, 3))
    frames_b = torch.ones((3, 32, 48, 3))
    output, winner, seed, report_json = build_synchronized_comparison(
        frames_a, frames_b, "a", "b", 1, 2, "A", "", True
    )
    report = json.loads(report_json)
    assert output.shape[0] == 3
    assert winner == "ABSTAIN" and seed == 0
    assert report["geometry_equal"] is False
    assert report["requested_winner"] == "A"


def test_creator_frontend_workflows_are_importable_documented_and_wired():
    root = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "workflows"
        / "11-studio-production"
    )
    expected = {
        "2026-08-22_H3_Creator_Workspace_Run_Window_Advanced.json": {
            "MiniMaxH3StudioTimelineT8Advanced",
            "MiniMaxH3CreatorShotOverrideT8Advanced",
            "MiniMaxH3CreatorWorkspaceT8Advanced",
            "MiniMaxH3CreatorWorkspaceShotSelectT8Advanced",
        },
        "2026-08-22_H3_Creator_Synchronized_AB_Advanced.json": {
            "MiniMaxH3CreatorSynchronizedCompareT8Advanced",
        },
        "2026-08-22_H3_Creator_Synchronized_AV_AB_Advanced.json": {
            "LoadVideo",
            "GetVideoComponents",
            "MiniMaxH3CreatorSynchronizedCompareT8Advanced",
            "MiniMaxH3AudioPerceptualDriftAuditT8Advanced",
            "PreviewAudio",
            "CreateVideo",
            "SaveVideo",
        },
        "2026-08-23_H3_Creator_Run_Receipt_Resume_Advanced.json": {
            "MiniMaxH3StudioTimelineT8Advanced",
            "MiniMaxH3CreatorWorkspaceT8Advanced",
            "MiniMaxH3CreatorRunReceiptT8Advanced",
            "MiniMaxH3CreatorResumePlanT8Advanced",
            "MiniMaxH3CreatorRetentionPlanT8Advanced",
            "MiniMaxH3CreatorArtifactQuarantineT8Advanced",
        },
        "2026-08-23_H3_Creator_Long_Video_Background_Bridge_Advanced_EXP.json": {
            "MiniMaxH3StudioTimelineT8Advanced",
            "MiniMaxH3CreatorShotOverrideT8Advanced",
            "MiniMaxH3CreatorWorkspaceT8Advanced",
            "MiniMaxH3CreatorBackgroundStartT8Advanced",
            "MiniMaxH3CreatorBackgroundRunSelectT8Advanced",
        },
    }
    for filename, required_types in expected.items():
        workflow = json.loads((root / filename).read_text(encoding="utf-8"))
        nodes = {node["id"]: node for node in workflow["nodes"]}
        node_types = {node["type"] for node in nodes.values()}
        assert required_types <= node_types
        assert "MarkdownNote" in node_types
        assert workflow["last_node_id"] == max(nodes)
        assert workflow["last_link_id"] == max(link[0] for link in workflow["links"])
        for link_id, source, output_slot, target, input_slot, link_type in workflow["links"]:
            assert nodes[target]["inputs"][input_slot]["link"] == link_id
            assert link_id in (nodes[source]["outputs"][output_slot].get("links") or [])
            assert nodes[source]["outputs"][output_slot]["type"] == link_type
            assert nodes[target]["inputs"][input_slot]["type"] in {link_type, "*"}

    run_window = json.loads(
        (root / "2026-08-22_H3_Creator_Workspace_Run_Window_Advanced.json").read_text(
            encoding="utf-8"
        )
    )
    by_type = {node["type"]: node for node in run_window["nodes"]}
    assert by_type["MiniMaxH3CreatorWorkspaceT8Advanced"]["widgets_values"][:2] == [1, 2]
    assert by_type["MiniMaxH3CreatorWorkspaceShotSelectT8Advanced"]["widgets_values"] == [
        0,
        0,
    ]

    comparison = json.loads(
        (root / "2026-08-22_H3_Creator_Synchronized_AB_Advanced.json").read_text(
            encoding="utf-8"
        )
    )
    compare_node = next(
        node
        for node in comparison["nodes"]
        if node["type"] == "MiniMaxH3CreatorSynchronizedCompareT8Advanced"
    )
    assert compare_node["widgets_values"][-3:] == [
        "ABSTAIN",
        "Watch the full clip before selecting A, B or TIE.",
        True,
    ]

    av_comparison = json.loads(
        (root / "2026-08-22_H3_Creator_Synchronized_AV_AB_Advanced.json").read_text(
            encoding="utf-8"
        )
    )
    av_by_type = {}
    for node in av_comparison["nodes"]:
        av_by_type.setdefault(node["type"], []).append(node)
    assert len(av_by_type["LoadVideo"]) == 2
    assert len(av_by_type["GetVideoComponents"]) == 2
    assert len(av_by_type["PreviewAudio"]) == 2
    assert av_by_type["MiniMaxH3CreatorSynchronizedCompareT8Advanced"][0][
        "widgets_values"
    ][-1] is True
    create_video = av_by_type["CreateVideo"][0]
    assert create_video["inputs"][2]["name"] == "audio"
    assert create_video["inputs"][2]["link"] is None
    notes = "\n".join(
        node["widgets_values"][0] for node in av_by_type["MarkdownNote"]
    )
    assert "并排视频故意不带声音" in notes
    assert "分别完整播放" in notes

    runtime = json.loads(
        (root / "2026-08-23_H3_Creator_Run_Receipt_Resume_Advanced.json").read_text(
            encoding="utf-8"
        )
    )
    runtime_by_type = {}
    for node in runtime["nodes"]:
        runtime_by_type.setdefault(node["type"], []).append(node)
    assert len(runtime_by_type["MiniMaxH3CreatorRunReceiptT8Advanced"]) == 2
    assert runtime_by_type["MiniMaxH3CreatorRunReceiptT8Advanced"][0][
        "widgets_values"
    ][3] == "completed"
    assert runtime_by_type["MiniMaxH3CreatorRunReceiptT8Advanced"][1][
        "widgets_values"
    ][3] == "accepted"
    assert len(runtime_by_type["MarkdownNote"]) == 4
    assert runtime_by_type["MiniMaxH3CreatorRetentionPlanT8Advanced"][0][
        "widgets_values"
    ] == [False]
    quarantine = runtime_by_type["MiniMaxH3CreatorArtifactQuarantineT8Advanced"][0]
    assert quarantine["mode"] == 4
    assert quarantine["widgets_values"] == ["prepare_only", "", "", 0, False, 8]
    runtime_notes = "\n".join(
        node["widgets_values"][0] for node in runtime_by_type["MarkdownNote"]
    )
    assert "拟删除清单" in runtime_notes
    assert "不会永久删除" in runtime_notes
    assert "recover_to_source" in runtime_notes

    background = json.loads(
        (
            root
            / "2026-08-23_H3_Creator_Long_Video_Background_Bridge_Advanced_EXP.json"
        ).read_text(encoding="utf-8")
    )
    background_by_type = {node["type"]: node for node in background["nodes"]}
    assert background_by_type["MiniMaxH3CreatorBackgroundStartT8Advanced"][
        "widgets_values"
    ][1] == "review_only"
    assert background_by_type["MiniMaxH3CreatorBackgroundRunSelectT8Advanced"][
        "widgets_values"
    ] == ["retry_as_variant_clamped"]
    background_notes = "\n".join(
        node["widgets_values"][0]
        for node in background["nodes"]
        if node["type"] == "MarkdownNote"
    )
    assert "Long Video Candidate Save" in background_notes
    assert "status/pause/resume/cancel" in background_notes
    assert "一个shot必须严格对应" in background_notes
