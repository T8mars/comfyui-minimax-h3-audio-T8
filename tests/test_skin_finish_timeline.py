from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
import torch

from h3_audio_t8_pkg import comfy_entrypoint
from h3_audio_t8_pkg.multiface_refine_advanced import (
    _hash_json,
    _json_safe,
    _source_contract,
)
from h3_audio_t8_pkg.nodes_skin_finish_timeline import (
    MiniMaxH3SkinFinishTimelineKeyframeT8Advanced,
    MiniMaxH3SkinFinishTimelineT8Advanced,
)
from h3_audio_t8_pkg.skin_finish import _tensor_proxy_sha256
from h3_audio_t8_pkg.skin_finish_multiface_parser import (
    SKIN_FINISH_MULTIFACE_SEMANTIC_SCHEMA,
)
from h3_audio_t8_pkg.skin_finish_timeline import (
    SKIN_FINISH_TIMELINE_PLAN_SCHEMA,
    _group_keyframes,
    _interpolated_parameters,
    build_skin_finish_timeline_keyframe,
    run_skin_finish_timeline,
)
from h3_audio_t8_pkg.studio_advanced import build_studio_timeline


def _timeline() -> dict:
    return build_studio_timeline(
        project_id="skin_timeline_test",
        shots_json=json.dumps([
            {"id": "close", "prompt": "A restrained close-up", "duration_seconds": 22 / 24},
            {"id": "turn", "prompt": "The subject turns", "duration_seconds": 22 / 24},
        ]),
        default_backend="minimax_h3",
        default_duration_seconds=22 / 24,
        default_aspect_ratio="16:9",
        base_seed=42,
        seed_policy="increment",
        split_long_shots=True,
        strict_exact_dialogue=True,
    )


def _frames() -> torch.Tensor:
    generator = torch.Generator().manual_seed(25082026)
    frames = torch.rand((44, 48, 96, 4), generator=generator) * 0.45 + 0.25
    frames[:, 8:40, 8:42, :3] += 0.12
    frames[:, 8:40, 54:88, :3] += 0.08
    return frames.clamp(0.0, 1.0)


def _track_plan(frames: torch.Tensor) -> dict:
    from comfy.ldm.sam3.tracker import pack_masks

    masks = torch.zeros((44, 2, 12, 24), dtype=torch.bool)
    masks[:, 0, 1:11, 1:11] = True
    masks[:, 1, 1:11, 13:23] = True
    shot = {
        "shot_id": 0,
        "start_frame": 0,
        "end_frame": 43,
        "frame_count": 44,
        "object_count": 2,
        "track_keys": ["0:0", "0:1"],
        "native_object_indices": [0, 1],
        "scores": [0.95, 0.95],
        "stats": [],
        "packed_masks": pack_masks(masks),
        "packed_masks_sha256": "skin-timeline-packed-test",
        "mask_size": [12, 24],
    }
    source = dict(_source_contract(frames))
    source["fps"] = 24.0
    plan = {
        "schema": "h3_t8_sam31_multiface_track_plan/v1",
        "status": "sam31_shot_local_tracks_ready",
        "source": source,
        "analysis": {"height": 12, "width": 24},
        "sam31": {"track_identity_scope": "shot_local_only"},
        "shots": [shot],
        "scene_cut_threshold": 0.28,
        "scene_cut_count": 0,
        "max_scene_delta": 0.0,
        "release": {"performed": True},
        "identity_assigned": False,
        "automatic_accept": False,
    }
    plan["sha256"] = _hash_json(_json_safe(plan))
    return plan


def _assignment(plan: dict) -> dict:
    assignment = {
        "schema": "h3_t8_multiface_identity_assignment/v1",
        "status": "identity_assignment_ready",
        "source": plan["source"],
        "track_plan_sha256": plan["sha256"],
        "mappings": [
            {"track_key": "0:0", "character_id": "Character_A"},
            {"track_key": "0:1", "character_id": "Character_B"},
        ],
        "identity_is_suggestion_not_proof": True,
        "automatic_accept": False,
    }
    assignment["sha256"] = _hash_json(_json_safe(assignment))
    return assignment


def _semantic(frames: torch.Tensor) -> torch.Tensor:
    mask = torch.zeros(frames.shape[:3], dtype=torch.float32)
    mask[:, 10:38, 10:40] = 0.85
    mask[:, 10:38, 56:86] = 0.85
    return mask


def _semantic_report(plan: dict, mask: torch.Tensor) -> str:
    return json.dumps({
        "schema": SKIN_FINISH_MULTIFACE_SEMANTIC_SCHEMA,
        "status": "READY",
        "source": plan["source"],
        "track_plan_sha256": plan["sha256"],
        "mask_proxy_sha256": _tensor_proxy_sha256(mask),
    })


def _key(
    timeline: dict,
    *,
    shot: int,
    frame: int,
    selector_type: str = "global",
    selector: str = "*",
    amount: float = 0.35,
    tone: float = 0.0,
    curve: str = "smoothstep",
    preset: str = "subtle",
    previous: dict | None = None,
) -> dict:
    plan, _ = build_skin_finish_timeline_keyframe(
        studio_timeline=timeline,
        selector_type=selector_type,
        selector=selector,
        studio_shot_index=shot,
        frame_in_shot=frame,
        interpolation_to_next=curve,
        preset=preset,
        amount=amount,
        texture_keep=0.90,
        shine_control=0.35,
        tone_adjust=tone,
        previous_plan=previous,
    )
    return plan


def test_keyframe_plan_is_hashed_canonical_and_bound_to_studio_timeline():
    timeline = _timeline()
    first = _key(timeline, shot=1, frame=10)
    second = _key(timeline, shot=0, frame=0, previous=first)
    assert second["schema"] == SKIN_FINISH_TIMELINE_PLAN_SCHEMA
    assert second["keyframe_count"] == 2
    assert [item["studio_shot_index"] for item in second["keyframes"]] == [0, 1]
    assert second["timeline_hash"] == timeline["timeline_hash"]
    with pytest.raises(ValueError, match="Duplicate Skin Finish keyframe"):
        _key(timeline, shot=0, frame=0, previous=second)
    changed = _timeline()
    changed["project_id"] = "tampered"
    with pytest.raises(Exception, match="hash mismatch"):
        _key(changed, shot=0, frame=1)


def test_continuous_values_interpolate_but_preset_is_categorical_and_hold_works():
    timeline = _timeline()
    plan = _key(
        timeline,
        shot=0,
        frame=0,
        amount=0.0,
        tone=-1.0,
        curve="smoothstep",
        preset="subtle",
    )
    plan = _key(
        timeline,
        shot=0,
        frame=10,
        amount=1.0,
        tone=1.0,
        preset="oil_control",
        previous=plan,
    )
    group = _group_keyframes(plan)[("global", "*", 0)]
    middle = _interpolated_parameters(group, 5)
    assert middle["amount"] == pytest.approx(0.5)
    assert middle["tone_adjust"] == pytest.approx(0.0)
    assert middle["preset"] == "subtle"
    assert _interpolated_parameters(group, 10)["preset"] == "oil_control"

    hold = [dict(group[0], interpolation_to_next="hold"), group[1]]
    held = _interpolated_parameters(hold, 9)
    assert held["amount"] == 0.0
    assert held["tone_adjust"] == -1.0


def test_keyframes_never_interpolate_across_studio_shots():
    timeline = _timeline()
    plan = _key(timeline, shot=0, frame=21, amount=0.0)
    plan = _key(timeline, shot=1, frame=10, amount=1.0, previous=plan)
    groups = _group_keyframes(plan)
    assert set(groups) == {("global", "*", 0), ("global", "*", 1)}
    assert _interpolated_parameters(groups[("global", "*", 0)], 21)["amount"] == 0.0
    assert _interpolated_parameters(groups[("global", "*", 1)], 0)["amount"] == 1.0


def test_timeline_executor_routes_exact_over_character_over_global_and_preserves_contracts():
    timeline = _timeline()
    frames = _frames()
    track_plan = _track_plan(frames)
    assignment = _assignment(track_plan)
    mask = _semantic(frames)
    plan = _key(timeline, shot=0, frame=0, amount=1.0, tone=1.0)
    plan = _key(
        timeline,
        shot=0,
        frame=0,
        selector_type="character_id",
        selector="Character_A",
        amount=0.0,
        previous=plan,
    )
    plan = _key(
        timeline,
        shot=1,
        frame=0,
        selector_type="shot_track",
        selector="0:0",
        amount=1.0,
        tone=-1.0,
        previous=plan,
    )
    plan = _key(timeline, shot=1, frame=0, amount=0.0, previous=plan)
    audio = {"waveform": torch.zeros((1, 2, 128)), "sample_rate": 32000}
    result = run_skin_finish_timeline(
        frames,
        timeline,
        plan,
        track_plan,
        mask,
        _semantic_report(track_plan, mask),
        "candidate_only",
        False,
        4,
        256,
        4,
        identity_assignment=assignment,
        audio=audio,
    )
    candidate, source, selected, returned_audio, used, rejected, preview, state, report = result
    parsed = json.loads(report)
    assert parsed["status"] == "CANDIDATE_READY"
    assert source is frames
    assert selected is frames
    assert returned_audio is audio
    assert state["mode"] == "per_person_studio_timeline_advanced"
    assert parsed["timeline"]["no_cross_shot_interpolation"] is True
    assert parsed["routing"]["precedence"] == (
        "sam_shot_track_over_character_id_over_global_over_source"
    )
    assert tuple(preview.shape) == (4, 48, 96, 3)
    assert torch.count_nonzero(rejected) == 0
    assert torch.count_nonzero(used) > 0
    assert torch.equal(candidate[..., 3:], frames[..., 3:])
    outside = used <= 0
    assert torch.equal(candidate[..., :3][outside], frames[..., :3][outside])
    # Character_A's zero-strength key overrides global during Studio shot 0.
    assert torch.equal(candidate[:22, :, :48], frames[:22, :, :48])
    # Character_B uses the global treatment during Studio shot 0.
    assert float((candidate[:22, :, 48:, :3] - frames[:22, :, 48:, :3]).abs().mean()) > 0.0
    # The exact SAM track key overrides the global zero-strength key in shot 1.
    assert float((candidate[22:, :, :48, :3] - frames[22:, :, :48, :3]).abs().mean()) > 0.0
    assert torch.equal(candidate[22:, :, 48:], frames[22:, :, 48:])


def test_timeline_source_mismatch_and_character_without_assignment_fail_closed():
    frames = _frames()
    timeline = _timeline()
    track_plan = _track_plan(frames)
    mask = _semantic(frames)
    plan = _key(
        timeline,
        shot=0,
        frame=0,
        selector_type="character_id",
        selector="Character_A",
    )
    result = run_skin_finish_timeline(
        frames,
        timeline,
        plan,
        track_plan,
        mask,
        _semantic_report(track_plan, mask),
        "candidate_only",
        False,
        4,
        256,
        2,
    )
    assert result[0] is frames
    assert result[2] is frames
    assert json.loads(result[-1])["status"] == (
        "ABSTAIN_CHARACTER_TIMELINE_REQUIRES_IDENTITY_ASSIGNMENT"
    )

    short_frames = frames[:-1]
    short_plan = _track_plan(torch.cat([short_frames, frames[-1:]], dim=0))
    result = run_skin_finish_timeline(
        short_frames,
        timeline,
        plan,
        short_plan,
        mask[:-1],
        _semantic_report(short_plan, mask[:-1]),
        "candidate_only",
        False,
        4,
        256,
        2,
    )
    assert result[0] is short_frames
    assert json.loads(result[-1])["status"] == "ABSTAIN_STUDIO_TIMELINE_SOURCE_MISMATCH"


def test_new_node_defaults_are_safe_and_registration_is_strictly_append_only():
    key_schema = MiniMaxH3SkinFinishTimelineKeyframeT8Advanced.define_schema()
    run_schema = MiniMaxH3SkinFinishTimelineT8Advanced.define_schema()
    key_inputs = {item.id: item for item in key_schema.inputs}
    run_inputs = {item.id: item for item in run_schema.inputs}
    assert key_schema.is_experimental is True
    assert run_schema.is_experimental is True
    assert key_inputs["selector_type"].default == "global"
    assert key_inputs["interpolation_to_next"].default == "smoothstep"
    assert run_inputs["accept_candidate"].default is False
    node_ids = [
        node.define_schema().node_id
        for node in asyncio.run(comfy_entrypoint().get_node_list())
    ]
    assert node_ids[199:211] == [
        "MiniMaxH3SkinFinishMultiPersonSemanticMaskT8Advanced",
        "MiniMaxH3SkinFinishPersonProfileT8Advanced",
        "MiniMaxH3SkinFinishPerPersonT8Advanced",
        "MiniMaxH3SkinFinishMultiPersonProfileSemanticMaskT8Advanced",
        "MiniMaxH3SkinFinishSafetyAuditT8Advanced",
        "MiniMaxH3SkinFinishFrequencySplitT8Advanced",
        "MiniMaxH3SkinFinishTimelineKeyframeT8Advanced",
        "MiniMaxH3SkinFinishTimelineT8Advanced",
        "MiniMaxH3SkinFinishQualityVideoStreamT8Advanced",
        "MiniMaxH3SkinFinishSpecularFrequencyT8Advanced",
        "MiniMaxH3SkinFinishSurfaceT8Advanced",
        "MiniMaxH3SkinFinishDichromaticT8Advanced",
    ]
    assert node_ids[211:213] == [
        "MiniMaxH3TurboSLAProfileRouterT8Advanced",
        "MiniMaxH3PDD8StepSetupT8Advanced",
    ]
    assert len(node_ids) == 221


def test_timeline_workflow_is_importable_documented_and_source_safe():
    path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "workflows"
        / "17-skin-finish"
        / "2026-08-25_H3_Skin_Finish_Studio_Timeline_Advanced_EXP.json"
    )
    workflow = json.loads(path.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in workflow["nodes"]}
    types = [node["type"] for node in workflow["nodes"]]
    assert workflow["version"] == 0.4
    assert workflow["last_node_id"] == max(nodes)
    assert workflow["last_link_id"] == max(link[0] for link in workflow["links"])
    assert types.count("MiniMaxH3StudioTimelineT8Advanced") == 1
    assert types.count("MiniMaxH3SkinFinishTimelineKeyframeT8Advanced") == 4
    assert types.count("MiniMaxH3SkinFinishTimelineT8Advanced") == 1
    assert types.count("MiniMaxH3SkinFinishSafetyAuditT8Advanced") == 1
    assert types.count("MarkdownNote") == 7
    timeline = next(
        node for node in workflow["nodes"] if node["type"] == "MiniMaxH3StudioTimelineT8Advanced"
    )
    keyframes = [
        node
        for node in workflow["nodes"]
        if node["type"] == "MiniMaxH3SkinFinishTimelineKeyframeT8Advanced"
    ]
    executor = next(
        node
        for node in workflow["nodes"]
        if node["type"] == "MiniMaxH3SkinFinishTimelineT8Advanced"
    )
    assert [item["name"] for item in timeline["inputs"]] == [
        "cast",
        "sound_canvas",
    ]
    assert [item["name"] for item in keyframes[0]["inputs"]] == [
        "studio_timeline",
        "previous_plan",
    ]
    assert [item["name"] for item in executor["inputs"]] == [
        "frames",
        "studio_timeline",
        "timeline_plan",
        "track_plan",
        "semantic_skin_mask",
        "semantic_report_json",
        "identity_assignment",
        "audio",
    ]
    assert [(node["widgets_values"][2], node["widgets_values"][3]) for node in keyframes] == [
        (0, 0),
        (0, 21),
        (1, 0),
        (1, 21),
    ]
    assert executor["widgets_values"] == ["", "candidate_only", False, 2, 640, 6]
    executor_inputs = {item["name"]: item for item in executor["inputs"]}
    assert executor_inputs["studio_timeline"]["link"] is not None
    assert executor_inputs["timeline_plan"]["link"] is not None
    assert executor_inputs["identity_assignment"]["link"] is not None
    assert executor_inputs["audio"]["link"] is not None
    assert "0.9166666667" in timeline["widgets_values"][1]
    notes = "\n".join(
        node["widgets_values"][0]
        for node in workflow["nodes"]
        if node["type"] == "MarkdownNote"
    )
    for required in (
        "Timeline总帧数",
        "Studio shot不是SAM shot",
        "绝不跨切镜",
        "smoothstep",
        "SAM shot:track > character_id > global > source",
        "默认仍然选择原片",
        "HDR、10-bit",
    ):
        assert required in notes
