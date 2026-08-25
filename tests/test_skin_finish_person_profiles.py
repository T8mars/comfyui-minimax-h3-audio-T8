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
    _mask_at_source,
    _source_contract,
)
from h3_audio_t8_pkg.nodes_skin_finish_person_profiles import (
    MiniMaxH3SkinFinishPerPersonT8Advanced,
    MiniMaxH3SkinFinishPersonProfileT8Advanced,
)
from h3_audio_t8_pkg.skin_finish import (
    _tensor_proxy_sha256,
    build_skin_finish_review,
)
from h3_audio_t8_pkg.skin_finish_multiface_parser import (
    SKIN_FINISH_MULTIFACE_SEMANTIC_SCHEMA,
)
from h3_audio_t8_pkg.skin_finish_person_profiles import (
    SKIN_FINISH_PERSON_PROFILES_SCHEMA,
    build_skin_finish_person_profile,
    run_skin_finish_per_person,
)


def _frames(frame_count: int = 2) -> torch.Tensor:
    generator = torch.Generator().manual_seed(25082026)
    frames = torch.rand((frame_count, 96, 192, 3), generator=generator) * 0.55 + 0.20
    frames[:, 18:78, 16:84] += torch.linspace(0.0, 0.18, 68).view(1, 1, 68, 1)
    frames[:, 18:78, 108:176] += torch.linspace(0.18, 0.0, 68).view(1, 1, 68, 1)
    return frames.clamp(0.0, 1.0)


def _track_plan(frames: torch.Tensor, *, overlap: bool = False) -> dict:
    from comfy.ldm.sam3.tracker import pack_masks

    frame_count = int(frames.shape[0])
    masks = torch.zeros((frame_count, 2, 24, 48), dtype=torch.bool)
    if overlap:
        masks[:, 0, 3:22, 7:31] = True
        masks[:, 1, 3:22, 17:41] = True
    else:
        masks[:, 0, 3:22, 2:23] = True
        masks[:, 1, 3:22, 25:46] = True
    packed = pack_masks(masks)
    source = dict(_source_contract(frames))
    source["fps"] = 24.0
    shot = {
        "shot_id": 0,
        "start_frame": 0,
        "end_frame": frame_count - 1,
        "frame_count": frame_count,
        "object_count": 2,
        "track_keys": ["0:0", "0:1"],
        "native_object_indices": [0, 1],
        "scores": [0.95, 0.95],
        "stats": [],
        "packed_masks": packed,
        "packed_masks_sha256": "person-profile-test-packed-mask",
        "mask_size": [24, 48],
    }
    plan = {
        "schema": "h3_t8_sam31_multiface_track_plan/v1",
        "status": "sam31_shot_local_tracks_ready",
        "source": source,
        "analysis": {"height": 24, "width": 48},
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


def _assignment_from_mappings(plan: dict, mappings: list[dict[str, str]]) -> dict:
    assignment = {
        "schema": "h3_t8_multiface_identity_assignment/v1",
        "status": "identity_assignment_ready",
        "source": plan["source"],
        "track_plan_sha256": plan["sha256"],
        "mappings": mappings,
        "identity_is_suggestion_not_proof": True,
        "automatic_accept": False,
    }
    assignment["sha256"] = _hash_json(_json_safe(assignment))
    return assignment


def _crossing_track_plan(frames: torch.Tensor) -> dict:
    from comfy.ldm.sam3.tracker import pack_masks

    frame_count = int(frames.shape[0])
    assert frame_count == 5
    masks = torch.zeros((frame_count, 2, 24, 48), dtype=torch.bool)
    left_positions = (2, 9, 16, 23, 30)
    right_positions = (36, 29, 22, 15, 8)
    for index, (left, right) in enumerate(zip(left_positions, right_positions)):
        masks[index, 0, 4:21, left : left + 10] = True
        masks[index, 1, 4:21, right : right + 10] = True
    source = dict(_source_contract(frames))
    source["fps"] = 24.0
    shot = {
        "shot_id": 0,
        "start_frame": 0,
        "end_frame": frame_count - 1,
        "frame_count": frame_count,
        "object_count": 2,
        "track_keys": ["0:0", "0:1"],
        "native_object_indices": [0, 1],
        "scores": [0.95, 0.95],
        "stats": [],
        "packed_masks": pack_masks(masks),
        "packed_masks_sha256": "person-profile-crossing-test-packed-mask",
        "mask_size": [24, 48],
    }
    plan = {
        "schema": "h3_t8_sam31_multiface_track_plan/v1",
        "status": "sam31_shot_local_tracks_ready",
        "source": source,
        "analysis": {"height": 24, "width": 48},
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


def _two_shot_track_plan(frames: torch.Tensor) -> dict:
    from comfy.ldm.sam3.tracker import pack_masks

    frame_count = int(frames.shape[0])
    assert frame_count == 4
    source = dict(_source_contract(frames))
    source["fps"] = 24.0
    shots = []
    for shot_id, start_frame in enumerate((0, 2)):
        masks = torch.zeros((2, 2, 24, 48), dtype=torch.bool)
        masks[:, 0, 3:22, 2:23] = True
        masks[:, 1, 3:22, 25:46] = True
        shots.append(
            {
                "shot_id": shot_id,
                "start_frame": start_frame,
                "end_frame": start_frame + 1,
                "frame_count": 2,
                "object_count": 2,
                "track_keys": [f"{shot_id}:0", f"{shot_id}:1"],
                "native_object_indices": [0, 1],
                "scores": [0.95, 0.95],
                "stats": [],
                "packed_masks": pack_masks(masks),
                "packed_masks_sha256": f"person-profile-shot-{shot_id}-packed-mask",
                "mask_size": [24, 48],
            }
        )
    plan = {
        "schema": "h3_t8_sam31_multiface_track_plan/v1",
        "status": "sam31_shot_local_tracks_ready",
        "source": source,
        "analysis": {"height": 24, "width": 48},
        "sam31": {"track_identity_scope": "shot_local_only"},
        "shots": shots,
        "scene_cut_threshold": 0.28,
        "scene_cut_count": 1,
        "max_scene_delta": 1.0,
        "release": {"performed": True},
        "identity_assigned": False,
        "automatic_accept": False,
    }
    plan["sha256"] = _hash_json(_json_safe(plan))
    return plan


def _semantic_from_track_plan(frames: torch.Tensor, plan: dict) -> torch.Tensor:
    frame_count, height, width = map(int, frames.shape[:3])
    mask = torch.zeros((frame_count, height, width), dtype=torch.float32)
    for shot in plan["shots"]:
        for frame_index in range(int(shot["start_frame"]), int(shot["end_frame"]) + 1):
            shot_local = frame_index - int(shot["start_frame"])
            for track_index in range(int(shot["object_count"])):
                mask[frame_index] = torch.maximum(
                    mask[frame_index],
                    _mask_at_source(
                        shot,
                        shot_local,
                        track_index,
                        height,
                        width,
                    ).float()
                    * 0.85,
                )
    return mask


def _semantic_mask(frames: torch.Tensor, *, overlap: bool = False) -> torch.Tensor:
    mask = torch.zeros(frames.shape[:3], dtype=torch.float32)
    if overlap:
        mask[:, 25:70, 76:116] = 0.85
    else:
        mask[:, 25:70, 20:78] = 0.85
        mask[:, 25:70, 114:172] = 0.85
    return mask


def _semantic_report(frames: torch.Tensor, plan: dict, mask: torch.Tensor) -> str:
    return json.dumps(
        {
            "schema": SKIN_FINISH_MULTIFACE_SEMANTIC_SCHEMA,
            "status": "READY",
            "source": plan["source"],
            "track_plan_sha256": plan["sha256"],
            "mask_proxy_sha256": _tensor_proxy_sha256(mask),
        }
    )


def _profile(
    selector: str,
    *,
    selector_type: str = "character_id",
    amount: float = 0.35,
    tone_adjust: float = 0.0,
    previous_profiles: dict | None = None,
) -> dict:
    profiles, _ = build_skin_finish_person_profile(
        selector_type=selector_type,
        selector=selector,
        preset="tone_even",
        amount=amount,
        texture_keep=0.90,
        shine_control=0.35,
        tone_adjust=tone_adjust,
        previous_profiles=previous_profiles,
    )
    return profiles


def _run(
    *,
    frames: torch.Tensor,
    plan: dict,
    mask: torch.Tensor,
    profiles: dict | None,
    assignment: dict | None,
    default_policy: str = "source_unmatched",
    accept_candidate: bool = False,
):
    return run_skin_finish_per_person(
        frames,
        plan,
        mask,
        _semantic_report(frames, plan, mask),
        default_policy,
        "subtle",
        0.35,
        0.90,
        0.35,
        0.0,
        "candidate_only",
        accept_candidate,
        1,
        256,
        2,
        profiles=profiles,
        identity_assignment=assignment,
    )


def test_profile_stack_is_hashed_chainable_and_rejects_duplicate_selector():
    first = _profile("Character_A", amount=0.20)
    second = _profile("0:1", selector_type="shot_track", previous_profiles=first)
    assert second["schema"] == SKIN_FINISH_PERSON_PROFILES_SCHEMA
    assert second["profile_count"] == 2
    assert second["profiles"][0]["selector"] == "Character_A"
    assert second["profiles"][1]["selector"] == "0:1"
    with pytest.raises(ValueError, match="Duplicate Skin Finish selector"):
        _profile("Character_A", previous_profiles=second)


def test_character_profiles_apply_distinct_parameters_and_preserve_source_by_default():
    frames = _frames()
    plan = _track_plan(frames)
    assignment = _assignment(plan)
    profiles = _profile("Character_A", amount=0.0)
    profiles = _profile(
        "Character_B",
        amount=1.0,
        tone_adjust=1.0,
        previous_profiles=profiles,
    )
    candidate, source, selected, _, used, rejected, preview, state, report = _run(
        frames=frames,
        plan=plan,
        mask=_semantic_mask(frames),
        profiles=profiles,
        assignment=assignment,
    )
    parsed = json.loads(report)
    assert parsed["status"] == "CANDIDATE_READY"
    assert source is frames
    assert selected is frames
    assert state["mode"] == "per_person_per_shot_advanced"
    assert tuple(preview.shape) == (2, 96, 192, 3)
    assert torch.count_nonzero(used[:, :, :96]) > 0
    assert torch.count_nonzero(used[:, :, 96:]) > 0
    assert torch.count_nonzero(rejected) == 0
    left_delta = (candidate[:, :, :96] - frames[:, :, :96]).abs().mean()
    right_delta = (candidate[:, :, 96:] - frames[:, :, 96:]).abs().mean()
    assert float(left_delta) == 0.0
    assert float(right_delta) > 0.0
    assert parsed["routing"]["precedence"] == (
        "shot_track_over_character_id_over_optional_default"
    )
    assert parsed["mechanical_gates"]["outside_mask_bit_exact"] is True
    assert parsed["mechanical_gates"]["candidate_selected"] is False


def test_exact_shot_track_profile_overrides_character_profile():
    frames = _frames()
    plan = _track_plan(frames)
    assignment = _assignment(plan)
    profiles = _profile("Character_A", amount=0.0)
    profiles = _profile(
        "0:0",
        selector_type="shot_track",
        amount=1.0,
        tone_adjust=1.0,
        previous_profiles=profiles,
    )
    candidate, _, _, _, _, _, _, _, report = _run(
        frames=frames,
        plan=plan,
        mask=_semantic_mask(frames),
        profiles=profiles,
        assignment=assignment,
    )
    parsed = json.loads(report)
    route = next(item for item in parsed["routing"]["tracks"] if item["track_key"] == "0:0")
    assert route["resolved_by"] == "shot_track"
    assert float((candidate[:, :, :96] - frames[:, :, :96]).abs().mean()) > 0.0


def test_unmatched_person_stays_exact_source_and_is_visible_as_rejected():
    frames = _frames()
    plan = _track_plan(frames)
    assignment = _assignment(plan)
    profiles = _profile("Character_A", amount=1.0, tone_adjust=1.0)
    candidate, _, _, _, used, rejected, _, _, report = _run(
        frames=frames,
        plan=plan,
        mask=_semantic_mask(frames),
        profiles=profiles,
        assignment=assignment,
    )
    parsed = json.loads(report)
    assert torch.count_nonzero(used[:, :, 96:]) == 0
    assert torch.count_nonzero(rejected[:, :, 96:]) > 0
    assert torch.equal(candidate[:, :, 96:], frames[:, :, 96:])
    assert "unmatched_person_skin_preserved_source" in parsed["findings"]


def test_cross_person_overlap_fails_closed_to_source_instead_of_color_bleeding():
    frames = _frames()
    plan = _track_plan(frames, overlap=True)
    assignment = _assignment(plan)
    profiles = _profile("Character_A", amount=1.0, tone_adjust=1.0)
    profiles = _profile("Character_B", amount=1.0, tone_adjust=-1.0, previous_profiles=profiles)
    mask = _semantic_mask(frames, overlap=True)
    candidate, _, selected, _, used, rejected, _, _, report = _run(
        frames=frames,
        plan=plan,
        mask=mask,
        profiles=profiles,
        assignment=assignment,
    )
    parsed = json.loads(report)
    assert parsed["status"] == "ABSTAIN_NO_PROFILED_SKIN_PIXELS"
    assert candidate is frames
    assert selected is frames
    assert torch.count_nonzero(used) == 0
    assert torch.count_nonzero(rejected) > 0
    assert parsed["routing"]["ambiguous_overlap_pixels"] > 0
    assert parsed["routing"]["overlap_policy"] == "source_on_any_multi_track_overlap"


def test_crossing_people_keep_character_routes_and_overlap_stays_exact_source():
    frames = _frames(frame_count=5)
    plan = _crossing_track_plan(frames)
    assignment = _assignment(plan)
    profiles = _profile("Character_A", amount=0.0)
    profiles = _profile(
        "Character_B",
        amount=1.0,
        tone_adjust=1.0,
        previous_profiles=profiles,
    )
    mask = _semantic_from_track_plan(frames, plan)
    candidate, _, _, _, used, rejected, _, _, report = _run(
        frames=frames,
        plan=plan,
        mask=mask,
        profiles=profiles,
        assignment=assignment,
    )
    parsed = json.loads(report)
    assert parsed["status"] == "CANDIDATE_READY"
    shot = plan["shots"][0]
    for frame_index in range(int(frames.shape[0])):
        track_a = _mask_at_source(shot, frame_index, 0, 96, 192)
        track_b = _mask_at_source(shot, frame_index, 1, 96, 192)
        unique_a = track_a & ~track_b & (mask[frame_index] > 1.0e-5)
        unique_b = track_b & ~track_a & (mask[frame_index] > 1.0e-5)
        overlap = track_a & track_b & (mask[frame_index] > 1.0e-5)
        assert torch.equal(candidate[frame_index][unique_a], frames[frame_index][unique_a])
        assert float(
            (candidate[frame_index][unique_b] - frames[frame_index][unique_b])
            .abs()
            .mean()
        ) > 0.0
        if bool(overlap.any()):
            assert torch.equal(
                candidate[frame_index][overlap], frames[frame_index][overlap]
            )
            assert torch.count_nonzero(used[frame_index][overlap]) == 0
            assert torch.count_nonzero(rejected[frame_index][overlap]) > 0
    routes = {item["selector"]: item for item in parsed["routing"]["routes"]}
    assert routes["Character_A"]["treatment_diagnostics"]["mean_abs_rgb_delta"] == 0.0
    assert routes["Character_B"]["treatment_diagnostics"]["mean_abs_rgb_delta"] > 0.0
    assert parsed["routing"]["ambiguous_overlap_pixels"] > 0


def test_cross_shot_character_binding_follows_reviewed_mapping_not_screen_side():
    frames = _frames(frame_count=4)
    plan = _two_shot_track_plan(frames)
    assignment = _assignment_from_mappings(
        plan,
        [
            {"track_key": "0:0", "character_id": "Character_A"},
            {"track_key": "0:1", "character_id": "Character_B"},
            {"track_key": "1:0", "character_id": "Character_B"},
            {"track_key": "1:1", "character_id": "Character_A"},
        ],
    )
    profiles = _profile("Character_A", amount=0.0)
    profiles = _profile(
        "Character_B",
        amount=1.0,
        tone_adjust=1.0,
        previous_profiles=profiles,
    )
    mask = _semantic_from_track_plan(frames, plan)
    candidate, _, _, _, _, _, _, _, report = _run(
        frames=frames,
        plan=plan,
        mask=mask,
        profiles=profiles,
        assignment=assignment,
    )
    parsed = json.loads(report)
    assert parsed["status"] == "CANDIDATE_READY"
    track_routes = {
        item["track_key"]: (item["character_id"], item["resolved_by"])
        for item in parsed["routing"]["tracks"]
    }
    assert track_routes == {
        "0:0": ("Character_A", "character_id"),
        "0:1": ("Character_B", "character_id"),
        "1:0": ("Character_B", "character_id"),
        "1:1": ("Character_A", "character_id"),
    }
    assert float((candidate[:2, :, :96] - frames[:2, :, :96]).abs().mean()) == 0.0
    assert float((candidate[:2, :, 96:] - frames[:2, :, 96:]).abs().mean()) > 0.0
    assert float((candidate[2:, :, :96] - frames[2:, :, :96]).abs().mean()) > 0.0
    assert float((candidate[2:, :, 96:] - frames[2:, :, 96:]).abs().mean()) == 0.0


def test_different_luma_people_report_diagnostics_without_automatic_fairness_claim():
    frames = torch.full((2, 96, 192, 3), 0.42, dtype=torch.float32)
    frames[:, 18:78, 16:84] = torch.tensor((0.16, 0.19, 0.22))
    frames[:, 18:78, 108:176] = torch.tensor((0.70, 0.74, 0.78))
    plan = _track_plan(frames)
    assignment = _assignment(plan)
    profiles = _profile("Character_A", amount=0.45)
    profiles = _profile(
        "Character_B",
        amount=0.45,
        previous_profiles=profiles,
    )
    mask = _semantic_mask(frames)
    candidate, _, _, _, used, _, _, _, report = _run(
        frames=frames,
        plan=plan,
        mask=mask,
        profiles=profiles,
        assignment=assignment,
    )
    parsed = json.loads(report)
    routes = {item["selector"]: item for item in parsed["routing"]["routes"]}
    dark = routes["Character_A"]["treatment_diagnostics"]
    light = routes["Character_B"]["treatment_diagnostics"]
    assert dark["metric_space"] == "display_referred_sdr_rec709_luma_proxy"
    assert dark["pixel_count"] > 0
    assert light["pixel_count"] > 0
    assert dark["source_mean_luma_proxy"] < light["source_mean_luma_proxy"]
    assert dark["candidate_low_clip_fraction"] == 0.0
    assert dark["candidate_high_clip_fraction"] == 0.0
    assert light["candidate_low_clip_fraction"] == 0.0
    assert light["candidate_high_clip_fraction"] == 0.0
    assert parsed["routing"]["diagnostic_contract"]["automatic_fairness_decision"] is False
    outside = used <= 0
    assert torch.equal(candidate[..., :3][outside], frames[..., :3][outside])


def test_character_profile_without_identity_assignment_abstains_exactly():
    frames = _frames()
    plan = _track_plan(frames)
    mask = _semantic_mask(frames)
    candidate, source, selected, _, used, _, _, _, report = _run(
        frames=frames,
        plan=plan,
        mask=mask,
        profiles=_profile("Character_A"),
        assignment=None,
    )
    parsed = json.loads(report)
    assert parsed["status"] == "ABSTAIN_CHARACTER_PROFILE_REQUIRES_IDENTITY_ASSIGNMENT"
    assert candidate is frames
    assert source is frames
    assert selected is frames
    assert torch.count_nonzero(used) == 0


def test_modified_profile_stack_fails_closed_and_marks_semantic_mask_rejected():
    frames = _frames()
    plan = _track_plan(frames)
    mask = _semantic_mask(frames)
    profiles = _profile("0:0", selector_type="shot_track")
    profiles["profiles"][0]["amount"] = 1.0
    result = _run(
        frames=frames,
        plan=plan,
        mask=mask,
        profiles=profiles,
        assignment=None,
    )
    parsed = json.loads(result[-1])
    assert parsed["status"] == "ABSTAIN_PERSON_PROFILES_INVALID"
    assert result[0] is frames
    assert result[2] is frames
    assert torch.count_nonzero(result[4]) == 0
    assert torch.equal(result[5], mask)


def test_semantic_mask_and_report_mismatch_abstains_before_processing():
    frames = _frames()
    plan = _track_plan(frames)
    mask = _semantic_mask(frames)
    wrong_report = _semantic_report(frames, plan, torch.zeros_like(mask))
    result = run_skin_finish_per_person(
        frames,
        plan,
        mask,
        wrong_report,
        "default_profile",
        "subtle",
        0.35,
        0.90,
        0.35,
        0.0,
        "candidate_only",
        False,
        1,
        256,
        2,
    )
    parsed = json.loads(result[-1])
    assert parsed["status"] == "ABSTAIN_SEMANTIC_MASK_REPORT_MISMATCH"
    assert result[0] is frames
    assert result[2] is frames


def test_per_person_state_and_report_remain_compatible_with_existing_preview_node():
    frames = _frames()
    plan = _track_plan(frames)
    mask = _semantic_mask(frames)
    result = _run(
        frames=frames,
        plan=plan,
        mask=mask,
        profiles=None,
        assignment=None,
        default_policy="default_profile",
    )
    candidate, source, _, audio, used, rejected, _, state, report = result
    review = build_skin_finish_review(
        source,
        candidate,
        used,
        rejected,
        state,
        report,
        frame_index=0,
        comparison_position=0.5,
        accept_candidate=False,
        audio_source=audio,
        audio_passthrough=audio,
    )
    assert review[0] is source
    assert json.loads(review[-1])["candidate_allowed_by_gate"] is True


def test_new_node_schemas_are_safe_and_registration_is_strictly_append_only():
    profile_schema = MiniMaxH3SkinFinishPersonProfileT8Advanced.define_schema()
    executor_schema = MiniMaxH3SkinFinishPerPersonT8Advanced.define_schema()
    inputs = {item.id: item for item in executor_schema.inputs}
    assert profile_schema.is_experimental is True
    assert executor_schema.is_experimental is True
    assert inputs["default_policy"].default == "source_unmatched"
    assert inputs["accept_candidate"].default is False
    assert inputs["profiles"].optional is True
    assert inputs["identity_assignment"].optional is True
    node_ids = [
        node.define_schema().node_id
        for node in asyncio.run(comfy_entrypoint().get_node_list())
    ]
    assert node_ids[199:205] == [
        "MiniMaxH3SkinFinishMultiPersonSemanticMaskT8Advanced",
        "MiniMaxH3SkinFinishPersonProfileT8Advanced",
        "MiniMaxH3SkinFinishPerPersonT8Advanced",
        "MiniMaxH3SkinFinishMultiPersonProfileSemanticMaskT8Advanced",
        "MiniMaxH3SkinFinishSafetyAuditT8Advanced",
        "MiniMaxH3SkinFinishFrequencySplitT8Advanced",
    ]
    assert node_ids[205:207] == [
        "MiniMaxH3SkinFinishTimelineKeyframeT8Advanced",
        "MiniMaxH3SkinFinishTimelineT8Advanced",
    ]


def test_per_person_workflow_is_importable_documented_and_source_safe():
    path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "workflows"
        / "17-skin-finish"
        / "2026-08-25_H3_Skin_Finish_Per_Person_Advanced_EXP.json"
    )
    workflow = json.loads(path.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in workflow["nodes"]}
    types = [node["type"] for node in workflow["nodes"]]
    assert workflow["version"] == 0.4
    assert workflow["last_node_id"] == max(nodes)
    assert workflow["last_link_id"] == max(link[0] for link in workflow["links"])
    assert types.count("MiniMaxH3SkinFinishPersonProfileT8Advanced") == 2
    assert types.count("MiniMaxH3SkinFinishPerPersonT8Advanced") == 1
    assert types.count("MiniMaxH3FaceTrackAssignT8Advanced") == 1
    assert (
        types.count(
            "MiniMaxH3SkinFinishMultiPersonProfileSemanticMaskT8Advanced"
        )
        == 1
    )
    assert types.count("MiniMaxH3SkinFinishSafetyAuditT8Advanced") == 1
    assert types.count("MarkdownNote") == 7
    profiles = [
        node
        for node in workflow["nodes"]
        if node["type"] == "MiniMaxH3SkinFinishPersonProfileT8Advanced"
    ]
    executor = next(
        node
        for node in workflow["nodes"]
        if node["type"] == "MiniMaxH3SkinFinishPerPersonT8Advanced"
    )
    parser = next(
        node
        for node in workflow["nodes"]
        if node["type"]
        == "MiniMaxH3SkinFinishMultiPersonProfileSemanticMaskT8Advanced"
    )
    guard = next(
        node
        for node in workflow["nodes"]
        if node["type"] == "MiniMaxH3SkinFinishTextureGuardT8Advanced"
    )
    finalizer = next(
        node
        for node in workflow["nodes"]
        if node["type"] == "MiniMaxH3SkinFinishVideoFinalizeT8Advanced"
    )
    safety_audit = next(
        node
        for node in workflow["nodes"]
        if node["type"] == "MiniMaxH3SkinFinishSafetyAuditT8Advanced"
    )
    assert profiles[0]["widgets_values"][:3] == [
        "character_id",
        "Character_A",
        "subtle",
    ]
    assert profiles[1]["widgets_values"][:3] == [
        "character_id",
        "Character_B",
        "oil_control",
    ]
    assert parser["widgets_values"][-3:] == [1.45, 0.50, 6]
    assert executor["widgets_values"] == [
        "",
        "source_unmatched",
        "subtle",
        0.35,
        0.90,
        0.35,
        0.0,
        "candidate_only",
        False,
        2,
        640,
        6,
    ]
    assert guard["widgets_values"][-1] is False
    assert safety_audit["widgets_values"] == [
        "unique_track_owner",
        "hard_gate",
        0.08,
        0.30,
        0.04,
        0.001,
        64,
        0.20,
        False,
    ]
    assert finalizer["widgets_values"][-1] is False
    notes = "\n".join(
        node["widgets_values"][0]
        for node in workflow["nodes"]
        if node["type"] == "MarkdownNote"
    )
    for required in (
        "shot:track > character_id > default",
        "source_unmatched",
        "SFace",
        "交叉",
        "tone_adjust",
        "payload",
        "gated_candidate",
        "只会自动拒绝",
    ):
        assert required in notes
