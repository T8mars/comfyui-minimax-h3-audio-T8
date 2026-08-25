from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

import h3_audio_t8_pkg.nodes_skin_finish_profile_crop as profile_crop_nodes
from h3_audio_t8_pkg.multiface_refine_advanced import (
    _hash_json,
    _json_safe,
    _source_contract,
)
from h3_audio_t8_pkg.nodes_skin_finish_multiface_parser import (
    MiniMaxH3SkinFinishMultiPersonSemanticMaskT8Advanced,
)
from h3_audio_t8_pkg.nodes_skin_finish_profile_crop import (
    MiniMaxH3SkinFinishMultiPersonProfileSemanticMaskT8Advanced,
)
from h3_audio_t8_pkg.skin_finish_multiface_parser import (
    FFHQ_FIVE_POINT_TEMPLATE_512,
    SKIN_FINISH_MULTIFACE_SEMANTIC_SCHEMA,
    _align_face,
    _normalized_five_points,
    run_multiface_semantic_skin_mask,
)


def _frames(frame_count: int = 2) -> torch.Tensor:
    generator = torch.Generator().manual_seed(8242026)
    return torch.rand((frame_count, 96, 192, 3), generator=generator) * 0.55 + 0.20


def _shot(
    frame_index: int,
    shot_id: int,
    *,
    duplicate_person_masks: bool = False,
) -> dict:
    from comfy.ldm.sam3.tracker import pack_masks

    masks = torch.zeros((1, 2, 24, 48), dtype=torch.bool)
    if duplicate_person_masks:
        masks[:, :, 1:23, 1:24] = True
    else:
        masks[:, 0, 1:23, 1:24] = True
        masks[:, 1, 1:23, 24:47] = True
    packed = pack_masks(masks)
    return {
        "shot_id": shot_id,
        "start_frame": frame_index,
        "end_frame": frame_index,
        "frame_count": 1,
        "object_count": 2,
        "track_keys": [f"{shot_id}:0", f"{shot_id}:1"],
        "native_object_indices": [0, 1],
        "scores": [0.95, 0.95],
        "stats": [],
        "packed_masks": packed,
        "packed_masks_sha256": "deterministic-test-packed-mask",
        "mask_size": [24, 48],
    }


def _track_plan(
    frames: torch.Tensor,
    *,
    duplicate_person_masks: bool = False,
) -> dict:
    source = dict(_source_contract(frames))
    source["fps"] = 24.0
    shots = [
        _shot(index, index, duplicate_person_masks=duplicate_person_masks)
        for index in range(int(frames.shape[0]))
    ]
    plan = {
        "schema": "h3_t8_sam31_multiface_track_plan/v1",
        "status": "sam31_shot_local_tracks_ready",
        "source": source,
        "analysis": {"height": 24, "width": 48},
        "sam31": {"track_identity_scope": "shot_local_only"},
        "shots": shots,
        "scene_cut_threshold": 0.28,
        "scene_cut_count": max(0, len(shots) - 1),
        "max_scene_delta": 1.0 if len(shots) > 1 else 0.0,
        "release": {"performed": True},
        "identity_assigned": False,
        "automatic_accept": False,
    }
    plan["sha256"] = _hash_json(_json_safe(plan))
    return plan


def _identity_assignment(plan: dict) -> dict:
    mappings = []
    for shot in plan["shots"]:
        mappings.extend(
            [
                {
                    "track_key": shot["track_keys"][0],
                    "character_id": "Character_A",
                },
                {
                    "track_key": shot["track_keys"][1],
                    "character_id": "Character_B",
                },
            ]
        )
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


def _exact_detection(x_shift: float = 0.0) -> dict:
    points = FFHQ_FIVE_POINT_TEMPLATE_512 / 5.0
    points[:, 0] += float(x_shift)
    return {
        "box": [20.0 + x_shift, 32.0, 82.0 + x_shift, 92.0],
        "confidence": 0.95,
        "landmarks_xy": points.tolist(),
    }


def _two_face_detections(frames: torch.Tensor, *_args, **_kwargs):
    detections = [
        [_exact_detection(0.0), _exact_detection(90.0)]
        for _ in range(int(frames.shape[0]))
    ]
    return detections, {
        "backend": "deterministic_test_yunet",
        "frame_count": len(detections),
        "detector_object_released": True,
        "five_point_landmarks": True,
    }


def _right_profile_landmark_rejection(frames: torch.Tensor, *_args, **_kwargs):
    right = _exact_detection(90.0)
    right["landmarks_xy"][1] = list(right["landmarks_xy"][0])
    detections = [
        [_exact_detection(0.0), right]
        for _ in range(int(frames.shape[0]))
    ]
    return detections, {
        "backend": "deterministic_profile_test_yunet",
        "frame_count": len(detections),
        "detector_object_released": True,
        "five_point_landmarks": True,
    }


class _FakeParser:
    def __call__(self, _value: torch.Tensor) -> torch.Tensor:
        logits = torch.full((1, 19, 512, 512), -6.0, dtype=torch.float32)
        logits[:, 0] = 4.0
        logits[:, 1, 180:400, 165:350] = 9.0
        logits[:, 2, 270:330, 235:280] = 12.0
        return logits

    def to(self, **_kwargs):
        return self


def _fake_parser_loader():
    return _FakeParser(), Path("parsing_parsenet.pth"), "deterministic-test-hash"


def _fixed_quality(*_args, **_kwargs):
    return 0.80, {
        "confidence": 0.95,
        "face_height_px": 60.0,
        "detail_score": 0.05,
        "person_overlap": 0.95,
        "profile_weight": 1.0,
        "quality_weight": 0.80,
    }


def _patch_runtime(monkeypatch, detections=_two_face_detections):
    import h3_audio_t8_pkg.skin_finish_multiface_parser as module

    monkeypatch.setattr(module, "_detect_local_opencv_yunet", detections)
    monkeypatch.setattr(module, "_load_pinned_parsenet", _fake_parser_loader)
    monkeypatch.setattr(module, "_quality_weight", _fixed_quality)


def test_five_point_normalization_sorts_viewpoint_dependent_pairs():
    detection = _exact_detection()
    points = detection["landmarks_xy"]
    detection["landmarks_xy"] = [points[1], points[0], points[2], points[4], points[3]]
    normalized = _normalized_five_points(detection)
    assert normalized[0, 0] < normalized[1, 0]
    assert normalized[3, 0] < normalized[4, 0]
    assert normalized[2].tolist() == pytest.approx(points[2])


def test_exact_five_point_alignment_is_finite_and_invertible():
    frame = np.zeros((96, 192, 3), dtype=np.float32)
    aligned, inverse, rms = _align_face(
        frame,
        _exact_detection(),
        maximum_alignment_rms=0.001,
    )
    assert tuple(aligned.shape) == (1, 512, 512, 3)
    assert inverse.shape == (2, 3)
    assert np.isfinite(inverse).all()
    assert rms < 1.0e-5


def test_two_people_two_shots_use_real_semantic_masks_and_optional_labels(monkeypatch):
    _patch_runtime(monkeypatch)
    frames = _frames()
    plan = _track_plan(frames)
    assignment = _identity_assignment(plan)
    mask, preview, report = run_multiface_semantic_skin_mask(
        frames,
        plan,
        identity_assignment=assignment,
        minimum_face_height_px=8.0,
        minimum_detail=0.0,
        maximum_skin_area_per_frame=1.0,
        minimum_ready_frame_fraction=1.0,
        preview_count=2,
    )
    parsed = json.loads(report)
    assert parsed["schema"] == SKIN_FINISH_MULTIFACE_SEMANTIC_SCHEMA
    assert parsed["status"] == "READY"
    assert parsed["mask_proxy_sha256"]
    assert parsed["selection"]["ready_frame_fraction"] == 1.0
    assert parsed["identity_assignment"]["track_to_character"] == {
        "0:0": "Character_A",
        "0:1": "Character_B",
        "1:0": "Character_A",
        "1:1": "Character_B",
    }
    assert tuple(mask.shape) == (2, 96, 192)
    assert tuple(preview.shape) == (2, 96, 192, 3)
    assert torch.count_nonzero(mask[:, :, :96]) > 0
    assert torch.count_nonzero(mask[:, :, 96:]) > 0
    for frame_index, shot in enumerate(plan["shots"]):
        from h3_audio_t8_pkg.multiface_refine_advanced import _mask_at_source

        allowed = _mask_at_source(shot, 0, 0, 96, 192) | _mask_at_source(
            shot, 0, 1, 96, 192
        )
        assert torch.count_nonzero(mask[frame_index][~allowed]) == 0
        assert all(track["status"] == "READY" for track in parsed["frames"][frame_index]["tracks"])


def test_stale_track_source_fails_before_detector_or_parser_load(monkeypatch):
    frames = _frames()
    plan = _track_plan(frames)
    changed = torch.zeros_like(frames)
    import h3_audio_t8_pkg.skin_finish_multiface_parser as module

    monkeypatch.setattr(
        module,
        "_detect_local_opencv_yunet",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("detector must not load for a stale source")
        ),
    )
    monkeypatch.setattr(
        module,
        "_load_pinned_parsenet",
        lambda: (_ for _ in ()).throw(
            AssertionError("parser must not load for a stale source")
        ),
    )
    mask, _, report = run_multiface_semantic_skin_mask(changed, plan)
    parsed = json.loads(report)
    assert parsed["status"] == "ABSTAIN_TRACK_PLAN_SOURCE_MISMATCH"
    assert torch.count_nonzero(mask) == 0
    assert parsed["parser"]["loaded"] is False


def test_invalid_identity_assignment_fails_before_models_load(monkeypatch):
    frames = _frames()
    plan = _track_plan(frames)
    assignment = _identity_assignment(plan)
    assignment["sha256"] = "corrupted"
    import h3_audio_t8_pkg.skin_finish_multiface_parser as module

    monkeypatch.setattr(
        module,
        "_detect_local_opencv_yunet",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("detector must not load for invalid identity assignment")
        ),
    )
    mask, _, report = run_multiface_semantic_skin_mask(
        frames,
        plan,
        identity_assignment=assignment,
    )
    parsed = json.loads(report)
    assert parsed["status"] == "ABSTAIN_IDENTITY_ASSIGNMENT_INVALID"
    assert torch.count_nonzero(mask) == 0


def test_one_face_detection_cannot_be_reused_by_two_tracks(monkeypatch):
    def one_face(frames: torch.Tensor, *_args, **_kwargs):
        return [[_exact_detection()] for _ in range(int(frames.shape[0]))], {
            "backend": "one-face-test",
            "detector_object_released": True,
        }

    _patch_runtime(monkeypatch, one_face)
    frames = _frames(frame_count=1)
    plan = _track_plan(frames, duplicate_person_masks=True)
    mask, _, report = run_multiface_semantic_skin_mask(
        frames,
        plan,
        minimum_face_height_px=8.0,
        minimum_detail=0.0,
        maximum_skin_area_per_frame=1.0,
        minimum_ready_frame_fraction=1.0,
    )
    parsed = json.loads(report)
    tracks = parsed["frames"][0]["tracks"]
    assert parsed["status"] == "READY"
    assert sum(track["status"] == "READY" for track in tracks) == 1
    assert sum(track["status"] != "READY" for track in tracks) == 1
    assert torch.count_nonzero(mask) > 0


def test_profile_crop_fallback_is_opt_in_and_preserves_strict_behavior(monkeypatch):
    _patch_runtime(monkeypatch, _right_profile_landmark_rejection)
    frames = _frames(frame_count=1)
    plan = _track_plan(frames)

    strict_mask, _, strict_report = run_multiface_semantic_skin_mask(
        frames,
        plan,
        minimum_face_height_px=8.0,
        minimum_detail=0.0,
        maximum_skin_area_per_frame=1.0,
        minimum_ready_frame_fraction=1.0,
    )
    strict = json.loads(strict_report)
    assert strict["alignment"]["policy"] == "five_point_strict"
    assert strict["selection"]["profile_crop_ready_counts"] == {}
    assert strict["frames"][0]["tracks"][0]["status"] == "READY"
    assert strict["frames"][0]["tracks"][1]["status"] == (
        "ABSTAIN_ALIGNMENT_OR_PARSE_FAILED"
    )
    assert torch.count_nonzero(strict_mask[:, :, :96]) > 0
    assert torch.count_nonzero(strict_mask[:, :, 96:]) == 0

    profile_mask, _, profile_report = run_multiface_semantic_skin_mask(
        frames,
        plan,
        alignment_policy="five_point_then_profile_crop",
        profile_crop_expansion=1.45,
        minimum_face_height_px=8.0,
        minimum_detail=0.0,
        maximum_skin_area_per_frame=1.0,
        minimum_ready_frame_fraction=1.0,
    )
    profile = json.loads(profile_report)
    right = profile["frames"][0]["tracks"][1]
    assert profile["status"] == "READY"
    assert profile["alignment"]["policy"] == "five_point_then_profile_crop"
    assert profile["selection"]["profile_crop_ready_counts"] == {"0:1": 1}
    assert right["status"] == "READY"
    assert right["alignment"] == "profile_bbox_crop_fallback"
    assert "YuNet eye distance is too small" in right["alignment_fallback_reason"]
    assert right["profile_crop_expansion"] == pytest.approx(1.45)
    assert torch.count_nonzero(profile_mask[:, :, :96]) > 0
    assert torch.count_nonzero(profile_mask[:, :, 96:]) > 0


def test_unknown_profile_alignment_policy_is_rejected_before_model_load(monkeypatch):
    frames = _frames(frame_count=1)
    plan = _track_plan(frames)
    import h3_audio_t8_pkg.skin_finish_multiface_parser as module

    monkeypatch.setattr(
        module,
        "_load_pinned_parsenet",
        lambda: (_ for _ in ()).throw(
            AssertionError("parser must not load for an invalid alignment policy")
        ),
    )
    with pytest.raises(ValueError, match="Unknown alignment_policy"):
        run_multiface_semantic_skin_mask(
            frames,
            plan,
            alignment_policy="unsafe_guess",
        )


def test_low_ready_coverage_zeroes_partial_mask_but_reports_observation(monkeypatch):
    def first_frame_only(frames: torch.Tensor, *_args, **_kwargs):
        return [[_exact_detection(), _exact_detection(90.0)], []], {
            "backend": "partial-coverage-test",
            "detector_object_released": True,
        }

    _patch_runtime(monkeypatch, first_frame_only)
    frames = _frames()
    plan = _track_plan(frames)
    mask, _, report = run_multiface_semantic_skin_mask(
        frames,
        plan,
        minimum_face_height_px=8.0,
        minimum_detail=0.0,
        maximum_skin_area_per_frame=1.0,
        minimum_ready_frame_fraction=1.0,
    )
    parsed = json.loads(report)
    assert parsed["status"] == "ABSTAIN_READY_FRAME_FRACTION_BELOW_MINIMUM"
    assert parsed["selection"]["ready_frame_fraction"] == 0.0
    assert (
        parsed["selection"]["observed_ready_frame_fraction_before_global_gate"]
        == 0.5
    )
    assert torch.count_nonzero(mask) == 0


def test_node_schema_is_append_only_review_input_with_safe_defaults():
    schema = MiniMaxH3SkinFinishMultiPersonSemanticMaskT8Advanced.define_schema()
    inputs = {item.id: item for item in schema.inputs}
    assert schema.is_experimental is True
    assert inputs["identity_assignment"].optional is True
    assert inputs["parser_model"].default == "facexlib_parsenet_v0.2.2_pinned"
    assert inputs["include_neck"].default is False
    assert inputs["minimum_ready_frame_fraction"].default == 0.50


def test_profile_crop_node_is_separate_experimental_with_strict_defaults():
    schema = (
        MiniMaxH3SkinFinishMultiPersonProfileSemanticMaskT8Advanced.define_schema()
    )
    inputs = {item.id: item for item in schema.inputs}
    assert schema.is_experimental is True
    assert inputs["identity_assignment"].optional is True
    assert inputs["maximum_alignment_rms"].default == 0.08
    assert inputs["profile_crop_expansion"].default == 1.45
    assert inputs["include_neck"].default is False


def test_profile_crop_node_locks_strict_first_policy(monkeypatch):
    captured = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return (
            torch.zeros((1, 8, 8)),
            torch.zeros((1, 8, 8, 3)),
            "{}",
        )

    monkeypatch.setattr(
        profile_crop_nodes,
        "run_multiface_semantic_skin_mask",
        fake_run,
    )
    MiniMaxH3SkinFinishMultiPersonProfileSemanticMaskT8Advanced.execute(
        frames=torch.zeros((1, 8, 8, 3)),
        track_plan={"status": "READY"},
        profile_crop_expansion=1.45,
    )
    assert captured["alignment_policy"] == "five_point_then_profile_crop"
    assert captured["profile_crop_expansion"] == 1.45


def test_multiface_semantic_workflow_is_importable_documented_and_source_safe():
    path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "workflows"
        / "17-skin-finish"
        / "2026-08-24_H3_Skin_Finish_MultiPerson_Semantic_Mask_Advanced_EXP.json"
    )
    workflow = json.loads(path.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in workflow["nodes"]}
    types = [node["type"] for node in workflow["nodes"]]
    assert workflow["version"] == 0.4
    assert workflow["last_node_id"] == max(nodes)
    assert workflow["last_link_id"] == max(link[0] for link in workflow["links"])
    assert types.count("MiniMaxH3SAM31MultiPersonTrackT8Advanced") == 1
    assert (
        types.count("MiniMaxH3SkinFinishMultiPersonSemanticMaskT8Advanced")
        == 1
    )
    assert types.count("MiniMaxH3SkinFinishAdvancedT8") == 1
    assert types.count("MiniMaxH3SkinFinishTextureGuardT8Advanced") == 1
    assert types.count("MiniMaxH3SkinFinishVideoFinalizeT8Advanced") == 1
    assert types.count("MarkdownNote") == 5
    parser = next(
        node
        for node in workflow["nodes"]
        if node["type"]
        == "MiniMaxH3SkinFinishMultiPersonSemanticMaskT8Advanced"
    )
    skin_finish = next(
        node
        for node in workflow["nodes"]
        if node["type"] == "MiniMaxH3SkinFinishAdvancedT8"
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
    assert parser["widgets_values"] == [
        "facexlib_parsenet_v0.2.2_pinned",
        0.45,
        32.0,
        0.010,
        0.20,
        0.10,
        0.55,
        3,
        False,
        0.00005,
        0.35,
        0.08,
        0.50,
        6,
    ]
    assert skin_finish["widgets_values"][0] == "external_exact"
    assert skin_finish["widgets_values"][7] is False
    assert guard["widgets_values"][-1] is False
    assert finalizer["widgets_values"][-1] is False
    notes = "\n".join(
        node["widgets_values"][0]
        for node in workflow["nodes"]
        if node["type"] == "MarkdownNote"
    )
    for required in (
        "SAM3.1",
        "五点",
        "ParseNet",
        "identity_assignment",
        "ABSTAIN",
        "payload",
    ):
        assert required in notes
