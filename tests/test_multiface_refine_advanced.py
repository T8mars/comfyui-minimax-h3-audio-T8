from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import h3_audio_t8_pkg.multiface_refine_advanced as module
from h3_audio_t8_pkg.multiface_refine_advanced import (
    ASSIGNMENT_SCHEMA,
    CHARACTER_PROFILE_SCHEMA,
    FACE_CAST_SCHEMA,
    TRACK_PLAN_SCHEMA,
    _assert_native_sam31_capability,
    assign_multiface_identities,
    build_multiface_repair_job,
    build_sam31_multiface_track_plan,
    composite_multiface_candidate,
    merge_multiface_cast,
)
from h3_audio_t8_pkg.nodes_multiface_refine_advanced import (
    MULTIFACE_REFINE_ADVANCED_NODE_CLASSES,
)


def _rehash(value):
    value.pop("sha256", None)
    value["sha256"] = module._hash_json(module._json_safe(value))
    return value


def _profile(character_id: str, value: float):
    embedding = torch.zeros(128)
    embedding[0] = value
    embedding[1] = 1.0 - value
    embedding /= torch.linalg.vector_norm(embedding)
    frames = torch.zeros((1, 64, 64, 3))
    profile = {
        "schema": CHARACTER_PROFILE_SCHEMA,
        "character_id": character_id,
        "rights_confirmed": True,
        "reference_images": frames,
        "reference_source": module._source_contract(frames),
        "reference_face_boxes": [[16.0, 12.0, 48.0, 52.0]],
        "identity_embedding": embedding,
        "identity_backend": {"license": "Apache-2.0"},
    }
    return _rehash(profile)


def _packed_people(frame_count=22, height=32, width=32):
    from comfy.ldm.sam3.tracker import pack_masks

    masks = torch.zeros((frame_count, 2, height, width))
    masks[:, 0, 4:30, 2:14] = 1
    masks[:, 1, 4:30, 18:30] = 1
    return pack_masks(masks)


def _track_plan(frames):
    packed = _packed_people(frames.shape[0])
    shot = {
        "shot_id": 0,
        "start_frame": 0,
        "end_frame": frames.shape[0] - 1,
        "frame_count": frames.shape[0],
        "object_count": 2,
        "track_keys": ["0:0", "0:1"],
        "native_object_indices": [0, 1],
        "scores": [0.9, 0.8],
        "stats": [],
        "packed_masks": packed,
        "packed_masks_sha256": module._tensor_sha256(packed),
        "mask_size": [32, 32],
    }
    plan = {
        "schema": TRACK_PLAN_SCHEMA,
        "status": "sam31_shot_local_tracks_ready",
        "source": {**module._source_contract(frames), "fps": 24.0},
        "analysis": {"height": 64, "width": 96},
        "sam31": {"maximum_people": 2},
        "shots": [shot],
        "scene_cut_threshold": 0.28,
        "scene_cut_count": 0,
        "max_scene_delta": 0.0,
        "release": {"performed": True},
        "identity_assigned": False,
        "automatic_accept": False,
    }
    return _rehash(plan)


def _cast():
    cast, *_ = merge_multiface_cast(_profile("Alice", 1.0))
    cast, *_ = merge_multiface_cast(_profile("Bob", 0.0), cast)
    return cast


def _manual_assignment(frames):
    assignment, *_ = assign_multiface_identities(
        frames=frames,
        track_plan=_track_plan(frames),
        face_cast=_cast(),
        identity_mode="manual_only",
        manual_assignments_json='{"0:0":"Alice","0:1":"Bob"}',
        minimum_similarity=0.4,
        minimum_margin=0.05,
        identity_samples_per_track=2,
        strict_identity=True,
        preview_stride=8,
    )
    return assignment


def test_multiface_nodes_are_append_only_advanced_contracts():
    schemas = [node.define_schema() for node in MULTIFACE_REFINE_ADVANCED_NODE_CLASSES]
    assert [schema.node_id for schema in schemas] == [
        "MiniMaxH3FaceCharacterProfileT8Advanced",
        "MiniMaxH3FaceCastMergeT8Advanced",
        "MiniMaxH3SAM31MultiPersonTrackT8Advanced",
        "MiniMaxH3FaceTrackAssignT8Advanced",
        "MiniMaxH3MultiFaceRepairJobT8Advanced",
        "MiniMaxH3MultiFaceCompositeT8Advanced",
    ]
    assert all(schema.is_experimental for schema in schemas)
    assert all(schema.node_id.endswith("Advanced") for schema in schemas)
    assert all("Face Refine Multi-Person" in schema.category for schema in schemas)


def test_native_sam31_capability_probe_accepts_only_multiplex_tracker():
    class SAM31Tracker:
        def track_video_with_detection(self):
            return None

    class Diffusion:
        tracker = SAM31Tracker()

        def forward_video(self):
            return None

    model = type(
        "Patcher",
        (),
        {"model": type("Base", (), {"diffusion_model": Diffusion()})()},
    )()
    report = _assert_native_sam31_capability(model)
    assert report["tracker_class"] == "SAM31Tracker"
    assert report["multiplex_text_detection"] is True

    Diffusion.tracker = type("SAM3Tracker", (), {})()
    with pytest.raises(ValueError, match="requires the current ComfyUI native SAM3.1"):
        _assert_native_sam31_capability(model)


def test_cast_is_limited_to_three_and_rejects_duplicates():
    cast = _cast()
    assert cast["schema"] == FACE_CAST_SCHEMA
    assert cast["character_ids"] == ["Alice", "Bob"]
    third, *_ = merge_multiface_cast(_profile("Carol", 0.5), cast)
    assert len(third["profiles"]) == 3
    try:
        merge_multiface_cast(_profile("Alice", 1.0), cast)
    except ValueError as error:
        assert "Duplicate character_id" in str(error)
    else:
        raise AssertionError("duplicate character IDs must fail closed")


def test_native_sam_plan_is_shot_local_sorted_and_selectively_unloaded(monkeypatch):
    frames = torch.zeros((22, 64, 96, 3))
    packed = _packed_people()
    monkeypatch.setattr(
        module,
        "_assert_native_sam31_capability",
        lambda model: {
            "tracker_class": "SAM31Tracker",
            "native_forward_video": True,
            "multiplex_text_detection": True,
            "contract_check": "test_double",
        },
    )
    monkeypatch.setattr(
        module,
        "_run_native_track",
        lambda *args, **kwargs: {
            "packed_masks": packed,
            "n_frames": 22,
            "scores": [0.91, 0.82],
        },
    )
    monkeypatch.setattr(
        module,
        "_selectively_unload_model",
        lambda model: {
            "policy": "offload_sam31_after_track",
            "scope": "selected_model_and_clones",
            "global_unload_called": False,
            "performed": True,
        },
    )
    plan, preview, report_json, shot_count, track_count = build_sam31_multiface_track_plan(
        frames=frames,
        model=object(),
        conditioning=[[torch.zeros(1), {}]],
        fps=24.0,
        maximum_people=2,
        detection_threshold=0.5,
        detect_interval=3,
        scene_cut_threshold=0.28,
        analysis_max_side=64,
        preview_stride=8,
        release_policy="offload_sam31_after_track",
    )
    assert plan["schema"] == TRACK_PLAN_SCHEMA
    assert plan["shots"][0]["track_keys"] == ["0:0", "0:1"]
    assert plan["release"]["scope"] == "selected_model_and_clones"
    assert preview.shape == (4, 64, 96, 3)
    assert (shot_count, track_count) == (1, 2)
    assert json.loads(report_json)["release"]["global_unload_called"] is False


def test_manual_identity_assignment_is_authoritative_and_strict():
    frames = torch.zeros((22, 64, 96, 3))
    assignment = _manual_assignment(frames)
    assert assignment["schema"] == ASSIGNMENT_SCHEMA
    assert [(item["track_key"], item["character_id"]) for item in assignment["mappings"]] == [
        ("0:0", "Alice"),
        ("0:1", "Bob"),
    ]
    assert all(item["source"] == "manual_override" for item in assignment["mappings"])


def test_repair_job_uses_person_mask_face_localization_and_h3_grid(monkeypatch):
    frames = torch.zeros((22, 64, 96, 3))
    assignment = _manual_assignment(frames)

    def detections(batch, *args, **kwargs):
        output = []
        for _ in range(batch.shape[0]):
            output.append(
                [
                    {
                        "box": [5.0, 8.0, 30.0, 48.0],
                        "confidence": 0.9,
                        "landmarks_xy": [[10, 20], [20, 20], [15, 28], [11, 36], [20, 36]],
                    },
                    {
                        "box": [62.0, 8.0, 90.0, 48.0],
                        "confidence": 0.9,
                        "landmarks_xy": [[68, 20], [82, 20], [75, 28], [69, 36], [82, 36]],
                    },
                ]
            )
        return output, {"backend": "test_yunet"}

    monkeypatch.setattr(module, "_detect_local_opencv_yunet", detections)
    result = build_multiface_repair_job(
        frames=frames,
        identity_assignment=assignment,
        character_id="Alice",
        shot_id=0,
        window_start_in_shot=0,
        window_frame_count=22,
        crop_factor=2.5,
        canvas_mode="manual_512",
        center_smooth_window=21,
        size_smooth_window=21,
        identity_guard="sam_track_only_exp",
        minimum_similarity=0.36,
        analysis_chunk_frames=4,
    )
    plan, window, crops, references, source_reference, preview = result[:6]
    assert plan["schema"] == "h3_t8_face_refine_parity_plan/v1"
    assert plan["multiface"]["character_id"] == "Alice"
    assert plan["multiface"]["track_key"] == "0:0"
    assert plan["source"]["frame_count"] == 22
    assert plan["canvas"]["width"] == 512
    assert window.shape == (22, 64, 96, 3)
    assert crops.shape == (22, 512, 512, 3)
    assert references.shape[0] == 1
    assert source_reference.shape == (1, 512, 512, 3)
    assert preview.shape[0] == 8


def test_repair_job_can_target_face_pixels_without_changing_legacy_default(monkeypatch):
    frames = torch.zeros((22, 64, 96, 3))
    assignment = _manual_assignment(frames)

    def detections(batch, *args, **kwargs):
        return [
            [
                {
                    "box": [5.0, 8.0, 30.0, 48.0],
                    "confidence": 0.9,
                    "landmarks_xy": [[10, 20], [20, 20], [15, 28], [11, 36], [20, 36]],
                }
            ]
            for _ in range(batch.shape[0])
        ], {"backend": "test_yunet"}

    monkeypatch.setattr(module, "_detect_local_opencv_yunet", detections)
    result = build_multiface_repair_job(
        frames=frames,
        identity_assignment=assignment,
        character_id="Alice",
        shot_id=0,
        window_start_in_shot=0,
        window_frame_count=22,
        crop_factor=2.5,
        canvas_mode="manual_512",
        center_smooth_window=21,
        size_smooth_window=21,
        identity_guard="sam_track_only_exp",
        minimum_similarity=0.36,
        analysis_chunk_frames=4,
        crop_scale_mode="target_face_px",
        target_face_px=300.0,
    )
    plan = result[0]
    report = json.loads(result[6])
    assert plan["parity_defaults"]["crop_scale_mode"] == "target_face_px"
    assert plan["parity_defaults"]["requested_crop_factor"] == 2.5
    assert plan["parity_defaults"]["crop_factor"] == pytest.approx(512.0 / 300.0)
    assert plan["metrics"]["crop_face_height_mean_px"] >= 300.0
    assert plan["metrics"]["source_boundary_limited_frames"] == 22
    assert report["target_face_px"] == 300.0
    assert report["achieved_crop_face_height_px"]["mean"] >= 300.0
    assert report["source_boundary_limited_frames"] == 22


def test_target_face_pixels_rejects_ambiguous_auto_canvas():
    with pytest.raises(ValueError, match="requires a manual canvas mode"):
        module._resolve_multiface_crop_scale(2.5, "auto_capped_768", "target_face_px", 300.0)


def test_composite_chains_disjoint_people_and_rejects_overlap(monkeypatch):
    frames = torch.zeros((22, 64, 96, 3))
    assignment = _manual_assignment(frames)

    def detections(batch, *args, **kwargs):
        return [
            [
                {
                    "box": [5.0, 8.0, 30.0, 48.0],
                    "confidence": 0.9,
                    "landmarks_xy": [[10, 20], [20, 20], [15, 28], [11, 36], [20, 36]],
                },
                {
                    "box": [62.0, 8.0, 90.0, 48.0],
                    "confidence": 0.9,
                    "landmarks_xy": [[68, 20], [82, 20], [75, 28], [69, 36], [82, 36]],
                },
            ]
            for _ in range(batch.shape[0])
        ], {"backend": "test_yunet"}

    monkeypatch.setattr(module, "_detect_local_opencv_yunet", detections)
    plans = []
    for character in ("Alice", "Bob"):
        plans.append(
            build_multiface_repair_job(
                frames,
                assignment,
                character,
                0,
                0,
                22,
                2.5,
                "manual_384",
                21,
                21,
                "sam_track_only_exp",
                0.36,
                4,
            )[0]
        )
    mask_a = torch.zeros((22, 64, 96))
    mask_b = torch.zeros_like(mask_a)
    mask_a[:, 8:48, 5:30] = 1
    mask_b[:, 8:48, 62:90] = 1
    candidate_a = frames.clone()
    candidate_b = frames.clone()
    candidate_a[mask_a.bool()] = 0.25
    candidate_b[mask_b.bool()] = 0.75
    out_a, state, _, count = composite_multiface_candidate(
        frames, candidate_a, mask_a, plans[0], True, "reject"
    )
    out_b, state, _, count = composite_multiface_candidate(
        frames, candidate_b, mask_b, plans[1], True, "reject", state
    )
    assert count == 2
    assert torch.all(out_b[mask_a.bool()] == 0.25)
    assert torch.all(out_b[mask_b.bool()] == 0.75)
    feather_tail = torch.zeros_like(mask_a)
    feather_tail[:, 52, 48] = 5e-7
    feather_candidate = frames.clone()
    feather_candidate[:, 52, 48] = 0.5
    feathered, _, feather_report, _ = composite_multiface_candidate(
        frames,
        feather_candidate,
        feather_tail,
        plans[0],
        False,
        "reject",
        state,
    )
    assert torch.equal(feathered, out_b)
    assert json.loads(feather_report)["outside_mask_bit_exact"] is True
    overlap = mask_a.clone()
    try:
        composite_multiface_candidate(
            frames, candidate_a, overlap, plans[0], True, "reject", state
        )
    except ValueError as error:
        assert "overlaps" in str(error)
    else:
        raise AssertionError("overlapping multi-person masks must fail closed")


def test_two_and_three_person_examples_are_review_gated_and_sequential():
    root = Path(__file__).resolve().parents[1]
    for count in (2, 3):
        api = json.loads(
            (root / "examples" / f"multiface_sam31_{count}person_advanced_api.json").read_text(
                encoding="utf-8"
            )
        )
        types = [node["class_type"] for node in api.values()]
        assert types.count("MiniMaxH3FaceCharacterProfileT8Advanced") == count
        assert types.count("MiniMaxH3MultiFaceRepairJobT8Advanced") == count
        assert types.count("MiniMaxH3MultiFaceCompositeT8Advanced") == count
        assert types.count("SamplerCustomAdvanced") == count
        assert "MiniMaxH3FaceRefineManual512RelativeBaselineT8Advanced" not in types

        track = next(
            node
            for node in api.values()
            if node["class_type"] == "MiniMaxH3SAM31MultiPersonTrackT8Advanced"
        )
        assert track["inputs"]["maximum_people"] == count
        assert track["inputs"]["analysis_max_side"] == 512
        assert track["inputs"]["release_policy"] == "offload_sam31_after_track"
        sam_prompt = next(
            node
            for node in api.values()
            if node["class_type"] == "CLIPTextEncode"
            and node["inputs"].get("clip") == [
                next(
                    node_id
                    for node_id, candidate in api.items()
                    if candidate["class_type"] == "CheckpointLoaderSimple"
                ),
                1,
            ]
        )
        assert sam_prompt["inputs"]["text"] == "front-facing person with a visible face"

        jobs = [
            node
            for node in api.values()
            if node["class_type"] == "MiniMaxH3MultiFaceRepairJobT8Advanced"
        ]
        assert all(node["inputs"]["window_frame_count"] == 73 for node in jobs)
        assert all(node["inputs"]["crop_scale_mode"] == "target_face_px" for node in jobs)
        assert all(node["inputs"]["target_face_px"] == 300.0 for node in jobs)

        schedulers = [
            node for node in api.values() if node["class_type"] == "BasicScheduler"
        ]
        assert len(schedulers) == count
        assert all(node["inputs"]["steps"] == 8 for node in schedulers)

        profiles = [
            node
            for node in api.values()
            if node["class_type"] == "MiniMaxH3FaceCharacterProfileT8Advanced"
        ]
        assert all(node["inputs"]["rights_confirmed"] is False for node in profiles)
        assignment = next(
            node
            for node in api.values()
            if node["class_type"] == "MiniMaxH3FaceTrackAssignT8Advanced"
        )
        assert assignment["inputs"]["identity_mode"] == "sface_cpu_suggest"
        assert len(json.loads(assignment["inputs"]["manual_assignments_json"])) == count

        composites = [
            (node_id, node)
            for node_id, node in api.items()
            if node["class_type"] == "MiniMaxH3MultiFaceCompositeT8Advanced"
        ]
        assert all(node["inputs"]["accept_candidate"] is False for _, node in composites)
        assert "previous_composite" not in composites[0][1]["inputs"]
        for index in range(1, count):
            assert composites[index][1]["inputs"]["previous_composite"] == [
                composites[index - 1][0],
                1,
            ]

        conditioning = [
            node
            for node in api.values()
            if node["class_type"] == "MiniMaxH3AudioConditioningT8"
        ]
        assert len(conditioning) == count
        assert all(node["inputs"]["length"] == 73 for node in conditioning)
        assert all(node["inputs"]["audio_mode"] == "lock_source" for node in conditioning)
        assert all(node["inputs"]["drive_audio"][1] == 1 for node in conditioning)

        frontend = json.loads(
            (
                root
                / "examples"
                / "workflows"
                / f"H3_SAM31_{count}Person_Face_Refine_Advanced_EXP.json"
            ).read_text(encoding="utf-8")
        )
        frontend_nodes = {node["id"]: node for node in frontend["nodes"]}
        assert frontend["last_node_id"] == max(frontend_nodes)
        note = next(node for node in frontend_nodes.values() if node["type"] == "MarkdownNote")
        assert "不是锐化/超分节点" in note["widgets_values"][0]
        assert "target_face_px=300" in note["widgets_values"][0]
        assert "8步" in note["widgets_values"][0]
        assert sum(
            node["type"] == "MiniMaxH3MultiFaceCompositeT8Advanced"
            for node in frontend_nodes.values()
        ) == count
        for link_id, origin, origin_slot, target, target_slot, link_type in frontend["links"]:
            assert link_id in frontend_nodes[origin]["outputs"][origin_slot]["links"]
            assert frontend_nodes[target]["inputs"][target_slot]["link"] == link_id
            assert frontend_nodes[target]["inputs"][target_slot]["type"] == link_type
