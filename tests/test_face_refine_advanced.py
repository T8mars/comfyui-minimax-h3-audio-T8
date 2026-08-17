from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import comfy.nested_tensor

import h3_audio_t8_pkg.face_refine_advanced as face_refine_module
from h3_audio_t8_pkg.face_refine_advanced import (
    ANIME_FACE_V14_N_RELATIVE,
    ANIME_FACE_V14_N_SHA256,
    MANUAL_DETECTOR,
    YUNET_2023MAR_RELATIVE,
    YUNET_2023MAR_SHA256,
    _detect_local_anime_onnx_exp,
    _detect_local_opencv_yunet,
    _select_track,
    build_face_refine_plan,
    inject_face_refine_video_latent,
    local_face_detector_options,
    setup_face_refine_sampling,
    stitch_face_refine_candidate,
)
from h3_audio_t8_pkg.nodes_face_refine_advanced import FACE_REFINE_ADVANCED_NODE_CLASSES


def _plan(frames, *, scene_cut_threshold=0.28, manual_x=0.0):
    return build_face_refine_plan(
        frames=frames,
        fps=24.0,
        detector_mode="manual_static_roi",
        detector_model=MANUAL_DETECTOR,
        detector_device="cpu",
        confidence=0.35,
        manual_roi_x=manual_x,
        manual_roi_y=0.10,
        manual_roi_width=0.30,
        manual_roi_height=0.45,
        scene_cut_threshold=scene_cut_threshold,
        max_track_jump=0.18,
        max_gap_frames=4,
        smoothing_radius=2,
        crop_context_scale=3.0,
        canvas_size="384",
        require_h3_grid=True,
        analysis_chunk_frames=2,
    )


class TinyVideoVAE:
    def encode(self, crops):
        frame_count, height, width = crops.shape[:3]
        latent_t = 2 if frame_count == 5 else ((frame_count - 5) // 17) * 5 + 2
        return torch.full((1, 24, latent_t, height // 16, width // 16), 0.25)


def _locked_av(frame_count=5, canvas=384):
    latent_t = 2 if frame_count == 5 else ((frame_count - 5) // 17) * 5 + 2
    video = torch.zeros((1, 24, latent_t, canvas // 16, canvas // 16))
    audio = torch.randn((1, 32, 2, 9))
    video_mask = torch.ones_like(video)
    audio_mask = torch.zeros_like(audio)
    return {
        "samples": comfy.nested_tensor.NestedTensor((video, audio)),
        "noise_mask": comfy.nested_tensor.NestedTensor((video_mask, audio_mask)),
        "marker": "preserve",
    }


def test_plan_uses_exact_edge_geometry_and_source_bound_hash():
    frames = torch.zeros((5, 64, 96, 3))
    plan, crops, preview, report_json, canvas_w, canvas_h, frame_count = _plan(
        frames, manual_x=0.0
    )

    assert plan["schema"] == "h3_t8_face_refine_plan/v1"
    assert len(plan["plan_sha256"]) == 64
    assert plan["limits"]["single_pass_safe"] is True
    assert plan["source"]["frame_count"] == 5
    assert plan["frames"][0]["source_crop_box_xyxy"][0] == 0.0
    assert plan["frames"][0]["crop_face_box_xyxy"][0] == 0.0
    assert crops.shape == (5, 384, 384, 3)
    assert preview.shape[0] == 5
    assert (canvas_w, canvas_h) == (384, 384)
    assert frame_count == 5
    assert json.loads(report_json)["plan_sha256"] == plan["plan_sha256"]


def test_plan_detects_hard_cut_and_never_smooths_it_away():
    frames = torch.zeros((22, 64, 96, 3))
    frames[11:] = 1.0
    plan, *_ = _plan(frames, scene_cut_threshold=0.20)

    assert plan["metrics"]["scene_cut_count"] == 1
    assert plan["shots"] == [
        {"shot_id": 0, "start_frame": 0, "end_frame": 10},
        {"shot_id": 1, "start_frame": 11, "end_frame": 21},
    ]
    assert plan["limits"]["single_pass_safe"] is False


def test_plan_rejects_non_h3_frame_count_and_long_tensor_route():
    with pytest.raises(ValueError, match=r"17n\+5"):
        _plan(torch.zeros((6, 32, 32, 3)))
    with pytest.raises(ValueError, match="capped at 362"):
        _plan(torch.zeros((379, 8, 8, 3)))


def test_opencv_yunet_backend_parses_boxes_landmarks_and_releases_local_model(
    monkeypatch, tmp_path
):
    model_root = tmp_path / "models"
    model_path = model_root / YUNET_2023MAR_RELATIVE
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"fake-yunet")
    calls = {"create": 0, "detect": 0}

    class FakeDetector:
        def detect(self, _image):
            calls["detect"] += 1
            return True, np.asarray(
                [[10, 12, 20, 24, 14, 18, 24, 18, 19, 24, 15, 30, 24, 30, 0.91]],
                dtype=np.float32,
            )

    class FakeFaceDetectorYN:
        @staticmethod
        def create(*_args):
            calls["create"] += 1
            return FakeDetector()

    monkeypatch.setattr(face_refine_module, "_model_root", lambda: model_root)
    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(FaceDetectorYN=FakeFaceDetectorYN))
    detections, report = _detect_local_opencv_yunet(
        torch.zeros((5, 64, 96, 3)), YUNET_2023MAR_RELATIVE, 0.35, "cuda_auto"
    )

    assert calls == {"create": 1, "detect": 5}
    assert detections[0][0]["box"] == pytest.approx([10.0, 12.0, 30.0, 36.0])
    assert detections[0][0]["confidence"] == pytest.approx(0.91)
    assert len(detections[0][0]["landmarks_xy"]) == 5
    assert report["backend"] == "local_opencv_yunet"
    assert report["effective_device"] == "cpu"
    assert report["cached_after_execute"] is False
    assert report["model_sha256"] != YUNET_2023MAR_SHA256


def test_detector_options_include_comfy_ultralytics_face_models(monkeypatch, tmp_path):
    model_root = tmp_path / "models"
    model_path = model_root / "ultralytics" / "bbox" / "face_yolov8m.pt"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"local-face-yolo")
    monkeypatch.setattr(face_refine_module, "_model_root", lambda: model_root)

    assert "ultralytics/bbox/face_yolov8m.pt" in local_face_detector_options()


def test_anime_onnx_backend_decodes_raw_yolov8_and_runs_cpu_only(monkeypatch, tmp_path):
    model_root = tmp_path / "models"
    model_path = model_root / ANIME_FACE_V14_N_RELATIVE
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"fake-anime-face")
    calls = {"sessions": 0, "runs": 0}

    class FakeSessionOptions:
        enable_mem_pattern = True
        enable_cpu_mem_arena = True

    class FakeSession:
        def __init__(self, _path, sess_options, providers):
            calls["sessions"] += 1
            assert providers == ["CPUExecutionProvider"]
            assert sess_options.enable_mem_pattern is False
            assert sess_options.enable_cpu_mem_arena is False

        @staticmethod
        def get_inputs():
            return [SimpleNamespace(name="images")]

        @staticmethod
        def get_outputs():
            return [SimpleNamespace(name="output0")]

        @staticmethod
        def run(_outputs, inputs):
            calls["runs"] += 1
            assert inputs["images"].shape == (1, 3, 640, 640)
            return [
                np.asarray(
                    [
                        [
                            [320.0, 322.0, 100.0],
                            [320.0, 322.0, 100.0],
                            [200.0, 200.0, 50.0],
                            [200.0, 200.0, 50.0],
                            [0.90, 0.80, 0.20],
                        ]
                    ],
                    dtype=np.float32,
                )
            ]

    fake_ort = SimpleNamespace(
        SessionOptions=FakeSessionOptions,
        InferenceSession=FakeSession,
    )
    monkeypatch.setattr(face_refine_module, "_model_root", lambda: model_root)
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

    detections, report = _detect_local_anime_onnx_exp(
        torch.zeros((5, 64, 96, 3)), ANIME_FACE_V14_N_RELATIVE, 0.35, "cuda_auto"
    )

    assert calls == {"sessions": 1, "runs": 5}
    assert len(detections[0]) == 1
    assert detections[0][0]["box"] == pytest.approx([33.0, 22.0, 63.0, 42.0])
    assert detections[0][0]["confidence"] == pytest.approx(0.90)
    assert report["backend"] == "local_anime_onnx_exp"
    assert report["domain"] == "anime_only_experimental"
    assert report["effective_device"] == "cpu"
    assert report["cached_after_execute"] is False
    assert report["model_sha256"] != ANIME_FACE_V14_N_SHA256


def test_tracker_does_not_lose_a_face_when_velocity_prediction_overshoots():
    detections = [
        [{"box": [0.0, 10.0, 10.0, 20.0], "confidence": 0.9}],
        [{"box": [40.0, 10.0, 50.0, 20.0], "confidence": 0.9}],
        [{"box": [45.0, 10.0, 55.0, 20.0], "confidence": 0.9}],
        [],
        [{"box": [48.0, 10.0, 58.0, 20.0], "confidence": 0.9}],
    ]
    boxes, states, weights, multi_face_frames = _select_track(
        detections, [(0, 4)], 100, 100, 0.30, 2
    )

    assert all(box is not None for box in boxes)
    assert states == ["detected", "detected", "detected", "interpolated", "reacquired"]
    assert weights[-1] == pytest.approx(0.85)
    assert multi_face_frames == 0


def test_latent_injection_preserves_audio_and_mask_objects_exactly():
    frames = torch.zeros((5, 64, 96, 3))
    plan, crops, *_ = _plan(frames)
    latent = _locked_av()
    source_video, source_audio = latent["samples"].unbind()
    source_mask = latent["noise_mask"]
    positive = [[torch.zeros((1, 1, 1)), {}]]

    output_positive, output, report_json = inject_face_refine_video_latent(
        positive,
        latent,
        crops,
        TinyVideoVAE(),
        plan,
        "require_locked",
        False,
    )
    output_video, output_audio = output["samples"].unbind()

    assert output_positive is positive
    assert output["marker"] == "preserve"
    assert output["noise_mask"] is source_mask
    assert output_audio.data_ptr() == source_audio.data_ptr()
    assert torch.equal(output_audio, source_audio)
    assert output_video.shape == source_video.shape
    assert torch.all(output_video == 0.25)
    report = json.loads(report_json)
    assert report["audio_tensor_reused"] is True
    assert report["noise_mask_object_reused"] is True
    assert report["implicit_temporal_fit"] is False


def test_latent_injection_refuses_multishot_unlocked_audio_and_temporal_mismatch():
    frames = torch.zeros((22, 64, 96, 3))
    frames[11:] = 1.0
    plan, crops, *_ = _plan(frames, scene_cut_threshold=0.20)
    latent = _locked_av(frame_count=22)
    positive = []

    with pytest.raises(ValueError, match="scene cuts"):
        inject_face_refine_video_latent(
            positive, latent, crops, TinyVideoVAE(), plan, "require_locked", False
        )

    unlocked = latent.copy()
    video_mask, audio_mask = latent["noise_mask"].unbind()
    unlocked["noise_mask"] = comfy.nested_tensor.NestedTensor(
        (video_mask, torch.ones_like(audio_mask))
    )
    with pytest.raises(ValueError, match="nonzero audio noise_mask"):
        inject_face_refine_video_latent(
            positive, unlocked, crops, TinyVideoVAE(), plan, "require_locked", True
        )

    with pytest.raises(ValueError, match="frame count"):
        inject_face_refine_video_latent(
            positive, latent, crops[:-1], TinyVideoVAE(), plan, "preserve_existing", True
        )


def test_stitch_keeps_every_pixel_outside_mask_bit_exact():
    frames = torch.full((5, 64, 96, 3), 0.25)
    plan, crops, *_ = _plan(frames)
    refined = (crops + 0.20).clamp(0.0, 1.0)

    candidate, changed_mask, fallback_mask, fallback_count, report_json = (
        stitch_face_refine_candidate(
            frames,
            refined,
            plan,
            "ellipse",
            4.0,
            1.0,
            0.0,
            0.50,
            0,
            "cpu_memory_safe",
        )
    )
    outside = changed_mask == 0

    assert candidate.shape == frames.shape
    assert changed_mask.shape == frames.shape[:3]
    assert torch.equal(candidate[outside], frames[outside])
    assert torch.count_nonzero(changed_mask) > 0
    assert torch.count_nonzero(fallback_mask) == 0
    assert fallback_count == 0
    assert json.loads(report_json)["mask_outside_bit_exact"] is True


def test_stitch_rejects_stale_source_and_falls_back_extreme_candidate():
    frames = torch.zeros((5, 64, 96, 3))
    plan, crops, *_ = _plan(frames)
    stale = torch.ones_like(frames)
    with pytest.raises(ValueError, match="fingerprint"):
        stitch_face_refine_candidate(
            stale, crops, plan, "ellipse", 4.0, 1.0, 0.0, 0.4, 0, "cpu_memory_safe"
        )

    extreme = torch.ones_like(crops)
    candidate, changed_mask, fallback_mask, fallback_count, report_json = (
        stitch_face_refine_candidate(
            frames,
            extreme,
            plan,
            "ellipse",
            4.0,
            1.0,
            0.0,
            0.05,
            1,
            "cpu_memory_safe",
        )
    )
    assert torch.equal(candidate, frames)
    assert torch.count_nonzero(changed_mask) == 0
    assert torch.count_nonzero(fallback_mask) == fallback_mask.numel()
    assert fallback_count == 5
    assert json.loads(report_json)["status"] == "candidate_requires_review"


def test_face_refine_nodes_are_append_only_advanced_experimental_contracts():
    schemas = [node.define_schema() for node in FACE_REFINE_ADVANCED_NODE_CLASSES]
    assert [schema.node_id for schema in schemas] == [
        "MiniMaxH3FaceRefinePlanT8Advanced",
        "MiniMaxH3FaceRefineConditioningT8Advanced",
        "MiniMaxH3FaceRefineSamplerT8Advanced",
        "MiniMaxH3FaceRefineStitchAuditT8Advanced",
    ]
    for schema in schemas:
        assert schema.node_id.endswith("Advanced")
        assert schema.is_experimental is True
        assert schema.category == "T8/MiniMax H3/Quality/Experimental"
    plan_inputs = {item.id: item for item in schemas[0].inputs}
    assert plan_inputs["detector_mode"].default == "local_opencv_yunet"
    assert plan_inputs["detector_device"].default == "cpu"
    conditioning_inputs = {item.id: item for item in schemas[1].inputs}
    assert conditioning_inputs["audio_policy"].default == "require_locked"
    assert conditioning_inputs["allow_multi_shot_exp"].default is False
    sampler_inputs = {item.id: item for item in schemas[2].inputs}
    assert sampler_inputs["denoise"].default == 0.45
    assert sampler_inputs["steps"].default == 12


def test_face_refine_sampler_selects_low_noise_tail_without_changing_stable_source(monkeypatch):
    captured = {}

    def fake_setup(model, latent, steps, shift_video, shift_audio, sampler_name, scheduler):
        captured["steps"] = steps
        return model, "sampler", torch.linspace(1.0, 0.0, steps + 1)

    monkeypatch.setattr(face_refine_module, "setup_dual_clock_sampling", fake_setup)
    model, sampler, sigmas, report_json = setup_face_refine_sampling(
        "model", {}, 4, 0.5, 12.0, 3.0, "dual_clock_euler", "native_flow"
    )

    assert captured["steps"] == 8
    assert model == "model"
    assert sampler == "sampler"
    assert sigmas.tolist() == pytest.approx([0.5, 0.375, 0.25, 0.125, 0.0])
    report = json.loads(report_json)
    assert report["requested_model_calls"] == 4
    assert report["full_schedule_steps"] == 8
    assert report["sampling_py_modified"] is False


def test_face_refine_api_and_frontend_examples_are_complete_and_link_valid():
    root = Path(__file__).resolve().parents[1]
    api = json.loads((root / "examples" / "face_refine_advanced_api.json").read_text("utf-8"))
    frontend = json.loads(
        (root / "examples" / "workflows" / "H3_Face_Refine_Advanced_EXP.json").read_text(
            "utf-8"
        )
    )
    api_types = {node["class_type"] for node in api.values()}
    assert {
        "MiniMaxH3FaceRefinePlanT8Advanced",
        "MiniMaxH3FaceRefineConditioningT8Advanced",
        "MiniMaxH3FaceRefineSamplerT8Advanced",
        "MiniMaxH3FaceRefineStitchAuditT8Advanced",
    } <= api_types
    assert api["17"]["inputs"]["base_frames"] == ["2", 0]
    assert api["18"]["inputs"]["audio"] == ["7", 2]
    for node in api.values():
        for value in node.get("inputs", {}).values():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                assert value[0] in api

    nodes = {node["id"]: node for node in frontend["nodes"]}
    assert frontend["last_node_id"] == max(nodes)
    assert frontend["last_link_id"] == max(link[0] for link in frontend["links"])
    assert len(nodes) == 17
    for link_id, origin, origin_slot, target, target_slot, _type in frontend["links"]:
        assert link_id >= 1
        assert origin in nodes and target in nodes
        assert origin_slot < len(nodes[origin]["outputs"])
        assert target_slot < len(nodes[target]["inputs"])


def test_anime_face_refine_examples_change_only_the_explicit_detector_route():
    root = Path(__file__).resolve().parents[1]
    examples = root / "examples"
    human_api = json.loads((examples / "face_refine_advanced_api.json").read_text("utf-8"))
    anime_api = json.loads(
        (examples / "face_refine_anime_advanced_api.json").read_text("utf-8")
    )
    assert human_api["3"]["inputs"]["detector_mode"] == "local_opencv_yunet"
    assert anime_api["3"]["inputs"]["detector_mode"] == "local_anime_onnx_exp"
    assert anime_api["3"]["inputs"]["detector_model"] == ANIME_FACE_V14_N_RELATIVE
    assert anime_api["3"]["inputs"]["detector_device"] == "cpu"

    anime_frontend = json.loads(
        (
            examples
            / "workflows"
            / "H3_Face_Refine_Anime_Advanced_EXP.json"
        ).read_text("utf-8")
    )
    plan = next(
        node
        for node in anime_frontend["nodes"]
        if node["type"] == "MiniMaxH3FaceRefinePlanT8Advanced"
    )
    assert plan["widgets_values"][:3] == [
        "local_anime_onnx_exp",
        ANIME_FACE_V14_N_RELATIVE,
        "cpu",
    ]
