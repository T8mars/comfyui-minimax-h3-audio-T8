from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

import comfy.nested_tensor

import h3_audio_t8_pkg.face_refine_parity_advanced as parity_module
from h3_audio_t8_pkg.face_refine_advanced import MANUAL_DETECTOR, canonical_json
from h3_audio_t8_pkg.core import align_frame_count, video_latent_t
from h3_audio_t8_pkg.face_refine_parity_advanced import (
    _gaussian_smooth,
    apply_face_refine_per_frame_denoise,
    build_face_refine_parity_plan,
    gate_face_refine_parity_candidate,
    inject_face_refine_parity_video_latent,
    stitch_face_refine_parity_candidate,
    validate_face_refine_manual512_relative_baseline,
)
from h3_audio_t8_pkg.nodes_face_refine_parity_advanced import (
    FACE_REFINE_PARITY_ADVANCED_NODE_CLASSES,
)


def _plan(frames: torch.Tensor):
    return build_face_refine_parity_plan(
        frames=frames,
        fps=24.0,
        detector_mode="manual_static_roi",
        detector_model=MANUAL_DETECTOR,
        detector_device="cpu",
        confidence=0.35,
        manual_roi_x=0.30,
        manual_roi_y=0.10,
        manual_roi_width=0.30,
        manual_roi_height=0.45,
        scene_cut_threshold=0.28,
        max_track_jump=0.18,
        max_gap_frames=4,
        center_smooth_window=21,
        size_smooth_window=51,
        crop_factor=3.0,
        canvas_mode="manual_384",
        require_h3_grid=True,
        analysis_chunk_frames=2,
    )


class TinyVideoVAE:
    @staticmethod
    def encode(crops):
        frame_count, height, width = crops.shape[:3]
        latent_t = video_latent_t(align_frame_count(frame_count))
        return torch.full((1, 24, latent_t, height // 16, width // 16), 0.25)


def _locked_av(frame_count=5, canvas=384):
    latent_t = 2 if frame_count == 5 else ((frame_count - 5) // 17) * 5 + 2
    video = torch.zeros((1, 24, latent_t, canvas // 16, canvas // 16))
    audio = torch.randn((1, 32, 2, 9))
    return {
        "samples": comfy.nested_tensor.NestedTensor((video, audio)),
        "noise_mask": comfy.nested_tensor.NestedTensor(
            (torch.ones_like(video), torch.zeros_like(audio))
        ),
        "marker": "preserve",
    }


def _manual512_relative_fixture(frame_count=5):
    frames = torch.zeros((frame_count, 64, 96, 3))
    plan, crops, *_ = build_face_refine_parity_plan(
        frames=frames,
        fps=24.0,
        detector_mode="manual_static_roi",
        detector_model=MANUAL_DETECTOR,
        detector_device="cpu",
        confidence=0.35,
        manual_roi_x=0.30,
        manual_roi_y=0.10,
        manual_roi_width=0.30,
        manual_roi_height=0.45,
        scene_cut_threshold=0.28,
        max_track_jump=0.18,
        max_gap_frames=4,
        center_smooth_window=21,
        size_smooth_window=51,
        crop_factor=2.5,
        canvas_mode="manual_512",
        require_h3_grid=frame_count % 17 == 5,
        analysis_chunk_frames=2,
    )
    plan["detector"] = {
        "backend": "local_ultralytics",
        "input_colour_space": "bgr_upstream_parity",
        "identity_verification": False,
    }
    _rehash(plan)
    latent = _locked_av(align_frame_count(frame_count), canvas=512)
    _, _, latent_report = inject_face_refine_parity_video_latent(
        [], latent, crops, TinyVideoVAE(), plan, "require_locked", False
    )
    denoised, denoise_report = apply_face_refine_per_frame_denoise(
        latent,
        plan,
        0.8,
        0.35,
        "relative_to_clip",
        30.0,
        120.0,
        1.0,
        9,
        "replace_video_parity",
        True,
    )
    del denoised
    stitch_report = canonical_json(
        {
            "schema": "h3_t8_face_refine_parity_stitch/v1",
            "plan_sha256": plan["plan_sha256"],
            "paste_region": "face_only",
            "mask_dilation": 24,
            "feather_source_px": 24.0,
            "colour_match": 1.0,
            "blend": 1.0,
            "undetected_frames": "fade_out",
            "fallback_count": 0,
            "alignment_tail_discarded_frames": align_frame_count(frame_count)
            - frame_count,
            "mask_outside_bit_exact": True,
            "audio_modified": False,
            "feather_canvas_policy": "upstream_chunk_midpoint_source_scale",
            "colour_match_space": "post_warp_source_coordinates",
        }
    )
    return frames, plan, latent_report, denoise_report, stitch_report


def _rehash(plan):
    plan.pop("plan_sha256", None)
    plan["plan_sha256"] = hashlib.sha256(canonical_json(plan).encode()).hexdigest()


def test_reflected_gaussian_reduces_jitter():
    values = torch.tensor([0.0, 8.0, 0.0, 8.0, 0.0], dtype=torch.float64).numpy()
    smoothed = _gaussian_smooth(values, 5)
    assert abs(smoothed[1] - smoothed[0]) < 8.0
    assert abs(smoothed[2] - smoothed[1]) < 8.0


def test_parity_plan_defaults_and_reference_crop():
    frames = torch.zeros((5, 64, 96, 3))
    plan, crops, reference, preview, report, width, height, count, reference_index = _plan(
        frames
    )

    assert plan["schema"] == "h3_t8_face_refine_parity_plan/v1"
    assert plan["metrics"]["smooth_method"] == "gaussian_reflect"
    assert plan["metrics"]["center_smooth_window"] == 21
    assert plan["metrics"]["size_smooth_window"] == 51
    assert plan["limits"]["reference_crop_is_identity_proof"] is False
    assert plan["parity_defaults"]["face_mask_geometry"] == (
        "upstream_centered_smoothed_face_rect"
    )
    assert crops.shape == (5, 384, 384, 3)
    assert reference.shape == (1, 384, 384, 3)
    assert preview.shape[0] == 5
    assert (width, height, count, reference_index) == (384, 384, 5, 0)
    assert json.loads(report)["plan_sha256"] == plan["plan_sha256"]


def test_latent_injection_reuses_audio_and_mask_object():
    frames = torch.zeros((5, 64, 96, 3))
    plan, crops, *_ = _plan(frames)
    latent = _locked_av()
    old_audio = latent["samples"].unbind()[1]
    old_mask = latent["noise_mask"]
    positive = [[torch.zeros((1, 1, 1)), {}]]

    result_positive, result, report = inject_face_refine_parity_video_latent(
        positive, latent, crops, TinyVideoVAE(), plan, "require_locked", False
    )

    assert result_positive is positive
    assert result["samples"].unbind()[1].data_ptr() == old_audio.data_ptr()
    assert result["noise_mask"] is old_mask
    assert result["marker"] == "preserve"
    assert json.loads(report)["implicit_temporal_fit"] is False


def test_per_frame_denoise_changes_only_video_mask_and_keeps_audio_zero():
    frames = torch.zeros((5, 64, 96, 3))
    plan, *_ = _plan(frames)
    crop_factor = plan["parity_defaults"]["crop_factor"]
    for index, record in enumerate(plan["frames"]):
        face_height = 30.0 + index * 22.5
        record["source_crop_box_xyxy"][1] = 0.0
        record["source_crop_box_xyxy"][3] = face_height * crop_factor
    _rehash(plan)
    latent = _locked_av()
    audio = latent["samples"].unbind()[1]

    result, report = apply_face_refine_per_frame_denoise(
        latent,
        plan,
        0.8,
        0.35,
        "absolute_px",
        30.0,
        120.0,
        1.0,
        1,
        "replace_video_parity",
        True,
    )
    video_mask, audio_mask = result["noise_mask"].unbind()

    assert float(video_mask[..., 0, :, :].mean()) > float(video_mask[..., -1, :, :].mean())
    assert torch.count_nonzero(audio_mask).item() == 0
    assert tuple(audio_mask.shape) == tuple(audio.shape)
    assert json.loads(report)["audio_samples_modified"] is False
    assert json.loads(report)["face_size_source"] == (
        "source_crop_height_divided_by_crop_factor"
    )


def test_ultralytics_parity_path_converts_comfy_rgb_to_bgr(monkeypatch):
    frames = torch.zeros((5, 64, 96, 3))
    frames[..., 0] = 1.0
    frames[..., 2] = 0.25
    observed = {}

    def fake_detector(received, detector_model, confidence, detector_device):
        observed["first"] = float(received[0, 0, 0, 0])
        observed["third"] = float(received[0, 0, 0, 2])
        return (
            [[{"box": [20.0, 8.0, 60.0, 56.0], "confidence": 0.9}] for _ in received],
            {"backend": "local_ultralytics", "model": detector_model},
        )

    monkeypatch.setattr(parity_module, "_detect_local_ultralytics", fake_detector)
    plan, *_ = build_face_refine_parity_plan(
        frames=frames,
        fps=24.0,
        detector_mode="local_ultralytics",
        detector_model="ultralytics/bbox/face_yolov8m.pt",
        detector_device="cpu",
        confidence=0.35,
        manual_roi_x=0.3,
        manual_roi_y=0.1,
        manual_roi_width=0.4,
        manual_roi_height=0.55,
        scene_cut_threshold=1.0,
        max_track_jump=0.18,
        max_gap_frames=4,
        center_smooth_window=21,
        size_smooth_window=51,
        crop_factor=2.5,
        canvas_mode="manual_512",
        require_h3_grid=True,
        analysis_chunk_frames=2,
    )

    assert observed == {"first": 0.25, "third": 1.0}
    assert plan["detector"]["input_colour_space"] == "bgr_upstream_parity"


def test_parity_face_mask_stays_centered_when_crop_is_edge_clamped():
    frames = torch.zeros((5, 64, 96, 3))
    plan, *_ = build_face_refine_parity_plan(
        frames=frames,
        fps=24.0,
        detector_mode="manual_static_roi",
        detector_model=MANUAL_DETECTOR,
        detector_device="cpu",
        confidence=0.35,
        manual_roi_x=0.30,
        manual_roi_y=0.62,
        manual_roi_width=0.30,
        manual_roi_height=0.36,
        scene_cut_threshold=1.0,
        max_track_jump=0.18,
        max_gap_frames=4,
        center_smooth_window=21,
        size_smooth_window=51,
        crop_factor=2.5,
        canvas_mode="manual_512",
        require_h3_grid=True,
        analysis_chunk_frames=2,
    )
    record = plan["frames"][0]
    face = record["crop_face_box_xyxy"]

    assert (face[1] + face[3]) / 2.0 == pytest.approx(256.0, abs=1e-6)
    assert record["source_crop_box_xyxy"][3] == pytest.approx(64.0)


def test_parity_stitch_is_bit_exact_outside_face_mask():
    frames = torch.rand((5, 64, 96, 3))
    plan, crops, *_ = _plan(frames)
    refined = (crops + 0.05).clamp(0.0, 1.0)

    result, changed, fallback, count, report = stitch_face_refine_parity_candidate(
        frames,
        refined,
        plan,
        "face_only",
        24,
        24.0,
        1.0,
        1.0,
        "fade_out",
        1.0,
        "cpu_memory_safe",
    )

    outside = changed == 0
    assert torch.equal(result[outside], frames[outside])
    assert fallback.shape == changed.shape
    assert count == 0
    assert json.loads(report)["mask_outside_bit_exact"] is True


def test_manual512_relative_baseline_guard_passes_through_exact_candidate():
    candidate, plan, latent_report, denoise_report, stitch_report = (
        _manual512_relative_fixture()
    )

    result, report = validate_face_refine_manual512_relative_baseline(
        candidate,
        plan,
        latent_report,
        denoise_report,
        stitch_report,
        "manual512_relative_author_parity_v2",
        200.0,
    )

    payload = json.loads(report)
    assert result is candidate
    assert payload["mechanical_baseline_matched"] is True
    assert payload["crop_face_height_min_px"] >= 200.0
    assert payload["quality_guaranteed"] is False
    assert payload["automatic_accept"] is False


def test_manual512_relative_baseline_explicitly_accepts_one_h3_alignment_tail():
    candidate, plan, latent_report, denoise_report, stitch_report = (
        _manual512_relative_fixture(frame_count=4)
    )

    result, report = validate_face_refine_manual512_relative_baseline(
        candidate,
        plan,
        latent_report,
        denoise_report,
        stitch_report,
        "manual512_relative_author_parity_v2",
        200.0,
    )

    payload = json.loads(report)
    assert result is candidate
    assert plan["source"]["h3_aligned_frame_count"] == 5
    assert payload["h3_alignment_tail_frames"] == 1
    assert payload["alignment_tail_discarded_frames"] == 1


def test_parity_stitch_discards_only_the_declared_one_frame_alignment_tail():
    frames = torch.zeros((4, 64, 96, 3))
    plan, crops, *_ = build_face_refine_parity_plan(
        frames=frames,
        fps=24.0,
        detector_mode="manual_static_roi",
        detector_model=MANUAL_DETECTOR,
        detector_device="cpu",
        confidence=0.35,
        manual_roi_x=0.30,
        manual_roi_y=0.10,
        manual_roi_width=0.30,
        manual_roi_height=0.45,
        scene_cut_threshold=0.28,
        max_track_jump=0.18,
        max_gap_frames=4,
        center_smooth_window=21,
        size_smooth_window=51,
        crop_factor=2.5,
        canvas_mode="manual_384",
        require_h3_grid=False,
        analysis_chunk_frames=2,
    )
    refined = torch.cat((crops, crops[-1:]), dim=0)

    result, _, _, count, report = stitch_face_refine_parity_candidate(
        frames,
        refined,
        plan,
        "face_only",
        24,
        24.0,
        1.0,
        1.0,
        "fade_out",
        1.0,
        "cpu_memory_safe",
    )

    payload = json.loads(report)
    assert result.shape[0] == 4
    assert count == 0
    assert payload["h3_aligned_frame_count"] == 5
    assert payload["alignment_tail_discarded_frames"] == 1


def test_manual512_relative_baseline_guard_rejects_wrong_scale_or_canvas():
    candidate, plan, latent_report, denoise_report, stitch_report = (
        _manual512_relative_fixture()
    )
    absolute = json.loads(denoise_report)
    absolute["scale_mode"] = "absolute_px"
    with pytest.raises(ValueError, match="scale_mode"):
        validate_face_refine_manual512_relative_baseline(
            candidate,
            plan,
            latent_report,
            canonical_json(absolute),
            stitch_report,
            "manual512_relative_author_parity_v2",
            200.0,
        )

    plan["canvas"]["mode"] = "auto_capped_768"
    _rehash(plan)
    with pytest.raises(ValueError, match="manual_512"):
        validate_face_refine_manual512_relative_baseline(
            candidate,
            plan,
            latent_report,
            denoise_report,
            stitch_report,
            "manual512_relative_author_parity_v2",
            200.0,
        )


def test_quality_gate_returns_source_when_ghost_candidate_fails_proxies():
    torch.manual_seed(7)
    frames = torch.rand((5, 64, 96, 3))
    plan, *_ = _plan(frames)
    ghost = torch.rand_like(frames)
    changed = torch.ones((5, 64, 96))

    result, accepted_mask, rejected_mask, accepted, rejected, report = (
        gate_face_refine_parity_candidate(
            frames,
            ghost,
            changed,
            plan,
            0.82,
            1.02,
            2.0,
            0.06,
            0.05,
            3,
            2,
        )
    )

    payload = json.loads(report)
    assert torch.equal(result, frames)
    assert torch.count_nonzero(accepted_mask).item() == 0
    assert torch.count_nonzero(rejected_mask).item() == rejected_mask.numel()
    assert (accepted, rejected) == (0, 5)
    assert payload["status"] == "no_frame_met_proxy_gate_source_returned"
    assert payload["source_returned_bit_exact"] is True
    assert payload["automatic_accept"] is False


def test_quality_gate_accepts_continuous_structurally_close_sharpness_gain():
    torch.manual_seed(11)
    texture = torch.rand((1, 3, 64, 96))
    smooth = torch.nn.functional.avg_pool2d(texture, 5, stride=1, padding=2)
    softer = torch.nn.functional.avg_pool2d(smooth, 3, stride=1, padding=1)
    sharper = (smooth + 0.25 * (smooth - softer)).clamp(0.0, 1.0)
    frames = smooth.permute(0, 2, 3, 1).repeat(5, 1, 1, 1)
    candidate = sharper.permute(0, 2, 3, 1).repeat(5, 1, 1, 1)
    plan, *_ = _plan(frames)
    changed = torch.ones((5, 64, 96))

    result, accepted_mask, rejected_mask, accepted, rejected, report = (
        gate_face_refine_parity_candidate(
            frames,
            candidate,
            changed,
            plan,
            0.80,
            1.001,
            5.0,
            0.20,
            0.10,
            3,
            0,
        )
    )

    payload = json.loads(report)
    assert torch.allclose(result, candidate, atol=1e-7, rtol=0.0)
    assert torch.count_nonzero(accepted_mask).item() == accepted_mask.numel()
    assert torch.count_nonzero(rejected_mask).item() == 0
    assert (accepted, rejected) == (5, 0)
    assert payload["status"] == "proxy_gated_candidate_requires_human_review"
    assert payload["quality_validated"] is False


def test_quality_gate_rejects_candidate_changes_outside_parity_mask():
    frames = torch.zeros((5, 64, 96, 3))
    candidate = frames.clone()
    candidate[:, 0, 0] = 1.0
    changed = torch.zeros((5, 64, 96))
    plan, *_ = _plan(frames)

    with pytest.raises(ValueError, match="outside changed_mask"):
        gate_face_refine_parity_candidate(
            frames,
            candidate,
            changed,
            plan,
            0.82,
            1.02,
            2.0,
            0.06,
            0.05,
            3,
            2,
        )


def test_parity_nodes_are_append_only_and_lock_upstream_defaults():
    ids = [node.define_schema().node_id for node in FACE_REFINE_PARITY_ADVANCED_NODE_CLASSES]
    assert ids == [
        "MiniMaxH3FaceRefineParityPlanT8Advanced",
        "MiniMaxH3FaceRefineParityLatentT8Advanced",
        "MiniMaxH3FaceRefinePerFrameDenoiseT8Advanced",
        "MiniMaxH3FaceRefineParityStitchT8Advanced",
        "MiniMaxH3FaceRefineQualityGateT8Advanced",
        "MiniMaxH3FaceRefineManual512RelativeBaselineT8Advanced",
    ]
    schemas = {node.define_schema().node_id: node.define_schema() for node in FACE_REFINE_PARITY_ADVANCED_NODE_CLASSES}
    assert all(schema.is_experimental for schema in schemas.values())


def test_parity_rejects_non_h3_frames_and_multi_shot_latent():
    with pytest.raises(ValueError, match=r"17n\+5"):
        _plan(torch.zeros((6, 64, 96, 3)))

    frames = torch.zeros((22, 64, 96, 3))
    frames[11:] = 1.0
    plan, crops, *_ = _plan(frames)
    with pytest.raises(ValueError, match="scene cuts"):
        inject_face_refine_parity_video_latent(
            [], _locked_av(22), crops, TinyVideoVAE(), plan, "require_locked", False
        )


def test_parity_examples_lock_reviewed_sampling_and_original_audio_mux():
    root = Path(__file__).resolve().parents[1]
    api = json.loads(
        (root / "examples" / "face_refine_parity_advanced_api.json").read_text(
            encoding="utf-8"
        )
    )
    by_type = {node["class_type"]: node for node in api.values()}
    assert by_type["KSamplerSelect"]["inputs"]["sampler_name"] == "er_sde"
    assert by_type["BasicScheduler"]["inputs"] == {
        "scheduler": "simple",
        "steps": 8,
        "denoise": 0.45,
        "model": ["10", 0],
    }
    assert by_type["LoraLoaderModelOnly"]["inputs"]["strength_model"] == 0.75
    assert by_type["LoraLoaderModelOnly"]["inputs"]["lora_name"] == (
        "minimax_h3_fl2v_turbo_4step_v0.1_comfyui_alpha8-T8-convert.safetensors"
    )
    assert by_type["UNETLoader"]["inputs"]["unet_name"] == (
        "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
    )
    assert by_type["RandomNoise"]["inputs"]["noise_seed"] == 42
    assert by_type["MiniMaxH3AudioConditioningT8"]["inputs"]["task_type"] == "Ref2VA"
    assert by_type["CreateVideo"]["inputs"]["audio"] == ["2", 1]
    assert by_type["MiniMaxH3FaceRefineParityPlanT8Advanced"]["inputs"]["crop_factor"] == 2.5
    assert by_type["MiniMaxH3FaceRefineParityPlanT8Advanced"]["inputs"]["canvas_mode"] == "manual_512"
    assert by_type["MiniMaxH3FaceRefineParityPlanT8Advanced"]["inputs"]["fps"] == 24.0
    assert (
        by_type["MiniMaxH3FaceRefineParityPlanT8Advanced"]["inputs"][
            "require_h3_grid"
        ]
        is False
    )
    assert by_type["MiniMaxH3FaceRefineParityPlanT8Advanced"]["inputs"]["detector_mode"] == "local_ultralytics"
    assert by_type["MiniMaxH3FaceRefineParityPlanT8Advanced"]["inputs"]["detector_model"] == (
        "ultralytics/bbox/face_yolov8m.pt"
    )
    conditioning_inputs = by_type["MiniMaxH3AudioConditioningT8"]["inputs"]
    assert conditioning_inputs["drive_audio"] == ["2", 1]
    assert conditioning_inputs["ref_images.ref_image_0"] == ["3", 0]
    assert conditioning_inputs["ref_images.ref_image_1"] == ["24", 0]
    denoise_inputs = by_type["MiniMaxH3FaceRefinePerFrameDenoiseT8Advanced"]["inputs"]
    assert denoise_inputs["scale_mode"] == "relative_to_clip"
    baseline_inputs = by_type[
        "MiniMaxH3FaceRefineManual512RelativeBaselineT8Advanced"
    ]["inputs"]
    assert baseline_inputs["candidate_frames"] == ["20", 0]
    assert baseline_inputs["face_plan"] == ["4", 0]
    assert baseline_inputs["latent_report_json"] == ["12", 2]
    assert baseline_inputs["denoise_report_json"] == ["13", 1]
    assert baseline_inputs["stitch_report_json"] == ["20", 4]
    assert by_type["CreateVideo"]["inputs"]["images"] == ["23", 0]

    workflow = json.loads(
        (root / "examples" / "workflows" / "H3_Face_Refine_Parity_Advanced_EXP.json")
        .read_text(encoding="utf-8")
    )
    nodes = {node["id"]: node for node in workflow["nodes"]}
    assert workflow["last_node_id"] == max(nodes)
    assert workflow["last_link_id"] == max(link[0] for link in workflow["links"])
    assert len(nodes) == 25
    assert {
        "MiniMaxH3FaceRefineParityPlanT8Advanced",
        "MiniMaxH3FaceRefineParityLatentT8Advanced",
        "MiniMaxH3FaceRefinePerFrameDenoiseT8Advanced",
        "MiniMaxH3FaceRefineParityStitchT8Advanced",
        "MiniMaxH3FaceRefineManual512RelativeBaselineT8Advanced",
    } <= {node["type"] for node in nodes.values()}
    note = next(node for node in nodes.values() if node["type"] == "MarkdownNote")
    assert "修复崩坏五官" in note["widgets_values"][0]
    assert "不是视频锐化器" in note["widgets_values"][0]
    assert "8步" in note["widgets_values"][0]
    for link_id, source, output_slot, target, input_slot, link_type in workflow["links"]:
        assert nodes[target]["inputs"][input_slot]["link"] == link_id
        assert link_id in (nodes[source]["outputs"][output_slot].get("links") or [])
        assert nodes[source]["outputs"][output_slot]["type"] == link_type
        assert nodes[target]["inputs"][input_slot]["type"] == link_type
