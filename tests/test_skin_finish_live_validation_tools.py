from __future__ import annotations

from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import run_skin_finish_live_sam31_validation as live_validation  # noqa: E402
import run_skin_finish_per_person_live_validation as person_validation  # noqa: E402
import run_skin_finish_sam31_cut_probe as cut_probe  # noqa: E402
import validate_skin_finish_speaking_representative as speaking_validation  # noqa: E402
import validate_skin_finish_quality_stream_representative as quality_stream_validation  # noqa: E402
import validate_skin_finish_quality_stream_long_representative as quality_stream_long_validation  # noqa: E402
import generate_skin_finish_oily_lora8_source as oily_lora8_validation  # noqa: E402
import validate_skin_finish_oil_control_stream as oil_control_validation  # noqa: E402


def test_full_live_validation_prompt_keeps_the_review_and_track_reports_visible():
    prompt = live_validation._build_prompt("source.mp4")

    assert prompt["1"] == {
        "class_type": "LoadVideo",
        "inputs": {"file": "source.mp4"},
    }
    assert prompt["5"]["inputs"]["maximum_people"] == 3
    assert prompt["5"]["inputs"]["scene_cut_threshold"] == pytest.approx(0.28)
    assert prompt["5"]["inputs"]["release_policy"] == "offload_sam31_after_track"
    assert prompt["6"]["class_type"] == (
        "MiniMaxH3SkinFinishMultiPersonSemanticMaskT8Advanced"
    )
    assert prompt["7"]["inputs"]["mask_source"] == "external_exact"
    assert prompt["8"]["class_type"] == (
        "MiniMaxH3SkinFinishTextureGuardT8Advanced"
    )
    assert prompt["9"]["inputs"]["accept_candidate"] is True
    assert prompt["18"] == {
        "class_type": "PreviewAny",
        "inputs": {"source": ["5", 2]},
    }
    assert prompt["19"] == {
        "class_type": "PreviewAny",
        "inputs": {"source": ["5", 3]},
    }
    sampled_indices = [
        node["inputs"]["batch_index"]
        for node in prompt.values()
        if node["class_type"] == "ImageFromBatch"
    ]
    assert sampled_indices == list(live_validation.SAMPLE_FRAMES) * 4


def test_cut_probe_is_sam_only_and_exposes_machine_readable_shot_counts():
    prompt = cut_probe._build_prompt(
        "cut.mp4",
        0.31,
        sam_text="person",
        detection_threshold=0.35,
        maximum_people=2,
    )

    assert prompt["5"]["inputs"]["scene_cut_threshold"] == pytest.approx(0.31)
    assert prompt["4"]["inputs"]["text"] == "person"
    assert prompt["5"]["inputs"]["detection_threshold"] == pytest.approx(0.35)
    assert prompt["5"]["inputs"]["maximum_people"] == 2
    assert prompt["5"]["inputs"]["preview_stride"] == 1
    assert prompt["5"]["inputs"]["release_policy"] == "offload_sam31_after_track"
    assert prompt["7"]["inputs"]["source"] == ["5", 2]
    assert prompt["8"]["inputs"]["source"] == ["5", 3]
    assert prompt["9"]["inputs"]["source"] == ["5", 4]
    class_types = {node["class_type"] for node in prompt.values()}
    assert "MiniMaxH3SkinFinishMultiPersonSemanticMaskT8Advanced" not in class_types
    assert "MiniMaxH3SkinFinishAdvancedT8" not in class_types
    assert "MiniMaxH3SkinFinishVideoFinalizeT8Advanced" not in class_types


def test_per_person_live_prompt_runs_two_explicit_routes_without_identity_guessing():
    prompt = person_validation._build_prompt("two-person.mp4")

    assert prompt["4"]["inputs"]["text"] == "person"
    assert prompt["5"]["inputs"]["maximum_people"] == 2
    assert prompt["5"]["inputs"]["detection_threshold"] == pytest.approx(0.35)
    assert prompt["5"]["inputs"]["release_policy"] == "offload_sam31_after_track"
    assert prompt["6"]["class_type"] == (
        "MiniMaxH3SkinFinishMultiPersonProfileSemanticMaskT8Advanced"
    )
    assert prompt["6"]["inputs"]["maximum_alignment_rms"] == pytest.approx(0.08)
    assert prompt["6"]["inputs"]["profile_crop_expansion"] == pytest.approx(1.45)
    assert prompt["7"]["inputs"] == {
        "selector_type": "shot_track",
        "selector": "0:0",
        "preset": "subtle",
        "amount": 0.25,
        "texture_keep": 0.95,
        "shine_control": 0.25,
        "tone_adjust": 0.0,
    }
    assert prompt["8"]["inputs"]["selector"] == "0:1"
    assert prompt["8"]["inputs"]["previous_profiles"] == ["7", 0]
    assert prompt["9"]["class_type"] == "MiniMaxH3SkinFinishPerPersonT8Advanced"
    assert prompt["9"]["inputs"]["semantic_report_json"] == ["6", 2]
    assert prompt["9"]["inputs"]["default_policy"] == "source_unmatched"
    assert prompt["9"]["inputs"]["accept_candidate"] is False
    assert "identity_assignment" not in prompt["9"]["inputs"]
    assert prompt["11"]["inputs"]["accept_candidate"] is True
    audit_id = str(person_validation.AUDIT_NODE_ID)
    audit_report_id = str(person_validation.AUDIT_REPORT_NODE_ID)
    audit_preview_id = str(person_validation.AUDIT_PREVIEW_NODE_ID)
    assert prompt["11"]["inputs"]["processed_frames"] == [audit_id, 1]
    assert prompt["17"]["inputs"]["source"] == ["5", 2]
    assert prompt["20"]["inputs"]["source"] == ["9", 8]
    assert prompt[audit_id] == {
        "class_type": "MiniMaxH3SkinFinishSafetyAuditT8Advanced",
        "inputs": {
            "source_frames": ["10", 1],
            "candidate_frames": ["10", 0],
            "used_skin_mask": ["10", 4],
            "audit_scope": "unique_track_owner",
            "temporal_policy": "hard_gate",
            "maximum_mean_abs_change": 0.08,
            "maximum_peak_abs_change": 0.30,
            "maximum_temporal_effect_jump": 0.04,
            "maximum_track_leak_fraction": 0.001,
            "minimum_temporal_pixels": 64,
            "scene_cut_reset_threshold": 0.20,
            "accept_candidate": False,
            "track_plan": ["5", 0],
            "audio_source": ["2", 1],
            "audio_passthrough": ["10", 3],
        },
    }
    assert prompt[audit_report_id] == {
        "class_type": "PreviewAny",
        "inputs": {"source": [audit_id, 7]},
    }
    assert prompt[audit_preview_id] == {
        "class_type": "PreviewImage",
        "inputs": {"images": [audit_id, 6]},
    }
    sampled_indices = [
        node["inputs"]["batch_index"]
        for node in prompt.values()
        if node["class_type"] == "ImageFromBatch"
    ]
    assert sampled_indices == list(person_validation.SAMPLE_FRAMES) * 4


@pytest.mark.parametrize("reader", [live_validation._history_text, cut_probe._history_text])
def test_validation_history_text_is_fail_closed(reader):
    history = {"outputs": {"7": {"text": ["2"]}}}

    assert reader(history, "7") == "2"
    with pytest.raises(RuntimeError, match="did not retain PreviewAny text"):
        reader(history, "8")


def test_validation_sources_and_model_contracts_are_explicit():
    assert live_validation.DEFAULT_SOURCE.name.endswith("832x736x124.mp4")
    assert cut_probe.DEFAULT_SOURCE.name.endswith("832x736x22.mp4")
    assert live_validation.SAM_MODEL.name == "sam3.1_multiplex_fp16.safetensors"
    assert live_validation.YUNET_MODEL.name == "face_detection_yunet_2023mar.onnx"
    assert live_validation.PARSENET_MODEL.name == "parsing_parsenet.pth"
    assert person_validation.DEFAULT_SOURCE.name == "ComfyUI_00001_qirzb_1786961596.mp4"
    assert (person_validation.TARGET_WIDTH, person_validation.TARGET_HEIGHT) == (
        960,
        704,
    )
    assert speaking_validation.DEFAULT_SOURCE.name == (
        "source_speaking_960x544_124f.mp4"
    )
    assert speaking_validation.EXPECTED_DIALOGUE == (
        "你在干嘛呢，我在这里呀，看看效果如何。"
    )
    assert quality_stream_validation.DEFAULT_SOURCE == speaking_validation.DEFAULT_SOURCE
    assert quality_stream_validation.EXPECTED_SOURCE_SHA256 == (
        "0330B4F36641777024509CA76135638860F52CC1899FB3A4068A5C48F8F4295F"
    )
    assert quality_stream_validation.DEFAULT_OUTPUT.name == (
        "skin-finish-quality-stream-probe-20260825"
    )
    assert quality_stream_long_validation.DEFAULT_SOURCE.name == (
        "H3_Unseen_32s_qipao_drum_dance_Interval1_r0008_cosine_bridge.mp4"
    )
    assert quality_stream_long_validation.EXPECTED_SOURCE_SHA256 == (
        "10CE6352F704700A3DBC24CBF19F503D1B6A6B244258FD6B14CCD98DF3D42BA0"
    )
    assert quality_stream_long_validation.DEFAULT_OUTPUT.name == (
        "skin-finish-quality-stream-long-32s-20260825"
    )
    assert oily_lora8_validation.LORA_NAME == (
        "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"
    )
    assert oily_lora8_validation.MODEL_NAME == (
        "minimax_h3_fl2va_int8_convrot.safetensors"
    )
    assert oily_lora8_validation.STEPS == 8


def test_oily_lora8_source_prompt_is_one_exact_eight_step_i2va_route():
    prompt = oily_lora8_validation.build_prompt("seed.png")

    assert prompt["2"]["class_type"] == "LoraLoaderBypassModelOnly"
    assert prompt["2"]["inputs"]["strength_model"] == pytest.approx(1.0)
    assert prompt["6"]["inputs"]["image"] == "seed.png"
    assert prompt["7"]["inputs"]["task_type"] == "I2VA"
    assert prompt["7"]["inputs"]["first_frame"] == ["6", 0]
    assert f"<d>{oily_lora8_validation.DIALOGUE}</d>" in (
        prompt["7"]["inputs"]["prompt"]
    )
    assert prompt["8"]["inputs"] == {
        "model": ["2", 0],
        "av_latent": ["7", 1],
        "steps": 8,
        "shift_video": 12.0,
        "shift_audio": 3.0,
        "sampler_name": "dual_clock_euler",
        "scheduler": "native_flow",
    }
    assert prompt["11"]["inputs"]["sigmas"] == ["8", 2]
    assert prompt["13"]["inputs"]["fps"] == pytest.approx(24.0)
    assert prompt["14"]["inputs"]["codec"]["codec"] == "h264"


def test_speaking_mouth_diagnostic_is_source_relative_and_non_oracular():
    import torch

    frames = torch.zeros((3, 64, 64, 3), dtype=torch.float32)
    candidate = frames.clone()
    plan = {
        "frames": [
            {"source_face_box_xyxy": [16.0, 12.0, 48.0, 52.0]}
            for _ in range(3)
        ]
    }
    detection = {
        "box": [16.0, 12.0, 48.0, 52.0],
        "landmarks_xy": [
            [26.0, 26.0],
            [38.0, 26.0],
            [32.0, 34.0],
            [27.0, 42.0],
            [37.0, 42.0],
        ],
    }
    report = speaking_validation._mouth_temporal_diagnostics(
        frames,
        candidate,
        plan,
        [[detection], [detection], [detection]],
    )

    assert report["detected_frame_count"] == 3
    assert report["protected_eye_mouth_roi_peak_abs_delta"] == 0.0
    assert report["mouth_motion_pair_count"] == 2
    assert report["mouth_motion_correlation"] is None
    assert "cannot prove phoneme correctness" in report["boundary"]


def test_speaking_candidate_encoder_reuses_the_packet_copy_finalizer_source():
    source = Path(speaking_validation.__file__).read_text(encoding="utf-8")

    assert "finalize_skin_finish_video" in source
    assert "InputImpl.VideoFromFile" in source
    assert "packet_payload_exact" in source
    assert '"-c:a",\n        "copy"' not in source
    assert 'parser.add_argument("--maximum-skin-area", type=float, default=0.20)' in source
    assert "0.20 <= float(args.maximum_skin_area) <= 0.50" in source


def test_oil_control_stream_validation_is_pinned_bounded_and_human_gated():
    source = Path(oil_control_validation.__file__).read_text(encoding="utf-8")

    assert oil_control_validation.EXPECTED_SOURCE_SHA256 == (
        "9467201FF32B491D9E45CFA823FE6FBC0AEB7C5A688D15F54FD70B69B16F1B2A"
    )
    assert 'preset="oil_control"' in source
    assert "amount=0.35" in source
    assert "texture_keep=0.90" in source
    assert "shine_control=0.35" in source
    assert "chunk_frames=2" in source
    assert "accept_candidate=True" in source
    assert '"automatic_accept": False' in source
    assert "build_review" in source
    assert "h3_model_loaded\": False" in source
    assert "sam_model_loaded\": False" in source
    assert "--confirm-run" in source
