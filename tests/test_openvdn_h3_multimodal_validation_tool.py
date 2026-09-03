from __future__ import annotations

from argparse import Namespace

from h3_audio_t8_pkg.tools import run_openvdn_h3_multimodal_validation as tool


def _args() -> Namespace:
    return Namespace(
        image="main.png",
        first_image="first.png",
        last_image="last.png",
        ref_image_1="one.png",
        ref_image_2="two.png",
        ref_video="reference.mp4",
    )


def _conditioning(variant: str):
    args = _args()
    args.width = 736
    args.height = 416
    args.frame_count = 39
    args.seed = 1
    graph = tool.build_variant_prompt(args, "test", variant_name=variant)
    return graph, graph["6"]["inputs"]


def test_validation_matrix_covers_requested_native_layouts():
    assert set(tool.VARIANTS) == {
        "i2va",
        "l2va",
        "fl2va",
        "ref2va",
        "multi_ref_images",
        "ref_video_audio",
        "ref_audio",
        "hybrid_first_audio",
    }


def test_first_last_and_multi_reference_graphs_are_wired_exactly():
    _graph, inputs = _conditioning("fl2va")
    assert inputs["task_type"] == "FL2VA"
    assert inputs["first_frame"] == ["20", 0]
    assert inputs["last_frame"] == ["21", 0]

    graph, inputs = _conditioning("multi_ref_images")
    assert inputs["task_type"] == "Ref2VA"
    assert inputs["ref_images.ref_image_0"] == ["20", 0]
    assert inputs["ref_images.ref_image_1"] == ["21", 0]
    assert graph["20"]["inputs"]["image"] == "one.png"
    assert graph["21"]["inputs"]["image"] == "two.png"


def test_reference_video_audio_uses_matching_components():
    graph, inputs = _conditioning("ref_video_audio")
    assert graph["20"]["class_type"] == "LoadVideo"
    assert graph["21"]["class_type"] == "GetVideoComponents"
    assert inputs["ref_videos.ref_video_0"] == ["21", 0]
    assert inputs["ref_video_audios.ref_video_audio_0"] == ["21", 1]
    assert "<Video 1>" in inputs["prompt"]
    assert "<Audio 1>" in inputs["prompt"]


def test_real_validation_also_uses_native_create_and_save_video():
    graph, _inputs = _conditioning("l2va")
    assert graph["4"]["inputs"]["unet_name"] == (
        "minimax_h3_fl2va_int8_convrot.safetensors"
    )
    assert graph["17"]["class_type"] == "CreateVideo"
    assert graph["17"]["inputs"]["images"] == ["12", 0]
    assert graph["17"]["inputs"]["audio"] == ["12", 1]
    assert graph["18"]["class_type"] == "SaveVideo"
    assert graph["18"]["inputs"]["video"] == ["17", 0]


def test_multimodal_validation_can_select_pruned_base_explicitly():
    args = _args()
    args.width = 320
    args.height = 192
    args.frame_count = 39
    args.seed = 1
    args.base_model = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
    graph = tool.build_variant_prompt(args, "test", variant_name="i2va")
    assert graph["4"]["inputs"]["unet_name"] == args.base_model


def test_standalone_audio_and_hybrid_use_reference_audio_slot():
    _graph, inputs = _conditioning("ref_audio")
    assert inputs["task_type"] == "Ref2VA"
    assert inputs["ref_audios.ref_audio_0"] == ["21", 1]

    _graph, inputs = _conditioning("hybrid_first_audio")
    assert inputs["task_type"] == "Hybrid"
    assert inputs["first_frame"] == ["20", 0]
    assert inputs["ref_audios.ref_audio_0"] == ["22", 1]


def test_adapter_integrity_requires_exact_shapes_and_clean_runtime_log():
    composition = {
        "adapters": [
            {
                "name": "default",
                "patch_targets": 104,
                "applied_targets": 104,
                "shape_validation": {
                    "checked_targets": 104,
                    "adaln_targets": 0,
                    "bias_diff_targets": 0,
                    "total_patch_targets": 104,
                    "all_shapes_exact": True,
                },
            },
            {
                "name": "turbo",
                "patch_targets": 259,
                "applied_targets": 259,
                "shape_validation": {
                    "checked_targets": 259,
                    "adaln_targets": 51,
                    "bias_diff_targets": 0,
                    "total_patch_targets": 259,
                    "all_shapes_exact": True,
                },
            },
        ]
    }

    assert all(tool.base.adapter_integrity_checks(composition, "clean log").values())
    checks = tool.base.adapter_integrity_checks(composition, "ERROR lora broken")
    assert checks["runtime_lora_errors_absent"] is False


def test_adapter_integrity_rejects_old_receipt_without_shape_proof():
    composition = {
        "adapters": [
            {"name": "default", "patch_targets": 104, "applied_targets": 104},
            {"name": "turbo", "patch_targets": 259, "applied_targets": 259},
        ]
    }

    checks = tool.base.adapter_integrity_checks(composition, "clean log")
    assert checks["default_104_shapes_exact"] is False
    assert checks["turbo_259_shapes_exact"] is False
    assert checks["turbo_51_adaln_shapes_exact"] is False


def test_curve_projected_adapter_requires_all_51_bias_residuals():
    composition = {
        "adapters": [
            {
                "name": "default",
                "variant": "native_full_width",
                "patch_targets": 104,
                "applied_targets": 104,
                "shape_validation": {
                    "checked_targets": 104,
                    "adaln_targets": 0,
                    "bias_diff_targets": 0,
                    "total_patch_targets": 104,
                    "all_shapes_exact": True,
                },
            },
            {
                "name": "turbo",
                "variant": "curve_projected",
                "patch_targets": 310,
                "applied_targets": 310,
                "shape_validation": {
                    "checked_targets": 259,
                    "adaln_targets": 51,
                    "bias_diff_targets": 51,
                    "total_patch_targets": 310,
                    "all_shapes_exact": True,
                },
            },
        ]
    }
    assert all(tool.base.adapter_integrity_checks(composition, "clean log").values())
    composition["adapters"][1]["shape_validation"]["bias_diff_targets"] = 50
    checks = tool.base.adapter_integrity_checks(composition, "clean log")
    assert checks["turbo_bias_residuals_exact"] is False
