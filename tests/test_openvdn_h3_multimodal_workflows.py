from __future__ import annotations

import json
from pathlib import Path

from h3_audio_t8_pkg.tools.build_openvdn_h3_multimodal_workflows import (
    OUTPUT_NAMES,
    build_all,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    PROJECT_ROOT
    / "examples"
    / "workflows"
    / "10-speed"
    / "2026-09-03_H3_OpenVDN_DMD8_T2VA_0p5MP_Advanced_EXP.json"
)


def _by_type(workflow: dict, node_type: str) -> list[dict]:
    return [node for node in workflow["nodes"] if node["type"] == node_type]


def _conditioning(workflow: dict) -> dict:
    return _by_type(workflow, "MiniMaxH3AudioConditioningT8")[0]


def test_builder_covers_formal_openvdn_native_task_matrix():
    built = build_all(json.loads(SOURCE.read_text(encoding="utf-8")))
    assert set(built) == set(OUTPUT_NAMES)
    assert len({workflow["id"] for workflow in built.values()}) == len(built)
    assert all(_by_type(workflow, "CreateVideo") for workflow in built.values())
    assert all(_by_type(workflow, "SaveVideo") for workflow in built.values())
    assert all(not _by_type(workflow, "VHS_VideoCombine") for workflow in built.values())
    for workflow in built.values():
        loader = _by_type(workflow, "UNETLoader")[0]
        assert loader["widgets_values"] == [
            "minimax_h3_fl2va_int8_convrot.safetensors",
            "default",
        ]
        note = _by_type(workflow, "MarkdownNote")[0]["widgets_values"][0]
        assert "adaln_t_table curve-basis/pruned" in note


def test_keyframe_and_multi_image_workflows_wire_native_slots():
    built = build_all(json.loads(SOURCE.read_text(encoding="utf-8")))
    fl_inputs = {item["name"]: item for item in _conditioning(built["fl2va"])["inputs"]}
    assert fl_inputs["first_frame"]["link"] is not None
    assert fl_inputs["last_frame"]["link"] is not None

    ref_inputs = {
        item["name"]: item
        for item in _conditioning(built["multi_ref_images"])["inputs"]
    }
    assert ref_inputs["ref_images.ref_image_0"]["link"] is not None
    assert ref_inputs["ref_images.ref_image_1"]["link"] is not None
    assert len(_by_type(built["multi_ref_images"], "LoadImage")) == 2


def test_reference_av_and_hybrid_workflows_keep_numbered_audio_contract():
    built = build_all(json.loads(SOURCE.read_text(encoding="utf-8")))
    ref_inputs = {
        item["name"]: item
        for item in _conditioning(built["ref_video_audio"])["inputs"]
    }
    assert ref_inputs["ref_videos.ref_video_0"]["link"] is not None
    assert ref_inputs["ref_video_audios.ref_video_audio_0"]["link"] is not None
    assert _by_type(built["ref_video_audio"], "Video Slice")[0]["widgets_values"] == [
        0.0,
        2.0,
        True,
    ]

    hybrid = built["hybrid_first_audio"]
    hybrid_inputs = {item["name"]: item for item in _conditioning(hybrid)["inputs"]}
    assert hybrid_inputs["first_frame"]["link"] is not None
    assert hybrid_inputs["ref_audios.ref_audio_0"]["link"] is not None
    assert _conditioning(hybrid)["widgets_values"][4] == "Hybrid"


def test_checked_in_workflows_are_exact_builder_outputs():
    built = build_all(json.loads(SOURCE.read_text(encoding="utf-8")))
    root = PROJECT_ROOT / "examples" / "workflows" / "10-speed"
    for variant, filename in OUTPUT_NAMES.items():
        checked_in = json.loads((root / filename).read_text(encoding="utf-8"))
        assert checked_in == built[variant]
