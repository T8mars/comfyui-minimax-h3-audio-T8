from __future__ import annotations

import json
import math

import pytest

from h3_audio_t8_pkg.nodes_prompt_relay_resource_estimate_advanced import (
    MiniMaxH3PromptRelayResourceEstimateT8Advanced,
)
from h3_audio_t8_pkg.prompt_relay_advanced import build_prompt_relay_plan
from h3_audio_t8_pkg.prompt_relay_resource_estimate_advanced import (
    estimate_prompt_relay_resources,
)


def _plan(local_prompts: str = "抬手\n转身\n拉远", *, query_route=None):
    plan, *_ = build_prompt_relay_plan(
        global_prompt="夜晚红衣人物，稳定电影镜头",
        local_prompts=local_prompts,
        length=124,
        timing_mode="auto_equal",
        time_ranges="",
        math_profile="paper_v1",
        epsilon=0.1,
        allow_gaps=False,
        allow_overlaps=False,
    )
    if query_route is not None:
        from h3_audio_t8_pkg.prompt_relay_advanced import (
            configure_prompt_relay_query_route,
        )

        plan, _ = configure_prompt_relay_query_route(plan, query_route)
    return plan


def _estimate(plan=None, **overrides):
    kwargs = {
        "prompt_relay_plan": plan or _plan(),
        "width": 736,
        "height": 416,
        "query_chunk_rows": 256,
        "precision": "bf16_fp16",
        "keyframe_stills": 0,
        "reference_images_match": 0,
        "reference_video_count": 0,
        "reference_video_frames_each": 124,
        "reference_video_has_audio": False,
        "reference_video_audio_seconds_each": 5.0,
        "standalone_reference_audio_count": 0,
        "standalone_reference_audio_seconds_each": 5.0,
        "additional_text_rows": 0,
        "manual_extra_packed_rows": 0,
    }
    kwargs.update(overrides)
    return estimate_prompt_relay_resources(**kwargs)


def test_known_736x416_124_frame_video_only_estimate():
    plan = _plan()
    returned_plan, seq_len, peak_mib, summary, report_json = _estimate(plan)
    report = json.loads(report_json)

    assert returned_plan == plan
    assert report["target"]["frame_rows"] == 23 * 13 == 299
    assert report["target"]["video_latent_t"] == 37
    assert report["target"]["target_video_rows"] == 11063
    assert report["target"]["audio_latent_t"] == 207
    assert report["target"]["target_audio_rows"] == 414
    expected_seq = len(plan["compiled_prompt"].encode("utf-8")) + 11063 + 414
    assert seq_len == expected_seq
    assert report["packed_sequence"]["estimated_seq_len"] == expected_seq
    assert report["relay_bias"]["route_chunk_count"] == 44
    expected_bytes = 256 * expected_seq * 2
    assert report["relay_bias"]["peak_explicit_bias_bytes"] == expected_bytes
    assert peak_mib == pytest.approx(expected_bytes / 1024**2)
    assert "NOT total VRAM" in summary
    assert report["execution"] == {
        "model_loaded": False,
        "media_encoded": False,
        "sampling_executed": False,
    }
    assert report["relay_bias"]["implementation_allocates_dense_sxs"] is False


def test_standard_canvas_matrix_reports_all_three_h3_profiles():
    plan = _plan()
    report = json.loads(_estimate(plan)[4])
    matrix = report["standard_canvas_matrix"]
    assert [(item["width"], item["height"]) for item in matrix] == [
        (736, 416),
        (1152, 640),
        (1920, 1088),
    ]
    assert [item["frame_rows"] for item in matrix] == [299, 720, 2040]
    assert [item["target_video_rows"] for item in matrix] == [
        37 * 299,
        37 * 720,
        37 * 2040,
    ]
    prompt_rows = len(plan["compiled_prompt"].encode("utf-8"))
    for item in matrix:
        expected_seq_len = prompt_rows + item["target_video_rows"] + 414
        assert item["estimated_seq_len"] == expected_seq_len
        assert item["peak_explicit_bias_bytes"] == 256 * expected_seq_len * 2
        assert item["route_chunk_count"] == math.ceil(
            item["target_video_rows"] / 256
        )
    assert [item["selected_canvas"] for item in matrix] == [True, False, False]


def test_joint_av_adds_audio_chunks_but_not_a_larger_peak_chunk():
    report = json.loads(_estimate(_plan(query_route="joint_av_exp"))[4])
    assert report["query_route"] == "joint_av_exp"
    assert report["relay_bias"]["potential_routed_query_rows"] == 11063 + 414
    assert report["relay_bias"]["route_chunk_count"] == 44 + 2
    assert report["relay_bias"]["peak_query_rows"] == 256
    assert any("not a mode validated" in item for item in report["warnings"])


@pytest.mark.parametrize("local_prompts", ["", "只有一个局部事件"])
def test_zero_or_one_event_reports_actual_no_patch_bias(local_prompts):
    report = json.loads(_estimate(_plan(local_prompts))[4])
    assert report["event_count"] <= 1
    assert report["relay_active"] is False
    assert report["relay_bias"]["routed_query_rows"] == 0
    assert report["relay_bias"]["route_chunk_count"] == 0
    assert report["relay_bias"]["peak_explicit_bias_bytes"] == 0
    assert all(
        item["peak_explicit_bias_bytes"] == 0
        and item["route_chunk_count"] == 0
        for item in report["standard_canvas_matrix"]
    )


def test_reference_row_breakdown_uses_h3_aligned_video_length():
    report = json.loads(
        _estimate(
            keyframe_stills=2,
            reference_images_match=1,
            reference_video_count=1,
            reference_video_frames_each=48,
            reference_video_has_audio=True,
            reference_video_audio_seconds_each=2.0,
            standalone_reference_audio_count=1,
            standalone_reference_audio_seconds_each=3.0,
        )[4]
    )
    rows = report["conditioning_rows"]
    assert rows["keyframe_rows"] == 598
    assert rows["reference_image_rows"] == 299
    assert rows["reference_video_frames_each_effective"] == 39
    assert rows["reference_video_latent_t_each"] == 12
    assert rows["reference_video_rows"] == 3588
    assert rows["reference_video_audio_rows"] == 160
    assert rows["standalone_reference_audio_rows"] == 240
    assert rows["conditioning_rows_total"] == 4885


@pytest.mark.parametrize(
    "overrides",
    [
        {"width": 735},
        {"height": 0},
        {"width": 1920, "height": 1120},
        {"query_chunk_rows": 31},
        {"precision": "int8"},
        {"reference_video_count": -1},
        {"reference_video_frames_each": 4},
        {"reference_video_audio_seconds_each": float("nan")},
        {"manual_extra_packed_rows": -1},
    ],
)
def test_invalid_resource_inputs_fail_closed(overrides):
    with pytest.raises(ValueError, match="Prompt Relay Resource Estimate|Unknown"):
        _estimate(**overrides)


def test_resource_node_is_append_only_experimental_output_node():
    schema = MiniMaxH3PromptRelayResourceEstimateT8Advanced.define_schema()
    assert schema.node_id == "MiniMaxH3PromptRelayResourceEstimateT8Advanced"
    assert schema.category == "T8/MiniMax H3/Conditioning/Experimental"
    assert schema.is_experimental is True
    assert schema.is_output_node is True
    inputs = {item.id: item for item in schema.inputs}
    assert inputs["width"].default == 736
    assert inputs["height"].default == 416
    assert inputs["query_chunk_rows"].default == 256
    assert inputs["precision"].default == "bf16_fp16"
    assert inputs["additional_text_rows"].default == 256
