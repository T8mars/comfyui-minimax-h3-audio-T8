from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
import torch

from h3_audio_t8_pkg import long_video_sampling_plan_advanced as plan_module
from h3_audio_t8_pkg.sampling import native_flow_sigmas


def test_disabled_plan_preserves_original_sigma_tensor_identity():
    base = native_flow_sigmas(8, 12.0)
    plan, _report = plan_module.build_long_video_sampling_plan(
        "disabled", 1, "video_sigma_linear", "0.5,0.412,0.35,0"
    )
    first, second, report = plan_module.resolve_long_video_sample_schedules(
        base, plan, shift_video=12.0, shift_audio=3.0
    )
    assert first is base
    assert second is None
    assert report["mode"] == "disabled"


def test_tail_subdivision_and_manual_second_pass_are_distinct_routes():
    base = native_flow_sigmas(8, 12.0)
    tail, _ = plan_module.build_long_video_sampling_plan(
        "tail_subdivide", 2, "video_sigma_linear", "0.5,0.412,0.35,0"
    )
    first, second, tail_report = plan_module.resolve_long_video_sample_schedules(
        base, tail, shift_video=12.0, shift_audio=3.0
    )
    assert first.numel() == base.numel() + 2
    assert second is None
    assert tail_report["first_pass_nfe"] == 10

    manual, report_json = plan_module.build_long_video_sampling_plan(
        "manual_second_pass", 1, "video_sigma_linear", "0.5,0.412,0.350,0"
    )
    first, second, manual_report = plan_module.resolve_long_video_sample_schedules(
        base, manual, shift_video=12.0, shift_audio=3.0
    )
    assert first is base
    assert torch.allclose(second, torch.tensor([0.5, 0.412, 0.35, 0.0]))
    assert manual_report["second_pass_nfe"] == 3
    assert json.loads(report_json)["audio_policy"] == "joint_av_dual_clock"


@pytest.mark.parametrize(
    "value",
    ["0.5,0.5,0", "0.5,0.3", "1.2,0", "hello,0"],
)
def test_manual_schedule_rejects_non_executable_values(value):
    with pytest.raises(ValueError):
        plan_module.build_long_video_sampling_plan(
            "manual_second_pass", 1, "video_sigma_linear", value
        )


def test_sampling_plan_node_is_appended_without_moving_old_registration_ids():
    from h3_audio_t8_pkg.nodes import MiniMaxH3AudioT8Extension

    nodes = asyncio.run(MiniMaxH3AudioT8Extension().get_node_list())
    ids = [node.define_schema().node_id for node in nodes]
    assert ids[-11] == "MiniMaxH3LongVideoSamplingPlanT8Advanced"
    assert ids[-10] == "MiniMaxH3ChunkedTwoPassGlobalNoisePlanT8Advanced"
    assert ids[-9] == "MiniMaxH3ChunkedTwoPassLowSigmaPlanT8Advanced"
    assert ids[-8] == "MiniMaxH3ChunkedTwoPassMaskedLowSigmaPlanT8Advanced"
    assert ids[-7] == "MiniMaxH3SubjectSafeRGBCompositeT8Advanced"
    assert ids[226:228] == [
        "MiniMaxH3FunControlLoaderT8Advanced",
        "MiniMaxH3FunControlApplyT8Advanced",
    ]


def test_manual_second_pass_frontend_workflow_is_optional_and_mirrored():
    root = Path(__file__).resolve().parents[1]
    relative = Path("04-long-video") / (
        "2026-08-30_H3_In_Node_Long_Video_Prompt_Relay_EAV_"
        "Manual_Second_Pass_Advanced_EXP.json"
    )
    source = root / "examples" / "workflows" / relative
    mirror = (
        root.parents[1]
        / "user"
        / "default"
        / "workflows"
        / "MiniMax H3 T8"
        / relative
    )
    assert source.read_bytes() == mirror.read_bytes()
    workflow = json.loads(source.read_text(encoding="utf-8"))
    runner = next(
        node
        for node in workflow["nodes"]
        if node["type"] == "MiniMaxH3LongVideoInNodeLoopEffectsT8Advanced"
    )
    plan = next(
        node
        for node in workflow["nodes"]
        if node["type"] == "MiniMaxH3LongVideoSamplingPlanT8Advanced"
    )
    optional_input = runner["inputs"][-1]
    assert optional_input["name"] == "long_video_sampling_plan"
    assert optional_input["link"] == 6
    assert plan["widgets_values"] == [
        "manual_second_pass",
        1,
        "video_sigma_linear",
        "0.5, 0.412, 0.350, 0",
    ]
    assert workflow["links"][-1][1:5] == [11, 0, 6, len(runner["inputs"]) - 1]
