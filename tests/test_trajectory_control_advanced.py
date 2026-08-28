from __future__ import annotations

import json

import pytest
import torch

from h3_audio_t8_pkg.nodes_trajectory_control_advanced import (
    TRAJECTORY_CONTROL_ADVANCED_NODE_CLASSES,
)
from h3_audio_t8_pkg.trajectory_control_advanced import (
    build_trajectory_control_plan,
    render_trajectory_control,
    validate_trajectory_control_plan,
)


KEYFRAMES = json.dumps(
    [
        {
            "frame": 0,
            "object_id": "hero",
            "x": 0.1,
            "y": 0.2,
            "width": 0.2,
            "height": 0.4,
            "strength": 0.8,
        },
        {
            "frame": 21,
            "object_id": "hero",
            "x": 0.7,
            "y": 0.3,
            "width": 0.2,
            "height": 0.4,
            "strength": 1.0,
        },
    ]
)


def test_nodes_are_append_only_planner_and_renderer():
    schemas = [node.define_schema() for node in TRAJECTORY_CONTROL_ADVANCED_NODE_CLASSES]
    assert [schema.node_id for schema in schemas] == [
        "MiniMaxH3TrajectoryControlPlanT8Advanced",
        "MiniMaxH3TrajectoryControlRenderT8Advanced",
    ]
    assert all(schema.is_experimental for schema in schemas)
    assert all(
        schema.category == "T8/MiniMax H3/Control/Experimental/Trajectory"
        for schema in schemas
    )


def test_smoothstep_interpolates_reviewable_h3_trajectory():
    plan, preview, report_json, object_count = build_trajectory_control_plan(
        keyframes_json=KEYFRAMES,
        width=320,
        height=192,
        length=22,
        fps=24.0,
        easing="smoothstep",
        clip_policy="clip_to_canvas",
    )
    assert object_count == 1
    assert preview.ndim == 4 and preview.shape[-1] == 3
    assert validate_trajectory_control_plan(plan)["plan_sha256"] == plan["plan_sha256"]
    middle = plan["tracks"][0]["frames"][11]
    assert 0.39 < middle["x"] < 0.45
    report = json.loads(report_json)
    assert report["object_count"] == 1
    assert "does not reproduce" in report["claim_boundary"]


def test_soft_region_renders_control_video_and_union_mask_without_audio():
    plan, *_ = build_trajectory_control_plan(
        keyframes_json=KEYFRAMES,
        width=320,
        height=192,
        length=22,
        fps=24.0,
        easing="linear",
        clip_policy="clip_to_canvas",
    )
    video, mask, preview, report_json = render_trajectory_control(
        trajectory_plan=plan,
        render_mode="soft_region",
        feather=0.01,
        line_width=4,
        background_level=0.0,
    )
    assert video.shape == (22, 192, 320, 3)
    assert mask.shape == (22, 192, 320)
    assert preview.shape[0] == 12
    assert float(mask.max()) > 0.8
    assert json.loads(report_json)["audio_modified"] is False


def test_reference_sprite_preserves_aspect_inside_planned_bbox():
    plan, *_ = build_trajectory_control_plan(
        keyframes_json=KEYFRAMES,
        width=320,
        height=192,
        length=22,
        fps=24.0,
        easing="linear",
        clip_policy="clip_to_canvas",
    )
    reference = torch.zeros(1, 40, 20, 3)
    reference[..., 0] = 1.0
    video, mask, _, _ = render_trajectory_control(
        trajectory_plan=plan,
        render_mode="reference_sprite",
        feather=0.0,
        line_width=4,
        background_level=0.0,
        reference_images=reference,
    )
    assert float(video[..., 0].max()) == pytest.approx(1.0)
    assert int(torch.count_nonzero(mask[0])) > 0
    assert float(video[..., 1:].max()) == 0.0


def test_invalid_h3_length_and_tampered_plan_fail_before_render():
    with pytest.raises(ValueError, match=r"17n\+5"):
        build_trajectory_control_plan(
            keyframes_json=KEYFRAMES,
            width=320,
            height=192,
            length=23,
            fps=24.0,
            easing="linear",
            clip_policy="clip_to_canvas",
        )
    plan, *_ = build_trajectory_control_plan(
        keyframes_json=KEYFRAMES,
        width=320,
        height=192,
        length=22,
        fps=24.0,
        easing="linear",
        clip_policy="clip_to_canvas",
    )
    plan["tracks"][0]["frames"][1]["x"] = 0.9
    with pytest.raises(ValueError, match="SHA-256"):
        validate_trajectory_control_plan(plan)
