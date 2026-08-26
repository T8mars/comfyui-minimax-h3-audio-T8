from __future__ import annotations

import json

import pytest
import torch

from h3_audio_t8_pkg.skin_finish_vretoucher_adapter import (
    VRETOUCHER_ADAPTER_SCHEMA,
    VRetouchAdapterUnavailable,
    build_vretoucher_context_plan,
    causal_context_indices,
    compose_vretoucher_current_frame,
    extract_vretoucher_context,
    square_face_crop_record,
)


def _source(frame_count: int = 12) -> torch.Tensor:
    y = torch.linspace(0.0, 1.0, 48).view(1, 48, 1, 1)
    x = torch.linspace(0.0, 1.0, 80).view(1, 1, 80, 1)
    rgb = torch.cat((x.expand(1, 48, 80, 1), y.expand(1, 48, 80, 1), (x + y) * 0.5), dim=-1)
    alpha = torch.full((1, 48, 80, 1), 0.73)
    return torch.cat((rgb, alpha), dim=-1).expand(frame_count, -1, -1, -1).clone()


def _evidence(frame_count: int = 12, key: str = "0:0"):
    return [key] * frame_count, [[24.0, 10.0, 52.0, 40.0]] * frame_count


def test_causal_six_frame_window_left_pads_and_never_wraps_tail():
    assert causal_context_indices(0, 0, 11) == [0, 0, 0, 0, 0, 0]
    assert causal_context_indices(2, 0, 11) == [0, 0, 0, 0, 1, 2]
    assert causal_context_indices(10, 10, 20) == [10, 10, 10, 10, 10, 10]
    assert causal_context_indices(14, 10, 20) == [10, 10, 11, 12, 13, 14]


def test_plan_is_shot_local_track_bound_and_fixed_to_current_output():
    frames = _source()
    keys, boxes = _evidence()
    plan = build_vretoucher_context_plan(
        frames,
        current_frame=2,
        shot_start=0,
        shot_end=11,
        track_key="0:0",
        frame_track_keys=keys,
        face_boxes=boxes,
    )
    assert plan["schema"] == VRETOUCHER_ADAPTER_SCHEMA
    assert plan["context_indices"] == [0, 0, 0, 0, 1, 2]
    assert plan["causal_only"] is True
    assert plan["tail_wrap"] is False
    assert plan["shot_boundary_reset"] is True
    assert plan["model_output_scope"] == "current_newest_frame_only"
    assert plan["automatic_accept"] is False
    assert len(plan["sha256"]) == 64


def test_track_discontinuity_or_missing_face_fails_closed():
    frames = _source()
    keys, boxes = _evidence()
    keys[1] = "0:1"
    with pytest.raises(VRetouchAdapterUnavailable) as error:
        build_vretoucher_context_plan(
            frames,
            current_frame=2,
            shot_start=0,
            shot_end=11,
            track_key="0:0",
            frame_track_keys=keys,
            face_boxes=boxes,
        )
    assert error.value.status == "ABSTAIN_TRACK_DISCONTINUITY"

    keys[1] = "0:0"
    boxes[2] = None
    with pytest.raises(VRetouchAdapterUnavailable) as error:
        build_vretoucher_context_plan(
            frames,
            current_frame=2,
            shot_start=0,
            shot_end=11,
            track_key="0:0",
            frame_track_keys=keys,
            face_boxes=boxes,
        )
    assert error.value.status == "ABSTAIN_FACE_LOCALIZATION_MISSING"


def test_square_crop_is_isotropic_and_uses_padding_instead_of_squashing():
    record = square_face_crop_record(
        [-4.0, 3.0, 18.0, 35.0],
        frame_width=80,
        frame_height=48,
        context_factor=1.5,
    )
    left, top, right, bottom = record["square_crop_xyxy_unclipped"]
    assert right - left == bottom - top == record["square_crop_side_px"]
    assert record["scale_x"] == pytest.approx(record["scale_y"])
    assert record["anisotropy"] == 1.0
    assert record["whole_frame_is_never_squashed"] is True
    assert sum(record["padding_ltrb"]) > 0


def test_extracted_context_is_six_normalized_512_square_crops():
    frames = _source()
    keys, boxes = _evidence()
    plan = build_vretoucher_context_plan(
        frames,
        current_frame=0,
        shot_start=0,
        shot_end=11,
        track_key="0:0",
        frame_track_keys=keys,
        face_boxes=boxes,
    )
    context = extract_vretoucher_context(frames, plan)
    assert context.shape == (6, 3, 512, 512)
    assert float(context.min()) >= -1.0
    assert float(context.max()) <= 1.0
    assert torch.equal(context[0], context[-1])


def test_context_matches_official_rgb_minus_one_one_preprocessing_on_square_input():
    axis = torch.linspace(0.0, 1.0, 512)
    red = axis.view(1, 512).expand(512, 512)
    green = axis.view(512, 1).expand(512, 512)
    blue = torch.full((512, 512), 0.25)
    frame = torch.stack((red, green, blue), dim=-1).unsqueeze(0)
    plan = build_vretoucher_context_plan(
        frame,
        current_frame=0,
        shot_start=0,
        shot_end=0,
        track_key="0:0",
        frame_track_keys=["0:0"],
        face_boxes=[[0.0, 0.0, 512.0, 512.0]],
        context_factor=1.0,
    )

    context = extract_vretoucher_context(frame, plan)
    official_reference = frame[0].movedim(-1, 0).mul(2.0).sub(1.0)

    assert context.shape == (6, 3, 512, 512)
    assert torch.equal(context[0], official_reference)
    assert torch.equal(context[-1], official_reference)
    assert float(context[0, 2].mean()) == pytest.approx(-0.5)


def test_compose_changes_only_reviewed_skin_and_preserves_auxiliary_channels():
    source = _source(frame_count=1)[0]
    record = square_face_crop_record(
        [24.0, 10.0, 52.0, 40.0],
        frame_width=80,
        frame_height=48,
    )
    proposal = torch.full((3, 512, 512), 0.9)
    semantic = torch.zeros((48, 80))
    semantic[13:37, 27:49] = 1.0
    person = torch.zeros((48, 80))
    person[8:44, 20:60] = 1.0
    output, effective, report_json = compose_vretoucher_current_frame(
        source,
        proposal,
        record,
        semantic,
        person_mask=person,
        feather_px=4,
    )
    report = json.loads(report_json)
    assert report["automatic_accept"] is False
    assert report["weight_and_model_inference_validated"] is False
    assert int(torch.count_nonzero(effective)) > 0
    outside = effective <= 0.0
    assert torch.equal(output[..., :3][outside], source[..., :3][outside])
    assert torch.equal(output[..., 3:], source[..., 3:])
    assert not torch.equal(output[..., :3][~outside], source[..., :3][~outside])


def test_zero_amount_and_equal_proposal_are_exact_noops():
    source = _source(frame_count=1)[0]
    record = square_face_crop_record(
        [24.0, 10.0, 52.0, 40.0],
        frame_width=80,
        frame_height=48,
    )
    semantic = torch.ones((48, 80))
    proposal = torch.zeros((3, 512, 512))
    output, _, _ = compose_vretoucher_current_frame(
        source, proposal, record, semantic, amount=0.0
    )
    assert torch.equal(output, source)

    crop_plan = build_vretoucher_context_plan(
        _source(frame_count=1),
        current_frame=0,
        shot_start=0,
        shot_end=0,
        track_key="0:0",
        frame_track_keys=["0:0"],
        face_boxes=[[24.0, 10.0, 52.0, 40.0]],
    )
    equal = extract_vretoucher_context(
        _source(frame_count=1), crop_plan, normalize_to_minus_one_one=False
    )[-1]
    output, _, _ = compose_vretoucher_current_frame(
        source, equal, record, semantic, feather_px=0
    )
    # Resize round trips are not bit-exact; the contract that matters is exact exterior.
    assert torch.equal(output[..., 3:], source[..., 3:])
