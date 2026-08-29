from __future__ import annotations

import json

import torch

from h3_audio_t8_pkg.long_video_seam_drift_advanced import (
    process_long_video_seam_drift,
)


def _textured(value: float, count: int = 8):
    grid = torch.arange(32 * 32, dtype=torch.float32).reshape(32, 32)
    texture = ((grid % 7) / 700.0)[..., None]
    frame = (torch.full((32, 32, 3), value) + texture).clamp(0, 1)
    return frame.repeat(count, 1, 1, 1)


def test_report_only_is_exact_source_identity():
    frames = torch.cat([_textured(0.4, 4), _textured(0.43, 4)])
    output, status, report = process_long_video_seam_drift(frames, "[4]")
    assert output is frames
    assert torch.equal(output, frames)
    assert status == "source_identity_report_only"
    assert json.loads(report)["audio_touched"] is False


def test_bounded_candidate_reduces_same_shot_color_step_without_texture_collapse():
    frames = torch.cat([_textured(0.40, 4), _textured(0.43, 8)])
    output, status, report_json = process_long_video_seam_drift(
        frames,
        "[4]",
        mode="bounded_candidate_exp",
        transition_frames=4,
        maximum_frame_change=0.08,
    )
    report = json.loads(report_json)
    seam = report["boundaries"][0]
    assert status == "candidate_applied"
    assert seam["seam_mad_after"] < seam["seam_mad_before"]
    assert seam["corrected_frames"] > 0
    assert torch.equal(output[:4], frames[:4])


def test_scene_cut_flash_black_and_hdr_abstain_to_source():
    cut = torch.cat([torch.zeros(4, 16, 16, 3), torch.ones(4, 16, 16, 3)])
    output, status, report = process_long_video_seam_drift(
        cut, "[4]", mode="bounded_candidate_exp"
    )
    assert status == "source_identity_abstain" and torch.equal(output, cut)
    assert "scene_cut" in json.loads(report)["boundaries"][0]["status"]

    hdr = _textured(0.4)
    output, status, report = process_long_video_seam_drift(
        hdr,
        "[4]",
        mode="bounded_candidate_exp",
        color_contract="unknown_or_hdr",
    )
    assert status == "source_identity_abstain" and torch.equal(output, hdr)
    assert "hdr" in json.loads(report)["boundaries"][0]["status"]


def test_boundary_frame_rollback_abstains_the_whole_transition_atomically():
    frames = torch.cat([_textured(0.40, 4), _textured(0.50, 8)])
    output, status, report_json = process_long_video_seam_drift(
        frames,
        "[4]",
        mode="bounded_candidate_exp",
        transition_frames=8,
        scene_cut_threshold=0.5,
        maximum_frame_change=0.001,
    )
    report = json.loads(report_json)
    assert status == "source_identity_abstain"
    assert torch.equal(output, frames)
    assert report["atomic_boundary_commit"] is True
    assert report["boundaries"][0]["status"] == "abstain_boundary_frame_rolled_back"


def test_roi_is_reported_without_hard_mask_paste_boundary():
    frames = torch.cat([_textured(0.35, 4), _textured(0.38, 4)])
    mask = torch.zeros(1, 32, 32)
    mask[:, 8:24, 8:24] = 1
    _output, _status, report = process_long_video_seam_drift(
        frames, "[4]", person_roi=mask
    )
    seam = json.loads(report)["boundaries"][0]
    assert seam["roi_luma_before"] is not None
    assert seam["roi_luma_after"] is not None


def test_bounded_tone_candidate_fades_out_instead_of_regrading_full_target():
    frames = torch.cat([_textured(0.40, 4), _textured(0.43, 12)])
    output, status, report_json = process_long_video_seam_drift(
        frames,
        "[4]",
        mode="bounded_candidate_exp",
        transition_frames=4,
        maximum_frame_change=0.08,
    )
    assert status == "candidate_applied"
    assert not torch.equal(output[4:8], frames[4:8])
    assert torch.equal(output[8:], frames[8:])
    report = json.loads(report_json)
    assert report["audio_touched"] is False
    assert report["detail_generation"] is False
