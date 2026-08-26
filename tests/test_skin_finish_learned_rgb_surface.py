from __future__ import annotations

import json

import pytest
import torch

from h3_audio_t8_pkg.skin_finish_frequency import _two_pass_box_lowpass
from h3_audio_t8_pkg.skin_finish_learned_rgb_surface import (
    SKIN_FINISH_LEARNED_RGB_SURFACE_SCHEMA,
    fuse_learned_low_frequency_rgb_surface,
)


def _fixture(*, frames: int = 3):
    y = torch.linspace(0.18, 0.72, 96).view(1, 1, 96, 1)
    x = torch.linspace(0.0, 1.0, 96).view(1, 1, 1, 96)
    texture = torch.sin(x * 68.0) * 0.018
    source_rgb = torch.cat((y * 1.02, y * 0.79, y * 0.63), dim=1) + texture
    broad = torch.cat(
        (
            torch.full_like(y, -0.025),
            torch.full_like(y, 0.025),
            torch.full_like(y, 0.045),
        ),
        dim=1,
    )
    proposal_rgb = source_rgb + broad
    alpha = torch.full((frames, 1, 96, 96), 0.73)
    source = torch.cat(
        (source_rgb.expand(frames, -1, -1, -1), alpha), dim=1
    ).movedim(1, -1).contiguous().clamp(0.0, 1.0)
    proposal = torch.cat(
        (proposal_rgb.expand(frames, -1, -1, -1), alpha), dim=1
    ).movedim(1, -1).contiguous().clamp(0.0, 1.0)
    mask = torch.zeros((frames, 96, 96))
    mask[:, 15:81, 15:81] = 1.0
    return source, proposal, mask


def test_broad_rgb_surface_is_visible_and_exterior_and_aux_are_exact():
    source, proposal, mask = _fixture(frames=1)
    candidate, effective, rejected, difference, report_json = (
        fuse_learned_low_frequency_rgb_surface(source, proposal, mask)
    )
    report = json.loads(report_json)
    assert report["schema"] == SKIN_FINISH_LEARNED_RGB_SURFACE_SCHEMA
    assert report["status"] == "PASS_REQUIRES_HUMAN_REVIEW"
    assert int(torch.count_nonzero(effective)) > 0
    assert int(torch.count_nonzero(rejected)) == 0
    assert float(difference.mean()) > 1.0e-4
    outside = effective <= 0.0
    assert torch.equal(candidate[..., :3][outside], source[..., :3][outside])
    assert torch.equal(candidate[..., 3:], source[..., 3:])
    assert report["mechanical_gates"]["automatic_accept"] is False


def test_source_high_frequency_phase_is_preserved():
    source, proposal, mask = _fixture(frames=1)
    candidate = fuse_learned_low_frequency_rgb_surface(
        source, proposal, mask, surface_radius_px=16
    )[0]
    source_nchw = source[..., :3].movedim(-1, 1)
    candidate_nchw = candidate[..., :3].movedim(-1, 1)
    source_high = source_nchw - _two_pass_box_lowpass(source_nchw, 2)
    candidate_high = candidate_nchw - _two_pass_box_lowpass(candidate_nchw, 2)
    source_interior = source_high[:, :, 20:76, 20:76].reshape(1, -1)
    candidate_interior = candidate_high[:, :, 20:76, 20:76].reshape(1, -1)
    similarity = torch.nn.functional.cosine_similarity(
        source_interior,
        candidate_interior,
    )
    assert float(similarity) > 0.995


def test_proposal_high_frequency_texture_is_not_pasted():
    source, proposal, mask = _fixture(frames=1)
    checker = (
        (torch.arange(96).view(1, 96, 1) + torch.arange(96).view(1, 1, 96)) % 2
    ).float()
    proposal[..., :3] = (proposal[..., :3] + (checker[..., None] - 0.5) * 0.30).clamp(
        0.0, 1.0
    )
    candidate = fuse_learned_low_frequency_rgb_surface(
        source, proposal, mask, surface_radius_px=16
    )[0]
    candidate_change = (candidate[..., :3] - source[..., :3]).abs().mean()
    direct_proposal_change = (proposal[..., :3] - source[..., :3]).abs().mean()
    assert float(candidate_change) < float(direct_proposal_change) * 0.35


def test_equal_proposal_zero_amount_and_chunk_parity_are_exact():
    source, proposal, mask = _fixture()
    equal = fuse_learned_low_frequency_rgb_surface(source, source, mask)[0]
    zero = fuse_learned_low_frequency_rgb_surface(source, proposal, mask, amount=0.0)[0]
    one = fuse_learned_low_frequency_rgb_surface(
        source, proposal, mask, chunk_frames=1
    )[0]
    three = fuse_learned_low_frequency_rgb_surface(
        source, proposal, mask, chunk_frames=3
    )[0]
    assert torch.equal(equal, source)
    assert torch.equal(zero, source)
    assert torch.equal(one, three)


def test_candidate_rgb_delta_is_direction_preserving_and_capped():
    source, proposal, mask = _fixture(frames=1)
    proposal[..., :3] = 1.0 - proposal[..., :3]
    candidate, effective, _, _, report_json = fuse_learned_low_frequency_rgb_surface(
        source,
        proposal,
        mask,
        amount=1.0,
        maximum_proposal_low_rgb_delta=0.50,
        candidate_rgb_delta_cap=0.08,
        maximum_peak_abs_change=0.081,
        minimum_texture_ratio=0.50,
        maximum_texture_ratio=2.0,
        maximum_new_clipped_fraction=0.05,
    )
    report = json.loads(report_json)
    assert report["status"] in {
        "PASS_REQUIRES_HUMAN_REVIEW",
        "ABSTAIN_ALL_FRAMES_REJECTED",
    }
    if int(torch.count_nonzero(effective)):
        assert float((candidate - source).abs().max()) <= 0.080001


def test_too_small_change_abstains_without_modifying_source():
    source, proposal, mask = _fixture(frames=1)
    proposal[..., :3] = (source[..., :3] + 1.0e-5).clamp(0.0, 1.0)
    candidate, effective, rejected, _, report_json = (
        fuse_learned_low_frequency_rgb_surface(source, proposal, mask)
    )
    report = json.loads(report_json)
    assert report["status"] == "ABSTAIN_NO_VISIBLE_CHANGE"
    assert torch.equal(candidate, source)
    assert int(torch.count_nonzero(effective)) == 0
    assert int(torch.count_nonzero(rejected)) == 0


def test_invalid_proposal_and_parameters_fail_closed():
    source, proposal, mask = _fixture(frames=1)
    with pytest.raises(ValueError, match="exactly match"):
        fuse_learned_low_frequency_rgb_surface(source, proposal[:, :-1], mask)
    broken = proposal.clone()
    broken[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN or Inf"):
        fuse_learned_low_frequency_rgb_surface(source, broken, mask)
    with pytest.raises(ValueError, match="surface_radius_px"):
        fuse_learned_low_frequency_rgb_surface(
            source, proposal, mask, surface_radius_px=1
        )
    with pytest.raises(ValueError, match="chunk_frames"):
        fuse_learned_low_frequency_rgb_surface(
            source, proposal, mask, chunk_frames=0
        )
