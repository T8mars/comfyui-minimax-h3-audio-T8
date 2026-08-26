from __future__ import annotations

import json

import pytest
import torch

from h3_audio_t8_pkg.skin_finish_frequency import _two_pass_box_lowpass
from h3_audio_t8_pkg.skin_finish_learned_mid_surface import (
    SKIN_FINISH_LEARNED_MID_SURFACE_SCHEMA,
    fuse_learned_skin_mid_surface,
)


def _fixture(*, frames: int = 3):
    y = torch.linspace(0.20, 0.68, 96).view(1, 1, 96, 1)
    x = torch.linspace(0.0, 1.0, 96).view(1, 1, 1, 96)
    fine = torch.sin(x * 72.0) * 0.014
    broad = torch.cat((y * 1.02, y * 0.82, y * 0.66), dim=1)
    source_rgb = broad + fine
    mid = torch.sin(x * 12.0) * 0.035
    proposal_rgb = broad + mid + fine * 1.7
    alpha = torch.full((frames, 1, 96, 96), 0.71)
    source = torch.cat(
        (source_rgb.expand(frames, -1, -1, -1), alpha), dim=1
    ).movedim(1, -1).contiguous().clamp(0.0, 1.0)
    proposal = torch.cat(
        (proposal_rgb.expand(frames, -1, -1, -1), alpha), dim=1
    ).movedim(1, -1).contiguous().clamp(0.0, 1.0)
    mask = torch.zeros((frames, 96, 96))
    mask[:, 15:81, 15:81] = 1.0
    return source, proposal, mask


def test_mid_surface_changes_skin_and_preserves_exterior_and_aux():
    source, proposal, mask = _fixture(frames=1)
    candidate, effective, rejected, difference, report_json = (
        fuse_learned_skin_mid_surface(
            source,
            proposal,
            mask,
            minimum_masked_mean_abs_change=0.002,
            minimum_applied_mid_rms=0.001,
        )
    )
    report = json.loads(report_json)
    assert report["schema"] == SKIN_FINISH_LEARNED_MID_SURFACE_SCHEMA
    assert report["status"] == "PASS_REQUIRES_IDENTITY_AND_HUMAN_REVIEW"
    assert int(torch.count_nonzero(effective)) > 0
    assert int(torch.count_nonzero(rejected)) == 0
    assert float(difference.mean()) > 1.0e-4
    outside = effective <= 0.0
    assert torch.equal(candidate[..., :3][outside], source[..., :3][outside])
    assert torch.equal(candidate[..., 3:], source[..., 3:])


def test_source_fine_phase_is_retained_in_mask_interior():
    source, proposal, mask = _fixture(frames=1)
    candidate = fuse_learned_skin_mid_surface(
        source,
        proposal,
        mask,
        minimum_masked_mean_abs_change=0.002,
        minimum_applied_mid_rms=0.001,
    )[0]
    source_nchw = source[..., :3].movedim(-1, 1)
    candidate_nchw = candidate[..., :3].movedim(-1, 1)
    source_fine = source_nchw - _two_pass_box_lowpass(source_nchw, 2)
    candidate_fine = candidate_nchw - _two_pass_box_lowpass(candidate_nchw, 2)
    a = source_fine[:, :, 22:74, 22:74].reshape(1, -1)
    b = candidate_fine[:, :, 22:74, 22:74].reshape(1, -1)
    assert float(torch.nn.functional.cosine_similarity(a, b)) > 0.995


def test_proposal_checkerboard_fine_detail_is_not_pasted():
    source, proposal, mask = _fixture(frames=1)
    checker = (
        (torch.arange(96).view(1, 96, 1) + torch.arange(96).view(1, 1, 96)) % 2
    ).float()
    proposal[..., :3] = (proposal[..., :3] + (checker[..., None] - 0.5) * 0.35).clamp(
        0.0, 1.0
    )
    candidate = fuse_learned_skin_mid_surface(
        source,
        proposal,
        mask,
        minimum_masked_mean_abs_change=0.002,
        minimum_applied_mid_rms=0.001,
        minimum_source_fine_cosine=0.98,
    )[0]
    candidate_change = (candidate[..., :3] - source[..., :3]).abs().mean()
    direct_change = (proposal[..., :3] - source[..., :3]).abs().mean()
    assert float(candidate_change) < float(direct_change) * 0.40


def test_equal_proposal_zero_amount_and_chunk_parity_are_exact():
    source, proposal, mask = _fixture()
    equal = fuse_learned_skin_mid_surface(source, source, mask)[0]
    zero = fuse_learned_skin_mid_surface(
        source, proposal, mask, broad_amount=0.0, mid_amount=0.0
    )[0]
    one = fuse_learned_skin_mid_surface(
        source,
        proposal,
        mask,
        chunk_frames=1,
        minimum_masked_mean_abs_change=0.002,
        minimum_applied_mid_rms=0.001,
    )[0]
    three = fuse_learned_skin_mid_surface(
        source,
        proposal,
        mask,
        chunk_frames=3,
        minimum_masked_mean_abs_change=0.002,
        minimum_applied_mid_rms=0.001,
    )[0]
    assert torch.equal(equal, source)
    assert torch.equal(zero, source)
    assert torch.equal(one, three)


def test_rgb_delta_is_capped_and_direction_is_not_channel_clipped_independently():
    source, proposal, mask = _fixture(frames=1)
    proposal[..., :3] = 1.0 - proposal[..., :3]
    candidate = fuse_learned_skin_mid_surface(
        source,
        proposal,
        mask,
        broad_amount=1.0,
        mid_amount=1.0,
        maximum_broad_component_delta=0.50,
        maximum_mid_component_delta=0.30,
        candidate_rgb_delta_cap=0.08,
        minimum_masked_mean_abs_change=0.002,
        minimum_applied_mid_rms=0.0,
        minimum_source_fine_cosine=0.90,
        minimum_texture_ratio=0.50,
        maximum_texture_ratio=2.0,
        maximum_new_clipped_fraction=0.05,
    )[0]
    assert float((candidate - source).abs().max()) <= 0.080001


def test_insufficient_mid_effect_abstains_exactly():
    source, proposal, mask = _fixture(frames=1)
    proposal[..., :3] = (source[..., :3] + 1.0e-5).clamp(0.0, 1.0)
    candidate, effective, rejected, _, report_json = fuse_learned_skin_mid_surface(
        source, proposal, mask
    )
    report = json.loads(report_json)
    assert report["status"] == "ABSTAIN_INSUFFICIENT_MID_SURFACE_EFFECT"
    assert torch.equal(candidate, source)
    assert int(torch.count_nonzero(effective)) == 0
    assert int(torch.count_nonzero(rejected)) == 0


def test_invalid_inputs_fail_closed():
    source, proposal, mask = _fixture(frames=1)
    with pytest.raises(ValueError, match="exactly match"):
        fuse_learned_skin_mid_surface(source, proposal[:, :-1], mask)
    broken = proposal.clone()
    broken[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN or Inf"):
        fuse_learned_skin_mid_surface(source, broken, mask)
    with pytest.raises(ValueError, match="broad_radius_px"):
        fuse_learned_skin_mid_surface(
            source, proposal, mask, fine_split_radius_px=3, broad_radius_px=4
        )
    with pytest.raises(ValueError, match="chunk_frames"):
        fuse_learned_skin_mid_surface(source, proposal, mask, chunk_frames=0)
