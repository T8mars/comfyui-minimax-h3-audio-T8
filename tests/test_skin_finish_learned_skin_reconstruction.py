from __future__ import annotations

import json

import pytest
import torch

from h3_audio_t8_pkg.skin_finish_learned_skin_reconstruction import (
    SKIN_FINISH_LEARNED_SKIN_RECONSTRUCTION_SCHEMA,
    fuse_bounded_semantic_skin_reconstruction,
)


def _fixture(*, frames: int = 3):
    y = torch.linspace(0.18, 0.72, 96).view(1, 1, 96, 1)
    x = torch.linspace(0.0, 1.0, 96).view(1, 1, 1, 96)
    fine = torch.sin(x * 68.0) * 0.022
    source_rgb = torch.cat((y * 1.02, y * 0.80, y * 0.64), dim=1) + fine
    proposal_rgb = torch.cat((y * 0.98, y * 0.83, y * 0.70), dim=1) + fine * 0.50
    alpha = torch.full((frames, 1, 96, 96), 0.69)
    source = torch.cat(
        (source_rgb.expand(frames, -1, -1, -1), alpha), dim=1
    ).movedim(1, -1).contiguous().clamp(0.0, 1.0)
    proposal = torch.cat(
        (proposal_rgb.expand(frames, -1, -1, -1), alpha), dim=1
    ).movedim(1, -1).contiguous().clamp(0.0, 1.0)
    mask = torch.zeros((frames, 96, 96))
    mask[:, 15:81, 15:81] = 1.0
    return source, proposal, mask


def test_semantic_skin_reconstruction_is_visible_and_exact_outside():
    source, proposal, mask = _fixture(frames=1)
    candidate, effective, rejected, difference, report_json = (
        fuse_bounded_semantic_skin_reconstruction(
            source,
            proposal,
            mask,
            minimum_masked_mean_abs_change=0.005,
            minimum_structural_gradient_cosine=0.80,
        )
    )
    report = json.loads(report_json)
    assert report["schema"] == SKIN_FINISH_LEARNED_SKIN_RECONSTRUCTION_SCHEMA
    assert report["status"] == "PASS_REQUIRES_IDENTITY_AND_HUMAN_REVIEW"
    assert int(torch.count_nonzero(effective)) > 0
    assert int(torch.count_nonzero(rejected)) == 0
    assert float(difference.mean()) > 1.0e-4
    outside = effective <= 0.0
    assert torch.equal(candidate[..., :3][outside], source[..., :3][outside])
    assert torch.equal(candidate[..., 3:], source[..., 3:])
    assert report["mechanical_gates"]["automatic_accept"] is False


def test_shifted_structural_edge_is_suppressed_near_source_edge():
    source, proposal, mask = _fixture(frames=1)
    source[:, :, 46:50, :3] *= 0.35
    proposal[:, :, 55:59, :3] *= 0.35
    candidate, effective, _, _, _ = fuse_bounded_semantic_skin_reconstruction(
        source,
        proposal,
        mask,
        minimum_masked_mean_abs_change=0.001,
        minimum_structural_gradient_cosine=0.50,
        minimum_texture_ratio=0.20,
        maximum_texture_ratio=2.0,
    )
    edge_change = (candidate[:, :, 44:52, :3] - source[:, :, 44:52, :3]).abs().mean()
    flat_change = (candidate[:, :, 24:36, :3] - source[:, :, 24:36, :3]).abs().mean()
    assert float(edge_change) < float(flat_change)
    assert float(effective[:, :, 46:50].mean()) < float(effective[:, :, 24:36].mean())


def test_equal_proposal_zero_amount_and_chunk_parity_are_exact():
    source, proposal, mask = _fixture()
    equal = fuse_bounded_semantic_skin_reconstruction(source, source, mask)[0]
    zero = fuse_bounded_semantic_skin_reconstruction(
        source, proposal, mask, amount=0.0
    )[0]
    one = fuse_bounded_semantic_skin_reconstruction(
        source,
        proposal,
        mask,
        chunk_frames=1,
        minimum_masked_mean_abs_change=0.005,
        minimum_structural_gradient_cosine=0.80,
    )[0]
    three = fuse_bounded_semantic_skin_reconstruction(
        source,
        proposal,
        mask,
        chunk_frames=3,
        minimum_masked_mean_abs_change=0.005,
        minimum_structural_gradient_cosine=0.80,
    )[0]
    assert torch.equal(equal, source)
    assert torch.equal(zero, source)
    assert torch.equal(one, three)


def test_candidate_delta_is_capped():
    source, proposal, mask = _fixture(frames=1)
    proposal[..., :3] = 1.0 - proposal[..., :3]
    candidate = fuse_bounded_semantic_skin_reconstruction(
        source,
        proposal,
        mask,
        amount=1.0,
        maximum_proposal_component_delta=0.50,
        candidate_rgb_delta_cap=0.08,
        minimum_masked_mean_abs_change=0.001,
        minimum_structural_gradient_cosine=0.50,
        minimum_texture_ratio=0.20,
        maximum_texture_ratio=2.0,
        maximum_new_clipped_fraction=0.05,
    )[0]
    assert float((candidate - source).abs().max()) <= 0.080001


def test_tiny_change_abstains_exactly():
    source, proposal, mask = _fixture(frames=1)
    proposal[..., :3] = (source[..., :3] + 1.0e-5).clamp(0.0, 1.0)
    candidate, effective, rejected, _, report_json = (
        fuse_bounded_semantic_skin_reconstruction(source, proposal, mask)
    )
    report = json.loads(report_json)
    assert report["status"] == "ABSTAIN_INSUFFICIENT_VISIBLE_CHANGE"
    assert torch.equal(candidate, source)
    assert int(torch.count_nonzero(effective)) == 0
    assert int(torch.count_nonzero(rejected)) == 0


def test_invalid_inputs_fail_closed():
    source, proposal, mask = _fixture(frames=1)
    with pytest.raises(ValueError, match="exactly match"):
        fuse_bounded_semantic_skin_reconstruction(source, proposal[:, :-1], mask)
    broken = proposal.clone()
    broken[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN or Inf"):
        fuse_bounded_semantic_skin_reconstruction(source, broken, mask)
    with pytest.raises(ValueError, match="proposal_prefilter_radius_px"):
        fuse_bounded_semantic_skin_reconstruction(
            source, proposal, mask, proposal_prefilter_radius_px=0
        )
    with pytest.raises(ValueError, match="chunk_frames"):
        fuse_bounded_semantic_skin_reconstruction(
            source, proposal, mask, chunk_frames=0
        )
