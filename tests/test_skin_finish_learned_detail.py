from __future__ import annotations

import json

import pytest
import torch

from h3_audio_t8_pkg.skin_finish_dichromatic import _srgb_to_linear
from h3_audio_t8_pkg.skin_finish_learned_detail import (
    SKIN_FINISH_LEARNED_DETAIL_SCHEMA,
    fuse_proposal_guided_skin_detail,
)


def _fixture(*, frames: int = 3):
    y = torch.linspace(0.12, 0.72, 96).view(1, 1, 96, 1)
    x = torch.linspace(0.0, 1.0, 96).view(1, 1, 1, 96)
    texture = torch.sin(x * 62.0) * 0.018
    rgb = torch.cat((y * 1.05, y * 0.78, y * 0.60), dim=1)
    source = (rgb + texture).clamp(0.0, 1.0).expand(frames, -1, -1, -1)
    proposal = (rgb + texture * 1.8).clamp(0.0, 1.0).expand(frames, -1, -1, -1)
    alpha = torch.full((frames, 1, 96, 96), 0.77)
    source = torch.cat((source, alpha), dim=1).movedim(1, -1).contiguous()
    proposal = torch.cat((proposal, alpha), dim=1).movedim(1, -1).contiguous()
    mask = torch.zeros((frames, 96, 96))
    mask[:, 16:80, 16:80] = 1.0
    return source, proposal, mask


def test_learned_detail_uses_source_phase_and_preserves_exterior_and_aux():
    source, proposal, mask = _fixture()
    candidate, effective, rejected, difference, report_json = (
        fuse_proposal_guided_skin_detail(source, proposal, mask, chunk_frames=1)
    )
    report = json.loads(report_json)
    assert report["schema"] == SKIN_FINISH_LEARNED_DETAIL_SCHEMA
    assert report["status"] == "PASS"
    assert int(torch.count_nonzero(effective)) > 0
    assert int(torch.count_nonzero(rejected)) == 0
    assert int(torch.count_nonzero(difference)) > 0
    outside = effective <= 0.0
    assert torch.equal(candidate[..., :3][outside], source[..., :3][outside])
    assert torch.equal(candidate[..., 3:], source[..., 3:])
    assert report["mechanical_gates"]["proposal_rgb_or_detail_phase_pasted"] is False


def test_linear_rgb_chromaticity_is_source_derived_for_changed_pixels():
    source, proposal, mask = _fixture(frames=1)
    candidate, effective, _, _, _ = fuse_proposal_guided_skin_detail(
        source, proposal, mask, chroma_amount=0.0
    )
    changed = effective[0] > 0.01
    source_linear = _srgb_to_linear(source[..., :3].movedim(-1, 1)).movedim(1, -1)
    candidate_linear = _srgb_to_linear(candidate[..., :3].movedim(-1, 1)).movedim(1, -1)
    source_chroma = source_linear / source_linear.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
    candidate_chroma = candidate_linear / candidate_linear.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
    assert torch.allclose(
        source_chroma[0][changed],
        candidate_chroma[0][changed],
        atol=3.0e-6,
        rtol=0.0,
    )


def test_zero_amount_equal_proposal_and_chunk_parity_are_exact():
    source, proposal, mask = _fixture()
    zero = fuse_proposal_guided_skin_detail(
        source,
        proposal,
        mask,
        amount=0.0,
        surface_amount=0.0,
        chroma_amount=0.0,
        chunk_frames=1,
    )[0]
    equal = fuse_proposal_guided_skin_detail(
        source, source, mask, chunk_frames=1
    )[0]
    one = fuse_proposal_guided_skin_detail(
        source, proposal, mask, chunk_frames=1
    )[0]
    three = fuse_proposal_guided_skin_detail(
        source, proposal, mask, chunk_frames=3
    )[0]
    assert torch.equal(zero, source)
    assert torch.equal(equal, source)
    assert torch.equal(one, three)


def test_low_frequency_geometry_mismatch_suppresses_the_proposal():
    source, proposal, mask = _fixture(frames=1)
    mismatched = proposal.clone()
    mismatched[:, 24:72, 24:72, :3] = 0.95
    _, effective, _, difference, report_json = fuse_proposal_guided_skin_detail(
        source,
        mismatched,
        mask,
        low_frequency_tolerance=0.005,
    )
    report = json.loads(report_json)
    assert float(effective[:, 32:64, 32:64].mean()) < 0.05
    assert float(difference[:, 32:64, 32:64].mean()) < 1.0e-3
    assert report["mechanical_gates"]["source_high_frequency_phase_preserved"] is True


def test_bounded_zero_mean_surface_trend_is_visible_without_chroma_transfer():
    source, proposal, mask = _fixture(frames=1)
    surface = proposal.clone()
    surface[:, 20:48, 20:76, :3] = (
        surface[:, 20:48, 20:76, :3] * 0.86
    )
    candidate, effective, _, difference, report_json = (
        fuse_proposal_guided_skin_detail(
            source,
            surface,
            mask,
            amount=0.0,
            surface_amount=0.45,
        )
    )
    report = json.loads(report_json)
    assert report["status"] == "PASS"
    assert int(torch.count_nonzero(effective)) > 0
    assert float(difference.mean()) > 1.0e-4
    assert float((candidate - source).abs().max()) <= 0.12
    assert report["mechanical_gates"]["proposal_low_frequency_luma_centered_and_bounded"] is True


def test_low_frequency_chroma_transfer_is_bounded():
    source, proposal, mask = _fixture(frames=1)
    tinted = proposal.clone()
    tinted[..., 0] = (tinted[..., 0] * 1.20).clamp(0.0, 1.0)
    _, _, _, _, report_json = fuse_proposal_guided_skin_detail(
        source,
        tinted,
        mask,
        amount=0.0,
        surface_amount=0.0,
        chroma_amount=0.20,
        maximum_chroma_component_delta=0.04,
    )
    report = json.loads(report_json)
    frame = report["frame_reports"][0]
    assert frame["maximum_abs_applied_chroma_component_delta"] <= 0.0080001
    assert frame["peak_abs_change"] <= 0.100001
    assert report["mechanical_gates"]["proposal_low_frequency_chroma_centered_and_bounded"] is True


def test_invalid_proposal_and_parameters_fail_closed():
    source, proposal, mask = _fixture(frames=1)
    with pytest.raises(ValueError, match="exactly match"):
        fuse_proposal_guided_skin_detail(source, proposal[:, :-1], mask)
    broken = proposal.clone()
    broken[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN or Inf"):
        fuse_proposal_guided_skin_detail(source, broken, mask)
    with pytest.raises(ValueError, match="maximum_detail_gain"):
        fuse_proposal_guided_skin_detail(
            source, proposal, mask, maximum_detail_gain=3.1
        )
    with pytest.raises(ValueError, match="chunk_frames"):
        fuse_proposal_guided_skin_detail(source, proposal, mask, chunk_frames=0)
