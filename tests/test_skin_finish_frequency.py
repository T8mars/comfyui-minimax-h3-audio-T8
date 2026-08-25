from __future__ import annotations

import json

import pytest
import torch

from h3_audio_t8_pkg.nodes_skin_finish_frequency import (
    MiniMaxH3SkinFinishFrequencySplitT8Advanced,
)
from h3_audio_t8_pkg.skin_finish_frequency import (
    SKIN_FINISH_FREQUENCY_SPLIT_SCHEMA,
    _two_pass_box_lowpass,
    separate_skin_finish_frequencies,
)


def _source(frame_count: int = 3, channels: int = 3) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.arange(48, dtype=torch.float32),
        torch.arange(64, dtype=torch.float32),
        indexing="ij",
    )
    checker = (((xx.to(torch.int64) + yy.to(torch.int64)) % 2) * 2 - 1).float()
    base = torch.stack(
        [0.46 + checker * 0.035, 0.50 + checker * 0.025, 0.54 + checker * 0.020],
        dim=-1,
    )
    source = base.unsqueeze(0).repeat(frame_count, 1, 1, 1)
    if channels == 3:
        return source
    alpha = torch.linspace(0.0, 1.0, 48 * 64).view(1, 48, 64, 1)
    return torch.cat([source, alpha.expand(frame_count, -1, -1, -1)], dim=-1)


def _candidate(source: torch.Tensor, level: float = 0.62) -> torch.Tensor:
    candidate = source.clone()
    candidate[..., :3] = level
    return candidate


def _mask(frame_count: int = 3) -> torch.Tensor:
    mask = torch.zeros((frame_count, 48, 64))
    mask[:, 10:38, 16:48] = 1.0
    return mask


def _audio() -> dict:
    return {
        "waveform": torch.linspace(-0.2, 0.2, 1024).view(1, 1, -1),
        "sample_rate": 32000,
    }


def test_frequency_split_uses_candidate_low_and_source_high_layers():
    source = _source(frame_count=1)
    candidate = _candidate(source)
    mask = _mask(frame_count=1)
    result, original, selected, _, effective, rejected, difference, report_json = (
        separate_skin_finish_frequencies(
            source,
            candidate,
            mask,
            low_frequency_strength=1.0,
            source_detail_gain=1.0,
            separation_radius_percent=1.0,
            maximum_radius_px=8,
            maximum_new_clipped_fraction=0.25,
            accept_candidate=False,
        )
    )
    report = json.loads(report_json)
    radius = report["parameters"]["actual_radius_px"]
    source_nchw = source[..., :3].movedim(-1, 1)
    candidate_nchw = candidate[..., :3].movedim(-1, 1)
    expected = (
        _two_pass_box_lowpass(candidate_nchw, radius)
        + source_nchw
        - _two_pass_box_lowpass(source_nchw, radius)
    ).movedim(1, -1)
    inside = mask > 0

    assert report["schema"] == SKIN_FINISH_FREQUENCY_SPLIT_SCHEMA
    assert report["method"] == "two_pass_box_lowpass_candidate_low_plus_source_high"
    assert report["status"] == "PASS"
    assert original is source
    assert selected is source
    assert torch.allclose(result[..., :3][inside], expected[inside], atol=1.0e-6)
    assert not torch.allclose(result[..., :3][inside], candidate[..., :3][inside])
    assert torch.equal(effective, mask)
    assert torch.count_nonzero(rejected) == 0
    assert float(difference.sum()) > 0.0


def test_frequency_split_preserves_outside_alpha_audio_and_explicit_selection():
    source = _source(channels=4)
    candidate = _candidate(source)
    mask = _mask()
    audio = _audio()
    result = separate_skin_finish_frequencies(
        source,
        candidate,
        mask,
        maximum_new_clipped_fraction=0.25,
        accept_candidate=True,
        audio=audio,
    )
    output, original, selected, audio_out, effective, _, _, report_json = result
    report = json.loads(report_json)
    outside = effective <= 0.0

    assert original is source
    assert selected is output
    assert audio_out is audio
    assert torch.equal(output[..., :3][outside], source[..., :3][outside])
    assert torch.equal(output[..., 3:], source[..., 3:])
    assert report["mechanical_gates"]["outside_effective_mask_bit_exact"] is True
    assert report["mechanical_gates"]["alpha_or_aux_channels_preserved"] is True
    assert report["mechanical_gates"]["audio_object_passthrough"] is True
    assert report["mechanical_gates"]["automatic_accept"] is False


def test_frequency_split_exact_noop_contracts_and_chunk_parity():
    source = _source()
    mask = _mask()
    same_candidate = source.clone()
    same = separate_skin_finish_frequencies(
        source,
        same_candidate,
        mask,
        maximum_new_clipped_fraction=0.25,
        chunk_frames=1,
    )[0]
    disabled = separate_skin_finish_frequencies(
        source,
        _candidate(source),
        mask,
        low_frequency_strength=0.0,
        source_detail_gain=1.0,
        maximum_new_clipped_fraction=0.25,
        chunk_frames=2,
    )[0]
    chunk_one = separate_skin_finish_frequencies(
        source,
        _candidate(source),
        mask,
        maximum_new_clipped_fraction=0.25,
        chunk_frames=1,
    )[0]
    chunk_three = separate_skin_finish_frequencies(
        source,
        _candidate(source),
        mask,
        maximum_new_clipped_fraction=0.25,
        chunk_frames=3,
    )[0]

    assert torch.equal(same, source)
    assert torch.equal(disabled, source)
    assert torch.equal(chunk_one, chunk_three)


def test_frequency_split_fails_closed_for_bad_mask_area_or_new_clipping():
    source = _source(frame_count=1)
    empty_mask = torch.zeros((1, 48, 64))
    empty_result = separate_skin_finish_frequencies(
        source,
        _candidate(source),
        empty_mask,
        maximum_new_clipped_fraction=0.25,
        accept_candidate=True,
    )
    empty_report = json.loads(empty_result[-1])
    assert empty_report["status"] == "ABSTAIN_ALL_FRAMES_REJECTED"
    assert torch.equal(empty_result[0], source)
    assert empty_result[2] is source
    assert empty_report["frame_reports"][0]["reasons"] == ["mask_area_gate_failed"]

    clipping_candidate = _candidate(source, level=0.99)
    clipping_result = separate_skin_finish_frequencies(
        source,
        clipping_candidate,
        _mask(frame_count=1),
        source_detail_gain=1.25,
        maximum_new_clipped_fraction=0.0,
        accept_candidate=True,
    )
    clipping_report = json.loads(clipping_result[-1])
    assert clipping_report["status"] == "ABSTAIN_ALL_FRAMES_REJECTED"
    assert "new_clipping_limit_failed" in clipping_report["frame_reports"][0]["reasons"]
    assert torch.equal(clipping_result[0], source)


def test_frequency_split_rejects_invalid_contracts():
    source = _source(frame_count=1)
    candidate = _candidate(source)
    mask = _mask(frame_count=1)
    with pytest.raises(ValueError, match="exactly match"):
        separate_skin_finish_frequencies(source, candidate[:, :, :-1], mask)
    with pytest.raises(ValueError, match="low_frequency_strength"):
        separate_skin_finish_frequencies(
            source, candidate, mask, low_frequency_strength=1.01
        )
    with pytest.raises(ValueError, match="NaN or Inf"):
        invalid = candidate.clone()
        invalid[0, 0, 0, 0] = float("nan")
        separate_skin_finish_frequencies(source, invalid, mask)


def test_frequency_split_schema_is_append_only_and_safe_by_default():
    schema = MiniMaxH3SkinFinishFrequencySplitT8Advanced.define_schema()
    inputs = {item.id: item for item in schema.inputs}
    assert schema.node_id == "MiniMaxH3SkinFinishFrequencySplitT8Advanced"
    assert schema.is_experimental is True
    assert inputs["low_frequency_strength"].default == 1.0
    assert inputs["source_detail_gain"].default == 1.0
    assert inputs["separation_radius_percent"].default == 1.0
    assert inputs["maximum_radius_px"].default == 32
    assert inputs["chunk_frames"].default == 4
    assert inputs["accept_candidate"].default is False
