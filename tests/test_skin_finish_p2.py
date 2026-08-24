from __future__ import annotations

import json

import torch
import torch.nn.functional as torch_functional

from h3_audio_t8_pkg.nodes_skin_finish_p2 import (
    MiniMaxH3SkinFinishTextureGuardT8Advanced,
)
from h3_audio_t8_pkg.skin_finish_p2 import (
    SKIN_FINISH_TEXTURE_GUARD_SCHEMA,
    guard_skin_finish_candidate,
)


def _source(frame_count: int = 3, channels: int = 3) -> torch.Tensor:
    generator = torch.Generator().manual_seed(260824)
    source = 0.18 + torch.rand((frame_count, 48, 64, 3), generator=generator) * 0.64
    source[:, 8:16, 8:20] = 0.02
    source[:, 8:16, 44:56] = 0.98
    if channels == 3:
        return source
    alpha = torch.linspace(0.0, 1.0, 48 * 64).view(1, 48, 64, 1)
    return torch.cat([source, alpha.expand(frame_count, -1, -1, -1)], dim=-1)


def _mask(frame_count: int = 3) -> torch.Tensor:
    mask = torch.zeros((frame_count, 48, 64))
    mask[:, 4:44, 4:60] = 1.0
    return mask


def _audio() -> dict:
    return {
        "waveform": torch.linspace(-0.25, 0.25, 1600).view(1, 1, -1),
        "sample_rate": 32000,
    }


def test_exposure_extremes_are_preserved_and_audio_is_same_object():
    source = _source(channels=4)
    candidate = source.clone()
    candidate[..., :3] = (candidate[..., :3] + 0.025).clamp(0.0, 1.0)
    audio = _audio()
    guarded, original, selected, audio_out, effective, rejected, _, report = (
        guard_skin_finish_candidate(
            source,
            candidate,
            _mask(),
            minimum_texture_ratio=0.0,
            maximum_new_clipped_fraction=0.25,
            accept_candidate=False,
            audio=audio,
        )
    )
    parsed = json.loads(report)
    assert parsed["schema"] == SKIN_FINISH_TEXTURE_GUARD_SCHEMA
    assert original is source
    assert selected is source
    assert audio_out is audio
    assert torch.equal(guarded[..., 3:], source[..., 3:])
    assert torch.equal(guarded[:, 8:16, 8:20], source[:, 8:16, 8:20])
    assert torch.equal(guarded[:, 8:16, 44:56], source[:, 8:16, 44:56])
    assert torch.count_nonzero(effective[:, 8:16, 8:20]) == 0
    assert torch.count_nonzero(effective[:, 8:16, 44:56]) == 0
    assert torch.count_nonzero(rejected) > 0
    assert parsed["mechanical_gates"]["automatic_accept"] is False


def test_new_clipping_rejects_each_affected_frame_to_source():
    source = _source()
    candidate = source.clone()
    candidate[:, 18:34, 18:46, :3] = 1.0
    guarded, _, _, _, effective, rejected, difference, report = (
        guard_skin_finish_candidate(
            source,
            candidate,
            _mask(),
            minimum_texture_ratio=0.0,
            maximum_new_clipped_fraction=0.0,
            accept_candidate=True,
        )
    )
    parsed = json.loads(report)
    assert parsed["status"] == "ABSTAIN_ALL_FRAMES_REJECTED"
    assert parsed["rejected_frame_count"] == source.shape[0]
    assert all(
        "new_clipping_limit_failed" in item["reasons"]
        for item in parsed["frame_reports"]
    )
    assert torch.equal(guarded, source)
    assert torch.count_nonzero(effective) == 0
    assert torch.equal(rejected, _mask())
    assert torch.count_nonzero(difference) == 0


def test_texture_floor_rejects_a_locally_flattened_candidate():
    source = _source()
    rgb = source[..., :3].movedim(-1, 1)
    flattened = torch_functional.avg_pool2d(rgb, 9, stride=1, padding=4).movedim(1, -1)
    candidate = source.clone()
    candidate[..., :3] = flattened
    guarded, _, _, _, effective, _, _, report = guard_skin_finish_candidate(
        source,
        candidate,
        _mask(),
        shadow_protection=0.0,
        highlight_protection=1.0,
        transition_width=0.01,
        minimum_texture_ratio=0.95,
        maximum_new_clipped_fraction=0.25,
        accept_candidate=True,
    )
    parsed = json.loads(report)
    assert parsed["rejected_frame_count"] == source.shape[0]
    assert all(
        "texture_floor_failed" in item["reasons"]
        for item in parsed["frame_reports"]
    )
    assert torch.equal(guarded, source)
    assert torch.count_nonzero(effective) == 0


def test_benign_low_frequency_candidate_passes_but_selection_is_explicit():
    source = _source()
    candidate = source.clone()
    candidate[:, 16:36, 16:48, :3] = (
        candidate[:, 16:36, 16:48, :3] * 0.99 + 0.004
    )
    guarded, _, selected, _, effective, _, difference, report = (
        guard_skin_finish_candidate(
            source,
            candidate,
            _mask(),
            minimum_texture_ratio=0.70,
            maximum_new_clipped_fraction=0.001,
            accept_candidate=True,
        )
    )
    parsed = json.loads(report)
    assert parsed["status"] == "PASS"
    assert parsed["accepted_frame_count"] == source.shape[0]
    assert selected is guarded
    assert torch.count_nonzero(effective) > 0
    assert float(difference.sum()) > 0.0
    outside = effective <= 0.0
    assert torch.equal(guarded[..., :3][outside], source[..., :3][outside])


def test_texture_guard_schema_is_append_only_and_safe_by_default():
    schema = MiniMaxH3SkinFinishTextureGuardT8Advanced.define_schema()
    inputs = {item.id: item for item in schema.inputs}
    assert schema.node_id == "MiniMaxH3SkinFinishTextureGuardT8Advanced"
    assert schema.is_experimental is True
    assert inputs["accept_candidate"].default is False
    assert inputs["chunk_frames"].default == 4
    assert inputs["minimum_texture_ratio"].default == 0.78
