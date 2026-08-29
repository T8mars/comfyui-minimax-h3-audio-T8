from __future__ import annotations

import asyncio
import json

import pytest
import torch

from h3_audio_t8_pkg import comfy_entrypoint
from h3_audio_t8_pkg.nodes_skin_finish_surface import (
    MiniMaxH3SkinFinishSurfaceT8Advanced,
)
from h3_audio_t8_pkg.skin_finish_surface import (
    FACIAL_SPECULAR_REFERENCE,
    GUIDED_FILTER_REFERENCE,
    SKIN_FINISH_SURFACE_SCHEMA,
    TONE_MAPPING_REFERENCE,
    _box_mean,
    finish_skin_surface,
)


def _fixture():
    source = torch.full((2, 64, 64, 4), 0.56, dtype=torch.float32)
    source[..., 3] = 0.73
    source[:, 18:28, 18:28, :3] = 0.88
    source[:, 36:46, 36:46, :3] = 0.26
    source[:, 12:52, 31:33, :3] = 0.72
    mask = torch.zeros((2, 64, 64), dtype=torch.float32)
    mask[:, 10:54, 10:54] = 1.0
    audio = {
        "waveform": torch.linspace(-0.5, 0.5, 64).view(1, 1, -1),
        "sample_rate": 24000,
    }
    return source, mask, audio


def test_separable_box_mean_matches_direct_two_dimensional_reference():
    generator = torch.Generator().manual_seed(731)
    value = torch.rand((2, 3, 31, 37), generator=generator)
    for radius in (1, 3, 7):
        actual = _box_mean(value, radius)
        kernel = radius * 2 + 1
        expected = torch.nn.functional.avg_pool2d(
            torch.nn.functional.pad(
                value,
                (radius, radius, radius, radius),
                mode="replicate",
            ),
            kernel_size=kernel,
            stride=1,
        )
        assert torch.allclose(actual, expected, atol=2.0e-7, rtol=2.0e-6)


def test_surface_zero_amount_is_exact_and_source_selected():
    source, mask, audio = _fixture()
    candidate, returned_source, selected, returned_audio, _, _, difference, report_json = (
        finish_skin_surface(
            source,
            mask,
            amount=0.0,
            chunk_frames=1,
            accept_candidate=False,
            audio=audio,
        )
    )
    assert torch.equal(candidate, source)
    assert returned_source is source
    assert selected is source
    assert returned_audio is audio
    assert int(torch.count_nonzero(difference)) == 0
    report = json.loads(report_json)
    assert report["schema"] == SKIN_FINISH_SURFACE_SCHEMA
    assert report["reference"] == GUIDED_FILTER_REFERENCE
    assert report["tone_mapping_reference"] == TONE_MAPPING_REFERENCE
    assert report["facial_specular_reference"] == FACIAL_SPECULAR_REFERENCE
    assert report["mechanical_gates"]["automatic_accept"] is False


def test_surface_balances_compact_highlight_and_blemish_without_touching_exterior():
    source, mask, audio = _fixture()
    candidate, _, selected, returned_audio, effective, _, _, report_json = (
        finish_skin_surface(
            source,
            mask,
            amount=0.85,
            surface_smoothing=0.80,
            texture_keep=0.78,
            highlight_compression=0.85,
            blemish_balance=0.65,
            surface_radius_percent=3.0,
            minimum_texture_ratio=0.70,
            chunk_frames=1,
            accept_candidate=False,
            audio=audio,
        )
    )
    report = json.loads(report_json)
    assert report["status"] == "PASS"
    assert selected is source
    assert returned_audio is audio
    assert float(candidate[:, 18:28, 18:28, :3].mean()) < float(
        source[:, 18:28, 18:28, :3].mean()
    )
    assert float(candidate[:, 36:46, 36:46, :3].mean()) > float(
        source[:, 36:46, 36:46, :3].mean()
    )
    outside = effective <= 0.0
    assert torch.equal(candidate[..., :3][outside], source[..., :3][outside])
    assert torch.equal(candidate[..., 3:], source[..., 3:])
    assert report["mechanical_gates"]["outside_effective_mask_bit_exact"] is True
    assert report["mechanical_gates"]["alpha_or_aux_channels_preserved"] is True


def test_surface_is_chunk_deterministic_and_fail_closed_on_change_gate():
    source, mask, _ = _fixture()
    one = finish_skin_surface(source, mask, chunk_frames=1)[0]
    two = finish_skin_surface(source, mask, chunk_frames=2)[0]
    assert torch.equal(one, two)
    rejected, _, selected, _, effective, rejected_mask, _, report_json = (
        finish_skin_surface(
            source,
            mask,
            amount=1.0,
            maximum_mean_abs_change=0.0,
            chunk_frames=1,
            accept_candidate=True,
        )
    )
    report = json.loads(report_json)
    assert report["status"] == "ABSTAIN_ALL_FRAMES_REJECTED"
    assert torch.equal(rejected, source)
    assert selected is source
    assert int(torch.count_nonzero(effective)) == 0
    assert int(torch.count_nonzero(rejected_mask)) > 0


def test_surface_broad_highlight_shoulder_uses_local_skin_contrast_not_global_tone():
    source = torch.full((1, 96, 96, 3), 0.48, dtype=torch.float32)
    source[:, 16:80, 16:80] = torch.tensor([0.76, 0.70, 0.66])
    source[:, 32:64, 32:64] = torch.tensor([0.90, 0.84, 0.80])
    mask = torch.zeros((1, 96, 96), dtype=torch.float32)
    mask[:, 16:80, 16:80] = 1.0
    without_shoulder = finish_skin_surface(
        source,
        mask,
        amount=0.8,
        surface_smoothing=0.0,
        highlight_compression=0.0,
        broad_highlight_compression=0.0,
        blemish_balance=0.0,
        surface_radius_percent=5.0,
    )[0]
    with_shoulder = finish_skin_surface(
        source,
        mask,
        amount=0.8,
        surface_smoothing=0.0,
        highlight_compression=0.0,
        broad_highlight_compression=0.75,
        broad_highlight_start=0.65,
        broad_highlight_end=0.92,
        blemish_balance=0.0,
        surface_radius_percent=5.0,
    )[0]
    assert torch.equal(without_shoulder, source)
    assert float(with_shoulder[:, 40:56, 40:56].mean()) < float(
        source[:, 40:56, 40:56].mean()
    ) - 0.01
    assert torch.equal(with_shoulder[:, 16, 16:80], source[:, 16, 16:80])
    assert torch.equal(with_shoulder[:, 16:80, 16], source[:, 16:80, 16])
    outside = mask <= 0.0
    assert torch.equal(with_shoulder[outside], source[outside])


def test_surface_uniform_bright_skin_is_not_mistaken_for_broad_highlight():
    source = torch.full((1, 96, 96, 3), 0.25, dtype=torch.float32)
    source[:, 16:80, 16:80] = torch.tensor([0.86, 0.80, 0.75])
    mask = torch.zeros((1, 96, 96), dtype=torch.float32)
    mask[:, 16:80, 16:80] = 1.0
    candidate = finish_skin_surface(
        source,
        mask,
        amount=0.8,
        surface_smoothing=0.0,
        highlight_compression=0.0,
        broad_highlight_compression=0.75,
        broad_highlight_start=0.65,
        broad_highlight_end=0.92,
        blemish_balance=0.0,
        surface_radius_percent=5.0,
    )[0]
    assert torch.equal(candidate, source)


def test_surface_boundary_gate_accepts_probability_valued_semantic_masks():
    source = torch.full((1, 96, 96, 3), 0.48, dtype=torch.float32)
    source[:, 16:80, 16:80] = torch.tensor([0.76, 0.70, 0.66])
    source[:, 32:64, 32:64] = torch.tensor([0.90, 0.84, 0.80])
    mask = torch.zeros((1, 96, 96), dtype=torch.float32)
    mask[:, 16:80, 16:80] = 0.35
    candidate, _, _, _, effective, _, _, report_json = finish_skin_surface(
        source,
        mask,
        amount=0.8,
        surface_smoothing=0.0,
        highlight_compression=0.0,
        broad_highlight_compression=0.75,
        broad_highlight_start=0.65,
        broad_highlight_end=0.92,
        blemish_balance=0.0,
        surface_radius_percent=5.0,
    )
    report = json.loads(report_json)
    assert report["status"] == "PASS"
    assert float(effective.max()) == pytest.approx(0.35)
    assert float(candidate[:, 40:56, 40:56].mean()) < float(
        source[:, 40:56, 40:56].mean()
    )


def test_surface_rejects_invalid_parameters():
    source, mask, _ = _fixture()
    with pytest.raises(ValueError, match="edge_epsilon"):
        finish_skin_surface(source, mask, edge_epsilon=0.0)
    with pytest.raises(ValueError, match="surface_radius_percent"):
        finish_skin_surface(source, mask, surface_radius_percent=5.1)
    with pytest.raises(ValueError, match="broad highlight limits"):
        finish_skin_surface(
            source,
            mask,
            broad_highlight_start=0.95,
            broad_highlight_end=0.90,
        )
    with pytest.raises(ValueError, match="chunk_frames"):
        finish_skin_surface(source, mask, chunk_frames=0)


def test_surface_node_is_append_only_and_safe_by_default():
    schema = MiniMaxH3SkinFinishSurfaceT8Advanced.define_schema()
    inputs = {item.id: item for item in schema.inputs}
    assert schema.is_experimental is True
    assert inputs["amount"].default == 0.65
    assert inputs["texture_keep"].default == 0.85
    assert inputs["broad_highlight_compression"].default == 0.45
    assert inputs["broad_highlight_start"].default == 0.68
    assert inputs["broad_highlight_end"].default == 0.94
    assert inputs["minimum_texture_ratio"].default == 0.82
    assert inputs["chunk_frames"].default == 2
    assert inputs["accept_candidate"].default is False
    ids = [
        node.define_schema().node_id
        for node in asyncio.run(comfy_entrypoint().get_node_list())
    ]
    assert len(ids) == 259
    assert ids[208] == "MiniMaxH3SkinFinishSpecularFrequencyT8Advanced"
    assert ids[209] == "MiniMaxH3SkinFinishSurfaceT8Advanced"
    assert ids[210] == "MiniMaxH3SkinFinishDichromaticT8Advanced"
