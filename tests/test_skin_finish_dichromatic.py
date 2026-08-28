from __future__ import annotations

import asyncio
import json

import pytest
import torch

from h3_audio_t8_pkg import comfy_entrypoint
from h3_audio_t8_pkg.nodes_skin_finish_dichromatic import (
    MiniMaxH3SkinFinishDichromaticT8Advanced,
)
from h3_audio_t8_pkg.skin_finish_dichromatic import (
    DICHROMATIC_REFERENCE,
    FACIAL_SPECULAR_REFERENCE,
    SKIN_FINISH_DICHROMATIC_SCHEMA,
    _dichromatic_specular_estimate,
    _linear_to_srgb,
    _srgb_to_linear,
    attenuate_skin_specular_dichromatic,
)


def _synthetic_fixture(*, frames: int = 2):
    linear = torch.full((frames, 96, 96, 3), 0.03, dtype=torch.float32)
    diffuse = torch.tensor([0.36, 0.16, 0.09], dtype=torch.float32)
    linear[:, 16:80, 16:80] = diffuse
    linear[:, 36:60, 36:60] = (diffuse + 0.14).clamp_max(1.0)
    source = _linear_to_srgb(linear.movedim(-1, 1)).movedim(1, -1)
    alpha = torch.full((frames, 96, 96, 1), 0.73, dtype=torch.float32)
    source = torch.cat([source, alpha], dim=-1)
    mask = torch.zeros((frames, 96, 96), dtype=torch.float32)
    mask[:, 16:80, 16:80] = 1.0
    audio = {
        "waveform": torch.linspace(-0.4, 0.4, 96).view(1, 1, -1),
        "sample_rate": 32000,
    }
    return source, mask, audio


def test_srgb_linear_round_trip_and_exact_dichromatic_solution():
    values = torch.linspace(0.0, 1.0, 257).view(1, 1, 1, -1)
    assert torch.allclose(
        _linear_to_srgb(_srgb_to_linear(values)),
        values,
        atol=1.0e-6,
        rtol=1.0e-6,
    )
    diffuse = torch.tensor([0.55, 0.30, 0.15]).view(1, 3, 1, 1)
    pixel = 0.42 * diffuse + 0.08
    specular, dilution, direction = _dichromatic_specular_estimate(
        pixel, diffuse
    )
    assert float(specular) == pytest.approx(0.08, abs=1.0e-6)
    assert float(dilution) > 0.0
    assert float(direction) > 0.99


def test_dichromatic_candidate_reduces_neutral_specular_without_touching_exterior():
    source, mask, audio = _synthetic_fixture()
    candidate, returned_source, selected, returned_audio, effective, _, _, report_json = (
        attenuate_skin_specular_dichromatic(
            source,
            mask,
            amount=1.0,
            specular_strength=1.0,
            diffuse_radius_percent=5.0,
            specular_threshold_linear=0.001,
            specular_softness_linear=0.010,
            chroma_dilution_threshold=0.0001,
            chroma_dilution_softness=0.005,
            minimum_diffuse_chroma=0.001,
            diffuse_chroma_softness=0.010,
            minimum_direction_cosine=0.50,
            minimum_texture_ratio=0.65,
            maximum_texture_ratio=2.0,
            chunk_frames=1,
            accept_candidate=False,
            audio=audio,
        )
    )
    report = json.loads(report_json)
    assert report["schema"] == SKIN_FINISH_DICHROMATIC_SCHEMA
    assert report["reference"] == DICHROMATIC_REFERENCE
    assert report["facial_specular_reference"] == FACIAL_SPECULAR_REFERENCE
    assert report["status"] == "PASS"
    assert returned_source is source
    assert selected is source
    assert returned_audio is audio
    assert float(candidate[:, 40:56, 40:56, :3].mean()) < float(
        source[:, 40:56, 40:56, :3].mean()
    ) - 0.01
    outside = effective <= 0.0
    assert torch.equal(candidate[..., :3][outside], source[..., :3][outside])
    assert torch.equal(candidate[..., 3:], source[..., 3:])
    assert report["mechanical_gates"]["automatic_accept"] is False


def test_uniform_and_same_chromaticity_brightness_are_not_specular():
    diffuse = torch.tensor([0.36, 0.16, 0.09], dtype=torch.float32)
    linear = torch.full((1, 96, 96, 3), 0.02, dtype=torch.float32)
    linear[:, 16:80, 16:80] = diffuse
    linear[:, 36:60, 36:60] = diffuse * 1.45
    source = _linear_to_srgb(linear.movedim(-1, 1)).movedim(1, -1)
    mask = torch.zeros((1, 96, 96), dtype=torch.float32)
    mask[:, 16:80, 16:80] = 1.0
    candidate = attenuate_skin_specular_dichromatic(
        source,
        mask,
        amount=1.0,
        specular_strength=1.0,
        diffuse_radius_percent=5.0,
        specular_threshold_linear=0.0,
        specular_softness_linear=0.001,
        chroma_dilution_threshold=0.0,
        chroma_dilution_softness=0.001,
        minimum_diffuse_chroma=0.0,
        diffuse_chroma_softness=0.001,
        minimum_direction_cosine=-0.5,
    )[0]
    assert torch.allclose(candidate, source, atol=2.0e-6, rtol=0.0)


def test_zero_amount_chunk_parity_and_fail_closed_change_gate():
    source, mask, _ = _synthetic_fixture(frames=3)
    zero, _, selected, _, _, _, difference, _ = (
        attenuate_skin_specular_dichromatic(
            source, mask, amount=0.0, accept_candidate=True
        )
    )
    assert torch.equal(zero, source)
    assert selected is zero
    assert int(torch.count_nonzero(difference)) == 0
    one = attenuate_skin_specular_dichromatic(
        source, mask, chunk_frames=1
    )[0]
    three = attenuate_skin_specular_dichromatic(
        source, mask, chunk_frames=3
    )[0]
    assert torch.equal(one, three)
    rejected, _, selected, _, effective, rejected_mask, _, report_json = (
        attenuate_skin_specular_dichromatic(
            source,
            mask,
            amount=1.0,
            specular_strength=1.0,
            maximum_mean_abs_change=0.0,
            accept_candidate=True,
        )
    )
    report = json.loads(report_json)
    assert report["status"] == "ABSTAIN_ALL_FRAMES_REJECTED"
    assert torch.equal(rejected, source)
    assert selected is source
    assert int(torch.count_nonzero(effective)) == 0
    assert int(torch.count_nonzero(rejected_mask)) > 0


def test_dichromatic_rejects_invalid_parameters():
    source, mask, _ = _synthetic_fixture(frames=1)
    with pytest.raises(ValueError, match="diffuse_radius_percent"):
        attenuate_skin_specular_dichromatic(
            source, mask, diffuse_radius_percent=8.1
        )
    with pytest.raises(ValueError, match="specular_softness_linear"):
        attenuate_skin_specular_dichromatic(
            source, mask, specular_softness_linear=0.0
        )
    with pytest.raises(ValueError, match="minimum_direction_cosine"):
        attenuate_skin_specular_dichromatic(
            source, mask, minimum_direction_cosine=1.0
        )
    with pytest.raises(ValueError, match="maximum_texture_ratio"):
        attenuate_skin_specular_dichromatic(
            source, mask, maximum_texture_ratio=2.1
        )
    with pytest.raises(ValueError, match="chunk_frames"):
        attenuate_skin_specular_dichromatic(source, mask, chunk_frames=0)


def test_dichromatic_node_is_append_only_and_safe_by_default():
    schema = MiniMaxH3SkinFinishDichromaticT8Advanced.define_schema()
    inputs = {item.id: item for item in schema.inputs}
    assert schema.is_experimental is True
    assert inputs["amount"].default == 0.80
    assert inputs["specular_strength"].default == 0.80
    assert inputs["minimum_texture_ratio"].default == 0.86
    assert inputs["maximum_texture_ratio"].default == 1.10
    assert inputs["chunk_frames"].default == 2
    assert inputs["accept_candidate"].default is False
    ids = [
        node.define_schema().node_id
        for node in asyncio.run(comfy_entrypoint().get_node_list())
    ]
    assert len(ids) == 244
    assert ids[209] == "MiniMaxH3SkinFinishSurfaceT8Advanced"
    assert ids[210] == "MiniMaxH3SkinFinishDichromaticT8Advanced"
