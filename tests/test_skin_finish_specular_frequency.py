from __future__ import annotations

import asyncio
import json

import pytest
import torch

from h3_audio_t8_pkg import comfy_entrypoint
from h3_audio_t8_pkg.nodes_skin_finish_specular_frequency import (
    MiniMaxH3SkinFinishSpecularFrequencyT8Advanced,
)
from h3_audio_t8_pkg.skin_finish_frequency import separate_skin_finish_frequencies
from h3_audio_t8_pkg.skin_finish_specular_frequency import (
    SKIN_FINISH_SPECULAR_FREQUENCY_SCHEMA,
    separate_skin_finish_specular_frequencies,
)


def _fixture():
    source = torch.full((2, 48, 48, 3), 0.72, dtype=torch.float32)
    source[:, 16:32, 16:32] = 0.88
    source[:, 22:26, 16:32] = 0.64  # negative detail must not be directly suppressed
    source[:, 10:14, 18:30] = 0.98  # positive specular detail
    candidate = source.clone()
    candidate[:, 6:38, 6:38] = (candidate[:, 6:38, 6:38] - 0.035).clamp_min(0.0)
    # Simulate a Skin Finish candidate that deliberately softens the compact highlight.
    # The ordinary split tends to restore part of this source-positive detail.
    candidate[:, 10:14, 18:30] = 0.86
    mask = torch.zeros((2, 48, 48), dtype=torch.float32)
    mask[:, 4:40, 4:40] = 1.0
    return source, candidate, mask


def test_zero_specular_strength_is_pixel_exact_ordinary_frequency_split():
    source, candidate, mask = _fixture()
    ordinary = separate_skin_finish_frequencies(
        source, candidate, mask, chunk_frames=1, accept_candidate=False
    )
    specular = separate_skin_finish_specular_frequencies(
        source,
        candidate,
        mask,
        highlight_detail_suppression=0.0,
        chunk_frames=1,
        accept_candidate=False,
    )
    assert torch.equal(specular[0], ordinary[0])
    assert torch.equal(specular[4], ordinary[4])
    assert torch.equal(specular[5], ordinary[5])
    assert torch.equal(specular[6], ordinary[6])
    report = json.loads(specular[7])
    assert report["schema"] == SKIN_FINISH_SPECULAR_FREQUENCY_SCHEMA
    assert report["mechanical_gates"]["zero_strength_exact_base_candidate"] is True


def test_specular_route_reduces_positive_bright_detail_and_preserves_exterior():
    source, candidate, mask = _fixture()
    ordinary = separate_skin_finish_frequencies(
        source, candidate, mask, maximum_mask_area=0.60
    )[0]
    specular, _, selected, _, effective, _, _, report_json = (
        separate_skin_finish_specular_frequencies(
            source,
            candidate,
            mask,
            highlight_detail_suppression=0.75,
            highlight_start=0.55,
            highlight_end=0.95,
            positive_detail_threshold=0.004,
            treatment_intent_scale=0.002,
            maximum_specular_delta=0.05,
            maximum_mask_area=0.60,
            chunk_frames=1,
            accept_candidate=False,
        )
    )
    report = json.loads(report_json)
    assert report["status"] == "PASS"
    assert selected is source
    assert float(specular[:, 10:14, 18:30].mean()) < float(
        ordinary[:, 10:14, 18:30].mean()
    )
    targeted = specular[:, 10:14, 18:30]
    targeted_base = ordinary[:, 10:14, 18:30]
    targeted_candidate = candidate[:, 10:14, 18:30]
    assert bool(
        (
            (targeted >= torch.minimum(targeted_base, targeted_candidate) - 1.0e-7)
            & (targeted <= torch.maximum(targeted_base, targeted_candidate) + 1.0e-7)
        ).all()
    )
    # The deliberately negative stripe is not directly targeted by the positive-detail gate.
    stripe_delta = (specular[:, 22:26, 16:32] - ordinary[:, 22:26, 16:32]).abs()
    assert float(stripe_delta.max()) < 1.0e-6
    outside = effective <= 0.0
    assert torch.equal(specular[..., :3][outside], source[..., :3][outside])
    assert report["mechanical_gates"]["outside_effective_mask_bit_exact"] is True
    assert report["mechanical_gates"]["negative_detail_directly_suppressed"] is False
    assert (
        report["mechanical_gates"][
            "bounded_between_frequency_base_and_input_candidate"
        ]
        is True
    )


def test_specular_route_is_chunk_deterministic_and_rejects_invalid_parameters():
    source, candidate, mask = _fixture()
    one = separate_skin_finish_specular_frequencies(
        source, candidate, mask, chunk_frames=1
    )[0]
    two = separate_skin_finish_specular_frequencies(
        source, candidate, mask, chunk_frames=2
    )[0]
    assert torch.equal(one, two)
    with pytest.raises(ValueError, match="highlight range"):
        separate_skin_finish_specular_frequencies(
            source, candidate, mask, highlight_start=0.9, highlight_end=0.8
        )
    with pytest.raises(ValueError, match="suppression"):
        separate_skin_finish_specular_frequencies(
            source, candidate, mask, highlight_detail_suppression=1.1
        )


def test_specular_node_is_append_only_and_safe_by_default():
    schema = MiniMaxH3SkinFinishSpecularFrequencyT8Advanced.define_schema()
    inputs = {item.id: item for item in schema.inputs}
    assert schema.is_experimental is True
    assert inputs["separation_radius_percent"].default == 3.0
    assert inputs["highlight_detail_suppression"].default == 0.65
    assert inputs["positive_detail_threshold"].default == 0.004
    assert inputs["accept_candidate"].default is False
    ids = [
        node.define_schema().node_id
        for node in asyncio.run(comfy_entrypoint().get_node_list())
    ]
    assert len(ids) == 298
    assert ids[207] == "MiniMaxH3SkinFinishQualityVideoStreamT8Advanced"
    assert ids[208] == "MiniMaxH3SkinFinishSpecularFrequencyT8Advanced"
    assert ids[209] == "MiniMaxH3SkinFinishSurfaceT8Advanced"
    assert ids[210] == "MiniMaxH3SkinFinishDichromaticT8Advanced"
