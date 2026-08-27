from __future__ import annotations

import asyncio

import pytest
import torch

import comfy.nested_tensor

from h3_audio_t8_pkg.nodes import MiniMaxH3AudioT8Extension
from h3_audio_t8_pkg.nodes_av_latent_builder_advanced import (
    MiniMaxH3AVLatentBuilderT8Advanced,
    _video_frame_count,
    build_h3_av_latent,
)


def _latents(
    video_shape=(1, 24, 107, 2, 3),
    audio_shape=(1, 32, 2, 603),
):
    return {"samples": torch.zeros(video_shape)}, {"samples": torch.zeros(audio_shape)}


def test_builder_keeps_stream_objects_and_builds_nested_latent():
    video_latent, audio_latent = _latents()

    result = build_h3_av_latent(video_latent, audio_latent)

    assert isinstance(result["samples"], comfy.nested_tensor.NestedTensor)
    video, audio = result["samples"].unbind()
    assert video is video_latent["samples"]
    assert audio is audio_latent["samples"]
    assert set(result) == {"samples"}


def test_builder_does_not_mutate_input_dictionaries():
    video_latent, audio_latent = _latents()
    video_latent["noise_mask"] = torch.ones(1)
    audio_latent["batch_index"] = [7]
    video_before = dict(video_latent)
    audio_before = dict(audio_latent)

    build_h3_av_latent(video_latent, audio_latent)

    assert video_latent == video_before
    assert audio_latent == audio_before


@pytest.mark.parametrize(
    ("video_shape", "audio_shape", "message"),
    [
        ((1, 24, 107, 2), (1, 32, 2, 603), "video latent must have shape"),
        ((1, 16, 107, 2, 3), (1, 32, 2, 603), "video latent must have shape"),
        ((0, 24, 107, 2, 3), (0, 32, 2, 603), "video latent must have shape"),
        ((1, 24, 107, 2, 3), (1, 32, 603), "audio latent must have shape"),
        ((1, 24, 107, 2, 3), (1, 16, 2, 603), "audio latent must have shape"),
        ((1, 24, 107, 2, 3), (1, 32, 1, 603), "audio latent must have shape"),
    ],
)
def test_builder_rejects_invalid_shapes(video_shape, audio_shape, message):
    video_latent, audio_latent = _latents(video_shape, audio_shape)

    with pytest.raises(ValueError, match=message):
        build_h3_av_latent(video_latent, audio_latent)


def test_builder_rejects_missing_tensor_and_mismatched_batches():
    _, audio_latent = _latents()
    with pytest.raises(ValueError, match="video_latent must contain"):
        build_h3_av_latent({}, audio_latent)

    video_latent, audio_latent = _latents(audio_shape=(2, 32, 2, 603))
    with pytest.raises(ValueError, match="batch sizes must match"):
        build_h3_av_latent(video_latent, audio_latent)


def test_builder_accepts_one_tick_rounding_and_rejects_timeline_mismatch():
    video_latent, audio_latent = _latents(audio_shape=(1, 32, 2, 604))
    assert build_h3_av_latent(video_latent, audio_latent)["samples"].unbind()[1] is audio_latent["samples"]

    video_latent, audio_latent = _latents(audio_shape=(1, 32, 2, 600))
    with pytest.raises(ValueError, match="timelines do not match"):
        build_h3_av_latent(video_latent, audio_latent)


@pytest.mark.parametrize(
    ("tokens", "frames"),
    [(1, 1), (2, 5), (5, 17), (7, 22), (107, 362)],
)
def test_video_token_timeline_matches_h3_cycle(tokens, frames):
    assert _video_frame_count(tokens) == frames


def test_schema_and_append_only_registration():
    schema = MiniMaxH3AVLatentBuilderT8Advanced.define_schema()
    assert schema.node_id == "MiniMaxH3AVLatentBuilderT8Advanced"
    node_list = asyncio.run(MiniMaxH3AudioT8Extension().get_node_list())
    assert MiniMaxH3AVLatentBuilderT8Advanced in node_list
    assert node_list.index(MiniMaxH3AVLatentBuilderT8Advanced) > node_list.index(
        next(
            node
            for node in node_list
            if node.define_schema().node_id
            == "MiniMaxH3LongVideoInNodeLoopEffectsT8Advanced"
        )
    )
