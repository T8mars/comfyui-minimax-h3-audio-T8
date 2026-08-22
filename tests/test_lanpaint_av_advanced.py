from __future__ import annotations

import json

import pytest
import torch

from h3_audio_t8_pkg.lanpaint_av_advanced import (
    audio_interval_mask,
    composite_lanpaint_av,
    parse_audio_intervals,
    prepare_lanpaint_av_latent,
)
from h3_audio_t8_pkg.nodes_lanpaint_av_advanced import (
    MiniMaxH3LanPaintAVCompositeT8Advanced,
    MiniMaxH3LanPaintAVPrepareT8Advanced,
)
from helpers import FakeAudioVAE, FakeVideoVAE, make_audio


def test_audio_intervals_clip_merge_and_cover_all_overlapping_bins():
    intervals = parse_audio_intervals(
        '[{"start":-1,"end":0.2},{"start":0.1,"end":0.4},{"start":0.8,"end":2}]',
        1.0,
    )
    assert intervals == [(0.0, 0.4), (0.8, 1.0)]
    mask = audio_interval_mask(intervals, 10, 10)
    assert mask.tolist() == [1, 1, 1, 1, 0, 0, 0, 0, 1, 1]


def test_prepare_report_can_drive_composite_intervals_without_duplication():
    report = json.dumps({"audio_intervals": [[0.5, 1.0], [0.9, 1.5]]})
    assert parse_audio_intervals(report, 2.0) == [(0.5, 1.5)]


def test_prepare_builds_exact_nested_h3_av_masks_without_dropping_short_video_event():
    frames = torch.zeros((22, 32, 32, 3))
    video_mask = torch.zeros((22, 32, 32))
    video_mask[9, 10:12, 10:12] = 1.0
    source_audio = make_audio(22 / 24, value=0.25, channels=2)
    latent, kept_frames, kept_audio, report = prepare_lanpaint_av_latent(
        frames,
        source_audio,
        FakeVideoVAE(),
        FakeAudioVAE(),
        '[{"start":0.25,"end":0.5}]',
        "strict",
        False,
        video_mask,
    )
    video, audio = latent["samples"].unbind()
    video_noise, audio_noise = latent["noise_mask"].unbind()
    assert video.shape == (1, 24, 7, 2, 2)
    assert audio.shape == (1, 32, 2, 37)
    assert video_noise.shape == video.shape
    assert audio_noise.shape == audio.shape
    assert torch.count_nonzero(video_noise) > 0
    assert torch.count_nonzero(audio_noise) > 0
    assert torch.count_nonzero(audio_noise) < audio_noise.numel()
    assert kept_frames is frames
    assert kept_audio is source_audio
    assert json.loads(report)["mask_semantics"] == "1=regenerate, 0=preserve"


def test_prepare_requires_explicit_grid_policy():
    frames = torch.zeros((23, 32, 32, 3))
    with pytest.raises(ValueError, match=r"17n\+5"):
        prepare_lanpaint_av_latent(
            frames,
            make_audio(1),
            FakeVideoVAE(),
            FakeAudioVAE(),
            "[]",
            "strict",
            False,
        )
    _latent, kept, *_ = prepare_lanpaint_av_latent(
        frames,
        make_audio(1),
        FakeVideoVAE(),
        FakeAudioVAE(),
        "[]",
        "trim_down",
        False,
    )
    assert kept.shape[0] == 22


def test_composite_replaces_only_declared_video_and_audio_regions():
    source_frames = torch.zeros((5, 16, 16, 3))
    repaired_frames = torch.ones_like(source_frames)
    video_mask = torch.zeros((5, 16, 16))
    video_mask[:, 6:10, 6:10] = 1.0
    source_audio = make_audio(1.0, sample_rate=100, value=0.0, channels=2)
    repaired_audio = make_audio(1.0, sample_rate=100, value=1.0, channels=2)
    frames, audio, report = composite_lanpaint_av(
        source_frames,
        repaired_frames,
        source_audio,
        repaired_audio,
        "0.25-0.50",
        1,
        0.0,
        video_mask,
    )
    assert frames[0, 0, 0, 0] == 0
    assert frames[0, 7, 7, 0] == 1
    assert torch.all(audio["waveform"][..., :25] == 0)
    assert torch.all(audio["waveform"][..., 25:50] == 1)
    assert torch.all(audio["waveform"][..., 50:] == 0)
    assert json.loads(report)["source_audio_preserved_outside_intervals"] is True


def test_lanpaint_nodes_are_explicit_external_sampler_bridges():
    prepare_schema = MiniMaxH3LanPaintAVPrepareT8Advanced.define_schema()
    inputs = {item.id: item for item in prepare_schema.inputs}
    assert inputs["require_lanpaint_sampler"].default is True
    assert inputs["frame_policy"].default == "strict"
    assert prepare_schema.is_experimental is True
    composite_schema = MiniMaxH3LanPaintAVCompositeT8Advanced.define_schema()
    assert composite_schema.is_experimental is True
    assert composite_schema.category == "T8/MiniMax H3/Repair/Experimental"
