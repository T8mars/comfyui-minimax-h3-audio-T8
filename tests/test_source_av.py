from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import comfy.nested_tensor

from h3_audio_t8_pkg.source_av import (
    frame_count_from_video_latent_t,
    prepare_source_media_window,
    prepare_source_av_latent,
    separate_source_av_latent,
)


def make_video(value=1.0, latent_t=37, dtype=torch.float32):
    return torch.full((1, 24, latent_t, 4, 6), value, dtype=dtype)


def make_audio(value=2.0, latent_t=207, dtype=torch.float32):
    return torch.full((1, 32, 2, latent_t), value, dtype=dtype)


def prepare(video_latent, audio_latent, **overrides):
    values = {
        "video_mode": "remix",
        "video_denoise_strength": 0.5,
        "audio_mode": "lock",
        "audio_denoise_strength": 0.35,
        "audio_fit_policy": "strict",
        "dtype_device_policy": "match_video",
    }
    values.update(overrides)
    return prepare_source_av_latent(video_latent, audio_latent, **values)


def test_video_latent_clock_inverse_is_strict():
    assert frame_count_from_video_latent_t(2) == 5
    assert frame_count_from_video_latent_t(7) == 22
    assert frame_count_from_video_latent_t(37) == 124
    with pytest.raises(ValueError, match=r"5n\+2"):
        frame_count_from_video_latent_t(36)


def test_source_media_window_resamples_frames_and_makes_reported_silence():
    frames = (
        torch.arange(12, dtype=torch.float32).div(11).reshape(12, 1, 1, 1).expand(-1, 8, 8, 3)
    )
    selected, audio, frame_count, duration, report = prepare_source_media_window(
        frames,
        source_fps=48.0,
        width=64,
        height=64,
        length=5,
        start_seconds=0.0,
        short_video_policy="strict",
        short_audio_policy="pad_silence",
        source_audio=None,
    )
    data = json.loads(report)

    assert selected.shape == (5, 64, 64, 3)
    assert torch.allclose(
        selected[:, 0, 0, 0],
        torch.tensor([0.0, 2 / 11, 4 / 11, 6 / 11, 8 / 11]),
        atol=0.003,
    )
    assert frame_count == 5
    assert duration == pytest.approx(5 / 24)
    assert audio["sample_rate"] == 32000
    assert audio["waveform"].shape == (1, 2, round(5 / 24 * 32000))
    assert torch.count_nonzero(audio["waveform"]) == 0
    assert data["facts"]["audio_source"] == "generated_silence"
    assert data["facts"]["first_source_frame_index"] == 0
    assert data["facts"]["last_source_frame_index"] == 8
    assert data["claims"]["streaming_decode"] is False


def test_source_media_window_short_policies_are_explicit():
    frames = torch.rand((3, 32, 32, 3))
    with pytest.raises(ValueError, match="too short"):
        prepare_source_media_window(
            frames,
            24.0,
            32,
            32,
            5,
            0.0,
            "strict",
            "pad_silence",
        )

    selected, _audio, _count, _duration, report = prepare_source_media_window(
        frames,
        24.0,
        32,
        32,
        5,
        0.0,
        "hold_last_frame",
        "pad_silence",
    )
    assert selected.shape[0] == 5
    assert torch.equal(selected[-1], selected[-2])
    assert json.loads(report)["facts"]["held_video_frames"] == 2


def test_source_media_window_audio_is_resampled_and_exactly_timed():
    frames = torch.rand((60, 32, 32, 3))
    source_audio = {
        "waveform": torch.ones((1, 1, 32000)),
        "sample_rate": 16000,
    }
    _frames, audio, frame_count, _duration, report = prepare_source_media_window(
        frames,
        24.0,
        32,
        32,
        22,
        0.5,
        "strict",
        "strict",
        source_audio,
    )
    expected_samples = round(frame_count / 24 * 32000)
    assert frame_count == 22
    assert audio["waveform"].shape == (1, 2, expected_samples)
    assert audio["sample_rate"] == 32000
    assert json.loads(report)["facts"]["input_audio_sample_rate"] == 16000


def test_prepare_exact_pair_preserves_storage_metadata_and_builds_masks():
    video = make_video()
    audio = make_audio()
    video_latent = {"samples": video, "source_hash": "video", "shared": "video_wins"}
    audio_latent = {"samples": audio, "audio_hash": "audio", "shared": "audio_loses"}

    av, video_out, audio_out, report = prepare(video_latent, audio_latent)
    out_video, out_audio = av["samples"].unbind()
    video_mask, audio_mask = av["noise_mask"].unbind()
    data = json.loads(report)

    assert out_video.data_ptr() == video.data_ptr()
    assert out_audio.data_ptr() == audio.data_ptr()
    assert torch.all(video_mask == 0.5)
    assert torch.all(audio_mask == 0.0)
    assert av["source_hash"] == "video"
    assert av["audio_hash"] == "audio"
    assert av["shared"] == "video_wins"
    assert video_out["samples"].data_ptr() == video.data_ptr()
    assert audio_out["samples"].data_ptr() == audio.data_ptr()
    assert data["facts"]["frame_count"] == 124
    assert data["facts"]["expected_audio_t"] == 207
    assert data["facts"]["audio_fit_action"] == "exact"
    assert data["facts"]["metadata_conflicts_kept_from_video"] == ["shared"]
    assert data["claims"]["memory_safe"] is False
    assert data["claims"]["denoise_strength_is_calibrated_linear_weight"] is False
    assert set(video_latent) == {"samples", "source_hash", "shared"}
    assert set(audio_latent) == {"samples", "audio_hash", "shared"}


def test_source_av_above_reference_area_is_warning_only():
    video = torch.zeros((1, 24, 2, 72, 130), dtype=torch.float32)
    audio = make_audio(latent_t=8)
    av, _video_out, _audio_out, report = prepare(
        {"samples": video},
        {"samples": audio},
    )
    out_video, _out_audio = av["samples"].unbind()
    data = json.loads(report)

    assert out_video.shape[-2:] == (72, 130)
    assert data["facts"]["canvas"] == [2080, 1152]
    assert any("execution remains allowed" in warning for warning in data["warnings"])


def test_prepare_respects_existing_masks_before_stream_strength():
    video = make_video()
    audio = make_audio()
    video_mask = torch.full((1, 1, 37, 4, 6), 0.4)
    audio_mask = torch.full((1, 1, 2, 207), 0.25)
    av, _, _, _ = prepare(
        {"samples": video, "noise_mask": video_mask},
        {"samples": audio, "noise_mask": audio_mask},
        video_denoise_strength=0.5,
        audio_mode="remix",
        audio_denoise_strength=0.4,
    )
    out_video_mask, out_audio_mask = av["noise_mask"].unbind()
    assert out_video_mask.shape == video.shape
    assert out_audio_mask.shape == audio.shape
    assert torch.allclose(out_video_mask, torch.full_like(video, 0.2))
    assert torch.allclose(out_audio_mask, torch.full_like(audio, 0.1))


def test_short_locked_audio_pads_only_tail_as_generatable():
    audio = make_audio(latent_t=200)
    av, _, audio_out, report = prepare(
        {"samples": make_video()},
        {"samples": audio},
        audio_fit_policy="pad_to_video_generate_tail",
    )
    _video, out_audio = av["samples"].unbind()
    _video_mask, audio_mask = av["noise_mask"].unbind()
    data = json.loads(report)

    assert out_audio.shape[-1] == 207
    assert torch.all(out_audio[..., :200] == 2.0)
    assert torch.all(out_audio[..., 200:] == 0.0)
    assert torch.all(audio_mask[..., :200] == 0.0)
    assert torch.all(audio_mask[..., 200:] == 1.0)
    assert torch.equal(audio_out["samples"], out_audio)
    assert data["facts"]["audio_fit_action"] == "padded_7_latent_steps_generate_tail"


def test_long_audio_requires_explicit_trim_policy():
    video_latent = {"samples": make_video()}
    audio_latent = {"samples": make_audio(latent_t=210)}
    with pytest.raises(ValueError, match="longer"):
        prepare(video_latent, audio_latent)

    av, _, _, report = prepare(
        video_latent,
        audio_latent,
        audio_fit_policy="trim_to_video",
    )
    assert av["samples"].unbind()[1].shape[-1] == 207
    assert json.loads(report)["facts"]["audio_fit_action"] == "trimmed_3_latent_steps"


def test_prepare_can_replace_audio_of_existing_av_without_changing_video():
    video = make_video(value=3.0)
    old_audio = make_audio(value=4.0)
    old_video_mask = torch.full_like(video, 0.8)
    existing = {
        "samples": comfy.nested_tensor.NestedTensor((video, old_audio)),
        "noise_mask": comfy.nested_tensor.NestedTensor(
            (old_video_mask, torch.ones_like(old_audio))
        ),
        "chain_id": "accepted-1",
    }
    replacement = make_audio(value=9.0)

    av, _, _, report = prepare(
        existing,
        {"samples": replacement},
        video_denoise_strength=0.5,
    )
    out_video, out_audio = av["samples"].unbind()
    video_mask, _audio_mask = av["noise_mask"].unbind()
    data = json.loads(report)

    assert out_video.data_ptr() == video.data_ptr()
    assert out_audio.data_ptr() == replacement.data_ptr()
    assert torch.all(video_mask == 0.4)
    assert av["chain_id"] == "accepted-1"
    assert data["facts"]["input_video_was_av"] is True


def test_match_video_policy_only_converts_audio_stream_dtype():
    video = make_video(dtype=torch.float32)
    audio = make_audio(dtype=torch.float16)
    av, _, _, report = prepare({"samples": video}, {"samples": audio})
    out_video, out_audio = av["samples"].unbind()

    assert out_video.data_ptr() == video.data_ptr()
    assert out_audio.dtype == torch.float32
    assert json.loads(report)["facts"]["audio_converted_to_video"] is True

    with pytest.raises(ValueError, match="strict mode"):
        prepare(
            {"samples": video},
            {"samples": audio},
            dtype_device_policy="strict",
        )


@pytest.mark.parametrize(
    ("video", "audio", "message"),
    [
        (torch.zeros((1, 23, 37, 4, 6)), make_audio(), "24 channels"),
        (make_video(latent_t=36), make_audio(), r"5n\+2"),
        (make_video(), torch.zeros((1, 32, 1, 207)), "stereo dimension 2"),
        (make_video().expand(2, -1, -1, -1, -1), make_audio(), "batch size 1"),
    ],
)
def test_prepare_rejects_malformed_h3_streams(video, audio, message):
    with pytest.raises(ValueError, match=message):
        prepare({"samples": video}, {"samples": audio})


def test_separate_is_vae_free_and_preserves_metadata_and_masks():
    video = make_video()
    audio = make_audio()
    video_mask = torch.full_like(video, 0.3)
    audio_mask = torch.full_like(audio, 0.7)
    av = {
        "samples": comfy.nested_tensor.NestedTensor((video, audio)),
        "noise_mask": comfy.nested_tensor.NestedTensor((video_mask, audio_mask)),
        "source_hash": "abc",
    }

    video_out, audio_out, report = separate_source_av_latent(av)
    data = json.loads(report)

    assert video_out["samples"].data_ptr() == video.data_ptr()
    assert audio_out["samples"].data_ptr() == audio.data_ptr()
    assert video_out["noise_mask"].data_ptr() == video_mask.data_ptr()
    assert audio_out["noise_mask"].data_ptr() == audio_mask.data_ptr()
    assert video_out["source_hash"] == audio_out["source_hash"] == "abc"
    assert data["facts"]["frame_count"] == 124
    assert data["facts"]["has_video_noise_mask"] is True
    assert data["facts"]["has_audio_noise_mask"] is True


def test_separate_rejects_mismatched_audio_clock():
    av = {
        "samples": comfy.nested_tensor.NestedTensor(
            (make_video(), make_audio(latent_t=206))
        )
    }
    with pytest.raises(ValueError, match="audio clock"):
        separate_source_av_latent(av)


def test_source_video_repaint_api_uses_prepared_latent_for_both_sampler_inputs():
    root = Path(__file__).resolve().parents[1]
    prompt = json.loads((root / "tests" / "fixtures" / "api" / "source_video_repaint_api.json").read_text("utf-8"))
    by_type = {node["class_type"]: node for node in prompt.values()}

    window = by_type["MiniMaxH3SourceMediaWindowT8"]
    prepare_node = by_type["MiniMaxH3SourceAVPrepareT8"]
    dual_clock = by_type["MiniMaxH3DualClockSamplerT8"]
    sampler = by_type["SamplerCustomAdvanced"]
    conditioning = by_type["MiniMaxH3AudioConditioningT8"]

    assert window["inputs"]["frames"] == ["2", 0]
    assert window["inputs"]["source_audio"] == ["2", 1]
    assert window["inputs"]["source_fps"] == ["2", 2]
    assert prepare_node["inputs"]["video_mode"] == "remix"
    assert prepare_node["inputs"]["audio_mode"] == "lock"
    assert dual_clock["inputs"]["av_latent"] == ["8", 0]
    assert sampler["inputs"]["latent_image"] == ["8", 0]
    assert conditioning["inputs"]["task_type"] == "T2VA"
    assert not any(node["class_type"].startswith("VHS_") for node in prompt.values())


def test_source_video_repaint_frontend_workflow_has_bidirectional_links():
    root = Path(__file__).resolve().parents[1]
    workflow = json.loads(
        (
            root
            / "examples"
            / "workflows"
            / "03-image-video-edit"
            / "2026-08-09_H3_Source_Video_Repaint_Stock20_EXP.json"
        ).read_text("utf-8")
    )
    nodes = {node["id"]: node for node in workflow["nodes"]}
    links = {link[0]: link for link in workflow["links"]}
    types = {node["type"] for node in nodes.values()}

    assert {
        "LoadVideo",
        "GetVideoComponents",
        "MiniMaxH3SourceMediaWindowT8",
        "MiniMaxH3SourceAVPrepareT8",
        "SamplerCustomAdvanced",
        "MiniMaxH3AVDecodeT8",
        "SaveVideo",
    } <= types
    assert not any(node_type.startswith("VHS_") for node_type in types)

    for link_id, origin_id, origin_slot, target_id, target_slot, link_type in workflow["links"]:
        assert links[link_id][5] == link_type
        assert link_id in nodes[origin_id]["outputs"][origin_slot]["links"]
        assert nodes[target_id]["inputs"][target_slot]["link"] == link_id
