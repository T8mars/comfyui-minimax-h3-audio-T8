from __future__ import annotations

import json

import pytest
import torch

from h3_audio_t8_pkg.conditioning import (
    HYBRID_KEYFRAME_SENTINEL,
    assert_hybrid_layout_contract,
    build_conditioning,
)
from helpers import FakeAudioVAE, FakeClip, FakeVideoVAE, make_audio


def base_args():
    return {
        "clip": FakeClip(),
        "video_vae": FakeVideoVAE(),
        "audio_vae": FakeAudioVAE(),
        "prompt": "Audio 1 drives the performance",
        "width": 128,
        "height": 128,
        "length": 124,
        "task_type": "auto",
        "audio_mode": "lock_source",
        "drive_audio": make_audio(),
    }


def test_source_audio_is_encoded_once_reused_and_locked():
    args = base_args()
    conditioning, latent, output_audio, prompt, media_map, report = build_conditioning(**args)
    assert len(args["audio_vae"].encode_calls) == 1
    video, audio = latent["samples"].unbind()
    video_mask, audio_mask = latent["noise_mask"].unbind()
    assert torch.all(audio[..., :200] == 0.25)
    assert torch.all(audio[..., 200:] == 0)
    assert torch.all(video_mask == 1)
    assert torch.all(audio_mask == 0)
    assert prompt == "<Audio 1> drives the performance"
    assert json.loads(media_map)["source_audio_ordinal"] == 1
    assert output_audio is args["drive_audio"]


def test_video_soundtracks_shift_primary_source_audio_tag():
    args = base_args()
    args.update({
        "ref_videos": {"ref_video_1": torch.zeros((48, 64, 64, 3)),
                       "ref_video_2": torch.zeros((48, 64, 64, 3))},
        "ref_video_audios": {"ref_video_audio_1": make_audio(2),
                             "ref_video_audio_2": make_audio(2)},
        "ref_audios": {"ref_audio_1": make_audio(1)},
    })
    conditioning, _, _, prompt, media_map, _ = build_conditioning(**args)
    media = json.loads(media_map)
    assert prompt.startswith("<Audio 3>")
    assert media["source_audio_ordinal"] == 3
    assert media["audios"] == {
        "1": "ref_video_audio_1", "2": "ref_video_audio_2",
        "3": "drive_audio (primary source)", "4": "ref_audio_1",
    }
    refs = conditioning[0][1]["minimax_refs"]
    assert [item["kind"] for item in refs] == ["video_audio", "video_audio", "audio", "audio"]


def test_soundtrack_pairing_uses_autogrow_ordinal_not_dense_position():
    args = base_args()
    args.update({
        "ref_videos": {"ref_video_1": torch.zeros((48, 64, 64, 3)),
                       "ref_video_2": torch.zeros((48, 64, 64, 3))},
        "ref_video_audios": {"ref_video_audio_2": make_audio(2)},
    })
    conditioning, *_ = build_conditioning(**args)
    refs = conditioning[0][1]["minimax_refs"]
    assert refs[0]["kind"] == "video"
    assert refs[1]["kind"] == "video_audio"


def test_exact_keyframe_and_audio_reference_use_guarded_hybrid_payload():
    assert_hybrid_layout_contract()
    args = base_args()
    args["first_frame"] = torch.zeros((1, 128, 128, 3))
    conditioning, _, _, _, media_map, report = build_conditioning(**args)
    metadata = conditioning[0][1]
    assert metadata["minimax_keyframes"][0]["resolved_frame_index"] == 0
    assert metadata["minimax_refs"][0]["kind"] == HYBRID_KEYFRAME_SENTINEL
    tokenize_kwargs = args["clip"].tokenize_calls[0][1]
    assert [item["type"] for item in tokenize_kwargs["minimax_ref_items"]] == ["image", "audio"]
    assert json.loads(media_map)["pictures"]["1"].startswith("first_frame")


def test_reference_only_keeps_blank_target_audio():
    args = base_args()
    args["audio_mode"] = "reference_only"
    _, latent, *_ = build_conditioning(**args)
    _, target_audio = latent["samples"].unbind()
    assert torch.all(target_audio == 0)
    assert "noise_mask" not in latent


def test_native_t2va_needs_no_source_audio():
    args = base_args()
    args.update({"audio_mode": "native", "drive_audio": None, "add_source_as_reference": False,
                 "prompt": "A person speaks naturally"})
    conditioning, latent, output_audio, prompt, media_map, report = build_conditioning(**args)
    assert output_audio is None
    assert "minimax_refs" not in conditioning[0][1]
    assert "task=t2va" in report


def test_orphan_video_soundtrack_is_rejected():
    args = base_args()
    args["ref_video_audios"] = {"ref_video_audio_2": make_audio(2)}
    with pytest.raises(ValueError, match="no same-numbered video"):
        build_conditioning(**args)


def test_canvas_allows_1080p_area_and_rejects_above_it():
    args = base_args()
    args.update({
        "width": 1920,
        "height": 1088,
        "length": 22,
        "audio_mode": "native",
        "prompt": "A quiet cinematic landscape",
        "drive_audio": None,
        "add_source_as_reference": False,
        "prompt_primary_audio_ordinal": 0,
    })
    _, latent, *_ = build_conditioning(**args)
    video, _audio = latent["samples"].unbind()
    assert video.shape[-2:] == (68, 120)

    args.update({"width": 1952, "height": 1088})
    with pytest.raises(ValueError, match="2,088,960"):
        build_conditioning(**args)
