from __future__ import annotations

import json

import pytest
import torch

from h3_audio_t8_pkg.still_image import (
    build_still_image_edit_conditioning,
    decode_still_image,
    empty_still_av_latent,
    resolve_still_canvas,
    resolve_still_target,
    run_still_preflight,
)
from h3_audio_t8_pkg.core import MAX_PIXELS
from helpers import FakeClip, FakeVideoVAE


def edit_args():
    return {
        "clip": FakeClip(),
        "video_vae": FakeVideoVAE(),
        "prompt": "Edit Picture 1 so the coat is red while preserving the person.",
        "canvas_mode": "custom",
        "width": 128,
        "height": 128,
        "target_mode": "direct_1_frame",
        "reference_strength": 0.999,
        "audio_target": "generate_and_discard",
        "strict_prompt_tags": True,
        "ref_image_size": "match",
        "edit_image": torch.zeros((1, 96, 160, 3)),
    }


def test_still_target_modes_have_expected_joint_latent_shapes():
    assert resolve_still_target("direct_1_frame") == (1, 1, 2)
    assert resolve_still_target("micro_video_5_frames") == (5, 2, 8)
    assert resolve_still_target("short_video_22_frames") == (22, 7, 37)
    assert resolve_still_target("trained_124_frames") == (124, 37, 207)

    direct, frames = empty_still_av_latent(
        128, 96, "direct_1_frame", "generate_and_discard"
    )
    video, audio = direct["samples"].unbind()
    assert frames == 1
    assert video.shape == (1, 24, 1, 6, 8)
    assert audio.shape == (1, 32, 2, 2)
    assert "noise_mask" not in direct


def test_reference_image_edit_builds_direct_single_frame_ref2va_payload():
    args = edit_args()
    conditioning, latent, prompt, media_map, report = build_still_image_edit_conditioning(
        **args
    )
    metadata = conditioning[0][1]
    video, audio = latent["samples"].unbind()

    assert prompt.startswith("Edit <Picture 1>")
    assert video.shape == (1, 24, 1, 8, 8)
    assert audio.shape == (1, 32, 2, 2)
    assert metadata["minimax_visual_cond_noise_aug"] == pytest.approx(0.999)
    assert [ref["kind"] for ref in metadata["minimax_refs"]] == ["image"]
    assert metadata["minimax_refs"][0]["latent"].shape[2] == 1
    assert args["clip"].tokenize_calls[0][1]["minimax_ref_items"][0]["type"] == "image"
    assert json.loads(media_map)["pictures"] == {"1": "edit_image (primary source)"}
    report_data = json.loads(report)
    assert report_data["target"]["frames"] == 1
    assert report_data["required_checkpoint_family"] == "MiniMax H3 Ref2VA"


def test_multi_reference_order_and_strict_picture_validation():
    args = edit_args()
    args["ref_images"] = {
        "ref_image_2": torch.ones((1, 64, 64, 3)),
        "ref_image_1": torch.full((1, 64, 64, 3), 0.5),
    }
    conditioning, _, _, media_map, _ = build_still_image_edit_conditioning(**args)
    assert len(conditioning[0][1]["minimax_refs"]) == 3
    assert json.loads(media_map)["pictures"] == {
        "1": "edit_image (primary source)",
        "2": "ref_image_1",
        "3": "ref_image_2",
    }

    args["prompt"] = "Use Picture 4 for the background"
    with pytest.raises(ValueError, match="Picture 4"):
        build_still_image_edit_conditioning(**args)


def test_locked_silence_preserves_video_denoising_and_locks_audio():
    args = edit_args()
    args["audio_target"] = "lock_silence"
    _, latent, *_ = build_still_image_edit_conditioning(**args)
    video_mask, audio_mask = latent["noise_mask"].unbind()
    assert torch.all(video_mask == 1)
    assert torch.all(audio_mask == 0)


def test_canvas_from_edit_image_and_custom_1080p_stay_inside_supported_cap():
    image = torch.zeros((1, 720, 1280, 3))
    width, height = resolve_still_canvas(image, "from_edit_image", 32, 32)
    assert width % 32 == 0 and height % 32 == 0
    assert width * height <= MAX_PIXELS

    assert resolve_still_canvas(image, "custom", 1920, 1088) == (1920, 1088)

    with pytest.raises(ValueError, match="2,088,960"):
        resolve_still_canvas(image, "custom", 1952, 1088)


def test_still_preflight_reports_missing_image_and_single_frame_ood_warning():
    ready, _, report = run_still_preflight(
        "from_edit_image",
        1344,
        768,
        "direct_1_frame",
        0.999,
        "generate_and_discard",
    )
    data = json.loads(report)
    assert ready is False
    assert any("requires edit_image" in error for error in data["errors"])

    ready, warning_count, report = run_still_preflight(
        "from_edit_image",
        1344,
        768,
        "direct_1_frame",
        0.999,
        "generate_and_discard",
        video_vae=FakeVideoVAE(),
        edit_image=torch.zeros((1, 96, 160, 3)),
    )
    data = json.loads(report)
    assert ready is True
    assert warning_count >= 3
    assert data["facts"]["target"]["video_latent_t"] == 1

    ready, _, report = run_still_preflight(
        "custom",
        128,
        128,
        "direct_1_frame",
        0.999,
        "generate_and_discard",
        video_vae=FakeVideoVAE(),
        edit_image=torch.zeros((1, 128, 128, 3)),
    )
    assert ready is True
    assert any("below 512" in warning for warning in json.loads(report)["warnings"])


def test_still_decode_selects_candidate_without_decoding_audio():
    latent, _ = empty_still_av_latent(
        128, 128, "micro_video_5_frames", "generate_and_discard"
    )
    selected, candidates, report = decode_still_image(
        latent, FakeVideoVAE(), "middle", 0
    )
    assert candidates.shape[0] == 5
    assert selected.shape[0] == 1
    assert json.loads(report)["selected_index"] == 2

    with pytest.raises(ValueError, match="outside the decoded range"):
        decode_still_image(latent, FakeVideoVAE(), "index", 5)


def test_native_short_22_frame_target_decodes_all_candidates():
    latent, frames = empty_still_av_latent(
        128, 128, "short_video_22_frames", "generate_and_discard"
    )
    video, audio = latent["samples"].unbind()
    assert frames == 22
    assert video.shape == (1, 24, 7, 8, 8)
    assert audio.shape == (1, 32, 2, 37)

    selected, candidates, report = decode_still_image(
        latent, FakeVideoVAE(), "middle", 0
    )
    assert candidates.shape[0] == 22
    assert selected.shape[0] == 1
    assert json.loads(report)["selected_index"] == 11
