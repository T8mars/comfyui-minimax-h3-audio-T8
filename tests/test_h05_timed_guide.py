"""Actual Core AddGuide contracts with test encoders; not GPU/quality evidence."""
import json

import pytest
import torch
from comfy_extras.nodes_minimax_h3 import MiniMaxH3AddGuide

from helpers import FakeAudioVAE, FakeClip, FakeVideoVAE, make_audio
from h3_audio_t8_pkg.conditioning import build_conditioning


def setup():
    result = build_conditioning(clip=FakeClip(), video_vae=FakeVideoVAE(),
        audio_vae=FakeAudioVAE(), prompt="Two voice references <Audio 1> and <Audio 2>.",
        width=128, height=128, length=124, task_type="Ref2VA", audio_mode="native",
        ref_audios={"ref_audio_0": make_audio(3.), "ref_audio_1": make_audio(2., value=.2)})
    positive, latent = result[:2]
    assert json.loads(result[4])["audios"] == {"1": "ref_audio_1", "2": "ref_audio_2"}
    assert len(positive[0][1]["minimax_refs"]) == 2
    return positive, latent


def test_midpoint_and_negative_endpoint_append_preserves_real_voice_ref_rows():
    positive, latent = setup()
    refs = positive[0][1]["minimax_refs"]
    video, audio = latent["samples"].tensors
    before = video.clone(), audio.clone()
    vae = FakeVideoVAE()
    image = torch.full((1, 128, 128, 3), .3)
    midpoint = MiniMaxH3AddGuide.execute(positive, latent, 72, vae=vae, image=image).result[0]
    endpoint = MiniMaxH3AddGuide.execute(midpoint, latent, -1, vae=vae, image=image).result[0]
    assert [row["resolved_frame_index"] for row in endpoint[0][1]["minimax_keyframes"]] == [72, 123]
    assert "minimax_keyframes" not in positive[0][1]
    for candidate in (midpoint, endpoint):
        assert candidate[0][1]["minimax_refs"] is refs
        assert all(candidate[0][1][key] is value for key, value in positive[0][1].items())
        assert candidate[0][0] is positive[0][0]
    # Actual Core Lanczos deliberately goes through uint8 PIL even at the same
    # size: float .3 becomes floor(.3*255)/255, not a float-pixel identity.
    expected_pixels = torch.full_like(image, 76/255)
    assert len(vae.encode_calls) == 2
    assert all(torch.equal(row, expected_pixels) for row in vae.encode_calls)
    assert torch.equal(image, torch.full_like(image, .3))
    assert torch.equal(video, before[0]) and torch.equal(audio, before[1])


def test_audio_guide_uses_same_target_frame_without_replacing_voice_references():
    positive, latent = setup()
    refs = positive[0][1]["minimax_refs"]
    vae = FakeAudioVAE()
    guided = MiniMaxH3AddGuide.execute(positive, latent, 72, audio_vae=vae,
                                      audio=make_audio(3.)).result[0]
    row, = guided[0][1]["minimax_keyframes"]
    assert row["resolved_frame_index"] == 72
    assert row["audio_latent"].shape == (1, 32, 2, 87)
    assert guided[0][1]["minimax_refs"] is refs and len(refs) == 2
    assert len(vae.encode_calls) == 1
    assert "minimax_keyframes" not in positive[0][1]


def test_out_of_range_guide_rejects_before_encoder_or_conditioning_mutation():
    positive, latent = setup()
    vae = FakeVideoVAE()
    for index in (124, -125):
        with pytest.raises(ValueError, match="outside"):
            MiniMaxH3AddGuide.execute(positive, latent, index, vae=vae,
                                    image=torch.zeros(1, 128, 128, 3))
    assert not vae.encode_calls and "minimax_keyframes" not in positive[0][1]
