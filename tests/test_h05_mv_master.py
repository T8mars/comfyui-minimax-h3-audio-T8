"""Three-role and nonzero-origin contracts; test doubles are not GPU evidence."""
import json

import pytest
import torch

from comfy_extras.nodes_audio import TrimAudioDuration
from h3_audio_t8_pkg.audio_ops import trim_av_output
from h3_audio_t8_pkg.conditioning import build_conditioning
from h3_audio_t8_pkg.source_av import prepare_source_media_window
from helpers import FakeAudioVAE, FakeClip, FakeVideoVAE


def source(value, samples=384000):
    waveform = torch.arange(samples, dtype=torch.float32).remainder(4000).div(4000)
    return {"sample_rate": 32000, "waveform": (waveform*value)[None, None].expand(1, 2, -1).clone()}


def trim(audio, start, duration):
    return TrimAudioDuration.execute(audio, start, duration)[0]


def test_nonzero_global_origin_separate_drive_reference_and_final_passthrough():
    vocal, mix = source(0.1), source(0.3)
    drive, final = trim(vocal, 5., 124/24), trim(mix, 5., 124/24)
    voice = trim(vocal, 1., 3.)
    vae = FakeAudioVAE()
    _, latent, mux, _, mapping, _ = build_conditioning(clip=FakeClip(), video_vae=FakeVideoVAE(),
        audio_vae=vae, prompt="<Audio 1> timing, <Audio 2> voice only", width=128, height=128,
        length=124, task_type="Ref2VA", audio_mode="lock_source", audio_denoise_strength=0.,
        add_source_as_reference=True, drive_audio=drive, final_audio=final,
        ref_audios={"ref_audio_0": voice})
    assert mux is final
    assert len(vae.encode_calls) == 2
    assert torch.equal(vae.encode_calls[0], drive["waveform"].movedim(1, -1))
    assert torch.equal(vae.encode_calls[1], voice["waveform"].movedim(1, -1))
    assert json.loads(mapping)["audios"] == {"1": "drive_audio (primary source)", "2": "ref_audio_1"}
    assert torch.count_nonzero(latent["noise_mask"].unbind()[1]) == 0
    frames = torch.zeros(124, 32, 32, 3)
    _, delivered, _ = trim_av_output(frames, 0., 5., mux, 24.)
    assert torch.equal(delivered["waveform"], mix["waveform"][..., 160000:320000])
    assert not torch.equal(delivered["waveform"], drive["waveform"][..., :160000])


def test_source_prepare_and_core_audio_trim_share_actual_nonzero_sample_origin():
    frames = torch.zeros(248, 32, 32, 3)
    mix = source(0.3)
    result = prepare_source_media_window(frames, 24., 32, 32, 124, 5., "strict", "strict", mix,
                                         return_frame_map=True)
    assert result[5]["source_indices"] == tuple(range(120, 244))
    assert result[5]["audio_start_sample_32k"] == 160000
    assert torch.equal(result[1]["waveform"], trim(mix, 5., 124/24)["waveform"])


def test_missing_audio_is_explicit_silence_and_short_master_is_not_preserved():
    frames = torch.zeros(248, 32, 32, 3)
    silent = prepare_source_media_window(frames, 24., 32, 32, 124, 5., "strict", "strict")
    assert json.loads(silent[4])["facts"]["audio_source"] == "generated_silence"
    assert torch.count_nonzero(silent[1]["waveform"]) == 0
    with pytest.raises(ValueError, match="too short"):
        prepare_source_media_window(frames, 24., 32, 32, 124, 5., "strict", "strict", source(0.3, 170000))
