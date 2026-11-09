"""Real source slices, tiny random Core encoders/layout/sampling; no Qwen quality claim."""
import json

import pytest
import torch
from comfy.ldm.minimax import vae as video_core
from comfy.ldm.minimax import model as model_core
from comfy.ldm.minimax.audio_vae import MiniMaxH3AudioVAE
from comfy.ldm.modules import attention
from comfy_extras.nodes_custom_sampler import BasicGuider, RandomNoise, SamplerCustomAdvanced

from h3_audio_t8_pkg.conditioning import build_conditioning
from h3_audio_t8_pkg.core import resize_image
from h3_audio_t8_pkg.sampling import setup_dual_clock_sampling
from h3_audio_t8_pkg.source_av import prepare_source_media_window
from helpers import FakeClip
from test_fast_h3_v2_core_sampler import model
from test_h21_core_vae_contracts import initialize


class RealVideoEncoderOwner:
    latent_channels, latent_dim = 24, 3

    def __init__(self):
        self.first_stage_model = video_core.MiniMaxH3VideoVAE(
            ch=32, ch_mult=(1, 1, 1, 1, 1, 1), num_res_blocks=1,
            encoder=True, decoder=False, tiling=False)
        initialize(self.first_stage_model, 5111)
        self.inputs = []

    def encode(self, frames):
        self.inputs.append(frames.clone())
        return self.first_stage_model.encode(frames.movedim(-1, 0).unsqueeze(0) * 2 - 1)


class RealAudioEncoderOwner:
    latent_channels, latent_dim, audio_sample_rate = 32, 2, 32000

    def __init__(self):
        self.first_stage_model = MiniMaxH3AudioVAE(encoder_dim=8, latent_dim=16, decoder_dim=128)
        initialize(self.first_stage_model, 5112)
        self.first_stage_model.latents_mean.zero_()
        self.first_stage_model.latents_std.fill_(1.)
        self.inputs = []

    def encode(self, audio):
        self.inputs.append(audio.movedim(-1, 1).clone())
        return self.first_stage_model.encode(audio.movedim(-1, 1))


def source():
    frames = torch.arange(90, dtype=torch.float32)[:, None, None, None].expand(-1, 32, 64, 3) / 90
    time = torch.arange(96000) / 32000
    audio = {"waveform": torch.stack((time / 6, torch.sin(time * 77) * .1))[None], "sample_rate": 32000}
    return frames, audio


def test_two_source_windows_feed_paired_native_reference_and_real_core_sampler(monkeypatch, tmp_path):
    monkeypatch.setattr(video_core, "optimized_attention", attention.attention_pytorch)
    monkeypatch.setattr(model_core, "optimized_attention", attention.attention_pytorch)
    video_vae, audio_vae = RealVideoEncoderOwner(), RealAudioEncoderOwner()
    frames, audio = source()
    original_frames, original_pcm = frames.clone(), audio["waveform"].clone()
    receipts, encoded = [], []
    for start in (.5, 1.):
        window, pcm, count, duration, raw, mapping = prepare_source_media_window(
            frames, 30., 64, 32, 22, start, "strict", "strict", audio, return_frame_map=True)
        indices = torch.tensor(mapping["source_indices"])
        # Existing Lanczos resize includes its documented RGB quantization.
        assert torch.equal(window, resize_image(frames[indices], 64, 32, "disabled"))
        first_sample = round(start * 32000)
        assert torch.equal(pcm["waveform"], audio["waveform"][..., first_sample:first_sample + round(duration*32000)])
        clip = FakeClip()  # Explicit tokenizer/embedding substitute; everything after it uses real Core.
        with torch.no_grad():
            positive, latent, _, _, media, _, detail = build_conditioning(
                clip, video_vae, audio_vae, "Video 1 with Audio 1", 64, 32, count,
                audio_mode="native", ref_videos={"ref_video_3": window},
                ref_video_audios={"ref_video_audio_3": pcm},
                reference_video_policy="model_minimum", return_details=True)
            block = detail["refs"][0]
            assert block["kind"] == "video_audio" and block["ref_audio_t"] > 0
            assert torch.equal(video_vae.inputs[-1], window)
            assert torch.equal(audio_vae.inputs[-1], pcm["waveform"])
            assert json.loads(media)["audios"] == {"1": "ref_video_audio_3"}
            assert clip.tokenize_calls[-1][1]["minimax_ref_items"][1]["timestamps"] == [0., .5]
            bare = model()
            observed = []
            def before_forward(module, args, kwargs):
                payload = kwargs["minimax_payload"]
                actual = payload["refs"][0]
                assert actual["kind"] == "video_audio"
                torch.testing.assert_close(actual["latent"], block["latent"], rtol=0, atol=0)
                torch.testing.assert_close(actual["audio_latent"], block["audio_latent"], rtol=0, atol=0)
                observed.append(True)
            hook = bare.model.diffusion_model.register_forward_pre_hook(before_forward, with_kwargs=True)
            try:
                prepared, sampler, sigmas = setup_dual_clock_sampling(bare, latent, 2, 12., 3.)
                result = SamplerCustomAdvanced.execute(RandomNoise.execute(5113).result[0],
                    BasicGuider.execute(prepared, positive).result[0], sampler, sigmas, latent).result[0]
            finally:
                hook.remove()
            assert len(observed) == 2
            assert all(bool(torch.isfinite(x).all()) for x in result["samples"].unbind())
        encoded.append((block["latent"], block["audio_latent"]))
        receipts.append({"source_start_seconds": start, "source_indices": list(mapping["source_indices"]),
            "pcm_start_sample": first_sample, "pcm_samples": pcm["waveform"].shape[-1],
            "video_latent_shape": list(block["latent"].shape), "audio_latent_shape": list(block["audio_latent"].shape),
            "native_ref_kind": block["kind"], "actual_forwards": len(observed), "window": json.loads(raw)})
    assert not torch.equal(encoded[0][0], encoded[1][0])
    assert not torch.equal(encoded[0][1], encoded[1][1])
    assert torch.equal(frames, original_frames) and torch.equal(audio["waveform"], original_pcm)
    (tmp_path / "native-slices.json").write_text(json.dumps({"windows": receipts,
        "encoders_and_sampler": "actual tiny random Core", "qwen": "explicit test embedding substitute",
        "trained_model": False, "human_reviewed": False, "cuda_initialized": torch.cuda.is_initialized()}, indent=2), encoding="utf-8")
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("short", ["video", "audio"])
def test_source_slice_rejects_missing_duration_before_encoding(short):
    frames, audio = source()
    if short == "video":
        frames = frames[:30]
    else:
        audio = {**audio, "waveform": audio["waveform"][..., :32000]}
    with pytest.raises(ValueError, match="too short"):
        prepare_source_media_window(frames, 30., 64, 32, 22, .5, "strict", "strict", audio)
