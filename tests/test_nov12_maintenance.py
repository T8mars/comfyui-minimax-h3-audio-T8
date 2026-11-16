"""Selected combined CPU consumers; no trained repair or tile-quality claim."""

import json

import pytest
import torch
import comfy.model_management as mm
from comfy.ldm.minimax import model as model_core
from comfy.ldm.minimax import vae as video_core
from comfy.ldm.modules import attention
from comfy_extras.nodes_custom_sampler import BasicGuider, RandomNoise, SamplerCustomAdvanced

from h3_audio_t8_pkg import chunked_two_pass_upscale_advanced as chunked
from h3_audio_t8_pkg.conditioning import build_conditioning
from h3_audio_t8_pkg.lanpaint_av_advanced import prepare_lanpaint_av_latent
from h3_audio_t8_pkg.sampling import setup_dual_clock_sampling
from helpers import FakeClip
from test_fast_h3_v2_core_sampler import model
from test_nov05_reference_slices import RealAudioEncoderOwner, RealVideoEncoderOwner, source
from test_progressive_sampling_runtime import conditioning
from test_chunked_two_pass_global_noise_advanced import _plan


@pytest.mark.parametrize("sparse", [False, True])
def test_moving_mask_with_distinct_source_audio_reference_reaches_actual_cpu_dit(
    monkeypatch, tmp_path, sparse,
):
    monkeypatch.setattr(video_core, "optimized_attention", attention.attention_pytorch)
    monkeypatch.setattr(model_core, "optimized_attention", attention.attention_pytorch)
    frames, pcm = source()
    frames = frames[:22]
    pcm = {**pcm, "waveform": pcm["waveform"][..., :round(22 / 24 * 32000)]}
    # The guide differs from the target audio, so a final PCM passthrough alone
    # cannot satisfy the reference-consumption check below.
    guide = {**pcm, "waveform": pcm["waveform"].flip(-1).clone() * .7}
    video_vae, audio_vae = RealVideoEncoderOwner(), RealAudioEncoderOwner()
    mask = torch.zeros(frames.shape[:3])
    for i in range(len(mask)):
        if not sparse or i % 3 == 0:
            x = i * 3 % 48
            mask[i, 8:24, x:x+12] = 1
    with torch.no_grad():
        latent, *_ = prepare_lanpaint_av_latent(
            frames, pcm, video_vae, audio_vae, "[]", "strict", False, mask,
        )
        positive, _, final_pcm, *_ = build_conditioning(
            FakeClip(), video_vae, audio_vae, "<Audio 1> speaks while the hand moves.",
            64, 32, 22, audio_mode="reference_only", drive_audio=guide,
            final_audio=pcm,
        )
        refs = positive[0][1]["minimax_refs"]
        expected = next(r for r in refs if r["kind"] == "audio")["audio_latent"]
        before = [t.clone() for t in latent["samples"].unbind()]
        bare = model()
        calls = []

        def observe(_module, _args, kwargs):
            actual = next(r for r in kwargs["minimax_payload"]["refs"] if r["kind"] == "audio")
            torch.testing.assert_close(actual["audio_latent"], expected, rtol=0, atol=0)
            calls.append(True)

        hook = bare.model.diffusion_model.register_forward_pre_hook(observe, with_kwargs=True)
        try:
            prepared, sampler, sigmas = setup_dual_clock_sampling(bare, latent, 2, 12., 3.)
            result = SamplerCustomAdvanced.execute(RandomNoise.execute(12091).result[0],
                BasicGuider.execute(prepared, positive).result[0], sampler, sigmas, latent).result[0]
        finally:
            hook.remove()
    assert len(calls) == 2 and final_pcm is pcm
    actual_video, actual_audio = result["samples"].unbind()
    video_mask, audio_mask = latent["noise_mask"].unbind()
    protected = video_mask.expand_as(actual_video) == 0
    torch.testing.assert_close(actual_video[protected], before[0][protected], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual_audio, before[1], rtol=1e-5, atol=1e-6)
    assert torch.count_nonzero(audio_mask) == 0
    assert all(torch.equal(old, current) for old, current in zip(before, latent["samples"].unbind()))
    (tmp_path / "combined-consumption.json").write_text(json.dumps(dict(
        moving_mask="sparse_authored" if sparse else "every_frame_authored",
        actual_Core_forwards=len(calls), actual_audio_reference=True,
        final_audio_not_reference_guide=True, trained_model=False,
        cuda_initialized=torch.cuda.is_initialized())), "utf8")
    assert not torch.cuda.is_initialized()


def test_actual_cpu_tile_interrupt_propagates_and_retry_keeps_source_and_global_noise():
    video = torch.zeros(1, 24, 2, 4, 4)
    audio = torch.linspace(-.1, .1, 512).reshape(1, 32, 2, 8)
    latent = {"samples": chunked.comfy.nested_tensor.NestedTensor((video, audio))}
    bare = model()
    prepared, sampler, sigmas = setup_dual_clock_sampling(bare, latent, 2, 12., 3.)
    noise = RandomNoise.execute(12092).result[0]
    global_noise = noise.generate_noise(latent).unbind()[0]
    before = global_noise.clone()
    calls = []

    def interrupt(_module, _args, _kwargs):
        calls.append(True)
        if len(calls) == 3:
            raise mm.InterruptProcessingException()

    def run():
        return chunked._spatial_resample(video, audio, conditioning(),
            _plan(chunked.build_chunked_two_pass_global_noise_plan), prepared, noise,
            sampler, sigmas, None, 1., chunk_noise_video=global_noise,
            rebind_shape_bound_sampler=True)

    hook = bare.model.diffusion_model.register_forward_pre_hook(interrupt, with_kwargs=True)
    try:
        with pytest.raises(mm.InterruptProcessingException):
            run()
    finally:
        hook.remove()
    assert len(calls) == 3
    assert torch.count_nonzero(video) == 0 and torch.equal(global_noise, before)
    retry_calls = []
    hook = bare.model.diffusion_model.register_forward_pre_hook(
        lambda *_args, **_kwargs: retry_calls.append(True), with_kwargs=True,
    )
    try:
        output, report = run()
    finally:
        hook.remove()
    assert len(retry_calls) == 8  # Four actual tiles, two Euler evaluations each.
    assert torch.isfinite(output).all() and report["rows"] == [0, 2] and report["cols"] == [0, 2]
    assert torch.count_nonzero(video) == 0 and torch.equal(global_noise, before)
    assert not torch.cuda.is_initialized()
