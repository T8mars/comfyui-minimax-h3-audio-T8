"""Real tiny Core sampling and native adapter loading; no trained quality claim."""
import json

import pytest
import torch
from safetensors.torch import save_file
from comfy_extras.nodes_custom_sampler import BasicGuider, RandomNoise, SamplerCustomAdvanced

from h3_audio_t8_pkg.core import nested_av_parts
from h3_audio_t8_pkg.h3_lora_compat_advanced import load_minimax_h3_lora_model
from h3_audio_t8_pkg.open_regenerate import prepare_refine, refine_sigmas
from test_fast_h3_v2_core_sampler import model
from test_h24_lora_file_alpha import check_forward
from test_progressive_sampling_runtime import conditioning, tiny_model


class SilenceVAE:
    audio_sample_rate = 32000

    def __init__(self):
        self.calls = []

    def encode(self, waveform):
        self.calls.append(waveform.clone())
        return torch.full((1, 32, 2, round(waveform.shape[1] / 800)), .125)


@pytest.mark.parametrize("preset,expected", [
    ("published_3step", [.909375, 6 / 7, 2 / 3, 0]),
    ("published_2step", [.7, 42 / 85, 0]),
])
def test_release_grids_are_raw_sigmas_not_old_two_pass_shifted_again(preset, expected):
    torch.testing.assert_close(refine_sigmas(preset), torch.tensor(expected))


def test_custom_shift6_and_one_step_are_explicit_and_unbounded_by_preset():
    torch.testing.assert_close(refine_sigmas("custom_shift6", 1, .8), torch.tensor([.8, 0]))
    values = refine_sigmas("custom_shift6", 4, .909375)
    torch.testing.assert_close(values, torch.tensor([.909375, .84144578, .73207547, .52669683, 0]))
    assert len(refine_sigmas("custom_shift6", 11, .8)) == 12
    for steps, start in ((0, .8), (3, float("nan")), (3, 0), (True, .8)):
        with pytest.raises(ValueError):
            refine_sigmas("custom_shift6", steps, start)


@pytest.mark.parametrize("preset,nfe", [("published_3step", 3), ("published_2step", 2)])
def test_actual_core_preserves_encoded_silence_and_source_with_exact_initial_blend(preset, nfe):
    bare, vae = model(), SilenceVAE()
    video = torch.linspace(-.1, .1, 1 * 24 * 2 * 2 * 2).reshape(1, 24, 2, 2, 2)
    before = video.clone()
    source = {"samples": video, "noise_mask": torch.zeros_like(video), "source_id": "unchanged"}
    prepared, sampler, sigmas, av, text = prepare_refine(bare, source, vae, preset)
    assert len(vae.calls) == 1 and not vae.calls[0].count_nonzero()
    av_video, av_audio = nested_av_parts(av)
    assert torch.equal(av_video, before) and torch.all(av_audio == .125)
    video_mask, audio_mask = av["noise_mask"].unbind()
    assert torch.all(video_mask == 1) and torch.all(audio_mask == 0)
    assert av['source_id'] == 'unchanged' and not source['noise_mask'].count_nonzero()
    noise = RandomNoise.execute(281028).result[0]
    random_video = noise.generate_noise(av).unbind()[0]
    normalized_video = bare.model.process_latent_in(av["samples"]).unbind()[0]
    expected = (1 - sigmas[0]) * normalized_video + sigmas[0] * random_video
    calls = []

    def observe(_module, args, kwargs):
        x = kwargs.get("x", args[0] if args else None)
        timestamp = kwargs.get("timestep", args[1] if len(args) > 1 else None)
        calls.append((x[0].clone(), timestamp.clone()))

    handle = bare.model.diffusion_model.register_forward_pre_hook(observe, with_kwargs=True)
    try:
        guider = BasicGuider.execute(prepared, conditioning()).result[0]
        output = SamplerCustomAdvanced.execute(noise, guider, sampler, sigmas, av).result[0]
    finally:
        handle.remove()
    assert len(calls) == nfe
    torch.testing.assert_close(calls[0][0], expected)
    torch.testing.assert_close(calls[0][1].flatten()[0], sigmas[0] * 1000)
    result_video, result_audio = nested_av_parts(output)
    torch.testing.assert_close(result_audio, av_audio, rtol=0, atol=2e-5)
    assert torch.isfinite(result_video).all() and not torch.equal(result_video, before)
    assert torch.equal(video, before) and not bare.object_patches
    assert json.loads(text)['trained_GPU_quality_verified'] is False
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("strength", [1., .35])
def test_native_kijai_fused_qkv_rank1152_keeps_baked_scale_and_user_strength(tmp_path, strength):
    generator = torch.Generator().manual_seed(281028)
    a = (torch.randn(1152, 8, generator=generator) * .002).bfloat16()
    b = (torch.randn(48, 1152, generator=generator) * .002).bfloat16()
    path = tmp_path / "user_selected.safetensors"
    module = "diffusion_model.blocks.0.attn.qkv_proj"
    save_file({module + '.lora_A.weight': a, module + '.lora_B.weight': b}, str(path), metadata={
        "converted_layout": "comfyui_minimax_h3", "baked_scale": "1.0",
        "split": "refiner ranks 0:384; alpha == rank, strength folded",
    })
    base = tiny_model()
    patched, report = load_minimax_h3_lora_model(base, path, strength)
    check_forward(patched, base, b.float() @ a.float(), strength)
    data = json.loads(report)
    assert data['applied_patch_count'] == 1 and data['missed_patch_target_count'] == 0
    assert data['selected_strength'] == strength
