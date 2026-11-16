"""A split must reproduce a single trajectory before spatial changes are introduced."""
import pytest
import torch
import json
from comfy.nested_tensor import NestedTensor
from comfy_extras.nodes_custom_sampler import BasicGuider, DisableNoise, RandomNoise, SamplerCustomAdvanced
from h3_audio_t8_pkg.sampling import setup_dual_clock_sampling
from h3_audio_t8_pkg.two_pass_audio_continuation import continue_audio
from test_fast_h3_v2_core_sampler import model
from test_progressive_sampling_runtime import conditioning, latent


@pytest.mark.parametrize('kind', ['dual_clock_euler', 'euler'])
@pytest.mark.parametrize('task', ['t2va', 'i2va'])
def test_split_trajectory_matches_unsplit_without_spatial_intervention(kind, task):
    source = latent()
    branch, sampler, sigmas = setup_dual_clock_sampling(model(), source, 8, 12., 3., sampler_name=kind)
    guider = BasicGuider.execute(branch, conditioning(task)).result[0]
    noise = RandomNoise.execute(170801).result[0]
    complete = SamplerCustomAdvanced.execute(noise, guider, sampler, sigmas, source).result[0]
    coarse = SamplerCustomAdvanced.execute(noise, guider, sampler, sigmas[:5], source).result[0]
    high, remaining_noise, _ = continue_audio(coarse, coarse, DisableNoise.execute().result[0], sigmas[:5], sigmas[4:], sampler)
    resumed = SamplerCustomAdvanced.execute(remaining_noise, guider, sampler, sigmas[4:], high).result[0]
    for stream in (0, 1):
        torch.testing.assert_close(complete['samples'].unbind()[stream], resumed['samples'].unbind()[stream], rtol=1e-5, atol=1e-5)


def test_direct_clock_normalization_preserves_high_known_audio_and_input_objects():
    high = latent()
    video, audio = high['samples'].unbind()
    audio.fill_(.75)
    mask = torch.ones_like(audio)
    mask[..., :2] = 0
    high['noise_mask'] = NestedTensor((torch.ones_like(video), mask))
    _, sampler, sigmas = setup_dual_clock_sampling(model(), high, 8, 12., 3.)
    continued = torch.full_like(audio, 3.25)
    coarse = {'samples': NestedTensor((video, continued))}
    prepared, _, report = continue_audio(coarse, high, DisableNoise.execute().result[0], sigmas[:5], sigmas[4:], sampler)
    actual = prepared['samples'].unbind()[1]
    torch.testing.assert_close(actual[..., 2:], torch.ones_like(actual[..., 2:]), rtol=1e-5, atol=1e-5)
    assert torch.equal(actual[..., :2], audio[..., :2])
    assert prepared['samples'].unbind()[0] is video and prepared['noise_mask'] is high['noise_mask']
    assert torch.all(continued == 3.25) and torch.all(audio == .75)
    assert json.loads(report)['low_output_clock'] == 'T8_direct_audio_clock_canonicalized'


def test_unlisted_sampler_metadata_does_not_block_or_claim_clock_certification():
    high = latent()
    prepared, noise, report = continue_audio(high, high, DisableNoise.execute().result[0],
                                            torch.tensor([1., .5]), torch.tensor([.5, 0.]), object())
    assert prepared['samples'].unbind()[1] is high['samples'].unbind()[1]
    assert json.loads(report)['low_output_clock'] == 'unlisted_sampler_preserved_clock_unverified'
    assert not noise.generate_noise(prepared).unbind()[1].count_nonzero()
