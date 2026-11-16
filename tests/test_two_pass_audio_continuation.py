"""Opt-in audio handoff; tiny real Core tests do not qualify trained sound."""
import json

import pytest
import torch
from comfy.nested_tensor import NestedTensor
from comfy_extras.nodes_custom_sampler import BasicGuider, DisableNoise, RandomNoise, SamplerCustomAdvanced

from h3_audio_t8_pkg.sampling import setup_dual_clock_sampling
from h3_audio_t8_pkg.modular_sampling import automatic_reuse
from h3_audio_t8_pkg.two_pass_audio_continuation import continue_audio
from test_dmad_recovery import callback_interrupt
from test_fast_h3_v2_core_sampler import model
from test_progressive_sampling_runtime import conditioning, latent


def test_real_core_split_audio_continuation_matches_complete_trajectory():
    source = latent()
    selected, sampler, sigmas = setup_dual_clock_sampling(model(), source, 8, 12., 3.)
    guider = BasicGuider.execute(selected, conditioning()).result[0]
    initial = RandomNoise.execute(56001).result[0]
    coarse = SamplerCustomAdvanced.execute(initial, guider, sampler, sigmas[:5], source).result[0]
    empty = DisableNoise.execute().result[0]
    expected = SamplerCustomAdvanced.execute(initial, guider, sampler, sigmas, source).result[0]
    prepared, noise, text = continue_audio(coarse, coarse, empty, sigmas[:5], sigmas[4:], sampler)
    result = SamplerCustomAdvanced.execute(noise, guider, sampler, sigmas[4:], prepared).result[0]
    for a, b in zip(expected['samples'].unbind(), result['samples'].unbind()):
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
    assert json.loads(text)['extra_denoiser_evaluations'] == 0
    assert not torch.cuda.is_initialized()


def test_unknown_noise_provider_delegates_and_keeps_visual_seed_and_masks():
    high = latent()
    video, audio = high['samples'].unbind()
    coarse = {'samples': NestedTensor((video, torch.full_like(audio, .25)))}
    high['noise_mask'] = NestedTensor((torch.ones_like(video), torch.ones_like(audio)))
    high['noise_mask'].unbind()[1][..., 0, :] = 0
    high['marker'] = object()

    class UnlistedNoise:
        seed = 99
        calls = 0
        def generate_noise(self, target):
            self.calls += 1
            return NestedTensor((torch.full_like(video, .75), torch.ones_like(audio)))

    parent = UnlistedNoise()
    output, noise, _ = continue_audio(coarse, high, parent, torch.tensor([1., .5]), torch.tensor([.5, 0.]))
    actual_video, actual_audio = noise.generate_noise(output).unbind()
    assert parent.calls == 1 and noise.seed == 99
    assert torch.all(actual_video == .75) and not actual_audio.count_nonzero()
    assert output['noise_mask'] is high['noise_mask'] and output['marker'] is high['marker']
    assert output['samples'].unbind()[0] is video
    assert not output['samples'].unbind()[1][..., 0, :].count_nonzero()
    assert torch.all(output['samples'].unbind()[1][..., 1, :] == .25)
    assert not audio.count_nonzero() and torch.all(coarse['samples'].unbind()[1] == .25)


def prepared_high():
    source = latent()
    selected, sampler, sigmas = setup_dual_clock_sampling(model(), source, 8, 12., 3.)
    guider = BasicGuider.execute(selected, conditioning()).result[0]
    parent = RandomNoise.execute(56002).result[0]
    coarse = SamplerCustomAdvanced.execute(parent, guider, sampler, sigmas[:5], source).result[0]
    high, noise, _ = continue_audio(coarse, coarse, parent, sigmas[:5], sigmas[4:], sampler)
    return guider, sampler, sigmas[4:], high, noise


def test_unknown_noise_identity_executes_without_fake_portable_cache(tmp_path):
    guider, sampler, sigmas, high, noise = prepared_high()
    first = automatic_reuse.sample(noise, guider, sampler, sigmas, high, tmp_path, mode='reuse_completed')
    second = automatic_reuse.sample(noise, guider, sampler, sigmas, high, tmp_path, mode='reuse_completed')
    assert json.loads(first[3])['status'] == json.loads(second[3])['status'] == 'uncertified_executed'
    assert not list(tmp_path.rglob('manifest.json'))
    for a, b in zip(first[0]['samples'].unbind(), second[0]['samples'].unbind()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_interrupted_high_retries_same_noise_and_keeps_inputs(monkeypatch):
    import comfy.model_management as mm
    import latent_preview
    guider, sampler, sigmas, high, noise = prepared_high()
    before = [value.clone() for value in high['samples'].unbind()]
    expected = SamplerCustomAdvanced.execute(noise, guider, sampler, sigmas, high).result[0]
    callbacks, original = callback_interrupt(monkeypatch)
    try:
        with pytest.raises(mm.InterruptProcessingException):
            SamplerCustomAdvanced.execute(noise, guider, sampler, sigmas, high)
    finally:
        mm.interrupt_current_processing(False)
        monkeypatch.setattr(latent_preview, 'prepare_callback', original)
    assert callbacks == [0, 1]
    result = SamplerCustomAdvanced.execute(noise, guider, sampler, sigmas, high).result[0]
    for a, b, old, current in zip(expected['samples'].unbind(), result['samples'].unbind(), before, high['samples'].unbind()):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
        assert torch.equal(old, current)


@pytest.mark.parametrize('kind', ['sigma', 'shape', 'nan'])
def test_actual_invalid_boundary_or_audio_still_fails(kind):
    high = latent()
    v, a = high['samples'].unbind()
    coarse = {'samples': NestedTensor((v, a.clone()))}
    end = .5
    if kind == 'sigma':
        end = .6
    if kind == 'shape':
        coarse['samples'] = NestedTensor((v, a[..., :1, :]))
    if kind == 'nan':
        coarse['samples'].unbind()[1].fill_(float('nan'))
    with pytest.raises(ValueError):
        continue_audio(coarse, high, RandomNoise.execute(1).result[0],
                       torch.tensor([1., end]), torch.tensor([.5, 0.]))
