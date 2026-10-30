"""Actual tiny Core LCM/FLOW_AV contracts, independent of trained quality."""
import asyncio
import copy
import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

import comfy.nested_tensor
from comfy.k_diffusion.sampling import sample_lcm
from comfy_extras.nodes_custom_sampler import BasicGuider, RandomNoise, SamplerCustomAdvanced
from h3_audio_t8_pkg.h3_lora_compat_advanced import load_minimax_h3_lora_model
from h3_audio_t8_pkg.nodes_pika_flex import MiniMaxH3PikaFlexSetupEXPT8
from h3_audio_t8_pkg.pika_flex import pika_sigmas, prepare_pika_flex
from test_fast_h3_v2_core_sampler import model
from test_progressive_sampling_runtime import conditioning, latent


def run(selected, sampler, sigmas, source, positive=None, seed=301030):
    noise = RandomNoise.execute(seed).result[0]
    guider = BasicGuider.execute(selected, positive or conditioning()).result[0]
    return SamplerCustomAdvanced.execute(noise, guider, sampler, sigmas, source).result[0]


def test_author_knots_have_one_shift_and_not_the_generic_six_step_grid():
    expected = torch.tensor([1, 36 / 37, 12 / 13, 4 / 5, 4 / 7, 1 / 3, 0])
    torch.testing.assert_close(pika_sigmas(), expected, rtol=0, atol=0)
    from h3_audio_t8_pkg.sampling import native_flow_sigmas
    assert not torch.equal(pika_sigmas(), native_flow_sigmas(6, 12))


def test_native_lcm_renoises_five_times_and_returns_the_last_clean_prediction():
    # An independent closed-form oracle for flow renoising: this is not an Euler
    # velocity update and not the sigma*sigma DDPM noise convention.
    class Flow:
        def noise_scaling(self, sigma, noise, clean):
            return sigma * noise + (1 - sigma) * clean
    class Oracle:
        inner_model = SimpleNamespace(model_patcher=SimpleNamespace(get_model_object=lambda _name: Flow()))
        def __init__(self):
            self.seen = []
        def __call__(self, x, sigma, **kwargs):
            self.seen.append(x.clone())
            return torch.full_like(x, len(self.seen) / 10)
    oracle, noises = Oracle(), []
    def next_noise(sigma, following):
        noises.append((float(sigma), float(following)))
        return torch.full((1, 3), len(noises) / 7)
    sigmas = pika_sigmas()
    result = sample_lcm(oracle, torch.ones(1, 3), sigmas, noise_sampler=next_noise, disable=True)
    assert len(oracle.seen) == 6 and len(noises) == 5
    for index in range(1, 6):
        expected = sigmas[index] * (index / 7) + (1 - sigmas[index]) * (index / 10)
        torch.testing.assert_close(oracle.seen[index], torch.full((1, 3), expected))
    torch.testing.assert_close(result, torch.full((1, 3), .6))


@pytest.mark.parametrize('task', ['t2va', 'i2va', 'fl2va', 'ref2va'])
def test_actual_native_six_forwards_preserve_conditioning_latent_and_bare_model(task):
    bare, source = model(), latent()
    positive = conditioning('i2va' if task in {'i2va', 'fl2va'} else 't2va')
    if task == 'fl2va':
        positive[0][1]['minimax_keyframes'].append({
            'resolved_frame_index': 4, 'latent': torch.ones(1, 24, 1, 4, 8) * .3})
    if task == 'ref2va':
        positive[0][1]['minimax_refs'] = [{'kind': 'image', 'latent_t': 1, 'latent_h': 4, 'latent_w': 8,
                                        'ref_audio_t': 0, 'latent': torch.ones(1, 24, 1, 4, 8) * .4}]
    before, original = copy.deepcopy(positive), copy.deepcopy(bare.model_options)
    selected, sampler, sigmas, report = prepare_pika_flex(bare, source)
    network, times = bare.model.diffusion_model, []
    def observe(_owner, args, kwargs):
        times.append(kwargs.get('timestep', args[1] if len(args) > 1 else None).clone())
    handle = network.register_forward_pre_hook(observe, with_kwargs=True)
    try:
        output = run(selected, sampler, sigmas, source, positive)
    finally:
        handle.remove()
    assert len(times) == 6
    torch.testing.assert_close(torch.stack(times).flatten(), sigmas[:-1] * 1000)
    assert all(torch.isfinite(x).all() for x in output['samples'].unbind())
    assert all(not x.count_nonzero() for x in source['samples'].unbind())
    assert bare.model_options == original and not bare.object_patches
    assert positive[0][1].keys() == before[0][1].keys()
    assert torch.equal(positive[0][0], before[0][0])
    for kind in ('minimax_keyframes', 'minimax_refs'):
        for old, new in zip(before[0][1].get(kind, []), positive[0][1].get(kind, [])):
            assert torch.equal(old['latent'], new['latent'])
    sampling = selected.get_model_object('model_sampling')
    assert sampling.shift == 12 and sampling.audio_shift == 3 and sampling.audio_scale == 4
    assert json.loads(report)['trained_GPU_quality_verified'] is False
    assert not torch.cuda.is_initialized()


def test_seed_replay_is_exact_and_another_seed_changes_both_streams():
    bare, source = model(), latent()
    selected, sampler, sigmas, _ = prepare_pika_flex(bare, source)
    a, b, c = [run(selected, sampler, sigmas, source, seed=seed)['samples'].unbind()
               for seed in (13, 13, 14)]
    for first, repeated, different in zip(a, b, c):
        assert torch.equal(first, repeated) and not torch.equal(first, different)


def test_actual_core_lcm_keeps_locked_source_audio_and_uses_both_native_clocks():
    bare, source = model(), latent()
    video, audio = source['samples'].unbind()
    audio = torch.linspace(-.1, .1, audio.numel()).reshape(audio.shape)
    source['samples'] = comfy.nested_tensor.NestedTensor((video, audio))
    source['noise_mask'] = comfy.nested_tensor.NestedTensor((torch.ones_like(video), torch.zeros_like(audio)))
    selected, sampler, sigmas, _ = prepare_pika_flex(bare, source)
    times = []
    handle = bare.model.diffusion_model.time_embedder.register_forward_pre_hook(
        lambda _owner, args: times.append(args[0].clone()))
    try:
        output = run(selected, sampler, sigmas, source)
    finally:
        handle.remove()
    torch.testing.assert_close(output['samples'].unbind()[1], audio, rtol=0, atol=3e-5)
    assert len(times) == 6
    for row, sigma in zip(times, sigmas[:-1]):
        audio_sigma = sigma / (4 - 3 * sigma)
        assert torch.isclose(row, 1 - sigma, atol=2e-6).any()
        assert torch.isclose(row, 1 - audio_sigma, atol=2e-6).any()
    assert not source['noise_mask'].unbind()[1].count_nonzero()


@pytest.mark.parametrize('fail', [False, True])
def test_user_forward_is_retained_and_real_error_propagates(fail):
    bare, source = model(), latent()
    selected_user, calls = bare.clone(), []
    def user(executor, *args, **kwargs):
        calls.append(1)
        if fail:
            raise RuntimeError('user kernel actually failed')
        return executor(*args, **kwargs)
    selected_user.add_wrapper_with_key('diffusion_model', 'unknown-user', user)
    selected, sampler, sigmas, _ = prepare_pika_flex(selected_user, source)
    assert selected.get_wrappers('diffusion_model', 'unknown-user') == [user]
    if fail:
        with pytest.raises(RuntimeError, match='user kernel actually failed'):
            run(selected, sampler, sigmas, source)
        assert len(calls) == 1
    else:
        run(selected, sampler, sigmas, source)
        assert len(calls) == 6
    assert selected_user.get_wrappers('diffusion_model', 'unknown-user') == [user]
    assert not bare.wrappers


@pytest.mark.parametrize('strength', [.35, 1.])
def test_native_fused_lora_is_kept_through_setup_and_changes_actual_forward(tmp_path, strength):
    bare, source = model(), latent()
    generator = torch.Generator().manual_seed(7)
    projection = bare.model.diffusion_model.blocks[0].attn.qkv_proj
    prefix = 'diffusion_model.blocks.0.attn.qkv_proj'
    weights = {prefix + '.lora_A.weight': torch.randn(384, projection.weight.shape[1], generator=generator) * .02,
               prefix + '.lora_B.weight': torch.randn(projection.weight.shape[0], 384, generator=generator) * .02}
    path = tmp_path / 'user-chosen-name.safetensors'
    save_file(weights, str(path), metadata={'converted_layout': 'comfyui_minimax_h3', 'baked_scale': '1.0'})
    adapted, raw = load_minimax_h3_lora_model(bare, path, strength)
    selected, sampler, sigmas, _ = prepare_pika_flex(adapted, source)
    assert selected.patches == adapted.patches and selected.patches
    assert json.loads(raw)['selected_strength'] == strength
    result = run(selected, sampler, sigmas, source)['samples'].unbind()
    plain, ps, sg, _ = prepare_pika_flex(bare, source)
    reference = run(plain, ps, sg, source)['samples'].unbind()
    assert any(not torch.equal(a, b) for a, b in zip(result, reference))
    assert not bare.patches


def test_real_channel_error_is_not_hidden_as_an_unverified_model_warning():
    source = latent()
    source['samples'] = comfy.nested_tensor.NestedTensor((torch.zeros(1, 23, 2, 4, 8), source['samples'].unbind()[1]))
    with pytest.raises(ValueError, match='AV channels'):
        prepare_pika_flex(model(), source)


def test_schema_and_original_published_registration_prefix_are_preserved():
    import h3_audio_t8_pkg as package
    from pathlib import Path
    info = MiniMaxH3PikaFlexSetupEXPT8.GET_NODE_INFO_V1()
    assert info['output'] == ['MODEL', 'SAMPLER', 'SIGMAS', 'STRING']
    actual = [node.define_schema().node_id for node in asyncio.run(package.comfy_entrypoint().get_node_list())]
    expected = json.loads((Path(__file__).parents[1] / 'features.json').read_text('utf-8'))['nodes']
    # Metadata may append this node as development proceeds; the released prefix
    # remains an exact ordered comparison, not a recaptured baseline.
    assert actual[:778] == expected[:778]
    assert actual[778:] == [MiniMaxH3PikaFlexSetupEXPT8.__name__, 'MiniMaxH3PikaTwoPassSamplerEXPT8']
