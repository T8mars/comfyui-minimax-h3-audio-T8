"""Real tiny Core verifies native-clock curve injection and clone restoration."""
import json
import copy

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
import comfy.lora

from h3_audio_t8_pkg import open_regenerate_recipe as recipe
from h3_audio_t8_pkg.open_regenerate import prepare_refine
from test_fast_h3_v2_core_sampler import model, sample
from test_open_regenerate import SilenceVAE
from test_progressive_sampling_runtime import latent
from h3_audio_t8_pkg.sampling import setup_dual_clock_sampling
import comfy.nested_tensor
import comfy.ops
from comfy.ldm.minimax.model import AdalnProj


def test_recipe_nested_model_options_match_portable_workflow_paths(monkeypatch):
    from h3_audio_t8_pkg import nodes_open_regenerate as node_module
    monkeypatch.setattr(node_module.folder_paths, 'get_filename_list', lambda _folder: [
        'Other\\custom.safetensors', 'OpenRegenerateAuthor\\sharp_curve_branches.safetensors'])
    node = node_module.MiniMaxH3OpenRegenerateRecipeEXPT8.GET_NODE_INFO_V1()
    choices = node['input']['required']['curve_lora_name'][1]['options']
    assert choices[0] == 'OpenRegenerateAuthor/sharp_curve_branches.safetensors'
    assert 'Other/custom.safetensors' in choices


def factors(network, seed=28):
    generator = torch.Generator().manual_seed(seed)
    a = torch.randn(3, 8, generator=generator) * .01
    b = torch.randn(network.blocks[0].adaln_proj.linear.out_features, 3, generator=generator) * .01
    table = torch.randn(17, 8, generator=generator) * .01
    return table, {'blocks.0.adaln_proj.linear': (a, b)}


def test_curve_interpolation_matches_author_formula_at_endpoints_and_fractional_times():
    table = torch.arange(17 * 8).reshape(17, 8).float()
    times = torch.tensor([0., .090625, 1 / 7, 1 / 3, .999, 1.])
    position = times * 16
    lower = position.floor().long().clamp(max=15)
    expected = torch.lerp(table[lower], table[lower + 1], (position - lower).unsqueeze(-1))
    torch.testing.assert_close(recipe.interpolate_curve(table, times), expected, rtol=0, atol=0)


@pytest.mark.parametrize('strength', [0., .4, 1., -1.])
def test_real_core_curve_uses_actual_clock_and_does_not_leak_into_bare_model(strength):
    bare = model()
    network = bare.model.diffusion_model
    table, branches = factors(network)
    previous = network.blocks[0].adaln_proj.linear.forward
    source = {'samples': latent()['samples'].unbind()[0]}
    plain, plain_sampler, plain_sigmas, plain_av, _ = prepare_refine(bare, source, SilenceVAE())
    baseline = sample(plain, plain_sampler, plain_sigmas, plain_av)[0]
    selected, runtime = recipe.attach_curves(bare, table, branches, strength)
    observed = []
    # Record the actual unmodified native time embedder arguments, not a
    # second expected-clock implementation.
    handle = network.time_embedder.register_forward_pre_hook(lambda _m, args: observed.append(args[0].clone()))
    try:
        prepared, sampler, sigmas, av, _ = prepare_refine(selected, source, SilenceVAE())
        output = sample(prepared, sampler, sigmas, av)[0]
    finally:
        handle.remove()
    assert runtime.completed_forwards == 3 and runtime.failed_forwards == 0
    assert len(observed) == 3
    for actual, report in zip(observed, runtime.forwards):
        torch.testing.assert_close(actual, torch.tensor(report['times']), rtol=0, atol=0)
    if strength:
        assert runtime.calls == {'blocks.0.adaln_proj.linear': 3}
        assert runtime.report()['all_selected_curve_targets_observed']
    assert all(torch.isfinite(value).all() for value in output['samples'].unbind())
    assert not bare.wrappers and not bare.object_patches
    assert recipe.KEY not in bare.attachments and recipe.ACTIVE.get() == {}
    # Core retains the active clone between sampling calls. Reusing the bare
    # branch must restore its actual callable and reproduce its prior output.
    before_reuse = dict(runtime.calls)
    repeated = sample(plain, plain_sampler, plain_sigmas, plain_av)[0]
    for actual, expected in zip(repeated['samples'].unbind(), baseline['samples'].unbind()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert runtime.calls == before_reuse and runtime.completed_forwards == 3
    assert network.blocks[0].adaln_proj.linear.forward == previous
    assert not torch.cuda.is_initialized()


def test_branch_numeric_order_matches_author_and_exception_releases_context():
    bare = model()
    table, branches = factors(bare.model.diffusion_model)
    selected, runtime = recipe.attach_curves(bare, table, branches, .7)
    name = 'blocks.0.adaln_proj.linear'
    times = torch.tensor([.09, 1.])
    coordinates = recipe.interpolate_curve(table, times)
    value = torch.ones(2, 24)
    original = bare.model.diffusion_model.blocks[0].adaln_proj.linear
    frame = {'coordinates': coordinates, 'seen': set(), 'native_embedding': None}
    token = recipe.ACTIVE.set({runtime.owner: frame})
    try:
        actual = runtime.apply_delta(value, original(value), name)
        a, b = branches[name]
        expected = original(value) + torch.nn.functional.linear(torch.nn.functional.linear(coordinates, a), b) * .7
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        recipe.ACTIVE.reset(token)
    with pytest.raises(ValueError, match='paired native diffusion wrapper'):
        runtime.apply_delta(value, original(value), name)
    assert recipe.ACTIVE.get() == {}


def test_unknown_compatible_projection_class_still_delegates_and_real_shape_error_is_reported():
    bare = model()
    layer = bare.model.diffusion_model.blocks[0].adaln_proj.linear
    calls = []
    class UserLinear(torch.nn.Linear):
        def forward(self, value):
            calls.append(value.shape)
            return super().forward(value)
    custom = UserLinear(layer.in_features, layer.out_features)
    custom.load_state_dict(layer.state_dict())
    bare.model.diffusion_model.blocks[0].adaln_proj.linear = custom
    table, branches = factors(bare.model.diffusion_model)
    selected, runtime = recipe.attach_curves(bare, table, branches)
    a, b = branches['blocks.0.adaln_proj.linear']
    with pytest.raises(ValueError, match='projection rows'):
        recipe.attach_curves(bare, table, {'blocks.0.adaln_proj.linear': (a, b[:-1])})
    assert runtime.completed_forwards == 0
    _, report = recipe.audit_recipe(selected, latent())
    assert json.loads(report)['status'] == 'unobserved'
    prepared, sampler, sigmas, av, _ = prepare_refine(selected,
        {'samples': latent()['samples'].unbind()[0]}, SilenceVAE())
    sample(prepared, sampler, sigmas, av)
    assert len(calls) == runtime.completed_forwards == 3


@pytest.mark.parametrize('user_scale', [1., 1.125])
def test_pruned_model_uses_selected_sharp_table_not_its_own_distinct_curve(user_scale):
    bare = model()
    network = bare.model.diffusion_model
    table, branches = factors(network)
    del network.time_embedder
    network.use_adaln_curves = True
    native_table = torch.linspace(-.1, .1, 17 * 8).reshape(17, 8)
    network.register_buffer('adaln_t_table', native_table)
    network.blocks[0].adaln_proj = AdalnProj(8, 24, 6, 3, apply_silu=False,
        dtype=torch.float32, operations=comfy.ops.disable_weight_init)
    network.final_layer.adaln_proj = AdalnProj(8, 24, 2, 1, apply_silu=False,
        dtype=torch.float32, operations=comfy.ops.disable_weight_init)
    with torch.no_grad():
        for part in [network.blocks[0].adaln_proj, network.final_layer.adaln_proj]:
            for parameter in part.parameters():
                parameter.fill_(.01)
    selected, runtime = recipe.attach_curves(bare, table, branches)
    observed = []
    linear = network.blocks[0].adaln_proj.linear
    user_hook = linear.register_forward_hook(lambda _module, _args, result: result * user_scale)
    # Observe the actual modulation consumed by the parent, after every
    # selected linear hook, instead of observing an earlier hook's input.
    handle = network.blocks[0].adaln_proj.register_forward_hook(lambda _module, args, result:
        observed.append((args[0].clone(), torch.cat(result, dim=-1).reshape(args[0].shape[0], -1).clone())))
    try:
        prepared, sampler, sigmas, av, _ = prepare_refine(selected,
            {'samples': latent()['samples'].unbind()[0]}, SilenceVAE())
        sample(prepared, sampler, sigmas, av)
    finally:
        handle.remove()
        user_hook.remove()
    assert runtime.completed_forwards == len(observed) == 3
    a, b = branches['blocks.0.adaln_proj.linear']
    for (actual_embedding, actual_output), forward in zip(observed, runtime.forwards):
        times = torch.tensor(forward['times'])
        torch.testing.assert_close(actual_embedding, recipe.interpolate_curve(native_table, times), rtol=0, atol=0)
        source_coordinates = recipe.interpolate_curve(table, times)
        expected = F.linear(actual_embedding, linear.weight, linear.bias) * user_scale + F.linear(F.linear(source_coordinates, a), b)
        torch.testing.assert_close(actual_output, expected, rtol=0, atol=0)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('shifts', [(1., 1.), (12., 3.)])
def test_fractional_video_and_audio_masks_use_the_actual_native_unique_clock_order(shifts):
    bare = model()
    table, branches = factors(bare.model.diffusion_model)
    selected, runtime = recipe.attach_curves(bare, table, branches)
    source = latent()
    video, audio = source['samples'].unbind()
    video_mask = torch.ones_like(video[:, :1])
    video_mask[..., :2, :2] = .25
    video_mask[..., 2:, 2:4] = .7
    audio_mask = torch.ones_like(audio[:, :1])
    audio_mask[..., :3] = 0
    audio_mask[..., 3:5] = .4
    source['noise_mask'] = comfy.nested_tensor.NestedTensor((video_mask, audio_mask))
    observed = []
    handle = bare.model.diffusion_model.time_embedder.register_forward_pre_hook(
        lambda _module, args: observed.append(args[0].clone()))
    try:
        prepared, sampler, sigmas = setup_dual_clock_sampling(selected, source, 3,
            shifts[0], shifts[1], sampler_name='euler')
        sample(prepared, sampler, sigmas, source)
    finally:
        handle.remove()
    assert len(observed) == runtime.completed_forwards == 3
    for actual, forward in zip(observed, runtime.forwards):
        torch.testing.assert_close(actual, torch.tensor(forward['times']), rtol=0, atol=0)
    assert recipe.ACTIVE.get() == {}


def test_later_user_timestep_wrapper_is_observed_at_actual_native_embedding():
    bare = model()
    table, branches = factors(bare.model.diffusion_model)
    selected, runtime = recipe.attach_curves(bare, table, branches)
    def user_clock(executor, x, timestep, context, transformer_options=None, **kwargs):
        return executor(x, timestep * .8, context, transformer_options, **kwargs)
    from comfy.patcher_extension import WrappersMP
    selected.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, 'user-later-clock', user_clock)
    actual = []
    hook = bare.model.diffusion_model.time_embedder.register_forward_pre_hook(
        lambda _m, args: actual.append(args[0].clone()))
    try:
        prepared, sampler, sigmas, av, _ = prepare_refine(selected,
            {'samples': latent()['samples'].unbind()[0]}, SilenceVAE())
        sample(prepared, sampler, sigmas, av)
    finally:
        hook.remove()
    assert len(actual) == runtime.completed_forwards == 3
    for observed, forward in zip(actual, runtime.forwards):
        torch.testing.assert_close(observed, torch.tensor(forward['times']), rtol=0, atol=0)
        assert forward['clock_alignment_verified']


def test_author_merged_namespace_native_parser_keeps_baked_turbo_scale_and_all_curve_factors(tmp_path):
    bare = model()
    network = bare.model.diffusion_model
    generator = torch.Generator().manual_seed(2910)
    raw = {}
    deltas = []
    rank, hidden, inner = 4, 24, 384
    for projection in ['q', 'k', 'v']:
        a = torch.randn(rank, hidden, generator=generator) * .002
        b = torch.randn(inner, rank, generator=generator) * .002
        # Three already-folded component slices: refiner2,Turbo1,Sharp1.
        b[:, 2:3] *= .0625
        stem = 'transformer.transformer_blocks.0.attn.to_' + projection
        raw[stem + '.lora_A.weight'] = a
        raw[stem + '.lora_B.weight'] = b
        deltas.append(b @ a)
    linear_file, curve_file, table_file = [tmp_path / name for name in ['merged.safetensors', 'curves.safetensors', 'source.safetensors']]
    save_file(raw, str(linear_file), metadata={'application': 'scale1,Turbo alpha/rank already folded in B'})
    table, branches = factors(network)
    a, b = branches['blocks.0.adaln_proj.linear']
    stem = 'transformer.transformer_blocks.0.adaln_proj.linear'
    save_file({stem + '.curve_A.weight': a, stem + '.curve_B.weight': b}, str(curve_file))
    save_file({'adaln_t_table': table}, str(table_file))
    source_sha = recipe.file_sha(linear_file)
    selected, text = recipe.load_recipe(bare, linear_file, curve_file, table_file, .4, .7)
    key = 'diffusion_model.blocks.0.attn.qkv_proj.weight'
    old = network.blocks[0].attn.qkv_proj.weight.detach().clone()
    actual = comfy.lora.calculate_weight(selected.patches[key], old.clone(), key)
    expected = old + torch.cat(deltas, dim=0) * .4
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-8)
    report = json.loads(text)
    assert report['linear_source_tensors'] == 6 and report['linear_registered_targets'] == 1
    assert report['curve_strength'] == .7 and report['curve_target_count'] == 1
    assert report['completed_model_forwards'] == 0 and not report['all_selected_curve_targets_observed']
    assert recipe.file_sha(linear_file) == source_sha


def test_two_selected_curve_owners_compose_without_leaking_or_replacing_each_other():
    bare = model()
    table1, branch1 = factors(bare.model.diffusion_model, 1)
    first, one = recipe.attach_curves(bare, table1, branch1, .4)
    table2, branch2 = factors(bare.model.diffusion_model, 2)
    second, two = recipe.attach_curves(first, table2, branch2, -.2)
    prepared, sampler, sigmas, av, _ = prepare_refine(second,
        {'samples': latent()['samples'].unbind()[0]}, SilenceVAE())
    sample(prepared, sampler, sigmas, av)
    assert one.completed_forwards == two.completed_forwards == 3
    assert one.calls == two.calls == {'blocks.0.adaln_proj.linear': 3}
    assert len(second.get_attachment(recipe.KEY)) == 2 and recipe.ACTIVE.get() == {}
    assert not bare.object_patches and not bare.wrappers


def test_native_forward_failure_releases_clock_context_and_bare_model_is_reusable():
    bare = model()
    source = {'samples': latent()['samples'].unbind()[0]}
    plain, sampler, sigmas, av, _ = prepare_refine(bare, source, SilenceVAE())
    before = sample(plain, sampler, sigmas, av)[0]
    table, branches = factors(bare.model.diffusion_model)
    selected, runtime = recipe.attach_curves(bare, table, branches)
    prepared, rsampler, rsigmas, rav, _ = prepare_refine(selected, source, SilenceVAE())
    def fail(_module, _args, _result):
        raise RuntimeError('Injected actual linear failure')
    hook = bare.model.diffusion_model.blocks[0].adaln_proj.linear.register_forward_hook(fail)
    try:
        with pytest.raises(RuntimeError, match='Injected actual linear failure'):
            sample(prepared, rsampler, rsigmas, rav)
    finally:
        hook.remove()
    assert runtime.completed_forwards == 0 and runtime.failed_forwards == 1
    assert recipe.ACTIVE.get() == {}
    repeated = sample(plain, sampler, sigmas, av)[0]
    for actual, expected in zip(repeated['samples'].unbind(), before['samples'].unbind()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize('parent', ['linear', 'adaln_proj', 'block'])
def test_selected_parent_linear_replacement_is_actually_delegated_before_any_sampling(parent):
    bare = model()
    network = bare.model.diffusion_model
    original_layer = network.blocks[0].adaln_proj.linear
    original_forward = original_layer.forward
    source_parent = {'linear': original_layer, 'adaln_proj': network.blocks[0].adaln_proj,
                     'block': network.blocks[0]}[parent]
    replacement = copy.deepcopy(source_parent)
    layer = replacement if parent == 'linear' else replacement.linear if parent == 'adaln_proj' else replacement.adaln_proj.linear
    calls = []
    previous = layer.forward

    def user_forward(value):
        calls.append(1)
        return previous(value) * 1.125

    layer.forward = user_forward
    user_model = bare.clone()
    path = {'linear': 'diffusion_model.blocks.0.adaln_proj.linear',
            'adaln_proj': 'diffusion_model.blocks.0.adaln_proj', 'block': 'diffusion_model.blocks.0'}[parent]
    user_model.add_object_patch(path, replacement)
    table, branches = factors(network)
    selected, runtime = recipe.attach_curves(user_model, table, branches, strength=0.)
    prepared, sampler, sigmas, av, _ = prepare_refine(selected,
        {'samples': latent()['samples'].unbind()[0]}, SilenceVAE())
    sample(prepared, sampler, sigmas, av)
    assert len(calls) == 3
    assert runtime.completed_forwards == 3
    assert not layer._forward_hooks
    assert not network.time_embedder._forward_pre_hooks
    plain, plain_sampler, plain_sigmas, plain_av, _ = prepare_refine(bare,
        {'samples': latent()['samples'].unbind()[0]}, SilenceVAE())
    sample(plain, plain_sampler, plain_sigmas, plain_av)
    assert network.blocks[0].adaln_proj.linear is original_layer
    assert original_layer.forward == original_forward and len(calls) == 3
