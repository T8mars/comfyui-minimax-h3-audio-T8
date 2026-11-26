"""Real Core CPU consumers; tiny random networks never qualify trained quality."""
import asyncio
import copy
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

import comfy.model_patcher
import comfy.samplers
import comfy.utils
import folder_paths
from comfy_extras.nodes_custom_sampler import BasicGuider, RandomNoise, SamplerCustomAdvanced
from h3_audio_t8_pkg import comfy_entrypoint
from h3_audio_t8_pkg import sulphur_corrector as adapter
from h3_audio_t8_pkg.nodes_sulphur_corrector import MiniMaxH3CorrectorApplyT8, MiniMaxH3CorrectorLoaderT8
from h3_audio_t8_pkg.sulphur_corrector_network import CorrectorConfig, H3Corrector, bound_residual
from h3_audio_t8_pkg.sampling import native_flow_sigmas, setup_dual_clock_sampling
from test_fast_h3_v2_core_sampler import model
from test_progressive_sampling_runtime import conditioning, latent


def tiny_corrector():
    torch.manual_seed(261126)
    config = CorrectorConfig(text_dim=8, width=16, depth=2, heads=2, head_dim=8,
                             mlp_ratio=2, time_embed_dim=8)
    network = H3Corrector(config).eval().requires_grad_(False)
    patcher = comfy.model_patcher.CoreModelPatcher(network, torch.device('cpu'), torch.device('cpu'))
    return {'network': network, 'patcher': patcher, 'name': 'arbitrary-user-name'}


def run(selected, source=None, sigmas=None, task='t2va', seed=42, sampler='euler'):
    source = source if source is not None else latent()
    selected, _, _ = setup_dual_clock_sampling(selected, source, 4, 12, 3, sampler_name=sampler)
    noise = RandomNoise.execute(seed).result[0]
    guider = BasicGuider.execute(selected, conditioning(task)).result[0]
    schedule = native_flow_sigmas(4, 12) if sigmas is None else sigmas
    return SamplerCustomAdvanced.execute(noise, guider, comfy.samplers.ksampler(sampler), schedule, source).result


def packed_args(corrector, sigma=.5, schedule=None):
    video = torch.linspace(-1, 1, 24 * 2 * 4 * 8).reshape(1, 24, 2, 4, 8)
    audio = torch.linspace(-.3, .2, 32 * 2 * 8).reshape(1, 32, 2, 8)
    noisy, shapes = comfy.utils.pack_latents([video, audio])
    prediction, _ = comfy.utils.pack_latents([video * .2, audio * .73])
    return dict(denoised=prediction, input=noisy, sigma=torch.tensor([sigma]),
                model=SimpleNamespace(latent_shapes=shapes),
                model_options={'transformer_options': {'sample_sigmas': torch.tensor([1., .5, 0.])
                                                        if schedule is None else schedule}},
                cond=[{'cross_attn': torch.zeros(1, 3, corrector['network'].config.text_dim)}])


def hook(corrector, **kwargs):
    selected, _ = adapter.apply_corrector(model(), corrector, **kwargs)
    return selected.model_options['sampler_post_cfg_function'][-1]


def test_loader_reads_metadata_exact_tensors_and_uses_fp32_even_under_global_bfloat16(tmp_path, monkeypatch):
    chosen = tiny_corrector()
    name = 'my-renamed-corrector.safetensors'
    save_file({k: v.half() for k, v in chosen['network'].state_dict().items()}, str(tmp_path / name),
              metadata=chosen['network'].config.to_metadata())
    monkeypatch.setitem(folder_paths.folder_names_and_paths, 'h3_correctors', ([str(tmp_path)], {'.safetensors'}))
    default = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        loaded, raw = adapter.load_corrector(name)
    finally:
        torch.set_default_dtype(default)
    assert loaded['name'] == Path(name).stem
    assert all(p.dtype == torch.float32 and not p.requires_grad for p in loaded['network'].parameters())
    for key, value in chosen['network'].state_dict().items():
        assert torch.equal(loaded['network'].state_dict()[key], value.half().float())
    assert json.loads(raw)['status'] == 'loaded_not_sampled'
    assert not torch.cuda.is_initialized()


def test_corrector_folder_retains_external_paths_and_extensions(tmp_path, monkeypatch):
    monkeypatch.setitem(folder_paths.folder_names_and_paths, 'h3_correctors', ([str(tmp_path)], {'.custom'}))
    adapter.register_corrector_folder()
    paths, extensions = folder_paths.folder_names_and_paths['h3_correctors']
    assert paths[0] == str(tmp_path)
    assert {'.custom', '.safetensors'} <= extensions


def test_loader_lists_and_loads_correctors_from_both_folders(tmp_path, monkeypatch):
    legacy, loras, extra = (tmp_path / name for name in ('h3_correctors', 'loras', 'extra_loras'))
    for folder in (legacy, loras, extra / 'nested'):
        folder.mkdir(parents=True)
    monkeypatch.setattr(folder_paths, 'models_dir', str(tmp_path))
    monkeypatch.setattr(folder_paths, 'filename_list_cache', {})
    monkeypatch.setitem(folder_paths.folder_names_and_paths, 'h3_correctors', ([str(legacy)], {'.safetensors'}))
    monkeypatch.setitem(folder_paths.folder_names_and_paths, 'loras', ([str(loras), str(extra)], {'.safetensors'}))

    network = tiny_corrector()['network']
    weights = network.state_dict()
    for file, value in [(legacy/'legacy.safetensors', 1.), (legacy/'shared.safetensors', 2.),
                        (loras/'shared.safetensors', 3.), (loras/'renamed.safetensors', 4.),
                        (extra/'nested/extra.safetensors', 5.)]:
        weights['patch_out.bias'].fill_(value)
        save_file(weights, str(file), metadata=network.config.to_metadata())
    save_file({'lora.weight': torch.ones(1)}, str(loras/'ordinary.safetensors'))
    (loras/'broken.safetensors').write_bytes(b'not a safetensors file')
    choices = MiniMaxH3CorrectorLoaderT8.GET_NODE_INFO_V1()['input']['required']['corrector'][1]['options']
    assert {Path(name).as_posix() for name in choices} == {
        'legacy.safetensors', 'shared.safetensors', 'renamed.safetensors', 'nested/extra.safetensors'}
    for filename, value in [('legacy.safetensors', 1.), ('shared.safetensors', 2.),
                            ('renamed.safetensors', 4.), ('nested/extra.safetensors', 5.)]:
        selected = MiniMaxH3CorrectorLoaderT8.execute(filename).result[0]
        assert torch.equal(selected['network'].patch_out.bias, torch.full_like(weights['patch_out.bias'], value))
    with pytest.raises(ValueError, match='architecture metadata'):
        MiniMaxH3CorrectorLoaderT8.execute('ordinary.safetensors')
    (legacy/'shared.safetensors').unlink()
    selected = MiniMaxH3CorrectorLoaderT8.execute('shared.safetensors').result[0]
    assert torch.equal(selected['network'].patch_out.bias, torch.full_like(weights['patch_out.bias'], 3.))


@pytest.mark.parametrize('fault', ['metadata', 'missing_tensor', 'wrong_shape'])
def test_actual_invalid_weight_files_fail(tmp_path, monkeypatch, fault):
    chosen = tiny_corrector()
    weights = chosen['network'].state_dict()
    metadata = chosen['network'].config.to_metadata()
    if fault == 'metadata':
        metadata = {}
    elif fault == 'missing_tensor':
        weights.pop('patch_out.bias')
    else:
        weights['patch_out.bias'] = torch.zeros(1)
    save_file(weights, str(tmp_path / 'invalid.safetensors'), metadata=metadata)
    monkeypatch.setitem(folder_paths.folder_names_and_paths, 'h3_correctors', ([str(tmp_path)], {'.safetensors'}))
    with pytest.raises((ValueError, RuntimeError)):
        adapter.load_corrector('invalid.safetensors')


@pytest.mark.parametrize('batch', [1, 2])
def test_post_cfg_matches_independent_velocity_formula_and_preserves_all_audio_bits(batch):
    chosen = tiny_corrector()
    args = packed_args(chosen)
    if batch == 2:
        for key in ('input', 'denoised'):
            args[key] = args[key].expand(batch, -1, -1).contiguous()
    video = comfy.utils.unpack_latents(args['input'], args['model'].latent_shapes)[0]
    prediction = comfy.utils.unpack_latents(args['denoised'], args['model'].latent_shapes)[0]
    velocity = (prediction - video) / .5
    with torch.no_grad():
        raw = chosen['network'](video, velocity, .5, 0., torch.zeros(batch, 3, 8))
    budget = velocity.square().flatten(1).mean(1).sqrt().clamp_min(.001) * .5
    factor = (1 + raw.square().flatten(1).mean(1) / budget.square()).rsqrt()
    expected = prediction + .5 * 1.3 * raw * factor.reshape(batch, 1, 1, 1, 1)
    before = args['denoised'].clone()
    out = hook(chosen, strength=1.3)(args)
    n = video[0].numel()
    torch.testing.assert_close(out[:, :, :n].reshape_as(video), expected, rtol=2e-6, atol=2e-6)
    assert torch.equal(out[:, :, n:].view(torch.uint8), before[:, :, n:].contiguous().view(torch.uint8))
    assert torch.equal(args['denoised'], before)


@pytest.mark.parametrize('sampler', ['euler', 'heun', 'lcm'])
@pytest.mark.parametrize('task', ['t2va', 'i2va'])
def test_real_core_sampler_invokes_corrector_with_its_actual_sigma_pairs(sampler, task):
    chosen, bare = tiny_corrector(), model()
    seen = []
    handle = chosen['network'].register_forward_pre_hook(lambda _m, a: seen.append((a[2], a[3])))
    try:
        selected, _ = adapter.apply_corrector(bare, chosen)
        result = run(selected, sampler=sampler, task=task)[0]
    finally:
        handle.remove()
    assert len(seen) >= 4 and seen[-1][1] == 0
    assert all(a >= b >= 0 for a, b in seen)
    assert all(torch.isfinite(t).all() for t in result['samples'].unbind())
    assert not bare.additional_models and not bare.model_options.get('sampler_post_cfg_function')


def test_strength_zero_is_exact_bypass_and_does_not_attach_corrector_vram():
    bare, chosen = model(), tiny_corrector()
    selected, raw = adapter.apply_corrector(bare, chosen, strength=0.)
    assert not selected.additional_models and json.loads(raw)['status'] == 'disabled'
    a, b = run(bare)[0]['samples'].unbind(), run(selected)[0]['samples'].unbind()
    assert all(torch.equal(x, y) for x, y in zip(a, b))


def test_inclusive_step_window_missing_schedule_fallback_and_terminal_zero(caplog):
    chosen = tiny_corrector()
    fn = hook(chosen, start_step=1, end_step=1, sigmas=torch.tensor([1., .5, 0.]))
    args = packed_args(chosen)
    assert fn({**args, 'sigma': torch.tensor([1.])}) is args['denoised']
    assert fn({**args, 'sigma': torch.tensor([0.])}) is args['denoised']
    expected = fn(args)
    assert not torch.equal(expected, args['denoised'])
    assert torch.equal(fn({**args, 'model_options': {}}), expected)
    missing = hook(chosen)
    assert missing({**args, 'model_options': {}}) is args['denoised']
    assert 'skipped' in caplog.text


def test_CPU_forward_stays_fp32_inside_an_external_autocast_context():
    chosen = tiny_corrector()
    fn, args = hook(chosen), packed_args(chosen)
    expected = fn(args)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        actual = fn(args)
    assert torch.equal(actual, expected)


def test_existing_hooks_models_lora_and_unknown_forward_are_retained_and_delegated():
    bare, chosen = model(), tiny_corrector()
    existing, calls = bare.clone(), []
    existing.set_additional_models('h3_corrector', [chosen['patcher']])
    def external_cfg(args):
        calls.append('cfg')
        return args['denoised']
    def external_forward(executor, *args, **kwargs):
        calls.append('forward')
        return executor(*args, **kwargs)
    existing.set_model_sampler_post_cfg_function(external_cfg)
    existing.add_wrapper_with_key('diffusion_model', 'renamed-third-party', external_forward)
    original = existing.model_options.copy()
    selected, _ = adapter.apply_corrector(existing, chosen)
    selected, _ = adapter.apply_corrector(selected, chosen, start_step=1, end_step=1)
    assert selected.model_options['sampler_post_cfg_function'][0] is external_cfg
    assert selected.patches == existing.patches
    assert selected.get_additional_models_with_key('h3_corrector')[0].model is chosen['network']
    assert len(selected.get_additional_models()) == 3
    run(selected)
    assert calls.count('cfg') == calls.count('forward') == 4
    assert existing.model_options == original


@pytest.mark.parametrize('fault', ['caption', 'shape', 'sigma', 'schedule'])
def test_real_runtime_contract_errors_are_not_hidden(fault):
    chosen = tiny_corrector()
    args = packed_args(chosen)
    if fault == 'caption':
        args['cond'] = [{'cross_attn': torch.zeros(1, 2, 7)}]
    elif fault == 'shape':
        args['model'].latent_shapes = []
    elif fault == 'sigma':
        args['sigma'] = torch.tensor([float('nan')])
    else:
        args['model_options']['transformer_options']['sample_sigmas'] = torch.tensor([0., 1.])
    with pytest.raises(ValueError):
        hook(chosen)(args)


def test_application_does_not_swallow_a_real_network_error(monkeypatch):
    chosen = tiny_corrector()
    def fail(*args):
        raise TypeError('actual corrector kernel failure')
    monkeypatch.setattr(chosen['network'], 'forward', fail)
    with pytest.raises(TypeError, match='actual corrector kernel failure'):
        hook(chosen)(packed_args(chosen))


def test_real_native_lora_weights_survive_corrector_and_change_the_actual_forward(tmp_path):
    from h3_audio_t8_pkg.h3_lora_compat_advanced import load_minimax_h3_lora_model
    chosen, bare = tiny_corrector(), model()
    name = 'diffusion_model.blocks.0.attn.qkv_proj.weight'
    value = bare.model.state_dict()[name]
    prefix = name.removesuffix('.weight')
    path = tmp_path / 'user-chosen-style.safetensors'
    save_file({prefix + '.lora_A.weight': torch.randn(4, value.shape[1]) * .1,
               prefix + '.lora_B.weight': torch.randn(value.shape[0], 4) * .1}, str(path),
              metadata={'converted_layout': 'comfyui_minimax_h3', 'baked_scale': '1.0'})
    adapted, _ = load_minimax_h3_lora_model(bare, path, .7)
    selected, _ = adapter.apply_corrector(adapted, chosen)
    assert selected.patches == adapted.patches and selected.patches
    a = run(selected)[0]['samples'].unbind()
    baseline, _ = adapter.apply_corrector(bare, chosen)
    b = run(baseline)[0]['samples'].unbind()
    assert any(not torch.equal(x, y) for x, y in zip(a, b))
    assert not bare.patches


def test_example_prompt_validates_against_actual_core_and_t8_input_schemas(monkeypatch):
    import execution
    import nodes
    from comfy_extras import nodes_custom_sampler, nodes_minimax_h3
    from h3_audio_t8_pkg.nodes import MiniMaxH3AudioConditioningT8, MiniMaxH3AVDecodeT8
    api = json.loads((Path(__file__).resolve().parents[1] /
                      'examples/workflows/99-sulphur3/S3_Corrector_T2VA_EXP_API.json').read_text('utf-8'))
    classes = asyncio.run(comfy_entrypoint().get_node_list())
    for module in (nodes_custom_sampler, nodes_minimax_h3):
        extension = asyncio.run(module.comfy_entrypoint())
        classes += asyncio.run(extension.get_node_list())
    for cls in classes:
        monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, cls.define_schema().node_id, cls)
    filenames = {
        'diffusion_models': ['s3.safetensors'],
        'text_encoders': ['qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors'],
        'vae': ['minimax_h3_video_vae_fp16.safetensors', 'minimax_h3_audio_vae_fp32.safetensors'],
        'h3_correctors': ['s3_corrector.safetensors'],
    }
    original = folder_paths.get_filename_list
    monkeypatch.setattr(folder_paths, 'get_filename_list', lambda category:
                        filenames[category] if category in filenames else original(category))
    validation = asyncio.run(execution.validate_prompt('sulphur-example', api, None))
    assert validation[0] is True, validation
    assert set(validation[2]) == {'15'}


def test_network_residual_budget_is_per_video_and_handles_zero_velocity():
    raw = torch.ones(2, 24, 2, 4, 8)
    velocity = raw.clone()
    velocity[0] *= 100
    velocity[1] *= 0
    applied, ratio = bound_residual(raw, velocity, .5, .001)
    assert applied[0].square().mean().sqrt() > applied[1].square().mean().sqrt()
    assert float(applied[1].square().mean().sqrt()) <= .0005 + 2e-10
    assert torch.isfinite(ratio).all()


def test_actual_stage_completion_and_save_do_not_reject_the_new_callable(tmp_path):
    from h3_audio_t8_pkg.modular_sampling.fast_h3_v2 import build_stage
    from h3_audio_t8_pkg.modular_sampling.results import sample_stage
    from h3_audio_t8_pkg.modular_sampling.storage import save_stage, load_stage
    source = latent()
    selected, _ = adapter.apply_corrector(model(), tiny_corrector())
    prepared, sampler, sigmas, context, _ = build_stage(selected, source, 'low_0_4', 'dense_compat_exp')
    guider = BasicGuider.execute(prepared, conditioning()).result[0]
    noise = RandomNoise.execute(66).result[0]
    output, denoised, result, raw = sample_stage(noise, guider, sampler, sigmas, source, context)
    receipt = result.verify()
    # A callable with a separately loaded network cannot borrow an old portable
    # MODEL identity; execution and Stage storage must remain available.
    assert receipt['portable_identity'] is False
    path, digest, _ = save_stage(result, tmp_path, 'corrector-stage')
    assert (tmp_path / path).is_file() and len(digest) == 64
    with pytest.raises(ValueError, match='unverified executable identity'):
        load_stage(tmp_path, path, digest, context.stage)


def test_dual_resolution_passes_have_independent_schedules_and_audio_masks():
    chosen, bare = tiny_corrector(), model()
    low = {'samples': comfy.nested_tensor.NestedTensor((torch.zeros(1, 24, 2, 2, 4), torch.zeros(1, 32, 2, 8)))}
    high = latent()
    video, audio = high['samples'].unbind()
    audio = torch.randn_like(audio)
    high['samples'] = comfy.nested_tensor.NestedTensor((video, audio))
    high['noise_mask'] = comfy.nested_tensor.NestedTensor((torch.ones_like(video), torch.zeros_like(audio)))
    selected_low, _ = adapter.apply_corrector(bare, chosen)
    selected_high, _ = adapter.apply_corrector(bare, chosen)
    seen = []
    handle = chosen['network'].register_forward_pre_hook(lambda _m, a: seen.append((tuple(a[0].shape), a[2], a[3])))
    try:
        run(selected_low, low, native_flow_sigmas(3, 12))
        high_out = run(selected_high, high, native_flow_sigmas(2, 12))[0]
    finally:
        handle.remove()
    assert [v[0][-2:] for v in seen] == [(2, 4)] * 3 + [(4, 8)] * 2
    assert seen[2][-1] == seen[-1][-1] == 0
    torch.testing.assert_close(high_out['samples'].unbind()[1], audio, rtol=0, atol=3e-5)
    assert not high['noise_mask'].unbind()[1].count_nonzero()


def test_appended_node_schemas_do_not_reorder_the_804_existing_nodes():
    nodes = asyncio.run(comfy_entrypoint().get_node_list())
    names = [n.define_schema().node_id for n in nodes]
    assert len(names) == len(set(names)) == 806
    root = Path(__file__).resolve().parents[1]
    prefix = json.loads((root / 'features.json').read_text('utf-8'))['nodes']
    for fixture in ['nov12_sampling_source.json', 'nov16_audio_handoff_source.json']:
        prefix += json.loads((root / 'tests/fixtures' / fixture).read_text('utf-8'))['appended_nodes']
    assert len(prefix) == 804 and names[:804] == prefix
    assert names[-2:] == ['MiniMaxH3CorrectorLoaderT8', 'MiniMaxH3CorrectorApplyT8']
    assert MiniMaxH3CorrectorApplyT8.define_schema().inputs[0].id == 'model'


@pytest.mark.skipif(not os.environ.get('T8_SULPHUR_UPSTREAM'), reason='Original optional node source not bundled')
def test_independent_implementation_matches_original_upstream_network_and_soft_cap():
    source = Path(os.environ['T8_SULPHUR_UPSTREAM']) / 'corrector.py'
    spec = importlib.util.spec_from_file_location('sulphur_original_network', source)
    upstream = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = upstream
    spec.loader.exec_module(upstream)
    ours = tiny_corrector()['network']
    original = upstream.H3Corrector(upstream.CorrectorConfig(**vars(ours.config))).eval()
    original.load_state_dict(ours.state_dict(), strict=True)
    torch.manual_seed(54)
    video, velocity, text = torch.randn(2, 24, 3, 4, 8), torch.randn(2, 24, 3, 4, 8), torch.randn(2, 5, 8)
    with torch.no_grad():
        actual = ours(video, velocity, .9274, .5678, text)
        expected = original(video, velocity, .9274, .5678, text)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    for a, b in zip(bound_residual(actual, velocity, .5, .001), upstream.bound_residual(expected, velocity, .5, .001)):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
