"""Actual installed BlockCache + actual small H3; CPU only, no external edits."""
import importlib
import importlib.util
import hashlib
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

import comfy.ops
from comfy.ldm.minimax import model as native
from comfy.ldm.modules import attention
from comfy.patcher_extension import WrapperExecutor
from comfy.model_management import InterruptProcessingException
from h3_audio_t8_pkg import enhance_a_video_advanced as eav
from h3_audio_t8_pkg.h3_core_compat import set_h3_attention_backend
from test_prompt_relay_core_compat import model_fixture, sparse


@pytest.fixture(scope='module')
def cache_nodes(record_testsuite_property):
    path = Path(os.environ.get('T8_BLOCKCACHE_ROOT') or
                (Path(importlib.util.find_spec('folder_paths').origin).resolve().parent /
                 'custom_nodes/comfyui-minimax-h3-blockcache-T8'))
    if not (path / 'nodes.py').is_file():
        pytest.skip('separately installed BlockCache is unavailable')
    name = 't8_block_cache_actual_core_probe'
    spec = importlib.util.spec_from_file_location(name, path / '__init__.py', submodule_search_locations=[str(path)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[name] = package
    spec.loader.exec_module(package)
    nodes = importlib.import_module(name + '.nodes')
    sources = [path / '__init__.py', path / 'nodes.py', path / 'h3_block_cache.py']
    before = {str(source): hashlib.sha256(source.read_bytes()).hexdigest() for source in sources}
    for source, digest in before.items():
        record_testsuite_property('external_cache_source:' + source, digest)
    yield nodes
    assert before == {str(source): hashlib.sha256(source.read_bytes()).hexdigest() for source in sources}


def cached_model(cache_nodes):
    model = model_fixture()
    model.model.diffusion_model.blocks = torch.nn.ModuleList([torch.nn.Identity() for _ in range(50)])
    for block in model.model.diffusion_model.blocks:
        block.attn = torch.nn.Identity()
    return cache_nodes.MiniMaxH3BlockCacheNode.execute(model, .12, 0., 1., 2, 'cpu', 1, False)[0]


def compose(model, mode='report_only'):
    sigmas = torch.cat((torch.linspace(1., .05, 20), torch.zeros(1)))
    return eav.build_eav_block_cache_model(model, sigmas, mode=mode, tau=.25,
        start_video_progress=0., end_video_progress=1., max_workspace_mib=4, g_hard_limit=1.5)


@pytest.mark.parametrize('backend', ['pytorch', 'sage'])
def test_ordinary_backend_and_disabled_identity(cache_nodes, backend):
    model = cached_model(cache_nodes)
    set_h3_attention_backend(model, getattr(attention, 'attention_' + backend))
    assert compose(model, 'disabled')[0] is model
    assert compose(model)[0] is not model


def executor_for(patched):
    wrappers = patched.get_wrappers('diffusion_model', eav.EAV_BLOCK_CACHE_WRAPPER_KEY)
    def never(*args, **kwargs):
        pytest.fail('missing payload or invalid owner must stop before diffusion')
    return WrapperExecutor.new_executor(never, wrappers)


@pytest.mark.parametrize('mutation', ['copied_marker', 'boundary', 'attn_hook'])
def test_unknown_runtime_owner_preserved_before_real_outer_state_error(cache_nodes, mutation, caplog):
    patched, _, _ = compose(cached_model(cache_nodes))
    options = patched.model_options['transformer_options']
    def foreign(*args, **kwargs):
        pytest.fail('foreign hook must never execute')
    if mutation == 'copied_marker':
        foreign._t8_h3_eav_patch_version = eav.EAV_PATCH_VERSION
        foreign._t8_h3_eav_block_cache_patch_version = eav.EAV_BLOCK_CACHE_PATCH_VERSION
        options['optimized_attention_override'] = foreign
    elif mutation == 'boundary':
        options['patches_replace']['dit'][('double_block', 0)] = foreign
    else:
        options.setdefault('patches', {})['attn1_patch'] = [foreign]
    # Unknown user owners are advisory. This intentionally incomplete invocation
    # must still report its actual missing execution-scoped cache state.
    with pytest.raises(RuntimeError, match='execution-scoped outer sample wrapper'):
        executor_for(patched).execute(None, None, None, options)
    assert 'continuing' in caplog.text
    if mutation == 'copied_marker':
        assert options['optimized_attention_override'] is foreign
    elif mutation == 'boundary':
        assert options['patches_replace']['dit'][('double_block', 0)] is foreign
    else:
        assert options['patches']['attn1_patch'] == [foreign]


def test_sparse_after_composer_keeps_current_sparse_owners_repeatedly(cache_nodes):
    patched, _, _ = compose(cached_model(cache_nodes))
    downstream = sparse(patched)
    for _ in range(2):
        downstream.prepare_state(torch.tensor(.5), downstream.model_options)
        options = downstream.model_options['transformer_options']
        expected = dict(options['patches_replace']['dit'])
        assert len(expected) == 50
        with pytest.raises(RuntimeError, match='execution-scoped outer sample wrapper'):
            executor_for(downstream).execute(None, None, None, options)
        assert options['patches_replace']['dit'] == expected
        assert all(options['patches_replace']['dit'][key] is value for key, value in expected.items())


@pytest.mark.parametrize('mode', ['disabled', 'report_only', 'apply_exp'])
def test_sparse_between_cache_and_composer_is_scoped_and_keeps_disabled_identity(cache_nodes, mode):
    original = sparse(cached_model(cache_nodes))
    original_dit = dict(original.model_options['transformer_options']['patches_replace']['dit'])
    original_override = original.model_options['transformer_options']['optimized_attention_override']
    patched, _, _ = compose(original, mode)
    assert original.model_options['transformer_options']['patches_replace']['dit'] == original_dit
    assert original.model_options['transformer_options']['optimized_attention_override'] is original_override
    if mode == 'disabled':
        assert patched is original
    else:
        assert patched is not original
        dit = patched.model_options['transformer_options']['patches_replace']['dit']
        assert len(dit) == 50 and dit == original_dit
        assert all(dit[key] is value for key, value in original_dit.items())


@pytest.mark.parametrize('mutation', ['wrapper', 'unknown_block'])
def test_sparse_composition_preserves_foreign_wrapper_or_block_without_reconstruction(cache_nodes, mutation):
    model = sparse(cached_model(cache_nodes))
    def foreign(*args, **kwargs):
        pytest.fail('foreign hook must not execute')
    if mutation == 'wrapper':
        model.wrappers['diffusion_model'][eav.BLOCK_CACHE_WRAPPER_KEY] = [foreign]
    else:
        model.model_options['transformer_options']['patches_replace']['dit'][('double_block', 25)] = foreign
    patched, _, _ = compose(model)
    assert patched is not model
    if mutation == 'unknown_block':
        assert model.model_options['transformer_options']['patches_replace']['dit'][('double_block', 25)] is foreign
        assert patched.model_options['transformer_options']['patches_replace']['dit'][('double_block', 25)] is foreign
    else:
        assert model.wrappers['diffusion_model'][eav.BLOCK_CACHE_WRAPPER_KEY] == [foreign]


def real_case(cache_nodes, monkeypatch, mode, sparse_position='none'):
    monkeypatch.setattr(native, 'optimized_attention', attention.attention_pytorch)
    torch.manual_seed(827)
    diffusion = native.MiniMaxH3Model(hidden_size=8, num_layers=50, token_refiner_num_layers=0,
        num_attention_heads=2, attention_head_dim=8, ffn_hidden_size=16, text_dim=8,
        timestep_input_dim=4, time_embed_hidden_size=8, time_embed_dim=4, rope_inv_freq_len=1,
        dtype=torch.float32, device='cpu', operations=comfy.ops.disable_weight_init)
    for parameter in diffusion.parameters():
        torch.nn.init.normal_(parameter, std=.05)
    diffusion.requires_grad_(False)
    diffusion.rope.inv_freq.fill_(1.)
    model = model_fixture()
    model.model.diffusion_model = diffusion
    cached = cache_nodes.MiniMaxH3BlockCacheNode.execute(model, .12, 0., 1., 2, 'cpu', 1, False)[0]
    if sparse_position == 'before':
        cached = sparse(cached)
    patched, runtime, _ = compose(cached, mode)
    if sparse_position == 'after':
        patched = sparse(patched)
    patched.prepare_state(torch.tensor(.5), patched.model_options)
    options = patched.model_options['transformer_options'].copy()
    options['wrappers'] = patched.wrappers
    options['sigmas'] = torch.tensor([.5])
    options['sample_sigmas'] = torch.tensor([1., .5, 0.])
    options[eav.BLOCK_CACHE_KEY] = options[eav.BLOCK_CACHE_KEY].clone().prepare(model.model.model_sampling)
    x = [torch.randn(1, 24, 2, 4, 4), torch.randn(1, 32, 2, 3)]
    t, context = torch.tensor([500.]), torch.randn(1, 2, 8)
    payload = {'layout': native.PackedLayout(2, 2, 4, 4, 3)}
    return patched, runtime, diffusion, options, x, t, context, payload


@pytest.mark.parametrize('mode', ['report_only', 'apply_exp'])
@pytest.mark.parametrize('sparse_position', ['none', 'before', 'after'])
def test_real_cache_full_then_hit_matches_identical_input_control(cache_nodes, monkeypatch, mode, sparse_position):
    patched, runtime, diffusion, options, x, t, context, payload = real_case(cache_nodes, monkeypatch, mode, sparse_position)
    original_final = diffusion.final_layer
    with torch.no_grad():
        full = diffusion(x, t, context, transformer_options=options, minimax_payload=payload)
        hit = diffusion(x, t, context, transformer_options=options, minimax_payload=payload)
    for a, b in zip(full, hit):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
    assert diffusion.final_layer is original_final
    expected = ['full', 'hit'] if sparse_position == 'none' else ['bypassed', 'bypassed']
    assert [row['block_cache_decision'] for row in runtime.snapshot(consume=False)['forwards']] == expected
    assert eav.EAV_RUNTIME_KEY not in options
    assert options[eav.BLOCK_CACHE_KEY].cache_hits == (1 if sparse_position == 'none' else 0)


@pytest.mark.parametrize('exception', [RuntimeError, InterruptProcessingException])
def test_actual_outer_cache_lifecycle_releases_state_on_cancel_and_fresh_retry(cache_nodes, monkeypatch, exception):
    patched, runtime, diffusion, options, x, t, context, payload = real_case(cache_nodes, monkeypatch, 'report_only')
    prototype = patched.model_options['transformer_options'][eav.BLOCK_CACHE_KEY]
    options[eav.BLOCK_CACHE_KEY] = prototype
    original_options = {'transformer_options': options}
    guider = SimpleNamespace(model_options=original_options, model_patcher=patched)
    outer = patched.get_wrappers('outer_sample', eav.BLOCK_CACHE_WRAPPER_KEY)
    assert outer == [cache_nodes.h3_block_cache_sample_wrapper]
    seen = []
    def sample(abort):
        live = guider.model_options['transformer_options']
        seen.append(live[eav.BLOCK_CACHE_KEY])
        assert seen[-1] is not prototype
        with torch.no_grad():
            first = diffusion(x, t, context, transformer_options=live, minimax_payload=payload)
            second = diffusion(x, t, context, transformer_options=live, minimax_payload=payload)
        for full, hit in zip(first, second):
            torch.testing.assert_close(full, hit, rtol=1e-5, atol=1e-6)
        assert seen[-1].cache_hits == 1
        if abort:
            raise exception('controlled-stop-after-real-cache-hit')
        return second
    executor = WrapperExecutor.new_class_executor(sample, guider, outer)
    with pytest.raises(exception, match='controlled-stop-after-real-cache-hit'):
        executor.execute(True)
    assert guider.model_options is original_options
    assert not seen[0].streams and seen[0].current is None
    assert not prototype.streams and prototype.total_forwards == 0
    runtime.snapshot(consume=True)
    result = executor.execute(False)
    assert len(result) == 2
    assert seen[1] is not seen[0]
    assert guider.model_options is original_options
    assert all(not cache.streams and cache.current is None for cache in seen)
    assert eav.EAV_RUNTIME_KEY not in options
