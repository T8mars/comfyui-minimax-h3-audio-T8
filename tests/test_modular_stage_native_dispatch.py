"""Actual Core patch/unpatch may restore class dispatch as an instance alias.

Only that exact bound alias is equivalent. Foreign owners, hooks, class code
and selected patches remain part of Stage's within-run mutation evidence.
"""
import inspect
from types import MethodType

import pytest
import torch
from comfy_extras.nodes_custom_sampler import SamplerCustomAdvanced

from h3_audio_t8_pkg.h3_memory_advanced import configure_chunk_feed_forward, configure_low_vram_attention
from h3_audio_t8_pkg.h3_memory_advanced import inspect_t8_memory_composition
from h3_audio_t8_pkg.modular_sampling.results import execution_identity, sample_stage
from h3_audio_t8_pkg.patch_stack_policy import model_identity_matches
from test_fast_h3_v2_core_sampler import model
from test_modular_native_explicit import inputs


def prepared():
    bare = configure_low_vram_attention(model(), 2)[0]
    bare = configure_chunk_feed_forward(bare, 2, 256)[0]
    args, _, original = inputs(bare=bare)
    return args, original


def test_actual_core_unpatch_native_alias_preserves_identity_without_mutating_selection():
    args, _ = prepared()
    selected = args[1].model_patcher
    initial = selected.model.model_sampling
    before = execution_identity(selected, initial)
    patches = dict(selected.object_patches)
    # Real Core stores a previously inherited bound method in its backup and
    # restores it with set_attr. No faux post-run receipt is supplied here.
    selected.patch_model(torch.device('cpu'), load_weights=False)
    selected.unpatch_model(unpatch_weights=False)
    block = selected.model.diffusion_model.blocks[0]
    assert isinstance(vars(block)['forward'], MethodType)
    assert vars(block)['forward'].__func__ is inspect.getattr_static(type(block), 'forward')
    after = execution_identity(selected, initial)
    assert model_identity_matches(before, after)
    assert selected.object_patches == patches
    assert before['portable_cache_reuse'] is after['portable_cache_reuse'] is False
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('mutation', ['foreign_bound_forward', 'wrong_owner', 'hook', 'class_forward'])
def test_native_alias_does_not_hide_real_execution_mutations(monkeypatch, mutation):
    args, _ = prepared()
    selected = args[1].model_patcher
    initial = selected.model.model_sampling
    before = execution_identity(selected, initial)
    selected.patch_model(torch.device('cpu'), load_weights=False)
    selected.unpatch_model(unpatch_weights=False)
    # Use a non-memory module so its unknown hook/owner cannot inherit the
    # three authenticated attention/MLP/block method slots.
    linear = selected.model.diffusion_model.blocks[0].attn.out_proj
    native = inspect.getattr_static(type(linear), 'forward')

    def foreign(self, *values, **options):
        return native(self, *values, **options)

    if mutation == 'foreign_bound_forward':
        linear.forward = MethodType(foreign, linear)
    elif mutation == 'wrong_owner':
        linear.forward = MethodType(native, selected.model.diffusion_model.blocks[0].attn.qkv_proj)
    elif mutation == 'hook':
        linear.forward = MethodType(native, linear)
        linear.register_forward_hook(lambda module, values, output: output)
    else:
        monkeypatch.setattr(type(linear), 'forward', foreign)
    after = execution_identity(selected, initial)
    assert not model_identity_matches(before, after)


def test_actual_core_memory_sampler_keeps_both_AV_outputs_and_step_count_exact():
    args, original = prepared()
    expected = SamplerCustomAdvanced.execute(*original).result
    actual = sample_stage(*args)
    for output, wanted in zip(actual[:2], expected, strict=True):
        assert all(torch.equal(a, b) for a, b in zip(output['samples'].unbind(), wanted['samples'].unbind(), strict=True))
    receipt = actual[2].verify()
    assert receipt['execution']['callbacks'] == list(range(args[-1].steps))
    assert receipt['execution']['denoiser_evaluations'] == args[-1].steps
    assert receipt['cache_reuse_authorized'] is False
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('group', [None, 'None'])
def test_anonymous_core_wrapper_key_is_not_coerced_and_loaded_memory_dispatch_is_equivalent(group):
    args, _ = prepared()
    selected = args[1].model_patcher

    def delegate(executor, *values, **options):
        return executor(*values, **options)

    selected.add_wrapper_with_key('diffusion_model', group, delegate)
    methods = inspect_t8_memory_composition(selected, allowed_wrapper_keys=(group,))
    assert methods is not None
    assert selected.get_wrappers('diffusion_model', group) == [delegate]
    # Explicitly allowing the OTHER key cannot grant this group recognition.
    other = 'None' if group is None else None
    assert inspect_t8_memory_composition(selected, allowed_wrapper_keys=(other,)) is None
    initial = selected.model.model_sampling
    before = execution_identity(selected, initial)
    selected.patch_model(torch.device('cpu'), load_weights=False)
    after = execution_identity(selected, initial)
    assert model_identity_matches(before, after)
    assert before['portable_cache_reuse'] is after['portable_cache_reuse'] is False


@pytest.mark.parametrize('mutation', ['hook', 'pre_hook', 'foreign_forward', 'wrapper'])
def test_loaded_memory_dispatch_retains_real_hook_and_owner_mutations(mutation):
    args, _ = prepared()
    selected = args[1].model_patcher
    selected.add_wrapper_with_key('diffusion_model', None, lambda executor, *v, **kw: executor(*v, **kw))
    initial = selected.model.model_sampling
    before = execution_identity(selected, initial)
    selected.patch_model(torch.device('cpu'), load_weights=False)
    block = selected.model.diffusion_model.blocks[0]
    if mutation == 'hook':
        block.register_forward_hook(lambda module, values, output: output)
    elif mutation == 'pre_hook':
        block.register_forward_pre_hook(lambda module, values: values)
    elif mutation == 'foreign_forward':
        original = block.forward
        block.forward = MethodType(lambda self, *v, **kw: original(*v, **kw), block)
    else:
        selected.wrappers['diffusion_model'][None][0] = lambda executor, *v, **kw: executor(*v, **kw)
    try:
        after = execution_identity(selected, initial)
    except ValueError as error:
        assert mutation == 'foreign_forward' and 'memory forward changed' in str(error)
    else:
        assert not model_identity_matches(before, after)
