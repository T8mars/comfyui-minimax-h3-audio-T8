"""Actual tiny Core contracts; trained upscaler is a labelled CPU test double."""
import copy
import json

import pytest
import torch

import comfy.nested_tensor
from h3_audio_t8_pkg.pika_two_pass import sample_pika_two_pass
from h3_audio_t8_pkg.pika_flex import prepare_pika_flex
from test_fast_h3_v2_core_sampler import model
from test_pika_flex import run
from test_progressive_sampling_runtime import conditioning, latent

pytest_plugins = ['test_progressive_sampling_runtime']


def execute(bare=None, source=None, split=4, **kwargs):
    bare, source = bare or model(), source or latent()
    return sample_pika_two_pass(bare, conditioning(), source, conditioning(), source,
                               'cpu-test-double', 301030, split, **kwargs)


@pytest.mark.parametrize('split', [1, 2, 3, 4, 5])
def test_same_canvas_split_matches_continuous_six_step_video_and_audio_exactly(split):
    bare, source = model(), latent()
    selected, sampler, sigmas, _ = prepare_pika_flex(bare, source)
    expected = run(selected, sampler, sigmas, source)['samples'].unbind()
    result, clean, raw = execute(bare, source, split)
    for actual, reference in zip(result['samples'].unbind(), expected):
        torch.testing.assert_close(actual, reference, atol=0, rtol=0)
    report = json.loads(raw)
    assert report['fresh_noise_draws'] == 5
    assert report['callbacks'] == {'low': split, 'high': 6-split}
    assert report['completed_apply_model_calls'] == {'low': split, 'high': 6-split}
    assert report['upscaler']['status'] == 'same_canvas_no_lift'
    assert report['shared_base_object']
    assert not bare.wrappers and not bare.object_patches
    assert all(not value.count_nonzero() for value in source['samples'].unbind())
    assert not torch.equal(clean['samples'].unbind()[1], result['samples'].unbind()[1])
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('split', [3, 4])
@pytest.mark.parametrize('task', ['t2va', 'fl2va', 'ref2va'])
def test_changed_canvas_six_calls_uses_clean_vae_space_lift_and_high_conditions(stub_lifter, split, task):
    bare, high = model(), latent()
    low = {'samples': comfy.nested_tensor.NestedTensor((torch.zeros(1, 24, 2, 2, 4), high['samples'].unbind()[1]))}
    pos_high, pos_low = conditioning(), conditioning()
    if task == 'fl2va':
        for pos, height, width in [(pos_low, 2, 4), (pos_high, 4, 8)]:
            pos[0][1].update(minimax_frame_count=5, minimax_keyframes=[
                {'resolved_frame_index': i, 'latent': torch.full((1, 24, 1, height, width), .2+i*.01)}
                for i in [0, 4]])
    if task == 'ref2va':
        for pos in [pos_low, pos_high]:
            pos[0][1]['minimax_refs'] = [{'kind': 'image', 'latent_t': 1, 'latent_h': 4, 'latent_w': 8,
                'ref_audio_t': 0, 'latent': torch.ones(1, 24, 1, 4, 8)*.4}]
    original = copy.deepcopy(pos_high)
    seen, notifications = [], []
    def observe_dit(owner, args, kwargs):
        streams = args[0] if args else kwargs['x']
        seen.append(tuple(streams[0].shape))
    hook = bare.model.diffusion_model.register_forward_pre_hook(observe_dit, with_kwargs=True)
    try:
        result, clean, raw = sample_pika_two_pass(bare, pos_low, low, pos_high, high,
            'cpu-test-double', 301030, split, callback=lambda i, p, s, n: notifications.append((i,n)))
    finally:
        hook.remove()
    assert len(seen) == 6 and notifications == [(i,6) for i in range(6)]
    assert seen[:split] == [(1,24,2,2,4)] * split
    assert seen[split:] == [(1,24,2,4,8)] * (6-split)
    assert result['samples'].unbind()[0].shape == high['samples'].unbind()[0].shape
    assert clean['samples'].unbind()[0].shape == low['samples'].unbind()[0].shape
    report = json.loads(raw)
    assert report['fresh_noise_draws'] == 5
    assert report['completed_apply_model_calls'] == {'low': split, 'high': 6-split}
    assert len(stub_lifter) == 1 and report['upscaler']['test_double']
    assert pos_high[0][1].keys() == original[0][1].keys()
    for key in ['minimax_keyframes', 'minimax_refs']:
        for actual, old in zip(pos_high[0][1].get(key, []), original[0][1].get(key, [])):
            assert torch.equal(actual['latent'], old['latent'])


@pytest.mark.parametrize('resize', [False, True])
def test_locked_audio_is_not_frozen_at_partial_prediction_or_changed_from_source(stub_lifter, resize):
    bare, high = model(), latent()
    v, a = high['samples'].unbind()
    a = torch.linspace(-.1, .1, a.numel()).reshape(a.shape)
    high['samples'] = comfy.nested_tensor.NestedTensor((v,a))
    high['noise_mask'] = comfy.nested_tensor.NestedTensor((torch.ones(1,1,2,4,8), torch.zeros(1,1,2,8)))
    low = high if not resize else {
        'samples': comfy.nested_tensor.NestedTensor((torch.zeros(1,24,2,2,4),a)),
        'noise_mask': comfy.nested_tensor.NestedTensor((torch.ones(1,1,2,2,4),torch.zeros(1,1,2,8)))}
    result, _, raw = sample_pika_two_pass(bare, conditioning(), low, conditioning(), high, 'test',31)
    torch.testing.assert_close(result['samples'].unbind()[1], a, atol=3e-5, rtol=0)
    assert json.loads(raw)['fresh_noise_draws'] == 5


def test_independent_model_user_wrappers_are_delegated_and_failure_leaves_callers_untouched():
    bare, source = model(), latent()
    low, high = bare.clone(), bare.clone()
    called=[]
    def user(executor,*args,**kwargs):
        called.append('low')
        return executor(*args,**kwargs)
    def broken(executor,*args,**kwargs):
        called.append('high')
        raise RuntimeError('actual user kernel failure')
    low.add_wrapper_with_key('diffusion_model','unfamiliar-low',user)
    high.add_wrapper_with_key('diffusion_model','unfamiliar-high',broken)
    with pytest.raises(RuntimeError,match='actual user kernel failure'):
        execute(low, source, model_high=high)
    assert called == ['low']*4 + ['high']
    assert low.get_wrappers('diffusion_model','unfamiliar-low') == [user]
    assert high.get_wrappers('diffusion_model','unfamiliar-high') == [broken]
    assert not bare.wrappers and not bare.object_patches


@pytest.mark.parametrize('split', [0,6,True])
def test_impossible_split_fails_with_actual_six_evaluation_requirement(split):
    with pytest.raises(ValueError,match='six-step ladder'):
        execute(split=split)


def test_actual_time_shape_mismatch_fails_before_sampling():
    source, high = latent(), latent()
    v,a = high['samples'].unbind()
    high['samples'] = comfy.nested_tensor.NestedTensor((v,torch.zeros(1,32,2,a.shape[-1]+1)))
    with pytest.raises(ValueError,match='timeline and audio shape'):
        sample_pika_two_pass(model(),conditioning(),source,conditioning(),high,'test',3)


def test_independent_multiple_loras_keep_strengths_and_do_not_leak_across_shared_base(tmp_path):
    from safetensors.torch import save_file
    from h3_audio_t8_pkg.h3_lora_compat_advanced import load_minimax_h3_lora_model
    bare, source = model(), latent()
    generator = torch.Generator().manual_seed(77)
    projection = bare.model.diffusion_model.blocks[0].attn.qkv_proj
    paths=[]
    for index in range(2):
        path=tmp_path/f'custom-{index}.safetensors'
        prefix='diffusion_model.blocks.0.attn.qkv_proj'
        save_file({prefix+'.lora_A.weight':torch.randn(12,projection.weight.shape[1],generator=generator)*.1,
                   prefix+'.lora_B.weight':torch.randn(projection.weight.shape[0],12,generator=generator)*.1},
                  str(path),metadata={'converted_layout':'comfyui_minimax_h3','baked_scale':'1.0'})
        paths.append(path)
    def branch(strengths):
        result=bare
        for path,strength in zip(paths,strengths):
            result,_=load_minimax_h3_lora_model(result,path,strength)
        return result
    low=branch([.35,.7])
    high=branch([.9,-.2])
    before_low, before_high = copy.deepcopy(low.patches),copy.deepcopy(high.patches)
    out, clean, raw=execute(low,source,model_high=high)
    other, other_clean, _=execute(low,source,model_high=branch([-.5,.2]))
    assert json.loads(raw)['shared_base_object']
    assert json.loads(raw)['low_model']['weight_patch_entries']==2
    assert json.loads(raw)['high_model']['weight_patch_entries']==2
    for a,b in zip(clean['samples'].unbind(),other_clean['samples'].unbind()):
        torch.testing.assert_close(a,b,atol=0,rtol=0)
    assert any(not torch.equal(a,b) for a,b in zip(out['samples'].unbind(),other['samples'].unbind()))
    for actual,expected in [(low.patches,before_low),(high.patches,before_high)]:
        assert list(actual)==list(expected)
        for key in actual:
            assert [entry[0] for entry in actual[key]] == [entry[0] for entry in expected[key]]
    assert not bare.patches and not bare.object_patches


def test_repeat_seed_is_exact_and_different_seed_changes_both_streams():
    bare,source=model(),latent()
    def run_seed(seed):
        return sample_pika_two_pass(bare,conditioning(),source,conditioning(),source,'unused',seed)[0]
    a,b,c=[run_seed(seed)['samples'].unbind() for seed in [31,31,32]]
    for x,y,z in zip(a,b,c):
        assert torch.equal(x,y) and not torch.equal(x,z)


def test_locked_video_uses_actual_high_clean_source_after_cross_canvas_restart(stub_lifter):
    bare,high=model(),latent()
    v,a=high['samples'].unbind()
    v=torch.linspace(-.1,.1,v.numel()).reshape(v.shape)
    high={'samples':comfy.nested_tensor.NestedTensor((v,a)),
          'noise_mask':comfy.nested_tensor.NestedTensor((torch.zeros(1,1,2,4,8),torch.ones(1,1,2,8)))}
    low={'samples':comfy.nested_tensor.NestedTensor((torch.zeros(1,24,2,2,4),a)),
         'noise_mask':comfy.nested_tensor.NestedTensor((torch.zeros(1,1,2,2,4),torch.ones(1,1,2,8)))}
    out,_,_=sample_pika_two_pass(bare,conditioning(),low,conditioning(),high,'test',3)
    torch.testing.assert_close(out['samples'].unbind()[0],v,atol=3e-5,rtol=0)


def test_real_upscaler_file_error_happens_before_any_model_sampling():
    high=latent()
    low={'samples':comfy.nested_tensor.NestedTensor((torch.zeros(1,24,2,2,4),high['samples'].unbind()[1]))}
    with pytest.raises((ValueError,FileNotFoundError),match='found|exist|path'):
        sample_pika_two_pass(model(),conditioning(),low,conditioning(),high,'missing-upscaler.safetensors',1)


@pytest.mark.parametrize('split', [3,4])
def test_fractional_mask_same_canvas_matches_continuous_native_sampling(split):
    bare,source=model(),latent()
    v,a=source['samples'].unbind()
    source['samples']=comfy.nested_tensor.NestedTensor((torch.full_like(v,.17),torch.full_like(a,-.13)))
    source['noise_mask']=comfy.nested_tensor.NestedTensor((torch.full((1,1,2,4,8),.4),
                                                         torch.full((1,1,2,8),.6)))
    source['user_metadata']={'untouched':True}
    selected,sampler,sigmas,_=prepare_pika_flex(bare,source)
    expected=run(selected,sampler,sigmas,source)
    actual,_,_=execute(bare,source,split)
    for a,b in zip(actual['samples'].unbind(),expected['samples'].unbind()):
        torch.testing.assert_close(a,b,atol=0,rtol=0)
    assert actual['user_metadata']=={'untouched':True}


def test_reference_video_payload_reaches_each_actual_stage_with_its_own_geometry(stub_lifter):
    bare,high=model(),latent()
    low={'samples':comfy.nested_tensor.NestedTensor((torch.zeros(1,24,2,2,4),high['samples'].unbind()[1]))}
    low_positive,high_positive=conditioning(),conditioning()
    for positive,height,width in [(low_positive,2,4),(high_positive,4,8)]:
        positive[0][1]['minimax_refs']=[{'kind':'video','latent_t':2,'latent_h':height,'latent_w':width,
            'ref_audio_t':0,'latent':torch.full((1,24,2,height,width),.27)}]
    seen=[]
    def observe(_owner,args,kwargs):
        refs=kwargs['minimax_payload']['refs']
        seen.append([(ref['kind'],ref['latent_t'],ref['latent_h'],ref['latent_w']) for ref in refs])
    hook=bare.model.diffusion_model.register_forward_pre_hook(observe,with_kwargs=True)
    try:
        output,_,_=sample_pika_two_pass(bare,low_positive,low,high_positive,high,'test',7)
    finally:
        hook.remove()
    assert seen==[[('video',2,2,4)]]*4+[[('video',2,4,8)]]*2
    assert output['samples'].unbind()[0].shape==high['samples'].unbind()[0].shape
    for positive in [low_positive,high_positive]:
        assert torch.all(positive[0][1]['minimax_refs'][0]['latent']==.27)
