"""Real Core bypass LoRA must survive native stage lifecycle without relaxing mutation checks."""
from copy import copy
import json
from types import FunctionType, MethodType

import pytest
import torch
import comfy.sd
from comfy_extras.nodes_custom_sampler import BasicGuider, RandomNoise

from h3_audio_t8_pkg import sampling
from h3_audio_t8_pkg.long_video_dual_identity import stage_model_identity
from h3_audio_t8_pkg.modular_sampling import core_bypass_identity as bypass, native_explicit, eav
from h3_audio_t8_pkg.modular_sampling.results import sample_stage, execution_identity
from h3_audio_t8_pkg.modular_sampling.storage import save_stage, load_stage
from h3_audio_t8_pkg.patch_stack_policy import UnverifiedModelStack, model_identity_matches
from test_fast_h3_v2_core_sampler import model
from test_modular_rf_restart import source_latent
from test_progressive_sampling_runtime import conditioning
from test_progressive_relay import paired


def prepared(dtype=torch.bfloat16, strength=1., bias=True, bare=None, targets=None):
    bare = model() if bare is None else bare
    state = bare.model_state_dict()
    targets = (['diffusion_model.blocks.0.attn.qkv_proj', 'diffusion_model.blocks.0.adaln_proj.linear']
               if targets is None else targets)
    values = {}
    for target in targets:
        out, width = state[target+'.weight'].shape
        values.update({target+'.lora_A.weight':torch.full((2,width),.02,dtype=dtype),
                       target+'.lora_B.weight':torch.full((out,2),.03,dtype=dtype),
                       target+'.alpha':torch.tensor(2.)})
        if bias and target+'.bias' in state:
            values[target+'.diff_b']=torch.full((out,),.004,dtype=dtype)
    patched, clip = comfy.sd.load_bypass_lora_for_models(bare,None,values,strength,0.)
    assert clip is None and not bare.patches and not bare.injections
    return patched


def arguments(stage='native_low', effects=False, dtype=torch.bfloat16):
    bare, positive = prepared(dtype), conditioning()
    source=source_latent(True)
    if effects:
        bare, positive, _=paired(bare)
    bare, sampler, sigmas=sampling.setup_dual_clock_sampling(bare,source,4,12.,3.)
    bound, sampler, sigmas, context, _=native_explicit.bind_stage(bare,sampler,sigmas,source,stage)
    runtime=None
    if effects:
        bound,runtime,_=eav.apply_stage_eav(bound,sigmas,source,context,
            eav.EAVConfig('apply_exp',tau=.2,start_video_progress=0.,end_video_progress=1.,g_hard_limit=3.))
    return (RandomNoise.execute(987).result[0],BasicGuider.execute(bound,positive).result[0],
            sampler,sigmas,source,context),runtime


@pytest.mark.parametrize('stage',native_explicit.STAGES)
@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
@pytest.mark.parametrize('effects',[False,True])
def test_real_bypass_stage_hot_cold_identity_and_saved_state(stage,dtype,effects,tmp_path):
    args,runtime=arguments(stage,effects,dtype)
    first=sample_stage(*args)[2]
    receipt=first.verify()
    assert receipt['portable_identity'] is receipt['verified_recipe_completion'] is True
    assert receipt['execution']['denoiser_evaluations']==4
    repeated=sample_stage(*args)[2].verify()
    assert repeated['request_sha256']==receipt['request_sha256']
    fresh=sample_stage(*arguments(stage,effects,dtype)[0])[2].verify()
    assert fresh['request_sha256']==receipt['request_sha256']
    path,digest,_=save_stage(first,tmp_path)
    loaded=load_stage(tmp_path,path,digest,stage)[0]
    assert all(torch.equal(a,b) for a,b in zip(first.output['samples'].unbind(),loaded['samples'].unbind()))
    if runtime:
        assert runtime.snapshot()['status']=='observed_apply_exp'


def test_projection_preserves_original_live_hook_and_regular_bias_patches():
    bare=prepared()
    original=bare.model
    before=stage_model_identity(bare)
    injection=bare.get_injections(bypass.KEY)[0]
    injection.inject(bare)
    try:
        module=bare.model.diffusion_model.blocks[0].attn.qkv_proj
        actual=module.forward
        view,contract=bypass.project(bare)
        assert module.forward is actual and bare.model is original
        assert view.model is not original and not view.get_injections(bypass.KEY)
        assert view.patches==bare.patches and contract['adapter_count']==2
        assert model_identity_matches(before,stage_model_identity(bare))
    finally:
        injection.eject(bare)
    assert model_identity_matches(before,stage_model_identity(bare))


@pytest.mark.parametrize('mutation',['weights','strength','extra_hook','foreign_injection','wrong_owner'])
def test_real_descriptor_mutation_never_disappears(mutation):
    bare=prepared()
    before=stage_model_identity(bare)
    injection=bare.get_injections(bypass.KEY)[0]
    manager=injection.inject.__closure__[0].cell_contents
    hook=manager.hooks[0]
    if mutation=='weights':
        hook.adapter.weights[0].add_(.001)
    elif mutation=='strength':
        hook.multiplier=.25
    elif mutation=='extra_hook':
        hook.unknown_callback=lambda x:x
    elif mutation=='foreign_injection':
        injection.inject=lambda model:None
    else:
        hook.module=copy(hook.module)
    after=stage_model_identity(bare)
    assert not model_identity_matches(before,after)
    if mutation!='weights':
        assert after['portable_cache_reuse'] is False
        with pytest.raises(UnverifiedModelStack):
            bypass.project(bare)


def test_foreign_prior_forward_is_preserved_and_not_portable():
    bare=prepared()
    module=bare.model.diffusion_model.blocks[0].attn.qkv_proj
    def foreign(self,*args,**kwargs):
        return type(self).forward(self,*args,**kwargs)
    module.forward=MethodType(foreign,module)
    old=module.forward
    view,_=bypass.project(bare)
    assert view.model.diffusion_model.blocks[0].attn.qkv_proj.forward is old
    assert module.forward is old and stage_model_identity(bare)['portable_cache_reuse'] is False


@pytest.mark.parametrize('owner,name',[
    (bypass.BypassForwardHook,'inject'),(bypass.BypassForwardHook,'_bypass_forward'),
    (bypass.BypassInjectionManager,'_get_module_by_key'),(bypass.LoRAAdapter,'h'),
    (bypass.core_base.WeightAdapterBase,'g'),(bypass.LoRAAdapter,'bypass_forward'),
])
def test_changed_class_method_cannot_masquerade_as_stock_core(monkeypatch,owner,name):
    bare=prepared()
    before=stage_model_identity(bare)
    original=getattr(owner,name)
    def foreign(*args,**kwargs):
        raise AssertionError('Identity inspection must never execute this delegate')
    # Even matching source filenames/globals are insufficient without exact code.
    forged=FunctionType(foreign.__code__.replace(co_filename=original.__code__.co_filename),
                        original.__globals__)
    monkeypatch.setattr(owner,name,forged)
    after=stage_model_identity(bare)
    assert after['portable_cache_reuse'] is False and not model_identity_matches(before,after)
    with pytest.raises(UnverifiedModelStack):
        bypass.project(bare)


def test_changed_native_move_defaults_cannot_be_authenticated(monkeypatch):
    bare=prepared()
    method=bypass.BypassForwardHook._move_adapter_weights_to_device
    monkeypatch.setattr(method,'__defaults__',(torch.float16,))
    assert stage_model_identity(bare)['portable_cache_reuse'] is False
    with pytest.raises(UnverifiedModelStack):
        bypass.project(bare)


def test_real_208_target_contract_is_bounded_and_binds_every_weight():
    original=model().model.model_config
    config=type(original)(dict(original.unet_config))
    config.unet_config.update(num_layers=50,token_refiner_num_layers=2)
    base=comfy.model_base.MiniMaxH3(config,device=torch.device('cpu'))
    with torch.no_grad():
        for parameter in base.parameters():
            parameter.zero_()
    base.diffusion_model.rope.inv_freq.fill_(1.)
    bare=comfy.model_patcher.ModelPatcher(base,torch.device('cpu'),torch.device('cpu'))
    targets=[key.removesuffix('.weight') for key in bare.model_state_dict()
             if key.endswith(('.attn.qkv_proj.weight','.attn.out_proj.weight','.mlp.fc1.weight','.mlp.fc2.weight'))]
    assert len(targets)==208
    patched=prepared(bare=bare,targets=targets,bias=False)
    _,before=bypass.project(patched)
    assert before['adapter_count']==208 and len(json.dumps(before).encode('utf8'))<1024
    manager=patched.get_injections(bypass.KEY)[0].inject.__closure__[0].cell_contents
    original=before['adapters_sha256']
    # Changes at the end of the real descriptor list are not lost to truncation.
    manager.hooks[-1].adapter.weights[0].add_(.001)
    _,after=bypass.project(patched)
    assert after['adapters_sha256']!=original and before['target_paths_sha256']==after['target_paths_sha256']


def test_original_plain_stage_identity_and_sampling_grid_unchanged():
    bare=model()
    source=source_latent(True)
    configured,sampler,sigmas=sampling.setup_dual_clock_sampling(bare,source,4,12.,3.)
    before=sigmas.clone()
    configured,_,table,context,_=native_explicit.bind_stage(configured,sampler,sigmas,source)
    initial=configured.model.model_sampling
    fingerprint=execution_identity(configured,initial)
    assert 'core_bypass_lora' not in fingerprint and table is sigmas and torch.equal(before,sigmas)
    assert json.loads(context.profile)['sampler_kind']=='dual_clock_euler'
