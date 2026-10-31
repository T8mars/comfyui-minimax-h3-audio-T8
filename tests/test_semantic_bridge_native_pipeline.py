"""Actual native H3/Bridge/SigmaShift/AV sampling plumbing, CPU only.

The H3 denoiser and public test bridge weights are tiny/random fixtures. This
checks software compatibility, not generated-media quality.
"""
import json

import pytest
import torch

import comfy.latent_formats
import comfy.model_base
import comfy.model_patcher
import comfy.ops
import comfy.supported_models_base
from comfy_extras import nodes_minimax_h3
from comfy_extras.nodes_custom_sampler import (
    BasicGuider, BasicScheduler, KSamplerSelect, RandomNoise, SamplerCustomAdvanced,
)
from h3_audio_t8_pkg import semantic_bridge as sb
from h3_audio_t8_pkg.long_video_dual_identity import _v2_sampling_identity
from h3_audio_t8_pkg.nodes_semantic_bridge import MiniMaxH3SemanticBridgeApplyT8
from test_progressive_sampling_runtime import latent
from test_semantic_bridge import weight_file  # noqa: F401


def raw_text_model():
    class Config(comfy.supported_models_base.BASE):
        latent_format = comfy.latent_formats.MiniMaxH3AV
        unet_extra_config = {}
        sampling_settings = {'shift':12., 'audio_shift':3.}
        custom_operations = comfy.ops.disable_weight_init
    config = Config(dict(hidden_size=24,num_layers=1,token_refiner_num_layers=0,
        num_attention_heads=3,attention_head_dim=128,ffn_hidden_size=32,
        text_dim=5120,timestep_input_dim=4,time_embed_hidden_size=24,
        time_embed_dim=24,rope_inv_freq_len=16,gate_compress=True,dtype=torch.float32))
    base = comfy.model_base.MiniMaxH3(config,device=torch.device('cpu'))
    generator = torch.Generator().manual_seed(261031)
    with torch.no_grad():
        for p in base.parameters():
            p.copy_(torch.randn(p.shape,generator=generator)*.02)
        base.diffusion_model.rope.inv_freq.fill_(1.)
    return comfy.model_patcher.ModelPatcher(base,torch.device('cpu'),torch.device('cpu'))


def run_bridge_pipeline(config, task='t2va'):
    bare, source = raw_text_model(), latent()
    native = torch.randn(1,8,5120,generator=torch.Generator().manual_seed(34))
    tags = torch.tensor([0,0,1,1,1,1,1,1])
    metadata = {'minimax_token_tags':tags}
    if task=='ref2va':
        metadata['minimax_refs']=[{'kind':'image','latent_t':1,'latent_h':4,'latent_w':8,
            'ref_audio_t':0,'latent':torch.ones(1,24,1,4,8)*.2}]
    items = [[native,metadata]]
    applied = MiniMaxH3SemanticBridgeApplyT8.execute(items,config)
    positive, report = applied.result
    selected = nodes_minimax_h3.MiniMaxH3SigmaShift.execute(bare,12.,3.).result[0]
    identity = _v2_sampling_identity(selected.get_model_object('model_sampling'))
    assert identity['protocol']=='core_minimax_h3_sigma_shift'
    sigmas = BasicScheduler.execute(selected,'simple',2,1.).result[0]
    sampler = KSamplerSelect.execute('euler').result[0]
    guider = BasicGuider.execute(selected,positive).result[0]
    output = SamplerCustomAdvanced.execute(RandomNoise.execute(31).result[0],
        guider,sampler,sigmas,source).result[0]
    assert all(torch.isfinite(t).all() for t in output['samples'].unbind())
    assert positive[0][1]['minimax_token_tags'] is tags
    assert sb.RECEIPT_KEY not in items[0][1]
    assert torch.equal(items[0][0],native)
    assert not bare.object_patches
    assert not torch.cuda.is_initialized()
    return json.loads(report), output


@pytest.mark.parametrize('task',['t2va','ref2va'])
def test_bridge_node_and_unchanged_native_shift_run_complete_AV_sampling(weight_file,task):  # noqa: F811
    config=sb.BridgeConfig(str(weight_file),sb.file_sha(weight_file),device='cpu',
                           token_scope='text_only_preserve_reference')
    report,output=run_bridge_pipeline(config,task)
    assert report['applied'] and report['items'][0]['shape']==[1,8,5120]
