"""Delivery frame endpoints select real temporal latent tokens before sampling."""
import json

import pytest
import torch
from comfy.nested_tensor import NestedTensor
from comfy_extras.nodes_custom_sampler import BasicGuider, RandomNoise, SamplerCustomAdvanced
import comfy.samplers

from h3_audio_t8_pkg.refine_endpoint_conditions import condition_refine_endpoints, token_frame_spans
from test_fast_h3_v2_core_sampler import model
from test_progressive_sampling_runtime import conditioning


def source(t=37):
    g=torch.Generator().manual_seed(971)
    return dict(samples=NestedTensor((torch.randn((1,24,t,2,2),generator=g)*.1,
                                    torch.randn((1,32,2,round(token_frame_spans(t)[-1][1]*40/24)),generator=g)*.1)))


@pytest.mark.parametrize("count,last_token",[(120,35),(121,36),(124,36)])
def test_actual_trim_tail_is_pinned_not_discarded_generation_tail(count,last_token):
    src=source()
    result,report=condition_refine_endpoints(src,True,0,count)
    report=json.loads(report)
    assert report["conditioned_latent_indices"] == [0,last_token]
    assert report["delivered_last_frame"] == count-1
    vm,am=result["noise_mask"].unbind()
    assert torch.all(vm[:,:,last_token]==0) and torch.all(am==1)
    if count==120:
        assert torch.all(vm[:,:,36]==1)
    assert result["samples"] is src["samples"] and "noise_mask" not in src


def test_disabled_exact_identity_and_existing_stronger_protection_audio_retained():
    src=source(7)
    assert condition_refine_endpoints(src)[0] is src
    v,a=src["samples"].unbind()
    src["noise_mask"]=NestedTensor((torch.full_like(v,.2),torch.zeros_like(a)))
    result,_=condition_refine_endpoints(src,True,1,19,2,1,.5)
    assert torch.equal(result["noise_mask"].unbind()[0],src["noise_mask"].unbind()[0])
    assert torch.equal(result["noise_mask"].unbind()[1],torch.zeros_like(a))


def test_real_core_sampler_obeys_endpoint_latent_conditions_without_rgb_postpaste():
    src=source(7)
    selected,_=condition_refine_endpoints(src,True,0,18)
    guider=BasicGuider.execute(model(),conditioning()).result[0]
    output=SamplerCustomAdvanced.execute(RandomNoise.execute(88).result[0],guider,
        comfy.samplers.ksampler("euler"),torch.tensor([.6,.3,0.]),selected).result[0]
    video=output["samples"].unbind()[0]
    original=src["samples"].unbind()[0]
    # Native H3 conditioning strength 0.999 adds its own tiny noise; this is
    # sampled preservation, intentionally not a claim of exact pasted pixels.
    assert (video[:,:,[0,5]]-original[:,:,[0,5]]).abs().max() < .01
    assert (video[:,:,1:5]-original[:,:,1:5]).abs().max() > .01


@pytest.mark.parametrize("start,count",[(120,5),(1,124),(125,0)])
def test_outside_delivery_frames_not_silently_relocated(start,count):
    with pytest.raises(ValueError,match="fit inside"):
        condition_refine_endpoints(source(),True,start,count)
