"""Real tiny Core full6 + independent Pika tail3; no trained-quality claim."""
import copy
import json

import pytest
import torch
import torch.nn.functional as F

import comfy.nested_tensor
from comfy_extras.nodes_custom_sampler import SplitSigmas
from h3_audio_t8_pkg.learned_latent_upscale_advanced import (
    audit_two_pass_h3_audio,
    reconcile_two_pass_h3_latent,
)
from h3_audio_t8_pkg.pika_flex import prepare_pika_flex
from test_fast_h3_v2_core_sampler import model
from test_pika_flex import run
from test_progressive_sampling_runtime import conditioning, latent


@pytest.mark.parametrize('task', ['t2va', 'fl2va', 'ref2va'])
def test_regular_full6_then_tail3_restarts_from_completed_video_and_locks_audio(task):
    bare, template = model(), latent()
    audio = template['samples'].unbind()[1]
    low_source = {'samples': comfy.nested_tensor.NestedTensor((torch.zeros(1,24,2,2,4), audio))}
    low_pos, high_pos = conditioning(), conditioning()
    for pos, height, width in [(low_pos,2,4),(high_pos,4,8)]:
        if task == 'fl2va':
            pos[0][1].update(minimax_frame_count=5, minimax_keyframes=[
                {'resolved_frame_index': i, 'latent': torch.full((1,24,1,height,width), .2)}
                for i in [0,4]])
        elif task == 'ref2va':
            pos[0][1]['minimax_refs'] = [{'kind':'image','latent_t':1,'latent_h':height,
                'latent_w':width,'ref_audio_t':0,'latent':torch.full((1,24,1,height,width),.3)}]
    original = copy.deepcopy(bare.model_options)
    forwards = []
    def before(owner, args, kwargs):
        x = kwargs.get('x', args[0] if args else None)
        t = kwargs.get('timestep', args[1] if len(args)>1 else None)
        forwards.append((tuple(x[0].shape), float(t.flatten()[0])))
    handle = bare.model.diffusion_model.register_forward_pre_hook(before, with_kwargs=True)
    try:
        low, solver, sigmas, _ = prepare_pika_flex(bare.clone(), low_source)
        finished = run(low, solver, sigmas, low_source, low_pos, seed=30)
        assert len(forwards)==6
        video, completed_audio = finished['samples'].unbind()
        # Explicit CPU test double, NOT the trained learned upscaler. The GPU
        # workflow separately executes the installed 322-tensor 3D model.
        lifted_video = F.interpolate(video, size=(2,4,8), mode='trilinear', align_corners=False)
        lifted = {**finished, 'samples': comfy.nested_tensor.NestedTensor((lifted_video,completed_audio))}
        first_snapshot = completed_audio.clone()
        source, positive, report = reconcile_two_pass_h3_latent(
            lifted, template, high_pos, 'first_pass', 'first_pass', 0.)
        assert json.loads(report)['second_pass_audio_locked']
        high, high_solver, full, _ = prepare_pika_flex(bare.clone(), source)
        tail = SplitSigmas.execute(full, 3).result[1]
        torch.testing.assert_close(tail, torch.tensor([.8,4/7,1/3,0]), atol=0, rtol=0)
        refined = run(high, high_solver, tail, source, positive, seed=31)
        verified, audit = audit_two_pass_h3_audio(source, refined, 0., True, 1e-5)
    finally:
        handle.remove()
    assert len(forwards)==9
    assert [x[0][-2:] for x in forwards]==[(2,4)]*6+[(4,8)]*3
    torch.testing.assert_close(torch.tensor([x[1] for x in forwards]), torch.cat((sigmas[:-1],tail[:-1]))*1000)
    assert json.loads(audit)['status']=='locked_audio_replaced_exact'
    assert torch.equal(verified['samples'].unbind()[1], first_snapshot)
    assert torch.equal(finished['samples'].unbind()[1], first_snapshot)
    assert torch.isfinite(verified['samples'].unbind()[0]).all()
    assert not torch.equal(verified['samples'].unbind()[0], lifted_video)
    assert bare.model_options==original and not bare.object_patches
    assert not torch.cuda.is_initialized()
