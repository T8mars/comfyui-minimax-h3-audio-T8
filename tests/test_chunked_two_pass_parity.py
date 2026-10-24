from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from h3_audio_t8_pkg import chunked_two_pass_parity as parity
from h3_audio_t8_pkg import chunked_two_pass_upscale_advanced as legacy


def _report():
    return {'node': 'MiniMaxH3LearnedTwoPassParityPlanT8Advanced',
        'status': 'upstream_schedule_reproduced', 'base_steps': 8,
        'coarse_steps': 4, 'refine_steps': 4, 'shift_video': 12., 'shift_audio': 3.,
        'coarse_video_sigmas': [1., .98, .96, .94, .92],
        'refine_video_sigmas': list(parity.UPSTREAM_REFINE_VIDEO_SIGMAS[4])}


def _plan(**updates):
    kwargs = dict(model_name='mock.safetensors', target_width=64, target_height=64,
        temporal_chunk_frames=34, temporal_overlap_frames=17, anchor_strength=.999,
        tile_width=64, tile_height=64, spatial_overlap=0, spatial_fade=0,
        minimum_tile_size=32, overlap_blend='smoothstep', precision='fp16',
        release_policy='offload_after', spatial_strategy='full_frame_safe',
        sampling_contract=parity.CONTRACT, parity_report_json=json.dumps(_report()))
    kwargs.update(updates)
    return legacy.build_chunked_two_pass_plan(**kwargs)[0]


@pytest.fixture
def harness(monkeypatch):
    calls = {'upscale': 0, 'noise': 0, 'sample': [], 'bind': []}
    nested = legacy.comfy.nested_tensor.NestedTensor
    def upscale(latent, *args):
        calls['upscale'] += 1
        v, a = latent['samples'].tensors
        out = dict(latent)
        out['samples'] = nested((v.repeat_interleave(2, -1).repeat_interleave(2, -2), a))
        return out, 64, 64, '{}'
    def sample(piece, positive, model, noise, sampler, sigmas, negative, cfg, *, prepared_noise):
        calls['sample'].append(piece)
        v, a = piece['samples'].tensors
        vm, am = piece['noise_mask'].tensors
        return nested((v + vm * 2, a + am * 3))
    def bind(model, piece, sampler):
        calls['bind'].append(tuple(t.shape for t in piece['samples'].tensors))
        return sampler
    class Noise:
        seed = 23
        def generate_noise(self, latent):
            calls['noise'] += 1
            return nested(tuple(torch.ones_like(t) for t in latent['samples'].tensors))
    def sample_minimax_h3_dual_clock_euler():
        pass
    sample_minimax_h3_dual_clock_euler._minimax_h3_shift_video = 12.
    sample_minimax_h3_dual_clock_euler._minimax_h3_shift_audio = 3.
    monkeypatch.setattr(legacy, 'learned_upscale_h3_av_latent', upscale)
    monkeypatch.setattr(legacy, 'sample_piece', sample)
    monkeypatch.setattr(parity, 'rebind_dual_clock_sampler', bind)
    sampler = SimpleNamespace(sampler_function=sample_minimax_h3_dual_clock_euler)
    def run(plan=None, tokens=15, audio_tokens=85, **updates):
        v = torch.zeros(1, 24, tokens, 2, 2)
        a = torch.zeros(1, 32, 2, audio_tokens)
        latent = {'samples': nested((v, a))}
        args = dict(model=object(), conditioning=[[torch.zeros(1), {}]],
            latent=latent, noise=Noise(), sampler=sampler,
            sigmas=torch.tensor(parity.UPSTREAM_REFINE_VIDEO_SIGMAS[4]), plan=plan or _plan())
        args.update(updates)
        output, report = legacy.execute_chunked_two_pass_upscale(**args)
        assert torch.equal(v, torch.zeros_like(v)) and torch.equal(a, torch.zeros_like(a))
        return output, json.loads(report)
    return calls, run


def test_joint4plus4_publishes_refined_audio_and_exact_overlap(harness):
    calls, run = harness
    output, report = run(audio_tokens=90)  # Include source padding at the last window.
    v, a = output['samples'].tensors
    assert torch.equal(v, torch.full_like(v, 2))
    assert torch.equal(a, torch.full_like(a, 3))
    assert a.shape[-1] == 90
    assert calls['upscale'] == calls['noise'] == 1
    assert len(calls['sample']) == len(calls['bind']) == report['segment_count'] == 2
    assert report['total_model_calls_including_coarse_contract'] == 12
    assert report['executor_refine_model_calls_contract'] == 8
    assert report['segments'][1]['locked_video_overlap_tokens'] == 5
    assert report['segments'][1]['locked_audio_overlap_tokens'] > 0
    assert report['audio_resampled'] and not report['audio_preserved_by_identity']
    assert not report['post_sampling_blend'] and not report['quality_qualified']


def test_single_window_has_standard8_contract(harness):
    calls, run = harness
    _, report = run(tokens=10, audio_tokens=57)
    assert len(calls['sample']) == 1
    assert report['total_model_calls_including_coarse_contract'] == 8


@pytest.mark.parametrize('sigmas', [torch.tensor([1., .5, .1, 0]), torch.linspace(1, 0, 9),
    torch.tensor([.9035, .8, .6, .3, 0])])
def test_wrong_refine_sigma_rejected_before_inference(harness, sigmas):
    calls, run = harness
    with pytest.raises(ValueError, match='four standard intervals'):
        run(sigmas=sigmas)
    assert calls['upscale'] == 0


@pytest.mark.parametrize('updates', [{'sampling_contract': 'bad'},
    {'spatial_strategy': 'independent_tiles_exp'}, {'parity_report_json': '{}'},
    {'temporal_chunk_frames': 0}, {'temporal_overlap_frames': -17}])
def test_wrong_plan_rejected(updates):
    with pytest.raises(ValueError):
        _plan(**updates)


def test_high_keyframes_wrong_canvas_fail_before_upscale(harness):
    calls, run = harness
    with pytest.raises(ValueError, match='target canvas'):
        run(conditioning=[[torch.zeros(1), {'minimax_keyframes': [{'latent': torch.zeros(1,24,1,2,2)}]}]])
    assert calls['upscale'] == 0


def test_unknown_sampler_and_clock_mismatch_fail_before_upscale(harness):
    calls, run = harness
    with pytest.raises(ValueError, match='dual-clock Euler'):
        run(sampler=object())
    report = _report()
    report['shift_video'] = 6
    with pytest.raises(ValueError, match='clocks'):
        run(plan=_plan(parity_report_json=json.dumps(report)))
    assert calls['upscale'] == 0


def test_gap_and_modified_locked_overlap_rejected():
    with pytest.raises(ValueError, match='gap'):
        parity._append_exact(torch.zeros(5), torch.zeros(3), 7, 0, 0)
    with pytest.raises(RuntimeError, match='read-only'):
        parity._append_exact(torch.zeros(5), torch.ones(3), 3, 0, 2)


def test_real_dual_clock_euler_roundoff_is_bounded_and_exactly_restored():
    from h3_audio_t8_pkg.sampling import sample_minimax_h3_dual_clock_euler as sample
    rng = torch.Generator().manual_seed(42)
    latent = torch.randn(1,1,1,1,1000, generator=rng)
    noise = torch.randn(latent.shape, generator=rng)
    mask = torch.ones_like(latent)
    mask[...,100:200] = mask[...,600:700] = 0
    class LockedModel:
        def __init__(self):
            self.latent_image, self.noise = latent, noise
        def __call__(self, x, timestep, **kwargs):
            return latent
    sigmas = torch.tensor(parity.UPSTREAM_REFINE_VIDEO_SIGMAS[4])
    initial = sigmas[0]*noise + (1-sigmas[0])*latent
    result = sample(LockedModel(), initial, sigmas, {'denoise_mask':mask}, disable=True,
        video_values=500, packed_values=1000, shift_video=12., shift_audio=3., audio_velocity_is_raw=True)
    assert not torch.equal(result[mask==0], latent[mask==0])
    restored, stats = parity._restore_zero_mask(result, latent, mask)
    assert stats['roundoff_changed_values'] > 0
    assert stats['max_abs_before_exact_restore'] <= 2*torch.finfo(torch.float32).eps
    assert torch.equal(restored[mask==0], latent[mask==0])
    assert torch.equal(restored[mask!=0], result[mask!=0])
    bad = result.clone()
    bad[...,100] += 1e-4
    with pytest.raises(RuntimeError, match='beyond FP32 roundoff'):
        parity._restore_zero_mask(bad, latent, mask)


def test_legacy_builder_default_unchanged():
    old = _plan(sampling_contract='video_only_legacy', parity_report_json='')
    assert old['schema'] == legacy.PLAN_SCHEMA_V1
    assert 'sampling_contract' not in old and 'parity_report' not in old
    assert old['audio_policy'] == 'exact_input_tensor_passthrough'


def test_nonfinite_input_and_unknown_metadata_fail_before_upscale(harness):
    calls, run = harness
    nested = legacy.comfy.nested_tensor.NestedTensor
    v, a = torch.zeros(1,24,15,2,2), torch.zeros(1,32,2,85)
    v[0,0,0,0,0] = float('nan')
    with pytest.raises(ValueError, match='NaN'):
        run(latent={'samples': nested((v, a))})
    v.zero_()
    with pytest.raises(ValueError, match='metadata'):
        run(latent={'samples': nested((v, a)), 'unknown_time_track': [4]})
    assert calls['upscale'] == 0


def test_inherited_video_audio_masks_survive_refinement(harness):
    _, run = harness
    nested = legacy.comfy.nested_tensor.NestedTensor
    v, a = torch.zeros(1,24,15,2,2), torch.zeros(1,32,2,85)
    vm, am = torch.ones(1,1,15,2,2), torch.ones_like(a)
    vm[:,:,:3] = 0
    am[...,:10] = 0
    output, _ = run(latent={'samples': nested((v,a)), 'noise_mask': nested((vm,am))})
    ov, oa = output['samples'].tensors
    assert not ov[:,:,:3].any() and not oa[...,:10].any()
    assert torch.equal(oa[...,10:], torch.full_like(oa[...,10:],3))


def test_public_plan_optional_fields_are_appended_and_default_legacy():
    from h3_audio_t8_pkg.nodes_chunked_two_pass_upscale_advanced import MiniMaxH3ChunkedTwoPassPlanT8Advanced as Node
    schema = Node.GET_NODE_INFO_V1()['input']
    assert list(schema['optional'])[:2] == ['sampling_contract', 'parity_report_json']
    assert list(schema['optional'])[2:] == ['size_mode', 'scale_by', 'target_megapixels',
        'aspect_policy', 'max_anisotropy', 'source_latent']
    assert schema['optional']['sampling_contract'][1]['default'] == 'video_only_legacy'
    assert schema['optional']['parity_report_json'][1]['forceInput']


def test_saved_workflow_is_nonpdd_exact4plus4_and_wiring_correct():
    root = Path(__file__).resolve().parents[1]
    path = root / 'examples/workflows/13-latent-upscale/2026-09-17_H3_NonPDD_Standard_4plus4_Chunked_EXP.json'
    w = json.loads(path.read_text(encoding='utf-8'))
    nodes = {n['id']: n for n in w['nodes']}
    assert not any('PDD' in n['type'] for n in nodes.values())
    assert not any(n['type'] == 'BasicScheduler' for n in nodes.values())
    assert nodes[9]['widgets_values'] == [8,4,4]
    assert nodes[14]['type'] == 'MiniMaxH3ChunkedTwoPassPlanT8Advanced'
    assert nodes[14]['widgets_values'][15] == parity.CONTRACT
    assert nodes[14]['widgets_values'][16:] == ['scale_by', 2.0, .70, 'preserve_source', 1.05]
    assert nodes[14]['widgets_values'][1:5] == [896,448,136,34]
    assert nodes[7]['widgets_values'][3] == nodes[13]['widgets_values'][3] == 192
    assert nodes[7]['widgets_values'][0] == nodes[13]['widgets_values'][0]
    edges = {e[0]: e for e in w['links']}
    assert edges[23][1:5] == [12,1,15,2]  # partial denoised_output, never noisy output0
    assert edges[26][1:5] == [9,1,15,5]   # unmodified original refine4
    assert edges[37][1:5] == [9,2,14,0]   # parity report handshake
    assert edges[39][1:5] == [12,1,14,1]  # actual partial source geometry
    high_inputs = {entry['name']: index for index, entry in enumerate(nodes[13]['inputs'])}
    assert edges[40][1:5] == [14,2,13,high_inputs['width']]  # synchronized HIGH width
    assert edges[41][1:5] == [14,3,13,high_inputs['height']]  # synchronized HIGH height
    assert edges[24][1] == 24 and edges[11][1] == 11
    for eid, source, slot, target, target_slot, _ in w['links']:
        assert nodes[target]['inputs'][target_slot]['link'] == eid
        assert eid in nodes[source]['outputs'][slot]['links']
