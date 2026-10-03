"""Actual old composites + exact full-source post-restriction, no diffusion."""
from dataclasses import replace
import json
import math

import pytest
import torch

from h3_audio_t8_pkg import visible_face_mask as runtime
from h3_audio_t8_pkg.nodes_visible_face_mask import NODES


def fixture(dtype=torch.float32):
    base = torch.linspace(.1, .7, 5 * 8 * 12 * 3).reshape(5, 8, 12, 3).to(dtype)
    alpha = torch.zeros((5, 8, 12))
    alpha[:, 1:7, 2:10] = .25
    candidate = base + .2 * alpha[..., None].to(dtype)
    visible = torch.zeros_like(alpha)
    visible[:, 2:6, 3:9] = .5
    visible[:, 3:5, 4:8] = 1
    return base, alpha, candidate, visible


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_endpoints_no_double_alpha_and_audio_same_object(dtype):
    base, alpha, candidate, visible = fixture(dtype)
    audio = {'waveform': torch.randn(1, 2, 4411), 'sample_rate': 44100}
    audio_before = audio['waveform'].clone()
    originals = [tensor.clone() for tensor in (base, alpha, candidate, visible)]
    binding = runtime.capture(base, visible, source_start_frame=124)
    output, returned_audio, effective, report_json = runtime.composite(binding, candidate, alpha, audio)
    expected = (base.to(torch.float64 if dtype == torch.float64 else torch.float32) +
                (candidate.to(torch.float64 if dtype == torch.float64 else torch.float32) -
                 base.to(torch.float64 if dtype == torch.float64 else torch.float32)) * visible[..., None]).to(dtype)
    expected = torch.where((visible == 1)[..., None], candidate, expected)
    expected = torch.where((effective > 0)[..., None], expected, base)
    assert torch.equal(output, expected)
    assert torch.equal(output[effective == 0], base[effective == 0])
    assert torch.equal(output[(visible == 1) & (alpha > 0)], candidate[(visible == 1) & (alpha > 0)])
    assert torch.equal(effective, alpha * visible)
    if dtype == torch.float32:
        selected = (visible == .5) & (alpha > 0)
        assert not torch.equal(output[selected], (base + (candidate - base) * alpha[..., None] * visible[..., None])[selected])
    report = json.loads(report_json)
    assert report['source_mask_binding']['frame_interval'] == [124, 129]
    assert report['sampling_nfe'] == 0 and report['automatic_accept'] is False
    assert report['occlusion_semantic_quality_verified'] is False
    assert report['original_alpha_not_applied_twice'] is True
    assert returned_audio is audio and torch.equal(audio['waveform'], audio_before)
    assert all(torch.equal(x, old) for x, old in zip((base, alpha, candidate, visible), originals, strict=True))


def test_black_returns_original_and_white_returns_candidate_exact():
    base, alpha, candidate, visible = fixture()
    for value, expected in [(0, base), (1, candidate)]:
        out, *_ = runtime.composite(runtime.capture(base, torch.full_like(visible, value)), candidate, alpha)
        assert torch.equal(out, expected)


def test_small_positive_support_is_not_epsilon_clipped_or_cast_to_RGB_dtype():
    base = torch.zeros((1, 2, 2, 3), dtype=torch.float16)
    alpha = torch.full((1, 2, 2), 1e-8)
    candidate = torch.full_like(base, .5)
    visible = torch.ones_like(alpha)
    out, _, effective, _ = runtime.composite(runtime.capture(base, visible), candidate, alpha)
    assert torch.equal(out, candidate) and bool((effective > 0).all())
    assert effective.dtype == torch.float32


def test_one_frame_mask_requires_explicit_broadcast_and_receipt_binds_actual_single_frame():
    base, alpha, candidate, visible = fixture()
    with pytest.raises(ValueError, match='broadcast'):
        runtime.capture(base, visible[:1])
    binding = runtime.capture(base, visible[:1], broadcast_single_mask=True)
    full = runtime.capture(base, visible)
    assert binding.verify()['visible_mask']['shape'] == [1, 8, 12]
    assert torch.equal(runtime.composite(binding, candidate, alpha)[0], runtime.composite(full, candidate, alpha)[0])


@pytest.mark.parametrize('clock_request', [{'fps': 23.976}, {'fps': True}, {'fps': float('nan')},
                                   {'source_start_frame': .0}, {'source_start_frame': True},
                                   {'source_start_frame': -1}, {'broadcast_single_mask': 1}])
def test_no_implicit_clock_or_boolean_conversion(clock_request):
    base, _, _, visible = fixture()
    with pytest.raises(ValueError):
        runtime.capture(base, visible, **clock_request)


@pytest.mark.parametrize('kind', ['source_hidden_pixel', 'visible', 'request', 'receipt', 'seal', 'implementation'])
def test_actual_full_bytes_and_exact_receipt_are_verified_on_every_use(kind, monkeypatch):
    base, alpha, candidate, visible = fixture()
    binding = runtime.capture(base, visible)
    if kind == 'source_hidden_pixel':
        base[-1, -1, -1, -1] += .001
    elif kind == 'visible':
        visible[-1, -1, -1] = .3
    elif kind == 'request':
        req = json.loads(binding.request_json)
        req['source_start_frame'] = 17
        binding = replace(binding, request_json=runtime.canonical(req))
    elif kind == 'receipt':
        receipt = json.loads(binding.contract_json)
        receipt['sampling_invalidated'] = True
        binding = replace(binding, contract_json=runtime.canonical(receipt))
    elif kind == 'seal':
        binding = replace(binding, contract_sha256='0' * 64)
    else:
        actual = runtime._implementation()
        monkeypatch.setattr(runtime, '_implementation', lambda: actual | {'new.py': '1' * 64})
    with pytest.raises(ValueError, match='bind again'):
        runtime.composite(binding, candidate, alpha)


@pytest.mark.parametrize('kind', ['candidate_short', 'candidate_width', 'candidate_dtype', 'mask_short',
                                 'mask_channel', 'mask_nan', 'mask_above_one', 'mask_negative', 'source_rgba',
                                 'source_integer', 'source_nan', 'candidate_nan', 'candidate_outside_alpha'])
def test_no_partial_window_crop_pad_resize_clamp_or_false_exterior(kind):
    base, alpha, candidate, visible = fixture()
    if kind == 'source_rgba':
        base = torch.cat((base, base[..., :1]), -1)
    elif kind == 'source_integer':
        base = (base * 255).to(torch.uint8)
    elif kind == 'source_nan':
        base[0, 0, 0, 0] = float('nan')
    with pytest.raises(ValueError):
        binding = runtime.capture(base, visible)
        if kind == 'candidate_short':
            candidate = candidate[:-1]
        elif kind == 'candidate_width':
            candidate = candidate[:, :, :-1]
        elif kind == 'candidate_dtype':
            candidate = candidate.double()
        elif kind == 'mask_short':
            alpha = alpha[:-1]
        elif kind == 'mask_channel':
            alpha = alpha[..., None]
        elif kind == 'mask_nan':
            alpha[0, 0, 0] = float('nan')
        elif kind == 'mask_above_one':
            alpha[0, 0, 0] = 1.1
        elif kind == 'mask_negative':
            alpha[0, 0, 0] = -.1
        elif kind == 'candidate_nan':
            candidate[0, 0, 0, 0] = float('nan')
        elif kind == 'candidate_outside_alpha':
            candidate[0, 0, 0, 0] += .1
        runtime.composite(binding, candidate, alpha)


@pytest.mark.parametrize('route', ['standard', 'parity'])
def test_actual_unchanged_old_stitch_output_is_restricted_without_alpha_squared(route):
    # Real original crop/plan/stitch functions, not a fabricated stitch mock.
    base = torch.rand((5, 64, 96, 3), generator=torch.Generator().manual_seed(7007)) * .6 + .1
    if route == 'standard':
        from test_face_refine_advanced import _plan
        from h3_audio_t8_pkg.face_refine_advanced import stitch_face_refine_candidate
        plan, crops, *_ = _plan(base)
        candidate, changed, *_ = stitch_face_refine_candidate(
            base, crops + .03, plan, 'ellipse', 8., .5, .0, 1., 0, 'cpu_memory_safe')
    else:
        from test_face_refine_parity_advanced import _plan
        from h3_audio_t8_pkg.face_refine_parity_advanced import stitch_face_refine_parity_candidate
        plan, crops, *_ = _plan(base)
        candidate, changed, *_ = stitch_face_refine_parity_candidate(
            base, crops + .03, plan, 'face_only', 0, 8., .0, .5, 'fade_out', 1., 'cpu_memory_safe')
    assert bool((changed > 0).any()) and not torch.equal(candidate, base)
    visible = torch.ones_like(changed)
    visible[:, :, 48:] = 0
    binding = runtime.capture(base, visible)
    output, _, effective, _ = runtime.composite(binding, candidate, changed)
    assert torch.equal(output[effective == 0], base[effective == 0])
    assert torch.equal(output[effective > 0], candidate[effective > 0])


def test_nodes_are_additive_no_sampling_no_cache_and_exactly_one_support_input():
    schemas = [node.define_schema() for node in NODES]
    assert [s.node_id for s in schemas] == ['MiniMaxH3VisibleFaceMaskBindEXPT8', 'MiniMaxH3VisibleFaceCompositeEXPT8']
    assert all(s.is_experimental for s in schemas)
    assert [item.id for item in schemas[0].inputs] == ['source_images', 'visible_mask', 'fps', 'source_start_frame', 'broadcast_single_mask']
    assert schemas[0].inputs[-1].default is False
    assert schemas[1].inputs[2].optional is True
    assert all(math.isnan(node.fingerprint_inputs()) for node in NODES)
    base, alpha, candidate, visible = fixture()
    binding = NODES[0].execute(base, visible).result[0]
    out = NODES[1].execute(binding, candidate, alpha).result
    assert torch.equal(out[0], runtime.composite(binding, candidate, alpha)[0])
    with pytest.raises(ValueError, match='exactly one'):
        NODES[1].execute(binding, candidate)
    with pytest.raises(ValueError, match='exactly one'):
        NODES[1].execute(binding, candidate, alpha, multiface_composite={})


def test_multiface_support_contract_not_double_blend_and_no_fake_window():
    from h3_audio_t8_pkg.multiface_refine_advanced import COMPOSITE_SCHEMA
    from h3_audio_t8_pkg.face_refine_advanced import source_proxy_sha256
    base, alpha, candidate, visible = fixture()
    binding = runtime.capture(base, visible)
    state = {'schema': COMPOSITE_SCHEMA, 'source_proxy_sha256': source_proxy_sha256(base),
             'frames': candidate, 'applied_mask': alpha > 0, 'automatic_accept': False, 'applied': []}
    actual = NODES[1].execute(binding, candidate, multiface_composite=state).result
    expected = runtime.composite(binding, candidate, alpha > 0)
    assert torch.equal(actual[0], expected[0])
    for bad in [state | {'frames': candidate[:-1]}, state | {'applied_mask': alpha},
                state | {'source_proxy_sha256': '0' * 64}, state | {'automatic_accept': True}]:
        with pytest.raises(ValueError):
            NODES[1].execute(binding, candidate, multiface_composite=bad)
