"""Actual tiny native H3/Euler continuation. No trained VAE/lifter/media claim."""

import copy
import json

import comfy.samplers
import pytest
import torch

from h3_audio_t8_pkg import progressive_continuation_runtime as continuation
from h3_audio_t8_pkg import progressive_sampling_runtime as runtime
from h3_audio_t8_pkg import long_video
from h3_audio_t8_pkg.sampling import native_flow_sigmas
from helpers import FakeClip, FakeVideoVAE, FakeAudioVAE, make_audio
from test_progressive_sampling_runtime import tiny_model, stub_lifter  # noqa: F401
from test_progressive_continuation import accepted, capture  # noqa: F401


def run(source, model, *, callback=None, options=None, model_hires=None, clip=None,
        prompt='Continue walking.', **sampler_options):
    return continuation.sample_progressive_continuation(source, model, comfy.samplers.ksampler('euler'),
        native_flow_sigmas(8, 12.), clip=clip or FakeClip(), video_vae=FakeVideoVAE(), audio_vae=FakeAudioVAE(),
        prompt=prompt, length=124, upscaler_model='test', seed=9,
        callback=callback, condition_options=options, model_hires=model_hires, **sampler_options)


@pytest.mark.parametrize('frames,steps', [(22, 7), (39, 12)])
@pytest.mark.parametrize('audio_mode', ['native', 'lock_source', 'remix_source'])
def test_actual_native_four_plus_four_uses_distinct_sources_and_completed_high_prefix(
        accepted, stub_lifter, monkeypatch, frames, steps, audio_mode):  # noqa: F811
    source = capture(accepted, context_frames=frames)
    first, second = tiny_model(), tiny_model()
    before = copy.deepcopy(first.model_options)
    calls, payloads = [], []
    original_stage, original_repair = runtime._native_stage, long_video.repair_long_video_payload

    def stage(model, sampler, sigmas, latent, noise, positive, negative, cfg, seed, callback,
              denoise_mask=None, *, preview_phase=None, preview_offset=0, preview_total=None):
        parts = tuple(p.clone() for p in latent.unbind())
        masks = None if denoise_mask is None else tuple(p.clone() for p in denoise_mask.unbind())
        assert (preview_phase, preview_offset, preview_total) == (
            ("low", 0, 8) if not calls else ("high", 4, 8))
        calls.append((model.model, parts, masks))
        return original_stage(model, sampler, sigmas, latent, noise, positive, negative, cfg, seed,
                              callback, denoise_mask=denoise_mask, preview_phase=preview_phase,
                              preview_offset=preview_offset, preview_total=preview_total)

    def repair(out, kwargs):
        result = original_repair(out, kwargs)
        payload = result['minimax_payload'].cond
        layout = payload['layout']
        payloads.append((payload.get('t8_long_video_patch_version'),
            [list(t.shape) for t in payload['cond_video_latents']],
            [float(layout.position_ids[a, 0]) for a, _, kind in layout.segments
             if kind in {'cond', 'ref_audio', 'audio', 'video'}]))
        return result

    monkeypatch.setattr(runtime, '_native_stage', stage)
    monkeypatch.setattr(long_video, 'repair_long_video_payload', repair)
    options = dict(audio_mode=audio_mode, add_source_as_reference=False,
        drive_audio=None if audio_mode == 'native' else make_audio(6))
    output, report_json = run(source, first, options=options, model_hires=second)
    report = json.loads(report_json)
    assert report['counts']['actual_forwards'] == {'low': 4, 'high': 4}
    assert report['continuation']['preparation']['additional_sampling_nfe'] == 0
    assert not report['continuation']['quality_qualified']
    assert calls[0][0] is first.model and calls[1][0] is second.model
    assert len(calls) == 2 and len(stub_lifter) == 1
    # LOW starts empty and free (or audio-only masked): never resize HIGH's
    # known-prefix into LOW. Previous accepted picture is supplied as guides.
    assert torch.count_nonzero(calls[0][1][0]) == 0
    assert calls[0][2] is None or torch.all(calls[0][2][0] == 1)
    assert torch.all(calls[1][2][0][:, :, :steps] == 0)
    assert torch.all(calls[1][2][0][:, :, steps:] == 1)
    _, high, _ = source.prepare_contexts(FakeVideoVAE())
    video, audio = output['samples'].unbind()
    torch.testing.assert_close(video[:, :, :steps], high['video_tail'][:, :, -steps:], rtol=0, atol=2e-7)
    assert torch.isfinite(video).all() and torch.isfinite(audio).all()
    assert payloads and all(p[0] == long_video.LONG_VIDEO_PATCH_VERSION for p in payloads)
    assert all(len(p[1]) == steps for p in payloads)
    assert payloads[0][2] == payloads[-1][2]
    assert payloads[0][1][0][-2:] == [2, 4] and payloads[-1][1][0][-2:] == [4, 8]
    assert first.model_options == before and not first.wrappers and not second.wrappers
    assert not first.object_patches and not second.object_patches


def test_cancellation_and_reuse_preserve_sources_and_base_options(accepted, stub_lifter):  # noqa: F811
    source, model = capture(accepted), tiny_model()
    baseline, _ = run(source, model)

    def cancel(step, *_args):
        if step == 5:
            raise InterruptedError('test cancellation in HIGH')

    with pytest.raises(InterruptedError):
        run(source, model, callback=cancel)
    restored, _ = run(source, model)
    for a, b in zip(restored['samples'].unbind(), baseline['samples'].unbind(), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert not model.wrappers and not model.object_patches
    source.revalidate()


@pytest.mark.parametrize('fault', ['positive', 'latent', 'tensor', 'canvas'])
def test_wrong_prepared_input_is_rejected_before_any_stage(accepted, stub_lifter, fault):  # noqa: F811
    source = capture(accepted)
    low, high, report = source.prepare_conditions(clip=FakeClip(), video_vae=FakeVideoVAE(),
        audio_vae=FakeAudioVAE(), prompt='test', length=124)
    prepared = continuation.PreparedProgressiveContinuation(source, low, high, report)
    positive, latent = high[0], high[1]
    if fault == 'positive':
        positive = copy.deepcopy(positive)
    elif fault == 'latent':
        latent = dict(latent)
    elif fault == 'tensor':
        low[0][0][1]['minimax_keyframes'][0]['latent'].add_(1.)
    with pytest.raises(ValueError):
        runtime.sample_progressive_h3(tiny_model(), positive, positive, latent,
            comfy.samplers.ksampler('euler'), native_flow_sigmas(8, 12.), upscaler_model='test', seed=9,
            low_evaluations=4, low_scale=.75 if fault == 'canvas' else .5,
            input_mode='initialized_av_exp', continuation=prepared)
    assert not stub_lifter


def test_t2va_input_still_rejects_motion_keyframes_without_continuation(accepted, stub_lifter):  # noqa: F811
    source = capture(accepted)
    _, high, _ = source.prepare_conditions(clip=FakeClip(), video_vae=FakeVideoVAE(), audio_vae=FakeAudioVAE(),
                                         prompt='test', length=124)
    with pytest.raises(ValueError, match='T2VA cannot contain keyframe conditioning'):
        runtime.sample_progressive_h3(tiny_model(), high[0], high[0], high[1],
            comfy.samplers.ksampler('euler'), native_flow_sigmas(8, 12.), upscaler_model='test', seed=9,
            input_mode='initialized_av_exp')


@pytest.mark.parametrize('fault', ['prompt', 'mux_audio', 'report'])
def test_prepared_delivery_description_is_content_bound(accepted, fault):  # noqa: F811
    source = capture(accepted)
    audio = make_audio(6)
    low, high, info = source.prepare_conditions(clip=FakeClip(), video_vae=FakeVideoVAE(),
        audio_vae=FakeAudioVAE(), prompt='Keep walking.', length=124, final_audio=audio)
    prepared = continuation.PreparedProgressiveContinuation(source, low, high, info)
    description = prepared.verify()['prepared_delivery']
    assert description['conditioned_prompt'] == high[3]
    assert description['mux_audio_identity'] == continuation._input_identity(audio)
    if fault == 'mux_audio':
        audio['waveform'][..., 0] += .1
    else:
        changed = list(prepared.high)
        if fault == 'prompt':
            changed[3] += 'Not encoded.'
        else:
            changed[5] = json.dumps({'not_the_original': True})
        prepared.high = tuple(changed)
    with pytest.raises(ValueError, match='prepared inputs changed'):
        prepared.verify()


@pytest.mark.parametrize('frames,steps', [(22, 7), (39, 12)])
def test_eav_reports_real_motion_payload_and_native_masks_without_altering_report_only(
        accepted, stub_lifter, frames, steps):  # noqa: F811
    source = capture(accepted, context_frames=frames)
    first, second = tiny_model(), tiny_model()
    last = {}

    def capture_final(step, prediction, state, total):
        if step == total - 1:
            last['prediction'] = prediction.unbind()[0].detach().clone()
            last['state'] = state.unbind()[0].detach().clone()

    def assert_native_final(output):
        # Core Euler does not return denoised directly on its final step:
        # x + ((x - denoised) / sigma) * (-sigma) has fp32 cancellation.
        # Reconstruct the actual final update, in the SAME operation order,
        # from the real callback. Do not clamp the sampler or relax equality.
        sigma = native_flow_sigmas(8, 12.)[-2].to(last['state'])
        expected = last['state'] + ((last['state'] - last['prediction']) / sigma) * (-sigma)
        latent_format = second.get_model_object('latent_format')
        torch.testing.assert_close(output['samples'].unbind()[0],
                                   latent_format.process_out(expected), rtol=0, atol=0)
        _, high, _ = source.prepare_contexts(FakeVideoVAE())
        anchor = latent_format.process_in(high['video_tail'][:, :, -steps:].float())
        torch.testing.assert_close(last['prediction'][:, :, :steps], anchor, rtol=0, atol=0)

    baseline, _ = run(source, first, model_hires=second, callback=capture_final)
    assert_native_final(baseline)
    diagnostic, text = run(source, first, model_hires=second, eav_mode='report_only')
    for a, b in zip(baseline['samples'].unbind(), diagnostic['samples'].unbind(), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    report = json.loads(text)
    for phase in ('low', 'high'):
        stage = report['eav'][phase]
        assert stage['verified_native_mask_forwards'] == 4
        assert stage['config']['long_video_contract']['context_frames'] == frames
        assert stage['config']['progressive_mask_contract']['accepted_source_sha256'] == source.sha256
        assert stage['config']['task_scope'] == ['LongVideoMotion']
    applied, text = run(source, first, model_hires=second, eav_mode='apply_exp', callback=capture_final)
    assert_native_final(applied)
    report = json.loads(text)
    assert report['eav']['summary']['full_schedule_nfe'] == 8
    assert report['eav']['summary']['output_gain_above_one_applied']
    av = applied['samples'].unbind()[0]
    bv = baseline['samples'].unbind()[0]
    assert not torch.equal(av[:, :, steps:], bv[:, :, steps:])


@pytest.mark.parametrize('frames', [22, 39])
@pytest.mark.parametrize('route', ['video_only_paper', 'joint_av_exp'])
def test_projected_relay_and_eav_execute_native_motion_stages(accepted, stub_lifter, frames, route):  # noqa: F811
    from test_prompt_relay_long_video_advanced import NativeLikeFakeClip
    from h3_audio_t8_pkg import prompt_relay_advanced as relay
    plan = relay.build_prompt_relay_plan('Scene.', 'Walk.\nStop.\nTurn.', 345,
        'auto_equal', '', 'paper_v1', .1, False, False)[0]
    plan['query_route'] = route
    plan.pop('plan_hash')
    plan['plan_hash'] = relay._sha256_json(plan)
    original = copy.deepcopy(plan)
    source = capture(accepted, context_frames=frames)
    first, second = tiny_model(), tiny_model()
    expected_calls = {'forward': 4, 'routed_attention': 4 * len(first.model.diffusion_model.blocks)}
    options = dict(clip=NativeLikeFakeClip(), prompt=None, prompt_relay_plan=plan,
                   query_chunk_rows=64, model_hires=second)
    baseline, text = run(source, first, **options)
    report = json.loads(text)
    low, high = (report['prompt_relay'][phase] for phase in ('low', 'high'))
    assert low['binding_hash'] != high['binding_hash']
    assert low['events'] == high['events']
    assert low['projection'] == high['projection']
    assert low['projection']['render_start_frame'] == 124 - frames
    assert low['projection']['accepted_start_frame'] == 124
    for event, global_event in zip(low['events'], plan['events'], strict=True):
        assert event['midpoint'] == pytest.approx(global_event['midpoint'] - (124 - frames) * 5 / 3)
        assert event['sigma'] == global_event['sigma']
    for stage in (low, high):
        assert stage['completed_calls'] == expected_calls
        assert stage['accepted_source_sha256'] == source.sha256
        assert stage['query_route'] == route
    diagnostic, _ = run(source, first, **options, eav_mode='report_only')
    for a, b in zip(baseline['samples'].unbind(), diagnostic['samples'].unbind(), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    applied, text = run(source, first, **options, eav_mode='apply_exp')
    report = json.loads(text)
    assert report['eav']['summary']['full_schedule_nfe'] == 8
    assert report['eav']['summary']['output_gain_above_one_applied']
    for phase in ('low', 'high'):
        assert report['eav'][phase]['verified_native_mask_forwards'] == 4
        assert report['prompt_relay'][phase]['completed_calls'] == expected_calls
    assert all(torch.isfinite(x).all() for x in applied['samples'].unbind())
    assert plan == original
    assert not first.wrappers and not second.wrappers
