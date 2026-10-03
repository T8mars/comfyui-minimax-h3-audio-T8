"""Actual external media and real tiny Core motion forwards; not trained quality."""
from copy import deepcopy
from dataclasses import replace, asdict
import json

import pytest
import torch
from comfy_extras.nodes_custom_sampler import BasicGuider, RandomNoise, SamplerCustomAdvanced

from h3_audio_t8_pkg import external_continuation as bridge, external_continuation_effects as effects
from h3_audio_t8_pkg import prompt_relay_advanced as relay, long_video, sampling
from h3_audio_t8_pkg import nodes_external_continuation_effects as public
from h3_audio_t8_pkg.modular_sampling import native_explicit, eav
from h3_audio_t8_pkg.modular_sampling.results import selected_model_identity
from h3_audio_t8_pkg.nodes_external_continuation import NODES as original_nodes
from h3_audio_t8_pkg.nodes_prompt_relay_long_video_advanced import MiniMaxH3PromptRelayLongVideoConditioningT8Advanced
from test_external_continuation import prepared as _prepared, stereo as _stereo, adopt
from test_fast_h3_v2_core_sampler import model
from test_prompt_relay_long_video_advanced import NativeLikeFakeClip
from helpers import FakeVideoVAE, FakeAudioVAE

prepared = _prepared
stereo = _stereo


def inputs(prepared, *, applied=False):
    context = bridge.prepare_context(adopt(prepared), FakeVideoVAE(), 96, 64)
    clip = NativeLikeFakeClip()
    plan = relay.build_prompt_relay_plan('Scene.', 'Walk.\nTurn.', 39,
        'auto_equal', '', 'paper_v1', .1, False, False)[0]
    projected = effects.project_relay(context, plan, 22, 5)
    options = dict(task_type='auto', audio_mode='native', audio_denoise_strength=.35,
        add_source_as_reference=True, prompt_primary_audio_ordinal=0, strict_prompt_tags=True,
        ref_image_size='match', reference_video_policy='official_2_to_15s',
        execution_mode='apply_exp' if applied else 'report_only', query_chunk_rows=64)
    result = effects.relay_condition(context, projected, model=model(), clip=clip,
        video_vae=context.video_vae, audio_vae=FakeAudioVAE(), length=22, **options)
    return context, projected, result


def configured(result, context):
    selected, positive, source = result[:3]
    scoped, scope, _ = effects.bind_scope(context, selected, positive, source)
    prepared, sampler, sigmas = sampling.setup_dual_clock_sampling(scoped, source, 4, 12., 3.)
    bound, sampler, sigmas, stage, _ = native_explicit.bind_stage(prepared, sampler, sigmas, source)
    return bound, sampler, sigmas, source, stage, positive, scope


def sample(args, selected=None, positive=None):
    bound, sampler, sigmas, source, _stage, conditions, _scope = args
    return SamplerCustomAdvanced.execute(RandomNoise.execute(38).result[0],
        BasicGuider.execute(selected or bound, positive or conditions).result[0], sampler, sigmas, source).result


def test_explicit_window_keeps_global_time_and_does_not_claim_native_ancestor(prepared):
    context, projected, result = inputs(prepared)
    report = projected.verify()
    assert report['generated_start_frame'] == 5
    assert report['context']['source']['frame_interval'] == [5, 10]
    assert report['projected']['long_video_projection']['render_start_frame'] == 0
    assert report['independent_relay_clock_not_external_movie_or_native_parent_clock']
    shifted = effects.project_relay(context, projected.global_plan, 22, 10)
    assert shifted.projected['long_video_projection']['render_start_frame'] == 5
    assert report['sha256'] != shifted.verify()['sha256']
    assert json.loads(result[-1])['external_motion_effects']['automatic_accept'] is False
    assert json.loads(result[-1])['status'] == 'report_only'
    assert not result[0].get_wrappers('diffusion_model', relay.PROMPT_RELAY_WRAPPER_KEY)
    assert result[0].object_patches['extra_conds'] is not None


@pytest.mark.parametrize('mode', ['disabled', 'report_only', 'apply_exp'])
@pytest.mark.parametrize('applied_relay', [False, True])
def test_actual_four_motion_forwards_composed_relay_eav_math_clock_and_masks(prepared, mode, applied_relay):
    context, _projected, result = inputs(prepared, applied=applied_relay)
    args = configured(result, context)
    bound, _sampler, sigmas, source, stage, _positive, scope = args
    before = effects._identity(source)
    baseline = sample(args)
    selected, runtime, raw = public.MiniMaxH3ExternalStageEAVApplyEXPT8.execute(
        bound, sigmas, source, stage, eav.EAVConfig(mode, .2, 0., 1., 32, 3.), scope).result
    actual = sample(args, selected)
    if mode == 'disabled':
        assert selected is bound and runtime.snapshot()['status'] == 'disabled_identity'
    else:
        report = runtime.snapshot()
        assert report['completed_forwards'] == report['selector_calls'] == 4
        assert report['clock_match'] is True
        assert report['status'] == ('observed_report_only' if mode == 'report_only' else 'observed_apply_exp')
        assert report['relay_required'] is applied_relay
        assert report['relay_attention_calls'] == (4 if applied_relay else 0)
        assert json.loads(raw)['long_video_contract']['segment_index'] == 1
        assert json.loads(raw)['external_motion_scope_sha256'] == scope.verify()['sha256']
    if mode != 'apply_exp':
        assert all(torch.equal(a, b) for a, b in zip(actual[0]['samples'].unbind(), baseline[0]['samples'].unbind()))
    else:
        assert not torch.equal(actual[0]['samples'].unbind()[0], baseline[0]['samples'].unbind()[0])
    assert effects._identity(source) == before
    assert context.verify()['source']['has_sampling_ancestor'] is False
    # Actual fake encoder executes normally but cannot become a portable VAE certificate.
    assert selected_model_identity(selected)['portable_cache_reuse'] is False
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('mutation', ['global', 'projected', 'clock', 'length', 'context'])
def test_projection_context_plan_and_explicit_clock_mutations_reject(prepared, mutation):
    context, projected, _result = inputs(prepared)
    if mutation == 'global':
        projected.global_plan['compiled_prompt'] += ' changed'
    elif mutation == 'projected':
        projected.projected['compiled_prompt'] += ' changed'
    elif mutation == 'context':
        context.context['video_tail'] = context.context['video_tail'].clone()
        context.context['video_tail'].flatten()[0] += .2
    else:
        projected = replace(projected, **({'generated_start_frame': 6} if mutation == 'clock' else {'length': 39}))
    with pytest.raises(ValueError):
        projected.verify()


@pytest.mark.parametrize('mutation', ['positive', 'motion', 'source', 'mask', 'context', 'project'])
def test_late_source_conditions_masks_project_mutation_cannot_reuse_scope(prepared, mutation):
    context, _projected, result = inputs(prepared)
    args = configured(result, context)
    if mutation == 'positive':
        result[1][0][0].flatten()[0] += .2
    elif mutation == 'motion':
        result[1][0][1]['minimax_keyframes'][0][long_video.MOTION_FRAME_INDEX] += 1
    elif mutation == 'source':
        result[2]['samples'].unbind()[0].flatten()[0] += .2
    elif mutation == 'mask':
        assert 'noise_mask' not in result[2]  # unchanged original motion conditioning
        result[2]['noise_mask'] = torch.ones_like(result[2]['samples'].unbind()[0])
    elif mutation == 'context':
        context.context['metadata']['width'] += 32
    else:
        store, project, *_rest = prepared
        saved = store.load(project['id'])
        saved['doc']['shots'][0]['simplePrompt'] = 'changed'
        store.save(saved, saved['revision'])
    with pytest.raises(ValueError):
        args[-1].verify(args[0], args[3])


def test_wrong_scope_stage_av_native_parent_or_runtime_guides_reject(prepared):
    context, projected, result = inputs(prepared)
    args = configured(result, context)
    bound, _sampler, sigmas, source, stage, positive, scope = args
    with pytest.raises(ValueError, match='different MODEL'):
        scope.verify(model())
    altered = {**source, 'noise_mask': None}
    with pytest.raises(ValueError, match='stage AV'):
        scope.verify(bound, altered)
    with pytest.raises(TypeError):
        eav.apply_stage_eav(bound, sigmas, source, stage, eav.EAVConfig(), external_scope=context)
    with pytest.raises(TypeError):
        eav.apply_stage_eav(bound, sigmas, source, stage, eav.EAVConfig(), external_scope=scope, p7_phase=object())
    payload = dict(keyframes=positive[0][1]['minimax_keyframes'], refs=[], frame_count=22)
    scope.validate_payload(payload)
    with pytest.raises(RuntimeError, match='runtime guides'):
        scope.validate_payload({**payload, 'keyframes': []})
    with pytest.raises(ValueError):
        effects.project_relay(context, projected.global_plan, 22, True)
    with pytest.raises(ValueError):
        effects.project_relay(context, projected.global_plan, 22, 4)


def test_append_only_schemas_retain_original_condition_inputs_defaults_and_outputs():
    assert [node.__name__ for node in original_nodes] == ['MiniMaxH3ExternalContinuationSourceEXPT8',
        'MiniMaxH3ExternalContextEncodeEXPT8', 'MiniMaxH3ExternalContinuationConditioningEXPT8']
    assert [node.define_schema().node_id for node in public.NODES] == [node.__name__ for node in public.NODES]
    old = MiniMaxH3PromptRelayLongVideoConditioningT8Advanced.define_schema()
    new = public.MiniMaxH3ExternalRelayConditioningEXPT8.define_schema()
    removed = {'context', 'prompt_relay_plan', 'segment_index', 'context_frames', 'context_audio', 'width', 'height'}
    expected = asdict(old.get_v1_info(MiniMaxH3PromptRelayLongVideoConditioningT8Advanced))
    current = asdict(new.get_v1_info(public.MiniMaxH3ExternalRelayConditioningEXPT8))
    assert expected['output'] == current['output'] and expected['output_name'] == current['output_name']
    for section, entries in expected['input'].items():
        for name, original in entries.items():
            if name in removed:
                continue
            original = deepcopy(original)
            if name == 'length':
                original[1]['forceInput'] = False
            assert current['input'][section][name] == original


@pytest.mark.parametrize('count', [5, 22, 39])
def test_actual_stereo_context_guides_and_plain_external_EAV_only(count, stereo):
    store, project, request, output, _path = stereo
    source = bridge.capture_source(store, output, project_id=project['id'], shot_id=project['current'],
        take_id=request['take_id'], context_frames=count, audio_policy='reencode_stereo_pcm_context')

    class AudioVAE:
        audio_sample_rate = 48000

        def encode(self, waveform):
            return torch.ones(1, 32, 2, round(waveform.shape[1] / 1200))

    audio = AudioVAE()
    context = bridge.prepare_context(source, FakeVideoVAE(), 96, 64, audio)
    result = bridge.condition(context, clip=NativeLikeFakeClip(), video_vae=context.video_vae,
        audio_vae=audio, prompt='Scene.', length=count+17)
    selected = long_video.patch_long_video_model(model())
    scoped, scope, _ = public.MiniMaxH3ExternalMotionEffectsBindEXPT8.execute(context, selected,
                                                                           result[0], result[1]).result
    prepared_model, sampler, sigmas = sampling.setup_dual_clock_sampling(scoped, result[1], 4, 12., 3.)
    bound, sampler, sigmas, stage, _ = native_explicit.bind_stage(prepared_model, sampler, sigmas, result[1])
    args = bound, sampler, sigmas, result[1], stage, result[0], scope
    before = effects._identity(result[1])
    original = sample(args)
    enhanced, runtime, _ = eav.apply_stage_eav(bound, sigmas, result[1], stage,
        eav.EAVConfig('report_only', .2, 0., 1., 32, 3.), external_scope=scope)
    actual = sample(args, enhanced)
    assert runtime.snapshot()['status'] == 'observed_report_only'
    assert runtime.snapshot()['completed_forwards'] == 4
    assert not runtime.snapshot()['relay_required']
    assert all(torch.equal(a,b) for a,b in zip(original[0]['samples'].unbind(),actual[0]['samples'].unbind()))
    assert effects._identity(result[1]) == before
    assert len(result[0][0][1]['minimax_refs']) == 1
    assert scope.verify()['context']['source']['frame_interval'] == [60-count, 60]
    assert scope.verify()['has_sampling_ancestor'] is False


def test_actual_runtime_different_motion_guides_abort_not_merely_same_shape(prepared):
    context, _projected, result = inputs(prepared)
    args = configured(result, context)
    bound, _sampler, sigmas, source, stage, positive, scope = args
    selected, runtime, _ = eav.apply_stage_eav(bound, sigmas, source, stage,
        eav.EAVConfig('report_only', .2, 0., 1., 32, 3.), external_scope=scope)
    different = deepcopy(positive)
    different[0][1]['minimax_keyframes'][0]['latent'] = different[0][1]['minimax_keyframes'][0]['latent'].clone()+.1
    with pytest.raises(RuntimeError, match='runtime guides'):
        sample(args, selected, different)
    assert runtime.snapshot()['status'] == 'aborted'


def test_changed_effect_implementation_cannot_revalidate_own_scope(prepared, monkeypatch):
    context, _projected, result = inputs(prepared)
    args = configured(result, context)
    original = effects._identity
    monkeypatch.setattr(effects, '_identity', lambda value: original(value))
    with pytest.raises(effects.UnverifiedModelStack, match='unknown executable closure'):
        args[-1].verify(args[0], args[3])
