"""Additive cast/cue/full-content contracts; mock loop is not GPU qualification."""
from contextlib import nullcontext
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch
import comfy.sd

from h3_audio_t8_pkg import mv_cast_solo as cast, mv_lipsync_advanced as mv
from h3_audio_t8_pkg.long_video_dual_identity import content_identity
from h3_audio_t8_pkg.nodes_mv_cast_solo import NODES


def audio(seconds=12.):
    time = torch.arange(round(seconds * 8000)) / 8000
    return {'sample_rate': 8000, 'waveform': (.2 * torch.sin(time * 2 * torch.pi * 220))[None, None]}


def plan():
    song = audio()
    scene = mv.build_mv_vocal_lock_scene_plan(song, song,
        min_scene_seconds=5., target_scene_seconds=6., max_scene_seconds=8.,
        analysis_hop_ms=50, vocal_active_ratio=.12, manual_boundaries_json='[6]')[0]
    assert scene['scene_count'] == 2
    result, _ = cast.build_plan(scene, 'a woman in red', 'a man in blue',
        json.dumps([{'scene_index': 0, 'performer_id': 'A', 'exact_vocal_text': ' First line\n第二行。 '},
                    {'scene_index': 1, 'performer_id': 'B', 'exact_vocal_text': 'His exact words.'}]),
        'A quiet studio performance.', 'cinematic realism')
    return result


def rehash(value):
    value.pop('prompt_plan_hash', None)
    value['prompt_plan_hash'] = mv._hash(value)
    return value


def test_explicit_two_cast_cues_literal_lyrics_and_original_clock():
    value = plan()
    assert cast.validate_plan(value) == value
    assert [row['performer_id'] for row in value['segments']] == ['A', 'B']
    for item, scene in zip(value['segments'], value['scene_plan']['scenes'], strict=True):
        assert (item['start_frame'], item['end_frame']) == (scene['start_frame'], scene['end_frame'])
        assert item['reset_policy'] == 'independent_ref2va_no_previous_tail'
        assert mv._SINGLE_SUBJECT_VISUAL_CONTRACT in item['prompt']
        assert '[English] ' + item['exact_vocal_text'] in item['prompt']
    assert value['audio_contract']['delivery'] == 'full_song_muxed_once'


@pytest.mark.parametrize('edit', [
    lambda value: value['segments'][1].update(performer_id='A'),
    lambda value: value['segments'][1].update(start_frame=0),
    lambda value: value['segments'][0].update(exact_vocal_text='changed words'),
    lambda value: value['segments'][1].update(cue_id='guess'),
    lambda value: value['segments'][1].update(reset_policy='previous_tail'),
    lambda value: value['performers'].__setitem__(0, None),
    lambda value: value['segments'].__setitem__(0, None),
    lambda value: value['performers'][1].update(description='a different actor'),
    lambda value: value.update(vocal_language='Chinese'),
])
def test_rehashed_inconsistent_cast_cue_or_literal_prompt_rejected(edit):
    value = plan()
    edit(value)
    with pytest.raises(ValueError):
        cast.validate_plan(rehash(value))


@pytest.mark.parametrize('assignments', [
    '[]', '[null,null]',
    '[{"scene_index":0,"scene_index":0,"performer_id":"A","exact_vocal_text":""},{}]',
    '[{"scene_index":true,"performer_id":"A","exact_vocal_text":""},{}]',
    '[{"scene_index":0,"performer_id":"A","exact_vocal_text":"<Picture 2>"},{}]',
    '[{"scene_index":0,"performer_id":"A","exact_vocal_text":""},{"scene_index":1,"performer_id":"A","exact_vocal_text":""}]',
])
def test_no_guessed_assignment_duplicate_fields_control_tags_or_single_cast(assignments):
    with pytest.raises(ValueError):
        cast.build_plan(plan()['scene_plan'], 'A actor', 'B actor', assignments, 'studio', 'realistic')


@pytest.mark.parametrize('bad', [torch.zeros(2, 32, 32, 3), torch.zeros(1, 32, 32, 2),
    torch.zeros(1, 15, 32, 3), torch.full((1, 32, 32, 3), float('nan')),
    torch.full((1, 32, 32, 3), -0.1), torch.full((1, 32, 32, 3), 1.1)])
def test_reference_image_alignment_is_explicit(bad):
    with pytest.raises(ValueError):
        cast.references(torch.zeros(1, 32, 32, 3), bad)


def test_separate_reference_canvases_are_not_implicitly_stacked_or_resized():
    first, second = torch.zeros(1, 32, 32, 3), torch.ones(1, 48, 64, 4)
    refs = cast.references(first, second)
    assert refs['A'] is first and refs['B'] is second


def fake_producers(monkeypatch):
    from h3_audio_t8_pkg.modular_sampling import results
    from h3_audio_t8_pkg import progressive_producers
    model = SimpleNamespace(weights=torch.zeros(7))
    components = [object.__new__(comfy.sd.CLIP), object.__new__(comfy.sd.VAE), object.__new__(comfy.sd.VAE)]
    for value in components:
        value.test_weights = torch.ones(7)
    calls = []
    monkeypatch.setattr(results, 'selected_model_identity', lambda value: {'state': content_identity(value.weights)})
    def producer(value, role):
        calls.append(role)
        return {'role': role, 'state': content_identity(value.test_weights),
                'crop_input': getattr(value, 'crop_input', None)}
    monkeypatch.setattr(progressive_producers, '_native_producer_description', producer)
    return model, components, calls


@pytest.mark.parametrize('target', ['A', 'B', 'song', 'vocal', 'model', 'clip', 'video_vae', 'audio_vae'])
def test_binding_uses_full_input_content_and_actual_selected_producers(monkeypatch, target):
    model, components, calls = fake_producers(monkeypatch)
    refs = cast.references(torch.zeros(1, 32, 32, 3), torch.ones(1, 32, 32, 3))
    song, vocal = audio(), audio()
    binding = cast.CastRuntimeBinding(model, *components, refs, song, vocal)
    assert calls == ['clip', 'video_vae', 'audio_vae']
    binding.verify()
    if target in refs:
        # Pixel1 is outside the legacy64-value sample used by earlier routes.
        before = mv._media_signature(refs[target])
        refs[target].flatten()[1] = .25
        assert mv._media_signature(refs[target]) == before
    elif target in {'song', 'vocal'}:
        value = song if target == 'song' else vocal
        before = mv._media_signature(value)
        value['waveform'].flatten()[1] = .45
        assert mv._media_signature(value) == before
    elif target == 'model':
        model.weights[1] = .5
    else:
        components[{'clip': 0, 'video_vae': 1, 'audio_vae': 2}[target]].test_weights[1] = .5
    with pytest.raises(ValueError, match='content changed'):
        binding.verify()


def mocked_loop(monkeypatch, tmp_path, prompt_plan, mutate=None):
    manifest = {'revision': 0, 'segments': []}
    descriptors, observed, muxes = {}, [], []
    monkeypatch.setattr(mv, 'long_video_chain_root', lambda _: tmp_path)
    monkeypatch.setattr(mv, '_exclusive_loop_lock', lambda _: nullcontext())
    monkeypatch.setattr(mv, 'load_delivery_manifest', lambda *a, **k: (manifest, 'primary'))
    monkeypatch.setattr(mv, '_reusable_candidate', lambda *a: None)
    monkeypatch.setattr(mv, '_available_candidate_id', lambda root, index, base: f'{base}-scene-{index}')
    monkeypatch.setattr(mv, '_release_segment_memory', lambda: None)
    def conditioning(*args, **kwargs):
        observed.append({'prompt': args[3], 'reference': args[19]['ref_image_1'],
                         'first_frame': args[17], 'last_frame': args[18], 'audio': args[15]})
        return args[3], {'samples': torch.zeros(1, 1)}, None, args[3], '{}', 'ok'
    monkeypatch.setattr(mv, 'build_conditioning', conditioning)
    def sample(model, positive, latent, **kwargs):
        if mutate is not None:
            mutate()
        return dict(latent)
    monkeypatch.setattr(mv, '_sample_one_segment', sample)
    monkeypatch.setattr(mv, 'decode_av_latent', lambda *a: (torch.zeros(158, 32, 32, 3), audio(), {}, {}))
    def save(frames, sound, latent, chain, index, start, save_context, parent, revision, candidate_id, model_id, summary, prompt, seed, *rest):
        descriptor = tmp_path / f'candidate-{index}.json'
        descriptor.write_text('{}', encoding='utf8')
        descriptors[str(descriptor)] = dict(index=index, candidate_id=candidate_id,
            parent_candidate_id=parent, parent_manifest_revision=revision,
            frame_count=len(frames), timeline_start_frame=round(start * 24),
            timeline_end_frame=round(start * 24) + len(frames), is_final_segment=not save_context,
            model_id=model_id, sampling_summary=summary, prompt=prompt, seed=seed, width=736, height=416)
        return str(descriptor), 'candidate.mp4', '{}'
    monkeypatch.setattr(mv, 'save_long_video_candidate', save)
    def accept(path, accepted, policy, strict):
        assert accepted is True and strict is True
        manifest['segments'].append(descriptors[path])
        manifest['revision'] += 1
        return 'preview', True, str(tmp_path / 'manifest.json'), '{}'
    monkeypatch.setattr(mv, 'accept_long_video_candidate', accept)
    def compose(*args):
        final = tmp_path / 'assembled.mp4'
        final.write_bytes(b'assembled')
        return str(final), '{}'
    monkeypatch.setattr(mv, 'compose_accepted_long_video', compose)
    def mux(path, song, total_frames, prefix):
        muxes.append((song, total_frames))
        final = tmp_path / 'final.mp4'
        final.write_bytes(b'master')
        return str(final), {'output_sha256': hashlib.sha256(b'master').hexdigest(), 'full_song_muxed_once': True}
    monkeypatch.setattr(mv, '_mux_master_audio', mux)
    return observed, muxes, manifest


def settings():
    return dict(chain_id='cast-test', width=736, height=416, base_seed=10, steps=4,
        shift_video=12., shift_audio=3., sampler_name='euler', scheduler='simple',
        resume_existing=True, filename_prefix='test', bit_depth=8, crf=18, model_id='test-label-only')


def test_actual_AB_dispatch_reset_resume_and_unique_original_song_mux(monkeypatch, tmp_path):
    prompt_plan = plan()
    model, components, _ = fake_producers(monkeypatch)
    first, second = torch.zeros(1, 32, 32, 3), torch.ones(1, 32, 32, 3)
    song, vocal = audio(), audio()
    observed, muxes, _ = mocked_loop(monkeypatch, tmp_path, prompt_plan)
    result = cast.render(model, *components, first, second, song, vocal, prompt_plan, **settings())
    assert result[2] == 2 and result[3] == 'complete'
    assert [row['reference'] is ref for row, ref in zip(observed, [first, second], strict=True)] == [True, True]
    assert all(row['first_frame'] is None and row['last_frame'] is None for row in observed)
    assert len(muxes) == 1 and muxes[0][0] is song and muxes[0][1] == 288
    assert json.loads(result[4])['human_quality_accepted'] is False
    repeated = cast.render(model, *components, first, second, song, vocal, prompt_plan, **settings())
    assert repeated[0] == result[0] and len(observed) == 2 and len(muxes) == 1
    second.flatten()[1] = .25
    with pytest.raises(ValueError, match='different contract'):
        cast.render(model, *components, first, second, song, vocal, prompt_plan, **settings())
    assert len(observed) == 2 and len(muxes) == 1


def test_mutation_during_sampler_rejects_before_save_accept_or_mux(monkeypatch, tmp_path):
    prompt_plan = plan()
    model, components, _ = fake_producers(monkeypatch)
    first, second = torch.zeros(1, 32, 32, 3), torch.ones(1, 32, 32, 3)
    song, vocal = audio(), audio()
    observed, muxes, manifest = mocked_loop(monkeypatch, tmp_path, prompt_plan, lambda: second.flatten().__setitem__(1, .3))
    with pytest.raises(ValueError, match='content changed'):
        cast.render(model, *components, first, second, song, vocal, prompt_plan, **settings())
    assert len(observed) == 1 and not muxes and not manifest['segments']


def test_node_schemas_are_independent_and_additive():
    assert [node.define_schema().node_id for node in NODES] == [
        'MiniMaxH3MVCastSoloPlanEXPT8', 'MiniMaxH3MVCastSoloRendererEXPT8']
    renderer = NODES[1].define_schema()
    assert {item.id for item in renderer.inputs} >= {'reference_a', 'reference_b', 'full_song', 'vocal_lock_audio'}
    assert next(item for item in renderer.inputs if item.id == 'steps').default == 4


def test_cast_runtime_cannot_silently_alter_old_route():
    with pytest.raises(ValueError, match='cannot alter'):
        mv.run_local_mv_in_node_loop(object(), object(), object(), object(),
            torch.zeros(1, 32, 32, 3), audio(), plan()['template'],
            prompt_plan_validator=mv.validate_mv_vocal_lock_visual_prompt_plan,
            cast_binding=object(), **settings())


def test_effective_original_H3_audio_crop_shim_is_bound_before_loop(monkeypatch, tmp_path):
    model, components, _ = fake_producers(monkeypatch)
    sound_vae = components[2]
    sound_vae.crop_input = True
    sound_vae.latent_channels, sound_vae.latent_dim, sound_vae.audio_sample_rate = 32, 2, 32000
    prompt_plan = plan()
    observed, _, manifest = mocked_loop(monkeypatch, tmp_path, prompt_plan,
                                        lambda: setattr(sound_vae, 'crop_input', True))
    def sample(model, positive, latent, **kwargs):
        assert sound_vae.crop_input is False  # old encode policy, before bind/sample
        sound_vae.crop_input = True
        return dict(latent)
    monkeypatch.setattr(mv, '_sample_one_segment', sample)
    with pytest.raises(ValueError, match=r'content changed.*crop_input'):
        cast.render(model, *components, torch.zeros(1, 32, 32, 3), torch.ones(1, 32, 32, 3),
                    audio(), audio(), prompt_plan, **settings())
    assert len(observed) == 1 and not manifest['segments']


@pytest.mark.parametrize('lora', [False, True])
def test_actual_Core_clock_container_preserves_source_clock_and_exact_AV(lora):
    from test_fast_h3_v2_core_sampler import model as native_model
    from test_progressive_sampling_runtime import conditioning, latent
    from h3_audio_t8_pkg.modular_sampling.results import selected_model_identity
    from h3_audio_t8_pkg.patch_stack_policy import model_identity_matches
    reference, source = native_model(), native_model()
    if lora:
        key = next(key for key in source.model_state_dict() if key.startswith('diffusion_model.') and key.endswith('.weight'))
        patch = torch.full_like(source.model_state_dict()[key], .003)
        for value in (reference, source):
            assert value.add_patches({key: ('diff', (patch,))}, .7) == [key]
    original_clock = source.model.model_sampling
    before = selected_model_identity(source)
    isolated = cast.independent_clock_container(source)
    assert isolated.model is not source.model
    assert isolated.model.diffusion_model is source.model.diffusion_model
    assert isolated.backup is source.backup and isolated.patches == source.patches
    assert isolated.object_patches_backup is not source.object_patches_backup
    options = dict(seed=31, steps=4, shift_video=12., shift_audio=3.,
                   sampler_name='euler', scheduler='simple')
    expected = mv._sample_one_segment(reference, conditioning(), latent(), **options)
    actual = mv._sample_one_segment(isolated, conditioning(), latent(), **options)
    for left, right in zip(actual['samples'].unbind(), expected['samples'].unbind(), strict=True):
        assert torch.equal(left, right) and torch.isfinite(left).all()
    assert source.model.model_sampling is original_clock
    assert not source.object_patches_backup
    assert model_identity_matches(before, selected_model_identity(source))
    repeated = mv._sample_one_segment(isolated, conditioning(), latent(), **options)
    assert source.model.model_sampling is original_clock
    assert model_identity_matches(before, selected_model_identity(source))
    for left, right in zip(actual['samples'].unbind(), repeated['samples'].unbind(), strict=True):
        assert torch.equal(left, right)
    if lora:
        patch.add_(.02)
    else:
        with torch.no_grad():
            next(source.model.diffusion_model.parameters()).add_(.02)
    assert not model_identity_matches(before, selected_model_identity(source))
    assert not torch.cuda.is_initialized()
