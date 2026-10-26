"""Actual media loaders and native conditioning boundaries; no learned VAE QA."""
from copy import deepcopy
import inspect
import json
from pathlib import Path
import shutil
import subprocess
import uuid
import wave

import numpy as np
import pytest
import torch
import folder_paths
import nodes as core_nodes
from comfy_extras.nodes_audio import LoadAudio, TrimAudioDuration
from comfy_extras.nodes_video import GetVideoComponents, LoadVideo

from helpers import FakeAudioVAE, FakeClip, FakeVideoVAE
from test_director_generation import _image, _patch_director_models, _store
from h3_audio_t8_pkg.conditioning import build_conditioning
from h3_audio_t8_pkg.director_generation import build_director_generation_prompt
from h3_audio_t8_pkg.director_project import ProjectStore, compile_project, new_project
from h3_audio_t8_pkg.ffmpeg_utils import resolve_ffmpeg
from h3_audio_t8_pkg.nodes import MiniMaxH3AudioWindowT8, MiniMaxH3OutputTrimT8


def _tone(store, hz):
    aid = str(uuid.uuid4())
    path = store.input_root / 't8_director' / aid / 'source.wav'
    path.parent.mkdir(parents=True)
    time = np.arange(96000) / 32000
    pcm = np.stack([.12 * np.sin(2*np.pi*hz*time), .1 * np.sin(2*np.pi*(hz+73)*time)], -1)
    with wave.open(str(path), 'wb') as out:
        out.setnchannels(2)
        out.setsampwidth(2)
        out.setframerate(32000)
        out.writeframes(np.asarray(np.rint(pcm * 32767), dtype='<i2').tobytes())
    return store.register_asset(path, aid, f'{hz}.wav')


def _video(store, color, sound):
    aid = str(uuid.uuid4())
    path = store.input_root / 't8_director' / aid / 'source.mp4'
    path.parent.mkdir(parents=True)
    arguments = ['-f', 'lavfi', '-i', f'color=c={color}:s=64x64:r=24:d=2']
    if sound:
        arguments += ['-f', 'lavfi', '-i', 'sine=frequency=431:sample_rate=32000:duration=2']
    arguments += ['-map', '0:v', '-c:v', 'libx264', '-pix_fmt', 'yuv420p']
    if sound:
        arguments += ['-map', '1:a', '-c:a', 'aac', '-ar', '32000', '-ac', '2']
    arguments += ['-t', '2', str(path)]
    subprocess.run([resolve_ffmpeg(), '-hide_banner', '-loglevel', 'error', '-nostdin', '-n',
                    *arguments], check=True, capture_output=True, timeout=30,
                   creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    result = store.register_asset(path, aid, f'{color}.mp4')
    assert result['has_audio'] is sound
    return result


@pytest.fixture
def sources(tmp_path, monkeypatch):
    store = _store(tmp_path)
    assets = {'a': _video(store, 'red', True), 'b': _video(store, 'blue', False),
              'v': _tone(store, 743), 'd': _tone(store, 1193),
              'first': _image(store, 'first.png'), 'last': _image(store, 'last.png')}
    monkeypatch.setattr(folder_paths, 'get_input_directory', lambda: str(store.input_root))
    _patch_director_models(monkeypatch)
    monkeypatch.setattr(folder_paths, 'get_filename_list', lambda folder: {
        'latent_upscale_models': ['minimax_h3_latent_upscaler_3d_fp16.safetensors']}.get(folder, []))
    monkeypatch.setattr(folder_paths, 'get_full_path', lambda folder, name: name)
    return store, assets


def _project(assets, sound='record', mode='first', order=('a', 'b')):
    project = new_project()
    project['assets'] = list(assets.values())
    shot = project['doc']['shots'][0]
    shot.update(mode=mode, sound=sound, first=assets['first']['id'], last=assets['last']['id'],
                audio=assets['d' if sound == 'record' else 'v']['id'], start=.25, end=2.25,
                manualDuration=2, duration=2, autoDuration=True,
                refs=[assets[key]['id'] for key in order],
                tray=[assets[key]['id'] for key in ('a', 'b', 'v', 'd', 'first', 'last')],
                simplePrompt='Use the motion reference and keep the scene continuous.')
    return project, shot


def _load_media(graph):
    """Execute the actual graph's media nodes, stopping before learned encoders."""
    cache = {}

    def read(link):
        node_id, slot = link
        if node_id not in cache:
            node = graph[node_id]
            values, kind = node['inputs'], node['class_type']
            if kind == 'LoadVideo':
                result = LoadVideo.execute(**values).result
            elif kind == 'GetVideoComponents':
                result = GetVideoComponents.execute(read(values['video'])).result
            elif kind == 'LoadAudio':
                result = LoadAudio.execute(**values).result
            elif kind == 'TrimAudioDuration':
                result = TrimAudioDuration.execute(**{**values, 'audio': read(values['audio'])}).result
            elif kind == 'LoadImage':
                result = core_nodes.LoadImage().load_image(**values)
            elif kind == 'MiniMaxH3AudioWindowT8':
                result = MiniMaxH3AudioWindowT8.execute(**{**values, 'audio': read(values['audio'])}).result
            else:
                raise AssertionError(f'Unexpected media producer: {kind}')
            cache[node_id] = result
        return cache[node_id][slot]

    return read


def _consume(graph, read):
    inputs = graph['5']['inputs']
    vae = FakeAudioVAE()
    kwargs = {'clip': FakeClip(), 'video_vae': FakeVideoVAE(), 'audio_vae': vae}
    for key, value in inputs.items():
        if key in kwargs:
            continue
        if '.' in key:
            field, slot = key.split('.', 1)
            kwargs.setdefault(field, {})[slot] = read(value)
        elif key in inspect.signature(build_conditioning).parameters:
            kwargs[key] = read(value) if isinstance(value, list) else value
    return build_conditioning(**kwargs), kwargs, vae


@pytest.mark.parametrize('order', [('a', 'b'), ('b', 'a'), ('b',)])
def test_record_reference_order_keeps_actual_video_soundtrack_and_drive_identity(sources, order):
    store, assets = sources
    project, shot = _project(assets, order=order)
    shot['refs'].append(assets['v']['id'])
    # Literal Audio1 belongs to video A when it is present. Director aliases
    # resolve V/D by stable ID before the native conditioner's optional remap.
    shot['simplePrompt'] = ('<Audio 1> is the video soundtrack. ' if 'a' in order else '') + \
        '@audio2 drives the performance. @audio1 provides a separate voice reference.'
    stored = store.save(project, 0)
    reopened = ProjectStore(store.root.parent, store.input_root).load(stored['id'])
    copied = deepcopy(reopened)
    copied.update(id=str(uuid.uuid4()), revision=0)
    copied = store.save(copied, 0)
    before = deepcopy(copied)
    plan = compile_project(copied, store, shot_id=shot['id'])
    assert plan['ready'], plan['errors']
    compiled = plan['shots'][0]
    sound_a = int('a' in order)
    audio_rows = [row for row in compiled['media_map'] if 'audio' in row['role']]
    assert [(row['asset_id'], row['native']) for row in audio_rows] == [
        *([(assets['a']['id'], '<Audio 1>')] if sound_a else []),
        (assets['d']['id'], f'<Audio {sound_a+1}>'),
        (assets['v']['id'], f'<Audio {sound_a+2}>')]
    built = build_director_generation_prompt(copied, shot['id'], store)
    graph, read = built['prompt'], _load_media(built['prompt'])
    (positive, latent, final, prompt, mapping, _), inputs, vae = _consume(graph, read)
    assert prompt == compiled['prompt']
    assert f'<Audio {sound_a+1}> drives' in prompt
    if sound_a:
        assert '<Audio 1> is the video soundtrack.' in prompt
    assert json.loads(mapping)['source_audio_ordinal'] == sound_a+1
    assert final is inputs['drive_audio'] is inputs['final_audio']
    assert torch.count_nonzero(latent['noise_mask'].unbind()[1]) == 0
    actual_refs = [row for row in positive[0][1]['minimax_refs'] if row['kind'] in {'video', 'video_audio'}]
    assert [row['kind'] for row in actual_refs] == ['video_audio' if key == 'a' else 'video' for key in order]
    # Encoder input captures are real decoded/resampled media, not dummy files.
    expected_audio = []
    if sound_a:
        ordinal = order.index('a')
        expected_audio.append(inputs['ref_video_audios'][f'ref_video_audio_{ordinal}'])
    expected_audio += [inputs['drive_audio'], inputs['ref_audios']['ref_audio_0']]
    assert len(vae.encode_calls) == len(expected_audio)
    for actual, expected in zip(vae.encode_calls, expected_audio):
        assert expected['sample_rate'] == 32000
        assert torch.equal(actual, expected['waveform'].movedim(1, -1))
    trim = graph['11']['inputs']
    images = torch.zeros(inputs['length'], 32, 32, 3)
    _, delivered, _ = MiniMaxH3OutputTrimT8.execute(
        images, read(trim['start_seconds']), trim['duration_seconds'], 24, final).result
    source = LoadAudio.execute(assets['d']['server_path']).result[0]
    assert torch.equal(delivered['waveform'], source['waveform'][..., 8000:72000])
    assert copied == before


@pytest.mark.parametrize('order', [('a', 'b'), ('b', 'a'), ('b',)])
def test_selected_voice_range_reaches_encoder_without_replacing_generated_track(sources, order):
    store, assets = sources
    project, shot = _project(assets, sound='voice', mode='refs', order=order)
    shot['simplePrompt'] = '@audio1 provides the voice. Keep the motion reference.'
    graph = build_director_generation_prompt(project, shot['id'], store)['prompt']
    (_, latent, final, prompt, mapping, _), inputs, vae = _consume(graph, _load_media(graph))
    expected = LoadAudio.execute(assets['v']['server_path']).result[0]['waveform'][..., 8000:72000]
    assert torch.equal(inputs['ref_audios']['ref_audio_0']['waveform'], expected)
    assert torch.equal(vae.encode_calls[-1], expected.movedim(1, -1))
    assert prompt.startswith(f'<Audio {1 + int("a" in order)}> provides')
    assert json.loads(mapping)['source_audio_ordinal'] is None
    # Native generation without a locked source has no denoise mask at all.
    assert 'noise_mask' not in latent
    assert final is None and 'drive_audio' not in inputs and 'final_audio' not in inputs
    assert graph['11']['inputs']['audio'] == ['10', 1]
    assert graph['5']['inputs']['length'] == 56


@pytest.mark.parametrize('mode', ['first', 'ends', 'refs', 'text'])
def test_record_visual_modes_use_existing_native_contract_and_keep_real_errors(sources, mode):
    store, assets = sources
    project, shot = _project(assets, mode=mode)
    built = build_director_generation_prompt(project, shot['id'], store)
    graph = built['prompt']
    _, inputs, _ = _consume(graph, _load_media(graph))
    assert inputs['task_type'] == ('Hybrid' if mode in {'first', 'ends'} else 'Ref2VA')
    assert graph['6']['inputs']['steps'] == 8
    shot['end'] = 99
    assert any(error['field'] == 'audio' for error in compile_project(project, store)['errors'])
    shot['end'] = 2.25
    if mode == 'first':
        shot['first'] = None
        assert any('首帧' in error['message'] for error in compile_project(project, store)['errors'])


@pytest.mark.parametrize('two_pass,relay', [(False, False), (True, False), (False, True), (True, True)])
def test_switch_record_voice_native_preserves_draft_and_final_track_for_each_stage(sources, two_pass, relay):
    store, assets = sources
    project, shot = _project(assets)
    if two_pass:
        project['doc']['sampling'] = {'mode': 'two_pass', 'output_mp': .4}
    if relay:
        shot['d3Inherit'] = False
        shot['d3'] = {'prompt_relay': {'enabled': True}}
    for sound in ('record', 'voice', 'native', 'record'):
        shot['sound'] = sound
        before = deepcopy(project)
        graph = build_director_generation_prompt(project, shot['id'], store)['prompt']
        condition_ids = [key for key, node in graph.items() if node['class_type'] in {
            'MiniMaxH3AudioConditioningT8', 'MiniMaxH3PromptRelayConditioningT8Advanced'}]
        assert len(condition_ids) == (2 if two_pass else 1)
        assert any(graph[key]['class_type'].startswith('MiniMaxH3PromptRelay') for key in condition_ids) is relay
        for key in condition_ids:
            inputs = graph[key]['inputs']
            assert ('drive_audio' in inputs) is (sound == 'record')
            assert ('final_audio' in inputs) is (sound == 'record')
            if sound == 'record':
                assert inputs['prompt_primary_audio_ordinal'] == 0
                assert inputs['drive_audio'] == inputs['final_audio']
        trim = graph['11']['inputs']
        if sound == 'record':
            assert trim['audio'][0] in condition_ids
            assert graph[trim['start_seconds'][0]]['class_type'] == 'MiniMaxH3AudioWindowT8'
        else:
            assert graph[trim['audio'][0]]['class_type'] == 'MiniMaxH3AVDecodeT8'
            assert not any(node['class_type'] == 'MiniMaxH3AudioWindowT8' for node in graph.values())
        assert project == before


def test_draft_preflight_accepts_computable_record_combinations_but_requires_selected_audio(sources):
    _, assets = sources
    project, shot = _project(assets, mode='ends')
    url = (Path(__file__).resolve().parents[1] / 'web/director/preflight_ui.mjs').as_uri()
    script = f"import {{directorDraftIssues}} from {json.dumps(url)};let s='';for await(const c of process.stdin)s+=c;const p=JSON.parse(s);console.log(JSON.stringify(directorDraftIssues(p.doc,p.doc.shots[0],p.assets)));"

    def check():
        result = subprocess.run([shutil.which('node'), '--input-type=module', '-e', script],
                                input=json.dumps(project), text=True, encoding='utf8', capture_output=True, timeout=20)
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)

    assert check() == []
    shot['audio'] = None
    assert any(row['field'] == 'audio' for row in check())
