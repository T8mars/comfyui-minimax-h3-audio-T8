"""Actual registered media -> saved adoption -> RGB/PCM -> original H3 builder."""
from copy import deepcopy
import json
import uuid

import av
import numpy as np
import pytest
import torch

from h3_audio_t8_pkg import external_continuation as bridge, director_external as takes
from h3_audio_t8_pkg.director_media import run_media, _result, MediaProcessError
from h3_audio_t8_pkg.director_project import sha, file_sha, new_project
from h3_audio_t8_pkg.long_video_dual_picture_context import tensor_sha
from h3_audio_t8_pkg.core import nested_av_parts
from h3_audio_t8_pkg.nodes_external_continuation import NODES
from h3_audio_t8_pkg.nodes_long_video_exp import MiniMaxH3LongVideoConditioningT8
from test_director_external import prepared as _prepared
from helpers import FakeVideoVAE, FakeAudioVAE, FakeClip

prepared = _prepared


def adopt(prepared):
    store, project, request, output, _original = prepared
    takes.register_external_take(store, request, output)
    project['doc']['shots'][0]['adoptedResultId'] = 'external:' + request['take_id']
    project['doc']['shots'][0]['filmTrim'] = {'in_frame': 2, 'out_frame': 10}
    store.save(project, project['revision'])
    return bridge.capture_source(store, output, project_id=project['id'], shot_id=project['current'],
                                 take_id=request['take_id'], context_frames=5)


def test_real_adopted_integer_tail_original_builder_and_no_native_ancestry(prepared):
    source = adopt(prepared)
    before = file_sha(prepared[-1])
    assert source.binding['frame_interval'] == [5, 10]
    assert source.binding['origin'] == 'external'
    assert source.binding['original_native_latent_available'] is source.binding['has_sampling_ancestor'] is False
    video, audio = FakeVideoVAE(), FakeAudioVAE()
    context = bridge.prepare_context(source, video, 96, 64)
    assert context.context['video_tail'].shape == (1, 24, 2, 4, 6)
    assert context.descriptor()['video_vae']['portable_cache_reuse'] is False  # unknown wrapper still executes
    with av.open(str(source.media)) as container:
        frames = [frame.to_ndarray(format='rgb24') for frame in container.decode(video=0)]
    expected = torch.from_numpy(frames[9]).float()/255
    assert torch.equal(context.last_frame[0], expected)
    old_tail = tensor_sha(context.context['video_tail'])
    result = bridge.condition(context, clip=FakeClip(), video_vae=video, audio_vae=audio,
                              prompt='continue the scene', length=22)
    assert result[0][0][1]['t8_external_continuation']['source'] == source.binding
    assert result[2] is None  # no silent original-PCM delivery substitution
    assert json.loads(result[5])['external_continuation']['RGB_PCM_reencode_not_original_latent'] is True
    assert nested_av_parts(result[1])[0].shape[2] == 7
    assert old_tail == tensor_sha(context.context['video_tail']) and file_sha(prepared[-1]) == before


@pytest.mark.parametrize('changed', ['project', 'adoption', 'trim', 'movie', 'receipt', 'source_hash'])
def test_current_full_source_adoption_trim_receipt_and_project_mutation_refused(prepared, changed):
    source = adopt(prepared)
    store, project, request, _output, _original = prepared
    if changed in ('project', 'adoption', 'trim'):
        saved = store.load(project['id'])
        if changed == 'project':
            saved['doc']['shots'][0]['simplePrompt'] = 'new local content'
        elif changed == 'adoption':
            saved['doc']['shots'][0]['adoptedResultId'] = None
        else:
            saved['doc']['shots'][0]['filmTrim']['out_frame'] = 9
        store.save(saved, saved['revision'])
    elif changed == 'movie':
        source.media.write_bytes(b'keep corrupt evidence')
    else:
        path = takes._receipt_path(store, project['id'], request['take_id'])
        receipt = json.loads(path.read_text(encoding='utf8'))
        if changed == 'receipt':
            receipt['sha256'] = '0'*64
        else:
            receipt['media_sha256'] = '0'*64
            receipt['sha256'] = sha({k: v for k, v in receipt.items() if k!='sha256'})
        path.write_text(json.dumps(receipt), encoding='utf8')
    with pytest.raises((ValueError, OSError)):
        source.verify()


def test_unadopted_short_mono_and_unknown_native_generation_inputs_refused(prepared):
    store, project, request, output, _original = prepared
    takes.register_external_take(store, request, output)
    args = dict(project_id=project['id'], shot_id=project['current'], take_id=request['take_id'], context_frames=5)
    with pytest.raises(ValueError, match='adopted'):
        bridge.capture_source(store, output, **args)
    source = adopt(prepared)
    with pytest.raises(ValueError, match='complete motion tail'):
        bridge.capture_source(store, output, **{**args, 'context_frames': 22})
    with pytest.raises(ValueError, match='stereo'):
        bridge.capture_source(store, output, **{**args, 'audio_policy': 'reencode_stereo_pcm_context'})
    with pytest.raises(ValueError, match='unknown generation'):
        bridge.current_binding(store, output, {**source.binding['request'], 'seed': 123})


@pytest.mark.parametrize('field', ['video_tail', 'audio_tail', 'last_frame', 'metadata'])
def test_encoded_context_finite_content_and_internal_adapter_identity_guard(prepared, field):
    context = bridge.prepare_context(adopt(prepared), FakeVideoVAE(), 96, 64)
    if field == 'metadata':
        context.context['metadata']['source_segment_index'] = 5
    elif field == 'last_frame':
        context.last_frame[0, 0, 0, 0] += .1
    else:
        context.context[field] = context.context[field].clone()
        context.context[field].flatten()[0] += .1
    with pytest.raises(ValueError, match='changed'):
        context.verify()


def test_actual_worker_tail_media_and_shape_facade_rejects_fabricated_result(prepared, tmp_path):
    source = adopt(prepared)
    output = tmp_path/'tail.npz'
    actual = run_media('external_tail', source.media, start_frame=5, end_frame=10, include_audio=False, output_npz=output)
    assert actual['frame_interval'] == [5, 10] and actual['audio_shape'] == [0, 0]
    assert actual['media_sha256'] == file_sha(source.media) and actual['output_sha256'] == file_sha(output)
    with np.load(output, allow_pickle=False) as data:
        assert data['frames'].shape == (5, 32, 64, 3) and data['audio'].shape == (0, 0)
    for key, changed in [('frames_shape', [True, 32, 64, 3]), ('frame_interval', [5, 12]),
                         ('audio_shape', [2, 200]), ('output_sha256', 'bad')]:
        fake = {**deepcopy(actual), key: changed}
        with pytest.raises(MediaProcessError):
            _result('external_tail', fake)
    with pytest.raises(ValueError, match='新绝对NPZ'):
        run_media('external_tail', source.media, start_frame=5, end_frame=10, include_audio=False, output_npz=output)


def test_original_conditioning_schema_unchanged_and_new_three_independent_sockets():
    old = MiniMaxH3LongVideoConditioningT8.define_schema()
    before = old.get_v1_info(MiniMaxH3LongVideoConditioningT8)
    new = NODES[-1].define_schema()
    assert MiniMaxH3LongVideoConditioningT8.define_schema().get_v1_info(MiniMaxH3LongVideoConditioningT8) == before
    names = {row.id for row in new.inputs}
    assert 'external_context' in names and {'model', 'clip', 'video_vae', 'audio_vae', 'prompt', 'semantic_bridge'} <= names
    assert not {'context', 'segment_index', 'context_frames', 'context_audio', 'width', 'height'} & names
    assert [node.define_schema().node_id for node in NODES] == [
        'MiniMaxH3ExternalContinuationSourceEXPT8', 'MiniMaxH3ExternalContextEncodeEXPT8',
        'MiniMaxH3ExternalContinuationConditioningEXPT8']
    assert NODES[0].define_schema().get_v1_info(NODES[0]).output[0] != before.input['required']['context'][0]


@pytest.fixture
def stereo(prepared):
    store, _project, _request, output, _original = prepared
    aid = str(uuid.uuid4())
    path = store.input_root/'t8_director'/aid/'source.mp4'
    path.parent.mkdir(parents=True)
    count, rate = 61, 48000
    with av.open(str(path), 'w') as container:
        video = container.add_stream('libx264', rate=24)
        video.width, video.height, video.pix_fmt = 64, 32, 'yuv420p'
        video.options = {'threads': '1', 'bf': '0'}
        sound = container.add_stream('aac', rate=rate)
        sound.layout = 'stereo'
        for index in range(count):
            pixels = np.full((32, 64, 3), (20+index, 70, 95), dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format='rgb24')
            for packet in video.encode(frame):
                container.mux(packet)
        for packet in video.encode():
            container.mux(packet)
        t = np.arange(count*rate//24)/rate
        pcm = np.stack([.15*np.sin(2*np.pi*220*t), .2*np.sin(2*np.pi*330*t)]).astype(np.float32)
        frame = av.AudioFrame.from_ndarray(pcm, format='fltp', layout='stereo')
        frame.sample_rate = rate
        for packet in sound.encode(frame):
            container.mux(packet)
        for packet in sound.encode():
            container.mux(packet)
    asset = store.register_asset(path, aid, 'stereo.mp4')
    project = new_project()
    project['assets'] = [asset]
    project['doc']['shots'][0].update(tray=[aid], selected=aid)
    project = store.save(project, 0)
    request = dict(project_id=project['id'], shot_id=project['current'], asset_id=aid,
                   take_id=str(uuid.uuid4()), expected_revision=project['revision'], project_sha256=sha(project),
                   label='Stereo真实PCM', provenance_category='external')
    takes.register_external_take(store, request, output)
    project['doc']['shots'][0].update(adoptedResultId='external:'+request['take_id'], filmTrim={'in_frame': 1, 'out_frame': 60})
    project = store.save(project, project['revision'])
    return store, project, request, output, path


@pytest.mark.parametrize('count', [5, 22, 39])
def test_actual_stereo_exact_PCM_window_and_original_source_preservation(stereo, tmp_path, count):
    store, project, request, output, path = stereo
    original = file_sha(path)
    source = bridge.capture_source(store, output, project_id=project['id'], shot_id=project['current'],
        take_id=request['take_id'], context_frames=count, audio_policy='reencode_stereo_pcm_context')
    archive = tmp_path/'pcm.npz'
    actual = run_media('external_tail', source.media, start_frame=60-count, end_frame=60,
                       include_audio=True, output_npz=archive)
    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        decoded = np.concatenate([f.to_ndarray() for f in container.decode(stream)], axis=1)
    with np.load(archive, allow_pickle=False) as data:
        assert np.array_equal(data['audio'], decoded[:, (60-count)*2000:120000])
    assert actual['sample_interval'] == [(60-count)*2000, 120000]

    class AudioVAE:
        audio_sample_rate = 48000

        def encode(self, waveform):
            self.input = waveform.clone()
            return torch.ones(1, 32, 2, round(waveform.shape[1]/48000*40))

    vae, video = AudioVAE(), FakeVideoVAE()
    context = bridge.prepare_context(source, video, 96, 64, vae)
    assert torch.equal(vae.input[0].movedim(-1, 0), torch.from_numpy(decoded[:, (60-count)*2000:120000]))
    assert context.context['metadata']['context_audio'] == 'video_and_audio'
    before_audio = context.context['audio_tail']
    result = bridge.condition(context, clip=FakeClip(), video_vae=video, audio_vae=vae,
                              prompt='continue naturally', length=124)
    assert result[2] is None and context.context['audio_tail'] is before_audio
    expected = max(0., round(count/24*40)-count*5/3)
    assert context.context['metadata']['audio_overhang'] == pytest.approx(expected)
    assert next(ref for ref in result[0][0][1]['minimax_refs'] if ref['kind']=='audio')['t8_long_video_audio_end_frame'] == pytest.approx(count+expected/(5/3))
    assert file_sha(path) == original and store.load(project['id']) == project
