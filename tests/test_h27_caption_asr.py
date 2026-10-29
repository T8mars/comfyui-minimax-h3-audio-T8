"""Actual frozen media/FFmpeg/sidecars; ASR injected here, real model separately."""
from fractions import Fraction
import uuid

import pytest

from h3_audio_t8_pkg import director_caption_asr as asr
from h3_audio_t8_pkg import director_captions as captions
from h3_audio_t8_pkg import speech_verification as speech
from h3_audio_t8_pkg.director_project import file_sha
from test_director_captions import frozen, payload, rational


@pytest.fixture
def scene(tmp_path, monkeypatch):
    store, output, project, records, film = frozen(tmp_path, trim=True)
    model = tmp_path / 'user_selected_unknown_model'
    model.mkdir()
    (model / 'config.json').write_text('{}')
    (model / 'model.bin').write_bytes(b'fixture only, never loaded')
    monkeypatch.setattr(speech, 'resolve_asr_model_directory', lambda _: model)
    body = {'film_sha256': film['sha256'], 'shot_id': film['entries'][0]['shot_id'],
            'model_directory': str(model), 'language': 'English', 'cpu_threads': 2}
    return store, output, project, records, film, body, model


def recognized(monkeypatch, calls):
    def consume(audio, *args, **kwargs):
        calls.append(audio['waveform'].clone())
        assert audio['sample_rate'] == 16000
        assert audio['waveform'].shape[-1] >= 8000  # full source, not the adopted crop
        kwargs['cancel_check']()
        return {'text': 'before hello world. outside.', 'language': 'en',
                'words': [{'word': 'before', 'start': .02, 'end': .1},
                          {'word': 'hello', 'start': .11, 'end': .22},
                          {'word': 'world.', 'start': .23, 'end': .31},
                          {'word': 'outside.', 'start': .6, 'end': .7}],
                'untimed_words': [{'word': 'missing', 'start': None, 'end': None}]}
    monkeypatch.setattr(speech, 'transcribe_audio', consume)


def start(scene):
    store, _, project, _, film, body, _ = scene
    job_id = str(uuid.uuid4())
    job, created = asr.reserve_asr(store, project['id'], film['id'], job_id, body)
    assert created
    return (store, project['id'], film['id'], job_id), job


def test_recognition_binding_editable_cues_and_trim_applied_once(scene, monkeypatch):
    calls = []
    recognized(monkeypatch, calls)
    args, job = start(scene)
    store, project_id, film_id, job_id = args
    film = scene[4]
    original = {p: file_sha(p) for p in scene[1].glob('*.mp4')}
    result = asr.run_asr(*args)
    assert result['state'] == 'success', result
    assert asr.run_asr(*args) == result and len(calls) == 1
    again, created = asr.reserve_asr(store, project_id, film_id, job_id, scene[5])
    assert not created and again == result
    content = asr.recognized_track(*args)
    track = content['track']
    assert content['untimed_words'][0]['word'] == 'missing'
    assert track['cues'][0]['text'] == 'before hello world.'
    before = asr._path(store, job_id).read_bytes()
    track['cues'][0]['text'] = 'User edited transcript <literal>'
    caption_id = str(uuid.uuid4())
    rendered, _ = captions.prepare_captions(store, project_id, film_id, caption_id, payload(film, [track]))
    assert rendered['text_status'] == 'recognized_audio_draft_editable_not_speaker_or_lipsync_verified'
    assert rational(rendered['cues'][0]['start']) == 0
    assert rational(rendered['cues'][0]['end']) == Fraction(31, 100) - Fraction(2, 24)
    assert rendered['cues'][0]['partially_trimmed']
    assert any(x['reason'] == 'outside_adopted_trim_omitted' for x in rendered['notices'])
    assert captions.load_captions(store, project_id, film_id, caption_id) == rendered
    assert asr._path(store, job_id).read_bytes() == before
    assert all(file_sha(p) == digest for p, digest in original.items())


@pytest.mark.parametrize('damage', ['audio_sha', 'media_sha', 'clock', 'missing_receipt'])
def test_recognized_source_cannot_be_forged_or_repointed(scene, monkeypatch, damage):
    recognized(monkeypatch, [])
    args, _ = start(scene)
    assert asr.run_asr(*args)['state'] == 'success'
    track = asr.recognized_track(*args)['track']
    if damage == 'audio_sha':
        track['source']['audio_sha256'] = '0' * 64
    elif damage == 'media_sha':
        track['media_sha256'] = '0' * 64
    elif damage == 'clock':
        track['clock']['rate'] = 8000
    else:
        track['source']['asr_job_id'] = str(uuid.uuid4())
    with pytest.raises((ValueError, FileNotFoundError)):
        captions.prepare_captions(args[0], args[1], args[2], str(uuid.uuid4()), payload(scene[4], [track]))


def test_cancelled_queued_job_never_decodes_or_transcribes(scene, monkeypatch):
    args, _ = start(scene)
    monkeypatch.setattr(asr, '_decode_pcm', lambda *_: pytest.fail('cancelled job decoded'))
    asr.cancel_asr(*args)
    assert asr.run_asr(*args)['state'] == 'cancelled'
    with pytest.raises(ValueError, match='成功识别'):
        asr.recognized_track(*args)


def test_failed_asr_does_not_publish_track_and_explicit_new_job_works(scene, monkeypatch):
    args, _ = start(scene)
    def fail(*_args, **_kwargs):
        raise RuntimeError('injected native failure')
    monkeypatch.setattr(speech, 'transcribe_audio', fail)
    assert asr.run_asr(*args)['state'] == 'error'
    assert 'injected native failure' in asr.asr_status(*args)['error']
    with pytest.raises(ValueError):
        asr.recognized_track(*args)
    recognized(monkeypatch, [])
    following, _ = start(scene)
    assert asr.run_asr(*following)['state'] == 'success'
    assert asr.asr_status(*args)['state'] == 'error'


def test_model_change_and_restart_do_not_retry_or_accept_old_identity(scene, monkeypatch):
    args, _ = start(scene)
    (scene[6] / 'model.bin').write_bytes(b'changed')
    monkeypatch.setattr(speech, 'transcribe_audio', lambda *_a, **_kw: pytest.fail('changed model consumed'))
    assert asr.run_asr(*args)['state'] == 'error'
    assert '模型内容已变' in asr.asr_status(*args)['error']
    following, _ = start(scene)
    assert asr.interrupt_asr(*following)['state'] == 'interrupted'
    assert asr.run_asr(*following)['state'] == 'interrupted'


def test_engine_change_after_reservation_is_not_labeled_a_matching_result(scene, monkeypatch):
    args, _ = start(scene)
    monkeypatch.setattr(asr, '_engine_identity', lambda: {'changed': True})
    monkeypatch.setattr(speech, 'transcribe_audio', lambda *_a, **_kw: pytest.fail('changed engine consumed'))
    assert asr.run_asr(*args)['state'] == 'error'
    assert '引擎或实现已变' in asr.asr_status(*args)['error']


def test_windows_cleanup_failure_preserves_primary_cancellation_and_evidence(scene, monkeypatch):
    args, _ = start(scene)
    def cancel_during_decode(*_args, **_kwargs):
        raise asr.ASRCancelled('explicit task cancellation')
    def locked(_path):
        raise PermissionError('injected Windows cleanup file lock')
    monkeypatch.setattr(asr, '_decode_pcm', cancel_during_decode)
    monkeypatch.setattr(asr.shutil, 'rmtree', locked)
    result = asr.run_asr(*args)
    assert result['state'] == 'cancelled' and result['error'] == 'explicit task cancellation'
    assert result['cleanup']['removed'] is False
    assert 'file lock' in result['cleanup']['error']
