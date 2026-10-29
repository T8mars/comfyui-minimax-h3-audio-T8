"""Actual adopted soundtrack -> immutable ASR evidence -> editable captions.

Full-source PCM uses media time zero. Caption compilation applies in/out once.
No speaker inference, lip-sync claim, media mutation or automatic adoption.
"""
from fractions import Fraction
from importlib.metadata import version
from contextlib import contextmanager
import json
import math
from pathlib import Path
import subprocess
import shutil
import tempfile
import threading
import time

from .director_film import film_media, load_film
from .director_project import atomic_json, contained, file_sha, identity, sha
from .ffmpeg_utils import resolve_ffmpeg


SCHEMA = 't8.director.caption_asr.v1'
_LOCK = threading.RLock()
_TERMINAL = {'success', 'error', 'cancelled', 'interrupted'}


class ASRCancelled(Exception):
    pass


def _path(store, job_id):
    return contained(store.root, f'film_asr/{identity(job_id)}.json')


def _model_identity(path):
    return {p.relative_to(path).as_posix(): file_sha(p) for p in sorted(path.rglob('*'))
            if p.is_file() and p.suffix not in {'.lock', '.pyc'} and '.cache' not in p.parts}


def _engine_identity():
    root = Path(__file__).parent
    return {'name': 'faster-whisper', 'version': version('faster-whisper'),
            'ctranslate2_version': version('ctranslate2'),
            'device': 'cpu', 'compute_type': 'int8', 'word_timestamps': True,
            'vad_filter': True, 'condition_on_previous_text': False,
            'sources': {name: file_sha(root/name) for name in
                        ('speech_verification.py', 'asr_process.py', 'asr_process_worker.py')}}


def reserve_asr(store, project_id, film_id, job_id, body):
    from .speech_verification import ASR_LANGUAGE_CODES, resolve_asr_model_directory
    if not isinstance(body, dict):
        raise ValueError('识别请求必须是对象')
    film = load_film(store, project_id, film_id)
    if body.get('film_sha256') != film['sha256']:
        raise ValueError('识别请求不属于当前冻结整片')
    entry = next((x for x in film['entries'] if x['shot_id'] == body.get('shot_id')), None)
    if entry is None or not entry['media'].get('has_audio'):
        raise ValueError('请选择冻结整片中有音轨的采用镜头')
    model = resolve_asr_model_directory(body.get('model_directory', ''))
    language, beam, threads = body.get('language', 'auto'), body.get('beam_size', 5), body.get('cpu_threads', 8)
    if language not in ASR_LANGUAGE_CODES or any(type(x) is not int or x < 1 for x in (beam, threads)):
        raise ValueError('识别语言无效，beam_size和cpu_threads须为正整数')
    request = {'film_sha256': film['sha256'], 'shot_id': entry['shot_id'],
        'version_id': entry['version_id'], 'media_sha256': entry['media_sha256'],
        'in_frame': entry['in_frame'], 'out_frame': entry['out_frame'],
        'model_directory': str(model), 'model_identity': _model_identity(model),
        'engine': _engine_identity(),
        'language': language, 'beam_size': beam, 'cpu_threads': threads,
        'decode': {'format': 'f32le', 'channels': 1, 'rate': 16000,
                   'stream': '0:a:0', 'sample_origin': 0,
                   'filter': 'aresample=16000:async=0:first_pts=0',
                   'time_origin': 'full_source_media_zero_not_adopted_trim'}}
    fingerprint = sha(request)
    with _LOCK:
        if _path(store, job_id).exists():
            current = asr_status(store, project_id, film_id, job_id)
            if current['fingerprint'] != fingerprint:
                raise ValueError('识别任务ID已用于另一份音轨或配置')
            return current, False
        job = {'schema': SCHEMA, 'id': identity(job_id), 'project_id': identity(project_id),
               'film_id': identity(film_id), 'request': request, 'fingerprint': fingerprint,
               'state': 'queued', 'cancel_requested': False, 'created_at': time.time()}
        atomic_json(_path(store, job_id), job)
        return job, True


def asr_status(store, project_id, film_id, job_id):
    job = json.loads(_path(store, job_id).read_text(encoding='utf-8'))
    if (job.get('schema') != SCHEMA or job.get('id') != identity(job_id)
            or job.get('project_id') != identity(project_id) or job.get('film_id') != identity(film_id)
            or job.get('fingerprint') != sha(job.get('request'))):
        raise ValueError('识别任务身份或输入快照损坏')
    if job.get('state') not in _TERMINAL | {'queued', 'running'}:
        raise ValueError('识别任务状态损坏')
    if job['state'] == 'success' and job.get('result_sha256') != sha(job.get('result')):
        raise ValueError('识别原始结果已损坏')
    return job


def cancel_asr(store, project_id, film_id, job_id):
    with _LOCK:
        job = asr_status(store, project_id, film_id, job_id)
        if job['state'] not in _TERMINAL:
            job['cancel_requested'] = True
            atomic_json(_path(store, job_id), job)
        return job


def interrupt_asr(store, project_id, film_id, job_id):
    with _LOCK:
        job = asr_status(store, project_id, film_id, job_id)
        if job['state'] not in _TERMINAL:
            job.update(state='interrupted', error='原识别进程不在当前Core，未自动重跑；请明确新建识别任务')
            atomic_json(_path(store, job_id), job)
        return job


def _decode_pcm(path, destination, decode, check, *, lifecycle=None):
    executable = resolve_ffmpeg()
    with destination.with_suffix('.log').open('w+b') as log:
        process = subprocess.Popen([executable, '-hide_banner', '-loglevel', 'error', '-nostdin', '-y',
            '-copyts', '-i', str(path), '-map', decode['stream'], '-vn', '-af', decode['filter'],
            '-ac', '1', '-ar', '16000', '-f', 'f32le', str(destination)],
            stdin=subprocess.DEVNULL, stdout=log, stderr=log,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        if lifecycle is not None:
            lifecycle.update(pid=process.pid, return_code=None)
        try:
            while process.poll() is None:
                check()
                time.sleep(.025)
            check()
            if process.returncode:
                log.seek(0)
                raise ValueError('音轨解码失败：' + log.read().decode('utf-8', errors='replace')[-2000:])
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
            if lifecycle is not None:
                lifecycle.update(return_code=process.returncode, reaped=True)
    if destination.stat().st_size == 0 or destination.stat().st_size % 4:
        raise ValueError('音轨没有完整可识别PCM')
    return {'executable_sha256': file_sha(Path(executable)), 'pcm_sha256': file_sha(destination)}


@contextmanager
def _workspace(work, job_id, cleanup):
    temporary = Path(tempfile.mkdtemp(prefix=job_id + '-', dir=work)).resolve()
    try:
        yield temporary
    finally:
        if temporary.parent != work.resolve() or temporary.is_symlink():
            raise ValueError('识别临时目录身份变化，未清理越界路径')
        try:
            shutil.rmtree(temporary)
            cleanup.update(removed=True)
        except OSError as error:
            # A delayed Windows file-handle release must not replace the
            # primary cancellation/error after the owned child was reaped.
            # Retain explicit cleanup evidence; never rerun recognition.
            cleanup.update(removed=False, directory=str(temporary), error=str(error))


def _track(job, raw):
    request = job['request']
    cues, notices = [], []
    for index, word in enumerate(raw['words']):
        start, end = word.get('start'), word.get('end')
        if (type(start) not in (int, float) or type(end) not in (int, float)
                or not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start):
            notices.append({'index': index, 'reason': 'word_has_no_valid_time', 'word': word})
            continue
        left, right = round(Fraction(str(start)) * 16000), round(Fraction(str(end)) * 16000)
        if right <= left:
            notices.append({'index': index, 'reason': 'word_below_sample_resolution', 'word': word})
            continue
        text = str(word.get('word', '')).strip()
        if (cues and left - cues[-1]['end'] <= 10400
                and right - cues[-1]['start'] <= 80000
                and len(cues[-1]['text']) + len(text) <= 42
                and not cues[-1]['text'].endswith(('.', '?', '!', '。', '？', '！', ';', '；'))):
            from .speech_verification import _uses_character_error_units
            separator = '' if text and _uses_character_error_units(text[0]) else ' '
            cues[-1].update(end=right, text=cues[-1]['text'] + separator + text)
        else:
            cues.append({'start': left, 'end': right, 'text': text})
    track = {'shot_id': request['shot_id'], 'version_id': request['version_id'],
        'media_sha256': request['media_sha256'], 'clock': {'unit': 'samples', 'rate': 16000},
        'source': {'kind': 'recognized_audio', 'asr_job_id': job['id'],
                   'report_sha256': job['result_sha256'], 'audio_sha256': job['result']['pcm']['pcm_sha256'],
                   'binding': 'actual_frozen_soundtrack_full_source_zero'}, 'cues': cues}
    return track, notices


def run_asr(store, project_id, film_id, job_id):
    import numpy as np
    import torch
    from .speech_verification import transcribe_audio
    with _LOCK:
        job = asr_status(store, project_id, film_id, job_id)
        if job['state'] != 'queued':
            return job
        job.update(state='running')
        atomic_json(_path(store, job_id), job)
    def check():
        if asr_status(store, project_id, film_id, job_id).get('cancel_requested'):
            raise ASRCancelled('识别已取消，原片和字幕稿保留')
    decoder, cleanup = {}, {}
    try:
        check()
        film = load_film(store, project_id, film_id)
        request = job['request']
        if film['sha256'] != request['film_sha256']:
            raise ValueError('冻结整片已变，不把识别重定向到新版本')
        index = next(i for i, x in enumerate(film['entries']) if x['shot_id'] == request['shot_id'])
        source = film_media(store, project_id, film_id, index)
        model = Path(request['model_directory'])
        if _model_identity(model) != request['model_identity']:
            raise ValueError('识别模型内容已变，请明确新建任务')
        if _engine_identity() != request.get('engine'):
            raise ValueError('识别引擎或实现已变，请明确新建任务')
        work = contained(store.root, 'film_asr_work')
        work.mkdir(parents=True, exist_ok=True)
        with _workspace(work, job_id, cleanup) as temporary:
            pcm = temporary / 'audio.f32le'
            pcm_identity = _decode_pcm(source, pcm, request['decode'], check, lifecycle=decoder)
            array = np.fromfile(pcm, dtype='<f4')
            if not np.isfinite(array).all():
                raise ValueError('解码音轨存在非有限数值')
            audio = {'waveform': torch.from_numpy(array).reshape(1, 1, -1), 'sample_rate': 16000}
            raw = transcribe_audio(audio, str(model), request['language'], request['beam_size'],
                                   request['cpu_threads'], cancel_check=check)
            check()
            if (file_sha(source) != request['media_sha256'] or _model_identity(model) != request['model_identity']
                    or _engine_identity() != request['engine']):
                raise ValueError('识别过程中媒体、模型或引擎已变化，未记录成功')
            result = {'request_sha256': job['fingerprint'], 'raw': raw,
                      'pcm': {**pcm_identity, 'sample_count': len(array), **request['decode']}}
        with _LOCK:
            check()
            job = asr_status(store, project_id, film_id, job_id)
            job.update(state='success', result=result, result_sha256=sha(result), finished_at=time.time(),
                       decoder=decoder, cleanup=cleanup)
            atomic_json(_path(store, job_id), job)
    except Exception as error:
        with _LOCK:
            job = asr_status(store, project_id, film_id, job_id)
            job.update(state='cancelled' if isinstance(error, ASRCancelled) else 'error',
                       error=str(error), finished_at=time.time(), decoder=decoder, cleanup=cleanup)
            atomic_json(_path(store, job_id), job)
    return job


def recognized_track(store, project_id, film_id, job_id):
    job = asr_status(store, project_id, film_id, job_id)
    film = load_film(store, project_id, film_id)
    if job['state'] != 'success' or job['request']['film_sha256'] != film['sha256']:
        raise ValueError('没有对应此冻结整片的成功识别结果')
    index = next(i for i, x in enumerate(film['entries']) if x['shot_id'] == job['request']['shot_id'])
    film_media(store, project_id, film_id, index)
    track, notices = _track(job, job['result']['raw'])
    return {'track': track, 'notices': notices, 'untimed_words': job['result']['raw'].get('untimed_words', [])}


def verify_track_source(store, film, track):
    expected = recognized_track(store, film['project_id'], film['id'], track['source'].get('asr_job_id'))['track']
    if any(track.get(key) != expected[key] for key in ('shot_id', 'version_id', 'media_sha256', 'clock', 'source')):
        raise ValueError('识别字幕来源、音轨或时钟与原始回执不一致')


def model_catalog():
    from .speech_verification import _candidate_asr_roots
    result = []
    for root in _candidate_asr_roots():
        if not root.is_dir():
            continue
        for path in [root, *sorted(p for p in root.iterdir() if p.is_dir())]:
            if (path / 'config.json').is_file() and (path / 'model.bin').is_file():
                value = {'name': path.name, 'directory': str(path.resolve())}
                if value not in result:
                    result.append(value)
    return result
