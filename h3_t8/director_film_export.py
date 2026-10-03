"""Explicit CPU deliverable encoding. Independent from H3 sampling/long-video chains."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
import time

from .director_film import film_media, inspect_media, load_film
from .director_project import atomic_json, contained, file_sha, identity, sha
from .ffmpeg_utils import resolve_ffmpeg


POLICY = 'h264_24fps_aac48k_stereo_contain_v1'


def _path(store, job_id):
    return contained(store.root, f'film_exports/{identity(job_id)}.json')


def reserve_export(store, project_id, film_id, job_id, options):
    manifest = load_film(store, project_id, film_id)
    if options.get('policy') != POLICY or options.get('confirm_encoding') is not True:
        raise ValueError('请确认导出规格：24fps H.264、48kHz双声道AAC、等比留边，无转场或音量处理')
    width, height = options.get('width'), options.get('height')
    if any(type(value) is not int or value % 2 or not 32 <= value <= 4096 for value in (width, height)):
        raise ValueError('导出宽高必须是 32–4096 内的偶数')
    if any(abs(entry['media']['fps']-24) > 0.00001 for entry in manifest['entries']):
        raise ValueError('当前导出只接受24fps成片，不会偷偷改变其它帧率')
    if any(entry.get('origin') == 'external' and entry.get('decoded_clock', {}).get('zero_origin_film_compatible') is not True
           for entry in manifest['entries']):
        raise ValueError('外片有非零视频/音频起点；请显式转换并登记新take，不会静默改写PTS或声音偏移')
    silent = options.get('allow_silent') is True
    if not silent and any(not entry['media']['has_audio'] for entry in manifest['entries']):
        raise ValueError('有镜头没有音轨；请明确允许为无音轨镜头补静音')
    specification = {'width': width, 'height': height, 'policy': POLICY, 'allow_silent': silent}
    fingerprint = sha({'manifest_sha256': manifest['sha256'], 'options': specification})
    path = _path(store, job_id)
    if path.exists():
        existing = export_status(store, project_id, film_id, job_id)
        if existing['fingerprint'] != fingerprint:
            raise ValueError('导出请求身份已用于其他规格')
        return existing, False
    job = {'schema': 't8.director.film_export.v1', 'id': identity(job_id), 'project_id': project_id,
           'film_id': film_id, 'manifest_sha256': manifest['sha256'], 'options': specification,
           'fingerprint': fingerprint, 'state': 'queued', 'done': 0, 'total': len(manifest['entries']),
           'created_at': time.time()}
    atomic_json(path, job)
    return job, True


def export_status(store, project_id, film_id, job_id):
    job = json.loads(_path(store, job_id).read_text(encoding='utf-8'))
    if (job.get('schema') != 't8.director.film_export.v1' or job.get('id') != job_id
            or job.get('project_id') != identity(project_id) or job.get('film_id') != identity(film_id)):
        raise ValueError('导出任务身份不一致')
    expected = sha({'manifest_sha256': job['manifest_sha256'], 'options': job['options']})
    if job.get('fingerprint') != expected:
        raise ValueError('导出规格快照已损坏')
    if job['state'] == 'success' and job.get('result_sha256') != sha({'media': job.get('media'), 'output_sha256': job.get('output_sha256')}):
        raise ValueError('导出成片回执已损坏')
    return job


def export_file(store, project_id, film_id, job_id):
    job = export_status(store, project_id, film_id, job_id)
    if job['state'] != 'success':
        raise ValueError('导出尚未成功')
    path = contained(store.root, f'film_exports/{identity(job_id)}.mp4')
    if not path.is_file() or file_sha(path) != job.get('output_sha256'):
        raise ValueError('导出成片已缺失或变化')
    return path


def _run(args, log):
    with log.open('ab') as stream:
        result = subprocess.run(args, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0), timeout=3600, check=False)
    if result.returncode:
        raise ValueError('FFmpeg 导出失败：'+log.read_text(encoding='utf-8', errors='replace')[-1800:])


def run_export(store, project_id, film_id, job_id):
    job = export_status(store, project_id, film_id, job_id)
    if job['state'] != 'queued':
        return job
    path = _path(store, job_id)
    try:
        manifest = load_film(store, project_id, film_id)
        if manifest['sha256'] != job['manifest_sha256']:
            raise ValueError('导出对应的整片清单已变化')
        executable = resolve_ffmpeg()
        width, height = job['options']['width'], job['options']['height']
        job['state'] = 'running'
        atomic_json(path, job)
        log = contained(store.root, f'film_exports/{identity(job_id)}.log')
        work_root = contained(store.root, 'film_export_work')
        work_root.mkdir(parents=True, exist_ok=True)
        # Only the newly allocated, export-owned temporary directory is removed.
        with tempfile.TemporaryDirectory(prefix=job_id+'-', dir=work_root) as temporary:
            work = Path(temporary)
            names = []
            frames_total = 0
            for index, entry in enumerate(manifest['entries']):
                source = film_media(store, project_id, film_id, index)
                frames = entry['out_frame']-entry['in_frame']
                frames_total += frames
                seconds = frames/24
                name = f'clip-{index:04}.mov'
                names.append(name)
                args = [executable, '-hide_banner', '-loglevel', 'error', '-nostdin', '-y', '-threads', '1', '-i', str(source)]
                audio_input = '0:a:0'
                if not entry['media']['has_audio']:
                    args += ['-f', 'lavfi', '-i', 'anullsrc=r=48000:cl=stereo']
                    audio_input = '1:a:0'
                vf = (f"trim=start_frame={entry['in_frame']}:end_frame={entry['out_frame']},setpts=N/(24*TB),"
                      f'scale={width}:{height}:force_original_aspect_ratio=decrease:force_divisible_by=2,'
                      f'pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1')
                start = entry['in_frame']/24 if entry['media']['has_audio'] else 0
                af = f'atrim=start={start:.12f}:duration={seconds:.12f},asetpts=PTS-STARTPTS,aresample=48000,apad,atrim=duration={seconds:.12f}'
                args += ['-map', '0:v:0', '-map', audio_input, '-vf', vf, '-af', af,
                         '-c:v', 'libx264', '-threads', '1', '-filter_threads', '1', '-preset', 'medium', '-crf', '18',
                         '-pix_fmt', 'yuv420p', '-bf', '0', '-refs', '1', '-r', '24', '-video_track_timescale', '24000',
                         '-c:a', 'pcm_s16le', '-ar', '48000', '-ac', '2', '-t', f'{seconds:.12f}', str(work/name)]
                _run(args, log)
                evidence = inspect_media(work/name)
                if evidence['frames'] != frames or (evidence['width'], evidence['height']) != (width, height):
                    raise ValueError('中间交付片帧数或尺寸不一致')
                if file_sha(source) != entry['media_sha256']:
                    raise ValueError('编码期间冻结媒体发生变化')
                job['done'] = index+1
                atomic_json(path, job)
            concat = work/'clips.txt'
            concat.write_text(''.join(f"file '{name}'\n" for name in names), encoding='utf-8')
            final = work/'film.mp4'
            _run([executable, '-hide_banner', '-loglevel', 'error', '-nostdin', '-y', '-f', 'concat', '-safe', '1',
                  '-i', str(concat), '-map', '0:v:0', '-map', '0:a:0', '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k',
                  '-ar', '48000', '-ac', '2', '-threads', '1', '-video_track_timescale', '24000', '-movflags', '+faststart', str(final)], log)
            evidence = inspect_media(final)
            if evidence['frames'] != frames_total or (evidence['width'], evidence['height']) != (width, height):
                raise ValueError('最终整片帧数或尺寸不一致')
            if not evidence['has_audio'] or abs(evidence['audio_samples']/evidence['audio_rate']-frames_total/24) > 2048/48000:
                raise ValueError('最终整片音频时长不一致')
            destination = contained(store.root, f'film_exports/{identity(job_id)}.mp4')
            if destination.exists():
                raise ValueError('导出目标已经存在，拒绝覆盖')
            final.replace(destination)
            job.update(state='success', media=evidence, output_sha256=file_sha(destination), finished_at=time.time())
            job['result_sha256'] = sha({'media': evidence, 'output_sha256': job['output_sha256']})
    except Exception as error:
        job.update(state='error', error=str(error), finished_at=time.time())
    atomic_json(path, job)
    return job
