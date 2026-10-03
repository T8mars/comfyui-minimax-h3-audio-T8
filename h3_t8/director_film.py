"""Frozen, adopted-version film playlists. CPU media only; never queue a model.

Source outputs are not changed. A content-addressed copy makes a review playlist
stable even when a Core output with the same filename is later replaced.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import threading
import uuid

from .director_project import atomic_json, contained, file_sha, identity, sha, validate_project


SCHEMA = 't8.director.film.v1'
MAX_BYTES = 20 * 1024**3
_LOCK = threading.RLock()
EXTENSIONS = {'.mp4', '.mov', '.mkv', '.webm'}


def output_videos(record):
    result = []
    outputs = record.get('outputs') or {}
    if not isinstance(outputs, dict):
        return result
    for output in outputs.values():
        if not isinstance(output, dict):
            continue
        for key in ('videos', 'gifs', 'video', 'images'):
            rows = output.get(key) or []
            if not isinstance(rows, list):
                continue
            for item in rows:
                if isinstance(item, dict) and Path(str(item.get('filename', ''))).suffix.lower() in EXTENSIONS:
                    result.append(item)
    return result


def result_key(record):
    if record.get('external_take_id'):
        return 'external:' + identity(record['external_take_id'])
    if record.get('prompt_id'):
        return record['prompt_id']
    videos = output_videos(record)
    if not videos:
        return None
    media = videos[0]
    return 'legacy:' + json.dumps([record['shot_id'], media.get('type') or 'output',
                                   (media.get('subfolder') or '').replace('\\', '/'), media['filename']],
                                  ensure_ascii=False, separators=(',', ':'))


def inspect_media(path):
    """Fully decode video/audio; report measured media, not requested shot timing."""
    from .director_media import inspect_media as inspect_isolated

    return inspect_isolated(path)


def _copy_media(store, source):
    digest = file_sha(source)
    suffix = source.suffix.lower()
    cache = contained(store.root, f'film_media/{digest}{suffix}')
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        if file_sha(cache) != digest:
            raise ValueError('已冻结的媒体副本校验失败；保留损坏证据，没有覆盖或换片')
    else:
        if shutil.disk_usage(cache.parent).free < source.stat().st_size + 64 * 1024**2:
            raise ValueError('磁盘空间不足以保存稳定成片副本')
        temporary = cache.with_name(cache.name + '.' + uuid.uuid4().hex + '.tmp')
        try:
            shutil.copyfile(source, temporary)
            if file_sha(temporary) != digest or file_sha(source) != digest:
                raise ValueError('成片在核对期间发生变化，请重新准备')
            temporary.replace(cache)
        finally:
            temporary.unlink(missing_ok=True)
    return cache, digest


def prepare_film(store, project, records, output_root):
    project = validate_project(project)
    owner = identity(project['id'])
    output_root = Path(output_root).resolve()
    entries, errors, sources, total_bytes = [], [], [], 0
    for index, shot in enumerate(project['doc']['shots']):
        entry = {'shot_id': shot['id'], 'shot_number': index + 1, 'name': shot['name'],
                 'version_id': shot.get('adoptedResultId')}
        try:
            if not entry['version_id']:
                raise ValueError('尚未采用成片版本；请先为此镜选择并采用一版')
            matches = [record for record in records if record.get('shot_id') == shot['id'] and result_key(record) == entry['version_id']]
            if len(matches) != 1 or matches[0].get('state') != 'success':
                raise ValueError('采用版本缺失或未成功，不会自动改用最新版')
            videos = output_videos(matches[0])
            if len(videos) != 1:
                raise ValueError('采用版本必须包含唯一成片，不能猜测多个输出中的哪一条')
            media = videos[0]
            if (media.get('type') or 'output') != 'output':
                raise ValueError('仅支持 Core output 中的持久成片')
            filename = media['filename']
            if Path(filename).name != filename or '/' in filename or '\\' in filename:
                raise ValueError('成片文件名必须是单一文件名')
            source = contained(output_root, str(Path((media.get('subfolder') or '').replace('\\', '/')) / filename))
            if not source.is_file():
                raise ValueError('采用版本的文件已缺失')
            total_bytes += source.stat().st_size
            if total_bytes > MAX_BYTES:
                raise ValueError('一次整片准备最多 20 GiB，请分批整理')
            sources.append((entry, shot, source))
        except (OSError, ValueError, TypeError) as error:
            errors.append({**entry, 'message': str(error)})
    if errors:
        return {'ready': False, 'errors': errors, 'entries': [], 'project_id': owner}
    with _LOCK:
        for entry, shot, source in sources:
            try:
                cache, digest = _copy_media(store, source)
                record = next(row for row in records if row.get('shot_id') == entry['shot_id'] and result_key(row) == entry['version_id'])
                external = record.get('origin') == 'external'
                if external and digest != record.get('media_sha256'):
                    raise ValueError('外片在固定副本前字节改变，拒绝更换已登记版本')
                evidence = inspect_media(cache)
                trim = shot.get('filmTrim') or {}
                start, end = trim.get('in_frame', 0), trim.get('out_frame', evidence['frames'])
                if type(start) is not int or type(end) is not int or not 0 <= start < end <= evidence['frames']:
                    raise ValueError('整片采用范围必须是成片内有效整数帧，结束帧不包含在范围中')
                entries.append({**entry, 'media_sha256': digest, 'media_file': cache.relative_to(store.root).as_posix(),
                                'source_file': source.relative_to(output_root).as_posix(), 'bytes': cache.stat().st_size,
                                'media': evidence, 'in_frame': start, 'out_frame': end,
                                'seconds': (end-start)/evidence['fps']})
                if external:
                    from copy import deepcopy
                    entries[-1].update(origin='external', can_resample=False,
                        decoded_clock=deepcopy(record['decoded_clock']), source_timing=deepcopy(record['source_timing']),
                        provenance_category=record['provenance_category'])
            except (OSError, ValueError, TypeError) as error:
                errors.append({**entry, 'message': '无法准备此镜：' + str(error)})
        if errors:
            return {'ready': False, 'errors': errors, 'entries': [], 'project_id': owner}
        manifest = {'schema': SCHEMA, 'id': str(uuid.uuid4()), 'project_id': owner,
                    'project_sha256': sha(project), 'title': project['title'], 'entries': entries,
                    'duration': sum(entry['seconds'] for entry in entries), 'ready': True}
        manifest['sha256'] = sha(manifest)
        atomic_json(contained(store.root, f"films/{manifest['id']}.json"), manifest)
        return manifest


def load_film(store, project_id, film_id):
    path = contained(store.root, f'films/{identity(film_id)}.json')
    manifest = json.loads(path.read_text(encoding='utf-8'))
    payload = {key: value for key, value in manifest.items() if key != 'sha256'}
    if (manifest.get('schema') != SCHEMA or manifest.get('id') != film_id
            or manifest.get('project_id') != identity(project_id) or sha(payload) != manifest.get('sha256')):
        raise ValueError('整片清单身份或校验失败')
    return manifest


def film_media(store, project_id, film_id, index):
    manifest = load_film(store, project_id, film_id)
    if type(index) is not int or not 0 <= index < len(manifest['entries']):
        raise ValueError('整片镜头序号无效')
    entry = manifest['entries'][index]
    path = contained(store.root, entry['media_file'])
    if not path.is_file() or file_sha(path) != entry['media_sha256']:
        raise ValueError('冻结成片副本已缺失或变化，拒绝替换播放')
    return path
