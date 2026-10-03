"""immutable external takes, no queue or StageResult."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re
import shutil
import threading
import time
import uuid

from .director_project import (
    ProjectConflict, _LOCK as PROJECT_LOCK, atomic_json, contained,
    file_sha, identity, sha,
)

SCHEMA = 't8.director.external_take.v1'
PROVENANCE = {'external', 'local_mux', 'upscaled', 'repaired'}
EXTENSIONS = {'.mp4', '.mov', '.mkv', '.webm'}
MAX_BYTES = 1024**3
_LOCK = threading.RLock()


def _request(value):
    fields = {'project_id', 'shot_id', 'asset_id', 'take_id', 'expected_revision',
              'project_sha256', 'label', 'provenance_category'}
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError('外部take请求字段不完整或包含未知生成参数')
    result = deepcopy(value)
    for key in ('project_id', 'shot_id', 'asset_id', 'take_id'):
        result[key] = identity(result[key])
    if type(result['expected_revision']) is not int or result['expected_revision'] < 1:
        raise ValueError('请先保存项目后登记外部take')
    if not isinstance(result['project_sha256'], str) or not re.fullmatch('[0-9a-f]{64}', result['project_sha256']):
        raise ValueError('外部take缺少完整项目SHA')
    label = result['label']
    if not isinstance(label, str) or not label.strip() or len(label) > 200 or any(ord(char) < 32 for char in label):
        raise ValueError('take名称须为1–200字符单行文字')
    if result['provenance_category'] not in PROVENANCE:
        raise ValueError('外部take来源类别无效')
    return result


def _receipt_path(store, project_id, take_id):
    return contained(store.root, f'external_takes/{identity(project_id)}/{identity(take_id)}.json')


def _read_receipt(path, project_id, take_id):
    if path.stat().st_size > 65536:
        raise ValueError('外部take回执超过64KiB')
    value = json.loads(path.read_text(encoding='utf8'))
    if not isinstance(value, dict):
        raise ValueError('外部take回执不是对象')
    claimed = value.get('sha256')
    body = {key: item for key, item in value.items() if key != 'sha256'}
    if (body.get('schema') != SCHEMA or body.get('project_id') != identity(project_id)
            or body.get('take_id') != identity(take_id) or sha(body) != claimed
            or body.get('request_fingerprint') != sha(body.get('request'))):
        raise ValueError('外部take回执身份或完整性不符')
    request = _request(body['request'])
    if any(request[key] != body[key] for key in ('project_id', 'shot_id', 'asset_id', 'take_id')):
        raise ValueError('外部take回执归属矛盾')
    if (body.get('origin') != 'external' or body.get('can_resample') is not False
            or body.get('snapshot_available') is not False):
        raise ValueError('外部take不能声明为生成快照或可精确重采样')
    expected = f'T8_Director_External/{project_id}/{take_id}{body["extension"]}'
    if body.get('output_file') != expected or body['extension'] not in EXTENSIONS:
        raise ValueError('外部take输出路径不符')
    from .director_media import _result
    _result('external', {key: body[key] for key in ('media', 'source_timing', 'decoded_clock')})
    return value


def _record(receipt, output_root):
    source = contained(Path(output_root), receipt['output_file'])
    if source.stat().st_size != receipt['size'] or file_sha(source) != receipt['media_sha256']:
        raise ValueError('外部take成片已缺失或字节改变，不替换其它版本')
    return {'shot_id': receipt['shot_id'], 'prompt_id': None,
            'external_take_id': receipt['take_id'], 'origin': 'external',
            'can_resample': False, 'snapshot_available': False,
            'request_id': None, 'seed': None, 'state': 'success',
            'outputs': {'external': {'images': [{'filename': source.name,
                'subfolder': str(Path(receipt['output_file']).parent).replace('\\', '/'), 'type': 'output'}]}},
            'recipe': '外部take · ' + receipt['request']['label'],
            'submitted_at': receipt['registered_at'], 'shot_rev': receipt['shot_rev'],
            'media_sha256': receipt['media_sha256'], 'media': deepcopy(receipt['media']),
            'source_timing': deepcopy(receipt['source_timing']),
            'decoded_clock': deepcopy(receipt['decoded_clock']),
            'provenance_category': receipt['request']['provenance_category'],
            'provenance_is_user_label_not_generation_proof': True,
            'source_asset_id': receipt['asset_id'], 'receipt_sha256': receipt['sha256']}


def register_external_take(store, value, output_root):
    """Append one explicit saved-project take; never edit/adopt the project."""
    request = _request(value)
    project_id, shot_id, asset_id, take_id = (request[key] for key in ('project_id', 'shot_id', 'asset_id', 'take_id'))
    output_root = Path(output_root).resolve()
    receipt_path = _receipt_path(store, project_id, take_id)

    def saved_project():
        project = store.load(project_id)
        if project['revision'] != request['expected_revision'] or sha(project) != request['project_sha256']:
            raise ProjectConflict('项目版本或内容已变化；保留草稿，请重新保存并登记take')
        shot = next((item for item in project['doc']['shots'] if item['id'] == shot_id), None)
        if shot is None:
            raise ValueError('外部take镜头不属于当前已保存项目')
        if not any(item['id'] == asset_id for item in project['assets']):
            raise ValueError('外部take素材不属于当前已保存项目')
        return project, shot

    with _LOCK:
        if receipt_path.exists():
            receipt = _read_receipt(receipt_path, project_id, take_id)
            if receipt['request_fingerprint'] != sha(request):
                raise ValueError('同一个take UUID已对应不同内容，拒绝覆盖')
            return {'record': _record(receipt, output_root), 'already_registered': True}
        _, shot = saved_project()
        asset = store.asset(asset_id, verify=True)
        if asset['kind'] != 'video':
            raise ValueError('外部take必须选中已上传的视频素材')
        source = contained(store.input_root, asset['server_path'])
        suffix = source.suffix.lower()
        if suffix not in EXTENSIONS or not 0 < asset['size'] <= MAX_BYTES:
            raise ValueError('外部take须为支持格式、最多1GiB的视频')
        relative = f'T8_Director_External/{project_id}/{take_id}{suffix}'
        target = contained(output_root, relative)
        if target.exists():
            raise ValueError('外部take输出已存在但无匹配回执，拒绝覆盖；请保留证据')
        target.parent.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(target.parent).free < asset['size'] + 64 * 1024**2:
            raise ValueError('空间不足以保存不可变外部take副本')
        temporary = target.with_name(take_id + '.' + uuid.uuid4().hex + '.tmp')
        try:
            shutil.copyfile(source, temporary)
            if file_sha(temporary) != asset['sha256'] or file_sha(source) != asset['sha256']:
                raise ValueError('外部take素材在复制时改变，未登记')
            # Explicit exact-clock operation; no header-only approval fallback.
            from .director_media import run_media
            evidence = run_media('external', temporary.resolve())
            if file_sha(temporary) != asset['sha256'] or file_sha(source) != asset['sha256']:
                raise ValueError('外部take素材在解码时改变，未登记')
            receipt = {'schema': SCHEMA, 'project_id': project_id, 'shot_id': shot_id,
                       'asset_id': asset_id, 'take_id': take_id, 'extension': suffix,
                       'request': request, 'request_fingerprint': sha(request),
                       'registered_at': time.time(), 'shot_rev': shot['rev'],
                       'origin': 'external', 'can_resample': False, 'snapshot_available': False,
                       'media_sha256': asset['sha256'], 'size': asset['size'], 'output_file': relative,
                       'media': evidence['media'], 'source_timing': evidence['source_timing'],
                       'decoded_clock': evidence['decoded_clock']}
            receipt['sha256'] = sha(receipt)
            # Use the existing project CAS lock for the final check+commit;
            # don't hold it while decoding or mutate the project revision.
            with PROJECT_LOCK:
                saved_project()
                store.asset(asset_id, verify=True)
                if target.exists() or receipt_path.exists():
                    raise ValueError('外部take目标已占用，拒绝覆盖')
                temporary.replace(target)
                atomic_json(receipt_path, receipt)
            return {'record': _record(receipt, output_root), 'already_registered': False}
        finally:
            temporary.unlink(missing_ok=True)


def external_results(store, project_id, output_root):
    """Never substitute an old/local asset for a damaged frozen take."""
    project_id = identity(project_id)
    if output_root is None:
        return []
    records = []
    folder = contained(store.root, f'external_takes/{project_id}')
    for path in sorted(folder.glob('*.json')):
        take_id = None
        receipt = None
        try:
            take_id = identity(path.stem)
            path = contained(store.root, path.relative_to(store.root))
            receipt = _read_receipt(path, project_id, take_id)
            records.append(_record(receipt, output_root))
        except (OSError, ValueError, TypeError, KeyError):
            # A corrupt receipt has no trustworthy shot owner; report a
            # project-level diagnostic, never invent a playable take.
            records.append({'shot_id': receipt['shot_id'] if receipt else None,
                'prompt_id': None, 'external_take_id': take_id,
                'origin': 'external', 'can_resample': False, 'snapshot_available': False,
                'state': 'error', 'outputs': {}, 'submitted_at': 0,
                'integrity_error': '外部take回执或固定成片已缺失/改变，未替换其它版本'})
    return records


def registration_absent(store, project_id, take_id, output_root):
    """Proof at response time only; a partial/orphan output stays ambiguous."""
    project_id, take_id = identity(project_id), identity(take_id)
    with _LOCK:
        if _receipt_path(store, project_id, take_id).exists():
            return False
        return not any(contained(Path(output_root),
            f'T8_Director_External/{project_id}/{take_id}{suffix}').exists() for suffix in EXTENSIONS)
