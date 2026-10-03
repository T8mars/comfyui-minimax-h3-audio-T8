"""Portable Director projects: explicit media manifests, no executable extraction.

No model imports, downloads or queue operations. Files are streamed into generated
private paths, then registered under fresh project/shot/asset identities.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import struct
import threading
import uuid
import zipfile

from .director_project import (
    atomic_json, canonical, contained, file_sha, identity, referenced_assets, sha, validate_project,
)
from .director_film import inspect_media, prepare_film

SCHEMA = 't8.director.bundle.v1'
MAX_BYTES = 20 * 1024**3
MAX_FILES = 2048
MAX_JSON = 8 * 1024**2
CHUNK = 1024**2
EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif', '.mp4', '.mov', '.mkv', '.webm',
              '.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'}
_LOCK = threading.RLock()

SHOT_FIELDS = set('id name mode sound writingMode simplePrompt prompt simpleInitialized advancedInitialized refs tray first last audio selected start end duration manualDuration autoDuration ratio ownRatio d3Inherit samplingInherit seed rev adoptedResultId'.split())
SAMPLING_FIELDS = set('mode preset resolution_mp output_mp upscaler variant hyperflow_file lora_mode'.split())
D3_FIELDS = {'semantic_bridge': 'enabled model_name alpha magnitude_match token_scope device chunk_tokens',
             'prompt_relay': 'enabled epsilon execution_mode query_chunk_rows',
             'fast_h3_v2': 'enabled profile min_tokens', 'memory': 'low_vram chunk_ffn head_chunks chunks seq_threshold'}


def _pick(value, fields):
    return {key: deepcopy(value[key]) for key in fields if key in value}


def _sampling(value):
    if value is None:
        return None
    result = _pick(value, SAMPLING_FIELDS)
    for key in ('loras', 'low_loras', 'high_loras'):
        if key in value:
            result[key] = [_pick(row, ('id', 'name', 'strength', 'enabled')) for row in value[key]]
    return result


def _d3(value):
    return {key: _pick(value[key], fields.split()) for key, fields in D3_FIELDS.items() if isinstance(value.get(key), dict)}


def _library(value):
    from .director_creation import KINDS, TEMPLATE_FIELDS
    result = {'version': 1, 'items': []}
    for item in value['items']:
        clean = _pick(item, {'id', 'name', 'kind'} | KINDS[item['kind']])
        if item['kind'] == 'template':
            clean['shot'] = _pick(item['shot'], TEMPLATE_FIELDS)
            clean['shot']['events'] = [_pick(event, ('id', 'start', 'end', 'text')) for event in item['shot']['events']]
        owner = clean['shot'] if item['kind'] == 'template' else clean
        for key, func in [('sampling', _sampling), ('d3', _d3)]:
            if owner.get(key) is not None:
                owner[key] = func(owner[key])
        result['items'].append(clean)
    return result


def portable_project(value, *, include_radar=False):
    """Explicit public settings only; keep inactive prompt/LoRA drafts, not metadata blobs."""
    value = validate_project(value)
    result = _pick(value, ('schema', 'version', 'id', 'title', 'revision', 'current'))
    doc = value['doc']
    target = _pick(doc, ('global', 'sharedRefs', 'sharedRatio', 'ratio'))
    if type(include_radar) is not bool:
        raise ValueError('证据/规则导出开关必须是布尔值')
    from .director_radar import DOC_FIELDS as RADAR_DOC_FIELDS, SHOT_FIELDS as RADAR_SHOT_FIELDS
    has_radar = any(key in doc for key in RADAR_DOC_FIELDS) or any(
        any(key in shot for key in RADAR_SHOT_FIELDS) for shot in doc['shots'])
    if include_radar and has_radar:
        target.update(_pick(doc, RADAR_DOC_FIELDS))
        result['radarPortability'] = {'schema': 't8.director.radar_portability.v1', 'include_evidence': True}
    if 'creationLibrary' in doc:
        target['creationLibrary'] = _library(doc['creationLibrary'])
    if 'generation' in doc:
        target['generation'] = _pick(doc['generation'], ('unet', 'clip', 'video_vae', 'audio_vae', 'lora_mode', 'lora', 'lora_strength', 'resolution_mp'))
        if 'loras' in doc['generation']:
            target['generation']['loras'] = [_pick(row, ('name', 'strength', 'enabled')) for row in doc['generation']['loras']]
    for source, destination in [(doc, target)]:
        if source.get('d3') is not None:
            destination['d3'] = _d3(source['d3'])
        if source.get('sampling') is not None:
            destination['sampling'] = _sampling(source['sampling'])
    target['shots'] = []
    for shot in doc['shots']:
        clean = _pick(shot, SHOT_FIELDS)
        if include_radar:
            clean.update(_pick(shot, RADAR_SHOT_FIELDS))
        seed = clean.get('seed')
        if seed is not None and (type(seed) is not int or not 0 <= seed <= 2**53-1):
            raise ValueError('镜头种子无法在页面精确表达，请先修正再打包')
        clean['events'] = [_pick(event, ('id', 'start', 'end', 'text')) for event in shot['events']]
        for key, func in [('d3', _d3), ('sampling', _sampling)]:
            if shot.get(key) is not None:
                clean[key] = func(shot[key])
        if shot.get('filmTrim') is not None:
            clean['filmTrim'] = _pick(shot['filmTrim'], ('in_frame', 'out_frame'))
        target['shots'].append(clean)
    result['doc'] = target
    excluded = set()
    if has_radar and not include_radar:
        from .director_radar import _packet_media
        evidence_assets = {row['asset_id'] for shot in doc['shots']
                           for evidence in shot.get('evidenceHistory', []) + ([shot['sourceEvidence']] if shot.get('sourceEvidence') else [])
                           for row in _packet_media(evidence['packet'])}
        keep = referenced_assets(result)
        def library_references(node):
            if isinstance(node, str):
                if node in evidence_assets:
                    keep.add(node)
            elif isinstance(node, dict):
                for child in node.values():
                    library_references(child)
            elif isinstance(node, list):
                for child in node:
                    library_references(child)
        library_references(target.get('creationLibrary'))
        excluded = evidence_assets - keep
    result['assets'] = [_pick(asset, ('id', 'name', 'kind', 'width', 'height', 'duration', 'has_audio', 'sha256', 'size', 'source_frame'))
                        for asset in value['assets'] if asset['id'] not in excluded]
    # Model names are retained for manual installation, not absolute machine paths or URLs.
    def model_names(node):
        if isinstance(node, dict):
            for key, item in node.items():
                if key in {'unet', 'clip', 'video_vae', 'audio_vae', 'upscaler', 'hyperflow_file', 'model_name', 'name', 'lora'}:
                    for name in item if isinstance(item, list) else [item]:
                        if isinstance(name, str) and (':' in name or name.startswith(('/', '\\')) or '..' in name.replace('\\', '/').split('/')):
                            raise ValueError('模型配置包含本机绝对路径或网址；请先改用模型列表中的相对文件名')
                else:
                    model_names(item)
        elif isinstance(node, list):
            for item in node:
                model_names(item)
    library_owners = [item.get('shot', item) for item in target.get('creationLibrary', {}).get('items', [])]
    for owner in [target, *target['shots'], *library_owners]:
        for field in ('generation', 'sampling', 'd3'):
            model_names(owner.get(field))
    return validate_project(result)


def _copy(source, target, digest):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if file_sha(target) != digest:
            raise ValueError('已有媒体身份或字节冲突，拒绝覆盖')
        return
    if shutil.disk_usage(target.parent).free < source.stat().st_size + 64*1024**2:
        raise ValueError('磁盘空间不足以保存工程素材副本')
    temporary = target.with_name(target.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        shutil.copyfile(source, temporary)
        if file_sha(temporary) != digest:
            raise ValueError('素材复制时发生变化，未创建工程包')
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _sealed(path):
    result = json.loads(path.read_text(encoding='utf-8'))
    if sha({key: value for key, value in result.items() if key != 'sha256'}) != result.get('sha256'):
        raise ValueError('工程包记录校验失败')
    return result


def _save_sealed(path, value):
    value = {key: item for key, item in value.items() if key != 'sha256'}
    value['sha256'] = sha(value)
    atomic_json(path, value)
    return value


def prepare_bundle(store, value, records, output_root, include_results=True, *, include_radar=False):
    if type(include_results) is not bool:
        raise ValueError('采用成片开关必须是布尔值')
    project = portable_project(value, include_radar=include_radar)
    rows, sources, errors = [], {}, []
    assets = {asset['id']: asset for asset in project['assets']}
    for missing in referenced_assets(project)-assets.keys():
        errors.append({'name': missing, 'message': '工程引用的素材不在素材清单中'})
    for asset in project['assets']:
        try:
            actual = store.asset(asset['id'], verify=True)
            if any(asset.get(key) != actual.get(key) for key in ('sha256', 'size', 'kind')):
                raise ValueError('草稿素材身份与服务端不一致，请先重连')
            source = contained(store.input_root, actual['server_path'])
            suffix = source.suffix.lower()
            if suffix not in EXTENSIONS:
                raise ValueError('不支持的素材文件格式')
            path = f"assets/{asset['id']}{suffix}"
            rows.append({'role': 'asset', 'asset_id': asset['id'], 'name': asset['name'], 'path': path,
                         'sha256': actual['sha256'], 'size': actual['size']})
            sources[path] = source
        except (OSError, ValueError) as error:
            errors.append({'name': asset.get('name', asset['id']), 'message': str(error)})
    adopted = [shot for shot in project['doc']['shots'] if shot.get('adoptedResultId')]
    if include_results and adopted:
        selected = deepcopy(project)
        selected['doc']['shots'], selected['current'] = adopted, adopted[0]['id']
        film = prepare_film(store, selected, records, output_root)
        if not film['ready']:
            errors.extend(film['errors'])
        else:
            for entry in film['entries']:
                source = contained(store.root, entry['media_file'])
                path = f"results/{entry['shot_id']}{source.suffix.lower()}"
                rows.append({'role': 'result', 'shot_id': entry['shot_id'], 'version_id': entry['version_id'],
                             'name': entry['name'], 'path': path, 'sha256': entry['media_sha256'], 'size': entry['bytes']})
                if entry.get('origin') == 'external':
                    rows[-1]['external'] = {'schema': 't8.director.external_bundle.v1', 'origin': 'external',
                        'can_resample': False, 'provenance_category': entry['provenance_category']}
                sources[path] = source
    if not include_results:
        for shot in project['doc']['shots']:
            shot.pop('adoptedResultId', None)
            shot.pop('filmTrim', None)
    total = sum(row['size'] for row in rows)
    if total > MAX_BYTES or len(rows) > MAX_FILES:
        errors.append({'message': '工程包最多20GiB媒体、2048个文件，请分项目整理'})
    if errors:
        return {'ready': False, 'errors': errors}
    with _LOCK:
        bundle_id = str(uuid.uuid4())
        cache_sources = {}
        for row in rows:
            target = contained(store.root, f"bundle_media/{row['sha256']}{Path(row['path']).suffix}")
            _copy(sources[row['path']], target, row['sha256'])
            cache_sources[row['path']] = target.relative_to(store.root).as_posix()
        manifest = {'schema': SCHEMA, 'project': project, 'files': rows}
        manifest['sha256'] = sha(manifest)
        if len(canonical(manifest).encode('utf-8')) > MAX_JSON:
            raise ValueError('工程设置超过8MiB，请拆分工程')
        _save_sealed(contained(store.root, f'bundle_exports/{bundle_id}.json'),
                     {'id': bundle_id, 'project_id': project['id'], 'manifest': manifest, 'sources': cache_sources})
        return {'ready': True, 'id': bundle_id, 'project_id': project['id'], 'title': project['title'],
                'files': rows, 'media_bytes': total, 'shots': len(project['doc']['shots']),
                'notice': '只含工程文字、模型文件名、输入素材和所选采用成片；不含模型权重、运行记录、其它历史或未知扩展字段。文字和媒体可能含私人内容，请自行检查后分享。'}


def build_bundle(store, project_id, bundle_id):
    with _LOCK:
        plan = _sealed(contained(store.root, f'bundle_exports/{identity(bundle_id)}.json'))
        if plan['project_id'] != identity(project_id):
            raise ValueError('工程包不属于此项目')
        path = contained(store.root, f'bundle_exports/{bundle_id}.zip')
        if path.exists():
            if file_sha(path) != plan.get('archive_sha256'):
                raise ValueError('已导出工程包损坏，拒绝覆盖')
            return path
        temporary = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
        try:
            if shutil.disk_usage(path.parent).free < sum(row['size'] for row in plan['manifest']['files'])+64*1024**2:
                raise ValueError('磁盘空间不足以生成ZIP')
            with zipfile.ZipFile(temporary, 'x', compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
                archive.writestr('manifest.json', canonical(plan['manifest']).encode('utf-8'))
                for row in plan['manifest']['files']:
                    source = contained(store.root, plan['sources'][row['path']])
                    if file_sha(source) != row['sha256']:
                        raise ValueError('固定素材副本已改变，拒绝打包')
                    digest = hashlib.sha256()
                    with source.open('rb') as incoming, archive.open(row['path'], 'w', force_zip64=True) as outgoing:
                        for chunk in iter(lambda: incoming.read(CHUNK), b''):
                            digest.update(chunk)
                            outgoing.write(chunk)
                    if digest.hexdigest() != row['sha256']:
                        raise ValueError('素材在打包期间变化')
            plan['archive_sha256'] = file_sha(temporary)
            os.replace(temporary, path)
            _save_sealed(contained(store.root, f'bundle_exports/{bundle_id}.json'), plan)
        finally:
            temporary.unlink(missing_ok=True)
        return path


def _json_unique(data):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('工程包JSON包含重复字段')
            result[key] = value
        return result
    return json.loads(data, object_pairs_hook=unique)


def _zip_limits(path):
    if path.stat().st_size > MAX_BYTES+MAX_JSON+4*1024**2:
        raise ValueError('ZIP超过20GiB上限')
    # Bound the central directory before ZipFile allocates one object per member.
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size-65557))
        tail = stream.read(65557)
    offset = tail.rfind(b'PK\x05\x06')
    if offset < 0 or len(tail)-offset < 22:
        raise ValueError('ZIP目录缺失')
    disk, start_disk, disk_count, count, size, _, comment = struct.unpack_from('<4H2LH', tail, offset+4)
    if disk or start_disk or disk_count != count or count > MAX_FILES+1 or size > 4*1024**2 or len(tail)-offset != 22+comment:
        raise ValueError('ZIP分卷或目录规模不受支持')


def inspect_bundle(store, upload_id):
    try:
        return _inspect_bundle(store, upload_id)
    except (zipfile.BadZipFile, EOFError, RuntimeError) as error:
        raise ValueError('ZIP结构或压缩数据损坏，工程未导入') from error


def _inspect_bundle(store, upload_id):
    """Validate everything before creating any project or globally registered asset."""
    upload_id = identity(upload_id)
    source = contained(store.root, f'bundle_imports/{upload_id}.zip')
    _zip_limits(source)
    archive_sha = file_sha(source)
    with _LOCK, zipfile.ZipFile(source) as archive:
        infos = archive.infolist()
        if len(infos) > MAX_FILES+1 or len({info.filename.casefold() for info in infos}) != len(infos):
            raise ValueError('ZIP文件名重复或文件数超限')
        total = 0
        for info in infos:
            name = info.filename
            if (info.is_dir() or info.orig_filename != name or '\\' in name or ':' in name or '\0' in name or name != str(PurePosixPath(name))
                    or PurePosixPath(name).is_absolute() or '..' in PurePosixPath(name).parts
                    or stat.S_ISLNK(info.external_attr >> 16) or info.flag_bits & 1
                    or info.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}):
                raise ValueError('ZIP包含危险路径、链接或不支持的压缩/加密文件')
            total += info.file_size
            if total > MAX_BYTES+MAX_JSON or info.file_size > max(1, info.compress_size)*200:
                raise ValueError('ZIP解压大小或压缩比例超限')
        metadata = archive.getinfo('manifest.json')
        if shutil.disk_usage(source.parent).free < total+64*1024**2:
            raise ValueError('磁盘空间不足以检查解压素材')
        if metadata.file_size > MAX_JSON:
            raise ValueError('工程清单超过8MiB')
        manifest = _json_unique(archive.read(metadata))
        if (not isinstance(manifest, dict) or manifest.get('schema') != SCHEMA or set(manifest) != {'schema', 'project', 'files', 'sha256'}
                or sha({key: value for key, value in manifest.items() if key != 'sha256'}) != manifest.get('sha256')):
            raise ValueError('工程包版本、字段或清单SHA校验失败')
        radar = manifest['project'].get('radarPortability')
        if radar is not None and radar != {'schema': 't8.director.radar_portability.v1', 'include_evidence': True}:
            raise ValueError('证据/规则工程包版本或导出确认记录无效')
        project = portable_project(manifest['project'], include_radar=radar is not None)
        if project != manifest['project']:
            raise ValueError('工程包包含未声明的扩展字段或本机路径')
        rows = manifest['files']
        if not isinstance(rows, list) or len(rows) > MAX_FILES:
            raise ValueError('文件清单无效')
        names, asset_ids, shot_ids = set(), set(), set()
        assets = {asset['id']: asset for asset in project['assets']}
        shots = {shot['id']: shot for shot in project['doc']['shots']}
        for row in rows:
            role = row['role']
            owner = identity(row.get('asset_id') if role == 'asset' else row.get('shot_id'))
            suffix = Path(row['path']).suffix.lower()
            expected = f"{'assets' if role == 'asset' else 'results'}/{owner}{suffix}"
            if (role not in {'asset', 'result'} or row['path'] != expected or suffix not in EXTENSIONS
                    or not re.fullmatch('[0-9a-f]{64}', row['sha256']) or type(row['size']) is not int or row['size'] <= 0
                    or row['path'] in names):
                raise ValueError('文件清单路径、身份、大小或SHA无效')
            names.add(row['path'])
            expected_fields = {'role', 'name', 'path', 'sha256', 'size', 'asset_id'} if role == 'asset' else {'role', 'name', 'path', 'sha256', 'size', 'shot_id', 'version_id'}
            if role == 'result' and 'external' in row:
                expected_fields.add('external')
                extra = row['external']
                from .director_external import PROVENANCE
                if (not isinstance(extra, dict) or set(extra) != {'schema', 'origin', 'can_resample', 'provenance_category'}
                        or extra.get('schema') != 't8.director.external_bundle.v1' or extra.get('origin') != 'external'
                        or extra.get('can_resample') is not False or extra.get('provenance_category') not in PROVENANCE):
                    raise ValueError('工程包外片来源合同无效，不可虚构生成祖先')
            if set(row) != expected_fields:
                raise ValueError('文件清单包含未知字段')
            if role == 'asset':
                if owner in asset_ids or owner not in assets or any(assets[owner].get(key) != row[key] for key in ('sha256', 'size')):
                    raise ValueError('素材身份重复或与工程不匹配')
                asset_ids.add(owner)
            else:
                if owner in shot_ids or owner not in shots or shots[owner].get('adoptedResultId') != row['version_id'] or suffix not in {'.mp4', '.mov', '.mkv', '.webm'}:
                    raise ValueError('采用成片身份重复或与镜头不匹配')
                shot_ids.add(owner)
            if archive.getinfo(row['path']).file_size != row['size']:
                raise ValueError('清单文件大小不匹配')
        if (names | {'manifest.json'} != {info.filename for info in infos} or asset_ids != set(assets)
                or referenced_assets(project)-asset_ids
                or shot_ids != {shot['id'] for shot in shots.values() if shot.get('adoptedResultId')}):
            raise ValueError('ZIP存在未声明/缺失文件，或工程引用缺失')
        for row in rows:
            target = contained(store.root, f"bundle_staging/{upload_id}/{row['path']}")
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name+'.'+uuid.uuid4().hex+'.tmp')
            digest, count = hashlib.sha256(), 0
            try:
                with archive.open(row['path']) as incoming, temporary.open('xb') as outgoing:
                    while chunk := incoming.read(CHUNK):
                        count += len(chunk)
                        if count > row['size']:
                            raise ValueError('ZIP实际解压超过声明大小')
                        digest.update(chunk)
                        outgoing.write(chunk)
                if count != row['size'] or digest.hexdigest() != row['sha256']:
                    raise ValueError('素材SHA不一致，工程未导入')
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
            if row['role'] == 'result':
                evidence = inspect_media(target)
                if 'external' in row:
                    from .director_media import run_media
                    run_media('external', target.resolve())
                trim = shots[row['shot_id']].get('filmTrim') or {}
                start, end = trim.get('in_frame', 0), trim.get('out_frame', evidence['frames'])
                if type(start) is not int or type(end) is not int or not 0 <= start < end <= evidence['frames']:
                    raise ValueError('采用成片裁剪范围无效')
            else:
                from PIL import Image
                kind = assets[row['asset_id']]['kind']
                if kind == 'image':
                    with Image.open(target) as image:
                        image.verify()
                elif kind in {'video', 'audio'}:
                    from .director_media import validate_media
                    validate_media(target, kind)
                else:
                    raise ValueError('素材种类无效')
        if file_sha(source) != archive_sha:
            raise ValueError('ZIP在校验期间改变')
        _save_sealed(contained(store.root, f'bundle_imports/{upload_id}.json'),
                     {'id': upload_id, 'archive_sha256': archive_sha, 'manifest': manifest})
        return {'ready': True, 'id': upload_id, 'title': project['title'], 'files': rows, 'shots': len(shots),
                'media_bytes': sum(row['size'] for row in rows), 'archive_sha256': archive_sha}


def import_bundle(store, output_root, upload_id, new_project_id):
    with _LOCK:
        upload_id, new_project_id = identity(upload_id), identity(new_project_id)
        plan = _sealed(contained(store.root, f'bundle_imports/{upload_id}.json'))
        project = validate_project(plan['manifest']['project'])
        if project['id'] == new_project_id:
            raise ValueError('导入必须创建新工程，不能覆盖来源工程')
        origin = {'archive_sha256': plan['archive_sha256'], 'source_project_id': project['id']}
        job_path = contained(store.root, f'bundle_import_jobs/{new_project_id}.json')
        previous_job = _sealed(job_path) if job_path.exists() else None
        try:
            existing = store.load(new_project_id)
        except FileNotFoundError:
            existing = None
        if existing:
            if existing.get('bundleSource') != origin and not (previous_job and previous_job['source'] == origin and previous_job['state'] == 'success'):
                raise ValueError('新工程身份已被使用，拒绝覆盖')
            return {'id': new_project_id, 'revision': existing['revision'], 'already_created': True}
        if previous_job and previous_job['source'] != origin:
            raise ValueError('此导入身份已用于其它工程包')
        _save_sealed(job_path, {'id': new_project_id, 'source': origin, 'state': 'importing'})
        namespace = uuid.UUID(new_project_id)
        def fresh(kind, old):
            return str(uuid.uuid5(namespace, kind+':'+old))
        asset_map = {asset['id']: fresh('asset', asset['id']) for asset in project['assets']}
        shot_map = {shot['id']: fresh('shot', shot['id']) for shot in project['doc']['shots']}
        assets, results = [], []
        # Re-verify every staged file before any public registration.
        for row in plan['manifest']['files']:
            source = contained(store.root, f"bundle_staging/{upload_id}/{row['path']}")
            if source.stat().st_size != row['size'] or file_sha(source) != row['sha256']:
                raise ValueError('已检查的工程包素材发生变化，请重新检查原包')
        for row in plan['manifest']['files']:
            source = contained(store.root, f"bundle_staging/{upload_id}/{row['path']}")
            suffix = source.suffix.lower()
            if row['role'] == 'asset':
                new_id = asset_map[row['asset_id']]
                original = next(item for item in project['assets'] if item['id'] == row['asset_id'])
                target = contained(store.input_root, f't8_director/{new_id}/source{suffix}')
                _copy(source, target, row['sha256'])
                try:
                    asset = store.asset(new_id, verify=True)
                except FileNotFoundError:
                    asset = store.register_asset(target, new_id, row['name'], source_frame=original.get('source_frame'))
                if asset['sha256'] != row['sha256']:
                    raise ValueError('导入资产UUID冲突')
                if asset.get('source_frame') != original.get('source_frame'):
                    raise ValueError('导入取帧素材来源冲突，拒绝覆盖')
                if asset['kind'] != original['kind']:
                    raise ValueError('素材真实类型与工程声明不符')
                assets.append(asset)
            else:
                shot_id = shot_map[row['shot_id']]
                version_id = fresh('version', row['shot_id'])
                folder = f'T8_Director_Imported/{new_project_id}'
                target = contained(Path(output_root), f'{folder}/{shot_id}{suffix}')
                _copy(source, target, row['sha256'])
                results.append({'shot_id': shot_id, 'prompt_id': version_id, 'state': 'success',
                                'outputs': {'bundle': {'images': [{'filename': target.name, 'subfolder': folder, 'type': 'output'}]}},
                                'recipe': '导入采用成片（原工程）', 'submitted_at': 0, 'shot_rev': None,
                                'snapshot_available': False, 'source_bundle_sha256': plan['archive_sha256'],
                                'media_sha256': row['sha256']})
                if 'external' in row:
                    from .director_media import run_media
                    measured = run_media('external', target.resolve())
                    if file_sha(target) != row['sha256']:
                        raise ValueError('外片在导入解码期间改变，拒绝登记')
                    results[-1].update(prompt_id=None, external_take_id=version_id, origin='external',
                        can_resample=False, seed=None, request_id=None,
                        recipe='导入外部成片（不可精确重采样）',
                        provenance_category=row['external']['provenance_category'],
                        provenance_is_user_label_not_generation_proof=True, **measured)
        project['doc']['sharedRefs'] = [asset_map[value] for value in project['doc']['sharedRefs']]
        for shot in project['doc']['shots']:
            old_id = shot['id']
            shot['id'] = shot_map[old_id]
            for key in ('tray', 'refs'):
                shot[key] = [asset_map[value] for value in shot[key]]
            for key in ('first', 'last', 'audio', 'selected'):
                if shot.get(key):
                    shot[key] = asset_map[shot[key]]
            for event in shot['events']:
                event['id'] = fresh('event', old_id+':'+event['id'])
            if shot.get('adoptedResultId'):
                external = any(row['role'] == 'result' and row['shot_id'] == old_id and 'external' in row for row in plan['manifest']['files'])
                shot['adoptedResultId'] = ('external:' if external else '') + fresh('version', old_id)
        if 'creationLibrary' in project['doc']:
            from .director_creation import remap_library
            remap_library(project['doc']['creationLibrary'], asset_map, fresh)
        from .director_radar import remap_project
        remap_project(project, asset_map, shot_map)
        project.update(id=new_project_id, revision=0, current=shot_map[project['current']], assets=assets,
                       title=project['title'][:180]+' · 导入副本', bundleSource=origin)
        _save_sealed(contained(store.root, f'bundle_results/{new_project_id}.json'),
                     {'project_id': new_project_id, 'source': origin, 'results': results})
        saved = store.save(project, 0)
        _save_sealed(job_path, {'id': new_project_id, 'source': origin, 'state': 'success'})
        return {'id': new_project_id, 'revision': saved['revision'], 'already_created': False}


def import_preview(store, upload_id):
    plan = _sealed(contained(store.root, f'bundle_imports/{identity(upload_id)}.json'))
    project, rows = plan['manifest']['project'], plan['manifest']['files']
    return {'ready': True, 'id': upload_id, 'title': project['title'], 'files': rows,
            'shots': len(project['doc']['shots']), 'media_bytes': sum(row['size'] for row in rows),
            'archive_sha256': plan['archive_sha256']}


def imported_results(store, project_id, output_root=None):
    path = contained(store.root, f'bundle_results/{identity(project_id)}.json')
    if not path.exists():
        return []
    value = _sealed(path)
    if value['project_id'] != project_id:
        raise ValueError('导入成片归属身份不符')
    records = value['results']
    for record in records:
        record['integrity_error'] = None
    if output_root is not None:
        for record in records:
            media = record['outputs']['bundle']['images'][0]
            try:
                source = contained(Path(output_root), media['subfolder']+'/'+media['filename'])
                if file_sha(source) != record['media_sha256']:
                    raise ValueError('导入成片SHA不符')
            except (OSError, ValueError):
                record.update(state='error', outputs={}, integrity_error='导入采用成片已缺失或字节改变；不会替换其它版本')
    return records
