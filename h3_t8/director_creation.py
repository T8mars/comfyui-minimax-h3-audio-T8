"""Bounded, project-owned creative collections. No model or filesystem execution."""
from copy import deepcopy
import math
import re

TEMPLATE_FIELDS = set('name mode sound writingMode simplePrompt prompt simpleInitialized advancedInitialized refs tray first last audio start end duration manualDuration autoDuration ratio ownRatio samplingInherit sampling d3Inherit d3 events'.split())
KINDS = {'snippet': {'text', 'bindings'}, 'template': {'shot', 'bindings'},
         'recipe': {'sampling', 'd3', 'ratio', 'duration'}, 'group': {'asset_ids', 'note'}}


def validate_library(value, validate_template):
    from .director_project import identity, canonical, RATIOS
    from .director_sampling_settings import normalize_sampling

    if not isinstance(value, dict) or set(value) != {'version', 'items'} or type(value['version']) is not int or value['version'] != 1:
        raise ValueError('创作收藏需要 version=1 和 items；未知格式请保留原项目')
    items = value['items']
    if not isinstance(items, list) or len(items) > 200:
        raise ValueError('创作收藏最多 200 项')
    if len(canonical(value).encode('utf-8')) > 2*1024**2:
        raise ValueError('创作收藏超过 2MiB，请拆分工程')
    seen = set()
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get('kind'), str) or item['kind'] not in KINDS:
            raise ValueError('未知创作收藏种类')
        if set(item) != {'id', 'name', 'kind'} | KINDS[item['kind']]:
            raise ValueError('创作收藏字段缺失或包含未声明字段')
        aid = identity(item['id'])
        if aid in seen:
            raise ValueError('创作收藏 UUID 重复')
        seen.add(aid)
        if not isinstance(item['name'], str) or not item['name'].strip() or len(item['name']) > 200:
            raise ValueError('收藏名称须为 1–200 字')
        if item['kind'] in {'snippet', 'template'}:
            bindings = item['bindings']
            if not isinstance(bindings, dict) or len(bindings) > 200:
                raise ValueError('收藏素材引用表无效或超过 200 项')
            for token, asset_id in bindings.items():
                if not re.fullmatch(r'@(image|video|audio)[1-9][0-9]*', token):
                    raise ValueError('收藏引用表只能保存明确的 @ 素材编号')
                identity(asset_id)
        if item['kind'] == 'snippet':
            if not isinstance(item['text'], str) or not item['text'].strip() or len(item['text']) > 100000:
                raise ValueError('文字片段须非空且不超过 100000 字符')
        elif item['kind'] == 'template':
            shot = item['shot']
            if not isinstance(shot, dict) or not set(shot) <= TEMPLATE_FIELDS:
                raise ValueError('镜头模板包含未声明字段')
            validate_template(deepcopy(shot), aid)
        elif item['kind'] == 'recipe':
            if not isinstance(item['sampling'], dict):
                raise ValueError('采样配方需要明确的采样设置')
            normalize_sampling(item['sampling'])
            if not isinstance(item['d3'], dict) or item['ratio'] not in RATIOS:
                raise ValueError('采样配方的增强设置/画幅无效')
            if type(item['duration']) not in (int, float) or not math.isfinite(item['duration']) or item['duration'] <= 0:
                raise ValueError('配方时长须为正数')
        elif item['kind'] == 'group':
            ids = item['asset_ids']
            if not isinstance(ids, list) or not ids or len(ids) > 200:
                raise ValueError('素材组合需要 1–200 项素材')
            if len(set(identity(aid) for aid in ids)) != len(ids):
                raise ValueError('素材组合不能重复引用同一素材')
            if not isinstance(item['note'], str) or len(item['note']) > 10000:
                raise ValueError('素材组合备注最多 10000 字符')


def remap_library(value, asset_map, fresh):
    """Fresh bundle identities, including explicitly missing references; no alias guessing."""
    def asset(aid):
        return asset_map.get(aid) or fresh('missing-collection-asset', aid)

    for item in value['items']:
        old_id = item['id']
        item['id'] = fresh('collection', old_id)
        if 'bindings' in item:
            item['bindings'] = {token: asset(aid) for token, aid in item['bindings'].items()}
        if item['kind'] == 'group':
            item['asset_ids'] = [asset(aid) for aid in item['asset_ids']]
        if item['kind'] == 'template':
            shot = item['shot']
            for field in ('refs', 'tray'):
                shot[field] = [asset(aid) for aid in shot[field]]
            for field in ('first', 'last', 'audio'):
                if shot.get(field):
                    shot[field] = asset(shot[field])
            for event in shot['events']:
                event['id'] = fresh('collection-event', old_id+':'+event['id'])


def validate_project_library(project, validate_project, new_project):
    value = project['doc'].get('creationLibrary')
    if value is None:
        if 'creationLibrary' in project['doc']:
            raise ValueError('creationLibrary 不能是 null')
        return

    def check(shot, item_id):
        probe = new_project()
        shot.update(id=item_id, rev=1)
        probe['doc']['shots'] = [shot]
        probe['current'] = item_id
        validate_project(probe)  # No collection in the isolated validation probe.

    validate_library(value, check)
