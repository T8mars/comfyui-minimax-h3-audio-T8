"""Subset identity/seed compatibility and preflight isolation, without GPU."""
from copy import deepcopy
import uuid
import asyncio
import json
import sys
import types

import pytest

from h3_audio_t8_pkg.director_batch import (
    batch_selection, create_batch, load_batch, compile_batch_selection,
    retry_batch_item, selected_project, batch_status,
)
from h3_audio_t8_pkg.director_project import ProjectStore, new_project, atomic_json


def project_with_three():
    project = new_project()
    first = project['doc']['shots'][0]
    first['simplePrompt'] = 'First scene'
    for name in ('Middle', 'Last'):
        shot = deepcopy(first)
        shot.update(id=str(uuid.uuid4()), name=name, simplePrompt=name)
        project['doc']['shots'].append(shot)
    return project


def test_subset_freezes_project_order_explicit_seeds_and_survives_retry(tmp_path):
    store = ProjectStore(tmp_path / 'user', tmp_path / 'input')
    project = project_with_three()
    first, middle, last = [shot['id'] for shot in project['doc']['shots']]
    shots, seeds = batch_selection(project, 100, [last, first], {first: 0, last: 900})
    assert [shot['id'] for shot in shots] == [first, last]
    plans = [{'prompt': {}, 'seed': seeds[shot['id']]} for shot in shots]
    batch_id = str(uuid.uuid4())
    batch = create_batch(store, batch_id, project, 100, plans, {'assets': {}, 'models': {}},
                         shot_ids=[last, first], seed_map=seeds)
    assert [item['shot_id'] for item in batch['items']] == [first, last]
    assert [item['seed'] for item in batch['items']] == [0, 900]
    assert len(batch['project']['doc']['shots']) == 3
    project['doc']['shots'].reverse()
    project['doc']['shots'][0]['simplePrompt'] = 'Changed draft'
    restored = load_batch(store, batch_id)
    assert restored['project']['doc']['shots'][0]['id'] == first
    assert restored['project']['doc']['shots'][1]['id'] == middle
    assert retry_batch_item(store, batch_id, 1)['items'][1]['seed'] == 900
    status = batch_status(store, batch_id, lambda _: {}, tmp_path)
    assert status['schema'].endswith('.v2')
    assert len(status['items']) == 2


@pytest.mark.parametrize('kind', ['duplicate', 'unknown', 'empty', 'extra-seed', 'missing-seed', 'boolean-seed', 'fractional-seed'])
def test_invalid_selection_is_rejected(kind):
    project = project_with_three()
    first, middle, last = [shot['id'] for shot in project['doc']['shots']]
    ids, seeds = [first, last], {first: 0, last: 99}
    if kind == 'duplicate':
        ids.append(first)
    elif kind == 'unknown':
        ids.append(str(uuid.uuid4()))
    elif kind == 'empty':
        ids = []
    elif kind == 'extra-seed':
        seeds[middle] = 25
    elif kind == 'missing-seed':
        seeds.pop(last)
    elif kind == 'boolean-seed':
        seeds[first] = True
    else:
        seeds[first] = 1.5
    with pytest.raises(ValueError):
        batch_selection(project, 1, ids, seeds)


def test_selected_preflight_ignores_other_incomplete_shots_and_assets():
    project = project_with_three()
    first, middle, last = project['doc']['shots']
    first.update(simplePrompt='', mode='ends', sound='record')
    missing = str(uuid.uuid4())
    first['tray'] = [missing]
    project['assets'].append({'id': missing, 'kind': 'invalid-unused-kind'})
    report = compile_batch_selection(project, None, [last['id']])
    assert report['ready'], report['errors']
    assert [shot['id'] for shot in report['shots']] == [last['id']]
    assert selected_project(project, [last])['assets'] == []
    last['simplePrompt'] = ''
    report = compile_batch_selection(project, None, [last['id']])
    assert report['errors'][0]['shot_number'] == 3


def test_legacy_base_seeds_still_use_original_position_for_subset():
    project = project_with_three()
    last = project['doc']['shots'][-1]['id']
    _, seeds = batch_selection(project, 20, [last])
    assert seeds[last] == 22


def test_seed_map_tampering_and_built_seed_disagreement_rejected(tmp_path):
    store = ProjectStore(tmp_path / 'user', tmp_path / 'input')
    project = project_with_three()
    first = project['current']
    bid = str(uuid.uuid4())
    args = (store, bid, project, 1, [{'prompt': {}, 'seed': 9}], {'assets': {}, 'models': {}})
    with pytest.raises(ValueError, match='不一致'):
        create_batch(*args, shot_ids=[first], seed_map={first: 10})
    batch = create_batch(*args, shot_ids=[first], seed_map={first: 9})
    batch['items'][0]['seed'] = 8
    atomic_json(store.root / 'batches' / f'{bid}.json', batch)
    with pytest.raises(ValueError, match='种子'):
        load_batch(store, bid)


def test_subset_route_prepares_only_selected_shot_and_reuses_exact_batch(tmp_path, monkeypatch):
    from h3_audio_t8_pkg import director_routes

    handlers = {}
    class Routes:
        def post(self, path):
            def register(handler):
                handlers[path] = handler
                return handler
            return register
        get = post

    server = types.SimpleNamespace(routes=Routes())
    monkeypatch.setitem(sys.modules, 'server', types.SimpleNamespace(PromptServer=types.SimpleNamespace(instance=server)))
    monkeypatch.setattr(director_routes, '_REGISTERED', False)
    store = ProjectStore(tmp_path / 'user', tmp_path / 'input')
    monkeypatch.setattr(director_routes, 'get_store', lambda: store)
    seen = []
    def build(project, shot_id, store, *, seed):
        seen.append((shot_id, seed, len(project['doc']['shots'])))
        return {'prompt': {}, 'seed': seed}
    monkeypatch.setattr(director_routes, 'build_director_generation_prompt', build)
    director_routes.register_director_routes()
    features = asyncio.run(handlers[director_routes.PREFIX + '/batch-features'](None))
    assert json.loads(features.text)['selection_version'] == 2
    project = project_with_three()
    project['doc']['shots'][0].update(mode='ends', simplePrompt='', sound='record')
    last = project['doc']['shots'][-1]['id']
    body = {'batch_id': str(uuid.uuid4()), 'project': project, 'seed': 123,
            'shot_ids': [last], 'seed_map': {last: 999}}
    class Request:
        async def json(self):
            return body

    handler = handlers[director_routes.PREFIX + '/batches']
    first = asyncio.run(handler(Request()))
    assert first.status == 200, first.text
    assert seen == [(last, 999, 3)]
    second = asyncio.run(handler(Request()))
    assert json.loads(second.text) == json.loads(first.text)
    assert len(seen) == 1
    body['seed_map'][last] = 998
    assert asyncio.run(handler(Request())).status == 400
    assert len(seen) == 1
