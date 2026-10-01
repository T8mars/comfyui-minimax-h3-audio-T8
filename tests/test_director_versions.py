import asyncio
from copy import deepcopy
import json
import uuid
import sys
import types

import pytest

from h3_audio_t8_pkg.director_project import ProjectStore, new_project, compile_project
from h3_audio_t8_pkg import director_routes
from h3_audio_t8_pkg.director_project import atomic_json, sha


def test_generation_snapshot_precedes_queue_and_is_immutable(tmp_path, monkeypatch):
    store = ProjectStore(tmp_path / 'user', tmp_path / 'input')
    project = new_project()
    project['doc']['shots'][0]['simplePrompt'] = 'Original prompt'
    request_id = str(uuid.uuid4())
    built = {'prompt': {'1': {'class_type': 'Test', 'inputs': {'seed': 23}}},
             'recipe': 'director_test', 'seed': 23, 'turbo_lora': None,
             'sampling': {'mode': 'single'}, 'report': {'ready': True}}
    original = deepcopy(project)
    submissions = []

    async def queue(prompt, client, prompt_id=None):
        receipt = json.loads((store.root / 'requests' / f'{request_id}.json').read_text(encoding='utf-8'))
        assert receipt['state'] == 'submitting'
        assert receipt['snapshot']['project'] == original
        submissions.append(prompt_id)
        return prompt_id

    monkeypatch.setattr(director_routes, 'queue_director_prompt', queue)
    result, status = asyncio.run(director_routes._submit_director_request_locked(
        store, project, project['current'], 23, request_id, built=built))
    assert status == 202
    second, status = asyncio.run(director_routes._submit_director_request_locked(
        store, project, project['current'], 23, request_id, built=built))
    assert status == 200 and second == result and len(submissions) == 1
    project['doc']['shots'][0]['simplePrompt'] = 'Edited later'
    built['prompt']['1']['inputs']['seed'] = 99
    snapshot = director_routes.director_result_snapshot(store, project['id'], request_id)
    assert snapshot['complete'] is True
    assert snapshot['snapshot']['project'] == original
    assert snapshot['snapshot']['prompt']['1']['inputs']['seed'] == 23
    with pytest.raises(ValueError, match='不属于'):
        director_routes.director_result_snapshot(store, str(uuid.uuid4()), request_id)
    path = store.root / 'requests' / f'{request_id}.json'
    receipt = json.loads(path.read_text(encoding='utf-8'))
    receipt['snapshot']['seed'] = 99
    path.write_text(json.dumps(receipt), encoding='utf-8')
    with pytest.raises(ValueError, match='校验失败'):
        director_routes.director_result_snapshot(store, project['id'], request_id)


def test_legacy_snapshot_is_not_invented_and_path_traversal_rejected(tmp_path):
    store = ProjectStore(tmp_path / 'user', tmp_path / 'input')
    project_id, request_id = str(uuid.uuid4()), str(uuid.uuid4())
    folder = store.root / 'requests'
    folder.mkdir(parents=True)
    (folder / f'{request_id}.json').write_text(json.dumps({'project_id': project_id}), encoding='utf-8')
    assert director_routes.director_result_snapshot(store, project_id, request_id)['complete'] is False
    with pytest.raises(ValueError):
        director_routes.director_result_snapshot(store, project_id, '../anything')


def test_preflight_errors_have_actionable_field_and_current_scope():
    project = new_project()
    selected = project['doc']['shots'][0]
    selected.update(mode='text', simplePrompt='')
    other = deepcopy(selected)
    other.update(id=str(uuid.uuid4()), mode='ends', sound='record')
    project['doc']['shots'].append(other)
    result = compile_project(project, shot_id=selected['id'])
    assert result['errors']
    assert all(error.get('shot_id') == selected['id'] for error in result['errors'])
    assert next(error for error in result['errors'] if '提示词' in error['message'])['field'] == 'prompt'


def test_copy_version_creates_idempotent_new_project_without_overwriting_source(tmp_path):
    store = ProjectStore(tmp_path/'user', tmp_path/'input')
    project = new_project()
    shot = project['doc']['shots'][0]
    shot.update(seed=1, adoptedResultId='old-film', filmTrim={'in_frame': 0, 'out_frame': 5}, simplePrompt='old text')
    store.save(project, 0)
    request_id, target_id = str(uuid.uuid4()), str(uuid.uuid4())
    snapshot = {'project': deepcopy(project), 'shot_id': shot['id'], 'seed': 42}
    atomic_json(store.root/'requests'/f'{request_id}.json', {'project_id': project['id'], 'shot_id': shot['id'],
                                                           'snapshot': snapshot, 'snapshot_sha256': sha(snapshot)})
    result = director_routes.copy_version_project(store, project['id'], request_id, target_id)
    assert result['revision'] == 1 and not result['already_created']
    copied = store.load(target_id)
    restored = copied['doc']['shots'][0]
    assert restored['seed'] == 42 and restored['simplePrompt'] == 'old text'
    assert 'adoptedResultId' not in restored and 'filmTrim' not in restored
    assert store.load(project['id'])['doc']['shots'][0]['seed'] == 1
    copied['doc']['shots'][0]['simplePrompt'] = 'edited new draft'
    store.save(copied, 1)
    assert director_routes.copy_version_project(store, project['id'], request_id, target_id)['already_created']
    assert store.load(target_id)['doc']['shots'][0]['simplePrompt'] == 'edited new draft'
    with pytest.raises(ValueError, match='覆盖'):
        director_routes.copy_version_project(store, project['id'], request_id, project['id'])


@pytest.mark.parametrize('problem', ['legacy', 'corrupt', 'unsafe-seed', 'missing-shot', 'occupied'])
def test_version_copy_refuses_incomplete_or_ambiguous_sources(tmp_path, problem):
    store = ProjectStore(tmp_path/'user', tmp_path/'input')
    project = new_project()
    request_id, target_id = str(uuid.uuid4()), str(uuid.uuid4())
    snapshot = {'project': deepcopy(project), 'shot_id': project['current'], 'seed': 12}
    if problem == 'unsafe-seed':
        snapshot['seed'] = 2**53
    if problem == 'missing-shot':
        snapshot['shot_id'] = str(uuid.uuid4())
    receipt = {'project_id': project['id'], 'shot_id': snapshot['shot_id'], 'snapshot': snapshot, 'snapshot_sha256': sha(snapshot)}
    if problem == 'legacy':
        receipt.pop('snapshot')
    if problem == 'corrupt':
        receipt['snapshot_sha256'] = 'bad'
    if problem == 'occupied':
        other = new_project()
        other['id'] = target_id
        store.save(other, 0)
    atomic_json(store.root/'requests'/f'{request_id}.json', receipt)
    with pytest.raises(ValueError):
        director_routes.copy_version_project(store, project['id'], request_id, target_id)


def test_version_copy_actual_http_routes(tmp_path, monkeypatch):
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    store = ProjectStore(tmp_path/'user', tmp_path/'input')
    project = new_project()
    request_id, target_id = str(uuid.uuid4()), str(uuid.uuid4())
    snapshot = {'project': deepcopy(project), 'shot_id': project['current'], 'seed': 0}
    atomic_json(store.root/'requests'/f'{request_id}.json', {'project_id': project['id'], 'shot_id': project['current'],
                                                           'snapshot': snapshot, 'snapshot_sha256': sha(snapshot)})
    routes = web.RouteTableDef()
    monkeypatch.setitem(sys.modules, 'server', types.SimpleNamespace(PromptServer=types.SimpleNamespace(instance=types.SimpleNamespace(routes=routes))))
    monkeypatch.setattr(director_routes, '_REGISTERED', False)
    monkeypatch.setattr(director_routes, 'get_store', lambda: store)
    director_routes.register_director_routes()

    async def scenario():
        app = web.Application()
        app.add_routes(routes)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            base = director_routes.PREFIX+f'/snapshots/{project["id"]}/{request_id}'
            response = await client.get(base)
            assert (await response.json())['complete']
            for repeated in (False, True):
                response = await client.post(base+'/copy', json={'new_project_id': target_id})
                assert response.status == 200, await response.text()
                assert (await response.json())['already_created'] == repeated
            response = await client.post(base+'/copy', json={'new_project_id': project['id']})
            assert response.status == 400
            response = await client.post(director_routes.PREFIX+f'/snapshots/{uuid.uuid4()}/{request_id}/copy', json={'new_project_id': str(uuid.uuid4())})
            assert response.status == 400
            assert store.load(target_id)['doc']['shots'][0]['seed'] == 0
        finally:
            await client.close()

    asyncio.run(scenario())


def test_version_copy_verifies_snapshot_asset_bytes_before_creating_project(tmp_path):
    from PIL import Image

    store = ProjectStore(tmp_path/'user', tmp_path/'input')
    project = new_project()
    asset_id, request_id, target_id = (str(uuid.uuid4()) for _ in range(3))
    path = store.input_root/'t8_director'/asset_id/'image.bmp'
    path.parent.mkdir(parents=True)
    Image.new('RGB', (32, 32), 'red').save(path)
    asset = store.register_asset(path, asset_id, 'image.bmp')
    project['assets'] = [asset]
    project['doc']['sharedRefs'] = [asset_id]
    snapshot = {'project': deepcopy(project), 'shot_id': project['current'], 'seed': 12}
    atomic_json(store.root/'requests'/f'{request_id}.json', {'project_id': project['id'], 'shot_id': project['current'],
                                                           'snapshot': snapshot, 'snapshot_sha256': sha(snapshot)})
    Image.new('RGB', (32, 32), 'blue').save(path)  # Same size, different actual bytes.
    assert path.stat().st_size == asset['size']
    with pytest.raises(ValueError, match='字节改变'):
        director_routes.copy_version_project(store, project['id'], request_id, target_id)
    with pytest.raises(FileNotFoundError):
        store.load(target_id)
