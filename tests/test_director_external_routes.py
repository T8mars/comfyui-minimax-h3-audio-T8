"""Actual guarded external registration, not a mocked generation endpoint."""
from copy import deepcopy
import uuid

import pytest

from h3_audio_t8_pkg import director_routes
from h3_audio_t8_pkg.director_project import new_project, ProjectStore
from test_director_media_worker_ops import video
from test_director_radar_routes import radar_routes as _radar_fixture, dispatch

radar_routes = _radar_fixture


@pytest.fixture
def external_route(radar_routes, tmp_path, monkeypatch):
    import folder_paths
    store, handlers = radar_routes
    output = tmp_path / 'output'
    output.mkdir()
    monkeypatch.setattr(folder_paths, 'get_output_directory', lambda: str(output))
    asset_id = str(uuid.uuid4())
    source = store.input_root / 't8_director' / asset_id / 'source.mp4'
    source.parent.mkdir(parents=True)
    video(source)
    asset = store.register_asset(source, asset_id, 'source.mp4')
    project = new_project()
    project['assets'] = [asset]
    project['doc']['shots'][0].update(tray=[asset_id], selected=asset_id)
    project = store.save(project, 0)
    body = {'project': project, 'shot_id': project['current'], 'asset_id': asset_id,
            'take_id': str(uuid.uuid4()), 'label': '外部原片🙂', 'provenance_category': 'external'}
    return store, output, source, body, handlers


def register(body, handlers):
    return dispatch(handlers[director_routes.PREFIX + '/external-takes'], body)


def test_actual_route_exact_retry_reopen_and_saved_project_not_mutated(external_route):
    store, output, source, body, handlers = external_route
    original = source.read_bytes()
    code, result = register(body, handlers)
    assert code == 200 and result['already_registered'] is False
    record = result['record']
    assert record['prompt_id'] is record['seed'] is record['request_id'] is None
    assert record['origin'] == 'external' and not record['can_resample']
    assert store.load(body['project']['id']) == body['project']
    assert register(body, handlers) == (200, {**result, 'already_registered': True})
    newer = deepcopy(body['project'])
    newer['doc']['shots'][0]['simplePrompt'] = '继续编辑'
    newer = store.save(newer, newer['revision'])
    assert register(body, handlers)[0] == 200  # Original frozen UUID, not a new mutation.
    fresh = ProjectStore(store.root.parent, store.input_root)
    results = director_routes.director_project_results(fresh, newer['id'], output_root=output)
    assert results['results'] == [record]
    assert source.read_bytes() == original and store.load(newer['id']) == newer


@pytest.mark.parametrize('damage', ['draft', 'revision', 'unknown-shot', 'label'])
def test_actual_route_failed_request_proves_no_take_and_never_queues(external_route, damage):
    store, output, source, body, handlers = external_route
    changed = deepcopy(body)
    if damage == 'draft':
        changed['project']['doc']['shots'][0]['simplePrompt'] = '未保存编辑'
    elif damage == 'revision':
        changed['project']['revision'] += 1
    elif damage == 'unknown-shot':
        changed['shot_id'] = str(uuid.uuid4())
    else:
        changed['label'] = '\n'
    code, result = register(changed, handlers)
    assert code == (409 if damage in ('draft', 'revision') else 400)
    assert result['registration_state'] == 'not_registered'
    assert not list(output.rglob('*.mp4'))
    assert store.load(body['project']['id']) == body['project'] and source.exists()


def test_actual_route_orphan_and_damage_keep_uncertain_identity(external_route):
    store, output, _source, body, handlers = external_route
    code, result = register(body, handlers)
    assert code == 200
    path = next(output.rglob('*.mp4'))
    path.write_bytes(b'preserve damaged output')
    code, result = register(body, handlers)
    assert code == 400 and result['registration_state'] == 'unknown_or_registered'
    records = director_routes.director_project_results(store, body['project']['id'], output_root=output)['results']
    assert len(records) == 1 and records[0]['state'] == 'error' and not records[0]['outputs']
    changed = deepcopy(body)
    changed['take_id'] = str(uuid.uuid4())
    orphan = path.with_name(changed['take_id'] + '.mp4')
    orphan.write_bytes(b'preserve orphan')
    code, result = register(changed, handlers)
    assert code == 400 and result['registration_state'] == 'unknown_or_registered'
    assert orphan.read_bytes() == b'preserve orphan'


def test_actual_route_rejects_generation_parameters_bad_identity_and_oversize(external_route):
    _store, output, _source, body, handlers = external_route
    handler = handlers[director_routes.PREFIX + '/external-takes']
    for changed in ({**body, 'seed': 123}, {**body, 'take_id': '../outside'}):
        assert dispatch(handler, changed)[0] == 400
    assert dispatch(handler, body, length=2 * 1024**2 + 1)[0] == 400
    assert not list(output.rglob('*.mp4'))
