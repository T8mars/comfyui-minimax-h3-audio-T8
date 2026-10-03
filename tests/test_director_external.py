"""Actual immutable external take integration and negative guards."""
from copy import deepcopy
import json
import uuid

import pytest

from h3_audio_t8_pkg import director_external as external
from h3_audio_t8_pkg import director_media
from h3_audio_t8_pkg.director_project import (
    ProjectStore, ProjectConflict, new_project, sha, file_sha,
)
from test_director_media_worker_ops import video


@pytest.fixture
def prepared(tmp_path):
    store = ProjectStore(tmp_path / 'user', tmp_path / 'input')
    asset_id = str(uuid.uuid4())
    source = store.input_root / 't8_director' / asset_id / 'source.mp4'
    source.parent.mkdir(parents=True)
    video(source)
    asset = store.register_asset(source, asset_id, 'source.mp4')
    project = new_project()
    project['assets'] = [asset]
    project['doc']['shots'][0]['tray'] = [asset_id]
    project['doc']['shots'][0]['selected'] = asset_id
    project = store.save(project, 0)
    request = {'project_id': project['id'], 'shot_id': project['current'],
               'asset_id': asset_id, 'take_id': str(uuid.uuid4()),
               'expected_revision': project['revision'], 'project_sha256': sha(project),
               'label': '外片原版', 'provenance_category': 'external'}
    output = tmp_path / 'output'
    output.mkdir()
    return store, project, request, output, source


def test_actual_external_register_is_immutable_idempotent_not_generation(prepared):
    store, project, request, output, source = prepared
    before = source.read_bytes()
    result = external.register_external_take(store, request, output)
    record = result['record']
    assert not result['already_registered']
    assert record['external_take_id'] == request['take_id']
    assert record['prompt_id'] is record['seed'] is record['request_id'] is None
    assert record['origin'] == 'external' and record['can_resample'] is False
    assert record['snapshot_available'] is False
    assert record['decoded_clock']['fully_decoded'] is True
    assert record['decoded_clock']['normalization_performed'] is False
    assert record['media_sha256'] == file_sha(source)
    assert store.load(project['id']) == project and source.read_bytes() == before
    assert 'adoptedResultId' not in project['doc']['shots'][0]
    assert external.register_external_take(store, request, output) == {'record': record, 'already_registered': True}
    assert external.external_results(store, project['id'], output) == [record]
    with pytest.raises(ValueError, match='拒绝覆盖'):
        external.register_external_take(store, {**request, 'label': '另一片'}, output)


@pytest.mark.parametrize('change', ['revision', 'project-content', 'other-shot', 'other-asset', 'extra-model', 'path', 'blank-label'])
def test_external_register_rejects_unowned_or_stale_request(prepared, change):
    store, project, request, output, _source = prepared
    changed = deepcopy(request)
    if change == 'revision':
        changed['expected_revision'] += 1
    elif change == 'project-content':
        changed['project_sha256'] = '0' * 64
    elif change == 'other-shot':
        changed['shot_id'] = str(uuid.uuid4())
    elif change == 'other-asset':
        changed['asset_id'] = str(uuid.uuid4())
    elif change == 'extra-model':
        changed['model_id'] = 'fake sampler ancestor'
    elif change == 'path':
        changed['take_id'] = '../outside'
    else:
        changed['label'] = '   '
    with pytest.raises(ValueError):
        external.register_external_take(store, changed, output)
    assert not list(output.rglob('*.mp4'))
    assert store.load(project['id']) == project


def test_project_changed_during_real_decode_never_commits_take(prepared, monkeypatch):
    store, project, request, output, source = prepared
    run = director_media.run_media
    original_sha = file_sha(source)

    def race(operation, path, **kwargs):
        result = run(operation, path, **kwargs)
        updated = deepcopy(project)
        updated['doc']['shots'][0]['simplePrompt'] = '用户新编辑'
        store.save(updated, project['revision'])
        return result

    monkeypatch.setattr(director_media, 'run_media', race)
    with pytest.raises(ProjectConflict):
        external.register_external_take(store, request, output)
    assert store.load(project['id'])['doc']['shots'][0]['simplePrompt'] == '用户新编辑'
    assert file_sha(source) == original_sha
    assert not list(output.rglob('*.mp4')) and not list(output.rglob('*.tmp'))
    assert not external.external_results(store, project['id'], output)


def test_frozen_copy_missing_or_changed_never_falls_back_to_source(prepared):
    store, project, request, output, source = prepared
    result = external.register_external_take(store, request, output)
    original = source.read_bytes()
    take = next(output.rglob('*.mp4'))
    take.write_bytes(b'damaged copy')
    records = external.external_results(store, project['id'], output)
    assert records[0]['external_take_id'] == result['record']['external_take_id']
    assert records[0]['shot_id'] == request['shot_id']
    assert records[0]['state'] == 'error' and records[0]['outputs'] == {}
    assert records[0]['integrity_error']
    with pytest.raises(ValueError, match='字节改变'):
        external.register_external_take(store, request, output)
    assert source.read_bytes() == original and take.read_bytes() == b'damaged copy'


def test_re_signed_receipt_cannot_claim_native_generation(prepared):
    store, project, request, output, _source = prepared
    external.register_external_take(store, request, output)
    receipt_path = external._receipt_path(store, project['id'], request['take_id'])
    value = json.loads(receipt_path.read_text(encoding='utf8'))
    value['can_resample'] = True
    value['sha256'] = sha({key: item for key, item in value.items() if key != 'sha256'})
    receipt_path.write_text(json.dumps(value), encoding='utf8')
    result = external.external_results(store, project['id'], output)[0]
    assert result['state'] == 'error' and not result['outputs']
    assert result['can_resample'] is False and not result['snapshot_available']


def test_orphan_target_is_not_overwritten(prepared):
    store, project, request, output, _source = prepared
    target = output / 'T8_Director_External' / project['id'] / (request['take_id'] + '.mp4')
    target.parent.mkdir(parents=True)
    target.write_bytes(b'preserve orphan evidence')
    with pytest.raises(ValueError, match='拒绝覆盖'):
        external.register_external_take(store, request, output)
    assert target.read_bytes() == b'preserve orphan evidence'


def test_malformed_receipt_name_does_not_hide_other_valid_versions(prepared):
    store, project, request, output, _source = prepared
    record = external.register_external_take(store, request, output)['record']
    folder = external._receipt_path(store, project['id'], request['take_id']).parent
    (folder / 'malformed.json').write_text('{}', encoding='utf8')
    records = external.external_results(store, project['id'], output)
    assert record in records
    bad = next(row for row in records if row['state'] == 'error')
    assert bad['shot_id'] is bad['external_take_id'] is None and bad['outputs'] == {}


def test_external_adoption_trim_frame_and_bundle_keep_original_provenance(prepared, tmp_path):
    from h3_audio_t8_pkg.director_film import prepare_film, result_key
    from h3_audio_t8_pkg.director_frame import extract_frame
    from h3_audio_t8_pkg.director_routes import director_project_results
    from h3_audio_t8_pkg.director_bundle import prepare_bundle, build_bundle, inspect_bundle, import_bundle
    from test_director_bundle import upload
    store, project, request, output, source = prepared
    original = source.read_bytes()
    record = external.register_external_take(store, request, output)['record']
    index = director_project_results(store, project['id'], output_root=output)['results']
    assert index == [record]
    shot = project['doc']['shots'][0]
    shot['adoptedResultId'] = result_key(record)
    shot['filmTrim'] = {'in_frame': 1, 'out_frame': record['media']['frames']}
    film = prepare_film(store, project, index, output)
    assert film['ready'] and film['entries'][0]['in_frame'] == 1
    assert film['entries'][0]['origin'] == 'external' and not film['entries'][0]['can_resample']
    frame = extract_frame(store, project['id'], film['id'], 0, record['media']['frames'] - 1)
    assert frame['asset']['kind'] == 'image' and frame['media_sha256'] == file_sha(source)
    plan = prepare_bundle(store, project, index, output)
    assert plan['ready'], plan
    row = next(row for row in plan['files'] if row['role'] == 'result')
    assert row['external']['origin'] == 'external' and row['external']['can_resample'] is False
    archive = build_bundle(store, project['id'], plan['id'])
    fresh = ProjectStore(tmp_path / 'fresh-user', tmp_path / 'fresh-input')
    fresh_output = tmp_path / 'fresh-output'
    fresh_output.mkdir()
    upload_id, new_id = upload(fresh, archive), str(uuid.uuid4())
    inspect_bundle(fresh, upload_id)
    imported = import_bundle(fresh, fresh_output, upload_id, new_id)
    assert imported['revision'] == 1
    saved = fresh.load(new_id)
    result = director_project_results(fresh, new_id, output_root=fresh_output)['results'][0]
    assert result['origin'] == 'external' and not result['can_resample'] and not result['snapshot_available']
    assert result['prompt_id'] is result['seed'] is result['request_id'] is None
    assert saved['doc']['shots'][0]['adoptedResultId'] == result_key(result)
    assert result_key(result) != result_key(record)
    assert result['media_sha256'] == record['media_sha256']
    assert result['decoded_clock'] == record['decoded_clock']
    reopened = prepare_film(fresh, saved, [result], fresh_output)
    assert reopened['ready'] and reopened['entries'][0]['in_frame'] == 1
    assert source.read_bytes() == original


def test_external_nonzero_origin_not_silently_normalized_for_export(prepared):
    from h3_audio_t8_pkg.director_film import prepare_film, result_key
    from h3_audio_t8_pkg.director_film_export import reserve_export, POLICY
    store, project, request, output, _source = prepared
    record = external.register_external_take(store, request, output)['record']
    # Exercise the export's exact explicit guard with a sealed film. The real
    # decoded offset acquisition is tested separately, not faked as native QA.
    record['decoded_clock']['zero_origin_film_compatible'] = False
    project['doc']['shots'][0]['adoptedResultId'] = result_key(record)
    film = prepare_film(store, project, [record], output)
    assert film['ready']
    with pytest.raises(ValueError, match='非零'):
        reserve_export(store, project['id'], film['id'], str(uuid.uuid4()),
            {'policy': POLICY, 'confirm_encoding': True, 'width': 64, 'height': 32})
