"""Caller-level failure containment with actual abruptly exiting child workers.

Normal repository conftest imports the plugin and may initialize a CUDA context.
These tests run no model, GPU inference, live Core, or user browser. CPU fixture
media and every mutation belong to pytest's temporary directory.
"""
from contextlib import contextmanager
import json
import os
import shutil
import uuid

import pytest

from h3_audio_t8_pkg import director_media
from h3_audio_t8_pkg.director_batch import batch_status
from h3_audio_t8_pkg.director_bundle import (
    build_bundle, import_bundle, inspect_bundle, prepare_bundle,
)
from h3_audio_t8_pkg.director_film import film_media, prepare_film
from h3_audio_t8_pkg.director_film_export import (
    POLICY, export_file, reserve_export, run_export,
)
from h3_audio_t8_pkg.director_frame import extract_frame
from h3_audio_t8_pkg.director_project import ProjectStore, file_sha
from test_director_batch import _create, _receipt, _setup, _write_video
from test_director_bundle import bundle_fixture, upload
from test_director_film import setup, video


def fingerprints(paths):
    return {str(path): file_sha(path) for path in paths}


def original_videos(output):
    return sorted(output.glob('*.mp4'))


@pytest.fixture
def crashing_worker(tmp_path, monkeypatch):
    """Patch the package used by all real callers, not a second module copy."""
    @contextmanager
    def crash(expected_operation):
        parent_pid = os.getpid()
        marker = tmp_path/('worker-events-'+uuid.uuid4().hex+'.jsonl')
        worker = tmp_path/('exit-worker-'+uuid.uuid4().hex+'.py')
        worker.write_text(
            'import json, os, sys\n'
            'request = json.load(sys.stdin)\n'
            f'with open({str(marker)!r}, "a", encoding="utf-8") as stream:\n'
            '    stream.write(json.dumps({"op":request["op"],"pid":os.getpid()})+"\\n")\n'
            '    stream.flush()\n'
            'os._exit(23)\n', encoding='utf-8',
        )
        with monkeypatch.context() as patch:
            patch.setattr(director_media, 'WORKER_PATH', worker)
            yield
        events = [json.loads(line) for line in marker.read_text(encoding='utf-8').splitlines()]
        assert events and all(event['op'] == expected_operation for event in events)
        assert all(event['pid'] != parent_pid for event in events)
        assert os.getpid() == parent_pid
    return crash


def test_batch_worker_failure_never_persists_success_and_next_poll_recovers(tmp_path, crashing_worker):
    store, batch_id, project, output = _setup(tmp_path)
    batch = _create(store, batch_id, project, 123)
    receipt_path, _record = _receipt(store, batch, 0)
    original_receipt = receipt_path.read_bytes()
    source = output/'complete.mp4'
    _write_video(source)
    before = file_sha(source)
    observed = {'state': 'success', 'outputs': {'save': {'images': [
        {'type': 'output', 'filename': source.name, 'subfolder': ''},
    ]}}}
    with crashing_worker('batch'):
        status = batch_status(store, batch_id, lambda _id: observed, output)
        assert status['items'][0]['state'] == 'needs_review'
        assert status['next_index'] == 0 and not status['complete']
        assert receipt_path.read_bytes() == original_receipt
        assert file_sha(source) == before
    status = batch_status(store, batch_id, lambda _id: observed, output)
    assert status['items'][0]['state'] == 'success' and status['next_index'] == 1
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    assert receipt['terminal']['state'] == 'success'
    assert receipt['media_evidence'][0]['sha256'] == before == file_sha(source)
    assert len(list((store.root/'requests').glob('*.json'))) == 1


def test_film_worker_failure_has_no_ready_manifest_and_explicit_prepare_recovers(tmp_path, crashing_worker):
    store, output, project, records = setup(tmp_path)
    sources = original_videos(output)
    before = fingerprints(sources)
    with crashing_worker('inspect'):
        result = prepare_film(store, project, records, output)
        assert not result['ready'] and result['errors'] and not result['entries']
        assert all('23' in error['message'] for error in result['errors'])
        assert not list((store.root/'films').glob('*.json'))
        assert fingerprints(sources) == before
    result = prepare_film(store, project, records, output)
    assert result['ready'] and len(result['entries']) == 2
    assert (store.root/'films'/f"{result['id']}.json").is_file()
    assert fingerprints(sources) == before


def test_frame_worker_failure_registers_no_asset_and_same_frame_can_recover(tmp_path, crashing_worker):
    store, output, project, records = setup(tmp_path)
    film = prepare_film(store, project, records, output)
    frozen = film_media(store, project['id'], film['id'], 0)
    sources = [*original_videos(output), frozen, store.root/'films'/f"{film['id']}.json"]
    before = fingerprints(sources)
    with crashing_worker('frame'):
        with pytest.raises(ValueError, match='23'):
            extract_frame(store, project['id'], film['id'], 0, 5)
        assert not list((store.root/'assets').glob('*.json'))
        assert not list(store.input_root.rglob('*.png'))
        assert fingerprints(sources) == before
    result = extract_frame(store, project['id'], film['id'], 0, 5)
    asset = store.asset(result['asset']['id'], verify=True)
    assert asset['source_frame']['media_sha256'] == file_sha(frozen)
    assert asset['source_frame']['frame'] == 5
    assert len(list((store.root/'assets').glob('*.json'))) == 1
    assert fingerprints(sources) == before


def test_asset_worker_failure_writes_no_manifest_and_registration_can_recover(tmp_path, crashing_worker):
    store = ProjectStore(tmp_path/'user', tmp_path/'input')
    asset_id = str(uuid.uuid4())
    source = store.input_root/'t8_director'/asset_id/'source.mp4'
    source.parent.mkdir(parents=True)
    video(source)
    before = file_sha(source)
    with crashing_worker('metadata'):
        with pytest.raises(ValueError, match='23'):
            store.register_asset(source, asset_id, 'original.mp4')
        assert not (store.root/'assets'/f'{asset_id}.json').exists()
        assert file_sha(source) == before
    asset = store.register_asset(source, asset_id, 'original.mp4')
    assert asset['kind'] == 'video' and asset['has_audio']
    assert store.asset(asset_id, verify=True)['sha256'] == before == file_sha(source)


@pytest.mark.parametrize('role', ['asset', 'result'])
def test_bundle_worker_failure_has_no_sealed_import_and_explicit_recheck_recovers(tmp_path, crashing_worker, role):
    if role == 'result':
        store, output, project, _records, _plan, archive = bundle_fixture(tmp_path)
    else:
        store, output, project, records = setup(tmp_path)
        asset_id = str(uuid.uuid4())
        source = store.input_root/'t8_director'/asset_id/'source.mp4'
        source.parent.mkdir(parents=True)
        shutil.copyfile(output/'0.mp4', source)
        project['assets'] = [store.register_asset(source, asset_id, 'reference.mp4')]
        plan = prepare_bundle(store, project, records, output, include_results=False)
        assert plan['ready']
        archive = build_bundle(store, project['id'], plan['id'])
    fresh = ProjectStore(tmp_path/'fresh-user', tmp_path/'fresh-input')
    upload_id = upload(fresh, archive)
    uploaded = fresh.root/'bundle_imports'/f'{upload_id}.zip'
    sources = [*original_videos(output), archive, uploaded]
    sources.extend(store.input_root.rglob('source.*'))
    before = fingerprints(sources)
    with crashing_worker('validate' if role == 'asset' else 'inspect'):
        with pytest.raises(ValueError, match='23'):
            inspect_bundle(fresh, upload_id)
        assert not (fresh.root/'bundle_imports'/f'{upload_id}.json').exists()
        assert not fresh.list()
        assert not list((fresh.root/'assets').glob('*.json'))
        assert not list((fresh.root/'bundle_results').glob('*.json'))
        assert fingerprints(sources) == before
    assert inspect_bundle(fresh, upload_id)['ready']
    new_id = str(uuid.uuid4())
    result = import_bundle(fresh, tmp_path/'fresh-output', upload_id, new_id)
    assert result['id'] == new_id and fresh.load(new_id)['revision'] == 1
    for asset in fresh.load(new_id)['assets']:
        assert fresh.asset(asset['id'], verify=True)['sha256'] == asset['sha256']
    assert fingerprints(sources) == before


def test_export_worker_failure_keeps_error_and_requires_new_explicit_job(tmp_path, crashing_worker):
    store, output, project, records = setup(tmp_path)
    film = prepare_film(store, project, records, output)
    sources = [*original_videos(output), *[film_media(store, project['id'], film['id'], index) for index in (0, 1)]]
    before = fingerprints(sources)
    job_id = str(uuid.uuid4())
    options = {'policy': POLICY, 'confirm_encoding': True, 'width': 64, 'height': 64}
    reserve_export(store, project['id'], film['id'], job_id, options)
    with crashing_worker('inspect'):
        result = run_export(store, project['id'], film['id'], job_id)
        assert result['state'] == 'error' and '23' in result['error']
        assert not {'media', 'output_sha256', 'result_sha256'} & result.keys()
        assert not (store.root/'film_exports'/f'{job_id}.mp4').exists()
        with pytest.raises(ValueError):
            export_file(store, project['id'], film['id'], job_id)
        assert fingerprints(sources) == before
    assert run_export(store, project['id'], film['id'], job_id)['state'] == 'error'
    next_id = str(uuid.uuid4())
    reserve_export(store, project['id'], film['id'], next_id, options)
    recovered = run_export(store, project['id'], film['id'], next_id)
    assert recovered['state'] == 'success', recovered.get('error')
    assert export_file(store, project['id'], film['id'], next_id).is_file()
    assert fingerprints(sources) == before
