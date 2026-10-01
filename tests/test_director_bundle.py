from copy import deepcopy
import json
import shutil
import uuid
import zipfile
import asyncio
import sys
import types

import pytest

from test_director_d1 import asset
from test_director_film import setup
from h3_audio_t8_pkg.director_project import ProjectStore, sha, file_sha
from h3_audio_t8_pkg.director_bundle import (
    portable_project, prepare_bundle, build_bundle, inspect_bundle, import_bundle, imported_results,
)
from h3_audio_t8_pkg.director_film import prepare_film
from h3_audio_t8_pkg.director_routes import director_project_results


def bundle_fixture(tmp_path, *, draft_flags=None):
    store, output, project, records = setup(tmp_path)
    media = asset(store)
    project['assets'] = [media]
    project['doc']['sharedRefs'] = [media['id']]
    first = project['doc']['shots'][0]
    first.update(mode='refs', refs=[media['id']], tray=[media['id']], selected=media['id'],
                 simplePrompt='@image1 微笑', seed=0, filmTrim={'in_frame': 2, 'out_frame': 10})
    first['events'] = [{'id': str(uuid.uuid4()), 'start': 0, 'end': 1, 'text': '保留未激活草稿 @image1'}]
    if draft_flags is not None:
        first['prompt'] = '独立高级原稿 @image1'
        second = project['doc']['shots'][1]
        second.update(writingMode='advanced', simplePrompt='', prompt='', events=[])
        for shot in project['doc']['shots']:
            if draft_flags:
                shot.update(simpleInitialized=True, advancedInitialized=True)
            else:
                shot.pop('simpleInitialized', None)
                shot.pop('advancedInitialized', None)
    store.save(project, 0)
    plan = prepare_bundle(store, project, records, output)
    assert plan['ready'], plan
    path = build_bundle(store, project['id'], plan['id'])
    return store, output, project, records, plan, path


def upload(store, archive):
    upload_id = str(uuid.uuid4())
    target = store.root/'bundle_imports'/f'{upload_id}.zip'
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(archive, target)
    return upload_id


def test_bundle_migrates_real_media_into_fresh_store_without_overwriting_original(tmp_path):
    store, output, project, records, plan, archive = bundle_fixture(tmp_path)
    original = deepcopy(store.load(project['id']))
    before = file_sha(archive)
    assert build_bundle(store, project['id'], plan['id']) == archive
    assert file_sha(archive) == before
    target = ProjectStore(tmp_path/'fresh-user', tmp_path/'fresh-input')
    fresh_output = tmp_path/'fresh-output'
    fresh_output.mkdir()
    upload_id = upload(target, archive)
    preview = inspect_bundle(target, upload_id)
    assert preview['shots'] == 2 and len(preview['files']) == 3
    assert not target.list()
    new_id = str(uuid.uuid4())
    result = import_bundle(target, fresh_output, upload_id, new_id)
    assert result['revision'] == 1 and not result['already_created']
    saved = target.load(new_id)
    assert saved['id'] != project['id']
    assert saved['assets'][0]['id'] != project['assets'][0]['id']
    first = saved['doc']['shots'][0]
    assert first['id'] != project['current'] and first['simplePrompt'] == '@image1 微笑' and first['seed'] == 0
    assert first['refs'] == saved['doc']['sharedRefs'] == [saved['assets'][0]['id']]
    assert first['events'][0]['id'] != project['doc']['shots'][0]['events'][0]['id']
    assert first['events'][0]['text'] == project['doc']['shots'][0]['events'][0]['text']
    recovered = director_project_results(target, new_id, output_root=fresh_output)['results']
    assert len(recovered) == 2 and all(row['state'] == 'success' and not row['snapshot_available'] for row in recovered)
    film = prepare_film(target, saved, recovered, fresh_output)
    assert film['ready'] and film['entries'][0]['in_frame'] == 2
    assert film['entries'][0]['out_frame'] == 10
    assert target.asset(saved['assets'][0]['id'], verify=True)['sha256'] == project['assets'][0]['sha256']
    saved['doc']['shots'][0]['simplePrompt'] = 'new work'
    saved.pop('bundleSource')  # A normal UI envelope need not carry backend provenance.
    target.save(saved, 1)
    assert import_bundle(target, fresh_output, upload_id, new_id)['already_created']
    assert target.load(new_id)['doc']['shots'][0]['simplePrompt'] == 'new work'
    assert store.load(project['id']) == original
    assert file_sha(archive) == before


def test_portable_manifest_excludes_private_metadata_paths_and_unrelated_files(tmp_path):
    store, output, project, records, _plan, _archive = bundle_fixture(tmp_path)
    project.update(token='PRIVATE-SECRET', api={'key': 'PRIVATE-SECRET'}, versionSource={'path': 'F:/private'})
    project['doc']['generation']['token'] = 'PRIVATE-SECRET'
    project['doc']['shots'][0]['private_key'] = 'PRIVATE-SECRET'
    project['assets'][0]['url'] = 'http://secret.invalid/?key=PRIVATE-SECRET'
    project['doc']['sampling'] = {'mode': 'single', 'high_loras': [{'name': 'inactive.safetensors', 'enabled': False, 'strength': 0.5}]}
    (output/'unrelated.secret').write_text('PRIVATE-SECRET')
    plan = prepare_bundle(store, project, records, output)
    archive = build_bundle(store, project['id'], plan['id'])
    with zipfile.ZipFile(archive) as z:
        text = z.read('manifest.json').decode('utf-8')
        assert 'PRIVATE-SECRET' not in text and 'server_path' not in text and 'F:/private' not in text
        assert set(z.namelist()) == {'manifest.json', *[row['path'] for row in plan['files']]}
        parsed = json.loads(text)
        assert parsed['project']['doc']['sampling']['high_loras'][0]['name'] == 'inactive.safetensors'
    project['doc']['generation']['unet'] = 'F:/private/model.safetensors'
    with pytest.raises(ValueError, match='绝对路径'):
        portable_project(project)


def test_portable_writing_initialization_flags_preserve_false_and_reject_invalid_values(tmp_path):
    _store, _output, project, _records, _plan, _archive = bundle_fixture(tmp_path)
    shot = project['doc']['shots'][0]
    shot.update(simpleInitialized=True, advancedInitialized=False)
    portable = portable_project(project)['doc']['shots'][0]
    assert portable['simpleInitialized'] is True
    assert portable['advancedInitialized'] is False
    for invalid in ('false', 1, None, []):
        shot['advancedInitialized'] = invalid
        with pytest.raises(ValueError, match='advancedInitialized 必须是布尔值'):
            portable_project(project)


def test_bundle_can_explicitly_export_draft_without_results_but_never_hide_missing_input(tmp_path):
    store, output, project, records, _plan, _archive = bundle_fixture(tmp_path)
    assert not prepare_bundle(store, project, [], output)['ready']
    plan = prepare_bundle(store, project, [], output, False)
    archive = build_bundle(store, project['id'], plan['id'])
    with zipfile.ZipFile(archive) as z:
        p = json.loads(z.read('manifest.json'))['project']
        assert all('adoptedResultId' not in shot and 'filmTrim' not in shot for shot in p['doc']['shots'])
        assert len(z.namelist()) == 2
    source = store.input_root/project['assets'][0]['server_path']
    source.rename(source.with_suffix('.missing'))
    failed = prepare_bundle(store, project, records, output, False)
    assert not failed['ready'] and failed['errors']


@pytest.mark.parametrize('seed', [2**53, -1, True, 1.5])
def test_portable_seed_must_remain_exact_in_browser(tmp_path, seed):
    _store, _output, project, _records, _plan, _archive = bundle_fixture(tmp_path)
    project['doc']['shots'][0]['seed'] = seed
    with pytest.raises(ValueError, match='精确表达'):
        portable_project(project)


def rewrite_zip(source, target, mutation):
    with zipfile.ZipFile(source) as z:
        data = {name: z.read(name) for name in z.namelist()}
    mutation(data)
    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_STORED) as z:
        for name, content in data.items():
            z.writestr(name, content)


@pytest.mark.parametrize('problem', ['traversal', 'absolute', 'backslash', 'unlisted', 'missing', 'bad-sha', 'duplicate-id', 'bad-json-sha', 'private-field', 'ratio', 'duplicate-name', 'symlink', 'garbage'])
def test_bundle_rejects_corrupt_or_dangerous_archive_before_project_creation(tmp_path, problem):
    _store, _output, _project, _records, _plan, archive = bundle_fixture(tmp_path)
    target = ProjectStore(tmp_path/'target', tmp_path/'target-input')
    upload_id = upload(target, archive)
    path = target.root/'bundle_imports'/f'{upload_id}.zip'

    def mutation(data):
        if problem in {'traversal', 'absolute', 'backslash', 'unlisted'}:
            data[{'traversal': '../escape', 'absolute': '/escape', 'backslash': 'a\\b', 'unlisted': 'secret.txt'}[problem]] = b'evil'
        manifest = json.loads(data['manifest.json'])
        row = manifest['files'][0]
        if problem == 'missing':
            data.pop(row['path'])
        if problem == 'bad-sha':
            content = bytearray(data[row['path']])
            content[-1] ^= 1
            data[row['path']] = content
        if problem == 'duplicate-id':
            manifest['files'].append(deepcopy(row))
        if problem == 'private-field':
            manifest['project']['token'] = 'secret'
        manifest.pop('sha256')
        manifest['sha256'] = 'bad' if problem == 'bad-json-sha' else sha(manifest)
        data['manifest.json'] = json.dumps(manifest).encode()

    rewrite_zip(archive, path, mutation)
    if problem in {'duplicate-name', 'symlink', 'ratio'}:
        with zipfile.ZipFile(path, 'a') as z:
            if problem == 'duplicate-name':
                z.writestr('manifest.json', b'{}')
            elif problem == 'symlink':
                link = zipfile.ZipInfo('link')
                link.external_attr = (0o120777 << 16)
                z.writestr(link, '/secret')
            else:
                z.writestr('bomb', b'0'*1048576, compress_type=zipfile.ZIP_DEFLATED)
    if problem == 'garbage':
        path.write_bytes(b'not a zip')
    with pytest.raises((ValueError, KeyError)):
        inspect_bundle(target, upload_id)
    assert not target.list()
    assert not (target.root/'assets').exists()


def test_staged_tamper_and_imported_media_tamper_are_not_silently_accepted(tmp_path):
    _store, _output, project, _records, _plan, archive = bundle_fixture(tmp_path)
    target = ProjectStore(tmp_path/'target', tmp_path/'target-input')
    output = tmp_path/'target-output'
    output.mkdir()
    upload_id = upload(target, archive)
    preview = inspect_bundle(target, upload_id)
    with pytest.raises(ValueError, match='不能覆盖'):
        import_bundle(target, output, upload_id, project['id'])
    staged = target.root/'bundle_staging'/upload_id/preview['files'][0]['path']
    staged.write_bytes(b'changed')
    with pytest.raises(ValueError, match='发生变化'):
        import_bundle(target, output, upload_id, str(uuid.uuid4()))
    inspect_bundle(target, upload_id)
    new_id = str(uuid.uuid4())
    import_bundle(target, output, upload_id, new_id)
    records = imported_results(target, new_id, output)
    media = records[0]['outputs']['bundle']['images'][0]
    (output/media['subfolder']/media['filename']).write_bytes(b'changed')
    changed = imported_results(target, new_id, output)
    assert changed[0]['state'] == 'error' and not changed[0]['outputs']
    assert changed[1]['state'] == 'success'


def test_bundle_registered_http_download_multipart_preview_apply_and_result_recovery(tmp_path, monkeypatch):
    from aiohttp import web, FormData
    from aiohttp.test_utils import TestClient, TestServer
    import folder_paths
    from h3_audio_t8_pkg import director_routes

    store, output, project, records, _plan, _archive = bundle_fixture(tmp_path)
    routes = web.RouteTableDef()
    monkeypatch.setitem(sys.modules, 'server', types.SimpleNamespace(PromptServer=types.SimpleNamespace(instance=types.SimpleNamespace(routes=routes))))
    monkeypatch.setattr(director_routes, '_REGISTERED', False)
    monkeypatch.setattr(director_routes, '_GENERATE_LOCK', asyncio.Lock())
    monkeypatch.setattr(director_routes, 'get_store', lambda: store)
    original_results = director_routes.director_project_results
    monkeypatch.setattr(director_routes, 'director_project_results', lambda s, pid, **kwargs: {'results': records} if pid == project['id'] else original_results(s, pid, **kwargs))
    monkeypatch.setattr(folder_paths, 'get_output_directory', lambda: str(output))
    director_routes.register_director_routes()

    async def scenario():
        app = web.Application()
        app.add_routes(routes)
        client = TestClient(TestServer(app))
        await client.start_server()
        base = director_routes.PREFIX
        try:
            response = await client.post(base+'/bundles/prepare', json={'project': project, 'include_results': True})
            assert response.status == 200, await response.text()
            plan = await response.json()
            address = base+f'/bundles/{project["id"]}/{plan["id"]}'
            assert (await client.get(address+'/file')).status == 400
            assert (await client.post(address+'/build', json={})).status == 400
            response = await client.post(address+'/build', json={'confirm_contents': True})
            assert response.status == 200
            downloaded = await client.get(address+'/file')
            assert downloaded.status == 200 and 'attachment' in downloaded.headers['Content-Disposition']
            data = await downloaded.read()
            assert (await client.get(base+f'/bundles/{uuid.uuid4()}/{plan["id"]}/file')).status == 400
            form = FormData()
            form.add_field('file', data, filename='portable.zip', content_type='application/zip')
            response = await client.post(base+'/bundle-imports', data=form)
            assert response.status == 200, await response.text()
            preview = await response.json()
            assert len(store.list()) == 1  # Inspection did not import.
            assert (await client.get(base+'/bundle-imports/'+preview['id'])).status == 200
            new_id = str(uuid.uuid4())
            address = base+'/bundle-imports/'+preview['id']+'/apply'
            assert (await client.post(address, json={'new_project_id': new_id})).status == 400
            for repeated in (False, True):
                response = await client.post(address, json={'new_project_id': new_id, 'confirm_new_project': True})
                assert response.status == 200, await response.text()
                assert (await response.json())['already_created'] == repeated
            assert (await client.get(base+'/projects/'+new_id)).status == 200
            response = await client.get(base+'/results/'+new_id)
            recovered = await response.json()
            assert len(recovered['results']) == 2 and all(row['state'] == 'success' for row in recovered['results'])
        finally:
            await client.close()

    asyncio.run(scenario())
