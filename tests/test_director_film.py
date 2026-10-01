from copy import deepcopy
import asyncio
import json
import sys
import types
import uuid

import av
import numpy as np
import pytest

from h3_audio_t8_pkg.director_film import prepare_film, load_film, film_media, result_key
from h3_audio_t8_pkg.director_project import ProjectStore, new_project, file_sha


def video(path, value=70, *, audio=True, width=64, height=32, tone=0, frames=12):
    with av.open(str(path), 'w') as container:
        stream = container.add_stream('libx264', rate=24)
        stream.width, stream.height, stream.pix_fmt = width, height, 'yuv420p'
        stream.options = {'threads': '1', 'bf': '0'}
        sound = container.add_stream('aac', rate=48000) if audio else None
        if sound:
            sound.layout = 'mono'
        for index in range(frames):
            frame = av.VideoFrame.from_ndarray(np.full((height, width, 3), value+index, dtype=np.uint8), format='rgb24')
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        if sound:
            samples = (0.2*np.sin(np.arange(frames*2000)*2*np.pi*tone/48000)).astype(np.float32)[None, :]
            frame = av.AudioFrame.from_ndarray(samples, format='fltp', layout='mono')
            frame.sample_rate = 48000
            for packet in sound.encode(frame):
                container.mux(packet)
            for packet in sound.encode():
                container.mux(packet)


def setup(tmp_path):
    store = ProjectStore(tmp_path/'user', tmp_path/'input')
    output = tmp_path/'output'
    output.mkdir()
    project = new_project()
    first = project['doc']['shots'][0]
    first.update(name='first', adoptedResultId='version-first')
    second = deepcopy(first)
    second.update(id=str(uuid.uuid4()), name='second', adoptedResultId='version-second')
    project['doc']['shots'].append(second)
    records = []
    for index, shot in enumerate(project['doc']['shots']):
        filename = f'{index}.mp4'
        video(output/filename, 70+index*80, width=32 if index else 64, height=64 if index else 32)
        records.append({'shot_id': shot['id'], 'prompt_id': shot['adoptedResultId'], 'state': 'success',
                        'outputs': {'save': {'images': [{'filename': filename, 'type': 'output'}]}}})
    return store, output, project, records


def test_film_freezes_adopted_order_hashes_and_trim_not_latest(tmp_path):
    store, output, project, records = setup(tmp_path)
    records.append({**deepcopy(records[0]), 'prompt_id': 'newest', 'outputs': {}})
    project['doc']['shots'][0]['filmTrim'] = {'in_frame': 2, 'out_frame': 10}
    manifest = prepare_film(store, project, records, output)
    assert manifest['ready'] and len(manifest['entries']) == 2
    assert [row['version_id'] for row in manifest['entries']] == ['version-first', 'version-second']
    assert manifest['duration'] == pytest.approx(20/24)
    assert manifest['entries'][0]['media']['frames'] == 12
    assert manifest['entries'][0]['media']['has_audio']
    frozen = film_media(store, project['id'], manifest['id'], 0)
    old_sha = file_sha(frozen)
    video(output/'0.mp4', 5)  # Simulate later Core output replacement, never change cache.
    project['doc']['shots'].reverse()
    assert file_sha(film_media(store, project['id'], manifest['id'], 0)) == old_sha
    assert load_film(store, project['id'], manifest['id']) == manifest


@pytest.mark.parametrize('kind', ['unadopted', 'missing-version', 'failed', 'missing-file', 'traversal', 'temporary', 'multiple'])
def test_film_never_substitutes_for_missing_or_unsafe_adoption(tmp_path, kind):
    store, output, project, records = setup(tmp_path)
    shot, item = project['doc']['shots'][0], records[0]['outputs']['save']['images'][0]
    if kind == 'unadopted':
        shot.pop('adoptedResultId')
    elif kind == 'missing-version':
        shot['adoptedResultId'] = 'unknown'
    elif kind == 'failed':
        records[0]['state'] = 'error'
    elif kind == 'missing-file':
        item['filename'] = 'gone.mp4'
    elif kind == 'traversal':
        item['subfolder'] = '../'
    elif kind == 'temporary':
        item['type'] = 'temp'
    else:
        records[0]['outputs']['save']['images'].append(dict(item))
    report = prepare_film(store, project, records, output)
    assert not report['ready'] and report['errors'][0]['shot_id'] == shot['id']
    assert not list((store.root/'films').glob('*.json'))


@pytest.mark.parametrize('trim', [{'in_frame': -1}, {'out_frame': 100}, {'in_frame': True}, {'in_frame': 3, 'out_frame': 3}, {'in_frame': 1.5}])
def test_film_invalid_trims_do_not_silently_round_or_extend(tmp_path, trim):
    store, output, project, records = setup(tmp_path)
    project['doc']['shots'][0]['filmTrim'] = trim
    assert not prepare_film(store, project, records, output)['ready']


def test_film_corrupt_manifest_media_and_foreign_project_rejected(tmp_path):
    store, output, project, records = setup(tmp_path)
    manifest = prepare_film(store, project, records, output)
    with pytest.raises(ValueError):
        load_film(store, str(uuid.uuid4()), manifest['id'])
    with pytest.raises(ValueError):
        film_media(store, project['id'], manifest['id'], -1)
    frozen = film_media(store, project['id'], manifest['id'], 0)
    frozen.write_bytes(b'changed fixture')
    with pytest.raises(ValueError, match='变化'):
        film_media(store, project['id'], manifest['id'], 0)
    assert not prepare_film(store, project, records, output)['ready']
    path = store.root/'films'/f"{manifest['id']}.json"
    manifest['entries'].reverse()
    path.write_text(json.dumps(manifest), encoding='utf-8')
    with pytest.raises(ValueError, match='校验失败'):
        load_film(store, project['id'], manifest['id'])


def test_legacy_result_key_matches_frontend_unicode_and_separator_convention():
    record = {'shot_id': 's', 'outputs': {'save': {'images': [{'filename': '片.mp4', 'subfolder': 'a\\b'}]}}}
    assert result_key(record) == 'legacy:["s","output","a/b","片.mp4"]'


def test_film_actual_http_routes_and_range_delivery(tmp_path, monkeypatch):
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer
    import folder_paths
    from h3_audio_t8_pkg import director_routes
    from h3_audio_t8_pkg.director_film_export import POLICY, reserve_export

    store, output, project, records = setup(tmp_path)
    routes = web.RouteTableDef()
    server = types.SimpleNamespace(routes=routes)
    monkeypatch.setitem(sys.modules, 'server', types.SimpleNamespace(PromptServer=types.SimpleNamespace(instance=server)))
    monkeypatch.setattr(director_routes, '_REGISTERED', False)
    monkeypatch.setattr(director_routes, '_GENERATE_LOCK', asyncio.Lock())
    monkeypatch.setattr(director_routes, '_FILM_EXPORT_LOCK', asyncio.Lock())
    monkeypatch.setattr(director_routes, '_FILM_EXPORT_TASKS', {})
    monkeypatch.setattr(director_routes, 'get_store', lambda: store)
    monkeypatch.setattr(director_routes, 'director_project_results', lambda *_args, **_kwargs: {'results': records})
    monkeypatch.setattr(folder_paths, 'get_output_directory', lambda: str(output))
    director_routes.register_director_routes()

    async def scenario():
        app = web.Application()
        app.add_routes(routes)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            response = await client.post(director_routes.PREFIX+'/films/prepare', json={'project': project})
            assert response.status == 200, await response.text()
            manifest = await response.json()
            assert manifest['ready']
            base = director_routes.PREFIX+f"/films/{project['id']}/{manifest['id']}"
            response = await client.get(base)
            assert await response.json() == manifest
            response = await client.get(base+'/media/0', headers={'Range': 'bytes=0-63'})
            assert response.status == 206
            assert await response.read() == film_media(store, project['id'], manifest['id'], 0).read_bytes()[:64]
            response = await client.get(director_routes.PREFIX+f"/films/{uuid.uuid4()}/{manifest['id']}/media/0")
            assert response.status == 400
            response = await client.post(base+'/frames', json={'index': 0, 'frame': 11})
            assert response.status == 200, await response.text()
            extracted = await response.json()
            assert extracted['asset']['kind'] == 'image' and extracted['frame'] == 11
            response = await client.get(director_routes.PREFIX+'/assets/'+extracted['asset']['id'])
            assert response.status == 200 and (await response.read()).startswith(b'\x89PNG')
            response = await client.post(base+'/frames', json={'index': 0, 'frame': True})
            assert response.status == 400
            response = await client.post(director_routes.PREFIX+f"/films/{uuid.uuid4()}/{manifest['id']}/frames", json={'index': 0, 'frame': 0})
            assert response.status == 400
            job_id = str(uuid.uuid4())
            options = {'policy': POLICY, 'width': 64, 'height': 64, 'confirm_encoding': True}
            job_url = base+'/exports/'+job_id
            response = await client.post(job_url, json=options)
            assert response.status == 200, await response.text()
            second = await client.post(job_url, json=options)
            assert second.status == 200  # Same request never creates a second worker.
            async with asyncio.timeout(20):
                while True:
                    response = await client.get(job_url)
                    progress = await response.json()
                    if progress['state'] not in {'queued', 'running'}:
                        break
                    await asyncio.sleep(0.05)
            assert progress['state'] == 'success', progress
            response = await client.get(job_url+'/file')
            assert response.status == 200 and 'attachment' in response.headers['Content-Disposition']
            assert len(await response.read()) > 100
            stale_id = str(uuid.uuid4())
            reserve_export(store, project['id'], manifest['id'], stale_id, options)
            response = await client.get(base+'/exports/'+stale_id)
            assert (await response.json())['state'] == 'interrupted'
            assert not (store.root/'film_exports'/f'{stale_id}.mp4').exists()
        finally:
            await client.close()

    asyncio.run(scenario())
