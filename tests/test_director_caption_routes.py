"""Actual loopback HTTP and CPU-encoded frozen media; no Core queue or CUDA."""
from copy import deepcopy
import asyncio
import sys
import types
import uuid

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
import pytest

from test_director_film import setup
from h3_audio_t8_pkg import director_routes
from h3_audio_t8_pkg.director_captions import INPUT_SCHEMA
from h3_audio_t8_pkg.director_film import film_media, prepare_film
from h3_audio_t8_pkg.director_film_export import POLICY, export_file, reserve_export, run_export
from h3_audio_t8_pkg.director_project import file_sha


def routes_for(monkeypatch, store):
    routes = web.RouteTableDef()
    server = types.SimpleNamespace(routes=routes)
    monkeypatch.setitem(sys.modules, 'server', types.SimpleNamespace(PromptServer=types.SimpleNamespace(instance=server)))
    monkeypatch.setattr(director_routes, '_REGISTERED', False)
    monkeypatch.setattr(director_routes, 'get_store', lambda: store)
    director_routes.register_director_routes()
    app = web.Application()
    app.add_routes(routes)
    return app


def payload_for(film):
    entry = film['entries'][0]
    return {'schema': INPUT_SCHEMA, 'film_sha256': film['sha256'], 'tracks': [{
        'shot_id': entry['shot_id'], 'version_id': entry['version_id'], 'media_sha256': entry['media_sha256'],
        'clock': {'unit': 'frames'}, 'source': {'kind': 'manual_planned_text'},
        'cues': [{'start': 0, 'end': 12, 'text': '你好，世界\n第二行 <script>不是代码</script>'}],
    }]}


def test_caption_http_publish_retry_download_and_independent_successful_mp4(tmp_path, monkeypatch):
    store, output, project, records = setup(tmp_path)
    film = prepare_film(store, project, records, output)
    film_path = store.root/'films'/f"{film['id']}.json"
    originals = [film_path, *(film_media(store, project['id'], film['id'], i) for i in range(2))]
    before = {path: file_sha(path) for path in originals}
    job_id = str(uuid.uuid4())
    options = {'policy': POLICY, 'width': 64, 'height': 64, 'confirm_encoding': True}
    reserve_export(store, project['id'], film['id'], job_id, options)
    run_export(store, project['id'], film['id'], job_id)
    mp4 = export_file(store, project['id'], film['id'], job_id)
    encoded_sha = file_sha(mp4)
    app = routes_for(monkeypatch, store)
    caption_id = str(uuid.uuid4())
    base = director_routes.PREFIX+f"/films/{project['id']}/{film['id']}"
    url = base+'/captions/'+caption_id
    payload = payload_for(film)

    async def scenario():
        async with TestClient(TestServer(app)) as client:
            response = await client.get(director_routes.PREFIX+'/ui-modules/caption_ui.mjs')
            assert response.status == 200 and 'createFilmCaptionEditor' in await response.text()
            response = await client.get(url)
            assert response.status == 404
            assert not (store.root/'film_captions'/caption_id).exists()
            response = await client.post(url, json=payload)
            assert response.status == 201, await response.text()
            result = await response.json()
            assert result['media_modified'] is False
            assert result['text_status'] == 'planned_text_not_asr_or_spoken_alignment_verified'
            response = await client.post(url, json=payload)
            assert response.status == 200 and await response.json() == result
            response = await client.get(url)
            assert response.status == 200 and await response.json() == result
            for format in ('srt', 'vtt'):
                response = await client.get(url+'/file/'+format)
                assert response.status == 200
                assert 'attachment' in response.headers['Content-Disposition']
                assert response.charset == 'utf-8'
                text = (await response.read()).decode('utf-8')
                assert '你好，世界\n第二行 &lt;script&gt;不是代码&lt;/script&gt;' in text
                assert text.startswith('WEBVTT\n\n' if format == 'vtt' else '1\n')
                response = await client.get(url+'/file/'+format, headers={'Range': 'bytes=0-3'})
                assert response.status == 206 and len(await response.read()) == 4
            changed = deepcopy(payload)
            changed['tracks'][0]['cues'][0]['text'] = 'different planned text'
            response = await client.post(url, json=changed)
            assert response.status == 400
            response = await client.get(url+'/file/html')
            assert response.status == 400
            response = await client.get(director_routes.PREFIX+f"/films/{uuid.uuid4()}/{film['id']}/captions/{caption_id}/file/srt")
            assert response.status == 400
            (store.root/'film_captions'/caption_id/'captions.srt').write_text('corrupt fixture', encoding='utf-8')
            response = await client.get(url+'/file/vtt')
            assert response.status == 400  # All files verified, not a half-published pair.
            response = await client.get(base+'/exports/'+job_id)
            assert response.status == 200 and (await response.json())['state'] == 'success'
            assert file_sha(mp4) == encoded_sha

    asyncio.run(scenario())
    assert {path: file_sha(path) for path in originals} == before


@pytest.mark.parametrize('defect', ['unconfirmed', 'wrong-shot', 'bad-coordinate', 'wrong-report'])
def test_speech_caption_import_rejects_unbound_or_malformed_without_publishing(tmp_path, monkeypatch, defect):
    store, output, project, records = setup(tmp_path)
    film = prepare_film(store, project, records, output)
    body = {'shot_id': film['entries'][0]['shot_id'], 'confirm_media_match': True,
            'report': {'schema': 'minimax_h3_t8_speech_timeline_v1', 'sample_rate': 48000,
                       'timeline': [{'speaker_id': 'A', 'text': '计划对白', 'start_sample': 0, 'end_sample': 24000}]}}
    if defect == 'unconfirmed':
        body['confirm_media_match'] = False
    elif defect == 'wrong-shot':
        body['shot_id'] = str(uuid.uuid4())
    elif defect == 'bad-coordinate':
        body['report']['timeline'][0]['end_sample'] = True
    else:
        body['report']['schema'] = 'not_a_speech_report'
    app = routes_for(monkeypatch, store)
    url = director_routes.PREFIX+f"/films/{project['id']}/{film['id']}/captions/speech-track"

    async def scenario():
        async with TestClient(TestServer(app)) as client:
            response = await client.post(url, json=body)
            assert response.status == 400, await response.text()
            body['shot_id'] = film['entries'][0]['shot_id']
            body['confirm_media_match'] = True
            body['report']['schema'] = 'minimax_h3_t8_speech_timeline_v1'
            body['report']['timeline'][0]['end_sample'] = 24000
            response = await client.post(url, json=body)
            assert response.status == 200, await response.text()
            track = await response.json()
            assert track['clock'] == {'unit': 'samples', 'rate': 48000}
            assert track['source']['binding'] == 'user_declared_media_match_not_independently_verified'
            assert track['cues'][0]['text'] == 'A: 计划对白'

    asyncio.run(scenario())
    assert not (store.root/'film_captions').exists()
