"""Real loopback task/cancel/download workflow; injected ASR, real FFmpeg."""
import asyncio
import time
import uuid

from aiohttp.test_utils import TestClient, TestServer

from test_director_caption_routes import routes_for
from test_h27_caption_asr import recognized
from test_director_captions import frozen
from h3_audio_t8_pkg import director_routes as routes
from h3_audio_t8_pkg import speech_verification as speech
from h3_audio_t8_pkg.director_project import file_sha


def test_http_asr_binding_cancel_old_identity_and_caption_download(tmp_path, monkeypatch):
    store, output, project, _, film = frozen(tmp_path, trim=True)
    model = tmp_path/'custom-layout-model'
    model.mkdir()
    (model/'config.json').write_text('{}')
    (model/'model.bin').write_bytes(b'not loaded')
    monkeypatch.setattr(speech, 'resolve_asr_model_directory', lambda _: model)
    calls = []
    recognized(monkeypatch, calls)
    monkeypatch.setattr(routes, '_FILM_ASR_TASKS', {})
    monkeypatch.setattr(routes, '_FILM_ASR_LOCK', asyncio.Lock())
    app = routes_for(monkeypatch, store)
    body = {'film_sha256': film['sha256'], 'shot_id': film['entries'][0]['shot_id'],
            'model_directory': str(model), 'language': 'English'}
    base = routes.PREFIX+f"/films/{project['id']}/{film['id']}/captions"
    before = {p:file_sha(p) for p in output.glob('*.mp4')}

    async def scenario():
        async with TestClient(TestServer(app)) as client:
            path = base+'/asr/'+str(uuid.uuid4())
            response = await client.post(path, json=body)
            assert response.status == 202, await response.text()
            for _ in range(200):
                response = await client.get(path)
                job = await response.json()
                if job['state'] not in {'queued', 'running'}:
                    break
                await asyncio.sleep(.025)
            assert job['state'] == 'success', job
            assert len(calls) == 1 and job['track']['source']['kind'] == 'recognized_audio'
            response = await client.post(path, json=body)
            assert response.status == 200 and len(calls) == 1
            response = await client.get(path.replace(project['id'], str(uuid.uuid4())))
            assert response.status == 400
            caption_path = base+'/'+str(uuid.uuid4())
            response = await client.post(caption_path, json={'schema':'t8.director.caption_input.v1',
                'film_sha256':film['sha256'], 'tracks':[job['track']]})
            assert response.status == 201, await response.text()
            response = await client.get(caption_path+'/file/srt')
            assert response.status == 200 and 'hello world.' in await response.text()
            # Waiting recognition cooperates with explicit API cancellation.
            def wait_for_cancel(*_args, **kwargs):
                while True:
                    kwargs['cancel_check']()
                    time.sleep(.01)
            monkeypatch.setattr(speech, 'transcribe_audio', wait_for_cancel)
            pending = base+'/asr/'+str(uuid.uuid4())
            assert (await client.post(pending, json=body)).status == 202
            await asyncio.sleep(.1)
            assert (await client.post(pending+'/cancel', json={})).status == 200
            for _ in range(200):
                cancelled = await (await client.get(pending)).json()
                if cancelled['state'] not in {'queued', 'running'}:
                    break
                await asyncio.sleep(.025)
            assert cancelled['state'] == 'cancelled', cancelled
            assert not cancelled.get('decoder') or (cancelled['decoder']['reaped']
                                                    and cancelled['decoder']['return_code'] is not None)
            assert cancelled.get('track') is None
            assert not routes._FILM_ASR_TASKS
    asyncio.run(scenario())
    assert all(file_sha(p)==digest for p,digest in before.items())
