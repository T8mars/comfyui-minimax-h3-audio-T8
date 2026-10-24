"""Real HTTP export status must not hide worker failure as a Core restart."""
import asyncio
import errno
import json
import sys
import threading
import types
import uuid

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
import pytest

from test_director_film import setup
from h3_audio_t8_pkg import director_film_export as exports
from h3_audio_t8_pkg import director_routes as routes_module
from h3_audio_t8_pkg.director_film import prepare_film
from h3_audio_t8_pkg.director_project import atomic_json, file_sha


def _routes(tmp_path, monkeypatch):
    store, output, project, records = setup(tmp_path)
    film = prepare_film(store, project, records, output)
    table = web.RouteTableDef()
    server = types.SimpleNamespace(routes=table)
    monkeypatch.setitem(sys.modules, 'server', types.SimpleNamespace(PromptServer=types.SimpleNamespace(instance=server)))
    monkeypatch.setattr(routes_module, '_REGISTERED', False)
    monkeypatch.setattr(routes_module, '_FILM_EXPORT_LOCK', asyncio.Lock())
    monkeypatch.setattr(routes_module, '_FILM_EXPORT_TASKS', {})
    monkeypatch.setattr(routes_module, 'get_store', lambda: store)
    routes_module.register_director_routes()
    base = routes_module.PREFIX + f"/films/{project['id']}/{film['id']}"
    return store, project, film, table, base


def _options():
    return {'policy': exports.POLICY, 'confirm_encoding': True, 'width': 64, 'height': 64}


async def _terminal(client, url):
    async with asyncio.timeout(20):
        while True:
            response = await client.get(url)
            assert response.status == 200, await response.text()
            result = await response.json()
            if result['state'] not in {'queued', 'running'}:
                return result
            await asyncio.sleep(0.01)


def test_export_final_status_write_failure_is_real_error_not_restart_or_repeat_encoding(tmp_path, monkeypatch):
    store, project, film, table, base = _routes(tmp_path, monkeypatch)
    original_atomic, original_export = exports.atomic_json, exports.run_export
    failure = PermissionError(errno.EACCES, 'explicit permanent final-status denial')
    failure.winerror = 5
    calls = []

    def final_status_denied(path, value):
        if value['state'] in {'success', 'error'}:
            raise failure
        return original_atomic(path, value)

    def counted_export(*args):
        calls.append(args[3])
        return original_export(*args)

    monkeypatch.setattr(exports, 'atomic_json', final_status_denied)
    monkeypatch.setattr(exports, 'run_export', counted_export)

    async def scenario():
        app = web.Application()
        app.add_routes(table)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            job_id = str(uuid.uuid4())
            url = base + '/exports/' + job_id
            response = await client.post(url, json=_options())
            assert response.status == 200, await response.text()
            result = await _terminal(client, url)
            assert result['state'] == 'error' and result['error'] == str(failure)
            assert result['status_persisted'] is False
            assert '重启' not in result['error']
            assert (result['project_id'], result['film_id'], result['id']) == (project['id'], film['id'], job_id)
            disk = json.loads((store.root / 'film_exports' / (job_id + '.json')).read_text(encoding='utf8'))
            assert disk['state'] == 'running' and disk['done'] == disk['total'] == 2
            assert 'status_persisted' not in disk
            output = store.root / 'film_exports' / (job_id + '.mp4')
            completed_sha = file_sha(output)
            task = routes_module._FILM_EXPORT_TASKS[job_id]
            assert task.done() and task.exception() is None and task.get_stack() == []
            assert task.result() == {'state': 'error', 'error': str(failure), 'status_persisted': False}
            response = await client.post(url, json=_options())
            assert response.status == 200 and calls == [job_id]
            assert (await _terminal(client, url))['error'] == str(failure)
            assert file_sha(output) == completed_sha
            wrong = routes_module.PREFIX + f"/films/{uuid.uuid4()}/{film['id']}/exports/{job_id}"
            assert (await client.get(wrong)).status == 400

            # A completed volatile error must not be counted as active encoding.
            def next_failure(*_args):
                raise ValueError('separate new request failure')
            monkeypatch.setattr(exports, 'run_export', next_failure)
            next_url = base + '/exports/' + str(uuid.uuid4())
            response = await client.post(next_url, json=_options())
            assert response.status == 200, await response.text()
            assert (await _terminal(client, next_url))['error'] == 'separate new request failure'
            assert calls == [job_id] and file_sha(output) == completed_sha

            # A genuine absent worker still uses the original interrupted state.
            stale = str(uuid.uuid4())
            exports.reserve_export(store, project['id'], film['id'], stale, _options())
            assert (await _terminal(client, base + '/exports/' + stale))['state'] == 'interrupted'
        finally:
            await client.close()

    asyncio.run(scenario())


def test_export_cancelled_error_is_not_swallowed_as_a_storage_failure(tmp_path, monkeypatch):
    store, project, film, table, base = _routes(tmp_path, monkeypatch)
    entered, release = threading.Event(), threading.Event()

    def cancelled(*_args):
        entered.set()
        assert release.wait(5)
        raise asyncio.CancelledError('actual cancellation remains cancellation')

    monkeypatch.setattr(exports, 'run_export', cancelled)

    async def scenario():
        app = web.Application()
        app.add_routes(table)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            job_id = str(uuid.uuid4())
            url = base + '/exports/' + job_id
            response = await client.post(url, json=_options())
            assert response.status == 200, await response.text()
            assert await asyncio.to_thread(entered.wait, 2)
            task = routes_module._FILM_EXPORT_TASKS[job_id]
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert job_id not in routes_module._FILM_EXPORT_TASKS
            result = await _terminal(client, url)
            assert result['state'] == 'interrupted' and 'status_persisted' not in result
            assert not (store.root / 'film_exports' / (job_id + '.mp4')).exists()
        finally:
            release.set()
            await client.close()

    asyncio.run(scenario())


def test_export_existing_durable_error_keeps_disk_authority_and_releases_task(tmp_path, monkeypatch):
    store, project, film, table, base = _routes(tmp_path, monkeypatch)

    def durable_error(selected_store, owner, selected_film, job_id):
        job = exports.export_status(selected_store, owner, selected_film, job_id)
        job.update(state='error', error='persisted original encoder error')
        atomic_json(store.root / 'film_exports' / (job_id + '.json'), job)
        return job

    monkeypatch.setattr(exports, 'run_export', durable_error)

    async def scenario():
        app = web.Application()
        app.add_routes(table)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            job_id = str(uuid.uuid4())
            url = base + '/exports/' + job_id
            assert (await client.post(url, json=_options())).status == 200
            result = await _terminal(client, url)
            assert result['state'] == 'error' and result['error'] == 'persisted original encoder error'
            assert 'status_persisted' not in result
            assert job_id not in routes_module._FILM_EXPORT_TASKS
        finally:
            await client.close()

    asyncio.run(scenario())
