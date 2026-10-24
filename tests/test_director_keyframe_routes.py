"""Actual registered aiohttp module delivery, including session-relative URL."""
import asyncio
from pathlib import Path

from aiohttp.test_utils import TestClient, TestServer

from h3_audio_t8_pkg import director_routes
from h3_audio_t8_pkg.director_project import ProjectStore
from test_director_caption_routes import routes_for


def test_current_keyframe_modules_serve_at_both_native_import_bases(tmp_path, monkeypatch):
    store = ProjectStore(tmp_path/'user', tmp_path/'input')
    app = routes_for(monkeypatch, store)
    root = Path(director_routes.__file__).resolve().parents[1]/'web/director'

    async def scenario():
        async with TestClient(TestServer(app)) as client:
            page = await client.get(director_routes.PREFIX+'/ui')
            assert page.status == 200 and await page.read() == (root/'index.html').read_bytes()
            for name in ('keyframes_state.mjs', 'keyframes_ui.mjs', 'keyframes_ui.css'):
                response = await client.get(director_routes.PREFIX+'/ui-modules/'+name)
                assert response.status == 200 and await response.read() == (root/name).read_bytes()
            for name in ('session.mjs', 'keyframes_state.mjs'):
                response = await client.get(director_routes.PREFIX+'/'+name)
                assert response.status == 200 and await response.read() == (root/name).read_bytes()
            assert not store.list()

    asyncio.run(scenario())
