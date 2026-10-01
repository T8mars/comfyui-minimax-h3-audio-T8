"""Actual CPU-encoded video played in isolated Chromium; no Core or GPU."""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

from playwright.sync_api import sync_playwright, expect

from test_director_film import setup
from h3_audio_t8_pkg.director_film import prepare_film, film_media
from h3_audio_t8_pkg.director_film_export import reserve_export, run_export, export_status


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_film_actual_playback_missing_adoption_and_navigation(tmp_path):
    store, output, project, records = setup(tmp_path)

    class Quiet(SimpleHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def guess_type(self, path):
            return 'text/javascript' if path.endswith('.mjs') else super().guess_type(path)

    server = ThreadingHTTPServer(('127.0.0.1', 0), partial(Quiet, directory=str(ROOT/'web/director')))
    Thread(target=server.serve_forever, daemon=True).start()
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport={'width': 1366, 'height': 768})
            errors, submissions, manifests = [], [], []
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.route('**/models', lambda route: route.fulfill(json={}))
            page.route('**/results/**', lambda route: route.fulfill(json={'results': records}))
            page.route('**/generate', lambda route: (submissions.append(route.request.url), route.fulfill(json={})))

            def prepare(route):
                manifest = prepare_film(store, route.request.post_data_json['project'], records, output)
                manifests.append(manifest)
                route.fulfill(json=manifest)

            def media(route):
                parts = route.request.url.split('/')
                path = film_media(store, parts[-4], parts[-3], int(parts[-1]))
                route.fulfill(body=path.read_bytes(), content_type='video/mp4')

            def export(route):
                parts = route.request.url.split('/')
                owner, film_id, job_id = parts[-4], parts[-3], parts[-1]
                if route.request.method == 'POST':
                    job, _ = reserve_export(store, owner, film_id, job_id, route.request.post_data_json)
                    run_export(store, owner, film_id, job_id)
                else:
                    job = export_status(store, owner, film_id, job_id)
                route.fulfill(json=job)

            page.route('**/films/prepare', prepare)
            page.route('**/films/*/*/media/*', media)
            page.route('**/films/*/*/exports/*', export)
            page.goto(f'http://127.0.0.1:{server.server_port}/index.html', wait_until='networkidle')
            page.evaluate("project=>window.postMessage({type:'t8-director:init',project},location.origin)", project)
            expect(page.locator('.o-shotbar [data-shot]')).to_have_count(2)
            page.locator('[data-film-open]').click()
            expect(page.locator('.w-film-player')).to_be_visible()
            page.wait_for_function("document.querySelector('.w-film-player').readyState>=1")
            assert manifests[0]['ready']
            page.locator('[data-film-restart]').click()
            expect(page.locator('[data-film-state]')).to_contain_text('整片播放结束', timeout=10000)
            assert page.locator('.w-film-player').evaluate('video=>video.videoWidth') == 32
            assert page.locator('[data-film-shot="1"]').get_attribute('aria-current') == 'true'
            page.locator('.w-film-export summary').click()
            page.locator('[data-film-width]').fill('64')
            page.locator('[data-film-height]').fill('64')
            page.locator('[data-film-encoding]').check()
            page.locator('[data-film-export]').click()
            expect(page.locator('[data-film-export-status]')).to_contain_text('导出已完成', timeout=15000)
            expect(page.locator('[data-film-export-status] a')).to_have_attribute('download', '')
            page.locator('[data-film-close]').click()
            expect(page.locator('[data-film-dialog]')).not_to_be_visible()
            project['doc']['shots'][0].pop('adoptedResultId')
            page.evaluate("project=>window.postMessage({type:'t8-director:init',project},location.origin)", project)
            page.locator('[data-film-open]').click()
            expect(page.locator('[data-film-state]')).to_contain_text('尚未准备好')
            page.locator('[data-film-last-export]').click()
            expect(page.locator('[data-film-export-status]')).to_contain_text('导出已完成')
            expect(page.locator('[data-film-locate]')).to_have_count(1)
            page.locator('[data-film-locate]').click()
            expect(page.locator('[data-film-dialog]')).not_to_be_visible()
            expect(page.locator('[data-workbench-heading]')).to_have_text('第 01 镜')
            assert not submissions and not errors
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
