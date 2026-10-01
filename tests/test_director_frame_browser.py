"""Actual browser UI with real decoded CPU video/PNG; isolated HTTP boundary."""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

from playwright.sync_api import sync_playwright, expect

from test_director_film import setup
from test_director_d1 import asset
from h3_audio_t8_pkg.director_film import prepare_film
from h3_audio_t8_pkg.director_frame import extract_frame
from h3_audio_t8_pkg.director_project import contained, validate_project

ROOT = Path(__file__).resolve().parents[1]


def test_frame_preview_cancel_replace_new_shot_and_undo(tmp_path):
    store, output, project, records = setup(tmp_path)
    old_asset = asset(store)
    project['assets'] = [old_asset]
    project['doc']['shots'][1].update(first=old_asset['id'], tray=[old_asset['id']], simplePrompt='原文 @image1', writingMode='simple')

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
            page = browser.new_page(viewport={'width': 1366, 'height': 900})
            errors, writes = [], []
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.on('request', lambda req: writes.append(req.url) if req.method != 'GET' else None)
            page.route('**/models', lambda route: route.fulfill(json={}))
            page.route('**/results/**', lambda route: route.fulfill(json={'results': records}))
            page.route('**/films/prepare', lambda route: route.fulfill(json=prepare_film(store, route.request.post_data_json['project'], records, output)))

            def extract(route):
                parts = route.request.url.split('/')
                data = route.request.post_data_json
                route.fulfill(json=extract_frame(store, parts[-3], parts[-2], data['index'], data['frame']))

            def image(route):
                record = store.asset(route.request.url.split('/')[-1])
                route.fulfill(body=contained(store.input_root, record['server_path']).read_bytes(), content_type='image/png')

            page.route('**/films/*/*/frames', extract)
            page.route('**/assets/*', image)
            page.goto(f'http://127.0.0.1:{server.server_port}/index.html', wait_until='networkidle')
            page.evaluate("project=>window.postMessage({type:'t8-director:init',project},location.origin)", project)
            expect(page.locator('.o-shotbar [data-shot]')).to_have_count(2)
            draft_js = "() => JSON.parse(localStorage.getItem(Object.keys(localStorage).find(key=>key.startsWith('t8director.draft:'))))"
            original = page.evaluate(draft_js)

            def open_frame():
                page.locator('[data-view="output"]').click()
                expect(page.locator('[data-frame-open]')).to_be_enabled()
                page.locator('[data-frame-open]').click()
                expect(page.locator('[data-frame-number]')).to_have_value('11')

            def preview():
                page.locator('[data-frame-extract]').click()
                expect(page.locator('[data-frame-status]')).to_contain_text('尚未改动')
                page.wait_for_function("document.querySelector('[data-frame-preview] img')?.naturalWidth===64")

            def undo():
                page.locator('.w-header-actions > .w-menu > summary').click()
                page.locator('[data-action="undo"]').click()

            open_frame()
            preview()
            assert page.evaluate(draft_js) == original
            expect(page.locator('[data-frame-apply]')).to_be_disabled()
            page.locator('[data-frame-close]').click()
            assert page.evaluate(draft_js) == original
            open_frame()
            preview()
            page.locator('[data-frame-replace]').check()
            expect(page.locator('[data-frame-apply]')).to_be_enabled()
            for width in (1366, 640):
                page.set_viewport_size({'width': width, 'height': 900})
                assert page.locator('[data-frame-dialog]').evaluate('d=>d.scrollWidth<=d.clientWidth')
            page.set_viewport_size({'width': 1366, 'height': 900})
            page.screenshot(path=str(tmp_path/'frame-picker.png'))
            print('FRAME_SCREENSHOT', tmp_path/'frame-picker.png')
            page.locator('[data-frame-apply]').click()
            after = page.evaluate(draft_js)
            validate_project(after)
            source, target = after['doc']['shots']
            assert source == original['doc']['shots'][0]
            assert target['first'] != old_asset['id'] and target['simplePrompt'] == '原文 @image1'
            assert target['adoptedResultId'] == original['doc']['shots'][1]['adoptedResultId']
            assert old_asset['id'] in target['tray'] and target['sound'] == 'native'
            assert after['current'] == target['id']
            frame_asset = next(item for item in after['assets'] if item['id'] == target['first'])
            assert frame_asset['source_frame']['frame'] == 11
            assert len(frame_asset['source_frame']['media_sha256']) == 64
            undo()
            undone = page.evaluate(draft_js)
            assert undone['doc'] == original['doc']
            assert len(undone['assets']) == 2  # Existing undo retains uploaded/extracted source files.
            page.locator(f'[data-shot="{source["id"]}"]').click()
            open_frame()
            preview()
            page.locator('[data-frame-number]').fill('0')
            expect(page.locator('[data-frame-apply]')).to_be_disabled()
            preview()
            page.locator('[data-frame-target]').select_option('new')
            page.locator('[data-frame-apply]').click()
            expect(page.locator('.o-shotbar [data-shot]')).to_have_count(3)
            after = page.evaluate(draft_js)
            validate_project(after)
            assert after['doc']['shots'][0] == original['doc']['shots'][0]
            assert after['doc']['shots'][2] == original['doc']['shots'][1]
            assert after['doc']['shots'][1]['mode'] == 'first'
            assert 'adoptedResultId' not in after['doc']['shots'][1]
            undo()
            assert page.evaluate(draft_js)['doc'] == original['doc']
            assert all(url.endswith(('/prepare', '/frames')) for url in writes)
            assert not errors
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
