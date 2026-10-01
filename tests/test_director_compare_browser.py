"""Real Chromium + CPU encoded media, actual snapshot and film functions, no Core/GPU."""
from copy import deepcopy
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
import uuid

from playwright.sync_api import sync_playwright, expect

from test_director_film import setup, video
from h3_audio_t8_pkg.director_film import prepare_film, film_media
from h3_audio_t8_pkg.director_routes import director_result_snapshot, copy_version_project
from h3_audio_t8_pkg.director_project import atomic_json, sha

ROOT = Path(__file__).resolve().parents[1]


def test_actual_compare_play_seek_audio_and_copy_version_does_not_replace_draft(tmp_path):
    store, output, project, records = setup(tmp_path)
    video(output/'0.mp4', frames=48, tone=440)
    video(output/'1.mp4', frames=48, tone=880)
    record = deepcopy(records[1])
    record.update(shot_id=project['current'], prompt_id='second-version')
    records.insert(0, record)
    request_id = str(uuid.uuid4())
    records[1].update(request_id=request_id, snapshot_available=True)
    snapshot = {'project': deepcopy(project), 'shot_id': project['current'], 'seed': 42, 'recipe': 'test'}
    atomic_json(store.root/'requests'/f'{request_id}.json', {'project_id': project['id'], 'shot_id': project['current'],
                                                           'snapshot': snapshot, 'snapshot_sha256': sha(snapshot)})

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
            errors, manifests, copies, submissions = [], [], [], []
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
                data = path.read_bytes()
                byte_range = route.request.headers.get('range')
                if byte_range:
                    start, end = byte_range.removeprefix('bytes=').split('-')
                    start, end = int(start or 0), int(end) if end else len(data)-1
                    route.fulfill(status=206, body=data[start:end+1], content_type='video/mp4',
                                  headers={'Accept-Ranges': 'bytes', 'Content-Range': f'bytes {start}-{end}/{len(data)}'})
                else:
                    route.fulfill(body=data, content_type='video/mp4', headers={'Accept-Ranges': 'bytes'})

            def version(route):
                if route.request.method == 'POST':
                    result = copy_version_project(store, project['id'], request_id, route.request.post_data_json['new_project_id'])
                    copies.append(result)
                else:
                    result = director_result_snapshot(store, project['id'], request_id)
                route.fulfill(json=result)

            page.route('**/films/prepare', prepare)
            page.route('**/films/*/*/media/*', media)
            page.route('**/snapshots/**', version)
            page.goto(f'http://127.0.0.1:{server.server_port}/index.html', wait_until='networkidle')
            page.evaluate("project=>window.postMessage({type:'t8-director:init',project},location.origin)", project)
            expect(page.locator('.o-shotbar [data-shot]')).to_have_count(2)
            page.locator('[data-view="output"]').click()
            expect(page.locator('[data-compare-open]')).to_be_enabled()
            page.locator('[data-compare-open]').click()
            expect(page.locator('[data-compare-status]')).to_contain_text('已固定两版')
            for width in (1366, 1920, 2560, 640):
                page.set_viewport_size({'width': width, 'height': 768})
                rect = page.locator('[data-compare-close]').bounding_box()
                assert rect['x'] >= 0 and rect['x']+rect['width'] <= width
                assert page.locator('[data-compare-dialog]').evaluate('d=>d.scrollWidth<=d.clientWidth')
            page.set_viewport_size({'width': 1366, 'height': 768})
            page.screenshot(path=str(tmp_path/'compare.png'))
            print('COMPARE_SCREENSHOT', tmp_path/'compare.png')
            a, b = page.locator('[data-compare-video-a]'), page.locator('[data-compare-video-b]')
            assert len(manifests) == 2 and all(item['ready'] for item in manifests)
            assert a.evaluate('v=>v.muted') is False and b.evaluate('v=>v.muted') is True
            page.locator('[data-compare-audio]').select_option('b')
            assert a.evaluate('v=>v.muted') is True and b.evaluate('v=>v.muted') is False
            a.evaluate('v=>v.muted=false')
            page.wait_for_function("document.querySelector('[data-compare-video-b]').muted")
            expect(page.locator('[data-compare-audio]')).to_have_value('a')
            a.evaluate('v=>v.currentTime=0.4')
            page.wait_for_function("Math.abs(document.querySelector('[data-compare-video-b]').currentTime-0.4)<0.08")
            a.evaluate('v=>v.play()')
            page.wait_for_function("document.querySelector('[data-compare-video-a]').currentTime>0.7")
            a.evaluate('v=>v.pause()')
            assert b.evaluate('v=>v.paused') is True
            assert abs(a.evaluate('v=>v.currentTime')-b.evaluate('v=>v.currentTime')) < 0.15
            page.locator('[data-compare-close]').click()
            expect(page.locator('[data-compare-dialog]')).not_to_be_visible()
            expect(page.locator('[data-adopt-version]')).to_have_text('已采用')
            page.locator('[data-version-details]').click()
            expect(page.locator('[data-service="copy-version"]')).to_be_visible()
            page.locator('[data-service="copy-version"]').click()
            expect(page.locator('[data-dialog]')).to_contain_text('版本新草稿已保存')
            assert len(copies) == 1 and copies[0]['id'] != project['id']
            copied = store.load(copies[0]['id'])
            assert copied['doc']['shots'][0]['seed'] == 42
            assert all('adoptedResultId' not in shot for shot in copied['doc']['shots'])
            expect(page.locator('.o-shotbar [data-shot]')).to_have_count(2)
            expect(page.locator('[data-adopt-version]')).to_have_text('已采用')
            assert not errors and not submissions
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
