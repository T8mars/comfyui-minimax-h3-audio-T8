"""Real Chromium clicks plus production registration/results/film handlers, no queue."""
from contextlib import contextmanager
from copy import deepcopy
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from threading import Thread
from urllib.parse import urlparse, parse_qs
import uuid

from playwright.sync_api import sync_playwright, expect

from h3_audio_t8_pkg import director_routes
from h3_audio_t8_pkg.director_project import contained
from test_director_external_routes import external_route as _route_fixture
from test_director_radar_routes import radar_routes as _radar_fixture, dispatch

radar_routes = _radar_fixture
external_route = _route_fixture
ROOT = Path(__file__).resolve().parents[1]
DRAFT = "() => JSON.parse(localStorage.getItem('t8director.draft:'+sessionStorage.getItem('t8director.tab')))"


@contextmanager
def actual_browser(external_route):
    store, output, source, original, handlers = external_route
    observed, errors = [], []

    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def guess_type(self, path):
            return 'text/javascript' if path.endswith('.mjs') else super().guess_type(path)

        def reply(self, value, status=200):
            data = json.dumps(value, ensure_ascii=False).encode('utf8')
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def media(self, path):
            data = path.read_bytes()
            self.send_response(200)
            self.send_header('Content-Type', 'video/mp4')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = urlparse(self.path).path
            observed.append(('GET', path, None))
            if path.endswith('/models'):
                return self.reply({})
            if '/results/' in path:
                return self.reply(director_routes.director_project_results(store, path.rsplit('/', 1)[-1], output_root=output))
            if path.endswith('/projects'):
                return self.reply({'projects': store.list()})
            if '/projects/' in path:
                return self.reply(store.load(path.rsplit('/', 1)[-1]))
            if '/assets/' in path:
                return self.media(source)
            if path == '/view':
                query = parse_qs(urlparse(self.path).query)
                return self.media(contained(output, Path(query.get('subfolder', [''])[0]) / query['filename'][0]))
            return super().do_GET()

        def do_POST(self):
            path = urlparse(self.path).path
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            observed.append(('POST', path, deepcopy(body)))
            api_path = director_routes.PREFIX + path
            if api_path in handlers and path.endswith(('/external-takes', '/films/prepare')):
                code, result = dispatch(handlers[api_path], body)
                return self.reply(result, code)
            if '/projects/' in path:
                return self.reply(store.save(body['project'], body['expected_revision']))
            errors.append('Unexpected mutation: ' + path)
            self.reply({'error': 'Unexpected mutation'}, 400)

    server = ThreadingHTTPServer(('127.0.0.1', 0), partial(Handler, directory=str(ROOT / 'web/director')))
    Thread(target=server.serve_forever, daemon=True).start()
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                page = browser.new_page(viewport={'width': 1440, 'height': 1000})
                page.on('pageerror', lambda error: errors.append(str(error)))
                page.goto(f'http://127.0.0.1:{server.server_port}/index.html', wait_until='networkidle')
                page.wait_for_function("!document.querySelector('#t8-obsidian-lab').inert")
                page.evaluate("p=>window.postMessage({type:'t8-director:init',project:p},location.origin)", original['project'])
                expect(page.locator('[data-external-take-open]')).to_be_visible()
                page.locator('[data-service="save"]').click()
                expect(page.locator('[data-save]')).to_contain_text('已真实保存')
                yield page, observed, errors
                assert not errors
                assert not any('/generate' in path or '/batches' in path or '/prompt' in path for _, path, _ in observed)
            finally:
                browser.close()
    finally:
        server.shutdown()
        server.server_close()


def open_registration(page):
    page.locator('[data-external-take-open]').click()
    expect(page.locator('[data-external-label]')).to_be_visible()


def test_actual_ui_registration_manual_adopt_trim_save_and_reopen(external_route):
    store, _output, source, original, _handlers = external_route
    original_bytes = source.read_bytes()
    with actual_browser(external_route) as (page, observed, _errors):
        before = page.evaluate(DRAFT)
        open_registration(page)
        page.locator('[data-external-label]').fill('后期修复 · 审核🙂')
        page.locator('[data-external-category]').select_option('repaired')
        page.locator('[data-service="register-external-take"]').click()
        expect(page.locator('[data-external-label]')).not_to_be_visible()
        assert page.evaluate(DRAFT) == before  # No implicit adoption or draft edit.
        assert store.load(before['id']) == before
        page.locator('[data-view="output"]').click()
        expect(page.locator('[data-result-version]')).to_contain_text('外部成片 · 不可重采样')
        expect(page.locator('[data-adopt-version]')).to_have_text('采用此版')
        key = page.locator('[data-result-version]').input_value()
        assert key.startswith('external:')
        page.locator('[data-version-details]').click()
        expect(page.locator('[data-dialog-title]')).to_have_text('外部成片版本')
        expect(page.locator('[data-dialog-body]')).to_contain_text('frame_pts_sha256')
        expect(page.locator('[data-service="copy-version"]')).to_have_count(0)
        page.locator('[data-action="close"]').click()
        page.locator('[data-adopt-version]').click()
        assert page.evaluate(DRAFT)['doc']['shots'][0]['adoptedResultId'] == key
        page.locator('[data-trim-open]').click()
        expect(page.locator('[data-trim-status]')).to_contain_text('12 帧 · 24 fps')
        page.locator('[data-trim-in]').fill('1')
        page.locator('[data-trim-out]').fill('10')
        page.locator('[data-trim-apply]').click()
        expect(page.locator('[data-trim-summary]')).to_contain_text('1–10 帧')
        page.locator('[data-service="save"]').click()
        expect(page.locator('[data-save]')).to_contain_text('已真实保存')
        saved = store.load(before['id'])
        assert saved['doc']['shots'][0]['filmTrim'] == {'in_frame': 1, 'out_frame': 10}
        page.reload(wait_until='networkidle')
        page.locator('[data-view="output"]').click()
        expect(page.locator('[data-adopt-version]')).to_have_text('已采用')
        expect(page.locator('[data-trim-summary]')).to_contain_text('1–10 帧')
        assert page.evaluate(DRAFT)['doc']['shots'][0]['adoptedResultId'] == key
        external_posts = [body for method, path, body in observed if method == 'POST' and path.endswith('/external-takes')]
        assert len(external_posts) == 1 and external_posts[0]['provenance_category'] == 'repaired'
        assert source.read_bytes() == original_bytes
        assert original['project']['revision'] == 1


def test_actual_ui_uncertain_response_retry_reuses_original_uuid(external_route):
    store, output, _source, original, _handlers = external_route
    with actual_browser(external_route) as (page, observed, _errors):
        calls = []

        def lose_first_response(route):
            body = route.request.post_data_json
            calls.append(deepcopy(body))
            response = route.fetch()
            assert response.status == 200, response.json()
            if len(calls) == 1:
                route.abort('failed')  # Actual handler committed; only response delivery failed.
            else:
                route.fulfill(response=response)

        page.route('**/external-takes', lose_first_response)
        before = page.evaluate(DRAFT)
        open_registration(page)
        page.locator('[data-external-label]').fill('原冻结请求')
        page.locator('[data-service="register-external-take"]').click()
        page.wait_for_function("Object.keys(sessionStorage).some(k=>k.startsWith('t8director.external:'))")
        # Wait until the request handler has re-enabled the failed-request button.
        expect(page.locator('[data-service="register-external-take"]')).to_be_enabled()
        assert len(calls) == 1
        page.locator('[data-action="close"]').click()
        page.reload(wait_until='networkidle')
        open_registration(page)
        expect(page.locator('[data-external-label]')).to_be_disabled()
        expect(page.locator('[data-external-label]')).to_have_value('原冻结请求')
        expect(page.locator('[data-service="register-external-take"]')).to_have_text('重试原登记请求')
        page.locator('[data-service="register-external-take"]').click()
        expect(page.locator('[data-external-label]')).not_to_be_visible()
        assert len(calls) == 2 and calls[0] == calls[1]
        assert page.evaluate(DRAFT)['doc'] == before['doc']
        records = director_routes.director_project_results(store, original['project']['id'], output_root=output)['results']
        assert len(records) == 1 and records[0]['external_take_id'] == calls[0]['take_id']
        assert not page.evaluate("Object.keys(sessionStorage).some(k=>k.startsWith('t8director.external:'))")
        assert len([row for row in observed if row[0] == 'POST' and row[1].endswith('/external-takes')]) == 2


def test_actual_ui_storage_failure_never_sends_unrecoverable_registration(external_route):
    _store, output, _source, _original, _handlers = external_route
    with actual_browser(external_route) as (page, observed, _errors):
        before = page.evaluate(DRAFT)
        page.evaluate("""() => { const original=Storage.prototype.setItem;
            Storage.prototype.setItem=function(key,value){
                if(key.startsWith('t8director.external:'))throw Error('fixture quota');
                return original.call(this,key,value);
            }; }""")
        open_registration(page)
        page.locator('[data-service="register-external-take"]').click()
        expect(page.locator('[data-notice]')).to_contain_text('未提交')
        assert not any(method == 'POST' and path.endswith('/external-takes') for method, path, _ in observed)
        assert not list(output.rglob('*.mp4')) and page.evaluate(DRAFT) == before


def test_actual_ui_late_registration_response_never_overwrites_new_project(external_route):
    store, output, _source, original, _handlers = external_route
    replacement = deepcopy(original['project'])
    replacement['id'] = str(uuid.uuid4())
    replacement['revision'] = 0
    replacement['doc']['shots'][0]['id'] = str(uuid.uuid4())
    replacement['current'] = replacement['doc']['shots'][0]['id']
    replacement['doc']['shots'][0]['simplePrompt'] = '切换后的新工程，不覆盖'
    with actual_browser(external_route) as (page, _observed, _errors):
        def delayed_response(route):
            response = route.fetch()
            assert response.status == 200, response.json()
            page.evaluate("p=>window.postMessage({type:'t8-director:init',project:p},location.origin)", replacement)
            page.wait_for_function("id=>JSON.parse(localStorage.getItem('t8director.draft:'+sessionStorage.getItem('t8director.tab'))).id===id", arg=replacement['id'])
            route.fulfill(response=response)

        page.route('**/external-takes', delayed_response)
        open_registration(page)
        page.locator('[data-service="register-external-take"]').click()
        expect(page.locator('[data-notice]')).to_contain_text('当前工程、镜头与编辑窗口保持不变')
        after = page.evaluate(DRAFT)
        assert after['id'] == replacement['id']
        assert after['doc']['shots'][0]['simplePrompt'] == replacement['doc']['shots'][0]['simplePrompt']
        assert 'adoptedResultId' not in after['doc']['shots'][0]
        assert len(director_routes.director_project_results(store, original['project']['id'], output_root=output)['results']) == 1
        assert director_routes.director_project_results(store, replacement['id'], output_root=output)['results'] == []
