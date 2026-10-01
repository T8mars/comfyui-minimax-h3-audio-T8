"""Actual Chromium controls and real package functions; HTTP/Core are isolated fixtures."""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from urllib.parse import urlparse
import zipfile
import json
import pytest
from email.parser import BytesParser
from email.policy import default

from playwright.sync_api import sync_playwright, expect

from test_director_bundle import bundle_fixture, upload
from h3_audio_t8_pkg.director_bundle import prepare_bundle, build_bundle, inspect_bundle, import_bundle, imported_results, import_preview

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('legacy_flags', [False, True])
def test_ui_package_export_preview_download_import_new_project_and_recover(tmp_path, legacy_flags):
    store, output, project, records, _plan, source_zip = bundle_fixture(tmp_path, draft_flags=not legacy_flags)

    class Quiet(SimpleHTTPRequestHandler):
        def do_POST(self):
            if self.path == '/bundle-imports':
                body = self.rfile.read(int(self.headers['Content-Length']))
                message = BytesParser(policy=default).parsebytes(('Content-Type: '+self.headers['Content-Type']+'\r\n\r\n').encode()+body)
                part = next(message.iter_parts())
                assert part.get_param('name', header='content-disposition') == 'file'
                assert part.get_payload(decode=True) == source_zip.read_bytes()
                data = json.dumps(inspect_bundle(store, upload(store, source_zip))).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            else:
                self.send_error(404)

        def do_GET(self):
            parts = urlparse(self.path).path.split('/')
            if len(parts) == 5 and parts[1] == 'bundles' and parts[-1] == 'file':
                data = build_bundle(store, parts[2], parts[3]).read_bytes()
                self.send_response(200)
                self.send_header('Content-Type', 'application/zip')
                self.send_header('Content-Disposition', 'attachment; filename="director-project.zip"')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            else:
                super().do_GET()

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
            errors, imported, submitted = [], [], []
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.on('dialog', lambda dialog: dialog.accept())
            page.route('**/models', lambda route: route.fulfill(json={}))
            page.route('**/generate', lambda route: (submitted.append(1), route.fulfill(json={})))
            page.route('**/projects/*', lambda route: route.fulfill(json=store.load(urlparse(route.request.url).path.split('/')[-1])))

            def result_route(route):
                pid = urlparse(route.request.url).path.split('/')[-1]
                route.fulfill(json={'results': records if pid == project['id'] else imported_results(store, pid, output)})

            page.route('**/results/**', result_route)
            page.route('**/bundles/prepare', lambda route: route.fulfill(json=prepare_bundle(store, route.request.post_data_json['project'], records, output, route.request.post_data_json['include_results'])))

            def build_route(route):
                parts = urlparse(route.request.url).path.split('/')
                archive = build_bundle(store, parts[-3], parts[-2])
                route.fulfill(json={'ready': True, 'bytes': archive.stat().st_size})

            def apply_route(route):
                parts = urlparse(route.request.url).path.split('/')
                result = import_bundle(store, output, parts[-2], route.request.post_data_json['new_project_id'])
                imported.append(result)
                route.fulfill(json=result)

            page.route('**/bundles/*/*/build', build_route)
            page.route('**/bundle-imports/*', lambda route: route.fulfill(json=import_preview(store, urlparse(route.request.url).path.split('/')[-1])))
            page.route('**/bundle-imports/*/apply', apply_route)
            page.goto(f'http://127.0.0.1:{server.server_port}/index.html', wait_until='networkidle')
            page.evaluate("project=>window.postMessage({type:'t8-director:init',project},location.origin)", project)
            expect(page.locator('.o-shotbar [data-shot]')).to_have_count(2)
            page.locator('.w-header-actions > .w-menu > summary').click()
            page.locator('[data-bundle-open]').click()
            page.locator('[data-bundle-prepare]').click()
            expect(page.locator('[data-bundle-status]')).to_contain_text('3 个媒体文件')
            expect(page.locator('.w-bundle-files tbody tr')).to_have_count(3)
            page.locator('[data-bundle-commit]').click()
            expect(page.locator('[data-bundle-status]')).to_contain_text('ZIP 已生成')
            with page.expect_download() as pending:
                page.locator('[data-bundle-body] a').click()
            assert pending.value.suggested_filename == 'director-project.zip'
            with zipfile.ZipFile(pending.value.path()) as downloaded:
                assert 'manifest.json' in downloaded.namelist()
            page.locator('[data-bundle-file]').set_input_files(source_zip)
            expect(page.locator('[data-bundle-status]')).to_contain_text('3 个媒体文件')
            assert len(store.list()) == 1
            for width in (1366, 1920, 2560, 640):
                page.set_viewport_size({'width': width, 'height': 768})
                assert page.locator('[data-bundle-dialog]').evaluate('d=>d.scrollWidth<=d.clientWidth')
            page.set_viewport_size({'width': 1366, 'height': 768})
            page.screenshot(path=str(tmp_path/'bundle.png'))
            print('BUNDLE_SCREENSHOT', tmp_path/'bundle.png')
            page.locator('[data-bundle-commit]').click()
            expect(page.locator('[data-bundle-status]')).to_contain_text('新工程已保存')
            expect(page.locator('[data-project-title]')).to_have_value(project['title'])
            assert len(store.list()) == 2
            page.locator('[data-bundle-last]').click()
            expect(page.locator('[data-bundle-commit]')).to_be_visible()
            page.locator('[data-bundle-commit]').click()
            expect(page.locator('[data-bundle-status]')).to_contain_text('新工程已保存')
            assert len(store.list()) == 2 and imported[-1]['already_created']
            page.locator('[data-bundle-body] [data-open-project]').click()
            expect(page.locator('[data-bundle-dialog]')).not_to_be_visible()
            expect(page.locator('[data-project-title]')).to_have_value(project['title']+' · 导入副本')
            # Real ZIP functions + real browser mode switching: preserve both drafts,
            # including intentionally empty text, in current and old flagless packages.
            expect(page.locator('[data-field="simplePrompt"]')).to_have_value('@image1 微笑')
            page.locator('[data-writing="advanced"]').click()
            expect(page.locator('[data-field="prompt"]')).to_have_value('独立高级原稿 @image1')
            page.locator('[data-writing="simple"]').click()
            expect(page.locator('[data-field="simplePrompt"]')).to_have_value('@image1 微笑')
            page.locator('.o-shotbar [data-shot]').nth(1).click()
            page.locator('[data-writing="simple"]').click()
            expect(page.locator('[data-field="simplePrompt"]')).to_have_value('')
            page.locator('[data-writing="advanced"]').click()
            expect(page.locator('[data-field="prompt"]')).to_have_value('')
            page.locator('.o-shotbar [data-shot]').first.click()
            page.locator('[data-view="output"]').click()
            expect(page.locator('[data-adopt-version]')).to_have_text('已采用')
            assert not errors and not submitted
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
