"""Real isolated browser collections, backend validation and no generation writes."""
import base64
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
import uuid

from playwright.sync_api import sync_playwright, expect
from h3_audio_t8_pkg.director_project import validate_project

ROOT = Path(__file__).resolve().parents[1]


def test_collections_snippet_template_recipe_group_and_one_step_undo(tmp_path):
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
            errors, posts = [], []
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.on('dialog', lambda dialog: dialog.accept())
            page.on('request', lambda request: posts.append(request.url) if request.method != 'GET' else None)
            page.route('**/models', lambda route: route.fulfill(json={}))
            page.route('**/results/**', lambda route: route.fulfill(json={'results': []}))
            png = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=')
            page.route('**/assets/**', lambda route: route.fulfill(content_type='image/png', body=png))
            page.goto(f'http://127.0.0.1:{server.server_port}/index.html', wait_until='networkidle')
            draft_js = "() => JSON.parse(localStorage.getItem(Object.keys(localStorage).find(key=>key.startsWith('t8director.draft:'))))"
            project = page.evaluate(draft_js)
            aids = [str(uuid.uuid4()) for _ in range(3)]
            project['assets'] = [dict(id=aid, name=f'素材{i}', kind='audio' if i == 2 else 'image', width=1, height=1, duration=4, sha256='a'*64)
                                 for i, aid in enumerate(aids)]
            shot = project['doc']['shots'][0]
            shot.update(mode='refs', tray=aids, refs=aids[:2], simplePrompt='前 @image2 后', seed=42)
            project['doc']['sharedRefs'] = aids[:1]
            project['doc']['sampling'] = {'mode': 'two_pass', 'output_mp': '0.6',
                'low_loras': [{'name': 'low.safetensors', 'strength': .8, 'enabled': True}, {'name': 'extra.safetensors', 'strength': .3, 'enabled': True}],
                'high_loras': [{'name': 'high.safetensors', 'strength': .5, 'enabled': True}]}
            page.evaluate("p=>window.postMessage({type:'t8-director:init',project:p},location.origin)", project)
            text = page.locator('[data-field="simplePrompt"]')
            expect(text).to_have_value('前 @image2 后')
            text.focus()
            text.evaluate('el=>el.setSelectionRange(2,9)')

            def more():
                menu = page.locator('.w-header-actions > .w-menu')
                if not menu.evaluate('el=>el.open'):
                    menu.locator(':scope > summary').click()

            def open_library(kind):
                more()
                page.locator('[data-creation-open]').click()
                page.locator('[data-creation-kind]').select_option(kind)

            def save(name):
                page.locator('[data-creation-name]').fill(name)
                page.locator('[data-creation-save]').click()
                expect(page.locator('[data-creation-status]')).to_contain_text('已收藏')

            def close():
                page.locator('[data-creation-close]').click()
                expect(page.locator('.w-header-actions > .w-menu > summary')).to_be_focused()

            open_library('snippet')
            page.locator('[data-creation-dialog]').press('Escape')
            expect(page.locator('[data-creation-dialog]')).not_to_be_visible()
            expect(page.locator('.w-header-actions > .w-menu > summary')).to_be_focused()
            open_library('snippet')
            save('角色引用')
            close()
            snapshot = page.evaluate(draft_js)
            validate_project(snapshot)
            item = snapshot['doc']['creationLibrary']['items'][0]
            assert item['text'] == '@image2' and item['bindings'] == {'@image2': aids[1]}
            text.fill('开始结束')
            text.evaluate('el=>el.setSelectionRange(2,2)')
            open_library('snippet')
            page.locator('[data-creation-use]').click()
            expect(page.locator('[data-creation-status]')).to_contain_text('只是预览')
            assert page.evaluate(draft_js)['doc']['shots'][0]['simplePrompt'] == '开始结束'
            page.locator('[data-creation-commit]').click()
            expect(text).to_have_value('开始@image2结束')
            more()
            page.locator('[data-action="undo"]').click()
            expect(text).to_have_value('开始结束')

            open_library('template')
            save('可复用镜头')
            page.locator('[data-creation-use]').click()
            expect(page.locator('[data-creation-preview]')).to_contain_text('新建模板副本')
            page.locator('[data-creation-commit]').click()
            expect(page.locator('.o-shotbar [data-shot]')).to_have_count(2)
            after = page.evaluate(draft_js)
            validate_project(after)
            new = after['doc']['shots'][1]
            assert new['id'] != shot['id'] and new['samplingInherit'] is False
            assert len(new['sampling']['low_loras']) == 2 and len(new['sampling']['high_loras']) == 1
            assert new.get('adoptedResultId') is None and 'seed' not in new
            assert after['doc']['shots'][0]['seed'] == 42

            open_library('recipe')
            save('双采配方')
            close()
            page.locator('[data-action="new"]').click()
            third = page.evaluate(draft_js)['current']
            page.locator(f'[data-shot-check="{shot["id"]}"]').check()
            page.locator(f'[data-shot-check="{third}"]').check()
            before = page.evaluate(draft_js)
            open_library('recipe')
            page.locator('[data-creation-targets]').select_option('selected')
            page.locator('[data-creation-use]').click()
            expect(page.locator('[data-creation-preview]')).to_contain_text('将修改 2 镜')
            page.screenshot(path=str(tmp_path/'creation-library.png'))
            page.locator('[data-creation-commit]').click()
            after = page.evaluate(draft_js)
            validate_project(after)
            assert after['doc']['shots'][1] == before['doc']['shots'][1]
            assert after['doc']['shots'][0]['samplingInherit'] is False
            assert after['doc']['shots'][2]['samplingInherit'] is False
            assert after['doc']['shots'][2]['sampling']['low_loras'] == new['sampling']['low_loras']
            assert after['doc']['generation'] == before['doc']['generation']
            assert after['doc']['shots'][0]['seed'] == 42

            page.locator(f'[data-shot="{shot["id"]}"]').click()
            open_library('group')
            save('人物素材组合')
            close()
            page.locator(f'[data-shot="{third}"]').click()
            open_library('group')
            page.locator('[data-creation-use]').click()
            page.locator('[data-creation-commit]').click()
            after = page.evaluate(draft_js)
            validate_project(after)
            assert set(after['doc']['shots'][2]['tray']) == set(aids)
            assert after['doc']['shots'][2]['sound'] == 'native' and after['doc']['shots'][2]['audio'] is None
            assert aids[1] in after['doc']['shots'][2]['refs']
            assert after['doc']['sharedRefs'] == aids[:1]
            page.reload(wait_until='networkidle')
            assert page.evaluate(draft_js)['doc']['creationLibrary'] == after['doc']['creationLibrary']

            # Deleting a saved group never removes actual source media or applied shots.
            open_library('group')
            page.locator('[data-creation-delete]').click()
            expect(page.locator('[data-creation-status]')).to_contain_text('没有删除原素材')
            close()
            deleted = page.evaluate(draft_js)
            assert deleted['assets'] == after['assets']
            assert deleted['doc']['shots'] == after['doc']['shots']
            more()
            page.locator('[data-action="undo"]').click()
            assert page.evaluate(draft_js)['doc']['creationLibrary'] == after['doc']['creationLibrary']
            # Global insertion cannot silently promote a shot-only bound asset.
            global_text = page.locator('[data-global]')
            global_text.evaluate("el=>el.closest('details').open=true")
            global_text.focus()
            global_text.evaluate('el=>el.setSelectionRange(0,0)')
            before_global = page.evaluate(draft_js)
            open_library('snippet')
            page.locator('[data-creation-use]').click()
            expect(page.locator('[data-creation-commit]')).to_be_disabled()
            assert page.evaluate(draft_js) == before_global
            page.locator('[data-creation-missing]').check()
            page.locator('[data-creation-commit]').click()
            changed = page.evaluate(draft_js)
            assert changed['doc']['global'].startswith('@missing_image2')
            assert changed['doc']['sharedRefs'] == aids[:1]
            more()
            page.locator('[data-action="undo"]').click()
            assert page.evaluate(draft_js)['doc'] == before_global['doc']
            # Direct parameter copy is still preview-first; changing scope invalidates it.
            open_library('recipe')
            page.locator('[data-creation-copy]').click()
            expect(page.locator('[data-creation-preview]')).to_be_visible()
            page.locator('[data-creation-targets]').select_option('all')
            expect(page.locator('[data-creation-preview]')).not_to_be_visible()
            close()
            assert page.evaluate(draft_js)['doc'] == before_global['doc']
            assert not posts and not errors
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
