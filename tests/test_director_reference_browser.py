"""Real isolated Chromium text/caret/library actions; no Core or generation."""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
import base64
import uuid

from playwright.sync_api import sync_playwright, expect

ROOT = Path(__file__).resolve().parents[1]


def test_library_search_and_reference_candidates_keep_scope_text_undo_and_modal_drafts():
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
            errors, submits = [], []
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.on('dialog', lambda dialog: dialog.accept())
            page.route('**/models', lambda route: route.fulfill(json={}))
            page.route('**/results/**', lambda route: route.fulfill(json={'results': []}))
            page.route('**/generate', lambda route: (submits.append(1), route.fulfill(json={})))
            png = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=')
            page.route('**/assets/**', lambda route: route.fulfill(content_type='image/png', body=png))
            page.goto(f'http://127.0.0.1:{server.server_port}/index.html', wait_until='networkidle')
            draft_js = "() => JSON.parse(localStorage.getItem(Object.keys(localStorage).find(key=>key.startsWith('t8director.draft:'))))"
            project = page.evaluate(draft_js)
            ids = [str(uuid.uuid4()) for _ in range(4)]
            project['assets'] = [dict(id=ids[i], name=name, kind=kind, width=1, height=1, duration=4, sha256='a'*64)
                                 for i, (name, kind) in enumerate([('共享图.png', 'image'), ('主角<img>.png', 'image'), ('配音.wav', 'audio'), ('库内视频.mp4', 'video')])]
            shot = project['doc']['shots'][0]
            shot.update(mode='refs', refs=[ids[1]], tray=ids[:3], selected=ids[0], simplePrompt='原文', writingMode='simple')
            project['doc']['sharedRefs'] = [ids[0]]
            page.evaluate("p=>window.postMessage({type:'t8-director:init',project:p},location.origin)", project)
            expect(page.locator('[data-thumbs] [data-tray-id]')).to_have_count(3)
            before = page.evaluate(draft_js)['doc']
            page.locator('.w-header-actions > .w-menu > summary').click()
            page.locator('[data-action="library"]').click()
            expect(page.locator('[data-library-asset]')).to_have_count(4)
            page.locator('[data-library-search]').fill('库内')
            page.locator('[data-library-kind]').select_option('video')
            page.locator('[data-library-scope]').select_option('unbound')
            expect(page.locator('[data-library-asset]:visible')).to_have_count(1)
            assert page.evaluate(draft_js)['doc'] == before
            page.locator('[data-library-search]').fill('无匹配')
            expect(page.locator('[data-library-count-status]')).to_contain_text('没有匹配')
            page.locator('[data-library-reset]').click()
            expect(page.locator('[data-library-asset]:visible')).to_have_count(4)
            page.locator(f'[data-library-asset="{ids[3]}"] [data-bind="refs"]').click()
            expect(page.locator('[data-dialog]')).not_to_be_visible()
            expect(page.locator('[data-thumbs] [data-tray-id]')).to_have_count(4)

            text = page.locator('[data-field="simplePrompt"]')
            text.fill('原文 @主角 后文')
            text.evaluate("el=>{el.setSelectionRange(6,6);el.dispatchEvent(new InputEvent('input',{bubbles:true}));}")
            expect(page.locator('.w-reference-popup')).to_be_visible()
            expect(page.locator('[data-reference-choice]')).to_have_count(1)
            expect(page.locator('[data-reference-choice] strong')).to_contain_text('主角<img>.png')
            text.press('Enter')
            expect(text).to_have_value('原文 @image2 后文')
            expect(text).to_be_focused()
            assert page.evaluate(draft_js)['doc']['shots'][0]['refs'] == [ids[1], ids[3]]
            # Undo insertion, not the other original text or asset binding.
            page.locator('.w-header-actions > .w-menu > summary').click()
            page.locator('[data-action="undo"]').click()
            expect(text).to_have_value('原文 @主角 后文')

            # Synthetic IME events do not open candidates mid-composition.
            text.focus()
            text.evaluate("el=>{el.dispatchEvent(new CompositionEvent('compositionstart',{bubbles:true}));el.value='@配';el.dispatchEvent(new InputEvent('input',{bubbles:true,isComposing:true}));}")
            expect(page.locator('.w-reference-popup')).not_to_be_visible()
            text.evaluate("el=>{el.setSelectionRange(2,2);el.dispatchEvent(new CompositionEvent('compositionend',{bubbles:true,data:'配'}));}")
            expect(page.locator('.w-reference-popup')).to_be_visible()
            text.press('Enter')
            expect(text).to_have_value('@audio1')
            assert page.evaluate(draft_js)['doc']['shots'][0]['sound'] == 'native'

            global_text = page.locator('[data-global]')
            page.locator('[data-global-zone] > summary').click()
            global_text.fill('全片 @主角')
            expect(page.locator('[data-reference-choice]')).to_have_count(1)
            expect(page.locator('[data-reference-choice] small')).to_contain_text('加入全片')
            global_text.press('Enter')
            expect(global_text).to_have_value('全片 @image2')
            assert page.evaluate(draft_js)['doc']['sharedRefs'] == ids[:2]
            page.locator('.w-header-actions > .w-menu > summary').click()
            page.locator('[data-action="undo"]').click()
            assert page.evaluate(draft_js)['doc']['sharedRefs'] == ids[:1]
            expect(global_text).to_have_value('全片 @主角')

            # Expanded editor promotions remain staged until Save; Esc closes only
            # candidates, and Cancel discards both the text and the sharing change.
            global_text.locator('..').locator('[data-expand]').click()
            editor = page.locator('[data-editor]')
            editor.fill('暂存 @主角')
            editor.press('Escape')
            expect(page.locator('[data-dialog]')).to_be_visible()
            expect(page.locator('.w-reference-popup')).not_to_be_visible()
            editor.fill('暂存 @主角')
            editor.press('Enter')
            expect(editor).to_have_value('暂存 @image2')
            assert page.evaluate(draft_js)['doc']['sharedRefs'] == ids[:1]
            page.locator('[data-dialog-body] [data-action="close"]').click()
            expect(global_text).to_have_value('全片 @主角')
            global_text.locator('..').locator('[data-expand]').click()
            editor.fill('保存 @主角')
            editor.press('Enter')
            page.locator('[data-action="editor-save"]').click()
            expect(global_text).to_have_value('保存 @image2')
            assert page.evaluate(draft_js)['doc']['sharedRefs'] == ids[:2]

            text.fill('@video')
            expect(page.locator('.w-reference-popup')).to_be_visible()
            page.evaluate("p=>window.postMessage({type:'t8-director:init',project:p},location.origin)", project)
            expect(page.locator('.w-reference-popup')).not_to_be_visible()
            # Promote an eleventh image: earlier aliases change digit lengths,
            # but the query replacement must still use the original caret slices.
            more_ids = [str(uuid.uuid4()) for _ in range(11)]
            project['assets'] = [dict(id=aid, name='末图.png' if i == 10 else f'图{i+1}.png', kind='image', width=1, height=1, sha256='a'*64)
                                 for i, aid in enumerate(more_ids)]
            project['doc']['sharedRefs'] = []
            project['doc']['shots'][0].update(tray=more_ids, refs=more_ids, selected=more_ids[0])
            page.evaluate("p=>window.postMessage({type:'t8-director:init',project:p},location.origin)", project)
            expect(page.locator('[data-thumbs] [data-tray-id]')).to_have_count(11)
            page.locator('[data-global-zone]').evaluate('el=>el.open=true')
            global_text.fill('@image10 before @末图 after @image1')
            global_text.evaluate("el=>{const pos=el.value.indexOf(' after');el.setSelectionRange(pos,pos);el.dispatchEvent(new InputEvent('input',{bubbles:true}));}")
            page.locator('[data-reference-choice]').click()
            expect(global_text).to_have_value('@image11 before @image1 after @image2')
            assert global_text.evaluate('el=>el.selectionStart') == len('@image11 before @image1')
            # Typing after programmatic insertion forms a new undo step.
            global_text.press_sequentially('继续')
            page.locator('.w-header-actions > .w-menu > summary').click()
            page.locator('[data-action="undo"]').click()
            expect(global_text).to_have_value('@image11 before @image1 after @image2')
            assert not errors and not submits
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
