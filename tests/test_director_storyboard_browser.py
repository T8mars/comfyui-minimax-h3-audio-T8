"""Explicit text import preview/confirm/undo through the real browser controls."""
from copy import deepcopy
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
import uuid

from playwright.sync_api import sync_playwright, expect

ROOT = Path(__file__).resolve().parents[1]


def test_storyboard_preview_does_not_mutate_and_import_is_one_undo(tmp_path):
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
            page.on('request', lambda request: writes.append(request.url) if request.method != 'GET' else None)
            page.route('**/models', lambda route: route.fulfill(json={}))
            page.route('**/results/**', lambda route: route.fulfill(json={'results': []}))
            page.goto(f'http://127.0.0.1:{server.server_port}/index.html', wait_until='networkidle')
            tiles = page.locator('.w-setting-tiles')
            assert tiles.evaluate("el=>getComputedStyle(el).gridTemplateColumns.split(' ').length") == 3
            page.set_viewport_size({'width': 1920, 'height': 900})
            assert tiles.evaluate("el=>getComputedStyle(el).gridTemplateColumns.split(' ').length") == 5
            page.set_viewport_size({'width': 1366, 'height': 900})
            draft_js = "() => JSON.parse(localStorage.getItem(Object.keys(localStorage).find(key=>key.startsWith('t8director.draft:'))))"
            original = page.evaluate(draft_js)
            source = '# <开场> | 4秒\n  人物微笑。\n\n# 回答 | 3.5s\n@image1 说话；<Audio 2>。'
            page.locator('.w-shot-actions summary').click()
            page.locator('[data-storyboard-open]').click()
            dialog = page.locator('[data-storyboard-dialog]')
            text = page.locator('[data-storyboard-text]')
            preview = page.locator('[data-storyboard-preview]')
            apply = page.locator('[data-storyboard-apply]')
            text.fill('不明确的文本')
            preview.click()
            expect(page.locator('[data-storyboard-state]')).to_contain_text('没有导入')
            expect(apply).to_be_disabled()
            assert page.evaluate(draft_js)['doc'] == original['doc']
            text.fill(source)
            preview.click()
            expect(page.locator('[data-storyboard-preview-body] article')).to_have_count(2)
            expect(page.locator('[data-storyboard-preview-body] h4').first).to_contain_text('<开场>')
            expect(page.locator('[data-storyboard-preview-body] pre').first).to_have_text('  人物微笑。', use_inner_text=False)
            expect(apply).to_be_disabled()
            page.locator('[data-storyboard-ack]').check()
            expect(apply).to_be_enabled()
            page.screenshot(path=str(tmp_path/'storyboard-preview.png'))
            assert page.evaluate(draft_js)['doc'] == original['doc']
            # Any input after preview invalidates confirmation.
            text.fill(source+'补充。')
            expect(apply).to_be_disabled()
            preview.click()
            page.locator('[data-storyboard-ack]').check()
            apply.click()
            expect(dialog).not_to_be_visible()
            expect(page.locator('.o-shotbar [data-shot]')).to_have_count(3)
            imported = page.evaluate(draft_js)
            assert imported['doc']['shots'][0] == original['doc']['shots'][0]
            assert imported['doc']['global'] == original['doc']['global']
            assert imported['doc']['generation'] == original['doc']['generation']
            one, two = imported['doc']['shots'][1:]
            assert one['name'] == '<开场>' and one['simplePrompt'] == '  人物微笑。'
            assert two['simplePrompt'] == '@missing_image1 说话；@missing_audio2。补充。'
            assert [one['duration'], two['duration']] == [4, 3.5]
            assert len({row['id'] for row in imported['doc']['shots']}) == 3
            assert all(row['writingMode'] == 'simple' and row['sound'] == 'native' and not row['autoDuration']
                       and row['samplingInherit'] and row['d3Inherit'] and not row['first'] and not row['audio']
                       and 'adoptedResultId' not in row for row in (one, two))
            page.locator(f'[data-shot="{two["id"]}"]').click()
            expect(page.locator('[data-draft-status]')).to_contain_text('待完善')
            page.locator('.w-header-actions > .w-menu > summary').click()
            page.locator('[data-action="undo"]').click()
            expect(page.locator('.o-shotbar [data-shot]')).to_have_count(1)
            assert page.evaluate(draft_js)['doc'] == original['doc']

            page.locator('.w-shot-actions summary').click()
            page.locator('[data-storyboard-open]').click()
            # File contents are read locally, not uploaded; valid preview then cancel.
            text.fill('')
            page.locator('[data-storyboard-file]').set_input_files({'name': '分镜.txt', 'mimeType': 'text/plain', 'buffer': '# 新镜 | 5秒\n正文'.encode()})
            expect(text).to_have_value('# 新镜 | 5秒\n正文')
            preview.click()
            expect(apply).to_be_enabled()
            page.locator('[data-storyboard-close]').click()
            assert page.evaluate(draft_js)['doc'] == original['doc']

            page.locator('.w-shot-actions summary').click()
            page.locator('[data-storyboard-open]').click()
            preview.click()
            changed = deepcopy(original)
            changed['id'] = str(uuid.uuid4())
            changed['doc']['global'] = '不同工程'
            page.evaluate("p=>window.postMessage({type:'t8-director:init',project:p},location.origin)", changed)
            apply.click()
            expect(page.locator('[data-storyboard-state]')).to_contain_text('工程已变化')
            expect(apply).to_be_disabled()
            assert len(page.evaluate(draft_js)['doc']['shots']) == 1
            assert not writes and not errors
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
