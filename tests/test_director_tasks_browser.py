"""Isolated real Chromium task drawer polling; mocked read-only Core API."""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
import uuid

from playwright.sync_api import sync_playwright, expect, Error

ROOT = Path(__file__).resolve().parents[1]


def test_task_drawer_polls_without_editing_or_submitting_and_discards_late_responses(tmp_path):
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
            errors, calls, posts, pending = [], [], [], []
            behavior = {'state': 'queued', 'fail': False, 'defer': False}
            records = []
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.on('request', lambda request: posts.append(request.url) if request.method != 'GET' else None)
            page.route('**/models', lambda route: route.fulfill(json={}))

            def results(route):
                calls.append(route.request.url)
                if behavior['defer']:
                    pending.append(route)
                elif behavior['fail']:
                    route.fulfill(status=503, json={'error': '测试断线'})
                else:
                    route.fulfill(json={'results': [dict(row, state=behavior['state']) for row in records]})

            page.route('**/results/**', results)
            page.goto(f'http://127.0.0.1:{server.server_port}/index.html', wait_until='networkidle')
            page.clock.install()
            draft_js = "() => JSON.parse(localStorage.getItem(Object.keys(localStorage).find(key=>key.startsWith('t8director.draft:'))))"
            project = page.evaluate(draft_js)
            records.append({'shot_id': project['current'], 'prompt_id': 'task-1', 'submitted_at': 1})
            button = page.locator('[data-service="task-list"]').first
            drawer = page.locator('[data-task-drawer]')
            text = page.locator('[data-field="simplePrompt"]')
            button.click()
            expect(drawer).to_be_visible()
            expect(page.locator('[data-task-content]')).to_contain_text('排队中')
            text.fill('后台查询不应覆盖这段草稿')
            text.evaluate("el=>el.dataset.identity='keep-this-field'")
            before = page.evaluate(draft_js)['doc']
            behavior['state'] = 'success'
            page.clock.fast_forward(5100)
            expect(page.locator('[data-task-content]')).to_contain_text('已完成')
            expect(text).to_be_focused()
            expect(text).to_have_attribute('data-identity', 'keep-this-field')
            assert page.evaluate(draft_js)['doc'] == before

            # No-op results should not replace the focused task link either.
            task_link = page.locator('[data-task-locate]').first
            task_link.focus()
            task_link.evaluate("el=>el.dataset.identity='keep-task-link'")
            page.clock.fast_forward(5100)
            expect(task_link).to_be_focused()
            expect(task_link).to_have_attribute('data-identity', 'keep-task-link')
            page.locator('[data-task-auto]').uncheck()
            count = len(calls)
            page.clock.fast_forward(16000)
            assert len(calls) == count

            behavior['fail'] = True
            page.locator('[data-task-refresh]').click()
            expect(page.locator('[data-task-state]')).to_contain_text('状态读取失败')
            expect(page.locator('[data-task-content]')).to_contain_text('已完成')
            behavior['fail'] = False
            page.locator('[data-task-refresh]').click()
            expect(page.locator('[data-task-state]')).to_contain_text('已更新')
            page.locator('[data-task-auto]').check()
            # All windows are now centered by design. The task window can cover
            # this pointer target; it stays non-modal, so keyboard activation of
            # the background editor must still work without losing task polling.
            expand = text.locator('..').locator('[data-expand]')
            expand.focus()
            expand.press('Enter')
            editor = page.locator('[data-editor]')
            editor.fill('还未保存的放大编辑稿')
            page.clock.fast_forward(5100)
            expect(editor).to_have_value('还未保存的放大编辑稿')
            expect(editor).to_be_focused()
            expect(page.locator('[data-dialog]')).to_be_visible()
            page.locator('[data-dialog-body] [data-action="close"]').click()
            page.locator('[data-task-auto]').uncheck()
            expect(text).to_have_value('后台查询不应覆盖这段草稿')
            for width in (640, 1366, 1920):
                page.set_viewport_size({'width': width, 'height': 900})
                assert drawer.evaluate('el=>{const r=el.getBoundingClientRect();return r.left>=12&&r.right<=innerWidth-12&&r.top>=12&&r.bottom<=innerHeight-12&&Math.abs(r.left+r.width/2-innerWidth/2)<1&&Math.abs(r.top+r.height/2-innerHeight/2)<1&&!el.matches(":modal")}')
            page.screenshot(path=str(tmp_path/'task-drawer.png'))

            # Closing during a pending read cancels it and cannot reopen the panel.
            behavior['defer'] = True
            page.locator('[data-task-refresh]').click()
            expect(page.locator('[data-task-refresh]')).to_be_disabled()
            page.locator('[data-task-close]').click()
            expect(drawer).not_to_be_visible()
            page.wait_for_timeout(50)
            assert pending
            for route in pending:
                try:
                    route.fulfill(json={'results': records})
                except Error:
                    pass  # Aborted fetches need not accept a response.
            pending.clear()
            count = len(calls)
            page.clock.fast_forward(16000)
            assert len(calls) == count
            expect(drawer).not_to_be_visible()

            behavior['defer'] = False
            button.click()
            expect(page.locator('[data-task-state]')).to_contain_text('已更新')
            # A different project epoch closes the old-project panel.
            project['id'] = str(uuid.uuid4())
            page.evaluate("p=>window.postMessage({type:'t8-director:init',project:p},location.origin)", project)
            expect(drawer).not_to_be_visible()
            assert not posts and not errors
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
