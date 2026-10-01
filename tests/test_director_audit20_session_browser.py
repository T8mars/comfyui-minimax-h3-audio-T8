"""Actual UI / browser storage, with isolated HTTP substitutes (no Core or GPU)."""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import pytest
from playwright.sync_api import sync_playwright, expect

ROOT = Path(__file__).resolve().parents[1]
DRAFT = "() => JSON.parse(localStorage.getItem('t8director.draft:'+sessionStorage.getItem('t8director.tab')))"
TAB = "sessionStorage.getItem('t8director.tab')"


@pytest.fixture
def director_page():
    class Quiet(SimpleHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def guess_type(self, path):
            return 'text/javascript' if path.endswith('.mjs') else super().guess_type(path)

    server = ThreadingHTTPServer(('127.0.0.1', 0), partial(Quiet, directory=str(ROOT/'web/director')))
    Thread(target=server.serve_forever, daemon=True).start()
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            context = browser.new_context(viewport={'width': 1440, 'height': 900})
            context.route('**/models', lambda route: route.fulfill(json={}))
            context.route('**/results/**', lambda route: route.fulfill(json={'results': []}))
            context.route('**/projects/**', lambda route: route.fulfill(status=404, json={'error': 'no remote draft'}))
            page = context.new_page()
            yield page, context, f'http://127.0.0.1:{server.server_port}/index.html'
            browser.close()
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize('locks', [True, False])
def test_opener_draft_isolation_and_refresh_preserve_both_unsaved_edits(director_page, locks):
    page, context, url = director_page
    if not locks:
        context.add_init_script("Object.defineProperty(navigator,'locks',{value:undefined})")
    page.goto(url, wait_until='networkidle')
    editor = page.locator('[data-field="simplePrompt"]')
    editor.fill('Original unsaved A')
    editor.press('Tab')
    original = page.evaluate(TAB)
    with context.expect_page() as opened:
        page.evaluate('window.open(location.href,"_blank")')
    other = opened.value
    other.wait_for_load_state('networkidle')
    other_editor = other.locator('[data-field="simplePrompt"]')
    expect(other_editor).to_have_value('Original unsaved A')
    assert other.evaluate(TAB) != original
    other_editor.fill('Other unsaved B')
    other_editor.press('Tab')
    assert page.evaluate(DRAFT)['doc']['shots'][0]['simplePrompt'] == 'Original unsaved A'
    assert other.evaluate(DRAFT)['doc']['shots'][0]['simplePrompt'] == 'Other unsaved B'
    page.reload(wait_until='networkidle')
    expect(editor).to_have_value('Original unsaved A')
    if locks:
        assert page.evaluate(TAB) == original
    other.reload(wait_until='networkidle')
    expect(other_editor).to_have_value('Other unsaved B')
    # Independent and noopener contexts do not share the live draft namespace.
    control = context.new_page()
    control.goto(page.url, wait_until='networkidle')
    assert control.evaluate(TAB) not in [page.evaluate(TAB), other.evaluate(TAB)]
    control.close()
    with context.expect_page() as opened:
        page.evaluate('window.open(location.href,"_blank","noopener")')
    control = opened.value
    control.wait_for_load_state('networkidle')
    assert control.evaluate(TAB) not in [page.evaluate(TAB), other.evaluate(TAB)]
    control.close()


def test_bfcache_lifecycle_reclaims_before_accepting_edits(director_page):
    page, context, url = director_page
    page.goto(url, wait_until='networkidle')
    page.locator('[data-field="simplePrompt"]').fill('A before suspension')
    page.locator('[data-field="simplePrompt"]').press('Tab')
    old = page.evaluate(TAB)
    # Synthetic lifecycle is explicitly not a native BFCache qualification.
    page.evaluate("dispatchEvent(new PageTransitionEvent('pagehide',{persisted:true}))")
    with context.expect_page() as opened:
        page.evaluate('window.open(location.href,"_blank")')
    other = opened.value
    other.wait_for_load_state('networkidle')
    assert other.evaluate(TAB) == old
    other.locator('[data-field="simplePrompt"]').fill('B holds old key')
    other.locator('[data-field="simplePrompt"]').press('Tab')
    page.evaluate("dispatchEvent(new PageTransitionEvent('pageshow',{persisted:true}))")
    page.wait_for_function("() => !document.querySelector('[data-field=simplePrompt]').closest('[inert]')")
    assert page.evaluate(TAB) != other.evaluate(TAB)
    assert page.evaluate(DRAFT)['doc']['shots'][0]['simplePrompt'] == 'A before suspension'
    assert other.evaluate(DRAFT)['doc']['shots'][0]['simplePrompt'] == 'B holds old key'


def test_save_during_generation_and_lost_response_retry(director_page):
    page, context, url = director_page
    submissions, saves = [], []
    lost, running = True, True

    def generate(route):
        nonlocal lost
        submissions.append(route.request.post_data_json)
        if lost:
            lost = False
            route.abort('failed')
        else:
            route.fulfill(json={'prompt_id': 'original', 'recipe': 'isolated-test'})

    def save(route):
        body = route.request.post_data_json
        saves.append(body)
        route.fulfill(json={'revision': body['expected_revision']+1})

    context.route('**/generate', generate)
    context.route('**/projects/**', save)
    context.route('**/jobs/original', lambda route: route.fulfill(json={'state': 'running' if running else 'error'}))
    page.goto(url, wait_until='networkidle')
    editor = page.locator('[data-field="simplePrompt"]')
    editor.fill('Frozen original input')
    editor.press('Tab')
    page.locator('[data-action="generate"]').click()
    expect(page.locator('[data-notice]')).to_contain_text('Failed to fetch')
    page.locator('[data-service="save"]').click()
    expect(page.locator('[data-save]')).to_contain_text('已真实保存')
    editor.fill('New editing after save')
    editor.press('Tab')
    with page.expect_response(lambda r: '/jobs/original' in r.url):
        page.locator('[data-action="generate"]').click()
    assert len(submissions) == 2
    assert submissions[0] == submissions[1]
    dialog = page.locator('[data-dialog]')
    if dialog.is_visible():
        page.locator('[data-dialog] [data-action="close"]').click()
    editor.fill('Further editing while running')
    editor.press('Tab')
    page.locator('[data-service="save"]').click()
    expect(page.locator('[data-save]')).to_contain_text('版本 2')
    assert len(saves) == 2
    assert saves[-1]['project']['doc']['shots'][0]['simplePrompt'] == 'Further editing while running'
    assert submissions[0]['project']['doc']['shots'][0]['simplePrompt'] == 'Frozen original input'
    running = False


@pytest.mark.parametrize('kind', ['sessionStorage', 'localStorage'])
def test_storage_security_error_keeps_editor_and_remote_save_usable(director_page, kind):
    page, context, url = director_page
    errors, saved = [], []
    page.on('pageerror', lambda e: errors.append(str(e)))
    context.add_init_script(f"Object.defineProperty(window,'{kind}',{{get(){{throw new DOMException('restricted','SecurityError')}}}})")

    def save(route):
        saved.append(route.request.post_data_json)
        route.fulfill(json={'revision': 1})

    context.route('**/projects/**', save)
    page.goto(url, wait_until='networkidle')
    editor = page.locator('[data-field="simplePrompt"]')
    editor.fill('Remote save without browser storage')
    editor.press('Tab')
    page.locator('[data-service="save"]').click()
    expect(page.locator('[data-save]')).to_contain_text('已真实保存')
    assert len(saved) == 1
    assert not errors
