"""Owned Chromium and real project/media compiler; never use user tabs or queue.

HTTP upload responses are registered real PNG fixtures, not a multipart-ingest
qualification. Deferred replies exercise the production XHR/owner transaction.
"""
from copy import deepcopy
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from threading import Thread
from urllib.parse import urlsplit
import uuid

from playwright.sync_api import sync_playwright, expect
import pytest

from h3_audio_t8_pkg.director_project import compile_project, validate_project
from h3_audio_t8_pkg.director_routes import _DIRECTOR_UI_MODULES
from test_director_d1 import asset
from test_director_keyframes import scene as shared_scene


ROOT = Path(__file__).resolve().parents[1]
PREFIX = '/minimax_h3_t8/director'
scene = shared_scene  # Register the shared real-PNG pytest fixture in this module.


@pytest.fixture
def keyframe_page(scene):
    store, project, shot, images = scene

    class Quiet(SimpleHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def guess_type(self, path):
            return 'text/javascript' if path.endswith('.mjs') else super().guess_type(path)

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == PREFIX+'/ui':
                self.path = '/index.html'
            elif path.startswith(PREFIX+'/ui-modules/'):
                name = path.rsplit('/', 1)[-1]
                if name not in _DIRECTOR_UI_MODULES:
                    return self.send_error(404)
                self.path = '/'+name
            elif path in {PREFIX+'/'+name for name in ('session.mjs', 'sampling_ui.mjs', 'radar_state.mjs', 'keyframes_state.mjs', 'control_state.mjs')}:
                self.path = '/'+path.rsplit('/', 1)[-1]
            return super().do_GET()

    server = ThreadingHTTPServer(('127.0.0.1', 0), partial(Quiet, directory=str(ROOT/'web/director')))
    Thread(target=server.serve_forever, daemon=True).start()
    state = dict(errors=[], network=[], compilations=[], submissions=[], hold_upload=False, pending=None, old_ack=False)
    replacement = asset(store, name='replacement.png')
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True, args=['--disable-gpu'])
            page = browser.new_page(viewport={'width': 1440, 'height': 900}, accept_downloads=True)
            page.on('pageerror', lambda error: state['errors'].append(str(error)))
            page.on('requestfailed', lambda request: state['network'].append((request.url, request.failure)))
            page.route('**/models', lambda route: route.fulfill(json={}))
            page.route('**/results/**', lambda route: route.fulfill(json={'results': []}))
            page.route('**/batch-features', lambda route: route.fulfill(json={'selection_version': 2}))

            def compile_draft(route):
                body = route.request.post_data_json
                report = compile_project(body['project'], store, shot_id=body.get('shot_id'))
                state['compilations'].append((deepcopy(body), deepcopy(report)))
                if state['old_ack']:
                    report.pop('extensions', None)
                route.fulfill(json=report)

            def serve_image(route):
                aid = urlsplit(route.request.url).path.rsplit('/', 1)[-1]
                path = store.input_root/store.asset(aid)['server_path']
                route.fulfill(body=path.read_bytes(), content_type='image/png')

            def upload(route):
                if state['hold_upload']:
                    state['pending'] = route
                else:
                    route.fulfill(json=replacement)

            def no_queue(route):
                state['submissions'].append(route.request.url)
                route.fulfill(status=400, json={'error': 'No real queue in browser test', 'submission_state': 'not_submitted'})

            page.route('**/compile', compile_draft)
            page.route('**/assets/*', serve_image)
            page.route('**/assets', upload)
            page.route('**/generate', no_queue)
            page.route('**/batches', no_queue)
            page.goto(f'http://127.0.0.1:{server.server_port}{PREFIX}/ui', wait_until='networkidle')
            try:
                expect(page.locator('[data-field="simplePrompt"]')).to_be_visible()
            except AssertionError as error:
                raise AssertionError({'errors': state['errors'], 'network': state['network']}) from error
            initialize(page, project)
            expect(page.locator('[data-workbench-heading]')).to_contain_text('01')
            expect(page.locator('[data-middle-row]')).to_have_count(1)
            yield page, state, store, project, replacement
            assert not state['errors'], state['errors']
            browser.close()
    finally:
        server.shutdown()
        server.server_close()


def initialize(page, project):
    page.evaluate("p=>window.postMessage({type:'t8-director:init',project:p},location.origin)", project)
    expect(page.locator('[data-field="simplePrompt"]')).to_have_value(project['doc']['shots'][0]['simplePrompt'])


def click_hidden_menu(page, selector):
    button = page.locator(selector)
    button.evaluate("e=>{for(let p=e.parentElement;p;p=p.parentElement)if(p.tagName==='DETAILS')p.open=true}")
    button.click()


def snapshot(page):
    with page.expect_download() as download:
        click_hidden_menu(page, '[data-service="project"]')
    return json.loads(Path(download.value.path()).read_bytes())


def open_editor(page):
    page.locator('[data-workbench-panel="source"]').click()
    detail = page.locator('[data-middle-editor]')
    if detail.get_attribute('open') is None:
        detail.locator(':scope > summary').click()
    return detail


def test_optional_folded_editor_real_native_module_paths_and_full_image_preview(keyframe_page):
    page, state, _, project, _ = keyframe_page
    page.locator('[data-workbench-panel="source"]').click()
    detail = page.locator('[data-middle-editor]')
    assert detail.get_attribute('open') is None
    detail.locator(':scope > summary').click()
    expect(page.locator('[data-middle-hint]')).to_contain_text('帧 90 · 3.750 秒')
    image = detail.locator('img')
    page.wait_for_function("document.querySelector('.k-image img').naturalWidth>0")
    assert image.evaluate("e=>getComputedStyle(e).objectFit") == 'contain'
    assert image.evaluate('e=>e.naturalWidth') == 512
    row = page.locator('[data-middle-row]')
    row.locator('[data-middle-key="position"]').fill('50')
    row.locator('[data-middle-key="position"]').press('Tab')
    expect(page.locator('[data-middle-hint]')).to_contain_text('帧 48')
    assert page.locator('[data-middle-editor]').get_attribute('open') is not None
    saved = snapshot(page)
    validate_project(saved)
    assert saved['doc']['shots'][0]['middleKeyframes']['items'][0]['position'] == 50
    assert saved['doc']['shots'][0]['simplePrompt'] == project['doc']['shots'][0]['simplePrompt']
    assert saved['aliasMaps'][saved['current']]['@image3'] == project['assets'][2]['id']
    assert not state['submissions']
    page.screenshot(path=str(ROOT/'artifacts/development/h21-implementation-20261021/d-keyframe-editor.png'))


def test_add_disable_collision_invalid_numeric_and_undo_preserve_explicit_requests(keyframe_page):
    page, state, _, project, _ = keyframe_page
    open_editor(page)
    page.locator('[data-keyframe-action="add"]').click()
    expect(page.locator('[data-middle-row]')).to_have_count(2)
    second = page.locator('[data-middle-row]').nth(1)
    second.locator('[data-middle-key="asset_id"]').select_option(project['assets'][2]['id'])
    second.locator('[data-middle-key="position"]').fill('95')
    second.locator('[data-middle-key="position"]').press('Tab')
    expect(page.locator('[data-middle-hint]').first).to_contain_text('与另一项同帧')
    second.locator('[data-middle-key="enabled"]').uncheck()
    expect(second.locator('[data-middle-hint]')).to_contain_text('已停用')
    before = snapshot(page)
    second.locator('[data-middle-key="position"]').fill('')
    second.locator('[data-middle-key="position"]').dispatch_event('change')
    expect(page.locator('[data-notice]')).to_contain_text('尚未保存')
    assert snapshot(page)['doc'] == before['doc']
    click_hidden_menu(page, '[data-action="undo"]')
    expect(page.locator('[data-middle-row]').nth(1).locator('[data-middle-key="enabled"]')).to_be_checked()
    assert snapshot(page)['doc']['shots'][0]['middleKeyframes']['items'][1]['position'] == 95
    assert not state['submissions']


def test_duplicate_row_ids_remove_asset_and_undo_are_real_editor_operations(keyframe_page):
    page, _, _, project, _ = keyframe_page
    source = snapshot(page)['doc']['shots'][0]
    click_hidden_menu(page, '[data-action="duplicate"]')
    copy = snapshot(page)
    assert len(copy['doc']['shots']) == 2
    rows = [s['middleKeyframes']['items'][0] for s in copy['doc']['shots']]
    assert rows[0]['id'] == source['middleKeyframes']['items'][0]['id']
    assert rows[1]['id'] != rows[0]['id'] and rows[1]['asset_id'] == rows[0]['asset_id']
    mid = project['assets'][2]['id']
    page.locator(f'[data-tray-id="{mid}"] [data-preview]').click()
    click_hidden_menu(page, f'[data-delete-asset="{mid}"]')
    page.locator('[data-confirm-delete]').click()
    deleted = snapshot(page)
    assert len(deleted['doc']['shots'][0]['middleKeyframes']['items']) == 1
    assert deleted['doc']['shots'][1]['middleKeyframes']['items'] == []
    assert any(a['id'] == mid for a in deleted['assets'])
    click_hidden_menu(page, '[data-action="undo"]')
    assert snapshot(page)['doc']['shots'][1]['middleKeyframes']['items'] == [rows[1]]


@pytest.mark.parametrize('changed', ['row_removed', 'slot_replaced', 'project_switched'])
def test_late_upload_does_not_overwrite_new_row_or_new_project(keyframe_page, changed):
    page, state, store, project, replacement = keyframe_page
    open_editor(page)
    state['hold_upload'] = True
    with page.expect_file_chooser() as chooser:
        page.locator('[data-keyframe-action="upload"]').click()
    with page.expect_request('**/assets'):
        chooser.value.set_files(str(store.input_root/project['assets'][2]['server_path']))
    page.wait_for_function("document.querySelector('[data-upload-cancel]')!==null")
    assert state['pending'] is not None
    if changed == 'row_removed':
        page.locator('[data-keyframe-action="remove"]').click()
    elif changed == 'slot_replaced':
        page.locator('[data-middle-key="asset_id"]').select_option(project['assets'][0]['id'])
    else:
        other = deepcopy(project)
        other['id'] = str(uuid.uuid4())
        other['doc']['shots'][0]['simplePrompt'] = '另一个工程'
        initialize(page, other)
    expected = snapshot(page)
    state['pending'].fulfill(json=replacement)
    expect(page.locator('[data-notice]')).to_contain_text('项目已切换' if changed == 'project_switched' else '没有覆盖新编辑')
    actual = snapshot(page)
    assert actual['doc'] == expected['doc'] and actual['id'] == expected['id']
    if changed == 'project_switched':
        assert replacement['id'] not in {a['id'] for a in actual['assets']}
    assert not state['submissions']


@pytest.mark.parametrize('all_shots', [False, True])
def test_old_server_protocol_cannot_queue_while_ignoring_keyframes(keyframe_page, all_shots):
    page, state, _, _, _ = keyframe_page
    state['old_ack'] = True
    if all_shots:
        click_hidden_menu(page, '[data-action="generate-all"]')
    else:
        page.locator('[data-action="generate"]').click()
    expect(page.locator('[data-notice]')).to_contain_text('未确认中间关键帧协议')
    assert state['compilations'] and not state['submissions']
    assert len(snapshot(page)['doc']['shots'][0]['middleKeyframes']['items']) == 1


@pytest.mark.parametrize('fault', ['future_version', 'bad_items', 'nonfinite_position'])
def test_unknown_middle_version_from_node_snapshot_preserves_current_project(keyframe_page, fault):
    page, _, _, project, _ = keyframe_page
    before = snapshot(page)
    unsupported = deepcopy(project)
    unsupported['id'] = str(uuid.uuid4())
    unsupported['doc']['shots'][0]['simplePrompt'] = '不应覆盖当前稿'
    middle = unsupported['doc']['shots'][0]['middleKeyframes']
    if fault == 'future_version':
        middle['version'] = 99
    elif fault == 'bad_items':
        middle['items'] = {'not': 'a list'}
    else:
        middle['items'][0]['position'] = None
    page.evaluate("p=>window.postMessage({type:'t8-director:init',project:p},location.origin)", unsupported)
    expect(page.locator('[data-notice]')).to_contain_text('节点快照未载入')
    after = snapshot(page)
    assert after['id'] == before['id'] and after['doc'] == before['doc']


def test_advanced_row_keeps_open_after_committing_noise_value(keyframe_page):
    page, _, _, _, _ = keyframe_page
    open_editor(page)
    page.evaluate("""()=>{
        window.middleToggleTrace=[];
        document.querySelector('#t8-obsidian-lab').addEventListener('toggle',e=>{
            if(e.target.tagName==='DETAILS')window.middleToggleTrace.push({
                advanced:e.target.dataset.middleAdvanced,outer:e.target.hasAttribute('data-middle-editor'),
                owner:e.target.closest('[data-middle-owner]')?.dataset.middleOwner,open:e.target.open});
        },true);
    }""")
    page.locator('[data-middle-advanced] > summary').click()
    page.locator('[data-middle-key="visual_noise_aug"]').fill('0.95')
    page.locator('[data-middle-key="visual_noise_aug"]').press('Tab')
    try:
        expect(page.locator('[data-middle-advanced]')).to_have_attribute('open', '')
    except AssertionError as error:
        raise AssertionError(page.evaluate('window.middleToggleTrace')) from error
