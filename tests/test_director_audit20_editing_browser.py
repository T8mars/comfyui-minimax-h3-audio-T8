"""Audit D08-D11 regressions: real isolated browser and CPU-decoded media."""
from contextlib import contextmanager
from copy import deepcopy
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
import uuid

from playwright.sync_api import expect, sync_playwright

from h3_audio_t8_pkg.director_film import prepare_film
from h3_audio_t8_pkg.director_project import compile_project, file_sha
from test_director_film import setup

ROOT = Path(__file__).resolve().parents[1]
DRAFT = "() => JSON.parse(localStorage.getItem('t8director.draft:'+sessionStorage.getItem('t8director.tab')))"


@contextmanager
def editing_browser(project, records):
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
            page = browser.new_page(viewport={'width': 1440, 'height': 960})
            errors = []
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.route('**/models', lambda route: route.fulfill(json={}))
            page.route('**/results/**', lambda route: route.fulfill(json={'results': records}))
            page.goto(f'http://127.0.0.1:{server.server_port}/index.html', wait_until='domcontentloaded')
            page.wait_for_selector('.w-shot-actions')
            page.wait_for_function("!document.querySelector('#t8-obsidian-lab').inert")
            page.evaluate("project=>window.postMessage({type:'t8-director:init',project},location.origin)", project)
            expect(page.locator('.o-shotbar [data-shot]')).to_have_count(len(project['doc']['shots']))
            yield page
            assert not errors
            browser.close()
    finally:
        server.shutdown()
        server.server_close()


def test_integer_event_frames_and_legacy_invalid_seconds_are_not_rounded(tmp_path):
    _, _, project, records = setup(tmp_path)
    shot = project['doc']['shots'][0]
    shot.update(writingMode='advanced', events=[{'id': str(uuid.uuid4()), 'start': 0, 'end': 1/24, 'text': 'hello'}])
    with editing_browser(project, records) as page:
        page.locator('[data-events]').evaluate('e=>e.parentElement.open=true')
        end = page.locator('[data-event="0"][data-key="end"]')
        expect(end).to_have_value('1')
        assert end.get_attribute('step') == '1'
        checks = page.locator('[data-event-frame]').evaluate_all('rows=>rows.map(e=>({value:e.value,valid:e.checkValidity(),message:e.validationMessage}))')
        assert all(row['valid'] for row in checks), checks
        page.locator('[data-action="event"]').click()
        after = page.evaluate(DRAFT)['doc']['shots'][0]
        assert after['events'][1]['start'] == 1/24
        assert after['events'][1]['end'] == 49/24
        compiled = compile_project(page.evaluate(DRAFT), shot_id=shot['id'])
        assert compiled['ready'], compiled['errors']
        assert compiled['shots'][0]['events'][1]['end_frame'] == 49
        end.focus()
        end.press('ArrowUp')
        end.press('Tab')
        after = page.evaluate(DRAFT)['doc']['shots'][0]
        assert after['events'][0]['end'] == 2/24
        assert after['events'][0]['end']*24 == 2
        before = deepcopy(after)
        end.fill('2.5')
        end.press('Tab')
        assert page.evaluate(DRAFT)['doc']['shots'][0] == before
        expect(end.locator('..').locator('[data-event-seconds]')).to_contain_text('原时间尚未更改')
        page.locator('[data-action="event"]').click()
        assert page.evaluate(DRAFT)['doc']['shots'][0] == before

    bad = deepcopy(project)
    bad['doc']['shots'][0]['events'][0]['end'] = 0.1
    with editing_browser(bad, records) as page:
        page.locator('[data-events]').evaluate('e=>e.parentElement.open=true')
        end = page.locator('[data-event="0"][data-key="end"]')
        expect(end).to_have_value('')
        expect(end.locator('..').locator('[data-event-seconds]')).to_contain_text('未自动舍入')
        page.locator('[data-action="event"]').click()
        assert len(page.evaluate(DRAFT)['doc']['shots'][0]['events']) == 1
        assert page.evaluate(DRAFT)['doc']['shots'][0]['events'][0]['end'] == 0.1
        end.fill('3')
        end.press('Tab')
        assert page.evaluate(DRAFT)['doc']['shots'][0]['events'][0]['end'] == 3/24


def test_trim_real_media_bounds_reset_copy_adopt_and_stale_context(tmp_path):
    store, output, project, records = setup(tmp_path)
    records.append({**deepcopy(records[0]), 'prompt_id': 'replacement-version'})
    first_id = project['doc']['shots'][0]['id']
    project['doc']['shots'][0]['filmTrim'] = {'in_frame': 2, 'out_frame': 10}
    media_sha = file_sha(output/'0.mp4')
    with editing_browser(project, records) as page:
        prepares = []

        def prepare(route):
            payload = route.request.post_data_json['project']
            prepares.append(payload)
            route.fulfill(json=prepare_film(store, payload, records, output))

        page.route('**/films/prepare', prepare)

        def open_trim():
            page.locator('[data-view="output"]').click()
            page.locator('[data-trim-open]').click()
            expect(page.locator('[data-trim-in]')).to_be_visible()

        open_trim()
        expect(page.locator('[data-trim-status]')).to_contain_text('12 帧 · 24 fps')
        expect(page.locator('[data-trim-in]')).to_have_value('2')
        expect(page.locator('[data-trim-out]')).to_have_value('10')
        assert 'filmTrim' not in prepares[-1]['doc']['shots'][0]
        for invalid in ('13', '2', '2.5', ''):
            page.locator('[data-trim-out]').fill(invalid)
            expect(page.locator('[data-trim-apply]')).to_be_disabled()
        page.locator('[data-trim-out]').fill('8')
        expect(page.locator('[data-trim-seconds]')).to_contain_text('共 6 帧')
        page.locator('[data-trim-apply]').click()
        assert page.evaluate(DRAFT)['doc']['shots'][0]['filmTrim'] == {'in_frame': 2, 'out_frame': 8}
        expect(page.locator('[data-trim-summary]')).to_contain_text('2–8 帧')
        page.locator('.w-header-actions > .w-menu > summary').click()
        page.locator('[data-action="undo"]').click()
        assert page.evaluate(DRAFT)['doc']['shots'][0]['filmTrim'] == {'in_frame': 2, 'out_frame': 10}
        page.locator('.w-header-actions > .w-menu > summary').click()
        page.locator('[data-action="redo"]').click()
        assert page.evaluate(DRAFT)['doc']['shots'][0]['filmTrim'] == {'in_frame': 2, 'out_frame': 8}
        for width in (1440, 390):
            page.set_viewport_size({'width': width, 'height': 960})
            open_trim()
            assert page.locator('[data-trim-dialog]').evaluate('d=>d.scrollWidth<=d.clientWidth')
            page.locator('[data-trim-close]').click()
        page.set_viewport_size({'width': 1440, 'height': 960})
        page.locator('.w-shot-actions .w-menu > summary').click()
        page.locator('[data-action="duplicate"]').click()
        copied = page.evaluate(DRAFT)['doc']['shots'][1]
        assert 'filmTrim' not in copied and 'adoptedResultId' not in copied
        page.locator(f'[data-shot="{first_id}"]').click()
        page.locator('[data-result-version]').select_option('replacement-version')
        page.locator('[data-adopt-version]').click()
        assert 'filmTrim' not in page.evaluate(DRAFT)['doc']['shots'][0]
        open_trim()
        expect(page.locator('[data-trim-out]')).to_have_value('12')
        page.locator('[data-trim-in]').fill('1')
        page.locator('[data-trim-apply]').click()
        open_trim()
        page.locator('[data-trim-reset]').click()
        assert 'filmTrim' not in page.evaluate(DRAFT)['doc']['shots'][0]

        # Pending metadata must never write its old project into a new project.
        pending = []
        page.unroute('**/films/prepare', prepare)
        page.route('**/films/prepare', lambda route: pending.append(route))
        page.locator('[data-trim-open]').click()
        page.wait_for_timeout(80)
        changed = deepcopy(project)
        changed['id'] = str(uuid.uuid4())
        changed['title'] = 'different project'
        page.evaluate("p=>window.postMessage({type:'t8-director:init',project:p},location.origin)", changed)
        page.wait_for_function('id=>JSON.parse(localStorage.getItem("t8director.draft:"+sessionStorage.getItem("t8director.tab"))).id===id', arg=changed['id'])
        for route in pending:
            route.fulfill(json=prepare_film(store, route.request.post_data_json['project'], records, output))
        expect(page.locator('[data-trim-dialog]')).not_to_be_visible()
        assert page.evaluate(DRAFT)['doc']['shots'][0]['filmTrim'] == {'in_frame': 2, 'out_frame': 10}
        assert file_sha(output/'0.mp4') == media_sha


def test_generic_menu_dialog_has_name_and_visible_return_focus(tmp_path):
    _, _, project, records = setup(tmp_path)
    with editing_browser(project, records) as page:
        summary = page.locator('.w-header-actions > .w-menu > summary')
        summary.click()
        page.locator('[data-action="library"]').click()
        dialog = page.locator('[data-dialog]')
        expect(dialog).to_be_visible()
        title = dialog.get_attribute('aria-labelledby')
        assert title and page.locator('#'+title).inner_text().startswith('素材库')
        page.keyboard.press('Escape')
        expect(summary).to_be_focused()
        # Existing expanded-editor explicit focus/caret restoration stays intact.
        prompt = page.locator('[data-field="simplePrompt"]')
        prompt.fill('unchanged prompt')
        prompt.press('Tab')
        prompt.locator('..').locator('[data-expand]').click()
        page.locator('[data-editor]').fill('saved prompt')
        page.locator('[data-action="editor-save"]').click()
        expect(prompt).to_be_focused()
        expect(prompt).to_have_value('saved prompt')
