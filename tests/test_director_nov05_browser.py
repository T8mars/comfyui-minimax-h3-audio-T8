"""Real isolated browser/storage/download combination; no Core queue or GPU."""
from copy import deepcopy
import json
import uuid

from playwright.sync_api import expect
from test_director_audit20_session_browser import director_page, DRAFT, TAB  # noqa: F401


def test_two_tabs_late_compile_new_shot_and_literal_project_export(director_page, tmp_path):  # noqa: F811
    page, context, url = director_page
    pending = []
    context.route('**/compile', lambda route: pending.append(route))
    page.goto(url, wait_until='networkidle')
    project = page.evaluate(DRAFT)
    first = project['doc']['shots'][0]
    second = deepcopy(first)
    second.update(id=str(uuid.uuid4()), name='Second', simplePrompt='second draft')
    project['doc']['shots'].append(second)
    page.evaluate("p=>window.postMessage({type:'t8-director:init',project:p},location.origin)", project)
    expect(page.locator('[data-shot]')).to_have_count(2)
    page.locator('.w-header-actions > .w-menu > summary').click()
    page.locator('.w-tools-menu > summary').click()
    with page.expect_request('**/compile'):
        page.locator('.w-tools-menu [data-action="check"]').click()
    assert len(pending) == 1
    with context.expect_page() as opened:
        page.evaluate('window.open(location.href,"_blank")')
    other = opened.value
    other.wait_for_load_state('networkidle')
    assert other.evaluate(TAB) != page.evaluate(TAB)
    other_editor = other.locator('[data-field="simplePrompt"]')
    other_editor.fill('other tab preserves its own first shot')
    other_editor.press('Tab')
    page.locator(f'[data-shot="{second["id"]}"]').click()
    literal = '<d>[Chinese]原句  两个空格。</d>\n第二行'
    editor = page.locator('[data-field="simplePrompt"]')
    editor.fill(literal)
    editor.press('Tab')
    pending.pop().fulfill(json={'ready': True, 'shots': [{'id': first['id'],
        'canvas': {'width': 64, 'height': 32, 'preprocessing': []}}]})
    expect(editor).to_have_value(literal)
    expect(page.locator('[data-dialog]')).not_to_be_visible()
    page.locator('.w-header-actions > .w-menu > summary').click()
    with page.expect_download() as downloaded:
        page.locator('[data-service="project"]').click()
    path = tmp_path/'project-export.json'
    downloaded.value.save_as(path)
    exported = json.loads(path.read_text(encoding='utf-8'))
    assert exported['current'] == second['id']
    assert exported['doc']['shots'][1]['simplePrompt'] == literal
    assert other.evaluate(DRAFT)['doc']['shots'][0]['simplePrompt'] == 'other tab preserves its own first shot'
    page.reload(wait_until='networkidle')
    expect(page.locator('[data-field="simplePrompt"]')).to_have_value(literal)
    assert other.evaluate(TAB) != page.evaluate(TAB)
