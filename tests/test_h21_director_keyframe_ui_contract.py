"""Independent owned Chromium edge sequences, no user tabs/server or GPU.

Reuse the actual registered-PNG / temporary-port fixture. Upload transport
receipts are explicitly intercepted; production DOM/XHR/history run for real.
"""
from copy import deepcopy
import uuid

from playwright.sync_api import expect
import pytest

import test_director_keyframes_browser as native_browser
from test_director_keyframes_browser import (
    click_hidden_menu, initialize, open_editor, snapshot,
)

scene = native_browser.scene
keyframe_page = native_browser.keyframe_page


@pytest.mark.usefixtures("keyframe_page")
def test_advanced_keyframe_controls_remain_open_during_a_two_field_edit(keyframe_page):
    page, _, _, _, _ = keyframe_page
    open_editor(page)
    advanced = page.locator('[data-middle-row] > details')
    advanced.locator(':scope > summary').click()
    value = page.locator('[data-middle-key="visual_noise_aug"]')
    value.fill('0.95')
    value.press('Tab')
    # A real rerender must not collapse the immediately adjacent second field.
    expect(page.locator('[data-middle-row] > details')).to_have_attribute('open', '')
    page.locator('[data-middle-key="resize_mode"]').select_option('stretch')
    actual = snapshot(page)['doc']['shots'][0]['middleKeyframes']['items'][0]
    assert actual['visual_noise_aug'] == .95 and actual['resize_mode'] == 'stretch'


def test_replacing_middle_image_marks_original_alias_missing_instead_of_redirecting_it(keyframe_page):
    page, _, _, project, _ = keyframe_page
    original = snapshot(page)
    old_middle = project['assets'][2]['id']
    assert old_middle in original['aliasMaps'][original['current']].values()
    open_editor(page)
    page.locator('[data-middle-key="asset_id"]').select_option(project['assets'][0]['id'])
    after = snapshot(page)
    assert '@missing_image3' in after['doc']['shots'][0]['simplePrompt']
    assert '@image1' not in after['doc']['shots'][0]['simplePrompt']
    click_hidden_menu(page, '[data-action="undo"]')
    restored = snapshot(page)
    assert restored['doc'] == original['doc'] and restored['aliasMaps'] == original['aliasMaps']


@pytest.mark.parametrize('change', ['shot_switched', 'shot_duplicated', 'project_round_trip'])
def test_deferred_upload_keeps_original_project_epoch_shot_and_row(keyframe_page, change):
    page, state, store, project, replacement = keyframe_page
    open_editor(page)
    state['hold_upload'] = True
    with page.expect_file_chooser() as chooser:
        page.locator('[data-keyframe-action="upload"]').click()
    with page.expect_request('**/assets'):
        chooser.value.set_files(str(store.input_root / project['assets'][2]['server_path']))
    page.wait_for_function("document.querySelector('[data-upload-cancel]')!==null")
    assert state['pending'] is not None
    original_id = project['doc']['shots'][0]['id']
    original_row_id = project['doc']['shots'][0]['middleKeyframes']['items'][0]['id']
    if change == 'shot_switched':
        click_hidden_menu(page, '[data-action="new"]')
    elif change == 'shot_duplicated':
        click_hidden_menu(page, '[data-action="duplicate"]')
    else:
        other = deepcopy(project)
        other['id'] = str(uuid.uuid4())
        other['doc']['shots'][0]['simplePrompt'] = '另一个工程'
        initialize(page, other)
        initialize(page, project)
    before_reply = snapshot(page)
    state['pending'].fulfill(json=replacement)
    expect(page.locator('[data-notice]')).to_contain_text(
        '项目已切换' if change == 'project_round_trip' else '已上传 1 项')
    result = snapshot(page)
    if change == 'project_round_trip':
        assert result['doc'] == before_reply['doc']
        assert result['id'] == before_reply['id'] == project['id']
        assert replacement['id'] not in {item['id'] for item in result['assets']}
    else:
        original = next(shot for shot in result['doc']['shots'] if shot['id'] == original_id)
        assert original['middleKeyframes']['items'][0]['id'] == original_row_id
        assert original['middleKeyframes']['items'][0]['asset_id'] == replacement['id']
        current = next(shot for shot in result['doc']['shots'] if shot['id'] == result['current'])
        assert current['id'] != original_id
        assert current.get('middleKeyframes') == next(
            shot for shot in before_reply['doc']['shots'] if shot['id'] == current['id']).get('middleKeyframes')
    assert not state['submissions']


def test_outer_fold_state_is_per_shot_and_survives_source_mode_round_trip(keyframe_page):
    page, _, _, project, _ = keyframe_page
    open_editor(page)
    original_id = project['doc']['shots'][0]['id']
    click_hidden_menu(page, '[data-action="duplicate"]')
    if not page.locator('[data-workbench-section="source"]').is_visible():
        page.locator('[data-workbench-panel="source"]').click()
    assert page.locator('[data-middle-editor]').get_attribute('open') is None
    page.locator(f'[data-shot="{original_id}"]').click()
    # Switching shots rerenders the source settings but preserves this shot's
    # already expanded editor, without expanding the duplicate automatically.
    assert page.locator('[data-middle-editor]').get_attribute('open') is not None
    page.locator('[data-mode="text"]').click()
    assert page.locator('[data-middle-editor]').get_attribute('open') is not None
    page.locator('[data-mode="ends"]').click()
    assert page.locator('[data-middle-editor]').get_attribute('open') is not None
