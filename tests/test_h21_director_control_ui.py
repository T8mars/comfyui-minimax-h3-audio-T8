"""Owned headless UI: real project/media state, no user tabs or GPU queue."""
from copy import deepcopy
import json
from pathlib import Path
import uuid

from playwright.sync_api import BrowserType, expect
import pytest

from h3_audio_t8_pkg.director_controls import normalize_controls
from h3_audio_t8_pkg.director_project import compile_project, validate_project
from test_director_keyframes_browser import (
    initialize, snapshot, click_hidden_menu, keyframe_page as shared_page,
)
from test_director_keyframes import scene as shared_scene


scene = shared_scene
keyframe_page = shared_page
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def owned_control_chromium(request, monkeypatch):
    binary = request.config.getoption('--director-chromium', default=None)
    if not binary:
        return  # Playwright may use an already installed default, never downloads.
    assert Path(binary).is_file(), 'Select an existing owned Chromium; no browser download.'
    launch = BrowserType.launch

    def selected_launch(self, *args, **kwargs):
        if self.name == 'chromium':
            kwargs.setdefault('executable_path', binary)
        return launch(self, *args, **kwargs)

    monkeypatch.setattr(BrowserType, 'launch', selected_launch)


def setup_control(page, project, replacement, *, enabled=True):
    value = deepcopy(project)
    value['assets'].append(deepcopy(replacement))
    value['doc']['shots'][0]['controlRegions'] = normalize_controls({
        'version': 1, 'items': [{'id': str(uuid.uuid4()), 'asset_id': replacement['id'],
                                'control_net_name': 'user-custom-control.safetensors',
                                'enabled': enabled}],
    })
    initialize(page, value)
    expect(page.locator('[data-control-row]')).to_have_count(1)
    return value


def open_control(page, advanced=False):
    source = page.locator('[data-workbench-panel="source"]')
    if source.get_attribute('aria-expanded') != 'true':
        source.click()
    detail = page.locator('[data-control-editor]')
    if detail.get_attribute('open') is None:
        detail.locator(':scope > summary').click()
    if advanced:
        item = page.locator('[data-control-advanced]').first
        if item.get_attribute('open') is None:
            item.locator(':scope > summary').click()
    return detail


def row_snapshot(page):
    return snapshot(page)['doc']['shots'][0]['controlRegions']['items'][0]


def test_control_optional_folded_no_alias_and_unknown_model_selectable(keyframe_page):
    page, state, _, project, replacement = keyframe_page
    page.locator('[data-workbench-panel="source"]').click()
    empty = page.locator('[data-control-editor]')
    assert empty.get_attribute('open') is None
    assert 'controlRegions' not in snapshot(page)['doc']['shots'][0]
    source = setup_control(page, project, replacement)
    assert empty.get_attribute('open') is None
    detail = open_control(page)
    expect(detail).to_contain_text('已经预处理')
    expect(detail.locator('[data-control-key="control_net_name"]')).to_have_value('user-custom-control.safetensors')
    detail.locator('[data-control-key="control_net_name"]').fill('my-unlisted-v2.safetensors')
    detail.locator('[data-control-key="control_net_name"]').press('Tab')
    value = snapshot(page)
    validate_project(value)
    assert value['doc']['shots'][0]['controlRegions']['items'][0]['control_net_name'] == 'my-unlisted-v2.safetensors'
    assert value['aliasMaps'] == snapshot_aliases(source)
    assert replacement['id'] not in value['doc']['shots'][0]['tray']
    assert not state['submissions']


def snapshot_aliases(project):
    # The three ordinary image roles remain the original reference sequence.
    shot = project['doc']['shots'][0]
    return {shot['id']: {'@image1': shot['first'], '@image2': shot['last'],
                         '@image3': shot['middleKeyframes']['items'][0]['asset_id']}}


def test_control_drag_live_preview_one_undo_and_cancel_no_commit(keyframe_page):
    page, state, _, project, replacement = keyframe_page
    setup_control(page, project, replacement)
    open_control(page, advanced=True)
    plane = page.locator('[data-control-plane="region"]')
    plane.scroll_into_view_if_needed()
    box = plane.bounding_box()
    start = (box['x']+box['width']*.1, box['y']+box['height']*.2)
    end = (box['x']+box['width']*.7, box['y']+box['height']*.8)
    original = row_snapshot(page)
    page.mouse.move(*start)
    page.mouse.down()
    page.mouse.move(*end, steps=8)
    live = plane.locator('[data-control-selection]')
    expect(live).to_be_visible()
    assert live.bounding_box()['width'] > box['width']*.5
    # A real browser pointercancel restores the original shape without history.
    plane.dispatch_event('pointercancel', {'pointerId': 1})
    page.mouse.up()
    assert row_snapshot(page) == original
    page.mouse.move(*start)
    page.mouse.down()
    page.mouse.move(*end, steps=8)
    page.mouse.up()
    edited = row_snapshot(page)
    assert edited['region']['mode'] == 'rect'
    assert abs(edited['region']['xyxy_normalized'][0]-.1) < .015
    assert abs(edited['region']['xyxy_normalized'][2]-.7) < .015
    click_hidden_menu(page, '[data-action="undo"]')
    assert row_snapshot(page) == original  # The entire drag was one transaction.
    assert not state['submissions']


def test_source_original_aspect_crop_exterior_and_target_independent(keyframe_page):
    page, state, _, project, replacement = keyframe_page
    setup_control(page, project, replacement)
    open_control(page, advanced=True)
    target = page.locator('[data-control-plane="region"]')
    target.scroll_into_view_if_needed()
    assert abs(target.bounding_box()['width']/target.bounding_box()['height']-16/9) < .02
    page.locator('[data-control-action="tab-crop"]').click()
    source = page.locator('[data-control-plane="crop"]')
    source.scroll_into_view_if_needed()
    image = source.locator('img')
    page.wait_for_function("document.querySelector('[data-control-plane] img').naturalWidth>0")
    assert image.evaluate('e=>getComputedStyle(e).objectFit') == 'contain'
    assert image.evaluate('e=>[e.naturalWidth,e.naturalHeight]') == [replacement['width'], replacement['height']]
    for i, value in enumerate([-.3, -.2, 1.4, 1.1]):
        element = page.locator(f'[data-control-key="crop.xyxy_normalized.{i}"]')
        element.fill(str(value))
        element.press('Tab')
    page.locator('[data-control-key="crop.enabled"]').check()
    edited = row_snapshot(page)
    assert edited['crop']['xyxy_normalized'] == [-.3, -.2, 1.4, 1.1]
    assert edited['region'] == {'mode': 'full', 'xyxy_normalized': [0, 0, 1, 1]}
    page.screenshot(path=str(ROOT/'artifacts/development/h21-implementation-20261021/c-control-source.png'))
    page.locator('[data-control-action="tab-region"]').click()
    expect(page.locator('[data-control-key="region.xyxy_normalized.0"]')).to_have_value('0')
    page.screenshot(path=str(ROOT/'artifacts/development/h21-implementation-20261021/c-control-editor.png'))
    assert not state['submissions']


def test_unrelated_pointer_cancel_does_not_cancel_primary_drag(keyframe_page):
    page, state, _, project, replacement = keyframe_page
    setup_control(page, project, replacement)
    detail = open_control(page, advanced=True)
    expect(detail.locator('[data-control-canvas-status]')).to_contain_text('服务端实际画布')
    plane = page.locator('[data-control-plane="region"]')
    plane.scroll_into_view_if_needed()
    box = plane.bounding_box()
    page.mouse.move(box['x']+box['width']*.2, box['y']+box['height']*.2)
    page.mouse.down()
    page.mouse.move(box['x']+box['width']*.8, box['y']+box['height']*.8, steps=5)
    plane.dispatch_event('pointercancel', {'pointerId': 99})
    expect(plane.locator('[data-control-selection]')).to_be_visible()
    page.mouse.up()
    assert row_snapshot(page)['region']['mode'] == 'rect'
    click_hidden_menu(page, '[data-action="undo"]')
    assert row_snapshot(page)['region']['mode'] == 'full'
    assert not state['submissions']


def test_canvas_reply_during_drag_applies_actual_ratio_after_cancel(keyframe_page):
    page, state, store, project, replacement = keyframe_page
    setup_control(page, project, replacement)
    pending = {}

    def delayed_canvas(route):
        body = route.request.post_data_json
        pending['report'] = compile_project(body['project'], store, shot_id=body['shot_id'])
        pending['route'] = route

    page.route('**/compile', delayed_canvas)
    detail = open_control(page, advanced=True)
    plane = detail.locator('[data-control-plane="region"]')
    plane.scroll_into_view_if_needed()
    assert pending.get('route') is not None
    box = plane.bounding_box()
    page.mouse.move(box['x']+box['width']*.1, box['y']+box['height']*.1)
    page.mouse.down()
    page.mouse.move(box['x']+box['width']*.8, box['y']+box['height']*.8, steps=5)
    with page.expect_response('**/compile'):
        pending['route'].fulfill(json=pending['report'])
    page.evaluate('()=>new Promise(requestAnimationFrame)')
    plane.dispatch_event('pointercancel', {'pointerId': 1})
    page.mouse.up()
    canvas = pending['report']['shots'][0]['canvas']
    expect(detail.locator('[data-control-canvas-status]')).to_contain_text(f'{canvas["width"]} × {canvas["height"]} · 服务端实际画布')
    actual = plane.bounding_box()
    assert abs(actual['width']/actual['height']-canvas['width']/canvas['height']) < .001
    assert row_snapshot(page)['region']['mode'] == 'full'
    assert not state['submissions']


def test_multiple_controls_duplicate_and_undo_preserve_unique_row_ids(keyframe_page):
    page, state, _, project, replacement = keyframe_page
    source = setup_control(page, project, replacement)
    open_control(page)
    page.locator('[data-control-action="add"]').click()
    expect(page.locator('[data-control-row]')).to_have_count(2)
    page.locator('[data-control-row]').nth(1).locator('[data-control-key="enabled"]').uncheck()
    saved = snapshot(page)
    original_rows = saved['doc']['shots'][0]['controlRegions']['items']
    assert len({r['id'] for r in original_rows}) == 2
    click_hidden_menu(page, '[data-action="duplicate"]')
    cloned = snapshot(page)
    rows = cloned['doc']['shots'][1]['controlRegions']['items']
    assert len(rows) == 2 and not {r['id'] for r in rows} & {r['id'] for r in original_rows}
    assert rows[0]['asset_id'] == replacement['id']
    assert cloned['doc']['shots'][0]['controlRegions']['items'] == original_rows
    assert source['doc']['shots'][0]['simplePrompt'] == cloned['doc']['shots'][0]['simplePrompt']
    click_hidden_menu(page, '[data-action="undo"]')
    assert len(snapshot(page)['doc']['shots']) == 1
    assert not state['submissions']


def test_bad_control_version_import_keeps_current_project(keyframe_page):
    page, state, _, project, replacement = keyframe_page
    setup_control(page, project, replacement)
    before = snapshot(page)
    invalid = deepcopy(before)
    invalid['doc']['shots'][0]['controlRegions']['version'] = 2
    initialize_error = json.dumps(invalid)
    page.evaluate("text=>window.postMessage({type:'t8-director:init',project:JSON.parse(text)},location.origin)", initialize_error)
    expect(page.locator('[data-notice]')).to_contain_text('局部控制版本')
    assert snapshot(page)['doc'] == before['doc']
    assert not state['submissions']


def test_template_real_dialog_keeps_controls_and_renews_ids_once(keyframe_page):
    page, state, _, project, replacement = keyframe_page
    source = setup_control(page, project, replacement, enabled=False)
    click_hidden_menu(page, '[data-creation-open]')
    page.locator('[data-creation-kind]').select_option('template')
    page.locator('[data-creation-name]').fill('局部控制模板')
    page.locator('[data-creation-save]').click()
    expect(page.locator('[data-creation-status]')).to_contain_text('已收藏')
    page.locator('[data-creation-use]').click()
    page.locator('[data-creation-commit]').click()
    saved = snapshot(page)
    validate_project(saved)
    old = source['doc']['shots'][0]['controlRegions']['items'][0]
    copied = saved['doc']['shots'][1]['controlRegions']['items'][0]
    assert copied['id'] != old['id']
    assert {k: v for k, v in copied.items() if k != 'id'} == {k: v for k, v in old.items() if k != 'id'}
    assert replacement['id'] not in saved['aliasMaps'][saved['current']].values()
    click_hidden_menu(page, '[data-action="undo"]')
    undone = snapshot(page)
    assert len(undone['doc']['shots']) == 1
    assert len(undone['doc']['creationLibrary']['items']) == 1
    assert not state['submissions']


@pytest.mark.parametrize('changed', ['row_removed', 'slot_replaced', 'project_switched'])
def test_control_upload_late_owner_never_replaces_new_edit(keyframe_page, changed):
    page, state, store, project, replacement = keyframe_page
    setup_control(page, project, replacement)
    open_control(page)
    state['hold_upload'] = True
    with page.expect_file_chooser() as chooser:
        page.locator('[data-control-action="upload"]').click()
    with page.expect_request('**/assets'):
        chooser.value.set_files(str(store.input_root/project['assets'][2]['server_path']))
    page.wait_for_function("document.querySelector('[data-upload-cancel]')!==null")
    assert state['pending'] is not None
    if changed == 'row_removed':
        page.locator('[data-control-action="remove"]').click()
    elif changed == 'slot_replaced':
        page.locator('[data-control-key="asset_id"]').select_option(project['assets'][0]['id'])
    else:
        other = deepcopy(project)
        other['id'] = str(uuid.uuid4())
        other['doc']['shots'][0]['simplePrompt'] = '另一工程保持原稿'
        initialize(page, other)
    before = snapshot(page)
    state['pending'].fulfill(json=replacement)
    expect(page.locator('[data-notice]')).to_contain_text('项目已切换' if changed == 'project_switched' else '没有覆盖新编辑')
    after = snapshot(page)
    assert after['doc'] == before['doc'] and after['id'] == before['id']
    if changed == 'project_switched':
        assert replacement['id'] not in {a['id'] for a in after['assets']}
    assert not state['submissions']


def test_control_real_upload_no_alias_and_delete_keeps_draft(keyframe_page):
    page, state, store, project, replacement = keyframe_page
    source = deepcopy(project)
    source['doc']['shots'][0]['controlRegions'] = normalize_controls({
        'version': 1, 'items': [{'id': str(uuid.uuid4()), 'enabled': False}],
    })
    initialize(page, source)
    expect(page.locator('[data-control-row]')).to_have_count(1)
    open_control(page)
    with page.expect_file_chooser() as chooser:
        page.locator('[data-control-action="upload"]').click()
    chooser.value.set_files(str(store.input_root/project['assets'][2]['server_path']))
    expect(page.locator('[data-control-key="asset_id"]')).to_have_value(replacement['id'])
    uploaded = snapshot(page)
    assert replacement['id'] not in uploaded['aliasMaps'][uploaded['current']].values()
    assert replacement['id'] not in uploaded['doc']['shots'][0]['tray']
    click_hidden_menu(page, f'[data-delete-asset="{replacement["id"]}"]')
    page.locator('[data-confirm-delete]').click()
    deleted = row_snapshot(page)
    assert deleted['id'] == source['doc']['shots'][0]['controlRegions']['items'][0]['id']
    assert deleted['asset_id'] is None
    click_hidden_menu(page, '[data-action="undo"]')
    assert row_snapshot(page)['asset_id'] == replacement['id']
    assert not state['submissions']


def test_resume_tail_low_only_incomplete_draft_is_inert_but_high_still_checked(keyframe_page):
    page, state, _, project, _ = keyframe_page
    source = deepcopy(project)
    source['doc']['sampling'] = {
        'mode': 'hyperflow', 'variant': 'continuous4plus4separate',
        'hyperflow_file': 'hyperflow/my-custom-adapter.safetensors', 'output_mp': .4,
        'stage_checkpoint': {'mode': 'resume_tail', 'artifact_path': 'frozen/head.safetensors',
                             'artifact_sha256': 'a'*64},
    }
    source['doc']['shots'][0]['controlRegions'] = normalize_controls({
        'version': 1, 'items': [{'id': str(uuid.uuid4()), 'asset_id': None,
                                'control_net_name': '', 'stages': {'low': True, 'high': False}}],
    })
    initialize(page, source)
    expect(page.locator('[data-control-row]')).to_have_count(1)
    detail = open_control(page, advanced=True)
    expect(detail.locator('.c-status')).to_contain_text('草稿保留')
    expect(detail.locator('.c-issues')).to_have_text('')
    expect(detail.locator('.c-stages')).to_contain_text('仅 HIGH 生效')
    before = row_snapshot(page)
    detail.locator('[data-control-key="stages.high"]').check()
    expect(detail.locator('.c-status')).to_contain_text('参与生成')
    expect(detail.locator('.c-issues')).to_contain_text('请选择 Fun Control 模型')
    expect(detail.locator('.c-issues')).to_contain_text('已预处理的控制图片')
    after = row_snapshot(page)
    assert after['asset_id'] is None and after['control_net_name'] == ''
    assert after['stages']['high'] and before['stages'] == {'low': True, 'high': False}
    assert not state['submissions']


def test_catalog_refresh_only_updates_options_not_uncommitted_input(keyframe_page):
    page, state, _, _, _ = keyframe_page
    actual = page.evaluate("""async()=>{
        const {createControlEditor}=await import('./ui-modules/control_ui.mjs');
        const {newControlRow}=await import('./ui-modules/control_state.mjs');
        const host=document.createElement('div');document.body.append(host);
        const shot={id:crypto.randomUUID(),ratio:'16:9',mode:'text',refs:[],controlRegions:{version:1,items:[newControlRow(crypto.randomUUID())]}};
        let catalog={},changes=0;
        const editor=createControlEditor({root:host,shot:()=>shot,doc:()=>({sampling:{mode:'single'}}),assets:()=>new Map(),catalog:()=>catalog,
            mutate:()=>{changes++},uuid:()=>crypto.randomUUID(),escape:v=>String(v??''),upload(){},notify(){}});
        editor.render(host);
        const field=host.querySelector('[data-control-key="control_net_name"]');field.value='正在输入尚未失焦.safetensors';
        catalog={fun_control_names:[{value:'new-control.safetensors',label:'New model'}]};editor.refreshCatalog();
        const result={value:field.value,sameField:field===host.querySelector('[data-control-key="control_net_name"]'),changes,options:[...host.querySelectorAll('datalist option')].map(e=>e.value)};
        editor.remember();host.remove();return result;
    }""")
    assert actual == {'value': '正在输入尚未失焦.safetensors', 'sameField': True,
                      'changes': 0, 'options': ['new-control.safetensors']}
    assert not state['submissions']
