"""All Director window geometry, using real Chromium and installed UI sources.

These are CSS-viewport tests, not an OS/browser-zoom or screen-reader claim.
No Core, queue, model or user browser is touched.
"""
import pytest
from playwright.sync_api import expect

import test_director_audit20_modal_browser as modal_browser

audit_page = modal_browser.audit_page


WINDOWS = (
    '[data-dialog]', '[data-model-zone]', '[data-film-dialog]',
    '[data-compare-dialog]', '[data-bundle-dialog]', '[data-creation-dialog]',
    '[data-storyboard-dialog]', '[data-frame-dialog]', '[data-trim-dialog]',
    '[data-task-drawer]',
)


def assert_centered(dialog):
    geometry = dialog.evaluate("""d=>{
        const r=d.getBoundingClientRect(),s=getComputedStyle(d);
        return {left:r.left,top:r.top,right:r.right,bottom:r.bottom,
            width:r.width,height:r.height,vw:innerWidth,vh:innerHeight,
            position:s.position,overflow:s.overflowY};
    }""")
    assert geometry['position'] == 'fixed', geometry
    assert abs((geometry['left'] + geometry['right']) / 2 - geometry['vw'] / 2) <= 1, geometry
    assert abs((geometry['top'] + geometry['bottom']) / 2 - geometry['vh'] / 2) <= 1, geometry
    assert geometry['left'] >= 12 and geometry['right'] <= geometry['vw'] - 12, geometry
    assert geometry['top'] >= 12 and geometry['bottom'] <= geometry['vh'] - 12, geometry
    assert geometry['overflow'] in ('auto', 'scroll'), geometry


@pytest.mark.parametrize(('width', 'height'), [
    (2560, 1440), (1920, 1080), (1366, 768), (1280, 720),
    (1093, 614), (911, 512), (390, 640), (320, 480),
])
def test_all_ten_windows_center_short_long_and_after_resize(audit_page, width, height):
    page = audit_page
    page.set_viewport_size({'width': width, 'height': height})
    # Fail when a newly added dialog has not been included in this inventory.
    expect(page.locator('#t8-obsidian-lab dialog')).to_have_count(len(WINDOWS))
    for selector in WINDOWS:
        # Copy production markup/styles, without lifecycle observers that correctly
        # reject a direct showModal() lacking an owner/project transaction. Real
        # entry-point/lifecycle checks below and in the other suites remain intact.
        page.locator(selector).evaluate("""d=>{
            const copy=d.cloneNode(true);copy.id='modal-layout-copy';
            copy.removeAttribute('open');d.parentElement.append(copy);
        }""")
        dialog = page.locator('#modal-layout-copy')
        for tall in (False, True):
            page.set_viewport_size({'width': width, 'height': height})
            # The filler isolates height/scroll behavior from network-dependent data.
            dialog.evaluate("""(d,tall)=>{
                const probe=document.createElement('section');probe.dataset.layoutProbe='';
                probe.innerHTML='<p>窗口内容</p><div></div><button>底部操作</button>';
                probe.querySelector('div').style.height=(tall?2200:60)+'px';
                d.append(probe);
                if(d.matches('[data-task-drawer]'))d.show();else d.showModal();
                d.scrollTop=0;
            }""", tall)
            assert_centered(dialog)
            if tall:
                assert dialog.evaluate('d=>d.scrollHeight>d.clientHeight')
                bottom = dialog.locator('[data-layout-probe] button')
                bottom.scroll_into_view_if_needed()
                bounds = bottom.bounding_box()
                assert bounds and bounds['y'] >= 0 and bounds['y'] + bounds['height'] <= height
                assert_centered(dialog)
            # An open dialog must re-center without being closed/reopened.
            page.set_viewport_size({'width': max(320, width - 100), 'height': max(400, height - 80)})
            assert_centered(dialog)
            dialog.evaluate("d=>{d.close();d.querySelector('[data-layout-probe]').remove()}")
        dialog.evaluate('d=>d.remove()')


def test_real_prompt_modal_ignores_opener_and_page_scroll(audit_page):
    page = audit_page
    page.set_viewport_size({'width': 640, 'height': 800})
    prompt = page.locator('[data-field="simplePrompt"]')
    prompt.fill('居中检查，取消后原稿保持。')
    opener = page.get_by_role('button', name='放大编辑完整提示词', exact=True)
    for _ in range(2):
        opener.scroll_into_view_if_needed()
        before = page.evaluate('()=>scrollY')
        opener.click()
        dialog = page.locator('[data-dialog]')
        expect(dialog).to_be_visible()
        assert_centered(dialog)
        assert dialog.get_attribute('style') in (None, '')
        assert page.evaluate('()=>scrollY') == before
        page.keyboard.press('Escape')
        expect(dialog).not_to_be_visible()
        expect(opener).to_be_focused()
        expect(prompt).to_have_value('居中检查，取消后原稿保持。')


def test_real_model_and_task_windows_use_same_center_without_modalizing_tasks(audit_page):
    page = audit_page
    page.locator('.w-header-actions [data-action="model-settings"]').click()
    model = page.locator('[data-model-zone]')
    expect(model).to_be_visible()
    assert_centered(model)
    page.locator('[data-sampling-mode="two_pass"]').click()
    assert_centered(model)
    page.locator('[data-action="cancel-model-settings"]').click()
    page.get_by_role('button', name='任务列表', exact=True).click()
    task = page.locator('[data-task-drawer]')
    expect(task).to_be_visible()
    assert_centered(task)
    assert not task.evaluate('d=>d.matches(":modal")')
    page.get_by_role('button', name='关闭任务列表', exact=True).click()
    expect(task).not_to_be_visible()
