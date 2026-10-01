"""Regression for post-acceptance audit, isolated UI, no user Core or queue."""

import pytest
import json
from pathlib import Path
from playwright.sync_api import expect
import test_director_audit20_modal_browser as modal_browser

audit_page = modal_browser.audit_page


@pytest.mark.parametrize("modal", ["prompt", "model"])
def test_modal_button_undo_redo_does_not_mutate_background(audit_page, modal):
    page = audit_page
    text = page.locator('[data-field="simplePrompt"]')
    text.fill("A")
    text.press("Tab")
    text.fill("B")
    text.press("Tab")
    if modal == "prompt":
        page.get_by_role("button", name="放大编辑完整提示词", exact=True).click()
        page.locator("[data-editor]").fill("C")
        button = page.locator('[data-action="editor-save"]')
        close = page.locator('[data-dialog-body] [data-action="close"]')
    else:
        page.locator('.w-header-actions [data-action="model-settings"]').click()
        button = close = page.locator('[data-action="cancel-model-settings"]')
    button.focus()
    button.press("Control+z")
    expect(text).to_have_value("B")
    button.press("Control+Shift+z")
    expect(text).to_have_value("B")
    close.click()
    expect(text).to_have_value("B")


@pytest.mark.parametrize("no_locks", [False, True])
def test_prompt_refresh_recovery_is_explicit_and_cancel_discards(audit_page, no_locks):
    page = audit_page
    text = page.locator('[data-field="simplePrompt"]')
    text.fill("Saved original")
    text.press("Tab")
    opener = page.get_by_role("button", name="放大编辑完整提示词", exact=True)
    opener.click()
    page.locator("[data-editor]").fill("Unsaved long draft")
    if no_locks:
        page.add_init_script(
            "Object.defineProperty(navigator,'locks',{value:undefined})"
        )
    page.reload()
    opener.click()
    expect(page.locator("[data-editor]")).to_have_value("Saved original")
    page.locator("[data-editor-restore]").click()
    expect(page.locator("[data-editor]")).to_have_value("Unsaved long draft")
    expect(text).to_have_value("Saved original")
    page.locator('[data-dialog-body] [data-action="close"]').click()
    opener.click()
    expect(page.locator("[data-editor-restore]")).to_have_count(0)
    expect(page.locator("[data-editor]")).to_have_value("Saved original")


def test_narrow_header_and_content_sized_task_window(audit_page):
    page = audit_page
    page.set_viewport_size({"width": 320, "height": 480})
    bounds = page.locator(".w-back").bounding_box()
    assert bounds["x"] >= 0 and bounds["x"] + bounds["width"] <= 320
    page.set_viewport_size({"width": 1366, "height": 768})
    page.get_by_role("button", name="任务列表", exact=True).click()
    bounds = page.locator("[data-task-drawer]").bounding_box()
    assert bounds["height"] < 400
    assert abs(bounds["y"] + bounds["height"] / 2 - 384) <= 1


def test_quota_recovery_downloads_exact_current_project_before_switch(audit_page):
    page = audit_page
    page.locator('[data-field="simplePrompt"]').fill("Current unsaved project")
    page.locator('[data-field="simplePrompt"]').press("Tab")
    original = page.evaluate(
        "()=>JSON.parse(localStorage.getItem('t8director.draft:'+sessionStorage.getItem('t8director.tab')))"
    )
    target = json.loads(json.dumps(original))
    target.update(id="bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb", title="Target project")
    target["doc"]["shots"][0]["simplePrompt"] = "Target text"
    page.route("**/projects", lambda r: r.fulfill(json={"projects": [target]}))
    page.route("**/projects/" + target["id"], lambda r: r.fulfill(json=target))
    page.evaluate(
        "()=>{Storage.prototype.setItem=function(){throw new DOMException('full','QuotaExceededError')}}"
    )
    page.on("dialog", lambda d: d.accept())
    opener = page.locator('[data-service="open"]')
    opener.evaluate("b=>b.closest('details').open=true")
    opener.click()
    page.locator("[data-open-project]").click()
    expect(page.locator('[data-service="backup-continue"]')).to_be_disabled()
    expect(page.locator('[data-field="simplePrompt"]')).to_have_value(
        "Current unsaved project"
    )
    with page.expect_download() as download_event:
        page.locator('[data-service="backup-export"]').click()
    downloaded = json.loads(
        Path(download_event.value.path()).read_text(encoding="utf-8")
    )
    assert downloaded == original
    page.locator('[data-service="backup-continue"]').click()
    expect(page.locator('[data-field="simplePrompt"]')).to_have_value("Target text")


def test_long_model_dialog_keeps_actions_visible(audit_page):
    page = audit_page
    page.set_viewport_size({"width": 911, "height": 512})
    page.locator('.w-header-actions [data-action="model-settings"]').click()
    page.locator('[data-sampling-mode="two_pass"]').click()
    for _ in range(8):
        page.locator('[data-sampling-action="add"][data-stage="low_loras"]').click()
    dialog = page.locator("[data-model-zone]")
    dialog.evaluate(
        "d=>{d.scrollTop=d.scrollHeight;const body=d.querySelector('[data-model-settings]');body.scrollTop=body.scrollHeight}"
    )
    bounds = dialog.bounding_box()
    for selector in (
        '[data-action="apply-model-settings"]',
        '[data-action="cancel-model-settings"]',
    ):
        button = page.locator(selector).bounding_box()
        assert button["y"] >= bounds["y"]
        assert button["y"] + button["height"] <= bounds["y"] + bounds["height"]
