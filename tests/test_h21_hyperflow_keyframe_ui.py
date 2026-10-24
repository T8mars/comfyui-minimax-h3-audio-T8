"""Owned browser recording+middle draft path, not training-weight acceptance."""
from copy import deepcopy

from playwright.sync_api import expect

import test_director_keyframes_browser as browser_cases
from test_director_keyframes_browser import initialize, open_editor, snapshot, ROOT
from test_director_generation import _audio
from h3_audio_t8_pkg.director_project import compile_project

scene = browser_cases.scene
keyframe_page = browser_cases.keyframe_page


def test_recording_middle_editor_preflight_and_saved_draft_agree(keyframe_page):
    page, state, store, project, _ = keyframe_page
    recording = _audio(store, seconds=4.)
    working = deepcopy(project)
    working["assets"].append(recording)
    shot = working["doc"]["shots"][0]
    shot.update(mode="first", last=None, sound="record", audio=recording["id"], start=0., end=4.,
        simplePrompt="The person speaks naturally.")
    initialize(page, working)
    editor = open_editor(page)
    expect(editor).to_contain_text("原音驱动")
    expect(page.locator('[data-middle-row]')).to_have_count(1)
    expect(page.locator('[data-middle-hint]')).to_contain_text("帧 90")
    saved = snapshot(page)
    report = compile_project(saved, store, shot_id=shot["id"])
    assert report["ready"], report["errors"]
    assert report["shots"][0]["audio_mode"] == "lock_source"
    current = saved["doc"]["shots"][0]
    assert current["last"] is None and current["first"] == shot["first"]
    assert current["middleKeyframes"] == shot["middleKeyframes"]
    assert current["sound"] == "record" and current["audio"] == recording["id"]
    assert not state["submissions"]
    page.screenshot(path=str(ROOT / 'artifacts/development/h21-implementation-20261021/hf-recording-middle-ui.png'))
