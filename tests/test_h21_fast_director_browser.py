"""Actual owned Chromium: Fast EXP selection and existing source state survive.

Real local HTML/modules and media fixtures; intercepted transport never queues
inference or touches the user's browser. No browser downloads.
"""

from copy import deepcopy
from pathlib import Path

from playwright.sync_api import BrowserType, expect
import pytest

from test_director_keyframes_browser import (
    initialize,
    keyframe_page as shared_page,
    snapshot,
)
from test_director_keyframes import scene  # noqa: F401 -- pytest fixture

keyframe_page = shared_page


@pytest.fixture(autouse=True)
def owned_chromium(request, monkeypatch):
    binary = request.config.getoption("--director-chromium", default=None)
    if not binary:
        return
    assert Path(binary).is_file(), "Choose an existing owned browser; never download."
    launch = BrowserType.launch

    def selected(self, *args, **kwargs):
        if self.name == "chromium":
            kwargs.setdefault("executable_path", binary)
        return launch(self, *args, **kwargs)

    monkeypatch.setattr(BrowserType, "launch", selected)


@pytest.mark.parametrize("mode", ("first", "ends", "refs"))
def test_fast_experiment_toggle_has_no_text_only_claim_and_preserves_nontext_project(
    keyframe_page, mode
):
    page, state, _, project, _ = keyframe_page
    value = deepcopy(project)
    shot = value["doc"]["shots"][0]
    shot.update(
        mode=mode,
        d3Inherit=False,
        d3={
            "fast_h3_v2": {
                "enabled": False,
                "profile": "dense_compat_exp",
                "min_tokens": 8192,
            }
        },
    )
    if mode == "refs":
        shot["refs"] = [value["assets"][0]["id"]]
    initialize(page, value)
    page.locator('[data-workbench-panel="d3"]').click()
    toggle = page.locator('[data-d3-toggle="fast_h3_v2"]')
    expect(toggle).to_be_enabled()
    expect(toggle.locator("..")).not_to_contain_text("仅文字")
    expect(page.locator('[data-workbench-section="d3"]')).to_contain_text("EXP")
    toggle.check()
    expect(page.locator('[data-workbench-section="d3"]')).to_contain_text(
        "非文字组合画质须单独验证"
    )
    if mode == "ends":
        page.screenshot(
            path=str(
                Path(__file__).resolve().parents[1]
                / "artifacts/development/h21-implementation-20261021/fast-director-policy-ui.png"
            )
        )
    expected = snapshot(page)
    selected = expected["doc"]["shots"][0]
    assert selected["d3"]["fast_h3_v2"] == {
        "enabled": True,
        "profile": "dense_compat_exp",
        "min_tokens": 8192,
    }
    assert selected["mode"] == mode and selected["first"] == shot["first"]
    assert (
        selected["last"] == shot["last"]
        and selected["middleKeyframes"] == shot["middleKeyframes"]
    )
    assert selected.get("refs", []) == shot.get("refs", [])
    initialize(page, expected)
    if (
        page.locator('[data-workbench-panel="d3"]').get_attribute("aria-expanded")
        != "true"
    ):
        page.locator('[data-workbench-panel="d3"]').click()
    expect(page.locator('[data-d3-toggle="fast_h3_v2"]')).to_be_checked()
    assert not state["submissions"]
