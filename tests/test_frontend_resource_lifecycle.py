from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_skin_finish_preview_releases_listeners_images_and_render_closure():
    source = (ROOT / "web" / "skin_finish_preview.js").read_text(encoding="utf-8")
    assert "_t8SkinFinishPreviewCleanup" in source
    assert source.count('removeEventListener("input"') == 1
    assert source.count('removeEventListener("click"') == 1
    assert source.count('removeEventListener("pointerdown"') == 1
    assert source.count('removeEventListener("wheel"') == 1
    assert 'sourceImage.removeAttribute("src")' in source
    assert 'candidateImage.removeAttribute("src")' in source
    assert "node._t8SkinFinishPreviewRender = null" in source
    assert "nodeType.prototype.onRemoved" in source


def test_studio_timeline_releases_dom_and_render_closure_on_removal():
    source = (ROOT / "web" / "studio_timeline_v2.js").read_text(encoding="utf-8")
    assert "_t8TimelineCleanup" in source
    assert "track.replaceChildren()" in source
    assert "rows.replaceChildren()" in source
    assert "root.replaceChildren()" in source
    assert "node._t8TimelineRender = null" in source
    assert "nodeType.prototype.onRemoved" in source
