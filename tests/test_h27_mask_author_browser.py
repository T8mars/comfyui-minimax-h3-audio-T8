"""Owned headless UI: live strokes, frame isolation, undo, saved reopen and centering."""

from urllib.parse import urlsplit

from playwright.sync_api import expect

import test_director_captions_browser as cases
from h3_audio_t8_pkg import director_mask_author as masks
from h3_audio_t8_pkg.director_project import file_sha
from h3_audio_t8_pkg.director_film import film_media

caption_page = cases.caption_page


def connect(page, store):
    calls = []

    def api(route):
        parts = urlsplit(route.request.url).path.split("/")
        parts = parts[parts.index("films") + 1 :]
        owner, fid, kind = parts[:3]
        try:
            if kind == "mask-frames":
                result = masks.preview_frame(
                    store, owner, fid, int(parts[3]), int(parts[4])
                )
            else:
                index, uid = int(parts[3]), parts[4]
                calls.append((owner, fid, index, uid))
                if len(parts) > 5 and parts[5] == "export":
                    path = masks.export_author(store, owner, fid, index, uid)
                    result = {"ready": True, "author_sha256": path.stem}
                elif route.request.method == "POST":
                    result = masks.save_author(
                        store, owner, fid, index, uid, route.request.post_data_json
                    )
                else:
                    result = masks.load_author(store, owner, fid, index, uid)
            route.fulfill(json=result)
        except ValueError as error:
            route.fulfill(status=400, json={"error": str(error)})

    page.route("**/films/*/*/masks/**", api)
    page.route("**/films/*/*/mask-frames/**", api)
    page.locator("[data-film-mask]").click()
    expect(page.locator("[data-mask-canvas]")).to_have_attribute("aria-busy", "false")
    return calls


def alpha(page):
    return page.locator("[data-mask-canvas]").evaluate(
        "e=>{const a=e.getContext('2d').getImageData(0,0,e.width,e.height).data;let n=0;for(let i=3;i<a.length;i+=4)n+=a[i]>0;return n;}"
    )


def test_live_brush_save_explicit_copy_undo_and_reopen_keep_source(
    caption_page, tmp_path
):
    page, state, store, project = caption_page
    film = state["films"][0]
    media = film_media(store, project["id"], film["id"], 0)
    before = file_sha(media)
    calls = connect(page, store)
    canvas = page.locator("[data-mask-canvas]")
    box = canvas.bounding_box()
    page.mouse.move(box["x"] + box["width"] * 0.2, box["y"] + box["height"] * 0.3)
    page.mouse.down()
    page.mouse.move(
        box["x"] + box["width"] * 0.5, box["y"] + box["height"] * 0.5, steps=5
    )
    assert alpha(page) > 0  # Visible BEFORE pointerup, not a delayed rectangle.
    page.mouse.up()
    page.locator("[data-mask-save]").click()
    expect(page.locator("[data-mask-status]")).to_contain_text("已保存")
    page.locator("[data-mask-next]").click()
    expect(canvas).to_have_attribute("aria-busy", "false")
    assert alpha(page) == 0
    page.locator("[data-mask-prev]").click()
    expect(canvas).to_have_attribute("aria-busy", "false")
    painted = alpha(page)
    assert painted > 0
    page.locator("[data-mask-copy-start]").fill("2")
    page.locator("[data-mask-copy-end]").fill("2")
    page.locator("[data-mask-copy]").click()
    page.locator("[data-mask-undo]").click()
    page.locator("[data-mask-copy]").click()
    page.locator("[data-mask-export]").click()
    expect(page.locator("[data-mask-download]")).to_be_visible()
    assert "sha=" in page.locator("[data-mask-download]").get_attribute("href")
    owner, fid, index, uid = calls[-1]
    saved = masks.load_author(store, owner, fid, index, uid)
    assert (
        saved["strokes"]["0"] == saved["strokes"]["2"] and "1" not in saved["strokes"]
    )
    modal = page.locator("[data-mask-author]")
    bounds = modal.bounding_box()
    assert bounds["y"] >= 0 and bounds["y"] + bounds["height"] <= 768.1
    assert abs(bounds["y"] + bounds["height"] / 2 - 384) < 4
    page.screenshot(path=str(tmp_path / "mask-author.png"))
    page.locator("[data-mask-close]").click()
    expect(modal).not_to_be_visible()
    page.locator("[data-film-mask]").click()
    expect(canvas).to_have_attribute("aria-busy", "false")
    assert alpha(page) == painted and file_sha(media) == before
    assert state["submissions"] == [] and state["errors"] == []


def test_unsaved_close_cancel_clear_and_undo_are_explicit(caption_page):
    page, _, store, _ = caption_page
    connect(page, store)
    canvas = page.locator("[data-mask-canvas]")
    box = canvas.bounding_box()
    page.mouse.click(box["x"] + box["width"] * 0.5, box["y"] + box["height"] * 0.5)
    assert alpha(page) > 0
    page.once("dialog", lambda d: d.dismiss())
    page.locator("[data-mask-close]").click()
    expect(page.locator("[data-mask-author]")).to_be_visible()
    painted = alpha(page)
    page.locator("[data-mask-clear]").click()
    assert alpha(page) == 0
    page.locator("[data-mask-undo]").click()
    assert alpha(page) == painted
    page.locator("[data-mask-save]").click()
    expect(page.locator("[data-mask-status]")).to_contain_text("已保存")
    page.locator("[data-mask-close]").click()
    expect(page.locator("[data-mask-author]")).not_to_be_visible()


def test_fast_scrub_keeps_one_preview_in_flight_and_discards_old_reply(caption_page):
    page, _, store, _ = caption_page
    connect(page, store)
    pending = []

    def delayed(route):
        pending.append(route)

    page.route("**/films/*/*/mask-frames/**", delayed)
    slider = page.locator("[data-mask-frame]")
    slider.evaluate(
        "e=>{e.value='1';e.dispatchEvent(new Event('input',{bubbles:true}));}"
    )
    page.wait_for_function(
        '() => document.querySelector("[data-mask-canvas]").getAttribute("aria-busy")==="true"'
    )
    # Let the first request reach the endpoint; later requests must be coalesced.
    for _ in range(50):
        if pending:
            break
        page.wait_for_timeout(10)
    assert len(pending) == 1
    slider.evaluate(
        "e=>{for(let i=2;i<=8;i++){e.value=String(i);e.dispatchEvent(new Event('input',{bubbles:true}));}}"
    )
    assert len(pending) == 1
    first = pending[0]
    parts = urlsplit(first.request.url).path.split("/")
    parts = parts[parts.index("films") + 1 :]
    first.fulfill(
        json=masks.preview_frame(
            store, parts[0], parts[1], int(parts[3]), int(parts[4])
        )
    )
    for _ in range(100):
        if len(pending) > 1:
            break
        page.wait_for_timeout(10)
    assert len(pending) == 2 and pending[1].request.url.endswith("/8")
    second = pending[1]
    second.fulfill(
        json=masks.preview_frame(store, parts[0], parts[1], int(parts[3]), 8)
    )
    expect(page.locator("[data-mask-canvas]")).to_have_attribute("aria-busy", "false")
    expect(page.locator("[data-mask-position]")).to_contain_text("第 8 帧")
