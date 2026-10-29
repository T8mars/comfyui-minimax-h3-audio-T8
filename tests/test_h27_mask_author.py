"""Actual frozen media, source-bound masks, immutable downloads and HTTP."""

import asyncio
import io
import json
import uuid
import zipfile

import numpy as np
import pytest
from PIL import Image
from aiohttp.test_utils import TestClient, TestServer

from h3_audio_t8_pkg.director_mask_author import (
    load_author,
    save_author,
    preview_frame,
    rasterize,
    export_author,
    export_file,
)
from h3_audio_t8_pkg.director_project import file_sha
from test_director_captions import frozen
from test_director_caption_routes import routes_for


def brush(tool="paint"):
    return {"tool": tool, "width": 0.2, "points": [[0.2, 0.3], [0.7, 0.6]]}


def test_mask_save_reopen_export_frame_map_empty_occlusion_and_originals(tmp_path):
    store, output, project, _, film = frozen(tmp_path, trim=True)
    uid = str(uuid.uuid4())
    before = {p: file_sha(p) for p in output.glob("*.mp4")}
    draft = load_author(store, project["id"], film["id"], 0, uid)
    assert draft["revision"] == 0
    draft["strokes"] = {"2": [brush()], "3": [], "5": [brush(), brush("erase")]}
    saved = save_author(store, project["id"], film["id"], 0, uid, draft)
    assert load_author(store, project["id"], film["id"], 0, uid) == saved
    old = export_author(store, project["id"], film["id"], 0, uid)
    with zipfile.ZipFile(old) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        assert [r["source_frame"] for r in manifest["frame_map"]] == list(
            range(draft["source"]["frames"])
        )
        masks = [
            np.array(Image.open(io.BytesIO(archive.read(row["file"]))))
            for row in manifest["frame_map"]
        ]
    assert masks[2].any() and not masks[3].any() and not masks[5].any()
    assert not masks[0].any() and not masks[-1].any()
    old_sha = file_sha(old)
    saved["strokes"]["2"] = []
    newer = save_author(store, project["id"], film["id"], 0, uid, saved)
    assert export_author(store, project["id"], film["id"], 0, uid) != old
    assert export_file(store, project["id"], film["id"], 0, uid, saved["sha256"]) == old
    assert file_sha(old) == old_sha
    saved["strokes"]["2"] = [brush()]
    with pytest.raises(ValueError, match="别处更新"):
        save_author(store, project["id"], film["id"], 0, uid, saved)
    assert newer["revision"] == 2 and before == {p: file_sha(p) for p in before}


@pytest.mark.parametrize("damage", ["version", "frame", "nan", "tool", "width"])
def test_actual_invalid_sources_and_strokes_are_not_silently_saved(tmp_path, damage):
    store, _, project, _, film = frozen(tmp_path, trim=True)
    uid = str(uuid.uuid4())
    draft = load_author(store, project["id"], film["id"], 0, uid)
    draft["strokes"] = {"2": [brush()]}
    if damage == "version":
        draft["source"]["version_id"] = "another-take"
    if damage == "frame":
        draft["strokes"] = {"0": [brush()]}
    if damage == "nan":
        draft["strokes"]["2"][0]["points"][0][0] = float("nan")
    if damage == "tool":
        draft["strokes"]["2"][0]["tool"] = "track"
    if damage == "width":
        draft["strokes"]["2"][0]["width"] = 0
    with pytest.raises(ValueError):
        save_author(store, project["id"], film["id"], 0, uid, draft)
    assert load_author(store, project["id"], film["id"], 0, uid)["revision"] == 0


def test_mask_preview_uses_real_adopted_source_and_export_corruption_is_preserved(
    tmp_path,
):
    store, _, project, _, film = frozen(tmp_path, trim=True)
    preview = preview_frame(store, project["id"], film["id"], 0, 2)
    assert (
        preview["frame"] == 2
        and preview["media_sha256"] == film["entries"][0]["media_sha256"]
    )
    assert preview["data_url"].startswith("data:image/png;base64,")
    uid = str(uuid.uuid4())
    draft = load_author(store, project["id"], film["id"], 0, uid)
    saved = save_author(store, project["id"], film["id"], 0, uid, draft)
    path = export_author(store, project["id"], film["id"], 0, uid)
    path.write_bytes(b"broken")
    with pytest.raises(ValueError):
        export_file(store, project["id"], film["id"], 0, uid, saved["sha256"])
    assert path.read_bytes() == b"broken"


def test_http_save_preview_and_version_bound_download(tmp_path, monkeypatch):
    store, _, project, _, film = frozen(tmp_path, trim=True)
    app = routes_for(monkeypatch, store)
    uid = str(uuid.uuid4())
    base = f"/minimax_h3_t8/director/films/{project['id']}/{film['id']}"

    async def run():
        async with TestClient(TestServer(app)) as client:
            response = await client.get(base + f"/masks/0/{uid}")
            assert response.status == 200, await response.text()
            draft = await response.json()
            draft["strokes"] = {"2": [brush()]}
            response = await client.post(base + f"/masks/0/{uid}", json=draft)
            assert response.status == 200, await response.text()
            saved = await response.json()
            response = await client.get(base + "/mask-frames/0/2")
            assert response.status == 200, await response.text()
            response = await client.post(base + f"/masks/0/{uid}/export", json={})
            assert response.status == 200, await response.text()
            result = await response.json()
            assert result["author_sha256"] == saved["sha256"]
            response = await client.get(
                base + f"/masks/0/{uid}/file?sha={saved['sha256']}"
            )
            assert response.status == 200 and zipfile.is_zipfile(
                io.BytesIO(await response.read())
            )
            response = await client.get(base + "/mask-frames/0/0")
            assert response.status == 400

    asyncio.run(run())


def test_mask_composite_keeps_outside_rgb_and_erase_is_explicit():
    source = np.full((32, 64, 3), 57, dtype=np.uint8)
    candidate = np.full_like(source, 204)
    mask = np.array(rasterize([brush()], 64, 32))
    composite = np.where(mask[:, :, None] > 0, candidate, source)
    assert np.array_equal(composite[mask == 0], source[mask == 0])
    assert np.array_equal(composite[mask > 0], candidate[mask > 0])
    assert not np.array(rasterize([brush(), brush("erase")], 64, 32)).any()


def test_exported_mask_really_loads_into_core_composite_and_rejects_changed_source(
    tmp_path, monkeypatch
):
    import av
    import torch
    import folder_paths
    from comfy_extras.nodes_mask import ImageCompositeMasked
    from h3_audio_t8_pkg.director_film import film_media
    from h3_audio_t8_pkg.nodes_authored_mask import MiniMaxH3AuthoredMaskSequenceEXPT8

    store, _, project, _, film = frozen(tmp_path, trim=True)
    uid = str(uuid.uuid4())
    draft = load_author(store, project["id"], film["id"], 0, uid)
    draft["strokes"] = {"2": [brush()]}
    save_author(store, project["id"], film["id"], 0, uid, draft)
    export_author(store, project["id"], film["id"], 0, uid)
    with av.open(str(film_media(store, project["id"], film["id"], 0))) as media:
        frames = torch.stack(
            [
                torch.from_numpy(frame.to_ndarray(format="rgb24")).float() / 255
                for frame in media.decode(video=0)
            ]
        )
    before = frames.clone()
    monkeypatch.setattr(
        folder_paths, "get_input_directory", lambda: str(store.input_root)
    )
    schema = MiniMaxH3AuthoredMaskSequenceEXPT8.define_schema()
    options = next(x for x in schema.inputs if x.id == "mask_archive").options
    assert len(options) == 1 and "first" in options[0]
    masks, report = MiniMaxH3AuthoredMaskSequenceEXPT8.execute(
        frames, options[0]
    ).result
    assert (
        masks.shape == frames.shape[:3]
        and masks[2].count_nonzero() > 0
        and masks[3].count_nonzero() == 0
    )
    candidate = torch.ones_like(frames) * 0.8
    result = ImageCompositeMasked.execute(frames, candidate, 0, 0, False, masks).result[
        0
    ]
    outside = (masks == 0).unsqueeze(-1).expand_as(frames)
    assert torch.equal(result[outside], before[outside]) and torch.equal(frames, before)
    altered = frames.clone()
    altered[0, 0, 0, 0] = 1 - altered[0, 0, 0, 0]
    with pytest.raises(ValueError, match="RGB24"):
        MiniMaxH3AuthoredMaskSequenceEXPT8.execute(altered, options[0])
    assert json.loads(report)["source_audio"] == "unchanged_not_included"


def test_explicit_retry_after_lost_save_reply_is_idempotent(tmp_path):
    store, _, project, _, film = frozen(tmp_path, trim=True)
    uid = str(uuid.uuid4())
    draft = load_author(store, project["id"], film["id"], 0, uid)
    draft["strokes"] = {"2": [brush()]}
    first = save_author(store, project["id"], film["id"], 0, uid, draft)
    second = save_author(store, project["id"], film["id"], 0, uid, draft)
    assert first == second and second["revision"] == 1
