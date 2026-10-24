"""Director middle-image draft clocks, content identities and portable bundles."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid
import zipfile

import pytest

from h3_audio_t8_pkg.director_keyframes import (
    FIELD, keyframe_asset_ids, portable_keyframes, resolve_keyframes, validate_keyframes,
)
from h3_audio_t8_pkg.director_project import (
    ProjectStore, compile_project, identity, new_project, referenced_assets, validate_project,
)
from h3_audio_t8_pkg.keyframe_positions import resolve_position
from test_director_d1 import asset


def entry(asset_id, mode="percent", position=50, **extra):
    return {"id": str(uuid.uuid4()), "asset_id": asset_id, "position_mode": mode,
            "position": position, "enabled": True, **extra}


def draft(items, basis="visible_delivery"):
    return {"version": 1, "time_basis": basis, "items": items}


@pytest.fixture
def scene(tmp_path):
    store = ProjectStore(tmp_path / "user", tmp_path / "input")
    images = [asset(store, name=name) for name in ("first.png", "last.png", "middle.png")]
    project = new_project()
    project["assets"] = images
    shot = project["doc"]["shots"][0]
    shot.update(mode="ends", first=images[0]["id"], last=images[1]["id"],
                simplePrompt="@image3 转身", manualDuration=4, duration=4, autoDuration=False)
    shot[FIELD] = draft([entry(images[2]["id"], position=95)])
    return store, project, shot, images


def test_old_project_without_field_and_empty_draft_preserve_original_inputs():
    project = new_project()
    shot = project["doc"]["shots"][0]
    shot["simplePrompt"] = "原文字"
    before = deepcopy(project)
    old = compile_project(project)
    assert "extensions" not in old
    assert FIELD not in validate_project(project)["doc"]["shots"][0]
    assert project == before and "middle_keyframes" not in old["shots"][0]
    shot[FIELD] = draft([])
    empty = compile_project(project)
    current = empty["shots"][0]
    assert current.pop("middle_keyframes") == [] and current == old["shots"][0]
    assert empty["extensions"] == {"middle_keyframes": 1}


def test_percent_uses_visible_clock_by_default_not_padded_generated_tail(scene):
    store, project, shot, images = scene
    source = deepcopy(project)
    report = compile_project(project, store)
    assert report["ready"], report["errors"]
    assert report["extensions"] == {"middle_keyframes": 1}
    compiled = report["shots"][0]
    middle = compiled["middle_keyframes"][0]
    assert compiled["time"]["delivery_trim_frames"] == middle["visible_frames"] == 96
    assert compiled["time"]["aligned_frames"] == middle["generated_frames"] == 107
    assert middle["position"] == 95 and middle["frame_index"] == 90 and middle["visible_in_delivery"]
    assert compiled["prompt"] == "<Picture 2> 转身"
    assert [row["role"] for row in compiled["media_map"]] == ["first_frame", "middle_keyframe", "last_frame"]
    assert compiled["media_map"][1]["keyframe_id"] == shot[FIELD]["items"][0]["id"]
    assert compiled["media_map"][1]["sha256"] == images[2]["sha256"]
    assert project == source


def test_generated_clock_is_explicit_keeps_offscreen_frame_with_visible_warning(scene):
    store, project, shot, _ = scene
    shot[FIELD]["time_basis"] = "generated_canvas"
    report = compile_project(project, store)
    assert report["ready"], report["errors"]
    middle = report["shots"][0]["middle_keyframes"][0]
    assert middle["frame_index"] == 101 and not middle["visible_in_delivery"]
    assert any("不在最终交付范围" in warning["message"] for warning in report["warnings"])


@pytest.mark.parametrize("mode,position,expected", [("frame", 48, 48), ("seconds", 2, 48), ("percent", 50, 48)])
def test_duration_changes_recompute_percent_but_keep_absolute_requests(scene, mode, position, expected):
    _, project, shot, images = scene
    row = entry(images[2]["id"], mode, position)
    shot[FIELD] = draft([row])
    assets = {item["id"]: item for item in images}
    assert resolve_keyframes(shot, assets)[0]["frame_index"] == expected
    shot["manualDuration"] = 8
    actual = resolve_keyframes(shot, assets)[0]
    assert actual["frame_index"] == (96 if mode == "percent" else 48)
    assert actual["position"] == position and actual["position_mode"] == mode


def test_shortened_duration_does_not_clamp_or_delete_absolute_keyframe(scene):
    _, project, shot, images = scene
    row = entry(images[2]["id"], "seconds", 3)
    shot[FIELD] = draft([row])
    shot["manualDuration"] = 2
    before = deepcopy(project)
    report = compile_project(project)
    assert not report["ready"] and any("frame 72" in error["message"] for error in report["errors"])
    assert project == before and shot[FIELD]["items"] == [row]


def test_same_frame_conflict_across_units_is_an_error_not_last_write_wins(scene):
    _, project, shot, images = scene
    shot[FIELD] = draft([entry(images[2]["id"], "seconds", 2), entry(images[2]["id"], "frame", 48)])
    report = compile_project(project)
    assert not report["ready"] and any("都定位到帧48" in error["message"] for error in report["errors"])


@pytest.mark.parametrize("fault", ["version", "basis", "boolean_position", "nan", "duplicate_id", "bad_asset_id", "nonbool_enabled", "noise", "resize"])
def test_versioned_field_rejects_real_bad_input(scene, fault):
    _, project, shot, _ = scene
    value = shot[FIELD]
    row = value["items"][0]
    if fault == "version":
        value["version"] = True
    elif fault == "basis":
        value["time_basis"] = "guess"
    elif fault == "boolean_position":
        row["position"] = True
    elif fault == "nan":
        row["position"] = float("nan")
    elif fault == "duplicate_id":
        value["items"].append(deepcopy(row))
    elif fault == "bad_asset_id":
        row["asset_id"] = "not-an-id"
    elif fault == "nonbool_enabled":
        row["enabled"] = 1
    elif fault == "noise":
        row["visual_noise_aug"] = 1.1
    else:
        row["resize_mode"] = "guess"
    with pytest.raises(ValueError):
        validate_project(project)


def test_disabled_empty_and_out_of_range_drafts_are_retained_without_execution(scene):
    _, project, shot, _ = scene
    shot[FIELD]["items"] = [entry(None, "seconds", 1000, enabled=False)]
    validate_keyframes(shot[FIELD], identity)
    shot["simplePrompt"] = "只是原首尾"
    report = compile_project(project)
    assert report["ready"] and report["shots"][0]["middle_keyframes"] == []
    assert shot[FIELD]["items"][0]["position"] == 1000


def test_more_than_seven_drafts_do_not_gain_an_invented_project_admission_cap(scene):
    _, project, shot, images = scene
    shot[FIELD]["items"] = [entry(images[2]["id"], "frame", index) for index in range(1, 10)]
    report = compile_project(project)
    assert report["ready"] and len(report["shots"][0]["middle_keyframes"]) == 9


def test_missing_endpoints_or_missing_image_preflight_never_silently_discards_plan(scene):
    _, project, shot, _ = scene
    shot["last"] = None
    assert any("中间关键帧需要" in error["message"] for error in compile_project(project)["errors"])
    shot[FIELD]["items"][0]["asset_id"] = str(uuid.uuid4())
    assert any("缺少实际图片" in error["message"] for error in compile_project(project)["errors"])


def test_keyframe_only_asset_is_referenced_saved_and_content_verified(scene):
    store, project, shot, images = scene
    assert keyframe_asset_ids(shot) == [images[2]["id"]]
    assert images[2]["id"] in referenced_assets(project)
    saved = store.save(project, 0)
    assert store.load(project["id"]) == saved
    (store.input_root / images[2]["server_path"]).write_bytes(b"changed")
    report = compile_project(saved, store)
    assert not report["ready"] and any(error.get("asset_id") == images[2]["id"] for error in report["errors"])


def test_portable_fields_strip_unknown_metadata_and_do_not_mutate_draft(scene):
    _, project, shot, _ = scene
    shot[FIELD]["secret"] = "PRIVATE"
    shot[FIELD]["items"][0]["token"] = "PRIVATE"
    before = deepcopy(shot[FIELD])
    assert "PRIVATE" not in json.dumps(portable_keyframes(shot[FIELD]))
    from h3_audio_t8_pkg.director_bundle import portable_project
    assert "PRIVATE" not in json.dumps(portable_project(project)) and shot[FIELD] == before


def test_real_bundle_export_import_remaps_keyframe_and_asset_identities(tmp_path):
    from test_director_bundle import bundle_fixture, upload
    from h3_audio_t8_pkg.director_bundle import build_bundle, import_bundle, inspect_bundle, prepare_bundle

    original_root = tmp_path / "original"
    original_root.mkdir()
    store, output, project, records, _, _ = bundle_fixture(original_root)
    middle = asset(store, name="middle-only.png")
    project["assets"].append(middle)
    shot = project["doc"]["shots"][0]
    shot[FIELD] = draft([entry(middle["id"]), entry(None, "seconds", 99, enabled=False)])
    before = deepcopy(project)
    plan = prepare_bundle(store, project, records, output)
    assert plan["ready"], plan
    archive = build_bundle(store, project["id"], plan["id"])
    with zipfile.ZipFile(archive) as stream:
        manifest = json.loads(stream.read("manifest.json"))
        assert manifest["project"]["doc"]["shots"][0][FIELD] == shot[FIELD]
    fresh = ProjectStore(tmp_path / "new-user", tmp_path / "new-input")
    fresh_output = tmp_path / "new-output"
    fresh_output.mkdir()
    new_id = str(uuid.uuid4())
    upload_id = upload(fresh, archive)
    inspect_bundle(fresh, upload_id)
    import_bundle(fresh, fresh_output, upload_id, new_id)
    reopened = fresh.load(new_id)
    rows = reopened["doc"]["shots"][0][FIELD]["items"]
    assert rows[0]["id"] != shot[FIELD]["items"][0]["id"]
    assert rows[0]["asset_id"] != middle["id"]
    assert fresh.asset(rows[0]["asset_id"], verify=True)["sha256"] == middle["sha256"]
    assert rows[1]["id"] != shot[FIELD]["items"][1]["id"] and rows[1]["asset_id"] is None
    assert rows[1]["position"] == 99 and not rows[1]["enabled"] and project == before


def test_native_position_math_keeps_original_aligned_canvas_meaning():
    assert resolve_position("percent", 95, 107) == 101
    assert resolve_position("percent", 95, 96) == 90
    assert resolve_position("seconds", .0625, 107) == 2
    with pytest.raises(ValueError, match="nonfinite frame"):
        resolve_position("seconds", 1e308, 107)


def test_project_keyframe_import_remains_free_of_torch_and_model_runtime():
    runtime = Path(__file__).resolve().parents[1] / "h3_t8"
    script = ("import sys,types; p=types.ModuleType('pure'); p.__path__=[sys.argv[1]]; "
              "sys.modules['pure']=p; import pure.director_project; "
              "assert 'torch' not in sys.modules; print('pure data import passed')")
    result = subprocess.run([sys.executable, "-B", "-c", script, str(runtime)],
                            capture_output=True, text=True, encoding="utf8", env={**os.environ, "PYTHONUTF8": "1"}, timeout=30)
    assert result.returncode == 0, result.stderr
