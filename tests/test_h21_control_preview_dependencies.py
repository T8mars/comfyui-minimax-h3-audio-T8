"""A selected preview is not a generation role; actual first/ref/tray stay strict."""
from copy import deepcopy
import uuid

import pytest

from h3_audio_t8_pkg.director_controls import FIELD
from h3_audio_t8_pkg.director_project import compile_project, referenced_assets
from test_h21_director_control_project import scene as shared_scene


scene = shared_scene


def selected_control(scene):
    store, project, shot, assets = scene
    shot["selected"] = assets[0]["id"]
    shot[FIELD]["items"][0]["enabled"] = False
    (store.input_root / assets[0]["server_path"]).unlink()
    return store, project, shot, assets


def test_control_preview_does_not_become_an_ordinary_dependency(scene):
    _, project, _, assets = selected_control(scene)
    assert referenced_assets(project, include_controls=False) == set()
    assert referenced_assets(project) == {assets[0]["id"]}


def test_save_retains_missing_control_preview_draft_for_explicit_reconnection(scene):
    store, project, shot, assets = selected_control(scene)
    before = deepcopy(project)
    saved = store.save(project, 0)
    assert saved["doc"]["shots"][0][FIELD] == shot[FIELD]
    assert saved["doc"]["shots"][0]["selected"] == assets[0]["id"]
    assert saved["assets"][0]["missing"] is True
    assert project == before


def test_compile_does_not_require_inactive_selected_preview_source(scene):
    store, project, _, _ = selected_control(scene)
    before = deepcopy(project)
    report = compile_project(project, store)
    assert report["ready"], report["errors"]
    assert "control_regions" not in report["shots"][0]
    assert report["shots"][0]["aliases"] == [] and project == before


@pytest.mark.parametrize("role", ["first", "last", "refs", "tray"])
def test_selected_preview_never_weakens_real_first_last_ref_or_tray_requirements(scene, role):
    store, project, shot, assets = selected_control(scene)
    missing = assets[0]["id"]
    if role == "first":
        shot.update(mode="first", first=missing)
    elif role == "last":
        shot.update(mode="ends", first=assets[1]["id"], last=missing)
    else:
        shot[role] = [missing]
        if role == "refs":
            shot["mode"] = "refs"
    with pytest.raises(ValueError, match="缺失"):
        store.save(project, 0)
    assert not compile_project(project, store)["ready"]


@pytest.mark.parametrize("legacy", ["no_controls", "unrelated_selected"])
def test_non_control_selected_keeps_original_ordinary_dependency_contract(scene, legacy):
    store, project, shot, assets = scene
    if legacy == "no_controls":
        shot.pop(FIELD)
    else:
        shot[FIELD]["items"][0]["enabled"] = False
    selected = assets[0 if legacy == "no_controls" else 1]
    shot["selected"] = selected["id"]
    (store.input_root / selected["server_path"]).unlink()
    assert selected["id"] in referenced_assets(project, include_controls=False)
    with pytest.raises(ValueError, match="缺失"):
        store.save(project, 0)
    assert not compile_project(project, store)["ready"]


def test_control_preview_does_not_hide_actual_active_control_source_error(scene):
    store, project, shot, _ = selected_control(scene)
    shot[FIELD]["items"][0]["enabled"] = True
    report = compile_project(project, store)
    assert not report["ready"]
    assert any(error.get("field") == FIELD for error in report["errors"])


def test_control_preview_dependency_is_still_required_by_another_shots_first_frame(scene):
    store, project, shot, assets = selected_control(scene)
    other = deepcopy(shot)
    other.pop(FIELD)
    other.update(id=str(uuid.uuid4()), mode="first", first=assets[0]["id"])
    project["doc"]["shots"].append(other)
    assert assets[0]["id"] in referenced_assets(project, include_controls=False)
    with pytest.raises(ValueError, match="缺失"):
        store.save(project, 0)
    assert not compile_project(project, store)["ready"]
