"""Real project/PNG/ZIP control lifecycle; no model, graph or GPU claim."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import uuid
import zipfile

from PIL import Image
import pytest

from h3_audio_t8_pkg.director_bundle import (
    build_bundle, import_bundle, inspect_bundle, portable_project, prepare_bundle,
)
from h3_audio_t8_pkg.director_controls import FIELD, normalize_controls
from h3_audio_t8_pkg.director_creation import TEMPLATE_FIELDS, remap_library
from h3_audio_t8_pkg.director_project import (
    ProjectStore, compile_project, file_sha, new_project, referenced_assets, validate_project,
)


def uid():
    return str(uuid.uuid4())


def control(asset_id=None, **extra):
    return {"id": uid(), "asset_id": asset_id,
            "control_net_name": "unfamiliar/control.safetensors", **extra}


def controls(*items):
    return {"version": 1, "items": list(items)}


def png(store, color, name="same.png"):
    asset_id = uid()
    path = store.input_root / "t8_director" / asset_id / "source.png"
    path.parent.mkdir(parents=True)
    Image.new("RGB", (64, 96), color).save(path)
    return store.register_asset(path, asset_id, name)


@pytest.fixture
def scene(tmp_path):
    store = ProjectStore(tmp_path / "user", tmp_path / "input")
    assets = [png(store, color) for color in [(20, 80, 130), (130, 40, 80)]]
    project = new_project()
    project["assets"] = assets
    shot = project["doc"]["shots"][0]
    shot.update(simplePrompt="普通文字\r\n🙂", autoDuration=False)
    shot[FIELD] = controls(control(assets[0]["id"]))
    return store, project, shot, assets


def template(project):
    shot = project["doc"]["shots"][0]
    snapshot = {key: deepcopy(value) for key, value in shot.items() if key in TEMPLATE_FIELDS}
    snapshot[FIELD] = deepcopy(shot[FIELD])
    return {"version": 1, "items": [{"id": uid(), "name": "控制模板", "kind": "template",
            "shot": snapshot, "bindings": {}}]}


def archive(scene, tmp_path):
    store, project, _, _ = scene
    output = tmp_path / "output"
    output.mkdir(exist_ok=True)
    prepared = prepare_bundle(store, project, [], output, include_results=False)
    assert prepared["ready"], prepared
    return build_bundle(store, project["id"], prepared["id"])


def uploaded(target, path):
    upload_id = uid()
    incoming = target.root / "bundle_imports" / f"{upload_id}.zip"
    incoming.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, incoming)
    return upload_id


def test_control_project_rejects_bad_version_before_save(scene):
    _, project, shot, _ = scene
    shot[FIELD]["version"] = 2
    with pytest.raises(ValueError, match="版本"):
        validate_project(project)


def test_compile_has_verified_control_rows_but_never_creates_reference_alias(scene):
    store, project, shot, assets = scene
    before = deepcopy(project)
    report = compile_project(project, store)
    assert report["ready"], report["errors"]
    compiled = report["shots"][0]
    rows = compiled["control_regions"]
    assert len(rows) == 1 and rows[0]["stage"] == "SINGLE"
    assert rows[0]["id"] == shot[FIELD]["items"][0]["id"]
    assert rows[0]["asset_sha256"] == assets[0]["sha256"]
    assert rows[0]["server_path"] == assets[0]["server_path"]
    assert rows[0]["asset_content_verified"] is True
    assert rows[0]["portable_cache_identity"] is False
    assert compiled["media_map"] == compiled["aliases"] == []
    assert report["extensions"] == {"control_regions_v1": True}
    assert project == before


def test_control_asset_dependencies_include_inactive_without_aliases(scene):
    _, project, shot, assets = scene
    shot[FIELD]["items"].append(control(assets[1]["id"], enabled=False))
    assert referenced_assets(project) == {asset["id"] for asset in assets}
    assert referenced_assets(project, include_controls=False) == set()


def test_portable_control_preserves_draft_but_strips_nested_private_data(scene):
    _, project, shot, _ = scene
    row = shot[FIELD]["items"][0]
    row.update(private_token="SECRET", source={"private_path": "C:/SECRET"})
    clean = portable_project(project)
    expected = normalize_controls(shot[FIELD])
    assert clean["doc"]["shots"][0][FIELD]["items"][0]["asset_id"] == row["asset_id"]
    assert clean["doc"]["shots"][0][FIELD]["items"][0]["source"]["start_seconds"] == expected["items"][0]["source"]["start_seconds"]
    assert "SECRET" not in json.dumps(clean)


def test_template_allows_control_field_and_real_zip_remaps_both_owners(scene, tmp_path):
    _, project, shot, assets = scene
    project["doc"]["creationLibrary"] = template(project)
    assert validate_project(project) == project
    bundle = archive(scene, tmp_path)
    before = file_sha(bundle)
    fresh = ProjectStore(tmp_path / "fresh-user", tmp_path / "fresh-input")
    output = tmp_path / "fresh-output"
    output.mkdir()
    incoming = uploaded(fresh, bundle)
    assert inspect_bundle(fresh, incoming)["ready"]
    target_id = uid()
    import_bundle(fresh, output, incoming, target_id)
    restored = fresh.load(target_id)
    live = restored["doc"]["shots"][0][FIELD]["items"][0]
    copy = restored["doc"]["creationLibrary"]["items"][0]["shot"][FIELD]["items"][0]
    assert live["asset_id"] == copy["asset_id"] != assets[0]["id"]
    assert len({live["id"], copy["id"], shot[FIELD]["items"][0]["id"]}) == 3
    assert fresh.asset(copy["asset_id"], verify=True)["sha256"] == assets[0]["sha256"]
    assert file_sha(bundle) == before


@pytest.mark.parametrize("sampling,expected", [
    ({"mode": "single"}, ["SINGLE"]),
    ({"mode": "two_pass", "output_mp": .6}, ["LOW", "HIGH"]),
    *[({"mode": "hyperflow", "variant": variant, "output_mp": .6,
        "hyperflow_file": "hyperflow/declared.safetensors"},
       ["SINGLE"] if variant == "single8" else ["LOW", "HIGH"])
      for variant in ["single8", "continuous4plus4", "continuous4plus4separate", "upscale8plus4", "upscale4plus4"]],
])
def test_compile_resolves_only_actual_sampling_stages_and_keeps_user_order(scene, sampling, expected):
    store, project, shot, assets = scene
    project["doc"]["sampling"] = sampling
    shot[FIELD]["items"] += [control(assets[1]["id"], stages={"low": False, "high": True})]
    report = compile_project(project, store)
    assert report["ready"], report["errors"]
    rows = report["shots"][0]["control_regions"]
    a, b = [row["id"] for row in shot[FIELD]["items"]]
    assert [(row["stage"], row["id"]) for row in rows] == (
        [("SINGLE", a), ("SINGLE", b)] if expected == ["SINGLE"]
        else [("LOW", a), ("HIGH", a), ("HIGH", b)])
    assert all(row["asset_content_verified"] for row in rows)


@pytest.mark.parametrize("low_fault", ["missing_asset", "missing_model", "geometry", "file"])
def test_resume_tail_compiles_only_high_controls_and_preserves_unused_low_draft(scene, low_fault):
    store, project, shot, assets = scene
    project["doc"]["sampling"] = {
        "mode": "hyperflow", "variant": "continuous4plus4separate",
        "hyperflow_file": "hyperflow/declared.safetensors",
        "stage_checkpoint": {"mode": "resume_tail", "artifact_path": "frozen/head.safetensors",
                             "artifact_sha256": "a" * 64},
    }
    low = control(assets[1]["id"], stages={"low": True, "high": False})
    if low_fault == "missing_asset":
        low["asset_id"] = uid()
    elif low_fault == "missing_model":
        low["control_net_name"] = ""
    elif low_fault == "geometry":
        low["region"] = {"mode": "rect", "xyxy_normalized": [1, 0, .2, 1]}
    else:
        (store.input_root / assets[1]["server_path"]).unlink()
    high = control(assets[0]["id"], stages={"low": False, "high": True})
    shot[FIELD] = controls(low, high)
    before = deepcopy(project)
    report = compile_project(project, store)
    assert report["ready"], report["errors"]
    rows = report["shots"][0]["control_regions"]
    assert [(row["stage"], row["id"]) for row in rows] == [("HIGH", high["id"])]
    assert rows[0]["asset_content_verified"] is True
    assert project == before


@pytest.mark.parametrize("draft", ["disabled", "zero", "no_dual_stage"])
def test_inactive_incomplete_drafts_save_restart_and_do_not_block_generation(scene, draft):
    store, project, shot, _ = scene
    item = control(uid(), control_net_name="", region={"mode": "rect", "xyxy_normalized": [4, 3, 2, 1]})
    if draft == "disabled":
        item["enabled"] = False
    elif draft == "zero":
        item["strength"] = 0
    else:
        item["stages"] = {"low": False, "high": False}
        project["doc"]["sampling"] = {"mode": "two_pass", "output_mp": .6}
    shot[FIELD] = controls(item)
    before = deepcopy(project)
    saved = store.save(project, 0)
    reopened = ProjectStore(store.root.parent, store.input_root).load(project["id"])
    assert reopened == saved and reopened["doc"]["shots"][0][FIELD] == before["doc"]["shots"][0][FIELD]
    report = compile_project(reopened, store)
    assert report["ready"], report["errors"]
    assert "control_regions" not in report["shots"][0]
    assert project == before


@pytest.mark.parametrize("fault", ["missing_asset", "missing_model", "region", "crop", "window", "file", "sha"])
def test_draft_saves_but_actual_generation_reports_concrete_control_error(scene, fault):
    store, project, shot, assets = scene
    row = shot[FIELD]["items"][0]
    if fault == "missing_asset":
        row["asset_id"] = uid()
    elif fault == "missing_model":
        row["control_net_name"] = ""
    elif fault == "region":
        row["region"] = {"mode": "rect", "xyxy_normalized": [-.1, 0, .8, 1]}
    elif fault == "crop":
        row["crop"] = {"enabled": True, "xyxy_normalized": [1, 0, .2, 1]}
    elif fault == "window":
        row.update(start_percent=.9, end_percent=.3)
    elif fault == "file":
        (store.input_root / assets[0]["server_path"]).unlink()
    else:
        path = store.input_root / assets[0]["server_path"]
        content = path.read_bytes()
        path.write_bytes(content[:-1] + bytes([content[-1] ^ 1]))
    saved = store.save(project, 0)
    assert saved["doc"]["shots"][0][FIELD] == shot[FIELD]
    report = compile_project(saved, store)
    assert not report["ready"]
    assert any(error.get("field") == FIELD for error in report["errors"])


def test_control_only_missing_asset_can_be_reconnected_without_relaxing_first_or_ref_checks(scene):
    store, project, shot, assets = scene
    original_sha = file_sha(store.input_root / assets[1]["server_path"])
    (store.input_root / assets[0]["server_path"]).unlink()
    saved = store.save(project, 0)
    assert saved["assets"][0]["missing"] is True
    assert not compile_project(saved, store)["ready"]
    selected_as_first = deepcopy(saved)
    selected_as_first["doc"]["shots"][0].update(mode="first", first=assets[0]["id"])
    with pytest.raises(ValueError, match="缺失"):
        store.save(selected_as_first, 1)
    reconnected = deepcopy(saved)
    reconnected["doc"]["shots"][0][FIELD]["items"][0]["asset_id"] = assets[1]["id"]
    reconnected = store.save(reconnected, 1)
    report = compile_project(reconnected, store)
    assert report["ready"], report["errors"]
    assert report["shots"][0]["control_regions"][0]["asset_sha256"] == assets[1]["sha256"]
    assert report["shots"][0]["aliases"] == []
    assert file_sha(store.input_root / assets[1]["server_path"]) == original_sha
    removed = deepcopy(reconnected)
    removed["doc"]["shots"][0][FIELD]["items"] = []
    assert "control_regions" not in compile_project(store.save(removed, 2), store)["shots"][0]


def test_selected_shot_does_not_compile_unselected_missing_control_sources(scene):
    store, project, shot, assets = scene
    other = deepcopy(shot)
    other["id"] = uid()
    other[FIELD] = controls(control(assets[1]["id"]))
    project["doc"]["shots"].append(other)
    (store.input_root / assets[1]["server_path"]).unlink()
    assert not compile_project(project, store)["ready"]
    selected = compile_project(project, store, shot_id=shot["id"])
    assert selected["ready"], selected["errors"]
    assert selected["shots"][0]["control_regions"][0]["asset_id"] == assets[0]["id"]
    assert not compile_project(project, store, shot_id=other["id"])["ready"]


def test_no_store_never_claims_bytes_read_and_many_unknown_controls_are_not_hard_banned(scene):
    _, project, shot, assets = scene
    shot[FIELD] = controls(*[control(assets[0]["id"], strength=-3.5,
        control_net_name="user-model/unknown.gguf") for _ in range(33)])
    report = compile_project(project)
    assert report["ready"], report["errors"]
    assert len(report["shots"][0]["control_regions"]) == 33
    assert all(not row["asset_content_verified"] and row["spec"]["strength"] == -3.5
               for row in report["shots"][0]["control_regions"])


def test_controls_do_not_renumber_existing_reference_alias_or_native_slot(scene):
    store, project, shot, assets = scene
    shot.update(mode="refs", refs=[assets[1]["id"]], tray=[assets[1]["id"]], simplePrompt="@image1 笑")
    before = deepcopy(project)
    before["doc"]["shots"][0].pop(FIELD)
    baseline = compile_project(before, store)["shots"][0]
    current = compile_project(project, store)["shots"][0]
    assert current["media_map"] == baseline["media_map"]
    assert current["aliases"] == baseline["aliases"]
    assert current["prompt"] == baseline["prompt"] == "<Picture 1> 笑"


def test_export_includes_control_only_media_and_same_name_distinct_sha(scene, tmp_path):
    _, project, shot, assets = scene
    shot[FIELD]["items"].append(control(assets[1]["id"], enabled=False))
    exported = archive(scene, tmp_path)
    with zipfile.ZipFile(exported) as stream:
        manifest = json.loads(stream.read("manifest.json"))
        assert len(manifest["files"]) == 2
        assert len({row["sha256"] for row in manifest["files"]}) == 2
        assert all(row["name"] == "same.png" for row in manifest["files"])
        assert len(manifest["project"]["doc"]["shots"][0][FIELD]["items"]) == 2
        assert "server_path" not in stream.read("manifest.json").decode("utf-8")


@pytest.mark.parametrize("name", ["G:/private/control.safetensors", "https://private.invalid/file", "../private.safetensors"])
def test_export_rejects_machine_paths_for_control_models_without_model_whitelist(scene, name):
    _, project, shot, _ = scene
    shot[FIELD]["items"][0]["control_net_name"] = name
    assert validate_project(project) == project
    with pytest.raises(ValueError, match="绝对路径"):
        portable_project(project)


def test_library_copy_and_explicit_missing_control_bindings_remap_without_alias_guessing(scene):
    _, project, shot, _ = scene
    missing = uid()
    shot[FIELD]["items"].append(control(missing, enabled=False))
    library = template(project)
    original = deepcopy(library)
    namespace = uuid.uuid4()
    def fresh(role, value):
        return str(uuid.uuid5(namespace, role+":"+value))
    remap_library(library, {}, fresh)
    rows = library["items"][0]["shot"][FIELD]["items"]
    assert all(row["id"] != old["id"] for row, old in zip(rows,
        original["items"][0]["shot"][FIELD]["items"], strict=True))
    assert rows[1]["asset_id"] == fresh("missing-collection-asset", missing)
    assert library["items"][0]["bindings"] == {}
    assert shot[FIELD] == project["doc"]["shots"][0][FIELD]


def test_imported_template_missing_control_stays_explicit_missing_not_another_same_name_asset(scene, tmp_path):
    _, project, _, assets = scene
    project["doc"]["creationLibrary"] = template(project)
    missing = uid()
    source = project["doc"]["creationLibrary"]["items"][0]["shot"][FIELD]["items"][0]
    source["asset_id"] = missing
    exported = archive(scene, tmp_path)
    fresh = ProjectStore(tmp_path / "fresh-user", tmp_path / "fresh-input")
    output = tmp_path / "fresh-output"
    output.mkdir()
    incoming = uploaded(fresh, exported)
    inspect_bundle(fresh, incoming)
    project_id = uid()
    import_bundle(fresh, output, incoming, project_id)
    restored = fresh.load(project_id)
    mapped = restored["doc"]["creationLibrary"]["items"][0]["shot"][FIELD]["items"][0]
    assert mapped["asset_id"] != missing
    assert mapped["asset_id"] not in {row["id"] for row in restored["assets"]}
    assert mapped["id"] != source["id"]
    assert len(restored["assets"]) == len(assets)


def test_project_control_lifecycle_import_stays_standard_library_model_free():
    root = Path(__file__).resolve().parents[1]
    code = r'''
import sys,types
package=types.ModuleType('h3_audio_t8_pkg')
package.__path__=[sys.argv[1]]
sys.modules['h3_audio_t8_pkg']=package
from h3_audio_t8_pkg.director_project import new_project,validate_project
from h3_audio_t8_pkg.director_bundle import portable_project
p=new_project();validate_project(p);portable_project(p)
assert 'torch' not in sys.modules and 'comfy.model_management' not in sys.modules
'''
    child = subprocess.run([sys.executable, "-X", "utf8", "-B", "-c", code, str(root / "h3_t8")],
        capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert child.returncode == 0, child.stdout + child.stderr
