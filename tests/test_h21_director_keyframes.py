"""Independent D clocks, real asset/ZIP identity and pure template regressions.

No GPU, server, browser or trained-weight inference. The one graph comparison
uses a model-catalog fixture only; media are real registered PNG/WAV bytes.
"""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import uuid
import wave
import zipfile

from PIL import Image
import pytest

from h3_audio_t8_pkg.director_bundle import (
    build_bundle, import_bundle, inspect_bundle, portable_project, prepare_bundle,
)
from h3_audio_t8_pkg.director_creation import TEMPLATE_FIELDS
from h3_audio_t8_pkg.director_keyframes import FIELD, resolve_keyframes
from h3_audio_t8_pkg.director_project import (
    ProjectStore, compile_project, file_sha, new_project, validate_project,
)
from h3_audio_t8_pkg.keyframe_positions import resolve_position


ROOT = Path(__file__).resolve().parents[1]


def uid():
    return str(uuid.uuid4())


def row(asset_id, position=48, mode="frame", **extra):
    return {"id": uid(), "asset_id": asset_id, "position_mode": mode,
            "position": position, "visual_noise_aug": .999,
            "resize_mode": "center_crop", "enabled": True, **extra}


def plan(items, basis="visible_delivery"):
    return {"version": 1, "time_basis": basis, "items": items}


def png(store, color, name="duplicate.png"):
    aid = uid()
    path = store.input_root / "t8_director" / aid / "source.png"
    path.parent.mkdir(parents=True)
    Image.new("RGB", (64, 96), color).save(path)
    return store.register_asset(path, aid, name)


@pytest.fixture
def scene(tmp_path):
    store = ProjectStore(tmp_path / "user", tmp_path / "input")
    images = [png(store, color) for color in [(190, 20, 30), (10, 180, 30), (20, 30, 170)]]
    project = new_project()
    project["assets"] = images
    shot = project["doc"]["shots"][0]
    shot.update(mode="ends", first=images[0]["id"], last=images[1]["id"],
                simplePrompt="保留原文\r\n🙂\t𠀀", manualDuration=4, duration=4, autoDuration=False)
    shot[FIELD] = plan([row(images[2]["id"])])
    return store, project, shot, images


def bundle(scene, tmp_path):
    store, project, _, _ = scene
    output = tmp_path / "output"
    output.mkdir(exist_ok=True)
    store.save(project, 0)
    prepared = prepare_bundle(store, project, [], output, include_results=False)
    assert prepared["ready"], prepared
    return build_bundle(store, project["id"], prepared["id"])


def upload(target, archive):
    upload_id = uid()
    path = target.root / "bundle_imports" / f"{upload_id}.zip"
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(archive, path)
    return upload_id


def template(project):
    shot = project["doc"]["shots"][0]
    snapshot = {key: deepcopy(value) for key, value in shot.items() if key in TEMPLATE_FIELDS}
    # Include the intended public field explicitly so a missing whitelist
    # cannot turn a regression test into an empty-plan success.
    snapshot[FIELD] = deepcopy(shot[FIELD])
    return {"version": 1, "items": [{"id": uid(), "name": "原中间帧模板", "kind": "template",
            "shot": snapshot, "bindings": {}}]}


def test_same_source_image_can_occupy_distinct_positions_and_reference_role(scene):
    store, project, shot, images = scene
    shot["last"] = shot["first"]
    first, middle = images[0]["id"], images[2]["id"]
    a, b, c = row(middle, 72), row(first, 20), row(middle, 48)
    shot[FIELD] = plan([a, b, c])
    shot["refs"] = [middle]
    before = deepcopy(project)
    report = compile_project(project, store)
    assert report["ready"], report["errors"]
    compiled = report["shots"][0]
    assert [item["id"] for item in compiled["middle_keyframes"]] == [b["id"], c["id"], a["id"]]
    assert [(item["asset_id"], item["role"]) for item in compiled["media_map"]] == [
        (first, "first_frame"), (first, "middle_keyframe"),
        (middle, "middle_keyframe"), (middle, "middle_keyframe"),
        (first, "last_frame"), (middle, "ref_image")]
    assert [item["native"] for item in compiled["media_map"]] == [f"<Picture {i}>" for i in range(1, 7)]
    assert len({item["sha256"] for item in compiled["media_map"] if item["role"] == "middle_keyframe"}) == 2
    assert project == before


def test_real_save_restart_and_json_preserve_more_than_seven_inactive_drafts(scene):
    store, project, shot, images = scene
    shot[FIELD] = plan([row(images[2]["id"], i + 1000, enabled=False) for i in range(11)])
    source = deepcopy(project)
    saved = store.save(project, 0)
    restarted = ProjectStore(store.root.parent, store.input_root)
    reopened = restarted.load(project["id"])
    roundtrip = validate_project(json.loads(json.dumps(reopened, ensure_ascii=False)))
    assert roundtrip == reopened == saved
    assert len(roundtrip["doc"]["shots"][0][FIELD]["items"]) == 11
    assert roundtrip["doc"]["shots"][0][FIELD] == source["doc"]["shots"][0][FIELD]
    assert project == source
    assert compile_project(roundtrip, restarted)["ready"]


@pytest.mark.parametrize("mode,value,expected", [("frame", 71, 71), ("seconds", 2.5, 60), ("percent", 50, 24)])
def test_effective_duration_follows_active_events_not_stale_hidden_writing(scene, mode, value, expected):
    _, project, shot, images = scene
    shot.update(manualDuration=8, duration=123, autoDuration=True, writingMode="advanced",
                prompt="高级稿", events=[{"id": uid(), "start": 0, "end": 2, "text": "动作"}])
    shot[FIELD] = plan([row(images[2]["id"], value, mode)], "generated_canvas" if mode != "percent" else "visible_delivery")
    assets = {asset["id"]: asset for asset in images}
    if mode != "percent":
        # The declared 2-second generated canvas is only 56 frames: these
        # absolute requests really are invalid, not auto-stretched to 8 sec.
        with pytest.raises(ValueError, match="resolves to frame"):
            resolve_keyframes(shot, assets)
    else:
        assert resolve_keyframes(shot, assets)[0]["frame_index"] == expected
    shot["writingMode"] = "simple"
    actual = resolve_keyframes(shot, assets)[0]
    assert actual["frame_index"] == (96 if mode == "percent" else expected)
    assert actual["visible_frames"] == 192 and shot["duration"] == 123


def test_recording_clock_can_be_resolved_without_claiming_avatar_middle_route(scene):
    store, _, shot, images = scene
    aid = uid()
    path = store.input_root / "t8_director" / aid / "source.wav"
    path.parent.mkdir(parents=True)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(8000)
        stream.writeframes(b"\x00\x00" * 8000 * 8)
    audio = store.register_asset(path, aid, "voice.wav")
    shot.update(sound="record", audio=audio["id"], start=2, end=5, autoDuration=True)
    shot[FIELD] = plan([row(images[2]["id"], 50, "percent")])
    actual = resolve_keyframes(shot, {item["id"]: item for item in [*images, audio]})[0]
    from h3_audio_t8_pkg.timing import make_timing_plan
    offset = round(make_timing_plan(2, 3, source_duration_seconds=8).final_trim_start_seconds * 24)
    assert actual["visible_frames"] == 72 and actual["frame_index"] == offset + 36
    assert (shot["start"], shot["end"]) == (2, 5)


@pytest.mark.parametrize("frame_count", [96, 107, 192])
def test_shared_math_preserves_half_up_ties_and_excludes_actual_endpoints(frame_count):
    assert resolve_position("seconds", .0625, frame_count) == 2
    assert resolve_position("seconds", 23.5 / 24, frame_count) == 24
    for endpoint in [0, frame_count - 1]:
        with pytest.raises(ValueError, match="middle keyframes"):
            resolve_position("frame", endpoint, frame_count)


def test_fractional_frame_does_not_gain_relative_tolerance_at_large_index():
    with pytest.raises(ValueError, match="integer"):
        resolve_position("frame", 1_000_000_000.25, 1_000_000_010)


def test_finite_but_overflowing_duration_is_actionable_input_not_uncaught_overflow(scene):
    _, project, shot, _ = scene
    shot["manualDuration"] = 1e308
    try:
        result = compile_project(project)
    except ValueError as error:
        assert any(word in str(error).lower() for word in ["frame", "finite", "时长", "时间", "帧"])
    else:
        assert not result["ready"] and result["errors"]


@pytest.mark.parametrize("field", ["time_basis", "position_mode", "resize_mode"])
@pytest.mark.parametrize("payload", [[], {}])
def test_malformed_json_enums_raise_real_input_error_not_unhashable_typeerror(scene, field, payload):
    _, project, shot, _ = scene
    holder = shot[FIELD] if field == "time_basis" else shot[FIELD]["items"][0]
    holder[field] = payload
    with pytest.raises(ValueError):
        validate_project(project)


def test_selected_shot_checks_its_middle_bytes_but_not_unselected_missing_input(scene):
    store, project, shot, images = scene
    sibling = deepcopy(shot)
    sibling["id"] = uid()
    sibling["first"] = sibling["last"] = None
    sibling[FIELD] = plan([row(uid())])
    project["doc"]["shots"].append(sibling)
    original = deepcopy(project)
    assert not compile_project(project, store)["ready"]
    good = compile_project(project, store, shot_id=shot["id"])
    assert good["ready"], good["errors"]
    path = store.input_root / images[2]["server_path"]
    payload = bytearray(path.read_bytes())
    payload[-1] ^= 1
    path.write_bytes(payload)
    bad = compile_project(project, store, shot_id=shot["id"])
    assert not bad["ready"] and any(item.get("asset_id") == images[2]["id"] for item in bad["errors"])
    assert project == original


def test_same_name_different_sha_is_not_deduplicated_by_bundle(scene, tmp_path):
    _, project, shot, images = scene
    assert len({item["sha256"] for item in images}) == 3
    shot[FIELD]["items"] += [row(images[2]["id"], 72)]
    source = deepcopy(project)
    archive = bundle(scene, tmp_path)
    original_digest = file_sha(archive)
    with zipfile.ZipFile(archive) as stream:
        manifest = json.loads(stream.read("manifest.json"))
        assert len(manifest["files"]) == 3
        assert len({item["sha256"] for item in manifest["files"]}) == 3
        assert manifest["project"]["doc"]["shots"][0][FIELD] == shot[FIELD]
    target = ProjectStore(tmp_path / "fresh-user", tmp_path / "fresh-input")
    output = tmp_path / "fresh-output"
    output.mkdir()
    imported_id = uid()
    upload_id = upload(target, archive)
    assert inspect_bundle(target, upload_id)["ready"]
    import_bundle(target, output, upload_id, imported_id)
    restored = target.load(imported_id)
    rows = restored["doc"]["shots"][0][FIELD]["items"]
    assert len(rows) == 2 and rows[0]["id"] != rows[1]["id"]
    assert rows[0]["asset_id"] == rows[1]["asset_id"] != images[2]["id"]
    assert all(item["id"] not in {old["id"] for old in shot[FIELD]["items"]} for item in rows)
    assert target.asset(rows[0]["asset_id"], verify=True)["sha256"] == images[2]["sha256"]
    assert project == source and file_sha(archive) == original_digest


def test_corrupt_middle_payload_in_zip_never_publishes_a_project(scene, tmp_path):
    _, project, _, images = scene
    archive = bundle(scene, tmp_path)
    source_digest = file_sha(archive)
    tampered = tmp_path / "tampered.zip"
    with zipfile.ZipFile(archive) as incoming, zipfile.ZipFile(tampered, "w") as outgoing:
        manifest = json.loads(incoming.read("manifest.json"))
        member = next(item["path"] for item in manifest["files"] if item.get("asset_id") == images[2]["id"])
        for info in incoming.infolist():
            payload = incoming.read(info.filename)
            if info.filename == member:
                payload = bytes([payload[0] ^ 1]) + payload[1:]
            outgoing.writestr(info, payload)
    target = ProjectStore(tmp_path / "fresh-user", tmp_path / "fresh-input")
    upload_id = upload(target, tampered)
    with pytest.raises(ValueError, match="SHA"):
        inspect_bundle(target, upload_id)
    assert not target.list() and file_sha(archive) == source_digest
    assert project["assets"][2]["sha256"] == images[2]["sha256"]


def test_template_field_is_valid_portable_and_real_zip_remaps_both_owners(scene, tmp_path):
    store, project, shot, images = scene
    project["doc"]["creationLibrary"] = template(project)
    before = deepcopy(project)
    assert validate_project(project) == project
    clean = portable_project(project)
    assert clean["doc"]["creationLibrary"]["items"][0]["shot"][FIELD] == shot[FIELD]
    archive = bundle(scene, tmp_path)
    target = ProjectStore(tmp_path / "fresh-user", tmp_path / "fresh-input")
    output = tmp_path / "fresh-output"
    output.mkdir()
    imported_id = uid()
    upload_id = upload(target, archive)
    assert inspect_bundle(target, upload_id)["ready"]
    import_bundle(target, output, upload_id, imported_id)
    restored = target.load(imported_id)
    live = restored["doc"]["shots"][0][FIELD]["items"][0]
    copy = restored["doc"]["creationLibrary"]["items"][0]["shot"][FIELD]["items"][0]
    assert live["asset_id"] == copy["asset_id"] != images[2]["id"]
    assert live["id"] != shot[FIELD]["items"][0]["id"]
    assert copy["id"] != shot[FIELD]["items"][0]["id"]
    assert target.asset(copy["asset_id"], verify=True)["sha256"] == images[2]["sha256"]
    assert store.load(project["id"])["doc"]["creationLibrary"] == project["doc"]["creationLibrary"]
    assert project == before


@pytest.mark.parametrize("action", ["capture", "copy", "reconnect", "missing"])
def test_actual_pure_js_template_operations_keep_middle_identity(scene, action):
    _, project, shot, images = scene
    node = shutil.which("node")
    assert node, "Node.js is required to execute the real pure template module"
    module_url = (ROOT / "web/director/creation_library.mjs").as_uri()
    payload = json.dumps({"doc": project["doc"], "shot": shot, "assets": images,
                          "action": action, "library": template(project)}, ensure_ascii=False)
    script = f"""
        import {{captureCreation,planCreation,remapCreationAsset}} from {json.dumps(module_url)};
        import {{randomUUID}} from 'node:crypto';
        let input=''; for await (const chunk of process.stdin) input+=chunk;
        const value=JSON.parse(input);
        let result;
        if(value.action==='capture'){{
            result=captureCreation({{...value,kind:'template',name:'captured',id:'collection',d3:{{}}}});
        }} else if(value.action==='reconnect'){{
            value.doc.creationLibrary=value.library;
            remapCreationAsset(value.doc,value.assets[2].id,'replacement-identity');
            result=value.doc.creationLibrary.items[0];
        }} else {{
            const item=value.library.items[0];
            const assets=value.action==='missing'?value.assets.slice(0,2):value.assets;
            try{{
                const planned=planCreation({{doc:value.doc,shotId:value.shot.id,assets,item,uuid:randomUUID}});
                result={{applied:true,shot:planned.doc.shots.find(row=>row.id===planned.current)}};
            }}catch(error){{ result={{applied:false,error:String(error)}}; }}
        }}
        process.stdout.write(JSON.stringify(result));
    """
    child = subprocess.run([node, "--input-type=module", "-e", script], input=payload,
                           capture_output=True, text=True, encoding="utf8", timeout=20)
    assert child.returncode == 0, child.stderr
    captured = json.loads(child.stdout)
    if action == "missing":
        assert captured["applied"] is False and captured["error"]
    elif action == "reconnect":
        assert captured["shot"][FIELD]["items"][0]["asset_id"] == "replacement-identity"
    else:
        source = shot[FIELD]["items"][0]
        actual = captured["shot"][FIELD]["items"][0]
        assert actual["asset_id"] == source["asset_id"]
        assert actual["position"] == source["position"]
        assert actual["id"] == source["id"] if action == "capture" else actual["id"] != source["id"]
        if action == "capture":
            assert images[2]["id"] in captured["shot"]["tray"]


@pytest.mark.parametrize("extension", [None, "empty", "disabled"])
def test_no_active_middle_leaves_actual_default_graph_unchanged(scene, monkeypatch, extension):
    from h3_audio_t8_pkg.director_generation import build_director_generation_prompt
    from test_director_generation import _patch_director_models

    store, project, shot, _ = scene
    _patch_director_models(monkeypatch)  # Catalog only; no model invocation is simulated.
    del shot[FIELD]
    expected = build_director_generation_prompt(project, shot["id"], store)["prompt"]
    if extension is None:
        shot[FIELD] = None
    else:
        shot[FIELD] = plan([] if extension == "empty" else [row(None, 1000, enabled=False)])
    actual = build_director_generation_prompt(project, shot["id"], store)["prompt"]
    assert actual == expected
