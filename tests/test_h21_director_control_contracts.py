"""Director regional-control data contracts; no graph/UI or GPU claim."""

from copy import deepcopy
import importlib
import json
from pathlib import Path
import subprocess
import sys
import uuid

import pytest


def module():
    return importlib.import_module("h3_audio_t8_pkg.director_controls")


def uid():
    return str(uuid.uuid4())


def row(**extra):
    return {"id": uid(), "asset_id": None, **extra}


def shot(*items):
    return {"controlRegions": {"version": 1, "items": list(items)}}


def scene(tmp_path, *, kind="image"):
    content = b"registered control input"
    path = (
        tmp_path
        / "t8_director"
        / uid()
        / ("control.png" if kind == "image" else "control.mp4")
    )
    path.parent.mkdir(parents=True)
    path.write_bytes(content)
    import hashlib

    asset = {
        "id": uid(),
        "kind": kind,
        "server_path": path.relative_to(tmp_path).as_posix(),
        "sha256": hashlib.sha256(content).hexdigest(),
        "width": 941,
        "height": 1672,
        "duration": 4.0 if kind == "video" else 0.0,
    }
    item = row(
        asset_id=asset["id"], control_net_name="unfamiliar/custom-control.safetensors"
    )
    return shot(item), asset, path


def test_absent_field_is_absent_and_normalization_is_lossless_for_the_caller():
    api = module()
    assert api.normalize_controls(None) is None
    assert api.control_asset_ids({}) == [] and api.resolve_controls({}, {}) == []
    value = {"version": 1, "items": [row()], "private": "preserved-draft"}
    before = deepcopy(value)
    normalized = api.normalize_controls(value)
    assert value == before and normalized is not value
    assert normalized["items"][0]["strength"] == 0.7
    assert normalized["items"][0]["stages"] == {"low": True, "high": True}
    assert normalized["private"] == "preserved-draft"


@pytest.mark.parametrize(
    "fault",
    [
        "version",
        "bool_version",
        "items",
        "duplicate",
        "id",
        "asset_id",
        "enabled",
        "strength_nan",
        "start_nan",
        "crop_nan",
        "region_nan",
        "stages",
        "model_type",
    ],
)
def test_invalid_version_identity_and_nonfinite_fields_are_never_accepted(fault):
    api = module()
    value = {"version": 1, "items": [row()]}
    item = value["items"][0]
    if fault == "version":
        value["version"] = 2
    elif fault == "bool_version":
        value["version"] = True
    elif fault == "items":
        value["items"] = {}
    elif fault == "duplicate":
        value["items"].append(deepcopy(item))
    elif fault == "id":
        item["id"] = "not-an-id"
    elif fault == "asset_id":
        item["asset_id"] = "bad-asset"
    elif fault == "enabled":
        item["enabled"] = 1
    elif fault == "strength_nan":
        item["strength"] = float("nan")
    elif fault == "start_nan":
        item["start_percent"] = float("inf")
    elif fault == "crop_nan":
        item["crop"] = {"xyxy_normalized": [0, 0, float("nan"), 1]}
    elif fault == "region_nan":
        item["region"] = {"xyxy_normalized": [0, 0, 1, float("inf")]}
    elif fault == "stages":
        item["stages"] = {"high": 1}
    else:
        item["control_net_name"] = {}
    with pytest.raises(ValueError):
        api.normalize_controls(value)


def test_disabled_and_zero_strength_drafts_do_not_need_a_model_or_asset():
    api = module()
    disabled = row(
        enabled=False, region={"mode": "rect", "xyxy_normalized": [4, 3, 2, 1]}
    )
    zero = row(strength=0, crop={"enabled": True, "xyxy_normalized": [9, 9, 0, 0]})
    value = shot(disabled, zero)
    before = deepcopy(value)
    assert api.resolve_controls(value, {}) == []
    assert api.portable_controls(value[api.FIELD])["items"][0]["region"][
        "xyxy_normalized"
    ] == [4, 3, 2, 1]
    assert value == before


def test_unknown_model_name_negative_strength_and_many_rows_are_not_admission_bans(
    tmp_path,
):
    api = module()
    value, asset, _ = scene(tmp_path)
    value[api.FIELD]["items"][0]["strength"] = -3.5
    for _ in range(32):
        value[api.FIELD]["items"].append(
            row(asset_id=asset["id"], control_net_name="anything.gguf", strength=7.0)
        )
    actual = api.resolve_controls(value, {asset["id"]: asset}, asset_root=tmp_path)
    assert len(actual) == 33 and actual[0]["spec"]["strength"] == -3.5
    assert (
        actual[0]["spec"]["control_net_name"] == "unfamiliar/custom-control.safetensors"
    )


def test_stages_filter_execution_without_discarding_other_stage_or_disabled_assets(
    tmp_path,
):
    api = module()
    value, asset, _ = scene(tmp_path)
    value[api.FIELD]["items"][0]["stages"] = {"low": False, "high": True}
    dormant = uid()
    value[api.FIELD]["items"].append(row(asset_id=dormant, enabled=False))
    assert api.control_asset_ids(value) == [asset["id"], dormant]
    assert api.control_asset_ids(value, active=True, stage="low") == []
    assert api.resolve_controls(value, {asset["id"]: asset}, stage="low") == []
    assert len(api.resolve_controls(value, {asset["id"]: asset}, stage="high")) == 1
    assert len(api.resolve_controls(value, {asset["id"]: asset}, stage="single")) == 1


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "wrong_kind",
        "wrong_id",
        "bad_sha",
        "unsafe_path",
        "missing_file",
        "changed_file",
        "bad_dimensions",
        "empty_model",
    ],
)
def test_active_control_requires_real_asset_contract_not_a_plausible_declaration(
    tmp_path, fault
):
    api = module()
    value, asset, path = scene(tmp_path)
    assets = {asset["id"]: asset}
    if fault == "missing":
        assets.clear()
    elif fault == "wrong_kind":
        asset["kind"] = "audio"
    elif fault == "wrong_id":
        asset["id"] = uid()
    elif fault == "bad_sha":
        asset["sha256"] = "invalid"
    elif fault == "unsafe_path":
        asset["server_path"] = "../control.png"
    elif fault == "missing_file":
        path.unlink()
    elif fault == "changed_file":
        path.write_bytes(b"changed")
    elif fault == "bad_dimensions":
        asset["width"] = True
    else:
        value[api.FIELD]["items"][0]["control_net_name"] = ""
    before = deepcopy(value)
    with pytest.raises((ValueError, FileNotFoundError)):
        api.resolve_controls(value, assets, asset_root=tmp_path)
    assert value == before


def test_actual_bytes_verification_and_metadata_only_status_are_not_confused(tmp_path):
    api = module()
    value, asset, _ = scene(tmp_path, kind="video")
    verified = api.resolve_controls(value, {asset["id"]: asset}, asset_root=tmp_path)[0]
    declared = api.resolve_controls(value, {asset["id"]: asset})[0]
    assert (
        verified["asset_content_verified"] is True
        and declared["asset_content_verified"] is False
    )
    assert verified["asset_sha256"] == asset["sha256"]
    assert verified["portable_cache_identity"] is False
    assert "source_fps" not in verified and verified["target_fps"] == 24


def test_source_crop_and_target_region_have_independent_stage_coordinate_spaces():
    api = module()
    item = row(
        region={"mode": "rect", "xyxy_normalized": [0.25, 0.25, 0.75, 0.75]},
        crop={
            "enabled": True,
            "xyxy_normalized": [0.1, 0.2, 0.9, 0.8],
            "fit": "contain",
            "flip_x": True,
        },
    )
    low = api.resolve_control_geometry(
        item, width=448, height=256, length=107, source_size=(941, 1672), stage="low"
    )
    high = api.resolve_control_geometry(
        item, width=896, height=512, length=107, source_size=(941, 1672), stage="high"
    )
    assert low["region"]["box_xyxy"] == [112.0, 64.0, 336.0, 192.0]
    assert high["region"]["box_xyxy"] == [224.0, 128.0, 672.0, 384.0]
    assert low["crop_plan"] == high["crop_plan"]
    assert low["crop_plan"]["source_size"] == [941, 1672]
    assert low["crop_plan"]["boxes"][0] == pytest.approx([94.1, 334.4, 846.9, 1337.6])
    assert json.loads(low["crop_plan_json"]) == low["crop_plan"]
    assert low["crop_plan"]["flip_x"] is True


def test_full_region_and_no_crop_keep_the_original_apply_fit_route():
    api = module()
    actual = api.resolve_control_geometry(
        row(), width=64, height=32, length=22, source_size=(75, 93)
    )
    assert (
        actual["region"] is None
        and actual["crop_plan"] is None
        and actual["crop_plan_json"] == ""
    )


def test_virtual_source_crop_is_not_clamped_or_confused_with_target_bounds():
    api = module()
    item = row(
        crop={
            "enabled": True,
            "xyxy_normalized": [-0.5, -0.25, 1.5, 1.25],
            "fit": "stretch",
        }
    )
    actual = api.resolve_control_geometry(
        item, width=64, height=32, length=22, source_size=(80, 40)
    )
    assert actual["crop_plan"]["boxes"] == [[-40.0, -10.0, 120.0, 50.0]]


@pytest.mark.parametrize(
    "fault",
    [
        "region_outside",
        "region_reversed",
        "crop_reversed",
        "time",
        "source_start",
        "width",
        "length",
        "source_size",
    ],
)
def test_actual_geometry_rejects_real_invalid_inputs_without_clamping(fault):
    api = module()
    item = row()
    kw = dict(width=64, height=32, length=22, source_size=(80, 40))
    if fault == "region_outside":
        item["region"] = {"mode": "rect", "xyxy_normalized": [0, 0, 1.1, 1]}
    elif fault == "region_reversed":
        item["region"] = {"mode": "rect", "xyxy_normalized": [0.8, 0, 0.2, 1]}
    elif fault == "crop_reversed":
        item["crop"] = {"enabled": True, "xyxy_normalized": [1, 0, 0, 1]}
    elif fault == "time":
        item.update(start_percent=0.8, end_percent=0.2)
    elif fault == "source_start":
        item["source"] = {"start_seconds": -1}
    elif fault == "width":
        kw["width"] = 63
    elif fault == "length":
        kw["length"] = 96
    else:
        kw["source_size"] = (True, 40)
    with pytest.raises(ValueError):
        api.resolve_control_geometry(item, **kw)


def test_portable_export_remap_and_copy_are_complete_and_atomic_for_drafts():
    api = module()
    first, dormant = uid(), uid()
    value = shot(
        row(asset_id=first, enabled=True), row(asset_id=dormant, enabled=False)
    )
    value[api.FIELD]["private"] = "SECRET"
    value[api.FIELD]["items"][0]["private"] = "SECRET"
    value[api.FIELD]["items"][0]["region"] = {"mode": "full", "private": "SECRET"}
    before = deepcopy(value)
    portable = api.portable_controls(value[api.FIELD])
    assert "SECRET" not in json.dumps(portable) and value == before
    with pytest.raises(KeyError):
        api.remap_controls(value, {first: uid()}, lambda kind, old: uid())
    assert value == before
    mapping = {first: uid(), dormant: uid()}
    tags = []

    def fresh(kind, old):
        tags.append((kind, old))
        return uid()

    api.remap_controls(value, mapping, fresh)
    assert [item["asset_id"] for item in value[api.FIELD]["items"]] == [
        mapping[first],
        mapping[dormant],
    ]
    assert all(
        new["id"] != old["id"]
        for new, old in zip(
            value[api.FIELD]["items"], before[api.FIELD]["items"], strict=True
        )
    )
    assert [kind for kind, _ in tags] == ["control", "control"]
    assert value[api.FIELD]["items"][1]["enabled"] is False


def test_pure_control_module_import_does_not_load_torch_or_models():
    runtime = Path(__file__).resolve().parents[1] / "h3_t8"
    code = (
        "import sys,types; p=types.ModuleType('pure'); p.__path__=[sys.argv[1]]; "
        "sys.modules['pure']=p; import pure.director_controls; "
        "assert 'torch' not in sys.modules; print('pure controls passed')"
    )
    actual = subprocess.run(
        [sys.executable, "-B", "-X", "utf8", "-c", code, str(runtime)],
        capture_output=True,
        text=True,
        encoding="utf8",
        timeout=30,
    )
    assert actual.returncode == 0, actual.stdout + actual.stderr


def test_remap_collision_never_partially_overwrites_the_original_ids():
    api = module()
    value = shot(row(), row())
    before, collision = deepcopy(value), uid()
    with pytest.raises(ValueError, match="重复 UUID"):
        api.remap_controls(value, {}, lambda kind, old: collision)
    assert value == before


@pytest.mark.parametrize(
    "relative",
    [
        ".",
        "C:/input/control.png",
        "folder/../../control.png",
        "//server/share/control.png",
    ],
)
def test_unsafe_or_empty_source_paths_are_rejected_even_without_file_verification(
    tmp_path, relative
):
    api = module()
    value, asset, _ = scene(tmp_path)
    asset["server_path"] = relative
    with pytest.raises(ValueError):
        api.resolve_controls(value, {asset["id"]: asset})
