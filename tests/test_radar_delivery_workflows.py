"""Actual schema-derived delivery graph; no model inference."""
from copy import deepcopy
import importlib
from pathlib import Path

import comfy.utils
import pytest

from tools.build_radar_delivery_workflows import FILENAMES, POST, SAFE, make_workflow
from tools.check_windows_paths import validate_paths


@pytest.fixture
def info(monkeypatch):
    core = Path(comfy.utils.__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(core))
    images = importlib.import_module("comfy_extras.nodes_images")
    video = importlib.import_module("comfy_extras.nodes_video")
    safe = importlib.import_module("h3_audio_t8_pkg.nodes_h3_av_delivery")
    post = importlib.import_module("h3_audio_t8_pkg.nodes_postprocess")
    classes = [images.ImageCropV2, images.ResizeAndPadImage, video.LoadVideo, video.GetVideoComponents,
               safe.MiniMaxH3SafeAVSaveT8Advanced, post.MiniMaxH3PostprocessSaveEXPT8]
    return {cls.define_schema().node_id: cls.GET_NODE_INFO_V1() for cls in classes}


def test_full_master_is_real_dependency_after_required_final_rgb_audio(info):
    workflow, prompt = make_workflow(info, cold=False)
    nodes = {node["id"]: node for node in workflow["nodes"]}
    assert nodes[1]["type"] == SAFE
    assert [(v["name"], v["link"]) for v in nodes[1]["inputs"][:2]] == [("images", None), ("audio", None)]
    assert [v["name"] for v in nodes[1]["inputs"]] == ["images", "audio"]
    assert nodes[1]["widgets_values"] == ["MiniMaxH3/Radar/master", 18]
    assert prompt["2"]["inputs"]["video"] == ["1", 0]
    assert prompt["4"]["inputs"]["master_video"] == ["1", 0]
    assert prompt["4"]["inputs"]["processed_frames"] == ["3", 0]
    assert prompt["4"]["inputs"]["confirm_postprocess"] is False
    assert all(item[0] != "audio" for item in prompt["4"]["inputs"].items())
    assert {node["class_type"] for node in prompt.values()} == {SAFE, "GetVideoComponents", "ImageCropV2", POST}
    assert len(workflow["links"]) == 4


def test_cold_core_boundingbox_serialization_is_not_a_fabricated_socket(info):
    workflow, prompt = make_workflow(info)
    crop = workflow["nodes"][2]
    assert crop["inputs"][1] == {"name": "crop_region", "type": "BOUNDING_BOX", "widget": {"name": "crop_region"}, "link": None}
    assert crop["widgets_values"] == [{"x": 0, "y": 4, "width": 1920, "height": 1080}, 0, 4, 1920, 1080]
    assert crop["widgets_values_named"]["crop_region"] == prompt["3"]["inputs"]["crop_region"]
    assert prompt["1"]["inputs"]["file"] == "SELECT_SAVED_MASTER.mp4"
    assert SAFE not in {node["class_type"] for node in prompt.values()}
    assert all(link[5] != "AUDIO" for link in workflow["links"])
    assert all("sampl" not in node["class_type"].lower() for node in prompt.values())
    modified = deepcopy(info)
    modified["ImageCropV2"]["input"]["required"]["crop_region"][1]["socketless"] = False
    with pytest.raises(ValueError, match="inspect native serialization"):
        make_workflow(modified)


def test_contain_is_an_explicit_other_policy_and_names_stay_short(info):
    workflow, prompt = make_workflow(info, contain=True)
    assert prompt["3"]["class_type"] == "ResizeAndPadImage"
    assert prompt["3"]["inputs"]["padding_color"] == "black"
    assert prompt["3"]["inputs"]["interpolation"] == "lanczos"
    assert workflow["extra"]["radar_r6_delivery"]["pixel_policy"] == "contain"
    assert workflow["extra"]["radar_r6_delivery"]["automatic_accept"] is False
    assert validate_paths(["examples/workflows/74-radar-r6-delivery/" + name for name in FILENAMES]) == []
    assert max(map(len, FILENAMES)) <= 64
