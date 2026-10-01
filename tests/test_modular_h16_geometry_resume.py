"""Frozen H16 learned geometry must match the cold-resume HIGH recipe."""

from copy import deepcopy
import json

import pytest

from h3_audio_t8_pkg.learned_latent_upscale_advanced import learned_upscale_geometry
from tools.build_modular_h16_geometry_resume import (
    PAIRS, corrected_pair, expected_high_geometry, source_pair,
)


@pytest.mark.parametrize("route", ["plain", "effects"])
def test_h16_resume_high_matches_production_learned_upscale_geometry(route):
    old_front, old_api, old_resume_front, old_resume_api = source_pair(route)
    high = learned_upscale_geometry(
        736 // 16, 416 // 16, "scale_by", 2.0, 1.0, 1280, 704,
        "preserve_source", 1.05)
    assert expected_high_geometry(old_api) == (high["output_width"],
                                               high["output_height"]) == (1472, 832)
    assert old_resume_api["14"]["inputs"]["width"] == 1344
    assert old_resume_api["14"]["inputs"]["height"] == 768
    fixed_front, fixed_api, resume_front, resume_api = corrected_pair(route)
    assert fixed_front == old_front and fixed_api == old_api
    assert resume_front != old_resume_front and resume_api != old_resume_api
    expected_front = deepcopy(old_resume_front)
    expected_api = deepcopy(old_resume_api)
    for node_id in PAIRS[route]["high_conditioners"]:
        front = next(node for node in expected_front["nodes"] if node["id"] == node_id)
        width_slot, height_slot = (1, 2) if node_id == 14 else (0, 1)
        front["widgets_values"][width_slot] = 1472
        front["widgets_values"][height_slot] = 832
        expected_api[str(node_id)]["inputs"].update(width=1472, height=832)
    assert resume_front == expected_front
    assert resume_api == expected_api
    assert not any(node["class_type"] == "SamplerCustomAdvanced"
                   for node in resume_api.values())


def test_h16_geometry_rejects_changed_freeze_recipe():
    _front, api, _resume_front, _resume_api = source_pair("plain")
    api["13"]["inputs"]["scale_by"] = 1.5
    with pytest.raises(ValueError, match="LOW/upscaler geometry"):
        expected_high_geometry(api)


def test_h16_geometry_rejects_unknown_route():
    with pytest.raises(ValueError, match="plain or effects"):
        corrected_pair("other")


@pytest.mark.parametrize("route", ["plain", "effects"])
def test_h16_corrected_saved_candidates_match_builder_and_preserve_freeze_bytes(route):
    config = PAIRS[route]
    generated = corrected_pair(route)
    freeze, resume = config["names"]
    files = (f"{freeze}.json", f"{freeze}.api.json",
             f"{resume}.json", f"{resume}.api.json")
    for name, expected in zip(files, generated, strict=True):
        source_path = config["source"] / name
        target_path = config["target"] / name
        actual = json.loads(target_path.read_bytes())
        assert actual == expected
        if name.startswith(freeze):
            assert target_path.read_bytes() == source_path.read_bytes()
