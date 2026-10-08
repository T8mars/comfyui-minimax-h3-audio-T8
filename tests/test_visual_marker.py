"""Source-image/text contracts, not GPU spatial-control or quality proof."""
from dataclasses import replace
import json

import pytest
import torch

from h3_audio_t8_pkg.nodes_visual_marker import NODES
from h3_audio_t8_pkg.visual_marker import (
    DEFAULT_MARKERS, compose_marker_text, parse_markers, prepare_marker_prompt, prepare_markers,
)


def source():
    return torch.linspace(0, 1, 64 * 96 * 3).reshape(1, 64, 96, 3)


def hand_spec():
    value = json.loads(DEFAULT_MARKERS)
    for marker in value["markers"]:
        marker.pop("xyxy")
    return json.dumps(value)


def test_render_deterministic_preserves_source_and_reports_actual_rectangle():
    image = source()
    before = image.clone()
    marked, plan, report = prepare_markers(image, DEFAULT_MARKERS)
    another, _, other_report = prepare_markers(image, DEFAULT_MARKERS)
    assert torch.equal(image, before) and torch.equal(marked, another)
    assert not torch.equal(marked, image) and report == other_report
    assert plan.verify() == report
    assert report["binding_verified"] is False and report["native_picture_ordinal"] is None
    assert report["source_dimensions"] == [96, 64]
    assert report["pixel_rectangles"][0]["source_pixel_xyxy"] == [14, 7, 44, 58]
    assert torch.equal(marked[0, 7, 14], torch.tensor([0., 223 / 255., 1.]))
    assert report["clean_rgb"] == report["source_rgb"]
    assert marked.data_ptr() != image.data_ptr() and plan.clean_image.data_ptr() != image.data_ptr()


def test_hand_drawn_image_exact_no_coordinates_no_fictitious_clean_source():
    image = source()
    marked, plan, report = prepare_markers(image, hand_spec(), mode="provided_marked")
    assert torch.equal(image, marked) and report["source_rgb"] == report["marked_rgb"]
    assert plan.clean_image is None and report["clean_rgb"] is None
    assert all(x["xyxy"] is None for x in report["spec"]["markers"])
    assert all(x["source_pixel_xyxy"] is None for x in report["pixel_rectangles"])
    assert report["line_width"] is None and report["pixel_rounding"] is None


def test_same_color_actor_target_and_multiple_actors_shared_target_are_legal():
    spec = json.loads(DEFAULT_MARKERS)
    spec["markers"].append({**spec["markers"][0], "marker_id": "actor_B", "role_id": "B"})
    spec["relations"].append({**spec["relations"][0], "actor_marker_id": "actor_B"})
    _, plan, _ = prepare_markers(source(), json.dumps(spec))
    full, _, actions, _, _ = prepare_marker_prompt(plan, "Keep the scene.")
    assert "Visual role A" in actions and "Visual role B" in actions
    assert "target_A" in full and len(plan.verify()["spec"]["relations"]) == 2


@pytest.mark.parametrize("coordinates", [
    [0, 0, 0, 1], [0, 0, 1, 0], [0.5, 0, 0.4, 1], [-.1, 0, .2, .2],
    [0, 0, 1.1, 1], [0, 0, float("nan"), 1], [0, 0, float("inf"), 1],
    [False, 0, 1, 1], [0, 0, 1], "0,0,1,1",
])
def test_bad_geometry_rejected_not_silently_clamped(coordinates):
    spec = json.loads(DEFAULT_MARKERS)
    spec["markers"][0]["xyxy"] = coordinates
    with pytest.raises(ValueError):
        parse_markers(json.dumps(spec), "render_rectangles")


@pytest.mark.parametrize("fault", ["duplicate_id", "missing_role", "unknown_field", "bad_color", "missing_relation", "wrong_target"])
def test_real_definition_errors_rejected(fault):
    spec = json.loads(DEFAULT_MARKERS)
    if fault == "duplicate_id":
        spec["markers"][1]["marker_id"] = "actor_A"
    elif fault == "missing_role":
        spec["markers"][0].pop("role_id")
    elif fault == "unknown_field":
        spec["markers"][0]["hidden_strength"] = .5
    elif fault == "bad_color":
        spec["markers"][0]["color"] = "#xyzxyz"
    elif fault == "missing_relation":
        spec["relations"] = []
    else:
        spec["relations"][0]["target_marker_id"] = "actor_A"
    with pytest.raises(ValueError):
        prepare_markers(source(), json.dumps(spec))


def test_duplicate_json_oversize_and_missing_render_geometry_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        parse_markers('{"markers":[],"markers":[],"relations":[]}', "provided_marked")
    with pytest.raises(ValueError, match="bounded"):
        parse_markers(" " * 65537, "provided_marked")
    with pytest.raises(ValueError, match="rendering needs xyxy"):
        prepare_markers(source(), hand_spec())


@pytest.mark.parametrize("fault", ["nonfinite", "range", "dtype", "batch", "channels"])
def test_invalid_rgb_before_render(fault):
    image = source()
    if fault == "nonfinite":
        image[0, 0, 0, 0] = float("nan")
    elif fault == "range":
        image[0, 0, 0, 0] = -1
    elif fault == "dtype":
        image = image.half()
    elif fault == "batch":
        image = image.repeat(2, 1, 1, 1)
    else:
        image = image[..., :2]
    with pytest.raises(ValueError):
        prepare_markers(image, DEFAULT_MARKERS)


def test_full_prompt_preserves_original_dialogue_bytes_and_separate_recipe():
    _, plan, _ = prepare_markers(source(), DEFAULT_MARKERS)
    original_record = plan.record_json
    base = '  雨天。\r\n<d>你好，今天真不错。</d>\nKeep native audio.  '
    full, legend, actions, recipe, report = prepare_marker_prompt(plan, base)
    assert full.startswith(base + "\n\n") and recipe.base_prompt.encode() == base.encode()
    assert full.count("<d>") == 1 and "<Picture 1>" in legend
    assert "walk to" in actions and plan.record_json == original_record
    assert not report["binding_verified"] and not report["automatic_translation"]
    assert not report["automatic_dialogue_or_audio_rules"]
    bound_full, bound_legend, _ = compose_marker_text(recipe, actual_picture_ordinal=3)
    assert bound_full.startswith(base + "\n\n") and "<Picture 3>" in bound_legend
    assert recipe.picture_ordinal == 1 and recipe.verify()["binding_verified"] is False


def test_manual_original_is_not_rewritten_or_given_audio_camera_rules():
    _, plan, _ = prepare_markers(source(), hand_spec(), mode="provided_marked")
    base = "Unchanged author prompt.\r\n"
    full, _, _, recipe, report = prepare_marker_prompt(plan, base, mode="manual_original")
    assert full == base and compose_marker_text(recipe, actual_picture_ordinal=4)[0] == base
    assert report["manual_original"]


def test_actual_pixels_and_independent_recipe_mutation_rejected():
    _, plan, _ = prepare_markers(source(), DEFAULT_MARKERS)
    recipe = prepare_marker_prompt(plan, "Keep the scene.")[3]
    with pytest.raises(ValueError, match="recipe changed"):
        replace(recipe, base_prompt="Different.").verify()
    plan.marked_image[0, 0, 0, 0] += .1
    with pytest.raises(ValueError, match="pixels changed"):
        recipe.verify()
    _, clean_plan, _ = prepare_markers(source(), DEFAULT_MARKERS)
    clean_plan.clean_image[0, 0, 0, 0] += .1
    with pytest.raises(ValueError, match="clean source changed"):
        clean_plan.verify()


def test_nodes_execute_without_model_or_detection_and_schema_has_distinct_outputs():
    prepared = NODES[0].execute(source(), markers_json=DEFAULT_MARKERS)
    result = NODES[1].execute(prepared[1], "A normal scene.")
    assert result[0].startswith("A normal scene.")
    assert not json.loads(prepared[2])["binding_verified"]
    for cls in NODES:
        info = cls.define_schema()
        assert info.node_id == cls.__name__ and info.is_experimental
    assert len(NODES[1].define_schema().outputs) == 5
