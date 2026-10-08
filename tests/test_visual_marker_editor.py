"""Source-bound editing/observation deltas; no old authoring suite replay."""
import json

import pytest
import torch

from h3_audio_t8_pkg.reference_package import tensor_record
from h3_audio_t8_pkg.visual_marker import DEFAULT_MARKERS, prepare_markers


def test_editor_actual_source_sha_matches_and_stale_preview_rejects():
    source = torch.full((1, 32, 64, 3), .5)
    sha = tensor_record(source)["sha256"]
    _, plan, report = prepare_markers(source, DEFAULT_MARKERS, expected_source_sha256=sha)
    assert plan.verify() == report and report["editor_preview_source_verified"]
    with pytest.raises(ValueError, match="preview and actual IMAGE differ"):
        prepare_markers(source + .1, DEFAULT_MARKERS, expected_source_sha256=sha)
    with pytest.raises(ValueError, match="preview and actual IMAGE differ"):
        prepare_markers(source, DEFAULT_MARKERS, expected_source_sha256="not a digest")
    _, _, manual = prepare_markers(source + .1, DEFAULT_MARKERS, expected_source_sha256="")
    assert not manual["editor_preview_source_verified"]


def test_handdrawn_missing_geometry_unobserved_not_detected_or_rendered():
    spec = json.loads(DEFAULT_MARKERS)
    for marker in spec["markers"]:
        marker.pop("xyxy")
    _, _, report = prepare_markers(torch.zeros(1, 32, 64, 3), json.dumps(spec), mode="provided_marked")
    assert all(item["geometry_observation"] == "unobserved" for item in report["pixel_rectangles"])


def test_overlapping_same_color_actors_warn_without_banning_shared_target():
    spec = json.loads(DEFAULT_MARKERS)
    spec["markers"].append({**spec["markers"][0], "marker_id": "actor_B", "role_id": "B"})
    spec["relations"].append({**spec["relations"][0], "actor_marker_id": "actor_B"})
    _, _, report = prepare_markers(torch.zeros(1, 32, 64, 3), json.dumps(spec))
    assert any("share a color" in message for message in report["warnings"])
    assert any("Overlapping" in message for message in report["warnings"])
