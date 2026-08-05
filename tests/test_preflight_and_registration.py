from __future__ import annotations

import asyncio
import json
from pathlib import Path

import torch

import h3_audio_t8_pkg
from h3_audio_t8_pkg.preflight import run_preflight
from helpers import FakeAudioVAE, FakeVideoVAE, make_audio


def test_all_nodes_register_with_unique_ids_and_valid_schemas():
    extension = h3_audio_t8_pkg.comfy_entrypoint()
    node_classes = asyncio.run(extension.get_node_list())
    schemas = [node.define_schema() for node in node_classes]
    ids = [schema.node_id for schema in schemas]
    assert len(ids) == 9
    assert len(ids) == len(set(ids))
    assert "MiniMaxH3AudioConditioningT8" in ids


def test_preflight_reports_alignment_audio_and_reference_guidance():
    ready, warning_count, report = run_preflight(
        1344, 768, 123, "lock_source", video_vae=FakeVideoVAE(), audio_vae=FakeAudioVAE(),
        drive_audio=make_audio(1, value=0),
        ref_videos={"ref_video_1": torch.zeros((20, 32, 32, 3))},
    )
    data = json.loads(report)
    assert ready is True
    assert warning_count >= 3
    assert data["facts"]["aligned_frames"] == 124


def test_preflight_blocks_oversize_canvas_and_missing_drive_audio():
    ready, _, report = run_preflight(1408, 768, 124, "lock_source")
    assert ready is False
    assert len(json.loads(report)["errors"]) == 2


def test_example_api_workflow_is_valid_and_references_existing_nodes():
    path = Path(__file__).resolve().parents[1] / "examples" / "audio_lock_api.json"
    workflow = json.loads(path.read_text(encoding="utf-8"))
    custom_types = {value["class_type"] for value in workflow.values() if value["class_type"].endswith("T8")}
    assert custom_types == {
        "MiniMaxH3AudioWindowT8", "MiniMaxH3AudioConditioningT8",
        "MiniMaxH3AVDecodeT8", "MiniMaxH3OutputTrimT8",
    }
    node_ids = set(workflow)
    for node in workflow.values():
        for value in node["inputs"].values():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                assert value[0] in node_ids
