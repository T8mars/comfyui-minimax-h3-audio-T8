from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from h3_audio_t8_pkg.nodes_optical_flow_advanced import OPTICAL_FLOW_ADVANCED_NODE_CLASSES
from h3_audio_t8_pkg.optical_flow_advanced import (
    _parse_indices,
    _unwrap_state_dict,
    propagate_masks_with_flows,
)


def test_optical_flow_nodes_are_append_only_and_explicitly_experimental():
    schemas = [node.define_schema() for node in OPTICAL_FLOW_ADVANCED_NODE_CLASSES]
    assert [schema.node_id for schema in schemas] == [
        "MiniMaxH3RAFTMotionAuditT8Advanced",
        "MiniMaxH3RAFTMaskPropagationT8Advanced",
    ]
    assert all(schema.is_experimental for schema in schemas)
    assert all(
        schema.category == "T8/MiniMax H3/Quality/Experimental/Optical Flow"
        for schema in schemas
    )
    propagation = schemas[1]
    inputs = {item.id: item for item in propagation.inputs}
    assert inputs["pair_batch_size"].default == 1
    assert inputs["analysis_max_side"].default == 640
    assert inputs["release_policy"].default == "offload_after"


def test_state_dict_unwraps_common_training_prefix_without_identity_gate():
    tensor = torch.ones(1)
    state = _unwrap_state_dict({"state_dict": {"module.feature_encoder.weight": tensor}})
    assert list(state) == ["feature_encoder.weight"]
    assert state["feature_encoder.weight"] is tensor


def test_keyframe_indices_are_strict_and_bound_to_mask_batch():
    assert _parse_indices("0, 4,8", 9, 3) == [0, 4, 8]
    with pytest.raises(ValueError, match="strictly increasing"):
        _parse_indices("0,4,4", 9, 3)
    with pytest.raises(ValueError, match="MASK batch"):
        _parse_indices("0,4", 9, 3)
    with pytest.raises(ValueError, match="within"):
        _parse_indices("0,9", 9, 2)


def test_bidirectional_flow_moves_square_between_two_reviewed_anchors():
    height, width = 32, 48
    first = torch.zeros(height, width)
    last = torch.zeros(height, width)
    first[10:20, 6:16] = 1.0
    last[10:20, 14:24] = 1.0
    keyframes = torch.stack((first, last))
    forward = []
    backward = []
    for _ in range(2):
        fwd = torch.zeros(2, height, width)
        fwd[0] = 4.0
        forward.append(fwd)
        bwd = torch.zeros(2, height, width)
        bwd[0] = -4.0
        backward.append(bwd)

    masks, confidence = propagate_masks_with_flows(
        keyframe_masks=keyframes,
        keyframe_indices=[0, 2],
        forward=forward,
        backward=backward,
        scene_deltas=[0.0, 0.0],
        scene_cut_threshold=0.2,
        consistency_threshold=1.0,
        minimum_confidence=0.05,
        extend_edges=True,
    )

    assert masks.shape == (3, height, width)
    assert float(masks[1, 10:20, 10:20].mean()) > 0.95
    assert float(masks[1, :, :8].mean()) < 0.05
    assert float(confidence[1].max()) > 0.95


def test_scene_cut_prevents_mask_transport_without_inventing_identity():
    height, width = 24, 32
    anchor = torch.zeros(1, height, width)
    anchor[0, 8:16, 8:16] = 1.0
    zero_flow = [torch.zeros(2, height, width) for _ in range(2)]
    masks, confidence = propagate_masks_with_flows(
        keyframe_masks=anchor,
        keyframe_indices=[0],
        forward=zero_flow,
        backward=[flow.clone() for flow in zero_flow],
        scene_deltas=[0.5, 0.0],
        scene_cut_threshold=0.2,
        consistency_threshold=1.0,
        minimum_confidence=0.05,
        extend_edges=True,
    )
    assert torch.equal(masks[0], anchor[0])
    assert int(torch.count_nonzero(masks[1:])) == 0
    assert int(torch.count_nonzero(confidence[1:])) == 0


def test_feature_manifest_records_scientific_boundary():
    feature_path = Path(__file__).resolve().parents[1] / "features.json"
    features = json.loads(feature_path.read_text(encoding="utf-8"))
    feature = features["optical_flow_motion_advanced"]
    assert feature["positions"] == [236, 237]
    assert "SEA-RAFT" in feature["scientific_boundary"]
    assert features["nodes"][236:238] == [
        "MiniMaxH3RAFTMotionAuditT8Advanced",
        "MiniMaxH3RAFTMaskPropagationT8Advanced",
    ]
