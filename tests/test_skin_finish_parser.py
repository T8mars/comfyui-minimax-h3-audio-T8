from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from h3_audio_t8_pkg.face_refine_advanced import (
    PLAN_SCHEMA,
    canonical_json,
    source_proxy_sha256,
)
from h3_audio_t8_pkg.nodes_skin_finish_parser import (
    MiniMaxH3SkinFinishSemanticMaskT8Advanced,
)
from h3_audio_t8_pkg import skin_finish_parser as parser_module
from h3_audio_t8_pkg.skin_finish_parser import (
    PARSENET_CLASS_NAMES,
    PARSENET_MODEL_NAME,
    PARSENET_MODEL_SHA256,
    SKIN_FINISH_SEMANTIC_MASK_SCHEMA,
    _ParserUnavailable,
    _load_pinned_parsenet,
    _semantic_local_masks,
    run_semantic_skin_mask,
)


def _frames(frame_count: int = 2) -> torch.Tensor:
    generator = torch.Generator().manual_seed(260824)
    return torch.rand((frame_count, 64, 64, 3), generator=generator)


def _plan(frames: torch.Tensor) -> dict:
    records = [
        {
            "frame_index": index,
            "shot_id": 0,
            "state": "detected",
            "source_face_box_xyxy": [16.0, 8.0, 48.0, 56.0],
            "source_crop_box_xyxy": [4.0, 0.0, 60.0, 64.0],
            "crop_face_box_xyxy": [96.0, 64.0, 416.0, 448.0],
            "paste_weight": 1.0,
        }
        for index in range(int(frames.shape[0]))
    ]
    plan = {
        "schema": PLAN_SCHEMA,
        "status": "experimental_candidate_plan",
        "source": {
            "frame_count": int(frames.shape[0]),
            "width": int(frames.shape[2]),
            "height": int(frames.shape[1]),
            "fps": 24.0,
            "proxy_sha256": source_proxy_sha256(frames),
        },
        "canvas": {"width": 512, "height": 512, "multiple": 32},
        "frames": records,
        "shots": [{"shot_id": 0, "start_frame": 0, "end_frame": len(records) - 1}],
    }
    plan["plan_sha256"] = hashlib.sha256(canonical_json(plan).encode("utf-8")).hexdigest()
    return plan


def _semantic_logits() -> torch.Tensor:
    logits = torch.full((1, 19, 512, 512), -9.0)
    logits[:, 0] = 3.0
    logits[:, 1, 90:430, 90:422] = 12.0
    logits[:, 2, 230:310, 230:282] = 15.0
    logits[:, 4, 180:225, 145:215] = 15.0
    logits[:, 5, 180:225, 297:367] = 15.0
    logits[:, 6, 145:175, 135:220] = 15.0
    logits[:, 7, 145:175, 292:377] = 15.0
    logits[:, 11, 325:350, 205:307] = 15.0
    logits[:, 12, 350:375, 205:307] = 15.0
    logits[:, 13, 45:145, 75:437] = 15.0
    logits[:, 17, 430:485, 185:327] = 15.0
    return logits


class _FakeParseNet(torch.nn.Module):
    def forward(self, value):
        logits = _semantic_logits().expand(int(value.shape[0]), -1, -1, -1).clone()
        return logits, torch.zeros_like(value)


def test_parsenet_mapping_is_the_verified_parsenet_order_not_bisenet_order():
    assert PARSENET_CLASS_NAMES[1] == "skin"
    assert PARSENET_CLASS_NAMES[2] == "nose"
    assert PARSENET_CLASS_NAMES[4:8] == (
        "left_eye",
        "right_eye",
        "left_eyebrow",
        "right_eyebrow",
    )
    assert PARSENET_CLASS_NAMES[13] == "hair"
    assert PARSENET_CLASS_NAMES[17] == "neck"
    assert PARSENET_CLASS_NAMES[18] == "cloth"


def test_semantic_mask_excludes_features_and_neck_by_default():
    skin, feature, report = _semantic_local_masks(
        _semantic_logits(),
        include_neck=False,
        minimum_class_probability=0.55,
        feature_protection_px=3,
    )
    assert skin[0, 260, 256] == 0  # nose
    assert skin[0, 200, 180] == 0  # eye
    assert skin[0, 350, 256] == 0  # lip plus protection dilation
    assert skin[0, 460, 256] == 0  # neck disabled
    assert skin[0, 280, 150] == 1  # cheek skin
    assert feature[0, 260, 256] == 1
    assert report["class_pixel_counts"]["nose"] > 0
    assert report["class_pixel_counts"]["hair"] > 0


def test_semantic_parser_end_to_end_is_source_bound_and_releases_model(monkeypatch):
    frames = _frames()
    model = _FakeParseNet()
    monkeypatch.setattr(
        parser_module,
        "_load_pinned_parsenet",
        lambda: (model, Path("pinned/parsing_parsenet.pth"), PARSENET_MODEL_SHA256),
    )
    mask, preview, report_json = run_semantic_skin_mask(
        frames,
        _plan(frames),
        minimum_skin_area=0.0001,
        maximum_skin_area=0.9,
        preview_count=2,
    )
    report = json.loads(report_json)
    assert report["schema"] == SKIN_FINISH_SEMANTIC_MASK_SCHEMA
    assert report["status"] == "READY"
    assert report["selection"]["accepted_frame_count"] == 2
    assert report["model"]["loaded"] is True
    assert report["model"]["unloaded_after_execute"] is True
    assert report["mechanical_gates"]["network_access_performed"] is False
    assert report["mechanical_gates"]["cuda_used"] is False
    assert tuple(mask.shape) == (2, 64, 64)
    assert tuple(preview.shape) == (2, 64, 64, 3)
    assert float(mask.sum()) > 0.0
    assert torch.equal(frames, frames.clone())


def test_missing_model_abstains_to_an_empty_mask(monkeypatch):
    frames = _frames()

    def unavailable():
        raise _ParserUnavailable("ABSTAIN_PARSENET_MODEL_MISSING", "missing")

    monkeypatch.setattr(parser_module, "_load_pinned_parsenet", unavailable)
    mask, preview, report_json = run_semantic_skin_mask(frames, _plan(frames))
    report = json.loads(report_json)
    assert report["status"] == "ABSTAIN_PARSENET_MODEL_MISSING"
    assert report["selection"]["accepted_frame_count"] == 0
    assert report["selection"]["rejected_frame_count"] == 2
    assert torch.count_nonzero(mask) == 0
    assert torch.equal(preview, frames[..., :3])


def test_stale_face_plan_abstains_without_loading_model(monkeypatch):
    frames = _frames()
    plan = _plan(frames)
    plan["frames"][0]["source_face_box_xyxy"] = [0.0, 0.0, 8.0, 8.0]
    called = False

    def load():
        nonlocal called
        called = True
        return _FakeParseNet(), Path("unused"), PARSENET_MODEL_SHA256

    monkeypatch.setattr(parser_module, "_load_pinned_parsenet", load)
    mask, _, report_json = run_semantic_skin_mask(frames, plan)
    assert json.loads(report_json)["status"] == "ABSTAIN_FACE_PLAN_MISSING_OR_INVALID"
    assert called is False
    assert torch.count_nonzero(mask) == 0


def test_wrong_sized_checkpoint_is_rejected_before_pickle_load(tmp_path):
    path = tmp_path / "parsing_parsenet.pth"
    path.write_bytes(b"not a checkpoint")
    with pytest.raises(_ParserUnavailable) as caught:
        _load_pinned_parsenet(path)
    assert caught.value.status == "REJECT_PARSENET_MODEL_SIZE_MISMATCH"


def test_semantic_parser_schema_is_append_only_and_safe_by_default():
    schema = MiniMaxH3SkinFinishSemanticMaskT8Advanced.define_schema()
    inputs = {item.id: item for item in schema.inputs}
    assert schema.node_id == "MiniMaxH3SkinFinishSemanticMaskT8Advanced"
    assert schema.is_experimental is True
    assert inputs["parser_model"].default == PARSENET_MODEL_NAME
    assert inputs["include_neck"].default is False
    assert inputs["crop_expansion"].default == 1.45
    assert inputs["minimum_class_probability"].default == 0.55
