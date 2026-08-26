from __future__ import annotations

import json

import pytest
import torch

from h3_audio_t8_pkg.skin_finish_vretoucher_adapter import VRetouchAdapterUnavailable
from h3_audio_t8_pkg.skin_finish_vretoucher_pipeline import (
    VRETOUCHER_PIPELINE_SCHEMA,
    VRetoucherWindowProcessor,
    VRetouchPipelineUnavailable,
)
from h3_audio_t8_pkg.skin_finish_vretoucher_runtime import VRetoucherRuntimeSession


def _source(frame_count: int = 8) -> torch.Tensor:
    frames = torch.zeros((frame_count, 48, 80, 4), dtype=torch.float32)
    for index in range(frame_count):
        frames[index, ..., 0] = 0.15 + index * 0.01
        frames[index, ..., 1] = 0.25
        frames[index, ..., 2] = 0.35
        frames[index, ..., 3] = 0.73
    return frames


class _BrightNewestModel(torch.nn.Module):
    def __init__(self, *, fail: bool = False):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.ones(1))
        self.fail = fail
        self.calls = 0
        self.last_inputs: list[torch.Tensor] | None = None

    def forward(self, inputs: list[torch.Tensor]):
        self.calls += 1
        self.last_inputs = [item.detach().clone() for item in inputs]
        if self.fail:
            raise RuntimeError("controlled model failure")
        result = torch.ones_like(inputs[-1])
        masks = [torch.ones((1, 1, 256, 256)) for _ in range(6)]
        flows = torch.zeros((5, 1, 64, 64, 2))
        return result, masks, flows


def _pipeline(model: torch.nn.Module) -> VRetoucherWindowProcessor:
    session = VRetoucherRuntimeSession(
        model,
        {"status": "TEST_MODEL", "checkpoint_loaded": False},
    )
    return VRetoucherWindowProcessor(session)


def _process(processor: VRetoucherWindowProcessor, frames: torch.Tensor):
    semantic = torch.zeros((48, 80), dtype=torch.float32)
    semantic[13:37, 27:49] = 1.0
    person = torch.zeros((48, 80), dtype=torch.float32)
    person[8:44, 20:60] = 1.0
    return processor.process(
        frames,
        current_frame=2,
        shot_start=0,
        shot_end=7,
        track_key="0:0",
        frame_track_keys=["0:0"] * 8,
        face_boxes=[[24.0, 10.0, 52.0, 40.0]] * 8,
        semantic_skin_mask=semantic,
        person_mask=person,
        feather_px=4,
    )


def test_single_window_pipeline_is_causal_current_only_and_source_safe():
    frames = _source()
    source_before = frames.clone()
    model = _BrightNewestModel()
    processor = _pipeline(model)
    with processor:
        output, effective, report_json = _process(processor, frames)
    report = json.loads(report_json)
    assert report["schema"] == VRETOUCHER_PIPELINE_SCHEMA
    assert report["context_indices"] == [0, 0, 0, 0, 1, 2]
    assert report["current_frame"] == 2
    assert report["current_frame_only"] is True
    assert report["source_batch_mutated"] is False
    assert report["automatic_accept"] is False
    assert report["candidate_selected"] is False
    assert report["quality_validated"] is False
    assert report["audio_touched"] is False
    assert model.calls == 1
    assert len(model.last_inputs) == 6
    assert all(tuple(item.shape) == (1, 3, 512, 512) for item in model.last_inputs)
    assert all(torch.equal(model.last_inputs[0], item) for item in model.last_inputs[1:4])
    assert not torch.equal(model.last_inputs[3], model.last_inputs[4])
    assert not torch.equal(model.last_inputs[4], model.last_inputs[5])
    assert torch.equal(frames, source_before)
    outside = effective <= 0.0
    assert torch.equal(output[..., :3][outside], frames[2, ..., :3][outside])
    assert torch.equal(output[..., 3:], frames[2, ..., 3:])
    assert not torch.equal(output[..., :3][~outside], frames[2, ..., :3][~outside])
    assert processor.closed is True
    assert processor.close_report["object_still_referenced_elsewhere"] is True


def test_pipeline_closes_session_when_model_inference_raises():
    model = _BrightNewestModel(fail=True)
    processor = _pipeline(model)
    with pytest.raises(RuntimeError, match="controlled model failure"):
        with processor:
            _process(processor, _source())
    assert model.calls == 1
    assert processor.closed is True
    assert processor.close_report["owner_reference_cleared"] is True


def test_pipeline_fails_track_discontinuity_before_model_call_and_closes():
    frames = _source()
    model = _BrightNewestModel()
    processor = _pipeline(model)
    semantic = torch.ones((48, 80), dtype=torch.float32)
    keys = ["0:0"] * 8
    keys[1] = "0:1"
    with pytest.raises(VRetouchAdapterUnavailable) as error:
        with processor:
            processor.process(
                frames,
                current_frame=2,
                shot_start=0,
                shot_end=7,
                track_key="0:0",
                frame_track_keys=keys,
                face_boxes=[[24.0, 10.0, 52.0, 40.0]] * 8,
                semantic_skin_mask=semantic,
            )
    assert getattr(error.value, "status", "") == "ABSTAIN_TRACK_DISCONTINUITY"
    assert model.calls == 0
    assert processor.closed is True


def test_closed_pipeline_rejects_before_allocating_context():
    processor = _pipeline(_BrightNewestModel())
    processor.close()
    with pytest.raises(VRetouchPipelineUnavailable) as error:
        _process(processor, _source())
    assert error.value.status == "ABSTAIN_VRETOUCHER_PIPELINE_CLOSED"
