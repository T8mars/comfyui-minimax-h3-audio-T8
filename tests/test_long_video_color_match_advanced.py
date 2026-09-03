from __future__ import annotations

import json
from pathlib import Path

import pytest
from safetensors import safe_open
import torch

import h3_audio_t8_pkg.long_video_color_match_advanced as color_match
from h3_audio_t8_pkg.long_video_color_match_advanced import (
    process_long_video_color_match,
)
from h3_audio_t8_pkg.nodes_long_video_color_match_advanced import (
    MiniMaxH3LongVideoColorMatchT8Advanced,
)


def _context(chain_id: str, segment_index: int) -> dict:
    if segment_index == 0:
        return {
            "schema": 1,
            "empty": True,
            "chain_id": chain_id,
            "target_segment_index": 0,
        }
    return {
        "schema": 1,
        "empty": False,
        "metadata": {
            "chain_id": chain_id,
            "source_segment_index": segment_index - 1,
            "target_segment_index": segment_index,
        },
    }


def _frames(value: float, count: int = 32, channels: int = 3) -> torch.Tensor:
    grid = torch.arange(16 * 24, dtype=torch.float32).reshape(16, 24)
    texture = ((grid % 11) / 1000.0)[..., None]
    rgb = (torch.full((16, 24, 3), value) + texture).clamp(0.0, 1.0)
    result = rgb.repeat(count, 1, 1, 1)
    if channels > 3:
        auxiliary = torch.full((count, 16, 24, channels - 3), 0.73)
        result = torch.cat((result, auxiliary), dim=-1)
    return result


@pytest.fixture
def isolated_state(tmp_path: Path, monkeypatch):
    def state_path(_chain_id: str, source_segment_index: int) -> Path:
        return tmp_path / f"segment_{source_segment_index:05d}.context.safetensors"

    monkeypatch.setattr(color_match, "context_state_path", state_path)
    return tmp_path


def test_segment_zero_is_exact_identity_and_writes_bounded_rgb_tail(isolated_state):
    frames = _frames(0.40, count=12)
    output, status, report_json = process_long_video_color_match(
        frames,
        _context("color-chain", 0),
        "color-chain",
        0,
    )
    report = json.loads(report_json)
    state = isolated_state / "segment_00000.color.safetensors"

    assert output is frames
    assert torch.equal(output, frames)
    assert status == "REFERENCE_INITIALIZED"
    assert report["source_identity"] is True
    assert report["audio_touched"] is False
    assert report["latent_touched"] is False
    assert state.is_file()
    with safe_open(str(state), framework="pt", device="cpu") as handle:
        assert set(handle.keys()) == {
            "tail_rgb_means",
            "tail_rgb_thumbnail",
            "tail_lab_mean",
            "tail_lab_std",
        }
        assert tuple(handle.get_tensor("tail_rgb_means").shape) == (5, 3)
        assert tuple(handle.get_tensor("tail_rgb_thumbnail").shape) == (3, 5, 8)
        assert tuple(handle.get_tensor("tail_lab_mean").shape) == (3,)
        assert tuple(handle.get_tensor("tail_lab_std").shape) == (3,)
        metadata = dict(handle.metadata() or {})
    assert metadata["chain_id"] == "color-chain"
    assert metadata["source_segment_index"] == "0"
    assert metadata["schema"] == "2"


def test_default_match_removes_head_mean_jump_then_fades_to_source(isolated_state):
    first = _frames(0.40, count=32, channels=4)
    process_long_video_color_match(
        first,
        _context("color-chain", 0),
        "color-chain",
        0,
    )
    continuation = _frames(0.43, count=32, channels=4)
    output, status, report_json = process_long_video_color_match(
        continuation,
        _context("color-chain", 1),
        "color-chain",
        1,
    )
    report = json.loads(report_json)

    assert status == "COLOR_MATCH_APPLIED"
    assert output is not continuation
    assert report["maximum_rgb_jump_before"] > 0.02
    assert report["maximum_rgb_jump_after"] < report["maximum_rgb_jump_before"]
    assert report["maximum_rgb_jump_after"] <= 0.010001
    assert max(abs(value) for value in report["applied_rgb_offset"]) <= 0.020001
    assert report["method"] == "bounded_uniform_reinhard_lab_spatial_rgb_with_fade"
    assert report["maximum_total_rgb_delta"] <= 0.020001
    assert torch.equal(output[..., 3:], continuation[..., 3:])
    assert not torch.equal(output[:24, ..., :3], continuation[:24, ..., :3])
    assert torch.equal(output[24:], continuation[24:])
    assert torch.isfinite(output).all()
    assert float(output.min()) >= 0.0 and float(output.max()) <= 1.0


def test_match_reduces_local_spatial_and_lab_distribution_jump(isolated_state):
    reference = _frames(0.38, count=32)
    reference[:, :, 12:, 0] += 0.14
    reference[:, :, 12:, 1] += 0.09
    reference[:, :, 12:, 2] += 0.04
    reference.clamp_(0.0, 1.0)
    process_long_video_color_match(
        reference,
        _context("spatial-chain", 0),
        "spatial-chain",
        0,
    )

    continuation = ((reference - 0.5) * 1.04 + 0.5).clamp(0.0, 1.0)
    continuation[:, :, :12, 0] += 0.012
    continuation[:, :, :12, 1] -= 0.008
    continuation[:, :, 12:, 0] -= 0.010
    continuation[:, :, 12:, 1] += 0.011
    continuation.clamp_(0.0, 1.0)
    output, status, report_json = process_long_video_color_match(
        continuation,
        _context("spatial-chain", 1),
        "spatial-chain",
        1,
    )
    report = json.loads(report_json)

    assert status == "COLOR_MATCH_APPLIED"
    assert report["maximum_spatial_rgb_jump_after"] < report[
        "maximum_spatial_rgb_jump_before"
    ]
    assert report["maximum_rgb_jump_after"] < report["maximum_rgb_jump_before"]
    assert max(abs(value) for value in report["lab_mean_jump_after"]) < max(
        abs(value) for value in report["lab_mean_jump_before"]
    )
    assert max(abs(value - 1.0) for value in report["lab_std_ratio_after"]) < max(
        abs(value - 1.0) for value in report["lab_std_ratio_before"]
    )
    assert report["spatial_grid"] == [5, 8]
    assert report["lab_scale_bounds"] == [0.85, 1.18]
    assert report["maximum_total_rgb_delta"] <= 0.020001
    assert torch.equal(output[24:], continuation[24:])


def test_disabled_is_exact_source_identity_but_records_actual_tail(isolated_state):
    frames = _frames(0.42, count=8)
    output, status, report_json = process_long_video_color_match(
        frames,
        _context("disabled-chain", 1),
        "disabled-chain",
        1,
        enabled=False,
    )
    report = json.loads(report_json)

    assert output is frames
    assert torch.equal(output, frames)
    assert status == "DISABLED_SOURCE_IDENTITY"
    assert report["enabled"] is False
    assert (isolated_state / "segment_00001.color.safetensors").is_file()


def test_enabled_continuation_requires_the_preceding_color_state(isolated_state):
    with pytest.raises(FileNotFoundError, match="Run segment 0"):
        process_long_video_color_match(
            _frames(0.4),
            _context("missing-chain", 1),
            "missing-chain",
            1,
        )


def test_scene_cut_abstains_without_modifying_pixels(isolated_state):
    first = _frames(0.1)
    process_long_video_color_match(first, _context("cut-chain", 0), "cut-chain", 0)
    cut = _frames(0.8)
    output, status, report_json = process_long_video_color_match(
        cut,
        _context("cut-chain", 1),
        "cut-chain",
        1,
    )
    report = json.loads(report_json)

    assert output is cut
    assert torch.equal(output, cut)
    assert status == "ABSTAIN_SCENE_CUT_OR_LARGE_COLOR_JUMP"
    assert report["applied"] is False


def test_context_binding_and_sdr_contract_fail_closed(isolated_state):
    wrong = _context("wrong-chain", 1)
    with pytest.raises(ValueError, match="binding failed"):
        process_long_video_color_match(
            _frames(0.4), wrong, "right-chain", 1, enabled=False
        )

    invalid = _frames(0.4)
    invalid[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite SDR"):
        process_long_video_color_match(
            invalid,
            _context("invalid-chain", 0),
            "invalid-chain",
            0,
        )


def test_node_contract_is_optional_default_on_and_post_decode_image_based():
    schema = MiniMaxH3LongVideoColorMatchT8Advanced.define_schema()
    inputs = {item.id: item for item in schema.inputs}

    assert schema.node_id == "MiniMaxH3LongVideoColorMatchT8Advanced"
    assert inputs["enabled"].default is True
    assert inputs["reference_frames"].default == 5
    assert inputs["transition_frames"].default == 24
    assert [output.id for output in schema.outputs] == ["frames", "status", "report_json"]
