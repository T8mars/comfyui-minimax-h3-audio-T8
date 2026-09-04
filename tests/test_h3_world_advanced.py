from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from h3_audio_t8_pkg import h3_world_advanced as world
from h3_audio_t8_pkg.nodes_h3_world_advanced import H3_WORLD_ADVANCED_NODE_CLASSES


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_NAME = "2026-09-04_H3_World_I2VA_832x480_124f_50step_Advanced.json"
SOURCE_WORKFLOW = ROOT / "examples" / "workflows" / "26-h3-world" / WORKFLOW_NAME
USER_WORKFLOW = (
    ROOT.parents[1]
    / "user"
    / "default"
    / "workflows"
    / "MiniMax H3 T8"
    / "26-h3-world"
    / WORKFLOW_NAME
)


def test_forward_plan_is_exact_37_latent_contract():
    plan, script_json, report_json = world.compile_action_plan("forward", "[]")

    assert plan["width"] == 832
    assert plan["height"] == 480
    assert plan["frame_count"] == 124
    assert plan["latent_t"] == 37
    assert plan["keys"] == [["W"]] * 37
    assert json.loads(script_json) == [
        "the man walks forward, camera follows him"
    ] * 37
    assert json.loads(report_json)["status"] == "READY"
    assert world.validate_action_plan(plan) is plan


@pytest.mark.parametrize(
    ("preset", "expected"),
    [
        ("still", "the man stands still, camera holds steady"),
        ("forward", "the man walks forward, camera follows him"),
        ("back", "the man walks backward, camera follows him"),
        ("strafe-left", "the man strafes left, camera follows him"),
        ("strafe-right", "the man strafes right, camera follows him"),
        ("tilt-down", "the man stands still, camera tilts down"),
        ("tilt-up", "the man stands still, camera tilts up"),
        ("pan-left", "the man stands still, camera pans left slowly"),
        ("pan-right", "the man stands still, camera pans right slowly"),
        ("pan-left-fast", "the man stands still, camera pans left sharply"),
        ("pan-right-fast", "the man stands still, camera pans right sharply"),
    ],
)
def test_all_upstream_action_presets_use_exact_sentences(preset, expected):
    plan, _script, _report = world.compile_action_plan(preset, "[]")

    assert plan["script"] == [expected] * 37


def test_compound_action_uses_upstream_fixed_clause_order():
    assert world.annotation_for_keys(["L", "F", "A", "W"]) == (
        "the man walks forward and strafes left, camera pans right sharply"
    )


def test_custom_timeline_tiles_all_latents_and_cancels_opposites():
    source = json.dumps(
        [
            {"start_latent": 0, "end_latent": 10, "keys": ["W", "S"]},
            {"start_latent": 10, "end_latent": 20, "keys": ["J", "F"]},
            {"start_latent": 20, "end_latent": 37, "keys": ["A"]},
        ]
    )
    plan, _script, _report = world.compile_action_plan("custom", source)

    assert plan["keys"][0] == []
    assert plan["script"][0] == "the man stands still, camera holds steady"
    assert plan["script"][10] == "the man stands still, camera pans left sharply"
    assert plan["script"][20] == "the man strafes left, camera follows him"


def test_custom_timeline_rejects_gaps_and_fast_without_pan():
    with pytest.raises(ValueError, match="without gaps"):
        world.compile_action_plan(
            "custom",
            '[{"start_latent":1,"end_latent":37,"keys":["W"]}]',
        )
    with pytest.raises(ValueError, match="only valid together"):
        world.compile_action_plan(
            "custom",
            '[{"start_latent":0,"end_latent":37,"keys":["F"]}]',
        )


def test_action_plan_hash_and_script_are_fail_closed():
    plan, _script, _report = world.compile_action_plan("forward", "[]")
    plan["keys"][3] = ["D"]
    with pytest.raises(ValueError, match="hash mismatch"):
        world.validate_action_plan(plan)


def test_directed_mask_only_exposes_action_to_its_own_video_latent():
    static = (-1, -1)
    action0 = (0, -1)
    action1 = (1, -1)
    video0 = (-1, 0)
    video1 = (-1, 1)

    def allows(query, key):
        return world.directed_mask_allows(query[0], key[0], query[1], key[1])

    assert allows(action0, static)
    assert allows(action0, action0)
    assert not allows(action0, action1)
    assert allows(action0, video0)
    assert not allows(action0, video1)
    assert allows(video0, action0)
    assert not allows(video1, action0)
    assert not allows(static, action0)
    assert allows(video0, video1)
    assert allows(video1, video0)


def test_action_positions_use_mirrored_video_time_grid():
    head_end = 24
    spans = [(head_end + index * 8, head_end + (index + 1) * 8) for index in range(37)]
    text_len = spans[-1][1]
    keyframe = {
        "resolved_frame_index": 0,
        "latent": torch.zeros((1, 24, 1, 30, 52)),
    }
    layout = world.minimax_model.PackedLayout(
        text_len, 37, 30, 52, 208, keyframes=[keyframe]
    )

    world._repair_layout_for_actions(layout, spans, head_end, 37)

    grid = world.minimax_model._video_t_grid(37, 0.0)
    origin = float(text_len) - float(grid[-1]) - 1.0
    for index, (start, stop) in enumerate(spans):
        assert torch.all(layout.position_ids[start:stop, 0] == origin + float(grid[index]))
    assert origin >= head_end


def test_refiner_spans_must_tile_the_action_text_block():
    valid = [(10 + index * 2, 12 + index * 2) for index in range(37)]
    assert world._validate_action_spans(valid, 10, 84) == valid
    broken = list(valid)
    broken[5] = (21, 24)
    with pytest.raises(RuntimeError, match="must tile"):
        world._validate_action_spans(broken, 10, 84)


def test_first_frame_uses_upstream_scale_to_cover_center_crop():
    image = torch.zeros((1, 480, 1664, 3), dtype=torch.float32)
    image[:, :, :416, 0] = 1.0
    image[:, :, 416:1248, 1] = 1.0
    image[:, :, 1248:, 2] = 1.0

    prepared = world._prepare_first_frame(image)

    assert tuple(prepared.shape) == (1, 480, 832, 3)
    assert float(prepared[..., 1].mean()) > 0.999
    assert float(prepared[..., 0].abs().max()) < 0.001
    assert float(prepared[..., 2].abs().max()) < 0.001


def test_safe_output_is_single_thread_strict_and_atomic(monkeypatch, tmp_path):
    monkeypatch.setattr(world, "WIDTH", 8)
    monkeypatch.setattr(world, "HEIGHT", 6)
    monkeypatch.setattr(world, "FRAME_COUNT", 2)
    monkeypatch.setattr(world, "FPS", 2)
    calls = []

    def fake_encode(path, chunks_factory, **kwargs):
        chunks = list(chunks_factory())
        assert len(chunks) == 2
        assert all(len(chunk) == 8 * 6 * 3 for chunk in chunks)
        calls.append(("encode", kwargs))
        path.write_bytes(b"video")

    def fake_strict(path, *, require_audio=True):
        calls.append(("strict", require_audio, path.name))

    def fake_audio(path, array):
        assert array.shape == (2, 4)
        calls.append(("audio", array.shape))
        path.write_bytes(b"audio")

    def fake_mux(
        video_path, raw_audio_path, output_path, *, sample_rate, duration_seconds
    ):
        assert video_path.read_bytes() == b"video"
        assert raw_audio_path.read_bytes() == b"audio"
        assert duration_seconds == 1.0
        calls.append(("mux", sample_rate))
        output_path.write_bytes(b"combined")

    monkeypatch.setattr(world, "_encode_rgb_frames_isolated", fake_encode)
    monkeypatch.setattr(world, "_strict_validate_mp4", fake_strict)
    monkeypatch.setattr(world, "_write_planar_audio_raw", fake_audio)
    monkeypatch.setattr(world, "_mux_h3_world_audio", fake_mux)
    images = torch.zeros((2, 6, 8, 3), dtype=torch.float32)
    audio = {"waveform": torch.zeros((1, 1, 3)), "sample_rate": 4}
    target = tmp_path / "world.mp4"

    output, report = world.save_h3_world_video_safe(images, audio, target, fps=2)

    assert output == target.resolve()
    assert output.read_bytes() == b"combined"
    assert calls[0][0] == "encode"
    assert calls[1][0:2] == ("strict", False)
    assert calls[-1][0:2] == ("strict", True)
    assert report["status"] == "ATOMICALLY_PUBLISHED"
    assert report["strict_decode_validated"] is True
    assert report["audio"]["source_channels"] == 1
    assert report["audio"]["encoded_channels"] == 2
    assert not list(tmp_path.glob(".*.tmp"))


def test_safe_output_rejects_non_contract_geometry(tmp_path):
    images = torch.zeros((2, 6, 7, 3), dtype=torch.float32)
    audio = {"waveform": torch.zeros((1, 2, 4)), "sample_rate": 4}
    with pytest.raises(ValueError, match="requires exactly 832x480x124"):
        world.save_h3_world_video_safe(images, audio, tmp_path / "world.mp4")


def test_h3_world_nodes_are_four_append_only_formal_nodes():
    ids = [node.define_schema().node_id for node in H3_WORLD_ADVANCED_NODE_CLASSES]
    assert ids == [
        "MiniMaxH3WorldActionTimelineT8Advanced",
        "MiniMaxH3WorldModelComposerT8Advanced",
        "MiniMaxH3WorldI2VAConditioningT8Advanced",
        "MiniMaxH3WorldSafeVideoSaveT8Advanced",
    ]
    for node in H3_WORLD_ADVANCED_NODE_CLASSES:
        schema = node.define_schema()
        assert schema.is_experimental is False
        assert schema.category == "T8/MiniMax H3/World"


def test_h3_world_frontend_workflow_is_fixed_contract_and_mirrored():
    workflow = json.loads(SOURCE_WORKFLOW.read_text(encoding="utf-8"))
    nodes = {node["type"]: node for node in workflow["nodes"]}

    assert workflow["last_node_id"] == 14
    assert "MiniMaxH3WorldActionTimelineT8Advanced" in nodes
    assert "MiniMaxH3WorldModelComposerT8Advanced" in nodes
    assert "MiniMaxH3WorldI2VAConditioningT8Advanced" in nodes
    assert "MiniMaxH3WorldSafeVideoSaveT8Advanced" in nodes
    assert nodes["MiniMaxH3WorldActionTimelineT8Advanced"]["widgets_values"][0] == "forward"
    assert (
        nodes["MiniMaxH3WorldI2VAConditioningT8Advanced"]["widgets_values"][0]
        == "A man in a yellow floral shirt stands in a dim, multi-level concrete parking garage."
    )
    assert nodes["RandomNoise"]["widgets_values"][0] == 2
    assert nodes["MiniMaxH3DualClockSamplerT8"]["widgets_values"][:3] == [
        50,
        12.0,
        3.0,
    ]
    assert USER_WORKFLOW.read_bytes() == SOURCE_WORKFLOW.read_bytes()
    assert (ROOT / "examples" / "workflows" / "README.md").read_bytes() == (
        ROOT.parents[1]
        / "user"
        / "default"
        / "workflows"
        / "MiniMax H3 T8"
        / "README.md"
    ).read_bytes()
    assert (
        ROOT
        / "examples"
        / "workflows"
        / "26-h3-world"
        / "assets"
        / "h3_world_official_first_frame.png"
    ).is_file()
