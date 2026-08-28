from __future__ import annotations

import json
from pathlib import Path

import torch

from h3_audio_t8_pkg import realbasicvsr_advanced as module


class _FakeRealBasicVSR(torch.nn.Module):
    def forward(self, value):
        batch, count, channels, height, width = value.shape
        flat = value.reshape(batch * count, channels, height, width)
        output = torch.nn.functional.interpolate(
            flat, scale_factor=4, mode="bilinear", align_corners=False
        )
        return output.reshape(batch, count, channels, height * 4, width * 4)


def test_window_ranges_cover_all_frames_without_duplicates_at_edges():
    assert module._window_ranges(17, 8, 2) == [(0, 8), (6, 14), (9, 17)]
    assert module._window_ranges(6, 8, 2) == [(0, 6)]


def test_restore_preserves_exact_audio_object_and_native_shape(monkeypatch):
    model = _FakeRealBasicVSR()
    monkeypatch.setattr(
        module,
        "_load_model",
        lambda *args, **kwargs: (model, torch.device("cpu"), torch.float32, False),
    )
    monkeypatch.setattr(
        module,
        "_release_model",
        lambda *args, **kwargs: {"policy": "offload_after", "device": "cpu"},
    )
    frames = torch.rand(9, 24, 32, 3)
    audio = {"waveform": torch.rand(1, 2, 128), "sample_rate": 32000}
    restored, source, audio_out, report_json = module.restore_realbasicvsr(
        frames,
        audio,
        model_path=Path("fake.pth"),
        model_name="fake.pth",
        output_mode="native_size_restore",
        strength=0.5,
        chunk_frames=4,
        overlap_frames=1,
        precision="fp32",
        checkpoint_branch="prefer_ema",
        release_policy="offload_after",
    )
    report = json.loads(report_json)
    assert restored.shape == frames.shape
    assert torch.equal(source, frames)
    assert audio_out is audio
    assert report["audio"]["exact_object_passthrough"] is True
    assert report["temporal_windows"]["ranges"] == [[0, 4], [3, 7], [5, 9]]


def test_restore_x4_mode_and_strength_zero_are_source_bicubic(monkeypatch):
    model = _FakeRealBasicVSR()
    monkeypatch.setattr(
        module,
        "_load_model",
        lambda *args, **kwargs: (model, torch.device("cpu"), torch.float32, True),
    )
    monkeypatch.setattr(module, "_release_model", lambda *args, **kwargs: {})
    frames = torch.rand(2, 16, 20, 4)
    restored, _, _, report_json = module.restore_realbasicvsr(
        frames,
        None,
        model_path=Path("fake.pth"),
        model_name="fake.pth",
        output_mode="x4_super_resolution",
        strength=0.0,
        chunk_frames=2,
        overlap_frames=0,
        precision="fp32",
        checkpoint_branch="prefer_generator",
        release_policy="keep_loaded",
    )
    report = json.loads(report_json)
    assert restored.shape == (2, 64, 80, 4)
    assert report["output"]["width"] == 80
    assert report["cache_hit"] is True


def test_unwrap_prefers_ema_without_filename_hash_or_size_gate():
    payload = {
        "state_dict": {
            "generator.conv.weight": torch.tensor([1.0]),
            "generator_ema.conv.weight": torch.tensor([2.0]),
            "step_counter": torch.tensor(3),
        }
    }
    state = module._unwrap_state_dict(payload, "prefer_ema")
    assert torch.equal(state["conv.weight"], torch.tensor([2.0]))
    assert "step_counter" not in state


def test_feature_manifest_registers_realbasicvsr():
    root = Path(__file__).resolve().parents[1]
    features = json.loads((root / "features.json").read_text(encoding="utf-8"))
    assert features["realbasicvsr_temporal_restore_advanced"]["position"] == 240
    assert features["nodes"][240] == "MiniMaxH3RealBasicVSRRestoreT8Advanced"


def test_frontend_workflow_is_native_and_documents_audio_passthrough():
    root = Path(__file__).resolve().parents[1]
    path = (
        root
        / "examples"
        / "workflows"
        / "07-motion-detail"
        / "2026-08-28_H3_RealBasicVSR_Temporal_Restore_Advanced_EXP.json"
    )
    workflow = json.loads(path.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in workflow["nodes"]}
    types = {node["type"] for node in nodes.values()}
    assert workflow["version"] == 0.4
    assert workflow["last_node_id"] == max(nodes)
    assert "MiniMaxH3RealBasicVSRRestoreT8Advanced" in types
    assert {"LoadVideo", "GetVideoComponents", "CreateVideo", "SaveVideo", "MarkdownNote"} <= types
    note = next(node for node in nodes.values() if node["type"] == "MarkdownNote")
    assert "AUDIO" in note["widgets_values"]
    assert "不重采样" in note["widgets_values"]
