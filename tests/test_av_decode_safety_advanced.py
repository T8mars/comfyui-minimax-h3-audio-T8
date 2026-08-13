from __future__ import annotations

import json

import pytest
import torch
from comfy.nested_tensor import NestedTensor

from h3_audio_t8_pkg.av_decode_safety_advanced import decode_av_safely, inspect_av_decode
from h3_audio_t8_pkg.core import empty_av_latent
from h3_audio_t8_pkg.nodes_av_decode_safety_advanced import (
    AV_DECODE_SAFETY_ADVANCED_NODE_CLASSES,
)


class FakeVideoModel:
    pass


class MiniMaxH3VideoVAE:
    tiling = True
    tile_size = 256


class MiniMaxH3AudioVAE:
    pass


class FakeVideoVAE:
    latent_channels = 24
    latent_dim = 3

    def __init__(self):
        self.first_stage_model = MiniMaxH3VideoVAE()
        self.decode_calls = 0
        self.tiled_calls = []

    def decode(self, latent):
        self.decode_calls += 1
        frames = 5 + 17 * ((latent.shape[2] - 2) // 5)
        return torch.zeros((1, frames, latent.shape[-2] * 16, latent.shape[-1] * 16, 3))

    def decode_tiled(self, latent, **kwargs):
        self.tiled_calls.append(kwargs)
        return self.decode(latent)


class FakeAudioVAE:
    latent_channels = 32
    latent_dim = 2
    audio_sample_rate = 32000
    audio_sample_rate_output = 32000

    def __init__(self):
        self.first_stage_model = MiniMaxH3AudioVAE()
        self.decode_calls = 0

    def decode(self, latent):
        self.decode_calls += 1
        samples = latent.shape[-1] * 800
        return torch.zeros((latent.shape[0], samples, 2))


def latent():
    return empty_av_latent(128, 128, 124)[0]


def runtime(free_mib=4096.0):
    return {
        "gpu": {"whole_device_free_mib": free_mib},
        "host": {"ram_available_gib": 64.0, "commit_headroom_gib": 64.0},
    }


def test_preflight_reports_shape_and_does_not_decode(monkeypatch):
    value = latent()
    video_vae = FakeVideoVAE()
    audio_vae = FakeAudioVAE()
    monkeypatch.setattr(
        "h3_audio_t8_pkg.av_decode_safety_advanced.runtime_snapshot",
        lambda: runtime(),
    )
    output = decode_av_safely(
        value,
        video_vae,
        audio_vae,
        mode="preflight_only",
        enforcement="report_only",
    )
    assert video_vae.decode_calls == 0
    assert audio_vae.decode_calls == 0
    assert output[0].shape == (0, 128, 128, 3)
    assert output[1]["waveform"].shape[-1] == 0
    report = json.loads(output[4])
    assert report["decoded"] is False
    assert report["latent"]["inferred_frames"] == 124
    assert report["memory_safe_claim"] is False


def test_regular_decode_matches_shapes_and_finite_output(monkeypatch):
    monkeypatch.setattr(
        "h3_audio_t8_pkg.av_decode_safety_advanced.runtime_snapshot",
        lambda: runtime(),
    )
    video_vae = FakeVideoVAE()
    audio_vae = FakeAudioVAE()
    frames, audio, video_latent, audio_latent, report_json = decode_av_safely(
        latent(),
        video_vae,
        audio_vae,
        mode="decode_regular",
    )
    assert frames.shape == (124, 128, 128, 3)
    assert audio["waveform"].shape == (1, 2, 207 * 800)
    assert video_latent["samples"].shape[1] == 24
    assert audio_latent["samples"].shape[1] == 32
    assert json.loads(report_json)["decoded"] is True


def test_tiled_decode_passes_explicit_video_tile_contract(monkeypatch):
    monkeypatch.setattr(
        "h3_audio_t8_pkg.av_decode_safety_advanced.runtime_snapshot",
        lambda: runtime(),
    )
    monkeypatch.setattr(
        "h3_audio_t8_pkg.av_decode_safety_advanced._h3_tiled_decode_core_contract",
        lambda: {
            "state": "supported",
            "adaptive_internal_tiling": True,
            "explicit_decode_tiled_alias": True,
        },
    )
    video_vae = FakeVideoVAE()
    output = decode_av_safely(
        latent(),
        video_vae,
        FakeAudioVAE(),
        mode="decode_tiled_exp",
        video_tile_size=24,
        video_tile_overlap=6,
        video_tile_temporal=37,
    )
    assert video_vae.tiled_calls == [
        {"tile_x": 24, "tile_y": 24, "overlap": 6, "tile_t": 37, "overlap_t": 1}
    ]
    report = json.loads(output[4])
    warning_codes = {item["code"] for item in report["issues"]["warnings"]}
    assert "h3_explicit_tile_controls_ignored" in warning_codes
    assert report["video_decode_route"]["explicit_tile_controls_effective"] is False


def test_regular_h3_decode_flags_internal_spatial_tiling_without_global_coordinates(monkeypatch):
    monkeypatch.setattr(
        "h3_audio_t8_pkg.av_decode_safety_advanced._h3_tiled_decode_core_contract",
        lambda: {
            "state": "unsupported",
            "adaptive_internal_tiling": True,
            "explicit_decode_tiled_alias": True,
        },
    )
    value = empty_av_latent(288, 256, 124)[0]
    report = inspect_av_decode(
        value,
        FakeVideoVAE(),
        FakeAudioVAE(),
        mode="decode_regular",
        minimum_current_headroom_mib=0.0,
        maximum_estimated_output_mib=8192.0,
        runtime=runtime(),
    )
    assert report["status"] == "high_risk"
    assert report["video_decode_route"]["internal_spatial_split_expected"] is True
    assert "h3_spatial_tiling_global_coordinates_missing" in {
        item["code"] for item in report["issues"]["high_risk"]
    }


def test_explicit_tiled_unknown_alias_is_not_assumed_effective(monkeypatch):
    monkeypatch.setattr(
        "h3_audio_t8_pkg.av_decode_safety_advanced._h3_tiled_decode_core_contract",
        lambda: {
            "state": "supported",
            "adaptive_internal_tiling": None,
            "explicit_decode_tiled_alias": None,
        },
    )
    report = inspect_av_decode(
        latent(),
        FakeVideoVAE(),
        FakeAudioVAE(),
        mode="decode_tiled_exp",
        minimum_current_headroom_mib=0.0,
        maximum_estimated_output_mib=8192.0,
        runtime=runtime(),
    )
    assert report["status"] == "unknown"
    assert report["video_decode_route"]["explicit_tile_controls_effective"] is None
    assert "h3_explicit_tile_controls_contract_unknown" in {
        item["code"] for item in report["issues"]["unknown"]
    }


def test_wrong_vae_and_low_headroom_fail_closed():
    class WrongVAE:
        latent_channels = 16
        latent_dim = 2

    report = inspect_av_decode(
        latent(),
        WrongVAE(),
        FakeAudioVAE(),
        mode="decode_regular",
        minimum_current_headroom_mib=512.0,
        maximum_estimated_output_mib=8192.0,
        runtime=runtime(128.0),
    )
    assert report["status"] == "blocked"
    assert {item["code"] for item in report["issues"]["hard"]} == {
        "video_vae_contract_mismatch"
    }
    with pytest.raises(ValueError, match="video_vae_contract_mismatch"):
        decode_av_safely(
            latent(),
            WrongVAE(),
            FakeAudioVAE(),
            mode="decode_regular",
            enforcement="block_known_unsafe",
        )


def test_output_estimate_gate_and_off_grid_audio_are_reported():
    value = latent()
    video, audio = value["samples"].unbind()
    value["samples"] = NestedTensor((video, audio[..., :-1]))
    report = inspect_av_decode(
        value,
        FakeVideoVAE(),
        FakeAudioVAE(),
        mode="preflight_only",
        minimum_current_headroom_mib=0.0,
        maximum_estimated_output_mib=1.0,
        runtime=runtime(),
    )
    warning_codes = {item["code"] for item in report["issues"]["warnings"]}
    risk_codes = {item["code"] for item in report["issues"]["high_risk"]}
    assert "audio_video_latent_duration_mismatch" in warning_codes
    assert "estimated_decode_output_exceeds_gate" in risk_codes


def test_node_schema_is_safe_by_default_and_appended():
    assert len(AV_DECODE_SAFETY_ADVANCED_NODE_CLASSES) == 1
    schema = AV_DECODE_SAFETY_ADVANCED_NODE_CLASSES[0].define_schema()
    inputs = {entry.id: entry for entry in schema.inputs}
    assert schema.node_id == "MiniMaxH3AVDecodeSafetyT8Advanced"
    assert schema.is_experimental is True
    assert inputs["mode"].default == "preflight_only"
    assert inputs["enforcement"].default == "report_only"
