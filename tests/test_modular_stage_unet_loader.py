"""Independent stage-ordered Core load never hides a sampling operation."""

import json

import pytest

from h3_audio_t8_pkg.modular_sampling import stage_unet_loader
from h3_audio_t8_pkg.modular_sampling.stage_unet_loader_nodes import MiniMaxH3StageUNETLoaderAfterEXPT8
from h3_audio_t8_pkg.modular_sampling.results import sample_stage
from test_modular_results import inputs


def test_loads_new_model_only_after_verified_completed_stage(monkeypatch):
    low = sample_stage(*inputs("low_0_4"))[2]
    calls = []
    fresh_model = object()

    def load(name, dtype):
        calls.append((name, dtype))
        return fresh_model

    monkeypatch.setattr(stage_unet_loader, "_load_unet", load)
    returned, report_json = stage_unet_loader.load_unet_after_stage(
        low, "v2-model.safetensors", "default")
    assert returned is fresh_model
    assert calls == [("v2-model.safetensors", "default")]
    report = json.loads(report_json)
    assert report["preceding_stage"] == "low_0_4"
    assert report["preceding_receipt_sha256"] == low.verify()["receipt_sha256"]
    assert MiniMaxH3StageUNETLoaderAfterEXPT8.GET_NODE_INFO_V1()["output"] == ["MODEL", "STRING"]


def test_invalid_or_incomplete_stage_never_loads_model(monkeypatch):
    calls = []
    monkeypatch.setattr(stage_unet_loader, "_load_unet", lambda *_: calls.append(True))
    with pytest.raises(ValueError, match="StageResult"):
        stage_unet_loader.load_unet_after_stage({"samples": 0}, "model", "default")
    args = list(inputs("low_0_4"))
    args[2].sampler_function = lambda denoiser, x, sigmas, **kwargs: x
    incomplete = sample_stage(*args)[2]
    with pytest.raises(ValueError, match="completed preceding stage"):
        stage_unet_loader.load_unet_after_stage(incomplete, "model", "default")
    completed = sample_stage(*inputs("low_0_4"))[2]
    completed.denoised_output["samples"].unbind()[0].add_(0.01)
    with pytest.raises(ValueError, match="mutated"):
        stage_unet_loader.load_unet_after_stage(completed, "model", "default")
    assert calls == []
