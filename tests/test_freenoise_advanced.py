from __future__ import annotations

import json
from pathlib import Path

import torch

import comfy.nested_tensor
from h3_audio_t8_pkg.freenoise_advanced import (
    build_free_noise_model,
    free_noise_config,
    reschedule_h3_noise,
)


class _FakeModel:
    def __init__(self):
        self.model_options = {"transformer_options": {"existing": "kept"}}

    def clone(self):
        output = _FakeModel()
        output.model_options = {
            "transformer_options": dict(self.model_options["transformer_options"])
        }
        return output


def test_model_plan_is_append_only_and_preserves_other_transformer_options():
    original = _FakeModel()
    patched, report_json = build_free_noise_model(
        original,
        mode="variance_preserving_blend",
        base_seed=77,
        reuse_ratio=0.65,
    )
    report = json.loads(report_json)
    assert free_noise_config(original) is None
    assert free_noise_config(patched)["base_seed"] == 77
    assert patched.model_options["transformer_options"]["existing"] == "kept"
    assert report["audio_noise"].startswith("native")


def test_paper_permutation_is_deterministic_and_audio_is_exactly_unchanged():
    model, _ = build_free_noise_model(
        _FakeModel(), mode="paper_permutation", base_seed=1234, reuse_ratio=0.2
    )
    config = free_noise_config(model)
    video = torch.randn(1, 24, 7, 4, 5)
    audio = torch.randn(1, 32, 2, 37)
    noise = comfy.nested_tensor.NestedTensor((video, audio))
    first, report_a = reschedule_h3_noise(noise, config=config, segment_index=2)
    second, report_b = reschedule_h3_noise(noise, config=config, segment_index=2)
    assert torch.equal(first.unbind()[0], second.unbind()[0])
    assert first.unbind()[1] is audio
    assert report_a == report_b
    assert report_a["reuse_ratio"] == 1.0


def test_segment_permutation_changes_video_pool_without_touching_audio():
    model, _ = build_free_noise_model(
        _FakeModel(), mode="variance_preserving_blend", base_seed=55, reuse_ratio=0.75
    )
    config = free_noise_config(model)
    video = torch.zeros(1, 24, 8, 3, 3)
    audio = torch.randn(1, 32, 2, 40)
    noise = comfy.nested_tensor.NestedTensor((video, audio))
    segment_zero, _ = reschedule_h3_noise(noise, config=config, segment_index=0)
    segment_one, _ = reschedule_h3_noise(noise, config=config, segment_index=1)
    assert not torch.equal(segment_zero.unbind()[0], segment_one.unbind()[0])
    assert segment_zero.unbind()[1] is audio
    assert segment_one.unbind()[1] is audio


def test_feature_manifest_registers_freenoise_after_all_old_nodes():
    root = Path(__file__).resolve().parents[1]
    features = json.loads((root / "features.json").read_text(encoding="utf-8"))
    feature = features["freenoise_long_video_advanced"]
    assert feature["position"] == 241
    assert "adaptation" in feature["scientific_boundary"]
    assert features["nodes"][241] == "MiniMaxH3FreeNoiseLongVideoT8Advanced"


def test_frontend_workflow_composes_free_noise_before_relay_eav_runner():
    root = Path(__file__).resolve().parents[1]
    path = (
        root
        / "examples"
        / "workflows"
        / "04-long-video"
        / "2026-08-28_H3_FreeNoise_Prompt_Relay_EAV_Long_Video_Advanced_EXP.json"
    )
    workflow = json.loads(path.read_text(encoding="utf-8"))
    nodes = {int(node["id"]): node for node in workflow["nodes"]}
    assert workflow["version"] == 0.4
    assert nodes[11]["type"] == "MiniMaxH3FreeNoiseLongVideoT8Advanced"
    assert nodes[6]["type"] == "MiniMaxH3LongVideoInNodeLoopEffectsT8Advanced"
    assert nodes[6]["inputs"][0]["link"] == 6
    assert [1, 1, 0, 11, 0, "MODEL"] in workflow["links"]
    assert [6, 11, 0, 6, 0, "MODEL"] in workflow["links"]
    assert "不等于论文完整复现" in nodes[12]["widgets_values"][0]
