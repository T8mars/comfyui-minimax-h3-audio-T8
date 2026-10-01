"""Current V2 recipe fingerprint changes with every editable cross-segment input."""

from dataclasses import replace
import hashlib
import json

import pytest
import torch

from h3_audio_t8_pkg import prompt_relay_advanced as relay
from h3_audio_t8_pkg.modular_sampling import node_classes
from h3_audio_t8_pkg.modular_sampling import fast_h3_v2_job as job
from h3_audio_t8_pkg.modular_sampling.eav import EAVConfig
from h3_audio_t8_pkg.modular_sampling.fast_h3_v2_job_nodes import (
    JOB_TYPE, MiniMaxH3FastH3V2CurrentRecipeEXPT8,
)


def _digest(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


@pytest.fixture
def inputs(monkeypatch):
    monkeypatch.setattr(job, "stage_model_identity", lambda value, **_kwargs: {"sha256": _digest(value)})
    monkeypatch.setattr(job, "_component_identity", lambda value: {"sha256": _digest(value)})
    monkeypatch.setattr(job, "implementation_identity", lambda: {"source.py": "1" * 64})
    plan = relay.build_prompt_relay_plan(
        "One scene.", "She walks.\nShe turns.", 193, "auto_equal", "",
        "paper_v1", .1, False, False)[0]
    geometry = job.learned_upscale_geometry(16, 24, "target_dimensions", 2., 1.,
                                             512, 768, "honor_dimensions_exp", 1.05)
    report = {"schema_version": 1, "node": "MiniMaxH3LearnedLatentUpscaleT8Advanced",
              "status": "ok", "geometry": geometry, "audio_preserved": True,
              "model": {"name": "learned3d.safetensors", "sha256": _digest("learned3d"),
                        "precision": "fp16", "contract": {"architecture": "test"}}}
    return {"model_pass1": "low-model", "model_pass2": "high-model",
            "clip": "qwen", "video_vae": "video-vae", "audio_vae": "audio-vae",
            "first_frame": torch.zeros(1, 32, 32, 3),
            "low_global_plan": plan, "high_global_plan": plan,
            "low_eav_config": EAVConfig(mode="disabled"),
            "high_eav_config": EAVConfig(mode="disabled"),
            "upscale_report_json": json.dumps(report), "chain_id": "current_v2_test",
            "low_width": 256, "low_height": 384, "width": 512, "height": 768,
            "total_accepted_frames": 192, "first_seed": 2609152201}


def test_recipe_is_current_and_stable_across_segments_not_a_parent_sha(inputs):
    recipe, digest, model_id, report_json = job.make_current_recipe(**inputs)
    assert recipe.verify()["schema"] == job.SCHEMA
    assert digest == recipe.sha256
    assert model_id == _digest("low-model")[:16] + ":" + _digest("high-model")[:16]
    assert job.make_current_recipe(**inputs)[1] == digest
    assert recipe.verify()["window"] == {"total_accepted_frames": 192,
        "first_render_frames": 124, "continuation_context_frames": 22,
        "first_seed": 2609152201, "seed_policy": "increment"}
    assert "parent_job_sha256" not in recipe.verify()
    assert "stage" not in recipe.verify()
    assert "no stage execution" in json.loads(report_json)["boundary"]
    assert MiniMaxH3FastH3V2CurrentRecipeEXPT8 in node_classes()
    assert MiniMaxH3FastH3V2CurrentRecipeEXPT8.GET_NODE_INFO_V1()["output"][0] == JOB_TYPE


def test_old_fixed_second_window_is_an_explicit_cross_segment_recipe_choice(inputs):
    compact = job.make_current_recipe(**inputs)[0]
    fixed = job.make_current_recipe(**inputs, continuation_render_frames=124)[0]
    assert compact.sha256 != fixed.sha256
    assert "continuation_render_frames" not in compact.verify()["window"]
    assert fixed.verify()["window"]["continuation_render_frames"] == 124
    with pytest.raises(ValueError, match="compact 90 or old fixed 124"):
        job.make_current_recipe(**inputs, continuation_render_frames=107)


@pytest.mark.parametrize("changed", [
    {"model_pass1": "another-low"}, {"model_pass2": "another-high"},
    {"clip": "another-qwen"}, {"video_vae": "another-video-vae"},
    {"audio_vae": "another-audio-vae"},
    {"chain_id": "different_chain"}, {"first_seed": 2609152202},
    {"low_eav_config": EAVConfig(mode="report_only")},
    {"high_eav_config": EAVConfig(mode="apply_exp")},
])
def test_recipe_changes_for_current_editable_inputs(inputs, changed):
    old = job.make_current_recipe(**inputs)[1]
    assert job.make_current_recipe(**{**inputs, **changed})[1] != old


def test_recipe_changes_for_first_frame_and_global_relay(inputs):
    old = job.make_current_recipe(**inputs)[1]
    pixels = inputs["first_frame"].clone()
    pixels[0, 0, 0, 0] = 1.
    assert job.make_current_recipe(**{**inputs, "first_frame": pixels})[1] != old
    other = relay.build_prompt_relay_plan(
        "Different scene.", "She walks.\nShe turns.", 193, "auto_equal", "",
        "paper_v1", .1, False, False)[0]
    for key in ("low_global_plan", "high_global_plan"):
        assert job.make_current_recipe(**{**inputs, key: other})[1] != old
    changed_report = json.loads(inputs["upscale_report_json"])
    changed_report["model"]["sha256"] = _digest("different-learned3d")
    assert job.make_current_recipe(**{
        **inputs, "upscale_report_json": json.dumps(changed_report)})[1] != old


def test_recipe_rejects_tampering_and_unsupported_bounds(inputs):
    recipe = job.make_current_recipe(**inputs)[0]
    payload = recipe.verify()
    payload["window"]["first_seed"] += 1
    with pytest.raises(ValueError, match="fingerprint changed"):
        replace(recipe, payload_json=job.canonical(payload)).verify()
    with pytest.raises(ValueError, match="finite"):
        job.make_current_recipe(**{**inputs, "first_frame": torch.full((1, 32, 32, 3), float("nan"))})
    with pytest.raises(ValueError, match="192-frame"):
        job.make_current_recipe(**{**inputs, "total_accepted_frames": 193})
    with pytest.raises(ValueError, match="geometry"):
        job.make_current_recipe(**{**inputs, "width": 255})
    bad_plan = dict(inputs["low_global_plan"])
    bad_plan["plan_hash"] = "0" * 64
    with pytest.raises(ValueError, match="plan hash mismatch"):
        job.make_current_recipe(**{**inputs, "low_global_plan": bad_plan})
    bad_upscale = json.loads(inputs["upscale_report_json"])
    bad_upscale["status"] = "noop"
    with pytest.raises(ValueError, match="completed learned 3D upscale"):
        job.make_current_recipe(**{**inputs, "upscale_report_json": json.dumps(bad_upscale)})
