from __future__ import annotations

import pytest

from h3_audio_t8_pkg.tools.build_h3_speed_quality_validation import (
    build_full_resolution_baseline,
    build_quality_pairs,
)


@pytest.fixture
def built():
    return build_quality_pairs(
        source_video="source.mp4",
        reference_image="reference.png",
    )


def test_quality_builder_emits_two_controlled_pairs(built):
    prompts, manifest = built
    assert set(prompts) == {
        "fl2va_baseline",
        "fl2va_speed",
        "ref2va_baseline",
        "ref2va_speed",
    }
    assert manifest["schema"] == "minimax_h3_speed_quality_pairs_v1"
    assert manifest["controlled"]["steps"] == 20
    assert all(
        all(contract.values()) for contract in manifest["contracts"].values()
    )


@pytest.mark.parametrize("name", ["fl2va", "ref2va"])
def test_quality_baseline_keeps_inputs_and_removes_speed_runtime(built, name):
    prompts, _manifest = built
    baseline = prompts[f"{name}_baseline"]
    speed = prompts[f"{name}_speed"]
    assert baseline["1"]["inputs"] == speed["1"]["inputs"]
    assert baseline["2"]["inputs"] == speed["2"]["inputs"]
    assert baseline["5"]["class_type"] == "MiniMaxH3AudioConditioningT8"
    assert baseline["6"]["class_type"] == "MiniMaxH3DualClockSamplerT8"
    assert baseline["7"]["class_type"] == "BasicGuider"
    assert baseline["19"]["class_type"] == (
        "MiniMaxH3SPEEDModalityStableNoiseT8Advanced"
    )
    assert baseline["20"]["class_type"] == "SamplerCustomAdvanced"
    assert baseline["13"]["inputs"]["av_latent"] == ["20", 0]
    assert baseline["6"]["inputs"]["steps"] == speed["5"]["inputs"]["steps"]
    assert baseline["19"]["inputs"]["seed"] == speed["7"]["inputs"]["seed"]
    assert "checkpoint_fingerprint" not in baseline["5"]["inputs"]
    assert "vae_fingerprint" not in baseline["5"]["inputs"]


def test_fl2va_baseline_preserves_remix_and_anchor_contract(built):
    prompts, _manifest = built
    baseline = prompts["fl2va_baseline"]
    speed = prompts["fl2va_speed"]
    conditioning = baseline["5"]["inputs"]
    assert conditioning["task_type"] == "FL2VA"
    assert conditioning["audio_mode"] == "remix_source"
    assert conditioning["drive_audio"] == ["10", 1]
    assert conditioning["first_frame"] == ["11", 0]
    assert conditioning["last_frame"] == ["12", 0]
    assert baseline["14"]["inputs"] == speed["14"]["inputs"]


def test_ref2va_baseline_preserves_visual_reference_contract(built):
    prompts, _manifest = built
    baseline = prompts["ref2va_baseline"]
    conditioning = baseline["5"]["inputs"]
    assert conditioning["task_type"] == "Ref2VA"
    assert conditioning["ref_images.ref_image_0"] == ["11", 0]
    assert "first_frame" not in conditioning
    assert "ref_videos.ref_video_0" not in conditioning


def test_baseline_builder_fails_closed_on_wrong_source_schema(built):
    prompts, _manifest = built
    broken = prompts["fl2va_speed"].copy()
    broken["6"] = {"class_type": "Other", "inputs": {}}
    with pytest.raises(ValueError, match="frozen SPEED stage source"):
        build_full_resolution_baseline(
            broken,
            width=1024,
            height=576,
            filename_prefix="test",
        )
