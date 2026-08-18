from __future__ import annotations

import pytest

from h3_audio_t8_pkg.tools.build_h3_speed_multimodal_validation import (
    build_multimodal_speed_prompts,
)


@pytest.fixture
def built():
    return build_multimodal_speed_prompts(source_video="source.mp4")


def test_p2_builder_covers_five_requirements_in_three_runs(built):
    prompts, manifest = built
    assert set(prompts) == {"i2va_lock_source", "fl2va_remix_source", "l2va_native"}
    coverage = {
        (item["task_type"], item["audio_mode"])
        for item in manifest["coverage"]
    }
    assert coverage == {
        ("I2VA", "lock_source"),
        ("FL2VA", "remix_source"),
        ("L2VA", "native"),
    }
    assert manifest["controlled"]["width"] == 1024
    assert manifest["controlled"]["height"] == 576
    assert manifest["controlled"]["length"] == 124
    assert manifest["controlled"]["steps"] == 20


def test_p2_prompts_use_stage_rebuild_and_exact_media_routes(built):
    prompts, _manifest = built
    expected = {
        "i2va_lock_source": ("I2VA", "lock_source", True, False, ["7", 1]),
        "fl2va_remix_source": ("FL2VA", "remix_source", True, True, ["13", 1]),
        "l2va_native": ("L2VA", "native", False, True, ["13", 1]),
    }
    for name, (task, mode, first, last, trim_audio) in expected.items():
        prompt = prompts[name]
        source = prompt["6"]["inputs"]
        assert source["task_type"] == task
        assert source["audio_mode"] == mode
        assert ("first_frame" in source) is first
        assert ("last_frame" in source) is last
        assert ("drive_audio" in source) is (mode != "native")
        assert source["add_source_as_reference"] is False
        assert prompt["7"]["inputs"]["execution_scope"] == "multimodal_research_exp"
        assert prompt["14"]["inputs"]["audio"] == trim_audio
        assert prompt["9"]["class_type"] == "Video Slice"
        assert prompt["9"]["inputs"]["strict_duration"] is True
        assert prompt["16"]["class_type"] == "SaveVideo"
        assert prompt["17"]["inputs"]["text"] == ["7", 4]
        assert prompt["18"]["inputs"]["text"] == ["6", 1]


def test_p2_builder_rejects_non_h3_grid_or_non_stock20():
    with pytest.raises(ValueError, match="multiples of 32"):
        build_multimodal_speed_prompts(source_video="source.mp4", width=1000)
    with pytest.raises(ValueError, match=r"17n\+5"):
        build_multimodal_speed_prompts(source_video="source.mp4", length=123)
    with pytest.raises(ValueError, match="Stock20"):
        build_multimodal_speed_prompts(source_video="source.mp4", steps=8)
