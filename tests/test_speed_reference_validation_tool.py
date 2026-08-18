from __future__ import annotations

import pytest

from h3_audio_t8_pkg.tools.build_h3_speed_reference_validation import (
    build_reference_speed_prompts,
)


@pytest.fixture
def built():
    return build_reference_speed_prompts(
        source_video="source.mp4", reference_image="reference.png"
    )


def test_p3_reference_builder_covers_three_reference_contracts(built):
    prompts, manifest = built
    assert set(prompts) == {
        "ref_image_native",
        "ref_video_audio_native",
        "hybrid_first_image_audio",
    }
    assert manifest["controlled"]["reference_video_seconds"] == 2.0
    assert manifest["controlled"]["steps"] == 20


def test_p3_reference_builder_uses_matching_numbered_video_soundtrack(built):
    prompts, _manifest = built
    source = prompts["ref_video_audio_native"]["6"]["inputs"]
    assert source["ref_videos.ref_video_0"] == ["10", 0]
    assert source["ref_video_audios.ref_video_audio_0"] == ["10", 1]
    assert "ref_images.ref_image_0" not in source
    assert prompts["ref_video_audio_native"]["9"]["inputs"]["duration"] == 2.0


def test_p3_hybrid_has_anchor_image_and_standalone_audio(built):
    prompts, _manifest = built
    source = prompts["hybrid_first_image_audio"]["6"]["inputs"]
    assert source["task_type"] == "Hybrid"
    assert source["first_frame"] == ["12", 0]
    assert source["ref_images.ref_image_0"] == ["11", 0]
    assert source["ref_audios.ref_audio_0"] == ["10", 1]
    assert source["add_source_as_reference"] is False


def test_p3_builder_rejects_invalid_grid_or_steps():
    with pytest.raises(ValueError, match="multiples of 32"):
        build_reference_speed_prompts(
            source_video="source.mp4", reference_image="reference.png", width=1000
        )
    with pytest.raises(ValueError, match="Stock20"):
        build_reference_speed_prompts(
            source_video="source.mp4", reference_image="reference.png", steps=8
        )
