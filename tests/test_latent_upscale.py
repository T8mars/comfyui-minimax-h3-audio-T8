from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import comfy.nested_tensor

from h3_audio_t8_pkg.latent_upscale import (
    aligned_upscale_geometry,
    upscale_latent_by_32,
)
from h3_audio_t8_pkg.nodes_latent_upscale import MiniMaxH3LatentUpscaleBy32T8


def test_best_aspect_h3_geometry_is_divisible_by_32_with_bounded_ratio_error():
    geometry = aligned_upscale_geometry(46, 26, 1.5, 16, "best_aspect")

    assert geometry["output_width"] % 32 == 0
    assert geometry["output_height"] % 32 == 0
    assert geometry["output_width"] == 1120
    assert geometry["output_height"] == 640

    source_aspect = 736 / 416
    naive_aspect = 1104 / 624
    selected_error = abs((geometry["output_aspect_ratio"] / source_aspect) - 1.0)
    naive_error = abs((naive_aspect / source_aspect) - 1.0)
    assert selected_error > naive_error
    # The official rounded target is closer in ratio here but is not a legal 32px canvas.
    assert 1104 % 32 != 0
    assert geometry["aspect_ratio_error_percent"] < 1.1


@pytest.mark.parametrize("policy", ["best_aspect", "nearest", "ceil", "floor"])
def test_all_alignment_policies_return_legal_nonshrinking_geometry(policy):
    geometry = aligned_upscale_geometry(45, 25, 1.01, 16, policy)
    assert geometry["output_width"] % 32 == 0
    assert geometry["output_height"] % 32 == 0
    assert geometry["output_width"] >= 45 * 16
    assert geometry["output_height"] >= 25 * 16


def test_plain_video_latent_resizes_only_spatial_axes_and_preserves_metadata():
    samples = torch.arange(2 * 4 * 3 * 26 * 46, dtype=torch.float32).reshape(2, 4, 3, 26, 46)
    noise_mask = torch.ones((2, 1, 3, 26, 46), dtype=torch.float32)
    batch_index = [5, 9]
    latent = {"samples": samples, "noise_mask": noise_mask, "batch_index": batch_index}

    output, width, height, report_text = upscale_latent_by_32(
        latent, "bilinear", 1.5, 16, "best_aspect"
    )

    assert (width, height) == (1120, 640)
    assert output["samples"].shape == (2, 4, 3, 40, 70)
    assert output["noise_mask"].shape == (2, 1, 3, 40, 70)
    assert output["batch_index"] is batch_index
    assert latent["samples"] is samples
    assert latent["noise_mask"] is noise_mask
    report = json.loads(report_text)
    assert report["noise_mask"] == "resized_nearest_exact"
    assert report["latent_kind"] == "plain"
    assert report["geometry"]["output_width"] % 32 == 0


def test_nested_h3_av_resizes_video_and_preserves_audio_values_and_timing():
    video = torch.randn((1, 24, 7, 26, 46))
    audio = torch.randn((1, 32, 2, 176))
    video_mask = torch.ones_like(video)
    audio_mask = torch.zeros_like(audio)
    samples = comfy.nested_tensor.NestedTensor((video, audio))
    mask = comfy.nested_tensor.NestedTensor((video_mask, audio_mask))
    latent = {"samples": samples, "noise_mask": mask, "custom": {"keep": True}}

    output, width, height, report_text = upscale_latent_by_32(
        latent, "bicubic", 1.5, 16, "best_aspect"
    )
    output_video, output_audio = tuple(output["samples"].unbind())
    output_video_mask, output_audio_mask = tuple(output["noise_mask"].unbind())

    assert (width, height) == (1120, 640)
    assert output_video.shape == (1, 24, 7, 40, 70)
    assert output_audio is audio
    assert output_audio.shape == (1, 32, 2, 176)
    assert output_video_mask.shape == (1, 24, 7, 40, 70)
    assert output_audio_mask is audio_mask
    assert output["custom"] is latent["custom"]
    report = json.loads(report_text)
    assert report["audio_latent_preserved"] is True
    assert report["latent_kind"] == "minimax_h3_joint_av"


def test_plain_sd_latent_uses_explicit_eight_pixel_scale():
    latent = {"samples": torch.randn((1, 4, 64, 96))}
    output, width, height, _ = upscale_latent_by_32(
        latent, "nearest-exact", 1.5, 8, "best_aspect"
    )
    assert width % 32 == 0
    assert height % 32 == 0
    assert output["samples"].shape[-1] == width // 8
    assert output["samples"].shape[-2] == height // 8


def test_matching_three_dimensional_noise_mask_is_resized():
    latent = {
        "samples": torch.randn((1, 4, 32, 48)),
        "noise_mask": torch.ones((1, 32, 48), dtype=torch.bool),
    }
    output, width, height, report_text = upscale_latent_by_32(
        latent, "nearest-exact", 2.0, 8, "nearest"
    )
    assert output["noise_mask"].shape == (1, height // 8, width // 8)
    assert output["noise_mask"].dtype == torch.bool
    assert json.loads(report_text)["noise_mask"] == "resized_nearest_exact"


def test_matching_two_dimensional_noise_mask_keeps_its_rank():
    latent = {
        "samples": torch.randn((1, 4, 32, 48)),
        "noise_mask": torch.ones((32, 48), dtype=torch.float32),
    }
    output, width, height, _ = upscale_latent_by_32(
        latent, "nearest-exact", 2.0, 8, "nearest"
    )
    assert output["noise_mask"].shape == (height // 8, width // 8)


def test_nonmatching_noise_mask_is_preserved_instead_of_guessed():
    noise_mask = torch.ones((1, 128, 192))
    latent = {"samples": torch.randn((1, 4, 32, 48)), "noise_mask": noise_mask}
    output, _, _, report_text = upscale_latent_by_32(
        latent, "nearest-exact", 2.0, 8, "nearest"
    )
    assert output["noise_mask"] is noise_mask
    assert json.loads(report_text)["noise_mask"] == "preserved_nonmatching_spatial_shape"


def test_nested_h3_requires_the_h3_pixel_scale_and_matching_nested_mask_contract():
    video = torch.zeros((1, 24, 7, 26, 46))
    audio = torch.zeros((1, 32, 2, 176))
    samples = comfy.nested_tensor.NestedTensor((video, audio))
    with pytest.raises(ValueError, match="requires pixels_per_latent=16"):
        upscale_latent_by_32(
            {"samples": samples}, "bicubic", 1.5, 8, "best_aspect"
        )
    with pytest.raises(ValueError, match="requires a nested video/audio noise_mask"):
        upscale_latent_by_32(
            {"samples": samples, "noise_mask": torch.ones_like(video)},
            "bicubic",
            1.5,
            16,
            "best_aspect",
        )


def test_schema_defaults_to_h3_and_best_aspect_without_changing_existing_nodes():
    schema = MiniMaxH3LatentUpscaleBy32T8.define_schema()
    inputs = {item.id: item for item in schema.inputs}
    assert schema.node_id == "MiniMaxH3LatentUpscaleBy32T8"
    assert schema.category == "T8/MiniMax H3/Latent"
    assert inputs["pixels_per_latent"].default == "16 - MiniMax H3"
    assert inputs["alignment_policy"].default == "best_aspect"
    assert inputs["scale_by"].default == 1.5


def test_node_execute_parses_the_ui_pixel_scale_and_returns_pixel_dimensions():
    result = MiniMaxH3LatentUpscaleBy32T8.execute(
        {"samples": torch.zeros((1, 4, 52, 92))},
        "bicubic",
        1.5,
        "8 - SD/SDXL/other",
        "best_aspect",
    )
    assert result[0]["samples"].shape == (1, 4, 80, 140)
    assert result[1:3] == (1120, 640)


def test_frontend_example_is_loadable_and_uses_plain_latent_scale_explicitly():
    workflow_path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "workflows"
        / "13-latent-upscale"
        / "2026-08-16_H3_Latent_Upscale_By32.json"
    )
    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    nodes = {node["type"]: node for node in workflow["nodes"]}
    assert workflow["version"] == 0.4
    assert "EmptyLatentImage" in nodes
    node = nodes["MiniMaxH3LatentUpscaleBy32T8"]
    assert node["widgets_values"] == [
        "bicubic",
        1.5,
        "8 - SD/SDXL/other",
        "best_aspect",
    ]
    assert "PreviewAny" in nodes
    assert workflow["links"] == [
        [1, 1, 0, 2, 0, "LATENT"],
        [2, 2, 3, 3, 0, "STRING"],
    ]
