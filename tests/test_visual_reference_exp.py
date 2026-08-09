from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import h3_audio_t8_pkg
from comfy.ldm.minimax.model import MiniMaxH3Model, VISUAL_COND_TIMESTEP
from h3_audio_t8_pkg.nodes_visual_reference_exp import (
    MiniMaxH3VisualReferenceStrengthEXPT8,
    apply_visual_reference_strength,
)


def conditioning_with(metadata):
    return [[torch.zeros((1, 2, 3), dtype=torch.float32), metadata]]


def test_visual_reference_strength_schema_and_registration_are_isolated_exp():
    schema = MiniMaxH3VisualReferenceStrengthEXPT8.define_schema()
    assert schema.node_id == "MiniMaxH3VisualReferenceStrengthEXPT8"
    assert schema.display_name == "MiniMax H3 Visual Reference Strength (EXP/T8)"
    assert schema.category == "T8/MiniMax H3/Conditioning/Experimental"
    assert schema.is_experimental is True
    assert [item.id for item in schema.inputs] == ["positive", "reference_strength"]
    assert schema.inputs[1].default == 0.999
    assert schema.inputs[1].min == 0.0
    assert schema.inputs[1].max == 1.0
    assert schema.inputs[1].step == 0.001
    assert [item.id for item in schema.outputs] == ["positive", "report"]

    node_classes = asyncio.run(h3_audio_t8_pkg.comfy_entrypoint().get_node_list())
    assert node_classes[35] is MiniMaxH3VisualReferenceStrengthEXPT8


@pytest.mark.parametrize("strength", [0.999, 0.995, 0.990, 0.980, 0.950])
def test_visual_reference_strength_preserves_three_decimal_values(strength):
    positive = conditioning_with(
        {"minimax_refs": [{"kind": "image", "latent": torch.ones(1)}]}
    )
    patched, report = apply_visual_reference_strength(positive, strength)
    assert patched[0][1]["minimax_visual_cond_noise_aug"] == strength
    assert json.loads(report)["reference_strength"] == strength


@pytest.mark.parametrize("strength", [-0.001, 1.001])
def test_visual_reference_strength_rejects_backend_out_of_range(strength):
    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        apply_visual_reference_strength(
            conditioning_with({"minimax_refs": [{"kind": "image"}]}),
            strength,
        )


def test_visual_reference_strength_rejects_no_visual_or_audio_only_refs():
    with pytest.raises(ValueError, match="visual conditioning is required"):
        apply_visual_reference_strength(conditioning_with({}), 0.99)
    with pytest.raises(ValueError, match="visual conditioning is required"):
        apply_visual_reference_strength(
            conditioning_with(
                {"minimax_refs": [{"kind": "audio", "audio_latent": torch.ones(1)}]}
            ),
            0.99,
        )


def test_visual_reference_strength_allows_keyframes_and_reports_global_risk():
    positive = conditioning_with(
        {
            "minimax_keyframes": [
                {"frame_index": 0, "latent": torch.zeros(1)},
                {"frame_index": 123, "latent": torch.ones(1)},
            ]
        }
    )
    patched, report_json = apply_visual_reference_strength(positive, 0.99)
    report = json.loads(report_json)
    assert patched[0][1]["minimax_visual_cond_noise_aug"] == 0.99
    assert report["has_keyframes"] is True
    assert report["keyframe_count"] == 2
    assert any("first_frame/last_frame" in item for item in report["warnings"])


def test_visual_reference_strength_low_value_has_aggressive_warning():
    _, report_json = apply_visual_reference_strength(
        conditioning_with({"minimax_refs": [{"kind": "video"}]}),
        0.949,
    )
    assert any("aggressive" in item for item in json.loads(report_json)["warnings"])


def test_visual_reference_strength_copies_metadata_and_preserves_other_fields():
    text_condition = torch.randn((1, 2, 3))
    refs = [
        {"kind": "image", "latent": torch.randn((1, 24, 1, 2, 2))},
        {"kind": "audio", "audio_latent": torch.randn((1, 32, 2, 5))},
    ]
    metadata = {
        "minimax_refs": refs,
        "minimax_audio_cond_noise_aug": 0.75,
        "custom_marker": {"preserve": True},
    }
    positive = [[text_condition, metadata]]
    patched, report_json = apply_visual_reference_strength(positive, 0.995)

    assert patched is not positive
    assert patched[0] is not positive[0]
    assert patched[0][1] is not metadata
    assert "minimax_visual_cond_noise_aug" not in metadata
    assert patched[0][0] is text_condition
    assert patched[0][1]["minimax_refs"] is refs
    assert patched[0][1]["minimax_audio_cond_noise_aug"] == 0.75
    assert patched[0][1]["custom_marker"] is metadata["custom_marker"]
    report = json.loads(report_json)
    assert report["visual_reference_count"] == 1
    assert report["audio_conditioning_changed"] is False


def test_explicit_0999_matches_current_h3_default_visual_noise_rows_exactly():
    assert VISUAL_COND_TIMESTEP == 0.999
    model_stub = SimpleNamespace(patch_size=(1, 2, 2))
    latent = torch.linspace(-1.0, 1.0, 24 * 2 * 4).reshape(1, 24, 1, 2, 4)
    baseline = MiniMaxH3Model._cond_video_rows(
        model_stub,
        {"cond_video_latents": [latent], "seed": 2608102201},
        torch.device("cpu"),
    )
    explicit = MiniMaxH3Model._cond_video_rows(
        model_stub,
        {
            "cond_video_latents": [latent],
            "seed": 2608102201,
            "visual_cond_noise_aug": 0.999,
        },
        torch.device("cpu"),
    )
    assert torch.equal(baseline, explicit)
    assert torch.max(torch.abs(baseline - explicit)).item() == 0.0


def test_visual_reference_strength_api_example_patches_only_positive_conditioning():
    root = Path(__file__).resolve().parents[1]
    workflow = json.loads(
        (root / "examples" / "ref2va_visual_reference_strength_exp_api.json").read_text(
            encoding="utf-8"
        )
    )
    types = {node["class_type"] for node in workflow.values()}
    assert "MiniMaxH3AudioConditioningT8" in types
    assert "MiniMaxH3VisualReferenceStrengthEXPT8" in types
    assert not any("Lora" in node_type for node_type in types)
    assert workflow["1"]["inputs"]["unet_name"] == (
        "minimax_h3_ref2va_int8_convrot.safetensors"
    )
    conditioning = workflow["6"]
    patch = workflow["7"]
    sampler = workflow["8"]
    assert conditioning["inputs"]["task_type"] == "Ref2VA"
    assert conditioning["inputs"]["ref_images.ref_image_0"] == ["5", 0]
    assert patch["inputs"] == {"positive": ["6", 0], "reference_strength": 0.99}
    assert sampler["inputs"]["av_latent"] == ["6", 1]
    assert sampler["inputs"]["steps"] == 20
    assert sampler["inputs"]["sampler_name"] == "dual_clock_euler"
    assert sampler["inputs"]["scheduler"] == "native_flow"
    assert workflow["10"]["inputs"]["conditioning"] == ["7", 0]


def test_visual_reference_strength_frontend_workflow_is_consistent_and_warns():
    root = Path(__file__).resolve().parents[1]
    workflow = json.loads(
        (
            root
            / "examples"
            / "workflows"
            / "H3_Ref2VA_Visual_Reference_Strength_EXP.json"
        ).read_text(encoding="utf-8")
    )
    nodes = {node["id"]: node for node in workflow["nodes"]}
    links = {link[0]: link for link in workflow["links"]}
    assert workflow["version"] == 0.4
    assert workflow["last_node_id"] == max(nodes)
    assert workflow["last_link_id"] == max(links)

    patch = next(
        node
        for node in nodes.values()
        if node["type"] == "MiniMaxH3VisualReferenceStrengthEXPT8"
    )
    assert patch["widgets_values"] == [0.99]
    assert "0.999 -> 0.995 -> 0.990" in patch["title"]
    assert "<=0.950" in patch["title"]

    conditioning = next(
        node for node in nodes.values() if node["type"] == "MiniMaxH3AudioConditioningT8"
    )
    assert conditioning["widgets_values"][4] == "Ref2VA"
    assert any(item["name"] == "ref_images.ref_image_0" for item in conditioning["inputs"])

    for link_id, source_id, source_slot, target_id, target_slot, link_type in workflow["links"]:
        source = nodes[source_id]["outputs"][source_slot]
        target = nodes[target_id]["inputs"][target_slot]
        assert link_id in (source.get("links") or [])
        assert target["link"] == link_id
        assert source["type"] == target["type"] == link_type
