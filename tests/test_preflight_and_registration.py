from __future__ import annotations

import asyncio
import json
from pathlib import Path

import torch

import h3_audio_t8_pkg
from h3_audio_t8_pkg.preflight import run_preflight
from helpers import FakeAudioVAE, FakeVideoVAE, make_audio


def test_all_nodes_register_with_unique_ids_and_valid_schemas():
    extension = h3_audio_t8_pkg.comfy_entrypoint()
    node_classes = asyncio.run(extension.get_node_list())
    schemas = [node.define_schema() for node in node_classes]
    ids = [schema.node_id for schema in schemas]
    assert len(ids) == 14
    assert len(ids) == len(set(ids))
    assert "MiniMaxH3AudioConditioningT8" in ids
    assert "MiniMaxH3DualClockSamplerT8" in ids
    assert "MiniMaxH3MultiRateSamplerEXPT8" in ids
    assert "MiniMaxH3StillConditioningT8" in ids
    assert "MiniMaxH3StillPreflightT8" in ids
    assert "MiniMaxH3StillDecodeT8" in ids
    exp_schema = schemas[ids.index("MiniMaxH3MultiRateSamplerEXPT8")]
    assert exp_schema.is_experimental is True
    assert exp_schema.category == "T8/MiniMax H3/Audio/Experimental"
    for still_id in {
        "MiniMaxH3StillConditioningT8",
        "MiniMaxH3StillPreflightT8",
        "MiniMaxH3StillDecodeT8",
    }:
        still_schema = schemas[ids.index(still_id)]
        assert still_schema.is_experimental is True
        assert still_schema.category == "T8/MiniMax H3/Still/Experimental"


def test_task_type_frontend_labels_preserve_canonical_backend_values():
    extension = h3_audio_t8_pkg.comfy_entrypoint()
    node_classes = asyncio.run(extension.get_node_list())
    conditioning = next(
        node for node in node_classes
        if node.define_schema().node_id == "MiniMaxH3AudioConditioningT8"
    )
    task_type = next(
        item for item in conditioning.define_schema().inputs
        if item.id == "task_type"
    )
    assert task_type.options == [
        "auto", "T2VA", "I2VA", "FL2VA", "L2VA", "Ref2VA", "Hybrid",
    ]

    package_root = Path(__file__).resolve().parents[1]
    assert h3_audio_t8_pkg.WEB_DIRECTORY == "./web"
    frontend = (package_root / "web" / "task_type_labels.js").read_text(encoding="utf-8")
    for label in {
        "auto — 自动判断",
        "T2VA — 文生音视频",
        "I2VA — 图生音视频（首帧）",
        "FL2VA — 首尾帧生音视频",
        "L2VA — 尾帧生音视频",
        "Ref2VA — 参考生音视频",
        "Hybrid — 关键帧+参考混合生成",
    }:
        assert label in frontend
    assert "toBackendValue(widget.value)" in frontend


def test_dual_clock_sampler_appends_optional_choices_without_reordering_legacy_widgets():
    extension = h3_audio_t8_pkg.comfy_entrypoint()
    node_classes = asyncio.run(extension.get_node_list())
    sampler_node = next(
        node for node in node_classes
        if node.define_schema().node_id == "MiniMaxH3DualClockSamplerT8"
    )
    inputs = sampler_node.define_schema().inputs

    assert [item.id for item in inputs] == [
        "model",
        "av_latent",
        "steps",
        "shift_video",
        "shift_audio",
        "sampler_name",
        "scheduler",
    ]
    sampler_name = inputs[-2]
    scheduler = inputs[-1]
    assert sampler_name.optional is True
    assert sampler_name.default == "dual_clock_euler"
    assert sampler_name.options[0] == "dual_clock_euler"
    assert "euler" in sampler_name.options
    assert scheduler.optional is True
    assert scheduler.default == "native_flow"
    assert scheduler.options[0] == "native_flow"
    assert "normal" in scheduler.options


def test_preflight_reports_alignment_audio_and_reference_guidance():
    ready, warning_count, report = run_preflight(
        1344, 768, 123, "lock_source", video_vae=FakeVideoVAE(), audio_vae=FakeAudioVAE(),
        drive_audio=make_audio(1, value=0),
        ref_videos={"ref_video_1": torch.zeros((20, 32, 32, 3))},
    )
    data = json.loads(report)
    assert ready is True
    assert warning_count >= 3
    assert data["facts"]["aligned_frames"] == 124
    assert data["facts"]["video_vae_kind"] == "video"
    assert not any("swapped" in warning for warning in data["warnings"])


def test_preflight_allows_1080p_area_and_blocks_only_above_it():
    ready, warning_count, report = run_preflight(1920, 1088, 124, "native")
    data = json.loads(report)
    assert ready is True
    assert warning_count >= 1
    assert data["facts"]["pixels"] == 1920 * 1088
    assert any("VRAM" in warning for warning in data["warnings"])

    ready, _, report = run_preflight(1952, 1088, 124, "lock_source")
    assert ready is False
    assert len(json.loads(report)["errors"]) == 2


def test_preflight_distinguishes_h3_video_and_audio_vaes_by_latent_contract():
    ready, _, report = run_preflight(
        1344,
        768,
        124,
        "native",
        video_vae=FakeVideoVAE(),
        audio_vae=FakeAudioVAE(),
    )
    data = json.loads(report)
    assert ready is True
    assert data["facts"]["video_vae_kind"] == "video"
    assert data["facts"]["audio_vae_kind"] == "audio"

    ready, _, report = run_preflight(
        1344,
        768,
        124,
        "native",
        video_vae=FakeAudioVAE(),
        audio_vae=FakeVideoVAE(),
    )
    data = json.loads(report)
    assert ready is False
    assert any("video_vae is an H3 audio VAE" in error for error in data["errors"])
    assert any("audio_vae is an H3 video VAE" in error for error in data["errors"])


def test_example_api_workflow_is_valid_and_references_existing_nodes():
    path = Path(__file__).resolve().parents[1] / "examples" / "audio_lock_api.json"
    workflow = json.loads(path.read_text(encoding="utf-8"))
    custom_types = {value["class_type"] for value in workflow.values() if value["class_type"].endswith("T8")}
    assert custom_types == {
        "MiniMaxH3AudioWindowT8", "MiniMaxH3AudioConditioningT8",
        "MiniMaxH3AVDecodeT8", "MiniMaxH3OutputTrimT8",
    }
    node_ids = set(workflow)
    for node in workflow.values():
        for value in node["inputs"].values():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                assert value[0] in node_ids


def test_dual_clock_example_uses_one_coherent_sampling_setup():
    path = Path(__file__).resolve().parents[1] / "examples" / "dual_clock_4step_api.json"
    workflow = json.loads(path.read_text(encoding="utf-8"))
    dual_nodes = [value for value in workflow.values() if value["class_type"] == "MiniMaxH3DualClockSamplerT8"]
    assert len(dual_nodes) == 1
    assert dual_nodes[0]["inputs"]["steps"] == 4
    assert not any(value["class_type"] in {"MiniMaxH3SigmaShift", "KSamplerSelect", "BasicScheduler"}
                   for value in workflow.values())
    node_ids = set(workflow)
    for node in workflow.values():
        for value in node["inputs"].values():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                assert value[0] in node_ids


def test_multirate_exp_example_is_independent_and_uses_eight_joint_calls():
    path = Path(__file__).resolve().parents[1] / "examples" / "multirate_exp_api.json"
    workflow = json.loads(path.read_text(encoding="utf-8"))
    exp_nodes = [
        value for value in workflow.values()
        if value["class_type"] == "MiniMaxH3MultiRateSamplerEXPT8"
    ]
    assert len(exp_nodes) == 1
    assert exp_nodes[0]["inputs"]["video_steps"] == 4
    assert exp_nodes[0]["inputs"]["audio_steps"] == 8
    assert not any(
        value["class_type"] in {
            "MiniMaxH3DualClockSamplerT8",
            "MiniMaxH3SigmaShift",
            "KSamplerSelect",
            "BasicScheduler",
        }
        for value in workflow.values()
    )
    node_ids = set(workflow)
    for node in workflow.values():
        for value in node["inputs"].values():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                assert value[0] in node_ids


def test_still_image_edit_example_uses_ref2va_without_incompatible_lora():
    path = Path(__file__).resolve().parents[1] / "examples" / "still_image_edit_api.json"
    workflow = json.loads(path.read_text(encoding="utf-8"))
    types = {value["class_type"] for value in workflow.values()}
    assert "MiniMaxH3StillConditioningT8" in types
    assert "MiniMaxH3StillPreflightT8" in types
    assert "MiniMaxH3StillDecodeT8" in types
    assert "MiniMaxH3DualClockSamplerT8" in types
    assert not any("Lora" in node_type for node_type in types)

    unet = next(value for value in workflow.values() if value["class_type"] == "UNETLoader")
    assert "ref2va" in unet["inputs"]["unet_name"]
    conditioning = next(
        value for value in workflow.values()
        if value["class_type"] == "MiniMaxH3StillConditioningT8"
    )
    assert conditioning["inputs"]["target_mode"] == "short_video_22_frames"
    assert conditioning["inputs"]["canvas_mode"] == "custom"
    assert conditioning["inputs"]["width"] == 512
    assert conditioning["inputs"]["height"] == 512
    sampler = next(
        value for value in workflow.values()
        if value["class_type"] == "MiniMaxH3DualClockSamplerT8"
    )
    assert sampler["inputs"]["steps"] == 20

    node_ids = set(workflow)
    for node in workflow.values():
        for value in node["inputs"].values():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                assert value[0] in node_ids


def test_frontend_workflows_cover_stable_and_both_exp_step_counts():
    workflow_dir = Path(__file__).resolve().parents[1] / "examples" / "workflows"
    expected = {
        "H3_Turbo_Stable_4V4A.json": ("MiniMaxH3DualClockSamplerT8", [4, 12.0, 3.0]),
        "H3_Turbo_EXP_4V8A.json": ("MiniMaxH3MultiRateSamplerEXPT8", [4, 8, 12.0, 3.0]),
        "H3_Turbo_EXP_4V10A.json": ("MiniMaxH3MultiRateSamplerEXPT8", [4, 10, 12.0, 3.0]),
    }

    for filename, (sampler_type, sampler_widgets) in expected.items():
        workflow = json.loads((workflow_dir / filename).read_text(encoding="utf-8"))
        assert workflow["version"] == 0.4
        assert workflow["last_node_id"] == max(node["id"] for node in workflow["nodes"])
        assert workflow["last_link_id"] == max(link[0] for link in workflow["links"])

        nodes = {node["id"]: node for node in workflow["nodes"]}
        types = {node["type"] for node in nodes.values()}
        assert "LoraLoaderBypassModelOnly" in types
        assert "MiniMaxH3SigmaShift" not in types
        assert "BasicScheduler" not in types
        assert "KSamplerSelect" not in types

        sampler_nodes = [node for node in nodes.values() if node["type"] == sampler_type]
        assert len(sampler_nodes) == 1
        assert sampler_nodes[0]["widgets_values"] == sampler_widgets

        unet = next(node for node in nodes.values() if node["type"] == "UNETLoader")
        assert unet["widgets_values"][0] == "minimax_h3_fl2va_int8_convrot.safetensors"
        lora = next(
            node for node in nodes.values() if node["type"] == "LoraLoaderBypassModelOnly"
        )
        assert lora["widgets_values"][0] == (
            "minimax_h3_turbo_4步加速ema_comfyui.safetensors"
        )

        for link_id, source_id, source_slot, target_id, target_slot, link_type in workflow["links"]:
            source = nodes[source_id]
            target = nodes[target_id]
            assert link_id in source["outputs"][source_slot]["links"]
            assert target["inputs"][target_slot]["link"] == link_id
            assert source["outputs"][source_slot]["type"] == link_type
            assert target["inputs"][target_slot]["type"] == link_type


def test_frontend_still_edit_workflow_uses_native_22_frame_ref2va_target():
    path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "workflows"
        / "H3_Still_Edit_22Frames_EXP.json"
    )
    workflow = json.loads(path.read_text(encoding="utf-8"))
    assert workflow["version"] == 0.4
    assert workflow["last_node_id"] == max(node["id"] for node in workflow["nodes"])
    assert workflow["last_link_id"] == max(link[0] for link in workflow["links"])

    nodes = {node["id"]: node for node in workflow["nodes"]}
    types = {node["type"] for node in nodes.values()}
    assert {
        "MiniMaxH3StillConditioningT8",
        "MiniMaxH3StillPreflightT8",
        "MiniMaxH3StillDecodeT8",
        "MiniMaxH3DualClockSamplerT8",
        "SaveImage",
    } <= types
    assert not any("Lora" in node_type for node_type in types)

    unet = next(node for node in nodes.values() if node["type"] == "UNETLoader")
    assert unet["widgets_values"][0] == (
        "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
    )
    conditioning = next(
        node for node in nodes.values()
        if node["type"] == "MiniMaxH3StillConditioningT8"
    )
    assert conditioning["widgets_values"][1:7] == [
        "custom",
        512,
        512,
        "short_video_22_frames",
        0.999,
        "generate_and_discard",
    ]
    sampler = next(
        node for node in nodes.values()
        if node["type"] == "MiniMaxH3DualClockSamplerT8"
    )
    assert sampler["widgets_values"] == [20, 12.0, 3.0]

    for link_id, source_id, source_slot, target_id, target_slot, link_type in workflow["links"]:
        source = nodes[source_id]
        target = nodes[target_id]
        assert link_id in source["outputs"][source_slot]["links"]
        assert target["inputs"][target_slot]["link"] == link_id
        assert (
            source["outputs"][source_slot]["type"] == link_type
            or link_type == "*"
        )
        assert target["inputs"][target_slot]["type"] == link_type
