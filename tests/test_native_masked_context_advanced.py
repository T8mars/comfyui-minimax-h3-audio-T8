from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import comfy.nested_tensor
import comfy.sampler_helpers
import comfy.utils
from comfy.model_base import MiniMaxH3

from h3_audio_t8_pkg.native_masked_context_advanced import (
    apply_native_masked_video_context,
    require_native_h3_av_mask_support,
)


CONTEXT_STEPS = {5: 2, 22: 7, 39: 12}


def _target(*, frames: int = 124, mask: bool = True):
    video_t = 2 + ((frames - 5) // 17) * 5
    audio_t = round(frames / 24 * 40)
    video = torch.arange(24 * video_t * 4 * 6, dtype=torch.float32).reshape(
        1, 24, video_t, 4, 6
    )
    audio = torch.arange(32 * 2 * audio_t, dtype=torch.float32).reshape(
        1, 32, 2, audio_t
    )
    latent = {
        "samples": comfy.nested_tensor.NestedTensor((video, audio)),
        "marker": {"preserved": True},
    }
    if mask:
        video_mask = torch.ones((1, 1, video_t, 4, 6), dtype=torch.float32)
        audio_mask = torch.full((1, 1, 2, audio_t), 0.0, dtype=torch.float32)
        latent["noise_mask"] = comfy.nested_tensor.NestedTensor(
            (video_mask, audio_mask)
        )
    return latent


def _context(context_frames: int, *, segment_index: int = 1):
    steps = CONTEXT_STEPS[context_frames]
    video_tail = torch.full((1, 24, steps, 4, 6), 9.0)
    audio_steps = round(context_frames / 24 * 40)
    audio_tail = torch.full((1, 32, 2, audio_steps), 7.0)
    return {
        "schema": 1,
        "empty": False,
        "video_tail": video_tail,
        "audio_tail": audio_tail,
        "metadata": {
            "schema": 1,
            "chain_id": "masked_plan_b",
            "source_segment_index": segment_index - 1,
            "target_segment_index": segment_index,
            "fps": 24,
            "source_total_frames": 124,
            "max_context_frames": context_frames,
            "audio_overhang": 0.0,
        },
    }


def _reports(
    context_frames: int,
    *,
    segment_index: int = 1,
    context_audio: str = "video_only",
    render_frames: int = 124,
):
    planner = {
        "chain_id": "masked_plan_b",
        "segment_index": segment_index,
        "render_frames": render_frames,
        "context_frames": context_frames,
        "trim_start_seconds": context_frames / 24,
        "timeline_start_seconds": 124 / 24,
        "timeline_end_seconds": (124 + render_frames - context_frames) / 24,
    }
    conditioning = {
        "schema": 1,
        "segment_index": segment_index,
        "context_active": True,
        "context_frames": context_frames,
        "context_audio": context_audio,
        "render_frames": render_frames,
        "motion_keyframes": CONTEXT_STEPS[context_frames],
        "timeline_audio_ref": context_audio == "video_and_audio",
    }
    return json.dumps(planner), json.dumps(conditioning)


@pytest.mark.parametrize("context_frames", [5, 22, 39])
def test_video_only_context_injects_native_tail_and_preserves_audio_identity(context_frames):
    latent = _target(mask=True)
    source_video, source_audio = tuple(latent["samples"].unbind())
    source_video_mask, source_audio_mask = tuple(latent["noise_mask"].unbind())
    planner, conditioning = _reports(context_frames)

    output, trim_frames, report_text = apply_native_masked_video_context(
        latent,
        _context(context_frames),
        planner,
        conditioning,
    )

    video, audio = tuple(output["samples"].unbind())
    video_mask, audio_mask = tuple(output["noise_mask"].unbind())
    steps = CONTEXT_STEPS[context_frames]
    report = json.loads(report_text)

    assert trim_frames == context_frames
    assert video is not source_video
    assert audio is source_audio
    assert audio_mask is source_audio_mask
    assert torch.all(video[:, :, :steps] == 9.0)
    assert torch.equal(video[:, :, steps:], source_video[:, :, steps:])
    assert torch.count_nonzero(video_mask[:, :, :steps]).item() == 0
    assert torch.equal(video_mask[:, :, steps:], source_video_mask[:, :, steps:])
    assert torch.count_nonzero(source_video == 9.0).item() == 1
    assert output["marker"] == {"preserved": True}
    assert report["status"] == "VIDEO_CONTEXT_APPLIED_AUDIO_UNTOUCHED"
    assert report["context_frames"] == context_frames
    assert report["audio_samples_reused"] is True
    assert report["audio_noise_mask_reused"] is True
    assert report["source_video_vae_roundtrip"] is False


def test_missing_noise_mask_creates_only_semantic_all_generate_audio_mask():
    latent = _target(mask=False)
    _source_video, source_audio = tuple(latent["samples"].unbind())
    planner, conditioning = _reports(22)

    output, _, report_text = apply_native_masked_video_context(
        latent,
        _context(22),
        planner,
        conditioning,
    )

    _video_mask, audio_mask = tuple(output["noise_mask"].unbind())
    assert tuple(output["samples"].unbind())[1] is source_audio
    assert torch.all(audio_mask == 1.0)
    report = json.loads(report_text)
    assert report["audio_noise_mask_reused"] is False
    assert report["audio_noise_mask_policy"] == "created_all_generate_equivalent"


def test_existing_locked_video_prefix_fails_closed_instead_of_overwriting_it():
    latent = _target(mask=True)
    video_mask, _audio_mask = tuple(latent["noise_mask"].unbind())
    video_mask[:, :, 0] = 0.25
    planner, conditioning = _reports(22)

    with pytest.raises(ValueError, match="already contains locked or partial video mask"):
        apply_native_masked_video_context(
            latent,
            _context(22),
            planner,
            conditioning,
        )


@pytest.mark.parametrize(
    ("planner_mutation", "conditioning_mutation", "message"),
    [
        ({"segment_index": 0}, {}, "continuation segments"),
        ({"chain_id": "wrong"}, {}, "chain_id"),
        ({"context_frames": 39}, {}, "context_frames"),
        ({}, {"context_audio": "video_and_audio"}, "requires context_audio=video_only"),
        ({}, {"context_active": False}, "active previous context"),
        ({}, {"render_frames": 107}, "render_frames"),
    ],
)
def test_report_and_audio_contract_mismatches_fail_closed(
    planner_mutation, conditioning_mutation, message
):
    planner_text, conditioning_text = _reports(22)
    planner = json.loads(planner_text)
    conditioning = json.loads(conditioning_text)
    planner.update(planner_mutation)
    conditioning.update(conditioning_mutation)

    with pytest.raises(ValueError, match=message):
        apply_native_masked_video_context(
            _target(mask=True),
            _context(22),
            json.dumps(planner),
            json.dumps(conditioning),
        )


def test_context_geometry_and_native_mask_shapes_fail_closed():
    planner, conditioning = _reports(22)
    wrong_canvas = _context(22)
    wrong_canvas["video_tail"] = torch.zeros((1, 24, 7, 5, 6))
    with pytest.raises(ValueError, match="same latent canvas"):
        apply_native_masked_video_context(
            _target(mask=True), wrong_canvas, planner, conditioning
        )

    pixel_mask = _target(mask=True)
    _video_mask, audio_mask = tuple(pixel_mask["noise_mask"].unbind())
    pixel_mask["noise_mask"] = comfy.nested_tensor.NestedTensor(
        (torch.ones((1, 1, 416, 736)), audio_mask)
    )
    with pytest.raises(ValueError, match="native latent-aligned video mask"):
        apply_native_masked_video_context(
            pixel_mask, _context(22), planner, conditioning
        )


def test_current_comfyui_exposes_native_h3_av_mask_contract():
    report = require_native_h3_av_mask_support()
    assert report["supported"] is True
    assert report["runtime_patch_installed"] is False


def test_output_mask_survives_current_comfy_sampler_pack_and_h3_token_pooling():
    latent = _target(mask=True)
    planner, conditioning = _reports(22)
    output, _, _ = apply_native_masked_video_context(
        latent,
        _context(22),
        planner,
        conditioning,
    )
    video, audio = tuple(output["samples"].unbind())
    video_mask, audio_mask = tuple(output["noise_mask"].unbind())
    prepared = [
        comfy.sampler_helpers.prepare_mask(video_mask, video.shape, "cpu"),
        comfy.sampler_helpers.prepare_mask(audio_mask, audio.shape, "cpu"),
    ]
    packed, latent_shapes = comfy.utils.pack_latents(prepared)

    h3 = object.__new__(MiniMaxH3)
    h3.diffusion_model = type("DummyDiffusion", (), {"patch_size": (1, 2, 2)})()
    token_masks = h3._token_grid_masks(packed, latent_shapes)
    assert token_masks[0].shape == video.shape
    assert token_masks[1].shape == audio.shape
    assert torch.count_nonzero(token_masks[0][:, :, :7]).item() == 0
    assert torch.all(token_masks[0][:, :, 7:] == 1.0)
    assert torch.count_nonzero(token_masks[1]).item() == 0
    conds = h3._denoise_mask_values(packed, latent_shapes)
    assert set(conds) == {"denoise_mask", "audio_denoise_mask"}
    assert torch.count_nonzero(conds["denoise_mask"][:, :, :7]).item() == 0
    assert torch.count_nonzero(conds["audio_denoise_mask"]).item() == 0


def test_plan_b_frontend_workflow_is_independent_video_only_and_report_bound():
    from tools.build_native_masked_context_workflow import NEW_EMA_B_LORA, build

    root = Path(__file__).resolve().parents[1]
    source_path = (
        root
        / "examples"
        / "workflows"
        / "04-long-video"
        / "2026-08-09_H3_Long_Video_22F_EXP.json"
    )
    path = (
        root
        / "examples"
        / "workflows"
        / "04-long-video"
        / "2026-09-02_H3_Native_Masked_Video_Context_Plan_B_Advanced_EXP.json"
    )
    source = json.loads(source_path.read_text(encoding="utf-8"))
    workflow = json.loads(path.read_text(encoding="utf-8"))
    assert build(source) == workflow
    assert source["id"] != workflow["id"]

    nodes = {node["id"]: node for node in workflow["nodes"]}
    by_type = {node["type"]: node for node in workflow["nodes"]}
    assert workflow["version"] == 0.4
    assert workflow["last_node_id"] == max(nodes)
    assert workflow["last_link_id"] == max(link[0] for link in workflow["links"])
    assert len(nodes) == len(workflow["nodes"])

    planner = by_type["MiniMaxH3LongVideoPlannerT8"]
    context_load = by_type["MiniMaxH3LongVideoContextLoadT8"]
    conditioning = by_type["MiniMaxH3LongVideoConditioningT8"]
    masked = by_type["MiniMaxH3NativeMaskedVideoContextT8Advanced"]
    dual_clock = by_type["MiniMaxH3DualClockSamplerT8"]
    sampler = by_type["SamplerCustomAdvanced"]
    context_save = by_type["MiniMaxH3LongVideoContextSaveT8"]
    output_trim = by_type["MiniMaxH3OutputTrimT8"]
    color_match = by_type["MiniMaxH3LongVideoColorMatchT8Advanced"]
    create_video = by_type["CreateVideo"]
    save_video = by_type["SaveVideo"]
    lora = by_type["LoraLoaderBypassModelOnly"]
    assert planner["widgets_values"][1] == 1
    assert conditioning["widgets_values"][0] == "video_only"
    assert save_video["widgets_values"][0] == (
        "MiniMaxH3/long_video_masked_plan_b_segment"
    )
    assert dual_clock["widgets_values"][3:] == ["euler", "native_flow"]
    assert context_save["widgets_values"][1] == (
        "4-step euler/native_flow ComfyUI ModelSamplingAV shift12/3"
    )
    assert lora["widgets_values"] == [NEW_EMA_B_LORA, 1.0]

    links = {link[0]: link for link in workflow["links"]}

    def input_source(node: dict, input_slot: int) -> tuple[int, int]:
        link = links[node["inputs"][input_slot]["link"]]
        return link[1], link[2]

    assert input_source(masked, 0) == (conditioning["id"], 2)
    assert input_source(masked, 1) == (context_load["id"], 0)
    assert input_source(masked, 2) == (planner["id"], 9)
    assert input_source(masked, 3) == (conditioning["id"], 6)
    assert input_source(dual_clock, 1) == (masked["id"], 0)
    assert input_source(sampler, 4) == (masked["id"], 0)
    assert input_source(color_match, 0) == (output_trim["id"], 0)
    assert input_source(color_match, 1) == (context_load["id"], 0)
    assert input_source(color_match, 2) == (planner["id"], 0)
    assert input_source(color_match, 3) == (planner["id"], 1)
    assert input_source(create_video, 0) == (color_match["id"], 0)
    assert color_match["widgets_values"] == [True, 5, 24, 1.0, 0.0005, 0.02, 0.18]
    assert not any(
        link[1] == conditioning["id"]
        and link[2] == 2
        and link[3] in {dual_clock["id"], sampler["id"]}
        for link in workflow["links"]
    )

    notes = "\n".join(
        node["widgets_values"][0]
        for node in workflow["nodes"]
        if node["type"] == "MarkdownNote"
    )
    for required in (
        "Plan B",
        "context_audio=video_only",
        "audio tensor",
        "Vocal Lock audio mask",
        "不替换现有 Long Video 默认路线",
        "当前核心",
        "原生 AV `euler + native_flow`",
        "完整续段试听",
        "通用 16GB 安全",
        "声音非常轻",
        "Color Match 默认开启",
        "只改显示域 RGB",
    ):
        assert required in notes

    for link_id, source_id, output_slot, target_id, input_slot, link_type in workflow[
        "links"
    ]:
        assert nodes[target_id]["inputs"][input_slot]["link"] == link_id
        assert link_id in (nodes[source_id]["outputs"][output_slot].get("links") or [])
        assert nodes[source_id]["outputs"][output_slot]["type"] == link_type
        assert nodes[target_id]["inputs"][input_slot]["type"] == link_type


def test_plan_b_segment_zero_starter_pins_the_same_native_av_sampler():
    from tools.build_native_masked_context_workflow import (
        NEW_EMA_B_LORA,
        build_starter,
    )

    root = Path(__file__).resolve().parents[1]
    source_path = (
        root
        / "examples"
        / "workflows"
        / "04-long-video"
        / "2026-08-09_H3_Long_Video_22F_EXP.json"
    )
    starter_path = (
        root
        / "examples"
        / "workflows"
        / "04-long-video"
        / "2026-09-02_H3_Native_Masked_Video_Context_Plan_B_Segment0_Starter_Advanced_EXP.json"
    )
    source = json.loads(source_path.read_text(encoding="utf-8"))
    starter = json.loads(starter_path.read_text(encoding="utf-8"))
    assert build_starter(source) == starter
    assert source["id"] != starter["id"]

    by_type = {node["type"]: node for node in starter["nodes"]}
    planner = by_type["MiniMaxH3LongVideoPlannerT8"]
    conditioning = by_type["MiniMaxH3LongVideoConditioningT8"]
    sampler_setup = by_type["MiniMaxH3DualClockSamplerT8"]
    context_save = by_type["MiniMaxH3LongVideoContextSaveT8"]
    context_load = by_type["MiniMaxH3LongVideoContextLoadT8"]
    output_trim = by_type["MiniMaxH3OutputTrimT8"]
    color_match = by_type["MiniMaxH3LongVideoColorMatchT8Advanced"]
    create_video = by_type["CreateVideo"]
    save_video = by_type["SaveVideo"]
    lora = by_type["LoraLoaderBypassModelOnly"]
    assert planner["widgets_values"][1] == 0
    assert planner["widgets_values"][6] is False
    assert conditioning["widgets_values"][0] == "video_only"
    assert sampler_setup["widgets_values"][3:] == ["euler", "native_flow"]
    assert context_save["widgets_values"][1] == (
        "4-step euler/native_flow ComfyUI ModelSamplingAV shift12/3"
    )
    assert save_video["widgets_values"][0] == (
        "MiniMaxH3/long_video_masked_plan_b_segment0"
    )
    assert lora["widgets_values"] == [NEW_EMA_B_LORA, 1.0]
    links = {link[0]: link for link in starter["links"]}

    def input_source(node: dict, input_slot: int) -> tuple[int, int]:
        link = links[node["inputs"][input_slot]["link"]]
        return link[1], link[2]

    assert input_source(color_match, 0) == (output_trim["id"], 0)
    assert input_source(color_match, 1) == (context_load["id"], 0)
    assert input_source(color_match, 2) == (planner["id"], 0)
    assert input_source(color_match, 3) == (planner["id"], 1)
    assert input_source(create_video, 0) == (color_match["id"], 0)
    assert color_match["widgets_values"] == [True, 5, 24, 1.0, 0.0005, 0.02, 0.18]
    assert "MiniMaxH3NativeMaskedVideoContextT8Advanced" not in by_type
    notes = "\n".join(
        node["widgets_values"][0]
        for node in starter["nodes"]
        if node["type"] == "MarkdownNote"
    )
    for required in (
        "第 0 段启动器",
        "segment_index=0",
        "is_final_segment=false",
        "原生 AV `euler + native_flow`",
        "完全相同的 `chain_id`",
        "旧 `dual_clock_euler`",
        "新版 step600 `EMA_B` LoRA",
        "Color Match 默认开启",
    ):
        assert required in notes

    source_sampler = next(
        node for node in source["nodes"] if node["type"] == "MiniMaxH3DualClockSamplerT8"
    )
    assert source_sampler["widgets_values"][3:] == ["dual_clock_euler", "native_flow"]
