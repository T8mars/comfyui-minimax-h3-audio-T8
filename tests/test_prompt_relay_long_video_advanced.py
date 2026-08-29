from __future__ import annotations

import json

import pytest
import torch

import h3_audio_t8_pkg.prompt_relay_advanced as prompt_relay_module
from comfy.model_patcher import ModelPatcher
from h3_audio_t8_pkg.long_video import LONG_VIDEO_CONDITIONING_KEY, LONG_VIDEO_SCHEMA
from h3_audio_t8_pkg.nodes_prompt_relay_long_video_advanced import (
    PROMPT_RELAY_LONG_VIDEO_ADVANCED_NODE_CLASSES,
)
from h3_audio_t8_pkg.prompt_relay_advanced import (
    PROMPT_RELAY_BINDING_KEY,
    PROMPT_RELAY_PAYLOAD_KEY,
    build_prompt_relay_plan,
)
from h3_audio_t8_pkg.prompt_relay_long_video_advanced import (
    PROMPT_RELAY_LONG_VIDEO_ATTACHMENT_KEY,
    build_prompt_relay_long_video_conditioning,
    project_prompt_relay_plan_to_long_video_window,
)
from helpers import FakeAudioVAE, FakeVideoVAE


class _ByteHF:
    def __init__(self):
        self.byte_decoder = {chr(0x100 + value): value for value in range(256)}

    @staticmethod
    def convert_ids_to_tokens(token_id):
        return chr(0x100 + int(token_id))


class _ByteInner:
    def __init__(self):
        self.tokenizer = _ByteHF()

    @staticmethod
    def tokenize_with_weights(text, **_kwargs):
        return [[(int(value), 1.0) for value in text.encode("utf-8")]]


class _OuterTokenizer:
    def __init__(self):
        self.qwen3vl_32b = _ByteInner()


class NativeLikeFakeClip:
    def __init__(self):
        self.tokenizer = _OuterTokenizer()

    @staticmethod
    def tokenize(prompt, **kwargs):
        prefix_count = 0
        if kwargs.get("images"):
            prefix_count += 2 * len(kwargs["images"])
        if kwargs.get("minimax_ref_items"):
            prefix_count += 2 * len(kwargs["minimax_ref_items"])
        return {
            "qwen3vl_32b": [[
                *[(1000 + index, 1.0) for index in range(prefix_count)],
                *[(int(value), 1.0) for value in prompt.encode("utf-8")],
            ]]
        }

    @staticmethod
    def encode_from_tokens_scheduled(tokens):
        entries = tokens["qwen3vl_32b"][0]
        tags = torch.tensor(
            [0 if int(entry[0]) >= 1000 else 1 for entry in entries],
            dtype=torch.long,
        )
        return [[torch.zeros((1, len(entries), 8)), {"minimax_token_tags": tags}]]


class MiniMaxH3Model(torch.nn.Module):
    pass


class _NativeH3Base(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.diffusion_model = MiniMaxH3Model()

    def extra_conds(self, **_kwargs):
        return {}


_NativeH3Base.extra_conds.__module__ = "comfy.model_base"


def _model_patcher():
    return ModelPatcher(
        _NativeH3Base(),
        load_device=torch.device("cpu"),
        offload_device=torch.device("cpu"),
    )


def _allow_fixture_core_contract(monkeypatch):
    monkeypatch.setattr(prompt_relay_module, "_source_sha256", lambda _source: "fixture")
    monkeypatch.setattr(prompt_relay_module, "ATTENTION_FORWARD_SHA256S", {"fixture"})
    monkeypatch.setattr(prompt_relay_module, "PACKED_LAYOUT_SHA256S", {"fixture"})
    monkeypatch.setattr(prompt_relay_module, "TOKENIZER_SHA256S", {"fixture"})
    monkeypatch.setattr(prompt_relay_module, "EXTRA_CONDS_SHA256S", {"fixture"})


def _global_plan(length=345):
    return build_prompt_relay_plan(
        global_prompt="同一个人物、场景与连续声音",
        local_prompts="人物抬手并响起钟声\n人物奔跑，只有脚步声\n镜头拉远并响起雷声",
        length=length,
        timing_mode="auto_equal",
        time_ranges="",
        math_profile="paper_v1",
        epsilon=0.1,
        allow_gaps=False,
        allow_overlaps=False,
    )[0]


def test_projection_uses_accepted_start_minus_context_and_preserves_global_sigma():
    source = _global_plan()
    projected, prompt, report = project_prompt_relay_plan_to_long_video_window(
        source,
        segment_index=1,
        length=124,
        context_frames=22,
        timeline_start_seconds=124 / 24,
        timeline_end_seconds=226 / 24,
    )
    parsed = json.loads(report)
    assert prompt == source["compiled_prompt"]
    assert projected["frame_count"] == 124
    assert projected["long_video_projection"]["render_start_frame"] == 102
    assert parsed["render_window_frames"] == [102, 226]
    assert parsed["accepted_window_frames"] == [124, 226]
    assert parsed["render_active_event_indices"] == [1, 2]
    assert parsed["accepted_active_event_indices"] == [2]

    coordinate_shift = (5 / 3) * 102
    for original, local in zip(source["events"], projected["events"], strict=True):
        assert local["midpoint"] + coordinate_shift == pytest.approx(original["midpoint"])
        assert local["window"] == original["window"]
        assert local["sigma"] == original["sigma"]
    # Event 1 crosses into the context head. It must not be clamped to local frame 0
    # and re-estimated as a short event.
    assert projected["events"][0]["start_frame"] == -102
    assert projected["events"][0]["end_frame_exclusive"] == 13


def test_projection_fails_closed_on_wrong_grid_or_global_duration():
    source = _global_plan(124)
    with pytest.raises(ValueError, match="context_frames=0"):
        project_prompt_relay_plan_to_long_video_window(source, 0, 124, 22, 0, 124 / 24)
    with pytest.raises(ValueError, match=r"17n\+5"):
        project_prompt_relay_plan_to_long_video_window(source, 0, 123, 0, 0, 123 / 24)
    with pytest.raises(ValueError, match="exceeds the global"):
        project_prompt_relay_plan_to_long_video_window(source, 0, 141, 0, 0, 130 / 24)


def test_combined_segment_zero_installs_both_scoped_contracts(monkeypatch):
    _allow_fixture_core_contract(monkeypatch)
    source = _global_plan(124)
    projected, *_ = project_prompt_relay_plan_to_long_video_window(
        source,
        segment_index=0,
        length=124,
        context_frames=0,
        timeline_start_seconds=0,
        timeline_end_seconds=124 / 24,
    )
    context = {
        "schema": LONG_VIDEO_SCHEMA,
        "empty": True,
        "chain_id": "relay-long-video",
        "target_segment_index": 0,
    }
    result = build_prompt_relay_long_video_conditioning(
        model=_model_patcher(),
        clip=NativeLikeFakeClip(),
        video_vae=FakeVideoVAE(),
        audio_vae=FakeAudioVAE(),
        context=context,
        prompt_relay_plan=projected,
        segment_index=0,
        context_frames=0,
        context_audio="video_and_audio",
        width=128,
        height=128,
        length=124,
        task_type="T2VA",
        audio_mode="native",
        audio_denoise_strength=0.35,
        add_source_as_reference=False,
        prompt_primary_audio_ordinal=0,
        strict_prompt_tags=True,
        ref_image_size="match",
        reference_video_policy="official_2_to_15s",
        execution_mode="apply_exp",
        query_chunk_rows=64,
    )
    patched, conditioning, latent, _audio, _prompt, _media, report = result
    metadata = conditioning[0][1]
    parsed = json.loads(report)
    assert metadata[LONG_VIDEO_CONDITIONING_KEY] == LONG_VIDEO_SCHEMA
    assert PROMPT_RELAY_BINDING_KEY in metadata
    assert metadata["model_conds"][PROMPT_RELAY_PAYLOAD_KEY].cond == (
        metadata[PROMPT_RELAY_BINDING_KEY]["binding_hash"]
    )
    assert getattr(
        patched.get_model_object("extra_conds"),
        "_t8_long_video_patch_version",
        None,
    ) == 1
    assert patched.get_attachment(PROMPT_RELAY_LONG_VIDEO_ATTACHMENT_KEY)[
        "projected_plan_hash"
    ] == projected["plan_hash"]
    assert parsed["status"] == "applied_exp"
    assert parsed["audio_mode"] == "native"
    assert json.loads(parsed["stable_conditioning_report"])["audio_mode"] == "native"
    assert parsed["render_window_frames"] == [0, 124]
    assert parsed["dense_s_by_s_mask_created"] is False
    assert latent["samples"].is_nested


@pytest.mark.parametrize(
    ("local_prompts", "expected_status"),
    [
        ("", "passthrough_no_events_long_video_only"),
        ("One global-length local event.", "passthrough_single_event_long_video_only"),
    ],
)
def test_zero_or_one_event_bypasses_only_relay_but_keeps_long_video_patch(
    monkeypatch,
    local_prompts,
    expected_status,
):
    _allow_fixture_core_contract(monkeypatch)
    source = build_prompt_relay_plan(
        global_prompt="One stable global scene.",
        local_prompts=local_prompts,
        length=124,
        timing_mode="auto_equal",
        time_ranges="",
        math_profile="paper_v1",
        epsilon=0.1,
        allow_gaps=False,
        allow_overlaps=False,
    )[0]
    projected, *_ = project_prompt_relay_plan_to_long_video_window(
        source,
        segment_index=0,
        length=124,
        context_frames=0,
        timeline_start_seconds=0,
        timeline_end_seconds=124 / 24,
    )
    result = build_prompt_relay_long_video_conditioning(
        model=_model_patcher(),
        clip=NativeLikeFakeClip(),
        video_vae=FakeVideoVAE(),
        audio_vae=FakeAudioVAE(),
        context={
            "schema": LONG_VIDEO_SCHEMA,
            "empty": True,
            "chain_id": "relay-long-video-bypass",
            "target_segment_index": 0,
        },
        prompt_relay_plan=projected,
        segment_index=0,
        context_frames=0,
        context_audio="video_and_audio",
        width=128,
        height=128,
        length=124,
        task_type="T2VA",
        audio_mode="native",
        audio_denoise_strength=0.35,
        add_source_as_reference=False,
        prompt_primary_audio_ordinal=0,
        strict_prompt_tags=True,
        ref_image_size="match",
        reference_video_policy="official_2_to_15s",
        execution_mode="apply_exp",
        query_chunk_rows=64,
    )
    patched, conditioning, _latent, _audio, _prompt, _media, report = result
    metadata = conditioning[0][1]
    parsed = json.loads(report)
    assert metadata[LONG_VIDEO_CONDITIONING_KEY] == LONG_VIDEO_SCHEMA
    assert PROMPT_RELAY_BINDING_KEY not in metadata
    assert PROMPT_RELAY_PAYLOAD_KEY not in metadata.get("model_conds", {})
    assert getattr(
        patched.get_model_object("extra_conds"),
        "_t8_long_video_patch_version",
        None,
    ) == 1
    assert patched.get_attachment(PROMPT_RELAY_LONG_VIDEO_ATTACHMENT_KEY) is None
    assert parsed["status"] == expected_status
    assert parsed["attention_patch_installed"] is False
    assert parsed["event_count"] == (1 if local_prompts else 0)


def test_combined_continuation_binds_repaired_motion_context_layout(monkeypatch):
    _allow_fixture_core_contract(monkeypatch)
    source = _global_plan(345)
    projected, *_ = project_prompt_relay_plan_to_long_video_window(
        source,
        segment_index=1,
        length=124,
        context_frames=22,
        timeline_start_seconds=124 / 24,
        timeline_end_seconds=226 / 24,
    )
    context = {
        "schema": LONG_VIDEO_SCHEMA,
        "empty": False,
        "video_tail": torch.zeros((1, 24, 12, 8, 8)),
        "audio_tail": torch.zeros((1, 32, 2, 65)),
        "metadata": {
            "source_segment_index": 0,
            "target_segment_index": 1,
            "max_context_frames": 39,
            "audio_overhang": 1 / 3,
        },
    }
    result = build_prompt_relay_long_video_conditioning(
        model=_model_patcher(),
        clip=NativeLikeFakeClip(),
        video_vae=FakeVideoVAE(),
        audio_vae=FakeAudioVAE(),
        context=context,
        prompt_relay_plan=projected,
        segment_index=1,
        context_frames=22,
        context_audio="video_and_audio",
        width=128,
        height=128,
        length=124,
        task_type="auto",
        audio_mode="native",
        audio_denoise_strength=0.35,
        add_source_as_reference=False,
        prompt_primary_audio_ordinal=0,
        strict_prompt_tags=True,
        ref_image_size="match",
        reference_video_policy="official_2_to_15s",
        execution_mode="apply_exp",
        query_chunk_rows=64,
    )
    _patched, conditioning, _latent, _audio, _prompt, _media, report = result
    metadata = conditioning[0][1]
    binding = metadata[PROMPT_RELAY_BINDING_KEY]
    parsed = json.loads(report)
    assert parsed["resolved_task"] == "i2va-motion"
    assert parsed["context_frames"] == 22
    assert parsed["render_window_frames"] == [102, 226]
    assert binding["keyframe_count"] == 7
    assert binding["reference_block_count"] == 1
    assert binding["layout_contract"]["segments"][-2][2] == "audio"
    assert binding["layout_contract"]["segments"][-1][2] == "video"


def test_combined_continuation_recovers_known_live_long_video_patch(monkeypatch):
    _allow_fixture_core_contract(monkeypatch)
    source_model = _model_patcher()
    base_model = source_model.model

    from h3_audio_t8_pkg.long_video import patch_long_video_model

    segment_zero_model = patch_long_video_model(source_model)
    live_patch = segment_zero_model.get_model_object("extra_conds")
    base_model.extra_conds = live_patch
    try:
        source = _global_plan(345)
        projected, *_ = project_prompt_relay_plan_to_long_video_window(
            source,
            segment_index=1,
            length=124,
            context_frames=22,
            timeline_start_seconds=124 / 24,
            timeline_end_seconds=226 / 24,
        )
        context = {
            "schema": LONG_VIDEO_SCHEMA,
            "empty": False,
            "video_tail": torch.zeros((1, 24, 12, 8, 8)),
            "audio_tail": torch.zeros((1, 32, 2, 65)),
            "metadata": {
                "source_segment_index": 0,
                "target_segment_index": 1,
                "max_context_frames": 39,
                "audio_overhang": 1 / 3,
            },
        }
        result = build_prompt_relay_long_video_conditioning(
            model=source_model,
            clip=NativeLikeFakeClip(),
            video_vae=FakeVideoVAE(),
            audio_vae=FakeAudioVAE(),
            context=context,
            prompt_relay_plan=projected,
            segment_index=1,
            context_frames=22,
            context_audio="video_and_audio",
            width=128,
            height=128,
            length=124,
            task_type="auto",
            audio_mode="native",
            audio_denoise_strength=0.35,
            add_source_as_reference=False,
            prompt_primary_audio_ordinal=0,
            strict_prompt_tags=True,
            ref_image_size="match",
            reference_video_policy="official_2_to_15s",
            execution_mode="apply_exp",
            query_chunk_rows=64,
        )
        patched = result[0]
        assert "extra_conds" in patched.object_patches
        normalized_patch = patched.get_model_object("extra_conds")
        assert getattr(
            normalized_patch,
            "_t8_long_video_patch_version",
            None,
        ) == 1
    finally:
        del base_model.extra_conds


def test_conditioning_rejects_a_projected_plan_from_another_segment():
    source = _global_plan(345)
    projected, *_ = project_prompt_relay_plan_to_long_video_window(
        source, 1, 124, 22, 124 / 24, 226 / 24
    )
    with pytest.raises(ValueError, match="does not match Conditioning inputs"):
        build_prompt_relay_long_video_conditioning(
            model=object(),
            clip=NativeLikeFakeClip(),
            video_vae=FakeVideoVAE(),
            audio_vae=FakeAudioVAE(),
            context={},
            prompt_relay_plan=projected,
            segment_index=2,
            context_frames=22,
            context_audio="video_only",
            width=128,
            height=128,
            length=124,
            task_type="auto",
            audio_mode="native",
            audio_denoise_strength=0.35,
            add_source_as_reference=False,
            prompt_primary_audio_ordinal=0,
            strict_prompt_tags=True,
            ref_image_size="match",
            reference_video_policy="official_2_to_15s",
            execution_mode="report_only",
            query_chunk_rows=64,
        )


def test_long_video_prompt_relay_nodes_are_append_only_advanced_nodes():
    schemas = [node.define_schema() for node in PROMPT_RELAY_LONG_VIDEO_ADVANCED_NODE_CLASSES]
    assert [schema.node_id for schema in schemas] == [
        "MiniMaxH3PromptRelayLongVideoPlanT8Advanced",
        "MiniMaxH3PromptRelayLongVideoConditioningT8Advanced",
    ]
    assert all(schema.is_experimental is True for schema in schemas)
    assert all(schema.category == "T8/MiniMax H3/Long Video/Experimental" for schema in schemas)
    conditioning_inputs = {item.id: item for item in schemas[1].inputs}
    assert conditioning_inputs["execution_mode"].default == "report_only"
    assert conditioning_inputs["audio_mode"].default == "native"
