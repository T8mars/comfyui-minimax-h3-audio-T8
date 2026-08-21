from __future__ import annotations

import json
import math
import types
from pathlib import Path

import pytest
import torch

import h3_audio_t8_pkg.prompt_relay_advanced as prompt_relay_module
from h3_audio_t8_pkg.conditioning import build_conditioning, build_packed_layout
from h3_audio_t8_pkg.nodes_prompt_relay_advanced import (
    PROMPT_RELAY_ADVANCED_NODE_CLASSES,
    build_prompt_relay_plan_with_optional_events,
)
from h3_audio_t8_pkg.nodes_prompt_relay_packet_advanced import (
    PROMPT_RELAY_PACKET_ADVANCED_NODE_CLASSES,
)
from h3_audio_t8_pkg.nodes_prompt_relay_preview_advanced import (
    PROMPT_RELAY_PREVIEW_ADVANCED_NODE_CLASSES,
)
from h3_audio_t8_pkg.prompt_relay_advanced import (
    ATTENTION_FORWARD_SHA256S,
    PACKED_LAYOUT_SHA256S,
    PROMPT_RELAY_BINDING_KEY,
    PROMPT_RELAY_PAYLOAD_KEY,
    PROMPT_RELAY_RUNTIME_KEY,
    PROMPT_RELAY_WRAPPER_KEY,
    TOKENIZER_SHA256S,
    _assert_core_contract,
    _bind_layout_contract,
    _runtime_route,
    _source_sha256,
    build_prompt_relay_binding,
    build_prompt_relay_conditioning,
    build_prompt_relay_plan,
    configure_prompt_relay_query_route,
    make_prompt_relay_bias,
    patch_prompt_relay_model,
    route_prompt_relay_attention,
)
from h3_audio_t8_pkg.prompt_relay_packet_advanced import (
    PROMPT_RELAY_EVENTS_TYPE,
    build_prompt_relay_event,
    build_prompt_relay_plan_from_packet,
)
from h3_audio_t8_pkg.prompt_relay_events_advanced import json_hash
from h3_audio_t8_pkg.prompt_relay_preview_advanced import preview_prompt_relay_plan
from h3_audio_t8_pkg.studio_advanced import compile_prompt_packet
import comfy.patcher_extension
from comfy.ldm.minimax.model import Attention, PackedLayout
from comfy.model_patcher import ModelPatcher
from comfy.text_encoders.minimax import MiniMaxH3Tokenizer
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

    def tokenize(self, prompt, **kwargs):
        prefix_count = 0
        if kwargs.get("images"):
            prefix_count += 2 * len(kwargs["images"])
        if kwargs.get("minimax_ref_items"):
            prefix_count += 2 * len(kwargs["minimax_ref_items"])
        return {
            "qwen3vl_32b": [
                [(1000 + index, 1.0) for index in range(prefix_count)]
                + [(int(value), 1.0) for value in prompt.encode("utf-8")]
            ]
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


def _native_h3_model_patcher() -> ModelPatcher:
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


def _plan(**overrides):
    kwargs = {
        "global_prompt": "夜晚，稳定人物与电影光线",
        "local_prompts": "女人抬手看向镜头\n女人转身跑向远处\n镜头快速拉远",
        "length": 124,
        "timing_mode": "auto_equal",
        "time_ranges": "",
        "math_profile": "paper_v1",
        "epsilon": 0.1,
        "allow_gaps": False,
        "allow_overlaps": False,
    }
    kwargs.update(overrides)
    return build_prompt_relay_plan(**kwargs)


def test_paper_v1_endpoint_weight_matches_epsilon():
    plan, _prompt, frame_count, _timeline, report = _plan()
    assert frame_count == 124
    for event in plan["events"]:
        assert event["endpoint_weight"] == pytest.approx(0.1, abs=1e-12)
    parsed = json.loads(report)
    assert parsed["paper_equation"] is True
    assert parsed["event_count"] == 3


def _prompt_packet(**overrides):
    kwargs = {
        "prompt": "Night street with one consistent heroine and stable lighting.",
        "backend": "minimax_h3",
        "duration_seconds": 5.0,
        "aspect_ratio": "16:9",
        "dialogue": "",
        "negative_prompt": "",
        "strict_exact_dialogue": True,
    }
    kwargs.update(overrides)
    return compile_prompt_packet(**kwargs)


def _packet_plan(packet=None, **overrides):
    kwargs = {
        "prompt_packet": packet or _prompt_packet(),
        "events_json": json.dumps(
            [
                {"prompt": "The heroine raises one hand."},
                {"prompt": "She turns and walks forward."},
            ]
        ),
        "timing_mode": "auto_equal",
        "math_profile": "paper_v1",
        "epsilon": 0.1,
        "allow_gaps": False,
        "allow_overlaps": False,
    }
    kwargs.update(overrides)
    return build_prompt_relay_plan_from_packet(**kwargs)


def test_prompt_packet_relay_uses_compiled_prompt_and_aligns_duration():
    packet = _prompt_packet(duration_seconds=5.0)
    plan, compiled, frame_count, timeline, report = _packet_plan(packet)

    assert frame_count == 124
    assert plan["global_prompt"] == packet["compiled_prompt"]
    assert plan["source_prompt_packet"] == {
        "schema": packet["schema"],
        "packet_hash": packet["packet_hash"],
        "backend": "minimax_h3",
        "duration_seconds": 5.0,
        "aspect_ratio": "16:9",
    }
    assert compiled.startswith("Global scene: " + packet["compiled_prompt"])
    assert len(json.loads(timeline)["events"]) == 2
    parsed = json.loads(report)
    assert parsed["requested_frame_count"] == 120
    assert parsed["aligned_frame_count"] == 124
    assert parsed["aligned_duration_seconds"] == pytest.approx(124 / 24)
    assert parsed["automatic_event_rewrite"] is False
    assert parsed["plan_hash"] == plan["plan_hash"]


@pytest.mark.parametrize(
    ("timing_mode", "events", "expected"),
    [
        (
            "frames",
            [
                {"prompt": "A", "start": 0, "end": 61},
                {"prompt": "B", "start": 62, "end": 123},
            ],
            [(0, 62), (62, 124)],
        ),
        (
            "seconds",
            [
                {"prompt": "A", "start": 0, "end": 62 / 24},
                {"prompt": "B", "start": 62 / 24, "end": 124 / 24},
            ],
            [(0, 62), (62, 124)],
        ),
        (
            "percent",
            [
                {"prompt": "A", "start": 0, "end": 50},
                {"prompt": "B", "start": 50, "end": 100},
            ],
            [(0, 62), (62, 124)],
        ),
    ],
)
def test_prompt_packet_relay_preserves_explicit_event_timing(
    timing_mode, events, expected
):
    plan, *_ = _packet_plan(
        _prompt_packet(duration_seconds=124 / 24),
        timing_mode=timing_mode,
        events_json=json.dumps(events),
    )
    assert [
        (event["start_frame"], event["end_frame_exclusive"])
        for event in plan["events"]
    ] == expected


def test_prompt_packet_relay_rejects_tampering_wrong_backend_and_hidden_timing():
    tampered = _prompt_packet()
    tampered["compiled_prompt"] += " changed"
    with pytest.raises(ValueError, match="hash mismatch"):
        _packet_plan(tampered)

    with pytest.raises(ValueError, match="only supports packets compiled for minimax_h3"):
        _packet_plan(_prompt_packet(backend="wan_2_2"))

    with pytest.raises(ValueError, match="includes start/end"):
        _packet_plan(
            events_json=json.dumps(
                [{"prompt": "A", "start": 0, "end": 100}]
            )
        )

    with pytest.raises(ValueError, match="requires start and end"):
        _packet_plan(
            timing_mode="percent",
            events_json=json.dumps([{"prompt": "A"}]),
        )


def test_prompt_packet_relay_rejects_duration_above_node_limit():
    with pytest.raises(ValueError, match="above the Prompt Relay limit"):
        _packet_plan(_prompt_packet(duration_seconds=200.0))


def test_prompt_relay_event_nodes_build_an_authenticated_append_only_chain():
    first, first_preview, first_report = build_prompt_relay_event(
        prompt="The heroine raises one hand.",
        start=0,
        end=40,
        enabled=True,
    )
    second, second_preview, second_report = build_prompt_relay_event(
        prompt="She turns and walks forward.",
        start=41,
        end=81,
        enabled=True,
        previous_events=first,
    )
    unchanged, *_ = build_prompt_relay_event(
        prompt="This disabled event must not be appended.",
        start=82,
        end=123,
        enabled=False,
        previous_events=second,
    )

    assert first["type"] == PROMPT_RELAY_EVENTS_TYPE
    assert [event["event_index"] for event in second["events"]] == [1, 2]
    assert [event["prompt"] for event in second["events"]] == [
        "The heroine raises one hand.",
        "She turns and walks forward.",
    ]
    assert unchanged == second
    assert json.loads(first_preview)["event_count"] == 1
    assert json.loads(second_preview)["event_count"] == 2
    assert json.loads(first_report)["status"] == "event_chain_ready"
    assert json.loads(second_report)["events_hash"] == second["events_hash"]


def test_standard_plan_keeps_legacy_inputs_exact_without_an_event_chain():
    kwargs = {
        "global_prompt": "One continuous shot.",
        "local_prompts": "A\nB",
        "length": 124,
        "timing_mode": "auto_equal",
        "time_ranges": "unused legacy widget text",
        "math_profile": "paper_v1",
        "epsilon": 0.1,
        "allow_gaps": False,
        "allow_overlaps": False,
    }
    assert build_prompt_relay_plan_with_optional_events(**kwargs) == (
        build_prompt_relay_plan(**kwargs)
    )


def test_standard_plan_prefers_the_connected_typed_event_chain():
    first, *_ = build_prompt_relay_event("A", 0, 61, True)
    second, *_ = build_prompt_relay_event("B", 62, 123, True, first)
    plan, _compiled, _length, _timeline, report = (
        build_prompt_relay_plan_with_optional_events(
            global_prompt="One continuous shot.",
            local_prompts="this fallback must be ignored",
            length=124,
            timing_mode="frames",
            time_ranges="this fallback must also be ignored",
            math_profile="paper_v1",
            epsilon=0.1,
            allow_gaps=False,
            allow_overlaps=False,
            prompt_relay_events=second,
        )
    )
    assert [event["local_prompt"] for event in plan["events"]] == ["A", "B"]
    assert [
        (event["start_frame"], event["end_frame_exclusive"])
        for event in plan["events"]
    ] == [(0, 62), (62, 124)]
    assert plan["source_events"] == {
        "source": "typed_event_chain",
        "events_hash": second["events_hash"],
        "event_count": 2,
    }
    assert configure_prompt_relay_query_route(plan, "video_only_paper")[0][
        "query_route"
    ] == "video_only_paper"
    parsed = json.loads(report)
    assert parsed["event_source"] == "typed_event_chain"
    assert parsed["plan_hash"] == plan["plan_hash"]


def test_all_disabled_event_chain_overrides_fallback_with_global_only_bypass():
    disabled, _preview, event_report = build_prompt_relay_event(
        "This event is disabled.",
        0,
        123,
        False,
    )
    plan, compiled, length, timeline, report = (
        build_prompt_relay_plan_with_optional_events(
            global_prompt="One continuous global scene.",
            local_prompts="this fallback must be ignored",
            length=124,
            timing_mode="frames",
            time_ranges="this fallback must also be ignored",
            math_profile="paper_v1",
            epsilon=0.1,
            allow_gaps=False,
            allow_overlaps=False,
            prompt_relay_events=disabled,
        )
    )
    assert json.loads(event_report)["event_count"] == 0
    assert compiled == "Global scene: One continuous global scene."
    assert length == 124
    assert plan["events"] == []
    assert plan["source_events"]["event_count"] == 0
    assert json.loads(timeline)["events"] == []
    assert json.loads(report)["status"] == "plan_bypass_no_events"


def test_plain_global_only_plan_requires_stale_ranges_to_be_cleared():
    plan, compiled, _length, _timeline, report = _plan(
        local_prompts="",
        time_ranges="",
    )
    assert plan["events"] == []
    assert compiled == "Global scene: 夜晚，稳定人物与电影光线"
    assert json.loads(report)["status"] == "plan_bypass_no_events"
    with pytest.raises(ValueError, match="cannot include time_ranges"):
        _plan(local_prompts="", timing_mode="frames", time_ranges="0-123")


def test_prompt_packet_relay_prefers_the_connected_typed_event_chain():
    first, *_ = build_prompt_relay_event("A", 0, 61, True)
    second, *_ = build_prompt_relay_event("B", 62, 123, True, first)
    plan, _compiled, _length, _timeline, report = _packet_plan(
        _prompt_packet(duration_seconds=124 / 24),
        events_json="this fallback is intentionally invalid JSON",
        timing_mode="frames",
        prompt_relay_events=second,
    )

    assert [
        (event["start_frame"], event["end_frame_exclusive"])
        for event in plan["events"]
    ] == [(0, 62), (62, 124)]
    assert plan["source_events"] == {
        "source": "typed_event_chain",
        "events_hash": second["events_hash"],
        "event_count": 2,
    }
    parsed = json.loads(report)
    assert parsed["event_source"] == "typed_event_chain"
    assert parsed["event_collection_hash"] == second["events_hash"]


@pytest.mark.parametrize("source", ["json", "typed_chain"])
def test_prompt_packet_relay_accepts_an_explicit_global_only_bypass(source):
    overrides = {"events_json": "[]"}
    if source == "typed_chain":
        disabled, *_ = build_prompt_relay_event("disabled", 0, 100, False)
        overrides = {
            "events_json": "this ignored fallback is invalid",
            "prompt_relay_events": disabled,
        }
    plan, compiled, _length, timeline, report = _packet_plan(**overrides)
    assert plan["events"] == []
    assert compiled == "Global scene: " + _prompt_packet()["compiled_prompt"]
    assert json.loads(timeline)["events"] == []
    parsed = json.loads(report)
    assert parsed["event_count"] == 0
    assert parsed["status"] == "packet_relay_plan_bypass_no_events"


def test_prompt_packet_relay_rejects_a_tampered_typed_event_chain():
    event_chain, *_ = build_prompt_relay_event("A", 0, 100, True)
    event_chain["events"][0]["prompt"] = "tampered"
    with pytest.raises(ValueError, match="event collection hash mismatch"):
        _packet_plan(prompt_relay_events=event_chain)


def test_prompt_relay_preview_validates_and_summarizes_without_sampling():
    plan, *_ = _plan()
    output_plan, ready, event_count, timeline_text, report_json = (
        preview_prompt_relay_plan(plan)
    )
    assert output_plan == plan
    assert ready is True
    assert event_count == 3
    assert "Prompt Relay Plan: READY" in timeline_text
    assert "frames 0-40" in timeline_text
    report = json.loads(report_json)
    assert report["covered_frame_count"] == 124
    assert report["gap_ranges_inclusive"] == []
    assert report["overlap_ranges_inclusive"] == []
    assert report["model_loaded"] is False
    assert report["sampling_executed"] is False


def test_prompt_relay_preview_reports_global_only_bypass_as_ready():
    plan, *_ = _plan(local_prompts="", time_ranges="")
    output_plan, ready, event_count, timeline_text, report_json = (
        preview_prompt_relay_plan(plan)
    )
    assert output_plan == plan
    assert ready is True
    assert event_count == 0
    assert "BYPASS (no active local events)" in timeline_text
    report = json.loads(report_json)
    assert report["status"] == "prompt_relay_plan_bypass"
    assert report["bypass_reason"] == "no_active_local_events"
    assert report["gap_ranges_inclusive"] == []
    assert report["model_loaded"] is False
    assert report["sampling_executed"] is False


def test_prompt_relay_preview_reports_allowed_gaps_and_rejects_semantic_tampering():
    gap_plan, *_ = _plan(
        local_prompts="A\nB",
        timing_mode="frames",
        time_ranges="0-40\n50-123",
        allow_gaps=True,
    )
    report = json.loads(preview_prompt_relay_plan(gap_plan)[4])
    assert report["gap_ranges_inclusive"] == [[41, 49]]

    forged = json.loads(json.dumps(gap_plan))
    forged.pop("plan_hash")
    forged["events"][0]["start_frame"] = -1
    forged["plan_hash"] = json_hash(forged)
    with pytest.raises(ValueError, match="invalid frame interval"):
        preview_prompt_relay_plan(forged)

    out_of_order = json.loads(json.dumps(gap_plan))
    out_of_order.pop("plan_hash")
    out_of_order["events"] = list(reversed(out_of_order["events"]))
    for index, event in enumerate(out_of_order["events"], 1):
        event["event_index"] = index
    out_of_order["plan_hash"] = json_hash(out_of_order)
    with pytest.raises(ValueError, match="chronological order"):
        preview_prompt_relay_plan(out_of_order)

    too_short = json.loads(json.dumps(gap_plan))
    too_short.pop("plan_hash")
    too_short["events"][0]["end_frame_exclusive"] = 4
    too_short["plan_hash"] = json_hash(too_short)
    with pytest.raises(ValueError, match="shorter than 5 frames"):
        preview_prompt_relay_plan(too_short)


def test_query_route_node_is_explicit_and_does_not_mutate_the_source_plan():
    plan, *_ = _plan()
    routed, report = configure_prompt_relay_query_route(plan, "joint_av_exp")
    assert "query_route" not in plan
    assert routed["query_route"] == "joint_av_exp"
    assert routed["query_route_schema"] == 1
    assert routed["plan_hash"] != plan["plan_hash"]
    assert json.loads(report)["paper_scope"] == (
        "experimental_h3_joint_audio_video_extension"
    )
    with pytest.raises(ValueError, match="Unknown Prompt Relay query route"):
        configure_prompt_relay_query_route(plan, "unknown")


def test_validated_comfy_h3_source_contracts_are_current():
    assert _source_sha256(Attention.forward) in ATTENTION_FORWARD_SHA256S
    assert _source_sha256(PackedLayout.__init__) in PACKED_LAYOUT_SHA256S
    assert _source_sha256(MiniMaxH3Tokenizer.tokenize_with_weights) in TOKENIZER_SHA256S


def test_lora_or_weight_patch_before_prompt_relay_fails_closed(monkeypatch):
    _allow_fixture_core_contract(monkeypatch)

    bypass_first = _native_h3_model_patcher()
    bypass_first.set_injections("bypass_lora", [object()])
    with pytest.raises(RuntimeError, match="apply LoRA downstream"):
        _assert_core_contract(bypass_first)

    weight_patch_first = _native_h3_model_patcher()
    weight_patch_first.patches["diffusion_model.fixture"] = [object()]
    with pytest.raises(RuntimeError, match="apply LoRA downstream"):
        _assert_core_contract(weight_patch_first)


def test_prompt_relay_rejects_unknown_live_extra_conds_patch(monkeypatch):
    _allow_fixture_core_contract(monkeypatch)
    source = _native_h3_model_patcher()

    def _foreign_extra_conds(_self, **_kwargs):
        return {}

    source.model.extra_conds = types.MethodType(_foreign_extra_conds, source.model)
    with pytest.raises(RuntimeError, match="instance-level extra_conds patch"):
        _assert_core_contract(source)


def test_prompt_relay_then_bypass_lora_clone_preserves_binding(monkeypatch):
    _allow_fixture_core_contract(monkeypatch)
    binding = {"binding_hash": "fixture-binding"}
    patched, core_hashes = patch_prompt_relay_model(
        _native_h3_model_patcher(),
        binding,
        query_chunk_rows=256,
    )

    downstream = patched.clone()
    downstream.set_injections("bypass_lora", [object()])

    assert patched.object_patches == {}
    assert downstream.object_patches == {}
    downstream_extra_conds = downstream.get_model_object("extra_conds")
    patched_extra_conds = patched.get_model_object("extra_conds")
    assert downstream_extra_conds.__func__ is patched_extra_conds.__func__
    assert downstream_extra_conds.__self__ is patched_extra_conds.__self__
    override = downstream.model_options["transformer_options"][
        "optimized_attention_override"
    ]
    assert override._t8_prompt_relay_binding_hash == "fixture-binding"
    wrappers = downstream.get_wrappers(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
        PROMPT_RELAY_WRAPPER_KEY,
    )
    assert len(wrappers) == 1
    assert downstream.get_attachment(PROMPT_RELAY_WRAPPER_KEY) == {
        "patch_version": prompt_relay_module.PROMPT_RELAY_PATCH_VERSION,
        "binding_hash": "fixture-binding",
        "binding": {"binding_hash": "fixture-binding"},
        "query_chunk_rows": 256,
        "core_hashes": core_hashes,
    }
    assert len(downstream.get_injections("bypass_lora")) == 1


def test_sequential_prompt_relay_models_do_not_patch_shared_extra_conds(monkeypatch):
    _allow_fixture_core_contract(monkeypatch)
    source = _native_h3_model_patcher()
    original_extra_conds = source.get_model_object("extra_conds")

    first, _ = patch_prompt_relay_model(
        source,
        {"binding_hash": "first-binding"},
        query_chunk_rows=64,
    )
    second, _ = patch_prompt_relay_model(
        source,
        {"binding_hash": "second-binding"},
        query_chunk_rows=64,
    )

    assert source.object_patches == {}
    assert first.object_patches == {}
    assert second.object_patches == {}
    for current in (source, first, second):
        current_extra_conds = current.get_model_object("extra_conds")
        assert current_extra_conds.__func__ is original_extra_conds.__func__
        assert current_extra_conds.__self__ is original_extra_conds.__self__


def test_legacy_profile_is_explicit_and_not_mislabeled_as_paper():
    plan, *_rest, report = _plan(math_profile="legacy_repo_compat")
    assert plan["math_profile"] == "legacy_repo_compat"
    assert json.loads(report)["paper_equation"] is False
    assert any(
        not math.isclose(event["endpoint_weight"], 0.1, rel_tol=1e-3)
        for event in plan["events"]
    )


def test_frame_ranges_are_inclusive_and_gaps_fail_closed():
    plan, *_ = _plan(
        local_prompts="A\nB",
        timing_mode="frames",
        time_ranges="0-61\n62-123",
    )
    assert [(event["start_frame"], event["end_frame_exclusive"]) for event in plan["events"]] == [
        (0, 62),
        (62, 124),
    ]
    with pytest.raises(ValueError, match="gap"):
        _plan(
            local_prompts="A\nB",
            timing_mode="frames",
            time_ranges="0-50\n60-123",
        )


@pytest.mark.parametrize(
    ("timing_mode", "time_ranges", "expected"),
    [
        ("seconds", f"0-{62 / 24}\n{62 / 24}-{124 / 24}", [(0, 62), (62, 124)]),
        ("percent", "0-50\n50-100", [(0, 62), (62, 124)]),
    ],
)
def test_standard_plan_maps_explicit_seconds_and_percent_boundaries(
    timing_mode, time_ranges, expected
):
    plan, *_ = _plan(
        local_prompts="A\nB",
        timing_mode=timing_mode,
        time_ranges=time_ranges,
    )
    assert [
        (event["start_frame"], event["end_frame_exclusive"])
        for event in plan["events"]
    ] == expected


def test_explicit_ranges_reject_fractional_frames_percent_overflow_and_bad_order():
    with pytest.raises(ValueError, match="integer frame indices"):
        _plan(
            local_prompts="A\nB",
            timing_mode="frames",
            time_ranges="0.5-61\n62-123",
        )
    with pytest.raises(ValueError, match="0..100"):
        _plan(
            local_prompts="A\nB",
            timing_mode="percent",
            time_ranges="0-50\n50-101",
        )
    with pytest.raises(ValueError, match="chronological order"):
        _plan(
            local_prompts="later\nearlier",
            timing_mode="frames",
            time_ranges="62-123\n0-61",
            allow_gaps=True,
            allow_overlaps=True,
        )


def test_six_auto_equal_events_cover_the_complete_timeline():
    plan, *_ = _plan(local_prompts="\n".join(f"Event {index}" for index in range(6)))
    ranges = [
        (event["start_frame"], event["end_frame_exclusive"])
        for event in plan["events"]
    ]
    assert len(ranges) == 6
    assert ranges[0][0] == 0
    assert ranges[-1][1] == 124
    assert all(end - start >= 5 for start, end in ranges)
    assert all(ranges[index][1] == ranges[index + 1][0] for index in range(5))


def test_chinese_token_binding_uses_authoritative_tail_and_byte_offsets():
    plan, prompt, *_ = _plan()
    clip = NativeLikeFakeClip()
    tokens = clip.tokenize(prompt)
    conditioning = clip.encode_from_tokens_scheduled(tokens)
    binding = build_prompt_relay_binding(clip, plan, prompt, conditioning, tokens)
    assert binding["text_len"] == len(prompt.encode("utf-8"))
    assert len(binding["events"]) == 3
    assert all(event["text_key_end"] > event["text_key_start"] for event in binding["events"])
    assert binding["events"][0]["text_key_end"] <= binding["events"][1]["text_key_start"]


def test_repeated_chinese_events_bind_to_distinct_authoritative_token_spans():
    plan, prompt, *_ = _plan(
        local_prompts="人物快速转身\n人物快速转身",
        timing_mode="frames",
        time_ranges="0-61\n62-123",
    )
    clip = NativeLikeFakeClip()
    tokens = clip.tokenize(prompt)
    conditioning = clip.encode_from_tokens_scheduled(tokens)
    binding = build_prompt_relay_binding(clip, plan, prompt, conditioning, tokens)
    first, second = binding["events"]
    assert first["text_key_end"] <= second["text_key_start"]
    assert (first["text_key_start"], first["text_key_end"]) != (
        second["text_key_start"],
        second["text_key_end"],
    )


def test_media_presentation_prefix_remains_outside_local_text_spans():
    plan, prompt, *_ = _plan(
        local_prompts=(
            "<Picture 1>中的女人抬手\n"
            "她转身跑向远处\n"
            "镜头快速拉远"
        )
    )
    clip = NativeLikeFakeClip()
    tokens = clip.tokenize(prompt, minimax_ref_items=[{"type": "image"}])
    conditioning = clip.encode_from_tokens_scheduled(tokens)
    binding = build_prompt_relay_binding(clip, plan, prompt, conditioning, tokens)
    assert binding["text_len"] > binding["prompt_token_count"]
    prefix_rows = binding["text_len"] - binding["prompt_token_count"]
    assert prefix_rows == 2
    assert all(event["text_key_start"] >= prefix_rows for event in binding["events"])


def test_multiple_media_tags_remain_text_while_all_presentation_rows_stay_global():
    plan, prompt, *_ = _plan(
        local_prompts=(
            "<Picture 1>中的人物看向<Video 1>\n"
            "<Audio 1>响起时人物转身"
        ),
        timing_mode="frames",
        time_ranges="0-61\n62-123",
    )
    clip = NativeLikeFakeClip()
    tokens = clip.tokenize(
        prompt,
        minimax_ref_items=[
            {"type": "image"},
            {"type": "video"},
            {"type": "audio"},
        ],
    )
    conditioning = clip.encode_from_tokens_scheduled(tokens)
    binding = build_prompt_relay_binding(clip, plan, prompt, conditioning, tokens)
    prefix_rows = binding["text_len"] - binding["prompt_token_count"]
    assert prefix_rows == 6
    assert all(event["text_key_start"] >= prefix_rows for event in binding["events"])
    assert binding["events"][0]["text_key_end"] <= binding["events"][1]["text_key_start"]


def test_layout_contract_rejects_keyframe_or_reference_drift():
    plan, prompt, *_ = _plan()
    clip = NativeLikeFakeClip()
    tokens = clip.tokenize(prompt)
    conditioning = clip.encode_from_tokens_scheduled(tokens)
    binding = build_prompt_relay_binding(clip, plan, prompt, conditioning, tokens)
    layout = build_packed_layout(binding["text_len"], 37, 26, 46, 207)
    binding = _bind_layout_contract(
        binding,
        layout,
        resolved_task="t2va",
        keyframes=[],
        refs=[],
    )
    route = _runtime_route(layout, binding, torch.device("cpu"))
    assert route["audio_end"] == route["video_start"]
    assert route["query_route"] == "video_only_paper"
    assert [segment["kind"] for segment in route["query_segments"]] == ["video"]

    drifted = build_packed_layout(
        binding["text_len"],
        37,
        26,
        46,
        207,
        keyframes=[
            {
                "resolved_frame_index": 0,
                "latent": torch.zeros((1, 24, 1, 26, 46)),
            }
        ],
    )
    with pytest.raises(RuntimeError, match="differs from the layout bound"):
        _runtime_route(drifted, binding, torch.device("cpu"))


def test_joint_av_runtime_route_uses_native_audio_and_video_time_grids():
    plan, prompt, *_ = _plan()
    plan, _report = configure_prompt_relay_query_route(plan, "joint_av_exp")
    clip = NativeLikeFakeClip()
    tokens = clip.tokenize(prompt)
    conditioning = clip.encode_from_tokens_scheduled(tokens)
    binding = build_prompt_relay_binding(clip, plan, prompt, conditioning, tokens)
    layout = build_packed_layout(binding["text_len"], 37, 26, 46, 207)
    binding = _bind_layout_contract(
        binding,
        layout,
        resolved_task="t2va",
        keyframes=[],
        refs=[],
    )
    route = _runtime_route(layout, binding, torch.device("cpu"))
    audio_segment, video_segment = route["query_segments"]
    assert route["query_route"] == "joint_av_exp"
    assert audio_segment["kind"] == "audio"
    assert video_segment["kind"] == "video"
    assert audio_segment["start"] == route["audio_start"]
    assert audio_segment["end"] == route["audio_end"]
    assert video_segment["start"] == route["video_start"]
    assert video_segment["end"] == route["video_end"]
    audio_times = audio_segment["query_times"]
    assert audio_times.shape == (414,)
    assert audio_times[0].item() == 0.0
    assert audio_times[206].item() == 206.0
    assert audio_times[207].item() == 0.0
    assert audio_times[-1].item() == 206.0
    assert video_segment["query_times"][0].item() == 0.0


def test_report_only_reuses_stable_conditioning_and_does_not_patch_model():
    plan, *_ = _plan()
    model = object()
    result = build_prompt_relay_conditioning(
        model=model,
        clip=NativeLikeFakeClip(),
        video_vae=FakeVideoVAE(),
        audio_vae=FakeAudioVAE(),
        prompt_relay_plan=plan,
        width=736,
        height=416,
        task_type="T2VA",
        audio_mode="native",
        audio_denoise_strength=0.35,
        add_source_as_reference=True,
        prompt_primary_audio_ordinal=1,
        strict_prompt_tags=True,
        ref_image_size="match",
        reference_video_policy="official_2_to_15s",
        execution_mode="report_only",
        query_chunk_rows=256,
    )
    returned_model, conditioning, latent, _audio, conditioned_prompt, _media, report = result
    assert returned_model is model
    assert PROMPT_RELAY_BINDING_KEY not in conditioning[0][1]
    assert latent["samples"].is_nested
    assert conditioned_prompt == plan["compiled_prompt"]
    parsed = json.loads(report)
    assert parsed["status"] == "report_only"
    assert parsed["dense_s_by_s_mask_created"] is False
    assert parsed["attention_patch_installed"] is False
    assert parsed["max_explicit_bias_bytes_bf16"] > 0


def test_single_event_apply_exp_is_an_exact_unpatched_model_conditioning_route(
    monkeypatch,
):
    def forbidden_patch(*_args, **_kwargs):
        raise AssertionError("single-event Prompt Relay must not patch MODEL")

    monkeypatch.setattr(
        prompt_relay_module,
        "patch_prompt_relay_model",
        forbidden_patch,
    )
    plan, *_ = _plan(local_prompts="One event covers the full video.")
    model = object()
    result = build_prompt_relay_conditioning(
        model=model,
        clip=NativeLikeFakeClip(),
        video_vae=FakeVideoVAE(),
        audio_vae=FakeAudioVAE(),
        prompt_relay_plan=plan,
        width=128,
        height=128,
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
    returned_model, conditioning, _latent, _audio, prompt, _media, report = result
    metadata = conditioning[0][1]
    parsed = json.loads(report)
    assert returned_model is model
    assert prompt == plan["compiled_prompt"]
    assert PROMPT_RELAY_BINDING_KEY not in metadata
    assert PROMPT_RELAY_PAYLOAD_KEY not in metadata.get("model_conds", {})
    assert parsed["status"] == "passthrough_single_event"
    assert parsed["attention_patch_installed"] is False
    assert parsed["core_hashes"] == {}
    assert any("single local event" in warning for warning in parsed["warnings"])


def test_no_event_apply_exp_matches_stable_global_prompt_conditioning(monkeypatch):
    def forbidden_patch(*_args, **_kwargs):
        raise AssertionError("empty Prompt Relay must not patch MODEL")

    monkeypatch.setattr(
        prompt_relay_module,
        "patch_prompt_relay_model",
        forbidden_patch,
    )
    plan, *_ = _plan(local_prompts="", time_ranges="")
    model = object()
    clip = NativeLikeFakeClip()
    video_vae = FakeVideoVAE()
    audio_vae = FakeAudioVAE()
    result = build_prompt_relay_conditioning(
        model=model,
        clip=clip,
        video_vae=video_vae,
        audio_vae=audio_vae,
        prompt_relay_plan=plan,
        width=128,
        height=128,
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
    stable = build_conditioning(
        clip,
        video_vae,
        audio_vae,
        plan["compiled_prompt"],
        128,
        128,
        124,
        "T2VA",
        "native",
        0.35,
        False,
        0,
        True,
        "match",
        "official_2_to_15s",
    )
    returned_model, conditioning, latent, output_audio, prompt, media, report = result
    assert returned_model is model
    assert prompt == stable[3] == plan["compiled_prompt"]
    assert output_audio == stable[2]
    assert media == stable[4]
    assert torch.equal(conditioning[0][0], stable[0][0][0])
    relay_metadata, stable_metadata = conditioning[0][1], stable[0][0][1]
    assert relay_metadata.keys() == stable_metadata.keys()
    for key in relay_metadata:
        if torch.is_tensor(relay_metadata[key]):
            assert torch.equal(relay_metadata[key], stable_metadata[key])
        else:
            assert relay_metadata[key] == stable_metadata[key]
    for relay_part, stable_part in zip(
        latent["samples"].unbind(),
        stable[1]["samples"].unbind(),
        strict=True,
    ):
        assert torch.equal(relay_part, stable_part)
    parsed = json.loads(report)
    assert parsed["status"] == "passthrough_no_events"
    assert parsed["attention_patch_installed"] is False
    assert parsed["core_hashes"] == {}


@pytest.mark.parametrize(
    ("task_type", "media", "expected_task", "expected_keyframes", "expected_refs"),
    [
        ("T2VA", {}, "t2va", 0, 0),
        (
            "I2VA",
            {"first_frame": torch.zeros((1, 128, 128, 3))},
            "i2va",
            1,
            0,
        ),
        (
            "FL2VA",
            {
                "first_frame": torch.zeros((1, 128, 128, 3)),
                "last_frame": torch.ones((1, 128, 128, 3)),
            },
            "fl2va",
            2,
            0,
        ),
        (
            "L2VA",
            {"last_frame": torch.ones((1, 128, 128, 3))},
            "l2va",
            1,
            0,
        ),
        (
            "Ref2VA",
            {"ref_images": {"ref_image_1": torch.zeros((1, 128, 128, 3))}},
            "ref2va",
            0,
            1,
        ),
        (
            "Hybrid",
            {
                "first_frame": torch.zeros((1, 128, 128, 3)),
                "ref_images": {"ref_image_1": torch.ones((1, 128, 128, 3))},
            },
            "hybrid",
            1,
            1,
        ),
    ],
)
def test_apply_exp_supports_all_native_visual_tasks_without_touching_prefix_rows(
    monkeypatch,
    task_type,
    media,
    expected_task,
    expected_keyframes,
    expected_refs,
):
    captured = {}

    def fake_patch(model, binding, query_chunk_rows):
        captured["binding"] = binding
        captured["query_chunk_rows"] = query_chunk_rows
        return "patched-model", {"test": "mock-core"}

    monkeypatch.setattr(prompt_relay_module, "patch_prompt_relay_model", fake_patch)
    plan, *_ = _plan()
    result = build_prompt_relay_conditioning(
        model=object(),
        clip=NativeLikeFakeClip(),
        video_vae=FakeVideoVAE(),
        audio_vae=FakeAudioVAE(),
        prompt_relay_plan=plan,
        width=128,
        height=128,
        task_type=task_type,
        audio_mode="native",
        audio_denoise_strength=0.35,
        add_source_as_reference=False,
        prompt_primary_audio_ordinal=0,
        strict_prompt_tags=True,
        ref_image_size="match",
        reference_video_policy="official_2_to_15s",
        execution_mode="apply_exp",
        query_chunk_rows=64,
        **media,
    )
    patched_model, conditioning, _latent, _audio, _prompt, _map, report = result
    parsed = json.loads(report)
    metadata = conditioning[0][1]
    binding = metadata[PROMPT_RELAY_BINDING_KEY]
    assert metadata["model_conds"][PROMPT_RELAY_PAYLOAD_KEY].cond == binding["binding_hash"]
    assert patched_model == "patched-model"
    assert captured["binding"]["binding_hash"] == binding["binding_hash"]
    assert binding["task"] == expected_task
    assert binding["keyframe_count"] == expected_keyframes
    assert binding["reference_block_count"] == expected_refs
    assert parsed["status"] == "applied_exp"
    assert parsed["task"] == expected_task
    assert parsed["keyframe_count"] == expected_keyframes
    assert parsed["reference_block_count"] == expected_refs
    assert parsed["target_audio_rows"] > 0
    assert parsed["target_video_rows"] > 0
    assert parsed["dense_s_by_s_mask_created"] is False
    assert parsed["attention_patch_installed"] is True
    assert parsed["packed_segments"][-2][2:] == ["audio"]
    assert parsed["packed_segments"][-1][2:] == ["video"]


@pytest.mark.parametrize(
    ("audio_mode", "strength", "add_reference", "expected_policy", "expected_mask"),
    [
        (
            "lock_source",
            0.35,
            False,
            "source_latent_locked_by_zero_noise_mask",
            0.0,
        ),
        (
            "remix_source",
            0.35,
            False,
            "source_latent_jointly_remixed_at_requested_strength",
            0.35,
        ),
        (
            "reference_only",
            0.35,
            True,
            "source_audio_reference_only_target_audio_regenerated",
            None,
        ),
    ],
)
def test_apply_exp_preserves_stable_audio_mode_contracts(
    monkeypatch,
    audio_mode,
    strength,
    add_reference,
    expected_policy,
    expected_mask,
):
    monkeypatch.setattr(
        prompt_relay_module,
        "patch_prompt_relay_model",
        lambda _model, _binding, _query_chunk_rows: (
            "patched-model",
            {"test": "mock-core"},
        ),
    )
    plan, *_ = _plan()
    result = build_prompt_relay_conditioning(
        model=object(),
        clip=NativeLikeFakeClip(),
        video_vae=FakeVideoVAE(),
        audio_vae=FakeAudioVAE(),
        prompt_relay_plan=plan,
        width=128,
        height=128,
        task_type="Ref2VA" if audio_mode == "reference_only" else "T2VA",
        audio_mode=audio_mode,
        audio_denoise_strength=strength,
        add_source_as_reference=add_reference,
        prompt_primary_audio_ordinal=1 if add_reference else 0,
        strict_prompt_tags=True,
        ref_image_size="match",
        reference_video_policy="official_2_to_15s",
        execution_mode="apply_exp",
        query_chunk_rows=64,
        drive_audio={
            "waveform": torch.zeros((1, 2, 32000)),
            "sample_rate": 32000,
        },
    )
    patched_model, _conditioning, latent, _audio, _prompt, _map, report = result
    parsed = json.loads(report)
    assert patched_model == "patched-model"
    assert parsed["audio_mode"] == audio_mode
    assert parsed["audio_policy"] == expected_policy
    assert parsed["audio_direct_prompt_relay_bias"] is False
    if expected_mask is None:
        assert parsed["audio_noise_mask_range"] is None
        assert "noise_mask" not in latent
    else:
        _video_mask, audio_mask = latent["noise_mask"].unbind()
        assert torch.allclose(audio_mask, torch.full_like(audio_mask, expected_mask))
        assert parsed["audio_noise_mask_range"] == pytest.approx(
            [expected_mask, expected_mask]
        )


def test_reference_only_rejects_disconnected_reference_semantics(monkeypatch):
    monkeypatch.setattr(
        prompt_relay_module,
        "patch_prompt_relay_model",
        lambda _model, _binding, _query_chunk_rows: (
            "patched-model",
            {"test": "mock-core"},
        ),
    )
    plan, *_ = _plan()
    with pytest.raises(ValueError, match="add_source_as_reference=true"):
        build_prompt_relay_conditioning(
            model=object(),
            clip=NativeLikeFakeClip(),
            video_vae=FakeVideoVAE(),
            audio_vae=FakeAudioVAE(),
            prompt_relay_plan=plan,
            width=128,
            height=128,
            task_type="T2VA",
            audio_mode="reference_only",
            audio_denoise_strength=0.35,
            add_source_as_reference=False,
            prompt_primary_audio_ordinal=0,
            strict_prompt_tags=True,
            ref_image_size="match",
            reference_video_policy="official_2_to_15s",
            execution_mode="apply_exp",
            query_chunk_rows=64,
            drive_audio={
                "waveform": torch.zeros((1, 2, 32000)),
                "sample_rate": 32000,
            },
        )


def test_joint_av_route_rejects_locked_audio_but_reports_native_audio(monkeypatch):
    monkeypatch.setattr(
        prompt_relay_module,
        "patch_prompt_relay_model",
        lambda _model, _binding, _query_chunk_rows: (
            "patched-model",
            {"test": "mock-core"},
        ),
    )
    plan, *_ = _plan()
    plan, _ = configure_prompt_relay_query_route(plan, "joint_av_exp")
    common = {
        "model": object(),
        "clip": NativeLikeFakeClip(),
        "video_vae": FakeVideoVAE(),
        "audio_vae": FakeAudioVAE(),
        "prompt_relay_plan": plan,
        "width": 128,
        "height": 128,
        "task_type": "T2VA",
        "audio_denoise_strength": 0.35,
        "add_source_as_reference": False,
        "prompt_primary_audio_ordinal": 0,
        "strict_prompt_tags": True,
        "ref_image_size": "match",
        "reference_video_policy": "official_2_to_15s",
        "execution_mode": "apply_exp",
        "query_chunk_rows": 64,
    }
    with pytest.raises(ValueError, match="cannot be combined with lock_source"):
        build_prompt_relay_conditioning(
            **common,
            audio_mode="lock_source",
            drive_audio={
                "waveform": torch.zeros((1, 2, 32000)),
                "sample_rate": 32000,
            },
        )
    result = build_prompt_relay_conditioning(**common, audio_mode="native")
    parsed = json.loads(result[-1])
    assert parsed["query_route"] == "joint_av_exp"
    assert parsed["audio_direct_prompt_relay_bias"] is True
    assert parsed["routed_query_rows"] == (
        parsed["target_audio_rows"] + parsed["target_video_rows"]
    )


def test_streamed_attention_matches_dense_reference(monkeypatch):
    from comfy.ldm.modules import attention as attention_module

    def pytorch_backend(q, k, v, heads, **kwargs):
        kwargs["_inside_attn_wrapper"] = True
        return attention_module.attention_pytorch(q, k, v, heads, **kwargs)

    monkeypatch.setattr(attention_module, "optimized_attention", pytorch_backend)
    generator = torch.Generator().manual_seed(8)
    q = torch.randn((1, 2, 9, 4), generator=generator)
    k = torch.randn((1, 2, 9, 4), generator=generator)
    v = torch.randn((1, 2, 9, 4), generator=generator)
    event = {
        "text_key_start": 1,
        "text_key_end": 3,
        "midpoint": 2.0,
        "window": 0.5,
        "sigma": 0.75,
    }
    route = {
        "seq_len": 9,
        "query_route": "video_only_paper",
        "video_start": 5,
        "video_end": 9,
        "query_segments": (
            {
                "kind": "video",
                "start": 5,
                "end": 9,
                "query_times": torch.tensor([0.0, 1.0, 2.0, 3.0]),
            },
        ),
        "events": (event,),
    }
    transformer_options = {PROMPT_RELAY_RUNTIME_KEY: route}
    actual = route_prompt_relay_attention(
        q,
        k,
        v,
        2,
        skip_reshape=True,
        transformer_options=transformer_options,
        query_chunk_rows=2,
    )
    dense_bias = torch.zeros((9, 9))
    dense_bias[5:] = make_prompt_relay_bias(
        route["query_segments"][0]["query_times"],
        9,
        route["events"],
        dtype=q.dtype,
    )
    expected = attention_module.attention_pytorch(
        q,
        k,
        v,
        2,
        mask=dense_bias,
        skip_reshape=True,
        _inside_attn_wrapper=True,
    )
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_joint_av_streamed_attention_matches_dense_reference(monkeypatch):
    from comfy.ldm.modules import attention as attention_module

    def pytorch_backend(q, k, v, heads, **kwargs):
        kwargs["_inside_attn_wrapper"] = True
        return attention_module.attention_pytorch(q, k, v, heads, **kwargs)

    monkeypatch.setattr(attention_module, "optimized_attention", pytorch_backend)
    generator = torch.Generator().manual_seed(9)
    q = torch.randn((1, 2, 9, 4), generator=generator)
    k = torch.randn((1, 2, 9, 4), generator=generator)
    v = torch.randn((1, 2, 9, 4), generator=generator)
    event = {
        "text_key_start": 1,
        "text_key_end": 3,
        "midpoint": 2.0,
        "window": 0.5,
        "sigma": 0.75,
    }
    audio_times = torch.tensor([0.0, 1.0])
    video_times = torch.tensor([0.0, 1.0, 2.0, 3.0])
    route = {
        "seq_len": 9,
        "query_route": "joint_av_exp",
        "query_segments": (
            {"kind": "audio", "start": 3, "end": 5, "query_times": audio_times},
            {"kind": "video", "start": 5, "end": 9, "query_times": video_times},
        ),
        "events": (event,),
    }
    actual = route_prompt_relay_attention(
        q,
        k,
        v,
        2,
        skip_reshape=True,
        transformer_options={PROMPT_RELAY_RUNTIME_KEY: route},
        query_chunk_rows=2,
    )
    dense_bias = torch.zeros((9, 9))
    dense_bias[3:5] = make_prompt_relay_bias(
        audio_times,
        9,
        route["events"],
        dtype=q.dtype,
    )
    dense_bias[5:9] = make_prompt_relay_bias(
        video_times,
        9,
        route["events"],
        dtype=q.dtype,
    )
    expected = attention_module.attention_pytorch(
        q,
        k,
        v,
        2,
        mask=dense_bias,
        skip_reshape=True,
        _inside_attn_wrapper=True,
    )
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_node_schemas_are_append_only_advanced_and_safe_by_default():
    schemas = [node.define_schema() for node in PROMPT_RELAY_ADVANCED_NODE_CLASSES]
    assert [schema.node_id for schema in schemas] == [
        "MiniMaxH3PromptRelayPlanT8Advanced",
        "MiniMaxH3PromptRelayConditioningT8Advanced",
        "MiniMaxH3PromptRelayQueryRouteT8Advanced",
    ]
    assert all(schema.is_experimental is True for schema in schemas)
    assert all(schema.category == "T8/MiniMax H3/Conditioning/Experimental" for schema in schemas)
    plan_inputs = {item.id: item for item in schemas[0].inputs}
    conditioning_inputs = {item.id: item for item in schemas[1].inputs}
    route_inputs = {item.id: item for item in schemas[2].inputs}
    assert plan_inputs["math_profile"].default == "paper_v1"
    assert plan_inputs["epsilon"].default == 0.1
    assert [item.id for item in schemas[0].inputs] == [
        "global_prompt",
        "local_prompts",
        "length",
        "timing_mode",
        "time_ranges",
        "math_profile",
        "epsilon",
        "allow_gaps",
        "allow_overlaps",
        "prompt_relay_events",
    ]
    assert conditioning_inputs["execution_mode"].default == "report_only"
    assert conditioning_inputs["query_chunk_rows"].default == 256
    assert route_inputs["query_route"].default == "video_only_paper"

    packet_schemas = [
        node.define_schema() for node in PROMPT_RELAY_PACKET_ADVANCED_NODE_CLASSES
    ]
    assert [schema.node_id for schema in packet_schemas] == [
        "MiniMaxH3PromptPacketRelayPlanT8Advanced",
        "MiniMaxH3PromptRelayEventT8Advanced",
    ]
    packet_schema = packet_schemas[0]
    assert packet_schema.is_experimental is True
    assert packet_schema.category == "T8/MiniMax H3/Conditioning/Experimental"
    packet_inputs = {item.id: item for item in packet_schema.inputs}
    assert packet_inputs["timing_mode"].default == "auto_equal"
    assert packet_inputs["math_profile"].default == "paper_v1"
    assert packet_inputs["epsilon"].default == 0.1
    event_inputs = {item.id: item for item in packet_schemas[1].inputs}
    assert event_inputs["start"].default == 0.0
    assert event_inputs["end"].default == 100.0
    assert event_inputs["enabled"].default is True

    preview_schemas = [
        node.define_schema() for node in PROMPT_RELAY_PREVIEW_ADVANCED_NODE_CLASSES
    ]
    assert [schema.node_id for schema in preview_schemas] == [
        "MiniMaxH3PromptRelayPreviewT8Advanced"
    ]
    assert preview_schemas[0].is_experimental is True
    assert preview_schemas[0].is_output_node is True
    assert preview_schemas[0].category == (
        "T8/MiniMax H3/Conditioning/Experimental"
    )


def test_frontend_workflows_are_importable_and_documented():
    root = Path(__file__).resolve().parents[1]
    workflow_root = root / "examples" / "workflows" / "14-prompt-relay"
    expected_workflows = {
        "2026-08-20_H3_Prompt_Relay_T2VA_Stock20_Advanced_EXP.json": {
            "task": "T2VA",
            "steps": 20,
            "turbo": False,
            "joint_av": False,
            "chunk": 256,
        },
        "2026-08-20_H3_Prompt_Relay_I2VA_Stock20_Advanced_EXP.json": {
            "task": "I2VA",
            "steps": 20,
            "turbo": False,
            "joint_av": False,
            "chunk": 256,
        },
        "2026-08-20_H3_Prompt_Relay_Ref2VA_Stock20_Advanced_EXP.json": {
            "task": "Ref2VA",
            "steps": 20,
            "turbo": False,
            "joint_av": False,
            "chunk": 256,
        },
        "2026-08-20_H3_Prompt_Relay_RefVideoAudio_Stock20_Advanced_EXP.json": {
            "task": "Ref2VA",
            "steps": 20,
            "turbo": False,
            "joint_av": False,
            "chunk": 256,
            "load_image": False,
        },
        "2026-08-20_H3_Prompt_Relay_RefAudio_Stock20_Advanced_EXP.json": {
            "task": "Ref2VA",
            "steps": 20,
            "turbo": False,
            "joint_av": False,
            "chunk": 256,
            "load_image": False,
        },
        "2026-08-20_H3_Prompt_Relay_T2VA_Turbo8_Advanced_EXP.json": {
            "task": "T2VA",
            "steps": 8,
            "turbo": True,
            "joint_av": False,
            "chunk": 256,
        },
        "2026-08-20_H3_Prompt_Relay_Joint_AV_Turbo8_Advanced_EXP.json": {
            "task": "T2VA",
            "steps": 8,
            "turbo": True,
            "joint_av": True,
            "chunk": 256,
        },
        "2026-08-20_H3_Prompt_Relay_FL2VA_Lock_Turbo8_Advanced_EXP.json": {
            "task": "FL2VA",
            "steps": 8,
            "turbo": True,
            "joint_av": False,
            "chunk": 256,
        },
        "2026-08-20_H3_Prompt_Relay_L2VA_Turbo8_Advanced_EXP.json": {
            "task": "L2VA",
            "steps": 8,
            "turbo": True,
            "joint_av": False,
            "chunk": 256,
        },
        "2026-08-20_H3_Prompt_Relay_Hybrid_Stock20_Advanced_EXP.json": {
            "task": "Hybrid",
            "steps": 20,
            "turbo": False,
            "joint_av": False,
            "chunk": 1024,
        },
    }
    for filename, expected in expected_workflows.items():
        workflow = json.loads((workflow_root / filename).read_text(encoding="utf-8-sig"))
        nodes = {node["id"]: node for node in workflow["nodes"]}
        types = {node["type"] for node in nodes.values()}
        links = {link[0]: link for link in workflow["links"]}
        assert workflow["version"] == 0.4
        assert workflow["last_node_id"] == max(nodes)
        assert workflow["last_link_id"] == max(link[0] for link in workflow["links"])
        assert "MiniMaxH3PromptRelayPlanT8Advanced" in types
        assert "MiniMaxH3PromptRelayConditioningT8Advanced" in types
        assert sum(
            node["type"] == "MiniMaxH3PromptRelayPreviewT8Advanced"
            for node in nodes.values()
        ) == 1
        assert sum(node["type"] == "MarkdownNote" for node in nodes.values()) == 3

        def source_for_input(node, name):
            item = next(value for value in node["inputs"] if value["name"] == name)
            if item["link"] is None:
                return None
            link = links[item["link"]]
            return nodes[link[1]], link[2]

        plan = next(
            node
            for node in nodes.values()
            if node["type"] == "MiniMaxH3PromptRelayPlanT8Advanced"
        )
        preview = next(
            node
            for node in nodes.values()
            if node["type"] == "MiniMaxH3PromptRelayPreviewT8Advanced"
        )
        assert source_for_input(preview, "prompt_relay_plan") == (plan, 0)
        conditioning = next(
            node
            for node in nodes.values()
            if node["type"] == "MiniMaxH3PromptRelayConditioningT8Advanced"
        )
        assert conditioning["widgets_values"][2] == expected["task"]
        assert conditioning["widgets_values"][-2:] == [
            "apply_exp",
            expected["chunk"],
        ]
        route_nodes = [
            node
            for node in nodes.values()
            if node["type"] == "MiniMaxH3PromptRelayQueryRouteT8Advanced"
        ]
        if expected["joint_av"]:
            assert len(route_nodes) == 1
            route_node = route_nodes[0]
            assert route_node["widgets_values"] == ["joint_av_exp"]
            preview_to_route = links[route_node["inputs"][0]["link"]]
            route_to_conditioning = links[conditioning["inputs"][4]["link"]]
            assert preview_to_route[1:5] == [
                preview["id"],
                0,
                route_node["id"],
                0,
            ]
            assert route_to_conditioning[1:5] == [
                route_node["id"],
                0,
                conditioning["id"],
                4,
            ]
        else:
            assert route_nodes == []
            assert source_for_input(conditioning, "prompt_relay_plan") == (preview, 0)
        sampler = next(
            node for node in nodes.values() if node["type"] == "MiniMaxH3DualClockSamplerT8"
        )
        assert sampler["widgets_values"][0] == expected["steps"]
        if expected.get("load_image", expected["task"] != "T2VA"):
            assert "LoadImage" in types
        lora_nodes = [
            node for node in nodes.values() if node["type"] == "LoraLoaderBypassModelOnly"
        ]
        if expected["turbo"]:
            assert len(lora_nodes) == 1
            lora = lora_nodes[0]
            assert lora["widgets_values"] == [
                "minimax_h3_fl2v_turbo_4step_v0.1_comfyui_alpha8-T8-convert.safetensors",
                1.0,
            ]
            assert conditioning["widgets_values"][:2] == [736, 416]
            loader = next(node for node in nodes.values() if node["type"] == "UNETLoader")
            assert loader["widgets_values"][0] == "minimax_h3_fl2va_int8_convrot.safetensors"

            conditioning_to_lora = links[lora["inputs"][0]["link"]]
            lora_to_sampler = links[sampler["inputs"][0]["link"]]
            assert conditioning_to_lora[1:5] == [conditioning["id"], 0, lora["id"], 0]
            assert lora_to_sampler[1:5] == [lora["id"], 0, sampler["id"], 0]
        else:
            assert lora_nodes == []

        if filename == "2026-08-20_H3_Prompt_Relay_T2VA_Stock20_Advanced_EXP.json":
            event_nodes = sorted(
                (
                    node
                    for node in nodes.values()
                    if node["type"] == "MiniMaxH3PromptRelayEventT8Advanced"
                ),
                key=lambda node: node["id"],
            )
            assert len(event_nodes) == 3
            assert plan["widgets_values"][3] == "frames"
            assert source_for_input(event_nodes[0], "previous_events") is None
            assert source_for_input(event_nodes[1], "previous_events") == (
                event_nodes[0],
                0,
            )
            assert source_for_input(event_nodes[2], "previous_events") == (
                event_nodes[1],
                0,
            )
            assert source_for_input(plan, "prompt_relay_events") == (
                event_nodes[2],
                0,
            )
            assert [node["widgets_values"][1:4] for node in event_nodes] == [
                [0, 40, True],
                [41, 81, True],
                [82, 123, True],
            ]

        decode = next(
            node for node in nodes.values() if node["type"] == "MiniMaxH3AVDecodeT8"
        )
        saver = next(node for node in nodes.values() if node["type"] == "VHS_VideoCombine")
        if expected["task"] == "FL2VA":
            assert conditioning["widgets_values"][3] == "lock_source"
            first = source_for_input(conditioning, "first_frame")
            last = source_for_input(conditioning, "last_frame")
            drive = source_for_input(conditioning, "drive_audio")
            assert first is not None and first[0]["type"] == "LoadImage"
            assert last is not None and last[0]["type"] == "LoadImage"
            assert first[0]["id"] != last[0]["id"]
            assert drive is not None and drive[0]["type"] == "LoadAudio"
            assert source_for_input(saver, "audio") == (conditioning, 3)
        elif expected["task"] == "L2VA":
            assert source_for_input(conditioning, "first_frame") is None
            last = source_for_input(conditioning, "last_frame")
            assert last is not None and last[0]["type"] == "LoadImage"
            assert "LoadAudio" not in types
            assert source_for_input(saver, "audio") == (decode, 1)
        elif expected["task"] == "Hybrid":
            first = source_for_input(conditioning, "first_frame")
            reference = source_for_input(conditioning, "ref_images.ref_image_0")
            assert first is not None and first[0]["type"] == "LoadImage"
            assert reference is not None and reference[0]["type"] == "LoadImage"
            assert first[0]["id"] != reference[0]["id"]
            loader = next(node for node in nodes.values() if node["type"] == "UNETLoader")
            assert loader["widgets_values"][0] == (
                "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
            )
            assert source_for_input(saver, "audio") == (decode, 1)
        if filename == (
            "2026-08-20_H3_Prompt_Relay_RefVideoAudio_Stock20_Advanced_EXP.json"
        ):
            assert {"LoadVideo", "Video Slice", "GetVideoComponents"} <= types
            reference_video = source_for_input(conditioning, "ref_videos.ref_video_0")
            reference_audio = source_for_input(
                conditioning, "ref_video_audios.ref_video_audio_0"
            )
            assert reference_video is not None
            assert reference_audio is not None
            assert reference_video[0] == reference_audio[0]
            assert reference_video[0]["type"] == "GetVideoComponents"
            assert reference_video[1] == 0
            assert reference_audio[1] == 1
            assert "LoadImage" not in types
        elif filename == (
            "2026-08-20_H3_Prompt_Relay_RefAudio_Stock20_Advanced_EXP.json"
        ):
            reference_audio = source_for_input(conditioning, "ref_audios.ref_audio_0")
            assert reference_audio is not None
            assert reference_audio[0]["type"] == "LoadAudio"
            assert reference_audio[1] == 0
            assert "LoadImage" not in types
        for link_id, source, output_slot, target, input_slot, link_type in workflow["links"]:
            assert nodes[target]["inputs"][input_slot]["link"] == link_id
            assert link_id in (nodes[source]["outputs"][output_slot].get("links") or [])
            assert nodes[source]["outputs"][output_slot]["type"] == link_type
            assert nodes[target]["inputs"][input_slot]["type"] == link_type


def test_prompt_packet_relay_frontend_workflow_connects_one_studio_fact_source():
    root = Path(__file__).resolve().parents[1]
    path = (
        root
        / "examples"
        / "workflows"
        / "14-prompt-relay"
        / "2026-08-20_H3_Studio_Prompt_Packet_Relay_Stock20_Advanced_EXP.json"
    )
    workflow = json.loads(path.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in workflow["nodes"]}
    by_type = {node["type"]: node for node in workflow["nodes"]}
    links = {link[0]: link for link in workflow["links"]}

    assert workflow["version"] == 0.4
    assert workflow["last_node_id"] == max(nodes)
    assert workflow["last_link_id"] == max(links)
    assert sum(node["type"] == "MarkdownNote" for node in nodes.values()) == 3
    assert "MiniMaxH3PromptRelayPlanT8Advanced" not in by_type
    assert "MiniMaxH3PromptPacketRelayPlanT8Advanced" in by_type
    event_nodes = sorted(
        (
            node
            for node in nodes.values()
            if node["type"] == "MiniMaxH3PromptRelayEventT8Advanced"
        ),
        key=lambda node: node["id"],
    )
    assert len(event_nodes) == 3

    def source_for_input(node, name):
        item = next(value for value in node["inputs"] if value["name"] == name)
        if item["link"] is None:
            return None
        link = links[item["link"]]
        return nodes[link[1]], link[2]

    cast = by_type["MiniMaxH3UnifiedCastT8Advanced"]
    canvas = by_type["MiniMaxH3SoundCanvasT8Advanced"]
    compiler = by_type["MiniMaxH3PromptCompilerT8Advanced"]
    bridge = by_type["MiniMaxH3PromptPacketRelayPlanT8Advanced"]
    preview = by_type["MiniMaxH3PromptRelayPreviewT8Advanced"]
    conditioning = by_type["MiniMaxH3PromptRelayConditioningT8Advanced"]
    assert source_for_input(compiler, "cast") == (cast, 0)
    assert source_for_input(compiler, "sound_canvas") == (canvas, 0)
    assert source_for_input(bridge, "prompt_packet") == (compiler, 0)
    assert source_for_input(event_nodes[0], "previous_events") is None
    assert source_for_input(event_nodes[1], "previous_events") == (
        event_nodes[0],
        0,
    )
    assert source_for_input(event_nodes[2], "previous_events") == (
        event_nodes[1],
        0,
    )
    assert source_for_input(bridge, "prompt_relay_events") == (event_nodes[2], 0)
    assert source_for_input(preview, "prompt_relay_plan") == (bridge, 0)
    assert source_for_input(conditioning, "prompt_relay_plan") == (preview, 0)
    assert compiler["widgets_values"][1] == "minimax_h3"
    assert compiler["widgets_values"][2] == pytest.approx(5.166667)
    assert bridge["widgets_values"][1:4] == ["frames", "paper_v1", 0.1]
    assert [node["widgets_values"][1:4] for node in event_nodes] == [
        [0, 40, True],
        [41, 81, True],
        [82, 123, True],
    ]
    assert conditioning["widgets_values"][-2:] == ["report_only", 256]
    assert by_type["MiniMaxH3DualClockSamplerT8"]["widgets_values"][0] == 20

    for link_id, source, output_slot, target, input_slot, link_type in workflow["links"]:
        assert nodes[target]["inputs"][input_slot]["link"] == link_id
        assert link_id in (nodes[source]["outputs"][output_slot].get("links") or [])
        assert nodes[source]["outputs"][output_slot]["type"] == link_type
        assert nodes[target]["inputs"][input_slot]["type"] == link_type


def test_prompt_relay_plan_preview_frontend_workflow_is_model_free_and_documented():
    root = Path(__file__).resolve().parents[1]
    path = (
        root
        / "examples"
        / "workflows"
        / "14-prompt-relay"
        / "2026-08-20_H3_Prompt_Relay_Plan_Preview_Advanced_EXP.json"
    )
    workflow = json.loads(path.read_text(encoding="utf-8-sig"))
    nodes = {node["id"]: node for node in workflow["nodes"]}
    links = {link[0]: link for link in workflow["links"]}
    by_type = {node["type"]: node for node in workflow["nodes"]}

    assert workflow["version"] == 0.4
    assert workflow["last_node_id"] == max(nodes)
    assert workflow["last_link_id"] == max(links)
    assert sum(node["type"] == "MarkdownNote" for node in nodes.values()) == 3
    assert sum(
        node["type"] == "MiniMaxH3PromptRelayEventT8Advanced"
        for node in nodes.values()
    ) == 3
    forbidden_types = {
        "UNETLoader",
        "CLIPLoader",
        "VAELoader",
        "MiniMaxH3DualClockSamplerT8",
        "MiniMaxH3PromptRelayConditioningT8Advanced",
    }
    assert forbidden_types.isdisjoint({node["type"] for node in nodes.values()})

    plan = by_type["MiniMaxH3PromptRelayPlanT8Advanced"]
    resource = by_type["MiniMaxH3PromptRelayResourceEstimateT8Advanced"]
    preview = by_type["MiniMaxH3PromptRelayPreviewT8Advanced"]
    resource_input = next(
        item for item in resource["inputs"] if item["name"] == "prompt_relay_plan"
    )
    plan_to_resource = links[resource_input["link"]]
    assert plan_to_resource[1:5] == [plan["id"], 0, resource["id"], 0]
    preview_input = next(
        item for item in preview["inputs"] if item["name"] == "prompt_relay_plan"
    )
    resource_to_preview = links[preview_input["link"]]
    assert resource_to_preview[1:5] == [resource["id"], 0, preview["id"], 0]
    assert resource["widgets_values"] == [
        736,
        416,
        256,
        "bf16_fp16",
        0,
        0,
        0,
        124,
        False,
        5.0,
        0,
        5.0,
        256,
        0,
    ]
    assert preview["outputs"][0].get("links") in (None, [])

    for link_id, source, output_slot, target, input_slot, link_type in workflow["links"]:
        assert nodes[target]["inputs"][input_slot]["link"] == link_id
        assert link_id in (nodes[source]["outputs"][output_slot].get("links") or [])
        assert nodes[source]["outputs"][output_slot]["type"] == link_type
        assert nodes[target]["inputs"][input_slot]["type"] == link_type
