"""LTX Relay's 8n+1 time base must not inherit H3's 17n+5 alignment."""
import asyncio
import json

import pytest
import torch

import h3_audio_t8_pkg
from h3_audio_t8_pkg.modular_sampling.ltx_relay_plan import (
    build_ltx_relay_plan,
    validate_ltx_relay_plan,
)
from h3_audio_t8_pkg.modular_sampling.ltx_relay_nodes import (
    MiniMaxH3LTXPromptRelayPlanEXPT8,
    MiniMaxH3LTXPromptRelayEncodeEXPT8,
    MiniMaxH3LTXPromptRelayApplyEXPT8,
    MiniMaxH3LTXPromptRelayAuditEXPT8,
)
from h3_audio_t8_pkg.prompt_relay_events_advanced import build_prompt_relay_event


def latent(t=15):
    return {"samples": torch.zeros(1, 128, t, 18, 32)}


def inputs(**changes):
    base = dict(ltx_latent=latent(), global_prompt="Continuity and original ambience",
                local_prompts="Subject enters.\nSubject turns.", timing_mode="auto_equal",
                time_ranges="", fps=24., epsilon=.1, allow_gaps=False,
                allow_overlaps=False)
    return {**base, **changes}


def test_exact_113_frame_grid_global_events_and_schema():
    plan, compiled, frames, timeline, report = build_ltx_relay_plan(**inputs())
    assert frames == 113 and plan["latent_shape"] == [1, 128, 15, 18, 32]
    assert plan["temporal_contract"] == "ltx_output_frames_8n_plus_1"
    assert [(e["start_frame"], e["end_frame_exclusive"]) for e in plan["events"]] == [
        (0, 56), (56, 113)]
    assert compiled == plan["compiled_prompt"]
    assert json.loads(timeline)["frame_count"] == 113
    assert json.loads(report)["attention_applied"] is False
    assert json.loads(report)["paper_qualified_for_ltx"] is False
    assert validate_ltx_relay_plan(plan, latent())["plan_hash"] == plan["plan_hash"]


@pytest.mark.parametrize("mode,ranges,expected", [
    ("frames", "0-39\n40-112", [(0, 40), (40, 113)]),
    ("seconds", "0-2\n2-4.7083333333333", [(0, 48), (48, 113)]),
    ("percent", "0-50\n50-100", [(0, 56), (56, 113)]),
])
def test_explicit_ranges_follow_actual_output_frames(mode, ranges, expected):
    plan, *_ = build_ltx_relay_plan(**inputs(timing_mode=mode, time_ranges=ranges))
    assert [(e["start_frame"], e["end_frame_exclusive"]) for e in plan["events"]] == expected


def test_event_chain_overrides_text_fields_and_zero_events_is_bypass():
    chain, *_ = build_prompt_relay_event("First beat", 0, 55, True)
    chain, *_ = build_prompt_relay_event("Second beat", 56, 112, True, chain)
    plan, *_ = build_ltx_relay_plan(**inputs(timing_mode="frames", time_ranges="bad",
                                              local_prompts="ignored", prompt_relay_events=chain))
    assert [event["local_prompt"] for event in plan["events"]] == ["First beat", "Second beat"]
    assert plan["source_events"]["event_count"] == 2
    bypass, *_ = build_ltx_relay_plan(**inputs(local_prompts=""))
    assert bypass["events"] == []


@pytest.mark.parametrize("changes,match", [
    ({"global_prompt": " "}, "global prompt"),
    ({"epsilon": 1.}, "epsilon"),
    ({"fps": float("nan")}, "fps"),
    ({"timing_mode": "frames", "time_ranges": "0-3\n4-112"}, "at least 5"),
    ({"timing_mode": "frames", "time_ranges": "0-40\n40-112"}, "overlap"),
])
def test_invalid_timeline_is_rejected(changes, match):
    with pytest.raises(ValueError, match=match):
        build_ltx_relay_plan(**inputs(**changes))


def test_hash_latent_binding_and_append_only_registration():
    plan, *_ = build_ltx_relay_plan(**inputs())
    changed = dict(plan)
    changed["frame_count"] = 124
    with pytest.raises(ValueError, match="hash"):
        validate_ltx_relay_plan(changed)
    with pytest.raises(ValueError, match="connected LTX latent"):
        validate_ltx_relay_plan(plan, latent(16))
    prior = [*asyncio.run(h3_audio_t8_pkg._HyperFlowLongVideoExtension().get_node_list()),
             *h3_audio_t8_pkg._modular_node_classes(), *h3_audio_t8_pkg._hyper_vae_2x_node_classes,
             *h3_audio_t8_pkg._audio_refine_effect_node_classes,
             *h3_audio_t8_pkg._ltx_rgb_source_node_classes,
             *h3_audio_t8_pkg._serial_video_io_node_classes,
             *h3_audio_t8_pkg._ltx_effect_node_classes]
    actual = asyncio.run(h3_audio_t8_pkg.comfy_entrypoint().get_node_list())
    assert len(prior) == 563 and actual[:567] == prior + [MiniMaxH3LTXPromptRelayPlanEXPT8,
                                                        MiniMaxH3LTXPromptRelayEncodeEXPT8,
                                                        MiniMaxH3LTXPromptRelayApplyEXPT8,
                                                        MiniMaxH3LTXPromptRelayAuditEXPT8]
    assert len(actual) == len({node.define_schema().node_id for node in actual})
    result = MiniMaxH3LTXPromptRelayPlanEXPT8.execute(**inputs()).result
    assert result[2] == 113 and result[0]["plan_hash"] == plan["plan_hash"]
