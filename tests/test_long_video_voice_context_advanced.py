from __future__ import annotations

import json

import pytest

from h3_audio_t8_pkg.long_video_voice_context_advanced import (
    build_long_video_voice_context_plan,
    release_long_video_voice_context_plan,
)


def _build(dialogue, *, policy="abstain", review=True):
    return build_long_video_voice_context_plan(
        "voice-test",
        226 / 24,
        124,
        22,
        "同一镜头，保持人物身份和环境声连续。",
        json.dumps(dialogue, ensure_ascii=False),
        json.dumps({"甲": 1, "B": 2}, ensure_ascii=False),
        policy,
        review,
    )


def test_multilingual_turns_compile_exact_tags_and_independent_pin_frames():
    result = _build(
        [
            {"speaker": "甲", "text": "你好。", "start_seconds": 0, "end_seconds": 1},
            {"speaker": "B", "text": "I am here!", "start_seconds": 6, "end_seconds": 7},
        ]
    )
    plan, prompt_json, pins_json, ready, review_required, report_json = result
    assert ready is True and review_required is True
    assert "<Audio 1>" in prompt_json and "<Audio 2>" in prompt_json
    pins = json.loads(pins_json)["audio_pin_frames"]
    assert pins[0]["render_local_start_frame"] == 0
    assert all(item["audio_injection"].startswith("reference_tag_only") for item in pins)
    assert plan["prompt_primary_audio_ordinal_required"] == 0
    assert json.loads(report_json)["turn_count"] == 2


def test_cross_boundary_long_sentence_abstains_without_silent_truncation():
    plan, prompt_json, _pins, ready, _review, report_json = _build(
        [{"speaker": "甲", "text": "这一句会跨越固定分段边界", "start_seconds": 4.5, "end_seconds": 5.5}]
    )
    assert ready is False
    assert plan["status"] == "abstain_cross_boundary_sentence"
    assert len(plan["cross_boundary_turns"]) == 1
    assert json.loads(report_json)["warnings"]
    assert "这一句" not in prompt_json


def test_experimental_duplicate_policy_preserves_exact_text_in_both_segments():
    plan, prompt_json, _pins, ready, _review, _report = _build(
        [{"speaker": "甲", "text": "连续长句", "start_seconds": 4.5, "end_seconds": 5.5}],
        policy="duplicate_exact_text_exp",
    )
    assert ready is True
    prompts = json.loads(prompt_json)["segments"]
    assert sum("连续长句" in item["prompt"] for item in prompts) == 2
    assert plan["adaptive_boundary_shift"] is False


def test_no_dialogue_is_valid_and_review_gate_is_hash_bound():
    plan, _prompts, _pins, ready, _review, _report = _build([])
    assert ready is True
    blocked = release_long_video_voice_context_plan(plan, False)
    assert blocked[0] == "" and blocked[2] is False
    released = release_long_video_voice_context_plan(plan, True)
    assert released[0] and released[2] is True
    plan["segments"][0]["prompt"] = "tampered"
    with pytest.raises(ValueError, match="hash"):
        release_long_video_voice_context_plan(plan, True)
