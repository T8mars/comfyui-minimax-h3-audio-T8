from __future__ import annotations

import json

import pytest

from h3_audio_t8_pkg.long_video_orchestration import (
    build_long_video_chain_plan,
    resolve_long_video_orchestration,
)
from h3_audio_t8_pkg.nodes_long_video_delivery_exp import (
    MiniMaxH3LongVideoOrchestratorT8,
)


def _manifest_for(segments, accepted_count):
    entries = []
    for segment in segments[:accepted_count]:
        plan = segment.plan
        entries.append(
            {
                "index": segment.index,
                "candidate_id": f"take-{segment.index}",
                "frame_count": plan.final_frame_count,
                "fps": 24,
                "timeline_start_frame": round(plan.timeline_start_seconds * 24),
                "timeline_end_frame": round(plan.timeline_end_seconds * 24),
                "is_final_segment": plan.is_final_segment,
                "video_path": f"accepted/segment_{segment.index:05d}.mp4",
                "prompt": f"conditioned {segment.prompt}",
                "seed": segment.seed,
                "sampling_summary": "4-step dual_clock_euler/native_flow shift12/3",
            }
        )
    return {
        "schema": 1,
        "chain_id": "chain",
        "revision": accepted_count,
        "segments": entries,
        "invalidated": [],
    }


def test_sixty_second_plan_uses_fixed_124_frame_windows_and_exact_final_frame():
    segments = build_long_video_chain_plan(
        "chain", 60.0, 124, 22, "global", "", 1000, "increment"
    )
    assert len(segments) == 14
    assert [segment.plan.final_frame_count for segment in segments] == (
        [124] + [102] * 12 + [92]
    )
    assert all(segment.plan.render_frames == 124 for segment in segments)
    assert segments[0].plan.context_frames == 0
    assert all(segment.plan.context_frames == 22 for segment in segments[1:])
    assert sum(segment.plan.final_frame_count for segment in segments) == 1440
    assert segments[-1].plan.timeline_end_seconds == pytest.approx(60.0)
    assert [segment.seed for segment in segments] == list(range(1000, 1014))
    assert all(segment.prompt == "global" for segment in segments)
    assert not any(segment.plan.is_final_segment for segment in segments[:-1])
    assert segments[-1].plan.is_final_segment is True
    assert segments[-1].plan.save_context is False


def test_short_total_duration_is_one_final_fixed_window_with_exact_trim():
    segments = build_long_video_chain_plan("short", 1.0, 124, 22)
    assert len(segments) == 1
    plan = segments[0].plan
    assert plan.render_frames == 124
    assert plan.final_frame_count == 24
    assert plan.hidden_tail_frames == 100
    assert plan.context_frames == 0
    assert plan.is_final_segment is True
    assert plan.save_context is False


def test_prompt_seed_overrides_and_hash_policy_are_deterministic():
    overrides = json.dumps(
        {
            "segments": {
                "1": {"prompt": "close-up", "seed": 77, "note": "shot B"},
                "2": "wide ending",
            }
        }
    )
    first = build_long_video_chain_plan(
        "中文 chain", 12.0, 124, 22, "global", overrides, 9, "hash_chain_segment"
    )
    second = build_long_video_chain_plan(
        "中文 chain", 12.0, 124, 22, "global", overrides, 9, "hash_chain_segment"
    )
    assert [item.seed for item in first] == [item.seed for item in second]
    assert first[1].prompt == "close-up"
    assert first[1].seed == 77
    assert first[1].note == "shot B"
    assert first[2].prompt == "wide ending"
    assert first[0].prompt == "global"


def test_orchestrator_resumes_at_first_unaccepted_manifest_segment(monkeypatch):
    import h3_audio_t8_pkg.long_video_orchestration as orchestration

    segments = build_long_video_chain_plan(
        "chain", 20.0, 124, 22, "global", "", 10, "increment"
    )
    manifest = _manifest_for(segments, 2)
    monkeypatch.setattr(
        orchestration, "load_delivery_manifest", lambda _chain: (manifest, "primary")
    )
    result, loaded = resolve_long_video_orchestration(
        "chain", 20.0, 124, 22, "global", "", 10, "increment"
    )
    assert loaded is manifest
    assert result.accepted_count == 2
    assert result.next_segment.index == 2
    assert result.progress == pytest.approx(2 / len(segments))
    plan = json.loads(result.plan_json(manifest))
    assert [item["status"] for item in plan["segments"][:4]] == [
        "accepted", "accepted", "next", "pending",
    ]
    assert plan["segments"][0]["accepted_candidate_id"] == "take-0"


def test_changed_timeline_settings_are_rejected_for_an_existing_chain(monkeypatch):
    import h3_audio_t8_pkg.long_video_orchestration as orchestration

    original = build_long_video_chain_plan("chain", 20.0, 124, 22)
    manifest = _manifest_for(original, 2)
    monkeypatch.setattr(
        orchestration, "load_delivery_manifest", lambda _chain: (manifest, "primary")
    )
    with pytest.raises(ValueError, match="conflicts with this total-duration plan"):
        resolve_long_video_orchestration("chain", 20.0, 141, 22)
    with pytest.raises(ValueError, match="sampling_summary"):
        resolve_long_video_orchestration(
            "chain", 20.0, 124, 22, steps=8, sampler_name="dual_clock_euler"
        )


def test_complete_manifest_returns_an_execution_blocker_before_sampling(monkeypatch):
    import h3_audio_t8_pkg.long_video_orchestration as orchestration

    segments = build_long_video_chain_plan("chain", 5.0, 124, 22, "done")
    manifest = _manifest_for(segments, len(segments))
    monkeypatch.setattr(
        orchestration, "load_delivery_manifest", lambda _chain: (manifest, "primary")
    )
    output = MiniMaxH3LongVideoOrchestratorT8.execute(
        "chain", 5.0, 124, 22, "done", "", 0, "increment"
    )
    assert output.block_execution is not None
    assert "is complete" in output.block_execution
    assert output.args[12] is False
    assert output.args[13] == pytest.approx(1.0)
    assert output.args[16:22] == (
        4, 12.0, 3.0, "dual_clock_euler", "native_flow",
        "4-step dual_clock_euler/native_flow shift12/3",
    )


def test_duration_quantization_and_invalid_plan_inputs_are_reported(monkeypatch):
    import h3_audio_t8_pkg.long_video_orchestration as orchestration

    def missing(_chain):
        raise FileNotFoundError

    monkeypatch.setattr(orchestration, "load_delivery_manifest", missing)
    result, manifest = resolve_long_video_orchestration("quantized", 1.01, 124, 22)
    assert manifest is None
    assert result.total_frame_count == 24
    assert result.quantized_total_duration_seconds == 1.0
    assert result.warnings
    tiny, _ = resolve_long_video_orchestration("one-frame", 0.001, 124, 22)
    assert tiny.total_frame_count == 1
    assert tiny.quantized_total_duration_seconds == pytest.approx(1 / 24)
    assert tiny.sampling_summary == "4-step dual_clock_euler/native_flow shift12/3"
    with pytest.raises(ValueError, match=r"17n\+5 grid"):
        build_long_video_chain_plan("bad", 5.0, 125, 22)
    with pytest.raises(ValueError, match="outside this chain"):
        build_long_video_chain_plan("bad", 1.0, 124, 22, "", '{"9": "unused"}')
