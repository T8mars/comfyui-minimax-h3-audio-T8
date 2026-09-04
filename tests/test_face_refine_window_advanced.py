from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
import torch

from h3_audio_t8_pkg.face_refine_window_advanced import (
    WINDOW_MAPPING_SCHEMA,
    WINDOW_PLAN_SCHEMA,
    apply_face_refine_manual_review,
    build_face_refine_window_plan,
    extract_face_refine_window,
)
from h3_audio_t8_pkg.nodes import comfy_entrypoint
from h3_audio_t8_pkg.nodes_face_refine_window_advanced import (
    FACE_REFINE_WINDOW_ADVANCED_NODE_CLASSES,
)


def _frames(count: int, *, height: int = 8, width: int = 12) -> torch.Tensor:
    base = torch.linspace(0.1, 0.2, count).view(count, 1, 1, 1)
    return base.expand(count, height, width, 3).contiguous()


def _plan(
    frames: torch.Tensor,
    ranges: str = "0-23",
    *,
    enabled: bool = True,
    overlap_policy: str = "reject",
    short_shot_policy: str = "edge_hold_exp",
    min_render_frames: int = 90,
    context_before_frames: int = 24,
    context_after_frames: int = 42,
):
    return build_face_refine_window_plan(
        base_frames=frames,
        fps=24.0,
        repair_ranges=ranges,
        range_mode="frames_inclusive",
        context_before_frames=context_before_frames,
        context_after_frames=context_after_frames,
        min_render_frames=min_render_frames,
        max_render_frames=362,
        scene_cut_threshold=0.9,
        overlap_policy=overlap_policy,
        short_shot_policy=short_shot_policy,
        enabled=enabled,
    )


def _extract(frames: torch.Tensor, plan: dict, audio=None):
    return extract_face_refine_window(
        base_frames=frames,
        window_plan=plan,
        window_index=0,
        pad_policy="edge_hold_exp",
        source_audio=audio,
    )


def test_89_frame_fixture_forms_legal_90_and_padding_is_not_acceptable():
    frames = _frames(89)
    plan, mask, count, report = _plan(frames)
    assert plan["schema"] == WINDOW_PLAN_SCHEMA
    assert count == 1
    assert mask.shape == (89, 8, 12)
    assert int(mask[:24].sum()) == 24 * 8 * 12
    assert int(mask[24:].sum()) == 0
    window = plan["windows"][0]
    assert window["render_frame_count"] == 90
    assert (window["render_frame_count"] - 5) % 17 == 0
    assert window["pre_pad_frames"] == 0
    assert window["post_pad_frames"] == 1
    assert window["accept_relative_ranges"] == [[0, 23]]
    assert json.loads(report)["plan_sha256"] == plan["plan_sha256"]

    render, _, mapping, *_ = _extract(frames, plan)
    assert render.shape[0] == 90
    assert torch.equal(render[:89], frames)
    assert torch.equal(render[89], frames[88])
    assert mapping["schema"] == WINDOW_MAPPING_SCHEMA
    assert mapping["frame_map"][89]["source_frame"] is None
    assert mapping["frame_map"][89]["kind"] == "padding"


@pytest.mark.parametrize("count", [22, 89, 90, 124, 311, 362])
def test_supported_source_lengths_produce_legal_windows(count: int):
    frames = _frames(count)
    end = min(count - 1, 10)
    plan, *_ = _plan(
        frames,
        f"0-{end}",
        context_before_frames=0,
        context_after_frames=0,
        min_render_frames=22,
    )
    window = plan["windows"][0]
    assert (window["render_frame_count"] - 5) % 17 == 0
    assert window["render_frame_count"] <= 362
    assert window["accept_relative_ranges"][0][0] >= 0
    assert window["accept_relative_ranges"][0][1] < window["render_frame_count"]


def test_window_shifts_at_beginning_middle_and_end_without_padding():
    frames = _frames(124)
    for repair, expected_start in [("0-5", 0), ("60-65", 28), ("118-123", 34)]:
        plan, *_ = _plan(
            frames,
            repair,
            short_shot_policy="reject",
            context_before_frames=32,
            context_after_frames=32,
        )
        window = plan["windows"][0]
        assert window["render_source_start_frame"] == expected_start
        assert window["render_source_end_frame"] == expected_start + 89
        start, end = window["repair_ranges_abs"][0]
        assert expected_start <= start <= end <= expected_start + 89


def test_overlap_reject_and_explicit_merge():
    frames = _frames(124)
    with pytest.raises(ValueError, match="overlap"):
        _plan(frames, "10-20,20-30")
    plan, *_ = _plan(frames, "10-20,20-30", overlap_policy="merge")
    assert plan["normalised_ranges_frames"] == [[10, 30]]
    assert plan["window_count"] == 1


def test_hard_cut_crossing_is_rejected():
    frames = torch.zeros((124, 8, 12, 3))
    frames[62:] = 1.0
    with pytest.raises(ValueError, match="hard cut"):
        build_face_refine_window_plan(
            base_frames=frames,
            fps=24.0,
            repair_ranges="60-64",
            range_mode="frames_inclusive",
            context_before_frames=0,
            context_after_frames=0,
            min_render_frames=22,
            max_render_frames=362,
            scene_cut_threshold=0.2,
            overlap_policy="reject",
            short_shot_policy="edge_hold_exp",
            enabled=True,
        )


def test_disabled_and_empty_plans_are_exact_noop_contracts():
    frames = _frames(89)
    disabled, disabled_mask, disabled_count, _ = _plan(
        frames, "this stale widget value is ignored", enabled=False
    )
    empty, empty_mask, empty_count, _ = _plan(frames, "")
    assert disabled["status"] == "disabled_noop"
    assert empty["status"] == "empty_noop"
    assert disabled_count == empty_count == 0
    assert torch.count_nonzero(disabled_mask) == 0
    assert torch.count_nonzero(empty_mask) == 0
    assert disabled["windows"] == empty["windows"] == []


def test_stale_plan_and_replaced_source_are_refused():
    frames = _frames(89)
    plan, *_ = _plan(frames)
    stale = dict(plan)
    stale["status"] = "tampered"
    with pytest.raises(ValueError, match="hash mismatch"):
        _extract(frames, stale)
    changed = frames.clone()
    changed[0] += 0.25
    with pytest.raises(ValueError, match="source-bound"):
        _extract(changed, plan)


def test_audio_uses_rational_frame_boundaries_and_zero_padding():
    frames = _frames(89)
    plan, *_ = _plan(frames)
    waveform = torch.arange(118_667, dtype=torch.float32).view(1, 1, -1)
    audio = {"waveform": waveform, "sample_rate": 32_000}
    _, render_audio, mapping, *_ = _extract(frames, plan, audio)
    assert render_audio["waveform"].shape == (1, 1, 120_000)
    assert torch.equal(render_audio["waveform"][..., :118_667], waveform)
    assert torch.count_nonzero(render_audio["waveform"][..., 118_667:]) == 0
    assert mapping["audio"] == {
        "connected": True,
        "sample_rate": 32_000,
        "source_start_sample": 0,
        "source_end_sample_exclusive": 118_667,
        "render_start_sample": 0,
        "render_end_sample_exclusive": 118_667,
        "target_sample_count": 120_000,
        "padding_samples_are_zero": True,
    }


def _candidate_fixture():
    frames = _frames(89)
    plan, *_ = _plan(frames)
    render, _, mapping, *_ = _extract(frames, plan)
    candidate = render.clone()
    changed = torch.zeros((90, 8, 12))
    changed[:24, 2:6, 3:9] = 1.0
    candidate[:24, 2:6, 3:9] = (candidate[:24, 2:6, 3:9] + 0.05).clamp(0, 1)
    return frames, candidate, changed, mapping


@pytest.mark.parametrize("decision", ["preview_only", "reject"])
def test_preview_and_reject_return_source_bit_exact(decision: str):
    frames, candidate, changed, mapping = _candidate_fixture()
    review, result, accepted, rejected, accepted_count, rejected_count, report = (
        apply_face_refine_manual_review(
            frames,
            candidate,
            changed,
            mapping,
            decision,
            "",
            False,
            2,
        )
    )
    assert review.shape == (90, 8, 24, 3)
    assert torch.equal(result, frames)
    assert torch.count_nonzero(accepted) == 0
    assert accepted_count == 0
    assert rejected_count == 24
    assert int(torch.count_nonzero(rejected)) == 24 * 4 * 6
    assert json.loads(report)["source_preserved_outside_accepted_mask_bit_exact"] is True


def test_accept_requires_confirmation_and_stays_inside_selected_range():
    frames, candidate, changed, mapping = _candidate_fixture()
    unconfirmed = apply_face_refine_manual_review(
        frames, candidate, changed, mapping, "accept_selected", "4-7", False, 2
    )
    assert torch.equal(unconfirmed[1], frames)
    assert json.loads(unconfirmed[-1])["status"] == "rejected_unconfirmed"

    accepted = apply_face_refine_manual_review(
        frames, candidate, changed, mapping, "accept_selected", "4-7", True, 1
    )
    result, accepted_mask = accepted[1], accepted[2]
    assert accepted[4] == 4
    assert torch.equal(result[:4], frames[:4])
    assert torch.equal(result[8:], frames[8:])
    assert torch.count_nonzero(accepted_mask[:4]) == 0
    assert torch.count_nonzero(accepted_mask[8:]) == 0
    assert not torch.equal(result[4:8], frames[4:8])


def test_context_padding_and_out_of_plan_subranges_cannot_be_accepted():
    frames, candidate, changed, mapping = _candidate_fixture()
    result = apply_face_refine_manual_review(
        frames, candidate, changed, mapping, "accept_selected", "24-30", True, 0
    )
    assert torch.equal(result[1], frames)
    assert json.loads(result[-1])["status"] == "rejected_invalid_selection"


def test_candidate_change_outside_mask_and_nonfinite_values_fail_closed():
    frames, candidate, changed, mapping = _candidate_fixture()
    candidate[10, 0, 0] += 0.1
    rejected = apply_face_refine_manual_review(
        frames, candidate, changed, mapping, "accept_selected", "0-23", True, 0
    )
    assert torch.equal(rejected[1], frames)
    assert json.loads(rejected[-1])["status"] == "rejected_contract"

    candidate, changed = _candidate_fixture()[1:3]
    candidate[10, 2, 3] = float("nan")
    rejected = apply_face_refine_manual_review(
        frames, candidate, changed, mapping, "accept_selected", "0-23", True, 0
    )
    assert torch.equal(rejected[1], frames)
    assert json.loads(rejected[-1])["status"] == "rejected_contract"


def test_stale_mapping_and_replaced_source_fail_closed_to_source():
    frames, candidate, changed, mapping = _candidate_fixture()
    stale = dict(mapping)
    stale["window_plan_sha256"] = "0" * 64
    result = apply_face_refine_manual_review(
        frames, candidate, changed, stale, "accept_selected", "0-23", True, 0
    )
    assert torch.equal(result[1], frames)
    assert json.loads(result[-1])["status"] == "rejected_contract"

    replaced = frames.clone()
    replaced[0] += 0.2
    result = apply_face_refine_manual_review(
        replaced, candidate, changed, mapping, "accept_selected", "0-23", True, 0
    )
    assert torch.equal(result[1], replaced)
    assert json.loads(result[-1])["status"] == "rejected_contract"


def test_three_node_schema_and_absolute_tail_registration():
    ids = [node.define_schema().node_id for node in FACE_REFINE_WINDOW_ADVANCED_NODE_CLASSES]
    assert ids == [
        "MiniMaxH3FaceRefineWindowPlanT8Advanced",
        "MiniMaxH3FaceRefineWindowExtractT8Advanced",
        "MiniMaxH3FaceRefineManualReviewT8Advanced",
    ]
    registered = [node.define_schema().node_id for node in asyncio.run(comfy_entrypoint().get_node_list())]
    assert registered[292:295] == ids
    assert len(registered) == len(set(registered))


def test_api_fixture_wires_the_full_source_audio_only_at_final_mux():
    root = Path(__file__).resolve().parents[1]
    graph = json.loads(
        (root / "tests" / "fixtures" / "api" / "face_refine_window_advanced_api.json").read_text(
            encoding="utf-8"
        )
    )
    types = [node["class_type"] for node in graph.values()]
    assert types.count("MiniMaxH3FaceRefineWindowPlanT8Advanced") == 1
    assert types.count("MiniMaxH3FaceRefineWindowExtractT8Advanced") == 1
    assert types.count("MiniMaxH3FaceRefineManualReviewT8Advanced") == 1
    extract = graph["5"]["inputs"]
    assert extract["base_frames"] == ["2", 0]
    assert extract["source_audio"] == ["2", 1]
    assert graph["13"]["inputs"]["drive_audio"] == ["5", 1]
    assert graph["24"]["inputs"]["base_frames"] == ["2", 0]
    assert graph["24"]["inputs"]["candidate_window_frames"] == ["23", 0]
    assert graph["24"]["inputs"]["changed_mask"] == ["22", 1]
    assert graph["25"]["inputs"]["images"] == ["24", 1]
    assert graph["25"]["inputs"]["audio"] == ["2", 1]


def test_frontend_workflow_matches_builder_and_uses_preview_by_default():
    from h3_audio_t8_pkg.tools.build_face_refine_window_workflow import build

    root = Path(__file__).resolve().parents[1]
    path = (
        root
        / "examples"
        / "workflows"
        / "06-face-refine"
        / "2026-09-05_H3_Face_Refine_Window_Manual_Review_Advanced_EXP.json"
    )
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved == build()
    by_type = {node["type"]: node for node in saved["nodes"]}
    assert by_type["MiniMaxH3FaceRefineManualReviewT8Advanced"]["widgets_values"] == [
        "preview_only",
        "0-23",
        False,
        2,
    ]
    assert by_type["MiniMaxH3FaceRefineWindowPlanT8Advanced"]["widgets_values"][-2:] == [
        "edge_hold_exp",
        True,
    ]
    node_ids = [node["id"] for node in saved["nodes"]]
    assert len(node_ids) == len(set(node_ids))
    link_ids = [link[0] for link in saved["links"]]
    assert len(link_ids) == len(set(link_ids))
