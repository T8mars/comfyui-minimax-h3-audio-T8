"""Offline contract tests for real PDD freeze -> cold audio-tail probe."""

from copy import deepcopy
import json

import pytest

from tools import run_modular_s26_pdd_resume_gpu as probe


@pytest.mark.parametrize("route,load,guard,save", (
    ("pdd8", "33", "34", "29"),
    ("pdd4plus4", "43", "44", "39"),
))
def test_resume_graph_only_fills_frozen_receipt_and_test_output(route, load, guard, save):
    original = json.loads(probe._source(route).read_text(encoding="utf-8"))
    receipt = {"checkpoint": {"relative_path": "frozen.h3latent.safetensors",
                              "file_sha256": "A" * 64,
                              "embedded_manifest": {"checkpoint_id": "audio_refine_firstpass"}}}
    graph, digest = probe.build_resume_graph(route, receipt)
    assert digest == probe.freeze._sha(probe._source(route))
    expected = deepcopy(original)
    expected[load]["inputs"].update(
        checkpoint_path="frozen.h3latent.safetensors",
        expected_manifest_json='{"checkpoint_id":"audio_refine_firstpass"}',
        expected_file_sha256="A" * 64)
    expected[guard]["inputs"]["expected_video_frame_count"] = 22
    expected[save]["inputs"]["filename_prefix"] = f"MiniMaxH3/S26_PDD_ColdAudio/{route}_selected"
    for added in ("200", "201", "202"):
        graph.pop(added)
    assert graph == expected
    assert sum(node["class_type"] == "SamplerCustomAdvanced" for node in graph.values()) == 1


@pytest.mark.parametrize("route,sampler,load,old", (
    ("pdd8", "21", "33", ["11"]),
    ("pdd4plus4", "31", "43", ["12", "19"]),
))
def test_tail_audit_requires_load_then_exact_four_tail_progress(route, sampler, load, old):
    phase = {"terminal": {"type": "execution_success"}, "events": [
        {"type": "executing", "node": load},
        {"type": "executing", "node": sampler},
        *({"type": "progress", "node": sampler} for _ in range(4)),
    ]}
    assert all(probe._tail_checks(route, phase).values())
    phase["events"].insert(0, {"type": "executing", "node": old[0]})
    assert probe._tail_checks(route, phase)["no_first_pass_sampler_nodes"] is False
    phase["events"].pop(0)
    phase["events"][-1]["node"] = "wrong"
    assert probe._tail_checks(route, phase)["exactly_four_audio_tail_steps"] is False


def test_delivery_audit_requires_changed_audio_stable_video_and_abstain():
    report = {"route": "pdd8", "load_status": "MATCH_EXTERNAL",
              "route_report": '{"decision":"ALLOW"}',
              "quality_decision": "ABSTAIN_HUMAN_REVIEW_REQUIRED"}
    phase = {"executed_outputs": {"22": {"text": [json.dumps({
        "audio_exact_equal": False, "audio_finite": True, "audio_rmse": .4,
        "video_finite": True, "video_max_abs": 4e-7})]}}}
    assert all(probe._delivery_checks(report, phase).values())
    phase["executed_outputs"]["22"]["text"] = [json.dumps({
        "audio_exact_equal": True, "audio_finite": True, "audio_rmse": 0,
        "video_finite": True, "video_max_abs": 4e-7})]
    assert probe._delivery_checks(report, phase)["candidate_audio_changed_and_finite"] is False
