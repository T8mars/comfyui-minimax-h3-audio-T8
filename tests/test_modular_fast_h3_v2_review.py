"""Standalone V2 review is an explicit step, not hidden in Candidate Save."""

import json
from pathlib import Path

from tools import build_modular_fast_h3_v2_review_workflow as builder
from tools import build_modular_fast_h3_v2_compose_workflow as composer


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "artifacts/development/modular-sampling-m1-v2-review-accept-v1-20260925"
COMPOSE_CANDIDATE = ROOT / "artifacts/development/modular-sampling-m1-v2-source-compose-v1-20260925"


def test_saved_review_graph_starts_preview_only_and_roundtrips_exactly():
    graph = builder.review_graph()
    assert len(graph) == 2
    assert graph["1"]["inputs"]["accept_candidate"] is False
    assert graph["1"]["class_type"] == "MiniMaxH3FastH3V2CurrentCandidateReviewAcceptEXPT8"
    assert graph["2"]["inputs"]["source"] == ["1", 7]
    audit = json.loads((CANDIDATE / "audit.json").read_text(encoding="utf-8"))
    assert audit["all_outputs_valid"] is True
    info = builder.base.load_live_info()
    expected, _ = builder.base.selected_frontend_schema(graph, info)
    name = audit["name"]
    assert json.loads((CANDIDATE / (name + ".api.json")).read_text(encoding="utf-8")) == expected
    _, frontend, _ = builder.build_candidate(graph, info)
    saved = json.loads((CANDIDATE / (name + ".json")).read_text(encoding="utf-8"))
    assert {key: value for key, value in saved.items() if key != "id"} == {
        key: value for key, value in frontend.items() if key != "id"}


def test_saved_compose_graph_gates_original_composer_on_two_source_bound_segments():
    graph = composer.compose_graph()
    assert len(graph) == 3
    assert graph["1"]["class_type"] == "MiniMaxH3FastH3V2CurrentAcceptedChainVerifyEXPT8"
    assert graph["2"]["class_type"] == "MiniMaxH3LongVideoComposeAcceptedT8"
    assert graph["2"]["inputs"]["chain_id"] == ["1", 0]
    assert graph["2"]["inputs"]["audio_seam_policy"] == "cosine_bridge"
    audit = json.loads((COMPOSE_CANDIDATE / "audit.json").read_text(encoding="utf-8"))
    assert audit["all_outputs_valid"] is True
    info = composer.base.load_live_info()
    expected, _ = composer.base.selected_frontend_schema(graph, info)
    name = audit["name"]
    assert json.loads((COMPOSE_CANDIDATE / (name + ".api.json")).read_text(encoding="utf-8")) == expected
    _, frontend, _ = composer.build_candidate(graph, info)
    saved = json.loads((COMPOSE_CANDIDATE / (name + ".json")).read_text(encoding="utf-8"))
    assert {key: value for key, value in saved.items() if key != "id"} == {
        key: value for key, value in frontend.items() if key != "id"}
