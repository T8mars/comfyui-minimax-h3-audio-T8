"""Fail-closed candidate Save As semantic audit, without a browser dependency."""
from __future__ import annotations

import json

import pytest

from tools import audit_modular_candidate_browser_save as audit


def _fixture(monkeypatch, tmp_path):
    root = tmp_path / "project"
    private = root / "artifacts/development"
    original = private / "candidate/source.json"
    saved = private / "profile/user/default/workflows/QA.json"
    original.parent.mkdir(parents=True)
    saved.parent.mkdir(parents=True)
    graph = {"nodes": [{"id": 1, "type": "TestNode", "mode": 0,
                        "inputs": [], "outputs": [], "widgets_values": ["LOW"]}],
             "links": []}
    original.write_text(json.dumps(graph), encoding="utf8")
    saved_graph = json.loads(original.read_text(encoding="utf8"))
    saved_graph["nodes"][0]["widgets_values_named"] = {"text": "HIGH"}
    saved_graph["nodes"][0]["widgets_values"] = ["HIGH"]
    saved.write_text(json.dumps(saved_graph), encoding="utf8")
    (root / "web").mkdir()
    (root / "web/task_type_labels.js").write_text("fixture", encoding="utf8")
    monkeypatch.setattr(audit, "ROOT", root)
    monkeypatch.setattr(audit, "PRIVATE_ROOT", private)
    case = {"original": "artifacts/development/candidate/source.json",
            "original_sha256": audit._sha(original), "saved": "QA.json",
            "edits": [{"node_id": 1, "widget_index": 0, "before": "LOW", "after": "HIGH"}]}
    return case, original, saved, private / "profile"


def test_candidate_browser_save_requires_pinned_edit_and_source(monkeypatch, tmp_path):
    case, original, saved, profile = _fixture(monkeypatch, tmp_path)
    result = audit.audit_case(case, profile=profile, current_nodes={})
    assert result["semantic_audit"]["edits"] == ["1:0"]
    assert result["original_sha256"] == audit._sha(original)
    assert result["saved_sha256"] == audit._sha(saved)

    without_edit = {**case, "edits": []}
    with pytest.raises(ValueError, match="widgets unexpectedly"):
        audit.audit_case(without_edit, profile=profile, current_nodes={})
    wrong_before = {**case, "edits": [{**case["edits"][0], "before": "different"}]}
    with pytest.raises(ValueError, match="not pinned"):
        audit.audit_case(wrong_before, profile=profile, current_nodes={})
    wrong_sha = {**case, "original_sha256": "0" * 64}
    with pytest.raises(ValueError, match="pinned private source"):
        audit.audit_case(wrong_sha, profile=profile, current_nodes={})


def test_candidate_browser_save_rejects_unpinned_path_and_extra_widget(monkeypatch, tmp_path):
    case, _, saved, profile = _fixture(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="bare JSON filename"):
        audit.audit_case({**case, "saved": "../QA.json"}, profile=profile, current_nodes={})
    graph = json.loads(saved.read_text(encoding="utf8"))
    graph["nodes"][0]["widgets_values"] = ["HIGH", "surprise"]
    graph["nodes"][0]["widgets_values_named"]["extra"] = "surprise"
    saved.write_text(json.dumps(graph), encoding="utf8")
    with pytest.raises((ValueError, KeyError), match="TestNode|widgets unexpectedly"):
        audit.audit_case(case, profile=profile, current_nodes={})
