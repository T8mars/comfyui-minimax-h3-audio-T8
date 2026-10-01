"""The M0 frozen-JSON structure audit must reject real graph damage."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

from tools.audit_modular_legacy_graph_structure import audit
from tools.audit_modular_sampling_compat import SCHEMA

ROOT = Path(__file__).resolve().parents[1]


def _graph():
    return {"nodes": [
        {"id": 1, "type": "StageLow", "inputs": [], "outputs": [{"name": "result"}]},
        {"id": 2, "type": "StageHigh", "inputs": [{"name": "result", "link": 1}],
         "outputs": []}],
        "links": [[1, 1, 0, 2, 0, "STAGE"]]}


def _frozen(tmp_path, graph):
    path = tmp_path / "examples" / "old.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(graph), encoding="utf8")
    baseline = {"schema": SCHEMA, "files": {
        "examples/old.json": hashlib.sha256(path.read_bytes()).hexdigest()}}
    return path, baseline


def test_current_frozen_corpus_has_no_broken_saved_graph_references():
    baseline = json.loads((ROOT / "artifacts/development/modular-sampling-m0-20260922/baseline.json")
                          .read_text(encoding="utf8"))
    report = audit(ROOT, baseline)
    assert report["status"] == "pass" and report["issues"] == []
    assert report["frozen_json_count"] == 382
    assert report["kinds"] == {"frontend": 306, "api": 66, "other": 10}
    assert report["frontend_nodes"] >= 4500 and report["frontend_links"] >= 5900


def test_valid_small_frontend_graph_passes(tmp_path):
    _, baseline = _frozen(tmp_path, _graph())
    report = audit(tmp_path, baseline)
    assert report["status"] == "pass"
    assert report["frontend_nodes"] == 2 and report["frontend_links"] == 1


def test_dangling_and_wrong_slot_fail_even_if_new_baseline_matches(tmp_path):
    graph = _graph()
    graph["links"][0][3] = 99
    _, baseline = _frozen(tmp_path, graph)
    assert "dangling_frontend_link" in {issue["kind"] for issue in audit(tmp_path, baseline)["issues"]}
    graph = deepcopy(_graph())
    graph["links"][0][4] = 3
    path = tmp_path / "examples" / "old.json"
    path.write_text(json.dumps(graph), encoding="utf8")
    baseline["files"]["examples/old.json"] = hashlib.sha256(path.read_bytes()).hexdigest()
    assert "invalid_frontend_slot" in {issue["kind"] for issue in audit(tmp_path, baseline)["issues"]}


def test_saved_input_pin_must_match_its_link_table_entry(tmp_path):
    graph = _graph()
    graph["nodes"][1]["inputs"][0]["link"] = 77
    _, baseline = _frozen(tmp_path, graph)
    assert "input_pin_link_mismatch" in {issue["kind"] for issue in
        audit(tmp_path, baseline)["issues"]}


def test_frozen_hash_and_path_escape_are_rejected(tmp_path):
    path, baseline = _frozen(tmp_path, _graph())
    path.write_text(path.read_text(encoding="utf8") + " ", encoding="utf8")
    assert "frozen_file_sha_changed" in {issue["kind"] for issue in audit(tmp_path, baseline)["issues"]}
    baseline["files"] = {"../outside.json": "0" * 64}
    assert "missing_or_escaping_frozen_file" in {
        issue["kind"] for issue in audit(tmp_path, baseline)["issues"]}
