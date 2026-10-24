"""History/path integrity only; fixture bytes are deliberately not real media."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location(
    "h21_control_media_test", Path(__file__).resolve().parents[1] / "tools/audit_h21_control_media.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


@pytest.fixture
def completed(tmp_path):
    attempt, case = "a" * 32, "single"
    graph = {"1": {"class_type": "MiniMaxH3SafeAVSaveT8Advanced", "inputs": {"crf": 18}}}
    folder = tmp_path / "attempts" / attempt / "executions" / case
    (folder / "observation").mkdir(parents=True)
    (tmp_path / "cases/single").mkdir(parents=True)
    (tmp_path / "output/own").mkdir(parents=True)
    (tmp_path / "output/own/movie.mp4").write_bytes(b"contract only, not a real movie")
    graph_path = tmp_path / "cases/single/prompt.json"
    graph_path.write_text(json.dumps(graph), encoding="utf-8")
    plan = {"schema": "t8.h21.control_probe.v1", "run_root": str(tmp_path),
            "cases": [{"name": case, "prompt": "cases/single/prompt.json"}],
            "sealed_files": {"cases/single/prompt.json": {"sha256": audit.digest(graph_path)}}}
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(plan), encoding="utf-8")
    history = {"status": {"status_str": "success", "completed": True},
               "prompt": [0, "one-actual-prompt", graph], "outputs": {
                   "1": {"images": [{"type": "output", "subfolder": "own", "filename": "movie.mp4"}]}}}
    for relative, value in {
        "history.json": history, "submitted.json": {"prompt_id": "one-actual-prompt"},
        "observation/terminal.json": {"prompt_id": "one-actual-prompt"},
    }.items():
        (folder / relative).write_text(json.dumps(value), encoding="utf-8")
    return manifest, attempt, case, folder, history


def test_completed_binding_returns_actual_movie_without_claiming_decode(completed):
    manifest, attempt, case, _, _ = completed
    _, _, _, movie, _ = audit.completed_case(manifest, attempt, case)
    assert movie.read_bytes().startswith(b"contract only")


@pytest.mark.parametrize("mutation, expected", [
    ("failed", "authoritative"), ("graph", "history graph"),
    ("prompt", "prompt IDs"), ("outside", "escaped"), ("no_output", "one actual"),
])
def test_incomplete_or_unrelated_receipts_do_not_qualify_media(completed, mutation, expected, tmp_path):
    manifest, attempt, case, folder, original = completed
    history = deepcopy(original)
    if mutation == "failed":
        history["status"]["status_str"] = "error"
    elif mutation == "graph":
        history["prompt"][2]["1"]["inputs"]["crf"] = True
    elif mutation == "prompt":
        history["prompt"][1] = "another-prompt"
    elif mutation == "outside":
        outside = tmp_path / "outside.mp4"
        outside.write_bytes(b"not owned output")
        history["outputs"]["1"]["images"][0]["subfolder"] = ".."
        history["outputs"]["1"]["images"][0]["filename"] = outside.name
    else:
        history["outputs"] = {}
    (folder / "history.json").write_text(json.dumps(history), encoding="utf-8")
    with pytest.raises(ValueError, match=expected):
        audit.completed_case(manifest, attempt, case)


def test_changed_frozen_graph_is_not_an_accepted_current_case(completed):
    manifest, attempt, case, _, _ = completed
    (manifest.parent / "cases/single/prompt.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA changed"):
        audit.completed_case(manifest, attempt, case)


@pytest.mark.parametrize("key", ["frames", "width", "height", "fps"])
@pytest.mark.parametrize("bad", [0, True])
def test_invalid_delivery_contract_fails_before_any_file_or_media_operation(tmp_path, key, bad):
    values = {"frames": 120, "width": 1024, "height": 576, "fps": 24}
    values[key] = bad
    target = tmp_path / "uncreated-audit"
    with pytest.raises(ValueError, match="positive integers"):
        audit.audit(tmp_path / "absent-manifest", "a" * 32, "case", target,
                    **values, ffmpeg=tmp_path / "unused-ffmpeg")
    assert not target.exists()
