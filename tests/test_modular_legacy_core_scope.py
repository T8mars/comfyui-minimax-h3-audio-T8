"""Fail-closed M0 Core-scope classification of the frozen old corpus."""

import asyncio
from copy import deepcopy
import hashlib
import json

import pytest

from tools.audit_modular_legacy_core_scope import (
    assess_core_result, audit, frozen_graphs, frontend_type_inventory,
)
from tools.audit_modular_sampling_compat import SCHEMA


class _Output:
    OUTPUT_NODE = True


class _Input:
    OUTPUT_NODE = False


def _fixture(tmp_path):
    frontend = {"nodes": [{"id": 1, "type": "MiniMaxH3Present"},
                          {"id": 2, "type": "ExternalMissing"}], "links": []}
    api = {"1": {"class_type": "Input", "inputs": {}},
           "2": {"class_type": "Output", "inputs": {"value": ["1", 0]}},
           "3": {"class_type": "Output", "inputs": {"value": ["1", 0]}}}
    data = {"kind": "fixture"}
    files = {}
    for name, value in (("frontend.json", frontend), ("api.json", api), ("data.json", data)):
        path = tmp_path / name
        path.write_text(json.dumps(value), encoding="utf8")
        files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"schema": SCHEMA, "files": files,
            "nodes": [{"id": "MiniMaxH3Present"}]}


def test_result_rejects_cores_partial_success_and_ignored_output():
    assert assess_core_result((True, None, ["2", "3"], {}), {"2", "3"})["status"] == "core_all_outputs_valid"
    result = assess_core_result((True, None, ["2"], {"3": {"errors": [
        {"type": "required_input_missing"}]}}), {"2", "3"})
    assert result["status"] == "core_invalid_or_ignored_output"
    assert result["error_kinds"] == ["required_input_missing"]
    assert assess_core_result((False, {"type": "prompt_no_outputs"}, [], {}), set())["status"] != "core_all_outputs_valid"


def test_scoped_audit_preserves_all_files_and_distinguishes_missing_external(tmp_path):
    baseline = _fixture(tmp_path)

    async def validate(_id, _graph, _partial):
        return True, None, ["2", "3"], {}

    report = asyncio.run(audit(tmp_path, baseline, {
        "MiniMaxH3Present": _Input, "Input": _Input, "Output": _Output}, validate))
    assert report["status"] == "scoped_audit_completed_not_workflow_qualification"
    assert report["counts"]["frontend"] == report["counts"]["api"] == report["counts"]["other"] == 1
    assert report["counts"]["api_all_outputs_valid"] == 1
    assert report["counts"]["frontend_needs_external_or_ui_types"] == 1
    assert report["missing_type_file_counts"] == {"ExternalMissing": 1}


def test_tampered_sha_and_escape_cannot_be_counted_as_qualified(tmp_path):
    baseline = _fixture(tmp_path)
    (tmp_path / "api.json").write_text("{}", encoding="utf8")
    kinds = {name: kind for name, kind, _ in frozen_graphs(tmp_path, baseline)}
    assert kinds["api.json"] == "frozen_file_sha_changed"
    escaped = deepcopy(baseline)
    escaped["files"] = {"../outside.json": "0" * 64}
    assert next(frozen_graphs(tmp_path, escaped))[1] == "missing_or_escaping_frozen_file"


def test_project_missing_is_decided_by_frozen_registry_not_name_prefix(tmp_path):
    baseline = _fixture(tmp_path)
    path = tmp_path / "frontend.json"
    graph = json.loads(path.read_text(encoding="utf8"))
    graph["nodes"][1]["type"] = "MiniMaxH3External"
    path.write_text(json.dumps(graph), encoding="utf8")
    baseline["files"]["frontend.json"] = hashlib.sha256(path.read_bytes()).hexdigest()

    async def validate(_id, _graph, _partial):
        return True, None, ["2", "3"], {}

    registry = {"MiniMaxH3Present": _Input, "Input": _Input, "Output": _Output}
    report = asyncio.run(audit(tmp_path, baseline, registry, validate))
    assert next(item for item in report["files"] if item["kind"] == "frontend")["status"] == "registry_scope_incomplete"
    baseline["nodes"].append({"id": "MiniMaxH3External"})
    report = asyncio.run(audit(tmp_path, baseline, registry, validate))
    assert next(item for item in report["files"] if item["kind"] == "frontend")["status"] == "missing_project_node"


def _subgraph_fixture():
    return {"nodes": [{"id": 1, "type": "outer"}], "links": [],
            "definitions": {"subgraphs": [
                {"id": "outer", "nodes": [{"id": 2, "type": "inner"}], "links": []},
                {"id": "inner", "nodes": [{"id": 3, "type": "MiniMaxH3Present"},
                                           {"id": 4, "type": "MarkdownNote"}], "links": []},
                {"id": "unused", "nodes": [{"id": 5, "type": "UnusedExternal"}], "links": []},
            ]}}


def test_bundled_subgraph_ids_are_not_external_and_unused_definitions_are_checked():
    graph = _subgraph_fixture()
    original = deepcopy(graph)
    types, definitions, count = frontend_type_inventory(graph)
    assert types == {"MiniMaxH3Present", "MarkdownNote", "UnusedExternal"}
    assert definitions == ["inner", "outer", "unused"]
    assert count == 4
    assert graph == original


def test_null_definition_is_empty_not_permission_to_ignore_an_unknown_type():
    graph = {"nodes": [{"id": 1, "type": "unbundled-uuid"}], "links": [],
             "definitions": None}
    assert frontend_type_inventory(graph) == ({"unbundled-uuid"}, [], 0)


@pytest.mark.parametrize("damage", ["cycle", "duplicate", "bad_definitions", "bad_list",
                                     "bad_nodes", "bad_type", "nested_declarations"])
def test_bad_embedded_definitions_fail_closed(damage):
    graph = _subgraph_fixture()
    definitions = graph["definitions"]["subgraphs"]
    if damage == "cycle":
        definitions[1]["nodes"][0]["type"] = "outer"
    elif damage == "duplicate":
        definitions[1]["id"] = "outer"
    elif damage == "bad_definitions":
        graph["definitions"] = []
    elif damage == "bad_list":
        graph["definitions"]["subgraphs"] = {}
    elif damage == "bad_nodes":
        definitions[0]["nodes"] = None
    elif damage == "bad_type":
        definitions[1]["nodes"][0]["type"] = None
    else:
        definitions[0]["definitions"] = {"subgraphs": [definitions[1]]}
    with pytest.raises(ValueError):
        frontend_type_inventory(graph)


@pytest.mark.parametrize("mode,status", [
    ("available", "embedded_types_available_not_browser_validated"),
    ("external_missing", "registry_scope_incomplete"),
    ("project_missing", "missing_project_node"),
    ("collision", "invalid_frontend_definitions"),
    ("cycle", "invalid_frontend_definitions"),
])
def test_embedded_inventory_never_certifies_missing_nodes_or_execution(tmp_path, mode, status):
    graph = _subgraph_fixture()
    if mode == "cycle":
        graph["definitions"]["subgraphs"][1]["nodes"][0]["type"] = "outer"
    path = tmp_path / "embedded.json"
    path.write_text(json.dumps(graph), encoding="utf-8")
    original = path.read_bytes()
    baseline = {"schema": SCHEMA, "files": {path.name: hashlib.sha256(original).hexdigest()},
                "nodes": [{"id": "MiniMaxH3Present"}]}
    registry = {"MiniMaxH3Present": _Input, "UnusedExternal": _Input}
    if mode == "external_missing":
        del registry["UnusedExternal"]
    elif mode == "project_missing":
        del registry["MiniMaxH3Present"]
    elif mode == "collision":
        registry["outer"] = _Input

    async def no_validate(*_):
        pytest.fail("Type inventory must not submit embedded graphs to API validation")

    result = asyncio.run(audit(tmp_path, baseline, registry, no_validate))
    assert result["files"][0]["status"] == status
    assert "outer" not in result["missing_type_file_counts"]
    assert result["counts"]["frontend"] == 1
    assert result["counts"]["frontend_all_types_registered"] == 0
    assert path.read_bytes() == original
