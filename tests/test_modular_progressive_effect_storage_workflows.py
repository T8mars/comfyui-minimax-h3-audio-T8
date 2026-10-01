"""Persisted effects graphs: every output validates and LOW truly disappears."""
import asyncio
from copy import deepcopy

import pytest

from tools import build_modular_progressive_effect_workflow as builder
from test_modular_workflows import ancestors


@pytest.fixture(scope="module")
def info():
    return builder.base.load_live_info()


@pytest.mark.parametrize("task", ["t2va", "i2va"])
@pytest.mark.parametrize("kind", builder.KINDS)
@pytest.mark.parametrize("scope", builder.SCOPES)
@pytest.mark.parametrize("variant", ["save", "resume_high", "load_high"])
def test_real_effect_storage_graphs_have_no_hidden_upstream_execution(info, task, kind, scope, variant):
    import execution
    graph, workflow, audit = builder.build_candidate(task, kind, scope, info, variant)
    for key in graph:
        assert key not in ancestors(graph, key)
    if variant == "save":
        assert graph["20"]["inputs"]["low_boundary"] == ["50", 0]
        assert graph["25"]["inputs"]["low_boundary"] == ["50", 0]
        assert graph["50"]["inputs"]["low_boundary"] == ["104" if scope != "high" else "13", 0]
        assert "51" in graph
    else:
        assert not {"1", "9", "10", "11", "12", "13", "26", "50", "100", "102", "103", "104", "105"} & set(graph)
    if variant == "resume_high":
        assert graph["20"]["inputs"]["low_boundary"] == ["60", 0]
        assert graph["25"]["inputs"]["low_boundary"] == ["60", 0]
        assert "29" in graph and "51" in graph
    elif variant == "load_high":
        assert not {"22", "23", "24", "25", "27", "28", "29", "92", "110", "112", "113"} & set(graph)
        assert "61" in graph
    validation = asyncio.run(execution.validate_prompt("progressive-effect-storage", deepcopy(graph), None))
    expected = {key for key, item in graph.items() if info[item["class_type"]].get("output_node")}
    assert validation[0] and not validation[3] and set(validation[2]) == expected, validation
    assert audit["nodes"] == len(graph)
    assert any(node["type"] == "MarkdownNote" for node in workflow["nodes"])
