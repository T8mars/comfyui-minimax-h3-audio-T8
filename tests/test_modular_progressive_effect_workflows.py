"""Independent effect graphs with real Core output validation; no inference."""
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
def test_graph_has_independent_effects_and_all_live_outputs(info, task, kind, scope):
    import execution
    graph, workflow, audit = builder.build_candidate(task, kind, scope, info)
    for key in graph:
        assert key not in ancestors(graph, key)
    assert not {"22", "24", "25", "27", "28", "29", "92", "110", "112", "113", "114"} & ancestors(graph, "13")
    for phase, group, sampler in (("low", 100, "13"), ("high", 110, "29")):
        active = scope in (phase, "both")
        parents = ancestors(graph, sampler)
        assert (str(group) in parents) == (active and kind in ("relay", "combined"))
        assert (str(group + 2) in parents) == (active and kind in ("eav", "combined"))
        if active and kind in ("relay", "combined"):
            stage = "12" if phase == "low" else "28"
            assert graph[sampler]["inputs"]["positive"] == [stage, 1]
            assert graph[sampler]["inputs"]["negative"] == [stage, 2]
    result = asyncio.run(execution.validate_prompt("test-progressive-external", deepcopy(graph), None))
    expected = {key for key, node in graph.items() if info[node["class_type"]].get("output_node")}
    assert result[0] and not result[3], result
    assert set(result[2]) == expected
    assert audit["nodes"] == len(graph)
    assert any(node["type"] == "MarkdownNote" for node in workflow["nodes"])


@pytest.mark.parametrize("kind,scope", [("none", "both"), ("eav", "all")])
def test_unsupported_effect_graph_is_not_silently_substituted(kind, scope):
    with pytest.raises(ValueError):
        builder.split_graph(kind=kind, scope=scope)
