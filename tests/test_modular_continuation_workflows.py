"""Every output validates; recovery graphs really remove earlier sampling bodies."""
import asyncio
from copy import deepcopy

import pytest

from tools import build_modular_continuation_workflow as builder
from test_modular_workflows import ancestors


@pytest.fixture(scope="module")
def info():
    return builder.base.load_live_info()


@pytest.mark.parametrize("frames", [22, 39])
@pytest.mark.parametrize("kind,scope", builder.COMBINATIONS)
@pytest.mark.parametrize("variant", builder.base.VARIANTS)
def test_all_outputs_and_separate_conditions_recovery(info, frames, kind, scope, variant):
    import execution
    graph, workflow, audit = builder.build_candidate(frames, kind, scope, variant, info)
    for key in graph:
        assert key not in ancestors(graph, key)
    assert graph["200"]["inputs"]["context_frames"] == frames
    if "13" in graph:
        assert graph["13"]["class_type"] == "MiniMaxH3ContinuationLowStageEXPT8"
        assert not {"22", "24", "25", "29", "92", "110", "111", "112", "113", "116"} & ancestors(graph, "13")
        assert graph["9"]["inputs"]["phase"] == "low"
    if variant in ("resume_high", "load_high"):
        assert not {"1", "9", "10", "11", "12", "13", "26", "50", "100", "101", "102", "103", "104", "105", "106"} & set(graph)
    if variant == "load_high":
        assert not {"1", "6", "22", "23", "24", "25", "27", "29", "90", "92", "201"} & set(graph)
        assert graph["203"]["inputs"]["high_result"] == ["61", 1]
    else:
        assert graph["25"]["inputs"]["prepared_phase"] == ["24", 0]
        assert graph["24"]["inputs"]["phase"] == "high"
    if variant == "resume_high":
        assert graph["25"]["inputs"]["low_boundary"] == ["60", 0]
    validation = asyncio.run(execution.validate_prompt("continuation-candidate", deepcopy(graph), None))
    expected = {key for key, item in graph.items() if info[item["class_type"]].get("output_node")}
    assert validation[0] and not validation[3] and set(validation[2]) == expected, validation
    assert audit["nodes"] == len(graph) and any(node["type"] == "MarkdownNote" for node in workflow["nodes"])
