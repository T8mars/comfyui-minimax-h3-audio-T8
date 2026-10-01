"""All Avatar output branches validate; original recording, stages and dependencies stay explicit."""
import asyncio
from copy import deepcopy

import pytest

from tools import build_modular_avatar_workflow as builder
from test_modular_workflows import ancestors


@pytest.fixture(scope="module")
def info():
    return builder.load_live_info()


@pytest.mark.parametrize("task", ["t2va", "i2va"])
@pytest.mark.parametrize("kind,scope", builder.COMBINATIONS)
@pytest.mark.parametrize("variant", builder.base.VARIANTS)
def test_avatar_candidate_all_outputs_and_true_recovery_graph(info, task, kind, scope, variant):
    import execution
    graph, workflow, audit = builder.build_candidate(task, kind, scope, variant, info)
    for key in graph:
        assert key not in ancestors(graph, key)
    assert graph["15"]["inputs"]["audio"] == ["134", 1]
    assert graph["134"]["inputs"]["original_recording"] == ["131", 0]
    assert graph["14"]["class_type"] == "VAEDecode"
    if "13" in graph:
        assert graph["13"]["class_type"] == "MiniMaxH3AvatarLowStageEXPT8"
        assert graph["26"]["inputs"]["input_mode"] == "initialized_av_exp"
        assert graph["26"]["inputs"]["high_source"] == ["133", 0]
        assert not {"22", "24", "25", "28", "29", "92", "110", "112", "113"} & ancestors(graph, "13")
    if variant in ("resume_high", "load_high"):
        assert not {"1", "9", "10", "11", "12", "13", "26", "50", "100", "102", "103", "104", "105"} & set(graph)
    if variant == "resume_high":
        assert graph["25"]["inputs"]["low_boundary"] == ["60", 0]
        assert {"132", "133", "29"} <= set(graph)
    elif variant == "load_high":
        assert not {"1", "6", "8", "22", "23", "24", "25", "27", "28", "29", "90", "92", "132", "133"} & set(graph)
        assert graph["134"]["inputs"]["high_result"] == ["61", 1]
    if "25" in graph:
        assert graph["25"]["class_type"] == "MiniMaxH3AvatarHighHandoffEXPT8"
        assert "high_source" not in graph["25"]["inputs"]
        assert graph["25"]["inputs"]["avatar_source"] == ["133", 1]
    validation = asyncio.run(execution.validate_prompt("avatar-candidate", deepcopy(graph), None))
    expected = {key for key, item in graph.items() if info[item["class_type"]].get("output_node")}
    assert validation[0] and not validation[3] and set(validation[2]) == expected, validation
    assert audit["nodes"] == len(graph)
    assert any(node["type"] == "MarkdownNote" for node in workflow["nodes"])
