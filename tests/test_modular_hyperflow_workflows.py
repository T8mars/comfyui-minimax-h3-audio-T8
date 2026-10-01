"""Public continuous stages and real-Core editable candidate graph checks."""
import asyncio
from copy import deepcopy
import json

import pytest
from legacy_schema_review import assert_reviewed_legacy_schemas

from tools import build_modular_hyperflow_workflow as builder
from test_modular_workflows import ancestors
from test_modular_hyperflow import inputs, equal
from h3_audio_t8_pkg.modular_sampling import hyperflow_nodes as nodes
from h3_audio_t8_pkg import hyperflow_two_pass_advanced as legacy


@pytest.fixture(scope="module")
def info():
    return builder.load_live_info()


@pytest.mark.parametrize("split", builder.SPLITS)
def test_candidate_every_output_serialization_and_independent_tail(info, split):
    import execution
    graph, workflow, audit = builder.build_candidate(split, info)
    result = asyncio.run(execution.validate_prompt("hyperflow-modular-candidate", deepcopy(graph), None))
    expected = {key for key, item in graph.items() if info[item["class_type"]].get("output_node")}
    assert result[0] and not result[3] and set(result[2]) == expected, result
    assert not {"24", "29", "102"} & ancestors(graph, "13")
    assert {"13", "24", "102"} <= ancestors(graph, "29")
    assert graph["101"]["inputs"]["model"] == graph["102"]["inputs"]["model"] == ["1", 0]
    assert graph["13"]["inputs"]["split_interval"] == split
    assert "noise" not in graph["29"]["inputs"]
    assert sum(item["class_type"] == "RandomNoise" for item in graph.values()) == 1
    assert not any("Upscale" in item["class_type"] or "SplitT8Advanced" in item["class_type"] for item in graph.values())
    assert all(key not in ancestors(graph, key) for key in graph)
    assert audit["nodes"] == len(graph)
    assert len(workflow["nodes"]) == len(graph) + 1


def test_public_registration_reviews_all420_schemas_without_rewriting_baseline(info):
    baseline = json.loads((builder.ROOT / "artifacts/development/modular-sampling-m3-hyperflow-20260923/before-registration.json").read_text(encoding="utf-8"))
    assert len(baseline["nodes"]) == 420
    assert_reviewed_legacy_schemas(baseline["nodes"], info)
    assert len(nodes.NODES) == 2
    assert all(node.__name__ in info for node in nodes.NODES)


@pytest.mark.parametrize("split", builder.SPLITS)
def test_actual_public_stage_execution_and_completed_result(monkeypatch, split):
    from comfy_extras.nodes_custom_sampler import RandomNoise
    low, high, source, positive = inputs(monkeypatch, distinct=True)
    expected, _ = legacy.sample_hyperflow_split(low, high, positive, source, seed=7, split_interval=split)
    noise = RandomNoise().get_noise(7)[0]
    boundary = nodes.MiniMaxH3HyperFlowHeadStageEXPT8.execute(low, source, noise, positive, [], split).result[0]
    output, report, completed = nodes.MiniMaxH3HyperFlowTailStageEXPT8.execute(boundary, high, positive, [], seed=7).result
    equal(output, expected)
    assert completed.output is output
    assert completed.verify()["sampling"] == json.loads(report)
    import torch
    with torch.inference_mode():
        output["samples"].unbind()[0].flatten()[0] += .1
    with pytest.raises(ValueError, match="changed"):
        completed.verify()
