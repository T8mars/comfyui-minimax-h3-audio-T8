"""Real editable topology, native schema, and all output branches validated."""
import asyncio
from copy import deepcopy

import pytest
from legacy_schema_review import assert_reviewed_legacy_schemas

from tools import build_modular_progressive_workflow as builder
from test_modular_workflows import ancestors


@pytest.mark.parametrize("task", ["t2va", "i2va"])
@pytest.mark.parametrize("variant", builder.VARIANTS)
def test_explicit_graphs_no_hidden_sampling_or_high_ancestor_of_low(task, variant):
    graph = builder.split_graph(variant, task)
    assert not any("Loop" in item["class_type"] or item["class_type"] == "MiniMaxH3ProgressiveSamplerEXPT8"
                   for item in graph.values())
    for key in graph:
        assert key not in ancestors(graph, key)
    if variant == "load_high":
        assert {item["class_type"] for item in graph.values()} == {
            "MiniMaxH3ProgressiveHighLoadEXPT8", "VAELoader", "MiniMaxH3AVDecodeT8", "CreateVideo", "SaveVideo", "PreviewAny"}
        return
    assert graph["10" if variant != "resume_high" else "92"]["inputs"]["sampler_name"] == "euler"
    assert graph["23"]["inputs"]["size_mode"] == "target_dimensions"
    assert graph["23"]["inputs"]["target_width"] == ["20", 1]
    assert graph["23"]["inputs"]["target_height"] == ["20", 2]
    assert graph["24"]["inputs"]["width"] == ["23", 1]
    assert graph["24"]["inputs"]["height"] == ["23", 2]
    assert graph["27"]["inputs"]["noise_seed"] == graph["29"]["inputs"]["seed"] + 1
    if variant == "resume_high":
        assert not {"1", "9", "10", "11", "12", "13", "26", "50"} & set(graph)
        assert graph["20"]["inputs"]["low_boundary"] == ["60", 0]
        assert graph["25"]["inputs"]["low_boundary"] == ["60", 0]
    else:
        low = ancestors(graph, "13")
        assert not {"22", "92", "20", "23", "24", "25", "27", "28", "29", "51"} & low
        assert {"90", "1", "9", "11", "26", "12"} <= low
        assert graph["26"]["inputs"]["high_source"] == ["90", 0]
        assert graph["26"]["inputs"]["low_evaluations"] == 10
        assert graph["11"]["inputs"]["noise_seed"] == graph["29"]["inputs"]["seed"]
    if task == "i2va":
        assert graph["24"]["inputs"]["first_frame"] == ["91", 0]
    assert {"92", "24", "27"} <= ancestors(graph, "29")


@pytest.fixture(scope="module")
def live_info():
    return builder.load_live_info()


@pytest.mark.parametrize("task", ["t2va", "i2va"])
@pytest.mark.parametrize("variant", builder.VARIANTS)
def test_real_core_validates_every_output_and_serialized_edge(live_info, task, variant):
    import execution
    graph, workflow, audit = builder.build_candidate(variant, task, live_info)
    result = asyncio.run(execution.validate_prompt("modular-progressive-test", deepcopy(graph), None))
    expected = {key for key, item in graph.items() if live_info[item["class_type"]].get("output_node")}
    assert result[0] and not result[3], result
    assert set(result[2]) == expected
    assert audit["nodes"] == len(graph)
    assert len([node for node in workflow["nodes"] if node["type"] == "MarkdownNote"]) == 1


def test_live_registration_appends_exact_public_progressive_nodes(live_info):
    import json
    from h3_audio_t8_pkg.modular_sampling.progressive_nodes import NODES
    previous = json.loads((builder.ROOT / "artifacts/development/modular-sampling-m3-progressive-public-20260923/before-registration.json").read_text(encoding="utf-8"))
    expected = [node.__name__ for node in NODES]
    assert len(expected) == 10 and len(set(expected)) == 10
    assert_reviewed_legacy_schemas(previous["nodes"], live_info)
    assert all(name in live_info for name in expected)


@pytest.mark.parametrize("variant,task", [("whole_loop", "t2va"), ("minimal", "ref2va")])
def test_unsupported_graph_not_misrepresented_as_native_contract(variant, task):
    with pytest.raises(ValueError):
        builder.split_graph(variant, task)
