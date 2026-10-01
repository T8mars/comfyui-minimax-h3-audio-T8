"""Live public storage nodes, all-output graph validation and true Core reuse."""
import asyncio
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from legacy_schema_review import assert_reviewed_legacy_schemas
import torch
from comfy_api.latest import io

from h3_audio_t8_pkg.modular_sampling import hyperflow as stages, hyperflow_nodes as public
from h3_audio_t8_pkg.modular_sampling import hyperflow_storage_nodes as stored
from test_modular_hyperflow import inputs, head, equal
from test_modular_hyperflow_core_cache import TinyHyperBase, SyntheticHyperLoader, HyperResultSink
from test_modular_progressive_core_cache import TinyCondition
from test_modular_workflows import ancestors
from tools import build_modular_hyperflow_storage_workflow as builder


@pytest.fixture(scope="module")
def info():
    return builder.continuous.load_live_info()


@pytest.mark.parametrize("variant", builder.VARIANTS)
@pytest.mark.parametrize("split", builder.continuous.SPLITS)
def test_all_outputs_and_pruned_dependencies(info, variant, split):
    import execution
    graph, workflow, audit = builder.build_candidate(variant, split, info)
    validation = asyncio.run(execution.validate_prompt("hyperflow-storage", deepcopy(graph), None))
    outputs = {key for key, node in graph.items() if info[node["class_type"]].get("output_node")}
    assert validation[0] and not validation[3] and set(validation[2]) == outputs
    if variant == "resume_tail":
        assert not {"13", "9", "11", "101", "90"} & set(graph)
        assert graph["50"]["class_type"] == "MiniMaxH3HyperFlowHeadLoadEXPT8"
    elif variant == "load_completed":
        assert not {"1", "6", "9", "11", "13", "24", "29", "50", "90", "101", "102"} & set(graph)
    elif variant == "save_head":
        assert outputs == {"50"} and not {"29", "24", "102", "51"} & set(graph)
    else:
        assert not {"24", "102", "29"} & ancestors(graph, "50")
    assert audit["nodes"] == len(graph)
    assert len(workflow["nodes"]) == len(graph) + 1


def test_all_422_prior_schemas_remain_exact_except_reviewed_named_deltas(info):
    root = Path(__file__).resolve().parents[1]
    baseline = json.loads((root / "artifacts/development/modular-sampling-m3-hyperflow-storage-20260923/before-registration.json").read_text(encoding="utf-8"))
    assert len(baseline["nodes"]) == 422
    assert_reviewed_legacy_schemas(baseline["nodes"], info)


def test_four_public_nodes_really_save_load_and_detect_mutation(tmp_path, monkeypatch):
    monkeypatch.setattr(stored, "_store_root", lambda: tmp_path)
    low, high, source, positive = inputs(monkeypatch)
    boundary = head(low, source, positive)
    saved = stored.MiniMaxH3HyperFlowHeadSaveEXPT8.execute(boundary).result
    loaded = stored.MiniMaxH3HyperFlowHeadLoadEXPT8.execute(saved[1], saved[2]).result
    assert loaded[0].receipt_json == boundary.receipt_json
    result, _ = stages.sample_tail_result(loaded[0], high, positive, [], seed=7)
    saved_tail = stored.MiniMaxH3HyperFlowTailSaveEXPT8.execute(result).result
    loaded_tail = stored.MiniMaxH3HyperFlowTailLoadEXPT8.execute(saved_tail[1], saved_tail[2]).result
    equal(loaded_tail[0], result.output)
    for kind, cls, values in (("head", stored.MiniMaxH3HyperFlowHeadLoadEXPT8, saved),
                              ("tail", stored.MiniMaxH3HyperFlowTailLoadEXPT8, saved_tail)):
        assert cls.fingerprint_inputs(values[1], values[2]) == values[2]
        path = tmp_path / values[1]
        with path.open("ab") as stream:
            stream.write(b"corrupted")
        assert cls.fingerprint_inputs(values[1], values[2]) != values[2]
        with pytest.raises(ValueError, match="SHA"):
            cls.execute(values[1], values[2])


@pytest.mark.parametrize("kind", ["head", "tail"])
def test_actual_core_frozen_load_cache_checks_current_file(tmp_path, monkeypatch, kind):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3]))
    import execution
    import nodes
    monkeypatch.setattr(stored, "_store_root", lambda: tmp_path)
    low, high, source, positive = inputs(monkeypatch)
    boundary = head(low, source, positive)
    if kind == "head":
        saved = stored.MiniMaxH3HyperFlowHeadSaveEXPT8.execute(boundary).result
        graph = builder.split_graph("resume_tail", artifact_path=saved[1], artifact_sha256=saved[2])
        # The actual dedicated installation still happens. Only model-file and
        # text-encoder provisioning are tiny deterministic test doubles.
        base = high.clone()
        from h3_audio_t8_pkg.modular_sampling.hyperflow_identity import project
        base, _, _ = project(base)
        # Remove the synthetic original HyperFlow LoRAs only on this test-only
        # fresh provisioning branch; installation will add its own once.
        base.patches = {}
        monkeypatch.setattr(TinyHyperBase, "execute", classmethod(lambda cls: io.NodeOutput(base)))
        graph["1"] = {"class_type": "TinyHyperBase", "inputs": {}}
        graph["102"] = {"class_type": "SyntheticHyperLoader", "inputs": {"model": ["1", 0], "patch": 0.}}
        graph["24"] = {"class_type": "TinyCondition", "inputs": {"width": 32, "height": 32, "prompt": 0.}}
        graph["100"] = {"class_type": "HyperResultSink", "inputs": {"result": ["29", 2]}}
    else:
        result, _ = stages.sample_tail_result(boundary, high, positive, [], seed=7)
        saved = stored.MiniMaxH3HyperFlowTailSaveEXPT8.execute(result).result
        graph = builder.split_graph("load_completed", artifact_path=saved[1], artifact_sha256=saved[2])
        graph["100"] = {"class_type": "HyperResultSink", "inputs": {"result": ["51", 1]}}
    graph = builder.continuous.common.prune(graph, ["100"])
    for cls in (*public.NODES, *stored.NODES, TinyHyperBase, SyntheticHyperLoader, TinyCondition, HyperResultSink):
        monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, cls.__name__, cls)
    server = SimpleNamespace(client_id=None, last_node_id=None, sockets_metadata={}, send_sync=lambda *a, **kw: None)
    executor = execution.PromptExecutor(server, cache_args={"ram": 0., "ram_inactive": 0.},
                                        asset_manager=SimpleNamespace(enabled=False))
    calls = []
    original = stages._run
    def run(model, plan, *a, **kw):
        assert plan.start_interval > 0, "A frozen graph cannot execute HEAD"
        calls.append("tail")
        return original(model, plan, *a, **kw)
    monkeypatch.setattr(stages, "_run", run)
    executor.execute(deepcopy(graph), "first-frozen", execute_outputs=["100"])
    assert executor.success, executor.status_messages
    assert calls == (["tail"] if kind == "head" else [])
    calls.clear()
    executor.execute(deepcopy(graph), "same-frozen", execute_outputs=["100"])
    assert executor.success and calls == []
    path = tmp_path / saved[1]
    with path.open("ab") as stream:
        stream.write(b"changed despite same path and UI SHA")
    executor.execute(deepcopy(graph), "corrupt-frozen", execute_outputs=["100"])
    assert not executor.success and calls == []
    assert any(event == "execution_error" for event, _ in executor.status_messages)
    assert not torch.cuda.is_initialized()
