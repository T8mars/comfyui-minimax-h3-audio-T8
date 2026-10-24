"""Actual callback evidence is distinct from declared grids or DiT forwards."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from h3_audio_t8_pkg import hyperflow_two_pass_advanced as split_sampling
from h3_audio_t8_pkg import nodes_hyperflow_advanced as hf_nodes
from h3_audio_t8_pkg.modular_sampling import hyperflow_nodes, nodes as stage_nodes
from test_modular_hyperflow import inputs


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "h21_hf_report_observer_test", ROOT / "tools/h21_control_probe.py"
)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)

REPORT_CLASSES = [
    (hf_nodes.MiniMaxH3HyperFlowLoaderT8Advanced, 1),
    (hf_nodes.MiniMaxH3HyperFlowPlanT8Advanced, 1),
    (hf_nodes.MiniMaxH3HyperFlowSamplerT8Advanced, 3),
    (hf_nodes.MiniMaxH3HyperFlowSplitT8Advanced, 1),
    (hf_nodes.MiniMaxH3HyperFlowTailPlanT8Advanced, 1),
    (hf_nodes.MiniMaxH3HyperFlowHeadPlanT8Advanced, 1),
    (hf_nodes.MiniMaxH3HyperFlowCoarseSamplerT8Advanced, 3),
    (hf_nodes.MiniMaxH3HyperFlowRefineSamplerT8Advanced, 3),
    (hf_nodes.MiniMaxH3HyperFlowPartialRefineSamplerT8Advanced, 3),
    (hyperflow_nodes.MiniMaxH3HyperFlowHeadStageEXPT8, 1),
    (hyperflow_nodes.MiniMaxH3HyperFlowTailStageEXPT8, 1),
    (stage_nodes.MiniMaxH3StageSamplerEXPT8, 3),
]


@pytest.mark.parametrize("split", [1, 4, 7])
def test_actual_split_callbacks_are_recorded_without_changing_av_or_user_callback(monkeypatch, split):
    low, high, latent, positive = inputs(monkeypatch, distinct=True)
    external = []

    def callback(index, prediction, state, total):
        for value in (prediction, state):
            parts = value.unbind()
            assert len(parts) == 2
            assert all(isinstance(part, torch.Tensor) for part in parts)
            assert [part.shape for part in parts] == [part.shape for part in latent["samples"].unbind()]
        external.append((index, total))

    observed, text = split_sampling.sample_hyperflow_split(
        low, high, positive, latent, seed=7, split_interval=split, callback=callback
    )
    reference, without_external = split_sampling.sample_hyperflow_split(
        low, high, positive, latent, seed=7, split_interval=split
    )
    assert external == [(index, 8) for index in range(8)]
    for left, right in zip(observed["samples"].unbind(), reference["samples"].unbind()):
        assert torch.equal(left, right)
    evidence = json.loads(text)["observed_sampler_callbacks"]
    assert evidence == json.loads(without_external)["observed_sampler_callbacks"]
    assert evidence["LOW"] == [
        {"local_index": index, "absolute_interval": index, "local_total": split}
        for index in range(split)
    ]
    assert evidence["HIGH"] == [
        {"local_index": index, "absolute_interval": split + index, "local_total": 8 - split}
        for index in range(8 - split)
    ]
    report = json.loads(text)
    assert report["stage_nfe"] == [split, 8 - split] and report["total_nfe"] == 8
    assert report["nfe_evidence"]["legacy_nfe_fields"] == "declared_trained_intervals"
    assert report["nfe_evidence"]["actual_DiT_forward_count_qualified"] is False


@pytest.mark.parametrize("node_class,port", REPORT_CLASSES, ids=lambda value: getattr(value, "__name__", str(value)))
def test_frozen_observer_consumes_actual_report_port_without_touching_producer(node_class, port):
    schema = node_class.GET_SCHEMA()
    assert schema.outputs[port].display_name == "report_json"
    assert schema.outputs[port].io_type == "STRING"
    graph = {
        "producer": {"class_type": schema.node_id, "inputs": {"seed": 99}},
        "sink": {"class_type": "PreviewAny", "inputs": {"source": ["producer", 0]}},
        "orphan": {"class_type": schema.node_id, "inputs": {"seed": 101}},
    }
    original = deepcopy(graph)
    observed, contract = probe.append_report_observers(graph, ["sink"])
    assert graph == original and all(observed[key] == node for key, node in original.items())
    assert contract["reports"] == [{
        "producer": "producer", "class_type": schema.node_id,
        "output": port, "preview": "h21_report_producer",
    }]
    assert observed["h21_report_producer"]["inputs"]["source"] == ["producer", port]
    assert contract["NFE_sigmas_noise_audio_routes_changed"] is False
