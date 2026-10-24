"""Real V3 node contracts and actual tiny Core INT8 AV, without CUDA."""

import asyncio
import json
import math

import pytest
import torch

from h3_audio_t8_pkg import nodes_int8_row_chunk as nodes
from test_h21_int8_row_chunk import actual_int8_model
from test_h21_projection_model import sample


@pytest.fixture(autouse=True)
def CPU_only():
    with torch.inference_mode():
        yield
    assert not torch.cuda.is_initialized()


def test_actual_V3_schemas_keep_zero_off_and_no_arbitrary_row_ceiling():
    apply = nodes.MiniMaxH3Int8RowChunksEXPT8.GET_SCHEMA()
    fields = {item.id: item for item in apply.inputs}
    assert apply.is_experimental and len(apply.outputs) == 2
    assert fields["rows"].min == 0 and fields["rows"].max is None
    assert fields["rows"].default == 2048
    assert fields["targets"].options == ["all", "mlp"]
    audit = nodes.MiniMaxH3Int8RowChunksAuditEXPT8.GET_SCHEMA()
    assert audit.is_output_node and audit.is_experimental
    assert math.isnan(nodes.MiniMaxH3Int8RowChunksAuditEXPT8.fingerprint_inputs())


@pytest.mark.parametrize("enabled,rows", [(False, 7), (True, 0)])
def test_disabled_or_zero_rows_preserves_original_MODEL_without_guessing_type(enabled, rows):
    original = object()
    selected, text = nodes.MiniMaxH3Int8RowChunksEXPT8.execute(
        original, enabled=enabled, rows=rows,
    ).result
    assert selected is original and json.loads(text)["status"] == "disabled"
    result = nodes.MiniMaxH3Int8RowChunksAuditEXPT8.execute(original, object())
    assert json.loads(result.result[0])["status"] == "not_selected"


def test_ordering_edge_alone_does_not_invent_execution_or_performance(tmp_path):
    selected, text = nodes.MiniMaxH3Int8RowChunksEXPT8.execute(
        actual_int8_model(tmp_path), rows=7,
    ).result
    assert json.loads(text)["status"] == "selected_not_executed"
    result = nodes.MiniMaxH3Int8RowChunksAuditEXPT8.execute(selected, {"samples": None})
    report = json.loads(result.result[0])
    assert report["completed_diffusion_calls"] == 0 and not report["calls"]
    for key in ("quality_accepted", "performance_qualified", "allocator_peak_measured",
                "portable_cache_identity"):
        assert report[key] is False


@pytest.mark.parametrize("targets", ["all", "mlp"])
def test_actual_tiny_INT8_AV_apply_sample_audit_and_original_next_request(tmp_path, targets):
    original = actual_int8_model(tmp_path)
    expected = sample(original)
    selected, _ = nodes.MiniMaxH3Int8RowChunksEXPT8.execute(
        original, rows=7, targets=targets, phase="HIGH",
    ).result
    output = sample(selected)
    result = nodes.MiniMaxH3Int8RowChunksAuditEXPT8.execute(selected, output)
    report = json.loads(result.result[0])
    assert report["phase"] == "HIGH" and report["completed_diffusion_calls"] == 2
    assert report["calls"] and report["cleanup_observed"]
    assert max(item["max_rows"] for item in report["calls"].values()) <= 7
    assert any("fused_int8" in name for name in report["calls"])
    if targets == "mlp":
        assert all("mlp" in name for name in report["calls"])
    assert not original.wrappers and not report["portable_cache_identity"]
    for actual in (output, sample(original)):
        for got, wanted in zip(actual["samples"].unbind(), expected["samples"].unbind(), strict=True):
            torch.testing.assert_close(got, wanted, rtol=0, atol=0)


def test_actual_Core_validates_apply_and_audit_graph_without_global_registration(monkeypatch):
    import execution
    import nodes as core_nodes

    class Source:
        RETURN_TYPES = ("MODEL", "LATENT")
        FUNCTION = "execute"

        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {}}

    monkeypatch.setitem(core_nodes.NODE_CLASS_MAPPINGS, "H21Int8SchemaSource", Source)
    for cls in nodes.NODES:
        monkeypatch.setitem(core_nodes.NODE_CLASS_MAPPINGS, cls.__name__, cls)
    graph = {
        "1": {"class_type": "H21Int8SchemaSource", "inputs": {}},
        "2": {"class_type": nodes.NODES[0].__name__, "inputs": {
            "model": ["1", 0], "enabled": True, "rows": 0,
            "targets": "all", "phase": "LOW",
        }},
        "3": {"class_type": nodes.NODES[1].__name__, "inputs": {
            "model": ["2", 0], "after_sampling": ["1", 1],
        }},
    }
    result = asyncio.run(execution.validate_prompt("h21-int8-row-nodes", graph, None))
    assert result[0], result
    assert result[2] == ["3"] and not result[3]
