"""CPU contracts for SHA-pinned Chunked v2/v3/v4 complete/cold probe."""
import json

import pytest

from tools import run_modular_chunked_v234_full_parity_gpu as pair


@pytest.mark.parametrize("variant", ("v2", "v3", "v4"))
def test_complete_graph_has_live_low_high_and_exact_pinned_sources(variant):
    spec = pair._spec(variant)
    graph = pair.build_graph(variant, "full")
    assert "312" in graph and graph["312"]["class_type"] == "SamplerCustomAdvanced"
    assert spec["load"] not in graph
    assert graph["200"]["inputs"]["source"] == [str(300 + int(spec["save"])), 4]
    assert graph["250"]["inputs"]["av_latent"] == [spec["audit"], 0]
    assert graph[spec["video"]]["inputs"]["filename_prefix"].endswith(f"/{variant}/full")
    for node in graph.values():
        for value in node["inputs"].values():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                assert value[0] in graph
    assert all(spec["adapter"]._sha(path) == sha for path, sha in spec["hashes"].items())


@pytest.mark.parametrize("variant", ("v2", "v3", "v4"))
def test_cold_graph_has_only_high_and_explicit_low_receipt(variant):
    receipt = {"path": "first.h3latent.safetensors", "manifest_json": "{}",
               "file_sha256": "a" * 64}
    spec = pair._spec(variant)
    graph = pair.build_graph(variant, "cold", receipt=receipt)
    assert "12" not in graph and "312" not in graph
    assert graph[spec["load"]]["inputs"]["checkpoint_path"] == receipt["path"]
    assert graph[spec["load"]]["inputs"]["expected_manifest_json"] == "{}"
    assert graph[spec["load"]]["inputs"]["expected_file_sha256"] == "a" * 64
    assert graph["250"]["inputs"]["av_latent"] == [spec["audit"], 0]
    assert graph[spec["video"]]["inputs"]["filename_prefix"].endswith(f"/{variant}/cold")


def test_composed_effect_gate_requires_relay_and_eav_actual_calls(monkeypatch):
    report = {"status": "observed_report_only", "completed_forwards": 4,
              "planned_forwards": 4, "selector_calls": 200,
              "relay_attention_calls": 200, "relay_required": True,
              "clock_match": True, "quality_accepted": False,
              "cache_reuse_authorized": False}
    monkeypatch.setattr(pair.capture, "_phase_text", lambda *_: json.dumps(report))
    phase = {"events": [{"type": "progress", "node": "26"}] * 4}
    assert pair._effect(phase, "26", "203", 4)
    report["relay_attention_calls"] = 0
    assert not pair._effect(phase, "26", "203", 4)


def test_unrecognized_variant_and_kind_fail_closed():
    with pytest.raises(ValueError, match="v2, v3 or v4"):
        pair.build_graph("v5", "full")
    with pytest.raises(ValueError, match="full or cold"):
        pair.build_graph("v2", "resume")
