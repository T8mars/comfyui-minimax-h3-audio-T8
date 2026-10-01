"""Read-only tests for the saved S26 PDD real-asset freeze probe."""

from copy import deepcopy
import json

import pytest

from tools import run_modular_s26_pdd_freeze_gpu as probe


@pytest.mark.parametrize("route,save_id,conditioning_id,count", (
    ("pdd8", "31", "6", 1),
    ("pdd4plus4", "41", "7", 2),
))
def test_probe_preserves_saved_pdd_route_except_declared_fixture_values(
        route, save_id, conditioning_id, count):
    original = json.loads(probe._candidate(route).read_text(encoding="utf-8"))
    graph, digest = probe.build_probe_graph(route, width=128, height=128, frames=22)
    assert digest == probe._sha(probe._candidate(route))
    assert sum(item["class_type"] == "SamplerCustomAdvanced" for item in graph.values()) == count
    assert graph["8"] == original["8"]
    expected = deepcopy(original)
    expected[conditioning_id]["inputs"].update(width=128, height=128, length=22)
    if route == "pdd4plus4":
        expected["14"]["inputs"]["length"] = 22
    expected[save_id]["inputs"].update(
        confirm_save=True, filename_prefix=f"s26_{route}_frozen_firstpass")
    for added in ("200", "201"):
        graph.pop(added)
    assert graph == expected


def test_probe_rejects_unknown_route_and_unaligned_canvas():
    with pytest.raises(ValueError):
        probe.build_probe_graph("other", width=128, height=128, frames=22)
    with pytest.raises(ValueError):
        probe.build_probe_graph("pdd8", width=127, height=128, frames=22)


def test_preflight_fails_closed_without_resources(monkeypatch, tmp_path):
    graph, digest = probe.build_probe_graph("pdd8", width=128, height=128, frames=22)
    options = probe.argparse.Namespace(route="pdd8", comfy_root=tmp_path,
                                       port=8862, min_free_vram_mib=12000)
    monkeypatch.setattr(probe.shared, "gpu_memory_mib", lambda: {"available": False})
    monkeypatch.setattr(probe.shared, "port_is_listening", lambda *_args: True)
    result = probe.preflight(options, graph, digest)
    assert result["ready"] is False
    assert result["checks"]["all_required_assets_exist"] is False
    assert result["checks"]["isolated_port_free"] is False
    assert result["checks"]["free_vram_gate"] is False


@pytest.mark.parametrize("route,sampler_nodes,ordered", (
    ("pdd8", ["11"] * 8, ["8", "11", "31"]),
    ("pdd4plus4", ["12"] * 4 + ["19"] * 4,
     ["8", "12", "13", "16", "19", "41"]),
))
def test_stage_audit_requires_exact_native_progress_and_handoff_order(
        route, sampler_nodes, ordered):
    phase = {"terminal": {"type": "execution_success"},
             "events": ([{"type": "executing", "node": item} for item in ordered] +
                        [{"type": "progress", "node": item} for item in sampler_nodes])}
    assert all(probe.stage_execution_checks(route, phase).values())
    phase["events"][-1]["node"] = "wrong_sampler"
    assert probe.stage_execution_checks(route, phase)["exact_eight_sampler_progress_events"] is False
