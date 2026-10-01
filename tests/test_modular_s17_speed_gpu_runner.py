"""Probe-only S17 SPEED checks; these tests never start a GPU Core."""

import asyncio
from argparse import Namespace
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest
from safetensors.torch import save_file

from tools import run_modular_s17_speed_gpu as speed_gpu
from tools import run_modular_s17_speed_resume_gpu as speed_resume


def test_saved_speed_pair_preserves_dct_and_both_external_effect_chains():
    graph, digest = speed_gpu.build_probe_graph(width=256, height=128, frames=22)
    assert len(digest) == 64
    assert graph["4"]["inputs"]["width"] == 256
    assert graph["4"]["inputs"]["height"] == 128
    assert {graph[key]["inputs"]["length"] for key in ("11", "23", "16", "28")} == {22}
    assert graph["27"]["inputs"]["completed_stage"] == ["21", 0]
    assert graph["27"]["inputs"]["next_stage"] == ["24", 5]
    assert graph["14"]["inputs"]["model"] == ["19", 0]
    assert graph["26"]["inputs"]["model"] == ["31", 0]
    assert graph["85"]["inputs"]["stage_result"] == ["26", 1]
    assert graph["86"]["inputs"]["source"] == ["85", 1]
    assert graph["82"]["inputs"]["filename_prefix"].startswith(speed_gpu.OUTPUT)


def test_s17_progress_and_handoff_reject_reordered_or_missing_stage():
    events = ([{"type": "executing", "node": key} for key in
               ("14", "21", "20", "24", "27", "26", "85", "32", "82")] +
              [{"type": "progress", "node": key} for key in ["14"] * 14 + ["26"] * 6])
    phase = {"terminal": {"type": "execution_success"}, "events": events}
    assert all(speed_gpu.stage_checks(phase).values())
    incomplete = deepcopy(phase)
    incomplete["events"].pop()
    assert not speed_gpu.stage_checks(incomplete)["exact_planned_low14_high6_progress"]
    wrong_order = deepcopy(phase)
    wrong_order["events"][4], wrong_order["events"][5] = (
        wrong_order["events"][5], wrong_order["events"][4])
    assert not speed_gpu.stage_checks(wrong_order)["dct_handoff_then_independent_high"]


def test_s17_saved_receipts_prove_exact_stages_when_core_progress_is_absent():
    events = [{"type": "executing", "node": key} for key in
              ("14", "21", "20", "24", "27", "26", "85", "32", "82")]
    phase = {"terminal": {"type": "execution_success"}, "events": events}

    def stage(steps, source=None):
        return {"receipt": {
            "receipt_sha256": "a" * 64 if source is None else "b" * 64,
            "callbacks": list(range(steps)), "sampler_known": True,
            "noise_provider": {} if source is None else {
                "source_receipt_sha256": source, "source_stage_index": 0},
            "effects": {"kind": "eav", "mode": "report_only",
                        "status": "observed_report_only", "aborted": False,
                        "clock_match": True, "completed_forwards": steps,
                        "planned_forwards": steps, "relay_required": True,
                        "relay_attention_calls": steps * 50,
                        "selector_calls": steps * 50}}}

    low, high = stage(14), stage(6, "a" * 64)
    assert all(speed_gpu.stage_checks(phase, low, high).values())
    high["receipt"]["callbacks"].pop()
    assert not speed_gpu.stage_checks(phase, low, high)["exact_planned_low14_high6_progress"]


def test_preflight_requires_installed_assets_and_separate_resource_gates(tmp_path, monkeypatch):
    args = Namespace(comfy_root=tmp_path, port=8871, min_free_vram_mib=12000,
                     min_free_ram_mib=95000)
    monkeypatch.setattr(speed_gpu.shared, "gpu_memory_mib",
                        lambda: {"available": True, "free_mib": 14000})
    monkeypatch.setattr(speed_gpu.native_gpu, "_free_physical_mib", lambda: 100000)
    monkeypatch.setattr(speed_gpu.shared, "port_is_listening", lambda *_: False)
    missing = speed_gpu.preflight(args, "a" * 64)
    assert not missing["ready"] and not missing["checks"]["all_assets_installed"]
    for name in missing["assets"].values():
        path = Path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    assert speed_gpu.preflight(args, "a" * 64)["ready"]
    args.min_free_ram_mib = 110000
    low_memory = speed_gpu.preflight(args, "a" * 64)
    assert not low_memory["ready"] and not low_memory["checks"]["system_free_ram_gate"]


def test_stage_artifact_requires_single_portable_exact_sha(tmp_path):
    store = tmp_path / "output/MiniMaxH3/modular_speed_stages/stage-0-test"
    store.mkdir(parents=True)
    state = store / "speed-stage.safetensors"
    request = {"plan_sha256": "p" * 64, "stage_index": 0}

    def digest(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                         ensure_ascii=False).encode()).hexdigest()

    receipt = {"schema": "t8.modular-sampling.speed-result.v1",
               "request": request, "request_sha256": digest(request),
               "verified_recipe_completion": True, "portable_identity": True,
               "execution": {"callbacks": list(range(14)), "sampler_known": True,
                             "effects": {}}, "output": None}
    receipt["receipt_sha256"] = digest(receipt)
    save_file({}, str(state), metadata={"speed_json": json.dumps({
        "schema": "t8.modular-sampling.frozen-speed-stage.v1",
        "receipt_json": json.dumps(receipt), "state": {}})})
    manifest = store / "manifest.json"
    manifest.write_text(json.dumps({
        "schema": "t8.modular-sampling.frozen-speed-stage.v1",
        "stage_index": 0, "portable_identity": True,
        "state_file": state.name, "state_bytes": state.stat().st_size,
        "state_sha256": speed_gpu.shared._sha256_file(state).lower(),
        "receipt_sha256": receipt["receipt_sha256"],
        "plan_sha256": request["plan_sha256"],
    }), encoding="utf-8")
    found = speed_gpu.speed_artifact(tmp_path, 0)
    assert found["path"] == "stage-0-test/manifest.json"
    assert found["receipt"]["callbacks"] == list(range(14))
    state.write_bytes(b"changed")
    with pytest.raises(ValueError, match="size/SHA"):
        speed_gpu.speed_artifact(tmp_path, 0)


def test_small_saved_speed_probe_graph_validates_in_current_cpu_core(monkeypatch):
    from tools.build_modular_fast_h3_v2_workflow import load_live_info
    from comfy_extras.nodes_preview_any import PreviewAny

    load_live_info()
    monkeypatch.syspath_prepend(str(speed_gpu.PROJECT.parents[1]))
    import execution
    import nodes
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "PreviewAny", PreviewAny)
    graph, _ = speed_gpu.build_probe_graph(width=256, height=128, frames=22)
    valid, _errors, outputs, diagnostics = asyncio.run(
        execution.validate_prompt("s17-speed-real-small", graph, None))
    assert valid and not diagnostics
    assert "82" in outputs and "86" in outputs


def test_small_saved_speed_cold_high_graph_validates_without_low_branch(monkeypatch):
    from tools.build_modular_fast_h3_v2_workflow import load_live_info
    from comfy_extras.nodes_preview_any import PreviewAny

    load_live_info()
    monkeypatch.syspath_prepend(str(speed_gpu.PROJECT.parents[1]))
    import execution
    import nodes
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "PreviewAny", PreviewAny)
    graph, digest = speed_resume.build_resume_graph(
        {"path": "stage-0-test/manifest.json", "sha256": "a" * 64},
        width=256, height=128, frames=22)
    assert len(digest) == 64
    assert not any(key in graph for key in ("10", "11", "12", "14", "21"))
    assert graph["5"]["inputs"]["artifact_sha256"] == "a" * 64
    assert graph["27"]["inputs"]["completed_stage"] == ["5", 0]
    valid, _errors, outputs, diagnostics = asyncio.run(
        execution.validate_prompt("s17-speed-cold-small", graph, None))
    assert valid and not diagnostics
    assert "82" in outputs and "86" in outputs
