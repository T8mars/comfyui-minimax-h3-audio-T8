"""Probe integrity tests; synthetic media here is not trained generation evidence."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from tools import run_modular_audio_tail_effect_gpu as probe


def receipt():
    return {"checkpoint": {"relative_path": "source.h3latent.safetensors", "file_sha256": "A" * 64,
            "embedded_manifest": {"frame_count": 22, "checkpoint_id": "audio_refine_firstpass"}}}


@pytest.mark.parametrize("route", probe.previous.ROUTES)
def test_saved_probe_preserves_all_inputs_except_explicit_receipts_effect_widgets_and_output(route):
    frontend, original, graph, sources = probe.build_graph(route, receipt())
    pins = probe.previous.ROUTES[route]
    allowed = {
        pins["load"]: {"checkpoint_path", "expected_manifest_json", "expected_file_sha256"},
        pins["guard"]: {"expected_video_frame_count"}, pins["save"]: {"filename_prefix"},
        probe._one(graph, probe.builder.RELAY_PLAN): {"local_prompts", "timing_mode", "time_ranges"},
        probe._one(graph, probe.builder.RELAY_ROUTE): {"query_route"},
        probe._one(graph, probe.builder.RELAY_COND): {"execution_mode", "query_chunk_rows"},
        probe._one(graph, probe.builder.CONFIG): {"mode", "tau", "start_video_progress", "end_video_progress"},
    }
    for key, node in original.items():
        changed = {k for k in node["inputs"].keys() | graph[key]["inputs"].keys()
                   if node["inputs"].get(k) != graph[key]["inputs"].get(k)}
        assert changed <= allowed.get(key, set())
    assert len(sources) == 3 and frontend["nodes"]
    assert probe._one(graph, "SamplerCustomAdvanced") == pins["sampler"]
    assert graph[pins["sampler"]] == original[pins["sampler"]]
    config = graph[probe._one(graph, probe.builder.CONFIG)]["inputs"]
    assert (config["tau"], config["g_hard_limit"], config["mode"]) == (.25, 1.5, "apply_exp")
    gate = graph[probe._one(graph, "MiniMaxH3AudioRefineQualityGateT8Advanced")]
    assert gate["inputs"]["accept_candidate"] is False
    for label, first in (("original", 801), ("candidate", 803)):
        decoder = gate["inputs"][label + "_audio"][0]
        assert graph[str(first)]["inputs"]["images"] == [decoder, 0]
        assert graph[str(first)]["inputs"]["audio"] == [decoder, 1]
        assert graph[str(first + 1)]["inputs"]["filename_prefix"].endswith("_" + label)
    assert all(probe.previous.freeze._sha(Path(p)) == sha for p, sha in sources.items())


def _reports():
    eav = {"status": "observed_apply_exp", "completed_forwards": 4, "planned_forwards": 4,
        "clock_match": True, "relay_required": True, "relay_attention_calls": 200,
        "forward_plan": {"known": True, "forwards": [{"attention_blocks": 50}] * 4},
        "feta": {"forwards": [{"active": True, "attention_count": 50}] * 4, "g_min": 1., "g_max": 1.2},
        "config": {"g_hard_limit": 1.5},
        "quality_accepted": False, "cache_reuse_authorized": False}
    return eav, {"status": "applied_exp"}, {"external_positive": True}


def test_effect_evidence_requires_actual_all_block_calls_not_only_success():
    assert all(probe.effect_checks(*_reports()).values())


def test_non_unit_gain_qualification_is_distinct_from_observed_routing():
    eav, relay, guider = _reports()
    assert all(probe.effect_checks(eav, relay, guider, require_non_identity=True).values())
    eav["feta"]["g_max"] = 1.
    assert all(probe.effect_checks(eav, relay, guider).values())
    assert not probe.effect_checks(eav, relay, guider, require_non_identity=True)["non_identity_gain_observed"]
    for bad in (float("nan"), float("inf"), 1.51, "1.2", True):
        eav["feta"]["g_max"] = bad
        assert not probe.gain_observation(eav)["finite_within_original_hard_limit"]


@pytest.mark.parametrize("tau", [float("nan"), float("inf"), 33, -33, True])
def test_bad_probe_tau_is_rejected_before_reading_graph(tau):
    with pytest.raises(ValueError, match="tau"):
        probe.build_graph("pdd8", receipt(), eav_tau=tau)


def test_standard_tau_probe_keeps_same_hard_limit_and_public_defaults():
    _frontend, original, graph, _sources = probe.build_graph("pdd8", receipt(), eav_tau=4.)
    config = probe._one(graph, probe.builder.CONFIG)
    assert graph[config]["inputs"]["tau"] == original[config]["inputs"]["tau"] == 4.
    assert graph[config]["inputs"]["g_hard_limit"] == original[config]["inputs"]["g_hard_limit"] == 1.5


@pytest.mark.parametrize("field,value", [("status", "unverified_relay_coverage"),
    ("completed_forwards", 3), ("relay_attention_calls", 199), ("clock_match", False),
    ("relay_required", False), ("quality_accepted", True), ("cache_reuse_authorized", True),
    ("feta", {"forwards": [{"active": True, "attention_count": 49}] * 4}),
    ("forward_plan", {"known": False, "forwards": [{"attention_blocks": 50}] * 4})])
def test_effect_evidence_rejects_missing_or_misleading_coverage(field, value):
    eav, relay, guider = _reports()
    eav[field] = value
    assert not all(probe.effect_checks(eav, relay, guider).values())


def test_reject_wrong_frame_receipt_and_accepted_candidate(monkeypatch):
    frozen = receipt()
    frozen["checkpoint"]["embedded_manifest"]["frame_count"] = 124
    with pytest.raises(ValueError, match="22-frame"):
        probe.build_graph("pdd8", frozen)
    frontend, graph, sources = probe.saved_graph("pdd8")
    graph[probe._one(graph, "MiniMaxH3AudioRefineQualityGateT8Advanced")]["inputs"]["accept_candidate"] = True
    monkeypatch.setattr(probe, "saved_graph", lambda _r: (frontend, deepcopy(graph), sources))
    with pytest.raises(ValueError, match="accept candidate"):
        probe.build_graph("pdd8", receipt())


def test_snapshot_writer_never_overwrites_existing_evidence(tmp_path):
    path = tmp_path / "report.json"
    probe._write(path, {"original": True})
    with pytest.raises(FileExistsError):
        probe._write(path, {"original": False})
    assert json.loads(path.read_text()) == {"original": True}


@pytest.mark.parametrize("route,canvas", [("pdd8", 128), ("pdd4plus4", 192)])
def test_complete_media_audit_checks_both_decoded_streams_and_default_selection(tmp_path, route, canvas):
    output = tmp_path / "output/MiniMaxH3/TailEffects"
    output.mkdir(parents=True)
    for label, frequency in (("original", 440), ("candidate", 660)):
        path = output / f"{route}_{label}_00001.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-n", "-f", "lavfi", "-i",
            f"color=c=blue:s={canvas}x{canvas}:r=24:d=0.916666667", "-f", "lavfi", "-i",
            f"sine=frequency={frequency}:sample_rate=44100:duration=0.916666667", "-frames:v", "22",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(path)],
            capture_output=True, check=True, timeout=30)
    original = output / f"{route}_original_00001.mp4"
    selected = output / f"{route}_selected_00001.mp4"
    shutil.copy2(original, selected)
    report = probe.media_audit(tmp_path, route)
    assert all(report["checks"].values()) and report["quality_accepted"] is False
    shutil.copy2(output / f"{route}_candidate_00001.mp4", selected)
    assert probe.media_audit(tmp_path, route)["checks"]["default_selected_media_equals_original"] is False


def test_preflight_retains_resource_gate_and_rejects_escaped_asset(tmp_path, monkeypatch):
    monkeypatch.setattr(probe.previous, "preflight", lambda *_a: {"checks": {"free_vram_gate": False}})
    graph = {"1": {"class_type": "UNETLoader", "inputs": {"unet_name": "../../escape"}}}
    args = SimpleNamespace(comfy_root=tmp_path)
    with pytest.raises(ValueError, match="escaped"):
        probe.preflight(args, graph, tmp_path / "freeze", {"source": "A" * 64})
    graph["1"]["inputs"]["unet_name"] = "absent.safetensors"
    report = probe.preflight(args, graph, tmp_path / "freeze", {"source": "A" * 64})
    assert report["ready"] is False and report["checks"]["free_vram_gate"] is False
