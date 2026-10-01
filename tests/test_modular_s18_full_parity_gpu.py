"""CPU-only contracts for the private S18 uninterrupted/cold real-weight probe."""
from argparse import Namespace
import json
from pathlib import Path

from tools import run_modular_s18_full_parity_gpu as pair
from tools import diagnose_modular_s18_decode as decode_probe


def test_s18_full_graph_keeps_three_explicit_segments_and_original_assets():
    graph = pair.build_graph("full")
    assert all(key in graph for key in ("12", "25", "28", "31", "40", "41", "55"))
    assert graph["20"]["inputs"]["first_pass_latent"] == ["12", 1]
    assert graph["28"]["inputs"]["previous_result"] == ["41", 1]
    assert graph["31"]["inputs"]["previous_result"] == ["28", 1]
    assert graph["55"]["inputs"]["segment_result"] == ["31", 1]
    assert graph["14"]["inputs"]["temporal_chunk_frames"] == 34
    assert graph["14"]["inputs"]["temporal_overlap_frames"] == 17
    assert graph["14"]["inputs"]["target_width"] == 256
    assert graph["19"]["inputs"]["filename_prefix"].endswith("/full")
    assert all(value[0] in graph for node in graph.values() for value in node["inputs"].values()
               if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str))
    assert all(pair.s18._sha(path) == digest for path, digest in pair.s18.HASHES.items())


def test_s18_cold_graph_uses_two_explicit_receipts_and_no_old_sampling():
    receipts = {"native": {"path": "first.h3latent.safetensors",
                           "sha256": "a" * 64, "manifest_json": "{}"},
                "segment": {"path": "segment/manifest.json", "sha256": "b" * 64}}
    graph = pair.build_graph("cold", receipts=receipts)
    assert "12" not in graph and "25" not in graph
    assert graph["40"]["inputs"]["expected_file_sha256"] == "a" * 64
    assert graph["41"]["inputs"]["artifact_sha256"] == "b" * 64
    assert graph["55"]["inputs"]["segment_result"] == ["31", 1]
    assert graph["19"]["inputs"]["filename_prefix"].endswith("/cold")


def test_s18_pair_preflight_requires_two_free_private_ports_and_resources(tmp_path, monkeypatch):
    args = Namespace(comfy_root=tmp_path, port=8877, min_free_vram_mib=12000,
                     min_free_ram_mib=95000)
    monkeypatch.setattr(pair.shared, "gpu_memory_mib", lambda: {"available": True, "free_mib": 14000})
    monkeypatch.setattr(pair.native_gpu, "_free_physical_mib", lambda: 100000)
    monkeypatch.setattr(pair.shared, "port_is_listening", lambda _host, port: port == 8878)
    missing = pair.preflight(args)
    assert not missing["ready"]
    assert not missing["checks"]["cold_port_free"]
    for asset in missing["assets"].values():
        path = Path(asset)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    monkeypatch.setattr(pair.shared, "port_is_listening", lambda *_: False)
    assert pair.preflight(args)["ready"]
    args.with_eav = True
    ready_eav = pair.preflight(args)
    assert ready_eav["ready"]
    assert ready_eav["candidate_sha256"]["relay_eav"] == pair.EAV_CANDIDATE_SHA256


def test_s18_pair_progress_filters_vhs_encoding_and_rejects_earlier_cold_stage():
    phase = {"events": [{"type": "progress", "node": key} for key in
                        (["28"] * 4 + ["31"] * 4 + ["19"] * 3)]}
    assert pair._progress(phase) == ["28"] * 4 + ["31"] * 4
    phase["events"].append({"type": "progress", "node": "12"})
    assert pair._progress(phase) != ["28"] * 4 + ["31"] * 4


def test_s18_decode_probe_can_rerender_final_receipt_without_any_sampling():
    original = pair.build_graph("cold")
    receipt = {"path": "final/manifest.json", "sha256": "c" * 64}
    graph = decode_probe.build_graph(original, receipt, with_media=True, kind="full")
    assert "19" in graph and "301" in graph
    assert all(key not in graph for key in ("12", "25", "28", "31", "41"))
    assert graph["55"]["inputs"]["artifact_sha256"] == "c" * 64
    assert graph["18"]["inputs"]["av_latent"] == ["55", 0]
    assert graph["19"]["inputs"]["filename_prefix"].endswith("/full")


def test_s18_relay_eav_full_graph_has_three_joint_audits_and_exact_model_pairing():
    graph = pair.build_graph("full", with_eav=True)
    assert pair.s18._sha(pair.EAV_CANDIDATE) == pair.EAV_CANDIDATE_SHA256
    assert graph["60"]["inputs"]["mode"] == "report_only"
    assert graph["60"]["inputs"]["g_hard_limit"] == 3.0
    for sampler, apply, audit, preview, project in (
            ("25", "61", "64", "205", "34"),
            ("28", "62", "65", "206", "36"),
            ("31", "63", "66", "207", "38")):
        assert graph[sampler]["inputs"]["model"] == [apply, 0]
        assert graph[sampler]["inputs"]["positive"] == [project, 1]
        assert graph[apply]["inputs"]["model"] == [project, 0]
        assert graph[apply]["inputs"]["eav_config"] == ["60", 0]
        assert graph[audit]["inputs"]["segment_result"] == [sampler, 1]
        assert graph[audit]["inputs"]["runtime"] == [apply, 1]
        assert graph[preview]["inputs"]["source"] == [audit, 1]
    assert all(node not in graph for node in ("35", "37", "39"))


def test_s18_relay_eav_cold_graph_has_only_later_effects_and_samplers():
    graph = pair.build_graph("cold", with_eav=True)
    assert all(node not in graph for node in ("12", "25", "35", "37", "39", "61", "64"))
    assert graph["28"]["inputs"]["model"] == ["62", 0]
    assert graph["31"]["inputs"]["model"] == ["63", 0]
    assert graph["202"]["inputs"]["source"] == ["65", 1]
    assert graph["203"]["inputs"]["source"] == ["66", 1]


def test_s18_relay_eav_audit_requires_all_real_calls_and_report_only(monkeypatch):
    def report(index):
        return {"chunked_segment_index": index, "status": "observed_report_only",
                "config": {"mode": "report_only"}, "completed_forwards": 4,
                "planned_forwards": 4, "selector_calls": 200,
                "relay_attention_calls": 200, "relay_required": True,
                "clock_match": True, "quality_accepted": False}

    reports = [report(index) for index in (0, 1, 2)]
    monkeypatch.setattr(pair.capture, "_phase_text", lambda _phase, node: json.dumps(
        reports[{"205": 0, "206": 1, "207": 2}[node]]))
    assert pair._eav({}, ("205", "206", "207"), (0, 1, 2))
    reports[2]["relay_attention_calls"] = 0
    assert not pair._eav({}, ("205", "206", "207"), (0, 1, 2))
