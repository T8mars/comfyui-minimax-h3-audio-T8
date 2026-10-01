"""S08 dense Relay/EAV control and cold-HIGH probe safety guards."""

import asyncio
import json

from tools import run_modular_s08_dense_eav_relay_gpu as probe


def test_only_external_eav_mode_changes_between_full_control_and_applied():
    before = probe.full.CANDIDATE.read_bytes()
    control, sha = probe.build_full("report_only")
    applied, other_sha = probe.build_full("apply_exp")
    assert sha == other_sha == probe.FULL_SHA
    assert probe.full.CANDIDATE.read_bytes() == before
    assert probe._same_except_eav_mode(control, applied)
    assert all(applied[node]["inputs"]["mode"] == "apply_exp" for node in ("41", "42"))
    assert all(applied[node]["inputs"]["tau"] == 4.0 for node in ("41", "42"))
    assert {applied[node]["class_type"] for node in ("40", "47")} == {
        "MiniMaxH3PromptRelayPlanT8Advanced"}
    assert all(applied[node]["inputs"]["profile"] == "dense_compat_exp"
               for node in ("10", "26"))
    applied["42"]["inputs"]["tau"] = 1.0
    assert not probe._same_except_eav_mode(control, applied)


def test_cold_graph_has_only_independent_high_with_relay_and_eav():
    original = probe.resume.CANDIDATE.read_bytes()
    receipt = {"path": "safe/manifest.json", "sha256": "ABC"}
    graph, sha = probe.build_cold(receipt)
    assert sha == probe.RESUME_SHA
    assert probe.resume.CANDIDATE.read_bytes() == original
    assert not any(node in graph for node in ("1", "9", "10", "13", "40", "41", "45", "50"))
    assert graph["60"]["inputs"]["artifact_path"] == receipt["path"]
    assert graph["29"]["class_type"] == "MiniMaxH3StageSamplerEXPT8"
    assert graph["47"]["class_type"] == "MiniMaxH3PromptRelayPlanT8Advanced"
    assert graph["42"]["inputs"]["mode"] == "apply_exp"


def test_dense_effect_gate_requires_real_relay_and_numeric_gain():
    audit = {"status": "observed_apply_exp", "completed_forwards": 4,
             "planned_forwards": 4, "relay_required": True,
             "relay_attention_calls": 200, "sparse_producer_calls": 0,
             "clock_match": True,
             "v2_dispatch": {"profile": "dense_compat_exp", "actual_vsa_dispatched": False},
             "feta": {"active_forward_count": 4, "g_max": 1.2}}
    assert all(probe._effect_checks(audit, "apply_exp").values())
    audit["relay_attention_calls"] = 0
    assert not probe._effect_checks(audit, "apply_exp")["separate_relay_executed"]
    audit["relay_attention_calls"] = 200
    audit["feta"]["g_max"] = 1.0
    assert not probe._effect_checks(audit, "apply_exp")["gain_above_identity"]
    audit["feta"]["g_max"] = 1.2
    audit["v2_dispatch"]["actual_vsa_dispatched"] = True
    assert not probe._effect_checks(audit, "apply_exp")["dense_not_silent_sparse_fallback"]


def test_full_and_cold_graphs_validate_in_current_cpu_core(monkeypatch):
    monkeypatch.syspath_prepend(str(probe.full.PROJECT.parents[1]))
    import h3_audio_t8_pkg
    import execution
    import nodes
    from comfy_extras import nodes_custom_sampler, nodes_preview_any, nodes_video

    classes = asyncio.run(h3_audio_t8_pkg.comfy_entrypoint().get_node_list())
    for cls in classes:
        monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, cls.define_schema().node_id, cls)
    for cls in (nodes_custom_sampler.RandomNoise, nodes_custom_sampler.BasicGuider,
                nodes_custom_sampler.SamplerCustomAdvanced,
                nodes_video.CreateVideo, nodes_video.SaveVideo):
        monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, cls.define_schema().node_id, cls)
    for name, cls in nodes_preview_any.NODE_CLASS_MAPPINGS.items():
        monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, name, cls)
    graphs = [probe.build_full("report_only")[0], probe.build_full("apply_exp")[0],
              probe.build_cold({"path": "safe/manifest.json", "sha256": "ABC"})[0]]
    for graph in graphs:
        valid = asyncio.run(execution.validate_prompt("s08-dense-eav-relay-static", graph, None))
        assert valid[0], json.dumps(valid, ensure_ascii=False, default=str)
        assert "16" in valid[2]
        assert not valid[3], valid
