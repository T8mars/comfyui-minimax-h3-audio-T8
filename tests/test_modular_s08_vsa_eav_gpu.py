"""S08 sparse/EAV private probe shape and current CPU Core schema guards."""

import asyncio
import json

from tools import run_modular_s08_vsa_eav_gpu as probe


def test_sparse_eav_variant_keeps_two_real_stage_branches_without_relay():
    original = json.loads(probe.base.CANDIDATE.read_text(encoding="utf8"))
    graph, source_sha = probe.build_probe_graph()
    assert source_sha == probe.base.shared._sha256_file(probe.base.CANDIDATE)
    assert original == json.loads(probe.base.CANDIDATE.read_text(encoding="utf8"))
    assert len(graph) == len(original) - 2
    assert {graph[node]["class_type"] for node in ("9", "24")} == {"MiniMaxH3AudioConditioningT8"}
    assert not any("PromptRelay" in node["class_type"] for node in graph.values())
    assert graph["10"]["inputs"]["model"] == ["1", 0]
    assert graph["26"]["inputs"]["model"] == ["22", 0]
    assert graph["9"]["inputs"]["length"] == graph["24"]["inputs"]["length"] == 22
    assert graph["12"]["inputs"]["conditioning"] == ["9", 0]
    assert graph["25"]["inputs"]["positive"] == ["24", 0]
    assert graph["13"]["inputs"]["stage_context"] == ["10", 3]
    assert graph["29"]["inputs"]["stage_context"] == ["26", 3]
    for stage, config in (("10", "41"), ("26", "42")):
        assert graph[stage]["inputs"]["profile"] == "trained_vsa_exp"
        assert graph[config]["inputs"]["mode"] == "apply_exp"
        assert graph[config]["inputs"]["tau"] == 4.0
    for node in graph.values():
        for value in node["inputs"].values():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                assert value[0] in graph


def test_sparse_effect_receipt_requires_real_gain_and_producer_coverage():
    value = {"status": "observed_apply_exp", "completed_forwards": 4, "planned_forwards": 4,
             "sparse_producer_calls": 200, "relay_required": False,
             "relay_attention_calls": 0, "clock_match": True,
             "v2_dispatch": {"actual_vsa_dispatched": True, "counts": {"vsa": 200},
                             "dense_reasons": {}}, "feta": {"g_max": 1.2}}
    assert all(probe._effect_checks(value).values())
    value["v2_dispatch"]["actual_vsa_dispatched"] = False
    assert not probe._effect_checks(value)["native_sparse_dispatched"]
    value["v2_dispatch"]["actual_vsa_dispatched"] = True
    value["v2_dispatch"]["dense_reasons"] = {"fallback": 1}
    assert not probe._effect_checks(value)["native_sparse_dispatched"]
    value["v2_dispatch"]["dense_reasons"] = {}
    value["feta"]["g_max"] = 1.0
    assert not probe._effect_checks(value)["gain_above_identity"]


def test_sparse_eav_variant_validates_with_current_cpu_core(monkeypatch):
    monkeypatch.syspath_prepend(str(probe.base.PROJECT.parents[1]))
    import h3_audio_t8_pkg
    import execution
    import nodes
    from comfy_extras import nodes_custom_sampler, nodes_preview_any, nodes_video

    graph, _ = probe.build_probe_graph()
    classes = asyncio.run(h3_audio_t8_pkg.comfy_entrypoint().get_node_list())
    for cls in classes:
        monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, cls.define_schema().node_id, cls)
    for cls in (nodes_custom_sampler.RandomNoise, nodes_custom_sampler.BasicGuider,
                nodes_custom_sampler.SamplerCustomAdvanced,
                nodes_video.CreateVideo, nodes_video.SaveVideo):
        monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, cls.define_schema().node_id, cls)
    for name, cls in nodes_preview_any.NODE_CLASS_MAPPINGS.items():
        monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, name, cls)
    valid = asyncio.run(execution.validate_prompt("s08-vsa-eav-static", graph, None))
    assert valid[0], valid
    assert {"16", "45", "46", "50", "51"} <= set(valid[2]), valid
    assert not valid[3], valid
