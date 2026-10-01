"""Guard the private S08 real VSA/EAV cold-HIGH execution copy without weights."""

import asyncio
import json

from tools import run_modular_s08_vsa_eav_resume_gpu as probe


RECEIPT = {"path": "FastH3V2/LOW-test/manifest.json", "sha256": "A" * 64}


def test_sparse_cold_high_keeps_exact_saved_low_and_removes_unsupported_relay():
    original = json.loads(probe.resume.CANDIDATE.read_text(encoding="utf8"))
    graph, source_sha = probe.build_probe_graph(RECEIPT)
    assert source_sha == probe.resume.shared._sha256_file(probe.resume.CANDIDATE)
    assert original == json.loads(probe.resume.CANDIDATE.read_text(encoding="utf8"))
    assert len(graph) == len(original) - 1
    assert not any("PromptRelay" in node["class_type"] for node in graph.values())
    assert not any(node in graph for node in ("1", "9", "10", "11", "12", "13", "50"))
    assert graph["60"]["inputs"]["artifact_path"] == RECEIPT["path"]
    assert graph["60"]["inputs"]["artifact_sha256"] == RECEIPT["sha256"]
    assert graph["22"]["inputs"]["completed_stage"] == ["60", 3]
    assert graph["26"]["inputs"]["model"] == ["22", 0]
    assert graph["26"]["inputs"]["profile"] == "trained_vsa_exp"
    assert graph["26"]["inputs"]["min_tokens"] == 0
    assert graph["42"]["inputs"]["mode"] == "apply_exp"
    assert graph["42"]["inputs"]["tau"] == 4.0
    assert graph["24"]["class_type"] == "MiniMaxH3AudioConditioningT8"
    assert graph["25"]["inputs"]["positive"] == ["24", 0]
    assert graph["29"]["inputs"]["stage_context"] == ["26", 3]
    assert graph["51"]["inputs"]["stage_result"] == ["29", 2]
    for node in graph.values():
        for value in node["inputs"].values():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                assert value[0] in graph


def test_sparse_cold_high_graph_validates_every_current_core_output(monkeypatch):
    monkeypatch.syspath_prepend(str(probe.resume.PROJECT.parents[1]))
    import h3_audio_t8_pkg
    import execution
    import nodes
    from comfy_extras import nodes_custom_sampler, nodes_preview_any, nodes_video

    graph, _ = probe.build_probe_graph(RECEIPT)
    classes = asyncio.run(h3_audio_t8_pkg.comfy_entrypoint().get_node_list())
    for cls in classes:
        monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, cls.define_schema().node_id, cls)
    for cls in (nodes_custom_sampler.RandomNoise, nodes_custom_sampler.BasicGuider,
                nodes_custom_sampler.SamplerCustomAdvanced,
                nodes_video.CreateVideo, nodes_video.SaveVideo):
        monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, cls.define_schema().node_id, cls)
    for name, cls in nodes_preview_any.NODE_CLASS_MAPPINGS.items():
        monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, name, cls)
    valid = asyncio.run(execution.validate_prompt("s08-vsa-eav-resume-static", graph, None))
    assert valid[0], valid
    assert {"16", "31", "32", "46", "51", "61"} <= set(valid[2]), valid
    assert not valid[3], valid


def test_sparse_cold_high_rejects_unqualified_full_status(tmp_path):
    (tmp_path / "report.json").write_text(json.dumps({"status": "fail"}), encoding="utf8")
    try:
        probe.resume.verified_low(tmp_path, expected_status=probe.FULL_STATUS)
    except ValueError as error:
        assert "mechanical gate" in str(error)
    else:
        raise AssertionError("An unqualified LOW was accepted")
