"""Saved-canvas S08 private execution copy stays source-identical."""

import asyncio
import json

from tools import run_modular_s08_saved_canvas_gpu as probe
from tools import run_modular_s08_saved_canvas_cold_gpu as cold


def test_authored_graph_changes_only_private_output_prefix():
    original_bytes = probe.base.CANDIDATE.read_bytes()
    original = json.loads(original_bytes)
    graph, source_sha = probe.build_graph()
    assert source_sha == probe.dense.FULL_SHA
    assert probe.base.CANDIDATE.read_bytes() == original_bytes
    assert graph["16"]["inputs"]["filename_prefix"] == probe.PREFIX
    graph["16"]["inputs"]["filename_prefix"] = original["16"]["inputs"]["filename_prefix"]
    assert graph == original


def test_cold_graph_changes_only_receipt_and_private_output_prefix():
    original_bytes = cold.resume.CANDIDATE.read_bytes()
    receipt = {"path": "safe/manifest.json", "sha256": "ABC"}
    graph, source_sha = cold.build_graph(receipt)
    assert source_sha == probe.dense.RESUME_SHA
    assert cold.resume.CANDIDATE.read_bytes() == original_bytes
    assert "13" not in graph and "1" not in graph
    assert graph["47"]["inputs"]["length"] == 73
    assert graph["26"]["inputs"]["min_tokens"] == 12288
    original = json.loads(original_bytes)
    graph["60"]["inputs"].update(artifact_path=original["60"]["inputs"]["artifact_path"],
                                 artifact_sha256=original["60"]["inputs"]["artifact_sha256"])
    graph["16"]["inputs"]["filename_prefix"] = original["16"]["inputs"]["filename_prefix"]
    assert graph == original


def test_authored_graph_current_cpu_core_validates_all_outputs(monkeypatch):
    monkeypatch.syspath_prepend(str(probe.base.PROJECT.parents[1]))
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
    for graph, expected in (
            (probe.build_graph()[0], {"16", "45", "46", "50", "51"}),
            (cold.build_graph({"path": "safe/manifest.json", "sha256": "ABC"})[0],
             {"16", "46", "51"})):
        valid = asyncio.run(execution.validate_prompt("s08-authored-canvas-static", graph, None))
        assert valid[0], valid
        assert expected <= set(valid[2]), valid
        assert not valid[3], valid
