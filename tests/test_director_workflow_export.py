"""Director split export stays editable without changing its queue path."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from pathlib import Path

import pytest

from h3_audio_t8_pkg.director_project import ProjectStore, new_project
from h3_audio_t8_pkg.director_workflow_export import export_director_split_workflow


def _registry(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3]))
    import nodes
    from comfy_extras.nodes_custom_sampler import BasicGuider, RandomNoise, SamplerCustomAdvanced
    import h3_audio_t8_pkg

    result = dict(nodes.NODE_CLASS_MAPPINGS)
    result.update({kind.__name__: kind for kind in
                   (BasicGuider, RandomNoise, SamplerCustomAdvanced)})
    result.update({cls.define_schema().node_id: cls
                   for cls in asyncio.run(h3_audio_t8_pkg.comfy_entrypoint().get_node_list())})
    return result


@pytest.mark.parametrize("resume", [False, True])
def test_director_export_is_native_separate_and_preserves_stage_effects(
        tmp_path, monkeypatch, resume):
    from h3_audio_t8_pkg import director_generation, director_hyperflow
    from h3_audio_t8_pkg.modular_sampling import hyperflow_storage

    monkeypatch.setattr(director_generation, "_pick", lambda _folder, names, _label: names[0])
    monkeypatch.setattr(director_generation, "_optional_turbo_lora", lambda: None)
    monkeypatch.setattr(director_generation, "resolve_ffmpeg", lambda: "ffmpeg")
    weight = tmp_path / "hf.safetensors"
    weight.write_bytes(b"header-only-export-test")
    monkeypatch.setattr(director_hyperflow, "_resolve", lambda _selection: weight)
    monkeypatch.setattr(director_hyperflow.folder_paths, "get_output_directory", lambda: str(tmp_path))
    digest = "a" * 64
    monkeypatch.setattr(hyperflow_storage, "fingerprint", lambda *_args: digest)
    project = new_project()
    project["doc"]["shots"][0]["simplePrompt"] = "A continuous scene."
    project["doc"]["d3"] = {"prompt_relay": {"enabled": True, "execution_mode": "apply_exp"}}
    project["doc"]["sampling"] = {
        "mode": "hyperflow", "variant": "continuous4plus4separate",
        "hyperflow_file": "hyperflow/hf.safetensors", "output_mp": 0.4,
        "stage_checkpoint": ({"mode": "resume_tail", "artifact_path": "frozen/head.safetensors",
                              "artifact_sha256": digest} if resume else {"mode": "save"}),
        "stage_eav": {"head": {"mode": "report_only", "tau": 2.0},
                      "tail": {"mode": "apply_exp", "tau": 4.0}},
        "stage_relay": {
            "head": {"mode": "custom", "global_prompt": "HEAD scene"},
            "tail": {"mode": "custom", "global_prompt": "TAIL scene"},
        },
    }
    original = deepcopy(project)
    registry = _registry(monkeypatch)
    if resume:
        import nodes

        for kind, cls in registry.items():
            monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, kind, cls)
    exported = export_director_split_workflow(
        project, project["current"], ProjectStore(tmp_path / "user", tmp_path / "input"),
        registry=None if resume else registry)
    assert project == original
    assert exported["schema"] == "t8.director.split-workflow-export.v1"
    api = exported["api_snapshot"]
    workflow = exported["workflow"]
    types = [node["class_type"] for node in api.values()]
    assert len(workflow["nodes"]) == len(api)
    assert len(workflow["links"]) == sum(
        isinstance(value, list) and len(value) == 2 and str(value[0]) in api
        for node in api.values() for value in node["inputs"].values())
    assert types.count("MiniMaxH3HyperFlowTailStageEXPT8") == 1
    assert types.count("MiniMaxH3HyperFlowHeadStageEXPT8") == (0 if resume else 1)
    assert types.count("MiniMaxH3HyperFlowHeadLoadEXPT8") == (1 if resume else 0)
    assert "MiniMaxH3HyperFlowSplitT8Advanced" not in types
    assert "MiniMaxH3DualClockSamplerT8" not in types
    assert types.count("MiniMaxH3StageEAVApplyEXPT8") == (1 if resume else 2)
    assert types.count("MiniMaxH3PromptRelayPlanT8Advanced") == (1 if resume else 2)
    assert {node["inputs"]["global_prompt"] for node in api.values()
            if node["class_type"] == "MiniMaxH3PromptRelayPlanT8Advanced"} == (
                {"TAIL scene"} if resume else {"HEAD scene", "TAIL scene"})
    assert workflow["extra"]["t8_director_split"]["recipe"] == exported["recipe"]


def test_director_editable_export_does_not_label_legacy_combined_route_as_split(
        tmp_path, monkeypatch):
    from h3_audio_t8_pkg import director_generation, director_hyperflow

    monkeypatch.setattr(director_generation, "_pick", lambda _folder, names, _label: names[0])
    monkeypatch.setattr(director_generation, "_optional_turbo_lora", lambda: None)
    monkeypatch.setattr(director_generation, "resolve_ffmpeg", lambda: "ffmpeg")
    weight = tmp_path / "hf.safetensors"
    weight.write_bytes(b"header-only-export-test")
    monkeypatch.setattr(director_hyperflow, "_resolve", lambda _selection: weight)
    project = new_project()
    project["doc"]["shots"][0]["simplePrompt"] = "A continuous scene."
    project["doc"]["sampling"] = {"mode": "hyperflow", "variant": "continuous4plus4",
                                  "hyperflow_file": "hyperflow/hf.safetensors", "output_mp": 0.4}
    with pytest.raises(ValueError, match="仅支持导出"):
        export_director_split_workflow(
            project, project["current"], ProjectStore(tmp_path / "user", tmp_path / "input"))
