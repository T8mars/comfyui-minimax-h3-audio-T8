"""Formal namespace, old registry preservation, native CPU graph and Relay baseline."""

import asyncio
import importlib.util
import json
from pathlib import Path
import sys

import h3_audio_t8_pkg
from h3_audio_t8_pkg import nodes_director, nodes_h16_chunked_pass2, prompt_relay_advanced as relay
from h3_audio_t8_pkg.nodes_fiveview_exp import FIVE_VIEW_EXP_NODE_CLASSES
from h3_audio_t8_pkg.nodes_hyperflow_advanced import HYPERFLOW_ADVANCED_NODE_CLASSES
from h3_audio_t8_pkg.hyperflow_long_video_exp.nodes import MiniMaxH3HyperFlowLongVideoEXPT8
from h3_audio_t8_pkg.hyperflow_long_video_exp.single8_node import MiniMaxH3HyperFlowSingle8LongVideoEXPT8
from h3_audio_t8_pkg.modular_sampling import node_classes as modular_node_classes
from h3_audio_t8_pkg.director_project import ProjectStore, new_project
from h3_audio_t8_pkg.director_routes import export_preflight_workflow

ROOT = Path(__file__).resolve().parents[1]


def backup_module(name, filename):
    spec = importlib.util.spec_from_file_location(
        "h3_audio_t8_pkg." + name,
        ROOT / "artifacts/director-d1-20260919/backups" / filename,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_old_354_registry_class_order_and_schemas_are_unchanged():
    old = backup_module("_director_before_nodes", "nodes.py")
    original = asyncio.run(old.comfy_entrypoint().get_node_list())
    current = asyncio.run(h3_audio_t8_pkg.comfy_entrypoint().get_node_list())
    assert len(original) == 354
    root_additions = [
        *h3_audio_t8_pkg._hyper_vae_2x_node_classes,
        *h3_audio_t8_pkg._audio_refine_effect_node_classes,
        *h3_audio_t8_pkg._ltx_rgb_source_node_classes,
        *h3_audio_t8_pkg._serial_video_io_node_classes,
        *h3_audio_t8_pkg._ltx_effect_node_classes,
        *h3_audio_t8_pkg._ltx_relay_node_classes,
        *h3_audio_t8_pkg._veda_sparse_node_classes,
        *h3_audio_t8_pkg._veda_heuristic_node_classes,
        *h3_audio_t8_pkg._prepared_ltx_effect_node_classes,
        *h3_audio_t8_pkg._prepared_ltx_relay_cache_node_classes,
        *h3_audio_t8_pkg._semantic_bridge_extra_node_classes,
        *h3_audio_t8_pkg._ltx_load_policy_node_classes,
        *h3_audio_t8_pkg._face_source_node_classes,
    ]
    assert len(root_additions) == 28
    assert len(current) == 581
    assert [c.define_schema().node_id for c in original] == [
        c.define_schema().node_id for c in current[:354]
    ]
    current_ids = [c.define_schema().node_id for c in current]
    assert len(current_ids) == len(set(current_ids))
    for old_class, current_class in zip(original, current):
        assert old_class.INPUT_TYPES() == current_class.INPUT_TYPES()
        assert old_class.RETURN_TYPES == current_class.RETURN_TYPES
        assert old_class.RETURN_NAMES == current_class.RETURN_NAMES
    assert current[354] is nodes_director.MiniMaxH3DirectorProjectT8
    assert current[355] is nodes_h16_chunked_pass2.DeciiaChunkedPass2Sampler
    assert current[356:] == [*FIVE_VIEW_EXP_NODE_CLASSES, *HYPERFLOW_ADVANCED_NODE_CLASSES,
                             MiniMaxH3HyperFlowLongVideoEXPT8,
                             MiniMaxH3HyperFlowSingle8LongVideoEXPT8,
                             *modular_node_classes(), *root_additions]
    assert h3_audio_t8_pkg.WEB_DIRECTORY == "./web"


def test_real_native_Core_validates_export_and_real_D1_node_executes_without_queue(
    tmp_path, monkeypatch
):
    store = ProjectStore(tmp_path / "user", tmp_path / "input")
    project = new_project()
    project["doc"]["shots"][0]["simplePrompt"] = "中文原文\r\n第二行  🙂"
    project = store.save(project, 0)
    graph = export_preflight_workflow(project, project["current"])
    # Core's own namespace must win over historical root/h3_t8 nodes.py copies.
    previous_paths = list(sys.path)
    try:
        sys.path.insert(0, str(ROOT.parents[1]))
        import nodes as native_nodes
        import execution

        assert Path(native_nodes.__file__).resolve() == ROOT.parents[1] / "nodes.py"
        monkeypatch.setitem(
            native_nodes.NODE_CLASS_MAPPINGS,
            "MiniMaxH3DirectorProjectT8",
            nodes_director.MiniMaxH3DirectorProjectT8,
        )
        validation = asyncio.run(
            execution.validate_prompt(
                "director-d1-cpu-only", graph["api_snapshot"], None
            )
        )
        assert validation[0], validation
    finally:
        sys.path[:] = previous_paths
    monkeypatch.setattr(nodes_director, "get_store", lambda: store)
    actual = nodes_director.MiniMaxH3DirectorProjectT8.execute(
        json.dumps(project), project["current"]
    ).result
    assert actual[0] == project["doc"]["shots"][0]["simplePrompt"]
    assert actual[1:4] == (800, 448, 107)
    assert json.loads(actual[-1])["gpu_queued"] is False


def test_H18_formal_ordinary_prompt_plan_and_token_inputs_are_exact_before_after():
    before = backup_module("_director_before_relay", "prompt_relay_advanced.py")
    path = (
        ROOT
        / "examples/workflows/34-semantic-bridge/2026-09-17_H3_SemanticBridge_DualIndependent_8s_Advanced.json"
    )
    workflow = json.loads(path.read_text(encoding="utf-8"))
    plan_node = next(
        n
        for n in workflow["nodes"]
        if n["type"] == "MiniMaxH3PromptRelayPlanT8Advanced"
    )
    values = plan_node["widgets_values"]
    old = before.build_prompt_relay_plan(*values)
    current = relay.build_prompt_relay_plan(*values)
    assert old == current
    assert old[0]["compiled_prompt"].encode() == current[0]["compiled_prompt"].encode()
