"""Public effect nodes, full authored topology and real Core cache execution."""
import asyncio
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from legacy_schema_review import assert_reviewed_legacy_schemas
import torch
from comfy_api.latest import io

from h3_audio_t8_pkg.modular_sampling import hyperflow as stages, hyperflow_nodes as public
from h3_audio_t8_pkg.modular_sampling import hyperflow_effect_nodes as effects, nodes as common
from test_modular_hyperflow import inputs, equal
from test_modular_hyperflow_effects import case
from test_modular_hyperflow_core_cache import TinyHyperBase, SyntheticHyperLoader, HyperResultSink
from test_modular_progressive_core_cache import TinyCondition
from test_modular_workflows import ancestors
from tools import build_modular_hyperflow_effect_workflow as builder


@pytest.fixture(scope="module")
def info():
    return builder.base.load_live_info()


@pytest.mark.parametrize("split", builder.base.SPLITS)
@pytest.mark.parametrize("kind", builder.KINDS)
@pytest.mark.parametrize("scope", builder.SCOPES)
@pytest.mark.parametrize("variant", builder.VARIANTS)
def test_every_output_and_dependency_valid(info, split, kind, scope, variant):
    import execution
    graph, workflow, audit = builder.build_candidate(split, kind, scope, variant, info)
    validation = asyncio.run(execution.validate_prompt("hyperflow-effects", deepcopy(graph), None))
    expected = {key for key, node in graph.items() if info[node["class_type"]].get("output_node")}
    assert validation[0] and not validation[3] and set(validation[2]) == expected
    assert audit["nodes"] == len(graph) and len(workflow["nodes"]) == len(graph) + 1
    if variant == "resume_tail":
        assert not {"9", "11", "13", "90", "101", "201", "202", "203", "204", "205"} & set(graph)
        assert graph["50"]["class_type"] == "MiniMaxH3HyperFlowHeadLoadEXPT8"
    else:
        assert not {"24", "29", "102", "211", "212", "213", "214", "215"} & ancestors(graph, "13")
    for phase, group in (("head", 200), ("tail", 210)):
        if phase == "head" and variant == "resume_tail":
            continue
        active = scope in (phase, "both")
        assert (str(group+2) in graph) == (active and kind in ("eav", "combined"))
        assert (str(group+5) in graph) == (active and kind in ("relay", "combined"))
    assert not any("Upscale" in node["class_type"] for node in graph.values())
    assert graph["29"]["inputs"].get("noise") is None


def test_all_426_prior_schemas_unchanged_except_reviewed_named_deltas(info):
    root = Path(__file__).resolve().parents[1]
    baseline = json.loads((root / "artifacts/development/modular-sampling-m3-hyperflow-effects-20260923/before-registration.json").read_text(encoding="utf-8"))
    assert len(baseline["nodes"]) == 426
    assert_reviewed_legacy_schemas(baseline["nodes"], info)


def test_four_public_nodes_execute_real_effect_stages(monkeypatch):
    from comfy_extras.nodes_custom_sampler import Noise_RandomNoise
    from test_progressive_relay import paired
    from h3_audio_t8_pkg.modular_sampling import hyperflow_effects as implementation
    low, high, source, positive = inputs(monkeypatch)
    import comfy.nested_tensor
    source = {"samples": comfy.nested_tensor.NestedTensor((torch.zeros(1, 24, 2, 4, 8), torch.zeros(1, 32, 2, 8)))}
    low, positive, _ = paired(low)
    high, hp, _ = paired(high, query_route="video_only_paper")
    bound = effects.MiniMaxH3HyperFlowHeadEffectsBindEXPT8.execute(low, source, positive, []).result
    config = common.MiniMaxH3StageEAVConfigEXPT8.execute("apply_exp", .2, 0., 1., 32, 3.).result[0]
    lm = common.MiniMaxH3StageEAVApplyEXPT8.execute(bound[0], bound[4], bound[3], bound[5], config).result[0]
    boundary = public.MiniMaxH3HyperFlowHeadStageEXPT8.execute(lm, source, Noise_RandomNoise(7), bound[1], bound[2]).result[0]
    audited = effects.MiniMaxH3HyperFlowHeadEffectsAuditEXPT8.execute(boundary).result
    assert audited[0] is boundary and json.loads(audited[1])["composition_verified"]
    bound = effects.MiniMaxH3HyperFlowTailEffectsBindEXPT8.execute(boundary, high, hp, []).result
    config = common.MiniMaxH3StageEAVConfigEXPT8.execute("report_only", .6, .2, .9, 32, 3.).result[0]
    hm = common.MiniMaxH3StageEAVApplyEXPT8.execute(bound[0], bound[4], bound[3], bound[5], config).result[0]
    output = public.MiniMaxH3HyperFlowTailStageEXPT8.execute(boundary, hm, bound[1], bound[2], seed=7).result
    audited = effects.MiniMaxH3HyperFlowTailEffectsAuditEXPT8.execute(output[2]).result
    equal(audited[1], output[0])
    assert audited[0] is output[2] and json.loads(audited[2])["composition_verified"]
    assert implementation.audit(boundary, "head")["relay"]["binding"]["query_route"] == "joint_av_exp"
    assert json.loads(audited[2])["relay"]["binding"]["query_route"] == "video_only_paper"


class TinyHyperRelay(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, inputs=[io.Model.Input("model"),
            io.Combo.Input("route", options=["joint_av_exp", "video_only_paper"]),
            io.Float.Input("prompt", default=0.)], outputs=[io.Model.Output(), io.Conditioning.Output()])

    @classmethod
    def execute(cls, model, route, prompt):
        from test_progressive_relay import paired
        selected, positive, _ = paired(model, query_route=route)
        positive[0][0].add_(prompt)
        return io.NodeOutput(selected, positive)


@pytest.mark.parametrize("change", ["tail_config", "head_config", "tail_disable", "tail_prompt", "tail_cancel",
                                    "tail_relay", "head_relay", "tail_cancel_relay"])
def test_actual_core_effect_edits_only_rerun_dependent_phases(monkeypatch, change):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3]))
    import execution
    import nodes
    import comfy.model_management
    from comfy_extras.nodes_custom_sampler import RandomNoise
    from comfy_extras.nodes_minimax_h3 import EmptyMiniMaxH3LatentAV
    from test_hyperflow_advanced import _tiny_native_model
    from test_progressive_sampling_runtime import tiny_model
    _, diffusion = _tiny_native_model(monkeypatch)
    base = tiny_model()
    base.model.diffusion_model = diffusion
    monkeypatch.setattr(TinyHyperBase, "execute", classmethod(lambda cls: io.NodeOutput(base)))
    for cls in (*public.NODES, *effects.NODES, TinyHyperBase, SyntheticHyperLoader, TinyCondition, TinyHyperRelay, HyperResultSink,
                common.MiniMaxH3StageEAVApplyEXPT8, common.MiniMaxH3StageEAVConfigEXPT8,
                RandomNoise, EmptyMiniMaxH3LatentAV):
        monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, cls.__name__, cls)
    with_relay = "relay" in change
    graph = builder.split_graph(kind="combined" if with_relay else "eav")
    graph["1"] = {"class_type": "TinyHyperBase", "inputs": {}}
    for key in ("101", "102"):
        graph[key] = {"class_type": "SyntheticHyperLoader", "inputs": {"model": ["1", 0], "patch": 0.}}
    for key in ("9", "24"):
        graph[key] = {"class_type": "TinyCondition", "inputs": {"width": 64, "height": 64, "prompt": 0.}}
    graph["90"]["inputs"] = {"width": 64, "height": 64, "length": 5}
    if with_relay:
        graph["90"]["inputs"]["width"] = 128
        for key, loader in (("9", "101"), ("24", "102")):
            graph[key] = {"class_type": "TinyHyperRelay", "inputs": {
                "model": [loader, 0], "route": "joint_av_exp", "prompt": 0.}}
    # Tiny spatial attention has a different FETA gain distribution from the
    # trained graph. Select explicit tiny-safe controls; retain production's
    # hard-limit refusal rather than disabling/clamping it for cache tests.
    for key in ("202", "212"):
        graph[key]["inputs"].update(tau=.2, g_hard_limit=3.)
    graph["100"] = {"class_type": "HyperResultSink", "inputs": {"result": ["214", 0]}}
    graph = builder.base.common.prune(graph, ["100"])
    server = SimpleNamespace(client_id=None, last_node_id=None, sockets_metadata={}, send_sync=lambda *a, **kw: None)
    executor = execution.PromptExecutor(server, cache_args={"ram": 0., "ram_inactive": 0.},
                                        asset_manager=SimpleNamespace(enabled=False))
    calls, receipts, cancel = [], [], {"enabled": False}
    original = stages._run
    def count(model, plan, *a, **kw):
        calls.append("tail" if plan.start_interval else "head")
        if plan.start_interval and cancel["enabled"]:
            def interrupt(*args):
                raise comfy.model_management.InterruptProcessingException()
            kw["callback"] = interrupt
        output = original(model, plan, *a, **kw)
        receipts.append(output[3]["effects"])
        return output
    monkeypatch.setattr(stages, "_run", count)
    def run(label):
        executor.execute(deepcopy(graph), label, execute_outputs=["100"])
        assert executor.success, executor.status_messages
    run("initial-effects")
    assert calls == ["head", "tail"]
    calls.clear()
    run("same-effects")
    assert not calls
    if with_relay:
        graph["9" if change == "head_relay" else "24"]["inputs"]["route"] = "video_only_paper"
    elif change == "tail_prompt":
        graph["24"]["inputs"]["prompt"] = .3
    elif change == "tail_disable":
        graph["212"]["inputs"]["mode"] = "disabled"
    else:
        graph["202" if change == "head_config" else "212"]["inputs"]["tau"] = .4
    if change in ("tail_cancel", "tail_cancel_relay"):
        cancel["enabled"] = True
        executor.execute(deepcopy(graph), "cancel-effects", execute_outputs=["100"])
        assert not executor.success and calls == ["tail"]
        calls.clear()
        cancel["enabled"] = False
    run("changed-effects")
    assert calls == (["head", "tail"] if change.startswith("head") else ["tail"])
    assert receipts[-1]["composition_verified"] is True
    assert ("eav" in receipts[-1]) == (change != "tail_disable")
    assert not torch.cuda.is_initialized()


def test_audit_rejects_swapped_typed_results(monkeypatch):
    _, high, _, _, hp, boundary, _, _ = case(monkeypatch)
    result, _ = stages.sample_tail_result(boundary, high, hp, [], seed=7)
    with pytest.raises(ValueError, match="matching completed"):
        effects.MiniMaxH3HyperFlowHeadEffectsAuditEXPT8.execute(result)
    with pytest.raises(ValueError, match="matching completed"):
        effects.MiniMaxH3HyperFlowTailEffectsAuditEXPT8.execute(boundary)
