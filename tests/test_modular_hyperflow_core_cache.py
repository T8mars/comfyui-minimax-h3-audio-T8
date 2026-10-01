"""Real Core dependency/cache; actual50+2 HyperFlow, synthetic weights/encoder."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from comfy_api.latest import io

from h3_audio_t8_pkg.modular_sampling import hyperflow as stages, hyperflow_nodes as public
from h3_audio_t8_pkg.hyperflow_runtime_advanced import install_hyperflow
from test_hyperflow_advanced import _tiny_native_model, _tiny_weights
from test_progressive_sampling_runtime import tiny_model
from test_modular_progressive_core_cache import TinyCondition
from tools.build_modular_hyperflow_workflow import split_graph
from tools.build_modular_progressive_workflow import prune


class TinyHyperBase(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, inputs=[], outputs=[io.Model.Output()])


class SyntheticHyperLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, inputs=[io.Model.Input("model"),
            io.Float.Input("patch", default=0.)], outputs=[io.Model.Output()])

    @classmethod
    def execute(cls, model, patch):
        branch = model.clone()
        if patch:
            key, weight = next((k, w) for k, w in branch.model.named_parameters() if w.ndim == 2)
            branch.add_patches({key: ("diff", (torch.full_like(weight, patch),))})
        selected, _, _ = install_hyperflow(branch, _tiny_weights(branch.get_model_object("diffusion_model")))
        return io.NodeOutput(selected)


class HyperResultSink(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, is_output_node=True,
            inputs=[io.Custom(public.RESULT).Input("result")], outputs=[io.String.Output()])

    @classmethod
    def execute(cls, result):
        result.verify()
        return io.NodeOutput(result.receipt_json)


@pytest.mark.parametrize("change", ["tail_prompt", "tail_patch", "tail_seed", "head_prompt", "head_noise", "tail_cancel"])
def test_actual_core_phase_dependency_cache_and_cancel(monkeypatch, change):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3]))
    import execution
    import nodes
    import comfy.model_management
    from comfy_extras.nodes_custom_sampler import RandomNoise
    from comfy_extras.nodes_minimax_h3 import EmptyMiniMaxH3LatentAV
    _, diffusion = _tiny_native_model(monkeypatch)
    base = tiny_model()
    base.model.diffusion_model = diffusion
    monkeypatch.setattr(TinyHyperBase, "execute", classmethod(lambda cls: io.NodeOutput(base)))
    for cls in (*public.NODES, TinyHyperBase, SyntheticHyperLoader, TinyCondition, HyperResultSink,
                RandomNoise, EmptyMiniMaxH3LatentAV):
        monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, cls.__name__, cls)
    graph = split_graph()
    graph["1"] = {"class_type": "TinyHyperBase", "inputs": {}}
    for key in ("101", "102"):
        graph[key] = {"class_type": "SyntheticHyperLoader", "inputs": {"model": ["1", 0], "patch": 0.}}
    for key in ("9", "24"):
        graph[key] = {"class_type": "TinyCondition", "inputs": {"width": 64, "height": 64, "prompt": 0.}}
    graph["90"]["inputs"] = {"width": 64, "height": 64, "length": 5}
    graph["100"] = {"class_type": "HyperResultSink", "inputs": {"result": ["29", 2]}}
    graph = prune(graph, ["100"])
    server = SimpleNamespace(client_id=None, last_node_id=None, sockets_metadata={}, send_sync=lambda *a, **k: None)
    executor = execution.PromptExecutor(server, cache_args={"ram": 0., "ram_inactive": 0.},
                                        asset_manager=SimpleNamespace(enabled=False))
    calls, cancel = [], {"enabled": False}
    original = stages._run
    def run_stage(model, plan, *args, **kwargs):
        phase = "tail" if plan.start_interval else "head"
        calls.append(phase)
        if phase == "tail" and cancel["enabled"]:
            def interrupt(*_):
                raise comfy.model_management.InterruptProcessingException()
            kwargs["callback"] = interrupt
        return original(model, plan, *args, **kwargs)
    monkeypatch.setattr(stages, "_run", run_stage)
    def run(label):
        executor.execute(deepcopy(graph), label, execute_outputs=["100"])
        assert executor.success, executor.status_messages
        assert any(event == "execution_success" for event, _ in executor.status_messages)
    run("initial")
    assert calls == ["head", "tail"]
    calls.clear()
    run("unchanged")
    assert calls == []
    if change in ("tail_prompt", "tail_cancel"):
        graph["24"]["inputs"]["prompt"] = .3
    elif change == "tail_patch":
        graph["102"]["inputs"]["patch"] = .01
    elif change == "tail_seed":
        graph["29"]["inputs"]["seed"] += 1
    elif change == "head_noise":
        graph["11"]["inputs"]["noise_seed"] += 1
    else:
        graph["9"]["inputs"]["prompt"] = .3
    if change == "tail_cancel":
        cancel["enabled"] = True
        executor.execute(deepcopy(graph), "cancel", execute_outputs=["100"])
        assert not executor.success
        assert any(event == "execution_interrupted" for event, _ in executor.status_messages)
        assert calls == ["tail"]
        calls.clear()
        cancel["enabled"] = False
    run("changed")
    assert calls == (["head", "tail"] if change.startswith("head") else ["tail"])
    assert not torch.cuda.is_initialized()
