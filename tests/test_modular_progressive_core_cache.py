"""Actual Core graph scheduler/cache with tiny H3; encoder/lift are explicit doubles."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
import comfy.nested_tensor
from comfy_api.latest import io

from h3_audio_t8_pkg.modular_sampling import progressive as stages
from h3_audio_t8_pkg.modular_sampling import progressive_nodes as public
from h3_audio_t8_pkg.nodes import MiniMaxH3DualClockSamplerT8
from test_progressive_sampling_runtime import tiny_model, conditioning
from tools.build_modular_progressive_workflow import split_graph, prune


class TinyModel(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, inputs=[io.Float.Input("patch", default=0.)], outputs=[io.Model.Output()])
    @classmethod
    def execute(cls, patch):
        model = tiny_model()
        if patch:
            key = next(iter(model.model_state_dict()))
            value = model.model_state_dict()[key]
            model.add_patches({key: ("diff", (torch.full_like(value, patch),))})
        return io.NodeOutput(model)


class TinyCondition(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, inputs=[io.Int.Input("width"), io.Int.Input("height"),
            io.Float.Input("prompt", default=0.)], outputs=[io.Conditioning.Output(), io.Latent.Output()])
    @classmethod
    def execute(cls, width, height, prompt):
        positive = conditioning()
        positive[0][0] = positive[0][0] + prompt
        source = {"samples": comfy.nested_tensor.NestedTensor((
            torch.zeros(1, 24, 2, height // 16, width // 16), torch.zeros(1, 32, 2, 8)))}
        return io.NodeOutput(positive, source)


class InterpolationDouble(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, inputs=[io.Latent.Input("av_latent"),
            io.Int.Input("target_width"), io.Int.Input("target_height")],
            outputs=[io.Latent.Output(), io.Int.Output(), io.Int.Output()])
    @classmethod
    def execute(cls, av_latent, target_width, target_height):
        video, audio = av_latent["samples"].unbind()
        lifted = F.interpolate(video, size=(video.shape[2], target_height // 16, target_width // 16),
                               mode="trilinear", align_corners=False)
        return io.NodeOutput({"samples": comfy.nested_tensor.NestedTensor((lifted, audio))}, target_width, target_height)


class ResultSink(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, is_output_node=True,
            inputs=[io.Custom(public.HIGH_RESULT).Input("result")], outputs=[io.String.Output()])
    @classmethod
    def execute(cls, result):
        result.verify()
        return io.NodeOutput(result.receipt_json)


def graph():
    value = split_graph()
    for key in ("1", "22"):
        value[key] = {"class_type": "TinyModel", "inputs": {"patch": 0.}}
    value["90"]["inputs"] = {"width": 128, "height": 64, "length": 5}
    for key in ("10", "92"):
        value[key]["inputs"]["steps"] = 4
    value["26"]["inputs"]["low_evaluations"] = 2
    for key in ("9", "24"):
        old = value[key]["inputs"]
        value[key] = {"class_type": "TinyCondition", "inputs": {
            "width": old["width"] if key == "24" else 128,
            "height": old["height"] if key == "24" else 64, "prompt": 0.}}
    old = value["23"]["inputs"]
    value["23"] = {"class_type": "InterpolationDouble", "inputs": {
        key: old[key] for key in ("av_latent", "target_width", "target_height")}}
    value["100"] = {"class_type": "ResultSink", "inputs": {"result": ["29", 2]}}
    return prune(value, ["100"])


@pytest.mark.parametrize("change", ["high_prompt", "high_model_patch", "high_video_noise", "low_prompt", "high_cancel"])
def test_actual_core_cache_only_invalidates_the_changed_phase(monkeypatch, change):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3]))
    import execution
    import nodes
    from comfy_extras.nodes_custom_sampler import RandomNoise
    from comfy_extras.nodes_minimax_h3 import EmptyMiniMaxH3LatentAV
    for node in (*public.NODES, TinyModel, TinyCondition, InterpolationDouble, ResultSink,
                 MiniMaxH3DualClockSamplerT8, RandomNoise, EmptyMiniMaxH3LatentAV):
        monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, node.__name__, node)
    server = SimpleNamespace(client_id=None, last_node_id=None, sockets_metadata={}, send_sync=lambda *a, **k: None)
    executor = execution.PromptExecutor(server, cache_args={"ram": 0., "ram_inactive": 0.}, asset_manager=object())
    calls = []
    cancel = {"enabled": False}
    original = stages._run
    def count(model, sampler, plan, phase, *args, **kwargs):
        calls.append(phase)
        if phase == "high" and cancel["enabled"]:
            callback = kwargs.get("callback")
            def interrupt(*values):
                if callback is not None:
                    callback(*values)
                raise comfy.model_management.InterruptProcessingException()
            kwargs["callback"] = interrupt
        return original(model, sampler, plan, phase, *args, **kwargs)
    monkeypatch.setattr(stages, "_run", count)
    def run(value, name):
        executor.execute(deepcopy(value), name, execute_outputs=["100"])
        assert executor.success, executor.status_messages
        assert any(event == "execution_success" for event, _ in executor.status_messages)
    value = graph()
    run(value, "initial")
    assert calls == ["low", "high"]
    calls.clear()
    run(value, "unchanged")
    assert calls == []
    if change in ("high_prompt", "high_cancel"):
        value["24"]["inputs"]["prompt"] = .3
    elif change == "high_model_patch":
        value["22"]["inputs"]["patch"] = .01
    elif change == "high_video_noise":
        value["27"]["inputs"]["noise_seed"] += 10
    else:
        value["9"]["inputs"]["prompt"] = .3
    if change == "high_cancel":
        cancel["enabled"] = True
        executor.execute(deepcopy(value), "interrupted", execute_outputs=["100"])
        assert not executor.success
        assert any(event == "execution_interrupted" for event, _ in executor.status_messages)
        assert calls == ["high"]
        calls.clear()
        cancel["enabled"] = False
    run(value, "changed")
    assert calls == (["low", "high"] if change == "low_prompt" else ["high"])
    assert not torch.cuda.is_initialized()
