"""Real Core dependency/cache execution for a two-segment v1 split graph."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from comfy.nested_tensor import NestedTensor
from comfy_api.latest import io
import torch

from h3_audio_t8_pkg import chunked_two_pass_upscale_advanced as old
from h3_audio_t8_pkg.modular_sampling.chunked_source_nodes import NODES as SOURCE_NODES
from h3_audio_t8_pkg.modular_sampling.chunked_stage_nodes import NODES as STAGE_NODES
from test_chunked_two_pass_global_noise_advanced import _plan, _CountingCoordinateNoise
from test_modular_chunked_source import _latent


PLAN = "T8_H3_CHUNKED_TWO_PASS_PLAN"
RESULT = "T8_CHUNKED_PASS2_RESULT"


class TinyChunkedSource(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, inputs=[], outputs=[
            io.Latent.Output("first_pass_latent"), io.Custom(PLAN).Output("plan"),
            io.Noise.Output("noise"), io.Sampler.Output("sampler"),
            io.Sigmas.Output("sigmas"),
        ])

    @classmethod
    def execute(cls):
        return io.NodeOutput(_latent(), _plan(old.build_chunked_two_pass_plan,
                                              strategy="full_frame_safe"),
                             _CountingCoordinateNoise(), object(), torch.tensor([1., 0.]))


class TinyChunkedModel(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, inputs=[io.Float.Input("offset", default=0.1)],
                         outputs=[io.Model.Output("model")])

    @classmethod
    def execute(cls, offset=0.1):
        return io.NodeOutput(SimpleNamespace(offset=float(offset)))


class TinyChunkedConditioning(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, inputs=[io.String.Input("prompt", default="first")],
                         outputs=[io.Conditioning.Output("positive")])

    @classmethod
    def execute(cls, prompt="first"):
        return io.NodeOutput([[torch.zeros(1), {"text": prompt}]])


class ChunkedResultSink(io.ComfyNode):
    seen = []

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, is_output_node=True,
                         inputs=[io.Latent.Input("av_latent"), io.Custom(RESULT).Input("result")],
                         outputs=[io.String.Output("report")])

    @classmethod
    def execute(cls, av_latent, result):
        video, audio = av_latent["samples"].unbind()
        assert result.index + 1 == result.count == 2
        cls.seen.append((video.clone(), audio.clone()))
        return io.NodeOutput(str(float(video.sum())))


def _graph():
    return {
        "1": {"class_type": "TinyChunkedSource", "inputs": {}},
        "2": {"class_type": "TinyChunkedModel", "inputs": {"offset": 0.1}},
        "3": {"class_type": "TinyChunkedModel", "inputs": {"offset": 0.2}},
        "4": {"class_type": "TinyChunkedConditioning", "inputs": {"prompt": "first"}},
        "5": {"class_type": "TinyChunkedConditioning", "inputs": {"prompt": "second"}},
        "10": {"class_type": "MiniMaxH3ChunkedPass2PrepareEXPT8", "inputs": {
            "first_pass_latent": ["1", 0], "plan": ["1", 1], "noise": ["1", 2]}},
        "11": {"class_type": "MiniMaxH3ChunkedSourceSegmentEXPT8", "inputs": {
            "first_pass_latent": ["1", 0], "plan": ["1", 1], "segment_index": 0}},
        "12": {"class_type": "MiniMaxH3ChunkedLearnedLiftEXPT8", "inputs": {
            "source_segment": ["11", 0], "segment_spec": ["11", 1],
            "pass2_context": ["10", 0], "plan": ["1", 1]}},
        "13": {"class_type": "MiniMaxH3ChunkedPass2SegmentEXPT8", "inputs": {
            "model": ["2", 0], "positive": ["4", 0],
            "source_segment": ["11", 0], "lifted_segment": ["12", 0],
            "segment_spec": ["11", 1], "pass2_context": ["10", 0],
            "plan": ["1", 1], "noise": ["1", 2], "sampler": ["1", 3],
            "sigmas": ["1", 4], "cfg": 1.0}},
        "21": {"class_type": "MiniMaxH3ChunkedSourceSegmentEXPT8", "inputs": {
            "first_pass_latent": ["1", 0], "plan": ["1", 1], "segment_index": 1}},
        "22": {"class_type": "MiniMaxH3ChunkedLearnedLiftEXPT8", "inputs": {
            "source_segment": ["21", 0], "segment_spec": ["21", 1],
            "pass2_context": ["10", 0], "plan": ["1", 1]}},
        "23": {"class_type": "MiniMaxH3ChunkedPass2SegmentEXPT8", "inputs": {
            "model": ["3", 0], "positive": ["5", 0],
            "source_segment": ["21", 0], "lifted_segment": ["22", 0],
            "segment_spec": ["21", 1], "pass2_context": ["10", 0],
            "plan": ["1", 1], "noise": ["1", 2], "sampler": ["1", 3],
            "sigmas": ["1", 4], "previous_result": ["13", 1], "cfg": 1.0}},
        "30": {"class_type": "ChunkedResultSink", "inputs": {
            "av_latent": ["23", 0], "result": ["23", 1]}},
    }


def test_core_v1_two_segment_high_only_edit_cache_and_cancel(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3]))
    import comfy.model_management
    import execution
    import nodes

    for cls in (*SOURCE_NODES, *STAGE_NODES, TinyChunkedSource,
                TinyChunkedModel, TinyChunkedConditioning, ChunkedResultSink):
        monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, cls.__name__, cls)

    def fake_lift(chunk, *_args):
        video, audio = chunk["samples"].unbind()
        return ({"samples": NestedTensor((
            video.repeat_interleave(2, -1).repeat_interleave(2, -2), audio))},
            64, 64, "{}")

    calls = []
    cancel = {"second": False}

    def fake_spatial(video, _audio, _positive, _plan, model, *_args, **_kwargs):
        calls.append(model.offset)
        if model.offset == 0.3 and cancel["second"]:
            raise comfy.model_management.InterruptProcessingException()
        return video + model.offset, {"mock": True}

    monkeypatch.setattr(old, "learned_upscale_h3_av_latent", fake_lift)
    monkeypatch.setattr(old, "_spatial_resample", fake_spatial)
    server = SimpleNamespace(client_id=None, last_node_id=None, sockets_metadata={},
                             send_sync=lambda *args, **kwargs: None)
    executor = execution.PromptExecutor(
        server, cache_args={"ram": 0., "ram_inactive": 0.},
        asset_manager=SimpleNamespace(enabled=False),
    )
    graph = _graph()
    ChunkedResultSink.seen.clear()

    def run(label, success=True):
        executor.execute(deepcopy(graph), label, execute_outputs=["30"])
        assert executor.success is success, executor.status_messages

    run("first")
    assert calls == [0.1, 0.2]
    first = ChunkedResultSink.seen[-1][0]
    calls.clear()
    run("cached")
    assert calls == []
    graph["3"]["inputs"]["offset"] = 0.3
    cancel["second"] = True
    run("second-cancel", success=False)
    assert calls == [0.3]
    calls.clear()
    cancel["second"] = False
    run("second-retry")
    assert calls == [0.3]
    assert not torch.equal(ChunkedResultSink.seen[-1][0], first)
    calls.clear()
    graph["2"]["inputs"]["offset"] = 0.4
    run("first-edit")
    assert calls == [0.4, 0.3]
