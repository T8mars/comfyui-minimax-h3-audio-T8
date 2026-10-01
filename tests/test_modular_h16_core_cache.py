"""Real Core graph cache/cancel dependency test for seven split H16 windows.

The learned lift and native sampler are deterministic CPU stand-ins.  Cache
invalidation and cancellation are exercised by the real PromptExecutor.
"""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from comfy.nested_tensor import NestedTensor
from comfy_api.latest import io
import torch

from h3_audio_t8_pkg import chunked_two_pass_upscale_advanced as legacy
from h3_audio_t8_pkg.modular_sampling.chunked_source_nodes import NODES as SOURCE_NODES
from h3_audio_t8_pkg.modular_sampling.chunked_stage_nodes import NODES as STAGE_NODES
from h3_audio_t8_pkg.modular_sampling.h16_nodes import NODES as H16_NODES
from h3_audio_t8_pkg.modular_sampling.storage import file_sha
from test_chunked_two_pass_global_noise_advanced import _CountingCoordinateNoise
from test_modular_h16_stages import _h16_source, _no_op_lift


RESULT = "T8_H16_PASS2_RESULT"
WINDOWS = 7


class TinyH16Source(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, inputs=[], outputs=[
            io.Latent.Output("first_pass_latent"), io.Noise.Output("noise"),
            io.Sampler.Output("sampler"), io.Sigmas.Output("sigmas"),
        ])

    @classmethod
    def execute(cls):
        source = _h16_source()
        video, audio = source["samples"].unbind()
        video_mask, audio_mask = source["noise_mask"].unbind()
        source["samples"] = NestedTensor((
            video[:, :, :1].repeat(1, 1, 37, 1, 1),
            audio[..., :1].repeat(1, 1, 1, 220),
        ))
        source["noise_mask"] = NestedTensor((
            video_mask[:, :, :1].repeat(1, 1, 37, 1, 1),
            audio_mask[..., :1].repeat(1, 1, 1, 220),
        ))
        return io.NodeOutput(source, _CountingCoordinateNoise(), object(),
                             torch.tensor([0.5, 0.0]))


class TinyH16Model(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, inputs=[
            io.Float.Input("offset", default=0.1)], outputs=[io.Model.Output("model")])

    @classmethod
    def execute(cls, offset=0.1):
        return io.NodeOutput(SimpleNamespace(offset=float(offset)))


class TinyH16Conditioning(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, inputs=[], outputs=[
            io.Conditioning.Output("positive")])

    @classmethod
    def execute(cls):
        return io.NodeOutput([[torch.zeros(1), {}]])


class H16ResultSink(io.ComfyNode):
    seen = []

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, is_output_node=True,
                         inputs=[io.Latent.Input("av_latent"),
                                 io.Custom(RESULT).Input("result")],
                         outputs=[io.String.Output("report")])

    @classmethod
    def execute(cls, av_latent, result):
        video, audio = av_latent["samples"].unbind()
        assert result.core_result.index + 1 == result.core_result.count == WINDOWS
        assert result.output_latent is av_latent
        cls.seen.append((video.clone(), audio.clone()))
        return io.NodeOutput(str(float(video.sum())))


def _graph():
    graph = {
        "1": {"class_type": "TinyH16Source", "inputs": {}},
        "2": {"class_type": "MiniMaxH3H16Pass2PlanEXPT8", "inputs": {
            "first_pass_latent": ["1", 0]}},
        "3": {"class_type": "TinyH16Conditioning", "inputs": {}},
        "10": {"class_type": "MiniMaxH3ChunkedPass2PrepareEXPT8", "inputs": {
            "first_pass_latent": ["1", 0], "plan": ["2", 0], "noise": ["1", 1]}},
    }
    for index in range(WINDOWS):
        segment = str(20 + index * 10)
        lift = str(21 + index * 10)
        model = str(22 + index * 10)
        stage = str(23 + index * 10)
        graph[segment] = {"class_type": "MiniMaxH3ChunkedSourceSegmentEXPT8", "inputs": {
            "first_pass_latent": ["1", 0], "plan": ["2", 0],
            "segment_index": index}}
        graph[lift] = {"class_type": "MiniMaxH3ChunkedLearnedLiftEXPT8", "inputs": {
            "source_segment": [segment, 0], "segment_spec": [segment, 1],
            "pass2_context": ["10", 0], "plan": ["2", 0]}}
        graph[model] = {"class_type": "TinyH16Model", "inputs": {
            "offset": round((index + 1) / 10, 1)}}
        inputs = {
            "model": [model, 0], "positive": ["3", 0],
            "source_segment": [segment, 0], "lifted_segment": [lift, 0],
            "segment_spec": [segment, 1], "pass2_context": ["10", 0],
            "plan": ["2", 0], "noise": ["1", 1],
            "sampler": ["1", 2], "sigmas": ["1", 3],
            "audio_output": "preserve_first_pass", "cfg": 1.0,
        }
        if index:
            inputs["previous_result"] = [str(23 + (index - 1) * 10), 1]
        graph[stage] = {"class_type": "MiniMaxH3H16Pass2WindowEXPT8", "inputs": inputs}
    graph["99"] = {"class_type": "H16ResultSink", "inputs": {
        "av_latent": ["83", 0], "result": ["83", 1]}}
    return graph


def test_core_h16_seven_windows_cache_cancel_and_upstream_edit(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3]))
    import comfy.model_management
    import execution
    import nodes

    for cls in (*SOURCE_NODES, *STAGE_NODES, *H16_NODES, TinyH16Source,
                TinyH16Model, TinyH16Conditioning, H16ResultSink):
        monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, cls.__name__, cls)

    calls = []
    cancel = {"last": False}

    def fake_spatial(video, _audio, _positive, _plan, model, *_args, **_kwargs):
        calls.append(model.offset)
        if model.offset == 0.8 and cancel["last"]:
            raise comfy.model_management.InterruptProcessingException()
        return video + model.offset, {"mock": True}

    monkeypatch.setattr(legacy, "learned_upscale_h3_av_latent", _no_op_lift)
    monkeypatch.setattr(legacy, "_spatial_resample", fake_spatial)
    server = SimpleNamespace(client_id=None, last_node_id=None, sockets_metadata={},
                             send_sync=lambda *args, **kwargs: None)
    executor = execution.PromptExecutor(
        server, cache_args={"ram": 0., "ram_inactive": 0.},
        asset_manager=SimpleNamespace(enabled=False),
    )
    graph = _graph()
    H16ResultSink.seen.clear()

    def run(label, success=True):
        executor.execute(deepcopy(graph), label, execute_outputs=["99"])
        assert executor.success is success, executor.status_messages

    run("h16-first")
    assert calls == [round((index + 1) / 10, 1) for index in range(WINDOWS)]
    first_video, first_audio = H16ResultSink.seen[-1]
    calls.clear()
    run("h16-cached")
    assert calls == []

    graph["82"]["inputs"]["offset"] = 0.8
    cancel["last"] = True
    run("h16-last-cancel", success=False)
    assert calls == [0.8]
    calls.clear()
    cancel["last"] = False
    run("h16-last-retry")
    assert calls == [0.8]
    assert not torch.equal(H16ResultSink.seen[-1][0], first_video)
    assert torch.equal(H16ResultSink.seen[-1][1], first_audio)

    calls.clear()
    graph["52"]["inputs"]["offset"] = 0.9
    run("h16-middle-edit")
    assert calls == [0.9, 0.5, 0.6, 0.8]

    calls.clear()
    graph["22"]["inputs"]["offset"] = 1.0
    run("h16-first-edit")
    assert calls == [1.0, 0.2, 0.3, 0.9, 0.5, 0.6, 0.8]


def test_core_h16_exact_artifact_resumes_only_later_windows_in_fresh_executor(
        monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3]))
    import execution
    import nodes
    from h3_audio_t8_pkg.modular_sampling import h16_nodes

    for cls in (*SOURCE_NODES, *STAGE_NODES, *H16_NODES, TinyH16Source,
                TinyH16Model, TinyH16Conditioning, H16ResultSink):
        monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, cls.__name__, cls)
    monkeypatch.setattr(legacy, "learned_upscale_h3_av_latent", _no_op_lift)
    monkeypatch.setattr(h16_nodes, "_h16_store_root", lambda: tmp_path)
    calls = []

    def fake_spatial(video, _audio, _positive, _plan, model, *_args, **_kwargs):
        calls.append(model.offset)
        return video + model.offset, {"mock": True}

    monkeypatch.setattr(legacy, "_spatial_resample", fake_spatial)
    server = SimpleNamespace(client_id=None, last_node_id=None, sockets_metadata={},
                             send_sync=lambda *args, **kwargs: None)

    def fresh_executor():
        return execution.PromptExecutor(
            server, cache_args={"ram": 0., "ram_inactive": 0.},
            asset_manager=SimpleNamespace(enabled=False),
        )

    freeze = _graph()
    freeze["100"] = {"class_type": "MiniMaxH3H16WindowSaveEXPT8", "inputs": {
        "window_result": ["23", 1], "source_segment": ["20", 0],
        "segment_spec": ["20", 1], "pass2_context": ["10", 0],
        "plan": ["2", 0], "confirm_save": True}}
    executor = fresh_executor()
    executor.execute(deepcopy(freeze), "h16-freeze", execute_outputs=["100"])
    assert executor.success, executor.status_messages
    assert calls == [0.1]
    manifests = list(tmp_path.rglob("manifest.json"))
    assert len(manifests) == 1
    manifest = manifests[0]
    path = manifest.relative_to(tmp_path).as_posix()
    digest = file_sha(manifest)

    resume = _graph()
    resume.pop("23")
    resume["101"] = {"class_type": "MiniMaxH3H16WindowLoadEXPT8", "inputs": {
        "source_segment": ["20", 0], "segment_spec": ["20", 1],
        "pass2_context": ["10", 0], "plan": ["2", 0],
        "artifact_path": path, "artifact_sha256": digest}}
    resume["33"]["inputs"]["previous_result"] = ["101", 1]
    calls.clear()
    H16ResultSink.seen.clear()
    executor = fresh_executor()
    executor.execute(deepcopy(resume), "h16-resume", execute_outputs=["99"])
    assert executor.success, executor.status_messages
    assert calls == [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    resumed_video, resumed_audio = H16ResultSink.seen[-1]

    calls.clear()
    executor = fresh_executor()
    executor.execute(_graph(), "h16-control", execute_outputs=["99"])
    assert executor.success, executor.status_messages
    assert calls == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    control_video, control_audio = H16ResultSink.seen[-1]
    assert torch.equal(resumed_video, control_video)
    assert torch.equal(resumed_audio, control_audio)
