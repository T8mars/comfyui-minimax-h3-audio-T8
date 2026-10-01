"""Actual Core cache with public Avatar nodes; explicitly labelled source/encoder/lift doubles."""
import comfy.nested_tensor
from comfy_api.latest import io
import pytest
import torch

from h3_audio_t8_pkg.modular_sampling import avatar, avatar_nodes
import test_modular_progressive_core_cache as core_cache
from test_modular_avatar import recording


class EncodedRecordingDouble(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, inputs=[io.Latent.Input("av_latent")],
                         outputs=[io.Latent.Output(), io.Audio.Output()])

    @classmethod
    def execute(cls, av_latent):
        video, audio = av_latent["samples"].unbind()
        source = {"samples": comfy.nested_tensor.NestedTensor((video, torch.linspace(-.2, .2, audio.numel()).reshape_as(audio))),
            "noise_mask": comfy.nested_tensor.NestedTensor((torch.ones_like(video), torch.zeros_like(audio)))}
        return io.NodeOutput(source, recording())


@pytest.mark.parametrize("change", ["high_prompt", "high_model_patch", "high_video_noise", "low_prompt", "high_cancel"])
def test_public_avatar_core_dependency_and_cancel_do_not_repeat_low(monkeypatch, change):
    original_graph = core_cache.graph
    def graph():
        value = original_graph()
        value["130"] = {"class_type": "EncodedRecordingDouble", "inputs": {"av_latent": ["90", 0]}}
        value["133"] = {"class_type": "MiniMaxH3AvatarSourceBindEXPT8", "inputs": {
            "high_source": ["130", 0], "original_recording": ["130", 1]}}
        value["26"]["inputs"].update(high_source=["133", 0], input_mode="initialized_av_exp")
        value["13"]["class_type"] = "MiniMaxH3AvatarLowStageEXPT8"
        value["13"]["inputs"]["avatar_source"] = ["133", 1]
        value["25"]["class_type"] = "MiniMaxH3AvatarHighHandoffEXPT8"
        value["25"]["inputs"].pop("high_source")
        value["25"]["inputs"]["avatar_source"] = ["133", 1]
        return value
    monkeypatch.setattr(core_cache, "graph", graph)
    monkeypatch.setattr(core_cache.public, "NODES", [*core_cache.public.NODES, *avatar_nodes.NODES, EncodedRecordingDouble])
    original_sink = core_cache.ResultSink.execute
    deliveries = []
    def sink(cls, result):
        pcm = recording()
        _, original, report = avatar.deliver(result, pcm)
        assert original is pcm
        deliveries.append(report)
        return original_sink(result)
    monkeypatch.setattr(core_cache.ResultSink, "execute", classmethod(sink))
    core_cache.test_actual_core_cache_only_invalidates_the_changed_phase(monkeypatch, change)
    assert len(deliveries) == 2
