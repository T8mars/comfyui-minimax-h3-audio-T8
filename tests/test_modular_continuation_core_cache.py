"""Actual public continuation nodes and Core cache; tiny/encoder/lift doubles explicit."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

from comfy_api.latest import io
import comfy.model_management
import pytest
import torch

from h3_audio_t8_pkg.modular_sampling import continuation as stages
from h3_audio_t8_pkg.modular_sampling import continuation_nodes as public
from h3_audio_t8_pkg.modular_sampling import progressive as shared
from h3_audio_t8_pkg.modular_sampling import progressive_nodes
from h3_audio_t8_pkg.modular_sampling.progressive_effect_nodes import NODES as audit_nodes
from h3_audio_t8_pkg.modular_sampling.nodes import MiniMaxH3StageEAVConfigEXPT8
from h3_audio_t8_pkg.nodes import MiniMaxH3DualClockSamplerT8
from h3_audio_t8_pkg.nodes_prompt_relay_advanced import MiniMaxH3PromptRelayPlanT8Advanced
from helpers import FakeVideoVAE, FakeAudioVAE
from test_prompt_relay_long_video_advanced import NativeLikeFakeClip
from test_progressive_continuation import accepted  # noqa: F401
from test_modular_progressive_core_cache import TinyModel, InterpolationDouble, ResultSink
from tools.build_modular_continuation_workflow import split_graph, base


class ContinuationEncoderDoubles(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, inputs=[], outputs=[io.Clip.Output(), io.Vae.Output(), io.Vae.Output()])

    @classmethod
    def execute(cls):
        return io.NodeOutput(NativeLikeFakeClip(), FakeVideoVAE(), FakeAudioVAE())


def graph(case, kind):
    value = split_graph(kind=kind)
    for key in ("1", "22"):
        value[key] = {"class_type": "TinyModel", "inputs": {"patch": 0.}}
    value["200"]["inputs"] = {**case.request, "previous_job_sha256": case.request["job_sha256"]}
    value["200"]["inputs"].pop("job_sha256")
    value["6"] = {"class_type": "ContinuationEncoderDoubles", "inputs": {}}
    value["201"]["inputs"]["video_vae"] = ["6", 1]
    for key in ("9", "24"):
        value[key]["inputs"].update(clip=["6", 0], video_vae=["6", 1], audio_vae=["6", 2])
    previous = value["23"]["inputs"]
    value["23"] = {"class_type": "InterpolationDouble", "inputs": {
        key: previous[key] for key in ("av_latent", "target_width", "target_height")}}
    value["900"] = {"class_type": "ResultSink", "inputs": {"result": ["29", 2]}}
    if kind == "combined":
        for key in ("102", "112"):
            value[key]["inputs"]["mode"] = "apply_exp"
    return base.prune(value, ["900", "203"])


@pytest.mark.parametrize("change", ["high_prompt", "high_model_patch", "high_video_noise", "low_prompt", "high_cancel", "parent_changed"])
@pytest.mark.parametrize("kind", ["plain", "combined"])
def test_actual_core_cache_and_revalidation(accepted, monkeypatch, change, kind):  # noqa: F811
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3]))
    import execution
    import nodes
    from comfy_extras.nodes_custom_sampler import RandomNoise
    from comfy_extras.nodes_minimax_h3 import EmptyMiniMaxH3LatentAV
    for node in (*public.NODES, *progressive_nodes.NODES, *audit_nodes, TinyModel, InterpolationDouble, ResultSink,
                 ContinuationEncoderDoubles, MiniMaxH3DualClockSamplerT8, RandomNoise, EmptyMiniMaxH3LatentAV,
                 MiniMaxH3StageEAVConfigEXPT8, MiniMaxH3PromptRelayPlanT8Advanced):
        monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, node.__name__, node)
    monkeypatch.setattr(stages.delivery, "long_video_chain_root", lambda chain_id: accepted.root)
    server = SimpleNamespace(client_id=None, last_node_id=None, sockets_metadata={}, send_sync=lambda *a, **k: None)
    executor = execution.PromptExecutor(server, cache_args={"ram": 0., "ram_inactive": 0.},
                                        asset_manager=SimpleNamespace(enabled=False))
    calls, cancel = [], {"enabled": False}
    original = shared._run
    def count(model, sampler, plan, phase, *args, **kwargs):
        calls.append(phase)
        if phase == "high" and cancel["enabled"]:
            def interrupt(*args):
                raise comfy.model_management.InterruptProcessingException()
            kwargs["callback"] = interrupt
        return original(model, sampler, plan, phase, *args, **kwargs)
    monkeypatch.setattr(shared, "_run", count)
    def run(value, name):
        executor.execute(deepcopy(value), name, execute_outputs=["900", "203"])
        assert executor.success, executor.status_messages
    value = graph(accepted, kind)
    run(value, "initial")
    assert calls == ["low", "high"]
    calls.clear()
    run(value, "unchanged")
    assert calls == []
    if change == "parent_changed":
        accepted.manifest["revision"] = 2
        accepted.manifest_path.write_text(json.dumps(accepted.manifest))
        executor.execute(deepcopy(value), "changed-parent", execute_outputs=["900", "203"])
        assert not executor.success and calls == []
        return
    if change in ("high_prompt", "high_cancel"):
        value["110" if kind == "combined" else "24"]["inputs"][
            "global_prompt" if kind == "combined" else "prompt"] = "Turn toward the window."
    elif change == "high_model_patch":
        value["22"]["inputs"]["patch"] = .01
    elif change == "high_video_noise":
        value["27"]["inputs"]["noise_seed"] += 10
    else:
        value["100" if kind == "combined" else "9"]["inputs"][
            "global_prompt" if kind == "combined" else "prompt"] = "Pause near the door."
    if change == "high_cancel":
        cancel["enabled"] = True
        executor.execute(deepcopy(value), "cancelled", execute_outputs=["900", "203"])
        assert not executor.success and calls == ["high"]
        calls.clear()
        cancel["enabled"] = False
    run(value, "changed")
    assert calls == (["low", "high"] if change == "low_prompt" else ["high"])
    assert not torch.cuda.is_initialized()
