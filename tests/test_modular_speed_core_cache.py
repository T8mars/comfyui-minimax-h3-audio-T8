"""Run the public SPEED split graph through Core's real dependency cache."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from comfy_api.latest import io

from h3_audio_t8_pkg.modular_sampling import speed_nodes, speed_stages, speed_storage
from h3_audio_t8_pkg.modular_sampling import speed_storage_nodes
from h3_audio_t8_pkg.modular_sampling import nodes as stage_nodes
from h3_audio_t8_pkg.modular_sampling.nodes import (
    MiniMaxH3StageEAVApplyEXPT8,
    MiniMaxH3StageEAVAuditEXPT8,
    MiniMaxH3StageEAVConfigEXPT8,
)
from h3_audio_t8_pkg.nodes_prompt_relay_advanced import MiniMaxH3PromptRelayPlanT8Advanced
from h3_audio_t8_pkg.prompt_relay_advanced import build_prompt_relay_plan
from h3_audio_t8_pkg.nodes_speed_advanced import (
    MiniMaxH3SPEEDModalityStableNoiseT8Advanced,
    MiniMaxH3SPEEDPlanT8Advanced,
    MiniMaxH3SPEEDSourceT8Advanced,
)
from helpers import FakeAudioVAE, FakeClip, FakeVideoVAE
from test_modular_speed_stages import _model
from test_prompt_relay_advanced import NativeLikeFakeClip
from tools.build_modular_progressive_workflow import prune
from tools.build_modular_speed_workflow import split_graph


class TinySpeedInputs(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, inputs=[],
                         outputs=[io.Clip.Output(), io.Vae.Output(), io.Vae.Output()])

    @classmethod
    def execute(cls):
        return io.NodeOutput(FakeClip(), FakeVideoVAE(), FakeAudioVAE())


class TinySpeedEffectInputs(TinySpeedInputs):
    @classmethod
    def execute(cls):
        return io.NodeOutput(NativeLikeFakeClip(), FakeVideoVAE(), FakeAudioVAE())


class TinySpeedModel(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, inputs=[], outputs=[io.Model.Output()])

    @classmethod
    def execute(cls):
        return io.NodeOutput(_model())


class SpeedResultSink(io.ComfyNode):
    seen = []
    results = []

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, is_output_node=True,
                         inputs=[io.Custom(speed_nodes.RESULT).Input("result")],
                         outputs=[io.String.Output()])

    @classmethod
    def execute(cls, result):
        receipt = result.verify_live()
        cls.seen.append(receipt)
        cls.results.append(result)
        return io.NodeOutput(receipt["receipt_sha256"])


def _graph(*, stages=2, resume_stage=None, with_effects=False):
    graph = split_graph(stages, resume_stage=resume_stage, with_effects=with_effects)
    graph["1"] = {"class_type": "TinySpeedEffectInputs" if with_effects else "TinySpeedInputs",
                  "inputs": {}}
    if with_effects:
        relay_settings = dict(global_prompt="One stable scene",
                              local_prompts="A woman waves.\nShe walks away.", length=22)
        _, compiled_prompt, *_ = build_prompt_relay_plan(
            **relay_settings, timing_mode="auto_equal", time_ranges="", math_profile="paper_v1",
            epsilon=0.1, allow_gaps=False, allow_overlaps=False,
        )
    for index in range(resume_stage or 0, stages):
        base = 10 + 12 * index
        graph[str(base + 1)]["inputs"].update(
            clip=["1", 0], video_vae=["1", 1], audio_vae=["1", 2],
            length=22 if with_effects else 5,
        )
        graph[str(base)] = {"class_type": "TinySpeedModel", "inputs": {}}
        if with_effects:
            graph[str(base + 1)]["inputs"]["prompt"] = compiled_prompt
            graph[str(base + 6)]["inputs"].update(relay_settings)
        if index:
            graph[str(base + 2)]["inputs"]["reuse_t2va_text"] = False
            graph[str(base + 5)]["inputs"]["dct_chunk_size"] = 4
    canvas = 64 if stages == 2 else 96
    graph["4"]["inputs"].update(width=canvas, height=canvas)
    last_sample = str(10 + 12 * (stages - 1) + 4)
    graph["100"] = {"class_type": "SpeedResultSink", "inputs": {"result": [last_sample, 1]}}
    outputs = ["100"]
    if with_effects:
        outputs.extend(str(10 + 12 * index + 10) for index in range(resume_stage or 0, stages))
    return prune(graph, outputs)


@pytest.mark.parametrize("stages", [2, 3])
def test_core_speed_graph_cache_edit_and_exact_resume(monkeypatch, tmp_path, stages):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3]))
    import execution
    import nodes
    import comfy.model_management

    for cls in (*speed_nodes.NODES, *speed_storage_nodes.NODES,
                MiniMaxH3SPEEDPlanT8Advanced, MiniMaxH3SPEEDSourceT8Advanced,
                MiniMaxH3SPEEDModalityStableNoiseT8Advanced,
                TinySpeedInputs, TinySpeedModel, SpeedResultSink):
        monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, cls.__name__, cls)
    original_root = speed_storage.stage_root
    monkeypatch.setattr(speed_storage, "stage_root",
                        lambda **kwargs: original_root(output_root=tmp_path))
    monkeypatch.setattr(speed_stages, "_apply_speed_scoped_headroom",
                        lambda: (None, {"applied": False}))
    original_sample = speed_nodes.sample_speed_stage
    calls = []
    cancel = {"high": False}

    def counted_sample(model, positive, av_latent, sampler, sigmas, noise, spec):
        calls.append(spec.index)
        if spec.index == stages - 1 and cancel["high"]:
            raise comfy.model_management.InterruptProcessingException()
        return original_sample(model, positive, av_latent, sampler, sigmas, noise, spec)

    monkeypatch.setattr(speed_nodes, "sample_speed_stage", counted_sample)
    server = SimpleNamespace(client_id=None, last_node_id=None, sockets_metadata={},
                             send_sync=lambda *args, **kwargs: None)

    def new_executor():
        return execution.PromptExecutor(
            server, cache_args={"ram": 0., "ram_inactive": 0.},
            asset_manager=SimpleNamespace(enabled=False),
        )

    executor = new_executor()
    SpeedResultSink.seen.clear()
    SpeedResultSink.results.clear()
    graph = _graph(stages=stages)

    def run(current, label, *, expected_success=True):
        executor.execute(deepcopy(current), label, execute_outputs=["100"])
        assert executor.success is expected_success, executor.status_messages

    run(graph, "initial")
    assert calls == list(range(stages))
    initial_output = SpeedResultSink.seen[-1]["output"]
    root = original_root(output_root=tmp_path)
    manifests = list(root.rglob("manifest.json"))
    assert len(manifests) == stages - 1
    preceding = next(path for path in manifests
                     if json.loads(path.read_text(encoding="utf-8"))["stage_index"] == stages - 2)
    artifact = preceding.relative_to(root).as_posix()
    digest = speed_storage.file_sha(preceding)

    calls.clear()
    run(graph, "unchanged")
    assert calls == []
    final_source = str(11 + 12 * (stages - 1))
    graph[final_source]["inputs"]["prompt"] += " Another camera direction."
    cancel["high"] = True
    run(graph, "high_cancel_before_forward", expected_success=False)
    assert calls == [stages - 1]
    assert any(event == "execution_interrupted" for event, _ in executor.status_messages)
    calls.clear()
    cancel["high"] = False
    run(graph, "high_prompt_edit")
    assert calls == [stages - 1]
    calls.clear()
    if stages == 3:
        graph["23"]["inputs"]["prompt"] += " Different middle action."
        run(graph, "middle_prompt_edit")
        assert calls == [1, 2]
        calls.clear()
    graph["11"]["inputs"]["prompt"] += " Different opening action."
    run(graph, "low_prompt_edit")
    assert calls == list(range(stages))

    resumed = _graph(stages=stages, resume_stage=stages - 1)
    assert "10" not in resumed and "14" not in resumed
    resumed["5"]["inputs"].update(artifact_path=artifact, artifact_sha256=digest)
    calls.clear()
    executor = new_executor()
    run(resumed, "resume_high")
    assert calls == [stages - 1]
    assert SpeedResultSink.seen[-1]["output"] == initial_output

    invalid = deepcopy(resumed)
    invalid["5"]["inputs"]["artifact_sha256"] = "0" * 64
    calls.clear()
    executor = new_executor()
    run(invalid, "bad_sha", expected_success=False)
    assert calls == []

    # Inference tensors have no version counter; the exact content receipt
    # must still reject an in-place output edit after execution.
    frozen_result = SpeedResultSink.results[-1]
    with torch.inference_mode():
        frozen_result.external_output.unbind()[0].add_(1)
    with pytest.raises(ValueError, match="output content differs"):
        frozen_result.verify_live()


@pytest.mark.parametrize("stages", [2, 3])
def test_core_speed_external_relay_eav_edit_and_exact_resume(monkeypatch, tmp_path, stages):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3]))
    import execution
    import nodes
    import comfy.model_management

    for cls in (*speed_nodes.NODES, *speed_storage_nodes.NODES,
                MiniMaxH3SPEEDPlanT8Advanced, MiniMaxH3SPEEDSourceT8Advanced,
                MiniMaxH3SPEEDModalityStableNoiseT8Advanced,
                MiniMaxH3PromptRelayPlanT8Advanced,
                MiniMaxH3StageEAVConfigEXPT8, MiniMaxH3StageEAVApplyEXPT8,
                MiniMaxH3StageEAVAuditEXPT8,
                TinySpeedEffectInputs, TinySpeedModel, SpeedResultSink):
        monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, cls.__name__, cls)
    original_root = speed_storage.stage_root
    monkeypatch.setattr(speed_storage, "stage_root",
                        lambda **kwargs: original_root(output_root=tmp_path))
    monkeypatch.setattr(speed_stages, "_apply_speed_scoped_headroom",
                        lambda: (None, {"applied": False}))
    original_sample = speed_nodes.sample_speed_stage
    calls = []
    evidence = {}
    cancel = {"high": False}
    audits = []
    original_audit = stage_nodes.audit_stage_eav

    def recorded_audit(av_latent, runtime):
        value, report = original_audit(av_latent, runtime)
        audits.append(json.loads(report))
        return value, report

    monkeypatch.setattr(stage_nodes, "audit_stage_eav", recorded_audit)

    def counted_sample(model, positive, av_latent, sampler, sigmas, noise, spec):
        calls.append(spec.index)
        if spec.index == stages - 1 and cancel["high"]:
            raise comfy.model_management.InterruptProcessingException()
        output = original_sample(model, positive, av_latent, sampler, sigmas, noise, spec)
        evidence[spec.index] = output[1].verify_live()
        return output

    monkeypatch.setattr(speed_nodes, "sample_speed_stage", counted_sample)
    server = SimpleNamespace(client_id=None, last_node_id=None, sockets_metadata={},
                             send_sync=lambda *args, **kwargs: None)

    def new_executor():
        return execution.PromptExecutor(
            server, cache_args={"ram": 0., "ram_inactive": 0.},
            asset_manager=SimpleNamespace(enabled=False),
        )

    executor = new_executor()
    SpeedResultSink.seen.clear()
    graph = _graph(stages=stages, with_effects=True)
    audit_ids = tuple(str(20 + 12 * index) for index in range(stages))
    last_base = 10 + 12 * (stages - 1)

    def run(current, label, *, outputs=None, expected_success=True):
        selected = ("100", *audit_ids) if outputs is None else outputs
        executor.execute(deepcopy(current), label, execute_outputs=list(selected))
        assert executor.success is expected_success, executor.status_messages

    run(graph, "effects_initial")
    assert calls == list(range(stages))
    assert all(evidence[index]["portable_identity"] for index in range(stages))
    assert all(evidence[index]["execution"]["effects"]["relay_attention_calls"] > 0
               for index in range(stages))
    assert len(audits) == stages
    assert all(item["status"] == "observed_report_only" for item in audits)
    initial_output = SpeedResultSink.seen[-1]["output"]
    root = original_root(output_root=tmp_path)
    manifests = list(root.rglob("manifest.json"))
    assert len(manifests) == stages - 1
    preceding = next(path for path in manifests
                     if json.loads(path.read_text(encoding="utf-8"))["stage_index"] == stages - 2)
    artifact = preceding.relative_to(root).as_posix()
    digest = speed_storage.file_sha(preceding)

    calls.clear()
    run(graph, "effects_unchanged")
    assert calls == []
    assert len(audits) == stages
    graph[str(last_base + 6)]["inputs"]["local_prompts"] = "A woman looks up.\nShe turns away."
    cancel["high"] = True
    run(graph, "high_relay_cancel_before_forward", expected_success=False)
    assert calls == [stages - 1]
    assert any(event == "execution_interrupted" for event, _ in executor.status_messages)
    calls.clear()
    cancel["high"] = False
    run(graph, "high_relay_edit")
    assert calls == [stages - 1]
    calls.clear()
    graph[str(last_base + 8)]["inputs"]["mode"] = "apply_exp"
    run(graph, "high_eav_edit")
    assert calls == [stages - 1]
    assert evidence[stages - 1]["execution"]["effects"]["mode"] == "apply_exp"
    assert audits[-1]["status"] == "observed_apply_exp"
    calls.clear()
    graph["16"]["inputs"]["local_prompts"] = "A woman starts moving.\nShe walks away."
    run(graph, "low_relay_edit")
    assert calls == list(range(stages))

    resumed = _graph(stages=stages, resume_stage=stages - 1, with_effects=True)
    resumed["5"]["inputs"].update(artifact_path=artifact, artifact_sha256=digest)
    calls.clear()
    executor = new_executor()
    run(resumed, "effects_resume", outputs=("100", audit_ids[-1]))
    assert calls == [stages - 1]
    assert SpeedResultSink.seen[-1]["output"] == initial_output

    invalid = deepcopy(resumed)
    invalid["5"]["inputs"]["artifact_sha256"] = "0" * 64
    calls.clear()
    executor = new_executor()
    run(invalid, "effects_bad_sha", outputs=("100", audit_ids[-1]), expected_success=False)
    assert calls == []
