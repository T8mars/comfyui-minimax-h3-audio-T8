"""Real Core dependency execution for separate V2 LOW/learned/HIGH stages.

The tiny gated V2 samplers are real. Conditioning and learned 3D weights are
test doubles; this verifies graph cache invalidation, not pretrained AV quality.
"""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

from comfy_api.latest import io

from h3_audio_t8_pkg.conditioning import build_conditioning
from h3_audio_t8_pkg.modular_sampling import nodes as modular_nodes
from h3_audio_t8_pkg.modular_sampling.fast_h3_v2_handoff_nodes import NODES as HANDOFF_NODES
from h3_audio_t8_pkg.modular_sampling.storage import file_sha
from h3_audio_t8_pkg.nodes_learned_latent_upscale_advanced import MiniMaxH3TwoPassLatentReconcileT8Advanced
from h3_audio_t8_pkg.nodes_prompt_relay_advanced import (
    MiniMaxH3PromptRelayConditioningT8Advanced,
    MiniMaxH3PromptRelayPlanT8Advanced,
)
from helpers import FakeAudioVAE, FakeVideoVAE
from test_fast_h3_v2_core_sampler import model
from test_modular_progressive_core_cache import TinyCondition, InterpolationDouble
from test_prompt_relay_advanced import NativeLikeFakeClip


class TinyV2Model(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, inputs=[io.Int.Input("revision", default=0)],
                         outputs=[io.Model.Output("model")])

    @classmethod
    def execute(cls, revision=0):
        return io.NodeOutput(model())


class TinyRelayAssets(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, inputs=[], outputs=[
            io.Clip.Output("clip"), io.Vae.Output("video_vae"), io.Vae.Output("audio_vae")])

    @classmethod
    def execute(cls):
        return io.NodeOutput(NativeLikeFakeClip(), FakeVideoVAE(), FakeAudioVAE())


class TinyLongCondition(TinyCondition):
    @classmethod
    def execute(cls, width, height, prompt):
        positive, latent, *_ = build_conditioning(
            NativeLikeFakeClip(), FakeVideoVAE(), FakeAudioVAE(),
            f"One stable scene {prompt}", width, height, 22,
            task_type="T2VA", audio_mode="native", add_source_as_reference=False,
            prompt_primary_audio_ordinal=0,
        )
        return io.NodeOutput(positive, latent)


class V2ResultSink(io.ComfyNode):
    seen = []

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, is_output_node=True,
                         inputs=[io.Custom("T8_STAGE_RESULT").Input("high_result")],
                         outputs=[io.String.Output("report_json")])

    @classmethod
    def execute(cls, high_result):
        receipt = high_result.verify()
        assert receipt["verified_recipe_completion"] is True
        cls.seen.append(receipt["outputs"])
        return io.NodeOutput(high_result.receipt_json)


def _graph():
    return {
        "1": {"class_type": "TinyV2Model", "inputs": {"revision": 0}},
        "2": {"class_type": "TinyCondition", "inputs": {"width": 128, "height": 64, "prompt": 0.}},
        "3": {"class_type": "MiniMaxH3FastH3V2StageSetupEXPT8", "inputs": {
            "model": ["1", 0], "av_latent": ["2", 1], "stage": "low_0_4",
            "profile": "dense_compat_exp", "min_tokens": 0}},
        "4": {"class_type": "RandomNoise", "inputs": {"noise_seed": 17}},
        "5": {"class_type": "BasicGuider", "inputs": {"model": ["3", 0], "conditioning": ["2", 0]}},
        "6": {"class_type": "MiniMaxH3StageSamplerEXPT8", "inputs": {
            "noise": ["4", 0], "guider": ["5", 0], "sampler": ["3", 1],
            "sigmas": ["3", 2], "latent_image": ["2", 1], "stage_context": ["3", 3]}},
        "7": {"class_type": "MiniMaxH3FastH3V2CompletedLowX0EXPT8", "inputs": {
            "low_stage_result": ["6", 2]}},
        "8": {"class_type": "InterpolationDouble", "inputs": {
            "av_latent": ["7", 0], "target_width": 256, "target_height": 128}},
        "9": {"class_type": "TinyV2Model", "inputs": {"revision": 0}},
        "10": {"class_type": "TinyCondition", "inputs": {
            "width": ["8", 1], "height": ["8", 2], "prompt": 0.}},
        "11": {"class_type": "MiniMaxH3TwoPassLatentReconcileT8Advanced", "inputs": {
            "learned_latent": ["8", 0], "highres_template": ["10", 1],
            "positive": ["10", 0], "audio_policy": "auto",
            "second_pass_audio_source": "legacy_policy", "second_pass_audio_strength": 0.}},
        "12": {"class_type": "MiniMaxH3FastH3V2StageSetupEXPT8", "inputs": {
            "model": ["9", 0], "av_latent": ["11", 0], "stage": "high_4_8",
            "profile": "dense_compat_exp", "min_tokens": 0}},
        "13": {"class_type": "RandomNoise", "inputs": {"noise_seed": 18}},
        "14": {"class_type": "BasicGuider", "inputs": {"model": ["12", 0], "conditioning": ["11", 1]}},
        "15": {"class_type": "MiniMaxH3StageSamplerEXPT8", "inputs": {
            "noise": ["13", 0], "guider": ["14", 0], "sampler": ["12", 1],
            "sigmas": ["12", 2], "latent_image": ["11", 0], "stage_context": ["12", 3]}},
        "16": {"class_type": "V2ResultSink", "inputs": {"high_result": ["15", 2]}},
    }


def test_core_cache_high_only_edit_and_low_change_recompute_correct_stages(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3]))
    import comfy.model_management
    import execution
    import nodes
    from comfy_extras.nodes_custom_sampler import RandomNoise, BasicGuider

    for cls in (modular_nodes.MiniMaxH3FastH3V2StageSetupEXPT8,
                modular_nodes.MiniMaxH3StageSamplerEXPT8, *HANDOFF_NODES,
                MiniMaxH3TwoPassLatentReconcileT8Advanced, RandomNoise, BasicGuider,
                TinyV2Model, TinyCondition, InterpolationDouble, V2ResultSink):
        monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, cls.define_schema().node_id, cls)

    calls = []
    cancel = {"high": False}
    original = modular_nodes.sample_stage

    def observed(*args):
        calls.append(args[-1].stage)
        if args[-1].stage == "high_4_8" and cancel["high"]:
            raise comfy.model_management.InterruptProcessingException()
        return original(*args)

    monkeypatch.setattr(modular_nodes, "sample_stage", observed)
    server = SimpleNamespace(client_id=None, last_node_id=None, sockets_metadata={},
                             send_sync=lambda *args, **kwargs: None)
    executor = execution.PromptExecutor(server, cache_args={"ram": 0., "ram_inactive": 0.},
                                        asset_manager=SimpleNamespace(enabled=False))
    graph = _graph()
    V2ResultSink.seen.clear()

    def run(label, success=True):
        executor.execute(deepcopy(graph), label, execute_outputs=["16"])
        assert executor.success is success, executor.status_messages

    run("v2-first")
    assert calls == ["low_0_4", "high_4_8"]
    first = V2ResultSink.seen[-1]
    calls.clear()
    run("v2-cached")
    assert calls == []

    graph["13"]["inputs"]["noise_seed"] = 19
    run("v2-high-noise-edit")
    assert calls == ["high_4_8"]
    assert V2ResultSink.seen[-1] != first
    calls.clear()

    graph["13"]["inputs"]["noise_seed"] = 20
    cancel["high"] = True
    run("v2-high-cancel", success=False)
    assert calls == ["high_4_8"]
    calls.clear()
    cancel["high"] = False
    run("v2-high-retry")
    assert calls == ["high_4_8"]
    calls.clear()

    graph["2"]["inputs"]["prompt"] = 0.1
    run("v2-low-prompt-edit")
    assert calls == ["low_0_4", "high_4_8"]


def test_fresh_core_executor_loads_frozen_low_and_executes_only_high(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3]))
    import execution
    import nodes
    from comfy_extras.nodes_custom_sampler import RandomNoise, BasicGuider

    for cls in (modular_nodes.MiniMaxH3FastH3V2StageSetupEXPT8,
                modular_nodes.MiniMaxH3StageSamplerEXPT8,
                modular_nodes.MiniMaxH3StageSaveEXPT8,
                modular_nodes.MiniMaxH3StageLoadEXPT8, *HANDOFF_NODES,
                modular_nodes.MiniMaxH3StageEAVConfigEXPT8,
                modular_nodes.MiniMaxH3StageEAVApplyEXPT8,
                modular_nodes.MiniMaxH3StageEAVAuditEXPT8,
                MiniMaxH3TwoPassLatentReconcileT8Advanced, RandomNoise, BasicGuider,
                TinyV2Model, TinyCondition, InterpolationDouble, V2ResultSink):
        monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, cls.define_schema().node_id, cls)
    monkeypatch.setattr(modular_nodes, "_stage_store_root", lambda: tmp_path)
    calls = []
    original = modular_nodes.sample_stage

    def observed(*args):
        calls.append(args[-1].stage)
        return original(*args)

    monkeypatch.setattr(modular_nodes, "sample_stage", observed)
    server = SimpleNamespace(client_id=None, last_node_id=None, sockets_metadata={},
                             send_sync=lambda *args, **kwargs: None)

    def fresh_executor():
        return execution.PromptExecutor(server, cache_args={"ram": 0., "ram_inactive": 0.},
                                        asset_manager=SimpleNamespace(enabled=False))

    freeze = _graph()
    freeze["17"] = {"class_type": "MiniMaxH3StageSaveEXPT8", "inputs": {
        "stage_result": ["6", 2], "prefix": "v2/LOW"}}
    executor = fresh_executor()
    executor.execute(deepcopy(freeze), "v2-freeze-low", execute_outputs=["17"])
    assert executor.success, executor.status_messages
    assert calls == ["low_0_4"]
    manifests = list(tmp_path.rglob("manifest.json"))
    assert len(manifests) == 1
    path = manifests[0].relative_to(tmp_path).as_posix()
    digest = file_sha(manifests[0])

    resume = {key: value for key, value in _graph().items() if int(key) >= 7}
    resume["7"]["inputs"]["low_stage_result"] = ["18", 3]
    resume["18"] = {"class_type": "MiniMaxH3StageLoadEXPT8", "inputs": {
        "artifact_path": path, "artifact_sha256": digest, "expected_stage": "low_0_4"}}
    assert not {"1", "2", "3", "4", "5", "6"} & set(resume)
    calls.clear()
    V2ResultSink.seen.clear()
    executor = fresh_executor()
    executor.execute(deepcopy(resume), "v2-resume-high", execute_outputs=["16"])
    assert executor.success, executor.status_messages
    assert calls == ["high_4_8"]
    resumed_outputs = V2ResultSink.seen[-1]

    calls.clear()
    executor = fresh_executor()
    executor.execute(deepcopy(_graph()), "v2-full-control", execute_outputs=["16"])
    assert executor.success, executor.status_messages
    assert calls == ["low_0_4", "high_4_8"]
    assert V2ResultSink.seen[-1] == resumed_outputs

    calls.clear()
    corrupt = deepcopy(resume)
    corrupt["18"]["inputs"]["artifact_sha256"] = "0" * 64
    executor = fresh_executor()
    executor.execute(corrupt, "v2-bad-frozen-low", execute_outputs=["16"])
    assert executor.success is False
    assert calls == []

    effect_resume = deepcopy(resume)
    effect_resume["19"] = {"class_type": "MiniMaxH3StageEAVConfigEXPT8", "inputs": {
        "mode": "report_only", "tau": 0.2, "start_video_progress": 0.,
        "end_video_progress": 1., "max_workspace_mib": 32, "g_hard_limit": 3.}}
    effect_resume["20"] = {"class_type": "MiniMaxH3StageEAVApplyEXPT8", "inputs": {
        "model": ["12", 0], "sigmas": ["12", 2], "av_latent": ["11", 0],
        "stage_context": ["12", 3], "eav_config": ["19", 0]}}
    effect_resume["14"]["inputs"]["model"] = ["20", 0]
    effect_resume["21"] = {"class_type": "MiniMaxH3StageEAVAuditEXPT8", "inputs": {
        "av_latent": ["15", 0], "runtime": ["20", 1]}}
    reports = []
    audit = modular_nodes.audit_stage_eav

    def observed_audit(*args):
        value, report = audit(*args)
        reports.append(json.loads(report))
        return value, report

    monkeypatch.setattr(modular_nodes, "audit_stage_eav", observed_audit)
    executor = fresh_executor()
    executor.execute(effect_resume, "v2-resume-high-eav", execute_outputs=["16", "21"])
    assert executor.success, executor.status_messages
    assert calls == ["high_4_8"]
    assert reports[-1]["status"] == "observed_report_only"
    assert reports[-1]["completed_forwards"] == 4
    assert reports[-1]["selector_calls"] == 4


def test_fresh_core_high_only_executes_external_relay_and_eav(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3]))
    import execution
    import nodes
    from comfy_extras.nodes_custom_sampler import RandomNoise, BasicGuider

    for cls in (modular_nodes.MiniMaxH3FastH3V2StageSetupEXPT8,
                modular_nodes.MiniMaxH3StageSamplerEXPT8,
                modular_nodes.MiniMaxH3StageSaveEXPT8,
                modular_nodes.MiniMaxH3StageLoadEXPT8, *HANDOFF_NODES,
                modular_nodes.MiniMaxH3StageEAVConfigEXPT8,
                modular_nodes.MiniMaxH3StageEAVApplyEXPT8,
                modular_nodes.MiniMaxH3StageEAVAuditEXPT8,
                MiniMaxH3PromptRelayPlanT8Advanced,
                MiniMaxH3PromptRelayConditioningT8Advanced,
                MiniMaxH3TwoPassLatentReconcileT8Advanced, RandomNoise, BasicGuider,
                TinyV2Model, TinyCondition, TinyLongCondition, TinyRelayAssets, InterpolationDouble,
                V2ResultSink):
        monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, cls.define_schema().node_id, cls)
    monkeypatch.setattr(modular_nodes, "_stage_store_root", lambda: tmp_path)
    server = SimpleNamespace(client_id=None, last_node_id=None, sockets_metadata={},
                             send_sync=lambda *args, **kwargs: None)

    def executor():
        return execution.PromptExecutor(server, cache_args={"ram": 0., "ram_inactive": 0.},
                                        asset_manager=SimpleNamespace(enabled=False))

    freeze = _graph()
    freeze["2"]["class_type"] = "TinyLongCondition"
    freeze["17"] = {"class_type": "MiniMaxH3StageSaveEXPT8", "inputs": {
        "stage_result": ["6", 2], "prefix": "v2/LOW-relay"}}
    first = executor()
    first.execute(deepcopy(freeze), "v2-freeze-low-for-relay", execute_outputs=["17"])
    assert first.success, first.status_messages
    manifest = next(tmp_path.rglob("manifest.json"))

    resume = {key: value for key, value in _graph().items() if int(key) >= 7}
    resume["7"]["inputs"]["low_stage_result"] = ["18", 3]
    resume["18"] = {"class_type": "MiniMaxH3StageLoadEXPT8", "inputs": {
        "artifact_path": manifest.relative_to(tmp_path).as_posix(),
        "artifact_sha256": file_sha(manifest), "expected_stage": "low_0_4"}}
    resume["22"] = {"class_type": "TinyRelayAssets", "inputs": {}}
    resume["23"] = {"class_type": "MiniMaxH3PromptRelayPlanT8Advanced", "inputs": {
        "global_prompt": "One stable scene", "local_prompts": "A woman waves.\nShe walks away.",
        "length": 22, "timing_mode": "auto_equal", "time_ranges": "",
        "math_profile": "paper_v1", "epsilon": 0.1,
        "allow_gaps": False, "allow_overlaps": False}}
    resume["10"] = {"class_type": "MiniMaxH3PromptRelayConditioningT8Advanced", "inputs": {
        "model": ["9", 0], "clip": ["22", 0], "video_vae": ["22", 1],
        "audio_vae": ["22", 2], "prompt_relay_plan": ["23", 0],
        "width": ["8", 1], "height": ["8", 2], "task_type": "T2VA",
        "audio_mode": "native", "audio_denoise_strength": 0.35,
        "add_source_as_reference": False, "prompt_primary_audio_ordinal": 0,
        "strict_prompt_tags": True, "ref_image_size": "match",
        "reference_video_policy": "official_2_to_15s", "execution_mode": "apply_exp",
        "query_chunk_rows": 64}}
    resume["11"]["inputs"].update(highres_template=["10", 2], positive=["10", 1])
    resume["12"]["inputs"]["model"] = ["10", 0]
    resume["19"] = {"class_type": "MiniMaxH3StageEAVConfigEXPT8", "inputs": {
        "mode": "report_only", "tau": 0.2, "start_video_progress": 0.,
        "end_video_progress": 1., "max_workspace_mib": 32, "g_hard_limit": 3.}}
    resume["20"] = {"class_type": "MiniMaxH3StageEAVApplyEXPT8", "inputs": {
        "model": ["12", 0], "sigmas": ["12", 2], "av_latent": ["11", 0],
        "stage_context": ["12", 3], "eav_config": ["19", 0]}}
    resume["14"]["inputs"]["model"] = ["20", 0]
    resume["21"] = {"class_type": "MiniMaxH3StageEAVAuditEXPT8", "inputs": {
        "av_latent": ["15", 0], "runtime": ["20", 1]}}
    assert not {"1", "2", "3", "4", "5", "6"} & set(resume)

    calls = []
    receipts = []
    original = modular_nodes.sample_stage

    def observed(*args):
        calls.append(args[-1].stage)
        result = original(*args)
        receipts.append(result[2].verify())
        return result

    monkeypatch.setattr(modular_nodes, "sample_stage", observed)
    reports = []
    audit = modular_nodes.audit_stage_eav

    def observed_audit(*args):
        value, report = audit(*args)
        reports.append(json.loads(report))
        return value, report

    monkeypatch.setattr(modular_nodes, "audit_stage_eav", observed_audit)
    fresh = executor()
    fresh.execute(deepcopy(resume), "v2-frozen-low-high-relay-eav", execute_outputs=["16", "21"])
    assert fresh.success, fresh.status_messages
    assert calls == ["high_4_8"]
    assert receipts[-1]["portable_identity"] is True
    assert "relay" in receipts[-1]["request"]["model"]["stage_effects"]
    assert reports[-1]["status"] == "observed_report_only"
    assert reports[-1]["relay_attention_calls"] == 4
    assert reports[-1]["completed_forwards"] == 4
    assert reports[-1]["selector_calls"] == 4
