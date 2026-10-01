"""Run the saved three-window Relay/EAV branches under the real Core executor.

Only heavyweight source/asset construction and the media terminal are replaced
in memory. The saved candidate files and production effect/window nodes run.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import comfy.nested_tensor
from comfy_api.latest import io
import folder_paths
import pytest
import torch

from h3_audio_t8_pkg import chunked_two_pass_parity as parity
from h3_audio_t8_pkg import chunked_two_pass_upscale_advanced as legacy
from h3_audio_t8_pkg import sampling
from h3_audio_t8_pkg.modular_sampling.chunked_v5 import lift_standard_joint
from h3_audio_t8_pkg.modular_sampling import chunked_v5_nodes, chunked_v5_relay
from h3_audio_t8_pkg.modular_sampling.nodes import MiniMaxH3StageEAVConfigEXPT8
from h3_audio_t8_pkg.modular_sampling.results import _input_identity
from h3_audio_t8_pkg.nodes_prompt_relay_advanced import (
    MiniMaxH3PromptRelayConditioningT8Advanced,
    MiniMaxH3PromptRelayPlanT8Advanced,
)
from h3_audio_t8_pkg.prompt_relay_advanced import PROMPT_RELAY_BINDING_KEY
from helpers import FakeAudioVAE, FakeVideoVAE
from test_modular_chunked_v5_saved_storage_candidate_core import (
    ROOT, _executor, _prune_api, _register, _saved_four, _saved_three,
)
from test_chunked_two_pass_parity import _plan
from test_modular_chunked_stages import _fake_lift
from test_modular_speed_stages import _model
from test_prompt_relay_advanced import NativeLikeFakeClip


class TinyV5EffectInputs(io.ComfyNode):
    case = None

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, inputs=[], outputs=[
            io.Latent.Output("partial4_av"),
            io.Custom("T8_H3_CHUNKED_TWO_PASS_PLAN").Output("plan"),
            io.Noise.Output("noise"), io.Model.Output("raw_model"),
            io.Sampler.Output("sampler"), io.Sigmas.Output("sigmas"),
            io.Clip.Output("clip"), io.Vae.Output("video_vae"),
            io.Vae.Output("audio_vae"), io.Image.Output("frame"),
        ])

    @classmethod
    def execute(cls):
        return io.NodeOutput(*cls.case)


class TinyV5EffectReceipt(io.ComfyNode):
    seen = []

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, is_output_node=True, inputs=[
            io.String.Input("native_status"), io.String.Input("native_path"),
            io.String.Input("native_sha"), io.String.Input("native_manifest"),
            io.String.Input("window_path"), io.String.Input("window_sha"),
            io.String.Input("first_eav_report"), io.String.Input("second_eav_report"),
            io.Conditioning.Input("first_relay_positive"),
            io.Conditioning.Input("second_relay_positive"),
        ], outputs=[io.String.Output("report")])

    @classmethod
    def execute(cls, native_status, native_path, native_sha, native_manifest,
                window_path, window_sha, first_eav_report, second_eav_report,
                first_relay_positive, second_relay_positive):
        cls.seen.append({
            "receipt": (native_path, native_sha, native_manifest, window_path, window_sha),
            "native_status": native_status,
            "audits": (json.loads(first_eav_report), json.loads(second_eav_report)),
            "relay_tasks": (
                first_relay_positive[0][1][PROMPT_RELAY_BINDING_KEY]["task"],
                second_relay_positive[0][1][PROMPT_RELAY_BINDING_KEY]["task"],
            ),
        })
        return io.NodeOutput(native_status)


class TinyV5FourEffectReceipt(io.ComfyNode):
    seen = []

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, is_output_node=True, inputs=[
            io.String.Input("native_status"), io.String.Input("native_path"),
            io.String.Input("native_sha"), io.String.Input("native_manifest"),
            io.String.Input("window_path"), io.String.Input("window_sha"),
            io.String.Input("first_eav_report"), io.String.Input("second_eav_report"),
            io.String.Input("third_eav_report"),
            io.Conditioning.Input("first_relay_positive"),
            io.Conditioning.Input("second_relay_positive"),
            io.Conditioning.Input("third_relay_positive"),
        ], outputs=[io.String.Output("report")])

    @classmethod
    def execute(cls, native_status, native_path, native_sha, native_manifest,
                window_path, window_sha, first_eav_report, second_eav_report,
                third_eav_report, first_relay_positive, second_relay_positive,
                third_relay_positive):
        cls.seen.append({
            "receipt": (native_path, native_sha, native_manifest, window_path, window_sha),
            "native_status": native_status,
            "audits": tuple(json.loads(report) for report in (
                first_eav_report, second_eav_report, third_eav_report,
            )),
            "relay_tasks": tuple(positive[0][1][PROMPT_RELAY_BINDING_KEY]["task"]
                                 for positive in (first_relay_positive,
                                                  second_relay_positive,
                                                  third_relay_positive)),
        })
        return io.NodeOutput(native_status)


class TinyV5EffectSink(io.ComfyNode):
    seen = []

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, is_output_node=True, inputs=[
            io.Latent.Input("av_latent"), io.String.Input("eav_report"),
            io.Conditioning.Input("relay_positive"),
        ], outputs=[io.String.Output("report")])

    @classmethod
    def execute(cls, av_latent, eav_report, relay_positive):
        video, audio = av_latent["samples"].unbind()
        cls.seen.append({
            "audit": json.loads(eav_report),
            "av_identity": (_input_identity(video), _input_identity(audio)),
            "relay_task": relay_positive[0][1][PROMPT_RELAY_BINDING_KEY]["task"],
        })
        return io.NodeOutput("observed")


def _case(monkeypatch, *, window_count=3):
    monkeypatch.setattr(legacy, "learned_upscale_h3_av_latent", _fake_lift)
    source = {"samples": comfy.nested_tensor.NestedTensor((
        torch.zeros(1, 24, 57, 2, 2), torch.zeros(1, 32, 2, 320),
    ))}
    plan = _plan(temporal_chunk_frames=85 if window_count == 4 else 102,
                 temporal_overlap_frames=34)

    class ZeroNoise:
        seed = 23

        def generate_noise(self, latent):
            return comfy.nested_tensor.NestedTensor(tuple(
                torch.zeros_like(value) for value in latent["samples"].unbind()
            ))

    lifted, receipt, _report = lift_standard_joint(source, plan)
    assert len(receipt.segments) == window_count
    raw_model, sampler, _sigmas = sampling.setup_dual_clock_sampling(
        _model(), lifted, 4, 12., 3.,
    )
    sigmas = torch.tensor(parity.UPSTREAM_REFINE_VIDEO_SIGMAS[4], dtype=torch.float32)
    noise = ZeroNoise()
    return (source, plan, noise, raw_model, sampler, sigmas,
            NativeLikeFakeClip(), FakeVideoVAE(), FakeAudioVAE(),
            torch.zeros((1, 64, 64, 3)))


def _effect_graph(saved, *, resume, receipt=None):
    graph = deepcopy(saved)
    graph["60"] = {"class_type": "TinyV5EffectInputs", "inputs": {}}
    source = ["49", 0] if resume else ["60", 0]
    if not resume:
        # The first pass was already completed. Keep the saved candidate's
        # native Save, global lift, window storage and both effect branches.
        graph["47"]["inputs"].update(av_latent=source, confirm_save=True)
        graph["48"]["inputs"].update(
            partial4_denoised_output=source, plan=["60", 1], confirm_save=True,
        )
    graph["28"]["inputs"].update(partial4_denoised_output=source, plan=["60", 1])
    graph["29"]["inputs"].update(
        partial4_denoised_output=source, plan=["60", 1], noise=["60", 2],
    )
    graph["34"]["inputs"].update(
        model=["60", 3], clip=["60", 6], video_vae=["60", 7],
        audio_vae=["60", 8], width=64, height=64,
        first_frame=["60", 9], last_frame=["60", 9],
    )
    projects = ("37",) if resume else ("35", "36")
    effects = ("45",) if resume else ("39", "42")
    windows = ("32",) if resume else ("30", "31")
    for project in projects:
        graph[project]["inputs"].update(
            raw_high_model=["60", 3], relay_model=["34", 0],
            relay_positive=["34", 1], relay_full_av_latent=["34", 2],
            prompt_relay_plan=["33", 0], partial4_denoised_output=source,
            plan=["60", 1], sigmas=["60", 5],
        )
    for effect in effects:
        graph[effect]["inputs"].update(
            sigmas=["60", 5], partial4_denoised_output=source, plan=["60", 1],
        )
    for window in windows:
        graph[window]["inputs"].update(
            partial4_denoised_output=source, plan=["60", 1],
            noise=["60", 2], sampler=["60", 4], sigmas=["60", 5],
        )
    if resume:
        graph["47"]["inputs"].update(
            checkpoint_path=receipt[0], expected_file_sha256=receipt[1],
            expected_manifest_json=receipt[2],
        )
        graph["48"]["inputs"].update(
            partial4_denoised_output=source, plan=["60", 1],
            artifact_path=receipt[3], artifact_sha256=receipt[4],
        )
        graph["61"] = {"class_type": "TinyV5EffectSink", "inputs": {
            "av_latent": ["46", 0], "eav_report": ["46", 1],
            "relay_positive": ["37", 1],
        }}
    else:
        graph["61"] = {"class_type": "TinyV5EffectReceipt", "inputs": {
            "native_status": ["47", 1], "native_path": ["47", 2],
            "native_sha": ["47", 3], "native_manifest": ["47", 4],
            "window_path": ["48", 2], "window_sha": ["48", 3],
            "first_eav_report": ["40", 1], "second_eav_report": ["43", 1],
            "first_relay_positive": ["35", 1],
            "second_relay_positive": ["36", 1],
        }}
    graph = _prune_api(graph, ("61",))
    if resume:
        assert not {"12", "30", "31", "35", "36", "39", "42"} & set(graph)
    else:
        assert "32" not in graph
    return graph


def _four_effect_graph(saved, *, resume, receipt=None):
    graph = deepcopy(saved)
    graph["60"] = {"class_type": "TinyV5EffectInputs", "inputs": {}}
    source = ["54", 0] if resume else ["60", 0]
    if not resume:
        graph["52"]["inputs"].update(av_latent=source, confirm_save=True)
        graph["53"]["inputs"].update(
            partial4_denoised_output=source, plan=["60", 1], confirm_save=True,
        )
    graph["28"]["inputs"].update(partial4_denoised_output=source, plan=["60", 1])
    graph["29"]["inputs"].update(
        partial4_denoised_output=source, plan=["60", 1], noise=["60", 2],
    )
    graph["35"]["inputs"].update(
        model=["60", 3], clip=["60", 6], video_vae=["60", 7],
        audio_vae=["60", 8], width=64, height=64,
        first_frame=["60", 9], last_frame=["60", 9],
    )
    projects = ("39",) if resume else ("36", "37", "38")
    effects = ("50",) if resume else ("41", "44", "47")
    windows = ("33",) if resume else ("30", "31", "32")
    for project in projects:
        graph[project]["inputs"].update(
            raw_high_model=["60", 3], relay_model=["35", 0],
            relay_positive=["35", 1], relay_full_av_latent=["35", 2],
            prompt_relay_plan=["34", 0], partial4_denoised_output=source,
            plan=["60", 1], sigmas=["60", 5],
        )
    for effect in effects:
        graph[effect]["inputs"].update(
            sigmas=["60", 5], partial4_denoised_output=source, plan=["60", 1],
        )
    for window in windows:
        graph[window]["inputs"].update(
            partial4_denoised_output=source, plan=["60", 1],
            noise=["60", 2], sampler=["60", 4], sigmas=["60", 5],
        )
    if resume:
        graph["52"]["inputs"].update(
            checkpoint_path=receipt[0], expected_file_sha256=receipt[1],
            expected_manifest_json=receipt[2],
        )
        graph["53"]["inputs"].update(
            partial4_denoised_output=source, plan=["60", 1],
            artifact_path=receipt[3], artifact_sha256=receipt[4],
        )
        graph["61"] = {"class_type": "TinyV5EffectSink", "inputs": {
            "av_latent": ["51", 0], "eav_report": ["51", 1],
            "relay_positive": ["39", 1],
        }}
    else:
        graph["61"] = {"class_type": "TinyV5FourEffectReceipt", "inputs": {
            "native_status": ["52", 1], "native_path": ["52", 2],
            "native_sha": ["52", 3], "native_manifest": ["52", 4],
            "window_path": ["53", 2], "window_sha": ["53", 3],
            "first_eav_report": ["42", 1],
            "second_eav_report": ["45", 1],
            "third_eav_report": ["48", 1],
            "first_relay_positive": ["36", 1],
            "second_relay_positive": ["37", 1],
            "third_relay_positive": ["38", 1],
        }}
    graph = _prune_api(graph, ("61",))
    if resume:
        assert not {"12", "30", "31", "32", "36", "37", "38",
                    "41", "44", "47"} & set(graph)
    else:
        assert "33" not in graph
    return graph


def _register_effects(patch):
    patch.syspath_prepend(str(ROOT.parents[1]))
    import nodes

    _register(patch)
    for cls in (MiniMaxH3PromptRelayPlanT8Advanced,
                MiniMaxH3PromptRelayConditioningT8Advanced,
                MiniMaxH3StageEAVConfigEXPT8, TinyV5EffectInputs,
                TinyV5EffectReceipt, TinyV5FourEffectReceipt,
                TinyV5EffectSink):
        patch.setitem(nodes.NODE_CLASS_MAPPINGS, cls.define_schema().node_id, cls)


def _cold_effect_resume(output_root, receipt):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
    torch.set_num_threads(1)
    with pytest.MonkeyPatch.context() as patch:
        _register_effects(patch)
        patch.setattr(folder_paths, "get_output_directory", lambda: output_root)
        TinyV5EffectInputs.case = _case(patch)
        TinyV5EffectSink.seen.clear()
        graph = _effect_graph(_saved_three("02_resume_window_2_DRAFT"),
                              resume=True, receipt=receipt)
        executor = _executor()
        executor.execute(graph, "v5-three-effects-cold-resume", execute_outputs=["61"])
        assert executor.success, executor.status_messages
        assert not torch.cuda.is_initialized()
        return TinyV5EffectSink.seen[-1]


def _cold_four_effect_resume(output_root, receipt):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
    torch.set_num_threads(1)
    with pytest.MonkeyPatch.context() as patch:
        _register_effects(patch)
        patch.setattr(folder_paths, "get_output_directory", lambda: output_root)
        TinyV5EffectInputs.case = _case(patch, window_count=4)
        TinyV5EffectSink.seen.clear()
        graph = _four_effect_graph(
            _saved_four("02_resume_window_3_DRAFT"), resume=True, receipt=receipt,
        )
        executor = _executor()
        executor.execute(graph, "v5-four-effects-cold-resume", execute_outputs=["61"])
        assert executor.success, executor.status_messages
        assert not torch.cuda.is_initialized()
        return TinyV5EffectSink.seen[-1]


def test_saved_four_window_effect_branches_execute_and_resume(monkeypatch, tmp_path):
    _register_effects(monkeypatch)
    torch.set_num_threads(1)
    output_root = tmp_path / "output"
    output_root.mkdir()
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(output_root))
    TinyV5EffectInputs.case = _case(monkeypatch, window_count=4)
    TinyV5FourEffectReceipt.seen.clear()
    TinyV5EffectSink.seen.clear()
    freeze = _four_effect_graph(_saved_four("01_freeze_window_2"), resume=False)
    executor = _executor()
    executor.execute(freeze, "v5-four-effects-freeze", execute_outputs=["61"])
    assert executor.success, executor.status_messages
    frozen = TinyV5FourEffectReceipt.seen[-1]
    assert frozen["native_status"] == "SAVED_VERIFIED"
    assert frozen["relay_tasks"] == ("i2va", "t2va", "t2va")
    for index, audit in enumerate(frozen["audits"]):
        assert audit["chunked_v5_window_index"] == index
        assert audit["status"] == "observed_report_only", audit
        assert audit["relay_required"] is True
        assert audit["relay_attention_calls"] > 0
        assert audit["completed_forwards"] == audit["planned_forwards"] == 4
    receipt = frozen["receipt"]

    bad_receipt = (*receipt[:4], "0" * 64)
    rejected = _four_effect_graph(
        _saved_four("02_resume_window_3_DRAFT"), resume=True,
        receipt=bad_receipt,
    )
    executor = _executor()
    executor.execute(rejected, "v5-four-effects-reject-window-sha",
                     execute_outputs=["61"])
    assert not executor.success
    assert not TinyV5EffectSink.seen

    sample_calls = []
    original_sample = chunked_v5_nodes.sample_standard_window

    def observed_sample(*args, **kwargs):
        sample_calls.append(int(args[9]))
        return original_sample(*args, **kwargs)

    monkeypatch.setattr(chunked_v5_nodes, "sample_standard_window", observed_sample)
    resume = _four_effect_graph(
        _saved_four("02_resume_window_3_DRAFT"), resume=True, receipt=receipt,
    )
    executor = _executor()
    executor.execute(deepcopy(resume), "v5-four-effects-resume", execute_outputs=["61"])
    assert executor.success, executor.status_messages
    assert sample_calls == [3]
    result = TinyV5EffectSink.seen[-1]
    assert result["relay_task"] == "l2va"
    audit = result["audit"]
    assert audit["chunked_v5_window_index"] == 3
    assert audit["status"] == "observed_report_only", audit
    assert audit["relay_required"] is True
    assert audit["relay_attention_calls"] > 0
    assert audit["completed_forwards"] == audit["planned_forwards"] == 4

    child_code = """
import json,runpy,sys
import comfy.options
comfy.options.enable_args_parsing()
args=sys.argv[1:]
sys.argv=['v5-four-effects-cold','--cpu','--use-pytorch-cross-attention']
import comfy.cli_args
sys.argv=['v5-four-effects-cold']
runpy.run_path('tests/conftest.py')
from test_modular_chunked_v5_saved_effects_core import _cold_four_effect_resume
print('RESULT='+json.dumps(_cold_four_effect_resume(args[0],args[1:6]),sort_keys=True))
"""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    env["PYTHONPATH"] = str(ROOT.parents[1])
    child = subprocess.run(
        [sys.executable, "-c", child_code, str(output_root), *receipt],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=90,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    cold = json.loads(next(line[7:] for line in child.stdout.splitlines()
                           if line.startswith("RESULT=")))
    assert cold["av_identity"] == list(result["av_identity"])
    assert cold["relay_task"] == "l2va"
    cold_audit = cold["audit"]
    assert cold_audit["chunked_v5_window_index"] == 3
    assert cold_audit["status"] == "observed_report_only"
    assert cold_audit["relay_required"] is True
    assert cold_audit["relay_attention_calls"] > 0
    assert cold_audit["completed_forwards"] == cold_audit["planned_forwards"] == 4

    applied = deepcopy(resume)
    applied["49"]["inputs"].update(
        mode="apply_exp", tau=0.5, start_video_progress=0.1, g_hard_limit=3.0,
    )
    executor.execute(applied, "v5-four-effects-last-window-apply",
                     execute_outputs=["61"])
    assert executor.success, executor.status_messages
    assert sample_calls == [3, 3]
    changed = TinyV5EffectSink.seen[-1]
    applied_audit = changed["audit"]
    assert applied_audit["chunked_v5_window_index"] == 3
    assert applied_audit["status"] == "observed_apply_exp", applied_audit
    assert applied_audit["relay_required"] is True
    assert applied_audit["relay_attention_calls"] > 0
    assert applied_audit["feta"]["active_forward_count"] > 0
    assert changed["av_identity"][0] != result["av_identity"][0]
    native_file = Path(output_root) / "MiniMaxH3/latent_checkpoints" / receipt[0]
    window_file = Path(output_root) / "MiniMaxH3/chunked_v5_window_artifacts" / receipt[3]
    assert hashlib.sha256(native_file.read_bytes()).hexdigest() == receipt[1].lower()
    assert hashlib.sha256(window_file.read_bytes()).hexdigest() == receipt[4].lower()


def test_saved_three_window_effect_branches_execute_and_resume(monkeypatch, tmp_path):
    _register_effects(monkeypatch)
    torch.set_num_threads(1)
    output_root = tmp_path / "output"
    output_root.mkdir()
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(output_root))
    TinyV5EffectInputs.case = _case(monkeypatch)
    TinyV5EffectReceipt.seen.clear()
    TinyV5EffectSink.seen.clear()
    freeze = _effect_graph(_saved_three("01_freeze_window_1"), resume=False)
    executor = _executor()
    executor.execute(freeze, "v5-three-effects-freeze", execute_outputs=["61"])
    assert executor.success, executor.status_messages
    result = TinyV5EffectReceipt.seen[-1]
    assert result["native_status"] == "SAVED_VERIFIED"
    assert result["relay_tasks"] == ("i2va", "t2va")
    receipt = result["receipt"]
    for index, audit in enumerate(result["audits"]):
        assert audit["chunked_v5_window_index"] == index
        assert audit["status"] == "observed_report_only", audit
        assert audit["relay_required"] is True
        assert audit["relay_attention_calls"] > 0
        assert audit["completed_forwards"] == audit["planned_forwards"] == 4

    resume = _effect_graph(_saved_three("02_resume_window_2_DRAFT"), resume=True,
                           receipt=receipt)
    bad_receipt = (*receipt[:4], "0" * 64)
    rejected = _effect_graph(_saved_three("02_resume_window_2_DRAFT"), resume=True,
                             receipt=bad_receipt)
    executor = _executor()
    executor.execute(rejected, "v5-three-effects-reject-window-sha",
                     execute_outputs=["61"])
    assert not executor.success
    assert not TinyV5EffectSink.seen
    sample_calls = []
    original_sample = chunked_v5_nodes.sample_standard_window

    def observed_sample(*args, **kwargs):
        sample_calls.append(int(args[9]))
        return original_sample(*args, **kwargs)

    monkeypatch.setattr(chunked_v5_nodes, "sample_standard_window", observed_sample)
    executor = _executor()
    executor.execute(deepcopy(resume), "v5-three-effects-resume", execute_outputs=["61"])
    assert executor.success, executor.status_messages
    assert sample_calls == [2]
    result = TinyV5EffectSink.seen[-1]
    audit = result["audit"]
    assert result["relay_task"] == "l2va"
    assert audit["chunked_v5_window_index"] == 2
    assert audit["status"] == "observed_report_only", audit
    assert audit["relay_required"] is True
    assert audit["relay_attention_calls"] > 0
    assert audit["completed_forwards"] == audit["planned_forwards"] == 4
    executor.execute(deepcopy(resume), "v5-three-effects-resume-repeat",
                     execute_outputs=["61"])
    assert executor.success, executor.status_messages
    assert sample_calls == [2]

    child_code = """
import json,runpy,sys
import comfy.options
comfy.options.enable_args_parsing()
args=sys.argv[1:]
sys.argv=['v5-effects-cold','--cpu','--use-pytorch-cross-attention']
import comfy.cli_args
sys.argv=['v5-effects-cold']
runpy.run_path('tests/conftest.py')
from test_modular_chunked_v5_saved_effects_core import _cold_effect_resume
print('RESULT='+json.dumps(_cold_effect_resume(args[0],args[1:6]),sort_keys=True))
"""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    env["PYTHONPATH"] = str(ROOT.parents[1])
    child = subprocess.run(
        [sys.executable, "-c", child_code, str(output_root), *receipt],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=90,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    cold = json.loads(next(line[7:] for line in child.stdout.splitlines()
                           if line.startswith("RESULT=")))
    assert cold["av_identity"] == list(result["av_identity"])
    assert cold["relay_task"] == "l2va"
    cold_audit = cold["audit"]
    assert cold_audit["chunked_v5_window_index"] == 2
    assert cold_audit["status"] == "observed_report_only"
    assert cold_audit["relay_required"] is True
    assert cold_audit["relay_attention_calls"] > 0
    assert cold_audit["completed_forwards"] == cold_audit["planned_forwards"] == 4

    native_file = Path(output_root) / "MiniMaxH3/latent_checkpoints" / receipt[0]
    window_file = Path(output_root) / "MiniMaxH3/chunked_v5_window_artifacts" / receipt[3]
    assert hashlib.sha256(native_file.read_bytes()).hexdigest() == receipt[1].lower()
    assert hashlib.sha256(window_file.read_bytes()).hexdigest() == receipt[4].lower()
    frozen_manifest = json.loads(window_file.read_text(encoding="utf-8"))
    assert frozen_manifest["implementation"]["chunked_v5_relay.py"] == (
        hashlib.sha256(Path(chunked_v5_relay.__file__).read_bytes()).hexdigest()
    )
    applied = deepcopy(resume)
    applied["44"]["inputs"].update(
        mode="apply_exp", tau=0.5, start_video_progress=0.1, g_hard_limit=3.0,
    )
    executor.execute(applied, "v5-three-effects-last-window-apply",
                     execute_outputs=["61"])
    assert executor.success, executor.status_messages
    assert sample_calls == [2, 2]
    changed = TinyV5EffectSink.seen[-1]
    applied_audit = changed["audit"]
    assert applied_audit["chunked_v5_window_index"] == 2
    assert applied_audit["status"] == "observed_apply_exp", applied_audit
    assert applied_audit["relay_required"] is True
    assert applied_audit["relay_attention_calls"] > 0
    assert applied_audit["feta"]["active_forward_count"] > 0
    assert changed["av_identity"][0] != result["av_identity"][0]
    executor.execute(deepcopy(applied), "v5-three-effects-last-window-apply-repeat",
                     execute_outputs=["61"])
    assert executor.success, executor.status_messages
    assert sample_calls == [2, 2]
    assert hashlib.sha256(native_file.read_bytes()).hexdigest() == receipt[1].lower()
    assert hashlib.sha256(window_file.read_bytes()).hexdigest() == receipt[4].lower()
