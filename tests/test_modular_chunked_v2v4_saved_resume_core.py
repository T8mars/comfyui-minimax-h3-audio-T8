"""Run saved v2-v4 branches with tiny inputs and native AV checkpoint I/O."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import comfy.nested_tensor
import comfy.model_management
from comfy_api.latest import io
import folder_paths
import pytest
import torch

from h3_audio_t8_pkg import chunked_two_pass_upscale_advanced as old
from h3_audio_t8_pkg.modular_sampling import chunked_stage_nodes
from h3_audio_t8_pkg.modular_sampling.chunked_source_nodes import NODES as SOURCE_NODES
from h3_audio_t8_pkg.modular_sampling.chunked_stage_nodes import NODES as STAGE_NODES
from h3_audio_t8_pkg.modular_sampling.nodes import MiniMaxH3StageEAVConfigEXPT8
from h3_audio_t8_pkg.nodes_native_latent_checkpoint_advanced import (
    MiniMaxH3NativeLatentCheckpointLoadT8Advanced,
    MiniMaxH3NativeLatentCheckpointSaveT8Advanced,
)
from h3_audio_t8_pkg.prompt_relay_advanced import PROMPT_RELAY_PLAN_TYPE
from test_chunked_two_pass_global_noise_advanced import _plan
from test_modular_chunked_v1_relay import _case
from tools.build_modular_chunked_v1_storage_workflow import _prune_api
from tools.build_modular_chunked_v2v4_resume_workflow import ROOT, build_pair


CANDIDATES = ROOT / "artifacts/development/modular-sampling-m4-chunked-v2v4-resume-20260924/candidate-v2"
HASHES = {
    "v2": ("f1a567dd77e65dc1f8dc969e4549e57b0ee676797cd2caa3a26e0897f530431a",
           "2d935d4ec3159f7babb3fe1593c4839a42c45ddcdd6852ca6bef3847ff620aff"),
    "v3": ("45c6e6c9cfddf0577e78865feb000a9192d5a14c4142e6e0b63c7dc7057e0380",
           "7313d04a7eb035f66a8e55835b8167365e1555d31640cf08e7d0d456966b4241"),
    "v4": ("153f1e47c35cd927ee82ff3b0f0497367d03b7ebb0408e6b219c13884f508dbc",
           "2ca8a3e5a38bbcfe9da2ed0f31628d2f80c082e944c79893027aed627ec171ea"),
}


class TinyChunkedV2V4Inputs(io.ComfyNode):
    case = None

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, inputs=[], outputs=[
            io.Latent.Output("first_pass_latent"),
            io.Custom("T8_H3_CHUNKED_TWO_PASS_PLAN").Output("plan"),
            io.Noise.Output("noise"), io.Model.Output("relay_model"),
            io.Conditioning.Output("relay_positive"),
            io.Latent.Output("relay_av_latent"),
            io.Custom(PROMPT_RELAY_PLAN_TYPE).Output("prompt_relay_plan"),
            io.Sampler.Output("sampler"), io.Sigmas.Output("sigmas"),
        ])

    @classmethod
    def execute(cls):
        return io.NodeOutput(*cls.case)


class TinyChunkedV2V4FirstPass(io.ComfyNode):
    calls = []

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, inputs=[io.Latent.Input("av_latent")],
                         outputs=[io.Latent.Output("output"),
                                  io.Latent.Output("denoised_output")])

    @classmethod
    def execute(cls, av_latent):
        cls.calls.append(1)
        return io.NodeOutput(av_latent, av_latent)


class TinyChunkedV2V4Receipt(io.ComfyNode):
    seen = []

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, is_output_node=True,
                         inputs=[io.String.Input("status"), io.String.Input("path"),
                                 io.String.Input("sha"), io.String.Input("manifest")],
                         outputs=[io.String.Output("report")])

    @classmethod
    def execute(cls, status, path, sha, manifest):
        cls.seen.append((status, path, sha, manifest))
        return io.NodeOutput(status)


class TinyChunkedV2V4Sink(io.ComfyNode):
    seen = []

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, is_output_node=True,
                         inputs=[io.Latent.Input("av_latent"),
                                 io.String.Input("audit_json")],
                         outputs=[io.String.Output("report")])

    @classmethod
    def execute(cls, av_latent, audit_json):
        video, audio = av_latent["samples"].unbind()
        digests = tuple(hashlib.sha256(item.detach().contiguous().numpy().tobytes()).hexdigest()
                        for item in (video, audio))
        cls.seen.append((digests, json.loads(audit_json)))
        return io.NodeOutput(audit_json)


def _saved(variant, label, expected_sha, expected):
    path = CANDIDATES / variant / f"{label}.api.json"
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == expected_sha
    graph = json.loads(raw)
    assert graph == expected
    return graph


def _inputs(monkeypatch, variant):
    raw, sampler, sigmas, relay_model, positive, relay_av, relay_plan, source, *_rest = (
        _case(monkeypatch, with_frames=True))
    assert raw is not None
    if variant == "v4":
        source = dict(source)
        video, audio = source["samples"].unbind()
        video_mask = torch.linspace(0.2, 1., video.shape[2]).reshape(1, 1, -1, 1, 1)
        source["noise_mask"] = comfy.nested_tensor.NestedTensor((
            video_mask.expand(1, 1, video.shape[2], video.shape[3], video.shape[4]).clone(),
            torch.ones_like(audio)))
    builder = {"v2": old.build_chunked_two_pass_global_noise_plan,
               "v3": old.build_chunked_two_pass_low_sigma_plan,
               "v4": old.build_chunked_two_pass_masked_low_sigma_plan}[variant]
    plan = _plan(builder, strategy="full_frame_safe", temporal_strategy="full_clip_safe")
    noise = _rest[1]
    return (source, plan, noise, relay_model, positive, relay_av, relay_plan,
            sampler, sigmas)


def _replace_large_upstream(graph, *, resume, receipt=None):
    graph = deepcopy(graph)
    graph["60"] = {"class_type": "TinyChunkedV2V4Inputs", "inputs": {}}
    if not resume:
        graph["12"] = {"class_type": "TinyChunkedV2V4FirstPass",
                       "inputs": {"av_latent": ["60", 0]}}
    for node in graph.values():
        for name, value in node["inputs"].items():
            if value == ["14", 0]:
                node["inputs"][name] = ["60", 1]
    by_type = {node["class_type"]: key for key, node in graph.items()}
    if not resume and "MiniMaxH3NativeLatentCheckpointSaveT8Advanced" in by_type:
        save = graph[by_type["MiniMaxH3NativeLatentCheckpointSaveT8Advanced"]]
        save["inputs"].update(confirm_save=True)
        graph["61"] = {"class_type": "TinyChunkedV2V4Receipt", "inputs": {
            "status": [by_type["MiniMaxH3NativeLatentCheckpointSaveT8Advanced"], 1],
            "path": [by_type["MiniMaxH3NativeLatentCheckpointSaveT8Advanced"], 2],
            "sha": [by_type["MiniMaxH3NativeLatentCheckpointSaveT8Advanced"], 3],
            "manifest": [by_type["MiniMaxH3NativeLatentCheckpointSaveT8Advanced"], 4]}}
        return _prune_api(graph, ("61",))
    if resume:
        load = graph[by_type["MiniMaxH3NativeLatentCheckpointLoadT8Advanced"]]
        load["inputs"].update(checkpoint_path=receipt[0],
                              expected_file_sha256=receipt[1],
                              expected_manifest_json=receipt[2])
    else:
        for kind in ("MiniMaxH3ChunkedSourceSegmentEXPT8",
                     "MiniMaxH3ChunkedPass2PrepareEXPT8"):
            graph[by_type[kind]]["inputs"]["first_pass_latent"] = ["12", 1]
    graph[by_type["MiniMaxH3ChunkedPass2PrepareEXPT8"]]["inputs"]["noise"] = ["60", 2]
    bind = graph[by_type["MiniMaxH3ChunkedPass2RelayBindEXPT8"]]["inputs"]
    bind.update(relay_model=["60", 3], relay_positive=["60", 4],
                relay_av_latent=["60", 5], prompt_relay_plan=["60", 6],
                sigmas=["60", 8])
    apply = graph[by_type["MiniMaxH3ChunkedPass2EAVApplyEXPT8"]]["inputs"]
    apply["sigmas"] = ["60", 8]
    sampled = graph[by_type["MiniMaxH3ChunkedPass2SegmentEXPT8"]]["inputs"]
    sampled.update(noise=["60", 2], sampler=["60", 7], sigmas=["60", 8])
    audit = by_type["MiniMaxH3ChunkedPass2EAVAuditEXPT8"]
    graph["61"] = {"class_type": "TinyChunkedV2V4Sink", "inputs": {
        "av_latent": [audit, 0], "audit_json": [audit, 1]}}
    return _prune_api(graph, ("61",))


def _executor():
    import execution

    server = SimpleNamespace(client_id=None, last_node_id=None, sockets_metadata={},
                             send_sync=lambda *args, **kwargs: None)
    return execution.PromptExecutor(
        server, cache_args={"ram": 0., "ram_inactive": 0.},
        asset_manager=SimpleNamespace(enabled=False))


def _run(graph, label):
    executor = _executor()
    executor.execute(deepcopy(graph), label, execute_outputs=["61"])
    assert executor.success, executor.status_messages


def _register_nodes(patch):
    import nodes

    for cls in (*SOURCE_NODES, *STAGE_NODES, MiniMaxH3StageEAVConfigEXPT8,
                MiniMaxH3NativeLatentCheckpointSaveT8Advanced,
                MiniMaxH3NativeLatentCheckpointLoadT8Advanced,
                TinyChunkedV2V4Inputs, TinyChunkedV2V4FirstPass,
                TinyChunkedV2V4Receipt, TinyChunkedV2V4Sink):
        patch.setitem(nodes.NODE_CLASS_MAPPINGS, cls.define_schema().node_id, cls)


def _cold_resume(output_root, variant, receipt):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
    torch.set_num_threads(1)
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(ROOT.parents[1]))
        _register_nodes(patch)
        patch.setattr(folder_paths, "get_output_directory", lambda: output_root)
        TinyChunkedV2V4Inputs.case = _inputs(patch, variant)
        TinyChunkedV2V4FirstPass.calls.clear()
        TinyChunkedV2V4Sink.seen.clear()
        expected_resume = build_pair(variant)[1][1]
        resume = _saved(variant, "02_resume_high_only_DRAFT", HASHES[variant][1],
                        expected_resume)
        graph = _replace_large_upstream(resume, resume=True, receipt=receipt)
        assert "12" not in graph
        _run(graph, f"{variant}-cold-resume")
        return {"av_sha256": TinyChunkedV2V4Sink.seen[-1][0],
                "first_pass_calls": len(TinyChunkedV2V4FirstPass.calls),
                "relay_attention_calls": TinyChunkedV2V4Sink.seen[-1][1][
                    "relay_attention_calls"],
                "cuda_initialized": torch.cuda.is_initialized()}


@pytest.mark.parametrize("variant", ("v2", "v3", "v4"))
def test_saved_graph_native_freeze_then_high_only_executor(monkeypatch, tmp_path, variant):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3]))
    _register_nodes(monkeypatch)
    torch.set_num_threads(1)
    output_root = tmp_path / "output"
    output_root.mkdir()
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(output_root))
    TinyChunkedV2V4Inputs.case = _inputs(monkeypatch, variant)
    cuda_initialized_before = torch.cuda.is_initialized()
    TinyChunkedV2V4FirstPass.calls.clear()
    TinyChunkedV2V4Receipt.seen.clear()
    TinyChunkedV2V4Sink.seen.clear()
    (expected_freeze, expected_resume) = build_pair(variant)
    freeze = _saved(variant, "01_freeze_first_pass", HASHES[variant][0], expected_freeze[1])
    resume = _saved(variant, "02_resume_high_only_DRAFT", HASHES[variant][1],
                    expected_resume[1])
    baseline = _replace_large_upstream(resume, resume=False)
    _run(baseline, f"{variant}-tiny-full")
    expected = TinyChunkedV2V4Sink.seen[-1]
    assert len(TinyChunkedV2V4FirstPass.calls) == 1

    frozen = _replace_large_upstream(freeze, resume=False)
    _run(frozen, f"{variant}-tiny-freeze")
    assert len(TinyChunkedV2V4FirstPass.calls) == 2
    assert len(TinyChunkedV2V4Receipt.seen) == 1
    status, path, sha, manifest = TinyChunkedV2V4Receipt.seen[0]
    assert status == "SAVED_VERIFIED"
    receipt = (path, sha, manifest)
    checkpoint = output_root / "MiniMaxH3" / "latent_checkpoints" / path
    assert checkpoint.is_file() and hashlib.sha256(checkpoint.read_bytes()).hexdigest() == sha.lower()

    recovered = _replace_large_upstream(resume, resume=True, receipt=receipt)
    assert "12" not in recovered
    _run(recovered, f"{variant}-tiny-resume")
    assert len(TinyChunkedV2V4FirstPass.calls) == 2
    assert TinyChunkedV2V4Sink.seen[-1][0] == expected[0]
    assert TinyChunkedV2V4Sink.seen[-1][1]["relay_attention_calls"] > 0
    assert torch.cuda.is_initialized() is cuda_initialized_before

    child_code = """
import json,runpy,sys
import comfy.options
comfy.options.enable_args_parsing()
args=sys.argv[1:]
sys.argv=['chunked-cold-resume','--cpu','--use-pytorch-cross-attention']
import comfy.cli_args
sys.argv=['chunked-cold-resume']
runpy.run_path('tests/conftest.py')
from test_modular_chunked_v2v4_saved_resume_core import _cold_resume
print('RESULT='+json.dumps(_cold_resume(args[0],args[1],args[2:5]),sort_keys=True))
"""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    env["PYTHONPATH"] = str(ROOT.parents[1])
    child = subprocess.run(
        [sys.executable, "-c", child_code, str(output_root), variant, *receipt],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=90,
    )
    assert child.returncode == 0, child.stderr
    cold = json.loads(next(line[7:] for line in child.stdout.splitlines()
                           if line.startswith("RESULT=")))
    assert cold == {"av_sha256": list(expected[0]), "first_pass_calls": 0,
                    "relay_attention_calls": expected[1]["relay_attention_calls"],
                    "cuda_initialized": False}

    sample_calls = []
    original_sample = chunked_stage_nodes.sample_chunked_pass2

    def observed_sample(*args):
        sample_calls.append(args[4].index)
        return original_sample(*args)

    monkeypatch.setattr(chunked_stage_nodes, "sample_chunked_pass2", observed_sample)
    for field, value in (("expected_file_sha256", "0" * 64),
                         ("expected_manifest_json", "{}")):
        invalid = deepcopy(recovered)
        load_key = next(key for key, node in invalid.items()
                        if node["class_type"] ==
                        "MiniMaxH3NativeLatentCheckpointLoadT8Advanced")
        invalid[load_key]["inputs"][field] = value
        failed = _executor()
        failed.execute(invalid, f"{variant}-bad-{field}", execute_outputs=["61"])
        assert not failed.success and sample_calls == []

    interrupted_calls = []
    successful_outputs_before_cancel = len(TinyChunkedV2V4Sink.seen)

    def interrupt_pass2(*args):
        interrupted_calls.append(args[4].index)
        raise comfy.model_management.InterruptProcessingException()

    monkeypatch.setattr(chunked_stage_nodes, "sample_chunked_pass2", interrupt_pass2)
    retry_executor = _executor()
    retry_executor.execute(deepcopy(recovered), f"{variant}-cancel-pass2",
                           execute_outputs=["61"])
    assert not retry_executor.success
    assert interrupted_calls == [0]
    assert len(TinyChunkedV2V4Sink.seen) == successful_outputs_before_cancel
    assert len(TinyChunkedV2V4FirstPass.calls) == 2
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == sha.lower()

    monkeypatch.setattr(chunked_stage_nodes, "sample_chunked_pass2", observed_sample)
    retry_executor.execute(deepcopy(recovered), f"{variant}-retry-pass2",
                           execute_outputs=["61"])
    assert retry_executor.success, retry_executor.status_messages
    assert sample_calls == [0]
    assert TinyChunkedV2V4Sink.seen[-1][0] == expected[0]
    assert len(TinyChunkedV2V4FirstPass.calls) == 2
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == sha.lower()

    sample_calls.clear()
    retry_executor.execute(deepcopy(recovered), f"{variant}-cached-high",
                           execute_outputs=["61"])
    assert retry_executor.success and sample_calls == []
    edited_high = deepcopy(recovered)
    high_key = next(key for key, node in edited_high.items()
                    if node["class_type"] == "MiniMaxH3ChunkedPass2SegmentEXPT8")
    edited_high[high_key]["inputs"]["cfg"] = 1.25
    retry_executor.execute(edited_high, f"{variant}-edited-high",
                           execute_outputs=["61"])
    assert retry_executor.success, retry_executor.status_messages
    assert sample_calls == [0]
    assert len(TinyChunkedV2V4FirstPass.calls) == 2
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == sha.lower()
