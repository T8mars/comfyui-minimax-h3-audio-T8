"""Execute the saved S22 v5 pair with real Core and tiny deterministic AV inputs.

Only the large upstream models, Relay/EAV and media terminal are bypassed in
the in-memory execution copy. Original private candidate files are SHA-pinned.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import comfy.nested_tensor
from comfy_api.latest import io
import folder_paths
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from h3_audio_t8_pkg.modular_sampling import chunked_v5
from h3_audio_t8_pkg.modular_sampling.chunked_v5_nodes import NODES as V5_NODES
from h3_audio_t8_pkg.modular_sampling.chunked_v5_native_source import verified_native_source
from h3_audio_t8_pkg.modular_sampling.chunked_v5_storage_nodes import NODES as STORAGE_NODES
from h3_audio_t8_pkg.modular_sampling.results import _input_identity
from h3_audio_t8_pkg.native_latent_checkpoint_advanced import (
    load_native_h3_av_checkpoint, save_native_h3_av_checkpoint,
)
from h3_audio_t8_pkg.nodes_native_latent_checkpoint_advanced import (
    MiniMaxH3NativeLatentCheckpointLoadT8Advanced,
    MiniMaxH3NativeLatentCheckpointSaveT8Advanced,
)
from test_modular_chunked_v5 import make_v5_harness
from test_chunked_two_pass_parity import _plan
from tools.build_modular_chunked_v5_storage_workflow import _prune_api, build_pair


ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ROOT / "artifacts/development/modular-sampling-m4-chunked-v5-storage-20260924/candidate-v2"
THREE_CANDIDATES = (
    ROOT / "artifacts/development/modular-sampling-m4-chunked-v5-storage-20260924"
    / "candidate-v4-three-windows"
)
FOUR_CANDIDATES = (
    ROOT / "artifacts/development/modular-sampling-m4-chunked-v5-storage-20260924"
    / "candidate-v6-four-windows"
)
HASHES = {
    "01_freeze_window_0": "ea242c3ad3db829cade78bab076d971a819b61b90d82afc82d190e738de94a62",
    "02_resume_window_1_DRAFT": "75701b14d8ac337a0f92972813db61133e939aab3fe47246b0f2979979915a5c",
}
THREE_HASHES = {
    "01_freeze_window_1": "0bedc867ae71c1adee1c81a4b8dee48ba5a699ca35079ae35ea34fcc59262058",
    "02_resume_window_2_DRAFT": "545186e891de69c8bf3e582add25328c69ab323d9ba5c509d60bdfdc2abd9025",
}
THREE_FRONTEND_HASHES = {
    "01_freeze_window_1": "23af579e5b6a1f6e49b18a777c6acaa4a3f7e292816aab4ac8d20bcb238ffbe1",
    "02_resume_window_2_DRAFT": "be109aa6eb0f54f265d9729cb3152dba9ac94ef5e51cd6fe52f7ea5419a2a450",
}
FOUR_HASHES = {
    "01_freeze_window_2": "ec81a29932884e7bc6a556dde02c6afe07850462adad023209b17d78bbe47c37",
    "02_resume_window_3_DRAFT": "6f79adfa04a598c71b5978805b447d6371f68e8912782e106b9cd5650018aa19",
}


class TinyV5Inputs(io.ComfyNode):
    case = None

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, inputs=[], outputs=[
            io.Latent.Output("partial4_av"),
            io.Custom("T8_H3_CHUNKED_TWO_PASS_PLAN").Output("plan"),
            io.Noise.Output("noise"), io.Model.Output("model"),
            io.Conditioning.Output("positive"), io.Sampler.Output("sampler"),
            io.Sigmas.Output("sigmas"),
        ])

    @classmethod
    def execute(cls):
        return io.NodeOutput(*cls.case)


class TinyV5FirstPass(io.ComfyNode):
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


class TinyV5EditableModel(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__,
                         inputs=[io.Float.Input("offset", default=0.1)],
                         outputs=[io.Model.Output("model")])

    @classmethod
    def execute(cls, offset=0.1):
        return io.NodeOutput(SimpleNamespace(offset=float(offset)))


class TinyV5Receipt(io.ComfyNode):
    seen = []

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, is_output_node=True,
                         inputs=[io.String.Input("native_status"),
                                 io.String.Input("native_path"),
                                 io.String.Input("native_sha"),
                                 io.String.Input("native_manifest"),
                                 io.String.Input("window_path"),
                                 io.String.Input("window_sha")],
                         outputs=[io.String.Output("report")])

    @classmethod
    def execute(cls, native_status, native_path, native_sha, native_manifest,
                window_path, window_sha):
        cls.seen.append((native_status, native_path, native_sha, native_manifest,
                         window_path, window_sha))
        return io.NodeOutput(native_status)


class TinyV5Sink(io.ComfyNode):
    seen = []

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, is_output_node=True,
                         inputs=[io.Latent.Input("av_latent")],
                         outputs=[io.String.Output("report")])

    @classmethod
    def execute(cls, av_latent):
        video, audio = av_latent["samples"].unbind()
        cls.seen.append((_input_identity(video), _input_identity(audio)))
        return io.NodeOutput("joint_av_seen")


def _saved(name):
    path = CANDIDATES / f"{name}.api.json"
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == HASHES[name]
    expected = build_pair()[name][1]
    graph = json.loads(raw)
    assert graph == expected
    return graph


def _saved_three(name):
    path = THREE_CANDIDATES / f"{name}.api.json"
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == THREE_HASHES[name]
    expected = build_pair(3)[name][1]
    graph = json.loads(raw)
    assert graph == expected
    return graph


def _saved_four(name):
    path = FOUR_CANDIDATES / f"{name}.api.json"
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == FOUR_HASHES[name]
    graph = json.loads(raw)
    assert graph == build_pair(4)[name][1]
    return graph


def test_saved_three_window_frontends_keep_only_the_required_stages():
    pair = build_pair(3)
    for name, (expected, _api) in pair.items():
        raw = (THREE_CANDIDATES / f"{name}.json").read_bytes()
        assert hashlib.sha256(raw).hexdigest() == THREE_FRONTEND_HASHES[name]
        assert json.loads(raw) == expected
    freeze = pair["01_freeze_window_1"][1]
    resume = pair["02_resume_window_2_DRAFT"][1]
    assert freeze["14"]["inputs"]["temporal_chunk_frames"] == 102
    assert freeze["14"]["inputs"]["temporal_overlap_frames"] == 34
    def stages(graph, kind):
        return sorted(item["inputs"]["window_index"] for item in graph.values()
                      if item["class_type"] == kind)
    for kind in ("MiniMaxH3ChunkedV5PASS2WindowEXPT8",
                 "MiniMaxH3ChunkedV5RelayProjectEXPT8",
                 "MiniMaxH3ChunkedV5EAVApplyEXPT8"):
        assert stages(freeze, kind) == [0, 1]
        assert stages(resume, kind) == [2]
    assert {item["inputs"]["window_result"][0] for item in freeze.values()
            if item["class_type"] == "MiniMaxH3ChunkedV5EAVAuditEXPT8"} == {
                "30", "31"}
    assert resume["48"]["inputs"]["expected_window_index"] == 1
    assert all(item["class_type"] != "SamplerCustomAdvanced"
               for item in resume.values())


def test_v5_window_storage_identity_stays_compatible_with_first_saved_candidate():
    from h3_audio_t8_pkg.modular_sampling.chunked_v5_storage_compat import FULL_REPORT_PROFILE
    from v5_storage_preimages import reviewed_preimage_bytes

    # Keep the original pin. The reviewed reader additions must reconstruct the
    # exact historical implementation, with real old-save/current-load covered
    # separately; replacing the old hash would conceal a cache format break.
    assert hashlib.sha256(reviewed_preimage_bytes(FULL_REPORT_PROFILE)).hexdigest() == (
        "a60d85616918df6691ed7523f30cf642c9763cf8d602324de0eef3b600b7d614"
    )


def _tiny_graph(saved, *, resume, receipt=None):
    graph = deepcopy(saved)
    graph["60"] = {"class_type": "TinyV5Inputs", "inputs": {}}
    if not resume:
        graph["12"] = {"class_type": "TinyV5FirstPass",
                       "inputs": {"av_latent": ["60", 0]}}
    source = ["44", 0] if resume else ["12", 1]
    graph["28"]["inputs"].update(partial4_denoised_output=source, plan=["60", 1])
    graph["29"]["inputs"].update(
        partial4_denoised_output=source, plan=["60", 1], noise=["60", 2])
    window = "31" if resume else "30"
    graph[window]["inputs"].update(
        model=["60", 3], positive=["60", 4], partial4_denoised_output=source,
        plan=["60", 1], noise=["60", 2], sampler=["60", 5], sigmas=["60", 6],
    )
    if resume:
        graph["42"]["inputs"].update(
            checkpoint_path=receipt[0], expected_file_sha256=receipt[1],
            expected_manifest_json=receipt[2],
        )
        graph["43"]["inputs"].update(
            partial4_denoised_output=source, plan=["60", 1],
            artifact_path=receipt[3], artifact_sha256=receipt[4],
        )
        graph["61"] = {"class_type": "TinyV5Sink",
                       "inputs": {"av_latent": ["31", 0]}}
        roots = ("61",)
    else:
        graph["42"]["inputs"].update(av_latent=source, confirm_save=True)
        graph["43"]["inputs"].update(
            partial4_denoised_output=source, plan=["60", 1], confirm_save=True,
        )
        graph["61"] = {"class_type": "TinyV5Receipt", "inputs": {
            "native_status": ["42", 1], "native_path": ["42", 2],
            "native_sha": ["42", 3], "native_manifest": ["42", 4],
            "window_path": ["43", 2], "window_sha": ["43", 3],
        }}
        roots = ("61",)
    graph = _prune_api(graph, roots)
    if resume:
        assert "12" not in graph and "30" not in graph
    else:
        assert "31" not in graph
    return graph


def _tiny_three_graph(saved, *, resume, receipt=None):
    graph = deepcopy(saved)
    graph["60"] = {"class_type": "TinyV5Inputs", "inputs": {}}
    if not resume:
        graph["12"] = {"class_type": "TinyV5FirstPass",
                       "inputs": {"av_latent": ["60", 0]}}
    source = ["49", 0] if resume else ["12", 1]
    graph["28"]["inputs"].update(partial4_denoised_output=source, plan=["60", 1])
    graph["29"]["inputs"].update(
        partial4_denoised_output=source, plan=["60", 1], noise=["60", 2])
    for window in (("32",) if resume else ("30", "31")):
        graph[window]["inputs"].update(
            model=["60", 3], positive=["60", 4],
            partial4_denoised_output=source, plan=["60", 1],
            noise=["60", 2], sampler=["60", 5], sigmas=["60", 6],
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
        graph["61"] = {"class_type": "TinyV5Sink",
                       "inputs": {"av_latent": ["32", 0]}}
    else:
        graph["47"]["inputs"].update(av_latent=source, confirm_save=True)
        graph["48"]["inputs"].update(
            partial4_denoised_output=source, plan=["60", 1], confirm_save=True,
        )
        graph["61"] = {"class_type": "TinyV5Receipt", "inputs": {
            "native_status": ["47", 1], "native_path": ["47", 2],
            "native_sha": ["47", 3], "native_manifest": ["47", 4],
            "window_path": ["48", 2], "window_sha": ["48", 3],
        }}
    graph = _prune_api(graph, ("61",))
    if resume:
        assert not {"12", "30", "31"} & set(graph)
    else:
        assert "32" not in graph
    return graph


def _tiny_four_graph(saved, *, resume, receipt=None):
    graph = deepcopy(saved)
    graph["60"] = {"class_type": "TinyV5Inputs", "inputs": {}}
    if not resume:
        graph["12"] = {"class_type": "TinyV5FirstPass",
                       "inputs": {"av_latent": ["60", 0]}}
    source = ["54", 0] if resume else ["12", 1]
    graph["28"]["inputs"].update(partial4_denoised_output=source, plan=["60", 1])
    graph["29"]["inputs"].update(
        partial4_denoised_output=source, plan=["60", 1], noise=["60", 2])
    for window in (("33",) if resume else ("30", "31", "32")):
        graph[window]["inputs"].update(
            model=["60", 3], positive=["60", 4],
            partial4_denoised_output=source, plan=["60", 1],
            noise=["60", 2], sampler=["60", 5], sigmas=["60", 6],
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
        graph["61"] = {"class_type": "TinyV5Sink",
                       "inputs": {"av_latent": ["33", 0]}}
    else:
        graph["52"]["inputs"].update(av_latent=source, confirm_save=True)
        graph["53"]["inputs"].update(
            partial4_denoised_output=source, plan=["60", 1], confirm_save=True,
        )
        graph["61"] = {"class_type": "TinyV5Receipt", "inputs": {
            "native_status": ["52", 1], "native_path": ["52", 2],
            "native_sha": ["52", 3], "native_manifest": ["52", 4],
            "window_path": ["53", 2], "window_sha": ["53", 3],
        }}
    graph = _prune_api(graph, ("61",))
    if resume:
        assert not {"12", "30", "31", "32"} & set(graph)
    else:
        assert "33" not in graph
    return graph


def _register(patch):
    import nodes

    for cls in (*V5_NODES, *STORAGE_NODES,
                MiniMaxH3NativeLatentCheckpointSaveT8Advanced,
                MiniMaxH3NativeLatentCheckpointLoadT8Advanced,
                TinyV5Inputs, TinyV5FirstPass, TinyV5EditableModel,
                TinyV5Receipt, TinyV5Sink):
        patch.setitem(nodes.NODE_CLASS_MAPPINGS, cls.define_schema().node_id, cls)


def _executor():
    import execution

    server = SimpleNamespace(client_id=None, last_node_id=None, sockets_metadata={},
                             send_sync=lambda *args, **kwargs: None)
    return execution.PromptExecutor(server, cache_args={"ram": 0., "ram_inactive": 0.},
                                    asset_manager=SimpleNamespace(enabled=False))


def _run(graph, label):
    executor = _executor()
    executor.execute(deepcopy(graph), label, execute_outputs=["61"])
    assert executor.success, executor.status_messages


def _direct_expected(case, *, window_count=2):
    source, plan, noise, model, positive, sampler, sigmas = case
    lifted, receipt, _ = chunked_v5.lift_standard_joint(source, plan)
    prepared, _ = chunked_v5.prepare_standard_joint(source, lifted, receipt, plan, noise)
    assert len(receipt.segments) == window_count
    previous = None
    for index in range(window_count):
        final, previous, _ = chunked_v5.sample_standard_window(
            model, positive, source, lifted, prepared, plan, noise,
            sampler, sigmas, index, previous,
        )
    return tuple(_input_identity(item) for item in final["samples"].unbind())


def _native_grid_case(patch, *, three_windows=False, four_windows=False):
    calls, _source, positive, noise, sampler, sigmas, _plan_15 = make_v5_harness(patch)
    # The generic parity fixture uses 15 video tokens, which is deliberately
    # not a complete 5n+2 checkpoint grid. The two-window probe needs 56
    # frames/17 tokens; the three-window saved candidate uses its actual
    # 192-frame/57-token plan and 320 audio tokens.
    if three_windows and four_windows:
        raise ValueError("Choose exactly one saved multi-window geometry")
    video_tokens, audio_tokens = (57, 320) if (three_windows or four_windows) else (17, 93)
    source = {"samples": comfy.nested_tensor.NestedTensor((
        torch.zeros(1, 24, video_tokens, 2, 2),
        torch.zeros(1, 32, 2, audio_tokens),
    ))}
    plan = (_plan(temporal_chunk_frames=85 if four_windows else 102,
                  temporal_overlap_frames=34)
            if (three_windows or four_windows) else
            _plan(temporal_chunk_frames=51, temporal_overlap_frames=17))
    return calls, (source, plan, noise, object(), positive, sampler, sigmas)


def _cold_resume(output_root, receipt, *, saved_index=0):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
    torch.set_num_threads(1)
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(ROOT.parents[1]))
        _register(patch)
        patch.setattr(folder_paths, "get_output_directory", lambda: output_root)
        calls, case = _native_grid_case(
            patch, three_windows=saved_index == 1, four_windows=saved_index == 2)
        TinyV5Inputs.case = case
        TinyV5FirstPass.calls.clear()
        TinyV5Sink.seen.clear()
        if saved_index == 2:
            graph = _tiny_four_graph(
                _saved_four("02_resume_window_3_DRAFT"), resume=True, receipt=receipt)
        elif saved_index == 1:
            graph = _tiny_three_graph(
                _saved_three("02_resume_window_2_DRAFT"), resume=True, receipt=receipt)
        else:
            graph = _tiny_graph(
                _saved("02_resume_window_1_DRAFT"), resume=True, receipt=receipt)
        _run(graph, "v5-cold-only-window-1")
        assert not torch.cuda.is_initialized()
        return {"av_identity": TinyV5Sink.seen[-1], "sample_calls": calls["sample"],
                "first_pass_calls": len(TinyV5FirstPass.calls), "cuda_initialized": False}


def test_saved_v5_pair_freeze_then_new_core_only_second_joint_window(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT.parents[1]))
    _register(monkeypatch)
    torch.set_num_threads(1)
    output_root = tmp_path / "output"
    output_root.mkdir()
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(output_root))
    calls, case = _native_grid_case(monkeypatch)
    TinyV5Inputs.case = case
    expected = _direct_expected(case)
    baseline_calls = calls["sample"]
    TinyV5FirstPass.calls.clear()
    TinyV5Receipt.seen.clear()
    TinyV5Sink.seen.clear()
    freeze = _tiny_graph(_saved("01_freeze_window_0"), resume=False)
    _run(freeze, "v5-freeze-window-0")
    assert calls["sample"] == baseline_calls + 1
    assert len(TinyV5FirstPass.calls) == 1
    assert len(TinyV5Receipt.seen) == 1
    status, native_path, native_sha, native_manifest, window_path, window_sha = (
        TinyV5Receipt.seen[0])
    assert status == "SAVED_VERIFIED"
    native_file = output_root / "MiniMaxH3" / "latent_checkpoints" / native_path
    window_file = output_root / "MiniMaxH3" / "chunked_v5_window_artifacts" / window_path
    assert native_file.is_file() and window_file.is_file()
    assert hashlib.sha256(native_file.read_bytes()).hexdigest() == native_sha.lower()
    assert hashlib.sha256(window_file.read_bytes()).hexdigest() == window_sha.lower()
    receipt = (native_path, native_sha, native_manifest, window_path, window_sha)

    for bad_receipt in (
        (native_path, "0" * 64, native_manifest, window_path, window_sha),
        (native_path, native_sha, native_manifest, window_path, "0" * 64),
    ):
        bad = _tiny_graph(_saved("02_resume_window_1_DRAFT"), resume=True,
                          receipt=bad_receipt)
        executor = _executor()
        executor.execute(bad, "v5-reject-wrong-receipt", execute_outputs=["61"])
        assert not executor.success
        assert calls["sample"] == baseline_calls + 1
        assert not TinyV5Sink.seen

    resume = _tiny_graph(_saved("02_resume_window_1_DRAFT"), resume=True,
                         receipt=receipt)
    _run(resume, "v5-resume-window-1")
    assert calls["sample"] == baseline_calls + 2
    assert len(TinyV5FirstPass.calls) == 1
    assert TinyV5Sink.seen[-1] == expected

    child_code = """
import json,runpy,sys
import comfy.options
comfy.options.enable_args_parsing()
args=sys.argv[1:]
sys.argv=['v5-cold-resume','--cpu','--use-pytorch-cross-attention']
import comfy.cli_args
sys.argv=['v5-cold-resume']
runpy.run_path('tests/conftest.py')
from test_modular_chunked_v5_saved_storage_candidate_core import _cold_resume
print('RESULT='+json.dumps(_cold_resume(args[0],args[1:6]),sort_keys=True))
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
    assert cold == {"av_identity": list(expected), "sample_calls": 1,
                    "first_pass_calls": 0, "cuda_initialized": False}


def test_saved_v5_resume_graph_only_invalidates_last_window_on_model_edit_and_retry(
        monkeypatch, tmp_path):
    import comfy.model_management
    from h3_audio_t8_pkg import chunked_two_pass_upscale_advanced as legacy

    monkeypatch.syspath_prepend(str(ROOT.parents[1]))
    _register(monkeypatch)
    torch.set_num_threads(1)
    output_root = tmp_path / "output"
    output_root.mkdir()
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(output_root))
    calls, case = _native_grid_case(monkeypatch)
    TinyV5Inputs.case = case
    TinyV5Receipt.seen.clear()
    TinyV5Sink.seen.clear()
    TinyV5FirstPass.calls.clear()
    _run(_tiny_graph(_saved("01_freeze_window_0"), resume=False), "v5-cache-freeze")
    status, native_path, native_sha, native_manifest, window_path, window_sha = (
        TinyV5Receipt.seen[-1])
    assert status == "SAVED_VERIFIED" and calls["sample"] == 1
    assert len(TinyV5FirstPass.calls) == 1
    native_file = output_root / "MiniMaxH3" / "latent_checkpoints" / native_path
    window_file = output_root / "MiniMaxH3" / "chunked_v5_window_artifacts" / window_path
    receipt = (native_path, native_sha, native_manifest, window_path, window_sha)

    original_sample = legacy.sample_piece
    cancel = {"after_sample": False}

    def editable_sample(piece, positive, model, noise, sampler, sigmas,
                        negative, cfg, *, prepared_noise):
        result = original_sample(piece, positive, model, noise, sampler, sigmas,
                                 negative, cfg, prepared_noise=prepared_noise)
        if cancel["after_sample"]:
            raise comfy.model_management.InterruptProcessingException()
        video, audio = result.unbind()
        video_mask, _audio_mask = piece["noise_mask"].unbind()
        return comfy.nested_tensor.NestedTensor((
            video + video_mask * model.offset, audio,
        ))

    monkeypatch.setattr(legacy, "sample_piece", editable_sample)
    graph = _tiny_graph(_saved("02_resume_window_1_DRAFT"), resume=True,
                        receipt=receipt)
    graph["62"] = {"class_type": "TinyV5EditableModel",
                    "inputs": {"offset": 0.1}}
    graph["31"]["inputs"]["model"] = ["62", 0]
    executor = _executor()

    def run(label, success=True):
        executor.execute(deepcopy(graph), label, execute_outputs=["61"])
        assert executor.success is success, executor.status_messages

    run("v5-resume-cache-first")
    assert calls["sample"] == 2
    first_video, first_audio = TinyV5Sink.seen[-1]
    run("v5-resume-cache-repeat")
    assert calls["sample"] == 2

    graph["62"]["inputs"]["offset"] = 0.2
    run("v5-resume-cache-edit-last-model")
    assert calls["sample"] == 3
    edited_video, edited_audio = TinyV5Sink.seen[-1]
    assert edited_video != first_video and edited_audio == first_audio

    graph["62"]["inputs"]["offset"] = 0.3
    cancel["after_sample"] = True
    run("v5-resume-cache-cancel-last-window", success=False)
    assert calls["sample"] == 4
    assert TinyV5Sink.seen[-1] == (edited_video, edited_audio)
    cancel["after_sample"] = False
    run("v5-resume-cache-retry-last-window")
    assert calls["sample"] == 5
    assert TinyV5Sink.seen[-1][0] != edited_video
    assert TinyV5Sink.seen[-1][1] == first_audio
    assert len(TinyV5FirstPass.calls) == 1
    assert hashlib.sha256(native_file.read_bytes()).hexdigest() == native_sha.lower()
    assert hashlib.sha256(window_file.read_bytes()).hexdigest() == window_sha.lower()


def test_three_v5_windows_freeze_after_second_then_cold_resume_only_third(
        monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT.parents[1]))
    _register(monkeypatch)
    torch.set_num_threads(1)
    output_root = tmp_path / "output"
    output_root.mkdir()
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(output_root))
    calls, case = _native_grid_case(monkeypatch, three_windows=True)
    TinyV5Inputs.case = case
    expected = _direct_expected(case, window_count=3)
    baseline_calls = calls["sample"]
    TinyV5FirstPass.calls.clear()
    TinyV5Receipt.seen.clear()
    TinyV5Sink.seen.clear()

    freeze = _tiny_three_graph(_saved_three("01_freeze_window_1"), resume=False)
    _run(freeze, "v5-three-freeze-through-window-1")
    assert calls["sample"] == baseline_calls + 2
    assert len(TinyV5FirstPass.calls) == 1
    assert len(TinyV5Receipt.seen) == 1
    status, native_path, native_sha, native_manifest, window_path, window_sha = (
        TinyV5Receipt.seen[-1])
    assert status == "SAVED_VERIFIED"
    receipt = (native_path, native_sha, native_manifest, window_path, window_sha)

    resume = _tiny_three_graph(_saved_three("02_resume_window_2_DRAFT"),
                               resume=True, receipt=receipt)
    bad = deepcopy(resume)
    bad["48"]["inputs"]["expected_window_index"] = 0
    executor = _executor()
    executor.execute(bad, "v5-three-reject-wrong-window", execute_outputs=["61"])
    assert not executor.success and calls["sample"] == baseline_calls + 2
    _run(resume, "v5-three-resume-only-window-2")
    assert calls["sample"] == baseline_calls + 3
    assert len(TinyV5FirstPass.calls) == 1
    assert TinyV5Sink.seen[-1] == expected

    child_code = """
import json,runpy,sys
import comfy.options
comfy.options.enable_args_parsing()
args=sys.argv[1:]
sys.argv=['v5-three-cold-resume','--cpu','--use-pytorch-cross-attention']
import comfy.cli_args
sys.argv=['v5-three-cold-resume']
runpy.run_path('tests/conftest.py')
from test_modular_chunked_v5_saved_storage_candidate_core import _cold_resume
print('RESULT='+json.dumps(_cold_resume(args[0],args[1:6],saved_index=1),sort_keys=True))
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
    assert cold == {"av_identity": list(expected), "sample_calls": 1,
                    "first_pass_calls": 0, "cuda_initialized": False}


def test_four_v5_saved_graphs_freeze_three_then_cold_resume_only_fourth(
        monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT.parents[1]))
    _register(monkeypatch)
    torch.set_num_threads(1)
    output_root = tmp_path / "output"
    output_root.mkdir()
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(output_root))
    calls, case = _native_grid_case(monkeypatch, four_windows=True)
    TinyV5Inputs.case = case
    expected = _direct_expected(case, window_count=4)
    baseline_calls = calls["sample"]
    TinyV5FirstPass.calls.clear()
    TinyV5Receipt.seen.clear()
    TinyV5Sink.seen.clear()

    freeze = _tiny_four_graph(_saved_four("01_freeze_window_2"), resume=False)
    _run(freeze, "v5-four-freeze-through-window-2")
    assert calls["sample"] == baseline_calls + 3
    assert len(TinyV5FirstPass.calls) == len(TinyV5Receipt.seen) == 1
    status, native_path, native_sha, native_manifest, window_path, window_sha = (
        TinyV5Receipt.seen[-1])
    assert status == "SAVED_VERIFIED"
    receipt = (native_path, native_sha, native_manifest, window_path, window_sha)
    native_file = output_root / "MiniMaxH3" / "latent_checkpoints" / native_path
    window_file = output_root / "MiniMaxH3" / "chunked_v5_window_artifacts" / window_path
    assert hashlib.sha256(native_file.read_bytes()).hexdigest() == native_sha.lower()
    assert hashlib.sha256(window_file.read_bytes()).hexdigest() == window_sha.lower()

    resume = _tiny_four_graph(_saved_four("02_resume_window_3_DRAFT"),
                              resume=True, receipt=receipt)
    bad = deepcopy(resume)
    bad["53"]["inputs"]["expected_window_index"] = 1
    executor = _executor()
    executor.execute(bad, "v5-four-reject-wrong-window", execute_outputs=["61"])
    assert not executor.success and calls["sample"] == baseline_calls + 3
    _run(resume, "v5-four-resume-only-window-3")
    assert calls["sample"] == baseline_calls + 4
    assert len(TinyV5FirstPass.calls) == 1
    assert TinyV5Sink.seen[-1] == expected

    child_code = """
import json,runpy,sys
import comfy.options
comfy.options.enable_args_parsing()
args=sys.argv[1:]
sys.argv=['v5-four-cold-resume','--cpu','--use-pytorch-cross-attention']
import comfy.cli_args
sys.argv=['v5-four-cold-resume']
runpy.run_path('tests/conftest.py')
from test_modular_chunked_v5_saved_storage_candidate_core import _cold_resume
print('RESULT='+json.dumps(_cold_resume(args[0],args[1:6],saved_index=2),sort_keys=True))
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
    assert cold == {"av_identity": list(expected), "sample_calls": 1,
                    "first_pass_calls": 0, "cuda_initialized": False}


def test_verified_native_source_rejects_missing_or_mismatched_receipts(tmp_path):
    source = {"samples": comfy.nested_tensor.NestedTensor((
        torch.zeros(1, 24, 17, 2, 2), torch.zeros(1, 32, 2, 93),
    ))}
    root = tmp_path / "native"
    _latent, status, path, file_sha, manifest, _report = save_native_h3_av_checkpoint(
        source, root, checkpoint_id="v5-bridge", confirm_save=True,
    )
    assert status == "SAVED_VERIFIED"
    loaded, status, verified, checkpoint_id, content_sha, file_sha, manifest, report = (
        load_native_h3_av_checkpoint(root, path, manifest, file_sha)
    )
    assert status == "MATCH_EXTERNAL" and verified
    args = (loaded, verified, checkpoint_id, content_sha, file_sha, manifest, report)
    restored, _verification = verified_native_source(*args)
    assert _input_identity(restored) == _input_identity(source)
    bad = [
        (loaded, False, checkpoint_id, content_sha, file_sha, manifest, report),
        (loaded, verified, checkpoint_id, "0" * 64, file_sha, manifest, report),
        (loaded, verified, checkpoint_id, content_sha, "0" * 64, manifest, report),
        (loaded, verified, checkpoint_id, content_sha, file_sha, "{}", report),
        (loaded, verified, checkpoint_id, content_sha, file_sha, manifest, "{}"),
        ({**loaded, "unknown": 1}, verified, checkpoint_id, content_sha,
         file_sha, manifest, report),
        ({**restored}, verified, checkpoint_id, content_sha, file_sha, manifest, report),
        ({**loaded, "samples": comfy.nested_tensor.NestedTensor((
            torch.ones(1, 24, 17, 2, 2), torch.zeros(1, 32, 2, 93),
        ))}, verified, checkpoint_id, content_sha, file_sha, manifest, report),
    ]
    for item in bad:
        with pytest.raises(ValueError):
            verified_native_source(*item)
