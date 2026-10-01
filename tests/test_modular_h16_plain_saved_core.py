"""Run the saved H16 no-effects freeze/resume pair under real CPU Core.

The candidate JSON stays SHA-bound. Only heavyweight upstream inputs and the
media terminal are replaced in an in-memory execution copy.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from comfy_api.latest import io
import folder_paths
import pytest
import torch

from h3_audio_t8_pkg import chunked_two_pass_upscale_advanced as legacy
from h3_audio_t8_pkg.modular_sampling import h16_nodes
from h3_audio_t8_pkg.modular_sampling.results import _input_identity
from test_modular_h16_saved_effects_core import (
    TinyH16EffectInputs, _executor, _numeric_effect_case,
    _register as _register_shared,
)
from test_modular_h16_stages import _no_op_lift
from test_progressive_sampling_runtime import conditioning
from tools.build_modular_chunked_v5_storage_workflow import _prune_api
from tools.build_modular_h16_storage_workflow import (
    TARGET_V2, candidate_api, freeze_frontend, resume_frontend,
)
from tools.build_modular_h16_workflow import TEMPLATE, split_frontend


LABELS = ("01_freeze_after_window_2", "02_resume_windows_3_to_6_DRAFT")
HASHES = {
    LABELS[0]: "41464d8ae9ad7f9ea50dd79ace53575c273c7ce7634fcb590e623948bae4ddaa",
    LABELS[1]: "9a779997f99db4cb5f5a74e4aedafbf28dcf64f77a57e6757c0261cad8909c6b",
}


class TinyPlainH16Reconcile(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__,
                         inputs=[io.Latent.Input("learned_latent")],
                         outputs=[io.Latent.Output("first_pass_latent"),
                                  io.Conditioning.Output("positive")])

    @classmethod
    def execute(cls, learned_latent):
        return io.NodeOutput(learned_latent, conditioning())


class TinyPlainH16FreezeReceipt(io.ComfyNode):
    seen = []

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, is_output_node=True,
                         inputs=[io.String.Input(name) for name in (
                             "native_status", "native_path", "native_sha",
                             "native_manifest", "window_path", "window_sha")]
                         + [io.Custom("T8_H16_PASS2_RESULT").Input("window_result")],
                         outputs=[io.String.Output("report")])

    @classmethod
    def execute(cls, native_status, native_path, native_sha, native_manifest,
                window_path, window_sha, window_result):
        cls.seen.append({
            "status": native_status,
            "receipt": (native_path, native_sha, native_manifest,
                        window_path, window_sha),
            "last_frozen_index": window_result.core_result.index,
        })
        return io.NodeOutput(native_status)


class TinyPlainH16ResumeReceipt(io.ComfyNode):
    seen = []

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, is_output_node=True,
                         inputs=[io.Latent.Input("av_latent"),
                                 io.Custom("T8_H16_PASS2_RESULT").Input("window_result"),
                                 io.String.Input("native_bridge_report"),
                                 io.String.Input("window_load_report")],
                         outputs=[io.String.Output("report")])

    @classmethod
    def execute(cls, av_latent, window_result, native_bridge_report,
                window_load_report):
        video, audio = av_latent["samples"].unbind()
        cls.seen.append({
            "av_identity": (_input_identity(video), _input_identity(audio)),
            "last_index": window_result.core_result.index,
            "audio_output": window_result.audio_output,
            "native_status": json.loads(native_bridge_report)["status"],
            "window_load_status": json.loads(window_load_report)["status"],
        })
        return io.NodeOutput("observed")


def _saved(label):
    path = TARGET_V2 / f"{label}.api.json"
    raw = path.read_bytes()
    base = split_frontend(json.loads(TEMPLATE.read_text(encoding="utf-8")))
    expected = freeze_frontend(base) if label.startswith("01") else resume_frontend(base)
    graph = json.loads(raw)
    assert graph == candidate_api(expected, freeze=label.startswith("01"))
    assert hashlib.sha256(raw).hexdigest() == HASHES[label]
    return graph


def _graph(saved, *, resume, receipt=None):
    graph = deepcopy(saved)
    graph["100"] = {"class_type": "TinyH16EffectInputs", "inputs": {}}
    if not resume:
        graph["13"] = {"class_type": "TinyH16Learned",
                       "inputs": {"source": ["100", 0]}}
    learned_source = graph["15"]["inputs"]["learned_latent"]
    graph["15"] = {"class_type": "TinyPlainH16Reconcile", "inputs": {
        "learned_latent": learned_source}}
    graph["16"] = {"class_type": "TinyH16SamplingSetup", "inputs": {
        "model": ["100", 1], "sampler": ["100", 2], "sigmas": ["100", 3]}}
    graph["29"]["inputs"]["noise"] = ["100", 4]
    for node in graph.values():
        if node["class_type"] == "MiniMaxH3H16Pass2WindowEXPT8":
            node["inputs"]["noise"] = ["100", 4]
    if resume:
        graph["51"]["inputs"].update(
            checkpoint_path=receipt[0], expected_file_sha256=receipt[1],
            expected_manifest_json=receipt[2],
        )
        graph["52"]["inputs"].update(
            artifact_path=receipt[3], artifact_sha256=receipt[4],
        )
        graph["101"] = {"class_type": "TinyPlainH16ResumeReceipt", "inputs": {
            "av_latent": ["50", 0], "window_result": ["50", 1],
            "native_bridge_report": ["53", 1],
            "window_load_report": ["52", 2],
        }}
    else:
        graph["51"]["inputs"]["confirm_save"] = True
        graph["52"]["inputs"]["confirm_save"] = True
        graph["101"] = {"class_type": "TinyPlainH16FreezeReceipt", "inputs": {
            "native_status": ["51", 1], "native_path": ["51", 2],
            "native_sha": ["51", 3], "native_manifest": ["51", 4],
            "window_path": ["52", 2], "window_sha": ["52", 3],
            "window_result": ["52", 1],
        }}
    graph = _prune_api(graph, ("101",))
    assert "12" not in graph and "14" not in graph
    if resume:
        assert "13" not in graph
        assert not {"32", "35", "38"} & set(graph)
    else:
        assert "41" not in graph
    return graph


def _register(monkeypatch):
    import nodes

    _register_shared(monkeypatch)
    for cls in (TinyPlainH16Reconcile, TinyPlainH16FreezeReceipt,
                TinyPlainH16ResumeReceipt):
        monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, cls.define_schema().node_id, cls)


def _cold_resume(output_root, receipt):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
    torch.set_num_threads(1)
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(Path(__file__).resolve().parents[3]))
        _register(patch)
        patch.setattr(folder_paths, "get_output_directory", lambda: output_root)
        patch.setattr(legacy, "learned_upscale_h3_av_latent", _no_op_lift)
        TinyH16EffectInputs.case = _numeric_effect_case(patch)
        TinyPlainH16ResumeReceipt.seen.clear()
        executor = _executor()
        executor.execute(_graph(_saved(LABELS[1]), resume=True, receipt=receipt),
                         "h16-plain-cold-resume", execute_outputs=["101"])
        assert executor.success, executor.status_messages
        assert not torch.cuda.is_initialized()
        return TinyPlainH16ResumeReceipt.seen[-1]


def test_saved_h16_plain_pair_freezes_and_cold_resumes_only_later_windows(
        monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3]))
    _register(monkeypatch)
    torch.set_num_threads(1)
    output_root = tmp_path / "output"
    output_root.mkdir()
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(output_root))
    monkeypatch.setattr(legacy, "learned_upscale_h3_av_latent", _no_op_lift)
    TinyH16EffectInputs.case = _numeric_effect_case(monkeypatch)
    original = h16_nodes.sample_h16_pass2
    calls = []

    def observed(*args, **kwargs):
        calls.append(int(args[4].index))
        return original(*args, **kwargs)

    monkeypatch.setattr(h16_nodes, "sample_h16_pass2", observed)
    TinyPlainH16FreezeReceipt.seen.clear()
    TinyPlainH16ResumeReceipt.seen.clear()
    freeze = _graph(_saved(LABELS[0]), resume=False)
    executor = _executor()
    executor.execute(freeze, "h16-plain-freeze", execute_outputs=["101"])
    assert executor.success, executor.status_messages
    assert calls == [0, 1, 2]
    frozen = TinyPlainH16FreezeReceipt.seen[-1]
    assert frozen["status"] == "SAVED_VERIFIED"
    assert frozen["last_frozen_index"] == 2
    receipt = frozen["receipt"]
    assert all(receipt)

    calls.clear()
    bad = _graph(_saved(LABELS[1]), resume=True,
                 receipt=(*receipt[:4], "0" * 64))
    executor = _executor()
    executor.execute(bad, "h16-plain-bad-window-sha", execute_outputs=["101"])
    assert not executor.success
    assert calls == [] and not TinyPlainH16ResumeReceipt.seen

    resume = _graph(_saved(LABELS[1]), resume=True, receipt=receipt)
    executor = _executor()
    executor.execute(resume, "h16-plain-resume", execute_outputs=["101"])
    assert executor.success, executor.status_messages
    assert calls == [3, 4, 5, 6]
    result = TinyPlainH16ResumeReceipt.seen[-1]
    assert result["last_index"] == 6
    assert result["audio_output"] == "refined_exp"
    assert result["native_status"] == "verified_native_source_for_h16"

    child_code = """
import json,runpy,sys
import comfy.options
comfy.options.enable_args_parsing()
args=sys.argv[1:]
sys.argv=['h16-plain-cold','--cpu','--use-pytorch-cross-attention']
import comfy.cli_args
sys.argv=['h16-plain-cold']
runpy.run_path('tests/conftest.py')
from test_modular_h16_plain_saved_core import _cold_resume
print('RESULT='+json.dumps(_cold_resume(args[0],args[1:6]),sort_keys=True))
"""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[3])
    child = subprocess.run(
        [sys.executable, "-c", child_code, str(output_root), *receipt],
        cwd=Path(__file__).resolve().parents[1], env=env,
        capture_output=True, text=True, timeout=90,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    cold = json.loads(next(line[7:] for line in child.stdout.splitlines()
                           if line.startswith("RESULT=")))
    assert cold == {**result, "av_identity": list(result["av_identity"])}

    native_file = output_root / "MiniMaxH3/latent_checkpoints" / receipt[0]
    window_file = output_root / "MiniMaxH3/h16_window_artifacts" / receipt[3]
    assert hashlib.sha256(native_file.read_bytes()).hexdigest() == receipt[1].lower()
    assert hashlib.sha256(window_file.read_bytes()).hexdigest() == receipt[4].lower()
