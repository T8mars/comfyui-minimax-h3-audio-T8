"""Saved effect graphs execute native tiny H3; loader/resource doubles are explicit.

These are tail-only mechanical checks, not pretrained first-pass/PDD or quality
qualification. Actual Plan/Conditioning, guards, effects and Core executor run.
"""
import asyncio
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import comfy.model_management
import comfy_extras.nodes_custom_sampler as custom
from comfy_api.latest import io
import pytest
import torch

import h3_audio_t8_pkg
from h3_audio_t8_pkg import nodes_native_latent_checkpoint_advanced as checkpoint_nodes
from h3_audio_t8_pkg import prompt_relay_advanced as relay
from h3_audio_t8_pkg.native_latent_checkpoint_advanced import save_native_h3_av_checkpoint
from test_modular_audio_refine_full_resume_core import (
    RuntimeAudit, RuntimeSetup, TinyRefineUNET, TinyRefineVAE, _native_av, _required_for,
)
from test_modular_audio_refine_resume_all import (
    RuntimeDualClockSetup, RuntimeDualModelSetup, TinyClipProjApply, TinyCompatCLIP,
    TinyReferenceImage, TinyTurbo4Lora,
)
from tools import build_modular_audio_tail_effect_workflows as builder
from tools.build_modular_audio_refine_resume_all import CHECKPOINT_GUARD, CHECKPOINT_LOAD

ROOT = Path(__file__).resolve().parents[1]
SAVED = ROOT / "artifacts/development/modular-audio-tail-effects-20260928/matrix-v1"
PATHS = sorted(builder.generated("relay"))


class CaptureTailEffects(io.ComfyNode):
    observations = []

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="CaptureTailEffects", is_output_node=True,
            inputs=[io.Latent.Input("original"), io.Latent.Input("candidate"),
                    io.Latent.Input("selected"), io.String.Input("decision"),
                    io.String.Input("relay_report"), io.String.Input("guider_report"),
                    io.String.Input("eav_report", default="{}")],
            outputs=[io.String.Output("decision")])

    @classmethod
    def execute(cls, **kwargs):
        cls.observations.append(kwargs)
        return io.NodeOutput(kwargs["decision"])


def _one(graph, kind):
    keys = [key for key, node in graph.items() if node["class_type"] == kind]
    assert len(keys) == 1, (kind, keys)
    return keys[0]


def _run_saved(tmp_path, monkeypatch, path_name, effect, mode, checkpoint):
    monkeypatch.syspath_prepend(str(ROOT.parents[1]))
    import execution
    import nodes

    path = next(path for path in PATHS if path.name == path_name)
    filename = path.stem + "_" + effect + "_TailEffects"
    saved_frontend = json.loads((SAVED / (filename + ".json")).read_text(encoding="utf8"))
    assert saved_frontend == builder.generated(effect)[path]
    graph = deepcopy(json.loads((SAVED / (filename + ".api.json")).read_text(encoding="utf8")))
    load = _one(graph, CHECKPOINT_LOAD)
    guard = _one(graph, CHECKPOINT_GUARD)
    plan, cond = _one(graph, builder.RELAY_PLAN), _one(graph, builder.RELAY_COND)
    query = _one(graph, builder.RELAY_ROUTE)
    guider = _one(graph, builder.GUIDER)
    gate = _one(graph, "MiniMaxH3AudioRefineQualityGateT8Advanced")
    sampler = _one(graph, "SamplerCustomAdvanced")
    graph[load]["inputs"].update(checkpoint_path=checkpoint[2],
        expected_manifest_json=checkpoint[4], expected_file_sha256=checkpoint[3])
    graph[guard]["inputs"]["expected_video_frame_count"] = 22
    assert graph[plan]["inputs"]["length"] == [guard, 3]
    assert [graph[cond]["inputs"][name] for name in ("width", "height")] == [[guard, 1], [guard, 2]]
    # Explicit experimental test opt-in, never modify saved draft defaults.
    graph[plan]["inputs"].update(local_prompts="Soft room ambience.\nClear footsteps.",
                                 timing_mode="auto_equal", time_ranges="")
    graph[query]["inputs"]["query_route"] = "joint_av_exp"
    graph[cond]["inputs"].update(execution_mode="apply_exp", query_chunk_rows=64)
    capture = {"original": [load, 0], "candidate": [sampler, 0], "selected": [gate, 0],
               "decision": [gate, 3], "relay_report": [cond, 6], "guider_report": [guider, 1]}
    if effect == "combined":
        config, audit = _one(graph, builder.CONFIG), _one(graph, builder.AUDIT)
        graph[config]["inputs"].update(mode=mode, tau=.25, start_video_progress=0., end_video_progress=1.)
        capture["eav_report"] = [audit, 1]
    graph["capture-tail-effects"] = {"class_type": "CaptureTailEffects", "inputs": capture}
    graph = _required_for(graph, "capture-tail-effects")
    assert sum(node["class_type"] == "SamplerCustomAdvanced" for node in graph.values()) == 1
    monkeypatch.setattr(checkpoint_nodes.folder_paths, "get_output_directory", lambda: str(tmp_path))
    monkeypatch.setattr(comfy.model_management, "intermediate_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(custom.latent_preview, "prepare_callback", lambda *_a, **_k: lambda *_v: None)
    classes = asyncio.run(h3_audio_t8_pkg.comfy_entrypoint().get_node_list())
    for cls in [*classes, TinyRefineUNET, TinyCompatCLIP, TinyRefineVAE, TinyReferenceImage,
                TinyClipProjApply, TinyTurbo4Lora, RuntimeAudit, RuntimeSetup,
                RuntimeDualClockSetup, RuntimeDualModelSetup, custom.SamplerCustomAdvanced, CaptureTailEffects]:
        key = cls.define_schema().node_id if hasattr(cls, "define_schema") else cls.__name__
        monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, key, cls)
    CaptureTailEffects.observations.clear()
    TinyRefineUNET.calls.clear()
    TinyTurbo4Lora.observations.clear()
    calls = []
    previous = relay.route_prompt_relay_attention

    def observed(*args, **kwargs):
        calls.append(kwargs["transformer_options"][relay.PROMPT_RELAY_RUNTIME_KEY]["binding_hash"])
        return previous(*args, **kwargs)

    monkeypatch.setattr(relay, "route_prompt_relay_attention", observed)
    executor = execution.PromptExecutor(
        SimpleNamespace(client_id=None, last_node_id=None, sockets_metadata={}, send_sync=lambda *_a, **_k: None),
        cache_args={"ram": 0., "ram_inactive": 0.}, asset_manager=SimpleNamespace(enabled=False))
    executor.execute(graph, "s26-fresh-tail-" + effect, execute_outputs=["capture-tail-effects"])
    assert executor.success, executor.status_messages
    assert len(TinyRefineUNET.calls) == 1
    assert len(CaptureTailEffects.observations) == 1
    result = CaptureTailEffects.observations[0]
    assert result["selected"] is result["original"]
    assert result["decision"] == "ABSTAIN_HUMAN_REVIEW_REQUIRED"
    relay_report = json.loads(result["relay_report"])
    assert relay_report["status"] == "applied_exp"
    assert json.loads(result["guider_report"])["external_positive"] is True
    assert len(calls) == 4 and len(set(calls)) == 1
    original_video, original_audio = result["original"]["samples"].unbind()
    candidate_video, candidate_audio = result["candidate"]["samples"].unbind()
    assert torch.allclose(candidate_video, original_video, atol=1e-5, rtol=0)
    assert not torch.equal(candidate_audio, original_audio)
    eav_report = json.loads(result.get("eav_report", "{}"))
    if effect == "combined":
        assert eav_report["completed_forwards"] == eav_report["relay_attention_calls"] == 4
        assert eav_report["status"] == {"report_only": "observed_report_only", "apply_exp": "observed_apply_exp"}[mode]
    assert not torch.cuda.is_initialized()
    return {"child_pid": os.getpid(), "path": path_name, "effect": effect, "mode": mode,
            "relay_calls": len(calls), "model_loads": len(TinyRefineUNET.calls),
            "candidate_audio_sha256": hashlib.sha256(candidate_audio.detach().contiguous().numpy().tobytes()).hexdigest(),
            "decision": result["decision"], "cuda_initialized": torch.cuda.is_initialized()}


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("effect,mode", [("relay", "disabled"), ("combined", "report_only"), ("combined", "apply_exp")])
def test_all_eight_saved_fresh_relay_routes_execute_real_tail(tmp_path, monkeypatch, path, effect, mode):
    checkpoint = save_native_h3_av_checkpoint(_native_av(), tmp_path / "MiniMaxH3/latent_checkpoints",
        filename_prefix="audio_refine_firstpass", checkpoint_id="audio_refine_firstpass", confirm_save=True)
    assert checkpoint[1] == "SAVED_VERIFIED"
    _run_saved(tmp_path, monkeypatch, path.name, effect, mode, checkpoint)


@pytest.mark.parametrize("path", PATHS)
def test_each_saved_combined_tail_runs_in_a_new_core_process(tmp_path, path):
    checkpoint = save_native_h3_av_checkpoint(_native_av(), tmp_path / "MiniMaxH3/latent_checkpoints",
        filename_prefix="audio_refine_firstpass", checkpoint_id="audio_refine_firstpass", confirm_save=True)
    code = '''
import json, os, runpy, sys
from pathlib import Path
root = Path.cwd()
sys.path.insert(0, str(root.parents[1]))
import comfy.cli_args
comfy.cli_args.args.cpu = True
import torch
torch.set_num_threads(2)
runpy.run_path('tests/conftest.py')
sys.path.insert(0, str(root))
os.chdir(root.parents[1])
from test_modular_audio_tail_effect_saved_core import _run_saved
import pytest
payload = json.load(sys.stdin)
with pytest.MonkeyPatch.context() as patch:
    result = _run_saved(Path(payload['root']), patch, payload['path'], 'combined', 'apply_exp', payload['checkpoint'])
print('RESULT=' + json.dumps(result, sort_keys=True))
'''
    payload = {"root": str(tmp_path), "path": path.name,
               "checkpoint": [None, None, checkpoint[2], checkpoint[3], checkpoint[4]]}
    child = subprocess.run([sys.executable, "-c", code], cwd=ROOT, input=json.dumps(payload),
        capture_output=True, text=True, timeout=120, check=False,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"})
    assert child.returncode == 0, child.stdout + child.stderr
    result = json.loads(next(line[7:] for line in child.stdout.splitlines() if line.startswith("RESULT=")))
    assert result["child_pid"] != os.getpid() and result["relay_calls"] == 4
    assert result["cuda_initialized"] is False
