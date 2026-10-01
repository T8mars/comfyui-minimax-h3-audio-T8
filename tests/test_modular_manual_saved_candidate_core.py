"""Run the saved S09 pair on real Core with explicit tiny CPU model doubles.

The candidate graph itself is loaded from disk. This is a stage-artifact and
dependency test, not a pretrained GPU, media, or image-quality qualification.
"""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import torch
from comfy_api.latest import io

from h3_audio_t8_pkg.modular_sampling import nodes as stage_nodes, stage_unet_loader
from h3_audio_t8_pkg.modular_sampling.stage_unet_loader_nodes import NODES as STAGE_LOADER_NODES
from h3_audio_t8_pkg.modular_sampling.storage import file_sha
from h3_audio_t8_pkg.nodes import MiniMaxH3AVDecodeT8
from h3_audio_t8_pkg.nodes_prompt_relay_advanced import (
    MiniMaxH3PromptRelayConditioningT8Advanced,
    MiniMaxH3PromptRelayPlanT8Advanced,
)
from test_fast_h3_v2_core_sampler import model
from test_modular_fast_h3_v2_saved_candidate_core import TinyCandidateCLIP, TinyCandidateVAE
from tools.build_modular_manual_pass_workflow import split_graph


ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ROOT / "artifacts/development/modular-sampling-m2-manual-pass-20260925/candidates-v3"
FULL = CANDIDATES / "Manual_Pass_save_effects_NativeNoise_EXP.api.json"
RESUME = CANDIDATES / "Manual_Pass_resume_effects_NativeNoise_EXP.api.json"


class TinyManualUNET(io.ComfyNode):
    calls = []

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="UNETLoader", inputs=[
            io.String.Input("unet_name"), io.String.Input("weight_dtype")],
            outputs=[io.Model.Output("model")])

    @classmethod
    def execute(cls, unet_name, weight_dtype):
        assert unet_name.endswith(".safetensors") and weight_dtype == "default"
        cls.calls.append(unet_name)
        return io.NodeOutput(model())


def _candidate(path, expected):
    assert path.is_file(), path
    graph = json.loads(path.read_text(encoding="utf-8"))
    assert set(graph) == set(expected)
    for key, node in graph.items():
        assert node["class_type"] == expected[key]["class_type"]
        actual_edges = {name: value for name, value in node["inputs"].items()
                        if isinstance(value, list) and len(value) == 2}
        planned_edges = {name: value for name, value in expected[key]["inputs"].items()
                         if isinstance(value, list) and len(value) == 2}
        assert actual_edges == planned_edges, key
    return graph


def _tiny(graph):
    graph = deepcopy(graph)
    for key, value in (("80", 128), ("81", 64), ("82", 22)):
        if key in graph:
            graph[key]["inputs"]["value"] = value
    for key in ("9", "24"):
        if key in graph:
            graph[key]["inputs"]["query_chunk_rows"] = 64
            for field, value in (("width", 128), ("height", 64)):
                if isinstance(graph[key]["inputs"].get(field), int):
                    graph[key]["inputs"][field] = value
    for key in ("40", "47", "104"):
        if key in graph:
            graph[key]["inputs"].update(global_prompt="A stable scene",
                                         local_prompts="A woman waves.\nShe walks away.")
            if isinstance(graph[key]["inputs"].get("length"), int):
                graph[key]["inputs"]["length"] = 22
    for key in ("41", "42", "100"):
        if key in graph:
            graph[key]["inputs"].update(tau=.2, start_video_progress=0.,
                                         end_video_progress=1., g_hard_limit=3.)
    return graph


def _cold_process(kind, artifact_root, artifact_path, artifact_sha256):
    code = """
import json,os,runpy,sys
from pathlib import Path
runpy.run_path('tools/check_director_d1_cpu.py',run_name='config')
runpy.run_path('tests/conftest.py')
sys.path.insert(0,str(Path.cwd()/'tests'))
os.chdir(Path.cwd().parents[1])
sys.path.insert(0,str(Path.cwd()))
from modular_m2_cold_helper import run_cold
print('RESULT='+json.dumps(run_cold(**json.load(sys.stdin)),sort_keys=True))
"""
    payload = {"kind": kind, "artifact_root": str(artifact_root),
               "artifact_path": artifact_path, "artifact_sha256": artifact_sha256}
    child = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                           input=json.dumps(payload), capture_output=True,
                           text=True, timeout=90, check=False)
    assert child.returncode == 0, child.stderr
    return json.loads(next(line[7:] for line in child.stdout.splitlines()
                           if line.startswith("RESULT=")))


def test_saved_manual_first_to_second_only_core_resume(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT.parents[1]))
    import execution
    import nodes
    from comfy_extras.nodes_custom_sampler import BasicGuider, RandomNoise
    from comfy_extras.nodes_primitive import Int
    from comfy_extras.nodes_preview_any import PreviewAny
    from comfy_extras.nodes_video import CreateVideo, SaveVideo

    full = _tiny(_candidate(FULL, split_graph("save_effects")))
    resume = _tiny(_candidate(RESUME, split_graph("resume_effects")))
    assert sum(item["class_type"] == "MiniMaxH3StageSamplerEXPT8" for item in resume.values()) == 1
    for cls in (TinyManualUNET, TinyCandidateCLIP, TinyCandidateVAE, Int,
                MiniMaxH3AVDecodeT8, PreviewAny, CreateVideo, SaveVideo,
                BasicGuider, RandomNoise,
                MiniMaxH3PromptRelayPlanT8Advanced,
                MiniMaxH3PromptRelayConditioningT8Advanced,
                stage_nodes.MiniMaxH3ManualPassStageSetupEXPT8,
                stage_nodes.MiniMaxH3StageNoiseEXPT8,
                stage_nodes.MiniMaxH3StageSamplerEXPT8,
                stage_nodes.MiniMaxH3StageSaveEXPT8,
                stage_nodes.MiniMaxH3StageLoadEXPT8,
                stage_nodes.MiniMaxH3StageEAVConfigEXPT8,
                stage_nodes.MiniMaxH3StageEAVApplyEXPT8,
                stage_nodes.MiniMaxH3StageEAVAuditEXPT8,
                *STAGE_LOADER_NODES):
        name = cls.define_schema().node_id if hasattr(cls, "define_schema") else cls.__name__
        monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, name, cls)
    monkeypatch.setattr(stage_nodes, "_stage_store_root", lambda: tmp_path)
    high_loads = []

    def load_high(name, dtype):
        high_loads.append((name, dtype))
        return model()

    monkeypatch.setattr(stage_unet_loader, "_load_unet", load_high)
    stages, receipts, audits = [], [], []
    original_sample = stage_nodes.sample_stage
    original_audit = stage_nodes.audit_stage_eav

    def observed_sample(*args):
        stages.append(args[-1].stage)
        result = original_sample(*args)
        receipts.append(result[2].verify())
        return result

    def observed_audit(*args):
        latent, report = original_audit(*args)
        audits.append(json.loads(report))
        return latent, report

    monkeypatch.setattr(stage_nodes, "sample_stage", observed_sample)
    monkeypatch.setattr(stage_nodes, "audit_stage_eav", observed_audit)
    TinyManualUNET.calls.clear()
    torch.set_num_threads(1)
    server = SimpleNamespace(client_id=None, last_node_id=None, sockets_metadata={},
                             send_sync=lambda *args, **kwargs: None)

    def executor():
        return execution.PromptExecutor(server, cache_args={"ram": 0., "ram_inactive": 0.},
                                        asset_manager=SimpleNamespace(enabled=False))

    core = executor()
    core.execute(deepcopy(full), "s09-full-tiny", execute_outputs=["45", "46", "50", "51"])
    assert core.success, core.status_messages
    assert stages == ["manual_first", "manual_second"]
    assert len(TinyManualUNET.calls) == len(high_loads) == 1
    assert len(audits) == 2
    assert all(item["status"] == "observed_report_only" for item in audits)
    assert [item["relay_attention_calls"] for item in audits] == [20, 3]
    expected_second = receipts[-1]
    edited = deepcopy(full)
    edited["47"]["inputs"]["local_prompts"] = "She looks up.\nShe turns away."
    stages.clear()
    core.execute(edited, "s09-second-plan-edit-tiny", execute_outputs=["46", "51"])
    assert core.success, core.status_messages
    assert stages == ["manual_second"]
    assert receipts[-1]["request_sha256"] != expected_second["request_sha256"]
    first_manifest = next(path for path in tmp_path.rglob("manifest.json")
                          if json.loads(path.read_text(encoding="utf-8"))["stage_context"]["stage"]
                          == "manual_first")
    resume["60"]["inputs"].update(artifact_path=first_manifest.relative_to(tmp_path).as_posix(),
                                  artifact_sha256=file_sha(first_manifest))
    stages.clear()
    audits.clear()
    core = executor()
    core.execute(deepcopy(resume), "s09-second-only-tiny", execute_outputs=["46", "51"])
    assert core.success, core.status_messages
    assert stages == ["manual_second"]
    assert len(TinyManualUNET.calls) == 2 and len(high_loads) == 1
    assert len(audits) == 1 and audits[0]["status"] == "observed_report_only"
    assert audits[0]["relay_attention_calls"] == 3
    assert receipts[-1]["request_sha256"] == expected_second["request_sha256"]
    assert receipts[-1]["outputs"] == expected_second["outputs"]
    cold = _cold_process("manual", tmp_path, resume["60"]["inputs"]["artifact_path"],
                         resume["60"]["inputs"]["artifact_sha256"])
    assert cold["stages"] == ["manual_second"]
    assert cold["request_sha256"] == expected_second["request_sha256"]
    assert cold["outputs"] == expected_second["outputs"]

    corrupted = deepcopy(resume)
    corrupted["60"]["inputs"]["artifact_sha256"] = "0" * 64
    stages.clear()
    core = executor()
    core.execute(corrupted, "s09-bad-manifest-tiny", execute_outputs=["46"])
    assert not core.success and not stages
    assert not torch.cuda.is_initialized()
