"""Run the saved S26 first-pass freeze graph with tiny assets before cold resume."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import av
import comfy.model_management
import comfy_extras.nodes_custom_sampler as custom_sampler_nodes
from comfy_api.latest import io
import folder_paths
import pytest
import torch

from h3_audio_t8_pkg import nodes_native_latent_checkpoint_advanced as checkpoint_nodes
from h3_audio_t8_pkg import prompt_relay_advanced as relay
from h3_audio_t8_pkg.nodes import MiniMaxH3DualClockSamplerT8
from h3_audio_t8_pkg.nodes_native_latent_checkpoint_advanced import (
    MiniMaxH3NativeLatentCheckpointSaveT8Advanced,
)
from h3_audio_t8_pkg.nodes_prompt_relay_advanced import (
    MiniMaxH3PromptRelayPlanT8Advanced,
    MiniMaxH3PromptRelayConditioningT8Advanced,
)
from h3_audio_t8_pkg.nodes_prompt_relay_preview_advanced import (
    MiniMaxH3PromptRelayPreviewT8Advanced,
)
from test_modular_audio_refine_full_resume_core import (
    TinyRefineUNET, TinyRefineVAE, TinyCandidateCLIP, RuntimeAudit, RuntimeSetup,
    CaptureFullResume, ObservedRefineBypass, RESUME, _required_for,
    make_tiny_refine_lora,
)
from h3_audio_t8_pkg.modular_sampling.audio_refine_nodes import NODES as STAGE_NODES
from h3_audio_t8_pkg.modular_sampling.audio_refine_storage_nodes import (
    MiniMaxH3AudioRefineFrozenFirstPassLoadEXPT8,
)
from h3_audio_t8_pkg.nodes import MiniMaxH3AVDecodeT8
from h3_audio_t8_pkg.nodes_audio_refine_advanced import (
    MiniMaxH3AudioRefineCompatibilityRouteT8Advanced,
    MiniMaxH3AudioRefineCompatibilityPlanT8Advanced,
    MiniMaxH3AudioRefineQualityGateT8Advanced,
)


ROOT = Path(__file__).resolve().parents[1]
FREEZE = (ROOT / "artifacts/development/modular-sampling-m4-audio-refine-independent-model-20260924"
          / "resume-pair-v3/01_freeze_first_pass.api.json")


class CaptureFrozenFirstPass(io.ComfyNode):
    observations = []

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="CaptureFrozenFirstPass", is_output_node=True,
                         inputs=[io.Latent.Input("latent"), io.String.Input("status"),
                                 io.String.Input("checkpoint_path"),
                                 io.String.Input("file_sha256"),
                                 io.String.Input("manifest_json")],
                         outputs=[io.String.Output("file_sha256")])

    @classmethod
    def execute(cls, latent, status, checkpoint_path, file_sha256, manifest_json):
        cls.observations.append((latent, status, checkpoint_path,
                                 file_sha256, manifest_json))
        return io.NodeOutput(file_sha256)


def _cold_resume(output_root, checkpoint_path, file_sha256, manifest_json,
                 refine_prompt, media_prefix, lora_strength):
    """Run the exact saved tail API in a fresh interpreter, not the parent cache."""
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
    sys.path.insert(0, str(ROOT.parents[1]))
    import execution
    import nodes
    from comfy_extras.nodes_video import CreateVideo, SaveVideo

    graph = deepcopy(json.loads(RESUME.read_text(encoding="utf-8")))
    assert all(node not in graph for node in ("1", "10", "17"))
    graph["39"]["inputs"].update(checkpoint_path=checkpoint_path,
        expected_manifest_json=manifest_json, expected_file_sha256=file_sha256)
    graph["35"]["inputs"].update(length=22, time_ranges="0-6\n7-13\n14-21")
    graph["37"]["inputs"].update(width=128, height=128, query_chunk_rows=64)
    graph["35"]["inputs"]["global_prompt"] = refine_prompt
    graph["38"]["inputs"]["strength_model"] = lora_strength
    graph["28"]["inputs"]["video_frame_count"] = 22
    graph["31"]["inputs"]["filename_prefix"] = media_prefix
    graph["200"] = {"class_type": "CaptureFullResume", "inputs": {
        "original": ["39", 0], "candidate": ["34", 0], "selected": ["28", 0],
        "decision": ["28", 3], "route_report": ["20", 3],
        "conditioning_report": ["37", 6]}}
    graph = _required_for(graph, "200", "31")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(checkpoint_nodes.folder_paths, "get_output_directory",
                      lambda: output_root)
        patch.setattr(comfy.model_management, "intermediate_device",
                      lambda: torch.device("cpu"))
        patch.setattr(custom_sampler_nodes.latent_preview, "prepare_callback",
                      lambda *_args, **_kwargs: lambda *_values: None)
        if lora_strength:
            lora_path = Path(output_root) / (Path(media_prefix).name + ".safetensors")
            make_tiny_refine_lora(lora_path)
            patch.setattr(folder_paths, "get_full_path_or_raise",
                          lambda category, _name: str(lora_path) if category == "loras"
                          else pytest.fail("Unexpected model asset lookup"))
        for cls in (TinyRefineUNET, TinyCandidateCLIP, TinyRefineVAE,
                    MiniMaxH3PromptRelayPlanT8Advanced,
                    MiniMaxH3PromptRelayConditioningT8Advanced,
                    RuntimeAudit,
                    MiniMaxH3AudioRefineCompatibilityRouteT8Advanced,
                    MiniMaxH3AudioRefineCompatibilityPlanT8Advanced,
                    RuntimeSetup, custom_sampler_nodes.SamplerCustomAdvanced,
                    MiniMaxH3AudioRefineQualityGateT8Advanced,
                    MiniMaxH3AudioRefineFrozenFirstPassLoadEXPT8,
                    MiniMaxH3AVDecodeT8, CreateVideo, SaveVideo,
                    CaptureFullResume, *STAGE_NODES):
            name = cls.define_schema().node_id if hasattr(cls, "define_schema") else cls.__name__
            patch.setitem(nodes.NODE_CLASS_MAPPINGS, name, cls)
        patch.setitem(nodes.NODE_CLASS_MAPPINGS, "LoraLoaderBypassModelOnly",
                      ObservedRefineBypass)
        TinyRefineUNET.calls.clear()
        CaptureFullResume.observations.clear()
        ObservedRefineBypass.observations.clear()
        relay_calls = []
        previous_route = relay.route_prompt_relay_attention

        def observed(*args, **kwargs):
            relay_calls.append(kwargs["transformer_options"][relay.PROMPT_RELAY_RUNTIME_KEY][
                "binding_hash"])
            return previous_route(*args, **kwargs)

        patch.setattr(relay, "route_prompt_relay_attention", observed)
        server = SimpleNamespace(client_id=None, last_node_id=None, sockets_metadata={},
                                 send_sync=lambda *_args, **_kwargs: None)
        executor = execution.PromptExecutor(server,
            cache_args={"ram": 0., "ram_inactive": 0.},
            asset_manager=SimpleNamespace(enabled=False))
        executor.execute(graph, "s26-cold-resume-child-cpu", execute_outputs=["200", "31"])
        assert executor.success, executor.status_messages
        assert len(TinyRefineUNET.calls) == 1
        assert ObservedRefineBypass.observations == [
            (lora_strength, 1 if lora_strength else 0)]
        assert len(relay_calls) == 4
        assert len(CaptureFullResume.observations) == 1
        loaded, candidate, selected, decision, route_report, cond_report = (
            CaptureFullResume.observations[0])
        assert selected is loaded and decision == "ABSTAIN_HUMAN_REVIEW_REQUIRED"
        assert json.loads(route_report)["decision"] == "ALLOW"
        assert json.loads(cond_report)["status"] == "applied_exp"
        loaded_video, loaded_audio = loaded["samples"].unbind()
        candidate_video, candidate_audio = candidate["samples"].unbind()
        assert torch.allclose(candidate_video, loaded_video, atol=1e-5, rtol=0)
        assert not torch.equal(candidate_audio, loaded_audio)
        files = list(Path(output_root).rglob(Path(media_prefix).name + "*.mp4"))
        assert len(files) == 1
        with av.open(str(files[0])) as container:
            assert len(container.streams.video) == len(container.streams.audio) == 1
            video_frames = len(list(container.decode(video=0)))
            container.seek(0)
            assert list(container.decode(audio=0))
        assert video_frames == 22
        assert not torch.cuda.is_initialized()
        return {"loaded_video_shape": list(loaded_video.shape),
                "loaded_audio_shape": list(loaded_audio.shape),
                "candidate_audio_sha256": hashlib.sha256(
                    candidate_audio.detach().contiguous().numpy().tobytes()).hexdigest(),
                "refine_model_loads": len(TinyRefineUNET.calls),
                "refine_lora_injections": ObservedRefineBypass.observations[0][1],
                "refine_relay_calls": len(relay_calls),
                "conditioning_report_sha256": hashlib.sha256(
                    cond_report.encode("utf-8")).hexdigest(),
                "media_video_frames": video_frames}


def test_saved_freeze_graph_runs_first_pass_and_writes_verified_av(tmp_path, monkeypatch):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
    monkeypatch.syspath_prepend(str(ROOT.parents[1]))
    import execution
    import nodes
    from comfy_extras.nodes_lora_debug import LoraLoaderBypassModelOnly

    graph = deepcopy(json.loads(FREEZE.read_text(encoding="utf-8")))
    graph["13"]["inputs"].update(length=22, time_ranges="0-6\n7-13\n14-21")
    graph["6"]["inputs"].update(width=128, height=128, query_chunk_rows=64)
    graph["17"]["inputs"]["strength_model"] = 0.0
    graph["33"]["inputs"]["confirm_save"] = True
    graph["200"] = {"class_type": "CaptureFrozenFirstPass", "inputs": {
        "latent": ["33", 0], "status": ["33", 1],
        "checkpoint_path": ["33", 2], "file_sha256": ["33", 3],
        "manifest_json": ["33", 4]}}
    graph = _required_for(graph, "200")
    monkeypatch.setattr(checkpoint_nodes.folder_paths, "get_output_directory",
                        lambda: str(tmp_path))
    monkeypatch.setattr(comfy.model_management, "intermediate_device",
                        lambda: torch.device("cpu"))
    monkeypatch.setattr(custom_sampler_nodes.latent_preview, "prepare_callback",
                        lambda *_args, **_kwargs: lambda *_values: None)
    for cls in (TinyRefineUNET, TinyCandidateCLIP, TinyRefineVAE,
                MiniMaxH3PromptRelayPlanT8Advanced,
                MiniMaxH3PromptRelayPreviewT8Advanced,
                MiniMaxH3PromptRelayConditioningT8Advanced,
                LoraLoaderBypassModelOnly, MiniMaxH3DualClockSamplerT8,
                custom_sampler_nodes.BasicGuider, custom_sampler_nodes.RandomNoise,
                custom_sampler_nodes.SamplerCustomAdvanced,
                MiniMaxH3NativeLatentCheckpointSaveT8Advanced,
                CaptureFrozenFirstPass):
        name = cls.define_schema().node_id if hasattr(cls, "define_schema") else cls.__name__
        monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, name, cls)
    CaptureFrozenFirstPass.observations.clear()
    server = SimpleNamespace(client_id=None, last_node_id=None, sockets_metadata={},
                             send_sync=lambda *_args, **_kwargs: None)
    executor = execution.PromptExecutor(server, cache_args={"ram": 0., "ram_inactive": 0.},
                                        asset_manager=SimpleNamespace(enabled=False))
    executor.execute(graph, "s26-saved-freeze-cpu", execute_outputs=["200"])
    assert executor.success, executor.status_messages
    assert len(CaptureFrozenFirstPass.observations) == 1
    latent, status, checkpoint_path, file_sha256, manifest_json = (
        CaptureFrozenFirstPass.observations[0])
    assert status == "SAVED_VERIFIED"
    assert len(file_sha256) == 64 and checkpoint_path
    assert json.loads(manifest_json)["checkpoint_id"] == "audio_refine_firstpass"
    video, audio = latent["samples"].unbind()
    assert video.shape[2] == 7 and audio.shape[-1] == 37
    code = """
import json,os,runpy,sys
from pathlib import Path
runpy.run_path('tools/check_director_d1_cpu.py',run_name='config')
runpy.run_path('tests/conftest.py')
os.chdir(Path.cwd().parents[1])
from test_modular_audio_refine_cold_process import _cold_resume
payload=json.load(sys.stdin)
result=_cold_resume(**payload)
print('RESULT='+json.dumps(result,sort_keys=True))
"""
    receipt = {"output_root": str(tmp_path), "checkpoint_path": checkpoint_path,
               "file_sha256": file_sha256, "manifest_json": manifest_json}
    cold_runs = []
    for suffix, prompt, strength in (
        ("a", "A quiet moonlit voice with stable ambience.", 0.),
        ("b", "A bright bell echoes across the distant valley.", 0.),
        ("c", "A quiet moonlit voice with stable ambience.", .7),
    ):
        payload = {**receipt, "refine_prompt": prompt,
                   "media_prefix": f"MiniMaxH3/AudioRefineCompat/cold_refine_{suffix}",
                   "lora_strength": strength}
        child = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                               input=json.dumps(payload), capture_output=True,
                               text=True, timeout=90, check=False)
        assert child.returncode == 0, child.stderr
        cold = json.loads(next(line[7:] for line in child.stdout.splitlines()
                               if line.startswith("RESULT=")))
        assert cold["loaded_video_shape"] == list(video.shape)
        assert cold["loaded_audio_shape"] == list(audio.shape)
        assert cold["refine_model_loads"] == 1
        assert cold["refine_lora_injections"] == (1 if strength else 0)
        assert cold["refine_relay_calls"] == 4
        assert cold["media_video_frames"] == 22
        cold_runs.append(cold)
    assert cold_runs[0]["conditioning_report_sha256"] != cold_runs[1][
        "conditioning_report_sha256"]
    assert cold_runs[0]["conditioning_report_sha256"] == cold_runs[2][
        "conditioning_report_sha256"]
    assert cold_runs[0]["candidate_audio_sha256"] != cold_runs[2][
        "candidate_audio_sha256"]
    assert len(list((tmp_path / "MiniMaxH3" / "latent_checkpoints").glob(
        "*.h3latent.safetensors"))) == 1
    assert not torch.cuda.is_initialized()
