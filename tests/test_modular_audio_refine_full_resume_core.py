"""Execute the saved S26 cold-resume chain with real Relay/Route/tiny sampling."""
from copy import deepcopy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import av
import comfy.model_management
import comfy.nested_tensor
import comfy_extras.nodes_custom_sampler as custom_sampler_nodes
from comfy_extras.nodes_lora_debug import LoraLoaderBypassModelOnly
from comfy_api.latest import io
import folder_paths
import pytest
from safetensors.torch import save_file
import torch

from h3_audio_t8_pkg import audio_refine_advanced as refine
from h3_audio_t8_pkg import nodes_native_latent_checkpoint_advanced as checkpoint_nodes
from h3_audio_t8_pkg import prompt_relay_advanced as relay
from h3_audio_t8_pkg.modular_sampling.audio_refine_nodes import NODES as STAGE_NODES
from h3_audio_t8_pkg.modular_sampling.audio_refine_storage_nodes import (
    MiniMaxH3AudioRefineFrozenFirstPassLoadEXPT8,
)
from h3_audio_t8_pkg.native_latent_checkpoint_advanced import save_native_h3_av_checkpoint
from h3_audio_t8_pkg.nodes import MiniMaxH3AVDecodeT8
from h3_audio_t8_pkg.nodes_audio_refine_advanced import (
    MiniMaxH3AudioRefineAuditT8Advanced,
    MiniMaxH3AudioRefineCompatibilityRouteT8Advanced,
    MiniMaxH3AudioRefineCompatibilityPlanT8Advanced,
    MiniMaxH3AudioRefineCompatibilitySetupT8Advanced,
    MiniMaxH3AudioRefineQualityGateT8Advanced,
)
from h3_audio_t8_pkg.nodes_prompt_relay_advanced import (
    MiniMaxH3PromptRelayPlanT8Advanced,
    MiniMaxH3PromptRelayConditioningT8Advanced,
)
from helpers import FakeAudioVAE, FakeVideoVAE
from test_audio_refine_advanced import _runtime
from test_fast_h3_v2_core_sampler import model
from test_modular_fast_h3_v2_saved_candidate_core import TinyCandidateCLIP


ROOT = Path(__file__).resolve().parents[1]
RESUME = (ROOT / "artifacts/development/modular-sampling-m4-audio-refine-independent-model-20260924"
          / "resume-pair-v3/02_resume_refine_only.api.json")


class TinyRefineUNET(io.ComfyNode):
    calls = []

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="UNETLoader", inputs=[
            io.String.Input("unet_name"), io.String.Input("weight_dtype")],
            outputs=[io.Model.Output("model")])

    @classmethod
    def execute(cls, unet_name, weight_dtype):
        cls.calls.append((unet_name, weight_dtype))
        return io.NodeOutput(model())


class TinyRefineVAE(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="VAELoader", inputs=[io.String.Input("vae_name")],
                         outputs=[io.Vae.Output("vae")])

    @classmethod
    def execute(cls, vae_name):
        return io.NodeOutput(FakeAudioVAE() if "audio" in vae_name else FakeVideoVAE())


class RuntimeAudit(MiniMaxH3AudioRefineAuditT8Advanced):
    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*refine.audit_audio_refine(**kwargs, runtime_snapshot_fn=_runtime))


class RuntimeSetup(MiniMaxH3AudioRefineCompatibilitySetupT8Advanced):
    @classmethod
    def execute(cls, plan, refine_model, positive, av_latent):
        result = refine.setup_audio_refine_compatibility(
            plan=plan, refine_model=refine_model, positive=positive,
            av_latent=av_latent, runtime_snapshot_fn=_runtime)
        return io.NodeOutput(result.model, result.noise, result.guider,
                             result.sampler, result.sigmas, result.latent,
                             result.report_json)


class CaptureFullResume(io.ComfyNode):
    observations = []

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="CaptureFullResume", is_output_node=True,
                         inputs=[io.Latent.Input("original"), io.Latent.Input("candidate"),
                                 io.Latent.Input("selected"), io.String.Input("decision"),
                                 io.String.Input("route_report"),
                                 io.String.Input("conditioning_report")],
                         outputs=[io.String.Output("decision")])

    @classmethod
    def execute(cls, original, candidate, selected, decision, route_report,
                conditioning_report):
        cls.observations.append((original, candidate, selected, decision,
                                 route_report, conditioning_report))
        return io.NodeOutput(decision)


class ObservedRefineBypass(LoraLoaderBypassModelOnly):
    observations = []

    def load_lora_model_only(self, model, lora_name, strength_model):
        result = super().load_lora_model_only(model, lora_name, strength_model)
        self.observations.append((strength_model,
                                  len(result[0].get_injections("bypass_lora") or [])))
        return result


def _native_av():
    video = torch.linspace(-.1, .1, 24 * 7 * 8 * 8).reshape(1, 24, 7, 8, 8)
    audio = torch.linspace(-.1, .1, 32 * 2 * 37).reshape(1, 32, 2, 37)
    return {"samples": comfy.nested_tensor.NestedTensor((video, audio))}


def _required_for(graph, *outputs):
    keep = set(outputs)
    pending = list(outputs)
    while pending:
        node = graph[pending.pop()]
        for value in node["inputs"].values():
            if isinstance(value, list) and len(value) == 2 and str(value[0]) in graph:
                source = str(value[0])
                if source not in keep:
                    keep.add(source)
                    pending.append(source)
    return {key: graph[key] for key in graph if key in keep}


def make_tiny_refine_lora(path):
    bare = model()
    key, weight = next((key, value) for key, value in bare.model_state_dict().items()
                       if key.startswith("diffusion_model.") and
                       key.endswith(".weight") and value.ndim == 2)
    prefix = "lora_unet_" + key[len("diffusion_model."):-len(".weight")].replace(".", "_")
    save_file({prefix + ".lora_up.weight": torch.full((weight.shape[0], 1), .02),
               prefix + ".lora_down.weight": torch.full((1, weight.shape[1]), .02)},
              str(path))


def _run_saved_resume(tmp_path, monkeypatch, export_media, lora_strength):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
    monkeypatch.syspath_prepend(str(ROOT.parents[1]))
    import execution
    import nodes
    from comfy_extras.nodes_video import CreateVideo, SaveVideo

    saved = json.loads(RESUME.read_text(encoding="utf-8"))
    graph = deepcopy(saved)
    assert "10" not in graph and "1" not in graph and "17" not in graph
    original = _native_av()
    checkpoint = save_native_h3_av_checkpoint(
        original, tmp_path / "MiniMaxH3" / "latent_checkpoints",
        filename_prefix="audio_refine_firstpass",
        checkpoint_id="audio_refine_firstpass", confirm_save=True)
    assert checkpoint[1] == "SAVED_VERIFIED"
    graph["39"]["inputs"].update(checkpoint_path=checkpoint[2],
        expected_manifest_json=checkpoint[4], expected_file_sha256=checkpoint[3])
    graph["35"]["inputs"].update(length=22, time_ranges="0-6\n7-13\n14-21")
    graph["37"]["inputs"].update(width=128, height=128, query_chunk_rows=64)
    graph["38"]["inputs"]["strength_model"] = lora_strength
    graph["28"]["inputs"]["video_frame_count"] = 22
    graph["200"] = {"class_type": "CaptureFullResume", "inputs": {
        "original": ["39", 0], "candidate": ["34", 0], "selected": ["28", 0],
        "decision": ["28", 3], "route_report": ["20", 3],
        "conditioning_report": ["37", 6]}}
    graph = _required_for(graph, "200", "31") if export_media else _required_for(graph, "200")
    monkeypatch.setattr(checkpoint_nodes.folder_paths, "get_output_directory",
                        lambda: str(tmp_path))
    monkeypatch.setattr(comfy.model_management, "intermediate_device",
                        lambda: torch.device("cpu"))
    monkeypatch.setattr(custom_sampler_nodes.latent_preview, "prepare_callback",
                        lambda *_args, **_kwargs: lambda *_values: None)
    if lora_strength:
        lora_path = tmp_path / "tiny_refine_lora.safetensors"
        make_tiny_refine_lora(lora_path)
        monkeypatch.setattr(folder_paths, "get_full_path_or_raise",
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
        monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, name, cls)
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "LoraLoaderBypassModelOnly",
                        ObservedRefineBypass)
    TinyRefineUNET.calls.clear()
    CaptureFullResume.observations.clear()
    ObservedRefineBypass.observations.clear()
    relay_calls = []
    previous_route = relay.route_prompt_relay_attention

    def observed(*args, **kwargs):
        relay_calls.append(kwargs["transformer_options"][relay.PROMPT_RELAY_RUNTIME_KEY]["binding_hash"])
        return previous_route(*args, **kwargs)

    monkeypatch.setattr(relay, "route_prompt_relay_attention", observed)
    server = SimpleNamespace(client_id=None, last_node_id=None, sockets_metadata={},
                             send_sync=lambda *_args, **_kwargs: None)
    executor = execution.PromptExecutor(server, cache_args={"ram": 0., "ram_inactive": 0.},
                                        asset_manager=SimpleNamespace(enabled=False))
    executor.execute(graph, f"s26-saved-full-resume-{export_media}-cpu",
                     execute_outputs=["200", "31"] if export_media else ["200"])
    assert executor.success, executor.status_messages
    assert len(TinyRefineUNET.calls) == 1
    assert ObservedRefineBypass.observations == [
        (lora_strength, 1 if lora_strength else 0)]
    assert len(CaptureFullResume.observations) == 1
    loaded, candidate, selected, decision, route_report, cond_report = (
        CaptureFullResume.observations[0])
    assert decision == "ABSTAIN_HUMAN_REVIEW_REQUIRED"
    assert json.loads(route_report)["decision"] == "ALLOW"
    assert json.loads(cond_report)["status"] == "applied_exp"
    assert len(relay_calls) == 4
    loaded_video, loaded_audio = loaded["samples"].unbind()
    original_video, original_audio = original["samples"].unbind()
    candidate_video, candidate_audio = candidate["samples"].unbind()
    assert torch.equal(loaded_video, original_video)
    assert torch.equal(loaded_audio, original_audio)
    assert torch.allclose(candidate_video, loaded_video, atol=1e-5, rtol=0)
    assert not torch.equal(candidate_audio, loaded_audio)
    assert selected is loaded
    if export_media:
        files = list(tmp_path.rglob("prompt_relay_turbo8_selected*.mp4"))
        assert len(files) == 1
        with av.open(str(files[0])) as container:
            assert len(container.streams.video) == len(container.streams.audio) == 1
            assert len(list(container.decode(video=0))) == 22
            container.seek(0)
            assert list(container.decode(audio=0))
    assert not torch.cuda.is_initialized()
    return candidate_audio.detach().clone()


@pytest.mark.parametrize("export_media,lora_strength", [
    (False, 0.), (True, 0.), (False, .7),
])
def test_saved_resume_executes_independent_relay_route_and_tail(tmp_path, monkeypatch,
                                                                export_media, lora_strength):
    _run_saved_resume(tmp_path, monkeypatch, export_media, lora_strength)


def test_nonzero_refine_lora_changes_only_the_tail_candidate(tmp_path):
    plain_dir, lora_dir = tmp_path / "plain", tmp_path / "lora"
    plain_dir.mkdir()
    lora_dir.mkdir()
    with pytest.MonkeyPatch.context() as patch:
        plain = _run_saved_resume(plain_dir, patch, False, 0.)
    with pytest.MonkeyPatch.context() as patch:
        patched = _run_saved_resume(lora_dir, patch, False, .7)
    assert not torch.equal(plain, patched)
    assert not torch.cuda.is_initialized()
