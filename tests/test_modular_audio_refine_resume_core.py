"""The saved cold-resume API actually loads AV and runs only the tail in Core."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import comfy.model_management
import comfy.nested_tensor
import comfy_extras.nodes_custom_sampler as custom_sampler_nodes
from comfy_api.latest import io
import pytest
import torch

from h3_audio_t8_pkg import audio_refine_advanced as refine
from h3_audio_t8_pkg import nodes_native_latent_checkpoint_advanced as checkpoint_nodes
from h3_audio_t8_pkg.modular_sampling.audio_refine_storage_nodes import (
    MiniMaxH3AudioRefineFrozenFirstPassLoadEXPT8,
)
from h3_audio_t8_pkg.native_latent_checkpoint_advanced import save_native_h3_av_checkpoint
from h3_audio_t8_pkg.modular_sampling.audio_refine_nodes import (
    MiniMaxH3AudioRefineCompatStageBindEXPT8,
    MiniMaxH3AudioRefineCompatStageAuditEXPT8,
    MiniMaxH3AudioRefineTailDeliveryAuditEXPT8,
)
from h3_audio_t8_pkg.nodes_audio_refine_advanced import MiniMaxH3AudioRefineQualityGateT8Advanced
from test_modular_audio_refine_saved_tail_core import (
    TinySignedSource, TinyCompatSetup, ObservedCoreSampler, CaptureTail, SAMPLER_CALLS,
)
from test_audio_refine_advanced import _audit, _positive, _audio, FakeModelPatcher


ROOT = Path(__file__).resolve().parents[1]
RESUME = (ROOT / "artifacts/development/modular-sampling-m4-audio-refine-independent-model-20260924"
          / "resume-pair-v3/02_resume_refine_only.api.json")


class TinyNativeSignedSource(TinySignedSource):
    @classmethod
    def execute(cls):
        video = torch.arange(1 * 24 * 7 * 2 * 2, dtype=torch.float32).reshape(1, 24, 7, 2, 2)
        audio = torch.arange(1 * 32 * 2 * 37, dtype=torch.float32).reshape(1, 32, 2, 37)
        original = {"samples": comfy.nested_tensor.NestedTensor((video, audio))}
        model = FakeModelPatcher()
        positive = _positive()
        audit, decision, _ = _audit(model=model, positive=positive, av_latent=original)
        assert decision == "ALLOW"
        routed, route, decision, _ = refine.route_audio_refine_compatibility(
            audit=audit, refine_model=model, positive=positive,
            generation_profile="turbo8", declared_first_pass_nfe=8)
        assert routed is model and decision == "ALLOW"
        plan, decision, _ = refine.plan_audio_refine_compatibility(route, 4, .5, 29)
        assert decision == "ALLOW"
        return io.NodeOutput(model, positive, original, plan, _audio())


def _only(graph, kind):
    matched = [node_id for node_id, node in graph.items() if node["class_type"] == kind]
    assert len(matched) == 1, (kind, matched)
    return matched[0]


def _tail_only(saved):
    setup = _only(saved, "MiniMaxH3AudioRefineCompatibilitySetupT8Advanced")
    bind = _only(saved, "MiniMaxH3AudioRefineCompatStageBindEXPT8")
    audit = _only(saved, "MiniMaxH3AudioRefineCompatStageAuditEXPT8")
    tail = _only(saved, "MiniMaxH3AudioRefineTailDeliveryAuditEXPT8")
    gate = _only(saved, "MiniMaxH3AudioRefineQualityGateT8Advanced")
    load = _only(saved, "MiniMaxH3AudioRefineFrozenFirstPassLoadEXPT8")
    samplers = [node_id for node_id, node in saved.items()
                if node["class_type"] == "SamplerCustomAdvanced"
                and node["inputs"].get("sigmas") == [bind, 4]]
    assert len(samplers) == 1
    sampler = samplers[0]
    graph = {node_id: deepcopy(saved[node_id])
             for node_id in (setup, bind, audit, tail, gate, load, sampler)}
    graph["100"] = {"class_type": "TinyNativeSignedSource", "inputs": {}}
    for node_id, fields in (
        (setup, {"plan": ["100", 3], "refine_model": ["100", 0],
                 "positive": ["100", 1], "av_latent": [load, 0]}),
        (bind, {"plan": ["100", 3], "original_av_latent": [load, 0]}),
        (audit, {"plan": ["100", 3], "original_av_latent": [load, 0]}),
        (gate, {"original_av_latent": [load, 0],
                "original_audio": ["100", 4], "candidate_audio": ["100", 4]}),
    ):
        graph[node_id]["inputs"].update(fields)
    graph[gate]["inputs"]["video_frame_count"] = 24
    graph["200"] = {"class_type": "CaptureTail", "inputs": {
        "original": [load, 0], "candidate": [audit, 0], "selected": [gate, 0],
        "stage_report": [audit, 1], "tail_report": [tail, 1],
        "quality_report": [gate, 4], "decision": [gate, 3]}}
    assert "10" not in graph
    return graph, load


@pytest.mark.parametrize("bad_evidence", [
    "none", "sha_bad", "sha_blank", "manifest_blank", "path_blank", "wrong_id",
])
def test_saved_resume_graph_loads_frozen_av_and_only_samples_refine(
        tmp_path, monkeypatch, bad_evidence):
    monkeypatch.syspath_prepend(str(ROOT.parents[1]))
    import execution
    import nodes

    saved = json.loads(RESUME.read_text(encoding="utf-8"))
    graph, load = _tail_only(saved)
    original = TinyNativeSignedSource.execute().result[2]
    store = tmp_path / "MiniMaxH3" / "latent_checkpoints"
    checkpoint = save_native_h3_av_checkpoint(
        original, store, filename_prefix="audio_refine_firstpass",
        checkpoint_id=("wrong_audio_refine_source" if bad_evidence == "wrong_id"
                       else "audio_refine_firstpass"), confirm_save=True)
    assert checkpoint[1] == "SAVED_VERIFIED"
    graph[load]["inputs"].update(
        checkpoint_path="" if bad_evidence == "path_blank" else checkpoint[2],
        expected_manifest_json="" if bad_evidence == "manifest_blank" else checkpoint[4],
        expected_file_sha256=("0" * 64 if bad_evidence == "sha_bad"
                              else "" if bad_evidence == "sha_blank" else checkpoint[3]))
    monkeypatch.setattr(checkpoint_nodes.folder_paths, "get_output_directory",
                        lambda: str(tmp_path))
    monkeypatch.setattr(comfy.model_management, "intermediate_device",
                        lambda: torch.device("cpu"))
    monkeypatch.setattr(custom_sampler_nodes.latent_preview, "prepare_callback",
                        lambda *_args, **_kwargs: lambda *_values: None)

    def cpu_audio_only(self, _noise, latent, _sampler, sigmas, *, denoise_mask,
                       callback, disable_pbar, seed):
        del callback, disable_pbar, seed
        assert len(sigmas) == 5
        video_mask, audio_mask = denoise_mask.unbind()
        assert not video_mask.count_nonzero() and bool(torch.all(audio_mask == 1))
        video, audio = latent.unbind()
        return comfy.nested_tensor.NestedTensor((video, audio + 0.25))

    monkeypatch.setattr(refine.AudioRefineBasicGuider, "sample", cpu_audio_only)
    for cls in (TinyNativeSignedSource, TinyCompatSetup, ObservedCoreSampler,
                MiniMaxH3AudioRefineCompatStageBindEXPT8,
                MiniMaxH3AudioRefineCompatStageAuditEXPT8,
                MiniMaxH3AudioRefineTailDeliveryAuditEXPT8,
                MiniMaxH3AudioRefineQualityGateT8Advanced,
                MiniMaxH3AudioRefineFrozenFirstPassLoadEXPT8,
                CaptureTail):
        monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, cls.define_schema().node_id, cls)
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "SamplerCustomAdvanced",
                        ObservedCoreSampler)
    SAMPLER_CALLS.clear()
    CaptureTail.observations.clear()
    server = SimpleNamespace(client_id=None, last_node_id=None, sockets_metadata={},
                             send_sync=lambda *_args, **_kwargs: None)
    executor = execution.PromptExecutor(server, cache_args={"ram": 0., "ram_inactive": 0.},
                                        asset_manager=SimpleNamespace(enabled=False))
    executor.execute(graph, f"s26-resume-{bad_evidence}-cpu",
                     execute_outputs=["200"])
    if bad_evidence != "none":
        assert not executor.success
        assert SAMPLER_CALLS == []
        assert CaptureTail.observations == []
    else:
        assert executor.success, executor.status_messages
        assert SAMPLER_CALLS == [5]
        assert len(CaptureTail.observations) == 1
        selected = CaptureTail.observations[0][2]
        assert selected is not original  # The AV came from the actual checkpoint Load.
        selected_video, selected_audio = selected["samples"].unbind()
        original_video, original_audio = original["samples"].unbind()
        assert torch.equal(selected_video, original_video)
        assert torch.equal(selected_audio, original_audio)  # Default Quality Gate abstains.
    assert not torch.cuda.is_initialized()
