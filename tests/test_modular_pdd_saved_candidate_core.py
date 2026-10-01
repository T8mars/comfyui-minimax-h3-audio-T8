"""Execute current saved PDD 4+4 pairs in Core with explicit tiny CPU assets.

The original dynamic PDD head/backbone and both stage samplers run. Downloaded
weights, learned 3D network, original geometry, and media quality are not tested.
"""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import comfy.nested_tensor
from comfy_api.latest import io
import pytest
import torch
import torch.nn.functional as F

from h3_audio_t8_pkg import sampling
from h3_audio_t8_pkg.modular_sampling import nodes as stage_nodes, stage_unet_loader
from h3_audio_t8_pkg.modular_sampling.stage_unet_loader_nodes import NODES as STAGE_LOADER_NODES
from h3_audio_t8_pkg.modular_sampling.storage import file_sha
from h3_audio_t8_pkg.nodes import MiniMaxH3AVDecodeT8
from h3_audio_t8_pkg.nodes_learned_latent_upscale_advanced import MiniMaxH3TwoPassLatentReconcileT8Advanced
from h3_audio_t8_pkg.nodes_pdd_advanced import MiniMaxH3PDD8StepSetupT8Advanced
from h3_audio_t8_pkg.nodes_prompt_relay_advanced import (
    MiniMaxH3PromptRelayConditioningT8Advanced,
    MiniMaxH3PromptRelayPlanT8Advanced,
)
from test_modular_fast_h3_v2_saved_candidate_core import TinyCandidateCLIP, TinyCandidateUpscale, TinyCandidateVAE
from test_modular_manual_saved_candidate_core import TinyManualUNET, _candidate, _cold_process, _tiny
from test_modular_pdd_dynamic import dynamic_model
from tools.build_modular_pdd_workflow import split_graph


ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ROOT / "artifacts/development/modular-sampling-m2-pdd-dynamic-20260923/candidates-v3"


class TinyReferenceImage(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="LoadImage", inputs=[io.String.Input("image")],
                         outputs=[io.Image.Output("image"), io.Mask.Output("mask")])

    @classmethod
    def execute(cls, image):
        assert image == "10A.jpg"
        return io.NodeOutput(torch.full((1, 128, 128, 3), .5), torch.zeros((1, 128, 128)))


class TinyPDDSetup(MiniMaxH3PDD8StepSetupT8Advanced):
    calls = []

    @classmethod
    def execute(cls, model, av_latent, pdd_lora_name, base_variant, strength):
        assert model is not None and pdd_lora_name.endswith("_comfyui_pdd.safetensors")
        assert strength == 1. and base_variant in ("FL2VA", "Ref2VA")
        cls.calls.append(base_variant)
        # Explicit synthetic weight double; the production PDD dynamic
        # injection, head selection, stage setup and sampler still execute.
        prepared = dynamic_model(strength, base_variant, base=model)
        prepared, sampler, sigmas = sampling.setup_dual_clock_sampling(
            prepared, av_latent, 8, 12., 3., "euler", "simple")
        return io.NodeOutput(prepared, sampler, sigmas, json.dumps({"asset_double": "synthetic_pdd"}))


class TinyPDDUpscale(TinyCandidateUpscale):
    @classmethod
    def execute(cls, av_latent, model_name, size_mode, scale_by, target_megapixels,
                target_width, target_height, aspect_policy, max_anisotropy, precision,
                release_policy):
        assert model_name.endswith("latent_upscaler_3d_fp16.safetensors")
        assert size_mode == "scale_by" and scale_by in (1.5, 2.)
        assert aspect_policy == "preserve_source" and precision == "fp16"
        assert release_policy == "offload_after"
        video, audio = av_latent["samples"].unbind()
        width = round(video.shape[-1] * 16 * scale_by)
        height = round(video.shape[-2] * 16 * scale_by)
        lifted = F.interpolate(video, size=(video.shape[2], height // 16, width // 16),
                               mode="trilinear", align_corners=False)
        return io.NodeOutput({"samples": comfy.nested_tensor.NestedTensor((lifted, audio))},
                             width, height, json.dumps({"network": "explicit_cpu_interpolation"}))


def _tiny_pdd(graph):
    graph = _tiny(graph)
    for key in ("97", "98"):
        if key in graph:
            graph[key]["inputs"].update(width=128, height=64)
    return graph


@pytest.mark.parametrize("base", ("FL2VA", "Ref2VA"))
def test_saved_pdd_pair_only_resumes_high(monkeypatch, tmp_path, base):
    monkeypatch.syspath_prepend(str(ROOT.parents[1]))
    import execution
    import nodes
    from comfy_extras.nodes_custom_sampler import BasicGuider, RandomNoise
    from comfy_extras.nodes_preview_any import PreviewAny
    from comfy_extras.nodes_video import CreateVideo, SaveVideo

    stem = f"PDD_{base}_4plus4"
    full = _tiny_pdd(_candidate(CANDIDATES / f"{stem}_save_effects_EXP.api.json",
                                split_graph(base, "save_effects")))
    resume = _tiny_pdd(_candidate(CANDIDATES / f"{stem}_resume_effects_EXP.api.json",
                                  split_graph(base, "resume_effects")))
    expected_frames = 124 if base == "FL2VA" else 22
    saved_full = json.loads((CANDIDATES / f"{stem}_save_effects_EXP.api.json").read_text(encoding="utf-8"))
    saved_resume = json.loads((CANDIDATES / f"{stem}_resume_effects_EXP.api.json").read_text(encoding="utf-8"))
    assert saved_full["40"]["inputs"]["length"] == expected_frames
    assert saved_full["47"]["inputs"]["length"] == expected_frames
    assert saved_resume["47"]["inputs"]["length"] == expected_frames
    assert resume["22"]["class_type"] == "UNETLoader"
    assert sum(node["class_type"] == "MiniMaxH3StageSamplerEXPT8" for node in resume.values()) == 1
    assert not {"1", "9", "10", "13", "40", "43", "45", "50", "90"} & set(resume)
    for cls in (TinyManualUNET, TinyCandidateCLIP, TinyCandidateVAE,
                TinyReferenceImage, TinyPDDSetup, TinyPDDUpscale,
                MiniMaxH3AVDecodeT8, MiniMaxH3TwoPassLatentReconcileT8Advanced,
                PreviewAny, CreateVideo, SaveVideo, BasicGuider, RandomNoise,
                MiniMaxH3PromptRelayPlanT8Advanced,
                MiniMaxH3PromptRelayConditioningT8Advanced,
                stage_nodes.MiniMaxH3PDDStageSetupEXPT8,
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
    high_loads, stages, receipts, audits = [], [], [], []

    def load_high(name, dtype):
        high_loads.append((name, dtype))
        from test_fast_h3_v2_core_sampler import model
        return model()

    monkeypatch.setattr(stage_unet_loader, "_load_unet", load_high)
    sample = stage_nodes.sample_stage
    audit = stage_nodes.audit_stage_eav

    def observed_sample(*args):
        stages.append(args[-1].stage)
        result = sample(*args)
        receipts.append(result[2].verify())
        return result

    def observed_audit(*args):
        latent, report = audit(*args)
        audits.append(json.loads(report))
        return latent, report

    monkeypatch.setattr(stage_nodes, "sample_stage", observed_sample)
    monkeypatch.setattr(stage_nodes, "audit_stage_eav", observed_audit)
    TinyManualUNET.calls.clear()
    TinyPDDSetup.calls.clear()
    torch.set_num_threads(1)
    server = SimpleNamespace(client_id=None, last_node_id=None, sockets_metadata={},
                             send_sync=lambda *args, **kwargs: None)

    def executor():
        return execution.PromptExecutor(server, cache_args={"ram": 0., "ram_inactive": 0.},
                                        asset_manager=SimpleNamespace(enabled=False))

    core = executor()
    core.execute(deepcopy(full), f"pdd-{base}-full-tiny", execute_outputs=["45", "46", "50", "51"])
    assert core.success, core.status_messages
    assert stages == ["pdd_low_0_4", "pdd_high_4_8"]
    assert TinyPDDSetup.calls == [base, base]
    assert len(TinyManualUNET.calls) == len(high_loads) == 1
    assert len(audits) == 2 and all(a["status"] == "observed_report_only" for a in audits)
    assert all(a["relay_attention_calls"] > 0 for a in audits)
    expected_high = receipts[-1]
    edited = deepcopy(full)
    edited["47"]["inputs"]["local_prompts"] = "She turns away.\nShe walks toward the light."
    stages.clear()
    core.execute(edited, f"pdd-{base}-high-plan-edit-tiny", execute_outputs=["46", "51"])
    assert core.success, core.status_messages
    assert stages == ["pdd_high_4_8"]
    assert receipts[-1]["request_sha256"] != expected_high["request_sha256"]
    first_manifest = next(path for path in tmp_path.rglob("manifest.json")
                          if json.loads(path.read_text(encoding="utf-8"))["stage_context"]["stage"]
                          == "pdd_low_0_4")
    resume["60"]["inputs"].update(artifact_path=first_manifest.relative_to(tmp_path).as_posix(),
                                  artifact_sha256=file_sha(first_manifest))
    stages.clear()
    audits.clear()
    core = executor()
    core.execute(deepcopy(resume), f"pdd-{base}-high-only-tiny", execute_outputs=["46", "51"])
    assert core.success, core.status_messages
    assert stages == ["pdd_high_4_8"]
    assert len(TinyPDDSetup.calls) == 4  # LOW, HIGH, edited HIGH, cold HIGH.
    assert len(TinyManualUNET.calls) == 2 and len(high_loads) == 1
    assert len(audits) == 1 and audits[0]["status"] == "observed_report_only"
    assert audits[0]["relay_attention_calls"] > 0
    assert receipts[-1]["request_sha256"] == expected_high["request_sha256"]
    assert receipts[-1]["outputs"] == expected_high["outputs"]
    cold = _cold_process(f"pdd_{base}_high4", tmp_path,
                         resume["60"]["inputs"]["artifact_path"],
                         resume["60"]["inputs"]["artifact_sha256"])
    assert cold["stages"] == ["pdd_high_4_8"]
    assert cold["request_sha256"] == expected_high["request_sha256"]
    assert cold["outputs"] == expected_high["outputs"]
    bad = deepcopy(resume)
    bad["60"]["inputs"]["artifact_sha256"] = "0" * 64
    stages.clear()
    core = executor()
    core.execute(bad, f"pdd-{base}-bad-manifest", execute_outputs=["46"])
    assert not core.success and not stages
    assert not torch.cuda.is_initialized()
