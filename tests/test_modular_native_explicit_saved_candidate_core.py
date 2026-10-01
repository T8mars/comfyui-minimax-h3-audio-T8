"""Execute saved S01/S03 native explicit pairs in Core with tiny CPU assets.

No trained weights, learned 3D checkpoint, media quality, or GPU is certified.
"""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from h3_audio_t8_pkg.modular_sampling import nodes as stage_nodes, stage_unet_loader
from h3_audio_t8_pkg.modular_sampling.stage_unet_loader_nodes import NODES as STAGE_LOADER_NODES
from h3_audio_t8_pkg.modular_sampling.storage import file_sha
from h3_audio_t8_pkg.nodes import MiniMaxH3AVDecodeT8, MiniMaxH3DualClockSamplerT8
from h3_audio_t8_pkg.nodes_h3_lora_compat_advanced import MiniMaxH3LoRACompatibilityLoaderT8Advanced
from h3_audio_t8_pkg.nodes_learned_latent_upscale_advanced import (
    MiniMaxH3LearnedTwoPassParityPlanT8Advanced,
    MiniMaxH3TwoPassAudioAuditT8Advanced,
    MiniMaxH3TwoPassLatentReconcileT8Advanced,
    MiniMaxH3TwoPassSigmaPlanT8Advanced,
)
from h3_audio_t8_pkg.nodes_prompt_relay_advanced import (
    MiniMaxH3PromptRelayConditioningT8Advanced,
    MiniMaxH3PromptRelayPlanT8Advanced,
)
from test_fast_h3_v2_core_sampler import model
from test_modular_fast_h3_v2_saved_candidate_core import (
    TinyCandidateCLIP, TinyCandidateUpscale, TinyCandidateVAE,
)
from test_modular_manual_saved_candidate_core import TinyManualUNET, _candidate, _cold_process, _tiny
from tools.build_modular_native_explicit_workflow import RECIPES, split_graph


ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ROOT / "artifacts/development/modular-sampling-m2-native-explicit-20260923/candidates-v4"


@pytest.mark.parametrize("recipe", RECIPES)
def test_saved_native_explicit_pair_and_cold_high_only(monkeypatch, tmp_path, recipe):
    monkeypatch.syspath_prepend(str(ROOT.parents[1]))
    import execution
    import nodes
    from comfy_extras.nodes_custom_sampler import BasicGuider, RandomNoise
    from comfy_extras.nodes_preview_any import PreviewAny
    from comfy_extras.nodes_video import CreateVideo, SaveVideo

    name, coarse, refine = recipe
    stem = f"Native_{name}_{coarse}plus{refine}"
    full = _tiny(_candidate(CANDIDATES / f"{stem}_save_effects_EXP.api.json",
                            split_graph(recipe, "save_effects")))
    resume = _tiny(_candidate(CANDIDATES / f"{stem}_resume_effects_EXP.api.json",
                              split_graph(recipe, "resume_effects")))
    for graph in (full, resume):
        for item in graph.values():
            if item["class_type"] == "MiniMaxH3LoRACompatibilityLoaderT8Advanced":
                item["inputs"]["lora_name"] = "disabled"
    assert "62" not in full and "62" not in resume
    assert resume["22"]["class_type"] == "UNETLoader"
    assert sum(item["class_type"] == "MiniMaxH3StageSamplerEXPT8"
               for item in resume.values()) == 1
    for cls in (TinyManualUNET, TinyCandidateCLIP, TinyCandidateVAE,
                TinyCandidateUpscale, MiniMaxH3LoRACompatibilityLoaderT8Advanced,
                MiniMaxH3AVDecodeT8, MiniMaxH3DualClockSamplerT8,
                MiniMaxH3TwoPassSigmaPlanT8Advanced,
                MiniMaxH3LearnedTwoPassParityPlanT8Advanced,
                MiniMaxH3TwoPassLatentReconcileT8Advanced,
                MiniMaxH3TwoPassAudioAuditT8Advanced,
                PreviewAny, CreateVideo, SaveVideo, BasicGuider, RandomNoise,
                MiniMaxH3PromptRelayPlanT8Advanced,
                MiniMaxH3PromptRelayConditioningT8Advanced,
                stage_nodes.MiniMaxH3NativeStageBindEXPT8,
                stage_nodes.MiniMaxH3StageSamplerEXPT8,
                stage_nodes.MiniMaxH3StageSaveEXPT8,
                stage_nodes.MiniMaxH3StageLoadEXPT8,
                stage_nodes.MiniMaxH3StageEAVConfigEXPT8,
                stage_nodes.MiniMaxH3StageEAVApplyEXPT8,
                stage_nodes.MiniMaxH3StageEAVAuditEXPT8,
                *STAGE_LOADER_NODES):
        node_id = cls.define_schema().node_id if hasattr(cls, "define_schema") else cls.__name__
        monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, node_id, cls)
    monkeypatch.setattr(stage_nodes, "_stage_store_root", lambda: tmp_path)
    high_loads, stages, receipts, audits = [], [], [], []

    def load_high(unet_name, weight_dtype):
        high_loads.append((unet_name, weight_dtype))
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
    torch.set_num_threads(1)
    server = SimpleNamespace(client_id=None, last_node_id=None, sockets_metadata={},
                             send_sync=lambda *args, **kwargs: None)

    def executor():
        return execution.PromptExecutor(server, cache_args={"ram": 0., "ram_inactive": 0.},
                                        asset_manager=SimpleNamespace(enabled=False))

    core = executor()
    core.execute(deepcopy(full), f"native-explicit-{stem}-full-tiny",
                 execute_outputs=["45", "46", "50", "51"])
    assert core.success, core.status_messages
    assert stages == ["native_low", "native_high"]
    assert len(TinyManualUNET.calls) == len(high_loads) == 1
    assert len(audits) == 2 and audits[-1]["status"] == "observed_report_only"
    expected_high = receipts[-1]
    first_manifest = next(path for path in tmp_path.rglob("manifest.json")
                          if json.loads(path.read_text(encoding="utf-8"))["stage_context"]["stage"]
                          == "native_low")
    resume["60"]["inputs"].update(artifact_path=first_manifest.relative_to(tmp_path).as_posix(),
                                  artifact_sha256=file_sha(first_manifest))
    stages.clear()
    audits.clear()
    core = executor()
    core.execute(deepcopy(resume), f"native-explicit-{stem}-high-only-tiny",
                 execute_outputs=["46", "51"])
    assert core.success, core.status_messages
    assert stages == ["native_high"]
    assert len(TinyManualUNET.calls) == 2 and len(high_loads) == 1
    assert len(audits) == 1 and audits[0]["status"] == "observed_report_only"
    assert receipts[-1]["request_sha256"] == expected_high["request_sha256"]
    assert receipts[-1]["outputs"] == expected_high["outputs"]
    cold = _cold_process(f"native_explicit_{name}{coarse}_high{refine}", tmp_path,
                         resume["60"]["inputs"]["artifact_path"],
                         resume["60"]["inputs"]["artifact_sha256"])
    assert cold["stages"] == ["native_high"]
    assert cold["request_sha256"] == expected_high["request_sha256"]
    assert cold["outputs"] == expected_high["outputs"]

    bad = deepcopy(resume)
    bad["60"]["inputs"]["artifact_sha256"] = "0" * 64
    stages.clear()
    core = executor()
    core.execute(bad, f"native-explicit-{stem}-bad-manifest-tiny", execute_outputs=["46"])
    assert not core.success and not stages
    assert not torch.cuda.is_initialized()
