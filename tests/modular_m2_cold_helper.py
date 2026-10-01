"""Fresh-interpreter CPU execution of exact saved S09/S29 resume candidates."""

import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import pytest
import torch

from h3_audio_t8_pkg.modular_sampling import nodes as stage_nodes, stage_unet_loader
from h3_audio_t8_pkg.modular_sampling.fast_h3_v2_handoff_nodes import NODES as V2_HANDOFF_NODES
from h3_audio_t8_pkg.modular_sampling.stage_unet_loader_nodes import NODES as STAGE_LOADER_NODES
from h3_audio_t8_pkg.nodes import MiniMaxH3AVDecodeT8, MiniMaxH3DualClockSamplerT8
from h3_audio_t8_pkg.nodes_learned_latent_upscale_advanced import (
    MiniMaxH3LearnedTwoPassParityPlanT8Advanced,
    MiniMaxH3TwoPassAudioAuditT8Advanced,
    MiniMaxH3TwoPassLatentReconcileT8Advanced,
    MiniMaxH3TwoPassSigmaPlanT8Advanced,
)
from h3_audio_t8_pkg.nodes_detail_sampling_advanced import (
    MiniMaxH3AVTailDetailScheduleT8Advanced,
    MiniMaxH3ModelTimeBiasSamplerT8Advanced,
    MiniMaxH3SpatioTemporalGuidanceT8Advanced,
)
from h3_audio_t8_pkg.nodes_h3_lora_compat_advanced import MiniMaxH3LoRACompatibilityLoaderT8Advanced
from h3_audio_t8_pkg.nodes_prompt_relay_advanced import (
    MiniMaxH3PromptRelayConditioningT8Advanced,
    MiniMaxH3PromptRelayPlanT8Advanced,
)
from test_modular_fast_h3_v2_saved_candidate_core import (
    TinyCandidateCLIP, TinyCandidateUpscale, TinyCandidateVAE,
)
from test_modular_manual_saved_candidate_core import TinyManualUNET, _tiny


ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "manual": ROOT / "artifacts/development/modular-sampling-m2-manual-pass-20260925"
              / "candidates-v3/Manual_Pass_resume_effects_NativeNoise_EXP.api.json",
    "rf_standalone": ROOT / "artifacts/development/modular-sampling-m2-rf-restart-20260925"
                     / "candidates-v3/RF_standalone_resume_effects_EXP.api.json",
    "rf_detail_mixer": ROOT / "artifacts/development/modular-sampling-m2-rf-restart-20260925"
                       / "candidates-v3/RF_detail_mixer_resume_effects_EXP.api.json",
    "rf_two_pass_detail_mixer": ROOT / "artifacts/development/modular-sampling-m2-rf-restart-20260925"
                                / "candidates-v3/RF_two_pass_detail_mixer_resume_effects_EXP.api.json",
}


def run_cold(kind, artifact_root, artifact_path, artifact_sha256):
    """Require one actual second-stage sample in a brand-new CPU Core process."""
    native = re.fullmatch(r"native_low(4|20)_high([345])", kind)
    explicit = re.fullmatch(r"native_explicit_(base_flow4|lbh4|complete8|complete20)_high([345])", kind)
    vdn_native = re.fullmatch(r"vdn_native_(stage_dmd_8nfe|stage_b_50nfe)_high([345])", kind)
    vdn_tail = re.fullmatch(r"vdn_tail_(stage_dmd_8nfe|stage_b_50nfe)_high([45])", kind)
    pdd = re.fullmatch(r"pdd_(FL2VA|Ref2VA)_high4", kind)
    if explicit is not None:
        family, high = explicit.groups()
        if family == "base_flow" and high != "4":
            raise ValueError("Unexpected base-flow candidate")
    assert (kind in FILES or native is not None or explicit is not None
            or vdn_native is not None or vdn_tail is not None or pdd is not None)
    assert not torch.cuda.is_initialized()
    # Candidate-builder imports may prepend the extension root; Core's
    # top-level `nodes` module must win before importing `execution`.
    sys.path.insert(0, str(ROOT.parents[1]))
    import execution
    import nodes
    from comfy_extras.nodes_custom_sampler import BasicGuider, RandomNoise
    from comfy_extras.nodes_primitive import Int
    from comfy_extras.nodes_preview_any import PreviewAny
    from comfy_extras.nodes_video import CreateVideo, SaveVideo

    if native:
        source = (ROOT / "artifacts/development/modular-sampling-m2-native-dual-20260922"
                  / f"candidates-v2/Native_Dual_LOW{native[1]}_HIGH{native[2]}_resume_effects_EXP.api.json")
    elif explicit:
        family, high = explicit.groups()
        route = ("base_flow", "4") if family == "base_flow4" else (
            ("lbh", "4") if family == "lbh4" else ("complete", family.removeprefix("complete")))
        source = (ROOT / "artifacts/development/modular-sampling-m2-native-explicit-20260923"
                  / f"candidates-v4/Native_{route[0]}_{route[1]}plus{high}_resume_effects_EXP.api.json")
    elif vdn_native:
        training, high = vdn_native.groups()
        source = (ROOT / "artifacts/development/modular-sampling-m2-vdn-relay-20260923"
                  / f"candidates-v2/VDNRelay_{training}_native{high}_resume_relay_eav_EXP.api.json")
    elif vdn_tail:
        if vdn_tail[2] == "5":
            if vdn_tail[1] != "stage_b_50nfe":
                raise ValueError("Only the original B50 own-tail uses five steps")
            source = None
        else:
            source = (ROOT / "artifacts/development/modular-sampling-m2-vdn-relay-20260923"
                      / f"candidates-v2/VDNRelay_{vdn_tail[1]}_vdn4_resume_relay_eav_EXP.api.json")
    elif pdd:
        source = (ROOT / "artifacts/development/modular-sampling-m2-pdd-dynamic-20260923"
                  / f"candidates-v3/PDD_{pdd[1]}_4plus4_resume_effects_EXP.api.json")
    else:
        source = FILES[kind]
    if source is None:
        from tools.build_modular_vdn_relay_workflow import split_graph as vdn_graph
        graph = _tiny(vdn_graph(vdn_tail[1], "vdn", 5, "resume_relay_eav"))
    else:
        graph = _tiny(json.loads(source.read_text(encoding="utf-8")))
    if vdn_native or vdn_tail:
        graph["140"]["inputs"].update(global_prompt="A stable scene",
                                       local_prompts="A woman waves.\nShe walks away.", length=22)
    if pdd:
        from test_modular_pdd_saved_candidate_core import _tiny_pdd
        graph = _tiny_pdd(graph)
    graph["60"]["inputs"].update(artifact_path=artifact_path,
                                  artifact_sha256=artifact_sha256)
    if kind == "rf_two_pass_detail_mixer" or native or explicit or vdn_native:
        for item in graph.values():
            if item["class_type"] == "MiniMaxH3LoRACompatibilityLoaderT8Advanced":
                item["inputs"]["lora_name"] = "disabled"
    assert sum(node["class_type"] == "MiniMaxH3StageSamplerEXPT8" for node in graph.values()) == 1
    assert not any(node["class_type"] == "MiniMaxH3StageUNETLoaderAfterEXPT8"
                   for node in graph.values())
    selected = [
        TinyManualUNET, TinyCandidateCLIP, TinyCandidateVAE, Int,
        MiniMaxH3AVDecodeT8, PreviewAny, CreateVideo, SaveVideo,
        BasicGuider, RandomNoise,
        MiniMaxH3PromptRelayPlanT8Advanced,
        MiniMaxH3PromptRelayConditioningT8Advanced,
        stage_nodes.MiniMaxH3StageNoiseEXPT8,
        stage_nodes.MiniMaxH3StageSamplerEXPT8,
        stage_nodes.MiniMaxH3StageSaveEXPT8,
        stage_nodes.MiniMaxH3StageLoadEXPT8,
        stage_nodes.MiniMaxH3StageEAVConfigEXPT8,
        stage_nodes.MiniMaxH3StageEAVApplyEXPT8,
        stage_nodes.MiniMaxH3StageEAVAuditEXPT8,
        *STAGE_LOADER_NODES,
    ]
    if kind == "manual":
        selected.append(stage_nodes.MiniMaxH3ManualPassStageSetupEXPT8)
        stage_name = "manual_second"
    elif native:
        selected.extend((MiniMaxH3LoRACompatibilityLoaderT8Advanced,
                         TinyCandidateUpscale,
                         stage_nodes.MiniMaxH3NativeDualStageSetupEXPT8,
                         stage_nodes.MiniMaxH3NativeDualHandoffEXPT8))
        stage_name = "dual_high_" + native[2]
    elif explicit:
        selected.extend((MiniMaxH3LoRACompatibilityLoaderT8Advanced,
                         TinyCandidateUpscale,
                         MiniMaxH3DualClockSamplerT8,
                         MiniMaxH3TwoPassSigmaPlanT8Advanced,
                         MiniMaxH3LearnedTwoPassParityPlanT8Advanced,
                         MiniMaxH3TwoPassLatentReconcileT8Advanced,
                         MiniMaxH3TwoPassAudioAuditT8Advanced,
                         stage_nodes.MiniMaxH3NativeStageBindEXPT8))
        stage_name = "native_high"
    elif vdn_native:
        selected.extend((MiniMaxH3LoRACompatibilityLoaderT8Advanced,
                         TinyCandidateUpscale,
                         MiniMaxH3DualClockSamplerT8,
                         MiniMaxH3LearnedTwoPassParityPlanT8Advanced,
                         MiniMaxH3TwoPassLatentReconcileT8Advanced,
                         MiniMaxH3TwoPassAudioAuditT8Advanced,
                         stage_nodes.MiniMaxH3NativeStageBindEXPT8))
        stage_name = "native_high"
    elif vdn_tail:
        from test_modular_vdn_native_saved_candidate_core import TinyVDNComposer
        selected.extend((TinyVDNComposer,
                         TinyCandidateUpscale,
                         MiniMaxH3TwoPassLatentReconcileT8Advanced,
                         MiniMaxH3TwoPassAudioAuditT8Advanced,
                         stage_nodes.MiniMaxH3VDNStageSetupEXPT8,
                         stage_nodes.MiniMaxH3VDNRelayApplyEXPT8,
                         stage_nodes.MiniMaxH3VDNRelayAuditEXPT8))
        stage_name = "vdn_refine"
    elif pdd:
        from test_modular_pdd_saved_candidate_core import TinyPDDSetup, TinyPDDUpscale, TinyReferenceImage
        selected.extend((TinyPDDSetup, TinyPDDUpscale, TinyReferenceImage,
                         MiniMaxH3TwoPassLatentReconcileT8Advanced,
                         stage_nodes.MiniMaxH3PDDStageSetupEXPT8))
        stage_name = "pdd_high_4_8"
    else:
        selected.extend((MiniMaxH3DualClockSamplerT8,
                         MiniMaxH3AVTailDetailScheduleT8Advanced,
                         MiniMaxH3ModelTimeBiasSamplerT8Advanced,
                         MiniMaxH3SpatioTemporalGuidanceT8Advanced,
                         stage_nodes.MiniMaxH3RFHandoffEXPT8,
                         stage_nodes.MiniMaxH3RFRestartStageSetupEXPT8))
        stage_name = "rf_restart"
        if kind == "rf_two_pass_detail_mixer":
            selected.extend((MiniMaxH3LoRACompatibilityLoaderT8Advanced,
                             TinyCandidateUpscale,
                             stage_nodes.MiniMaxH3NativeDualStageSetupEXPT8,
                             stage_nodes.MiniMaxH3NativeDualHandoffEXPT8,
                             *V2_HANDOFF_NODES))
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(ROOT.parents[1]))
        for cls in selected:
            name = cls.define_schema().node_id if hasattr(cls, "define_schema") else cls.__name__
            patch.setitem(nodes.NODE_CLASS_MAPPINGS, name, cls)
        patch.setattr(stage_nodes, "_stage_store_root", lambda: Path(artifact_root))
        patch.setattr(stage_unet_loader, "_load_unet",
                      lambda *_args: pytest.fail("Cold resume must use direct UNETLoader"))
        calls, receipts, audits = [], [], []
        sample = stage_nodes.sample_stage
        audit = stage_nodes.audit_stage_eav

        def observed_sample(*args):
            calls.append(args[-1].stage)
            result = sample(*args)
            receipts.append(result[2].verify())
            return result

        def observed_audit(*args):
            latent, report = audit(*args)
            audits.append(json.loads(report))
            return latent, report

        patch.setattr(stage_nodes, "sample_stage", observed_sample)
        patch.setattr(stage_nodes, "audit_stage_eav", observed_audit)
        TinyManualUNET.calls.clear()
        torch.set_num_threads(1)
        server = SimpleNamespace(client_id=None, last_node_id=None, sockets_metadata={},
                                 send_sync=lambda *args, **kwargs: None)
        executor = execution.PromptExecutor(server,
            cache_args={"ram": 0., "ram_inactive": 0.},
            asset_manager=SimpleNamespace(enabled=False))
        outputs = (["102", "103"] if kind == "rf_two_pass_detail_mixer" else
                   ["46", "51", "113"] if vdn_tail else ["46", "51"])
        executor.execute(graph, "m2-cold-" + kind, execute_outputs=outputs)
        assert executor.success, executor.status_messages
        assert calls == [stage_name]
        assert len(TinyManualUNET.calls) == len(audits) == 1
        assert audits[0]["status"] == "observed_report_only"
        assert audits[0]["relay_attention_calls"] > 0
        assert not torch.cuda.is_initialized()
        return {"stages": calls, "request_sha256": receipts[0]["request_sha256"],
                "outputs": receipts[0]["outputs"], "audit": audits[0]["status"]}
