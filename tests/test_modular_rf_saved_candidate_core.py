"""Execute the saved S29 standalone RF pair with a tiny CPU model."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from h3_audio_t8_pkg.modular_sampling import nodes as stage_nodes, stage_unet_loader
from h3_audio_t8_pkg.modular_sampling.fast_h3_v2_handoff_nodes import NODES as V2_HANDOFF_NODES
from h3_audio_t8_pkg.modular_sampling.stage_unet_loader_nodes import NODES as STAGE_LOADER_NODES
from h3_audio_t8_pkg.modular_sampling.storage import file_sha
from h3_audio_t8_pkg.nodes import MiniMaxH3AVDecodeT8, MiniMaxH3DualClockSamplerT8
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
from test_fast_h3_v2_core_sampler import model
from test_modular_fast_h3_v2_saved_candidate_core import (
    TinyCandidateCLIP, TinyCandidateUpscale, TinyCandidateVAE,
)
from test_modular_manual_saved_candidate_core import TinyManualUNET, _candidate, _cold_process, _tiny
from tools.build_modular_rf_workflow import split_graph


ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ROOT / "artifacts/development/modular-sampling-m2-rf-restart-20260925/candidates-v3"
@pytest.mark.parametrize("entry", ["standalone", "detail_mixer", "two_pass_detail_mixer"])
def test_saved_rf_base_to_restart_only_core_resume(monkeypatch, tmp_path, entry):
    monkeypatch.syspath_prepend(str(ROOT.parents[1]))
    import execution
    import nodes
    from comfy_extras.nodes_custom_sampler import BasicGuider, RandomNoise
    from comfy_extras.nodes_primitive import Int
    from comfy_extras.nodes_preview_any import PreviewAny
    from comfy_extras.nodes_video import CreateVideo, SaveVideo

    full_path = CANDIDATES / f"RF_{entry}_save_effects_EXP.api.json"
    resume_path = CANDIDATES / f"RF_{entry}_resume_effects_EXP.api.json"
    full = _tiny(_candidate(full_path, split_graph(entry, "save_effects")))
    resume = _tiny(_candidate(resume_path, split_graph(entry, "resume_effects")))
    if entry == "two_pass_detail_mixer":
        # Keep real source/Handoff/Stage logic, substitute only the heavy
        # pretrained LoRA and learned 3D weights in the CPU execution copy.
        for graph in (full, resume):
            for item in graph.values():
                if item["class_type"] == "MiniMaxH3LoRACompatibilityLoaderT8Advanced":
                    item["inputs"]["lora_name"] = "disabled"
    assert sum(item["class_type"] == "MiniMaxH3StageSamplerEXPT8" for item in resume.values()) == 1
    for cls in (TinyManualUNET, TinyCandidateCLIP, TinyCandidateVAE, Int,
                MiniMaxH3AVDecodeT8, MiniMaxH3DualClockSamplerT8,
                MiniMaxH3AVTailDetailScheduleT8Advanced,
                MiniMaxH3ModelTimeBiasSamplerT8Advanced,
                MiniMaxH3SpatioTemporalGuidanceT8Advanced,
                MiniMaxH3LoRACompatibilityLoaderT8Advanced,
                TinyCandidateUpscale,
                PreviewAny, CreateVideo, SaveVideo, BasicGuider, RandomNoise,
                MiniMaxH3PromptRelayPlanT8Advanced,
                MiniMaxH3PromptRelayConditioningT8Advanced,
                stage_nodes.MiniMaxH3RFBaseStageSetupEXPT8,
                stage_nodes.MiniMaxH3NativeDualStageSetupEXPT8,
                stage_nodes.MiniMaxH3NativeDualHandoffEXPT8,
                stage_nodes.MiniMaxH3RFHandoffEXPT8,
                stage_nodes.MiniMaxH3RFRestartStageSetupEXPT8,
                stage_nodes.MiniMaxH3StageNoiseEXPT8,
                stage_nodes.MiniMaxH3StageSamplerEXPT8,
                stage_nodes.MiniMaxH3StageSaveEXPT8,
                stage_nodes.MiniMaxH3StageLoadEXPT8,
                stage_nodes.MiniMaxH3StageEAVConfigEXPT8,
                stage_nodes.MiniMaxH3StageEAVApplyEXPT8,
                stage_nodes.MiniMaxH3StageEAVAuditEXPT8,
                *STAGE_LOADER_NODES, *V2_HANDOFF_NODES):
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
    full_outputs = ["45", "46", "50", "51"]
    resume_outputs = ["46", "51"]
    if entry == "two_pass_detail_mixer":
        full_outputs += ["102", "103"]
        resume_outputs = ["102", "103"]
    core.execute(deepcopy(full), "s29-" + entry + "-full-tiny", execute_outputs=full_outputs)
    assert core.success, core.status_messages
    assert stages == (["dual_low_4", "rf_base", "rf_restart"]
                      if entry == "two_pass_detail_mixer" else ["rf_base", "rf_restart"])
    assert len(TinyManualUNET.calls) == 1
    assert len(high_loads) == (2 if entry == "two_pass_detail_mixer" else 1)
    assert len(audits) == (3 if entry == "two_pass_detail_mixer" else 2)
    assert all(item["status"] == "observed_report_only" for item in audits)
    expected_restart = receipts[-1]
    first_manifest = next(path for path in tmp_path.rglob("manifest.json")
                          if json.loads(path.read_text(encoding="utf-8"))["stage_context"]["stage"]
                          == "rf_base")
    resume["60"]["inputs"].update(artifact_path=first_manifest.relative_to(tmp_path).as_posix(),
                                  artifact_sha256=file_sha(first_manifest))
    stages.clear()
    audits.clear()
    core = executor()
    core.execute(deepcopy(resume), "s29-" + entry + "-restart-only-tiny",
                 execute_outputs=resume_outputs)
    assert core.success, core.status_messages
    assert stages == ["rf_restart"]
    assert len(TinyManualUNET.calls) == 2
    assert len(high_loads) == (2 if entry == "two_pass_detail_mixer" else 1)
    assert len(audits) == 1 and audits[0]["status"] == "observed_report_only"
    assert receipts[-1]["request_sha256"] == expected_restart["request_sha256"]
    assert receipts[-1]["outputs"] == expected_restart["outputs"]
    cold = _cold_process("rf_" + entry, tmp_path, resume["60"]["inputs"]["artifact_path"],
                         resume["60"]["inputs"]["artifact_sha256"])
    assert cold["stages"] == ["rf_restart"]
    assert cold["request_sha256"] == expected_restart["request_sha256"]
    assert cold["outputs"] == expected_restart["outputs"]

    corrupted = deepcopy(resume)
    corrupted["60"]["inputs"]["artifact_sha256"] = "0" * 64
    stages.clear()
    core = executor()
    core.execute(corrupted, "s29-" + entry + "-bad-manifest-tiny",
                 execute_outputs=[resume_outputs[0]])
    assert not core.success and not stages
    assert not torch.cuda.is_initialized()
