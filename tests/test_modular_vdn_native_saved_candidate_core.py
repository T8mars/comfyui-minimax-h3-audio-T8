"""Execute saved VDN LOW -> native or own-grid HIGH graphs with tiny CPU assets.

The actual Composer/VDN branch and native sampler execute. The downloaded
weights, learned 3D network, original dimensions, and media are not certified.
"""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from comfy_api.latest import io

from h3_audio_t8_pkg.modular_sampling import nodes as stage_nodes
from h3_audio_t8_pkg.modular_sampling.storage import file_sha
from h3_audio_t8_pkg.nodes import MiniMaxH3AVDecodeT8, MiniMaxH3DualClockSamplerT8
from h3_audio_t8_pkg.nodes_h3_lora_compat_advanced import MiniMaxH3LoRACompatibilityLoaderT8Advanced
from h3_audio_t8_pkg.nodes_learned_latent_upscale_advanced import (
    MiniMaxH3LearnedTwoPassParityPlanT8Advanced,
    MiniMaxH3TwoPassAudioAuditT8Advanced,
    MiniMaxH3TwoPassLatentReconcileT8Advanced,
)
from h3_audio_t8_pkg.nodes_prompt_relay_advanced import (
    MiniMaxH3PromptRelayConditioningT8Advanced,
    MiniMaxH3PromptRelayPlanT8Advanced,
)
from test_modular_fast_h3_v2_saved_candidate_core import (
    TinyCandidateCLIP, TinyCandidateUpscale, TinyCandidateVAE,
)
from test_modular_manual_saved_candidate_core import TinyManualUNET, _candidate, _cold_process, _tiny
from test_modular_vdn_baseline import tiny_vdn
from tools.build_modular_vdn_relay_workflow import split_graph


ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ROOT / "artifacts/development/modular-sampling-m2-vdn-relay-20260923/candidates-v2"


class TinyVDNComposer(io.ComfyNode):
    calls = []

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3VDNModelComposerT8Advanced", inputs=[
            io.Model.Input("model"), io.String.Input("vdn_root"), io.String.Input("stage"),
            io.Boolean.Input("verify_hashes"), io.Boolean.Input("allow_structural_base")],
            outputs=[io.Model.Output("model"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, model, vdn_root, stage, verify_hashes, allow_structural_base):
        assert model is not None and vdn_root == "OpenVDN/vdn-minimax-h3"
        assert verify_hashes and allow_structural_base
        cls.calls.append(stage)
        # The original Composer is exercised inside tiny_vdn with an actual
        # random VDN branch; this explicit asset double supplies tiny weights.
        prepared, _branch = tiny_vdn(stage)
        return io.NodeOutput(prepared, json.dumps({"asset_double": "tiny_random_vdn_branch"}))


@pytest.mark.parametrize("training,backend,refine", [
    (training, backend, refine)
    for training in ("stage_dmd_8nfe", "stage_b_50nfe")
    for backend, refine in (("native", 3), ("native", 4), ("native", 5), ("vdn", 4))
] + [("stage_b_50nfe", "vdn", 5)])
def test_saved_vdn_relay_pair_only_resumes_high(monkeypatch, tmp_path, request, training, backend, refine):
    monkeypatch.syspath_prepend(str(ROOT.parents[1]))
    import execution
    import nodes
    from comfy_extras.nodes_custom_sampler import BasicGuider, RandomNoise
    from comfy_extras.nodes_preview_any import PreviewAny
    from comfy_extras.nodes_video import CreateVideo, SaveVideo

    stem = f"VDNRelay_{training}_{backend}{refine}"
    if (backend, refine) == ("vdn", 5):
        # This exact old B50 tail was omitted from the earlier 32 candidate files.
        full = _tiny(split_graph(training, backend, refine, "save_relay_eav"))
        resume = _tiny(split_graph(training, backend, refine, "resume_relay_eav"))
    else:
        full = _tiny(_candidate(CANDIDATES / f"{stem}_save_relay_eav_EXP.api.json",
                                split_graph(training, backend, refine, "save_relay_eav")))
        resume = _tiny(_candidate(CANDIDATES / f"{stem}_resume_relay_eav_EXP.api.json",
                                  split_graph(training, backend, refine, "resume_relay_eav")))
    for graph in (full, resume):
        if "140" in graph:
            graph["140"]["inputs"].update(global_prompt="A stable scene",
                                           local_prompts="A woman waves.\nShe walks away.", length=22)
        for item in graph.values():
            if item["class_type"] == "MiniMaxH3LoRACompatibilityLoaderT8Advanced":
                item["inputs"]["lora_name"] = "disabled"
    assert sum(node["class_type"] == "MiniMaxH3StageSamplerEXPT8" for node in resume.values()) == 1
    assert not {"1", "9", "10", "13", "40", "43", "50", "70", "110", "112"} & set(resume)
    assert resume["22"]["class_type"] == "UNETLoader"
    for cls in (TinyManualUNET, TinyCandidateCLIP, TinyCandidateVAE,
                TinyCandidateUpscale, TinyVDNComposer,
                MiniMaxH3LoRACompatibilityLoaderT8Advanced,
                MiniMaxH3AVDecodeT8, MiniMaxH3DualClockSamplerT8,
                MiniMaxH3LearnedTwoPassParityPlanT8Advanced,
                MiniMaxH3TwoPassLatentReconcileT8Advanced,
                MiniMaxH3TwoPassAudioAuditT8Advanced,
                PreviewAny, CreateVideo, SaveVideo, BasicGuider, RandomNoise,
                MiniMaxH3PromptRelayPlanT8Advanced,
                MiniMaxH3PromptRelayConditioningT8Advanced,
                stage_nodes.MiniMaxH3VDNStageSetupEXPT8,
                stage_nodes.MiniMaxH3VDNRelayApplyEXPT8,
                stage_nodes.MiniMaxH3VDNRelayAuditEXPT8,
                stage_nodes.MiniMaxH3NativeStageBindEXPT8,
                stage_nodes.MiniMaxH3StageSamplerEXPT8,
                stage_nodes.MiniMaxH3StageSaveEXPT8,
                stage_nodes.MiniMaxH3StageLoadEXPT8,
                stage_nodes.MiniMaxH3StageEAVConfigEXPT8,
                stage_nodes.MiniMaxH3StageEAVApplyEXPT8,
                stage_nodes.MiniMaxH3StageEAVAuditEXPT8):
        node_id = cls.define_schema().node_id if hasattr(cls, "define_schema") else cls.__name__
        monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, node_id, cls)
    monkeypatch.setattr(stage_nodes, "_stage_store_root", lambda: tmp_path)
    TinyManualUNET.calls.clear()
    TinyVDNComposer.calls.clear()
    stages, receipts, relay_audits, eav_audits = [], [], [], []
    sample = stage_nodes.sample_stage
    relay_audit = stage_nodes.vdn_relay.audit
    eav_audit = stage_nodes.audit_stage_eav

    def observed_sample(*args):
        stages.append(args[-1].stage)
        result = sample(*args)
        receipts.append(result[2].verify())
        return result

    def observed_relay(*args):
        latent, report = relay_audit(*args)
        relay_audits.append(json.loads(report))
        return latent, report

    def observed_eav(*args):
        latent, report = eav_audit(*args)
        eav_audits.append(json.loads(report))
        return latent, report

    monkeypatch.setattr(stage_nodes, "sample_stage", observed_sample)
    monkeypatch.setattr(stage_nodes.vdn_relay, "audit", observed_relay)
    monkeypatch.setattr(stage_nodes, "audit_stage_eav", observed_eav)
    original_threads = torch.get_num_threads()
    request.addfinalizer(lambda: torch.set_num_threads(original_threads))
    torch.set_num_threads(1)
    server = SimpleNamespace(client_id=None, last_node_id=None, sockets_metadata={},
                             send_sync=lambda *args, **kwargs: None)

    def executor():
        return execution.PromptExecutor(server, cache_args={"ram": 0., "ram_inactive": 0.},
                                        asset_manager=SimpleNamespace(enabled=False))

    core = executor()
    full_outputs = ["46", "50", "51", "112", *(["113"] if backend == "vdn" else [])]
    core.execute(deepcopy(full), f"vdn-native-{stem}-full-tiny",
                 execute_outputs=full_outputs)
    assert core.success, core.status_messages
    high_stage = "native_high" if backend == "native" else "vdn_refine"
    assert stages == ["vdn_complete", high_stage]
    assert TinyVDNComposer.calls and set(TinyVDNComposer.calls) == {training}
    full_composer_count = len(TinyVDNComposer.calls)
    # Core may deduplicate identical bare UNETLoader values across the two
    # branches; the Composer/LoRA create independently editable MODEL clones.
    assert len(TinyManualUNET.calls) == 1
    assert len(relay_audits) == (1 if backend == "native" else 2)
    assert all(item["status"] == "observed_apply_exp" for item in relay_audits)
    assert len(eav_audits) == 2 and eav_audits[-1]["status"] == "observed_report_only"
    expected_high = receipts[-1]
    first_manifest = next(path for path in tmp_path.rglob("manifest.json")
                          if json.loads(path.read_text(encoding="utf-8"))["stage_context"]["stage"]
                          == "vdn_complete")
    resume["60"]["inputs"].update(artifact_path=first_manifest.relative_to(tmp_path).as_posix(),
                                  artifact_sha256=file_sha(first_manifest))
    stages.clear()
    relay_audits.clear()
    eav_audits.clear()
    core = executor()
    resume_outputs = ["46", "51", *(["113"] if backend == "vdn" else [])]
    core.execute(deepcopy(resume), f"vdn-native-{stem}-resume-tiny", execute_outputs=resume_outputs)
    assert core.success, core.status_messages
    assert stages == [high_stage]
    assert len(TinyVDNComposer.calls) == full_composer_count + (backend == "vdn")
    assert len(TinyManualUNET.calls) == 2
    assert len(relay_audits) == (backend == "vdn") and len(eav_audits) == 1
    assert receipts[-1]["request_sha256"] == expected_high["request_sha256"]
    assert receipts[-1]["outputs"] == expected_high["outputs"]
    kind = (f"vdn_native_{training}_high{refine}" if backend == "native" else
            f"vdn_tail_{training}_high{refine}")
    cold = _cold_process(kind, tmp_path,
                         resume["60"]["inputs"]["artifact_path"],
                         resume["60"]["inputs"]["artifact_sha256"])
    assert cold["stages"] == [high_stage]
    assert cold["request_sha256"] == expected_high["request_sha256"]
    assert cold["outputs"] == expected_high["outputs"]

    bad = deepcopy(resume)
    bad["60"]["inputs"]["artifact_sha256"] = "0" * 64
    stages.clear()
    core = executor()
    core.execute(bad, f"vdn-native-{stem}-bad-manifest", execute_outputs=["46"])
    assert not core.success and not stages
    assert not torch.cuda.is_initialized()
