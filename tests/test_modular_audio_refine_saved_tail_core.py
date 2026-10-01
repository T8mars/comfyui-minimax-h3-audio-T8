"""Execute the saved S26 tail wiring in Core with explicit CPU source doubles.

The original candidate API JSON is read, not changed. Its signed Setup, stage
Bind, external Core sampler, stage Audit and old Quality Gate stay connected.
Only the large upstream generation and pretrained denoising are replaced.
"""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import comfy.model_management
import comfy.nested_tensor
import comfy.samplers
import comfy_extras.nodes_custom_sampler as custom_sampler_nodes
from comfy_api.latest import io
import pytest
import torch

from h3_audio_t8_pkg import audio_refine_advanced as refine
from h3_audio_t8_pkg.modular_sampling.audio_refine_nodes import (
    MiniMaxH3AudioRefineDualClockStageAuditEXPT8,
    MiniMaxH3AudioRefineDualClockStageBindEXPT8,
    MiniMaxH3AudioRefineDualModelStageAuditEXPT8,
    MiniMaxH3AudioRefineDualModelStageBindEXPT8,
    MiniMaxH3AudioRefineCompatStageAuditEXPT8,
    MiniMaxH3AudioRefineCompatStageBindEXPT8,
    MiniMaxH3AudioRefineTailDeliveryAuditEXPT8,
)
from h3_audio_t8_pkg.nodes_audio_refine_advanced import (
    MiniMaxH3AudioRefineCompatibilitySetupT8Advanced,
    MiniMaxH3AudioRefineDualClockSetupT8Advanced,
    MiniMaxH3AudioRefineDualModelSetupT8Advanced,
    MiniMaxH3AudioRefineQualityGateT8Advanced,
)
from test_audio_refine_advanced import (
    _abstain_bundle,
    _allow_bundle,
    _audit,
    _audio,
    _full_video_sigmas,
    _latent,
    _phase2_bundle,
    _positive,
    _runtime,
    FakeModelPatcher,
)


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_DIR = (ROOT / "artifacts/development/modular-sampling-m4-audio-refine-abstain-20260924"
                 / "candidate-v3")
CANDIDATES = sorted(path.name for path in CANDIDATE_DIR.glob("*.api.json"))
TURBO4 = "2026-08-26_H3_Audio_Refine_Turbo4_Plus_Refine4_Advanced_EXP_StageBound_EXP.api.json"
SAMPLER_CALLS = []

FAMILIES = {
    "MiniMaxH3AudioRefineDualClockSetupT8Advanced": (
        "dual_clock", MiniMaxH3AudioRefineDualClockStageBindEXPT8,
        MiniMaxH3AudioRefineDualClockStageAuditEXPT8),
    "MiniMaxH3AudioRefineDualModelSetupT8Advanced": (
        "dual_model", MiniMaxH3AudioRefineDualModelStageBindEXPT8,
        MiniMaxH3AudioRefineDualModelStageAuditEXPT8),
    "MiniMaxH3AudioRefineCompatibilitySetupT8Advanced": (
        "compatibility", MiniMaxH3AudioRefineCompatStageBindEXPT8,
        MiniMaxH3AudioRefineCompatStageAuditEXPT8),
}


class TinySignedSource(io.ComfyNode):
    mode = "allow"
    variant = "dual_clock"
    strategy = "base_without_turbo"

    @classmethod
    def define_schema(cls):
        plan_type = {
            "dual_clock": refine.AUDIO_REFINE_PLAN_TYPE,
            "dual_model": refine.AUDIO_REFINE_PHASE2_PLAN_TYPE,
            "compatibility": refine.AUDIO_REFINE_COMPAT_PLAN_TYPE,
        }[cls.variant]
        return io.Schema(node_id=cls.__name__, inputs=[], outputs=[
            io.Model.Output(), io.Conditioning.Output(), io.Latent.Output(),
            io.Custom(plan_type).Output(), io.Audio.Output()])

    @classmethod
    def execute(cls):
        if cls.variant == "dual_clock":
            model, positive, latent, plan = (
                _allow_bundle() if cls.mode == "allow" else _abstain_bundle())
        elif cls.variant == "dual_model":
            first_model, model, positive, latent, _, plan = _phase2_bundle(
                strategy=cls.strategy)
            if cls.mode == "abstain":
                audit, decision, _ = _audit(
                    model=first_model, positive=positive, av_latent=latent,
                    conditioning_report="task=I2VA\naudio_mode=lock_source\nframes=22")
                assert decision == "ABSTAIN"
                _, route, decision, _ = refine.route_audio_refine_model(
                    audit=audit, first_pass_model=first_model, refine_model=model,
                    route_strategy=cls.strategy, declared_first_pass_nfe=4)
                assert decision == "ABSTAIN"
                plan, decision, _ = refine.plan_audio_refine_phase2(route, 4, .5, 2608260404)
                assert decision == "ABSTAIN"
        else:
            model = FakeModelPatcher()
            positive, latent = _positive(), _latent()
            audit, decision, _ = _audit(
                model=model, positive=positive, av_latent=latent,
                **({"conditioning_report": "task=I2VA\naudio_mode=lock_source\nframes=22"}
                   if cls.mode == "abstain" else {}))
            assert decision == cls.mode.upper()
            _, route, decision, _ = refine.route_audio_refine_compatibility(
                audit=audit, refine_model=model, positive=positive,
                generation_profile="turbo8", declared_first_pass_nfe=8)
            assert decision == cls.mode.upper()
            plan, decision, _ = refine.plan_audio_refine_compatibility(route, 4, .5, 29)
            assert decision == cls.mode.upper()
        return io.NodeOutput(model, positive, latent, plan, _audio())


def _cpu_setup(setup_fn, *, plan, model, positive, av_latent, dual_model=False):
    def cpu_sampling(connected_model, _latent, steps, *_args):
        def unexpected_denoiser(*_args, **_kwargs):
            raise AssertionError("test denoiser must be mediated by the CPU guider double")

        return (connected_model.clone(), comfy.samplers.KSAMPLER(unexpected_denoiser),
                _full_video_sigmas(steps))

    kwargs = {"plan": plan, "positive": positive, "av_latent": av_latent,
              "setup_sampling_fn": cpu_sampling, "runtime_snapshot_fn": _runtime}
    kwargs["refine_model" if dual_model else "model"] = model
    result = setup_fn(**kwargs)
    return io.NodeOutput(result.model, result.noise, result.guider, result.sampler,
                         result.sigmas, result.latent, result.report_json)


class TinyResourceSetup(MiniMaxH3AudioRefineDualClockSetupT8Advanced):
    @classmethod
    def execute(cls, plan, model, positive, av_latent):
        return _cpu_setup(refine.setup_audio_refine, plan=plan, model=model,
                          positive=positive, av_latent=av_latent)


class TinyDualModelSetup(MiniMaxH3AudioRefineDualModelSetupT8Advanced):
    @classmethod
    def execute(cls, plan, refine_model, positive, av_latent):
        return _cpu_setup(refine.setup_audio_refine_dual_model, plan=plan,
                          model=refine_model, positive=positive, av_latent=av_latent,
                          dual_model=True)


class TinyCompatSetup(MiniMaxH3AudioRefineCompatibilitySetupT8Advanced):
    @classmethod
    def execute(cls, plan, refine_model, positive, av_latent):
        return _cpu_setup(refine.setup_audio_refine_compatibility, plan=plan,
                          model=refine_model, positive=positive, av_latent=av_latent,
                          dual_model=True)


SETUPS = {
    "dual_clock": TinyResourceSetup,
    "dual_model": TinyDualModelSetup,
    "compatibility": TinyCompatSetup,
}


def test_saved_s26_candidate_inventory_is_complete():
    if not CANDIDATE_DIR.is_dir():
        pytest.skip("private saved S26 candidates are absent")
    assert len(CANDIDATES) == 10
    assert TURBO4 in CANDIDATES


class ObservedCoreSampler(custom_sampler_nodes.SamplerCustomAdvanced):
    @classmethod
    def execute(cls, noise, guider, sampler, sigmas, latent_image):
        SAMPLER_CALLS.append(len(sigmas))
        return super().execute(noise, guider, sampler, sigmas, latent_image)


class CaptureTail(io.ComfyNode):
    observations = []

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, is_output_node=True, inputs=[
            io.Latent.Input("original"), io.Latent.Input("candidate"),
            io.Latent.Input("selected"), io.String.Input("stage_report"),
            io.String.Input("tail_report"), io.String.Input("quality_report"),
            io.String.Input("decision")],
            outputs=[io.String.Output()])

    @classmethod
    def execute(cls, original, candidate, selected, stage_report, tail_report,
                quality_report, decision):
        cls.observations.append((original, candidate, selected,
                                 json.loads(stage_report), json.loads(tail_report),
                                 json.loads(quality_report), decision))
        return io.NodeOutput(decision)


def _one(saved, class_type):
    matches = [key for key, node in saved.items() if node["class_type"] == class_type]
    assert len(matches) == 1, (class_type, matches)
    return matches[0]


def _tail_graph(candidate_name):
    candidate = CANDIDATE_DIR / candidate_name
    if not candidate.is_file():
        pytest.skip("private saved S26 candidate is absent")
    saved = json.loads(candidate.read_text(encoding="utf-8"))
    setup_types = [name for name in FAMILIES if any(
        node["class_type"] == name for node in saved.values())]
    assert len(setup_types) == 1
    setup_type = setup_types[0]
    variant, bind_cls, audit_cls = FAMILIES[setup_type]
    setup_id = _one(saved, setup_type)
    bind_id = _one(saved, bind_cls.define_schema().node_id)
    audit_id = _one(saved, audit_cls.define_schema().node_id)
    tail_id = _one(saved, "MiniMaxH3AudioRefineTailDeliveryAuditEXPT8")
    gate_id = _one(saved, "MiniMaxH3AudioRefineQualityGateT8Advanced")
    samplers = [key for key, node in saved.items()
                if node["class_type"] == "SamplerCustomAdvanced"
                and node["inputs"].get("sigmas") == [bind_id, 4]]
    assert len(samplers) == 1
    sampler_id = samplers[0]
    selected = (setup_id, sampler_id, tail_id, gate_id, bind_id, audit_id)
    graph = {key: deepcopy(saved[key]) for key in selected}
    graph["100"] = {"class_type": "TinySignedSource", "inputs": {}}
    model_field = "model" if variant == "dual_clock" else "refine_model"
    for key, field, slot in ((setup_id, "plan", 3), (setup_id, model_field, 0),
                             (setup_id, "positive", 1), (setup_id, "av_latent", 2),
                             (bind_id, "plan", 3), (bind_id, "original_av_latent", 2),
                             (audit_id, "plan", 3), (audit_id, "original_av_latent", 2),
                             (gate_id, "original_av_latent", 2),
                             (gate_id, "original_audio", 4),
                             (gate_id, "candidate_audio", 4)):
        graph[key]["inputs"][field] = ["100", slot]
    graph[gate_id]["inputs"]["video_frame_count"] = 24
    graph["200"] = {"class_type": "CaptureTail", "inputs": {
        "original": ["100", 2], "candidate": [audit_id, 0],
        "selected": [gate_id, 0], "stage_report": [audit_id, 1],
        "tail_report": [tail_id, 1], "quality_report": [gate_id, 4],
        "decision": [gate_id, 3]}}
    assert graph[audit_id]["inputs"]["candidate_av_latent"] == [sampler_id, 0]
    assert graph[tail_id]["inputs"]["second_pass_output"] == [audit_id, 0]
    assert graph[gate_id]["inputs"]["candidate_av_latent"] == [tail_id, 0]
    return graph, variant, SETUPS[variant], bind_cls, audit_cls


@pytest.mark.parametrize("candidate_name,mode", [
    *((name, mode) for name in CANDIDATES for mode in ("allow", "abstain")),
])
def test_saved_s26_tail_executes_sampler_audit_and_default_quality_gate(
        candidate_name, mode, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT.parents[1]))
    import execution
    import nodes

    graph, variant, setup_cls, bind_cls, audit_cls = _tail_graph(candidate_name)
    TinySignedSource.mode = mode
    TinySignedSource.variant = variant
    TinySignedSource.strategy = ("same_turbo_stack" if "Phase2_Same" in candidate_name
                                 else "base_without_turbo")
    SAMPLER_CALLS.clear()
    CaptureTail.observations.clear()
    guider_requests = []

    if mode == "allow":
        def cpu_audio_only(self, _noise, latent, _sampler, sigmas, *, denoise_mask,
                           callback, disable_pbar, seed):
            del callback, disable_pbar, seed
            assert len(sigmas) == 5
            video_mask, audio_mask = denoise_mask.unbind()
            assert not video_mask.count_nonzero() and bool(torch.all(audio_mask == 1))
            video, audio = latent.unbind()
            guider_requests.append(len(sigmas) - 1)
            return comfy.nested_tensor.NestedTensor((video, audio + 0.25))

        monkeypatch.setattr(refine.AudioRefineBasicGuider, "sample", cpu_audio_only)

    monkeypatch.setattr(custom_sampler_nodes.latent_preview, "prepare_callback",
                        lambda *_args, **_kwargs: lambda *_values: None)
    monkeypatch.setattr(comfy.model_management, "intermediate_device",
                        lambda: torch.device("cpu"))
    for cls in (TinySignedSource, setup_cls, ObservedCoreSampler,
                bind_cls, audit_cls,
                MiniMaxH3AudioRefineTailDeliveryAuditEXPT8,
                MiniMaxH3AudioRefineQualityGateT8Advanced, CaptureTail):
        monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, cls.define_schema().node_id, cls)
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "SamplerCustomAdvanced",
                        ObservedCoreSampler)

    server = SimpleNamespace(client_id=None, last_node_id=None, sockets_metadata={},
                             send_sync=lambda *_args, **_kwargs: None)
    executor = execution.PromptExecutor(server, cache_args={"ram": 0., "ram_inactive": 0.},
                                        asset_manager=SimpleNamespace(enabled=False))
    executor.execute(graph, f"s26-{candidate_name}-{mode}-cpu", execute_outputs=["200"])
    assert executor.success, executor.status_messages
    assert SAMPLER_CALLS == ([5] if mode == "allow" else [0])
    assert len(CaptureTail.observations) == 1
    original, candidate, selected, stage_report, tail_report, quality_report, decision = (
        CaptureTail.observations[0])
    source_video, source_audio = original["samples"].unbind()
    candidate_video, candidate_audio = candidate["samples"].unbind()
    selected_video, selected_audio = selected["samples"].unbind()
    assert torch.equal(candidate_video, source_video)
    assert torch.equal(selected_video, source_video)
    assert torch.equal(selected_audio, source_audio)
    assert selected is original
    assert decision == "ABSTAIN_HUMAN_REVIEW_REQUIRED"
    assert quality_report["candidate_selected"] is False
    if mode == "allow":
        assert guider_requests == [4]
        assert not torch.equal(candidate_audio, source_audio)
        assert stage_report["status"] == "candidate_received"
        assert tail_report["node"] == "MiniMaxH3TwoPassAudioAuditT8Advanced"
        assert tail_report["status"] == "measured"
    else:
        assert guider_requests == []
        assert torch.equal(candidate_audio, source_audio)
        assert stage_report["status"] == "abstain_passthrough"
        assert tail_report["status"] == "abstain_no_sample"
        assert tail_report["sampled"] is False
