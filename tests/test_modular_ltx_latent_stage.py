"""S27 learned-latent stage contract; synthetic adapter, no model inference."""

from copy import deepcopy
import json
from pathlib import Path

import comfy.samplers
from comfy.nested_tensor import NestedTensor
import pytest
import torch

from h3_audio_t8_pkg.h3_ltx_adapter_runtime import MODEL_SHA, MODEL_REVISION, SOURCE_REVISION
from h3_audio_t8_pkg.h3_ltx_latent_contract import convert_video_latent
from h3_audio_t8_pkg.modular_sampling.ltx_latent_nodes import NODES
from h3_audio_t8_pkg.modular_sampling.ltx_latent_sample_nodes import NODES as SAMPLE_NODES
from h3_audio_t8_pkg.modular_sampling.ltx_latent_stage import (
    audit_ltx_latent_stage, bind_ltx_latent_stage, decode_original_h3_audio,
)
from h3_audio_t8_pkg.modular_sampling.ltx_latent_sample_stage import (
    audit_completed_ltx_latent_stage, sample_ltx_latent_stage,
)
from h3_audio_t8_pkg import sol_engine_h3_super_advanced as sol
from tools.build_modular_ltx_latent_workflow import build_prompt, STEM


class Adapter:
    def convert(self, video, *, pixel_frames, **kwargs):
        count = (pixel_frames - 1 + 7) // 8 + 1
        return torch.ones(video.shape[0], 128, count, video.shape[-2] // 2, video.shape[-1] // 2)


def _case(*, frames=73, policy="exact", joint=True):
    count = ((frames - 5) // 17) * 5 + 2
    video = torch.ones(1, 24, count, 4, 6)
    audio = torch.ones(1, 32, 2, 122)
    source = {"samples": NestedTensor((video, audio)) if joint else video,
              "noise_mask": "H3 only"}
    ltx, original, report = convert_video_latent(source, adapter=Adapter(),
        source_frames=frames, frame_policy=policy)
    report.update(model_sha256=MODEL_SHA, model_revision=MODEL_REVISION,
                  source_revision=SOURCE_REVISION)
    adapter_report = json.dumps(report, sort_keys=True)
    model = type("LTXModel", (), {"model_options": {}})()
    _, sigmas, _, setup = sol.setup_ltx_stage2_refiner(model, enabled=False)
    noise = object()
    guider = comfy.samplers.CFGGuider(model)
    sampler = comfy.samplers.KSAMPLER(lambda *_args, **_kwargs: None)
    args = (source, original, ltx, adapter_report, model, noise, guider,
            sampler, sigmas, setup)
    return args


def test_learned_latent_stage_keeps_external_sampler_and_original_av():
    args = _case()
    bound = bind_ltx_latent_stage(*args)
    assert all(left is right for left, right in zip(bound[:5],
        (args[5], args[6], args[7], args[8], args[2])))
    assert not json.loads(bound[6])["sampled"]
    candidate = {"samples": args[2]["samples"] + .1}
    audited = audit_ltx_latent_stage(bound[5], *args, candidate)
    assert audited[0] is candidate and audited[1] is args[0]
    assert audited[2] == pytest.approx(73 / 24)
    assert not json.loads(audited[3])["portable_cache_reuse_authorized"]


@pytest.mark.parametrize("mutation", ["source", "adapter_report", "model", "sigmas", "candidate", "forged"])
def test_source_or_controls_changed_fail_closed(mutation):
    args = list(_case())
    boundary = bind_ltx_latent_stage(*args)[5]
    candidate = {"samples": args[2]["samples"] + .1}
    if mutation == "source":
        changed = deepcopy(args[0])
        changed["samples"].unbind()[0][0, 0, 0, 0, 0] = .25
        args[0] = changed
        args[1] = changed
    elif mutation == "adapter_report":
        report = json.loads(args[3])
        report["model_sha256"] = "f" * 64
        args[3] = json.dumps(report)
    elif mutation == "model":
        args[4] = object()
    elif mutation == "sigmas":
        args[8] = args[8].clone()
        args[8][0] = .7
    elif mutation == "candidate":
        candidate = {"samples": torch.zeros(1, 128, 9, 2, 3)}
    else:
        boundary = deepcopy(boundary)
        boundary["output_fps"] = 30
    with pytest.raises(ValueError):
        audit_ltx_latent_stage(boundary, *args, candidate)


@pytest.mark.parametrize("frames,policy", [(124, "pad_to_ltx_grid"), (124, "crop_to_ltx_grid")])
def test_duration_changed_requires_explicit_audio_policy(frames, policy):
    with pytest.raises(ValueError, match="audio reconciliation"):
        bind_ltx_latent_stage(*_case(frames=frames, policy=policy))


def test_original_audio_decoder_uses_h3_audio_latent_only(monkeypatch):
    args = _case()
    seen = []

    def decode(vae, latent):
        seen.append((vae, latent))
        return {"waveform": torch.ones(1, 2, 100), "sample_rate": 32000}

    monkeypatch.setattr("comfy_extras.nodes_audio.vae_decode_audio", decode)
    vae = object()
    audio, report = decode_original_h3_audio(args[1], args[3], vae)
    assert seen[0][0] is vae
    assert seen[0][1]["samples"] is args[0]["samples"].unbind()[1]
    assert audio["sample_rate"] == 32000
    assert json.loads(report)["status"] == "original_h3_audio_decoded"
    video_only = _case(joint=False)
    with pytest.raises(ValueError, match="joint AV audio"):
        decode_original_h3_audio(video_only[1], video_only[3], vae)


def test_nodes_are_append_only_and_typed():
    assert [node.define_schema().node_id for node in NODES] == [
        "MiniMaxH3LTXLearnedStageBindEXPT8",
        "MiniMaxH3LTXLearnedStageAuditEXPT8",
        "MiniMaxH3LTXOriginalAudioDecodeEXPT8",
    ]
    assert [node.define_schema().node_id for node in SAMPLE_NODES] == [
        "MiniMaxH3LTXLearnedStageSampleEXPT8",
        "MiniMaxH3LTXLearnedStageCertifiedAuditEXPT8"]


def test_learned_latent_stage_proves_one_native_sampler_return(monkeypatch):
    args = list(_case())

    class Noise:
        seed = 42

        def generate_noise(self, latent):
            return torch.zeros_like(latent["samples"])

    args[5] = Noise()
    boundary = bind_ltx_latent_stage(*args)[5]
    calls = []

    def fake_sample(_actual_noise, *_unused, callback=None, **_kwargs):
        calls.append("sample")
        for step in range(3):
            callback(step)
        return args[2]["samples"] + .1

    monkeypatch.setattr(args[6], "sample", fake_sample)
    monkeypatch.setattr("comfy_extras.nodes_custom_sampler.comfy.sample.fix_empty_latent_channels",
                        lambda _model, latent, *_args: latent)
    monkeypatch.setattr("comfy_extras.nodes_custom_sampler.latent_preview.prepare_callback",
                        lambda *_args: lambda *_items: None)
    monkeypatch.setattr("comfy_extras.nodes_custom_sampler.comfy.model_management.intermediate_device",
                        lambda: torch.device("cpu"))
    output, _denoised, proof, sample_report = sample_ltx_latent_stage(
        args[5], args[6], args[7], args[8], args[2], boundary)
    assert calls == ["sample"]
    assert json.loads(sample_report)["observed_step_callbacks"] == [0, 1, 2]
    audited = audit_completed_ltx_latent_stage(boundary, *args, output, proof)
    assert json.loads(audited[3])["sampler_execution_proven"] is True
    assert json.loads(audited[3])["portable_cache_reuse_authorized"] is False
    changed = {"samples": output["samples"].clone()}
    with pytest.raises(ValueError, match="actual sampler proof"):
        audit_completed_ltx_latent_stage(boundary, *args, changed, proof)


def test_learned_latent_stage_rejects_missing_native_step(monkeypatch):
    args = list(_case())

    class Noise:
        seed = 42

        def generate_noise(self, latent):
            return torch.zeros_like(latent["samples"])

    args[5] = Noise()
    boundary = bind_ltx_latent_stage(*args)[5]

    def fake_sample(_actual_noise, *_unused, callback=None, **_kwargs):
        for step in (0, 1):
            callback(step)
        return args[2]["samples"]

    monkeypatch.setattr(args[6], "sample", fake_sample)
    monkeypatch.setattr("comfy_extras.nodes_custom_sampler.comfy.sample.fix_empty_latent_channels",
                        lambda _model, latent, *_args: latent)
    monkeypatch.setattr("comfy_extras.nodes_custom_sampler.latent_preview.prepare_callback",
                        lambda *_args: lambda *_items: None)
    monkeypatch.setattr("comfy_extras.nodes_custom_sampler.comfy.model_management.intermediate_device",
                        lambda: torch.device("cpu"))
    with pytest.raises(ValueError, match="completion was not observed"):
        sample_ltx_latent_stage(args[5], args[6], args[7], args[8], args[2], boundary)


def test_private_candidate_uses_joint_av_checkpoint_external_sampler_and_original_audio():
    root = Path(__file__).resolve().parents[1]
    candidate = root / "artifacts/development/modular-sampling-m4-ltx-latent-20260924/candidate-v1"
    api = json.loads((candidate / (STEM + ".api.json")).read_text(encoding="utf8"))
    frontend = json.loads((candidate / (STEM + ".json")).read_text(encoding="utf8"))
    assert api == build_prompt()
    assert len(api) == 21 and len(frontend["nodes"]) == 22 and len(frontend["links"]) == 50
    assert api["1"]["class_type"] == "MiniMaxH3NativeLatentCheckpointLoadT8Advanced"
    assert api["2"]["inputs"]["h3_latent"] == ["1", 0]
    assert api["12"]["inputs"]["original_h3_av"] == ["2", 1]
    assert api["13"]["inputs"]["latent_image"] == ["12", 4]
    assert api["14"]["inputs"]["candidate_latent"] == ["13", 0]
    assert api["16"]["inputs"]["original_h3_av"] == ["14", 1]
    assert api["19"]["inputs"]["audio"] == ["16", 0]
    assert api["20"]["inputs"]["audio"] == ["19", 1]
    assert api["21"]["inputs"]["video"] == ["20", 0]
    types = [node["class_type"] for node in api.values()]
    assert "LTXVLatentUpsampler" not in types
    assert "MiniMaxH3SolEngineDraftToLTXT8Advanced" not in types
    assert "VAEEncode" not in types


def test_private_candidate_v3_wires_live_sampler_proof_without_changing_v1():
    root = Path(__file__).resolve().parents[1]
    candidate = root / "artifacts/development/modular-sampling-m4-ltx-latent-20260924/candidate-v3"
    api = json.loads((candidate / (STEM + ".api.json")).read_text(encoding="utf8"))
    frontend = json.loads((candidate / (STEM + ".json")).read_text(encoding="utf8"))
    assert api == build_prompt(certified_sample=True)
    assert len(api) == 21 and len(frontend["nodes"]) == 22
    assert api["13"]["class_type"] == "MiniMaxH3LTXLearnedStageSampleEXPT8"
    assert api["14"]["class_type"] == "MiniMaxH3LTXLearnedStageCertifiedAuditEXPT8"
    assert api["13"]["inputs"]["stage_boundary"] == ["12", 5]
    assert api["14"]["inputs"]["candidate_latent"] == ["13", 0]
    assert api["14"]["inputs"]["sample_proof"] == ["13", 2]
    assert build_prompt()["13"]["class_type"] == "SamplerCustomAdvanced"
    assert "sample_proof" not in build_prompt()["14"]["inputs"]
