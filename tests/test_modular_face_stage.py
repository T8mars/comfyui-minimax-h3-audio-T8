"""Standard Face Refine stage binding; other family variants stay separate."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch
from comfy_extras.nodes_custom_sampler import BasicGuider, RandomNoise, SamplerCustomAdvanced

from h3_audio_t8_pkg.face_refine_advanced import setup_face_refine_sampling
from h3_audio_t8_pkg.modular_sampling import face_stage
from h3_audio_t8_pkg.modular_sampling.results import sample_stage
from h3_audio_t8_pkg.modular_sampling.storage import save_stage, load_stage
from h3_audio_t8_pkg.modular_sampling.eav import EAVConfig, apply_stage_eav, audit_stage_eav
from h3_audio_t8_pkg.modular_sampling.face_nodes import NODES
from test_face_refine_advanced import _plan, _locked_av
from test_fast_h3_v2_core_sampler import model
from test_progressive_sampling_runtime import conditioning
from tools.build_modular_face_workflow import SOURCE, TARGET, split_frontend, split_api


def _inputs():
    frames = torch.zeros((5, 64, 96, 3))
    plan = _plan(frames)[0]
    latent = _locked_av()
    prepared, sampler, sigmas, _ = setup_face_refine_sampling(
        model(), latent, 2, .5, 12., 3., "dual_clock_euler", "native_flow")
    return frames, plan, latent, prepared, sampler, sigmas


def test_face_stage_matches_existing_single_sampler_and_audits_locked_source():
    frames, plan, latent, prepared, sampler, sigmas = _inputs()
    bound, selected, table, context, report = face_stage.bind_standard_face_stage(
        plan, frames, prepared, sampler, sigmas, latent)
    assert selected is sampler and table is sigmas
    assert json.loads(report)["sampled"] is False
    assert json.loads(context.profile)["face_refine"]["plan_sha256"] == plan["plan_sha256"]
    noise = RandomNoise.execute(145).result[0]
    expected = SamplerCustomAdvanced.execute(
        noise, BasicGuider.execute(prepared, conditioning()).result[0], sampler, sigmas, latent).result
    result = sample_stage(noise, BasicGuider.execute(bound, conditioning()).result[0],
                          selected, table, latent, context)[2]
    for actual, reference in zip((result.output, result.denoised_output), expected):
        assert all(torch.equal(x, y) for x, y in zip(actual["samples"].unbind(), reference["samples"].unbind()))
    candidate, audit = face_stage.audit_standard_face_stage(result, plan, frames, latent)
    assert candidate is result.output
    assert json.loads(audit)["delivery_audio"] == "original_source_audio_only"
    assert result.verify()["execution"]["denoiser_evaluations"] == 2
    assert [node.define_schema().node_id for node in NODES][:2] == [
        "MiniMaxH3FaceStageBindEXPT8", "MiniMaxH3FaceStageAuditEXPT8"]


@pytest.mark.parametrize("mutation", ["plan", "source", "time", "audio_mask", "policy"])
def test_face_bind_rejects_wrong_plan_source_and_audio_contract(mutation):
    frames, plan, latent, prepared, sampler, sigmas = _inputs()
    if mutation == "plan":
        plan = deepcopy(plan)
        plan["canvas"]["width"] = 352
    elif mutation == "source":
        frames = frames.clone()
        frames[0] = 1
    elif mutation == "time":
        video, audio = latent["samples"].unbind()
        import comfy.nested_tensor
        latent["samples"] = comfy.nested_tensor.NestedTensor((video[:, :, :1], audio))
    elif mutation == "audio_mask":
        latent["noise_mask"].unbind()[1][0, 0, 0, 0] = 1
    else:
        with pytest.raises(ValueError, match="audio policy"):
            face_stage.bind_standard_face_stage(plan, frames, prepared, sampler, sigmas,
                                                latent, "replace_audio")
        return
    with pytest.raises(ValueError):
        face_stage.bind_standard_face_stage(plan, frames, prepared, sampler, sigmas, latent)


def test_face_audit_rejects_changed_source_after_real_sampling():
    frames, plan, latent, prepared, sampler, sigmas = _inputs()
    bound, sampler, sigmas, context, _ = face_stage.bind_standard_face_stage(
        plan, frames, prepared, sampler, sigmas, latent)
    result = sample_stage(RandomNoise.execute(145).result[0],
                          BasicGuider.execute(bound, conditioning()).result[0],
                          sampler, sigmas, latent, context)[2]
    changed = frames.clone()
    changed[0] = 1
    with pytest.raises(ValueError, match="source frames"):
        face_stage.audit_standard_face_stage(result, plan, changed, latent)


def test_face_bound_stage_accepts_external_report_only_eav_without_changing_output():
    frames, plan, latent, prepared, sampler, sigmas = _inputs()
    bound, sampler, sigmas, context, _ = face_stage.bind_standard_face_stage(
        plan, frames, prepared, sampler, sigmas, latent)
    effected, runtime, _ = apply_stage_eav(bound, sigmas, latent, context,
        EAVConfig("report_only", tau=-8., start_video_progress=0., end_video_progress=1., g_hard_limit=3.))
    result = sample_stage(RandomNoise.execute(145).result[0],
                          BasicGuider.execute(effected, conditioning()).result[0],
                          sampler, sigmas, latent, context)[2]
    face_stage.audit_standard_face_stage(result, plan, frames, latent)
    _, audit = audit_stage_eav(result.output, runtime)
    assert json.loads(audit)["status"] == "observed_report_only"


def test_face_result_can_be_explicitly_frozen_then_source_audited(tmp_path, monkeypatch):
    frames, plan, latent, prepared, sampler, sigmas = _inputs()
    bound, sampler, sigmas, context, _ = face_stage.bind_standard_face_stage(
        plan, frames, prepared, sampler, sigmas, latent)
    result = sample_stage(RandomNoise.execute(145).result[0],
                          BasicGuider.execute(bound, conditioning()).result[0],
                          sampler, sigmas, latent, context)[2]
    assert result.verify()["portable_identity"] is True
    path, digest, _ = save_stage(result, tmp_path, "face-standard")
    frozen, _, frozen_context, restored, _ = load_stage(tmp_path, path, digest, "native_high")
    assert frozen_context == context
    assert all(torch.equal(x, y) for x, y in zip(frozen["samples"].unbind(), result.output["samples"].unbind()))
    face_stage.audit_standard_face_stage(restored, plan, frames, latent)
    monkeypatch.setattr(face_stage, "_face_implementation_sha256", lambda: "0" * 64)
    with pytest.raises(ValueError, match="changed after stage binding"):
        face_stage.audit_standard_face_stage(restored, plan, frames, latent)


def test_private_face_candidate_keeps_legacy_graph_and_source_audio_separate():
    original = json.loads(SOURCE.read_text(encoding="utf-8"))
    graph = split_frontend(deepcopy(original))
    api = split_api(graph)
    saved = json.loads((TARGET / "Face_Standard_Separate_Stage_EXP.json").read_text(encoding="utf-8"))
    assert graph == saved
    assert api == json.loads((TARGET / "Face_Standard_Separate_Stage_EXP.api.json").read_text(encoding="utf-8"))
    assert len(graph["nodes"]) == 19 and len(graph["links"]) == 44
    assert next(node for node in graph["nodes"] if node["id"] == 13)["type"] == "MiniMaxH3StageSamplerEXPT8"
    assert api["12"]["inputs"]["model"] == ["18", 0]
    assert api["13"]["inputs"]["stage_context"] == ["18", 3]
    assert api["14"]["inputs"]["av_latent"] == ["19", 0]
    assert api["16"]["inputs"]["audio"] == ["7", 2]
    assert original["nodes"][12]["type"] == "SamplerCustomAdvanced"
    assert Path(SOURCE).is_file()
