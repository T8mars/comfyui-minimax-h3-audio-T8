"""Source-bound Parity/Per-Frame Face sampling without certifying er_sde reuse."""
from copy import deepcopy
import json

import pytest
import torch
from comfy_extras.nodes_custom_sampler import BasicGuider, BasicScheduler, KSamplerSelect, RandomNoise, SamplerCustomAdvanced

from h3_audio_t8_pkg import sampling
from h3_audio_t8_pkg.face_refine_parity_advanced import apply_face_refine_per_frame_denoise
from h3_audio_t8_pkg.face_refine_sampler_mask_advanced import apply_face_refine_sampler_mask_patch
from h3_audio_t8_pkg.modular_sampling.face_stage import bind_parity_face_stage, audit_parity_face_stage
from h3_audio_t8_pkg.modular_sampling.results import sample_stage
from h3_audio_t8_pkg.modular_sampling.storage import load_stage, save_stage
from h3_audio_t8_pkg.modular_sampling.face_nodes import NODES
from test_face_refine_parity_advanced import _plan, _locked_av, _manual512_relative_fixture
from test_fast_h3_v2_core_sampler import model
from test_progressive_sampling_runtime import conditioning
from tools.build_modular_face_parity_workflow import SOURCE, TARGET, split_api, split_frontend


def _inputs(*, per_frame=False):
    frames = torch.zeros((5, 64, 96, 3))
    plan = _plan(frames)[0]
    latent = _locked_av()
    if per_frame:
        latent = apply_face_refine_per_frame_denoise(latent, plan, .8, .35, "absolute_px",
            30., 120., 1., 9, "replace_video_parity", True)[0]
    prepared, _, _ = sampling.setup_dual_clock_sampling(
        model(), latent, 8, 12., 3., "er_sde", "simple")
    sampler = KSamplerSelect.execute("er_sde").result[0]
    sigmas = BasicScheduler.execute(prepared, "simple", 8, .45).result[0]
    return frames, plan, latent, prepared, sampler, sigmas


@pytest.mark.parametrize("per_frame", [False, True])
def test_parity_stage_keeps_connected_er_sde_and_per_frame_video_mask(per_frame, tmp_path):
    frames, plan, latent, prepared, sampler, sigmas = _inputs(per_frame=per_frame)
    bound, selected, table, context, report = bind_parity_face_stage(
        plan, frames, prepared, sampler, sigmas, latent)
    assert selected is sampler and table is sigmas
    assert json.loads(report)["portable_completion_sampler_adapted"] is True
    assert json.loads(context.profile)["face_refine"]["h3_aligned_frame_count"] == 5
    noise = RandomNoise.execute(317).result[0]
    torch.manual_seed(317)
    expected = SamplerCustomAdvanced.execute(
        noise, BasicGuider.execute(prepared, conditioning()).result[0], sampler, sigmas, latent).result
    torch.manual_seed(317)
    result = sample_stage(noise, BasicGuider.execute(bound, conditioning()).result[0],
                          sampler, sigmas, latent, context)[2]
    for actual, reference in zip((result.output, result.denoised_output), expected):
        assert all(torch.equal(x, y) for x, y in zip(actual["samples"].unbind(), reference["samples"].unbind()))
    candidate, audit = audit_parity_face_stage(result, plan, frames, latent)
    assert candidate is result.output
    assert json.loads(audit)["variant"] == "parity"
    assert result.verify()["portable_identity"] is True
    path, digest, _ = save_stage(result, tmp_path, "parity-er-sde")
    _, _, loaded_context, loaded, _ = load_stage(tmp_path, path, digest, "native_high")
    assert loaded_context == context
    assert loaded.verify()["receipt_sha256"] == result.verify()["receipt_sha256"]
    audit_parity_face_stage(loaded, plan, frames, latent)
    with pytest.raises(ValueError, match="SHA"):
        load_stage(tmp_path, path, "0" * 64, "native_high")


def test_parity_audit_does_not_accept_standard_plan_or_mutated_source():
    frames, plan, latent, prepared, sampler, sigmas = _inputs()
    bound, sampler, sigmas, context, _ = bind_parity_face_stage(
        plan, frames, prepared, sampler, sigmas, latent)
    result = sample_stage(RandomNoise.execute(317).result[0],
        BasicGuider.execute(bound, conditioning()).result[0], sampler, sigmas, latent, context)[2]
    changed = frames.clone()
    changed[0] = 1
    with pytest.raises(ValueError, match="source frames"):
        audit_parity_face_stage(result, plan, changed, latent)
    assert [node.define_schema().node_id for node in NODES][2:4] == [
        "MiniMaxH3FaceParityStageBindEXPT8", "MiniMaxH3FaceParityStageAuditEXPT8"]


def test_parity_one_frame_alignment_tail_is_bound_without_implicit_latent_trim():
    frames, plan, *_ = _manual512_relative_fixture(frame_count=4)
    latent = _locked_av(frame_count=5, canvas=512)
    prepared, _, _ = sampling.setup_dual_clock_sampling(model(), latent, 8, 12., 3., "er_sde", "simple")
    sampler = KSamplerSelect.execute("er_sde").result[0]
    sigmas = BasicScheduler.execute(prepared, "simple", 8, .45).result[0]
    _, _, _, context, _ = bind_parity_face_stage(plan, frames, prepared, sampler, sigmas, latent)
    assert json.loads(context.profile)["face_refine"]["h3_aligned_frame_count"] == 5
    assert context.video_shape[2] == 2


def test_enabled_sampler_mask_patch_remains_external_to_parity_stage():
    frames = torch.zeros((5, 64, 96, 3))
    plan = _plan(frames)[0]
    latent, denoise_report = apply_face_refine_per_frame_denoise(
        _locked_av(), plan, .8, .35, "absolute_px", 30., 120., 1., 9,
        "replace_video_parity", True)
    prepared, _, _ = sampling.setup_dual_clock_sampling(
        model(), latent, 8, 12., 3., "er_sde", "simple")
    patched, returned, patch_report = apply_face_refine_sampler_mask_patch(
        prepared, latent, denoise_report, enabled=True)
    assert returned is latent
    assert json.loads(patch_report)["enabled"] is True
    sampler = KSamplerSelect.execute("er_sde").result[0]
    sigmas = BasicScheduler.execute(patched, "simple", 8, .45).result[0]
    bound, selected, table, context, _ = bind_parity_face_stage(
        plan, frames, patched, sampler, sigmas, latent)
    noise = RandomNoise.execute(318).result[0]
    torch.manual_seed(318)
    reference = SamplerCustomAdvanced.execute(
        noise, BasicGuider.execute(patched, conditioning()).result[0], sampler,
        sigmas, latent).result
    torch.manual_seed(318)
    result = sample_stage(noise, BasicGuider.execute(bound, conditioning()).result[0],
                          selected, table, latent, context)[2]
    for actual, expected in zip((result.output, result.denoised_output), reference):
        assert all(torch.equal(x, y) for x, y in zip(
            actual["samples"].unbind(), expected["samples"].unbind()))
    assert audit_parity_face_stage(result, plan, frames, latent)[0] is result.output


def test_private_parity_candidate_keeps_old_graph_and_external_masks():
    original = json.loads(SOURCE.read_text(encoding="utf-8"))
    graph = split_frontend(deepcopy(original))
    api = split_api(graph)
    saved = json.loads((TARGET / "Face_Parity_Separate_Stage_EXP.json").read_text(encoding="utf-8"))
    assert graph == saved
    assert api == json.loads((TARGET / "Face_Parity_Separate_Stage_EXP.api.json").read_text(encoding="utf-8"))
    assert len(graph["nodes"]) == 28 and len(graph["links"]) == 57
    assert api["13"]["inputs"]["av_latent"] == ["12", 1]
    assert api["32"]["inputs"]["av_latent"] == ["13", 0]
    assert api["33"]["inputs"]["model"] == ["32", 0]
    assert api["33"]["inputs"]["av_latent"] == ["32", 1]
    assert api["18"]["inputs"]["stage_context"] == ["33", 3]
    assert api["19"]["inputs"]["av_latent"] == ["34", 0]
    assert api["21"]["inputs"]["audio"] == ["2", 1]
    assert api["32"]["inputs"]["enabled"] is False
    assert next(node for node in original["nodes"] if node["id"] == 18)["type"] == "SamplerCustomAdvanced"
    assert next(node for node in original["nodes"] if node["id"] == 10)["type"] == "MiniMaxH3SigmaShift"
    nodes = {node["id"]: node for node in graph["nodes"]}
    for link_id, source, output_slot, target, input_slot, link_type in graph["links"]:
        assert nodes[target]["inputs"][input_slot]["link"] == link_id
        assert link_id in nodes[source]["outputs"][output_slot]["links"]
        assert nodes[source]["outputs"][output_slot]["type"] == link_type
        assert nodes[target]["inputs"][input_slot]["type"] == link_type
