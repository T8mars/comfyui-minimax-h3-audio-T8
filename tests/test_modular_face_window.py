"""Window/Studio Face stages bind the exact parent, frame map and source audio."""
from copy import deepcopy
import json

import pytest
import torch
from comfy_extras.nodes_custom_sampler import BasicGuider, BasicScheduler, KSamplerSelect, RandomNoise, SamplerCustomAdvanced

from h3_audio_t8_pkg import face_refine_window_advanced as window, sampling
from h3_audio_t8_pkg.face_refine_parity_advanced import apply_face_refine_per_frame_denoise
from h3_audio_t8_pkg.modular_sampling.face_stage import bind_window_face_stage, audit_window_face_stage
from h3_audio_t8_pkg.modular_sampling.results import sample_stage
from h3_audio_t8_pkg.modular_sampling.face_nodes import NODES
from test_face_refine_parity_advanced import _plan, _locked_av
from test_fast_h3_v2_core_sampler import model
from test_progressive_sampling_runtime import conditioning
from tools.build_modular_face_window_workflows import TARGET, VARIANTS, source_paths, split_api, split_frontend


def _job(*, padded=False):
    count = 21 if padded else 22
    parent = torch.zeros((count, 64, 96, 3))
    parent[:, 0, 0, 0] = torch.arange(count) / count
    audio = {"waveform": torch.arange(40_000, dtype=torch.float32).view(1, 1, -1) / 40_000,
             "sample_rate": 32_000}
    plan, *_ = window.build_face_refine_window_plan(
        parent, 24., "0-3", "frames_inclusive", 0, 0, 22, 22, 1.,
        "reject", "edge_hold_exp" if padded else "reject", True)
    source, window_audio, mapping, *_ = window.extract_face_refine_window(
        parent, plan, 0, "edge_hold_exp" if padded else "reject", audio)
    face_plan = _plan(source)[0]
    return parent, audio, plan, source, window_audio, mapping, face_plan


def _bind(job, latent):
    parent, audio, plan, source, window_audio, mapping, face_plan = job
    prepared, _, _ = sampling.setup_dual_clock_sampling(model(), latent, 8, 12., 3., "er_sde", "simple")
    sampler = KSamplerSelect.execute("er_sde").result[0]
    sigmas = BasicScheduler.execute(prepared, "simple", 8, .45).result[0]
    result = bind_window_face_stage(face_plan, source, parent, plan, mapping, audio,
                                    window_audio, prepared, sampler, sigmas, latent)
    return prepared, sampler, sigmas, result


def test_window_stage_matches_external_core_sampler_and_binds_original_audio():
    job = _job()
    parent, audio, plan, source, window_audio, mapping, face_plan = job
    latent = apply_face_refine_per_frame_denoise(
        _locked_av(frame_count=22), face_plan, .8, .35, "relative_to_clip",
        30., 120., 1., 9, "replace_video_parity", True)[0]
    prepared, sampler, sigmas, (bound, selected, table, context, report) = _bind(job, latent)
    assert json.loads(report)["variant"] == "window"
    assert json.loads(context.profile)["face_refine"]["window_index"] == 0
    noise = RandomNoise.execute(320).result[0]
    torch.manual_seed(320)
    reference = SamplerCustomAdvanced.execute(
        noise, BasicGuider.execute(prepared, conditioning()).result[0], sampler, sigmas, latent).result
    torch.manual_seed(320)
    result = sample_stage(noise, BasicGuider.execute(bound, conditioning()).result[0],
                          selected, table, latent, context)[2]
    for actual, expected in zip((result.output, result.denoised_output), reference):
        assert all(torch.equal(x, y) for x, y in zip(actual["samples"].unbind(), expected["samples"].unbind()))
    candidate, audit = audit_window_face_stage(result, face_plan, source, parent, plan, mapping,
                                               audio, window_audio, latent)
    assert candidate is result.output
    assert json.loads(audit)["delivery_audio"] == "original_source_audio_only"
    changed_audio = deepcopy(audio)
    changed_audio["waveform"] = audio["waveform"].clone()
    changed_audio["waveform"][0, 0, 0] = 1
    with pytest.raises(ValueError, match="audio"):
        audit_window_face_stage(result, face_plan, source, parent, plan, mapping,
                                changed_audio, window_audio, latent)
    assert [node.define_schema().node_id for node in NODES][-2:] == [
        "MiniMaxH3FaceWindowStageBindEXPT8", "MiniMaxH3FaceWindowStageAuditEXPT8"]


def test_window_stage_checks_edge_padding_and_rehashed_frame_map():
    job = _job(padded=True)
    parent, audio, plan, source, window_audio, mapping, face_plan = job
    assert plan["windows"][0]["post_pad_frames"] == 1
    assert torch.equal(source[-1], parent[-1])
    latent = _locked_av(frame_count=22)
    prepared, sampler, sigmas, _ = _bind(job, latent)
    forged = deepcopy(mapping)
    forged["frame_map"][0]["source_frame"] = 1
    forged.pop("mapping_sha256")
    forged = window._signed(forged, "mapping_sha256")
    with pytest.raises(ValueError, match="frame map"):
        bind_window_face_stage(face_plan, source, parent, plan, forged, audio,
                               window_audio, prepared, sampler, sigmas, latent)
    wrong_window_audio = deepcopy(window_audio)
    wrong_window_audio["waveform"] = window_audio["waveform"].clone()
    wrong_window_audio["waveform"][0, 0, 1] = -1
    with pytest.raises(ValueError, match="audio"):
        bind_window_face_stage(face_plan, source, parent, plan, mapping, audio,
                               wrong_window_audio, prepared, sampler, sigmas, latent)


@pytest.mark.parametrize("variant,node_count,link_count,api_count", [
    ("Manual", 31, 75, 30), ("Studio_Serial", 34, 84, 33)])
def test_private_window_graph_preserves_manual_or_studio_review(
        variant, node_count, link_count, api_count):
    source_path, _fixture = source_paths(variant)
    original = json.loads(source_path.read_text(encoding="utf-8"))
    graph = split_frontend(deepcopy(original))
    api = split_api(graph)
    stem = f"Face_Window_{variant}_Separate_Stage_EXP"
    assert graph == json.loads((TARGET / f"{stem}.json").read_text(encoding="utf-8"))
    assert api == json.loads((TARGET / f"{stem}.api.json").read_text(encoding="utf-8"))
    assert len(graph["nodes"]) == node_count and len(graph["links"]) == link_count
    assert len(api) == api_count
    assert next(node for node in original["nodes"] if node["id"] == 18)["type"] == "SamplerCustomAdvanced"
    assert api["10"]["inputs"]["av_latent"] == ["13", 0]
    assert api["33"]["inputs"]["source_frames"] == ["27", 0]
    assert api["33"]["inputs"]["window_mapping"] == ["27", 2]
    assert api["33"]["inputs"]["source_audio"] == ["2", 1]
    assert api["33"]["inputs"]["window_audio"] == ["27", 1]
    assert api["18"]["inputs"]["stage_context"] == ["33", 3]
    assert api["19"]["inputs"]["av_latent"] == ["34", 0]
    assert api["21"]["inputs"]["audio"] == ["2", 1]
    assert api["32"]["inputs"]["enabled"] is False
    if variant == "Manual":
        assert api["28"]["inputs"]["window_mapping"] == ["27", 2]
        assert api["28"]["inputs"]["decision"] == "preview_only"
    else:
        assert api["27"]["inputs"]["window_index"] == ["28", 0]
        assert api["29"]["inputs"]["window_mapping"] == ["27", 2]
        assert api["29"]["inputs"]["decision"] == "preview_only"
        assert api["30"]["inputs"]["commit_barrier"] == ["29", 9]
    mapped = {node["id"]: node for node in graph["nodes"]}
    for link_id, source_id, output_slot, target_id, input_slot, dtype in graph["links"]:
        assert mapped[target_id]["inputs"][input_slot]["link"] == link_id
        assert link_id in mapped[source_id]["outputs"][output_slot]["links"]
        assert mapped[source_id]["outputs"][output_slot]["type"] == dtype
        assert mapped[target_id]["inputs"][input_slot]["type"] == dtype
    assert set(VARIANTS) == {"Manual", "Studio_Serial"}
