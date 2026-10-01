"""S25 separate Motion Recovery pass 2: native output and source-bound audit."""
from copy import deepcopy
import json

import pytest
import torch
from comfy_extras.nodes_custom_sampler import BasicGuider, RandomNoise, SamplerCustomAdvanced

from h3_audio_t8_pkg import motion_recovery_advanced as motion, sampling
from h3_audio_t8_pkg.modular_sampling.motion_stage import bind_motion_stage, audit_motion_stage
from h3_audio_t8_pkg.modular_sampling.motion_nodes import NODES
from h3_audio_t8_pkg.modular_sampling.results import sample_stage
from helpers import FakeAudioVAE, FakeVideoVAE, make_audio
from test_fast_h3_v2_core_sampler import model
from test_motion_recovery_advanced import _frames, _manual_plan
from test_progressive_sampling_runtime import conditioning
from tools.build_modular_motion_recovery_workflows import SOURCES, source_path, split_api, split_frontend


def _source(*, windowed=False, seed_mode="none_invent_exp",
            sampler_name="er_sde", scheduler="simple"):
    frames = _frames()
    audio = make_audio(seconds=124 / motion.FPS, sample_rate=32000, value=.125, channels=2)
    plan = _manual_plan(ranges="20-75:3")
    parent = {"parent_plan": plan, "parent_frames": frames, "parent_audio": audio} if windowed else {}
    if windowed:
        frames, plan, _, _, audio, *_ = motion.plan_motion_segment(
            frames, audio, plan, 90, 0, 4, "hot_ranges_only")
    av, smeared, seed_audio, _, report = motion.prepare_motion_retiming(
        frames, plan, FakeVideoVAE(),
        FakeAudioVAE() if seed_mode != "none_invent_exp" else None, audio, seed_mode)
    configured, sampler, sigmas = sampling.setup_dual_clock_sampling(
        model(), av, 4, 12., 3., sampler_name, scheduler)
    composed, selected, _, _, _ = motion.compose_motion_recovery(
        av, sigmas, plan, "apply_exp", .5, 2)
    return plan, frames, audio, smeared, seed_audio, report, configured, sampler, selected, composed, parent


@pytest.mark.parametrize("windowed", [False, True])
def test_motion_stage_matches_existing_external_sampler_and_audits_source(windowed):
    source = _source(windowed=windowed)
    plan, frames, audio, smeared, seed_audio, report, configured, sampler, sigmas, av, parent = source
    prepared, selected, table, context, bind_report = bind_motion_stage(
        plan, frames, audio, smeared, seed_audio, report, configured, sampler, sigmas, av, **parent)
    assert json.loads(bind_report)["variant"] == ("windowed" if windowed else "full_clip")
    noise = RandomNoise.execute(927).result[0]
    torch.manual_seed(927)
    reference = SamplerCustomAdvanced.execute(
        noise, BasicGuider.execute(configured, conditioning()).result[0], sampler, sigmas, av).result
    torch.manual_seed(927)
    result = sample_stage(noise, BasicGuider.execute(prepared, conditioning()).result[0],
                          selected, table, av, context)[2]
    for actual, expected in zip((result.output, result.denoised_output), reference):
        assert all(torch.equal(left, right) for left, right in
                   zip(actual["samples"].unbind(), expected["samples"].unbind()))
    candidate, audit = audit_motion_stage(result, plan, frames, audio, smeared, seed_audio, report, av,
                                          **parent)
    assert candidate is result.output
    assert json.loads(audit)["delivery_audio"] == "pass1_original_default_only"
    assert [node.define_schema().node_id for node in NODES] == [
        "MiniMaxH3MotionStageBindEXPT8", "MiniMaxH3MotionStageAuditEXPT8",
        "MiniMaxH3MotionRelayLengthEXPT8", "MiniMaxH3MotionRelayBindEXPT8"]

    changed = frames.clone()
    changed[0, 0, 0, 0] = 1
    with pytest.raises(ValueError, match="smeared frames"):
        audit_motion_stage(result, plan, changed, audio, smeared, seed_audio, report, av, **parent)
    changed_audio = deepcopy(audio)
    changed_audio["waveform"] = audio["waveform"].clone()
    changed_audio["waveform"][0, 0, 0] = .8
    with pytest.raises(ValueError, match="audio|source, plan|input AV"):
        audit_motion_stage(result, plan, frames, changed_audio, smeared, seed_audio, report, av, **parent)
    if windowed:
        wrong_parent = dict(parent)
        wrong_parent["parent_frames"] = parent["parent_frames"].clone()
        wrong_parent["parent_frames"][0, 0, 0, 0] = 1
        with pytest.raises(ValueError, match="parent slice|source, plan"):
            audit_motion_stage(result, plan, frames, audio, smeared, seed_audio, report, av,
                               **wrong_parent)


def test_motion_stage_rejects_wrong_audio_seed_and_signed_plan():
    plan, frames, audio, smeared, seed_audio, report, configured, sampler, sigmas, av, _ = _source()
    changed_seed = deepcopy(seed_audio)
    changed_seed["waveform"] = seed_audio["waveform"].clone()
    changed_seed["waveform"][0, 0, 0] = 1
    with pytest.raises(ValueError, match="stereo silence"):
        bind_motion_stage(plan, frames, audio, smeared, changed_seed, report, configured, sampler, sigmas, av)
    wrong_plan = deepcopy(plan)
    wrong_plan["holds"][0] += 1
    with pytest.raises(ValueError, match="SHA-256"):
        bind_motion_stage(wrong_plan, frames, audio, smeared, seed_audio, report,
                          configured, sampler, sigmas, av)


@pytest.mark.parametrize("windowed", [False, True])
def test_motion_stage_accepts_existing_follow_original_audio_seed_and_rejects_mask_change(windowed):
    plan, frames, audio, smeared, seed_audio, report, configured, sampler, sigmas, av, parent = _source(
        windowed=windowed, seed_mode="follow_original_0p5")
    _, _, _, context, _ = bind_motion_stage(
        plan, frames, audio, smeared, seed_audio, report, configured, sampler, sigmas, av, **parent)
    assert json.loads(context.profile)["motion_recovery"]["audio_seed_mode"] == "follow_original_0p5"
    changed = dict(av)
    video_mask, audio_mask = av["noise_mask"].unbind()
    changed["noise_mask"] = type(av["noise_mask"])((video_mask, audio_mask + .1))
    with pytest.raises(ValueError, match="noise masks"):
        bind_motion_stage(plan, frames, audio, smeared, seed_audio, report,
                          configured, sampler, sigmas, changed, **parent)


@pytest.mark.parametrize("variant", list(SOURCES))
def test_private_motion_graph_keeps_old_branch_audio_and_separates_pass2(variant):
    source = json.loads(source_path(variant).read_text(encoding="utf-8"))
    graph = split_frontend(source, variant)
    api = split_api(graph)
    nodes = {node["id"]: node for node in graph["nodes"]}
    links = {item[0]: item for item in graph["links"]}
    old_sampler = 19 if variant == "Windowed" else 18
    stage = nodes[old_sampler]
    assert stage["type"] == "MiniMaxH3StageSamplerEXPT8"
    assert nodes[9]["type"] == "SamplerCustomAdvanced"
    assert len([node for node in nodes.values() if node["type"] == "MiniMaxH3MotionStageBindEXPT8"]) == 1
    assert len([node for node in nodes.values() if node["type"] == "MiniMaxH3MotionStageAuditEXPT8"]) == 1
    audit = next(node for node in nodes.values() if node["type"] == "MiniMaxH3MotionStageAuditEXPT8")
    decoder = nodes[20 if variant == "Windowed" else 19]
    assert links[decoder["inputs"][0]["link"]][1:3] == [audit["id"], 0]
    recovered = nodes[21 if variant == "Windowed" else 20]
    original_audio = recovered["inputs"][2]["link"]
    assert links[original_audio][1:3] == ([13, 4] if variant == "Windowed" else [10, 1])
    assert api[str(audit["id"])]["inputs"]["stage_result"] == [str(old_sampler), 2]
    assert len(graph["links"]) == len({link[0] for link in graph["links"]})
    for link_id, source_id, output_slot, target_id, input_slot, dtype in graph["links"]:
        assert nodes[source_id]["outputs"][output_slot]["type"] == dtype
        assert nodes[target_id]["inputs"][input_slot]["type"] == dtype
        assert nodes[target_id]["inputs"][input_slot]["link"] == link_id
