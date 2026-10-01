"""S25 external Relay/EAV candidate contracts; no trained-media claim."""
from copy import deepcopy
import json

import pytest
from comfy_extras.nodes_custom_sampler import BasicGuider, RandomNoise

from h3_audio_t8_pkg.modular_sampling.eav import EAVConfig, apply_stage_eav, audit_stage_eav
from h3_audio_t8_pkg.modular_sampling.motion_effects import (
    bind_motion_relay, retimed_relay_length,
)
from h3_audio_t8_pkg.modular_sampling.motion_stage import bind_motion_stage, audit_motion_stage
from h3_audio_t8_pkg.modular_sampling.results import sample_stage
from h3_audio_t8_pkg.prompt_relay_advanced import (
    PROMPT_RELAY_BINDING_KEY, build_prompt_relay_conditioning, build_prompt_relay_plan,
)
from helpers import FakeAudioVAE, FakeVideoVAE
from test_modular_motion_stage import _source
from test_prompt_relay_advanced import NativeLikeFakeClip, _allow_fixture_core_contract
from tools.build_modular_motion_effect_workflows import effect_api, effect_frontend
from tools.run_modular_motion_effect_gpu import build_probe_graph


def _paired(monkeypatch, *, windowed=False, with_stage=False):
    _allow_fixture_core_contract(monkeypatch)
    plan, frames, audio, smeared, seed_audio, report, configured, sampler, sigmas, av, parent = (
        _source(windowed=windowed, sampler_name="dual_clock_euler", scheduler="native_flow"))
    length, length_report = retimed_relay_length(plan)
    assert json.loads(length_report)["expanded_frames"] == length
    relay_plan, *_ = build_prompt_relay_plan(
        "Night road, same rider", "Rider starts to pedal\nRider accelerates smoothly",
        length, "auto_equal", "", "paper_v1", .1, False, False)
    width = frames.shape[2]
    height = frames.shape[1]
    relay_model, positive, relay_av, *_ = build_prompt_relay_conditioning(
        model=configured, clip=NativeLikeFakeClip(), video_vae=FakeVideoVAE(),
        audio_vae=FakeAudioVAE(), prompt_relay_plan=relay_plan,
        width=width, height=height, task_type="T2VA", audio_mode="native",
        audio_denoise_strength=1., add_source_as_reference=False,
        prompt_primary_audio_ordinal=0, strict_prompt_tags=True,
        ref_image_size="match", reference_video_policy="official_2_to_15s",
        execution_mode="apply_exp", query_chunk_rows=64)
    prepared, selected, table, context, _ = bind_motion_stage(
        plan, frames, audio, smeared, seed_audio, report, relay_model, sampler,
        sigmas, av, **parent)
    args = (prepared, positive, relay_av, relay_plan, plan, av, context)
    if with_stage:
        return args, (selected, table, frames, audio, smeared, seed_audio, report, parent)
    return args


@pytest.mark.parametrize("windowed", [False, True])
def test_external_relay_pairs_with_exact_retimed_motion_source(monkeypatch, windowed):
    args = _paired(monkeypatch, windowed=windowed)
    model, positive, report_json = bind_motion_relay(*args)
    assert model is args[0] and positive is args[1]
    report = json.loads(report_json)
    assert report["status"] == "paired_retained_model_and_conditioning"
    assert report["frame_count"] == args[4]["expanded_length"]
    assert report["sampled"] is False and report["quality_accepted"] is False


def test_external_relay_refuses_wrong_plan_conditioning_and_source(monkeypatch):
    args = list(_paired(monkeypatch))
    wrong_plan, *_ = build_prompt_relay_plan(
        "Other prompt", "A\nB", 124, "auto_equal", "", "paper_v1", .1, False, False)
    with pytest.raises(ValueError, match="timeline differs"):
        bind_motion_relay(*args[:3], wrong_plan, *args[4:])
    wrong_positive = deepcopy(args[1])
    wrong_positive[0][1][PROMPT_RELAY_BINDING_KEY] = {}
    with pytest.raises(ValueError, match="CONDITIONING binding differ"):
        bind_motion_relay(args[0], wrong_positive, *args[2:])
    wrong_source = dict(args[5])
    wrong_source["samples"] = type(args[5]["samples"])(
        tuple(part.clone() for part in args[5]["samples"].unbind()))
    wrong_source["samples"].unbind()[0][0, 0, 0, 0, 0] = 1
    with pytest.raises(ValueError, match="StageContext does not bind"):
        bind_motion_relay(*args[:5], wrong_source, args[6])


@pytest.mark.parametrize("mode", ["report_only", "apply_exp"])
def test_external_relay_and_eav_observe_actual_motion_pass2_forward(monkeypatch, mode):
    args, extra = _paired(monkeypatch, windowed=True, with_stage=True)
    relay_model, positive, _ = bind_motion_relay(*args)
    selected, table, frames, audio, smeared, seed_audio, report, parent = extra
    effected, runtime, _ = apply_stage_eav(
        relay_model, table, args[5], args[6], EAVConfig(mode=mode))
    guider = BasicGuider.execute(effected, positive).result[0]
    result = sample_stage(RandomNoise.execute(929).result[0], guider,
                          selected, table, args[5], args[6])[2]
    candidate, source_report = audit_motion_stage(
        result, args[4], frames, audio, smeared, seed_audio, report, args[5], **parent)
    _, effect_report = audit_stage_eav(candidate, runtime)
    assert json.loads(source_report)["status"] == "source_bound_candidate_audited"
    observed = json.loads(effect_report)
    assert observed["status"] == "observed_" + mode, {
        key: observed[key] for key in ("status", "completed_forwards", "planned_forwards",
                                       "selector_calls", "sparse_producer_calls",
                                       "relay_attention_calls", "clock_match")}
    assert observed["relay_required"] is True
    assert observed["relay_attention_calls"] > 0


@pytest.mark.parametrize("variant", ["Fullclip", "Windowed"])
def test_private_motion_effect_graph_keeps_independent_editable_nodes(variant):
    graph = effect_frontend(variant)
    api = effect_api(graph, variant)
    nodes = {node["id"]: node for node in graph["nodes"]}
    links = {link[0]: link for link in graph["links"]}
    kinds = [node["type"] for node in nodes.values()]
    for kind in ("MiniMaxH3MotionRelayLengthEXPT8",
                 "MiniMaxH3PromptRelayPlanT8Advanced",
                 "MiniMaxH3PromptRelayConditioningT8Advanced",
                 "MiniMaxH3MotionRelayBindEXPT8",
                 "MiniMaxH3StageEAVConfigEXPT8",
                 "MiniMaxH3StageEAVApplyEXPT8",
                 "MiniMaxH3StageEAVAuditEXPT8"):
        assert kinds.count(kind) == 1
    assert kinds.count("SamplerCustomAdvanced") == 1
    assert kinds.count("MiniMaxH3StageSamplerEXPT8") == 1
    assert kinds.count("VHS_VideoCombine") == 2
    if variant == "Windowed":
        assert nodes[13]["widgets_values"] == [209, 0, "fixed", 12, "hot_ranges_only"]
        assert api["13"]["inputs"]["handle_frames"] == 12
        assert api["13"]["inputs"]["coverage"] == "hot_ranges_only"
        assert "control_after_generate" not in api["13"]["inputs"]
    decoder = nodes[20 if variant == "Windowed" else 19]
    assert nodes[links[decoder["inputs"][0]["link"]][1]]["type"] == "MiniMaxH3StageEAVAuditEXPT8"
    recovered = nodes[21 if variant == "Windowed" else 20]
    assert links[recovered["inputs"][2]["link"]][1:3] == (
        [13, 4] if variant == "Windowed" else [10, 1])
    assert len(api) < len(nodes)
    assert {int(node_id) for node_id in api} <= set(nodes)
    for link_id, source_id, output_slot, target_id, input_slot, dtype in graph["links"]:
        assert nodes[source_id]["outputs"][output_slot]["type"] == dtype
        assert nodes[target_id]["inputs"][input_slot]["type"] == dtype
        assert nodes[target_id]["inputs"][input_slot]["link"] == link_id


@pytest.mark.parametrize("variant", ["Fullclip", "Windowed"])
@pytest.mark.parametrize("mode", ["report_only", "apply_exp"])
def test_sealed_small_real_weight_probe_keeps_two_pass_source_and_effects(variant, mode):
    graph, source_sha = build_probe_graph(variant, eav_mode=mode)
    assert len(source_sha) == 64
    kinds = [node["class_type"] for node in graph.values()]
    assert kinds.count("SamplerCustomAdvanced") == 1
    assert kinds.count("MiniMaxH3StageSamplerEXPT8") == 1
    assert kinds.count("MiniMaxH3MotionRelayBindEXPT8") == 1
    assert kinds.count("MiniMaxH3StageEAVApplyEXPT8") == 1
    assert graph["12"]["inputs"]["mode"] == "manual_ranges"
    config = graph["36" if variant == "Windowed" else "34"]["inputs"]
    assert config["mode"] == mode
    assert config["tau"] == (10. if mode == "apply_exp" else .2)
    assert graph["5"]["inputs"]["length"] == 22
    if variant == "Windowed":
        assert graph["13"]["inputs"]["handle_frames"] == 12
        assert graph["13"]["inputs"]["coverage"] == "hot_ranges_only"
