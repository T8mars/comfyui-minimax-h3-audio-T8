"""Full-clip external Relay projected to each native v1 temporal segment."""
import json

import comfy.nested_tensor
import comfy.samplers
import pytest
import torch

from h3_audio_t8_pkg import chunked_two_pass_upscale_advanced as old
from h3_audio_t8_pkg import sampling
from h3_audio_t8_pkg.modular_sampling.chunked_effects import audit_chunked_eav, bind_chunked_eav
from h3_audio_t8_pkg.modular_sampling.chunked_source import slice_chunked_source
from h3_audio_t8_pkg.modular_sampling.chunked_v1_relay_nodes import (
    MiniMaxH3ChunkedV1RelayAuditEXPT8, MiniMaxH3ChunkedV1RelayProjectEXPT8,
)
from h3_audio_t8_pkg.modular_sampling.chunked_stages import (
    lift_chunked_segment, prepare_chunked_pass2, sample_chunked_pass2,
)
from h3_audio_t8_pkg.modular_sampling.chunked_v1_relay import (
    audit_v1_local_relay, project_v1_local_relay,
)
from h3_audio_t8_pkg.modular_sampling.eav import EAVConfig
from h3_audio_t8_pkg.prompt_relay_advanced import (
    build_prompt_relay_conditioning, build_prompt_relay_plan,
)
from helpers import FakeAudioVAE, FakeVideoVAE
from test_chunked_two_pass_global_noise_advanced import _CountingCoordinateNoise, _plan
from test_modular_chunked_stages import _fake_lift
from test_modular_speed_stages import _model
from test_prompt_relay_advanced import NativeLikeFakeClip
from tools.build_modular_chunked_v1_relay_workflow import AUDIT, PASS, PROJECT, build_pair


def _case(monkeypatch, *, with_frames=False):
    monkeypatch.setattr(old, "learned_upscale_h3_av_latent", _fake_lift)
    plan = _plan(old.build_chunked_two_pass_plan, strategy="full_frame_safe")
    relay_plan, *_ = build_prompt_relay_plan(
        global_prompt=("A continuous room with <Picture 1> first and <Picture 2> last"
                       if with_frames else "A continuous room"),
        local_prompts="A woman waves.\nShe walks away.", length=56,
        timing_mode="auto_equal", time_ranges="", math_profile="paper_v1",
        epsilon=0.1, allow_gaps=False, allow_overlaps=False,
    )
    provisional = {"samples": comfy.nested_tensor.NestedTensor((
        torch.zeros(1, 24, 17, 2, 2), torch.zeros(1, 32, 2, 94),
    ))}
    raw_model, _shape_bound_sampler, sigmas = sampling.setup_dual_clock_sampling(
        _model(), provisional, 3, 12., 3.,
    )
    sampler = comfy.samplers.sampler_object("euler")
    relay_model, positive, full_av, *_ = build_prompt_relay_conditioning(
        model=raw_model, clip=NativeLikeFakeClip(), video_vae=FakeVideoVAE(),
        audio_vae=FakeAudioVAE(), prompt_relay_plan=relay_plan,
        width=64, height=64, task_type="FL2VA" if with_frames else "T2VA",
        audio_mode="native", audio_denoise_strength=0.35,
        add_source_as_reference=False, prompt_primary_audio_ordinal=0,
        strict_prompt_tags=True, ref_image_size="match",
        reference_video_policy="official_2_to_15s", execution_mode="apply_exp",
        query_chunk_rows=64,
        **({"first_frame": torch.zeros((1, 64, 64, 3)),
            "last_frame": torch.ones((1, 64, 64, 3))} if with_frames else {}),
    )
    full_video, full_audio = full_av["samples"].unbind()
    source = {"samples": comfy.nested_tensor.NestedTensor((
        torch.zeros(1, 24, full_video.shape[2], 2, 2), full_audio.clone(),
    ))}
    noise = _CountingCoordinateNoise()
    context, _ = prepare_chunked_pass2(source, plan, noise)
    return (raw_model, sampler, sigmas, relay_model, positive, full_av,
            relay_plan, source, plan, noise, context)


def _segment(case, index):
    _raw, _sampler, _sigmas, _relay, _positive, _full, _relay_plan, source, plan, _noise, context = case
    segment, spec, _ = slice_chunked_source(source, plan, index)
    lifted, _ = lift_chunked_segment(segment, spec, context, plan)
    return segment, lifted, spec


def _project(case, index, previous=None):
    raw, _sampler, sigmas, relay_model, positive, full_av, relay_plan, _source, plan, _noise, context = case
    segment, lifted, spec = _segment(case, index)
    model, paired, runtime, report = project_v1_local_relay(
        raw, relay_model, positive, full_av, relay_plan,
        segment, lifted, spec, context, plan, sigmas, previous,
    )
    return model, paired, runtime, json.loads(report), segment, lifted, spec


@pytest.mark.parametrize("with_frames", [False, True])
def test_v1_three_segments_project_full_relay_and_sample_actual_calls(monkeypatch, with_frames):
    case = _case(monkeypatch, with_frames=with_frames)
    raw, sampler, sigmas, relay_model, positive, _full, _rplan, source, plan, noise, context = case
    assert _segment(case, 0)[2].count == 3
    first_binding = None
    previous = None
    for index in range(3):
        model, paired, runtime, report, segment, lifted, spec = _project(case, index, previous)
        assert report["sampled"] is False
        assert report["previous_anchor_inserted"] is (index > 0)
        binding = paired[0][1]["minimax_prompt_relay_binding"]
        if first_binding is None:
            first_binding = binding
        else:
            assert binding["events"][0]["midpoint"] == pytest.approx(
                first_binding["events"][0]["midpoint"] - (5.0 / 3.0) * spec.start_frame,
            )
            assert binding["keyframe_count"] >= 1
        output, previous, _ = sample_chunked_pass2(
            model, paired, segment, lifted, spec, context, plan,
            noise, sampler, sigmas, previous,
        )
        audited, audit_json = audit_v1_local_relay(previous, spec, runtime)
        assert audited is output
        audit = json.loads(audit_json)
        assert audit["status"] == "observed_relay_calls_quality_unverified", audit
        assert audit["actual_calls"] == audit["expected_calls"]
        assert audit["actual_calls"]["routed_attention_calls"] > 0
    assert output["samples"].unbind()[1] is context.original_audio
    assert raw.get_attachment("t8_modular_chunked_v1_local_relay_v1") is None
    assert relay_model.get_attachment("t8_modular_chunked_v1_local_relay_v1") is None
    assert positive[0][1]["minimax_prompt_relay_binding"]["plan_hash"] == first_binding["plan_hash"]


def test_v1_rejects_unprojected_stock_relay_and_wrong_previous(monkeypatch):
    case = _case(monkeypatch)
    raw, sampler, sigmas, relay_model, positive, full_av, relay_plan, _source, plan, noise, context = case
    segment0, lifted0, spec0 = _segment(case, 0)
    with pytest.raises(ValueError, match="Project external Prompt Relay"):
        sample_chunked_pass2(
            relay_model, positive, segment0, lifted0, spec0,
            context, plan, noise, sampler, sigmas,
        )
    segment1, lifted1, spec1 = _segment(case, 1)
    with pytest.raises(ValueError, match="exact prior PASS2 result"):
        project_v1_local_relay(
            raw, relay_model, positive, full_av, relay_plan,
            segment1, lifted1, spec1, context, plan, sigmas,
        )
    model0, paired0, _runtime0, _report0, *_ = _project(case, 0)
    _, previous, _ = sample_chunked_pass2(
        model0, paired0, segment0, lifted0, spec0, context, plan,
        noise, sampler, sigmas,
    )
    model1, paired1, _runtime1, _report1, *_ = _project(case, 1, previous)
    with pytest.raises(ValueError, match="exact prior PASS2 result"):
        sample_chunked_pass2(
            model1, paired1, segment1, lifted1, spec1, context, plan,
            noise, sampler, sigmas,
        )
    with pytest.raises(ValueError, match="not bound to this exact segment"):
        sample_chunked_pass2(
            model1, paired1, segment1, lifted1, spec1, context, plan,
            noise, sampler, sigmas * 0.9, previous,
        )


def test_v1_all_three_segments_relay_and_external_eav_actual_calls(monkeypatch):
    case = _case(monkeypatch)
    _raw, sampler, sigmas, _relay_model, _positive, _full, _rplan, _source, plan, noise, context = case
    previous = None
    for index in range(3):
        model, paired, _runtime, _report, segment, lifted, spec = _project(case, index, previous)
        eav_model, eav_runtime, _report, _stage = bind_chunked_eav(
            model, sigmas, segment, lifted, spec, context, plan,
            EAVConfig(mode="report_only", start_video_progress=0.1, g_hard_limit=3.0),
        )
        _, result, _ = sample_chunked_pass2(
            eav_model, paired, segment, lifted, spec, context, plan,
            noise, sampler, sigmas, previous,
        )
        audit = json.loads(audit_chunked_eav(result, spec, eav_runtime)[1])
        assert audit["relay_required"] is True
        assert audit["relay_attention_calls"] > 0
        assert audit["completed_forwards"] == audit["planned_forwards"] == 3
        previous = result


def test_v1_relay_eav_third_segment_obeys_original_gain_hard_limit(monkeypatch):
    case = _case(monkeypatch)
    _raw, sampler, sigmas, _relay_model, _positive, _full, _rplan, _source, plan, noise, context = case
    previous = None
    for index in range(3):
        model, paired, _runtime, _report, segment, lifted, spec = _project(case, index, previous)
        eav_model, _eav_runtime, _report, _stage = bind_chunked_eav(
            model, sigmas, segment, lifted, spec, context, plan,
            EAVConfig(mode="report_only", start_video_progress=0.1, g_hard_limit=1.5),
        )
        inputs = (eav_model, paired, segment, lifted, spec, context, plan,
                  noise, sampler, sigmas, previous)
        if index == 2:
            with pytest.raises(RuntimeError, match="configured hard limit 1.500000"):
                sample_chunked_pass2(*inputs)
        else:
            _, previous, _ = sample_chunked_pass2(*inputs)


def test_v1_public_project_and_audit_are_registered():
    project = MiniMaxH3ChunkedV1RelayProjectEXPT8.define_schema()
    audit = MiniMaxH3ChunkedV1RelayAuditEXPT8.define_schema()
    assert project.node_id == "MiniMaxH3ChunkedV1RelayProjectEXPT8"
    assert audit.node_id == "MiniMaxH3ChunkedV1RelayAuditEXPT8"
    assert "previous_result" in [item.id for item in project.inputs]


def test_v1_three_segment_candidate_has_paired_projection_and_prior_dependency():
    frontend, api = build_pair()
    nodes = {node["id"]: node for node in frontend["nodes"]}
    links = {link[0]: link for link in frontend["links"]}
    front_passes = [node for node in frontend["nodes"] if node["type"] == PASS]
    front_projects = [node for node in frontend["nodes"] if node["type"] == PROJECT]
    front_audits = [node for node in frontend["nodes"] if node["type"] == AUDIT]
    assert len(front_passes) == len(front_projects) == len(front_audits) == 3
    assert sum(node["type"] == "MiniMaxH3PromptRelayPlanT8Advanced"
               for node in frontend["nodes"]) == 1
    assert sum(node["type"] == "MiniMaxH3PromptRelayConditioningT8Advanced"
               for node in frontend["nodes"]) == 1
    for index, stage in enumerate(front_passes):
        def incoming(name):
            link_id = next(item["link"] for item in stage["inputs"] if item["name"] == name)
            return links[link_id]
        model_link = incoming("model")
        paired_link = incoming("positive")
        assert model_link[1] == paired_link[1]
        assert model_link[2] == 0 and paired_link[2] == 1
        projected = nodes[model_link[1]]
        assert projected["type"] == PROJECT
        previous_input = next(item for item in projected["inputs"]
                              if item["name"] == "previous_result")
        if index == 0:
            assert previous_input["link"] is None
        else:
            assert links[previous_input["link"]][1:3] == [front_passes[index - 1]["id"], 1]
    api_passes = [(key, node) for key, node in api.items() if node["class_type"] == PASS]
    api_projects = [(key, node) for key, node in api.items() if node["class_type"] == PROJECT]
    api_audits = [(key, node) for key, node in api.items() if node["class_type"] == AUDIT]
    assert len(api_passes) == len(api_projects) == len(api_audits) == 3
    for index, (pass_id, stage) in enumerate(api_passes):
        project_id = stage["inputs"]["model"][0]
        assert stage["inputs"]["positive"] == [project_id, 1]
        project_inputs = api[project_id]["inputs"]
        if index:
            assert project_inputs["previous_result"] == [api_passes[index - 1][0], 1]
        else:
            assert "previous_result" not in project_inputs
        assert any(node["inputs"]["segment_result"] == [pass_id, 1]
                   and node["inputs"]["runtime"] == [project_id, 2]
                   for _id, node in api_audits)


def test_v1_three_segment_relay_eav_candidate_uses_joint_actual_call_audits():
    frontend, api = build_pair(with_eav=True)
    nodes = {node["id"]: node for node in frontend["nodes"]}
    links = {link[0]: link for link in frontend["links"]}
    passes = [node for node in frontend["nodes"] if node["type"] == PASS]
    projects = [node for node in frontend["nodes"] if node["type"] == PROJECT]
    eav_apply = [node for node in frontend["nodes"]
                 if node["type"] == "MiniMaxH3ChunkedPass2EAVApplyEXPT8"]
    eav_audit = [node for node in frontend["nodes"]
                 if node["type"] == "MiniMaxH3ChunkedPass2EAVAuditEXPT8"]
    assert len(passes) == len(projects) == len(eav_apply) == len(eav_audit) == 3
    assert not any(node["type"] == AUDIT for node in frontend["nodes"])
    config = next(node for node in frontend["nodes"]
                  if node["type"] == "MiniMaxH3StageEAVConfigEXPT8")
    assert config["widgets_values"][-1] == 3.0
    for stage in passes:
        model_input = next(item for item in stage["inputs"] if item["name"] == "model")
        apply = nodes[links[model_input["link"]][1]]
        assert apply["type"] == "MiniMaxH3ChunkedPass2EAVApplyEXPT8"
        applied_model = next(item for item in apply["inputs"] if item["name"] == "model")
        assert nodes[links[applied_model["link"]][1]]["type"] == PROJECT
        positive_input = next(item for item in stage["inputs"] if item["name"] == "positive")
        assert links[positive_input["link"]][1] == links[applied_model["link"]][1]
        assert any(links[item["link"]][1] == stage["id"]
                   for audit in eav_audit for item in audit["inputs"]
                   if item["name"] == "segment_result")
    api_passes = [(key, node) for key, node in api.items() if node["class_type"] == PASS]
    assert len(api_passes) == 3
    assert sum(node["class_type"] == "MiniMaxH3ChunkedPass2EAVAuditEXPT8"
               for node in api.values()) == 3
    assert not any(node["class_type"] == AUDIT for node in api.values())
    assert next(node["inputs"]["g_hard_limit"] for node in api.values()
                if node["class_type"] == "MiniMaxH3StageEAVConfigEXPT8") == 3.0
    for _id, stage in api_passes:
        eav_id = stage["inputs"]["model"][0]
        apply = api[eav_id]
        assert apply["class_type"] == "MiniMaxH3ChunkedPass2EAVApplyEXPT8"
        project_id = apply["inputs"]["model"][0]
        assert api[project_id]["class_type"] == PROJECT
        assert stage["inputs"]["positive"] == [project_id, 1]
