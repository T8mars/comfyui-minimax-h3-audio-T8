"""H16 full-clip Relay projection and Relay+EAV actual window calls."""
import json
from pathlib import Path

import comfy.nested_tensor
import pytest
import torch

from h3_audio_t8_pkg import chunked_two_pass_upscale_advanced as legacy
from h3_audio_t8_pkg import sampling
from h3_audio_t8_pkg.modular_sampling.chunked_source import slice_chunked_source
from h3_audio_t8_pkg.modular_sampling.chunked_stages import (
    lift_chunked_segment, prepare_chunked_pass2,
)
from h3_audio_t8_pkg.modular_sampling.eav import EAVConfig
from h3_audio_t8_pkg.modular_sampling.h16_effects import audit_h16_eav, bind_h16_eav
from h3_audio_t8_pkg.modular_sampling.h16_nodes import (
    MiniMaxH3H16RelayAuditEXPT8, MiniMaxH3H16RelayProjectEXPT8,
)
from h3_audio_t8_pkg.modular_sampling.h16_relay import (
    assert_h16_relay_binding, audit_h16_relay, project_h16_relay,
)
from h3_audio_t8_pkg.modular_sampling.h16_stages import build_h16_plan, sample_h16_pass2
from h3_audio_t8_pkg.prompt_relay_advanced import (
    build_prompt_relay_conditioning, build_prompt_relay_plan,
)
from helpers import FakeAudioVAE, FakeVideoVAE
from test_chunked_two_pass_global_noise_advanced import _CountingCoordinateNoise
from test_modular_h16_stages import _h16_source, _no_op_lift
from test_modular_speed_stages import _model
from test_prompt_relay_advanced import NativeLikeFakeClip
from tools.build_modular_h16_workflow import (
    ROOT, TEMPLATE, WINDOW_COUNT, add_eav_frontend, add_relay_frontend,
    split_api, split_frontend,
)


def _case(monkeypatch, *, with_frame=False, last_frame=False, length=51):
    monkeypatch.setattr(legacy, "learned_upscale_h3_av_latent", _no_op_lift)
    raw_model, sampler, sigmas = sampling.setup_dual_clock_sampling(
        _model(), _h16_source(), 3, 12., 3.,
    )
    relay_plan, *_ = build_prompt_relay_plan(
        global_prompt=("One continuous room with <Picture 1> first"
                       + (" and <Picture 2> last" if last_frame else "")
                       if with_frame else "One continuous room"),
        local_prompts="A woman waves.\nShe walks away.",
        length=length, timing_mode="auto_equal", time_ranges="", math_profile="paper_v1",
        epsilon=0.1, allow_gaps=False, allow_overlaps=False,
    )
    relay_model, positive, full_av, *_ = build_prompt_relay_conditioning(
        model=raw_model, clip=NativeLikeFakeClip(), video_vae=FakeVideoVAE(),
        audio_vae=FakeAudioVAE(), prompt_relay_plan=relay_plan,
        width=256, height=256,
        task_type="FL2VA" if last_frame else "I2VA" if with_frame else "T2VA",
        audio_mode="native",
        audio_denoise_strength=0.35, add_source_as_reference=False,
        prompt_primary_audio_ordinal=0, strict_prompt_tags=True,
        ref_image_size="match", reference_video_policy="official_2_to_15s",
        execution_mode="apply_exp", query_chunk_rows=64,
        **({"first_frame": torch.zeros((1, 256, 256, 3)),
            **({"last_frame": torch.ones((1, 256, 256, 3))} if last_frame else {})}
           if with_frame else {}),
    )
    video, audio = full_av["samples"].unbind()
    source = {
        "samples": comfy.nested_tensor.NestedTensor((video.clone(), audio.clone())),
        "noise_mask": comfy.nested_tensor.NestedTensor((
            torch.ones_like(video[:, :1]), torch.ones_like(audio),
        )),
    }
    plan, _ = build_h16_plan(source)
    noise = _CountingCoordinateNoise()
    context, _ = prepare_chunked_pass2(source, plan, noise)
    return raw_model, sampler, sigmas, relay_model, positive, full_av, relay_plan, source, plan, noise, context


def _window(case, index):
    _raw, _sampler, _sigmas, _relay, _positive, _full, _rplan, source, plan, _noise, context = case
    segment, spec, _ = slice_chunked_source(source, plan, index)
    lifted, _ = lift_chunked_segment(segment, spec, context, plan)
    return segment, spec, lifted


def _project(case, index, previous=None, audio_output="preserve_first_pass"):
    raw, _sampler, sigmas, relay_model, positive, full_av, relay_plan, _source, plan, _noise, context = case
    segment, spec, lifted = _window(case, index)
    model, paired, runtime, report_json = project_h16_relay(
        raw, relay_model, positive, full_av, relay_plan, segment, lifted,
        spec, context, plan, sigmas, audio_output, previous,
    )
    return model, paired, runtime, json.loads(report_json), segment, spec, lifted


def test_h16_relay_first_and_guarded_second_window_actual_calls(monkeypatch):
    case = _case(monkeypatch)
    raw, sampler, sigmas, relay_model, positive, _full, _rplan, _source, plan, noise, context = case
    model0, paired0, runtime0, report0, segment0, spec0, lifted0 = _project(case, 0)
    assert report0["window_frames"][0] == 0 and report0["sampled"] is False
    with pytest.raises(ValueError, match="Project full-clip Prompt Relay"):
        sample_h16_pass2(relay_model, positive, segment0, lifted0, spec0,
                         context, plan, noise, sampler, sigmas)
    _output0, previous, _core, _ = sample_h16_pass2(
        model0, paired0, segment0, lifted0, spec0,
        context, plan, noise, sampler, sigmas,
    )
    audit0 = json.loads(audit_h16_relay(previous, spec0, runtime0)[1])
    assert audit0["status"] == "observed_relay_calls_quality_unverified", audit0
    assert audit0["actual_calls"] == audit0["expected_calls"]
    assert audit0["actual_calls"]["completed_forwards"] == 3
    model1, paired1, runtime1, report1, segment1, spec1, lifted1 = _project(case, 1, previous)
    assert report1["locked_overlap_tokens"] > 0
    assert report1["transition_overlap_tokens"] > 0
    assert report1["window_frames"][0] > 0
    first = paired0[0][1]["minimax_prompt_relay_binding"]
    second = paired1[0][1]["minimax_prompt_relay_binding"]
    assert second["events"][0]["midpoint"] == pytest.approx(
        first["events"][0]["midpoint"] - 5.0 / 3.0 * spec1.start_frame,
    )
    _output1, result1, _core1, _ = sample_h16_pass2(
        model1, paired1, segment1, lifted1, spec1,
        context, plan, noise, sampler, sigmas, previous,
    )
    audit1 = json.loads(audit_h16_relay(result1, spec1, runtime1)[1])
    assert audit1["status"] == "observed_relay_calls_quality_unverified", audit1
    assert audit1["actual_calls"] == audit1["expected_calls"]
    assert raw.get_attachment("t8_modular_h16_relay_v1") is None


def test_h16_i2va_first_frame_uses_one_live_reanchor_per_window(monkeypatch):
    case = _case(monkeypatch, with_frame=True)
    _raw, sampler, sigmas, _relay_model, positive, _full, _rplan, _source, plan, noise, context = case
    assert len(positive[0][1]["minimax_keyframes"]) == 1
    previous = None
    for index in range(2):
        model, paired, runtime, _report, segment, spec, lifted = _project(case, index, previous)
        assert len(paired[0][1]["minimax_keyframes"]) == 1
        _output, previous, _core, _ = sample_h16_pass2(
            model, paired, segment, lifted, spec, context, plan,
            noise, sampler, sigmas, previous,
        )
        audit = json.loads(audit_h16_relay(previous, spec, runtime)[1])
        assert audit["status"] == "observed_relay_calls_quality_unverified", audit


def test_h16_relay_all_seven_guarded_windows_and_final_refined_audio(monkeypatch):
    case = _case(monkeypatch, with_frame=True, length=124)
    _raw, sampler, sigmas, _relay_model, _positive, _full, _rplan, source, plan, noise, context = case
    assert slice_chunked_source(source, plan, 0)[1].count == WINDOW_COUNT
    previous = None
    for index in range(WINDOW_COUNT):
        model, paired, runtime, report, segment, spec, lifted = _project(
            case, index, previous, "refined_exp",
        )
        assert report["window_index"] == index
        output, previous, _core, _ = sample_h16_pass2(
            model, paired, segment, lifted, spec, context,
            plan, noise, sampler, sigmas, previous,
            audio_output="refined_exp",
        )
        audited, audit_json = audit_h16_relay(previous, spec, runtime)
        audit = json.loads(audit_json)
        assert audited is output
        assert audit["status"] == "observed_relay_calls_quality_unverified", audit
        assert audit["actual_calls"]["completed_forwards"] == 3
    final_audio = output["samples"].tensors[1]
    assert final_audio.shape == context.original_audio.shape
    assert final_audio is not context.original_audio


def test_h16_relay_eav_all_seven_report_only_and_refined_audio(monkeypatch):
    case = _case(monkeypatch, with_frame=True, length=124)
    _raw, sampler, sigmas, _relay_model, _positive, _full, _rplan, _source, plan, noise, context = case
    previous = None
    for index in range(WINDOW_COUNT):
        model, paired, _standalone, _report, segment, spec, lifted = _project(
            case, index, previous, "refined_exp",
        )
        eav_model, runtime, _pre_report, _stage = bind_h16_eav(
            model, sigmas, segment, lifted, spec, context, plan, previous,
            "refined_exp", EAVConfig(mode="report_only", start_video_progress=0.1,
                                     g_hard_limit=3.0),
            paired,
        )
        output, previous, _core, _ = sample_h16_pass2(
            eav_model, paired, segment, lifted, spec, context,
            plan, noise, sampler, sigmas, previous, audio_output="refined_exp",
        )
        audited, audit_json = audit_h16_eav(previous, spec, runtime)
        audit = json.loads(audit_json)
        assert audited is output
        assert audit["status"] == "observed_report_only", audit
        assert audit["relay_required"] is True
        assert audit["relay_attention_calls"] > 0
        assert audit["completed_forwards"] == audit["planned_forwards"] == 3
    assert output["samples"].tensors[1] is not context.original_audio


def test_h16_fl2va_relay_eav_projects_first_middle_last_tasks(monkeypatch):
    case = _case(monkeypatch, with_frame=True, last_frame=True)
    _raw, sampler, sigmas, _relay_model, _positive, _full, _rplan, source, plan, noise, context = case
    count = slice_chunked_source(source, plan, 0)[1].count
    assert count >= 3
    observed = []
    previous = None
    for index in range(count):
        model, paired, _standalone, _report, segment, spec, lifted = _project(
            case, index, previous,
        )
        observed.append(paired[0][1]["minimax_prompt_relay_binding"]["task"])
        eav_model, runtime, _pre_report, _stage = bind_h16_eav(
            model, sigmas, segment, lifted, spec, context, plan, previous,
            "preserve_first_pass", EAVConfig(mode="report_only",
                                             start_video_progress=0.1, g_hard_limit=3.0),
            paired,
        )
        _output, previous, _core, _ = sample_h16_pass2(
            eav_model, paired, segment, lifted, spec, context,
            plan, noise, sampler, sigmas, previous,
        )
        assert json.loads(audit_h16_eav(previous, spec, runtime)[1])["status"] == "observed_report_only"
    assert observed[0] == "i2va" and observed[-1] == "l2va"
    assert all(task == "t2va" for task in observed[1:-1])


@pytest.mark.parametrize("window_index", [0, 1])
def test_h16_relay_plus_eav_joint_actual_calls(monkeypatch, window_index):
    case = _case(monkeypatch)
    _raw, sampler, sigmas, _relay_model, _positive, _full, _rplan, _source, plan, noise, context = case
    previous = None
    if window_index:
        model0, paired0, _runtime0, _report0, segment0, spec0, lifted0 = _project(case, 0)
        _output0, previous, _core0, _ = sample_h16_pass2(
            model0, paired0, segment0, lifted0, spec0,
            context, plan, noise, sampler, sigmas,
        )
    model, paired, _standalone_runtime, _report, segment, spec, lifted = _project(
        case, window_index, previous,
    )
    eav_model, eav_runtime, _pre_report, _stage = bind_h16_eav(
        model, sigmas, segment, lifted, spec, context, plan, previous,
        "preserve_first_pass",
        EAVConfig(mode="apply_exp", start_video_progress=0.1, g_hard_limit=3.0),
        paired,
    )
    _output, result, _core, _ = sample_h16_pass2(
        eav_model, paired, segment, lifted, spec, context,
        plan, noise, sampler, sigmas, previous,
    )
    report = json.loads(audit_h16_eav(result, spec, eav_runtime)[1])
    assert report["status"] == "observed_apply_exp", report
    assert report["relay_required"] is True
    assert report["relay_attention_calls"] > 0
    assert report["completed_forwards"] == report["planned_forwards"] == 3


def test_h16_relay_rejects_wrong_pair_plan_and_window_before_sampling(monkeypatch):
    case = _case(monkeypatch)
    raw, _sampler, sigmas, relay_model, positive, full_av, relay_plan, _source, plan, _noise, context = case
    segment, spec, lifted = _window(case, 0)
    wrong_plan, *_ = build_prompt_relay_plan(
        global_prompt="Changed room", local_prompts="A woman waves.\nShe walks away.",
        length=51, timing_mode="auto_equal", time_ranges="", math_profile="paper_v1",
        epsilon=0.1, allow_gaps=False, allow_overlaps=False,
    )
    with pytest.raises(ValueError, match="another Plan"):
        project_h16_relay(raw, relay_model, positive, full_av, wrong_plan,
                          segment, lifted, spec, context, plan, sigmas)
    unpaired = [[tensor, dict(metadata)] for tensor, metadata in positive]
    unpaired[0][1].pop("minimax_prompt_relay_binding")
    with pytest.raises(ValueError, match="MODEL and CONDITIONING"):
        project_h16_relay(raw, relay_model, unpaired, full_av, relay_plan,
                          segment, lifted, spec, context, plan, sigmas)
    model, paired, _runtime, _report, _segment, _spec, _lifted = _project(case, 0)
    with pytest.raises(ValueError, match="exact window"):
        assert_h16_relay_binding(model, paired, segment, lifted, spec, context,
                                 plan, None, sigmas + 0.01, "preserve_first_pass")
    with pytest.raises(ValueError, match="matching projected CONDITIONING"):
        bind_h16_eav(model, sigmas, segment, lifted, spec, context, plan,
                     None, "preserve_first_pass", EAVConfig(mode="report_only"))


def test_h16_relay_public_node_contract():
    project = MiniMaxH3H16RelayProjectEXPT8.GET_NODE_INFO_V1()
    audit = MiniMaxH3H16RelayAuditEXPT8.GET_NODE_INFO_V1()
    assert project["name"] == "MiniMaxH3H16RelayProjectEXPT8"
    assert project["output_name"] == ["model", "positive", "runtime", "report_json"]
    assert audit["output_name"] == ["cumulative_av_latent", "report_json"]


@pytest.mark.parametrize("with_eav", [False, True])
def test_h16_relay_candidate_seven_explicit_windows_and_combined_wiring(with_eav):
    original_bytes = TEMPLATE.read_bytes()
    graph = add_relay_frontend(split_frontend(json.loads(original_bytes)),
                               with_audit=not with_eav)
    if with_eav:
        graph = add_eav_frontend(graph)
    assert TEMPLATE.read_bytes() == original_bytes
    nodes = {node["id"]: node for node in graph["nodes"]}
    links = {link[0]: link for link in graph["links"]}
    windows = sorted((node for node in nodes.values()
                      if node["type"] == "MiniMaxH3H16Pass2WindowEXPT8"),
                     key=lambda node: node["id"])
    projects = sorted((node for node in nodes.values()
                       if node["type"] == "MiniMaxH3H16RelayProjectEXPT8"),
                      key=lambda node: node["id"])
    assert len(windows) == len(projects) == WINDOW_COUNT
    assert sum(node["type"] == "MiniMaxH3PromptRelayPlanT8Advanced"
               for node in nodes.values()) == 1
    assert sum(node["type"] == "MiniMaxH3PromptRelayConditioningT8Advanced"
               for node in nodes.values()) == 1
    applies = sorted((node for node in nodes.values()
                      if node["type"] == "MiniMaxH3H16EAVApplyEXPT8"),
                     key=lambda node: node["id"])
    audits = sorted((node for node in nodes.values()
                     if node["type"] == ("MiniMaxH3H16EAVAuditEXPT8" if with_eav
                                         else "MiniMaxH3H16RelayAuditEXPT8")),
                    key=lambda node: node["id"])
    assert len(audits) == WINDOW_COUNT
    assert len(applies) == (WINDOW_COUNT if with_eav else 0)

    def source(node, name):
        link_id = next(item["link"] for item in node["inputs"] if item["name"] == name)
        return None if link_id is None else links[link_id][1:3]

    for index, (window, project) in enumerate(zip(windows, projects, strict=True)):
        assert source(window, "positive") == [project["id"], 1]
        assert source(project, "previous_result") == (
            None if index == 0 else [windows[index - 1]["id"], 1]
        )
        if with_eav:
            apply = applies[index]
            assert source(window, "model") == [apply["id"], 0]
            assert source(apply, "model") == [project["id"], 0]
            assert source(apply, "relay_positive") == [project["id"], 1]
            assert source(audits[index], "runtime") == [apply["id"], 1]
        else:
            assert source(window, "model") == [project["id"], 0]
            assert source(audits[index], "runtime") == [project["id"], 2]
        assert source(audits[index], "window_result") == [window["id"], 1]
    assert source(nodes[20], "av_latent") == [audits[-1]["id"], 0]
    api = split_api(graph)
    assert api["h16_report"]["inputs"]["source"] == [str(audits[-1]["id"]), 1]
    assert sum(node["class_type"] == "MiniMaxH3H16RelayProjectEXPT8"
               for node in api.values()) == WINDOW_COUNT
    folder = "candidate-relay-eav-v3" if with_eav else "candidate-relay-v1"
    suffix = "_Relay_EAV_EXP" if with_eav else "_Relay_EXP"
    candidate = ROOT / Path("artifacts/development/modular-sampling-m4-h16-relay-20260923") \
        / folder / f"H16_124F_Seven_Windows_Separate_PASS2{suffix}.json"
    assert json.loads(candidate.read_text(encoding="utf-8")) == graph
