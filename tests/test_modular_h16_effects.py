"""H16 external EAV: exact guarded-overlap AV input, identity and audit."""
import json
from pathlib import Path

import pytest
import torch

from h3_audio_t8_pkg import chunked_two_pass_upscale_advanced as legacy
from h3_audio_t8_pkg import sampling
from h3_audio_t8_pkg.modular_sampling.chunked_source import slice_chunked_source
from h3_audio_t8_pkg.modular_sampling.chunked_stages import (
    lift_chunked_segment, prepare_chunked_pass2,
)
from h3_audio_t8_pkg.modular_sampling.eav import EAVConfig, KEY as EAV_KEY
from h3_audio_t8_pkg.modular_sampling.h16_effects import (
    assert_h16_eav_binding, audit_h16_eav, bind_h16_eav,
)
from h3_audio_t8_pkg.modular_sampling.h16_nodes import (
    MiniMaxH3H16EAVApplyEXPT8, MiniMaxH3H16EAVAuditEXPT8,
)
from h3_audio_t8_pkg.modular_sampling.h16_stages import (
    build_h16_plan, h16_window_piece, sample_h16_pass2,
)
from h3_audio_t8_pkg.modular_sampling.results import _input_identity
from test_chunked_two_pass_global_noise_advanced import _CountingCoordinateNoise
from test_modular_h16_stages import _h16_source, _native_like_piece, _no_op_lift
from test_modular_speed_stages import _model
from test_progressive_sampling_runtime import conditioning
from tools.build_modular_h16_workflow import (
    ROOT, TEMPLATE, WINDOW_COUNT, add_eav_frontend, split_frontend, split_api,
)


def test_h16_each_effect_piece_matches_actual_sampler_and_guarded_mask(monkeypatch):
    monkeypatch.setattr(legacy, "learned_upscale_h3_av_latent", _no_op_lift)
    monkeypatch.setattr(legacy, "rebind_dual_clock_sampler", lambda _model, _piece, sampler: sampler)
    seen = []

    def capture(piece, *args, **kwargs):
        seen.append(_input_identity(piece))
        return _native_like_piece(piece, *args, **kwargs)

    monkeypatch.setattr(legacy, "sample_piece", capture)
    source = _h16_source()
    plan, _ = build_h16_plan(source)
    noise = _CountingCoordinateNoise()
    context, _ = prepare_chunked_pass2(source, plan, noise)
    first_segment, first_spec, _ = slice_chunked_source(source, plan, 0)
    first_lifted, _ = lift_chunked_segment(first_segment, first_spec, context, plan)
    model, sampler, sigmas = sampling.setup_dual_clock_sampling(
        _model(), first_lifted, 1, 12., 3.,
    )
    previous = None
    for index in range(first_spec.count):
        segment, spec, _ = slice_chunked_source(source, plan, index)
        lifted, _ = lift_chunked_segment(segment, spec, context, plan)
        expected, locked, transition = h16_window_piece(
            segment, lifted, spec, context, plan, previous, "refined_exp",
        )
        video_mask, audio_mask = expected["noise_mask"].unbind()
        assert torch.count_nonzero(audio_mask) == audio_mask.numel()
        if index:
            assert locked > 0 and transition > 0
            assert torch.count_nonzero(video_mask[:, :, :max(1, locked // 2)]) == 0
        patched, runtime, _report, stage = bind_h16_eav(
            model, sigmas, segment, lifted, spec, context, plan,
            previous, "refined_exp", EAVConfig(mode="report_only"),
        )
        assert json.loads(stage.profile)["locked_overlap_tokens"] == locked
        assert json.loads(stage.profile)["transition_overlap_tokens"] == transition
        assert_h16_eav_binding(patched, segment, lifted, spec, context, plan,
                               previous, sigmas, "refined_exp")
        output, previous, _core, _ = sample_h16_pass2(
            patched, [], segment, lifted, spec, context, plan,
            noise, sampler, sigmas, previous, audio_output="refined_exp",
        )
        assert seen[-1] == _input_identity(expected)
        audited, audit_json = audit_h16_eav(previous, spec, runtime)
        assert audited is output
        assert json.loads(audit_json)["status"] == "unverified_incomplete_stage_coverage"
    assert len(seen) == first_spec.count
    assert previous.output_latent["samples"].tensors[1] is output["samples"].tensors[1]


def test_h16_effect_rejects_wrong_window_policy_and_mutation_before_sampling(monkeypatch):
    monkeypatch.setattr(legacy, "learned_upscale_h3_av_latent", _no_op_lift)
    source = _h16_source()
    plan, _ = build_h16_plan(source)
    noise = _CountingCoordinateNoise()
    context, _ = prepare_chunked_pass2(source, plan, noise)
    segment, spec, _ = slice_chunked_source(source, plan, 0)
    lifted, _ = lift_chunked_segment(segment, spec, context, plan)
    model, sampler, sigmas = sampling.setup_dual_clock_sampling(_model(), lifted, 1, 12., 3.)
    patched, runtime, _report, _ = bind_h16_eav(
        model, sigmas, segment, lifted, spec, context, plan,
        None, "preserve_first_pass", EAVConfig(mode="report_only"),
    )
    with pytest.raises(ValueError, match="different audio policy|exact window"):
        assert_h16_eav_binding(patched, segment, lifted, spec, context, plan,
                               None, sigmas, "refined_exp")
    with pytest.raises(ValueError, match="exact window"):
        assert_h16_eav_binding(patched, segment, lifted, spec, context, plan,
                               None, sigmas + 0.01, "preserve_first_pass")
    unwrapped = patched.clone()
    unwrapped.remove_wrappers_with_key("diffusion_model", EAV_KEY)
    with pytest.raises(ValueError, match="wrapper was removed"):
        assert_h16_eav_binding(unwrapped, segment, lifted, spec, context, plan,
                               None, sigmas, "preserve_first_pass")
    changed_clock = patched.clone()
    changed_clock.model_options["transformer_options"]["minimax_h3_sigma_shift_video"] = 13.0
    with pytest.raises(ValueError, match="exact window"):
        assert_h16_eav_binding(changed_clock, segment, lifted, spec, context, plan,
                               None, sigmas, "preserve_first_pass")
    lifted["samples"].tensors[0][0, 0, 0, 0, 0] += 1
    with pytest.raises(ValueError, match="exact window"):
        sample_h16_pass2(patched, [], segment, lifted, spec, context, plan,
                         noise, sampler, sigmas)
    assert runtime.snapshot()["status"] == "unverified_incomplete_stage_coverage"


def test_h16_first_window_real_tiny_h3_eav_report_only_calls(monkeypatch):
    monkeypatch.setattr(legacy, "learned_upscale_h3_av_latent", _no_op_lift)
    source = _h16_source()
    video, audio = source["samples"].unbind()
    source["samples"] = type(source["samples"])((video / 100000, audio / 100000))
    plan, _ = build_h16_plan(source)
    noise = _CountingCoordinateNoise()
    context, _ = prepare_chunked_pass2(source, plan, noise)
    segment, spec, _ = slice_chunked_source(source, plan, 0)
    lifted, _ = lift_chunked_segment(segment, spec, context, plan)
    model, sampler, sigmas = sampling.setup_dual_clock_sampling(_model(), lifted, 3, 12., 3.)
    baseline, *_ = sample_h16_pass2(
        model, conditioning(), segment, lifted, spec, context, plan,
        noise, sampler, sigmas,
    )
    patched, runtime, _report, _ = bind_h16_eav(
        model, sigmas, segment, lifted, spec, context, plan,
        None, "preserve_first_pass",
        EAVConfig(mode="report_only", start_video_progress=0.1),
    )
    output, result, _core, _ = sample_h16_pass2(
        patched, conditioning(), segment, lifted, spec, context, plan,
        noise, sampler, sigmas,
    )
    for actual, expected in zip(output["samples"].unbind(), baseline["samples"].unbind()):
        assert torch.equal(actual, expected)
    audited, report_json = audit_h16_eav(result, spec, runtime)
    report = json.loads(report_json)
    assert audited is output
    assert report["status"] == "observed_report_only", report
    assert report["completed_forwards"] == report["planned_forwards"] == 3
    assert report["selector_calls"] > 0


@pytest.mark.parametrize("mode", ["report_only", "apply_exp"])
def test_h16_guarded_second_window_real_tiny_h3_eav_calls(monkeypatch, mode):
    monkeypatch.setattr(legacy, "learned_upscale_h3_av_latent", _no_op_lift)
    source = _h16_source()
    video, audio = source["samples"].unbind()
    source["samples"] = type(source["samples"])((video / 100000, audio / 100000))
    plan, _ = build_h16_plan(source)
    noise = _CountingCoordinateNoise()
    context, _ = prepare_chunked_pass2(source, plan, noise)
    segment0, spec0, _ = slice_chunked_source(source, plan, 0)
    lifted0, _ = lift_chunked_segment(segment0, spec0, context, plan)
    model0, sampler0, sigmas0 = sampling.setup_dual_clock_sampling(_model(), lifted0, 3, 12., 3.)
    _output0, previous, _core0, _ = sample_h16_pass2(
        model0, conditioning(), segment0, lifted0, spec0, context, plan,
        noise, sampler0, sigmas0,
    )
    segment1, spec1, _ = slice_chunked_source(source, plan, 1)
    lifted1, _ = lift_chunked_segment(segment1, spec1, context, plan)
    model1, sampler1, sigmas1 = sampling.setup_dual_clock_sampling(_model(), lifted1, 3, 12., 3.)
    piece, locked, transition = h16_window_piece(
        segment1, lifted1, spec1, context, plan, previous,
    )
    mask, audio_mask = piece["noise_mask"].unbind()
    assert locked > 0 and transition > 0
    assert mask[:, :, :max(1, locked // 2)].count_nonzero() == 0
    assert audio_mask.count_nonzero() == audio_mask.numel()
    baseline, *_ = sample_h16_pass2(
        model1, conditioning(), segment1, lifted1, spec1, context,
        plan, noise, sampler1, sigmas1, previous,
    )
    patched, runtime, _report, _ = bind_h16_eav(
        model1, sigmas1, segment1, lifted1, spec1, context, plan,
        previous, "preserve_first_pass",
        EAVConfig(mode=mode, start_video_progress=0.1,
                  g_hard_limit=3.0 if mode == "apply_exp" else 1.5),
    )
    output, result, _core1, _ = sample_h16_pass2(
        patched, conditioning(), segment1, lifted1, spec1, context,
        plan, noise, sampler1, sigmas1, previous,
    )
    if mode == "report_only":
        for actual, expected in zip(output["samples"].unbind(), baseline["samples"].unbind()):
            assert torch.equal(actual, expected)
    _audited, report_json = audit_h16_eav(result, spec1, runtime)
    report = json.loads(report_json)
    assert report["status"] == f"observed_{mode}", report
    assert report["completed_forwards"] == report["planned_forwards"] == 3
    assert report["selector_calls"] > 0
    if mode == "apply_exp":
        assert report["feta"]["active_forward_count"] > 0


def test_h16_effect_nodes_are_append_only():
    apply_schema = MiniMaxH3H16EAVApplyEXPT8.GET_NODE_INFO_V1()
    audit_schema = MiniMaxH3H16EAVAuditEXPT8.GET_NODE_INFO_V1()
    assert apply_schema["name"] == "MiniMaxH3H16EAVApplyEXPT8"
    assert audit_schema["name"] == "MiniMaxH3H16EAVAuditEXPT8"
    assert apply_schema["input"]["required"]["audio_output"][1]["default"] == "preserve_first_pass"
    assert audit_schema["output_name"] == ["cumulative_av_latent", "report_json"]


def test_h16_effect_candidate_has_seven_independent_config_apply_audit_chains():
    original_bytes = TEMPLATE.read_bytes()
    graph = add_eav_frontend(split_frontend(json.loads(original_bytes)))
    assert TEMPLATE.read_bytes() == original_bytes
    nodes = {node["id"]: node for node in graph["nodes"]}
    links = {link[0]: link for link in graph["links"]}
    windows = sorted((node for node in nodes.values()
                      if node["type"] == "MiniMaxH3H16Pass2WindowEXPT8"),
                     key=lambda node: node["id"])
    configs = [node for node in nodes.values() if node["type"] == "MiniMaxH3StageEAVConfigEXPT8"]
    applies = [node for node in nodes.values() if node["type"] == "MiniMaxH3H16EAVApplyEXPT8"]
    audits = [node for node in nodes.values() if node["type"] == "MiniMaxH3H16EAVAuditEXPT8"]
    assert len(windows) == len(configs) == len(applies) == len(audits) == WINDOW_COUNT
    for index, window in enumerate(windows):
        apply = applies[index]
        audit = audits[index]
        model_link = next(item["link"] for item in window["inputs"] if item["name"] == "model")
        assert links[model_link][1:3] == [apply["id"], 0]
        config_link = next(item["link"] for item in apply["inputs"] if item["name"] == "eav_config")
        assert links[config_link][1:3] == [configs[index]["id"], 0]
        previous_link = next(item["link"] for item in apply["inputs"] if item["name"] == "previous_result")
        assert previous_link is None if index == 0 else links[previous_link][1:3] == [windows[index - 1]["id"], 1]
        result_link = next(item["link"] for item in audit["inputs"] if item["name"] == "window_result")
        assert links[result_link][1:3] == [window["id"], 1]
    final_link = next(item["link"] for item in nodes[20]["inputs"] if item["name"] == "av_latent")
    assert links[final_link][1:3] == [audits[-1]["id"], 0]
    api = split_api(graph)
    assert sum(node["class_type"] == "MiniMaxH3H16EAVApplyEXPT8" for node in api.values()) == WINDOW_COUNT
    assert api["h16_report"]["inputs"]["source"] == [str(audits[-1]["id"]), 1]
    candidate = ROOT / Path(
        "artifacts/development/modular-sampling-m4-h16-eav-20260923/candidate-v1/"
        "H16_124F_Seven_Windows_Separate_PASS2_EAV_EXP.json"
    )
    assert json.loads(candidate.read_text(encoding="utf-8")) == graph
