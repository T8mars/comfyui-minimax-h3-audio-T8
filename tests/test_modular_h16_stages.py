"""S23 H16-3 one-window separation against the unchanged all-in-one path."""
import json
from pathlib import Path
import hashlib

import comfy.nested_tensor
import pytest
import torch

from h3_audio_t8_pkg import chunked_two_pass_upscale_advanced as legacy
from h3_audio_t8_pkg import nodes_h16_chunked_pass2 as old_h16
from h3_audio_t8_pkg.modular_sampling.chunked_source import slice_chunked_source
from h3_audio_t8_pkg.modular_sampling.chunked_stages import (
    lift_chunked_segment, prepare_chunked_pass2,
)
from h3_audio_t8_pkg.modular_sampling.h16_nodes import (
    MiniMaxH3H16Pass2PlanEXPT8, MiniMaxH3H16Pass2WindowEXPT8,
)
from h3_audio_t8_pkg.modular_sampling.h16_stages import (
    H16Pass2Result, build_h16_plan, sample_h16_pass2,
)
from test_chunked_two_pass_global_noise_advanced import _CountingCoordinateNoise
from test_modular_chunked_source import _latent
from tools.build_modular_h16_workflow import (
    TEMPLATE, WINDOW_COUNT, split_frontend, split_api,
)


def _h16_source():
    source = _latent(masked=True)
    video, audio = source["samples"].unbind()
    video_mask, audio_mask = source["noise_mask"].unbind()
    source["samples"] = comfy.nested_tensor.NestedTensor((
        video.repeat_interleave(8, -1).repeat_interleave(8, -2), audio,
    ))
    source["noise_mask"] = comfy.nested_tensor.NestedTensor((
        video_mask.repeat_interleave(8, -1).repeat_interleave(8, -2), audio_mask,
    ))
    return source


def _no_op_lift(piece, *_args):
    video, audio = piece["samples"].unbind()
    lifted = {"samples": comfy.nested_tensor.NestedTensor((video.clone(), audio))}
    if "noise_mask" in piece:
        video_mask, audio_mask = piece["noise_mask"].unbind()
        lifted["noise_mask"] = comfy.nested_tensor.NestedTensor((
            video_mask.clone(), audio_mask,
        ))
    return lifted, int(video.shape[-1]) * 16, int(video.shape[-2]) * 16, "{}"


def _native_like_piece(piece, _positive, _model, _noise, _sampler, _sigmas,
                       _negative, _cfg, *, prepared_noise=None):
    video, audio = piece["samples"].unbind()
    video_mask, audio_mask = piece["noise_mask"].unbind()
    assert prepared_noise is not None
    return comfy.nested_tensor.NestedTensor((
        video + video_mask * 0.1, audio + audio_mask * 0.5,
    ))


@pytest.mark.parametrize("audio_output", ["preserve_first_pass", "refined_exp"])
def test_h16_explicit_windows_match_old_video_and_audio(monkeypatch, audio_output):
    monkeypatch.setattr(legacy, "learned_upscale_h3_av_latent", _no_op_lift)
    monkeypatch.setattr(legacy, "rebind_dual_clock_sampler", lambda _model, _piece, sampler: sampler)
    monkeypatch.setattr(legacy, "sample_piece", _native_like_piece)
    source = _h16_source()
    plan, _ = build_h16_plan(source)
    assert plan["schema"] == legacy.PLAN_SCHEMA_MASKED_LOW_SIGMA_V4
    assert plan["spatial_strategy"] == "full_frame_safe"
    old_noise = _CountingCoordinateNoise()
    args = (object(), [[torch.zeros(1), {}]], source, old_noise, object(),
            torch.tensor([0.5, 0.0]), plan)
    if audio_output == "refined_exp":
        old_output, _ = old_h16._run_refined_audio(*args, None, 1.0)
    else:
        old_output, _ = legacy.execute_chunked_two_pass_upscale(*args)
    new_noise = _CountingCoordinateNoise()
    context, _ = prepare_chunked_pass2(source, plan, new_noise)
    previous = None
    reports = []
    video, _ = source["samples"].unbind()
    segments, _ = legacy.compute_temporal_segments(video.shape[2], 34, 17)
    for index in range(len(segments)):
        segment, spec, _ = slice_chunked_source(source, plan, index)
        lifted, _ = lift_chunked_segment(segment, spec, context, plan)
        output, previous, core, report_json = sample_h16_pass2(
            object(), [[torch.zeros(1), {}]], segment, lifted, spec,
            context, plan, new_noise, object(), torch.tensor([0.5, 0.0]),
            previous, audio_output=audio_output,
        )
        assert core.index == index
        reports.append(json.loads(report_json))
    assert type(previous) is H16Pass2Result
    assert previous.core_result.index + 1 == previous.core_result.count == len(segments)
    actual_video, actual_audio = output["samples"].unbind()
    old_video, old_audio = old_output["samples"].unbind()
    assert torch.equal(actual_video, old_video)
    assert torch.equal(actual_audio, old_audio)
    assert reports[-1]["audio_output"] == audio_output
    if audio_output == "refined_exp":
        assert reports[-1]["refined_audio_chunks"] == len(segments)
        assert len(previous.audio_chunks) == len(segments)
        assert reports[-1]["audio_merge"] == "absolute_frame_rescale_crossfade_energy_gate"
    else:
        assert actual_audio is source["samples"].tensors[1]
        assert reports[-1]["audio_merge"] == "preserve_first_pass"


def test_h16_audio_policy_mismatch_and_mutated_capture_fail_before_next_window(monkeypatch):
    monkeypatch.setattr(legacy, "learned_upscale_h3_av_latent", _no_op_lift)
    monkeypatch.setattr(legacy, "rebind_dual_clock_sampler", lambda _model, _piece, sampler: sampler)
    monkeypatch.setattr(legacy, "sample_piece", _native_like_piece)
    source = _h16_source()
    plan, _ = build_h16_plan(source)
    noise = _CountingCoordinateNoise()
    context, _ = prepare_chunked_pass2(source, plan, noise)
    segment0, spec0, _ = slice_chunked_source(source, plan, 0)
    lifted0, _ = lift_chunked_segment(segment0, spec0, context, plan)
    _, previous, _, _ = sample_h16_pass2(
        object(), [], segment0, lifted0, spec0, context, plan, noise,
        object(), torch.tensor([0.5, 0.0]), audio_output="refined_exp",
    )
    segment1, spec1, _ = slice_chunked_source(source, plan, 1)
    lifted1, _ = lift_chunked_segment(segment1, spec1, context, plan)
    args = (object(), [], segment1, lifted1, spec1, context, plan, noise,
            object(), torch.tensor([0.5, 0.0]), previous)
    with pytest.raises(ValueError, match="different audio policy"):
        sample_h16_pass2(*args, audio_output="preserve_first_pass")
    previous.audio_chunks[0][..., 0] += 1
    with pytest.raises(ValueError, match="changed"):
        sample_h16_pass2(*args, audio_output="refined_exp")


def test_h16_explicit_refined_merge_failure_falls_back_without_resampling(monkeypatch):
    monkeypatch.setattr(legacy, "learned_upscale_h3_av_latent", _no_op_lift)
    monkeypatch.setattr(legacy, "rebind_dual_clock_sampler", lambda _model, _piece, sampler: sampler)
    calls = []

    def counted_piece(*args, **kwargs):
        calls.append(1)
        return _native_like_piece(*args, **kwargs)

    monkeypatch.setattr(legacy, "sample_piece", counted_piece)
    source = _h16_source()
    plan, _ = build_h16_plan(source)
    noise = _CountingCoordinateNoise()
    context, _ = prepare_chunked_pass2(source, plan, noise)
    monkeypatch.setattr(old_h16, "_merge_audio_segments",
                        lambda *_args: (_ for _ in ()).throw(RuntimeError("test merge fault")))
    previous = None
    for index in range(2):
        segment, spec, _ = slice_chunked_source(source, plan, index)
        lifted, _ = lift_chunked_segment(segment, spec, context, plan)
        output, previous, _core, report_json = sample_h16_pass2(
            object(), [], segment, lifted, spec, context, plan, noise,
            object(), torch.tensor([0.5, 0.0]), previous,
            audio_output="refined_exp",
        )
    assert len(calls) == 2
    assert output["samples"].tensors[1] is source["samples"].tensors[1]
    assert json.loads(report_json)["audio_merge"] == "fallback_preserve_first_pass"
    assert "test merge fault" in json.loads(report_json)["audio_merge_error"]


def test_h16_public_node_schemas_are_additive_and_old_schema_unchanged():
    old = old_h16.DeciiaChunkedPass2Sampler.GET_NODE_INFO_V1()
    assert old["output_name"] == ["output", "denoised_output"]
    assert old["input"]["required"]["audio_output"][1]["default"] == "preserve_first_pass"
    plan = MiniMaxH3H16Pass2PlanEXPT8.GET_NODE_INFO_V1()
    stage = MiniMaxH3H16Pass2WindowEXPT8.GET_NODE_INFO_V1()
    assert plan["name"] == "MiniMaxH3H16Pass2PlanEXPT8"
    assert stage["name"] == "MiniMaxH3H16Pass2WindowEXPT8"
    assert stage["output_name"] == ["cumulative_av_latent", "window_result",
                                    "core_segment_result", "report_json"]
    assert stage["input"]["required"]["audio_output"][1]["default"] == "preserve_first_pass"


def test_fixed_h16_candidate_is_seven_real_windows_and_high_lora_is_independent():
    source_bytes = TEMPLATE.read_bytes()
    original = json.loads(source_bytes)
    graph = split_frontend(original)
    assert TEMPLATE.read_bytes() == source_bytes
    assert hashlib.sha256(TEMPLATE.read_bytes()).hexdigest() == hashlib.sha256(source_bytes).hexdigest()
    assert not any(node["type"] == "DeciiaChunkedPass2Sampler" for node in graph["nodes"])
    nodes = {node["id"]: node for node in graph["nodes"]}
    links = {link[0]: link for link in graph["links"]}
    stages = sorted((node for node in nodes.values()
                     if node["type"] == "MiniMaxH3H16Pass2WindowEXPT8"),
                    key=lambda node: node["id"])
    assert len(stages) == WINDOW_COUNT
    assert len(legacy.compute_temporal_segments(37, 34, 17)[0]) == WINDOW_COUNT
    assert all(node["widgets_values"] == ["refined_exp", 1.0] for node in stages)
    for index, stage in enumerate(stages):
        previous = next(item for item in stage["inputs"] if item["name"] == "previous_result")
        if index == 0:
            assert previous["link"] is None
        else:
            assert links[previous["link"]][1:3] == [stages[index - 1]["id"], 1]
        source = next(item for item in stage["inputs"] if item["name"] == "source_segment")
        assert nodes[links[source["link"]][1]]["type"] == "MiniMaxH3ChunkedSourceSegmentEXPT8"
    final_link = next(item for item in nodes[20]["inputs"] if item["name"] == "av_latent")
    assert links[final_link["link"]][1:3] == [stages[-1]["id"], 0]
    mixer_input = next(item for item in nodes[16]["inputs"] if item["name"] == "model")
    high_lora_id = links[mixer_input["link"]][1]
    assert high_lora_id != 5 and nodes[high_lora_id]["type"] == nodes[5]["type"]
    assert nodes[high_lora_id]["widgets_values"] == nodes[5]["widgets_values"]
    api = split_api(graph)
    assert sum(node["class_type"] == "MiniMaxH3H16Pass2WindowEXPT8"
               for node in api.values()) == WINDOW_COUNT
    assert api["h16_report"]["inputs"]["source"] == [str(stages[-1]["id"]), 3]
    candidate = Path(__file__).resolve().parents[1] / (
        "artifacts/development/modular-sampling-m4-h16-20260923/candidate-v1/"
        "H16_124F_Seven_Windows_Separate_PASS2_EXP.json"
    )
    assert json.loads(candidate.read_text(encoding="utf-8")) == graph


def test_h16_fixed_builder_rejects_changed_source_length():
    original = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    next(node for node in original["nodes"] if node["id"] == 7)["widgets_values"][3] = 73
    with pytest.raises(ValueError, match="frozen"):
        split_frontend(original)
