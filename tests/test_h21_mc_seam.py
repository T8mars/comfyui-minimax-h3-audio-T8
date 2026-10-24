"""Actual AV numerical contracts, not latent quality/performance claims."""
import json
import asyncio
from types import SimpleNamespace

import comfy.model_management as mm
from comfy.nested_tensor import NestedTensor
import pytest
import torch

from h3_audio_t8_pkg import mc_seam_experimental as mc
from h3_audio_t8_pkg.nodes_mc_seam_experimental import (
    MiniMaxH3MCJoinMatchEXPT8, MiniMaxH3MCRefineBlendEXPT8, MiniMaxH3MCSigmaTailEXPT8,
)


def latent(batch=1, count=25, kind=NestedTensor, seed=2110217):
    generator = torch.Generator().manual_seed(seed)
    video = torch.randn(batch, 3, count, 4, 5, generator=generator)
    sound = torch.randn(batch, 4, 2, 83, generator=generator)
    return {"samples": kind((video, sound)), "noise_mask": object(),
            "batch_index": [4], "custom_metadata": {"foreign": object()}}


def parts(value):
    sample = value["samples"]
    return tuple(sample.unbind()) if getattr(sample, "is_nested", False) else tuple(sample)


def preserved(source, output):
    for key in source:
        if key != "samples":
            assert output[key] is source[key]


@pytest.mark.parametrize("steps", [0, 1, 2, 3, 4, 5, 6, 7, 12, 10000, 1000000000000000000])
def test_exact_cycle_has_no_search_size_limit(steps):
    assert mc.steps_for_frames(mc.pixel_frames(steps)) == steps
    if steps < 15:
        assert mc.pixel_frames(steps) == sum((1, 4, 4, 4, 4)[k % 5] for k in range(steps))
    assert mc.steps_for_frames(22) == 7


@pytest.mark.parametrize("frames", [-1, 2, 3, 4, 6, 44, 22.9])
def test_actual_non_native_boundary_is_clear_error(frames):
    with pytest.raises(ValueError, match="native|integer|>="):
        mc.steps_for_frames(frames)


@pytest.mark.parametrize("operator", [mc.refine_blend, mc.join_match, mc.sigma_tail])
def test_disabled_is_original_object_without_reading_unused_inputs(operator):
    value = object()
    output, report = operator(value)
    assert output is value and report["source_identity"]
    assert report["own_diffusion_nfe"] == 0 and not report["sampler_called"]
    assert not report["quality_accepted"] and not report["portable_cache_identity"]


@pytest.mark.parametrize("kind", [list, tuple, NestedTensor])
def test_refine_native_weights_exact_pin_original_audio_and_all_metadata(kind):
    source, refined = latent(kind=kind), latent(kind=kind, seed=2110218)
    before, sound = parts(source)
    target, _ = parts(refined)
    old, target_old = before.clone(), target.clone()
    output, report = mc.refine_blend(source, refined, enabled=True)
    actual, actual_sound = parts(output)
    expected = before.clone()
    weights = []
    for k in range(before.shape[2]):
        frame = sum(mc.FRAME_PER_TOKEN[j % 5] for j in range(k))
        weight = 0 if frame < 22 else min(1, max(0, (frame - 22) / 22))
        weights.append(weight)
        if weight == 1:
            expected[:, :, k] = target[:, :, k]
        elif weight > 0:
            expected[:, :, k] = (1 - weight) * before[:, :, k] + weight * target[:, :, k]
    assert torch.equal(actual, expected)
    assert torch.equal(actual[:, :, :7], before[:, :, :7])
    assert torch.equal(actual[:, :, :7].contiguous().view(torch.uint8), before[:, :, :7].contiguous().view(torch.uint8))
    assert report["held_native_steps"] == 7 and report["zero_weight_steps"] == 8
    assert report["weights"] == weights
    assert actual_sound is sound and not report["audio_changed"]
    assert type(output["samples"]) is kind
    preserved(source, output)
    assert torch.equal(before, old) and torch.equal(target, target_old)


def test_nested_container_extra_state_survives():
    class UnknownContainer(NestedTensor):
        pass
    source, refined = latent(kind=UnknownContainer), latent(kind=UnknownContainer, seed=13)
    source["samples"].foreign = object()
    output, _ = mc.refine_blend(source, refined, enabled=True)
    assert type(output["samples"]) is UnknownContainer
    assert output["samples"].foreign is source["samples"].foreign
    assert output["samples"].tensors is not source["samples"].tensors


def test_first_clip_explicit_refined_audio_and_original_metadata():
    source, refined = latent(), latent(seed=13)
    output, report = mc.refine_blend(source, refined, enabled=True, clip_index=0, hold_frames=44, audio="refined")
    video, sound = parts(output)
    target, target_sound = parts(refined)
    assert torch.equal(video, target) and sound is target_sound
    assert report["first_clip"] and report["hold_frames_used"] == 0
    assert report["audio_changed"] and report["explicit_upstream_refine_cost_not_included_in_own_nfe"]
    preserved(source, output)


def test_hard_cut_refine_zero_ramp_is_explicit_and_pin_exact():
    source, refined = latent(), latent(seed=18)
    output, _ = mc.refine_blend(source, refined, enabled=True, blend_frames=0)
    actual, _ = parts(output)
    original, _ = parts(source)
    target, _ = parts(refined)
    assert torch.equal(actual[:, :, :7], original[:, :, :7])
    assert torch.equal(actual[:, :, 7:], target[:, :, 7:])


@pytest.mark.parametrize("variant", ["source_whole_clip_exp", "keep_pin_T8_exp"])
def test_join_mean_only_has_independent_native_phase_and_explicit_pin_policy(variant):
    source, previous = latent(batch=2), latent(batch=2, count=20, seed=15)
    original, sound = parts(source)
    prior, _ = parts(previous)
    original_copy, previous_copy = original.clone(), prior.clone()
    output, report = mc.join_match(source, previous, enabled=True, variant=variant,
                                  tuning="manual", min_excess=0, strength=1.7,
                                  match_spread=False, match_detail=False)
    actual, actual_sound = parts(output)
    expected = []
    for index in range(2):
        current, old = original[index:index + 1], prior[index:index + 1]
        mean = current[0, :, 9].mean((-2, -1))
        step = mean - old[0, :, 19].mean((-2, -1))
        target = mean - 1.7 * step
        expected.append(current - (mean - target).view(1, -1, 1, 1, 1))
    expected = torch.cat(expected)
    if variant == "keep_pin_T8_exp":
        expected[:, :, :7] = original[:, :, :7]
    assert torch.equal(actual, expected)
    assert torch.equal(actual[:, :, :7], original[:, :, :7]) == (variant == "keep_pin_T8_exp")
    assert (report["previous_step"], report["measured_step"], report["next_same_phase_step"]) == (19, 9, 14)
    assert report["batch_independent_extension"] and len(report["per_batch"]) == 2
    assert actual_sound is sound and not report["audio_changed"]
    assert torch.equal(original, original_copy) and torch.equal(prior, previous_copy)
    preserved(source, output)


@pytest.mark.parametrize("tuning", ["auto", "manual"])
def test_join_full_detail_spread_has_per_batch_independent_result(tuning):
    source, previous = latent(batch=2), latent(batch=2, seed=19)
    output, report = mc.join_match(source, previous, enabled=True, tuning=tuning)
    for index in range(2):
        single = {"samples": NestedTensor(tuple(x[index:index + 1] for x in parts(source)))}
        single_previous = {"samples": NestedTensor(tuple(x[index:index + 1] for x in parts(previous)))}
        expected, note = mc.join_match(single, single_previous, enabled=True, tuning=tuning)
        assert torch.equal(parts(output)[0][index:index + 1], parts(expected)[0])
        assert report["per_batch"][index] == note["per_batch"][0]
    assert torch.equal(parts(output)[0][:, :, :7], parts(source)[0][:, :, :7])


def test_join_zero_first_or_short_clip_are_same_object():
    source, previous = latent(count=12), latent(count=20)
    for arguments in ({"strength": 0}, {"context_length": 0}, {"context_latent": None}, {}):
        kwargs = {"context_latent": previous, **arguments}
        output, report = mc.join_match(source, enabled=True, **kwargs)
        assert output is source and report["source_identity"]


@pytest.mark.parametrize("strength", [-.75, .5, 5.0])
def test_sigma_source_formula_actual_dtype_prefix_zero_and_input_unchanged(strength):
    source = torch.tensor([1., .95, .9, .8, .6, .4, .2, 0.], dtype=torch.float64)
    old = source.clone()
    output, report = mc.sigma_tail(source, enabled=True, tail_steps=3, strength=strength)
    expected = source.clone()
    anchor = float(source[3])
    for index in (4, 5, 6):
        expected[index] = anchor * (float(source[index]) / anchor) ** (1 / (1 + strength))
    assert torch.equal(output, expected)
    assert torch.equal(output[:4], source[:4]) and output[-1] == source[-1] == 0
    assert torch.equal(source, old) and output is not source
    assert output.dtype == source.dtype and report["own_diffusion_nfe"] == 0
    assert report["strict_order_in_actual_dtype"] and report["explicit_modified_schedule_not_trained_table"]
    assert not report["trained_sigma_table_modified"]


def test_sigma_zero_and_short_tables_identity_and_no_fixed_tail_size_cap():
    source = torch.tensor([1., .7, 0.])
    assert mc.sigma_tail(source, enabled=True, strength=0)[0] is source
    assert mc.sigma_tail(source, enabled=True, tail_steps=0)[0] is source
    only = torch.tensor([0.])
    assert mc.sigma_tail(only, enabled=True)[0] is only
    output, report = mc.sigma_tail(source, enabled=True, tail_steps=1000000)
    assert output[0] == 1 and output[-1] == 0 and report["changed_indices"] == [1]


@pytest.mark.parametrize("source,arguments,match", [
    (torch.tensor([1., float("nan"), 0.]), {}, "nonfinite"),
    (torch.tensor([1., -.1, 0.]), {}, "nonnegative"),
    (torch.zeros(2, 2), {}, "one-dimensional"),
    (torch.tensor([1., .5, 0.]), {"strength": -1}, "zero denominator"),
    (torch.tensor([1., .5, 0., 0.]), {"strength": -2}, "undefined"),
    (torch.tensor([1., .5, 0.]), {"strength": float("inf")}, "finite"),
])
def test_real_bad_sigma_math_still_errors(source, arguments, match):
    with pytest.raises(ValueError, match=match):
        mc.sigma_tail(source, enabled=True, **arguments)


def test_real_bad_AV_dimensions_dtype_and_nan_are_errors_without_input_writes():
    source, refined = latent(), latent(seed=22)
    before = parts(source)[0].clone()
    target, sound = parts(refined)
    for bad, match in (({"samples": torch.zeros(2, 3, 4, 5, 6)}, "explicit video/audio"),
                       ({"samples": [target, sound, sound]}, "exactly two"),
                       ({"samples": [target, sound[..., :1, :]]}, "H3 shape"),
                       ({"samples": [target.to(torch.int32), sound]}, "floating"),
                       ({"samples": [target[:, :, :-1], sound]}, "shape")):
        with pytest.raises(ValueError, match=match):
            mc.refine_blend(source, bad, enabled=True)
    target[0, 0, -1, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        mc.refine_blend(source, refined, enabled=True)
    assert torch.equal(parts(source)[0], before)
    with pytest.raises(ValueError, match="explicit upstream"):
        mc.refine_blend(source, enabled=True)
    with pytest.raises(ValueError, match="more native"):
        mc.refine_blend(source, latent(), enabled=True, hold_frames=mc.pixel_frames(100))
    with pytest.raises(ValueError, match="spatial sizes"):
        other = latent()
        other["samples"].tensors[0] = parts(other)[0][..., :-1]
        mc.join_match(source, other, enabled=True)


def test_actual_Core_cancel_once_preserves_inputs_and_next_request(monkeypatch):
    source, refined = latent(), latent(seed=23)
    before = parts(source)[0].clone()
    calls = []
    def cancel():
        calls.append(True)
        raise mm.InterruptProcessingException()
    with monkeypatch.context() as selected:
        selected.setattr(mm, "throw_exception_if_processing_interrupted", cancel)
        with pytest.raises(mm.InterruptProcessingException):
            mc.refine_blend(source, refined, enabled=True)
    assert calls == [True] and torch.equal(parts(source)[0], before)
    output, report = mc.refine_blend(source, refined, enabled=True)
    assert report["status"] == "computed_unreviewed" and output is not source
    assert not torch.cuda.is_initialized()


def test_actual_Core_cancel_after_partial_local_refine_keeps_inputs_exact(monkeypatch):
    source, refined = latent(), latent(seed=24)
    before, target = parts(source)[0].clone(), parts(refined)[0].clone()
    expected, _ = mc.refine_blend(source, refined, enabled=True)
    calls = []
    original = mm.throw_exception_if_processing_interrupted
    def cancel():
        calls.append(True)
        if len(calls) == 17:
            raise mm.InterruptProcessingException()
        original()
    with monkeypatch.context() as selected:
        selected.setattr(mm, "throw_exception_if_processing_interrupted", cancel)
        with pytest.raises(mm.InterruptProcessingException):
            mc.refine_blend(source, refined, enabled=True)
    assert len(calls) == 17
    assert torch.equal(parts(source)[0], before) and torch.equal(parts(refined)[0], target)
    actual, _ = mc.refine_blend(source, refined, enabled=True)
    assert torch.equal(parts(actual)[0], parts(expected)[0])


def test_actual_Core_schemas_lazy_disabled_and_execute_reports():
    source, refined = latent(), latent(seed=19)
    for node in (MiniMaxH3MCRefineBlendEXPT8, MiniMaxH3MCJoinMatchEXPT8, MiniMaxH3MCSigmaTailEXPT8):
        schema = node.GET_SCHEMA()
        assert schema.node_id == node.__name__ and schema.is_experimental
        enabled = next(value for value in schema.inputs if value.id == "enabled")
        assert enabled.default is False
    blend = MiniMaxH3MCRefineBlendEXPT8
    assert blend.check_lazy_status(source, enabled=False) == []
    assert blend.check_lazy_status(source, enabled=True) == ["refined"]
    assert blend.check_lazy_status(source, enabled=True, refined=refined) == []
    disabled = blend.execute(source)
    assert disabled.result[0] is source and json.loads(disabled.result[1])["status"] == "disabled_identity"
    enabled = blend.execute(source, enabled=True, refined=refined)
    assert torch.equal(parts(enabled.result[0])[0][:, :, :7], parts(source)[0][:, :, :7])
    join = MiniMaxH3MCJoinMatchEXPT8
    assert join.check_lazy_status(source) == []
    assert join.check_lazy_status(source, enabled=True, strength=0) == []
    assert join.check_lazy_status(source, enabled=True) == ["context_latent"]
    assert json.loads(join.execute(source).result[1])["audio_changed"] is False
    sigma = MiniMaxH3MCSigmaTailEXPT8
    table = torch.tensor([1., .6, .3, 0.])
    assert sigma.execute(table).result[0] is table
    assert sigma.execute(table, enabled=True).result[0] is not table


def test_lazy_optional_unwired_is_not_requested_from_real_prompt(monkeypatch):
    prompt = {"41": {"inputs": {"enabled": True}}}
    for cls in (MiniMaxH3MCRefineBlendEXPT8, MiniMaxH3MCJoinMatchEXPT8):
        monkeypatch.setattr(cls, "hidden", SimpleNamespace(prompt=prompt, unique_id="41"))
        assert cls.check_lazy_status(latent(), enabled=True) == []
    assert mc.join_match(latent(), enabled=True)[1]["source_identity"]
    prompt["41"]["inputs"]["refined"] = ["40", 0]
    assert MiniMaxH3MCRefineBlendEXPT8.check_lazy_status(latent(), enabled=True) == ["refined"]
    prompt["41"]["inputs"]["context_latent"] = ["40", 0]
    assert MiniMaxH3MCJoinMatchEXPT8.check_lazy_status(latent(), enabled=True) == ["context_latent"]


@pytest.mark.parametrize("enabled", [False, True])
def test_actual_Core_validates_explicit_three_operator_graph_without_registration(monkeypatch, enabled):
    import execution
    import nodes as core_nodes
    from comfy_api.latest import io

    class Inputs(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(node_id="H21MCOnlyTestInputs", inputs=[], outputs=[
                io.Latent.Output("original"), io.Latent.Output("refined"),
                io.Latent.Output("previous"), io.Sigmas.Output("sigmas")])
        @classmethod
        def execute(cls):
            return io.NodeOutput(latent(), latent(seed=21), latent(seed=22), torch.tensor([1., .6, .3, 0.]))

    class Reports(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(node_id="H21MCOnlyTestReports", is_output_node=True,
                             inputs=[io.String.Input("blend"), io.String.Input("join"), io.String.Input("tail")], outputs=[])
        @classmethod
        def execute(cls, blend, join, tail):
            return io.NodeOutput(ui={"text": [blend, join, tail]})

    classes = [MiniMaxH3MCRefineBlendEXPT8, MiniMaxH3MCJoinMatchEXPT8, MiniMaxH3MCSigmaTailEXPT8]
    for cls in [Inputs, Reports, *classes]:
        monkeypatch.setitem(core_nodes.NODE_CLASS_MAPPINGS, cls.GET_SCHEMA().node_id, cls)
    graph = {"1": {"class_type": Inputs.GET_SCHEMA().node_id, "inputs": {}}}
    for node_id, cls in zip(("2", "3", "4"), classes):
        graph[node_id] = {"class_type": cls.__name__, "inputs": {
            field.id: field.default for field in cls.GET_SCHEMA().inputs if not field.optional and hasattr(field, "default")}}
        graph[node_id]["inputs"]["enabled"] = enabled
    graph["2"]["inputs"].update(original=["1", 0], refined=["1", 1])
    graph["3"]["inputs"].update(samples=["2", 0], context_latent=["1", 2])
    graph["4"]["inputs"].update(sigmas=["1", 3])
    graph["5"] = {"class_type": Reports.GET_SCHEMA().node_id,
                  "inputs": {"blend": ["2", 1], "join": ["3", 1], "tail": ["4", 1]}}
    result = asyncio.run(execution.validate_prompt("h21-mc-separate-operators", graph, None))
    assert result[0], result
    assert result[2] == ["5"] and not result[3]
