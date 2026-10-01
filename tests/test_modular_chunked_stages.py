"""Real v1-v4 legacy math path split at source / learned lift / PASS2."""
import json

import comfy.nested_tensor
import pytest
import torch

from h3_audio_t8_pkg import chunked_two_pass_upscale_advanced as old
from h3_audio_t8_pkg.modular_sampling.chunked_source import slice_chunked_source
from h3_audio_t8_pkg.modular_sampling.chunked_stages import (
    lift_chunked_segment, prepare_chunked_pass2, sample_chunked_pass2,
)
from test_chunked_two_pass_global_noise_advanced import _plan, _CountingCoordinateNoise
from test_modular_chunked_source import _latent


def _fake_lift(chunk, *_args):
    video, audio = chunk["samples"].unbind()
    lifted = {"samples": comfy.nested_tensor.NestedTensor((
        video.repeat_interleave(2, -1).repeat_interleave(2, -2), audio,
    ))}
    if "noise_mask" in chunk:
        source_mask, audio_mask = chunk["noise_mask"].unbind()
        lifted["noise_mask"] = comfy.nested_tensor.NestedTensor((
            source_mask.repeat_interleave(2, -1).repeat_interleave(2, -2), audio_mask,
        ))
    return lifted, 64, 64, "{}"


def _fake_spatial(video, audio, *_args, **kwargs):
    value = video + 0.25
    if kwargs["chunk_noise_video"] is not None:
        value = value + kwargs["chunk_noise_video"] * 0.001
    if kwargs["chunk_temporal_mask"] is not None:
        value = value + kwargs["chunk_temporal_mask"] * 0.01
    if kwargs["chunk_inherited_video_mask"] is not None:
        value = value + kwargs["chunk_inherited_video_mask"] * 0.02
    return value, {"audio_shape": list(audio.shape)}


@pytest.mark.parametrize("builder,temporal_strategy", [
    (old.build_chunked_two_pass_plan, None),
    (old.build_chunked_two_pass_global_noise_plan, "full_clip_safe"),
    (old.build_chunked_two_pass_low_sigma_plan, "full_clip_safe"),
    (old.build_chunked_two_pass_masked_low_sigma_plan, "full_clip_safe"),
    (old.build_chunked_two_pass_masked_low_sigma_plan, "guarded_overlap_exp"),
])
def test_explicit_lift_and_each_pass2_match_legacy(monkeypatch, builder, temporal_strategy):
    monkeypatch.setattr(old, "learned_upscale_h3_av_latent", _fake_lift)
    monkeypatch.setattr(old, "_spatial_resample", _fake_spatial)
    plan = _plan(builder, strategy="full_frame_safe", temporal_strategy=temporal_strategy)
    source = _latent(masked=builder is old.build_chunked_two_pass_masked_low_sigma_plan)
    old_noise = _CountingCoordinateNoise()
    old_output, old_report = old.execute_chunked_two_pass_upscale(
        object(), [[torch.zeros(1), {}]], source, old_noise, object(),
        torch.tensor([1., 0.]), plan,
    )
    new_noise = _CountingCoordinateNoise()
    context, context_report = prepare_chunked_pass2(source, plan, new_noise)
    assert old_noise.calls == new_noise.calls == (0 if builder is old.build_chunked_two_pass_plan else 1)
    assert json.loads(context_report)["sampled"] is False
    previous = None
    for index in range(json.loads(old_report)["segment_count"]):
        chunk, spec, _ = slice_chunked_source(source, plan, index)
        lifted, lift_report = lift_chunked_segment(chunk, spec, context, plan)
        assert json.loads(lift_report)["sampled"] is False
        new_output, previous, stage_report = sample_chunked_pass2(
            object(), [[torch.zeros(1), {}]], chunk, lifted, spec, context,
            plan, new_noise, object(), torch.tensor([1., 0.]), previous,
        )
        assert json.loads(stage_report)["completed"] == (index + 1 == spec.count)
    old_video, old_audio = old_output["samples"].unbind()
    new_video, new_audio = new_output["samples"].unbind()
    assert torch.equal(new_video, old_video)
    assert new_audio is old_audio is source["samples"].tensors[1]
    assert previous.index + 1 == previous.count


def test_chunked_pass2_rejects_stale_source_and_out_of_order(monkeypatch):
    monkeypatch.setattr(old, "learned_upscale_h3_av_latent", _fake_lift)
    monkeypatch.setattr(old, "_spatial_resample", _fake_spatial)
    plan = _plan(old.build_chunked_two_pass_plan, strategy="full_frame_safe")
    source = _latent()
    noise = _CountingCoordinateNoise()
    context, _ = prepare_chunked_pass2(source, plan, noise)
    chunk, spec, _ = slice_chunked_source(source, plan, 1)
    lifted, _ = lift_chunked_segment(chunk, spec, context, plan)
    args = (object(), [[torch.zeros(1), {}]], chunk, lifted, spec,
            context, plan, noise, object(), torch.tensor([1., 0.]))
    with pytest.raises(ValueError, match="out of order"):
        sample_chunked_pass2(*args)
    source["samples"].tensors[0][0, 0, 0, 0, 0] += 1
    with pytest.raises(ValueError, match="mutated"):
        lift_chunked_segment(chunk, spec, context, plan)


def test_chunked_pass2_rejects_changed_mask_and_noise_seed(monkeypatch):
    monkeypatch.setattr(old, "learned_upscale_h3_av_latent", _fake_lift)
    monkeypatch.setattr(old, "_spatial_resample", _fake_spatial)
    plan = _plan(old.build_chunked_two_pass_masked_low_sigma_plan, strategy="full_frame_safe")
    source = _latent(masked=True)
    noise = _CountingCoordinateNoise()
    context, _ = prepare_chunked_pass2(source, plan, noise)
    chunk, spec, _ = slice_chunked_source(source, plan, 0)
    lifted, _ = lift_chunked_segment(chunk, spec, context, plan)
    other_noise = _CountingCoordinateNoise()
    other_noise.seed += 1
    args = (object(), [[torch.zeros(1), {}]], chunk, lifted, spec,
            context, plan, other_noise, object(), torch.tensor([1., 0.]))
    with pytest.raises(ValueError, match="seed differs"):
        sample_chunked_pass2(*args)
    lifted_mask, _ = lifted["noise_mask"].unbind()
    lifted_mask[0, 0, 0, 0, 0] = 0.9
    with pytest.raises(ValueError, match="changed inherited"):
        sample_chunked_pass2(*args[:7], noise, *args[8:])


def test_chunked_v1_previous_result_content_is_checked(monkeypatch):
    monkeypatch.setattr(old, "learned_upscale_h3_av_latent", _fake_lift)
    monkeypatch.setattr(old, "_spatial_resample", _fake_spatial)
    plan = _plan(old.build_chunked_two_pass_plan, strategy="full_frame_safe")
    source = _latent()
    prepare_noise = _CountingCoordinateNoise()
    context, _ = prepare_chunked_pass2(source, plan, prepare_noise)
    first, first_spec, _ = slice_chunked_source(source, plan, 0)
    first_lift, _ = lift_chunked_segment(first, first_spec, context, plan)
    _output, result, _ = sample_chunked_pass2(
        object(), [], first, first_lift, first_spec, context,
        plan, prepare_noise, object(), torch.tensor([1., 0.]),
    )
    second, second_spec, _ = slice_chunked_source(source, plan, 1)
    second_lift, _ = lift_chunked_segment(second, second_spec, context, plan)
    result.output_latent["samples"].tensors[0][0, 0, 0, 0, 0] += 1
    with pytest.raises(ValueError, match="stale"):
        sample_chunked_pass2(
            object(), [], second, second_lift, second_spec, context,
            plan, prepare_noise, object(), torch.tensor([1., 0.]), result,
        )


@pytest.mark.parametrize("builder", [
    old.build_chunked_two_pass_global_noise_plan,
    old.build_chunked_two_pass_low_sigma_plan,
    old.build_chunked_two_pass_masked_low_sigma_plan,
])
def test_split_pass2_runs_original_spatial_tile_loop(monkeypatch, builder):
    monkeypatch.setattr(old, "learned_upscale_h3_av_latent", _fake_lift)
    monkeypatch.setattr(old, "rebind_dual_clock_sampler", lambda _model, _piece, sampler: sampler)
    captures = []

    def fake_sample_piece(piece, _positive, _model, _noise, _sampler, _sigmas,
                          _negative, _cfg, *, prepared_noise=None):
        video, audio = piece["samples"].unbind()
        video_mask, audio_mask = piece["noise_mask"].unbind()
        noise_video, noise_audio = prepared_noise.unbind()
        captures.append((video_mask.clone(), audio_mask.clone(), noise_audio.clone()))
        return comfy.nested_tensor.NestedTensor((video + video_mask * 0.1 + noise_video * 0.001, audio))

    monkeypatch.setattr(old, "sample_piece", fake_sample_piece)
    plan = _plan(builder, strategy="independent_tiles_exp")
    source = _latent(masked=builder is old.build_chunked_two_pass_masked_low_sigma_plan)
    old_noise = _CountingCoordinateNoise()
    old_output, _ = old.execute_chunked_two_pass_upscale(
        object(), [], source, old_noise, object(), torch.tensor([0.5, 0.0]), plan,
    )
    old_calls = captures.copy()
    captures.clear()
    new_noise = _CountingCoordinateNoise()
    context, _ = prepare_chunked_pass2(source, plan, new_noise)
    chunk, spec, _ = slice_chunked_source(source, plan, 0)
    lifted, _ = lift_chunked_segment(chunk, spec, context, plan)
    output, result, _ = sample_chunked_pass2(
        object(), [], chunk, lifted, spec, context, plan, new_noise, object(),
        torch.tensor([0.5, 0.0]),
    )
    assert result.index == 0 and result.count == 1
    assert len(captures) == len(old_calls) == 4
    for actual, expected in zip(captures, old_calls):
        for tensor, reference in zip(actual, expected):
            assert torch.equal(tensor, reference)
    assert torch.equal(output["samples"].tensors[0], old_output["samples"].tensors[0])
    assert output["samples"].tensors[1] is source["samples"].tensors[1]
