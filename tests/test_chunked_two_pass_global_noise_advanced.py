from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
import torch

from h3_audio_t8_pkg import chunked_two_pass_upscale_advanced as chunked
from h3_audio_t8_pkg.nodes import comfy_entrypoint
from h3_audio_t8_pkg.nodes_chunked_two_pass_global_noise_advanced import (
    MiniMaxH3ChunkedTwoPassGlobalNoisePlanT8Advanced,
    MiniMaxH3ChunkedTwoPassLowSigmaPlanT8Advanced,
    MiniMaxH3ChunkedTwoPassMaskedLowSigmaPlanT8Advanced,
)
from tools import run_community_update_real_validation as real_validation


def _plan(
    builder,
    *,
    strategy="independent_tiles_exp",
    temporal_strategy=None,
    second_pass_audio_policy=None,
    video_mask_policy=None,
):
    args = [
        "mock.safetensors",
        64,
        64,
        34,
        17,
        0.999,
        32,
        32,
        0,
        0,
        32,
        "smoothstep",
        "fp16",
        "offload_after",
        strategy,
    ]
    if temporal_strategy is not None:
        args.append(temporal_strategy)
    if second_pass_audio_policy is not None:
        if temporal_strategy is None:
            args.append("full_clip_safe")
        args.append(second_pass_audio_policy)
    if video_mask_policy is not None:
        if temporal_strategy is None:
            args.append("full_clip_safe")
        if second_pass_audio_policy is None:
            args.append("joint_av_preserve_input")
        args.append(video_mask_policy)
    return builder(*args)[0]


class _CountingCoordinateNoise:
    seed = 12345

    def __init__(self):
        self.calls = 0
        self.video = None

    def generate_noise(self, latent):
        self.calls += 1
        video, audio = latent["samples"].tensors
        self.video = torch.arange(video.numel(), dtype=video.dtype).reshape(
            video.shape
        )
        return chunked.comfy.nested_tensor.NestedTensor(
            (self.video, torch.ones_like(audio))
        )


def test_v2_plan_is_opt_in_and_keeps_v1_schema_unchanged():
    v1 = _plan(chunked.build_chunked_two_pass_plan)
    v2 = _plan(chunked.build_chunked_two_pass_global_noise_plan)

    assert v1["schema"] == chunked.PLAN_SCHEMA_V1
    assert "noise_policy" not in v1
    assert v2["schema"] == chunked.PLAN_SCHEMA_GLOBAL_NOISE_V2
    assert v2["noise_policy"] == chunked.GLOBAL_NOISE_POLICY
    assert v2["audio_noise_policy"].startswith("zero_per_piece")
    assert "ancestral" in v2["sampler_boundary"]
    assert v2["temporal_strategy"] == "full_clip_safe"
    assert v2["temporal_merge_policy"] == chunked.TEMPORAL_FULL_CLIP_POLICY
    assert "complete source timeline" in v2["temporal_overlap_policy"]


def test_v3_plan_requires_complete_first_pass_and_defaults_to_joint_av_context():
    v3 = _plan(chunked.build_chunked_two_pass_low_sigma_plan)

    assert v3["schema"] == chunked.PLAN_SCHEMA_LOW_SIGMA_V3
    assert v3["first_pass_contract"] == "complete_trajectory_to_zero_before_upscale"
    assert v3["recommended_refine"] == {
        "scheduler": "simple",
        "steps": 3,
        "denoise": 0.30,
        "upstream_readme_max_denoise": 0.40,
    }
    assert v3["second_pass_audio_policy"] == "joint_av_preserve_input"
    assert v3["final_audio_policy"] == "return_exact_first_pass_audio_tensor"


def test_v3_plan_keeps_locked_audio_as_an_explicit_diagnostic_option():
    v3 = _plan(
        chunked.build_chunked_two_pass_low_sigma_plan,
        second_pass_audio_policy="locked_input_audio",
    )

    assert v3["second_pass_audio_policy"] == "locked_input_audio"
    assert v3["audio_noise_policy"] == "zero_per_piece_with_zero_audio_noise_mask"


def test_v4_plan_requires_inherited_video_mask_without_changing_v3():
    v3 = _plan(chunked.build_chunked_two_pass_low_sigma_plan)
    v4 = _plan(chunked.build_chunked_two_pass_masked_low_sigma_plan)

    assert v3["schema"] == chunked.PLAN_SCHEMA_LOW_SIGMA_V3
    assert "video_mask_policy" not in v3
    assert v4["schema"] == chunked.PLAN_SCHEMA_MASKED_LOW_SIGMA_V4
    assert v4["video_mask_policy"] == "inherit_required"
    assert v4["video_mask_resize"] == "nearest_exact_spatial_only"
    assert v4["video_mask_combine"] == (
        "inherited_times_spatial_ownership_times_temporal_ownership"
    )
    assert v4["second_pass_audio_policy"] == "joint_av_preserve_input"


def test_static_pixel_mask_normalizes_to_single_channel_latent_grid():
    video = torch.zeros((1, 24, 4, 2, 3), dtype=torch.float32)
    audio = torch.zeros((1, 32, 2, 8), dtype=torch.float32)
    static = torch.zeros((16, 1, 8, 12), dtype=torch.float32)
    static[..., 2:6, 4:8] = 1.0
    latent = {
        "samples": chunked.comfy.nested_tensor.NestedTensor((video, audio)),
        "noise_mask": chunked.comfy.nested_tensor.NestedTensor(
            (static, torch.ones_like(audio))
        ),
    }

    normalized, report = chunked._normalize_inherited_video_mask(
        latent,
        video,
        policy="inherit_required",
    )

    assert tuple(normalized.shape) == (1, 1, 4, 2, 3)
    assert torch.equal(normalized[:, :, 0], normalized[:, :, -1])
    assert report["temporal_policy"] == "verified_static_then_expanded"
    assert report["channel_policy"] == "single_channel"
    assert report["source_shape"] == [16, 1, 8, 12]


def test_dynamic_nonlatent_time_mask_is_rejected_instead_of_interpolated():
    video = torch.zeros((1, 24, 4, 2, 3), dtype=torch.float32)
    audio = torch.zeros((1, 32, 2, 8), dtype=torch.float32)
    dynamic = torch.zeros((16, 1, 8, 12), dtype=torch.float32)
    dynamic[8:] = 1.0
    latent = {
        "samples": chunked.comfy.nested_tensor.NestedTensor((video, audio)),
        "noise_mask": chunked.comfy.nested_tensor.NestedTensor(
            (dynamic, torch.ones_like(audio))
        ),
    }

    with pytest.raises(ValueError, match="temporal interpolation is forbidden"):
        chunked._normalize_inherited_video_mask(
            latent,
            video,
            policy="inherit_required",
        )


def test_v4_required_policy_refuses_a_missing_mask_before_sampling():
    video = torch.zeros((1, 24, 4, 2, 2), dtype=torch.float32)
    audio = torch.zeros((1, 32, 2, 22), dtype=torch.float32)
    latent = {"samples": chunked.comfy.nested_tensor.NestedTensor((video, audio))}

    with pytest.raises(ValueError, match="requires a nested first-pass noise_mask"):
        chunked.execute_chunked_two_pass_upscale(
            object(),
            [],
            latent,
            object(),
            object(),
            torch.tensor([0.5, 0.0]),
            _plan(chunked.build_chunked_two_pass_masked_low_sigma_plan),
        )


def test_inherited_mask_multiplies_spatial_and_temporal_ownership(monkeypatch):
    video = torch.zeros((1, 24, 2, 4, 4), dtype=torch.float32)
    audio = torch.zeros((1, 32, 2, 8), dtype=torch.float32)
    inherited = torch.zeros((1, 1, 2, 4, 4), dtype=torch.float32)
    inherited[..., 1:3, 1:3] = 1.0
    temporal = torch.ones((1, 1, 2, 1, 1), dtype=torch.float32)
    temporal[:, :, :1] = 0.25
    captured = {}

    def fake_sample_piece(piece, *_args, **_kwargs):
        captured["video_mask"] = piece["noise_mask"].tensors[0].clone()
        return piece["samples"]

    monkeypatch.setattr(chunked, "sample_piece", fake_sample_piece)
    plan = _plan(
        chunked.build_chunked_two_pass_masked_low_sigma_plan,
        strategy="full_frame_safe",
    )
    plan.update(
        {
            "tile_width": 64,
            "tile_height": 64,
            "minimum_tile_size": 64,
        }
    )
    output, report = chunked._spatial_resample(
        video,
        audio,
        [],
        plan,
        object(),
        object(),
        object(),
        torch.tensor([0.5, 0.0]),
        None,
        1.0,
        chunk_temporal_mask=temporal,
        chunk_inherited_video_mask=inherited,
    )

    assert torch.equal(output, video)
    assert torch.equal(captured["video_mask"], inherited * temporal)
    assert report["inherited_video_mask_applied"] is True
    assert report["video_mask_combine"] == (
        "inherited_times_spatial_ownership_times_temporal_ownership"
    )


def test_v4_executor_propagates_resized_mask_and_preserves_audio_identity(monkeypatch):
    video = torch.zeros((1, 24, 4, 2, 2), dtype=torch.float32)
    audio = torch.randn((1, 32, 2, 22), dtype=torch.float32)
    video_mask = torch.zeros((1, 1, 4, 2, 2), dtype=torch.float32)
    video_mask[..., 0, 0] = 1.0
    latent = {
        "samples": chunked.comfy.nested_tensor.NestedTensor((video, audio)),
        "noise_mask": chunked.comfy.nested_tensor.NestedTensor(
            (video_mask, torch.ones_like(audio))
        ),
    }
    captured = {}

    def fake_upscale(value, *_args, **_kwargs):
        source_video, source_audio = value["samples"].tensors
        upscaled_video = torch.zeros(
            (*source_video.shape[:-2], 4, 4), dtype=source_video.dtype
        )
        output = dict(value)
        output["samples"] = chunked.comfy.nested_tensor.NestedTensor(
            (upscaled_video, source_audio)
        )
        source_mask, source_audio_mask = value["noise_mask"].tensors
        output["noise_mask"] = chunked.comfy.nested_tensor.NestedTensor(
            (
                chunked._resize_video_mask_spatial_only(source_mask, 4, 4),
                source_audio_mask,
            )
        )
        return output, 64, 64, json.dumps({"status": "mock_upscale"})

    def fake_spatial(chunk_video, *_args, chunk_inherited_video_mask=None, **_kwargs):
        captured["mask"] = chunk_inherited_video_mask.clone()
        return chunk_video, {"mock": True}

    monkeypatch.setattr(chunked, "learned_upscale_h3_av_latent", fake_upscale)
    monkeypatch.setattr(chunked, "_spatial_resample", fake_spatial)
    output, report_json = chunked.execute_chunked_two_pass_upscale(
        object(),
        [],
        latent,
        _CountingCoordinateNoise(),
        object(),
        torch.tensor([0.5, 0.0]),
        _plan(chunked.build_chunked_two_pass_masked_low_sigma_plan),
    )
    report = json.loads(report_json)
    expected = chunked._resize_video_mask_spatial_only(video_mask, 4, 4)

    assert torch.equal(captured["mask"], expected)
    assert output["samples"].tensors[1] is audio
    assert report["audio_preserved_by_identity"] is True
    assert report["inherited_video_mask"]["status"] == "normalized"
    assert report["inherited_video_mask"]["target_shape"] == [1, 1, 4, 4, 4]


def test_temporal_overlap_profile_keeps_a_guard_and_smoothly_takes_over():
    mask, locked, transition = chunked._temporal_overlap_mask(
        10,
        5,
        dtype=torch.float32,
        device="cpu",
    )

    assert locked == 2
    assert transition == 3
    assert torch.count_nonzero(mask[:, :, :locked]) == 0
    assert 0 < float(mask[0, 0, 2, 0, 0]) < 1
    assert 0 < float(mask[0, 0, 3, 0, 0]) < 1
    assert float(mask[0, 0, 4, 0, 0]) == 1
    assert torch.all(mask[:, :, 5:] == 1)


def test_global_target_noise_is_generated_once_at_full_target_shape():
    video = torch.zeros((1, 24, 15, 2, 2), dtype=torch.float16)
    audio = torch.zeros((1, 32, 2, 85), dtype=torch.float16)
    latent = {
        "samples": chunked.comfy.nested_tensor.NestedTensor((video, audio)),
        "batch_index": [0],
    }
    noise = _CountingCoordinateNoise()

    generated, report = chunked._build_global_target_video_noise(
        noise,
        latent,
        video,
        audio,
        _plan(chunked.build_chunked_two_pass_global_noise_plan),
    )

    assert noise.calls == 1
    assert tuple(generated.shape) == (1, 24, 15, 4, 4)
    assert torch.equal(generated, noise.video)
    assert report["generate_noise_calls"] == 1
    assert report["target_video_noise_shape"] == [1, 24, 15, 4, 4]
    assert report["seed"] == 12345


def test_v3_global_target_noise_keeps_matching_audio_noise_for_model_context():
    video = torch.zeros((1, 24, 15, 2, 2), dtype=torch.float16)
    audio = torch.zeros((1, 32, 2, 85), dtype=torch.float16)
    latent = {"samples": chunked.comfy.nested_tensor.NestedTensor((video, audio))}
    noise = _CountingCoordinateNoise()

    video_noise, audio_noise, report = chunked._build_global_target_av_noise(
        noise,
        latent,
        video,
        audio,
        _plan(chunked.build_chunked_two_pass_low_sigma_plan),
    )

    assert noise.calls == 1
    assert tuple(video_noise.shape) == (1, 24, 15, 4, 4)
    assert torch.count_nonzero(audio_noise) == audio_noise.numel()
    assert report["target_audio_noise_shape"] == [1, 32, 2, 85]
    assert report["audio_noise"] == "generated_once_full_timeline"


def test_spatial_tiles_receive_exact_slices_and_zero_audio_noise(monkeypatch):
    video = torch.zeros((1, 24, 2, 4, 4), dtype=torch.float32)
    audio = torch.randn((1, 32, 2, 8), dtype=torch.float32)
    global_noise = torch.arange(video.numel(), dtype=video.dtype).reshape(video.shape)
    captured = []

    def fake_sample_piece(piece, *_args, prepared_noise=None, **_kwargs):
        assert prepared_noise is not None
        tile_noise, audio_noise = prepared_noise.tensors
        captured.append((tile_noise.clone(), audio_noise.clone()))
        return piece["samples"]

    monkeypatch.setattr(chunked, "sample_piece", fake_sample_piece)
    plan = _plan(chunked.build_chunked_two_pass_global_noise_plan)
    output, report = chunked._spatial_resample(
        video,
        audio,
        [],
        plan,
        object(),
        object(),
        object(),
        torch.tensor([0.5, 0.0]),
        None,
        1.0,
        chunk_noise_video=global_noise,
    )

    expected = [
        global_noise[..., 0:2, 0:2],
        global_noise[..., 0:2, 2:4],
        global_noise[..., 2:4, 0:2],
        global_noise[..., 2:4, 2:4],
    ]
    assert len(captured) == 4
    assert all(torch.equal(actual[0], wanted) for actual, wanted in zip(captured, expected))
    assert all(torch.count_nonzero(actual[1]) == 0 for actual in captured)
    assert torch.equal(output, video)
    assert report["noise_policy"] == chunked.GLOBAL_NOISE_POLICY


def test_v3_joint_av_refine_reuses_audio_noise_but_returns_video_only_from_piece(
    monkeypatch,
):
    video = torch.zeros((1, 24, 2, 4, 4), dtype=torch.float32)
    audio = torch.randn((1, 32, 2, 8), dtype=torch.float32)
    video_noise = torch.arange(video.numel(), dtype=video.dtype).reshape(video.shape)
    audio_noise = torch.full_like(audio, 7.0)
    captured = []

    def fake_sample_piece(piece, *_args, prepared_noise=None, **_kwargs):
        assert prepared_noise is not None
        video_mask, audio_mask = piece["noise_mask"].tensors
        sampled_video_noise, sampled_audio_noise = prepared_noise.tensors
        captured.append(
            (
                video_mask.clone(),
                audio_mask.clone(),
                sampled_video_noise.clone(),
                sampled_audio_noise.clone(),
            )
        )
        return piece["samples"]

    monkeypatch.setattr(chunked, "sample_piece", fake_sample_piece)
    plan = _plan(chunked.build_chunked_two_pass_low_sigma_plan)
    output, report = chunked._spatial_resample(
        video,
        audio,
        [],
        plan,
        object(),
        object(),
        object(),
        torch.tensor([0.5, 0.0]),
        None,
        1.0,
        chunk_noise_video=video_noise,
        chunk_noise_audio=audio_noise,
        audio_sampling_policy="joint_av_preserve_input",
    )

    assert len(captured) == 4
    assert all(torch.all(item[1] == 1) for item in captured)
    assert all(torch.equal(item[3], audio_noise) for item in captured)
    assert torch.equal(output, video)
    assert report["audio_sampling_policy"] == "joint_av_preserve_input"


def test_v3_spatial_sampling_rebinds_shape_bound_sampler_per_piece(monkeypatch):
    video = torch.zeros((1, 24, 4, 2, 2), dtype=torch.float32)
    audio = torch.zeros((1, 32, 2, 8), dtype=torch.float32)
    rebound = object()
    captured = {}

    def fake_rebind(model, piece, sampler):
        captured["model"] = model
        captured["piece"] = piece
        captured["sampler"] = sampler
        return rebound

    def fake_sample_piece(
        piece, _conditioning, _model, _noise, sampler, *_args, **_kwargs
    ):
        captured["effective_sampler"] = sampler
        return piece["samples"]

    monkeypatch.setattr(chunked, "rebind_dual_clock_sampler", fake_rebind)
    monkeypatch.setattr(chunked, "sample_piece", fake_sample_piece)
    model = object()
    sampler = object()
    output, _report = chunked._spatial_resample(
        video,
        audio,
        [],
        _plan(chunked.build_chunked_two_pass_low_sigma_plan),
        model,
        object(),
        sampler,
        torch.tensor([0.5, 0.0]),
        None,
        1.0,
        rebind_shape_bound_sampler=True,
    )

    assert captured["model"] is model
    assert captured["sampler"] is sampler
    assert captured["effective_sampler"] is rebound
    assert torch.equal(output, video)


def test_temporal_ownership_mask_is_applied_to_video_but_not_audio(monkeypatch):
    video = torch.zeros((1, 24, 4, 2, 2), dtype=torch.float32)
    audio = torch.randn((1, 32, 2, 8), dtype=torch.float32)
    temporal_mask = torch.ones((1, 1, 4, 1, 1), dtype=torch.float32)
    temporal_mask[:, :, :2] = 0
    captured = {}

    def fake_sample_piece(piece, *_args, **_kwargs):
        video_mask, audio_mask = piece["noise_mask"].tensors
        captured["video_mask"] = video_mask.clone()
        captured["audio_mask"] = audio_mask.clone()
        return piece["samples"]

    monkeypatch.setattr(chunked, "sample_piece", fake_sample_piece)
    plan = _plan(
        chunked.build_chunked_two_pass_global_noise_plan,
        strategy="full_frame_safe",
    )
    output, _report = chunked._spatial_resample(
        video,
        audio,
        [],
        plan,
        object(),
        object(),
        object(),
        torch.tensor([0.5, 0.0]),
        None,
        1.0,
        chunk_temporal_mask=temporal_mask,
    )

    assert torch.equal(output, video)
    assert tuple(captured["video_mask"].shape) == (1, 1, 4, 2, 2)
    assert torch.count_nonzero(captured["video_mask"][:, :, :2]) == 0
    assert torch.all(captured["video_mask"][:, :, 2:] == 1)
    assert torch.count_nonzero(captured["audio_mask"]) == 0


def test_executor_slices_one_global_noise_across_overlapping_time_chunks(monkeypatch):
    video = torch.zeros((1, 24, 15, 4, 4), dtype=torch.float32)
    audio = torch.randn((1, 32, 2, 85), dtype=torch.float32)
    latent = {"samples": chunked.comfy.nested_tensor.NestedTensor((video, audio))}
    noise = _CountingCoordinateNoise()
    captured_noise = []
    captured_video = []
    captured_masks = []

    monkeypatch.setattr(
        chunked,
        "learned_upscale_h3_av_latent",
        lambda value, *_args, **_kwargs: (
            value,
            64,
            64,
            json.dumps({"status": "mock_upscale"}),
        ),
    )

    def fake_spatial(
        chunk_video,
        _chunk_audio,
        *_args,
        chunk_noise_video=None,
        chunk_temporal_mask=None,
        **_kwargs,
    ):
        call_index = len(captured_noise)
        captured_noise.append(chunk_noise_video.clone())
        captured_video.append(chunk_video.clone())
        captured_masks.append(
            None if chunk_temporal_mask is None else chunk_temporal_mask.clone()
        )
        offset = 1.0 if call_index == 0 else 10.0
        if chunk_temporal_mask is None:
            return chunk_video + offset, {"mock": True}
        return chunk_video + offset * chunk_temporal_mask, {"mock": True}

    monkeypatch.setattr(chunked, "_spatial_resample", fake_spatial)
    output, report_json = chunked.execute_chunked_two_pass_upscale(
        object(),
        [],
        latent,
        noise,
        object(),
        torch.tensor([0.5, 0.0]),
        _plan(
            chunked.build_chunked_two_pass_global_noise_plan,
            temporal_strategy="guarded_overlap_exp",
        ),
    )
    report = json.loads(report_json)

    assert noise.calls == 1
    assert len(captured_noise) == 2
    assert torch.equal(captured_noise[0], noise.video[:, :, 0:10])
    assert torch.equal(captured_noise[1], noise.video[:, :, 5:15])
    assert captured_masks[0] is None
    assert torch.count_nonzero(captured_masks[1][:, :, :2]) == 0
    assert torch.all(captured_masks[1][:, :, 2:4] > 0)
    assert torch.all(captured_masks[1][:, :, 2:4] < 1)
    assert torch.all(captured_masks[1][:, :, 4:] == 1)
    assert torch.all(captured_video[1][:, :, :5] == 1)
    output_video = output["samples"].tensors[0]
    assert tuple(output_video.shape) == tuple(video.shape)
    assert torch.all(output_video[:, :, :7] == 1)
    assert torch.all(output_video[:, :, 7:9] > 1)
    assert torch.all(output_video[:, :, 7:9] < 11)
    assert torch.all(output_video[:, :, 9:10] == 11)
    assert torch.all(output_video[:, :, 10:] == 10)
    assert output["samples"].tensors[1] is audio
    assert report["global_noise"]["generate_noise_calls"] == 1
    assert report["audio_preserved_by_identity"] is True
    assert report["temporal_merge_policy"] == chunked.TEMPORAL_OWNERSHIP_POLICY
    assert report["segments"][0]["locked_overlap_tokens"] == 0
    assert report["segments"][0]["published_new_tokens"] == 10
    assert report["segments"][1]["locked_overlap_tokens"] == 2
    assert report["segments"][1]["transition_overlap_tokens"] == 3
    assert report["segments"][1]["published_new_tokens"] == 5


def test_default_v2_executor_uses_one_full_temporal_trajectory(monkeypatch):
    video = torch.zeros((1, 24, 15, 4, 4), dtype=torch.float32)
    audio = torch.zeros((1, 32, 2, 85), dtype=torch.float32)
    latent = {"samples": chunked.comfy.nested_tensor.NestedTensor((video, audio))}
    noise = _CountingCoordinateNoise()
    sampled_shapes = []

    monkeypatch.setattr(
        chunked,
        "learned_upscale_h3_av_latent",
        lambda value, *_args, **_kwargs: (
            value,
            64,
            64,
            json.dumps({"status": "mock_upscale"}),
        ),
    )

    def fake_spatial(chunk_video, *_args, **_kwargs):
        sampled_shapes.append(tuple(chunk_video.shape))
        return chunk_video, {"mock": True}

    monkeypatch.setattr(chunked, "_spatial_resample", fake_spatial)
    _output, report_json = chunked.execute_chunked_two_pass_upscale(
        object(),
        [],
        latent,
        noise,
        object(),
        torch.tensor([0.5, 0.0]),
        _plan(chunked.build_chunked_two_pass_global_noise_plan),
    )
    report = json.loads(report_json)

    assert sampled_shapes == [tuple(video.shape)]
    assert report["segment_count"] == 1
    assert report["temporal_strategy"] == "full_clip_safe"
    assert report["temporal_merge_policy"] == chunked.TEMPORAL_FULL_CLIP_POLICY


def test_old_v1_executor_never_builds_global_noise(monkeypatch):
    video = torch.zeros((1, 24, 5, 4, 4))
    audio = torch.zeros((1, 32, 2, 29))
    latent = {"samples": chunked.comfy.nested_tensor.NestedTensor((video, audio))}
    monkeypatch.setattr(
        chunked,
        "_build_global_target_video_noise",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("v1 must not build global noise")
        ),
    )
    monkeypatch.setattr(
        chunked,
        "learned_upscale_h3_av_latent",
        lambda value, *_args, **_kwargs: (
            value,
            64,
            64,
            json.dumps({"status": "mock_upscale"}),
        ),
    )
    monkeypatch.setattr(
        chunked,
        "_spatial_resample",
        lambda value, *_args, **_kwargs: (value, {"mock": True}),
    )
    output, report_json = chunked.execute_chunked_two_pass_upscale(
        object(),
        [],
        latent,
        object(),
        object(),
        torch.tensor([0.5, 0.0]),
        _plan(chunked.build_chunked_two_pass_plan, strategy="full_frame_safe"),
    )

    assert output["samples"].tensors[1] is audio
    assert json.loads(report_json)["global_noise"] is None


def test_v2_node_is_append_only_and_reuses_existing_plan_socket():
    schema = MiniMaxH3ChunkedTwoPassGlobalNoisePlanT8Advanced.define_schema()
    assert schema.is_experimental is True
    assert schema.node_id == "MiniMaxH3ChunkedTwoPassGlobalNoisePlanT8Advanced"
    node_ids = [
        node.define_schema().node_id
        for node in asyncio.run(comfy_entrypoint().get_node_list())
    ]
    assert node_ids[-7] == schema.node_id
    assert node_ids[-6] == "MiniMaxH3ChunkedTwoPassLowSigmaPlanT8Advanced"
    assert node_ids[-5] == (
        "MiniMaxH3ChunkedTwoPassMaskedLowSigmaPlanT8Advanced"
    )
    assert node_ids[-4] == "MiniMaxH3SubjectSafeRGBCompositeT8Advanced"
    v3_schema = MiniMaxH3ChunkedTwoPassLowSigmaPlanT8Advanced.define_schema()
    assert v3_schema.is_experimental is True
    assert v3_schema.node_id == node_ids[-6]
    v4_schema = MiniMaxH3ChunkedTwoPassMaskedLowSigmaPlanT8Advanced.define_schema()
    assert v4_schema.is_experimental is True
    assert v4_schema.node_id == node_ids[-5]
    assert [item.id for item in v4_schema.inputs][-2:] == [
        "second_pass_audio_policy",
        "video_mask_policy",
    ]


def test_real_validation_prealigns_keyframes_and_defaults_to_full_context_routes():
    args = real_validation._parser().parse_args(
        ["--mode", "chunked_two_pass_global_noise"]
    )
    prompt, _reports = real_validation._chunked_prompt(args, "test")
    plan_inputs = prompt["14"]["inputs"]

    assert args.spatial_strategy == "full_frame_safe"
    assert args.temporal_strategy == "full_clip_safe"
    assert plan_inputs["spatial_strategy"] == "full_frame_safe"
    assert plan_inputs["temporal_strategy"] == "full_clip_safe"
    assert prompt["6"] == {
        "class_type": "ImageScale",
        "inputs": {
            "image": ["5", 0],
            "upscale_method": "lanczos",
            "width": args.target_width,
            "height": args.target_height,
            "crop": "center",
        },
    }
    for node_id in ("7", "13"):
        assert prompt[node_id]["inputs"]["first_frame"] == ["6", 0]
        assert prompt[node_id]["inputs"]["last_frame"] == ["6", 0]


def test_low_sigma_validation_completes_pass1_before_upscale_and_uses_three_step_refine():
    args = real_validation._parser().parse_args(
        ["--mode", "chunked_two_pass_low_sigma", "--save-draft"]
    )
    prompt, reports = real_validation._chunked_low_sigma_prompt(args, "test")

    assert reports == {"plan": "18", "execution": "19"}
    assert prompt["7"]["class_type"] == "LoraLoaderBypassModelOnly"
    assert prompt["7"]["inputs"]["lora_name"] == real_validation.TURBO_ALPHA8
    assert prompt["9"]["inputs"]["steps"] == 8
    assert prompt["12"]["inputs"]["sigmas"] == ["9", 2]
    assert not any(
        node["class_type"] == "SplitSigmas" for node in prompt.values()
    )
    assert prompt["14"]["class_type"] == (
        "MiniMaxH3ChunkedTwoPassLowSigmaPlanT8Advanced"
    )
    assert prompt["14"]["inputs"]["second_pass_audio_policy"] == (
        "joint_av_preserve_input"
    )
    assert prompt["15"] == {
        "class_type": "BasicScheduler",
        "inputs": {
            "model": ["9", 0],
            "scheduler": "simple",
            "steps": 3,
            "denoise": 0.30,
        },
    }
    assert prompt["17"]["inputs"]["latent"] == ["12", 1]
    assert prompt["17"]["inputs"]["sigmas"] == ["15", 0]
    assert prompt["23"]["inputs"]["filename_prefix"].endswith(
        "chunked_two_pass_low_sigma_draft_first_pass"
    )


def test_masked_low_sigma_validation_inherits_one_mask_across_both_passes():
    args = real_validation._parser().parse_args(
        [
            "--mode",
            "chunked_two_pass_masked_low_sigma",
            "--width",
            "576",
            "--height",
            "320",
            "--save-draft",
        ]
    )
    prompt, reports = real_validation._chunked_masked_low_sigma_prompt(args, "test")

    assert reports == {"plan": "18", "execution": "19"}
    assert prompt["14"]["class_type"] == (
        "MiniMaxH3ChunkedTwoPassMaskedLowSigmaPlanT8Advanced"
    )
    assert prompt["14"]["inputs"]["video_mask_policy"] == "inherit_required"
    assert prompt["26"] == {
        "class_type": "LoadImageMask",
        "inputs": {
            "image": real_validation.STATIC_BACKGROUND_MASK,
            "channel": "red",
        },
    }
    assert prompt["27"]["inputs"] == {
        "samples": ["25", 0],
        "mask": ["26", 0],
    }
    assert prompt["29"]["inputs"] == {
        "video_latent": ["27", 0],
        "audio_latent": ["28", 1],
    }
    assert prompt["9"]["inputs"]["av_latent"] == ["29", 0]
    assert prompt["12"]["inputs"]["latent_image"] == ["29", 0]
    assert prompt["17"]["inputs"]["latent"] == ["12", 1]


def test_masked_validation_accepts_an_input_relative_precise_mask_without_changing_default():
    precise = "codex_h3_precise_subject_mask_576x320.png"
    default_args = real_validation._parser().parse_args(
        ["--mode", "chunked_two_pass_masked_low_sigma"]
    )
    precise_args = real_validation._parser().parse_args(
        [
            "--mode",
            "chunked_two_pass_masked_low_sigma",
            "--video-mask",
            precise,
        ]
    )

    default_prompt, _ = real_validation._chunked_masked_low_sigma_prompt(
        default_args, "default"
    )
    precise_prompt, _ = real_validation._chunked_masked_low_sigma_prompt(
        precise_args, "precise"
    )

    assert default_prompt["26"]["inputs"]["image"] == (
        real_validation.STATIC_BACKGROUND_MASK
    )
    assert precise_prompt["26"]["inputs"]["image"] == precise


def test_upscale_only_control_keeps_the_masked_first_pass_and_has_no_refine_sampler():
    precise = "codex_h3_precise_subject_mask_576x320.png"
    args = real_validation._parser().parse_args(
        [
            "--mode",
            "chunked_two_pass_upscale_only_control",
            "--video-mask",
            precise,
            "--save-draft",
        ]
    )
    prompt, reports = real_validation._chunked_upscale_only_prompt(args, "test")

    assert reports == {"upscale": "34"}
    assert prompt["9"]["inputs"]["av_latent"] == ["29", 0]
    assert prompt["12"]["inputs"]["latent_image"] == ["29", 0]
    assert prompt["26"]["inputs"]["image"] == precise
    assert prompt["31"]["class_type"] == "LTXVSeparateAVLatent"
    assert prompt["32"]["class_type"] == "LTXVConcatAVLatent"
    assert prompt["33"]["class_type"] == (
        "MiniMaxH3LearnedLatentUpscaleT8Advanced"
    )
    assert prompt["33"]["inputs"]["av_latent"] == ["32", 0]
    assert prompt["35"]["inputs"]["av_latent"] == ["33", 0]
    assert prompt["38"]["inputs"]["filename_prefix"].endswith(
        "chunked_two_pass_upscale_only_control_draft_first_pass"
    )
    assert sum(
        node["class_type"] == "SamplerCustomAdvanced"
        for node in prompt.values()
    ) == 1
    assert not any(
        node["class_type"]
        in {
            "MiniMaxH3ChunkedTwoPassLowSigmaPlanT8Advanced",
            "MiniMaxH3ChunkedTwoPassMaskedLowSigmaPlanT8Advanced",
            "MiniMaxH3ChunkedTwoPassUpscaleT8Advanced",
            "H3LoopingSampler",
        }
        for node in prompt.values()
    )


def test_masked_low_sigma_frontend_workflow_is_importable_and_explicit():
    root = Path(__file__).resolve().parents[1]
    path = (
        root
        / "examples"
        / "workflows"
        / "13-latent-upscale"
        / "2026-08-30_H3_Mask_Preserving_Low_Sigma_TwoPass_v4_Advanced_EXP.json"
    )
    workflow = json.loads(path.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in workflow["nodes"]}
    types = [node["type"] for node in workflow["nodes"]]

    assert workflow["version"] == 0.4
    assert workflow["last_node_id"] == max(nodes)
    assert workflow["last_link_id"] == max(link[0] for link in workflow["links"])
    assert len(workflow["links"]) == len({link[0] for link in workflow["links"]})
    assert all(link[1] in nodes and link[3] in nodes for link in workflow["links"])
    assert types.count("MiniMaxH3ChunkedTwoPassMaskedLowSigmaPlanT8Advanced") == 1
    assert types.count("MiniMaxH3ChunkedTwoPassUpscaleT8Advanced") == 1
    assert types.count("LoadImageMask") == 1
    assert types.count("SetLatentNoiseMask") == 1
    assert types.count("RepeatImageBatch") == 1
    assert types.count("LTXVSeparateAVLatent") == 1
    assert types.count("LTXVConcatAVLatent") == 1
    assert types.count("MarkdownNote") == 3

    plan = next(
        node
        for node in workflow["nodes"]
        if node["type"] == "MiniMaxH3ChunkedTwoPassMaskedLowSigmaPlanT8Advanced"
    )
    dual_clock = next(
        node
        for node in workflow["nodes"]
        if node["type"] == "MiniMaxH3DualClockSamplerT8"
    )
    scheduler = next(
        node for node in workflow["nodes"] if node["type"] == "BasicScheduler"
    )
    assert plan["widgets_values"][-4:] == [
        "full_frame_safe",
        "full_clip_safe",
        "joint_av_preserve_input",
        "inherit_required",
    ]
    assert dual_clock["widgets_values"][:3] == [8, 12.0, 3.0]
    assert scheduler["widgets_values"] == ["simple", 3, 0.3]

    notes = "\n".join(
        node["widgets_values"][0]
        for node in workflow["nodes"]
        if node["type"] == "MarkdownNote"
    )
    for required in (
        "黑色 `0`",
        "白色 `1`",
        "inherit_required",
        "v1/v2/v3",
        "201 MiB",
        "人工观看",
    ):
        assert required in notes


def test_upstream_exact_validation_is_a_same_schedule_full_context_control():
    args = real_validation._parser().parse_args(
        ["--mode", "chunked_two_pass_upstream_exact", "--save-draft"]
    )
    prompt, reports = real_validation._chunked_upstream_exact_prompt(args, "test")

    assert reports == {}
    assert prompt["12"]["inputs"]["sigmas"] == ["9", 2]
    assert prompt["13"]["class_type"] == "MiniMaxH3LearnedLatentUpscaleT8Advanced"
    assert prompt["15"]["inputs"]["av_latent"] == ["13", 0]
    assert prompt["16"]["inputs"] == {
        "model": ["15", 0],
        "scheduler": "simple",
        "steps": 3,
        "denoise": 0.30,
    }
    assert prompt["19"]["class_type"] == "H3LoopingSampler"
    assert prompt["24"] == {
        "class_type": "KSamplerSelect",
        "inputs": {"sampler_name": "euler"},
    }
    assert prompt["19"]["inputs"]["sampler"] == ["24", 0]
    assert prompt["19"]["inputs"]["horizontal_tiles"] == 1
    assert prompt["19"]["inputs"]["vertical_tiles"] == 1
    assert prompt["19"]["inputs"]["adain_factor"] == 0.0
    assert prompt["21"]["inputs"]["images"] == ["20", 0]
    assert prompt["23"]["inputs"]["filename_prefix"].endswith(
        "chunked_two_pass_upstream_exact_draft_first_pass"
    )


def test_full_frame_euler_control_changes_only_the_upstream_wrapper():
    args = real_validation._parser().parse_args(
        ["--mode", "chunked_two_pass_full_frame_euler_control", "--save-draft"]
    )
    control, reports = real_validation._chunked_full_frame_euler_control_prompt(
        args, "test"
    )
    upstream, _ = real_validation._chunked_upstream_exact_prompt(args, "test")

    assert reports == {}
    assert control["19"] == {
        "class_type": "SamplerCustomAdvanced",
        "inputs": {
            "noise": ["18", 0],
            "guider": ["17", 0],
            "sampler": ["24", 0],
            "sigmas": ["16", 0],
            "latent_image": ["13", 0],
        },
    }
    assert control["20"]["inputs"]["av_latent"] == ["19", 1]
    assert control["24"] == upstream["24"]
    assert control["16"] == upstream["16"]
    assert control["18"] == upstream["18"]
    assert control["13"] == upstream["13"]
    assert control["14"] == upstream["14"]
    assert control["15"] == upstream["15"]
    assert control["12"] == upstream["12"]
    assert not any(
        node["class_type"] == "H3LoopingSampler" for node in control.values()
    )


def test_upstream_example_validation_keeps_the_same_masked_first_pass_but_not_a_tiled_semantic_mask():
    precise = "codex_h3_precise_subject_mask_576x320.png"
    args = real_validation._parser().parse_args(
        [
            "--mode",
            "chunked_two_pass_upstream_example",
            "--width",
            "576",
            "--height",
            "320",
            "--target-width",
            "1152",
            "--target-height",
            "640",
            "--video-mask",
            precise,
        ]
    )
    prompt, reports = real_validation._chunked_upstream_example_prompt(args, "test")

    assert reports == {}
    assert prompt["9"]["inputs"]["av_latent"] == ["29", 0]
    assert prompt["12"]["inputs"]["latent_image"] == ["29", 0]
    assert prompt["26"]["inputs"]["image"] == precise
    assert prompt["32"] == {
        "class_type": "SolidMask",
        "inputs": {"value": 1.0, "width": 1152, "height": 640},
    }
    assert prompt["34"]["inputs"] == {
        "video_latent": ["33", 0],
        "audio_latent": ["31", 1],
    }
    assert prompt["15"]["inputs"]["av_latent"] == ["34", 0]
    assert prompt["35"] == {
        "class_type": "KSamplerSelect",
        "inputs": {"sampler_name": "euler"},
    }
    assert prompt["19"]["inputs"] == {
        "noise": ["18", 0],
        "guider": ["17", 0],
        "sampler": ["35", 0],
        "sigmas": ["16", 0],
        "latent_image": ["34", 0],
        "temporal_tile_size": 101,
        "temporal_overlap": 49,
        "temporal_overlap_strength": 0.99,
        "horizontal_tiles": 3,
        "vertical_tiles": 3,
        "spatial_overlap": 24,
        "adain_factor": 0.0,
    }
    assert prompt["20"]["inputs"]["av_latent"] == ["19", 1]


def test_v2_frontend_workflow_is_importable_and_prealigns_keyframe_canvas():
    path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "workflows"
        / "13-latent-upscale"
        / "2026-08-30_H3_Chunked_TwoPass_Global_Noise_v2_Advanced_EXP.json"
    )
    workflow = json.loads(path.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in workflow["nodes"]}
    types = [node["type"] for node in workflow["nodes"]]

    assert workflow["version"] == 0.4
    assert workflow["last_node_id"] == max(nodes)
    assert workflow["last_link_id"] == max(link[0] for link in workflow["links"])
    assert types.count("MiniMaxH3ChunkedTwoPassGlobalNoisePlanT8Advanced") == 1
    assert types.count("MiniMaxH3ChunkedTwoPassUpscaleT8Advanced") == 1
    assert types.count("ImageScale") == 1
    assert types.count("MarkdownNote") == 3
    plan = next(
        node
        for node in workflow["nodes"]
        if node["type"] == "MiniMaxH3ChunkedTwoPassGlobalNoisePlanT8Advanced"
    )
    assert plan["widgets_values"][1:5] == [832, 512, 136, 17]
    assert plan["widgets_values"][6:10] == [832, 512, 0, 0]
    assert plan["widgets_values"][-2:] == ["full_frame_safe", "full_clip_safe"]
    keyframe_canvas = next(node for node in workflow["nodes"] if node["type"] == "ImageScale")
    assert keyframe_canvas["widgets_values"] == ["lanczos", 832, 512, "center"]
    conditionings = [
        node for node in workflow["nodes"] if node["type"] == "MiniMaxH3AudioConditioningT8"
    ]
    assert len(conditionings) == 2
    image_output_links = set(keyframe_canvas["outputs"][0]["links"])
    for conditioning in conditionings:
        first = next(item for item in conditioning["inputs"] if item["name"] == "first_frame")
        last = next(item for item in conditioning["inputs"] if item["name"] == "last_frame")
        assert first["link"] in image_output_links
        assert last["link"] in image_output_links
    notes = "\n".join(
        node["widgets_values"][0]
        for node in workflow["nodes"]
        if node["type"] == "MarkdownNote"
    )
    assert "一次生成完整目标噪声" in notes
    assert "音频不加噪" in notes
    assert "前半只读" in notes
