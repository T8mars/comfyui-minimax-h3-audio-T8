"""S18-S21 first-pass source slicing against the original Chunked executor."""
import json

import comfy.nested_tensor
import pytest
import torch

from h3_audio_t8_pkg import chunked_two_pass_upscale_advanced as old
from h3_audio_t8_pkg.modular_sampling.chunked_source import slice_chunked_source
from h3_audio_t8_pkg.modular_sampling.chunked_source_nodes import MiniMaxH3ChunkedSourceSegmentEXPT8
from test_chunked_two_pass_global_noise_advanced import _plan


def _latent(*, masked=False):
    video = torch.arange(1 * 24 * 15 * 2 * 2, dtype=torch.float32).reshape(1, 24, 15, 2, 2)
    audio = torch.arange(1 * 32 * 2 * 90, dtype=torch.float32).reshape(1, 32, 2, 90)
    latent = {"samples": comfy.nested_tensor.NestedTensor((video, audio))}
    if masked:
        mask = torch.linspace(0, 1, video.shape[2]).reshape(1, 1, -1, 1, 1).expand(1, 1, 15, 2, 2).clone()
        latent["noise_mask"] = comfy.nested_tensor.NestedTensor((mask, torch.ones_like(audio)))
    return latent


def test_public_chunked_source_matches_each_legacy_pre_upscale_chunk(monkeypatch):
    plan = _plan(old.build_chunked_two_pass_plan, strategy="full_frame_safe")
    source = _latent()
    captured = []

    def fake_upscale(chunk, *_args):
        captured.append(chunk)
        video, audio = chunk["samples"].unbind()
        lifted = {"samples": comfy.nested_tensor.NestedTensor((
            video.repeat_interleave(2, -1).repeat_interleave(2, -2), audio,
        ))}
        return lifted, 64, 64, "{}"

    monkeypatch.setattr(old, "learned_upscale_h3_av_latent", fake_upscale)
    monkeypatch.setattr(old, "_spatial_resample", lambda video, *_a, **_k: (video, {}))
    old.execute_chunked_two_pass_upscale(
        object(), [[torch.zeros(1), {}]], source, object(), object(),
        torch.tensor([1., 0.]), plan,
    )
    assert len(captured) > 1
    for index, expected in enumerate(captured):
        actual, spec, report_json = MiniMaxH3ChunkedSourceSegmentEXPT8.execute(
            source, plan, index,
        ).result
        for real, reference in zip(actual["samples"].unbind(), expected["samples"].unbind()):
            assert torch.equal(real, reference)
        report = json.loads(report_json)
        assert report["video_tokens"] == [spec.start_token, spec.end_token]
        assert report["audio_tokens_read_only"] == [spec.audio_start, spec.audio_end]
        assert report["sampled"] is False and report["portable_stage_result"] is False


@pytest.mark.parametrize("builder", [
    old.build_chunked_two_pass_global_noise_plan,
    old.build_chunked_two_pass_low_sigma_plan,
    old.build_chunked_two_pass_masked_low_sigma_plan,
])
def test_later_chunked_contracts_keep_full_clip_source_and_v4_mask(builder):
    plan = _plan(builder, strategy="full_frame_safe")
    source = _latent(masked=builder is old.build_chunked_two_pass_masked_low_sigma_plan)
    chunk, spec, report_json = slice_chunked_source(source, plan, 0)
    source_video, source_audio = source["samples"].unbind()
    chunk_video, chunk_audio = chunk["samples"].unbind()
    assert spec.count == 1 and spec.start_token == 0 and spec.end_token == source_video.shape[2]
    assert torch.equal(chunk_video, source_video)
    assert torch.equal(chunk_audio, source_audio[..., spec.audio_start:spec.audio_end])
    assert json.loads(report_json)["segment_count"] == 1
    if plan["schema"] == old.PLAN_SCHEMA_MASKED_LOW_SIGMA_V4:
        source_mask, _ = source["noise_mask"].unbind()
        chunk_mask, audio_mask = chunk["noise_mask"].unbind()
        assert torch.equal(chunk_mask, source_mask)
        assert torch.equal(audio_mask, torch.ones_like(chunk_audio))
    else:
        assert "noise_mask" not in chunk
    with pytest.raises(ValueError, match="outside"):
        slice_chunked_source(source, plan, 1)


def test_chunked_slice_rejects_unknown_contract_without_sampling():
    with pytest.raises(ValueError, match="v1-v4"):
        slice_chunked_source(_latent(), {"schema": "other"}, 0)
