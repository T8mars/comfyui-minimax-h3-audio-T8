"""Actual pinned cache CPU ownership; no GPU/quality claims or copied cache math."""
from __future__ import annotations

import os

import pytest
import torch

from h3_audio_t8_pkg.freevideo_h21.cache_state import HISTORY_POLICY, snapshot_history
from h3_audio_t8_pkg.freevideo_h21.source import Source


@pytest.fixture
def cache_module():
    location = os.environ.get("T8_H21_FREEVIDEO_SOURCE")
    if not location:
        pytest.skip("Requires installed exact source, never an implicit download")
    return Source(location).module("step_cache")


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_actual_sampled_input_and_output_history_owns_storage(cache_module, dtype):
    original_class = cache_module.StepCache
    video = torch.ones(10, 24, dtype=dtype)
    audio = torch.full((6, 32), 2., dtype=dtype)
    video_out, audio_out = video * 2, audio * 3
    with snapshot_history(cache_module):
        cache = cache_module.StepCache("easycache-av", .2, 20)
        cache.record_forward(video, audio, (.8, .6), video_out, audio_out)
        inputs = tuple(value.clone() for value in cache.prev_input)
        outputs = tuple(value.clone() for value in cache.prev_out)
        residuals = tuple(value.clone() for value in cache.prev_diff)
        video.add_(1)
        audio.add_(2)
        video_out.add_(3)
        audio_out.add_(4)
        assert all(torch.equal(a, b) for a, b in zip(inputs, cache.prev_input))
        assert all(torch.equal(a, b) for a, b in zip(outputs, cache.prev_out))
        assert all(torch.equal(a, b) for a, b in zip(residuals, cache.prev_diff))
        assert cache.prev_input[0].shape == (10, 3)  # row/channel sampling unchanged
        assert cache.prev_input[1].shape == (6, 4)
        assert cache.stats()["history_policy"] == HISTORY_POLICY
        assert cache.threshold == .2 and cache.steps == 20
    assert cache_module.StepCache is original_class


def test_original_FP32_counterexample_is_not_mistaken_for_zero_change(cache_module):
    """Same storage pattern as real reference_sampler's generated-row assignment."""
    with snapshot_history(cache_module):
        cache = cache_module.StepCache("easycache-av", .2, 20)
        video, audio = torch.ones(8, 24), torch.ones(4, 32) * 2
        cache.record_forward(video, audio, None, video * 2, audio * 3)
        video[:] = video + .1
        audio[:] = audio + .2
        cache.record_forward(video, audio, None, video * 2, audio * 3)
        assert cache.rate == pytest.approx([2., 3.], abs=2e-5)
        video[:] = video + .4
        audio[:] = audio + .8
        cache.step = 4
        assert cache.skip_forward(video, audio, None) is None
        assert cache.hits == [] and cache.decisions[-1]["hit"] is False


def test_valid_hit_still_uses_original_residual_and_exact_modality_values(cache_module):
    with snapshot_history(cache_module):
        cache = cache_module.StepCache("easycache-av", .2, 20)
        v0, a0 = torch.ones(8, 24), torch.ones(4, 32) * 2
        cache.record_forward(v0, a0, None, v0 * 2, a0 * 3)
        v1, a1 = v0 + .1, a0 + .2
        cache.record_forward(v1, a1, None, v1 * 2, a1 * 3)
        cache.step = 4
        video, audio = v1 + .001, a1 + .002
        actual = cache.skip_forward(video, audio, None)
        assert actual is not None and cache.hits == [4]
        assert torch.equal(actual[0], video + cache.prev_diff[0])
        assert torch.equal(actual[1], audio + cache.prev_diff[1])


@pytest.mark.parametrize("interrupted", [False, True])
def test_nested_scope_and_exception_restore_exact_class(cache_module, interrupted):
    original = cache_module.StepCache
    try:
        with snapshot_history(cache_module):
            outer = cache_module.StepCache
            with snapshot_history(cache_module):
                assert cache_module.StepCache is not outer
            assert cache_module.StepCache is outer
            if interrupted:
                raise InterruptedError("actual cache scope interruption")
    except InterruptedError:
        assert interrupted
    assert cache_module.StepCache is original


def test_disabled_scope_and_block_cache_are_not_reconfigured(cache_module):
    original = cache_module.StepCache
    with snapshot_history(cache_module, enabled=False):
        assert cache_module.StepCache is original
    with snapshot_history(cache_module):
        block_cache = cache_module.StepCache("fbcache", .1, 20)
        assert "history_policy" not in block_cache.stats()
        assert block_cache.block_level and block_cache.threshold == .1
    assert cache_module.StepCache is original
