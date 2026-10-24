"""Only the two bounded independent MC review topics: state and FP64 stats."""
import json
import math

import pytest
import torch

from h3_audio_t8_pkg import mc_seam_experimental as mc
from h3_audio_t8_pkg.nodes_mc_seam_experimental import MiniMaxH3MCJoinMatchEXPT8 as Join
from test_h21_mc_seam import latent, parts, preserved


class TaggedTuple(tuple):
    pass


class TaggedList(list):
    __slots__ = ("tag", "attrs")


@pytest.mark.parametrize("kind", [TaggedTuple, TaggedList])
def test_real_pair_subclass_keeps_its_own_metadata_without_writing_input(kind):
    source, refined = latent(kind=kind), latent(kind=kind, seed=19)
    source["samples"].tag = object()
    source["samples"].attrs = {"user": object()}
    original_container = source["samples"]
    video, audio = parts(source)
    before = video.clone()
    output, report = mc.refine_blend(source, refined, enabled=True)
    assert type(output["samples"]) is kind
    assert getattr(output["samples"], "tag", None) is source["samples"].tag
    assert getattr(output["samples"], "attrs", None) is source["samples"].attrs
    assert output["samples"] is not original_container
    assert source["samples"] is original_container and parts(source)[0] is video
    assert torch.equal(video, before) and parts(output)[1] is audio
    assert report["metadata_preserved"]
    preserved(source, output)


def _scaled_pair(scale):
    source, previous = latent(), latent(count=20, seed=23)
    for item in (source, previous):
        item["samples"].tensors[0] = parts(item)[0].double() * scale
    return source, previous


def _finite_report(value):
    if isinstance(value, float):
        assert math.isfinite(value)
    elif isinstance(value, dict):
        for item in value.values():
            _finite_report(item)
    elif isinstance(value, list):
        for item in value:
            _finite_report(item)


@pytest.mark.parametrize("full_statistics", [False, True])
def test_real_finite_FP64_can_produce_finite_join_and_strict_JSON_without_square_overflow(full_statistics):
    source, previous = _scaled_pair(1e200)
    before, prior_before = parts(source)[0].clone(), parts(previous)[0].clone()
    options = {"enabled": True, "tuning": "manual", "min_excess": 0,
               "variant": "source_whole_clip_exp", "match_spread": full_statistics,
               "match_detail": full_statistics, "context_latent": previous}
    output, text = Join.execute(source, **options).result
    report = json.loads(text)
    _finite_report(report)
    assert torch.isfinite(parts(output)[0]).all()
    if full_statistics:
        baseline, previous_baseline = _scaled_pair(1.)
        options["context_latent"] = previous_baseline
        expected, _ = Join.execute(baseline, **options).result
        # New stress oracle only: normalization of a 1e200 multiplication adds
        # FP64 rounding. Original ordinary378 source equality remains exact.
        torch.testing.assert_close(parts(output)[0] / 1e200, parts(expected)[0], rtol=1e-14, atol=1e-14)
    else:
        video, old = before, prior_before
        mean = video[0, :, 9].mean((-2, -1))
        step = mean - old[0, :, 19].mean((-2, -1))
        target = mean - step
        expected = video - (mean - target).view(1, -1, 1, 1, 1)
        assert torch.equal(parts(output)[0], expected)
    assert parts(output)[1] is parts(source)[1]
    assert torch.equal(parts(source)[0], before) and torch.equal(parts(previous)[0], prior_before)
    preserved(source, output)


def test_actual_unrepresentable_FP64_mean_step_explains_computation_not_generic_JSON():
    source, previous = _scaled_pair(1.)
    parts(source)[0].fill_(1e308)
    parts(previous)[0].fill_(-1e308)
    before, prior_before = parts(source)[0].clone(), parts(previous)[0].clone()
    with pytest.raises(ValueError, match=r"Join Match .*?(nonfinite|overflow|RMS|mean)"):
        Join.execute(source, enabled=True, context_latent=previous, tuning="manual",
                     min_excess=0, match_detail=False, match_spread=False)
    assert torch.equal(parts(source)[0], before) and torch.equal(parts(previous)[0], prior_before)
