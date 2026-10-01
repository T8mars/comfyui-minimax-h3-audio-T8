from __future__ import annotations

from copy import deepcopy

import comfy.nested_tensor
import pytest
import torch

from h3_audio_t8_pkg.modular_sampling.speed_transition import transition_speed_stage
from h3_audio_t8_pkg.speed_advanced import (
    build_speed_plan,
    dct_expand_official,
    recover_raw_flow_state,
    reindex_joint_audio_state,
    solve_segment_noise,
)


def _plan():
    return build_speed_plan(
        width=128,
        height=128,
        steps=20,
        scales="0.4,0.7,1.0",
        transition_mode="manual_sigmas",
        manual_transition_sigmas="0.94,0.78",
        delta=0.01,
        shift_video=12.0,
        transform="dct",
        profile_policy="require_validated_profile",
        fallback_policy="error",
    )[0]


def _pair(plan, stage_index):
    before = plan["stages"][stage_index]
    after = plan["stages"][stage_index + 1]
    generator = torch.Generator().manual_seed(90 + stage_index)

    def random(*shape):
        return torch.randn(shape, generator=generator)

    output = comfy.nested_tensor.NestedTensor((
        random(1, 24, 2, before["latent_height"], before["latent_width"]),
        random(1, 32, 2, 8),
    ))
    target = {"samples": comfy.nested_tensor.NestedTensor((
        random(1, 24, 2, after["latent_height"], after["latent_width"]),
        random(1, 32, 2, 8),
    ))}
    return output, target


@pytest.mark.parametrize("stage_index", [0, 1])
def test_explicit_speed_transition_matches_original_runner_math(stage_index):
    plan = _plan()
    output, next_latent = _pair(plan, stage_index)
    transition = plan["transitions"][stage_index]
    audio_scale = 3.0
    noise_scale = 1.25
    seed = 12345
    actual, report = transition_speed_stage(
        plan, stage_index, output, next_latent,
        audio_scale=audio_scale, noise_scale=noise_scale, seed=seed, dct_chunk_size=7,
    )

    raw = recover_raw_flow_state(
        output, sigma=transition["sigma"], audio_scale=audio_scale
    )
    raw_video, raw_audio = raw.unbind()
    next_shape = plan["stages"][stage_index + 1]
    expanded_video, sigma_to, _ = dct_expand_official(
        raw_video, next_shape["latent_height"], next_shape["latent_width"],
        sigma=transition["sigma"], ratio=transition["ratio"],
        seed=seed + (stage_index + 1) * 10_000, chunk_size=7,
    )
    target_video, target_audio = next_latent["samples"].unbind()
    expected_audio = reindex_joint_audio_state(
        raw_audio, target_audio * audio_scale,
        sigma_from=transition["sigma"], sigma_to=sigma_to,
    )
    expected = solve_segment_noise(
        comfy.nested_tensor.NestedTensor((expanded_video, expected_audio)),
        next_latent["samples"], sigma=sigma_to,
        audio_scale=audio_scale, noise_scale=noise_scale,
    )
    for got, wanted in zip(actual.unbind(), expected.unbind()):
        assert torch.equal(got, wanted)
    again, _ = transition_speed_stage(
        plan, stage_index, output, next_latent,
        audio_scale=audio_scale, noise_scale=noise_scale, seed=seed, dct_chunk_size=7,
    )
    for got, repeated in zip(actual.unbind(), again.unbind()):
        assert torch.equal(got, repeated)
    assert report["from_stage"] == stage_index
    assert report["to_stage"] == stage_index + 1
    assert report["sampled"] is False
    assert report["cache_reuse_authorized"] is False
    assert report["audio_spatial_noise_expansion"] is False


def test_transition_rejects_wrong_stage_canvas_and_tampered_schedule():
    plan = _plan()
    output, next_latent = _pair(plan, 0)
    kwargs = {"audio_scale": 3.0, "noise_scale": 1.0, "seed": 8}
    with pytest.raises(ValueError, match="out of range"):
        transition_speed_stage(plan, 2, output, next_latent, **kwargs)
    with pytest.raises(ValueError, match="planned canvas"):
        wrong_output, _ = _pair(plan, 1)
        transition_speed_stage(plan, 0, wrong_output, next_latent, **kwargs)
    changed = deepcopy(plan)
    changed["segments"][1]["sigmas"][0] += 0.01
    with pytest.raises(ValueError, match="segment endpoints"):
        transition_speed_stage(changed, 0, output, next_latent, **kwargs)
    with pytest.raises(ValueError, match="finite and positive"):
        transition_speed_stage(plan, 0, output, next_latent, **{**kwargs, "noise_scale": 0.0})
    bad_audio = {"samples": comfy.nested_tensor.NestedTensor((
        next_latent["samples"].unbind()[0], torch.zeros(1, 32, 2, 9),
    ))}
    with pytest.raises(ValueError, match="AV batch"):
        transition_speed_stage(plan, 0, output, bad_audio, **kwargs)
