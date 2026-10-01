"""Real tiny Core sampling parity, not trained-model or learned-upscaler QA."""
from dataclasses import FrozenInstanceError, replace
import json

import pytest
import torch

from h3_audio_t8_pkg import fast_h3_v2_advanced as legacy
from h3_audio_t8_pkg.modular_sampling.contracts import StageContext
from h3_audio_t8_pkg.modular_sampling.fast_h3_v2 import build_stage
from h3_audio_t8_pkg.modular_sampling.nodes import MiniMaxH3FastH3V2StageSetupEXPT8
from test_fast_h3_v2_core_sampler import model, sample
from test_progressive_sampling_runtime import latent


@pytest.mark.parametrize("profile", ["dense_compat_exp", "trained_vsa_exp"])
def test_two_public_stage_setups_match_existing_private_absolute_windows(profile):
    bare, source = model(), latent()
    low, sampler, sigmas, context, raw = build_stage(bare, source, "low_0_4", profile)
    reference, ref_sampler, full, _ = legacy.build_fast_h3_v2_setup(bare, source, profile)
    ref_sampler.extra_options.update(stage_start=0, stage_end=4)
    low_result = sample(low, sampler, sigmas, source)
    expected = sample(reference, ref_sampler, full[:5], source)
    for actual_latent, expected_latent in zip(low_result, expected):
        for actual, wanted in zip(actual_latent["samples"].unbind(), expected_latent["samples"].unbind()):
            assert torch.equal(actual, wanted)
    assert sigmas[-1] > 0
    assert context.start == 0 and context.end == 4 and context.steps == 4
    report = json.loads(raw)
    assert report["sampled"] is False and report["cache_reuse_authorized"] is False
    assert report["planned_nfe"] == 4
    # Real learned upscaler is a separate existing node, not replaced here.
    # This checks the stage math with same-size x0 and does not qualify upscale.
    high_source = low_result[1]
    high, high_sampler, high_sigmas, high_context, _ = build_stage(bare, high_source, "high_4_8", profile)
    ref_high, ref_high_sampler, ref_ladder, _ = legacy.build_fast_h3_v2_setup(bare, high_source, profile)
    ref_high_sampler.extra_options.update(stage_start=4, stage_end=8)
    actual = sample(high, high_sampler, high_sigmas, high_source)[0]
    expected = sample(ref_high, ref_high_sampler, ref_ladder[4:], high_source)[0]
    for value, wanted in zip(actual["samples"].unbind(), expected["samples"].unbind()):
        assert torch.equal(value, wanted)
        assert torch.isfinite(value).all()
    assert not torch.equal(actual["samples"].unbind()[1], high_source["samples"].unbind()[1])
    assert high_context.start == 4 and high_context.end == 8
    assert high_sigmas[-1] == 0
    assert legacy.capture_fast_h3_v2_owner(low).runtime is not legacy.capture_fast_h3_v2_owner(high).runtime
    assert not bare.wrappers and not bare.object_patches
    assert not torch.cuda.is_initialized()


def test_context_is_immutable_roundtrippable_and_not_a_completion_receipt():
    _, _, sigmas, context, _ = build_stage(model(), latent())
    restored = StageContext.from_dict(context.to_dict())
    assert restored == context
    assert restored.descriptor_sha256 == context.descriptor_sha256
    with pytest.raises(FrozenInstanceError):
        context.start = 1
    with pytest.raises(ValueError):
        replace(context, trajectory_sigmas=list(context.trajectory_sigmas))
    with pytest.raises(ValueError):
        replace(context, video_shape=(False, 24))
    with pytest.raises(ValueError):
        replace(context, start=True)
    sigmas[0] = 0
    assert context.trajectory_sigmas[0] > 0


@pytest.mark.parametrize("arguments", [{"stage": "full8"}, {"profile": "official_comfy_template_exp"}])
def test_different_recipe_is_not_silently_reinterpreted(arguments):
    with pytest.raises(ValueError):
        build_stage(model(), latent(), **arguments)


def test_public_ports_keep_native_sampler_and_editable_model_branch():
    info = MiniMaxH3FastH3V2StageSetupEXPT8.GET_NODE_INFO_V1()
    assert info["output"] == ["MODEL", "SAMPLER", "SIGMAS", "T8_STAGE_CONTEXT", "STRING"]
    assert list(info["input"]["required"]) == ["model", "av_latent", "stage", "profile", "min_tokens"]
