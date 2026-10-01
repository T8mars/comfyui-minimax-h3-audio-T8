"""Real tiny Core comparison: hidden old RF branch vs independent restart only."""
import json

import pytest
import torch
import comfy.nested_tensor
from comfy_extras.nodes_custom_sampler import BasicGuider, RandomNoise, SamplerCustomAdvanced

from h3_audio_t8_pkg import detail_sampling_advanced as old
from h3_audio_t8_pkg.sampling import setup_dual_clock_sampling
from h3_audio_t8_pkg.modular_sampling import rf_restart as split
from test_fast_h3_v2_core_sampler import model
from test_progressive_sampling_runtime import latent, conditioning


def source_latent(masked=False, batch=1):
    source = latent()
    generator = torch.Generator().manual_seed(841)
    streams = [torch.randn((batch, *x.shape[1:]), generator=generator) * .2 for x in source["samples"].unbind()]
    source["samples"] = comfy.nested_tensor.NestedTensor(streams)
    source["batch_index"] = list(reversed(range(batch)))
    if masked:
        masks = [torch.ones_like(x) for x in streams]
        masks[0][:, :, :1] = 0.
        source["noise_mask"] = comfy.nested_tensor.NestedTensor(masks)
    return source


def run(prepared, sampler, sigmas, source, seed=73):
    guider = BasicGuider.execute(prepared, conditioning()).result[0]
    return SamplerCustomAdvanced.execute(RandomNoise.execute(seed).result[0], guider, sampler, sigmas, source).result


def mixer_kwargs():
    return dict(shift_video=12., shift_audio=3., enable_tail=True, extra_tail_steps=2,
        tail_spacing="video_sigma_linear", enable_model_time_bias=False, bias=-.1,
        bias_start_progress=.2, bias_end_progress=.8, bias_domain="video_sigma",
        enable_stg=False, stg_scale=.3, stg_double_blocks="0", stg_start_progress=.2,
        stg_end_progress=.8, enable_restart=True, restart_video_sigma=.15, restart_steps=3, restart_seed=17)


@pytest.mark.parametrize("entry", ["standalone", "detail_mixer", "two_pass_detail_mixer"])
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("batch_index", [0, 3])
@pytest.mark.parametrize("effects", ["none", "bias", "stg", "bias_stg"])
def test_split_matches_both_av_streams_and_denoised_for_all_old_entries(entry, masked, batch_index, effects):
    bare, source = model(), source_latent(masked)
    source["batch_index"] = [batch_index]
    kwargs = mixer_kwargs()
    kwargs.update(enable_model_time_bias="bias" in effects, enable_stg="stg" in effects)
    if entry == "standalone":
        # Standalone RF also accepts already-patched models; match mixer order.
        if kwargs["enable_model_time_bias"]:
            bare = old.setup_model_time_bias_sampling(bare, source, steps=4, shift_video=12., shift_audio=3.,
                bias=-.1, start_progress=.2, end_progress=.8, bias_domain="video_sigma")[0]
        if kwargs["enable_stg"]:
            bare = old.apply_h3_spatiotemporal_guidance(bare, scale=.3, double_blocks="0",
                start_progress=.2, end_progress=.8, shift_video=12., rescale=0.)[0]
    if entry == "standalone":
        prepared, sampler, sigmas, _ = old.setup_rectified_flow_restart_sampling(bare, source, steps=4,
            **{key: kwargs[key] for key in ("shift_video", "shift_audio", "restart_video_sigma", "restart_steps", "restart_seed")})
        base_model, base_sampler, base_sigmas = setup_dual_clock_sampling(bare, source, 4, 12., 3.)
    else:
        factory = old.setup_detail_mixer_sampling if entry == "detail_mixer" else old.setup_two_pass_detail_mixer_sampling
        extra = {"steps": 4, "profile": "custom_strict"} if entry == "detail_mixer" else {"refine_sigmas": torch.tensor([.5, .35, .1, 0.])}
        prepared, sampler, sigmas, *_ = factory(bare, source, **kwargs, **extra)
        base_model, base_sampler, base_sigmas, *_ = factory(bare, source, **{**kwargs, "enable_restart": False}, **extra)
    expected = run(prepared, sampler, sigmas, source)
    first = run(base_model, base_sampler, base_sigmas, source)[0]
    handoff = split.prepare_handoff(first, source)
    branch, restart, tail, raw = split.build_restart_stage(base_model, handoff, restart_seed=17)
    actual = run(branch, restart, tail, first)
    for left, right in zip(actual, expected):
        for a, b in zip(left["samples"].unbind(), right["samples"].unbind()):
            assert torch.equal(a, b), (entry, masked, batch_index, effects, (a - b).abs().max().item())
    assert json.loads(raw)["planned_stage_calls"] == 3
    assert json.loads(raw)["diffusion_calls_in_setup"] == 0
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("disabled", ["sigma", "steps"])
def test_disabled_restart_has_zero_calls_and_identity_samples(disabled, monkeypatch):
    bare, source = model(), source_latent(True)
    # Legacy bypass permits fractional masks; no stricter enabled-only check.
    source["noise_mask"].unbind()[0].fill_(.5)
    handoff = split.prepare_handoff(source, source)
    options = {"restart_video_sigma": 0.} if disabled == "sigma" else {"restart_steps": 0}
    prepared, sampler, sigmas, _ = split.build_restart_stage(bare, handoff, **options)
    assert sigmas.numel() == 0
    def forbidden(*args, **kwargs):
        raise AssertionError("Disabled RF must not call the model")
    monkeypatch.setattr(bare.model.diffusion_model, "forward", forbidden)
    actual = run(prepared, sampler, sigmas, source)[0]
    assert all(torch.equal(a, b) for a, b in zip(actual["samples"].unbind(), source["samples"].unbind()))


def test_handoff_mismatch_and_mutation_are_rejected():
    first, original = source_latent(), source_latent()
    handoff = split.prepare_handoff(first, original)
    original["samples"].unbind()[1].add_(.1)
    with pytest.raises(ValueError, match="changed"):
        handoff.verify()
    wrong = source_latent()
    wrong["samples"] = comfy.nested_tensor.NestedTensor((wrong["samples"].unbind()[0][..., :4], wrong["samples"].unbind()[1]))
    with pytest.raises(ValueError, match="matching layouts"):
        split.prepare_handoff(first, wrong)


def test_existing_single_batch_restriction_is_preserved():
    source = source_latent(batch=2)
    with pytest.raises(ValueError, match="batch size 1"):
        old.setup_rectified_flow_restart_sampling(model(), source, steps=4, shift_video=12., shift_audio=3.,
            restart_video_sigma=.15, restart_steps=3, restart_seed=17)
    with pytest.raises(ValueError, match="batch size 1"):
        split.prepare_handoff(source, source)


@pytest.mark.parametrize("mask", ["audio_locked", "video_fractional"])
def test_original_mask_restrictions_remain_real_errors(mask):
    source = source_latent(True)
    source["noise_mask"].unbind()[1 if mask == "audio_locked" else 0].fill_(0 if mask == "audio_locked" else .5)
    with pytest.raises(ValueError):
        split.build_restart_stage(model(), split.prepare_handoff(source, source))


@pytest.mark.parametrize("sigma,steps,restart_seed,video_shift,audio_shift,noise_scale", [
    (.01, 1, 0, 12., 3., 1.), (.15, 3, 17, 10., 2., .7),
    (.5, 8, 2**64 - 1, 8., 3., 1.), (.3, 2, 19, 3., 3., .4)])
@pytest.mark.parametrize("masked", [False, True])
def test_independent_restart_exact_calls_clocks_rng_and_original_noise_scale(
        sigma, steps, restart_seed, video_shift, audio_shift, noise_scale, masked):
    bare, source = model(), source_latent(masked)
    bare.model.model_sampling.set_noise_scale(noise_scale)
    options = dict(shift_video=video_shift, shift_audio=audio_shift,
                   restart_video_sigma=sigma, restart_steps=steps, restart_seed=restart_seed)
    old_model, old_sampler, old_sigmas, _ = old.setup_rectified_flow_restart_sampling(bare, source, steps=4, **options)
    expected = run(old_model, old_sampler, old_sigmas, source)[0]
    first_model, first_sampler, first_sigmas = setup_dual_clock_sampling(bare, source, 4, video_shift, audio_shift)
    first = run(first_model, first_sampler, first_sigmas, source)[0]
    branch, sampler, sigmas, _ = split.build_restart_stage(bare, split.prepare_handoff(first, source), **options)
    calls = []
    hook = bare.model.diffusion_model.register_forward_pre_hook(lambda *_: calls.append(1))
    try:
        actual = run(branch, sampler, sigmas, first)[0]
    finally:
        hook.remove()
    assert len(calls) == steps  # Not base + restart, not a hidden extra first pass.
    assert all(torch.equal(a, b) for a, b in zip(actual["samples"].unbind(), expected["samples"].unbind()))
    packed, _ = comfy.utils.pack_latents(first["samples"].unbind())
    before = torch.random.get_rng_state().clone()
    split._renoise(packed, video_values=first["samples"].unbind()[0].numel(),
                   video_sigma=sigma, audio_sigma=sigma, restart_seed=restart_seed, mask=None)
    assert torch.equal(before, torch.random.get_rng_state())


def test_replacing_original_anchor_with_completed_endpoint_would_change_legacy_audio():
    bare, source = model(), source_latent()
    prepared, sampler, sigmas = setup_dual_clock_sampling(bare, source, 4, 12., 3.)
    first = run(prepared, sampler, sigmas, source)[0]
    good = split.build_restart_stage(bare, split.prepare_handoff(first, source))
    wrong = split.build_restart_stage(bare, split.prepare_handoff(first, first))
    actual = run(*good[:3], first)[0]["samples"].unbind()[1]
    changed = run(*wrong[:3], first)[0]["samples"].unbind()[1]
    assert not torch.equal(actual, changed)
