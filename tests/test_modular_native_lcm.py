"""One opt-in Core LCM stage; no ELM weights or subjective quality claim."""
from copy import copy
import json
from types import SimpleNamespace

import torch
import comfy.k_diffusion.sampling as k_sampling
import comfy.model_sampling
import comfy.samplers
from comfy_extras.nodes_custom_sampler import BasicGuider, RandomNoise, SamplerCustomAdvanced

from h3_audio_t8_pkg import sampling
from h3_audio_t8_pkg.modular_sampling import native_explicit as native, eav
from h3_audio_t8_pkg.modular_sampling.results import sample_stage
from h3_audio_t8_pkg.modular_sampling.storage import load_stage, save_stage
from test_fast_h3_v2_core_sampler import model
from test_modular_rf_restart import source_latent
from test_progressive_sampling_runtime import conditioning


def test_native_lcm_keeps_core_math_masks_and_frozen_completion(tmp_path):
    bare, source = model(), source_latent(True)
    configured, sampler, sigmas = sampling.setup_dual_clock_sampling(bare, source, 4, 12., 3., "lcm")
    noise = RandomNoise.execute(31).result[0]
    expected = SamplerCustomAdvanced.execute(noise, BasicGuider.execute(configured, conditioning()).result[0],
                                             sampler, sigmas, source).result
    bound, selected, table, context, report = native.bind_stage(configured, sampler, sigmas, source)
    assert selected is sampler and table is sigmas
    assert json.loads(context.profile)["sampler_kind"] == "lcm_exact_source"
    assert json.loads(report)["eav_forward_coverage_adapter"] == "core_lcm_exact_source"
    bound, runtime, _ = eav.apply_stage_eav(bound, sigmas, source, context,
        eav.EAVConfig("report_only", tau=.2, g_hard_limit=3.))
    result = sample_stage(noise, BasicGuider.execute(bound, conditioning()).result[0],
                          selected, sigmas, source, context)[2]
    for actual, reference in zip((result.output, result.denoised_output), expected):
        assert all(torch.equal(a, b) for a, b in zip(actual["samples"].unbind(), reference["samples"].unbind()))
    receipt = result.verify()
    assert receipt["verified_recipe_completion"] is receipt["portable_identity"] is True
    assert receipt["execution"]["callbacks"] == [0, 1, 2, 3]
    assert receipt["execution"]["denoiser_evaluations"] == 4
    assert runtime.snapshot()["status"] == "observed_report_only"
    path, digest, _ = save_stage(result, tmp_path)
    restored = load_stage(tmp_path, path, digest, "native_low")[0]
    assert all(torch.equal(a, b) for a, b in zip(restored["samples"].unbind(), result.output["samples"].unbind()))
    assert all(torch.equal(a, b) for a, b in zip(restored["noise_mask"].unbind(), source["noise_mask"].unbind()))
    assert not bare.object_patches and not torch.cuda.is_initialized()


def test_core_lcm_fresh_transition_matches_both_modality_formulas():
    class NativeAV(comfy.model_sampling.ModelSamplingAV, comfy.model_sampling.CONST):
        pass
    selected = NativeAV()
    selected.set_parameters(shift=12., audio_shift=3.)
    clean = torch.tensor([[[.2, .3, .7, .8]]])
    packed_clean = clean.clone()
    packed_clean[..., 2:] *= 4.
    states, draws = [], []

    def denoiser(x, sigma, **kwargs):
        states.append(x.clone())
        return packed_clean.clone()

    denoiser.inner_model = SimpleNamespace(model_patcher=SimpleNamespace(get_model_object=lambda name: selected))

    def noise_sampler(sigma, next_sigma):
        value = torch.tensor([[[.1, -.2, .4, -.5]]]) * (len(draws) + 1)
        draws.append(value)
        return value

    sigmas = sampling.native_flow_sigmas(4, 12.)
    result = k_sampling.sample_lcm(denoiser, torch.zeros_like(clean), sigmas,
                                   disable=True, noise_sampler=noise_sampler)
    assert len(states) == 4 and len(draws) == 3 and torch.equal(result, packed_clean)
    for i in range(1, 4):
        sv = sigmas[i]
        sa = sampling.time_shift_sigma(sv, 12., 3.)
        assert torch.allclose(states[i][..., :2], (1-sv)*clean[..., :2] + sv*draws[i-1][..., :2])
        actual_audio = states[i][..., 2:] / (4-3*sv)
        assert torch.allclose(actual_audio, (1-sa)*clean[..., 2:] + sa*draws[i-1][..., 2:], atol=1e-6)


def test_modified_lcm_is_unverified_and_bound_forward_plan_loses_coverage(monkeypatch):
    sampler = comfy.samplers.sampler_object("lcm")
    assert native._exact_source_lcm(sampler)
    for attribute, value in (("extra_options", {"s_noise": .9}), ("inpaint_options", {"random": True}),
                             ("sampler_function", lambda *args, **kwargs: None)):
        changed = copy(sampler)
        setattr(changed, attribute, value)
        assert not native._exact_source_lcm(changed)
    source = source_latent()
    configured, sampler, sigmas = sampling.setup_dual_clock_sampling(model(), source, 4, 12., 3., "lcm")
    context = native.bind_stage(configured, sampler, sigmas, source)[3]
    monkeypatch.setattr(native, "_lcm_core_source_hash", lambda: "changed")
    assert not native._exact_source_lcm(sampler)
    assert native.forward_plan(context)["known"] is False
