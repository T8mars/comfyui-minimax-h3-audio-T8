"""Existing Core RES joint AV and its latent-only resume boundary; CPU only."""
from types import SimpleNamespace

import pytest
import torch
import comfy.k_diffusion.sampling as core_solver
import comfy.model_sampling
import comfy.samplers
from comfy_extras.nodes_custom_sampler import BasicGuider, RandomNoise, SamplerCustomAdvanced

from h3_audio_t8_pkg import sampling, trajectory_probe_advanced as trajectory
from test_fast_h3_v2_core_sampler import model
from test_modular_rf_restart import source_latent
from test_progressive_sampling_runtime import conditioning


def test_existing_core_res_executes_joint_AV_with_native_common_clock_without_input_mutation():
    bare, source = model(), source_latent(masked=True)
    before = [value.clone() for value in source['samples'].unbind()]
    masks = [value.clone() for value in source['noise_mask'].unbind()]
    branch, sampler, sigmas = sampling.setup_dual_clock_sampling(
        bare, source, 8, 12., 3., sampler_name='res_multistep', scheduler='native_flow')
    selected = branch.get_model_object('model_sampling')
    assert isinstance(selected, comfy.model_sampling.ModelSamplingAV)
    assert selected.shift == 12. and selected.audio_shift == 3.
    assert sampler.sampler_function is core_solver.sample_res_multistep
    assert torch.equal(sigmas, sampling.native_flow_sigmas(8, 12.))
    output, prediction = SamplerCustomAdvanced.execute(
        RandomNoise.execute(2610070101).result[0], BasicGuider.execute(branch, conditioning()).result[0],
        sampler, sigmas, source).result
    for result in (output, prediction):
        for actual, original in zip(result['samples'].unbind(), before, strict=True):
            assert actual.shape == original.shape and torch.isfinite(actual).all()
    assert all(torch.equal(a, b) for a, b in zip(source['samples'].unbind(), before, strict=True))
    assert all(torch.equal(a, b) for a, b in zip(source['noise_mask'].unbind(), masks, strict=True))
    assert not bare.object_patches and not torch.cuda.is_initialized()


class NonlinearDenoiser:
    """Analytic per-element CPU witness, not a trained H3 or vLLM double."""
    def __init__(self):
        self.inner_model = SimpleNamespace(model_patcher=SimpleNamespace(
            get_model_object=lambda name: SimpleNamespace(noise_scale=1.)))
        self.calls = 0

    def __call__(self, values, sigma, **options):
        self.calls += 1
        return .2 * values + .03 * values.square() + .3 * sigma[:, None].sin()


def test_actual_core_multistep_latent_only_split_loses_history_but_common_clock_packing_is_exact():
    source = torch.tensor([[.2, -.1, .4, -.3]], dtype=torch.float32)
    sigmas = torch.tensor([1., .88, .71, .59, .43, .32, .19, .08, 0.])
    full_model, callbacks = NonlinearDenoiser(), []
    full = core_solver.sample_res_multistep(full_model, source.clone(), sigmas, disable=True,
                                          callback=lambda values: callbacks.append(values['i']))
    assert full_model.calls == 8 and callbacks == list(range(8))
    # Carry the actual x_sigma, not x0, but no old_denoised/old_sigma_down.
    first = core_solver.sample_res_multistep(NonlinearDenoiser(), source.clone(), sigmas[:5], disable=True)
    restarted = core_solver.sample_res_multistep(NonlinearDenoiser(), first, sigmas[4:], disable=True)
    assert not torch.equal(full, restarted)
    assert (full - restarted).abs().max().item() > 1e-6
    # The deterministic eta=0 solver itself can keep two streams on one clock;
    # this is not equivalence to two separate physical H3 clocks or vLLM solvers.
    independent = torch.cat([core_solver.sample_res_multistep(NonlinearDenoiser(), part.clone(),
        sigmas, disable=True) for part in source.split(2, dim=1)], dim=1)
    assert torch.equal(full, independent)
    assert not torch.cuda.is_initialized()


def test_existing_euler_trajectory_guard_does_not_accept_RES_latent_only_resume():
    with pytest.raises(ValueError, match='hidden history'):
        trajectory._sampler_identity(comfy.samplers.sampler_object('res_multistep'))
    with pytest.raises(ValueError, match='hidden history'):
        trajectory._sampler_identity(comfy.samplers.sampler_object('res_multistep_ancestral'))
    branch, _, _ = sampling.setup_dual_clock_sampling(model(), source_latent(), 8, 12., 3.,
                                                      sampler_name='res_multistep', scheduler='native_flow')
    with pytest.raises(ValueError, match='stable dual_clock_euler'):
        trajectory.prepare_trajectory_model(branch)
    assert not torch.cuda.is_initialized()
