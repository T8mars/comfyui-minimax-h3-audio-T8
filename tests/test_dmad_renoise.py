import copy
import json
from types import SimpleNamespace

import pytest
import torch
import comfy.nested_tensor
from comfy.k_diffusion.sampling import sample_lcm
from comfy_extras.nodes_custom_sampler import (
    BasicGuider,
    RandomNoise,
    SamplerCustomAdvanced,
)

from h3_audio_t8_pkg.dmad_renoise import prepare_dmad_renoise
from h3_audio_t8_pkg.sampling import native_flow_sigmas, time_shift_sigma
from test_fast_h3_v2_core_sampler import model
from test_progressive_sampling_runtime import conditioning, latent


@pytest.mark.parametrize("steps", [4, 8])
def test_native_consumer_matches_independent_two_clock_renoise_oracle(steps):
    selected, sampler, sigmas, report = prepare_dmad_renoise(model(), latent(), steps)
    sampling = selected.get_model_object("model_sampling")
    # The packed audio is carried on the video clock. Compare decoded inputs
    # with direct audio-clock flow updates, not with the packed carrier itself.
    scale = sampling.audio_scale
    seen, noises = [], []

    class Oracle:
        inner_model = SimpleNamespace(
            model_patcher=SimpleNamespace(get_model_object=lambda _key: sampling)
        )

        def __call__(self, x, sigma, **kwargs):
            seen.append(x.clone())
            clean = x.new_tensor([[len(seen) / 10, scale * len(seen) / 20]])
            return clean

    def noise(_current, _next):
        value = torch.tensor([[len(noises) / 7 + 0.1, -len(noises) / 11 - 0.2]])
        noises.append(value)
        return value

    result = sample_lcm(
        Oracle(), torch.zeros(1, 2), sigmas, noise_sampler=noise, disable=True
    )
    assert len(seen) == steps and len(noises) == steps - 1
    for i in range(1, steps):
        sv = sigmas[i]
        sa = time_shift_sigma(sv, 12.0, 2.0)
        expected_video = (1 - sv) * (i / 10) + sv * noises[i - 1][0, 0]
        expected_audio = (1 - sa) * (i / 20) + sa * noises[i - 1][0, 1]
        torch.testing.assert_close(seen[i][0, 0], expected_video)
        torch.testing.assert_close(
            seen[i][0, 1] / (sv + (1 - sv) * scale), expected_audio
        )
    torch.testing.assert_close(result, torch.tensor([[steps / 10, scale * steps / 20]]))
    assert torch.equal(sigmas, native_flow_sigmas(steps, 12.0))
    assert sampler.sampler_function is sample_lcm
    assert json.loads(report)["trained_GPU_quality_verified"] is False


@pytest.mark.parametrize("steps", [4, 8, 7])
def test_real_core_uses_requested_forwards_and_restores_sources(steps):
    bare, source, positive = model(), latent(), conditioning("i2va")
    before, options = copy.deepcopy(positive), copy.deepcopy(bare.model_options)
    selected, sampler, sigmas, _ = prepare_dmad_renoise(bare, source, steps)
    calls = []
    handle = bare.model.diffusion_model.register_forward_pre_hook(
        lambda *_: calls.append(1)
    )
    try:
        guider = BasicGuider.execute(selected, positive).result[0]
        output = SamplerCustomAdvanced.execute(
            RandomNoise.execute(913).result[0], guider, sampler, sigmas, source
        ).result[0]
    finally:
        handle.remove()
    assert len(calls) == steps
    assert all(torch.isfinite(x).all() for x in output["samples"].unbind())
    assert all(not x.count_nonzero() for x in source["samples"].unbind())
    assert bare.model_options == options and not bare.object_patches
    assert torch.equal(positive[0][0], before[0][0])
    assert torch.equal(
        positive[0][1]["minimax_keyframes"][0]["latent"],
        before[0][1]["minimax_keyframes"][0]["latent"],
    )


def test_frozen_audio_mask_is_still_consumed_by_native_sampler():
    source = latent()
    video, audio = source["samples"].unbind()
    audio.fill_(0.27)
    source["noise_mask"] = comfy.nested_tensor.NestedTensor(
        (torch.ones_like(video), torch.zeros_like(audio))
    )
    selected, sampler, sigmas, _ = prepare_dmad_renoise(model(), source, 4)
    guider = BasicGuider.execute(selected, conditioning()).result[0]
    output = SamplerCustomAdvanced.execute(
        RandomNoise.execute(913).result[0], guider, sampler, sigmas, source
    ).result[0]
    torch.testing.assert_close(output["samples"].unbind()[1], audio, rtol=0, atol=1e-6)


def test_unknown_wrapper_is_delegated_and_actual_errors_propagate():
    bare = model()
    calls = []

    def wrapper(function, inputs):
        calls.append(1)
        raise RuntimeError("selected user's wrapper failure")

    bare.set_model_unet_function_wrapper(wrapper)
    selected, sampler, sigmas, _ = prepare_dmad_renoise(bare, latent(), 1)
    assert selected.model_options["model_function_wrapper"] is wrapper
    with pytest.raises(RuntimeError, match="user's wrapper failure"):
        SamplerCustomAdvanced.execute(
            RandomNoise.execute(913).result[0],
            BasicGuider.execute(selected, conditioning()).result[0],
            sampler,
            sigmas,
            latent(),
        )
    assert calls


def test_schema_exposes_sampler_without_loading_adapter():
    from h3_audio_t8_pkg.nodes_dmad_renoise import MiniMaxH3DMADRenoiseSetupEXPT8

    assert (
        MiniMaxH3DMADRenoiseSetupEXPT8.define_schema().node_id
        == "MiniMaxH3DMADRenoiseSetupEXPT8"
    )
