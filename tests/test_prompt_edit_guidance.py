from copy import deepcopy
import json

import pytest
import torch
from comfy_extras.nodes_custom_sampler import BasicGuider, RandomNoise, SamplerCustomAdvanced

from h3_audio_t8_pkg.prompt_edit_guidance import PromptEditGuider, sampling_step, validate_text_only_edit
from h3_audio_t8_pkg.sampling import setup_dual_clock_sampling
from test_fast_h3_v2_core_sampler import model
from test_progressive_sampling_runtime import conditioning, latent


def edited():
    return [[torch.full((1, 5, 8), 0.1), {"minimax_token_tags": torch.ones(5, dtype=torch.long)}]]


def sample(guider, sampler, sigmas, source):
    return SamplerCustomAdvanced.execute(RandomNoise.execute(51).result[0], guider, sampler, sigmas, source).result[0]


@pytest.mark.parametrize("strength", [0.0, 1.0])
def test_real_core_zero_one_match_basic_with_different_text_lengths(strength):
    source = latent()
    selected, sampler, sigmas = setup_dual_clock_sampling(model(), source, 4, 12.0, 3.0)
    original, changed = conditioning(), edited()
    expected = sample(BasicGuider.execute(selected, original if strength == 0 else changed).result[0], sampler, sigmas, source)
    guider = PromptEditGuider(selected, original, changed, strength)
    actual = sample(guider, sampler, sigmas, source)
    for a, b in zip(actual["samples"].unbind(), expected["samples"].unbind()):
        assert torch.equal(a, b)
    report = json.loads(guider.report())
    assert len(report["actual_predict_calls"]) == 4
    assert report["physical_model_forwards"] == 4


def test_real_core_delayed_two_branch_guidance_consumes_schedule_and_media():
    source = latent()
    selected, sampler, sigmas = setup_dual_clock_sampling(model(), source, 4, 12.0, 3.0)
    original, changed = conditioning("i2va"), edited()
    changed[0][1].update({k: v for k, v in original[0][1].items() if k != "minimax_token_tags"})
    changed[0][1]["minimax_token_tags"][0] = 0
    guider = PromptEditGuider(selected, original, changed, 1.5, first_step=2)
    actual = sample(guider, sampler, sigmas, source)
    assert all(torch.isfinite(x).all() for x in actual["samples"].unbind())
    report = json.loads(guider.report())
    assert [len(x["branches"]) for x in report["actual_predict_calls"]] == [1, 1, 2, 2]
    assert report["physical_model_forwards"] == 6
    assert "model_function_wrapper" not in selected.model_options


def test_equal_reference_counts_with_changed_pixels_are_rejected():
    original = conditioning("i2va")
    changed = deepcopy(original)
    changed[0][1]["minimax_keyframes"][0]["latent"][0, 0, 0, 0, 0] += 0.1
    with pytest.raises(ValueError, match="changed media"):
        validate_text_only_edit(original, changed)


@pytest.mark.parametrize("key,value", [("minimax_frame_count", 17), ("clock", {"origin": 2.0})])
def test_time_and_nontext_metadata_cannot_be_silently_changed(key, value):
    original, changed = conditioning(), conditioning()
    changed[0][1][key] = value
    with pytest.raises(ValueError, match="changed media"):
        validate_text_only_edit(original, changed)


def test_actual_step_uses_schedule_intervals_not_number_of_model_calls():
    options = {"transformer_options": {"sample_sigmas": torch.tensor([1., .7, .2, 0.])}}
    assert [sampling_step(torch.tensor([x]), options) for x in [1., .8, .7, .5, .2]] == [0, 0, 1, 1, 2]
    with pytest.raises(ValueError, match="actual sample_sigmas"):
        sampling_step(torch.tensor([.5]), {})


def test_existing_model_wrapper_is_delegated():
    source = latent()
    selected, sampler, sigmas = setup_dual_clock_sampling(model(), source, 2, 12., 3.)
    calls = []
    def wrapper(apply, options):
        calls.append(True)
        return apply(options["input"], options["timestep"], **options["c"])
    selected.set_model_unet_function_wrapper(wrapper)
    guider = PromptEditGuider(selected, conditioning(), edited(), 1.1)
    sample(guider, sampler, sigmas, source)
    assert len(calls) == guider.physical_model_forwards == 4


def test_cached_guider_reuse_reports_current_invocation_only():
    source = latent()
    selected, sampler, sigmas = setup_dual_clock_sampling(model(), source, 2, 12., 3.)
    guider = PromptEditGuider(selected, conditioning(), edited(), 1.1)
    first = sample(guider, sampler, sigmas, source)
    second = sample(guider, sampler, sigmas, source)
    assert len(guider.calls) == 2 and guider.physical_model_forwards == 4
    assert all(torch.equal(a, b) for a, b in zip(first["samples"].unbind(), second["samples"].unbind()))
