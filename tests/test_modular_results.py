"""Typed stage evidence comes from a real single native sampler execution."""
import json

import pytest
import torch
from comfy_extras.nodes_custom_sampler import BasicGuider, RandomNoise, SamplerCustomAdvanced

from h3_audio_t8_pkg.modular_sampling.fast_h3_v2 import build_stage
from h3_audio_t8_pkg.modular_sampling.results import sample_stage
from h3_audio_t8_pkg.modular_sampling.eav import EAVConfig, apply_stage_eav
from test_fast_h3_v2_core_sampler import model
from test_progressive_sampling_runtime import conditioning, latent


def inputs(stage="low_0_4", profile="dense_compat_exp"):
    source = latent()
    prepared, sampler, sigmas, context, _ = build_stage(model(), source, stage, profile)
    guider = BasicGuider.execute(prepared, conditioning()).result[0]
    noise = RandomNoise.execute(31).result[0]
    return noise, guider, sampler, sigmas, source, context


@pytest.mark.parametrize("stage", ["low_0_4", "high_4_8"])
@pytest.mark.parametrize("profile", ["dense_compat_exp", "trained_vsa_exp"])
def test_capture_keeps_native_outputs_bit_exact_and_observes_actual_stage(stage, profile):
    args = inputs(stage, profile)
    reference = SamplerCustomAdvanced.execute(*args[:5]).result
    actual = sample_stage(*args)
    for output, expected in zip(actual[:2], reference):
        for value, wanted in zip(output["samples"].unbind(), expected["samples"].unbind()):
            assert torch.equal(value, wanted)
    receipt = actual[2].verify()
    assert receipt["execution"]["callbacks"] == [0, 1, 2, 3]
    assert receipt["execution"]["denoiser_evaluations"] == 4
    assert receipt["verified_recipe_completion"] is True
    assert receipt["cache_reuse_authorized"] is False
    assert receipt["request"]["stage_context"]["stage"] == stage
    assert actual[2].output is actual[0] and actual[2].denoised_output is actual[1]
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("which", [0, 1])
def test_downstream_tensor_mutation_cannot_keep_a_valid_result_receipt(which):
    result = sample_stage(*inputs())
    result[which]["samples"].unbind()[0].add_(.1)
    with pytest.raises(ValueError, match="mutated"):
        result[2].verify()


def test_noise_change_invalidates_request_but_not_native_source_latent():
    args = list(inputs())
    original = args[4]["samples"].unbind()[0].clone()
    first = sample_stage(*args)[2].verify()
    args[0] = RandomNoise.execute(32).result[0]
    second = sample_stage(*args)[2].verify()
    assert first["request"]["noise"] != second["request"]["noise"]
    assert first["request_sha256"] != second["request_sha256"]
    assert torch.equal(args[4]["samples"].unbind()[0], original)


def test_original_backend_error_never_produces_a_completion_result():
    args = list(inputs())

    def fail(*unused, **kwargs):
        raise RuntimeError("intentional interrupted sampler")

    args[2].sampler_function = fail
    with pytest.raises(RuntimeError, match="intentional interrupted sampler"):
        sample_stage(*args)


def test_unknown_sampler_runs_fresh_but_is_not_certified_as_the_v2_recipe():
    args = list(inputs())

    def passthrough(denoiser, x, sigmas, **kwargs):
        return x

    args[2].sampler_function = passthrough
    _, _, result, _ = sample_stage(*args)
    receipt = result.verify()
    assert receipt["verified_recipe_completion"] is False
    assert receipt["portable_identity"] is False
    assert receipt["execution"]["denoiser_evaluations"] == 0


def test_external_eav_executes_and_binds_its_authenticated_portable_identity():
    noise, guider, sampler, sigmas, source, context = inputs("high_4_8")
    applied, runtime, _ = apply_stage_eav(guider.model_patcher, sigmas, source, context, EAVConfig(tau=.2))
    guider = BasicGuider.execute(applied, conditioning()).result[0]
    _, _, result, raw = sample_stage(noise, guider, sampler, sigmas, source, context)
    receipt = json.loads(raw)
    assert runtime.snapshot()["completed_forwards"] == 4
    assert receipt["verified_recipe_completion"] is True
    assert receipt["portable_identity"] is True
    assert receipt["request"]["model"]["stage_effects"]["eav"]["tau"] == .2
    assert result.verify()["outputs"] == receipt["outputs"]


@pytest.mark.parametrize("profile", ["dense_compat_exp", "trained_vsa_exp"])
def test_cold_warm_and_rebuilt_native_request_identity_are_equal_without_unpatching(profile):
    args = inputs(profile=profile)
    first = sample_stage(*args)[2].verify()
    prepared = args[1].model_patcher
    installed = prepared.model.model_sampling
    backups = dict(prepared.object_patches_backup)
    # A fresh guider creates different Core allocation UUIDs, not new conditions.
    second_args = list(args)
    second_args[1] = BasicGuider.execute(prepared, conditioning()).result[0]
    second = sample_stage(*second_args)[2].verify()
    rebuilt = sample_stage(*inputs(profile=profile))[2].verify()
    assert first["portable_identity"] is second["portable_identity"] is rebuilt["portable_identity"] is True
    assert first["request_sha256"] == second["request_sha256"] == rebuilt["request_sha256"]
    assert prepared.model.model_sampling is installed
    assert prepared.object_patches_backup == backups


@pytest.mark.parametrize("mutation", ["weight", "callback", "condition", "sampling_buffer", "sigmas", "cfg"])
def test_real_during_run_mutations_still_reject_completion(mutation):
    args = inputs()
    guider = args[1]
    original = guider.sample

    def mutate_after(*items, **kwargs):
        value = original(*items, **kwargs)
        if mutation == "weight":
            with torch.no_grad():
                next(guider.model_patcher.model.diffusion_model.parameters()).add_(.01)
        elif mutation == "callback":
            guider.model_patcher.model_options["sampler_post_cfg_function"] = [lambda state: state["denoised"]]
        elif mutation == "condition":
            guider.original_conds["positive"][0]["cross_attn"].add_(.01)
        elif mutation == "sampling_buffer":
            guider.model_patcher.object_patches["model_sampling"].sigmas.add_(.01)
        elif mutation == "sigmas":
            args[3][0] -= .01
        elif mutation == "cfg":
            guider.cfg = 2.
        return value

    guider.sample = mutate_after
    with pytest.raises(ValueError, match="changed during sampling"):
        sample_stage(*args)


def test_unknown_native_sampling_extension_is_preserved_but_not_portable():
    args = inputs()
    selected = args[1].model_patcher.object_patches["model_sampling"]
    selected.foreign_observer = object()
    first = sample_stage(*args)[2].verify()
    second = sample_stage(*args)[2].verify()
    assert not first["portable_identity"] and not second["portable_identity"]
    assert selected.foreign_observer is not None


def test_real_nonzero_lora_keeps_cold_hot_identity_and_parameter_binding():
    from comfy.weight_adapter.lora import LoRAAdapter
    bare = model()
    key, weight = next((key, value) for key, value in bare.model_state_dict().items()
                       if key.endswith("weight") and value.ndim == 2)
    adapter = LoRAAdapter({"up", "down"}, (torch.ones(weight.shape[0], 1) * .02,
                                          torch.ones(1, weight.shape[1]) * .02, 1., None, None, None))
    assert bare.add_patches({key: adapter}, strength_patch=.7) == [key]
    source = latent()
    prepared, sampler, sigmas, context, _ = build_stage(bare, source, "low_0_4", "dense_compat_exp")
    guider = BasicGuider.execute(prepared, conditioning()).result[0]
    args = (RandomNoise.execute(31).result[0], guider, sampler, sigmas, source, context)
    first = sample_stage(*args)[2].verify()
    second = sample_stage(*args)[2].verify()
    assert key in prepared.backup and prepared.patches[key][0][1] is adapter
    assert first["portable_identity"] is second["portable_identity"] is True
    assert first["request_sha256"] == second["request_sha256"]
    unpatched = sample_stage(*inputs())[2].verify()
    assert first["request"]["model"] != unpatched["request"]["model"]
