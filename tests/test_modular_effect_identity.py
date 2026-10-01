"""Real CPU stage persistence with authenticated external effect owners."""
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from comfy.nested_tensor import NestedTensor
from comfy_extras.nodes_custom_sampler import BasicGuider, RandomNoise, SamplerCustomAdvanced

from h3_audio_t8_pkg.modular_sampling import eav
from h3_audio_t8_pkg.modular_sampling import effect_identity
from h3_audio_t8_pkg import prompt_relay_advanced as relay
from h3_audio_t8_pkg.prompt_relay_long_video_advanced import (
    PROMPT_RELAY_LONG_VIDEO_ATTACHMENT_KEY,
    PROMPT_RELAY_LONG_VIDEO_PROJECTION_SCHEMA,
)
from h3_audio_t8_pkg.patch_stack_policy import UnverifiedModelStack
from h3_audio_t8_pkg.vdn_attention_compat import _factory_closure
from h3_audio_t8_pkg.modular_sampling.fast_h3_v2 import build_stage
from h3_audio_t8_pkg.modular_sampling.results import sample_stage, selected_model_identity
from h3_audio_t8_pkg.modular_sampling.storage import load_stage, save_stage
from test_fast_h3_v2_core_sampler import model
from test_progressive_sampling_runtime import conditioning, latent
from test_progressive_relay import paired


def effect_inputs(stage="low_0_4", profile="dense_compat_exp", mode="apply_exp", tau=.2,
                  source=None, bare=None, positive=None):
    source = latent() if source is None else source
    bare = model() if bare is None else bare
    positive = conditioning() if positive is None else positive
    prepared, sampler, sigmas, context, _ = build_stage(bare, source, stage, profile)
    applied, runtime, _ = eav.apply_stage_eav(prepared, sigmas, source, context,
        eav.EAVConfig(mode, tau=tau, start_video_progress=0., end_video_progress=1., g_hard_limit=3.))
    guider = BasicGuider.execute(applied, positive).result[0]
    return (RandomNoise.execute(31).result[0], guider, sampler, sigmas, source, context), runtime


def test_long_video_relay_descriptor_projects_only_with_matching_bound_owner():
    class Clone:
        removed = None

        def remove_attachments(self, key):
            self.removed = key

    binding = {"binding_hash": "a" * 64, "plan_hash": "b" * 64}
    attachment = {"schema": PROMPT_RELAY_LONG_VIDEO_PROJECTION_SCHEMA,
                  "global_plan_hash": "c" * 64, "projected_plan_hash": binding["plan_hash"],
                  "binding_hash": binding["binding_hash"], "segment_index": 0}
    clone = Clone()
    contract = effect_identity._project_long_video_relay_attachment(clone, attachment, binding)
    assert clone.removed == PROMPT_RELAY_LONG_VIDEO_ATTACHMENT_KEY
    assert contract["long_video_projection"] == attachment
    assert len(contract["long_video_builder_sha256"]) == 64
    assert attachment["binding_hash"] == binding["binding_hash"]
    for altered, owner in (({**attachment, "binding_hash": "d" * 64}, binding),
                           ({**attachment, "projected_plan_hash": "d" * 64}, binding),
                           (attachment, None)):
        with pytest.raises(UnverifiedModelStack, match="Long Video Relay"):
            effect_identity._project_long_video_relay_attachment(Clone(), altered, owner)


@pytest.mark.parametrize("stage", ["low_0_4", "high_4_8"])
@pytest.mark.parametrize("profile", ["dense_compat_exp", "trained_vsa_exp"])
@pytest.mark.parametrize("mode", ["report_only", "apply_exp"])
def test_cold_hot_rebuilt_effect_identity_roundtrip_preserves_live_owners(tmp_path, stage, profile, mode):
    args, runtime = effect_inputs(stage, profile, mode)
    prepared = args[1].model_patcher
    override = prepared.model_options["transformer_options"]["optimized_attention_override"]
    callbacks = {role: dict(groups) for role, groups in prepared.callbacks.items()}
    first_result = sample_stage(*args)[2]
    reference = SamplerCustomAdvanced.execute(*effect_inputs(stage, profile, mode)[0][:5]).result
    for actual, expected in zip((first_result.output, first_result.denoised_output), reference):
        assert all(torch.equal(a, b) for a, b in zip(actual["samples"].unbind(), expected["samples"].unbind()))
    first = first_result.verify()
    hot = sample_stage(*args)[2].verify()
    rebuilt = sample_stage(*effect_inputs(stage, profile, mode)[0])[2].verify()
    assert first["portable_identity"] is hot["portable_identity"] is rebuilt["portable_identity"] is True
    assert first["request_sha256"] == hot["request_sha256"] == rebuilt["request_sha256"]
    assert first["outputs"] == hot["outputs"] == rebuilt["outputs"]
    assert prepared.get_attachment(eav.KEY) is runtime
    assert prepared.model_options["transformer_options"]["optimized_attention_override"] is override
    assert prepared.callbacks == callbacks and len(prepared.get_wrappers("diffusion_model", eav.KEY)) == 1
    path, digest, _ = save_stage(first_result, tmp_path)
    assert load_stage(tmp_path, path, digest, stage)[3].verify() == first
    assert not torch.cuda.is_initialized()  # trained profile is CPU-eligible Dense, not VSA proof.


def test_config_changes_bind_different_requests_and_disabled_keeps_base_identity():
    first = sample_stage(*effect_inputs(tau=.2)[0])[2].verify()
    changed = sample_stage(*effect_inputs(tau=.3)[0])[2].verify()
    assert first["request_sha256"] != changed["request_sha256"]
    assert first["request"]["model"]["stage_effects"] != changed["request"]["model"]["stage_effects"]
    disabled_args, _ = effect_inputs(mode="disabled")
    assert "stage_effects" not in selected_model_identity(disabled_args[1].model_patcher)


def test_native_masks_are_bound_and_roundtrip_with_eav(tmp_path):
    source = latent()
    vm, am = [torch.ones_like(value) for value in source["samples"].unbind()]
    vm[:, :, :1] = 0
    am[..., :4] = 0
    source["noise_mask"] = NestedTensor((vm, am))
    args, _ = effect_inputs(source=source)
    result = sample_stage(*args)[2]
    receipt = result.verify()
    assert receipt["portable_identity"] is True
    assert receipt["request"]["model"]["stage_effects"]["mask"] is not None
    path, digest, _ = save_stage(result, tmp_path)
    restored = load_stage(tmp_path, path, digest, "low_0_4")[1]
    assert all(torch.equal(a, b) for a, b in zip(restored["noise_mask"].unbind(), (vm, am)))


@pytest.mark.parametrize("change", ["config_object", "config_value", "runtime_method", "telemetry_method"])
def test_during_run_effect_mutations_cannot_get_a_completion_receipt(change):
    args, runtime = effect_inputs()
    guider = args[1]
    original = guider.sample

    def changed_after(*items, **kwargs):
        value = original(*items, **kwargs)
        if change == "config_object":
            runtime.config = replace(runtime.config, tau=.3)
        elif change == "config_value":
            object.__setattr__(runtime.config, "tau", .3)
        elif change == "runtime_method":
            runtime.prepare = lambda *a: None
        else:
            runtime.telemetry.record = lambda *a, **k: None
        return value

    guider.sample = changed_after
    with pytest.raises(ValueError, match="changed during sampling"):
        sample_stage(*args)


def test_unknown_delegate_runs_but_does_not_gain_portable_identity():
    bare = model()
    calls = []

    def unknown(function, *args, **kwargs):
        calls.append(True)
        return function(*args, **kwargs)

    bare.model_options["transformer_options"]["optimized_attention_override"] = unknown
    args, _ = effect_inputs(bare=bare)
    receipt = sample_stage(*args)[2].verify()
    assert len(calls) == 4 and receipt["portable_identity"] is False


def test_relay_composition_keeps_its_real_bias_and_binds_its_separate_identity():
    bound, positive, _ = paired(model())
    args, runtime = effect_inputs(bare=bound, positive=positive)
    receipt = sample_stage(*args)[2].verify()
    assert runtime.snapshot()["relay_attention_calls"] == 4
    assert receipt["portable_identity"] is True
    assert "relay" in receipt["request"]["model"]["stage_effects"]


def test_new_process_loads_effect_low_and_only_runs_effect_high(tmp_path):
    low = sample_stage(*effect_inputs()[0])[2]
    path, digest, _ = save_stage(low, tmp_path)
    expected = sample_stage(*effect_inputs("high_4_8", source=low.denoised_output)[0])[2].verify()
    code = """
import json,runpy,sys
runpy.run_path('tools/check_director_d1_cpu.py',run_name='config')
runpy.run_path('tests/conftest.py')
from test_modular_effect_identity import effect_inputs
from h3_audio_t8_pkg.modular_sampling.storage import load_stage
from h3_audio_t8_pkg.modular_sampling.results import sample_stage
import torch
low=load_stage(sys.argv[1],sys.argv[2],sys.argv[3],'low_0_4')[1]
args,runtime=effect_inputs('high_4_8',source=low)
receipt=sample_stage(*args)[2].verify()
assert receipt['execution']['denoiser_evaluations']==4
assert runtime.snapshot()['completed_forwards']==4 and receipt['portable_identity']
assert not torch.cuda.is_initialized()
print('RESULT='+json.dumps(receipt['outputs']))
"""
    child = subprocess.run([sys.executable, "-c", code, str(tmp_path), path, digest],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=45)
    assert child.returncode == 0, child.stderr
    actual = json.loads(next(line[7:] for line in child.stdout.splitlines() if line.startswith("RESULT=")))
    assert actual == expected["outputs"]


def relay_inputs(stage="low_0_4", with_eav=True, query_route="joint_av_exp", source=None):
    bound, positive, _ = paired(model(), query_route=query_route)
    if with_eav:
        return effect_inputs(stage, source=source, bare=bound, positive=positive)[0]
    source = latent() if source is None else source
    prepared, sampler, sigmas, context, _ = build_stage(bound, source, stage, "dense_compat_exp")
    guider = BasicGuider.execute(prepared, positive).result[0]
    return RandomNoise.execute(31).result[0], guider, sampler, sigmas, source, context


@pytest.mark.parametrize("stage", ["low_0_4", "high_4_8"])
@pytest.mark.parametrize("with_eav", [False, True])
@pytest.mark.parametrize("query_route", ["joint_av_exp", "video_only_paper"])
def test_plain_relay_and_composed_eav_roundtrip_with_cold_hot_rebuilt_identity(tmp_path, stage, with_eav, query_route):
    args = relay_inputs(stage, with_eav, query_route)
    prepared = args[1].model_patcher
    runtime = eav.v2.capture_fast_h3_v2_owner(prepared).runtime
    original_override = runtime.override
    first_result = sample_stage(*args)[2]
    reference = SamplerCustomAdvanced.execute(*relay_inputs(stage, with_eav, query_route)[:5]).result
    for actual, expected in zip((first_result.output, first_result.denoised_output), reference):
        assert all(torch.equal(a, b) for a, b in zip(actual["samples"].unbind(), expected["samples"].unbind()))
    first = first_result.verify()
    second = sample_stage(*args)[2].verify()
    rebuilt = sample_stage(*relay_inputs(stage, with_eav, query_route))[2].verify()
    assert first["portable_identity"] is second["portable_identity"] is rebuilt["portable_identity"] is True
    assert first["request_sha256"] == second["request_sha256"] == rebuilt["request_sha256"]
    assert first["outputs"] == second["outputs"] == rebuilt["outputs"]
    assert eav.v2.capture_fast_h3_v2_owner(prepared).runtime is runtime and runtime.override is original_override
    path, digest, _ = save_stage(first_result, tmp_path)
    assert load_stage(tmp_path, path, digest, stage)[3].verify() == first
    assert not torch.cuda.is_initialized()


def test_relay_route_is_part_of_request_identity():
    first = sample_stage(*relay_inputs(query_route="joint_av_exp"))[2].verify()
    second = sample_stage(*relay_inputs(query_route="video_only_paper"))[2].verify()
    assert first["request_sha256"] != second["request_sha256"]
    assert first["request"]["model"]["stage_effects"]["relay"] != second["request"]["model"]["stage_effects"]["relay"]


def test_unknown_relay_observer_executes_but_is_not_given_a_portable_owner():
    args = relay_inputs(with_eav=False)
    wrapper = args[1].model_patcher.get_wrappers("diffusion_model", relay.PROMPT_RELAY_WRAPPER_KEY)[0]
    calls = []
    index = wrapper.__code__.co_freevars.index("execution_observer")
    wrapper.__closure__[index].cell_contents = calls.append
    receipt = sample_stage(*args)[2].verify()
    assert calls.count("forward") == 4 and receipt["portable_identity"] is False


def test_relay_binding_mutation_after_sampling_cannot_keep_completion():
    args = relay_inputs()
    guider = args[1]
    wrapper = guider.model_patcher.get_wrappers("diffusion_model", eav.KEY)[0]
    captured = _factory_closure(wrapper, eav.apply_stage_eav, "stage_wrapper")
    original = guider.sample

    def changed_after(*items, **kwargs):
        value = original(*items, **kwargs)
        captured["binding"]["events"][0]["midpoint"] += .1
        return value

    guider.sample = changed_after
    with pytest.raises(ValueError, match="changed"):
        sample_stage(*args)


def test_sparse_relay_does_not_inherit_dense_portable_identity():
    bound, positive, _ = paired(model())
    args, _ = effect_inputs(profile="trained_vsa_exp", bare=bound, positive=positive)
    receipt = sample_stage(*args)[2].verify()
    assert receipt["portable_identity"] is False


def test_nonzero_lora_with_relay_and_eav_remains_portable_without_unpatching_live_weights():
    from comfy.weight_adapter.lora import LoRAAdapter
    bare = model()
    key, weight = next((key, value) for key, value in bare.model_state_dict().items()
                       if key.endswith("weight") and value.ndim == 2)
    adapter = LoRAAdapter({"up", "down"}, (torch.ones(weight.shape[0], 1) * .02,
                                          torch.ones(1, weight.shape[1]) * .02, 1., None, None, None))
    assert bare.add_patches({key: adapter}, strength_patch=.7) == [key]
    bound, positive, _ = paired(bare)
    args, runtime = effect_inputs(bare=bound, positive=positive)
    first = sample_stage(*args)[2].verify()
    live = args[1].model_patcher
    backup = live.backup[key]
    second = sample_stage(*args)[2].verify()
    assert live.backup[key] is backup and live.patches[key][0][1] is adapter
    assert runtime.snapshot()["relay_attention_calls"] == 4
    assert first["portable_identity"] is second["portable_identity"] is True
    assert first["request_sha256"] == second["request_sha256"]
    plain = sample_stage(*relay_inputs())[2].verify()
    assert first["request"]["model"] != plain["request"]["model"]


def test_actual_new_process_reads_relay_eav_low_and_runs_only_relay_eav_high(tmp_path):
    low = sample_stage(*relay_inputs())[2]
    path, digest, _ = save_stage(low, tmp_path)
    expected = sample_stage(*relay_inputs("high_4_8", source=low.denoised_output))[2].verify()
    code = """
import json,runpy,sys
runpy.run_path('tools/check_director_d1_cpu.py',run_name='config')
runpy.run_path('tests/conftest.py')
from test_modular_effect_identity import relay_inputs
from h3_audio_t8_pkg.modular_sampling.storage import load_stage
from h3_audio_t8_pkg.modular_sampling.results import sample_stage
import torch
low=load_stage(sys.argv[1],sys.argv[2],sys.argv[3],'low_0_4')[1]
receipt=sample_stage(*relay_inputs('high_4_8',source=low))[2].verify()
assert receipt['execution']['denoiser_evaluations']==4 and receipt['portable_identity']
assert not torch.cuda.is_initialized()
print('RESULT='+json.dumps(receipt['outputs']))
"""
    child = subprocess.run([sys.executable, "-c", code, str(tmp_path), path, digest],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=45)
    assert child.returncode == 0, child.stderr
    actual = json.loads(next(line[7:] for line in child.stdout.splitlines() if line.startswith("RESULT=")))
    assert actual == expected["outputs"]
