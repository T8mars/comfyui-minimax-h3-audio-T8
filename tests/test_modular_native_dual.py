"""Original Dual stages versus public stages on real tiny CPU Core H3."""
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from comfy.nested_tensor import NestedTensor
from comfy_extras.nodes_custom_sampler import BasicGuider, RandomNoise, SamplerCustomAdvanced

from h3_audio_t8_pkg.long_video_dual_model_runner import DualModelSegmentRunner
from h3_audio_t8_pkg.modular_sampling import native_dual as native, eav
from h3_audio_t8_pkg.modular_sampling.results import sample_stage, selected_model_identity
from h3_audio_t8_pkg.modular_sampling.storage import save_stage, load_stage
from test_fast_h3_v2_core_sampler import model
from test_progressive_sampling_runtime import latent, conditioning
from test_progressive_relay import paired


def original_runner(bare, coarse=4, refine=4, shifts=(12., 3.), source="auto", strength=0.):
    return DualModelSegmentRunner(bare, bare, contract={}, low_width=64, low_height=32,
        upscaler_model="test_not_loaded", coarse_steps=coarse, refine_steps=refine,
        first_shift_video=shifts[0], first_shift_audio=shifts[1],
        second_shift_video=shifts[0], second_shift_audio=shifts[1],
        second_audio_source=source, second_audio_strength=strength)


def inputs(stage="dual_low_4", effects="none", source=None, shifts=(12., 3.), bare=None):
    bare, positive = model() if bare is None else bare, conditioning()
    if "relay" in effects:
        bare, positive, _ = paired(bare)
    source = latent() if source is None else source
    prepared, sampler, sigmas, context, _ = native.build_stage(bare, source, stage, *shifts)
    runtime = None
    if "eav" in effects:
        prepared, runtime, _ = eav.apply_stage_eav(prepared, sigmas, source, context,
            eav.EAVConfig("apply_exp", tau=.2, start_video_progress=0., end_video_progress=1., g_hard_limit=3.))
    args = (RandomNoise.execute(31).result[0], BasicGuider.execute(prepared, positive).result[0],
            sampler, sigmas, source, context)
    return args, runtime


@pytest.mark.parametrize("coarse", [4, 20])
@pytest.mark.parametrize("refine", [3, 4, 5])
@pytest.mark.parametrize("first", [False, True])
@pytest.mark.parametrize("shifts", [(12., 3.), (8., 2.)])
def test_all_original_tables_and_real_tiny_outputs_match(coarse, refine, first, shifts):
    source, bare = latent(), model()
    before = dict(bare.object_patches)
    old, old_sampler, old_sigmas, _ = original_runner(bare, coarse, refine, shifts)._stage_sampling(bare, source, first)
    name = f"dual_low_{coarse}" if first else f"dual_high_{refine}"
    new, sampler, sigmas, context, raw = native.build_stage(bare, source, name, *shifts)
    assert torch.equal(old_sigmas, sigmas)
    assert context.steps == (coarse if first else refine)
    assert bool(sigmas[-1] > 0) == (first and coarse == 4)
    assert bare.object_patches == before and bare.get_attachment(native.KEY) is None
    noise = RandomNoise.execute(31).result[0]
    reference = SamplerCustomAdvanced.execute(noise, BasicGuider.execute(old, conditioning()).result[0],
                                              old_sampler, old_sigmas, source).result
    actual = sample_stage(noise, BasicGuider.execute(new, conditioning()).result[0], sampler, sigmas, source, context)
    for output, expected in zip(actual[:2], reference):
        assert all(torch.equal(a, b) for a, b in zip(output["samples"].unbind(), expected["samples"].unbind()))
    receipt = actual[2].verify()
    assert receipt["execution"]["callbacks"] == list(range(context.steps))
    assert receipt["execution"]["denoiser_evaluations"] == context.steps
    assert receipt["verified_recipe_completion"] is receipt["portable_identity"] is True
    assert json.loads(raw)["sampled"] is False and not torch.cuda.is_initialized()


@pytest.mark.parametrize("stage", native.STAGES)
@pytest.mark.parametrize("effects", ["none", "eav", "relay", "relay_eav"])
def test_cold_hot_rebuilt_effect_identity_and_frozen_result(tmp_path, stage, effects):
    args, runtime = inputs(stage, effects)
    selected = args[1].model_patcher
    owner = native.capture_owner(selected)
    result = sample_stage(*args)[2]
    cold = result.verify()
    hot = sample_stage(*args)[2].verify()
    rebuilt = sample_stage(*inputs(stage, effects)[0])[2].verify()
    assert cold["portable_identity"] is hot["portable_identity"] is rebuilt["portable_identity"] is True
    assert cold["request_sha256"] == hot["request_sha256"] == rebuilt["request_sha256"]
    assert cold["outputs"] == hot["outputs"] == rebuilt["outputs"]
    assert native.capture_owner(selected) is owner
    if runtime is not None:
        audit = runtime.snapshot()
        assert audit["status"] == "observed_apply_exp"
        assert audit["completed_forwards"] == args[-1].steps
        assert audit["v2_dispatch"] is None
        if "relay" in effects:
            assert audit["relay_attention_calls"] == args[-1].steps
    path, digest, _ = save_stage(result, tmp_path)
    assert load_stage(tmp_path, path, digest, stage)[3].verify() == cold


@pytest.mark.parametrize("coarse", [4, 20])
@pytest.mark.parametrize("source,strength", [("auto", 0.), ("first_pass", 0.), ("first_pass", .4),
                                           ("highres_template", .7), ("legacy_policy", 0.)])
def test_exact_old_audio_handoff_with_locked_and_partial_masks(coarse, source, strength):
    learned, template = latent(), latent()
    video, audio = learned["samples"].unbind()
    learned["samples"] = NestedTensor((video + .1, audio + .3))
    tv, ta = template["samples"].unbind()
    template["samples"] = NestedTensor((tv + .6, ta + .9))
    am = torch.ones_like(ta)
    am[..., :2], am[..., 2:4] = 0, .5
    template["noise_mask"] = NestedTensor((torch.ones_like(tv), am))
    positive = conditioning()
    runner = original_runner(None, coarse, source=source, strength=strength)
    expected, expected_cond, _ = runner._reconcile(learned, template, positive)
    actual, actual_cond, raw = native.reconcile(learned, template, positive, coarse, source, strength)
    for key in ("samples", "noise_mask"):
        assert all(torch.equal(a, b) for a, b in zip(actual[key].unbind(), expected[key].unbind()))
    assert actual_cond == expected_cond
    assert json.loads(raw)["audio_policy"] == runner.audio_policy
    assert torch.equal(template["samples"].unbind()[1], ta + .9)
    if runner.audio[0] == "legacy_policy":
        out = actual["samples"].unbind()[1]
        assert torch.all(out[..., :2] == .9) and torch.all(out[..., 2:] == .3)


@pytest.mark.parametrize("mutation", ["context", "sigmas", "sampling", "shifts"])
def test_own_receipt_mismatches_are_not_silently_accepted(mutation):
    args = list(inputs()[0])
    selected = args[1].model_patcher
    if mutation == "context":
        args[-1] = replace(args[-1], audio_shift=2.)
    elif mutation == "sigmas":
        args[3] = args[3].clone() * .9
    elif mutation == "sampling":
        selected.object_patches["model_sampling"].sigmas.add_(.01)
    else:
        selected.model_options["transformer_options"]["minimax_h3_sigma_shift_audio"] = 2.
    with pytest.raises(ValueError):
        sample_stage(*args)


def test_foreign_sampler_keeps_fresh_execution_without_false_completion():
    args = list(inputs()[0])
    calls = []

    def passthrough(model_wrap, x, sigmas, **kwargs):
        calls.append(True)
        return x

    args[2].sampler_function = passthrough
    receipt = sample_stage(*args)[2].verify()
    assert calls == [True]
    assert receipt["portable_identity"] is receipt["verified_recipe_completion"] is False


def test_unknown_model_hook_remains_available_but_not_portable():
    args = list(inputs()[0])
    selected = args[1].model_patcher
    def callback(value):
        return value["denoised"]
    selected.model_options["sampler_post_cfg_function"] = [callback]
    assert selected_model_identity(selected)["portable_cache_reuse"] is False
    assert sample_stage(*args)[2].verify()["portable_identity"] is False
    assert selected.model_options["sampler_post_cfg_function"] == [callback]


@pytest.mark.parametrize("stage", native.STAGES)
@pytest.mark.parametrize("mode", ["disabled", "report_only"])
def test_inactive_native_eav_preserves_unobserved_core_outputs(stage, mode):
    args, _ = inputs(stage)
    reference = SamplerCustomAdvanced.execute(*args[:5]).result
    noise, guider, sampler, sigmas, source, context = args
    applied, runtime, _ = eav.apply_stage_eav(guider.model_patcher, sigmas, source, context,
        eav.EAVConfig(mode, tau=.2, start_video_progress=0., end_video_progress=1., g_hard_limit=3.))
    if mode == "disabled":
        assert applied is guider.model_patcher
    actual = sample_stage(noise, BasicGuider.execute(applied, conditioning()).result[0],
                          sampler, sigmas, source, context)
    for output, expected in zip(actual[:2], reference):
        assert all(torch.equal(a, b) for a, b in zip(output["samples"].unbind(), expected["samples"].unbind()))
    assert runtime.snapshot()["status"] == ("disabled_identity" if mode == "disabled" else "observed_report_only")


@pytest.mark.parametrize("coarse", [4, 20])
@pytest.mark.parametrize("refine", [3, 4, 5])
def test_new_process_reads_low_and_executes_only_high_with_real_audio_handoff(tmp_path, coarse, refine):
    low = sample_stage(*inputs(f"dual_low_{coarse}", "relay_eav")[0])[2]
    path, digest, _ = save_stage(low, tmp_path)
    reconciled = native.reconcile(low.denoised_output, latent(), conditioning(), coarse)[0]
    expected = sample_stage(*inputs(f"dual_high_{refine}", "relay_eav", source=reconciled)[0])[2].verify()
    code = """
import json,runpy,sys
runpy.run_path('tools/check_director_d1_cpu.py',run_name='config')
runpy.run_path('tests/conftest.py')
from test_modular_native_dual import inputs,latent,conditioning,native
from h3_audio_t8_pkg.modular_sampling.storage import load_stage
from h3_audio_t8_pkg.modular_sampling.results import sample_stage
import torch
coarse,refine=int(sys.argv[4]),int(sys.argv[5])
low=load_stage(sys.argv[1],sys.argv[2],sys.argv[3],f'dual_low_{coarse}')[1]
original=native.build_stage
def high_only(*args,**kwargs):
    assert args[2].startswith('dual_high_')
    return original(*args,**kwargs)
native.build_stage=high_only
source=native.reconcile(low,latent(),conditioning(),coarse)[0]
args,runtime=inputs(f'dual_high_{refine}','relay_eav',source=source)
receipt=sample_stage(*args)[2].verify()
assert receipt['execution']['denoiser_evaluations']==refine and receipt['portable_identity']
assert runtime.snapshot()['relay_attention_calls']==refine
assert not torch.cuda.is_initialized()
print('RESULT='+json.dumps(receipt['outputs']))
"""
    child = subprocess.run([sys.executable, "-c", code, str(tmp_path), path, digest, str(coarse), str(refine)],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=45)
    assert child.returncode == 0, child.stderr
    actual = json.loads(next(line[7:] for line in child.stdout.splitlines() if line.startswith("RESULT=")))
    assert actual == expected["outputs"]


def test_nonzero_lora_and_native_av_masks_remain_portable_with_both_effects(tmp_path):
    from comfy.weight_adapter.lora import LoRAAdapter
    bare = model()
    key, weight = next((key, value) for key, value in bare.model_state_dict().items()
                       if key.endswith("weight") and value.ndim == 2)
    adapter = LoRAAdapter({"up", "down"}, (torch.ones(weight.shape[0], 1) * .02,
                                          torch.ones(1, weight.shape[1]) * .02, 1., None, None, None))
    assert bare.add_patches({key: adapter}, strength_patch=.7) == [key]
    source = latent()
    vm, am = [torch.ones_like(value) for value in source["samples"].unbind()]
    vm[:, :, :1], am[..., :4] = 0, 0
    source["noise_mask"] = NestedTensor((vm, am))
    args, runtime = inputs("dual_high_5", "relay_eav", source=source, bare=bare)
    result = sample_stage(*args)[2]
    receipt = result.verify()
    selected = args[1].model_patcher
    backup = selected.backup[key]
    assert sample_stage(*args)[2].verify()["request_sha256"] == receipt["request_sha256"]
    assert selected.backup[key] is backup and selected.patches[key][0][1] is adapter
    assert receipt["portable_identity"] is True
    assert runtime.snapshot()["relay_attention_calls"] == 5
    path, digest, _ = save_stage(result, tmp_path)
    restored = load_stage(tmp_path, path, digest, "dual_high_5")[1]
    assert all(torch.equal(a, b) for a, b in zip(restored["noise_mask"].unbind(), (vm, am)))


def test_during_run_owner_replacement_cannot_get_a_completion_receipt():
    args, _ = inputs()
    guider = args[1]
    original = guider.sample

    def replace_after(*values, **kwargs):
        output = original(*values, **kwargs)
        selected = guider.model_patcher
        owner = selected.get_attachment(native.KEY)
        selected.set_attachments(native.KEY, replace(owner, context=replace(owner.context, audio_shift=2.)))
        return output

    guider.sample = replace_after
    with pytest.raises(ValueError, match="clock shifts"):
        sample_stage(*args)
