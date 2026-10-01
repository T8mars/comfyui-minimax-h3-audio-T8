"""Original tiny VDN window + linear, own-grid stages and frozen AV results."""
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from comfy_extras.nodes_custom_sampler import BasicGuider, RandomNoise

from h3_audio_t8_pkg import vdn_h3_advanced as vdn, vdn_two_pass
from h3_audio_t8_pkg.modular_sampling import vdn_stages as split, vdn_identity, eav
from h3_audio_t8_pkg.modular_sampling.results import sample_stage
from h3_audio_t8_pkg.modular_sampling.storage import save_stage, load_stage
from test_modular_vdn_baseline import tiny_vdn, source, run
from test_progressive_sampling_runtime import conditioning


def inputs(training="stage_dmd_8nfe", stage="vdn_complete", frames=2, tail=4):
    model, branch = tiny_vdn(training)
    latent = source(frames)
    prepared, sampler, sigmas, context, _ = split.build_stage(model, latent, stage, tail, latent)
    args = (RandomNoise.execute(43).result[0], BasicGuider.execute(prepared, conditioning()).result[0],
            sampler, sigmas, latent, context)
    return args, model, branch


@pytest.mark.parametrize("training,frames,tail", [
    ("stage_dmd_8nfe", 2, 4), ("stage_dmd_8nfe", 16, 4), ("stage_b_50nfe", 2, 5)])
@pytest.mark.parametrize("stage", split.STAGES)
def test_real_legacy_stage_parity_cold_hot_rebuilt_and_frozen(training, frames, tail, stage, tmp_path):
    args, model, branch = inputs(training, stage, frames, tail)
    reference = (vdn.setup_vdn_execution(model, args[4]) if stage == split.STAGES[0]
                 else vdn_two_pass.setup_vdn_refine(model, args[4], args[4], tail))
    expected = run(*reference[:3], args[4])
    result = sample_stage(*args)[2]
    for actual, wanted in zip((result.output, result.denoised_output), expected):
        assert all(torch.equal(a, b) for a, b in zip(actual["samples"].unbind(), wanted["samples"].unbind()))
    receipt = result.verify()
    assert receipt["portable_identity"] is receipt["verified_recipe_completion"] is True
    assert receipt["execution"]["denoiser_evaluations"] == args[-1].steps
    assert sample_stage(*args)[2].verify()["request_sha256"] == receipt["request_sha256"]
    rebuilt, _, _ = inputs(training, stage, frames, tail)
    assert sample_stage(*rebuilt)[2].verify()["request_sha256"] == receipt["request_sha256"]
    path, sha, _ = save_stage(result, tmp_path)
    loaded = load_stage(tmp_path, path, sha, stage)[0]
    assert all(torch.equal(a, b) for a, b in zip(loaded["samples"].unbind(), result.output["samples"].unbind()))
    assert model.additional_models[vdn.ADDITIONAL_MODEL_KEY][0] is branch
    assert not torch.cuda.is_initialized()


def test_readonly_projection_does_not_remove_live_vdn():
    args, model, branch = inputs()
    before = vdn_identity.inspect(model)
    view, contract = vdn_identity.project(model)
    assert contract == before and not view.additional_models
    assert model.additional_models[vdn.ADDITIONAL_MODEL_KEY][0] is branch
    assert model.get_wrappers("diffusion_model", vdn.WRAPPER_KEY) == [vdn._layout_wrapper]
    assert json.loads(args[-1].profile)["vdn"]["identity"] == before


def test_cold_hot_branch_identity_diagnostic():
    args, model, _ = inputs()
    before = vdn_identity.inspect(model)
    run(*vdn.setup_vdn_execution(model, args[4])[:3], args[4])
    after = vdn_identity.inspect(model)
    def differences(a, b, path=""):
        if type(a) is dict and type(b) is dict:
            return [change for key in a.keys() | b.keys()
                    for change in differences(a.get(key), b.get(key), path + "/" + str(key))]
        if type(a) is list and type(b) is list and len(a) == len(b):
            return [change for i, (x, y) in enumerate(zip(a, b)) for change in differences(x, y, path + "/" + str(i))]
        return [] if a == b else [(path, a, b)]
    assert before == after, differences(before, after)


@pytest.mark.parametrize("stage", split.STAGES)
@pytest.mark.parametrize("frames", [2, 16])
@pytest.mark.parametrize("mode", ["report_only", "apply_exp"])
def test_vdn_eav_actual_window_and_linear_projection_path(stage, frames, mode, tmp_path):
    args, _, branch = inputs(stage=stage, frames=frames)
    original = sample_stage(*args)[2]
    model = args[1].model_patcher
    stage_branch = model.additional_models[vdn.ADDITIONAL_MODEL_KEY][0]
    options = model.model_options["transformer_options"]
    hooks = options[vdn.OWNER_HOOKS_KEY]
    applied, runtime, _ = eav.apply_stage_eav(model, args[3], args[4], args[5],
        eav.EAVConfig(mode, tau=.5, start_video_progress=0., end_video_progress=1., g_hard_limit=3.))
    assert vdn_identity.inspect(applied) == vdn_identity.inspect(model)
    configured = (args[0], BasicGuider.execute(applied, conditioning()).result[0], *args[2:])
    result = sample_stage(*configured)[2]
    assert result.verify()["portable_identity"] is True
    audit = runtime.snapshot()
    assert audit["status"] == "observed_" + mode
    assert audit["selector_calls"] == 0 and audit["sparse_producer_calls"] == args[-1].steps
    assert options[vdn.OWNER_HOOKS_KEY] == hooks
    assert model.additional_models[vdn.ADDITIONAL_MODEL_KEY][0] is stage_branch
    assert stage_branch.model is branch.model
    same = all(torch.equal(x, y) for x, y in zip(result.output["samples"].unbind(), original.output["samples"].unbind()))
    assert same is (mode == "report_only")
    assert sample_stage(*configured)[2].verify()["request_sha256"] == result.verify()["request_sha256"]
    path, digest, _ = save_stage(result, tmp_path)
    assert load_stage(tmp_path, path, digest, stage)[3].verify()["portable_identity"]


def effect_args(training, stage, latent, first=None):
    model, _ = tiny_vdn(training)
    prepared, sampler, sigmas, context, _ = split.build_stage(model, latent, stage, 4, first)
    applied, runtime, _ = eav.apply_stage_eav(prepared, sigmas, latent, context,
        eav.EAVConfig("apply_exp", tau=.5, start_video_progress=0., end_video_progress=1., g_hard_limit=3.))
    return (RandomNoise.execute(43).result[0], BasicGuider.execute(applied, conditioning()).result[0],
            sampler, sigmas, latent, context), runtime


def high_from_completed(training, first):
    from h3_audio_t8_pkg import learned_latent_upscale_advanced as learned
    frames = first["samples"].unbind()[0].shape[2]
    # Same-size handoff exercises the real reconcile/mask, not a learned model.
    high, _, _ = learned.reconcile_two_pass_h3_latent(first, source(frames), conditioning(), "first_pass",
        second_pass_audio_source="first_pass", second_pass_audio_strength=0.)
    return effect_args(training, split.STAGES[1], high, first)


@pytest.mark.parametrize("training", ["stage_dmd_8nfe", "stage_b_50nfe"])
@pytest.mark.parametrize("frames", [2, 16])
@pytest.mark.parametrize("cpu_threads", [1, 2])
def test_actual_new_process_load_completed_low_only_runs_high_with_eav(tmp_path, training, frames, cpu_threads):
    torch.set_num_threads(cpu_threads)
    from h3_audio_t8_pkg import learned_latent_upscale_advanced as learned
    low_args, low_effect = effect_args(training, split.STAGES[0], source(frames))
    result = sample_stage(*low_args)[2]
    assert low_effect.snapshot()["status"] == "observed_apply_exp"
    path, digest, _ = save_stage(result, tmp_path)
    args, runtime = high_from_completed(training, result.output)
    high = sample_stage(*args)[2]
    expected = high.verify()["outputs"]
    audited, report = learned.audit_two_pass_h3_audio(args[4], high.output, 0., True, 1e-5)
    assert json.loads(report)["audio_relocked_exact"]
    assert torch.equal(audited["samples"].unbind()[1], result.output["samples"].unbind()[1])
    assert runtime.snapshot()["status"] == "observed_apply_exp"
    code = """
import json,runpy,sys
runpy.run_path('tools/check_director_d1_cpu.py',run_name='config')
runpy.run_path('tests/conftest.py')
from test_modular_vdn_stages import high_from_completed
from h3_audio_t8_pkg.modular_sampling.storage import load_stage
from h3_audio_t8_pkg.modular_sampling.results import sample_stage
from h3_audio_t8_pkg import learned_latent_upscale_advanced as learned
import torch
torch.set_num_threads(int(sys.argv[5]))
first=load_stage(sys.argv[1],sys.argv[2],sys.argv[3],'vdn_complete')[0]
args,runtime=high_from_completed(sys.argv[4],first)
result=sample_stage(*args)[2]
receipt=result.verify()
assert receipt['execution']['denoiser_evaluations']==4 and receipt['portable_identity']
assert runtime.snapshot()['status']=='observed_apply_exp'
audited,_=learned.audit_two_pass_h3_audio(args[4],result.output,0.,True,1e-5)
assert torch.equal(first['samples'].unbind()[1],audited['samples'].unbind()[1])
assert not torch.cuda.is_initialized()
print('RESULT='+json.dumps(receipt['outputs']))
"""
    child = subprocess.run([sys.executable, "-c", code, str(tmp_path), path, digest, training,
                            str(torch.get_num_threads())],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=60)
    assert child.returncode == 0, child.stderr
    actual = json.loads(next(line[7:] for line in child.stdout.splitlines() if line.startswith("RESULT=")))
    assert actual == expected


@pytest.mark.parametrize("damage", ["weight", "module_field", "forward", "hook", "wrapper", "receipt", "branch_patch"])
def test_bound_vdn_execution_changes_cannot_reuse_old_stage_contract(damage):
    args, _, _ = inputs()
    prepared = args[1].model_patcher
    branch = prepared.additional_models[vdn.ADDITIONAL_MODEL_KEY][0]
    if damage == "weight":
        with torch.no_grad():
            next(branch.model.parameters()).add_(.01)
    elif damage == "module_field":
        branch.model.blocks[0].softmax_gate.custom_setting = 1
    elif damage == "forward":
        branch.model.blocks[0].softmax_gate.forward = lambda x: x
    elif damage == "hook":
        prepared.set_model_patch_replace(lambda args, extra: extra["original_block"](args), "dit", "double_block", 0)
    elif damage == "wrapper":
        prepared.remove_wrappers_with_key("diffusion_model", vdn.WRAPPER_KEY)
    elif damage == "receipt":
        receipt = dict(prepared.get_attachment(vdn.ATTACHMENT_KEY), test_changed=True)
        prepared.set_attachments(vdn.ATTACHMENT_KEY, receipt)
    else:
        key, weight = next(iter(branch.model_state_dict().items()))
        branch.add_patches({key: ("diff", (torch.ones_like(weight) * .01,))})
    with pytest.raises(ValueError, match="changed after"):
        split.validate_stage(prepared, args[3], args[4], args[5])


@pytest.mark.parametrize("damage", ["sigma_value", "sigma_dtype", "geometry", "context"])
def test_mismatched_stage_inputs_rejected_before_sampling(damage):
    args, _, _ = inputs()
    model, sigmas, latent, context = args[1].model_patcher, args[3], args[4], args[5]
    if damage == "sigma_value":
        sigmas = sigmas.clone()
        sigmas[1] -= .001
    elif damage == "sigma_dtype":
        sigmas = sigmas.double()
    elif damage == "geometry":
        latent = source(3)
    else:
        from dataclasses import replace
        context = replace(context, stage=split.STAGES[1])
    with pytest.raises(ValueError, match="SIGMAS|geometry|context"):
        split.validate_stage(model, sigmas, latent, context)


def test_unknown_branch_callback_is_preserved_and_fresh_execution_not_falsely_portable():
    from comfy.patcher_extension import CallbacksMP
    model, branch = tiny_vdn()
    calls = []
    def callback(*a, **kw):
        calls.append("loaded")
    branch.add_callback_with_key(CallbacksMP.ON_LOAD, "user", callback)
    latent = source()
    bound, sampler, sigmas, context, _ = split.build_stage(model, latent)
    args = (RandomNoise.execute(43).result[0], BasicGuider.execute(bound, conditioning()).result[0],
            sampler, sigmas, latent, context)
    result = sample_stage(*args)[2].verify()
    assert calls and result["portable_identity"] is False
    assert branch.get_callbacks(CallbacksMP.ON_LOAD, "user") == [callback]


@pytest.mark.parametrize("stage", split.STAGES)
def test_nonzero_user_lora_kept_and_changes_actual_vdn_output(stage):
    from comfy.weight_adapter.lora import LoRAAdapter
    args, _, _ = inputs(stage=stage)
    prepared = args[1].model_patcher
    key, value = next((key, value) for key, value in prepared.model_state_dict().items()
        if ".blocks.0.mlp." in key and key.endswith("weight") and value.ndim == 2)
    adapter = LoRAAdapter({"up", "down"}, (torch.ones(value.shape[0], 1) * .2,
        torch.ones(1, value.shape[1]) * .2, 1., None, None, None))
    prepared.add_patches({key: adapter}, .7)
    result = sample_stage(*args)[2]
    assert result.verify()["portable_identity"] is True
    assert prepared.patches[key][0][1] is adapter
    assert sample_stage(*args)[2].verify()["request_sha256"] == result.verify()["request_sha256"]
    plain = sample_stage(*inputs(stage=stage)[0])[2]
    assert any(not torch.equal(a, b) for a, b in zip(result.output["samples"].unbind(), plain.output["samples"].unbind()))


@pytest.mark.parametrize("stage", split.STAGES)
def test_additional_branch_user_weight_patch_cold_hot_save_and_actual_effect(stage, tmp_path):
    model, branch = tiny_vdn()
    # Resolve the real gate weight by shape rather than assume a vendor name.
    key, weight = next((key, value) for key, value in branch.model_state_dict().items()
                      if "softmax_gate" in key and value.ndim == 2)
    patch = ("diff", (torch.ones_like(weight) * .15,))
    assert branch.add_patches({key: patch}, .7) == [key]
    latent = source(16)
    bound, sampler, sigmas, context, _ = split.build_stage(model, latent, stage, 4, latent)
    args = (RandomNoise.execute(43).result[0], BasicGuider.execute(bound, conditioning()).result[0],
            sampler, sigmas, latent, context)
    result = sample_stage(*args)[2]
    receipt = result.verify()
    assert receipt["portable_identity"] is True
    assert sample_stage(*args)[2].verify()["request_sha256"] == receipt["request_sha256"]
    assert branch.patches[key][0][1] is patch
    path, digest, _ = save_stage(result, tmp_path)
    assert load_stage(tmp_path, path, digest, stage)[3].verify()["portable_identity"]
    plain = sample_stage(*inputs(stage=stage, frames=16)[0])[2]
    assert any(not torch.equal(a, b) for a, b in zip(result.output["samples"].unbind(), plain.output["samples"].unbind()))


@pytest.mark.parametrize("stage", split.STAGES)
def test_disabled_eav_keeps_original_vdn_object_and_hooks(stage):
    args, _, _ = inputs(stage=stage)
    model = args[1].model_patcher
    hooks = model.model_options["transformer_options"][vdn.OWNER_HOOKS_KEY]
    selected, _, _ = eav.apply_stage_eav(model, args[3], args[4], args[5], eav.EAVConfig("disabled"))
    assert selected is model and selected.model_options["transformer_options"][vdn.OWNER_HOOKS_KEY] is hooks


def test_foreign_failure_is_preserved_and_error_propagates_then_actual_retry_completes():
    args, _, _ = inputs()
    model = args[1].model_patcher
    def failure(executor, *a, **kw):
        raise RuntimeError("user-vdn-forward-error")
    model.add_wrapper_with_key("diffusion_model", "user_failure", failure)
    with pytest.raises(RuntimeError, match="user-vdn-forward-error"):
        sample_stage(*args)
    assert model.get_wrappers("diffusion_model", "user_failure") == [failure]
    model.remove_wrappers_with_key("diffusion_model", "user_failure")
    assert sample_stage(*args)[2].verify()["verified_recipe_completion"]


@pytest.mark.parametrize("damage", ["missing_first", "zero_tail", "full_tail", "nan", "shrink"])
def test_original_vdn_refine_real_errors_not_relaxed(damage):
    model, _ = tiny_vdn()
    first, latent, tail = source(), source(), 4
    if damage == "missing_first":
        first = None
    elif damage == "zero_tail":
        tail = 0
    elif damage == "full_tail":
        tail = 8
    elif damage == "nan":
        latent["samples"].unbind()[0][0, 0, 0, 0, 0] = float("nan")
    else:
        latent["samples"].tensors[0] = torch.zeros(1, 24, 2, 2, 4)
    with pytest.raises(ValueError, match="completed first-pass|refine_steps|NaN|shrink"):
        split.build_stage(model, latent, split.STAGES[1], tail, first)
