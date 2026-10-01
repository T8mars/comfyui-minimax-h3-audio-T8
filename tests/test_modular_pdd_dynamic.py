"""Legacy PDD real tiny heads/backbone injection; no full artifact/GPU claim."""
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from comfy_extras.nodes_custom_sampler import BasicGuider, RandomNoise, SamplerCustomAdvanced

from h3_audio_t8_pkg import pdd_advanced as pdd, sampling
from h3_audio_t8_pkg.modular_sampling import pdd_stages as split, eav
from h3_audio_t8_pkg.modular_sampling.results import sample_stage
from h3_audio_t8_pkg.modular_sampling.storage import save_stage, load_stage
from test_fast_h3_v2_core_sampler import model
from test_modular_rf_restart import source_latent
from test_progressive_sampling_runtime import conditioning
from test_progressive_relay import paired


def dynamic_model(strength=.5, variant="FL2VA", dtype=torch.float32, base=None):
    bare = model() if base is None else base
    final = bare.get_model_object("diffusion_model.final_layer")
    generator = torch.Generator().manual_seed(783)
    banks = {}
    for stream in ("video", "audio"):
        for field in ("weight", "bias"):
            value = getattr(getattr(final, stream + "_out"), field)
            banks[f"pdd.final_layer.{stream}_out.{field}"] = torch.randn(
                (32, *value.shape), generator=generator, dtype=dtype) * .03
    key, weight = next((key, value) for key, value in bare.model_state_dict().items()
        if ".blocks.0.attn." in key and key.endswith("weight") and value.ndim == 2)
    target = key.removesuffix(".weight")
    state = {**banks, target + ".lora_A.weight": torch.ones(1, weight.shape[1], dtype=dtype) * .02,
             target + ".lora_B.weight": torch.ones(weight.shape[0], 1, dtype=dtype) * .02}
    patched, hooks, _ = pdd._apply_dynamic_lora(bare, state, strength)
    assert hooks == 1
    # Only tiny header dimensions are overridden, not PDD fusion or execution.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(pdd, "PDD_HEAD_SPECS", {key: (tuple(value.shape), value.dtype) for key, value in banks.items()})
        head = pdd.PDDHeadFinalLayer(final, *banks.values(), strength=strength, variant=variant)
    injection = pdd._create_pdd_runtime_injection(patched.get_injections(pdd.PDD_INJECTION_KEY)[0], final, head)
    patched.set_injections(pdd.PDD_INJECTION_KEY, [injection])
    patched.model_options["transformer_options"]["minimax_h3_pdd_final"] = head
    patched.add_wrapper_with_key("diffusion_model", pdd.PDD_WRAPPER_KEY, pdd.pdd_forward_wrapper)
    patched.set_attachments(pdd.PDD_ATTACHMENT_KEY, {"schema": "t8_minimax_h3_pdd_8step_setup_v2",
        "lora": {"application_mode": split.DYNAMIC, "strength": strength},
        "base": {"variant_declared_by_user": variant}, "test_boundary": "tiny_not_pretrained"})
    return patched


def stage_inputs(stage=split.STAGES[0], strength=.5, variant="FL2VA", effects="none", source=None, dtype=torch.float32):
    prepared = dynamic_model(strength, variant, dtype)
    source = source_latent(True) if source is None else source
    positive = conditioning()
    if "relay" in effects:
        prepared, positive, _ = paired(prepared)
    prepared, sampler, full = sampling.setup_dual_clock_sampling(prepared, source, 8, 12., 3., "euler", "simple")
    bound, sampler, sigmas, context, report = split.build_stage(prepared, source, full, stage)
    runtime = None
    if "eav" in effects:
        bound, runtime, _ = eav.apply_stage_eav(bound, sigmas, source, context,
            eav.EAVConfig("apply_exp" if "apply" in effects else "report_only", tau=.2,
                          start_video_progress=0., end_video_progress=1., g_hard_limit=3.))
    return (RandomNoise.execute(73).result[0], BasicGuider.execute(bound, positive).result[0], sampler,
            sigmas, source, context), runtime, json.loads(report)


@pytest.mark.parametrize("stage", split.STAGES)
@pytest.mark.parametrize("strength", [.5, 1.])
def test_real_core_dynamic_pdd_single_stage_completes(stage, strength):
    args, _, _ = stage_inputs(stage, strength)
    result = SamplerCustomAdvanced.execute(*args[:-1]).result
    assert all(torch.isfinite(x).all() for x in result[0]["samples"].unbind())
    head = args[1].model_patcher.model_options["transformer_options"]["minimax_h3_pdd_final"]
    assert head.selection_count == 4
    assert head.block_index == args[-1].end - 1


@pytest.mark.parametrize("stage", split.STAGES)
@pytest.mark.parametrize("strength", [0., .5, 1.])
@pytest.mark.parametrize("effects", ["none", "relay_eav", "relay_eav_apply"])
def test_dynamic_stage_cold_hot_rebuilt_portable_identity(stage, strength, effects):
    args, runtime, report = stage_inputs(stage, strength, effects=effects)
    from h3_audio_t8_pkg.modular_sampling import pdd_dynamic
    pdd_dynamic.describe(args[1].model_patcher)  # Surface the actual adapter gap, not a generic False.
    result = sample_stage(*args)[2]
    receipt = result.verify()
    if "apply" not in effects:
        # Independent old public graph reference: loader Euler/simple full
        # table, then HIGH's original separate dual-clock geometry setup.
        original = dynamic_model(strength)
        positive = conditioning()
        if "relay" in effects:
            original, positive, _ = paired(original)
        original, sampler, full = sampling.setup_dual_clock_sampling(original, args[-2], 8, 12., 3., "euler", "simple")
        begin = 0 if stage == split.STAGES[0] else 4
        if begin:
            original, sampler, _ = sampling.setup_dual_clock_sampling(original, args[-2], 8, 12., 3.,
                                                                     "dual_clock_euler", "native_flow")
        expected = SamplerCustomAdvanced.execute(args[0], BasicGuider.execute(original, positive).result[0],
                                                  sampler, full[begin:begin + 5], args[-2]).result
        for actual, reference in zip((result.output, result.denoised_output), expected):
            assert all(torch.equal(a, b) for a, b in zip(actual["samples"].unbind(), reference["samples"].unbind()))
    assert receipt["portable_identity"] is receipt["verified_recipe_completion"] is True
    assert report["legacy_dynamic_persistent_identity_adapted"] is True
    assert receipt["execution"]["denoiser_evaluations"] == 4
    assert sample_stage(*args)[2].verify()["request_sha256"] == receipt["request_sha256"]
    assert sample_stage(*stage_inputs(stage, strength, effects=effects)[0])[2].verify()["request_sha256"] == receipt["request_sha256"]
    if runtime is not None:
        assert runtime.snapshot()["status"] == ("observed_apply_exp" if "apply" in effects else "observed_report_only")


@pytest.mark.parametrize("stage", split.STAGES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_effective_backbone_cast_and_frozen_native_av_roundtrip(stage, dtype, tmp_path):
    args, _, _ = stage_inputs(stage, dtype=dtype)
    result = sample_stage(*args)[2]
    assert result.verify()["portable_identity"] is True
    assert sample_stage(*args)[2].verify()["request_sha256"] == result.verify()["request_sha256"]
    path, digest, _ = save_stage(result, tmp_path)
    loaded = load_stage(tmp_path, path, digest, stage)[0]
    assert all(torch.equal(a, b) for a, b in zip(result.output["samples"].unbind(), loaded["samples"].unbind()))


@pytest.mark.parametrize("variant", pdd.PDD_VARIANTS)
@pytest.mark.parametrize("strength", [.5, 1.])
def test_new_process_dynamic_high_only_with_real_backbone_relay_eav(tmp_path, variant, strength):
    from h3_audio_t8_pkg import learned_latent_upscale_advanced as learned
    low = sample_stage(*stage_inputs(strength=strength, variant=variant, effects="relay_eav_apply")[0])[2]
    path, digest, _ = save_stage(low, tmp_path)
    high, _, _ = learned.reconcile_two_pass_h3_latent(low.denoised_output, source_latent(True), conditioning(), "auto")
    args, runtime, _ = stage_inputs(split.STAGES[1], strength, variant, "relay_eav_apply", high)
    expected = sample_stage(*args)[2].verify()["outputs"]
    assert runtime.snapshot()["status"] == "observed_apply_exp"
    code = """
import json,runpy,sys
runpy.run_path('tools/check_director_d1_cpu.py',run_name='config')
runpy.run_path('tests/conftest.py')
from test_modular_pdd_dynamic import stage_inputs,source_latent,conditioning
from h3_audio_t8_pkg.modular_sampling.storage import load_stage
from h3_audio_t8_pkg.modular_sampling.results import sample_stage
from h3_audio_t8_pkg import learned_latent_upscale_advanced as learned
import torch
low=load_stage(sys.argv[1],sys.argv[2],sys.argv[3],'pdd_low_0_4')[1]
high,_,_=learned.reconcile_two_pass_h3_latent(low,source_latent(True),conditioning(),'auto')
args,runtime,_=stage_inputs('pdd_high_4_8',float(sys.argv[5]),sys.argv[4],'relay_eav_apply',high)
receipt=sample_stage(*args)[2].verify()
assert receipt['execution']['denoiser_evaluations']==4
assert receipt['portable_identity'] and runtime.snapshot()['status']=='observed_apply_exp'
assert not torch.cuda.is_initialized()
print('RESULT='+json.dumps(receipt['outputs']))
"""
    child = subprocess.run([sys.executable, "-c", code, str(tmp_path), path, digest, variant, str(strength)],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=45)
    assert child.returncode == 0, child.stderr
    actual = json.loads(next(line[7:] for line in child.stdout.splitlines() if line.startswith("RESULT=")))
    assert actual == expected


def test_projection_never_ejects_or_mutates_live_heads_backbone_or_clone():
    from h3_audio_t8_pkg.modular_sampling import pdd_dynamic
    args, _, _ = stage_inputs()
    selected = args[1].model_patcher
    before = pdd_dynamic.describe(selected)
    sample_stage(*args)
    # Exercise the explicit active lifecycle as well as Core's cleanup path.
    selected.inject_model()
    final = selected.get_model_object("diffusion_model.final_layer")
    head = selected.model_options["transformer_options"]["minimax_h3_pdd_final"]
    marker = final._t8_pdd_final_layer_injection_owner
    original_forward = final.forward
    count = head.selection_count
    try:
        for candidate in (selected, selected.clone()):
            view, contract = pdd_dynamic.project(candidate)
            assert contract == before
            assert view.model is not selected.model
            assert view.get_model_object("diffusion_model.final_layer") is not final
            assert final.forward == original_forward and final._t8_pdd_final_layer_injection_owner is marker
            assert head.selection_count == count
            assert selected.get_injections(pdd.PDD_INJECTION_KEY)
    finally:
        selected.eject_model()
    assert pdd_dynamic.describe(selected) == before
    assert sample_stage(*args)[2].verify()["portable_identity"] is True


@pytest.mark.parametrize("damage", ["backbone", "strength", "head", "forward", "offload", "extra_member"])
def test_changes_to_bound_dynamic_execution_cannot_restore_old_identity(damage):
    from h3_audio_t8_pkg.modular_sampling import pdd_dynamic
    args, _, _ = stage_inputs()
    selected = args[1].model_patcher
    runtime, _ = pdd_dynamic.injection(selected.get_injections(pdd.PDD_INJECTION_KEY)[0],
                                      pdd._create_pdd_runtime_injection)
    backbone, _ = pdd_dynamic.injection(runtime["backbone_injection"], pdd._create_offloading_bypass_injections)
    manager = pdd_dynamic.closure(backbone["native_injection"].inject,
        pdd_dynamic.BypassInjectionManager.create_injections, "inject_all")["self"]
    adapter = manager.hooks[0].adapter
    head = selected.model_options["transformer_options"]["minimax_h3_pdd_final"]
    if damage == "backbone":
        adapter.weights[0][0, 0] += .1
    elif damage == "strength":
        adapter.multiplier = .9
    elif damage == "head":
        head.video_weight[0, 0, 0] += .1
    elif damage == "forward":
        head.forward = lambda *a, **kw: None
    elif damage == "offload":
        runtime["backbone_injection"].eject = lambda *a: None
    else:
        adapter.custom_call = lambda *a: None
    with pytest.raises(ValueError, match="changed after"):
        split.validate_stage(selected, args[3], args[4], args[5])


def test_unknown_extra_injection_preserved_and_sampling_error_propagates_then_recovers():
    from comfy.patcher_extension import PatcherInjection
    args, _, _ = stage_inputs()
    selected = args[1].model_patcher
    calls = []
    extra = PatcherInjection(lambda patcher: calls.append("in"), lambda patcher: calls.append("out"))
    selected.set_injections("user_extra", [extra])
    result = sample_stage(*args)[2]
    assert calls and result.verify()["portable_identity"] is False
    assert selected.get_injections("user_extra") == [extra]
    def fail(executor, *a, **kw):
        raise RuntimeError("user-forward-error")
    selected.add_wrapper_with_key("diffusion_model", "user_failure", fail)
    with pytest.raises(RuntimeError, match="user-forward-error"):
        sample_stage(*args)
    selected.remove_wrappers_with_key("diffusion_model", "user_failure")
    before_fields = [key for key, value in vars(args[1]).items() if callable(value)]
    recovered = sample_stage(*args)[2].verify()
    assert recovered["verified_recipe_completion"] is True, {
        "execution": recovered["execution"], "callable_guider_fields_before": before_fields}


@pytest.mark.parametrize("block", range(8))
def test_four_and_seven_argument_final_calls_have_identical_legacy_math(block):
    prepared = dynamic_model(.5)
    head = prepared.model_options["transformer_options"]["minimax_h3_pdd_final"]
    head.select_for_sigma(float(pdd.pdd_runtime_sigmas()[block]))
    x = torch.arange(4 * 24, dtype=torch.float32).reshape(4, 24) / 100.
    t = torch.ones(2, 24)
    segments = ((0, 2, 0), (2, 4, 1))
    old = head(x, t, *segments)
    new = head(x, t, *segments, pdd.pdd_runtime_sigmas()[block], pdd.pdd_runtime_sigmas(), (12., 3.))
    assert all(torch.equal(a, b) for a, b in zip(old, new))


@pytest.mark.parametrize("field", ["inner_model", "custom_callback"])
def test_recovery_does_not_authenticate_an_arbitrary_callable_guider_field(field):
    args, _, _ = stage_inputs()
    setattr(args[1], field, lambda *a, **kw: None)
    result = sample_stage(*args)[2].verify()
    assert result["portable_identity"] is result["verified_recipe_completion"] is False
    assert result["execution"]["denoiser_evaluations"] == 4


@pytest.mark.parametrize("stage", split.STAGES)
def test_extra_user_weight_lora_is_not_removed_by_dynamic_projection(stage):
    from comfy.weight_adapter.lora import LoRAAdapter
    args, _, _ = stage_inputs(stage)
    selected = args[1].model_patcher
    key, value = next((key, value) for key, value in selected.model_state_dict().items()
        if ".blocks.0.mlp." in key and key.endswith("weight") and value.ndim == 2)
    adapter = LoRAAdapter({"up", "down"}, (torch.ones(value.shape[0], 1) * .2,
        torch.ones(1, value.shape[1]) * .2, 1., None, None, None))
    assert selected.add_patches({key: adapter}, .7) == [key]
    result = sample_stage(*args)[2]
    assert result.verify()["portable_identity"] is True
    assert selected.patches[key][0][1] is adapter
    plain = sample_stage(*stage_inputs(stage)[0])[2]
    assert any(not torch.equal(a, b) for a, b in zip(result.output["samples"].unbind(), plain.output["samples"].unbind()))
