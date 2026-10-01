"""Reuse actual public native schedules; binding must not change their math."""
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from comfy_extras.nodes_custom_sampler import BasicGuider, RandomNoise, SamplerCustomAdvanced

from h3_audio_t8_pkg import learned_latent_upscale_advanced as learned, sampling
from h3_audio_t8_pkg.modular_sampling import native_explicit as native, eav, native_dual, manual_pass, rf_stages
from h3_audio_t8_pkg.modular_sampling.results import sample_stage
from h3_audio_t8_pkg.modular_sampling.storage import save_stage, load_stage
from test_fast_h3_v2_core_sampler import model
from test_modular_rf_restart import source_latent
from test_modular_rf_stages import effects_model
from test_progressive_sampling_runtime import conditioning
from test_progressive_relay import paired

ROUTES = (("base_flow", 4, 4), *(('lbh', 4, n) for n in (3, 4, 5)),
          *(('complete', n, r) for n in (8, 20) for r in (3, 4, 5)))


def inputs(route=ROUTES[0], stage="native_low", effects="none", *, source=None, bare=None):
    name, coarse, refine = route
    source = source_latent(True) if source is None else source
    bare, positive = model() if bare is None else bare, conditioning()
    if "relay" in effects:
        bare, positive, _ = paired(bare)
    if "detail" in effects:
        bare = effects_model(bare, source, "bias_stg")
    count = coarse if name == "complete" else 8
    prepared, sampler, full = sampling.setup_dual_clock_sampling(bare, source, count, 12., 3.)
    if name == "base_flow":
        low, high, _ = learned.build_two_pass_sigma_plan(prepared, coarse, refine, .5)
    else:
        low, high, _ = learned.build_learned_two_pass_parity_plan(prepared, 8, 4, refine)
        if name == "complete":
            low = full
    sigmas = low if stage == "native_low" else high
    original = (RandomNoise.execute(31).result[0], BasicGuider.execute(prepared, positive).result[0], sampler, sigmas, source)
    bound, selected, table, context, _ = native.bind_stage(prepared, sampler, sigmas, source, stage)
    assert selected is sampler and table is sigmas
    runtime = None
    if "eav" in effects:
        mode = "apply_exp" if "apply" in effects else "report_only"
        bound, runtime, _ = eav.apply_stage_eav(bound, sigmas, source, context,
            eav.EAVConfig(mode, tau=.2, start_video_progress=0., end_video_progress=1., g_hard_limit=3.))
    return (original[0], BasicGuider.execute(bound, positive).result[0], sampler, sigmas, source, context), runtime, original


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("stage", native.STAGES)
@pytest.mark.parametrize("effects", ["none", "relay_eav", "relay_eav_detail"])
def test_original_public_stage_pair_outputs_effects_and_frozen_state(tmp_path, route, stage, effects):
    args, runtime, original = inputs(route, stage, effects)
    expected = SamplerCustomAdvanced.execute(*original).result
    result = sample_stage(*args)[2]
    for actual, reference in zip((result.output, result.denoised_output), expected):
        assert all(torch.equal(a, b) for a, b in zip(actual["samples"].unbind(), reference["samples"].unbind()))
    receipt = result.verify()
    assert receipt["portable_identity"] is receipt["verified_recipe_completion"] is True
    assert receipt["execution"]["denoiser_evaluations"] == args[-1].steps
    assert sample_stage(*args)[2].verify()["request_sha256"] == receipt["request_sha256"]
    assert sample_stage(*inputs(route, stage, effects)[0])[2].verify()["request_sha256"] == receipt["request_sha256"]
    if runtime is not None:
        assert runtime.snapshot()["status"] == "observed_report_only"
    path, digest, _ = save_stage(result, tmp_path)
    restored = load_stage(tmp_path, path, digest, stage)[0]
    assert all(torch.equal(a, b) for a, b in zip(restored["samples"].unbind(), result.output["samples"].unbind()))


@pytest.mark.parametrize("route", ROUTES)
def test_new_process_only_restores_high_from_low_denoised(tmp_path, route):
    first_args = inputs(route, "native_low", "relay_eav_detail_apply")[0]
    first = sample_stage(*first_args)[2]
    path, digest, _ = save_stage(first, tmp_path)
    # Same-size tiny handoff is deliberate, NOT a full learned-network test.
    high, positive, _ = learned.reconcile_two_pass_h3_latent(first.denoised_output, source_latent(True),
                                                            conditioning(), "auto")
    args = inputs(route, "native_high", "relay_eav_detail_apply", source=high)[0]
    expected = sample_stage(*args)[2].verify()["outputs"]
    code = """
import json,runpy,sys
runpy.run_path('tools/check_director_d1_cpu.py',run_name='config')
runpy.run_path('tests/conftest.py')
from test_modular_native_explicit import inputs,learned,source_latent,conditioning
from h3_audio_t8_pkg.modular_sampling.storage import load_stage
from h3_audio_t8_pkg.modular_sampling.results import sample_stage
import torch
low=load_stage(sys.argv[1],sys.argv[2],sys.argv[3],'native_low')[1]
high,_,_=learned.reconcile_two_pass_h3_latent(low,source_latent(True),conditioning(),'auto')
args,runtime,_=inputs(tuple(json.loads(sys.argv[4])),'native_high','relay_eav_detail_apply',source=high)
receipt=sample_stage(*args)[2].verify()
assert receipt['execution']['denoiser_evaluations']==json.loads(sys.argv[4])[2]
assert receipt['portable_identity'] and runtime.snapshot()['status']=='observed_apply_exp'
assert not torch.cuda.is_initialized()
print('RESULT='+json.dumps(receipt['outputs']))
"""
    child = subprocess.run([sys.executable, "-c", code, str(tmp_path), path, digest, json.dumps(route)],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=45)
    assert child.returncode == 0, child.stderr
    actual = json.loads(next(line[7:] for line in child.stdout.splitlines() if line.startswith("RESULT=")))
    assert actual == expected


@pytest.mark.parametrize("change", ["values", "dtype", "stage", "geometry"])
def test_own_stage_binding_rejects_changed_contract(change):
    from dataclasses import replace
    args, _, _ = inputs()
    args = list(args)
    if change == "values":
        args[3] = args[3].clone()
        args[3][1] -= 1e-5
    elif change == "dtype":
        args[3] = args[3].double()
    elif change == "stage":
        args[-1] = replace(args[-1], stage="native_high")
    else:
        args[-1] = replace(args[-1], video_shape=(1,24,2,4,4))
    with pytest.raises(ValueError):
        sample_stage(*args)


def test_unadapted_heun_runs_without_false_native_completion():
    source = source_latent()
    prepared, sampler, sigmas = sampling.setup_dual_clock_sampling(model(), source, 3, 12., 3., "heun")
    prepared, sampler, sigmas, context, report = native.bind_stage(prepared, sampler, sigmas, source)
    prepared, runtime, _ = eav.apply_stage_eav(prepared, sigmas, source, context,
        eav.EAVConfig("report_only", tau=.2, g_hard_limit=3.))
    result = sample_stage(RandomNoise.execute(31).result[0], BasicGuider.execute(prepared, conditioning()).result[0],
                          sampler, sigmas, source, context)[2]
    assert json.loads(report)["portable_completion_sampler_adapted"] is False
    assert result.verify()["execution"]["denoiser_evaluations"] == 5
    assert result.verify()["portable_identity"] is result.verify()["verified_recipe_completion"] is False
    assert runtime.snapshot()["status"] == "unverified_incomplete_stage_coverage"


@pytest.mark.parametrize("adapter", [native_dual, manual_pass, rf_stages])
def test_rebinding_changes_only_owned_inert_descriptors_on_new_branch(adapter):
    args, _, _ = inputs()
    previous, source = args[1].model_patcher, args[-2]
    marker = object()
    previous.set_attachments("user_attachment", marker)
    if adapter is rf_stages:
        built = adapter.build_base_stage(previous, source, torch.tensor([1., .5, 0.]))
    else:
        built = adapter.build_stage(previous, source)
    assert previous.get_attachment(native.KEY) is not None
    assert built[0].get_attachment(native.KEY) is None
    assert built[0].get_attachment("user_attachment") is marker
    bound = native.bind_stage(built[0], built[1], built[2], source)[0]
    assert bound.get_attachment(adapter.KEY) is None and bound.get_attachment(native.KEY) is not None
    assert built[0].get_attachment(adapter.KEY) is not None


@pytest.mark.parametrize("shifts", [(6., 2.), (18., 4.)])
@pytest.mark.parametrize("family", ["base_flow", "lbh"])
def test_each_model_uses_its_own_clock_not_a_shared_or_guessed_lbh_shift(shifts, family):
    source = source_latent()
    prepared, sampler, _ = sampling.setup_dual_clock_sampling(model(), source, 8, *shifts)
    tables = (learned.build_two_pass_sigma_plan(prepared, 4, 4, .4) if family == "base_flow"
              else learned.build_learned_two_pass_parity_plan(prepared, 8, 4, 4))
    built = native.bind_stage(prepared, sampler, tables[1], source, "native_high")
    assert (built[3].video_shift, built[3].audio_shift) == shifts
    if family == "lbh":
        assert torch.equal(built[2], torch.tensor([.9035, .8, .6316, .3158, 0.]))
    noise = RandomNoise.execute(31).result[0]
    expected = SamplerCustomAdvanced.execute(noise, BasicGuider.execute(prepared, conditioning()).result[0],
                                              sampler, tables[1], source).result[0]
    actual = sample_stage(noise, BasicGuider.execute(built[0], conditioning()).result[0],
                          built[1], built[2], source, built[3])[0]
    assert all(torch.equal(a, b) for a, b in zip(actual["samples"].unbind(), expected["samples"].unbind()))


@pytest.mark.parametrize("stage", native.STAGES)
def test_nonzero_lora_native_mask_and_effects_are_not_removed_for_portable_identity(stage, tmp_path):
    from comfy.weight_adapter.lora import LoRAAdapter
    bare = model()
    key, weight = next((key, value) for key, value in bare.model_state_dict().items()
                       if key.endswith("weight") and value.ndim == 2)
    adapter = LoRAAdapter({"up", "down"}, (torch.ones(weight.shape[0], 1) * .02,
                                          torch.ones(1, weight.shape[1]) * .02, 1., None, None, None))
    assert bare.add_patches({key: adapter}, strength_patch=.7) == [key]
    args, runtime, _ = inputs(stage=stage, effects="relay_eav_detail_apply", bare=bare)
    result = sample_stage(*args)[2]
    selected = args[1].model_patcher
    backup = selected.backup[key]
    assert result.verify()["portable_identity"] is True
    assert sample_stage(*args)[2].verify()["request_sha256"] == result.verify()["request_sha256"]
    assert selected.backup[key] is backup and selected.patches[key][0][1] is adapter
    assert runtime.snapshot()["status"] == "observed_apply_exp"
    path, digest, _ = save_stage(result, tmp_path)
    loaded = load_stage(tmp_path, path, digest, stage)[0]
    assert all(torch.equal(a, b) for a, b in zip(loaded["noise_mask"].unbind(), args[-2]["noise_mask"].unbind()))
