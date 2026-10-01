"""Public RF stage semantics, actual effects, and anchored frozen state."""
import json
from dataclasses import replace
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from comfy_extras.nodes_custom_sampler import BasicGuider, RandomNoise, SamplerCustomAdvanced

from h3_audio_t8_pkg import detail_sampling_advanced as detail
from h3_audio_t8_pkg.modular_sampling import rf_stages as rf, eav, detail_effects
from h3_audio_t8_pkg.modular_sampling.results import sample_stage
from h3_audio_t8_pkg.modular_sampling.results import _identity_difference_path
from h3_audio_t8_pkg.modular_sampling.storage import save_stage, load_stage
from h3_audio_t8_pkg.sampling import native_flow_sigmas
from test_modular_rf_restart import source_latent, mixer_kwargs, run
from test_fast_h3_v2_core_sampler import model
from test_progressive_sampling_runtime import conditioning
from test_progressive_relay import paired


def effects_model(bare, source, effects):
    if "bias" in effects:
        bare = detail.setup_model_time_bias_sampling(bare, source, steps=4, shift_video=12., shift_audio=3.,
            bias=-.1, start_progress=0., end_progress=1., bias_domain="video_sigma")[0]
    if "stg" in effects:
        bare = detail.apply_h3_spatiotemporal_guidance(bare, scale=.3, double_blocks="0",
            start_progress=0., end_progress=1., shift_video=12., rescale=0.)[0]
    return bare


def inputs(stage="rf_base", effects="none", relay=False, mode=None, source=None, original=None, restart_steps=3,
           bare=None, sigma_dtype=torch.float32):
    source = source_latent(True) if source is None else source
    original = source_latent(True) if original is None else original
    bare, positive = model() if bare is None else bare, conditioning()
    if relay:
        bare, positive, _ = paired(bare)
    bare = effects_model(bare, source, effects)
    built = (rf.build_base_stage(bare, source, native_flow_sigmas(4, 12.).to(sigma_dtype)) if stage == "rf_base"
             else rf.build_restart_stage(bare, rf.handoff(source, original), restart_steps=restart_steps))
    prepared, sampler, sigmas, context, _ = built
    runtime = None
    if mode is not None:
        prepared, runtime, _ = eav.apply_stage_eav(prepared, sigmas, source, context,
            eav.EAVConfig(mode, tau=.2, start_video_progress=0., end_video_progress=1., g_hard_limit=3.))
    return (RandomNoise.execute(73).result[0], BasicGuider.execute(prepared, positive).result[0],
            sampler, sigmas, source, context), runtime


@pytest.mark.parametrize("stage", rf.STAGES)
@pytest.mark.parametrize("effects", ["none", "bias", "stg", "bias_stg"])
@pytest.mark.parametrize("relay", [False, True])
@pytest.mark.parametrize("mode", ["report_only", "apply_exp"])
def test_own_effect_clocks_portable_cold_hot_results_and_restore(tmp_path, stage, effects, relay, mode):
    args, runtime = inputs(stage, effects, relay, mode)
    contract = detail_effects.describe(args[1].model_patcher)
    assert contract["known"] is True
    result = sample_stage(*args)[2]
    receipt = result.verify()
    assert receipt["verified_recipe_completion"] is receipt["portable_identity"] is True
    assert sample_stage(*args)[2].verify()["request_sha256"] == receipt["request_sha256"]
    assert sample_stage(*inputs(stage, effects, relay, mode)[0])[2].verify()["request_sha256"] == receipt["request_sha256"]
    audit = runtime.snapshot()
    assert audit["status"] == ("observed_report_only" if mode == "report_only" else "observed_apply_exp")
    assert audit["planned_integrator_intervals"] == args[-1].steps
    assert audit["completed_forwards"] == args[-1].steps * (2 if "stg" in effects else 1)
    assert rf.ANCHOR in result.output
    if mode == "report_only":
        reference = SamplerCustomAdvanced.execute(*inputs(stage, effects, relay)[0][:5]).result
        for actual, expected in zip((result.output, result.denoised_output), reference):
            assert all(torch.equal(a, b) for a, b in zip(actual["samples"].unbind(), expected["samples"].unbind()))
    path, digest, _ = save_stage(result, tmp_path)
    loaded = load_stage(tmp_path, path, digest, stage)[0]
    assert rf.handoff(loaded).template_identity == rf.handoff(result.output).template_identity
    assert not torch.cuda.is_initialized()


def test_zero_step_is_explicit_empty_identity_and_never_one_fake_step(tmp_path, monkeypatch):
    args, runtime = inputs("rf_restart", "bias_stg", True, "apply_exp", restart_steps=0)
    context = args[-1]
    assert context.steps == 0 and context.trajectory_sigmas == ()
    assert context.input_semantics == context.output_semantics == context.denoised_semantics == "identity_noop"
    def forbidden(*args, **kwargs):
        raise AssertionError("No-op must not call the model")
    monkeypatch.setattr(args[1].model_patcher.model.diffusion_model, "forward", forbidden)
    result = sample_stage(*args)[2]
    assert result.verify()["verified_recipe_completion"] is True
    assert result.verify()["execution"]["denoiser_evaluations"] == 0
    assert runtime.snapshot()["completed_forwards"] == 0
    assert all(torch.equal(a, b) for a, b in zip(result.output["samples"].unbind(), args[-2]["samples"].unbind()))
    # The deliberate foreign model hook is nonportable, not a reason to block.
    assert result.verify()["portable_identity"] is False
    with pytest.raises(ValueError):
        replace(context, input_semantics="pretend_sampling")


def test_unknown_time_wrapper_is_preserved_and_runs_without_false_clock_or_portability():
    source = source_latent()
    bare = model()
    calls = []
    def user_wrapper(apply_model, options):
        calls.append(1)
        return apply_model(options["input"], options["timestep"] * .99, **options["c"])
    bare.set_model_unet_function_wrapper(user_wrapper)
    prepared, sampler, sigmas, context, _ = rf.build_base_stage(bare, source, native_flow_sigmas(4, 12.))
    prepared, runtime, _ = eav.apply_stage_eav(prepared, sigmas, source, context,
        eav.EAVConfig("report_only", tau=.2, g_hard_limit=3.))
    result = sample_stage(RandomNoise.execute(73).result[0], BasicGuider.execute(prepared, conditioning()).result[0],
                          sampler, sigmas, source, context)[2]
    assert calls and prepared.model_options["model_function_wrapper"] is user_wrapper
    assert result.verify()["portable_identity"] is False
    assert runtime.snapshot()["status"] == "unverified_incomplete_stage_coverage"


def test_anchor_and_actual_restart_seed_are_bound_not_just_display_labels():
    args, _ = inputs("rf_restart")
    profile = json.loads(args[-1].profile)
    assert profile["restart_report"]["restart_seed"] == 1234
    sampler = args[2]
    changed = json.loads(sampler.report_json)
    changed["restart_seed"] += 1
    sampler.report_json = json.dumps(changed)
    with pytest.raises(ValueError, match="parameters differ"):
        sample_stage(*args)


@pytest.mark.parametrize("effects", ["none", "bias_stg"])
@pytest.mark.parametrize("sigma_dtype", [torch.float32, torch.float64])
def test_new_process_loads_base_anchor_and_only_executes_restart(
        tmp_path, effects, sigma_dtype, single_thread_rf):
    first = sample_stage(*inputs("rf_base", effects, True, "apply_exp", sigma_dtype=sigma_dtype)[0])[2]
    path, digest, _ = save_stage(first, tmp_path)
    expected_receipt = sample_stage(*inputs("rf_restart", effects, True, "apply_exp", source=first.output,
                                            original=first.output[rf.ANCHOR])[0])[2].verify()
    expected = expected_receipt["outputs"]
    code = """
import json,runpy,sys
runpy.run_path('tools/check_director_d1_cpu.py',run_name='config')
from comfy.cli_args import args as comfy_args
comfy_args.use_pytorch_cross_attention = sys.argv[5] == 'attention_pytorch'
runpy.run_path('tests/conftest.py')
from test_modular_rf_stages import inputs,rf
from h3_audio_t8_pkg.modular_sampling.storage import load_stage
from h3_audio_t8_pkg.modular_sampling.results import sample_stage
import torch
first=load_stage(sys.argv[1],sys.argv[2],sys.argv[3],'rf_base')[0]
def forbidden(*args,**kwargs):
    raise AssertionError('Restoring RF must not execute or set up the base')
rf.build_base_stage=forbidden
handoff=rf.handoff(first)
args,runtime=inputs('rf_restart',sys.argv[4],True,'apply_exp',source=first,original=handoff.original_template)
receipt=sample_stage(*args)[2].verify()
assert receipt['execution']['denoiser_evaluations']==3 and receipt['portable_identity']
assert runtime.snapshot()['status']=='observed_apply_exp'
assert not torch.cuda.is_initialized()
print('RESULT='+json.dumps(receipt['outputs']))
print('REQUEST='+receipt['request_sha256'])
print('REQUEST_BODY='+json.dumps(receipt['request']))
"""
    backend = expected_receipt["request"]["model"]["backend"]["name"]
    assert backend in {"attention_pytorch", "attention_sub_quad"}
    child = subprocess.run([sys.executable, "-c", code, str(tmp_path), path, digest, effects, backend],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=45)
    assert child.returncode == 0, child.stderr
    actual = json.loads(next(line[7:] for line in child.stdout.splitlines() if line.startswith("RESULT=")))
    request_sha = next(line[8:] for line in child.stdout.splitlines() if line.startswith("REQUEST="))
    request_body = json.loads(next(line[13:] for line in child.stdout.splitlines()
                                   if line.startswith("REQUEST_BODY=")))
    assert request_sha == expected_receipt["request_sha256"], _identity_difference_path(
        expected_receipt["request"], request_body) + ": " + str((
            expected_receipt["request"]["model"]["backend"]["name"],
            request_body["model"]["backend"]["name"]))
    assert actual == expected, _identity_difference_path(expected, actual)


@pytest.fixture
def single_thread_rf():
    """Match the child Core's one-thread CPU math for exact hash comparison."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def test_registered_node_slots_connect_without_running_a_hidden_first_pass():
    from h3_audio_t8_pkg.modular_sampling.nodes import (
        MiniMaxH3RFBaseStageSetupEXPT8 as Base, MiniMaxH3RFHandoffEXPT8 as Handoff,
        MiniMaxH3RFRestartStageSetupEXPT8 as Restart)
    source, bare = source_latent(), model()
    base = Base.execute(bare, source, native_flow_sigmas(4, 12.)).result
    first = sample_stage(RandomNoise.execute(73).result[0], BasicGuider.execute(base[0], conditioning()).result[0],
                         base[1], base[2], source, base[3])[0]
    hand = Handoff.execute(first).result
    second = Restart.execute(bare, hand[0]).result
    assert second[3] is first and second[4].stage == "rf_restart"
    assert second[4].steps == 3
    assert json.loads(hand[2])["model_calls"] == 0


@pytest.mark.parametrize("entry", ["standalone", "detail_mixer", "two_pass_detail_mixer"])
@pytest.mark.parametrize("effects", ["none", "bias", "stg", "bias_stg"])
@pytest.mark.parametrize("masked", [False, True])
def test_public_rf_pair_matches_all_three_old_entrypoints(entry, effects, masked):
    bare, source = model(), source_latent(masked)
    kwargs = mixer_kwargs()
    if entry == "standalone":
        bare = effects_model(bare, source, effects)
        prepared, sampler, sigmas, _ = detail.setup_rectified_flow_restart_sampling(bare, source,
            steps=4, shift_video=12., shift_audio=3., restart_video_sigma=.15, restart_steps=3, restart_seed=17)
        stage_model = bare
    else:
        kwargs.update(enable_model_time_bias="bias" in effects, enable_stg="stg" in effects)
        factory = detail.setup_detail_mixer_sampling if entry == "detail_mixer" else detail.setup_two_pass_detail_mixer_sampling
        extra = {"steps": 4, "profile": "custom_strict"} if entry == "detail_mixer" else {"refine_sigmas": torch.tensor([.5, .35, .1, 0.])}
        prepared, sampler, sigmas, *_ = factory(bare, source, **kwargs, **extra)
        stage_model = factory(bare, source, **{**kwargs, "enable_restart": False}, **extra)[0]
    expected = run(prepared, sampler, sigmas, source)
    first_model, first_sampler, first_sigmas, first_context, _ = rf.build_base_stage(stage_model, source, sigmas)
    first = sample_stage(RandomNoise.execute(73).result[0], BasicGuider.execute(first_model, conditioning()).result[0],
        first_sampler, first_sigmas, source, first_context)[2]
    prepared, sampler, sigmas, context, _ = rf.build_restart_stage(stage_model, rf.handoff(first.output), restart_seed=17)
    final = sample_stage(RandomNoise.execute(73).result[0], BasicGuider.execute(prepared, conditioning()).result[0],
        sampler, sigmas, first.output, context)[2]
    assert first.verify()["portable_identity"] is final.verify()["portable_identity"] is True
    for actual, target in zip((final.output, final.denoised_output), expected):
        assert all(torch.equal(a, b) for a, b in zip(actual["samples"].unbind(), target["samples"].unbind()))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("effects", ["none", "bias_stg"])
def test_restart_preserves_external_base_sigma_precision_across_frozen_result(tmp_path, dtype, masked, effects):
    source = source_latent(masked)
    bare = effects_model(model(), source, effects)
    prepared, sampler, sigmas, _ = detail.setup_rectified_flow_restart_sampling(bare, source,
        steps=4, shift_video=12., shift_audio=3., restart_video_sigma=.3, restart_steps=3, restart_seed=17)
    sigmas = sigmas.to(dtype=dtype)
    expected = run(prepared, sampler, sigmas, source)[0]
    built = rf.build_base_stage(bare, source, sigmas)
    first = sample_stage(RandomNoise.execute(73).result[0], BasicGuider.execute(built[0], conditioning()).result[0],
        built[1], built[2], source, built[3])[2]
    raw = rf.rf_restart.retained_endpoint(first.output)
    assert raw is not None
    assert raw.dtype == (torch.float64 if dtype == torch.float64 else torch.float32)
    assert rf.rf_restart.RAW_ENDPOINT not in first.denoised_output
    if dtype == torch.float64:
        assert not torch.equal(raw, raw.float().double()), "Fixture must exercise Core's lossy public boundary"
    path, digest, _ = save_stage(first, tmp_path)
    loaded = load_stage(tmp_path, path, digest, "rf_base")[0]
    built = rf.build_restart_stage(bare, rf.handoff(loaded), restart_video_sigma=.3, restart_seed=17)
    assert built[2].dtype == dtype
    actual = sample_stage(RandomNoise.execute(73).result[0], BasicGuider.execute(built[0], conditioning()).result[0],
        built[1], built[2], loaded, built[3])[0]
    assert all(torch.equal(a, b) for a, b in zip(actual["samples"].unbind(), expected["samples"].unbind()))


@pytest.mark.parametrize("target", ["packed", "visible"])
def test_retained_endpoint_rejects_tampering_even_before_handoff(target):
    output = sample_stage(*inputs()[0])[0]
    if target == "packed":
        output[rf.rf_restart.RAW_ENDPOINT]["packed"].add_(.01)
    else:
        output["samples"].unbind()[0].add_(.01)
    with pytest.raises(ValueError, match="retained endpoint"):
        rf.handoff(output)


def test_external_standard_base_can_bind_template_and_explicit_sigma_dtype_without_inventing_raw_state():
    source, bare = source_latent(), model()
    built = rf.build_base_stage(bare, source, native_flow_sigmas(4, 12.).double())
    first = run(built[0], built[1], built[2], source)[0]
    assert rf.rf_restart.RAW_ENDPOINT not in first and rf.SIGMA_DTYPE not in first
    bound = rf.handoff(first, source, built[2])
    assert rf.rf_restart.retained_endpoint(bound.completed_av) is None
    assert rf.SIGMA_DTYPE not in first  # Binding must not mutate the user's LATENT.
    prepared = rf.build_restart_stage(bare, bound)
    assert prepared[2].dtype == torch.float64


@pytest.mark.parametrize("stage", rf.STAGES)
def test_nonzero_lora_native_video_mask_and_all_external_effects_remain_portable(tmp_path, stage):
    from comfy.weight_adapter.lora import LoRAAdapter
    bare = model()
    key, weight = next((key, value) for key, value in bare.model_state_dict().items()
                       if key.endswith("weight") and value.ndim == 2)
    adapter = LoRAAdapter({"up", "down"}, (torch.ones(weight.shape[0], 1) * .02,
                                          torch.ones(1, weight.shape[1]) * .02, 1., None, None, None))
    assert bare.add_patches({key: adapter}, strength_patch=.7) == [key]
    args, runtime = inputs(stage, "bias_stg", True, "apply_exp", bare=bare)
    result = sample_stage(*args)[2]
    selected = args[1].model_patcher
    backup = selected.backup[key]
    assert result.verify()["portable_identity"] is True
    assert sample_stage(*args)[2].verify()["request_sha256"] == result.verify()["request_sha256"]
    assert selected.backup[key] is backup and selected.patches[key][0][1] is adapter
    assert runtime.snapshot()["status"] == "observed_apply_exp"
    path, digest, _ = save_stage(result, tmp_path)
    loaded = load_stage(tmp_path, path, digest, stage)[0]
    assert rf.rf_restart.retained_endpoint(loaded) is not None
    assert all(torch.equal(a, b) for a, b in zip(loaded["noise_mask"].unbind(), args[-2]["noise_mask"].unbind()))
