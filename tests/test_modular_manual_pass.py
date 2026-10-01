"""Real tiny CPU sampling against both old manual-second-pass execution paths."""
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from comfy_extras.nodes_custom_sampler import BasicGuider, RandomNoise

from h3_audio_t8_pkg import long_video_in_node_loop_advanced as plain
from h3_audio_t8_pkg import long_video_in_node_loop_effects_advanced as effects_loop
from h3_audio_t8_pkg import freenoise_advanced as freenoise
from h3_audio_t8_pkg.long_video_sampling_plan_advanced import build_long_video_sampling_plan, resolve_long_video_sample_schedules
from h3_audio_t8_pkg.sampling import setup_dual_clock_sampling
from h3_audio_t8_pkg.modular_sampling import manual_pass, eav
from h3_audio_t8_pkg.modular_sampling.noise import build_noise
from h3_audio_t8_pkg.modular_sampling.results import sample_stage
from h3_audio_t8_pkg.modular_sampling.storage import save_stage, load_stage
from test_fast_h3_v2_core_sampler import model
from test_progressive_sampling_runtime import latent, conditioning
from test_progressive_relay import paired

MANUAL = "0.5,0.412,0.35,0"


def inputs(stage="manual_first", *, source=None, bare=None, positive=None, sampler_name="dual_clock_euler",
           scheduler="native_flow", first_steps=4, effects=False, noise_mode="disabled", segment=2, seed=31):
    bare = model() if bare is None else bare
    positive = conditioning() if positive is None else positive
    source = latent() if source is None else source
    prepared, sampler, sigmas, context, _ = manual_pass.build_stage(bare, source, stage, first_steps, MANUAL,
                                                                  12., 3., sampler_name, scheduler)
    runtime = None
    if effects:
        prepared, runtime, _ = eav.apply_stage_eav(prepared, sigmas, source, context,
            eav.EAVConfig("apply_exp", tau=.2, start_video_progress=0., end_video_progress=1., g_hard_limit=3.))
    noise = build_noise(RandomNoise.execute(seed).result[0], noise_mode, 71, .65, segment, model=bare)[0]
    return (noise, BasicGuider.execute(prepared, positive).result[0], sampler, sigmas, source, context), runtime


@pytest.mark.parametrize("backend", ["plain", "effects"])
@pytest.mark.parametrize("sampler_name", ["dual_clock_euler", "euler"])
@pytest.mark.parametrize("scheduler", ["native_flow", "simple"])
@pytest.mark.parametrize("noise_mode", [None, "paper_permutation", "variance_preserving_blend"])
def test_two_public_calls_match_original_manual_two_pass(backend, sampler_name, scheduler, noise_mode):
    bare, source, positive = model(), latent(), conditioning()
    if noise_mode is not None:
        bare, _ = freenoise.build_free_noise_model(bare, mode=noise_mode, base_seed=71, reuse_ratio=.65)
    plan = build_long_video_sampling_plan("manual_second_pass", 0, "video_sigma_linear", MANUAL)[0]
    if backend == "plain":
        expected = plain._sample_one_segment(bare, positive, source, seed=31, steps=4, shift_video=12.,
            shift_audio=3., sampler_name=sampler_name, scheduler=scheduler, segment_index=2, sampling_plan=plan)
    else:
        # Exact outer-loop numerical call sequence (delivery/VAE not executed).
        first, sampler, base = setup_dual_clock_sampling(bare, source, 4, 12., 3., sampler_name, scheduler)
        sigmas, second_sigmas, _ = resolve_long_video_sample_schedules(base, plan, shift_video=12., shift_audio=3.)
        output = effects_loop._sample_prepared_segment(first, positive, source, sampler=sampler, sigmas=sigmas,
                                                      seed=31, segment_index=2)
        second, sampler, _ = setup_dual_clock_sampling(bare, output, 3, 12., 3., sampler_name, scheduler)
        expected = effects_loop._sample_prepared_segment(second, positive, output, sampler=sampler, sigmas=second_sigmas,
                                                         seed=31, segment_index=2)
    options = dict(bare=bare, positive=positive, sampler_name=sampler_name, scheduler=scheduler,
                   noise_mode="from_model_plan", segment=2)
    first_result = sample_stage(*inputs(source=source, **options)[0])[2]
    args, _ = inputs("manual_second", source=first_result.output, **options)
    actual = sample_stage(*args)[2]
    assert all(torch.equal(a, b) for a, b in zip(actual.output["samples"].unbind(), expected["samples"].unbind()))
    assert first_result.verify()["execution"]["denoiser_evaluations"] == 4
    assert actual.verify()["execution"]["denoiser_evaluations"] == 3
    assert actual.verify()["portable_identity"] is True
    assert args[-1].input_semantics == "first_pass_output_with_fresh_noise"


@pytest.mark.parametrize("stage", manual_pass.STAGES)
@pytest.mark.parametrize("with_relay", [False, True])
@pytest.mark.parametrize("with_eav", [False, True])
@pytest.mark.parametrize("sampler_name", ["dual_clock_euler", "euler"])
def test_external_effects_and_noise_roundtrip(tmp_path, stage, with_relay, with_eav, sampler_name):
    bare, positive = model(), conditioning()
    if with_relay:
        bare, positive, _ = paired(bare)
    args, runtime = inputs(stage, bare=bare, positive=positive, sampler_name=sampler_name,
                           effects=with_eav, noise_mode="variance_preserving_blend")
    result = sample_stage(*args)[2]
    cold = result.verify()
    hot = sample_stage(*args)[2].verify()
    assert cold["portable_identity"] is hot["portable_identity"] is True
    assert cold["request_sha256"] == hot["request_sha256"]
    if with_eav:
        audit = runtime.snapshot()
        assert audit["status"] == "observed_apply_exp" and audit["v2_dispatch"] is None
        if with_relay:
            assert audit["relay_attention_calls"] == args[-1].steps
    path, digest, _ = save_stage(result, tmp_path)
    assert load_stage(tmp_path, path, digest, stage)[3].verify() == cold


def test_first_stage_does_not_depend_on_unused_second_schedule():
    bare, source = model(), latent()
    a = manual_pass.build_stage(bare, source, manual_sigmas=MANUAL)[3]
    b = manual_pass.build_stage(bare, source, manual_sigmas="unused text not parsed for first")[3]
    assert a == b
    with pytest.raises(ValueError):
        manual_pass.build_stage(bare, source, "manual_second", manual_sigmas="0.5,0.5,0")


def test_other_sampler_executes_without_false_euler_completion():
    args, runtime = inputs("manual_second", sampler_name="heun", effects=True)
    result = sample_stage(*args)[2].verify()
    assert result["verified_recipe_completion"] is result["portable_identity"] is False
    assert result["execution"]["denoiser_evaluations"] == 5
    assert runtime.snapshot()["status"] == "unverified_incomplete_stage_coverage"
    assert json.loads(args[-1].profile)["sampler"] == "heun"


@pytest.mark.parametrize("sampler_name", ["dual_clock_euler", "euler"])
def test_new_process_only_runs_second_with_external_relay_eav_and_freenoise(tmp_path, sampler_name):
    bare, positive, _ = paired(model())
    options = dict(bare=bare, positive=positive, effects=True, sampler_name=sampler_name,
                   noise_mode="variance_preserving_blend")
    first = sample_stage(*inputs(**options)[0])[2]
    path, digest, _ = save_stage(first, tmp_path)
    expected = sample_stage(*inputs("manual_second", source=first.output, **options)[0])[2].verify()
    code = """
import json,runpy,sys
runpy.run_path('tools/check_director_d1_cpu.py',run_name='config')
runpy.run_path('tests/conftest.py')
from test_modular_manual_pass import inputs,paired,model,manual_pass
from h3_audio_t8_pkg.modular_sampling.storage import load_stage
from h3_audio_t8_pkg.modular_sampling.results import sample_stage
import torch
first=load_stage(sys.argv[1],sys.argv[2],sys.argv[3],'manual_first')[0]
original=manual_pass.build_stage
def second_only(*args,**kwargs):
    assert args[2]=='manual_second'
    return original(*args,**kwargs)
manual_pass.build_stage=second_only
bare,positive,_=paired(model())
args,runtime=inputs('manual_second',source=first,bare=bare,positive=positive,effects=True,
                   sampler_name=sys.argv[4],noise_mode='variance_preserving_blend')
receipt=sample_stage(*args)[2].verify()
assert receipt['execution']['denoiser_evaluations']==3 and receipt['portable_identity']
assert runtime.snapshot()['relay_attention_calls']==3
assert not torch.cuda.is_initialized()
print('RESULT='+json.dumps(receipt['outputs']))
"""
    child = subprocess.run([sys.executable, "-c", code, str(tmp_path), path, digest, sampler_name],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=45)
    assert child.returncode == 0, child.stderr
    actual = json.loads(next(line[7:] for line in child.stdout.splitlines() if line.startswith("RESULT=")))
    assert actual == expected["outputs"]


@pytest.mark.parametrize("noise_mode", [None, "variance_preserving_blend"])
def test_actual_effects_outer_loop_runs_both_calls_with_tiny_core_sampling(monkeypatch, tmp_path, noise_mode):
    # Keep the old outer orchestration/call site live. Only conditioning, EAV
    # composer and file/VAE delivery are labelled boundary doubles from the
    # existing fixture. Native setup, noise and all 46 denoiser calls are real.
    from test_long_video_in_node_loop_effects_advanced import _install_fake_combined_runtime, _kwargs, _relay_plan
    real_sample = effects_loop._sample_prepared_segment
    manifest, events = _install_fake_combined_runtime(monkeypatch, tmp_path)
    segment_models, observed = {}, []
    fake_conditioning = effects_loop.build_prompt_relay_long_video_conditioning

    def prepare(**kwargs):
        values = list(fake_conditioning(**kwargs))
        selected = model()
        if noise_mode is not None:
            selected, _ = freenoise.build_free_noise_model(selected, mode=noise_mode, base_seed=71, reuse_ratio=.65)
        segment_models[int(kwargs["segment_index"])] = selected
        values[0], values[1], values[2] = selected, conditioning(), latent()
        return tuple(values)

    def compose(selected, sigmas, **kwargs):
        index = int(kwargs["segment_index"])
        events.append(("compose_owner", index))
        return selected, {"segment_index": index}, json.dumps({"status": "test_double_identity_eav"})

    def observe(selected, positive, source, **kwargs):
        output = real_sample(selected, positive, source, **kwargs)
        observed.append((kwargs["segment_index"], kwargs["seed"], kwargs["sigmas"].clone(), output))
        return output

    monkeypatch.setattr(effects_loop, "build_prompt_relay_long_video_conditioning", prepare)
    monkeypatch.setattr(effects_loop, "setup_dual_clock_sampling", setup_dual_clock_sampling)
    monkeypatch.setattr(effects_loop, "build_eav_prompt_relay_long_video_model", compose)
    monkeypatch.setattr(effects_loop, "_sample_prepared_segment", observe)
    kwargs = _kwargs(_relay_plan())
    kwargs["long_video_sampling_plan"] = build_long_video_sampling_plan(
        "manual_second_pass", 0, "video_sigma_linear", MANUAL)[0]
    result = effects_loop.run_long_video_in_node_loop_effects(object(), object(), object(), object(), **kwargs)
    assert result[3] == "complete" and len(manifest["segments"]) == 2
    assert [(index, seed, len(sigmas) - 1) for index, seed, sigmas, _ in observed] == [
        (0, 17, 20), (0, 17, 3), (1, 18, 20), (1, 18, 3)]
    for index in (0, 1):
        options = dict(bare=segment_models[index], first_steps=20, noise_mode="from_model_plan", segment=index, seed=17 + index)
        first = sample_stage(*inputs(**options)[0])[0]
        second = sample_stage(*inputs("manual_second", source=first, **options)[0])[0]
        assert all(torch.equal(a, b) for a, b in zip(second["samples"].unbind(), observed[index * 2 + 1][3]["samples"].unbind()))
