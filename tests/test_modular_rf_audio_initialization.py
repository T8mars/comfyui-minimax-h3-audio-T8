"""One-time AV clock initialization, without changing legacy RF semantics."""
import json
from pathlib import Path
import subprocess
import sys
import pytest
import torch
from h3_audio_t8_pkg.modular_sampling import rf_restart as rf, rf_stages
from h3_audio_t8_pkg import sampling
from test_modular_rf_restart import model, source_latent, run


def corrected_args(source, original, effects="none", relay=False, mode=None, *, steps=3, sigma=.15):
    from comfy_extras.nodes_custom_sampler import BasicGuider, RandomNoise
    from test_modular_rf_stages import effects_model
    from test_progressive_sampling_runtime import conditioning
    from test_progressive_relay import paired
    from h3_audio_t8_pkg.modular_sampling import eav
    bare, positive = model(), conditioning()
    if relay:
        bare, positive, _ = paired(bare)
    bare = effects_model(bare, source, effects)
    prepared, sampler, sigmas, context, _ = rf_stages.build_restart_stage(bare,
        rf_stages.handoff(source, original), restart_steps=steps, restart_video_sigma=sigma,
        audio_start_policy="already_joint_renoised")
    runtime = None
    if mode:
        prepared, runtime, _ = eav.apply_stage_eav(prepared, sigmas, source, context,
            eav.EAVConfig(mode, tau=.2, start_video_progress=0., end_video_progress=1., g_hard_limit=3.))
    return (RandomNoise.execute(73).result[0], BasicGuider.execute(prepared, positive).result[0],
            sampler, sigmas, source, context), runtime


def test_new_node_is_explicit_and_reuses_legacy_ports_without_mutating_old_schema():
    from h3_audio_t8_pkg.modular_sampling.nodes import MiniMaxH3RFRestartStageSetupEXPT8 as Old
    from h3_audio_t8_pkg.modular_sampling.rf_audio_clock_nodes import MiniMaxH3RFRestartJointClockSetupEXPT8 as New
    old, new = Old.define_schema(), New.define_schema()
    assert old.node_id == 'MiniMaxH3RFRestartStageSetupEXPT8'
    assert new.node_id == 'MiniMaxH3RFRestartJointClockSetupEXPT8'
    assert [p.id for p in old.inputs] == [p.id for p in new.inputs]
    assert [p.id for p in old.outputs] == [p.id for p in new.outputs]
    source, bare = source_latent(), model()
    first = run(*sampling.setup_dual_clock_sampling(bare, source, 4, 12., 3.), source)[0]
    handoff = rf_stages.handoff(first, source)
    output = New.execute(bare, handoff).result
    assert json.loads(output[1].report_json)['audio_start_policy'] == 'already_joint_renoised'


@pytest.mark.parametrize('masked', [False, True])
@pytest.mark.parametrize('shifts', [(12., 3.), (3., 12.), (3., 3.)])
def test_explicit_joint_clock_initialization_reaches_first_forward_once(monkeypatch, masked, shifts):
    bare, original = model(), source_latent(masked)
    first_setup = sampling.setup_dual_clock_sampling(bare, original, 4, *shifts)
    completed = run(*first_setup, original)[0]
    handoff = rf.prepare_handoff(completed, original)
    before = torch.random.get_rng_state().clone()
    branch, restart, sigmas, raw = rf.build_restart_stage(bare, handoff,
        shift_video=shifts[0], shift_audio=shifts[1], audio_start_policy='already_joint_renoised')
    assert torch.equal(torch.random.get_rng_state(), before)
    rebase = sampling._rebase_partial_audio_start
    observed = []
    def observe(bound_model, x, sigma, **kwargs):
        actual = rebase(bound_model, x, sigma, **kwargs)
        observed.append((bound_model, x.clone(), actual.clone()))
        return actual
    monkeypatch.setattr(sampling, '_rebase_partial_audio_start', observe)
    actual = run(branch, restart, sigmas, completed)
    assert len(observed) == 1
    bound, start, after = observed[0]
    assert type(bound) is rf.JointClockInitializedModel
    assert torch.equal(start, after)  # ALREADY clock-initialized, not rebased.
    assert hasattr(bound.inpaint, 'noise') and hasattr(bound.inpaint, 'latent_image')
    original_packed, _ = rf.comfy.utils.pack_latents(original['samples'].unbind())
    expected_anchor = branch.model.process_latent_in(original_packed)
    assert torch.equal(bound.inpaint.latent_image.cpu(), expected_anchor)
    packed, _ = rf.comfy.utils.pack_latents(completed['samples'].unbind())
    endpoint = branch.model.process_latent_in(packed)
    mask = None
    if masked:
        mask, _ = rf.comfy.utils.pack_latents(completed['noise_mask'].unbind())
    report = json.loads(restart.report_json)
    expected = rf._renoise(endpoint, video_values=original['samples'].unbind()[0].numel(),
        video_sigma=.15, audio_sigma=report['restart_audio_sigma'], restart_seed=1234, mask=mask)
    assert torch.equal(start.cpu(), expected.cpu())
    assert json.loads(raw)['planned_stage_calls'] == 3
    assert all(torch.isfinite(x).all() for result in actual for x in result['samples'].unbind())
    assert not torch.cuda.is_initialized()


def test_corrected_policy_has_separate_authenticated_identity_and_legacy_report_unchanged():
    bare, source = model(), source_latent()
    first = run(*sampling.setup_dual_clock_sampling(bare, source, 4, 12., 3.), source)[0]
    handoff = rf_stages.handoff(first, source)
    old = rf_stages.build_restart_stage(bare, handoff)
    new = rf_stages.build_restart_stage(bare, handoff, audio_start_policy='already_joint_renoised')
    assert 'audio_start_policy' not in json.loads(old[1].report_json)
    assert json.loads(new[1].report_json)['audio_start_policy'] == 'already_joint_renoised'
    assert old[3] != new[3]
    assert rf_stages.sampler_is_known(old[0], old[1], old[3])
    assert rf_stages.sampler_is_known(new[0], new[1], new[3])
    with pytest.raises(ValueError, match='differ from context'):
        rf_stages.sampler_is_known(new[0], old[1], new[3])
    with pytest.raises(ValueError, match='Unknown RF'):
        rf.build_restart_stage(bare, handoff, audio_start_policy='guess')


@pytest.mark.parametrize("effects,relay,mode", [
    ("none", False, None), ("bias_stg", True, "report_only"), ("bias_stg", True, "apply_exp")])
def test_corrected_stage_keeps_effects_original_anchor_and_portable_saved_AV(tmp_path, effects, relay, mode):
    from test_modular_rf_stages import inputs
    from h3_audio_t8_pkg.modular_sampling.results import sample_stage
    from h3_audio_t8_pkg.modular_sampling.storage import save_stage, load_stage
    first = sample_stage(*inputs("rf_base", effects, relay, mode)[0])[2]
    source, original = first.output, first.output[rf_stages.ANCHOR]
    args, runtime = corrected_args(source, original, effects, relay, mode)
    result = sample_stage(*args)[2]
    receipt = result.verify()
    assert receipt["portable_identity"] is receipt["verified_recipe_completion"] is True
    assert receipt["execution"]["denoiser_evaluations"] == 3
    assert json.loads(args[-1].profile)["restart_report"]["audio_start_policy"] == "already_joint_renoised"
    if runtime:
        audit = runtime.snapshot()
        assert audit["status"] == "observed_" + mode
        assert audit["completed_forwards"] == 6
    assert sample_stage(*args)[2].verify()["request_sha256"] == receipt["request_sha256"]
    path, digest, _ = save_stage(result, tmp_path)
    loaded = load_stage(tmp_path, path, digest, "rf_restart")[0]
    for key in ("samples", "noise_mask"):
        assert all(torch.equal(a, b) for a, b in zip(loaded[key].unbind(), result.output[key].unbind()))
    assert rf_stages.handoff(loaded).template_identity == rf_stages.handoff(source).template_identity
    assert rf.retained_endpoint(loaded) is not None
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("disabled", ["sigma", "steps"])
def test_explicit_corrected_noop_keeps_both_streams_and_zero_model_calls(monkeypatch, disabled):
    from h3_audio_t8_pkg.modular_sampling.results import sample_stage
    source = source_latent(True)
    options = {"sigma": 0.} if disabled == "sigma" else {"steps": 0}
    args, runtime = corrected_args(source, source, "bias_stg", True, "apply_exp", **options)
    def forbidden(*args, **kwargs):
        raise AssertionError("Corrected RF no-op must not evaluate the model")
    monkeypatch.setattr(args[1].model_patcher.model.diffusion_model, "forward", forbidden)
    result = sample_stage(*args)[2]
    assert result.verify()["execution"]["denoiser_evaluations"] == 0
    assert runtime.snapshot()["completed_forwards"] == 0
    assert args[-1].steps == 0 and args[-1].trajectory_sigmas == ()
    assert all(torch.equal(a, b) for a, b in zip(result.output["samples"].unbind(), source["samples"].unbind()))


def test_fresh_process_corrected_restart_loads_saved_base_with_all_external_effects(tmp_path):
    from test_modular_rf_stages import inputs
    from h3_audio_t8_pkg.modular_sampling.results import sample_stage
    from h3_audio_t8_pkg.modular_sampling.storage import save_stage
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        first = sample_stage(*inputs("rf_base", "bias_stg", True, "apply_exp")[0])[2]
        path, digest, _ = save_stage(first, tmp_path)
        args, _ = corrected_args(first.output, first.output[rf_stages.ANCHOR], "bias_stg", True, "apply_exp")
        expected = sample_stage(*args)[2].verify()
        code = """
import json,runpy,sys
runpy.run_path('tools/check_director_d1_cpu.py',run_name='config')
from comfy.cli_args import args as comfy_args
comfy_args.use_pytorch_cross_attention = sys.argv[4] == 'attention_pytorch'
runpy.run_path('tests/conftest.py')
from test_modular_rf_audio_initialization import corrected_args,rf_stages
from h3_audio_t8_pkg.modular_sampling.results import sample_stage
from h3_audio_t8_pkg.modular_sampling.storage import load_stage
import torch
first=load_stage(sys.argv[1],sys.argv[2],sys.argv[3],'rf_base')[0]
def forbidden(*args,**kwargs):
    raise AssertionError('Corrected restart must not regenerate a saved BASE')
rf_stages.build_base_stage=forbidden
args,runtime=corrected_args(first,rf_stages.handoff(first).original_template,'bias_stg',True,'apply_exp')
receipt=sample_stage(*args)[2].verify()
assert receipt['execution']['denoiser_evaluations']==3 and receipt['portable_identity']
assert runtime.snapshot()['status']=='observed_apply_exp'
assert not torch.cuda.is_initialized()
print('RESULT='+json.dumps(receipt['outputs']))
print('REQUEST='+receipt['request_sha256'])
"""
        backend = expected["request"]["model"]["backend"]["name"]
        assert backend in {"attention_pytorch", "attention_sub_quad"}
        child = subprocess.run([sys.executable, "-c", code, str(tmp_path), path, digest, backend],
            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, encoding="utf8", timeout=45)
        assert child.returncode == 0, child.stderr
        actual = json.loads(next(line[7:] for line in child.stdout.splitlines() if line.startswith("RESULT=")))
        request = next(line[8:] for line in child.stdout.splitlines() if line.startswith("REQUEST="))
        assert actual == expected["outputs"] and request == expected["request_sha256"]
    finally:
        torch.set_num_threads(previous)
