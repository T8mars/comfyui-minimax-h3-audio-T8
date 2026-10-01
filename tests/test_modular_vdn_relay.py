"""True Core Composer/window/linear Relay, not an attention-selector test double."""
import json
from copy import deepcopy
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from comfy_extras.nodes_custom_sampler import BasicGuider, RandomNoise

from h3_audio_t8_pkg import prompt_relay_advanced as relay
from h3_audio_t8_pkg.modular_sampling import vdn_stages, vdn_relay, eav, effect_identity
from h3_audio_t8_pkg.modular_sampling.results import sample_stage
from h3_audio_t8_pkg.modular_sampling.storage import save_stage, load_stage
from test_modular_vdn_baseline import tiny_vdn, source
from test_progressive_sampling_runtime import conditioning


def inputs(stage="vdn_complete", frames=2, with_eav=False, neutral=False, enabled=True, latent=None,
           training="stage_dmd_8nfe", query_route="joint_av_exp"):
    model, _ = tiny_vdn(training)
    latent = source(frames) if latent is None else latent
    positive = conditioning()
    positive[0][0][0, 0] = torch.linspace(-1, 1, 8)
    positive[0][0][0, 1] = torch.linspace(1, -1, 8)
    layout = relay.build_packed_layout(2, frames, 4, 4, 8, keyframes=[], refs=[], frame_count=(frames - 1)*4 + 1)
    binding = relay._bind_layout_contract({"schema": relay.PROMPT_RELAY_PATCH_VERSION,
        "plan_hash": "tiny-vdn-events-neutral" if neutral else "tiny-vdn-events", "text_len": 2,
        "query_route": query_route, "events": [dict(text_key_start=i, text_key_end=i+1,
            midpoint=float(i*4), window=1000. if neutral else .1, sigma=.2) for i in range(2)]},
        layout, resolved_task="t2va", keyframes=[], refs=[])
    positive = [[value, {**meta, relay.PROMPT_RELAY_BINDING_KEY: binding}] for value, meta in positive]
    positive = relay._attach_binding_model_cond(positive, binding["binding_hash"])
    model, _ = relay.patch_prompt_relay_model(model, binding, 32)
    model, sampler, sigmas, context, _ = vdn_stages.build_stage(model, latent, stage, 4, latent)
    before = model
    applied, runtime, _ = vdn_relay.apply(model, sigmas, latent, context,
        vdn_relay.Config("apply_exp" if enabled else "disabled", 64))
    eav_runtime = None
    if with_eav:
        applied, eav_runtime, _ = eav.apply_stage_eav(applied, sigmas, latent, context,
            eav.EAVConfig("apply_exp", tau=.5, start_video_progress=0., end_video_progress=1., g_hard_limit=3.))
    args = (RandomNoise.execute(43).result[0], BasicGuider.execute(applied, positive).result[0],
            sampler, sigmas, latent, context)
    return args, runtime, eav_runtime, before


@pytest.mark.parametrize("stage", vdn_stages.STAGES)
@pytest.mark.parametrize("frames", [2, 16])
@pytest.mark.parametrize("with_eav", [False, True])
@pytest.mark.parametrize("neutral", [False, True])
def test_actual_vdn_relay_short_full_cover_long_linear_identity_and_frozen(tmp_path, stage, frames, with_eav, neutral):
    args, runtime, eav_runtime, before = inputs(stage, frames, with_eav, neutral)
    effect_identity.project_stage_effects(args[1].model_patcher)  # expose exact adapter failures
    result = sample_stage(*args)[2]
    receipt = result.verify()
    assert receipt["portable_identity"] is receipt["verified_recipe_completion"] is True
    assert receipt["execution"]["denoiser_evaluations"] == args[-1].steps
    report = runtime.snapshot()
    assert report["status"] == ("observed_neutral" if neutral else "observed_apply_exp")
    assert report["completed_blocks"] == args[-1].steps
    if not neutral:
        assert report["stats"]["window_calls"] >= args[-1].steps
        assert report["stats"].get("linear_frames", 0) == (14 * args[-1].steps if frames == 16 else 0)
    if eav_runtime is not None:
        assert eav_runtime.snapshot()["status"] == "observed_apply_exp"
        assert eav_runtime.snapshot()["relay_attention_calls"] == args[-1].steps
    plain = sample_stage(*inputs(stage, frames, with_eav, neutral, False)[0])[2]
    equal = all(torch.equal(a, b) for a, b in zip(result.output["samples"].unbind(), plain.output["samples"].unbind()))
    assert equal is neutral
    assert sample_stage(*args)[2].verify()["request_sha256"] == receipt["request_sha256"]
    assert sample_stage(*inputs(stage, frames, with_eav, neutral)[0])[2].verify()["request_sha256"] == receipt["request_sha256"]
    assert before.get_attachment(vdn_relay.KEY) is None
    path, digest, _ = save_stage(result, tmp_path)
    assert load_stage(tmp_path, path, digest, stage)[3].verify()["request_sha256"] == receipt["request_sha256"]
    assert json.loads(vdn_relay.audit(result.output, runtime)[1])["status"] == report["status"]
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("training", ["stage_dmd_8nfe", "stage_b_50nfe"])
@pytest.mark.parametrize("frames", [2, 16])
@pytest.mark.parametrize("cpu_threads", [1, 2])
def test_new_process_only_high_with_actual_vdn_relay_and_eav(tmp_path, training, frames, cpu_threads):
    torch.set_num_threads(cpu_threads)
    from h3_audio_t8_pkg import learned_latent_upscale_advanced as learned
    args, runtime, _, _ = inputs(frames=frames, with_eav=True, training=training)
    low = sample_stage(*args)[2]
    assert runtime.snapshot()["status"] == "observed_apply_exp"
    path, digest, _ = save_stage(low, tmp_path)
    high, _, _ = learned.reconcile_two_pass_h3_latent(low.output, source(frames), conditioning(), "first_pass",
        second_pass_audio_source="first_pass", second_pass_audio_strength=0.)
    args, runtime, _, _ = inputs("vdn_refine", frames, True, latent=high, training=training)
    result = sample_stage(*args)[2]
    expected = result.verify()["outputs"]
    audited, _ = learned.audit_two_pass_h3_audio(high, result.output, 0., True, 1e-5)
    assert torch.equal(audited["samples"].unbind()[1], low.output["samples"].unbind()[1])
    code = """
import json,runpy,sys
runpy.run_path('tools/check_director_d1_cpu.py',run_name='config')
runpy.run_path('tests/conftest.py')
from test_modular_vdn_relay import inputs,source,conditioning
from h3_audio_t8_pkg.modular_sampling.storage import load_stage
from h3_audio_t8_pkg.modular_sampling.results import sample_stage
from h3_audio_t8_pkg import learned_latent_upscale_advanced as learned
import torch
torch.set_num_threads(int(sys.argv[6]))
first=load_stage(sys.argv[1],sys.argv[2],sys.argv[3],'vdn_complete')[0]
frames=int(sys.argv[5])
high,_,_=learned.reconcile_two_pass_h3_latent(first,source(frames),conditioning(),'first_pass',
    second_pass_audio_source='first_pass',second_pass_audio_strength=0.)
args,runtime,eav,_=inputs('vdn_refine',frames,True,latent=high,training=sys.argv[4])
result=sample_stage(*args)[2]
receipt=result.verify()
assert receipt['portable_identity'] and receipt['execution']['denoiser_evaluations']==4
assert runtime.snapshot()['status']==eav.snapshot()['status']=='observed_apply_exp'
audited,_=learned.audit_two_pass_h3_audio(high,result.output,0.,True,1e-5)
assert torch.equal(first['samples'].unbind()[1],audited['samples'].unbind()[1])
assert not torch.cuda.is_initialized()
print('RESULT='+json.dumps(receipt['outputs']))
"""
    child = subprocess.run([sys.executable, "-c", code, str(tmp_path), path, digest, training, str(frames),
                            str(torch.get_num_threads())],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=60)
    assert child.returncode == 0, child.stderr
    actual = json.loads(next(line[7:] for line in child.stdout.splitlines() if line.startswith("RESULT=")))
    assert actual == expected


def test_disabled_adapter_original_object_and_live_projection_keeps_callbacks_and_hooks():
    args, runtime, _, before = inputs(enabled=False)
    assert args[1].model_patcher is before and runtime.snapshot()["status"] == "disabled_identity"
    args, runtime, _, _ = inputs(with_eav=True)
    model = args[1].model_patcher
    hooks = dict(model.model_options["transformer_options"]["patches_replace"]["dit"])
    callbacks = {key: list(value[vdn_relay.KEY]) for key, value in model.callbacks.items() if vdn_relay.KEY in value}
    effect_identity.project_stage_effects(model)
    assert model.get_attachment(vdn_relay.KEY) is runtime
    assert model.model_options["transformer_options"]["patches_replace"]["dit"] == hooks
    assert all(model.callbacks[key][vdn_relay.KEY] == value for key, value in callbacks.items())


def test_changed_binding_raises_actual_error_records_abort_then_retry_works():
    args, runtime, _, _ = inputs()
    original = runtime.binding_hash
    runtime.binding_hash = "changed"
    with pytest.raises(ValueError, match="binding hashes differ"):
        sample_stage(*args)
    assert runtime.snapshot()["status"] == "aborted"
    runtime.binding_hash = original
    assert sample_stage(*args)[2].verify()["portable_identity"]


@pytest.mark.parametrize("query_route", ["joint_av_exp", "video_only_paper"])
def test_b50_tail_actual_relay_query_modes(query_route):
    args, runtime, _, _ = inputs("vdn_refine", 16, training="stage_b_50nfe", query_route=query_route)
    result = sample_stage(*args)[2]
    assert result.verify()["portable_identity"] and runtime.snapshot()["status"] == "observed_apply_exp"
    assert runtime.snapshot()["stats"]["linear_frames"] == 56


def test_foreign_dit_owner_preserved_and_not_false_relay_coverage():
    args, _, _, before = inputs()
    original = before.model_options["transformer_options"]["patches_replace"]["dit"][("double_block", 0)]
    calls = []
    def foreign(block_args, extra):
        calls.append(1)
        return original(block_args, extra)
    before.set_model_patch_replace(foreign, "dit", "double_block", 0)
    # Rebind the new execution selection; do not bypass old immutable receipts.
    before, sampler, sigmas, context, _ = vdn_stages.build_stage(before, args[4])
    model, runtime, _ = vdn_relay.apply(before, sigmas, args[4], context, vdn_relay.Config())
    guider = type(args[1])(model)
    guider.original_conds = deepcopy(args[1].original_conds)  # already converted by Core
    fresh = (args[0], guider, sampler, sigmas, args[4], context)
    result = sample_stage(*fresh)[2]
    assert calls and result.verify()["portable_identity"] is False
    assert model.model_options["transformer_options"]["patches_replace"]["dit"][("double_block", 0)] is foreign
    assert runtime.snapshot()["status"] == "unverified_incomplete_coverage"


def test_actual_linear_workspace_error_is_not_silent_fallback_and_recovery_works():
    args, runtime, _, _ = inputs(frames=16)
    runtime.config = vdn_relay.Config("apply_exp", 4)
    with pytest.raises(ValueError, match="one linear head"):
        sample_stage(*args)
    assert runtime.snapshot()["status"] == "aborted"
    assert runtime.snapshot()["stats"]["window_calls"] > 0
    runtime.config = vdn_relay.Config("apply_exp", 64)
    result = sample_stage(*args)[2]
    assert result.verify()["portable_identity"] and runtime.snapshot()["status"] == "observed_apply_exp"
