"""Native motion references + independent EAV/Relay; tiny CPU, not media quality."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import comfy.samplers
import pytest
import torch

from h3_audio_t8_pkg import prompt_relay_advanced as relay
from h3_audio_t8_pkg.modular_sampling import continuation as split
from h3_audio_t8_pkg.modular_sampling import continuation_effects as effects
from h3_audio_t8_pkg.modular_sampling import progressive as stages
from h3_audio_t8_pkg.modular_sampling import progressive_effects as shared
from h3_audio_t8_pkg.modular_sampling import progressive_storage as storage
from h3_audio_t8_pkg.modular_sampling import progressive_high_result as high_results
from h3_audio_t8_pkg.modular_sampling.eav import EAVConfig
from helpers import FakeVideoVAE, FakeAudioVAE, make_audio
from test_prompt_relay_long_video_advanced import NativeLikeFakeClip
from test_modular_continuation import low_sample, high_restart
from test_progressive_continuation import accepted  # noqa: F401
from test_progressive_continuation_runtime import run as legacy_run
from test_progressive_sampling_runtime import tiny_model, stub_lifter  # noqa: F401


def inputs(case, frames=22, events="Walk.\nStop.\nTurn.", audio_mode="native"):
    clip = NativeLikeFakeClip()
    source = split.capture_source(case.root, **{**case.request, "context_frames": frames})
    contexts = split.prepare_contexts(source, FakeVideoVAE())
    global_plan = relay.build_prompt_relay_plan("Scene.", events, 345,
        "auto_equal", "", "paper_v1", .1, False, False)[0]
    projected = effects.project_relay(contexts, global_plan, 124)
    common = dict(clip=clip, video_vae=FakeVideoVAE(), audio_vae=FakeAudioVAE(), length=124,
                  prompt=projected.projected["compiled_prompt"], audio_mode=audio_mode,
                  drive_audio=None if audio_mode == "native" else make_audio(6), add_source_as_reference=False)
    low = split.prepare_phase(contexts, "low", **common)
    high = split.prepare_phase(contexts, "high", **common)
    from h3_audio_t8_pkg.sampling import native_flow_sigmas
    plan = split.build_plan(contexts, 124, native_flow_sigmas(8, 12.))
    return source, contexts, low, high, plan, clip, global_plan, projected


def select(model, phase, plan, clip, projected, kind, restart=None, mode="apply_exp"):
    pos = neg = phase.result[0]
    if kind != "eav":
        model, pos, neg = effects.apply_relay(model, phase, plan, clip, projected, query_chunk_rows=64)
    if kind != "relay":
        model = effects.apply_eav(model, EAVConfig(mode, .2, 0., 1.), phase, plan, restart=restart)
    return model, pos, neg


@pytest.mark.parametrize("frames", [22, 39])
@pytest.mark.parametrize("kind", ["eav", "relay", "combined"])
def test_separate_effects_equal_old_motion_composer_and_are_saveable(accepted, stub_lifter, tmp_path, frames, kind):  # noqa: F811
    source, _, low, high, plan, clip, global_plan, projected = inputs(accepted, frames)
    model, sampler = tiny_model(), comfy.samplers.ksampler("euler")
    reference = dict(model_hires=tiny_model(), clip=clip,
        options={"add_source_as_reference": False}, eav_tau=.2)
    if kind != "relay":
        reference["eav_mode"] = "apply_exp"
    if kind != "eav":
        reference.update(prompt=None, prompt_relay_plan=global_plan, query_chunk_rows=64)
    else:
        reference["prompt"] = projected.projected["compiled_prompt"]
    expected, _ = legacy_run(source, model, **reference)
    selected, pos, neg = select(model, low, plan, clip, projected, kind)
    before = stages.native_model_identity(selected, sampler)
    assert stages.portable(before)
    boundary = low_sample(low, plan, selected, positive=pos, negative=neg)
    assert stages.native_model_identity(selected, sampler) == before
    path, digest, _ = storage.save_boundary(boundary, tmp_path)
    restored, _ = storage.load_boundary(tmp_path, path, digest)
    state = high_restart(high, restored, model, sampler)
    selected, pos, neg = select(model, high, plan, clip, projected, kind, state)
    result, _ = split.sample_high(high, state, selected, sampler, positive=pos, negative=neg, seed=9)
    for actual, wanted in zip(result.output["samples"].unbind(), expected["samples"].unbind()):
        torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
    for phase, complete in (("low", boundary), ("high", result)):
        report = shared.audit(complete, phase)
        assert report["status"] == "verified_stage_execution_quality_unverified"
        assert report["portable_effect_identity"] and complete.verify()["portable_identity"]
        if kind != "relay":
            assert report["eav"]["verified_native_mask_forwards"] == 4
            assert report["eav"]["config"]["task_scope"] == ["LongVideoMotion"]
            assert report["eav"]["config"]["progressive_mask_contract"]["accepted_source_sha256"] == source.sha256
        if kind != "eav":
            assert report["relay"]["completed_calls"] == {"forward": 4, "routed_attention": 4}
            assert report["relay"]["projection"]["render_start_frame"] == 124 - frames
    path, digest, _ = high_results.save_result(result, tmp_path)
    loaded, _ = high_results.load_result(tmp_path, path, digest)
    assert shared.audit(loaded, "high") == shared.audit(result, "high")
    assert not model.object_patches and not model.wrappers


@pytest.mark.parametrize("phase", ["low", "high"])
@pytest.mark.parametrize("kind", ["eav", "combined"])
def test_one_stage_eav_report_only_exact_and_disabled_identity(accepted, stub_lifter, phase, kind):  # noqa: F811
    _, _, low, high, plan, clip, _, projected = inputs(accepted)
    model, sampler = tiny_model(), comfy.samplers.ksampler("euler")
    boundary = low_sample(low, plan, model)
    state = high_restart(high, boundary, model, sampler)
    prepared = low if phase == "low" else high
    plain, pos, neg = (model, prepared.result[0], prepared.result[0])
    if kind == "combined":
        plain, pos, neg = effects.apply_relay(model, prepared, plan, clip, projected, query_chunk_rows=64)
    assert effects.apply_eav(plain, EAVConfig("disabled"), prepared, plan, restart=state) is plain
    diagnostic = effects.apply_eav(plain, EAVConfig("report_only", .2, 0., 1.), prepared, plan,
                                   restart=state if phase == "high" else None)
    def sample(selected):
        if phase == "low":
            return low_sample(low, plan, selected, positive=pos, negative=neg)
        return split.sample_high(high, state, selected, sampler, positive=pos, negative=neg, seed=9)[0]
    baseline, actual = sample(plain), sample(diagnostic)
    def tensors(item):
        return item.tensors.values() if phase == "low" else item.output["samples"].unbind()
    for a, b in zip(tensors(baseline), tensors(actual)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert shared.audit(actual, phase)["eav"]["verified_native_mask_forwards"] == 4


def test_independent_high_relay_plan_does_not_change_low_or_global_source(accepted, stub_lifter, monkeypatch):  # noqa: F811
    _, contexts, low, _, plan, clip, global_plan, projected = inputs(accepted)
    original = copy.deepcopy(global_plan)
    model, sampler = tiny_model(), comfy.samplers.ksampler("euler")
    selected, pos, neg = select(model, low, plan, clip, projected, "combined")
    boundary = low_sample(low, plan, selected, positive=pos, negative=neg)
    frozen = boundary.verify()
    other = relay.build_prompt_relay_plan("Other scene.", "Run.\nJump.", 345,
        "auto_equal", "", "paper_v1", .1, False, False)[0]
    high_plan = effects.project_relay(contexts, other, 124)
    high = split.prepare_phase(contexts, "high", clip=clip, video_vae=FakeVideoVAE(),
        audio_vae=FakeAudioVAE(), prompt=high_plan.projected["compiled_prompt"], length=124)
    monkeypatch.setattr(stages, "sample_low", lambda *a, **k: pytest.fail("LOW reran"))
    state = high_restart(high, boundary, model, sampler)
    selected, pos, neg = select(model, high, plan, clip, high_plan, "combined", state)
    result, _ = split.sample_high(high, state, selected, sampler, positive=pos, negative=neg, seed=9)
    assert shared.audit(result, "high")["relay"]["events"] != shared.audit(boundary, "low")["relay"]["events"]
    assert boundary.verify() == frozen and global_plan == original


def test_mismatched_phase_plan_and_changed_projection_rejected(accepted):  # noqa: F811
    _, _, low, high, plan, clip, _, projected = inputs(accepted)
    model, _, _ = effects.apply_relay(tiny_model(), low, plan, clip, projected)
    with pytest.raises(ValueError, match="same independent phase"):
        effects.apply_eav(model, EAVConfig("apply_exp"), high, plan)
    projected.projected["compiled_prompt"] += " unencoded"
    with pytest.raises(ValueError, match="projection changed"):
        effects.apply_relay(tiny_model(), low, plan, clip, projected)


@pytest.mark.parametrize("kind", ["eav", "relay", "combined"])
def test_cancel_retry_and_unknown_user_delegate_keep_actual_phase_execution(accepted, kind):  # noqa: F811
    _, _, low, _, plan, clip, _, projected = inputs(accepted)
    base = tiny_model()
    selected, pos, neg = select(base, low, plan, clip, projected, kind)
    host = shared.owner(selected)
    def cancel(*args):
        raise InterruptedError("continuation effect cancellation")
    with pytest.raises(InterruptedError):
        low_sample(low, plan, selected, positive=pos, negative=neg, callback=cancel)
    assert host.last_report["status"] == "aborted" and not host.lock.locked()
    baseline = low_sample(low, plan, selected, positive=pos, negative=neg)
    again = low_sample(low, plan, selected, positive=pos, negative=neg)
    for a, b in zip(baseline.tensors.values(), again.tensors.values()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert shared.audit(again, "low")["assigned_nfe"] == 4
    calls = []
    def user(executor, *args, **kwargs):
        calls.append(True)
        return executor(*args, **kwargs)
    selected.add_wrapper_with_key("apply_model", "user", user)
    result = low_sample(low, plan, selected, positive=pos, negative=neg)
    assert len(calls) == 4 and not result.verify()["portable_identity"]
    assert selected.get_wrappers("apply_model", "user") == [user]


def test_joint_audio_route_rejects_actual_locked_source_not_unknown_stack(accepted):  # noqa: F811
    _, contexts, _, _, plan, clip, global_plan, _ = inputs(accepted)
    global_plan["query_route"] = "joint_av_exp"
    global_plan.pop("plan_hash")
    global_plan["plan_hash"] = relay._sha256_json(global_plan)
    projected = effects.project_relay(contexts, global_plan, 124)
    low = split.prepare_phase(contexts, "low", clip=clip, video_vae=FakeVideoVAE(), audio_vae=FakeAudioVAE(),
        prompt=projected.projected["compiled_prompt"], length=124, audio_mode="lock_source", drive_audio=make_audio(6))
    with pytest.raises(ValueError, match="locked source audio"):
        effects.apply_relay(tiny_model(), low, plan, clip, projected)


@pytest.mark.parametrize("kind", ["eav", "combined"])
@pytest.mark.parametrize("change_high", [False, True])
@pytest.mark.parametrize("cpu_threads", [1, 2])
def test_new_process_effect_low_to_independently_changed_high_only(
        accepted, tmp_path, stub_lifter, kind, change_high, cpu_threads):  # noqa: F811
    torch.set_num_threads(cpu_threads)
    source, contexts, low, high, plan, clip, _, projected = inputs(accepted)
    model, sampler = tiny_model(), comfy.samplers.ksampler("euler")
    selected, pos, neg = select(model, low, plan, clip, projected, kind)
    boundary = low_sample(low, plan, selected, positive=pos, negative=neg)
    path, digest, _ = storage.save_boundary(boundary, tmp_path)
    if change_high:
        other = relay.build_prompt_relay_plan("Other scene.", "Run.\nJump.", 345,
            "auto_equal", "", "paper_v1", .1, False, False)[0]
        projected = effects.project_relay(contexts, other, 124)
        high = split.prepare_phase(contexts, "high", clip=clip, video_vae=FakeVideoVAE(), audio_vae=FakeAudioVAE(),
            prompt=projected.projected["compiled_prompt"], length=124)
        key, weight = next(iter(dict(model.model.named_parameters()).items()))
        model.add_patches({key: ("diff", (torch.full_like(weight, .01),))})
    state = high_restart(high, boundary, model, sampler)
    selected, pos, neg = select(model, high, plan, clip, projected, kind, state)
    expected, _ = split.sample_high(high, state, selected, sampler, positive=pos, negative=neg, seed=9)
    code = """
import json, runpy, sys
runpy.run_path('tools/check_director_d1_cpu.py', run_name='config')
runpy.run_path('tests/conftest.py')
import comfy.samplers, pytest, torch
torch.set_num_threads(int(sys.argv[8]))
from helpers import FakeVideoVAE, FakeAudioVAE
from test_prompt_relay_long_video_advanced import NativeLikeFakeClip
from test_progressive_sampling_runtime import tiny_model, stub_lifter
from test_modular_continuation import high_restart
from test_modular_continuation_effects import select
from h3_audio_t8_pkg import prompt_relay_advanced as relay
from h3_audio_t8_pkg.modular_sampling import continuation as split, continuation_effects as effects, progressive as stages
from h3_audio_t8_pkg.modular_sampling import progressive_storage as storage, progressive_high_result as high_results
low, _ = storage.load_boundary(sys.argv[1], sys.argv[2], sys.argv[3])
source = split.capture_source(sys.argv[4], **json.loads(sys.argv[5]))
contexts = split.prepare_contexts(source, FakeVideoVAE())
changed = sys.argv[7] == 'True'
global_plan = relay.build_prompt_relay_plan('Other scene.' if changed else 'Scene.',
    'Run.\\nJump.' if changed else 'Walk.\\nStop.\\nTurn.', 345, 'auto_equal', '', 'paper_v1', .1, False, False)[0]
projected = effects.project_relay(contexts, global_plan, 124)
clip = NativeLikeFakeClip()
high = split.prepare_phase(contexts, 'high', clip=clip, video_vae=FakeVideoVAE(), audio_vae=FakeAudioVAE(),
    prompt=projected.projected['compiled_prompt'], length=124, add_source_as_reference=False)
plan = stages.plan_from_dict(low.verify()['request']['plan'])
model, sampler = tiny_model(), comfy.samplers.ksampler('euler')
if changed:
    key, weight = next(iter(dict(model.model.named_parameters()).items()))
    model.add_patches({key: ('diff', (torch.full_like(weight, .01),))})
def forbidden(*args, **kwargs):
    raise AssertionError('LOW/whole sampling or encoding during completed HIGH delivery')
with pytest.MonkeyPatch.context() as patch:
    stub_lifter.__wrapped__(patch)
    patch.setattr(stages, 'sample_low', forbidden)
    patch.setattr(stages.legacy, 'sample_progressive_h3', forbidden)
    state = high_restart(high, low, model, sampler)
    selected, pos, neg = select(model, high, plan, clip, projected, sys.argv[6], state)
    result, report = split.sample_high(high, state, selected, sampler, positive=pos, negative=neg, seed=9)
assert json.loads(report)['execution']['actual_apply_calls'] == 4 and result.verify()['portable_identity']
path, digest, _ = high_results.save_result(result, sys.argv[1])
stages.sample_high = split.sample_high = split.prepare_contexts = split.prepare_phase = forbidden
loaded, _ = high_results.load_result(sys.argv[1], path, digest)
output, _, _, _, delivery = split.deliver(loaded, source)
assert json.loads(delivery)['sampling_calls'] == 0 and not torch.cuda.is_initialized()
print('RESULT=' + json.dumps(stages.snapshot(output)))
"""
    child = subprocess.run([sys.executable, "-c", code, str(tmp_path), path, digest,
        str(accepted.root), json.dumps(source.binding["request"]), kind, str(change_high),
        str(torch.get_num_threads())],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=90)
    assert child.returncode == 0, child.stdout + child.stderr
    actual = json.loads(next(line[7:] for line in child.stdout.splitlines() if line.startswith("RESULT=")))
    assert actual == stages.snapshot(expected.output)
