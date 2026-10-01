"""Authentic motion owner and frozen stages; no pretend portable user patches."""
import json
from pathlib import Path
import subprocess
import sys
from types import MethodType

import comfy.cli_args
import comfy.samplers
import pytest
import torch

from h3_audio_t8_pkg import long_video
from h3_audio_t8_pkg.modular_sampling import continuation as split
from h3_audio_t8_pkg.modular_sampling import continuation_identity as identity
from h3_audio_t8_pkg.modular_sampling import progressive as stages
from h3_audio_t8_pkg.modular_sampling import progressive_storage as storage
from h3_audio_t8_pkg.modular_sampling import progressive_high_result as high_results
from test_modular_continuation import prepared, low_sample, high_restart
from test_progressive_continuation import accepted  # noqa: F401
from test_progressive_sampling_runtime import tiny_model, stub_lifter  # noqa: F401


def test_actual_motion_owner_projection_is_stable_and_read_only(accepted):  # noqa: F811
    _, _, low, _, plan = prepared(accepted)
    base = tiny_model()
    model = long_video.patch_long_video_model(base)
    sampler = comfy.samplers.ksampler("euler")
    patch = model.object_patches["extra_conds"]
    clone, contract = identity.project(model)
    assert contract and clone is not model and not clone.object_patches
    before = stages.native_model_identity(model, sampler)
    assert stages.portable(before)
    result = low_sample(low, plan, model)
    assert result.verify()["portable_identity"]
    assert stages.native_model_identity(model, sampler) == before
    assert model.object_patches["extra_conds"] is patch and not base.object_patches
    rebuilt = long_video.patch_long_video_model(tiny_model())
    assert stages.native_model_identity(rebuilt, sampler) == before


@pytest.mark.parametrize("fault", ["label_only", "delegate", "attribute", "bound_host", "defaults"])
def test_fake_or_changed_owner_never_granted_portable_identity(fault):
    model = long_video.patch_long_video_model(tiny_model())
    patch = model.object_patches["extra_conds"]
    function = patch.__func__
    if fault == "label_only":
        def counterfeit(self, **kwargs):
            return function(self, **kwargs)
        counterfeit.__dict__.update(function.__dict__)
        model.add_object_patch("extra_conds", MethodType(counterfeit, model.model))
    elif fault == "delegate":
        function._t8_long_video_original_extra_conds = lambda **kwargs: {}
    elif fault == "attribute":
        function.extra_owner = object()
    elif fault == "bound_host":
        model.add_object_patch("extra_conds", MethodType(function, tiny_model().model))
    else:
        function.__kwdefaults__ = {"unknown": 1}
    assert not stages.portable(stages.native_model_identity(model, comfy.samplers.ksampler("euler")))
    assert "extra_conds" in model.object_patches


def test_unknown_extra_conds_delegate_still_executes_without_portable_claim(accepted):  # noqa: F811
    _, _, low, _, plan = prepared(accepted)
    model, calls = tiny_model(), []
    original = model.get_model_object("extra_conds")
    def custom(_self, **kwargs):
        calls.append(True)
        return original(**kwargs)
    method = MethodType(custom, model.model)
    model.add_object_patch("extra_conds", method)
    result = low_sample(low, plan, model)
    assert calls and result.verify()["execution"]["actual_apply_calls"] == 4
    assert not result.verify()["portable_identity"]
    assert model.object_patches["extra_conds"] is method


def test_native_lora_difference_and_payload_function_change_invalidate(monkeypatch):
    model = long_video.patch_long_video_model(tiny_model())
    sampler = comfy.samplers.ksampler("euler")
    before = stages.native_model_identity(model, sampler)
    assert stages.portable(before)
    key, weight = next(iter(dict(model.model.named_parameters()).items()))
    model.add_patches({key: ("diff", (torch.full_like(weight, .01),))})
    after = stages.native_model_identity(model, sampler)
    assert stages.portable(after) and not stages.model_identity_matches(before, after)
    original = long_video.repair_long_video_payload
    monkeypatch.setattr(long_video, "repair_long_video_payload", lambda *a, **k: original(*a, **k))
    changed = stages.native_model_identity(model, sampler)
    assert not stages.portable(changed) and not stages.model_identity_matches(after, changed)


@pytest.mark.parametrize("change_high", [False, True])
def test_new_process_frozen_low_only_runs_high_and_completed_high_loads_without_source_encoding(
        accepted, tmp_path, stub_lifter, change_high):  # noqa: F811
    _, _, low, high, plan = prepared(accepted)
    model, sampler = tiny_model(), comfy.samplers.ksampler("euler")
    low_result = low_sample(low, plan, model)
    assert low_result.verify()["portable_identity"]
    path, digest, _ = storage.save_boundary(low_result, tmp_path)
    if change_high:
        key, weight = next(iter(dict(model.model.named_parameters()).items()))
        model.add_patches({key: ("diff", (torch.full_like(weight, .01),))})
    state = high_restart(high, low_result, model, sampler)
    expected, _ = split.sample_high(high, state, model, sampler, seed=9)
    code = """
import json, runpy, sys
import comfy.cli_args
comfy.cli_args.args.use_pytorch_cross_attention = sys.argv[7] == '1'
runpy.run_path('tools/check_director_d1_cpu.py', run_name='config')
runpy.run_path('tests/conftest.py')
import comfy.samplers, pytest, torch
torch.set_num_threads(int(sys.argv[8]))
from helpers import FakeClip, FakeVideoVAE, FakeAudioVAE
from test_progressive_sampling_runtime import tiny_model, stub_lifter
from test_modular_continuation import high_restart
from h3_audio_t8_pkg.modular_sampling import continuation as split, progressive as stages
from h3_audio_t8_pkg.modular_sampling import progressive_storage as storage, progressive_high_result as high_results
low, _ = storage.load_boundary(sys.argv[1], sys.argv[2], sys.argv[3])
source = split.capture_source(sys.argv[4], **json.loads(sys.argv[5]))
contexts = split.prepare_contexts(source, FakeVideoVAE())
high = split.prepare_phase(contexts, 'high', clip=FakeClip(), video_vae=FakeVideoVAE(),
    audio_vae=FakeAudioVAE(), prompt='Continue walking.', length=124)
model, sampler = tiny_model(), comfy.samplers.ksampler('euler')
if sys.argv[6] == 'True':
    key, weight = next(iter(dict(model.model.named_parameters()).items()))
    model.add_patches({key: ('diff', (torch.full_like(weight, .01),))})
def forbidden(*args, **kwargs):
    raise AssertionError('LOW/whole sampling or encoding during completed HIGH load')
with pytest.MonkeyPatch.context() as patch:
    stub_lifter.__wrapped__(patch)
    patch.setattr(stages, 'sample_low', forbidden)
    patch.setattr(stages.legacy, 'sample_progressive_h3', forbidden)
    result, report = split.sample_high(high, high_restart(high, low, model, sampler), model, sampler, seed=9)
assert json.loads(report)['execution']['actual_apply_calls'] == 4
assert result.verify()['portable_identity']
path, digest, _ = high_results.save_result(result, sys.argv[1])
stages.sample_high = split.sample_high = split.prepare_contexts = split.prepare_phase = forbidden
loaded, report = high_results.load_result(sys.argv[1], path, digest)
assert json.loads(report)['sampling_calls'] == 0 and not torch.cuda.is_initialized()
print('RESULT=' + json.dumps(stages.snapshot(loaded.output)))
"""
    attention_backend = str(int(bool(comfy.cli_args.args.use_pytorch_cross_attention)))
    intraop_threads = str(torch.get_num_threads())
    child = subprocess.run([sys.executable, "-c", code, str(tmp_path), path, digest,
        str(accepted.root), json.dumps(accepted.request), str(change_high), attention_backend,
        intraop_threads],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=90)
    assert child.returncode == 0, child.stdout + child.stderr
    actual = json.loads(next(line[7:] for line in child.stdout.splitlines() if line.startswith("RESULT=")))
    assert actual == stages.snapshot(expected.output)


def test_saved_low_cannot_resume_after_accepted_parent_changes(accepted, tmp_path, stub_lifter):  # noqa: F811
    _, _, low, high, plan = prepared(accepted)
    model, sampler = tiny_model(), comfy.samplers.ksampler("euler")
    boundary = low_sample(low, plan, model)
    path, digest, _ = storage.save_boundary(boundary, tmp_path)
    restored, _ = storage.load_boundary(tmp_path, path, digest)
    accepted.manifest["revision"] = 2
    accepted.manifest_path.write_text(json.dumps(accepted.manifest))
    with pytest.raises(ValueError, match="revision"):
        high_restart(high, restored, model, sampler)


@pytest.mark.parametrize("explicit_audio", [False, True])
def test_completed_high_delivery_is_source_checked_without_reencoding_or_auto_accept(
        accepted, tmp_path, stub_lifter, monkeypatch, explicit_audio):  # noqa: F811
    from helpers import make_audio
    audio = make_audio(6) if explicit_audio else None
    source, _, low, high, plan = prepared(accepted, options={"final_audio": audio})
    model, sampler = tiny_model(), comfy.samplers.ksampler("euler")
    boundary = low_sample(low, plan, model)
    result, _ = split.sample_high(high, high_restart(high, boundary, model, sampler), model, sampler, seed=9)
    files = {str(path.relative_to(accepted.root)): path.read_bytes() for path in accepted.root.rglob("*") if path.is_file()}
    output, pcm, prompt, _, report = split.deliver(result, source)
    assert output is result.output and pcm is audio and prompt == high.result[3]
    assert json.loads(report)["audio_policy"] == ("explicit_selected_pcm" if explicit_audio else "decode_generated_audio")
    path, digest, _ = high_results.save_result(result, tmp_path)
    def forbidden(*args, **kwargs):
        pytest.fail("completed delivery must not encode or sample")
    monkeypatch.setattr(split, "prepare_contexts", forbidden)
    monkeypatch.setattr(split, "prepare_phase", forbidden)
    monkeypatch.setattr(stages, "sample_low", forbidden)
    monkeypatch.setattr(stages, "sample_high", forbidden)
    loaded, _ = high_results.load_result(tmp_path, path, digest)
    _, restored_pcm, _, _, _ = split.deliver(loaded, source)
    assert stages.snapshot(restored_pcm) == stages.snapshot(audio)
    assert files == {str(path.relative_to(accepted.root)): path.read_bytes() for path in accepted.root.rglob("*") if path.is_file()}
    accepted.manifest["revision"] = 2
    accepted.manifest_path.write_text(json.dumps(accepted.manifest))
    with pytest.raises(ValueError, match="revision"):
        split.deliver(loaded, source)
