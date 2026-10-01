"""Actual owner projection and persisted native effect stages; CPU tiny models."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from comfy.ldm.modules import attention

from h3_audio_t8_pkg.h3_core_compat import set_h3_attention_backend
from h3_audio_t8_pkg.modular_sampling import progressive as stages
from h3_audio_t8_pkg.modular_sampling import progressive_effects as effects
from h3_audio_t8_pkg.modular_sampling import progressive_effect_identity as identity
from h3_audio_t8_pkg.modular_sampling import progressive_storage as storage
from h3_audio_t8_pkg.modular_sampling import progressive_high_result as high_results
from h3_audio_t8_pkg.modular_sampling.eav import EAVConfig
from test_modular_progressive_stages import inputs, low_only, restart
from test_modular_progressive_effects import selected, sample
import test_progressive_sampling_runtime as fixtures

stub_lifter = fixtures.stub_lifter


@pytest.fixture(autouse=True)
def report_identity_mismatch(monkeypatch):
    original = stages.model_identity_matches
    def compare(left, right):
        matched = original(left, right)
        if not matched:
            def differences(a, b, path="model"):
                if isinstance(a, dict) and isinstance(b, dict):
                    return [row for key in set(a) | set(b)
                            for row in differences(a.get(key), b.get(key), path + "." + key)]
                return [] if a == b else [(path, a, b)]
            print("IDENTITY_DIFF=" + json.dumps(differences(left, right), default=str))
        return matched
    monkeypatch.setattr(stages, "model_identity_matches", compare)


def prepare(case, phase, kind, state=None):
    return selected(case, phase, source=state, with_relay=kind != "eav",
                    config=EAVConfig("apply_exp", .2, 0., 1.) if kind != "relay" else None)


@pytest.mark.parametrize("task", ["t2va", "i2va"])
@pytest.mark.parametrize("kind", ["relay", "eav", "combined"])
@pytest.mark.parametrize("phase", ["low", "high"])
@pytest.mark.parametrize("backend", ["automatic", "pytorch"])
def test_exact_projection_stable_before_after_native_execution(stub_lifter, task, kind, phase, backend):
    case = inputs(task)
    if backend == "pytorch":
        set_h3_attention_backend(case["model"], attention.attention_pytorch)
    state = restart(case, low_only(case)) if phase == "high" else None
    model, pos, neg = prepare(case, phase, kind, state)
    wrappers = {k: {n: list(v) for n, v in names.items()} for k, names in model.wrappers.items()}
    selector = model.model_options["transformer_options"]["optimized_attention_override"]
    host = effects.owner(model)
    clone, contract = identity.project(model)
    assert clone is not model and contract
    before = stages.native_model_identity(model, case["sampler"])
    assert stages.portable(before)
    actual = sample(case, phase, model, pos, neg, state=state)
    assert actual.verify()["portable_identity"]
    assert stages.native_model_identity(model, case["sampler"]) == before
    assert model.wrappers == wrappers and effects.owner(model) is host
    assert model.model_options["transformer_options"]["optimized_attention_override"] is selector
    # Different graph objects, same authentic stage; counters cannot affect identity.
    rebuilt, _, _ = prepare(case, phase, kind, state)
    assert stages.native_model_identity(rebuilt, case["sampler"]) == before


@pytest.mark.parametrize("kind", ["relay", "eav", "combined"])
def test_effect_low_and_completed_high_save_load_without_sampling(stub_lifter, tmp_path, monkeypatch, kind):
    case = inputs()
    boundary = sample(case, "low", *prepare(case, "low", kind))
    low_path, low_sha, _ = storage.save_boundary(boundary, tmp_path)
    restored, _ = storage.load_boundary(tmp_path, low_path, low_sha)
    state = restart(case, restored)
    high = sample(case, "high", *prepare(case, "high", kind, state), state=state)
    path, digest, _ = high_results.save_result(high, tmp_path)
    monkeypatch.setattr(stages, "sample_low", lambda *a, **k: pytest.fail("Load executed LOW"))
    monkeypatch.setattr(stages, "sample_high", lambda *a, **k: pytest.fail("Load executed HIGH"))
    read, report = high_results.load_result(tmp_path, path, digest)
    assert json.loads(report)["sampling_calls"] == 0
    for a, b in zip(read.output["samples"].unbind(), high.output["samples"].unbind()):
        assert torch.equal(a, b)
    assert effects.audit(read, "high") == effects.audit(high, "high")


@pytest.mark.parametrize("kind", ["relay", "eav", "combined"])
def test_foreign_user_wrapper_is_preserved_but_never_made_portable(kind):
    case = inputs()
    model, pos, neg = prepare(case, "low", kind)
    calls = []
    def foreign(executor, *args, **kwargs):
        calls.append(1)
        return executor(*args, **kwargs)
    model.add_wrapper_with_key("apply_model", "user", foreign)
    before = deepcopy(model.patches)
    result = sample(case, "low", model, pos, neg)
    assert not result.verify()["portable_identity"] and len(calls) == 2
    assert model.get_wrappers("apply_model", "user") == [foreign]
    assert model.patches == before


@pytest.mark.parametrize("kind", ["relay", "eav", "combined"])
@pytest.mark.parametrize("changed", ["selector", "wrapper", "host_method", "runtime_state", "container"])
def test_modified_effect_owners_cannot_authorize_portable_reuse(kind, changed):
    case = inputs()
    model, _, _ = prepare(case, "low", kind)
    host = effects.owner(model)
    selector = model.model_options["transformer_options"]["optimized_attention_override"]
    if changed == "selector":
        model.model_options["transformer_options"]["optimized_attention_override"] = lambda *a, **k: None
    elif changed == "wrapper":
        key = next(k for k, v in model.wrappers["diffusion_model"].items() if v)
        model.wrappers["diffusion_model"][key] = [lambda executor, *a, **k: executor(*a, **k)]
    elif changed == "host_method":
        host.report = lambda *a: {}
    elif changed == "runtime_state":
        if host.eav_runtime:
            host.eav_runtime.begin_forward = lambda *a, **k: 0
        else:
            host.relay_report["completed_calls"] = dict(host.relay_report["completed_calls"])
    else:
        selector.container_function = lambda *a, **k: None
    assert not stages.portable(stages.native_model_identity(model, case["sampler"]))


@pytest.mark.parametrize("owner", ["stage", "eav", "mask", "delegate"])
def test_replaced_effect_class_method_changes_identity(monkeypatch, owner):
    case = inputs()
    set_h3_attention_backend(case["model"], attention.attention_pytorch)
    model, _, _ = prepare(case, "low", "combined")
    host = effects.owner(model)
    choices = {"stage": (type(host), "reset"), "eav": (type(host.eav_runtime), "begin_forward"),
               "mask": (type(host.mask), "report"), "delegate": (identity._PlainDelegate, "attention")}
    before = stages.native_model_identity(model, case["sampler"])
    assert stages.portable(before)
    kind, method = choices[owner]
    original = getattr(kind, method)
    monkeypatch.setattr(kind, method, lambda *a, **k: original(*a, **k))
    after = stages.native_model_identity(model, case["sampler"])
    assert not stages.portable(after) and not stages.model_identity_matches(before, after)


def test_lazy_sage_delegate_is_unverified_before_and_after_cache_creation():
    from h3_audio_t8_pkg.patch_stack_policy import UnverifiedModelStack
    backend = identity._PlainDelegate(None, "sage")
    for child in (None, object()):
        backend.masked_delegate = child
        with pytest.raises(UnverifiedModelStack, match="Sage"):
            identity._delegate(backend)


@pytest.mark.parametrize("kind", ["relay", "eav", "combined"])
@pytest.mark.parametrize("change_high", [False, True])
def test_new_process_saved_effect_low_runs_only_independent_high(tmp_path, stub_lifter, kind, change_high):
    case = inputs()
    low = sample(case, "low", *prepare(case, "low", kind))
    path, digest, _ = storage.save_boundary(low, tmp_path)
    if change_high:
        key, weight = next(iter(dict(case["model"].model.named_parameters()).items()))
        case["model"].add_patches({key: ("diff", (torch.full_like(weight, .01),))})
    state = restart(case, low)
    expected = sample(case, "high", *prepare(case, "high", kind, state), state=state)
    code = """
import json, runpy, sys
runpy.run_path('tools/check_director_d1_cpu.py', run_name='config')
runpy.run_path('tests/conftest.py')
import pytest, torch
from test_modular_progressive_effect_identity import prepare, sample, inputs, restart
from test_progressive_sampling_runtime import stub_lifter
from h3_audio_t8_pkg.modular_sampling import progressive as stages, progressive_storage as storage
from h3_audio_t8_pkg.modular_sampling import progressive_effects as effects, progressive_high_result as high
low, _ = storage.load_boundary(sys.argv[1], sys.argv[2], sys.argv[3])
case = inputs()
if sys.argv[5] == 'True':
    key, weight = next(iter(dict(case['model'].model.named_parameters()).items()))
    case['model'].add_patches({key: ('diff', (torch.full_like(weight, .01),))})
def forbidden(*a, **k):
    raise AssertionError('LOW/whole sampler ran in a HIGH-only restored graph')
with pytest.MonkeyPatch.context() as patch:
    stub_lifter.__wrapped__(patch)
    patch.setattr(stages, 'sample_low', forbidden)
    patch.setattr(stages.legacy, 'sample_progressive_h3', forbidden)
    state = restart(case, low)
    result = sample(case, 'high', *prepare(case, 'high', sys.argv[4], state), state=state)
assert result.verify()['sampling']['execution']['actual_apply_calls'] == 2
audit = effects.audit(result, 'high')
assert audit['status'] == 'verified_stage_execution_quality_unverified'
if sys.argv[4] != 'eav':
    assert audit['relay']['completed_calls'] == {'forward': 2, 'routed_attention': 2}
path, digest, _ = high.save_result(result, sys.argv[1])
stages.sample_low = stages.sample_high = forbidden
restored, report = high.load_result(sys.argv[1], path, digest)
assert json.loads(report)['sampling_calls'] == 0 and not torch.cuda.is_initialized()
print('RESULT=' + json.dumps(stages.snapshot(restored.output)))
"""
    child = subprocess.run([sys.executable, "-c", code, str(tmp_path), path, digest, kind, str(change_high)],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=60)
    assert child.returncode == 0, child.stdout + child.stderr
    actual = json.loads(next(line[7:] for line in child.stdout.splitlines() if line.startswith("RESULT=")))
    assert actual == stages.snapshot(expected.output)
