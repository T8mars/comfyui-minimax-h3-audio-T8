"""Exact typed state, corrupt-store rejection and actual new-process TAIL only."""
from dataclasses import replace
import json
from pathlib import Path
import struct
import subprocess
import sys

import comfy.patcher_extension
import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from h3_audio_t8_pkg.modular_sampling import hyperflow as stages
from h3_audio_t8_pkg.modular_sampling import hyperflow_storage as storage
from h3_audio_t8_pkg.modular_sampling.storage import _lease
from test_modular_hyperflow import inputs, head, equal


@pytest.fixture
def saved(tmp_path, monkeypatch):
    low, high, source, positive = inputs(monkeypatch)
    boundary = head(low, source, positive)
    path, digest, report = storage.save_boundary(boundary, tmp_path)
    assert json.loads(report)["portable_identity"] is True
    return boundary, path, digest, high, positive


def test_head_roundtrip_preserves_raw_and_scaffold_separately(tmp_path, saved):
    boundary, path, digest, _, _ = saved
    result, report = storage.load_boundary(tmp_path, path, digest)
    assert type(result) is stages.ContinuousBoundary
    assert result.receipt_json == boundary.receipt_json
    assert torch.equal(result.x_sigma, boundary.x_sigma)
    equal(result.scaffold, boundary.scaffold)
    assert json.loads(report)["sampling_calls"] == 0
    assert storage.fingerprint(tmp_path, path, "head") == digest
    another, _, _ = storage.save_boundary(boundary, tmp_path)
    assert path != another and storage.file_sha(tmp_path / path) == digest


@pytest.mark.parametrize("split", [1, 4, 7])
@pytest.mark.parametrize("distinct", [False, True])
def test_new_process_uses_only_tail_with_independent_content_lora_and_prompt(tmp_path, monkeypatch, split, distinct):
    low, high, source, positive = inputs(monkeypatch, distinct=distinct)
    boundary = head(low, source, positive, split)
    path, digest, _ = storage.save_boundary(boundary, tmp_path)
    if distinct:
        positive[0][0].add_(.3)
    expected, _ = stages.sample_tail(boundary, high, positive, [], seed=19)
    code = r'''
import json, runpy, sys
runpy.run_path('tools/check_director_d1_cpu.py', run_name='config')
runpy.run_path('tests/conftest.py')
import pytest, torch
from test_modular_hyperflow import inputs
from h3_audio_t8_pkg.modular_sampling import hyperflow as stages
from h3_audio_t8_pkg.modular_sampling.hyperflow_storage import load_boundary
from h3_audio_t8_pkg import hyperflow_two_pass_advanced as legacy
import comfy.sample
def forbidden(*a, **kw):
    raise AssertionError('HEAD, old whole split or fresh noise executed during TAIL-only replay')
with pytest.MonkeyPatch.context() as patch:
    _, high, _, positive = inputs(patch, distinct=sys.argv[4] == 'True')
    if sys.argv[4] == 'True':
        positive[0][0].add_(.3)
    boundary, _ = load_boundary(sys.argv[1], sys.argv[2], sys.argv[3])
    patch.setattr(stages, 'sample_head', forbidden)
    patch.setattr(legacy, 'sample_hyperflow_split', forbidden)
    patch.setattr(comfy.sample, 'prepare_noise', forbidden)
    output, text = stages.sample_tail(boundary, high, positive, [], seed=19)
report = json.loads(text)
assert report['execution']['actual_apply_intervals'] == list(range(int(sys.argv[5]), 8))
assert report['execution']['portable_identity'] is True
assert report['head_executed'] is report['fresh_noise'] is False
assert not torch.cuda.is_initialized()
print('RESULT=' + json.dumps(stages.snapshot(output)))
'''
    child = subprocess.run([sys.executable, "-c", code, str(tmp_path), path, digest, str(distinct), str(split)],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=60)
    assert child.returncode == 0, child.stdout + child.stderr
    actual = json.loads(next(line[7:] for line in child.stdout.splitlines() if line.startswith("RESULT=")))
    assert actual == stages.snapshot(expected)


def test_completed_tail_roundtrip_has_no_models_or_sampling(tmp_path, saved, monkeypatch):
    boundary, _, _, high, positive = saved
    result, _ = stages.sample_tail_result(boundary, high, positive, [], seed=7)
    path, digest, _ = storage.save_result(result, tmp_path)
    monkeypatch.setattr(stages, "sample_head", lambda *a, **kw: pytest.fail("HEAD replay"))
    monkeypatch.setattr(stages, "sample_tail", lambda *a, **kw: pytest.fail("TAIL replay"))
    loaded, report = storage.load_result(tmp_path, path, digest)
    assert loaded.receipt_json == result.receipt_json
    equal(loaded.output, result.output)
    assert json.loads(report)["sampling_calls"] == 0
    assert storage.fingerprint(tmp_path, path, "tail") == digest
    with pytest.raises(ValueError, match="expected completed"):
        storage.load_boundary(tmp_path, path, digest)


@pytest.mark.parametrize("digest", ["", "invalid", "0" * 64])
def test_sha_required(tmp_path, saved, digest):
    with pytest.raises(ValueError, match="SHA"):
        storage.load_boundary(tmp_path, saved[1], digest)


@pytest.mark.parametrize("path", ["../hyperflow-head.safetensors", "x/../../hyperflow-head.safetensors",
                                  "x/hyperflow-head.safetensors.partial", "x/progressive-low.safetensors"])
def test_path_and_type_contract(tmp_path, path):
    with pytest.raises(ValueError):
        storage.load_boundary(tmp_path, path, "a" * 64)


@pytest.mark.parametrize("fault", ["short", "header", "extra_tensor", "changed_tensor", "kind", "schema", "metadata", "receipt"])
def test_corrupt_artifacts_cannot_claim_completion(tmp_path, saved, fault):
    _, relative, _, _, _ = saved
    path = tmp_path / relative
    if fault == "short":
        path.write_bytes(b"bad")
    elif fault == "header":
        path.write_bytes(struct.pack("<Q", storage.MAX_JSON + 1) + b"a" * 32)
    else:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            tensors = {key: handle.get_tensor(key) for key in handle.keys()}
            metadata = handle.metadata()
        payload = json.loads(metadata["hyperflow_json"])
        if fault == "extra_tensor":
            tensors["unregistered"] = torch.zeros(1)
        elif fault == "changed_tensor":
            key = next(iter(tensors))
            tensors[key] = tensors[key] + .1
        elif fault == "kind":
            payload["kind"] = "tail"
        elif fault == "schema":
            payload["schema"] = "ordinary.clean.latent"
        elif fault == "receipt":
            receipt = json.loads(payload["receipt_json"])
            receipt["execution"]["callbacks"] = []
            payload["receipt_json"] = json.dumps(receipt)
        metadata = {"hyperflow_json": json.dumps(payload)} if fault != "metadata" else {"foreign": "owner"}
        save_file(tensors, str(path), metadata=metadata)
    with pytest.raises(ValueError):
        storage.load_boundary(tmp_path, relative, storage.file_sha(path))


def test_cancellation_before_commit_does_not_publish_completed_file(tmp_path, saved, monkeypatch):
    root = tmp_path / "cancelled"
    def fail(*args):
        raise RuntimeError("cancel before commit")
    monkeypatch.setattr(storage.os, "replace", fail)
    with pytest.raises(RuntimeError, match="cancel before commit"):
        storage.save_boundary(saved[0], root)
    assert list(root.rglob("*.partial")) and not list(root.rglob("hyperflow-head.safetensors"))


def test_lease_protects_load_and_fingerprint(tmp_path, saved):
    _, path, digest, _, _ = saved
    with _lease((tmp_path / path).parent):
        with pytest.raises(RuntimeError, match="busy"):
            storage.load_boundary(tmp_path, path, digest)
        with pytest.raises(RuntimeError, match="busy"):
            storage.fingerprint(tmp_path, path, "head")
    storage.load_boundary(tmp_path, path, digest)


def test_unknown_wrapper_executes_but_cannot_acquire_portable_resume(tmp_path, monkeypatch):
    low, high, source, positive = inputs(monkeypatch)
    calls = []
    def user(executor, *a, **kw):
        calls.append(True)
        return executor(*a, **kw)
    low.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.APPLY_MODEL, "user", user)
    boundary = head(low, source, positive)
    assert calls and boundary.verify()["execution"]["portable_identity"] is False
    path, digest, report = storage.save_boundary(boundary, tmp_path)
    assert json.loads(report)["status"] == "archived_unverified_identity"
    with pytest.raises(ValueError, match="Unverified executable"):
        storage.load_boundary(tmp_path, path, digest)
    # The original same-process user stack remains executable. Completed AV
    # must not forget its unverified HEAD provenance either.
    result, _ = stages.sample_tail_result(boundary, high, positive, [], seed=7)
    path, digest, report = storage.save_result(result, tmp_path)
    assert json.loads(report)["portable_identity"] is False
    with pytest.raises(ValueError, match="Unverified executable"):
        storage.load_result(tmp_path, path, digest)


def test_forged_portable_flag_is_rejected(tmp_path, saved):
    boundary = saved[0]
    receipt = json.loads(boundary.receipt_json)
    receipt["execution"]["actual_apply_intervals"] = []
    receipt.pop("receipt_sha256")
    receipt["receipt_sha256"] = stages.sha(receipt)
    forged = replace(boundary, receipt_json=json.dumps(receipt))
    with pytest.raises(ValueError, match="portable identity"):
        storage.save_boundary(forged, tmp_path)
