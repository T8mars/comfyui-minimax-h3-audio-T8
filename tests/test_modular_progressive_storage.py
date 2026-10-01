"""Typed LOW persistence and actual new-process HIGH-only tiny Core replay."""
from dataclasses import replace
import json
from pathlib import Path
import struct
import subprocess
import sys

import pytest
import torch
from safetensors.torch import save_file

from h3_audio_t8_pkg.modular_sampling import progressive as stages
from h3_audio_t8_pkg.modular_sampling import progressive_storage as storage
from h3_audio_t8_pkg.modular_sampling.storage import _lease
from test_modular_progressive_stages import inputs, low_only, initialized, restart, high_only
import test_progressive_sampling_runtime as runtime_fixtures

stub_lifter = runtime_fixtures.stub_lifter


@pytest.fixture
def saved(tmp_path):
    boundary = low_only(inputs())
    path, digest, _ = storage.save_boundary(boundary, tmp_path)
    return boundary, path, digest


def test_exact_roundtrip_is_typed_not_an_ordinary_latent(tmp_path, saved):
    boundary, path, digest = saved
    loaded, text = storage.load_boundary(tmp_path, path, digest)
    assert type(loaded) is stages.ProgressiveBoundary
    assert loaded.receipt_json == boundary.receipt_json
    assert json.loads(text)["low_executed"] is False
    assert json.loads(text)["automatic_cache_reuse"] is False
    assert storage.fingerprint_boundary(tmp_path, path) == digest
    assert set(loaded.tensors) == {"clean_video", "audio_next", "anchor_audio_noise"}
    for key in loaded.tensors:
        assert torch.equal(loaded.tensors[key], boundary.tensors[key])


def test_saving_again_never_overwrites_prior_artifact(tmp_path, saved):
    boundary, path, digest = saved
    new_path, new_digest, _ = storage.save_boundary(boundary, tmp_path)
    assert new_path != path
    assert storage.file_sha(tmp_path / path) == digest
    assert storage.file_sha(tmp_path / new_path) == new_digest


@pytest.mark.parametrize("kind", ["empty", "avatar"])
@pytest.mark.parametrize("change_high", [False, True])
def test_new_process_load_runs_only_high_and_matches_same_process(tmp_path, stub_lifter, kind, change_high):
    case = inputs() if kind == "empty" else inputs(source=initialized("avatar"), mode="initialized_av_exp")
    boundary = low_only(case)
    path, digest, _ = storage.save_boundary(boundary, tmp_path)
    if change_high:
        with torch.no_grad():
            next(case["model"].model.parameters()).add_(.1)
        case["hp"][0][0] = torch.ones_like(case["hp"][0][0]) * .3
    expected, _ = high_only(case, restart(case, boundary))
    code = """
import json, runpy, sys
runpy.run_path('tools/check_director_d1_cpu.py', run_name='config')
runpy.run_path('tests/conftest.py')
import pytest, torch
from test_modular_progressive_stages import inputs, initialized, restart, high_only
from test_progressive_sampling_runtime import stub_lifter
from h3_audio_t8_pkg.modular_sampling import progressive as stages
from h3_audio_t8_pkg.modular_sampling.progressive_storage import load_boundary
from h3_audio_t8_pkg import progressive_sampling_runtime as legacy
boundary, _ = load_boundary(sys.argv[1], sys.argv[2], sys.argv[3])
case = inputs() if sys.argv[4] == 'empty' else inputs(source=initialized('avatar'), mode='initialized_av_exp')
if sys.argv[5] == 'True':
    with torch.no_grad():
        next(case['model'].model.parameters()).add_(.1)
    case['hp'][0][0] = torch.ones_like(case['hp'][0][0]) * .3
def forbidden(*args, **kwargs):
    raise AssertionError('LOW/full executor ran during explicit HIGH replay')
with pytest.MonkeyPatch.context() as patch:
    stub_lifter.__wrapped__(patch)
    patch.setattr(stages, 'sample_low', forbidden)
    patch.setattr(legacy, 'sample_progressive_h3', forbidden)
    output, report = high_only(case, restart(case, boundary))
assert not torch.cuda.is_initialized()
report = json.loads(report)
assert report['execution']['actual_apply_calls'] == 2 and not report['low_executed']
print('RESULT=' + json.dumps(stages.snapshot(output)))
"""
    child = subprocess.run([sys.executable, "-c", code, str(tmp_path), path, digest, kind, str(change_high)],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=60)
    assert child.returncode == 0, child.stdout + child.stderr
    actual = json.loads(next(line[7:] for line in child.stdout.splitlines() if line.startswith("RESULT=")))
    assert actual == stages.snapshot(expected)


@pytest.mark.parametrize("digest", ["", "abc", "0" * 64])
def test_exact_sha_required(tmp_path, saved, digest):
    with pytest.raises(ValueError, match="SHA"):
        storage.load_boundary(tmp_path, saved[1], digest)


@pytest.mark.parametrize("path", ["../progressive-low.safetensors", "x/../../progressive-low.safetensors",
                                "x/progressive-low.safetensors.partial", "x/state.safetensors"])
def test_wrong_type_partial_or_outside_path_rejected(tmp_path, path):
    with pytest.raises(ValueError):
        storage.load_boundary(tmp_path, path, "a" * 64)


@pytest.mark.parametrize("kind", ["truncated", "header", "changed_tensor", "extra_tensor", "metadata", "schema"])
def test_malformed_artifact_never_becomes_a_completed_boundary(tmp_path, saved, kind):
    boundary, relative, _ = saved
    path = tmp_path / relative
    if kind == "truncated":
        path.write_bytes(b"bad")
    elif kind == "header":
        path.write_bytes(struct.pack("<Q", storage.MAX_JSON + 1) + b"a" * 32)
    else:
        tensors = dict(boundary.tensors)
        payload = {"schema": storage.SCHEMA, "receipt_json": boundary.receipt_json}
        if kind == "changed_tensor":
            tensors["audio_next"] = tensors["audio_next"] + .1
        elif kind == "extra_tensor":
            tensors["unregistered"] = torch.zeros(1)
        elif kind == "schema":
            payload["schema"] = "ordinary-completed-latent"
        metadata = {"wrong": "type"} if kind == "metadata" else {"progressive_json": json.dumps(payload)}
        save_file(tensors, str(path), metadata=metadata)
    with pytest.raises(ValueError):
        storage.load_boundary(tmp_path, relative, storage.file_sha(path))


def test_save_interruption_has_no_completion_point(tmp_path, monkeypatch):
    boundary = low_only(inputs())
    def interrupted(*args):
        raise RuntimeError("test before commit")
    monkeypatch.setattr(storage.os, "replace", interrupted)
    with pytest.raises(RuntimeError, match="before commit"):
        storage.save_boundary(boundary, tmp_path)
    assert list(tmp_path.rglob("*.partial"))
    assert not list(tmp_path.rglob(storage.FILENAME))


def test_locked_store_cannot_be_read_or_fingerprinted(tmp_path, saved):
    _, path, digest = saved
    with _lease((tmp_path / path).parent):
        with pytest.raises(RuntimeError, match="busy"):
            storage.load_boundary(tmp_path, path, digest)
        with pytest.raises(RuntimeError, match="busy"):
            storage.fingerprint_boundary(tmp_path, path)
    storage.load_boundary(tmp_path, path, digest)


def test_unknown_wrapper_can_archive_but_cannot_claim_persistent_reuse(tmp_path):
    case = inputs()
    case["model"].set_model_unet_function_wrapper(
        lambda execute, args: execute(args["input"], args["timestep"], **args["c"]))
    boundary = low_only(case)
    path, digest, text = storage.save_boundary(boundary, tmp_path)
    assert json.loads(text)["status"] == "archived_unverified_identity"
    with pytest.raises(ValueError, match="Unverified executable"):
        storage.load_boundary(tmp_path, path, digest)


def test_incomplete_forward_evidence_not_saved(tmp_path):
    boundary = low_only(inputs())
    receipt = boundary.verify()
    receipt["execution"]["actual_apply_calls"] = 0
    receipt["portable_identity"] = False
    receipt.pop("receipt_sha256")
    receipt["receipt_sha256"] = stages.sha(receipt)
    boundary = replace(boundary, receipt_json=stages.canonical(receipt))
    with pytest.raises(ValueError, match="Incomplete LOW"):
        storage.save_boundary(boundary, tmp_path)


def test_query_missing_does_not_create_storage(tmp_path):
    absent = tmp_path / "absent"
    with pytest.raises(FileNotFoundError):
        storage.fingerprint_boundary(absent, "x/" + storage.FILENAME)
    assert not absent.exists()


def test_metadata_bounds_checked_before_commit(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "MAX_JSON", 16)
    with pytest.raises(ValueError, match="size limit"):
        storage.save_boundary(low_only(inputs()), tmp_path)
    assert not list(tmp_path.rglob(storage.FILENAME))


def test_tensor_archive_cannot_be_saved_as_wrong_public_type(tmp_path):
    with pytest.raises(ValueError, match="typed ProgressiveBoundary"):
        storage.save_boundary({"samples": torch.zeros(1)}, tmp_path)


def test_source_input_tamper_prevents_save(tmp_path):
    boundary = low_only(inputs())
    boundary.tensors["audio_next"] = boundary.tensors["audio_next"] + 1
    with pytest.raises(ValueError, match="contents changed"):
        storage.save_boundary(boundary, tmp_path)


def test_mutation_during_serialization_never_commits(tmp_path, monkeypatch):
    boundary = low_only(inputs())
    save = storage.save_file
    def mutated(*args, **kwargs):
        save(*args, **kwargs)
        boundary.tensors["audio_next"] = boundary.tensors["audio_next"] + .1
    monkeypatch.setattr(storage, "save_file", mutated)
    with pytest.raises(ValueError, match="contents changed"):
        storage.save_boundary(boundary, tmp_path)
    assert not list(tmp_path.rglob(storage.FILENAME))


def test_actual_killed_lock_owner_releases_os_lease(tmp_path, saved):
    _, path, digest = saved
    directory = (tmp_path / path).parent
    code = """
import runpy,sys
from pathlib import Path
runpy.run_path('tools/check_director_d1_cpu.py',run_name='config')
runpy.run_path('tests/conftest.py')
from h3_audio_t8_pkg.modular_sampling.storage import _lease
with _lease(Path(sys.argv[1])):
    print('LEASE_READY',flush=True)
    sys.stdin.readline()
"""
    child = subprocess.Popen([sys.executable, "-u", "-c", code, str(directory)],
        cwd=Path(__file__).resolve().parents[1], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True)
    try:
        while True:
            line = child.stdout.readline()
            assert line, child.stderr.read()
            if line.strip() == "LEASE_READY":
                break
        with pytest.raises(RuntimeError, match="busy"):
            storage.load_boundary(tmp_path, path, digest)
        child.kill()  # Only this test's explicitly owned child process.
        child.communicate(timeout=15)
        storage.load_boundary(tmp_path, path, digest)
    finally:
        if child.poll() is None:
            child.kill()
        child.communicate(timeout=15)
