"""Completed HIGH roundtrip, corruption, atomicity and no-model fresh loading."""
from dataclasses import replace
import json
from pathlib import Path
import struct
import subprocess
import sys

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from h3_audio_t8_pkg.modular_sampling import progressive as stages
from h3_audio_t8_pkg.modular_sampling import progressive_high_result as results
from h3_audio_t8_pkg.modular_sampling import progressive_nodes as nodes
from h3_audio_t8_pkg.modular_sampling.storage import _lease
from test_modular_progressive_stages import inputs, low_only, restart
import test_progressive_sampling_runtime as runtime_fixtures

stub_lifter = runtime_fixtures.stub_lifter


@pytest.fixture
def completed(stub_lifter):
    case = inputs()
    case["source"]["extra"] = {"fps": 24., "index": (1, 3), "labels": ["a", None, True], "tensor": torch.ones(2)}
    state = restart(case, low_only(case))
    result, _ = results.sample_high_result(state, case["model"], case["sampler"], case["hp"], case["hn"], seed=7)
    return result


@pytest.fixture
def saved(completed, tmp_path):
    path, digest, _ = results.save_result(completed, tmp_path)
    return completed, path, digest


def test_high_nodes_roundtrip_preserves_all_av_and_metadata(saved, tmp_path, monkeypatch):
    completed, path, digest = saved
    monkeypatch.setattr(nodes, "_store_root", lambda: tmp_path)
    output, loaded, report = nodes.MiniMaxH3ProgressiveHighLoadEXPT8.execute(path, digest).result
    assert loaded.receipt_json == completed.receipt_json
    assert stages.snapshot(output) == stages.snapshot(completed.output)
    assert json.loads(report)["sampling_calls"] == 0
    assert nodes.MiniMaxH3ProgressiveHighLoadEXPT8.fingerprint_inputs(path, digest) == digest
    output2, path2, digest2, _ = nodes.MiniMaxH3ProgressiveHighSaveEXPT8.execute(loaded, "new/HIGH").result
    assert output2 is output and path2 != path
    assert results.file_sha(tmp_path / path) == digest and results.file_sha(tmp_path / path2) == digest2


def test_new_process_loads_final_av_without_any_sampler_or_model(saved, tmp_path):
    completed, path, digest = saved
    code = """
import json, runpy, sys
runpy.run_path('tools/check_director_d1_cpu.py', run_name='config')
runpy.run_path('tests/conftest.py')
import torch
from h3_audio_t8_pkg.modular_sampling import progressive as stages
from h3_audio_t8_pkg.modular_sampling.progressive_high_result import load_result
def forbidden(*a, **k):
    raise AssertionError('Sampling/model/lift ran during completed HIGH load')
stages.sample_low = stages.sample_high = stages.legacy.sample_progressive_h3 = forbidden
stages.native_model_identity = stages.legacy._lift_video = forbidden
result, report = load_result(sys.argv[1], sys.argv[2], sys.argv[3])
assert json.loads(report)['sampling_calls'] == 0
assert not torch.cuda.is_initialized()
print('RESULT=' + json.dumps(stages.snapshot(result.output)))
"""
    child = subprocess.run([sys.executable, "-c", code, str(tmp_path), path, digest],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=60)
    assert child.returncode == 0, child.stdout + child.stderr
    actual = json.loads(next(line[7:] for line in child.stdout.splitlines() if line.startswith("RESULT=")))
    assert actual == stages.snapshot(completed.output)


@pytest.mark.parametrize("field", ["video", "audio", "metadata", "schema", "callback", "forwards", "portable"])
def test_mutated_result_is_not_savable(completed, tmp_path, field):
    with torch.inference_mode():
        if field in ("video", "audio"):
            completed.output["samples"].unbind()[field == "audio"].add_(.1)
        elif field == "metadata":
            completed.output["extra"]["tensor"].add_(.1)
        else:
            receipt = json.loads(completed.receipt_json)
            if field == "schema":
                receipt["schema"] = "ordinary_latent"
            elif field == "callback":
                receipt["sampling"]["execution"]["callbacks"] = [0]
            elif field == "forwards":
                receipt["sampling"]["execution"]["actual_apply_calls"] = 0
            else:
                receipt["portable_identity"] = False
            receipt.pop("receipt_sha256")
            receipt["receipt_sha256"] = results.sha(receipt)
            completed = replace(completed, receipt_json=results.canonical(receipt))
    with pytest.raises(ValueError):
        results.save_result(completed, tmp_path)
    assert not list(tmp_path.rglob(results.FILENAME))


@pytest.mark.parametrize("digest", ["", "abc", "0" * 64])
def test_exact_sha_required(saved, tmp_path, digest):
    with pytest.raises(ValueError, match="SHA"):
        results.load_result(tmp_path, saved[1], digest)


@pytest.mark.parametrize("path", ["../progressive-high.safetensors", "a/progressive-low.safetensors",
                                "a/progressive-high.safetensors.partial", "a/state.safetensors"])
def test_wrong_type_partial_and_escape_rejected(tmp_path, path):
    with pytest.raises(ValueError):
        results.load_result(tmp_path, path, "0" * 64)


@pytest.mark.parametrize("kind", ["truncated", "header", "unused", "changed_tensor", "metadata", "schema", "duplicate"])
def test_corrupted_payload_with_fresh_file_sha_still_rejected(saved, tmp_path, kind):
    _, path, _ = saved
    target = tmp_path / path
    with safe_open(str(target), framework="pt", device="cpu") as handle:
        # Release the Windows mmap before deliberately truncating the fixture.
        tensors = {key: handle.get_tensor(key).clone() for key in handle.keys()}
        metadata = handle.metadata()
    if kind == "truncated":
        target.write_bytes(b"bad")
    elif kind == "header":
        target.write_bytes(struct.pack("<Q", results.MAX_JSON + 1) + b"a" * 32)
    else:
        if kind == "unused":
            tensors["not_referenced"] = torch.zeros(2)
        elif kind == "changed_tensor":
            key = next(iter(tensors))
            tensors[key] = tensors[key] + .1
        elif kind == "metadata":
            metadata = {"wrong": "type"}
        elif kind == "duplicate":
            value = metadata["progressive_high_json"]
            metadata["progressive_high_json"] = '{"schema":"wrong",' + value[1:]
        else:
            value = json.loads(metadata["progressive_high_json"])
            value["schema"] = "ordinary-completed-latent"
            metadata["progressive_high_json"] = json.dumps(value)
        save_file(tensors, str(target), metadata=metadata)
    with pytest.raises(ValueError):
        results.load_result(tmp_path, path, results.file_sha(target))


@pytest.mark.parametrize("fault", ["write", "mutation"])
def test_failed_save_never_creates_completed_artifact(completed, tmp_path, monkeypatch, fault):
    original = results.save_file
    def fail(tensors, path, **kwargs):
        original(tensors, path, **kwargs)
        if fault == "write":
            raise OSError("interrupted writer")
        with torch.inference_mode():
            completed.output["samples"].unbind()[1].add_(.1)
    monkeypatch.setattr(results, "save_file", fail)
    with pytest.raises((OSError, ValueError)):
        results.save_result(completed, tmp_path)
    assert not list(tmp_path.rglob(results.FILENAME))
    assert list(tmp_path.rglob(results.FILENAME + ".partial"))


def test_busy_file_refuses_reuse_then_loads_after_release(saved, tmp_path):
    _, path, digest = saved
    with _lease((tmp_path / path).parent):
        with pytest.raises(RuntimeError, match="busy"):
            results.load_result(tmp_path, path, digest)
        with pytest.raises(RuntimeError, match="busy"):
            results.fingerprint_result(tmp_path, path)
    assert results.load_result(tmp_path, path, digest)[0].verify()


def test_missing_fingerprint_does_not_create_store(tmp_path, monkeypatch):
    target = tmp_path / "absent"
    monkeypatch.setattr(nodes, "_store_root", lambda: target)
    value = nodes.MiniMaxH3ProgressiveHighLoadEXPT8.fingerprint_inputs("a/" + results.FILENAME, "0" * 64)
    assert value != value and not target.exists()


def test_only_typed_completed_result_allowed(tmp_path):
    with pytest.raises(ValueError, match="typed"):
        results.save_result({"samples": torch.zeros(2)}, tmp_path)


def test_unknown_identity_archives_but_does_not_certify_persistent_reuse(stub_lifter, tmp_path):
    import comfy.patcher_extension
    case = inputs()
    state = restart(case, low_only(case))
    calls = []
    def delegate(executor, *args, **kwargs):
        calls.append(1)
        return executor(*args, **kwargs)
    case["model"].add_wrapper_with_key(comfy.patcher_extension.WrappersMP.APPLY_MODEL, "user", delegate)
    result, _ = results.sample_high_result(state, case["model"], case["sampler"], case["hp"], case["hn"], seed=7)
    assert calls == [1, 1] and not result.verify()["portable_identity"]
    path, digest, text = results.save_result(result, tmp_path)
    assert json.loads(text)["status"] == "archived_unverified_identity"
    with pytest.raises(ValueError, match="Unverified"):
        results.load_result(tmp_path, path, digest)
