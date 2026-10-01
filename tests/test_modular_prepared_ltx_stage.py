"""Synthetic split-stage state tests; no model inference or media qualification."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from h3_audio_t8_pkg import prepared_generation_contract as contract
from h3_audio_t8_pkg import prepared_generation_runtime as legacy
from h3_audio_t8_pkg.modular_sampling import prepared_ltx_stage as stage
from tests.test_prepared_generation_contract import bundle as ltx_bundle  # noqa: F401


@pytest.fixture
def rig(ltx_bundle, tmp_path, monkeypatch):  # noqa: F811
    bundle = ltx_bundle
    monkeypatch.setattr(contract, "environment_identity", lambda: {"synthetic": True})
    monkeypatch.setattr(stage, "engine_sources", lambda: {"synthetic_backend": "a" * 64})
    monkeypatch.setattr(stage, "verify_assets", lambda *args: None)

    class Reader:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def sample(self):
            return {"gpu_uuid": "SYNTHETIC-no-GPU"}

    class Guard:
        def __init__(self, *args):
            pass

        def observe(self, *args, **kwargs):
            return None

    monkeypatch.setattr(stage, "NvmlResourceReader", Reader)
    monkeypatch.setattr(stage, "ResourceGuard", Guard)
    calls = []

    def worker(directory, spec, request, *args):
        directory.mkdir()
        calls.append((spec[0], deepcopy(request)))
        path = directory / spec[2]
        path.write_bytes(b"SYNTHETIC ONLY: " + spec[0].encode())
        return legacy.artifact_record(path, {"synthetic": True})

    monkeypatch.setattr(stage, "run_worker", worker)
    kwargs = {"output_directory": tmp_path, "chain_id": "split_fixture", "noise_seed": 8301,
              "resume_existing": True, "lease_path": str(tmp_path / "serial.lock"),
              "interrupt": lambda: None}
    return bundle, kwargs, calls, worker


def test_generate_and_decode_are_separate_and_cache_independent(rig):
    bundle, kwargs, calls, _ = rig
    receipt, generation = stage.run_generation(bundle, **kwargs)
    assert [name for name, _ in calls] == ["ltx_prepared_refinement_worker.py"]
    assert generation["generation_ran"] and not generation["decode_ran"]
    assert receipt["schema"] == stage.RECEIPT_SCHEMA
    assert stage.NAMESPACE in receipt["state_path"]
    assert "generation" in json.loads(Path(receipt["state_path"]).read_text())["stages"]
    again, cached = stage.run_generation(bundle, **kwargs)
    assert again == receipt and not cached["generation_ran"] and len(calls) == 1
    movie, report = stage.run_decode(bundle, receipt, output_directory=kwargs["output_directory"],
                                      lease_path=kwargs["lease_path"], interrupt=lambda: None)
    assert [name for name, _ in calls] == ["ltx_prepared_refinement_worker.py", "ltx_refined_decode_worker.py"]
    assert calls[1][1]["latent"] == receipt["latent_path"]
    assert calls[1][1]["original_video"] == bundle["decode"]["original_video"]
    assert movie.exists() and report["decode_ran"] and not report["generation_ran"]
    _, cached_decode = stage.run_decode(bundle, receipt, output_directory=kwargs["output_directory"],
                                         lease_path=kwargs["lease_path"], interrupt=lambda: None)
    assert not cached_decode["decode_ran"] and len(calls) == 2


def test_explicit_load_after_generation_has_no_worker_or_gpu_reader(rig, monkeypatch):
    bundle, kwargs, calls, _ = rig
    receipt, _ = stage.run_generation(bundle, **kwargs)
    monkeypatch.setattr(stage, "NvmlResourceReader", lambda: pytest.fail("load started GPU reader"))
    loaded, report = stage.load_generation(bundle, output_directory=kwargs["output_directory"],
        chain_id=kwargs["chain_id"], noise_seed=kwargs["noise_seed"],
        expected_sha256=receipt["sha256"], interrupt=lambda: None)
    assert loaded == receipt and not report["generation_ran"] and len(calls) == 1


def test_decode_never_runs_missing_generation(rig):
    bundle, kwargs, calls, _ = rig
    fake = {"schema": stage.RECEIPT_SCHEMA, "fingerprint": "a" * 64,
            "chain_id": kwargs["chain_id"], "noise_seed": 8301,
            "state_path": str(kwargs["output_directory"] / "missing"),
            "latent_path": str(kwargs["output_directory"] / "missing.safetensors"), "sha256": "b" * 64}
    with pytest.raises((ValueError, FileNotFoundError)):
        stage.run_decode(bundle, fake, output_directory=kwargs["output_directory"],
            lease_path=kwargs["lease_path"], interrupt=lambda: None)
    assert calls == []


@pytest.mark.parametrize("change", ["sha", "path", "seed", "prompt", "backend", "asset", "state"])
def test_changed_or_forged_generation_cannot_decode(rig, monkeypatch, change):
    bundle, kwargs, calls, _ = rig
    receipt, _ = stage.run_generation(bundle, **kwargs)
    forged = deepcopy(receipt)
    modified_bundle = deepcopy(bundle)
    if change == "sha":
        forged["sha256"] = "f" * 64
    elif change == "path":
        forged["latent_path"] = bundle["generation"]["inputs"]
    elif change == "seed":
        forged["noise_seed"] += 1
    elif change == "prompt":
        modified_bundle["generation"]["prompt"] = "another prepared prompt"
    elif change == "backend":
        monkeypatch.setattr(stage, "engine_sources", lambda: {"synthetic_backend": "b" * 64})
    elif change == "asset":
        Path(receipt["latent_path"]).write_bytes(b"changed content")
    else:
        state_file = Path(receipt["state_path"])
        state_value = json.loads(state_file.read_text())
        state_value["stages"]["generation"]["sha256"] = "f" * 64
        legacy.write_json(state_file, state_value)
    with pytest.raises((ValueError, FileNotFoundError)):
        stage.run_decode(modified_bundle, forged, output_directory=kwargs["output_directory"],
                         lease_path=kwargs["lease_path"], interrupt=lambda: None)
    assert len(calls) == 1


def test_generation_failure_is_bound_and_resume_only_generation(rig, monkeypatch):
    bundle, kwargs, calls, worker = rig
    def fail(*args):
        raise RuntimeError("synthetic generation failure")
    monkeypatch.setattr(stage, "run_worker", fail)
    with pytest.raises(RuntimeError, match="generation failure"):
        stage.run_generation(bundle, **kwargs)
    state_file = stage._root(kwargs["output_directory"], kwargs["chain_id"]) / "state.json"
    assert json.loads(state_file.read_text())["stages"] == {}
    monkeypatch.setattr(stage, "run_worker", worker)
    receipt, _ = stage.run_generation(bundle, **kwargs)
    assert receipt["latent_path"] and len(calls) == 1
    kwargs["noise_seed"] += 1
    with pytest.raises(ValueError, match="new chain_id"):
        stage.run_generation(bundle, **kwargs)
    assert len(calls) == 1


def test_cancel_after_generation_worker_does_not_promote_latent(rig):
    bundle, kwargs, calls, _ = rig
    checks = 0

    def interrupt():
        nonlocal checks
        checks += 1
        if checks == 2:
            raise RuntimeError("synthetic late cancellation")

    kwargs["interrupt"] = interrupt
    with pytest.raises(RuntimeError, match="late cancellation"):
        stage.run_generation(bundle, **kwargs)
    state_file = stage._root(kwargs["output_directory"], kwargs["chain_id"]) / "state.json"
    assert json.loads(state_file.read_text())["stages"] == {}
    assert len(calls) == 1


def test_decode_failure_keeps_generation_and_retries_only_decode(rig, monkeypatch):
    bundle, kwargs, calls, worker = rig
    receipt, _ = stage.run_generation(bundle, **kwargs)

    def fail_decode(directory, spec, request, *args):
        if spec[0] == "ltx_refined_decode_worker.py":
            raise RuntimeError("synthetic decode failure")
        return worker(directory, spec, request, *args)

    monkeypatch.setattr(stage, "run_worker", fail_decode)
    with pytest.raises(RuntimeError, match="decode failure"):
        stage.run_decode(bundle, receipt, output_directory=kwargs["output_directory"],
                         lease_path=kwargs["lease_path"], interrupt=lambda: None)
    state_file = Path(receipt["state_path"])
    assert set(json.loads(state_file.read_text())["stages"]) == {"generation"}
    monkeypatch.setattr(stage, "run_worker", worker)
    _, report = stage.run_decode(bundle, receipt, output_directory=kwargs["output_directory"],
                                 lease_path=kwargs["lease_path"], interrupt=lambda: None)
    assert report["decode_ran"] and len(calls) == 2


def test_old_prepared_namespace_and_runtime_are_not_modified(rig):
    bundle, kwargs, calls, _ = rig
    stage.run_generation(bundle, **kwargs)
    assert not (kwargs["output_directory"] / "MiniMaxH3-Prepared").exists()
    assert len(calls) == 1
