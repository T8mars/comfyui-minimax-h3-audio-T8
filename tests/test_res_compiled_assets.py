"""Data-only boundary asset persistence; no CUDA/execution qualification."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
from safetensors import safe_open
from safetensors.torch import load_file, save_file
import torch
import triton.knobs as knobs

from h3_audio_t8_pkg import res_compiled_assets as assets
from h3_audio_t8_pkg import res_history_exp as res
from h3_audio_t8_pkg.patch_stack_policy import UnverifiedModelStack
from h3_audio_t8_pkg.res_model_identity import loaded_model_identity
from test_res_history_exp import Analytic, CONTRACT, fixture
from test_res_history_setup import model, source_latent, options, run
from test_res_sol_normalization import selected
from test_res_triton_cache_identity import structural_program
from test_res_kj_sage_identity import actual_source_chain  # noqa: F401 -- native source/ops fixture


def boundary(tmp_path, snapshot):
    full, noise, latent, mask = fixture()
    written = []

    def post(state):
        if state.completed_steps == 4:
            written.append(res.save_checkpoint(tmp_path, "assets.h3res.safetensors", state, full,
                original_noise=noise, original_latent_image=latent, denoise_mask=mask,
                run_contract=CONTRACT, compiled_assets=snapshot, hash_chunk_bytes=128))

    res.sample_res_history(Analytic(), noise.clone(), full, extra_args={"seed": 42},
                           post_step=post, disable=True)
    return written[0]


def test_actual_MODEL_collector_remains_cold_warm_stable_and_boundary_files_are_read_only(actual_source_chain, tmp_path, monkeypatch):  # noqa: F811
    module, _, _ = actual_source_chain
    branch = selected(module, tmp_path, dense_blocks="0")
    cold_observed = {}
    cold = loaded_model_identity(branch, observed_artifacts=cold_observed)
    assert cold_observed and all(not item["observed_programs"] for item in cold_observed.values())
    program, files, _ = structural_program(module._quant_query_per_thread_int8_i64_kernel, tmp_path)
    warm_observed = {}
    assert loaded_model_identity(branch, observed_artifacts=warm_observed) == cold
    snapshot = assets.freeze_compiled_assets(warm_observed)
    assert snapshot != assets.freeze_compiled_assets(cold_observed)
    assert not snapshot["portable_resume_qualified"] and not cold["portable_cache_reuse"]
    saved = boundary(tmp_path / "boundaries", snapshot)
    assert saved["payload"]["compiled_assets"] == snapshot
    module._quant_query_per_thread_int8_i64_kernel.device_caches.clear()
    current = {}
    assert loaded_model_identity(branch, observed_artifacts=current) == cold
    monkeypatch.setattr(knobs.cache, "dir", str(tmp_path))
    before = {path: Path(path).read_bytes() for path in files.values()}
    result = assets.validate_bound_compiled_assets(snapshot, current)
    assert result["compiled_programs"] == 1 and result["file_contents_verified"]
    assert not result["portable_resume_qualified"] and not result["kernel_launched"]
    assert before == {path: Path(path).read_bytes() for path in files.values()}
    assert res.read_checkpoint(saved["path"].parent, saved["relative_path"])["file_sha256"] == saved["file_sha256"]
    assert program._run is None and program.module is None and program.function is None
    assert not module._quant_query_per_thread_int8_i64_kernel.device_caches and not torch.cuda.is_initialized()


def test_bound_producer_tamper_changed_files_and_foreign_roots_are_rejected_without_execution(actual_source_chain, tmp_path, monkeypatch):  # noqa: F811
    module, _, _ = actual_source_chain
    branch = selected(module, tmp_path, dense_blocks="0")
    _, files, _ = structural_program(module._quant_query_per_thread_int8_i64_kernel, tmp_path)
    observed = {}
    loaded_model_identity(branch, observed_artifacts=observed)
    snapshot = assets.freeze_compiled_assets(observed)
    monkeypatch.setattr(knobs.cache, "dir", str(tmp_path))
    corrupted = deepcopy(snapshot)
    corrupted["sha256"] = "0" * 64
    with pytest.raises(UnverifiedModelStack, match="snapshot digest"):
        assets.validate_bound_compiled_assets(corrupted, observed)
    changed = deepcopy(observed)
    changed[next(iter(changed))]["producer_sha256"] = "0" * 64
    with pytest.raises(UnverifiedModelStack, match="producer content changed"):
        assets.validate_bound_compiled_assets(snapshot, changed)
    with monkeypatch.context() as other:
        other.setattr(knobs.cache, "dir", str(tmp_path / "other_root"))
        with pytest.raises(UnverifiedModelStack, match="outside its current native root"):
            assets.validate_bound_compiled_assets(snapshot, observed)
    calls = []
    def foreign():
        calls.append(True)
        raise AssertionError("unknown cache path factory must not run")
    descriptor = vars(type(knobs.cache))["dir"]
    with monkeypatch.context() as other:
        other.setattr(descriptor, "default_factory", foreign)
        with pytest.raises(UnverifiedModelStack, match="live executable"):
            assets.validate_bound_compiled_assets(snapshot, observed)
    with monkeypatch.context() as other:
        other.setattr(knobs.cache, "manager_class", foreign)
        with pytest.raises(UnverifiedModelStack, match="custom manager"):
            assets.validate_bound_compiled_assets(snapshot, observed)
    assert not calls
    cubin = Path(files[module._quant_query_per_thread_int8_i64_kernel.fn.__name__ + ".cubin"])
    cubin.write_bytes(b"tampered structural fixture")
    with pytest.raises(UnverifiedModelStack, match="file content changed"):
        assets.validate_bound_compiled_assets(snapshot, observed)
    saved = boundary(tmp_path / "boundaries", snapshot)
    with safe_open(str(saved["path"]), framework="pt", device="cpu") as handle:
        payload = json.loads(handle.metadata()[res.METADATA_KEY])
    payload["compiled_assets"]["sha256"] = "0" * 64
    bad = saved["path"].parent / "bad.h3res.safetensors"
    save_file(load_file(str(saved["path"])), str(bad), metadata={res.METADATA_KEY: json.dumps(payload)})
    with pytest.raises(UnverifiedModelStack, match="snapshot digest"):
        res.read_checkpoint(saved["path"].parent, bad.name)
    assert not torch.cuda.is_initialized()


def test_real_Core_setup_commits_boundary_observations_not_cold_contract_and_validates_before_resume(tmp_path, monkeypatch):
    from h3_audio_t8_pkg import res_history_setup as setup
    original = setup.loaded_model_identity
    calls = []
    producer = "a" * 64
    def capture(branch, *, observed_artifacts=None):
        calls.append(branch)
        if observed_artifacts is not None:
            observed_artifacts["data_only_integration_fixture"] = {
                "producer_sha256": producer, "observed_programs": [],
                "CUDA_kernel_launched_by_inspection": False}
        return original(branch, observed_artifacts=observed_artifacts)
    monkeypatch.setattr(setup, "loaded_model_identity", capture)
    bare, source = model(), source_latent(masked=True)
    checkpointed = setup.setup_res_history_sampling(bare, source,
        **options(tmp_path, mode="checkpoint", confirm_checkpoint_write=True))
    result = run(*checkpointed[:3], source)
    saved = res.read_checkpoint(tmp_path / "boundaries", "native.h3res.safetensors")
    assert len(calls) == 2 and "compiled_assets" not in saved["payload"]["run_contract"]
    assert saved["payload"]["compiled_assets"]["kernels"]["data_only_integration_fixture"]["producer_sha256"] == producer
    resumed = setup.setup_res_history_sampling(bare, source, **options(tmp_path, mode="resume"))
    restored = run(*resumed[:3], source)
    for left, right in zip(result, restored, strict=True):
        assert all(torch.equal(a, b) for a, b in zip(left["samples"].unbind(), right["samples"].unbind(), strict=True))
    producer = "b" * 64
    resumed = setup.setup_res_history_sampling(bare, source, **options(tmp_path, mode="resume"))
    with pytest.raises(UnverifiedModelStack, match="producer content changed"):
        run(*resumed[:3], source)
    assert res.read_checkpoint(tmp_path / "boundaries", "native.h3res.safetensors")["file_sha256"] == saved["file_sha256"]
    assert not torch.cuda.is_initialized()
