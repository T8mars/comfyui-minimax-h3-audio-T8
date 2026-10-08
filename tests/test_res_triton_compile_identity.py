"""Native hash/default parity and input tamper checks, not CUDA qualification."""
from copy import deepcopy
import importlib
import json
from pathlib import Path

import pytest
import torch
import triton.runtime.cache as native_cache

from h3_audio_t8_pkg.patch_stack_policy import UnverifiedModelStack
from h3_audio_t8_pkg.res_kj_sage_identity import _KJ_SOURCE
from h3_audio_t8_pkg.res_triton_identity import inspect_native_jit
from h3_audio_t8_pkg.res_triton_compile_identity import derive_key, runtime
from test_res_triton_cache_identity import structural_program
from test_res_triton_state_identity import snapshot
from test_res_kj_sage_identity import actual_source_chain  # noqa: F401 -- real source-backed native JIT

native_driver = importlib.import_module("triton.runtime.driver")


def test_native_compile_key_matches_actual_default_tuple_external_library_and_package_hash(actual_source_chain, tmp_path, monkeypatch):  # noqa: F811
    module, _, _ = actual_source_chain
    kernel = module._quant_query_per_thread_int8_i64_kernel
    cold = inspect_native_jit(kernel, module, _KJ_SOURCE)
    program, files, _ = structural_program(kernel, tmp_path)
    before, driver_before = snapshot(kernel), dict(vars(native_driver.driver))
    native_cache_before = native_cache.triton_key.cache_info()
    observed = {}
    assert inspect_native_jit(kernel, module, _KJ_SOURCE, observed_artifacts=observed) == cold
    artifact = observed[next(iter(observed))]["observed_programs"][0]
    assert artifact["compiler_key_independently_derived"]
    producer, root = runtime()
    assert producer["package_key"] == native_cache.triton_key()
    assert "Built on" in producer["ptxas"]["full_version_stdout"]
    assert artifact["actual_compile_key"]["compile_hash"] == program.hash
    assert type(artifact["actual_compile_key"]["options"]["extern_libs"]) is tuple
    assert type(artifact["actual_compile_key"]["options"]["supported_fp8_dtypes"]) is tuple
    assert snapshot(kernel) == before and vars(native_driver.driver) == driver_before
    # The sole additional native LRU hit above belongs to the explicit parity
    # comparison in this test, not to inspection.
    assert native_cache.triton_key.cache_info().misses == native_cache_before.misses
    assert program.module is None and program._run is None and not torch.cuda.is_initialized()
    with monkeypatch.context() as changed:
        changed.setattr(program, "hash", "0" * 64)
        with pytest.raises(UnverifiedModelStack, match="metadata source/target/hash differs"):
            inspect_native_jit(kernel, module, _KJ_SOURCE)
    json_path = Path(files[kernel.fn.__name__ + ".json"])
    original_bytes = json_path.read_bytes()
    metadata = json.loads(original_bytes)
    metadata["num_stages"] += 1
    json_path.write_text(json.dumps(metadata), encoding="utf8")
    with pytest.raises(UnverifiedModelStack, match="runtime metadata differs"):
        inspect_native_jit(kernel, module, _KJ_SOURCE)
    json_path.write_bytes(original_bytes)
    altered = deepcopy(producer)
    altered["ptxas"]["full_version_stdout"] = "12.8"
    assert derive_key(program.src, cold["native_parameter_cache_owners"]["native_dependency_hash_sha256"],
                      "{'num_warps': 4}", vars(program.metadata.target), altered, root)["compile_hash"] != program.hash


def test_foreign_environment_defaults_hooks_and_executable_option_literals_are_not_run(actual_source_chain, tmp_path, monkeypatch):  # noqa: F811
    module, _, _ = actual_source_chain
    kernel = module._quant_key_per_thread_int8_i64_kernel
    contract = inspect_native_jit(kernel, module, _KJ_SOURCE)
    program, _, _ = structural_program(kernel, tmp_path)
    producer, root = runtime()
    calls = []
    def foreign(*args, **kwargs):
        calls.append(True)
        raise AssertionError("unknown compiler owner must not execute during identity inspection")
    import triton.knobs as knobs
    import triton.backends.nvidia.compiler as cuda
    import triton.compiler.compiler as compiler
    for owner, name, value, message in (
            (knobs, "getenv", foreign, "environment reader has another native owner"),
            (cuda.CUDAOptions, "num_stages", property(foreign), "opaque executable value"),
            (type(knobs.compilation), "listener", property(foreign), "listener or pipeline stage owner"),
            (compiler, "get_cache_invalidating_env_vars", foreign, "environment reader has another native owner")):
        with monkeypatch.context() as changed:
            changed.setattr(owner, name, value)
            if name == "listener":
                # An instance None does not mask a foreign data descriptor in
                # native compilation. Inspect the class before trusting it.
                changed.setitem(vars(knobs.compilation), "listener", None)
            with pytest.raises(UnverifiedModelStack, match=message):
                inspect_native_jit(kernel, module, _KJ_SOURCE)
    with pytest.raises(UnverifiedModelStack, match="bounded literal data"):
        derive_key(program.src, contract["native_parameter_cache_owners"]["native_dependency_hash_sha256"],
                   "{'num_warps': __import__('os').getpid()}", vars(program.metadata.target), producer, root)
    from h3_audio_t8_pkg import res_triton_compile_identity as identity
    def unavailable(*args, **kwargs):
        raise identity.subprocess.TimeoutExpired(args[0], 5)
    with monkeypatch.context() as changed:
        changed.setattr(identity.subprocess, "run", unavailable)
        with pytest.raises(UnverifiedModelStack, match="version query unavailable"):
            identity._version.__wrapped__(str(root / "backends/nvidia/bin/ptxas.exe"), identity._PTXAS)
    assert not calls and program.module is None and program._run is None and not torch.cuda.is_initialized()
