"""Native C-source generation and unknown-owner refusals; no GPU/extension load."""
import hashlib

import pytest
import torch

from h3_audio_t8_pkg.patch_stack_policy import UnverifiedModelStack
from h3_audio_t8_pkg.res_kj_sage_identity import _KJ_SOURCE
from h3_audio_t8_pkg.res_triton_identity import inspect_native_jit
from h3_audio_t8_pkg.res_triton_launcher_identity import generated_source, runtime
from test_res_triton_cache_identity import structural_program
from test_res_triton_state_identity import snapshot
from test_res_kj_sage_identity import actual_source_chain  # noqa: F401 -- actual source/native JIT fixture


def test_actual_generated_C_source_is_bound_without_compile_or_loading_handles(actual_source_chain, tmp_path):  # noqa: F811
    module, _, _ = actual_source_chain
    kernel = module._quant_query_per_thread_int8_i64_kernel
    cold = inspect_native_jit(kernel, module, _KJ_SOURCE)
    program, _, _ = structural_program(kernel, tmp_path)
    before, observations = snapshot(kernel), {}
    assert inspect_native_jit(kernel, module, _KJ_SOURCE, observed_artifacts=observations) == cold
    artifact = observations[next(iter(observations))]["observed_programs"][0]
    producer, driver, _ = runtime()
    c_source = generated_source(program.src, {}, driver)
    assert "PyInit___triton_launcher" in c_source and "cuLaunchKernelEx" in c_source
    assert hashlib.sha256(c_source.encode()).hexdigest() == artifact["actual_launcher"]["generated_C_sha256"]
    assert not artifact["actual_launcher"]["native_launcher_extension_content_bound"]
    assert not artifact["actual_launcher"]["loaded_GPU_handle_types_observed"]
    assert not producer["driver_compiler_or_launcher_called"]
    assert snapshot(kernel) == before and program.module is None and program._run is None
    assert not torch.cuda.is_initialized()


def test_foreign_C_generator_launcher_global_hook_and_native_export_are_not_invoked(actual_source_chain, tmp_path, monkeypatch):  # noqa: F811
    module, _, _ = actual_source_chain
    kernel = module._quant_key_per_thread_int8_i64_kernel
    inspect_native_jit(kernel, module, _KJ_SOURCE)
    program, _, _ = structural_program(kernel, tmp_path)
    _, driver, _ = runtime()
    import triton.knobs as knobs
    calls = []
    def foreign(*args, **kwargs):
        calls.append(True)
        raise AssertionError("inspection must not execute foreign launcher code")
    with monkeypatch.context() as changed:
        changed.setattr(driver, "make_launcher", foreign)
        with pytest.raises(UnverifiedModelStack, match="live executable"):
            inspect_native_jit(kernel, module, _KJ_SOURCE)
    with monkeypatch.context() as changed:
        changed.setattr(driver.CudaLauncher, "__call__", foreign)
        with pytest.raises(UnverifiedModelStack, match="live executable"):
            inspect_native_jit(kernel, module, _KJ_SOURCE)
    with monkeypatch.context() as changed:
        changed.setattr(driver.CudaLauncher, "launch", property(foreign), raising=False)
        with pytest.raises(UnverifiedModelStack, match="live executable"):
            inspect_native_jit(kernel, module, _KJ_SOURCE)
    with monkeypatch.context() as changed:
        changed.setattr(knobs.HookChain, "calls", property(foreign), raising=False)
        with pytest.raises(UnverifiedModelStack, match="live executable"):
            inspect_native_jit(kernel, module, _KJ_SOURCE)
    with monkeypatch.context() as changed:
        changed.setattr(knobs.runtime.launch_enter_hook, "calls", [foreign])
        with pytest.raises(UnverifiedModelStack, match="additional global execution hook"):
            inspect_native_jit(kernel, module, _KJ_SOURCE)
    with monkeypatch.context() as changed:
        changed.setitem(driver.FLOAT_STORAGE_TYPE, "fp32", "wrong")
        with pytest.raises(UnverifiedModelStack, match="literal dependency"):
            inspect_native_jit(kernel, module, _KJ_SOURCE)
    with monkeypatch.context() as changed:
        changed.setattr(program, "_run", foreign)
        with pytest.raises(UnverifiedModelStack, match="opaque execution owner"):
            inspect_native_jit(kernel, module, _KJ_SOURCE)
    # Constructor-free structure is only a negative fixture, never presented as
    # an actual launched GPU program or trusted binary producer.
    launcher = object.__new__(driver.CudaLauncher)
    expected = dict(num_ctas=1, global_scratch_size=0, global_scratch_align=1,
                    profile_scratch_size=0, profile_scratch_align=1,
                    launch_cooperative_grid=False, launch_pdl=False)
    launcher.__dict__.update(expected, launch=foreign)
    with monkeypatch.context() as changed:
        changed.setattr(program, "_run", launcher)
        # Missing original warm-launcher fields must not be guessed or accepted.
        with pytest.raises(UnverifiedModelStack, match="launcher metadata differs"):
            inspect_native_jit(kernel, module, _KJ_SOURCE)
    assert not calls and program.module is None and program._run is None and not torch.cuda.is_initialized()
