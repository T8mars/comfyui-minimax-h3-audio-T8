"""Actual native cache/binder structure only; CPU fake bytes are NOT a cubin certificate."""
from copy import copy
import base64
import hashlib
import importlib
import json

import pytest
import torch
from triton.backends.compiler import GPUTarget
from triton.backends.nvidia.compiler import CUDABackend
from triton.compiler.compiler import ASTSource, CompiledKernel, compile, get_cache_invalidating_env_vars
from triton.runtime.cache import get_cache_key
import triton.runtime.jit as jit

from h3_audio_t8_pkg.patch_stack_policy import UnverifiedModelStack
from h3_audio_t8_pkg.res_kj_sage_identity import _KJ_SOURCE
from h3_audio_t8_pkg.res_triton_identity import inspect_native_jit
from test_res_triton_state_identity import snapshot
from test_res_kj_sage_identity import actual_source_chain  # noqa: F401 -- source-backed native JIT fixture


def warm_binder(kernel):
    """Only a private native binder factory; no create_binder/driver/compile call."""
    target = GPUTarget("cuda", 89, 32)
    backend = CUDABackend(target)
    parameters = []
    for original in kernel.params:
        private = copy(original)
        private.__dict__ = dict(vars(original))
        parameters.append(private)
    binder = jit.create_function_from_signature(kernel.signature, parameters, backend)
    kernel.CompiledKernel, kernel.ASTSource, kernel.compile = CompiledKernel, ASTSource, compile
    kernel.device_caches[0] = {}, {}, target, backend, binder
    return kernel.device_caches[0]


def structural_program(kernel, tmp_path):
    cached, key_cache, target, backend, _ = warm_binder(kernel)
    specialization = [(("constexpr", 64 if parameter.name == "C" else 128)
                       if parameter.is_constexpr else ("*fp32", "D")) for parameter in kernel.params]
    signature = {name: item[0] for name, item in zip(kernel.arg_names, specialization, strict=True)}
    constants = {(index,): item[1] for index, item in enumerate(specialization) if item[0] == "constexpr"}
    attrs = {(index,): backend.parse_attr(item[1]) for index, item in enumerate(specialization)
             if type(item[1]) is str}
    source = ASTSource(kernel, signature, constants, attrs)
    name = kernel.fn.__name__
    # Actual native pure hash/default construction, not the implementation being
    # tested. Mutating cache_key is accessed only on this private JIT copy.
    private = copy(kernel)
    private.__dict__ = dict(vars(kernel))
    private.hash, private.used_global_vals = None, {}
    private_source = ASTSource(private, signature, constants, attrs)
    options = backend.parse_options({"num_warps": 4})
    environment = get_cache_invalidating_env_vars()
    compiled_hash = hashlib.sha256(get_cache_key(private_source, backend, options, environment).encode()).hexdigest()
    metadata = dict(**vars(options), **environment, name=name, hash=compiled_hash, target=vars(target),
                    shared=0, triton_version="3.6.0", tensordesc_meta=[])
    directory = tmp_path / base64.b32encode(bytes.fromhex(compiled_hash)).decode().rstrip("=")
    directory.mkdir()
    files = {name + suffix: str(directory / (name + suffix)) for suffix in (".json", ".cubin", ".ttir")}
    for filename, path in files.items():
        from pathlib import Path
        payload = (json.dumps(metadata).encode() if filename.endswith(".json") else
                   b"CPU structure fixture, not CUDA binary" if filename.endswith(".cubin") else b"CPU IR fixture\r\n")
        Path(path).write_bytes(payload)
    program = CompiledKernel(source, files, metadata["hash"])
    key = jit.compute_cache_key(key_cache, specialization, {"num_warps": 4})
    cached[key] = program
    return program, files, key


def test_native_cold_and_binder_hot_identity_is_stable_and_never_populates_selected_cache(actual_source_chain):  # noqa: F811
    module, _, _ = actual_source_chain
    kernel = module._quant_query_per_thread_int8_i64_kernel
    before, observed = snapshot(kernel), {}
    cold = inspect_native_jit(kernel, module, _KJ_SOURCE, observed_artifacts=observed)
    assert snapshot(kernel) == before and not observed[next(iter(observed))]["observed_programs"]
    warm_binder(kernel)
    before = snapshot(kernel)
    hot = inspect_native_jit(kernel, module, _KJ_SOURCE, observed_artifacts=observed)
    assert hot == cold and snapshot(kernel) == before
    assert len(observed[next(iter(observed))]["observed_native_binders"]) == 1
    assert not hot["compiled_device_cache_content_qualified"] and not torch.cuda.is_initialized()


def test_native_compiled_structure_binds_specialization_memory_files_but_not_CUDA_qualification(actual_source_chain, tmp_path, monkeypatch):  # noqa: F811
    module, _, _ = actual_source_chain
    kernel = module._quant_key_per_thread_int8_i64_kernel
    cold = inspect_native_jit(kernel, module, _KJ_SOURCE)
    program, files, _ = structural_program(kernel, tmp_path)
    before, observed = snapshot(kernel), {}
    assert inspect_native_jit(kernel, module, _KJ_SOURCE, observed_artifacts=observed) == cold
    evidence = observed[next(iter(observed))]["observed_programs"][0]
    assert evidence["native_files_and_memory_content_consistent"]
    assert evidence["compiler_key_independently_derived"]
    assert evidence["actual_compile_key"]["compile_key_is_not_a_CUDA_execution_certificate"]
    assert not evidence["native_launcher_and_loaded_handle_qualified"]
    assert snapshot(kernel) == before and program.module is None and program.function is None and program._run is None
    with monkeypatch.context() as changed:
        changed.setattr(program.src, "constants", {(kernel.arg_names.index("C"),): 96})
        with pytest.raises(UnverifiedModelStack, match="constants/attributes differ"):
            inspect_native_jit(kernel, module, _KJ_SOURCE)
    with monkeypatch.context() as changed:
        changed.setattr(program, "kernel", b"different memory bytes")
        with pytest.raises(UnverifiedModelStack, match="binary differs"):
            inspect_native_jit(kernel, module, _KJ_SOURCE)
    class ForeignEquality:
        def __eq__(self, other):
            raise AssertionError("native artifacts must not execute foreign equality")
    with monkeypatch.context() as changed:
        changed.setattr(program.src, "name", ForeignEquality())
        with pytest.raises(UnverifiedModelStack, match="source does not belong"):
            inspect_native_jit(kernel, module, _KJ_SOURCE)
    with monkeypatch.context() as changed:
        changed.setitem(program.asm, "cubin", ForeignEquality())
        with pytest.raises(UnverifiedModelStack, match="binary differs"):
            inspect_native_jit(kernel, module, _KJ_SOURCE)
    from pathlib import Path
    cubin = Path(files[kernel.fn.__name__ + ".cubin"])
    cubin.write_bytes(b"different disk bytes")
    with pytest.raises(UnverifiedModelStack, match="binary differs"):
        inspect_native_jit(kernel, module, _KJ_SOURCE)
    assert program.module is None and program._run is None and not torch.cuda.is_initialized()


def test_foreign_binder_compiler_parser_and_async_program_are_never_invoked(actual_source_chain, monkeypatch):  # noqa: F811
    module, _, _ = actual_source_chain
    kernel = module._quant_query_per_thread_int8_i64_kernel
    inspect_native_jit(kernel, module, _KJ_SOURCE)
    row = warm_binder(kernel)
    calls = []
    def foreign(*args, **kwargs):
        calls.append(True)
        raise AssertionError("cache inspection must not invoke unknown execution")
    with monkeypatch.context() as changed:
        changed.setitem(kernel.device_caches, 0, (*row[:-1], foreign))
        with pytest.raises(UnverifiedModelStack, match="generated binder dependencies"):
            inspect_native_jit(kernel, module, _KJ_SOURCE)
    with monkeypatch.context() as changed:
        changed.setattr(jit, "create_function_from_signature", foreign)
        with pytest.raises(UnverifiedModelStack, match="live executable"):
            inspect_native_jit(kernel, module, _KJ_SOURCE)
    with monkeypatch.context() as changed:
        changed.setattr(CUDABackend, "parse_attr", staticmethod(foreign))
        with pytest.raises(UnverifiedModelStack, match="live executable"):
            inspect_native_jit(kernel, module, _KJ_SOURCE)
    class Future:
        def result(self):
            foreign()
    specialization = [("constexpr", 128) if parameter.is_constexpr else ("*fp32", "D") for parameter in kernel.params]
    key = jit.compute_cache_key(row[1], specialization, {})
    row[0][key] = Future()
    with pytest.raises(UnverifiedModelStack, match="opaque or asynchronous"):
        inspect_native_jit(kernel, module, _KJ_SOURCE)
    assert not calls and not torch.cuda.is_initialized()


def test_compiler_source_pin_and_specialization_keys_are_checked_not_only_recorded(actual_source_chain, monkeypatch):  # noqa: F811
    from h3_audio_t8_pkg import res_triton_cache_identity as cache_identity
    module, _, _ = actual_source_chain
    kernel = module._quant_query_per_thread_int8_i64_kernel
    inspect_native_jit(kernel, module, _KJ_SOURCE)
    with monkeypatch.context() as changed:
        changed.setitem(cache_identity._PINS, "triton.compiler.compiler", "0" * 64)
        with pytest.raises(UnverifiedModelStack, match="source requires a fresh audit"):
            inspect_native_jit(kernel, module, _KJ_SOURCE)
    row = warm_binder(kernel)
    specialization = [("constexpr", 128) if parameter.is_constexpr else ("*fp32", "D") for parameter in kernel.params]
    jit.compute_cache_key(row[1], specialization, {})
    key = next(iter(row[1]))
    row[1][key] = "wrong key"
    with pytest.raises(UnverifiedModelStack, match="specialization key differs"):
        inspect_native_jit(kernel, module, _KJ_SOURCE)
    native = importlib.import_module("triton._C.libtriton")
    with monkeypatch.context() as changed:
        changed.setattr(jit, "native_specialize_impl", lambda *args: None)
        with pytest.raises(UnverifiedModelStack, match="another native binary owner"):
            inspect_native_jit(kernel, module, _KJ_SOURCE)
    assert type(native.native_specialize_impl).__name__ == "builtin_function_or_method"
    assert not torch.cuda.is_initialized()
