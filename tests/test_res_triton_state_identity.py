"""Native JIT parameter/cache-owner inspection, never a device kernel launch."""
from collections import defaultdict
from copy import copy

import pytest
import torch
from triton.runtime.jit import JITCallable, JITFunction

from h3_audio_t8_pkg.patch_stack_policy import UnverifiedModelStack
from h3_audio_t8_pkg.res_kj_sage_identity import _KJ_SOURCE
from h3_audio_t8_pkg.res_triton_identity import inspect_native_jit
from test_res_kj_sage_identity import actual_source_chain  # noqa: F401 -- source-backed native JIT fixture


def snapshot(kernel):
    return (kernel.hash, dict(kernel.used_global_vals), dict(kernel.device_caches),
            kernel.device_caches.default_factory, [dict(vars(param)) for param in kernel.params])


def test_actual_native_parameter_and_dependency_hash_cold_hot_do_not_write_selected_JIT(actual_source_chain):  # noqa: F811
    module, _, _ = actual_source_chain
    for kernel in (module._quant_query_per_thread_int8_i64_kernel, module._quant_key_per_thread_int8_i64_kernel):
        before = snapshot(kernel)
        identity = inspect_native_jit(kernel, module, _KJ_SOURCE)
        assert snapshot(kernel) == before
        assert identity["native_parameter_cache_owners"]["native_parameter_and_default_factory_verified"]
        assert not identity["compiled_device_cache_content_qualified"]
        # This native property warms only a CPU hash, not the device compiler.
        # Normal cold/hot source hashing must not change the content identity.
        assert kernel.cache_key == identity["native_parameter_cache_owners"]["native_dependency_hash_sha256"]
        warm = snapshot(kernel)
        assert inspect_native_jit(kernel, module, _KJ_SOURCE) == identity
        assert snapshot(kernel) == warm and not torch.cuda.is_initialized()


def test_wrong_actual_parameter_signature_metadata_and_cached_hash_are_not_authenticated(actual_source_chain, monkeypatch):  # noqa: F811
    module, _, _ = actual_source_chain
    kernel = module._quant_query_per_thread_int8_i64_kernel
    assert inspect_native_jit(kernel, module, _KJ_SOURCE)["native_parameter_cache_owners"]["native_parameter_and_default_factory_verified"]
    for name, value, reason in (("constexprs", [0], "constexpr order"), ("hash", "wrong", "dependency hash"),
            ("signature", kernel.signature.replace(parameters=[]), "parameter signature")):
        with monkeypatch.context() as changed:
            changed.setattr(kernel, name, value)
            with pytest.raises(UnverifiedModelStack, match=reason):
                inspect_native_jit(kernel, module, _KJ_SOURCE)
            assert getattr(kernel, name) is value
    for name, value in (("num", 9), ("name", "another_argument"), ("is_constexpr", True)):
        with monkeypatch.context() as changed:
            parameter = kernel.params[0]
            changed.setattr(parameter, name, value, raising=False)
            with pytest.raises(UnverifiedModelStack, match="parameter metadata/cache"):
                inspect_native_jit(kernel, module, _KJ_SOURCE)
            assert getattr(parameter, name) is value
    with monkeypatch.context() as changed:
        parameters = list(kernel.params)
        parameters[0] = copy(parameters[0])
        parameters[0].do_not_specialize = True
        changed.setattr(kernel, "params", parameters)
        with pytest.raises(UnverifiedModelStack, match="parameter metadata/cache"):
            inspect_native_jit(kernel, module, _KJ_SOURCE)
    assert not torch.cuda.is_initialized()


def test_foreign_factory_runtime_and_global_cache_remain_uninvoked(actual_source_chain, monkeypatch):  # noqa: F811
    module, _, _ = actual_source_chain
    kernel = module._quant_key_per_thread_int8_i64_kernel
    assert inspect_native_jit(kernel, module, _KJ_SOURCE)["native_parameter_cache_owners"]["native_parameter_and_default_factory_verified"]
    calls = []
    def foreign(*args, **kwargs):
        calls.append(True)
        raise AssertionError("inspection must not invoke foreign compiler/cache owners")
    class ForeignEquality:
        def __eq__(self, other):
            foreign()

    with monkeypatch.context() as changed:
        changed.setattr(kernel.fn, "__annotations__", {"C": ForeignEquality()})
        with pytest.raises(UnverifiedModelStack, match="parameter annotations"):
            inspect_native_jit(kernel, module, _KJ_SOURCE)
    with monkeypatch.context() as changed:
        changed.setattr(kernel.params[0], "num", ForeignEquality())
        with pytest.raises(UnverifiedModelStack, match="parameter metadata/cache"):
            inspect_native_jit(kernel, module, _KJ_SOURCE)
    with monkeypatch.context() as changed:
        cache = defaultdict(foreign)
        changed.setattr(kernel, "device_caches", cache)
        with pytest.raises(UnverifiedModelStack, match="default-factory owner"):
            inspect_native_jit(kernel, module, _KJ_SOURCE)
        assert kernel.device_caches is cache and not cache
    with monkeypatch.context() as changed:
        changed.setattr(JITFunction, "create_binder", foreign)
        with pytest.raises(UnverifiedModelStack, match="runtime class executable"):
            inspect_native_jit(kernel, module, _KJ_SOURCE)
        assert JITFunction.create_binder is foreign
    with monkeypatch.context() as changed:
        changed.setattr(JITCallable, "src", property(foreign))
        with pytest.raises(UnverifiedModelStack, match="runtime class executable"):
            inspect_native_jit(kernel, module, _KJ_SOURCE)
    for name, value, reason in (("arg_names", [ForeignEquality()], "metadata has opaque"),
            ("_fn_name", ForeignEquality(), "source-location/hash"),
            ("pre_run_hooks", ForeignEquality(), "another launch hook")):
        with monkeypatch.context() as changed:
            changed.setattr(kernel, name, value)
            with pytest.raises(UnverifiedModelStack, match=reason):
                inspect_native_jit(kernel, module, _KJ_SOURCE)
            assert getattr(kernel, name) is value
    with monkeypatch.context() as changed:
        changed.setattr(kernel, "used_global_vals", {("opaque", 123): (object(), {})})
        with pytest.raises(UnverifiedModelStack, match="foreign cached global"):
            inspect_native_jit(kernel, module, _KJ_SOURCE)
        assert kernel.used_global_vals
    assert not calls and not torch.cuda.is_initialized()
