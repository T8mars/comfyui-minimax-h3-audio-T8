"""RES-only identity of an actual source-backed Triton JIT, no launch."""
import hashlib
import inspect
import ast
from collections import defaultdict
from functools import cached_property
from pathlib import Path
import re
import sys
import textwrap
from types import FunctionType, MethodType

from .long_video_dual_identity import content_identity
from .modular_sampling.progressive_effect_identity import _code_identity
from .patch_stack_policy import UnverifiedModelStack
from .relay_kj_backend import _codes, _live_code_matches
from .res_kitchen_identity import _digest


_JIT_SOURCE = "0b70358d901afd70937a344a7811e4a2bcd10edca8a7c241e47842dc9ac660c8"


def _native_runtime(jit, require):
    """Authenticate class descriptors before reading any selected JIT property."""
    path = Path(jit.__file__).resolve()
    source = path.read_bytes()
    require(hashlib.sha256(source).hexdigest() == _JIT_SOURCE,
            "native runtime source requires a fresh cache-owner audit")
    compiled = tuple(_codes(compile(source, str(path), "exec", dont_inherit=True)))
    for cls in (jit.JITFunction, jit.JITCallable, jit.KernelInterface, jit.KernelParam, jit.DependenciesFinder):
        for value in vars(cls).values():
            functions = ([value] if type(value) is FunctionType else
                         [part for part in (value.fget, value.fset) if part is not None]
                         if type(value) is property else
                         [value.func] if type(value) is cached_property else [])
            for function in functions:
                require(_live_code_matches(function, compiled, jit),
                        "native runtime class executable changed")
    for name in ("_normalize_ty", "get_full_name"):
        require(_live_code_matches(getattr(jit, name), compiled, jit),
                "native parameter helper executable changed")
    require(jit.inspect is inspect and jit.ast is ast and jit.hashlib is hashlib
            and jit.re is re and jit.textwrap is textwrap and jit.defaultdict is defaultdict,
            "native runtime dependencies changed")


def _native_state(kernel, jit, require):
    """Authenticate parameters/default factory/hash without touching live caches.

    Only the already-audited kernels with native tl globals are eligible for a
    private CPU-only dependency hash. Compiled device artifacts are a separate
    pending gate, not certified by their container types or this cold inspection.
    """
    require(not (set(vars(kernel.fn)) & {"__signature__", "__wrapped__"}),
            "Python parameter signature has another owner")
    closure = inspect.getclosurevars(kernel.fn)
    require(not closure.nonlocals and set(closure.globals) <= {"tl", "triton"},
            "private dependency hash requires the audited language-only globals")
    require(type(kernel.fn.__annotations__) is dict and all(
        type(key) is str and (value is kernel.__globals__["tl"].constexpr
                             or (type(value) is str and value == "tl.constexpr"))
        for key, value in kernel.fn.__annotations__.items()), "parameter annotations have another owner")
    private = jit.JITFunction(kernel.fn, version=kernel.version,
        do_not_specialize=list(kernel.do_not_specialize),
        do_not_specialize_on_alignment=list(kernel.do_not_specialize_on_alignment),
        debug=kernel.debug, noinline=kernel.noinline)
    def literal_equal(actual, expected):
        return actual is expected or (type(actual) in (str, int, float, bool, type(None))
                                      and type(actual) is type(expected) and actual == expected)

    require(type(kernel.signature) is inspect.Signature
            and list(kernel.signature.parameters) == list(private.signature.parameters)
            and literal_equal(kernel.signature.return_annotation, private.signature.return_annotation),
            "actual parameter signature differs from the source-backed function")
    for name, original in kernel.signature.parameters.items():
        expected = private.signature.parameters[name]
        require(type(original) is inspect.Parameter and all(
            literal_equal(getattr(original, field), getattr(expected, field))
            for field in ("name", "kind", "annotation", "default")),
            "actual parameter signature differs from the source-backed function")
    require(type(kernel.params) is list and len(kernel.params) == len(private.params),
            "parameter metadata set changed")
    for current, expected in zip(kernel.params, private.params, strict=True):
        require(type(current) is jit.KernelParam, "parameter metadata has another owner")
        # Compute cached properties only on the private native object. Inspection
        # must not fill properties, hashes or device entries on the selected JIT.
        for name in ("name", "annotation", "annotation_type", "is_constexpr", "is_const"):
            getattr(expected, name)
        require(current._param is kernel.signature.parameters[expected.name]
                and set(vars(current)) <= set(vars(expected)) and all(
            name == "_param" or (type(value) is type(vars(expected)[name])
                                 and literal_equal(value, vars(expected)[name]))
            for name, value in vars(current).items()), "actual parameter metadata/cache changed")
    require(kernel.arg_names == private.arg_names and kernel.constexprs == private.constexprs,
            "actual argument or constexpr order changed")
    cache = kernel.device_caches
    factory = cache.default_factory if type(cache) is defaultdict else None
    require(type(cache) is defaultdict and type(factory) is MethodType
            and factory.__self__ is kernel and factory.__func__ is jit.JITFunction.create_binder,
            "device cache has another container/default-factory owner")
    require(type(kernel.used_global_vals) is dict, "used-global cache has another owner")
    expected_hash = private.cache_key
    require(kernel.hash is None or (type(kernel.hash) is str and kernel.hash == expected_hash),
            "native cached dependency hash changed")
    require(not private.used_global_vals and not kernel.used_global_vals,
            "audited language-only kernel has foreign cached global dependencies")
    require(all(literal_equal(getattr(kernel, name), getattr(private, name))
                for name in ("_fn_name", "module", "starting_line_number"))
            and type(kernel.raw_src) is list and len(kernel.raw_src) == len(private.raw_src)
            and all(type(line) is str and line == expected
                    for line, expected in zip(kernel.raw_src, private.raw_src, strict=True)),
            "native source-location/hash inputs changed")
    return {"native_parameter_and_default_factory_verified": True,
            "native_dependency_hash_sha256": expected_hash,
            "live_JIT_parameter_hash_or_device_cache_written": False}


def inspect_native_jit(kernel, module, source_sha256, *, observed_artifacts=None):
    """Bind source/default/compile settings; existing compiled device caches persist."""
    try:
        import triton
        import triton.runtime.jit as jit
        from triton.runtime.jit import JITFunction
        import triton.language as tl
    except ImportError as error:
        raise UnverifiedModelStack("RES Triton native compiler is unavailable") from error

    def require(condition, message):
        if not condition:
            raise UnverifiedModelStack("RES Triton " + message)

    _native_runtime(jit, require)
    require(type(kernel) is JITFunction and type(kernel.fn) is FunctionType, "selected quantizer is opaque")
    path = Path(module.__file__).resolve()
    source = path.read_bytes()
    require(hashlib.sha256(source).hexdigest() == source_sha256, "source requires a fresh quantizer audit")
    compiled = tuple(_codes(compile(source, str(path), "exec", dont_inherit=True)))
    require(sys.modules.get(module.__name__) is module
            and _live_code_matches(kernel.fn, compiled, module)
            and kernel.__globals__ is vars(module), "live source/code/global owner changed")
    require(module.triton is triton and module.tl is tl, "compiler/language dependencies changed")
    actual = textwrap.dedent(inspect.getsource(kernel.fn))
    match = re.search(r"^def\s+\w+\s*\(", actual, re.MULTILINE)
    require(match is not None and kernel.src == actual[match.start():], "JIT source differs from its actual Python definition")
    require(type(kernel.pre_run_hooks) is list and not kernel.pre_run_hooks
            and kernel.launch_metadata is None and kernel._repr is None,
            "JIT has another launch hook/metadata owner")
    require(not (set(vars(kernel)) & {"run", "__getitem__", "create_binder"}), "JIT execution method has been replaced")
    settings = {name: getattr(kernel, name) for name in ("version", "debug", "noinline",
        "do_not_specialize", "do_not_specialize_on_alignment", "arg_names", "constexprs")}
    for name in ("do_not_specialize", "do_not_specialize_on_alignment", "arg_names", "constexprs"):
        require(type(settings[name]) is list, "JIT compile parameters have foreign containers")
    require(all(type(value) in (str, int) for name in (
        "do_not_specialize", "do_not_specialize_on_alignment") for value in settings[name]),
        "JIT specialization parameters have opaque values")
    require(all(type(value) is str for value in settings["arg_names"])
            and all(type(value) is int for value in settings["constexprs"]),
            "JIT argument and constexpr metadata has opaque values")
    require(type(settings["debug"]) in (bool, type(None))
            and type(settings["noinline"]) in (bool, type(None))
            and type(settings["version"]) in (str, int, type(None)), "JIT compile settings are opaque")
    state = _native_state(kernel, jit, require)
    from .res_triton_cache_identity import inspect_device_cache
    producer, observed = inspect_device_cache(kernel, state["native_dependency_hash_sha256"])
    identity = {"kind": "source_backed_native_Triton_JIT", "source_sha256": source_sha256,
            "code_sha256": _digest(_code_identity(kernel.fn.__code__)),
            "jit_source_sha256": hashlib.sha256(kernel.src.encode()).hexdigest(),
            "defaults": content_identity(kernel.fn.__defaults__),
            "compile_settings": content_identity(settings), "triton_version": triton.__version__,
            "compiler_jit_source_sha256": hashlib.sha256(Path(inspect.getfile(JITFunction)).read_bytes()).hexdigest(),
            "provider_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "native_parameter_cache_owners": state,
            "derived_cache_inspector": producer,
            "compiled_device_cache_content_qualified": False, "CUDA_kernel_launched": False}
    if observed_artifacts is not None:
        require(type(observed_artifacts) is dict, "artifact collector has another owner")
        observed_artifacts[module.__name__ + ":" + kernel.fn.__name__] = {
            **observed, "producer_sha256": _digest(identity)}
    return identity
