"""RES-only inspection of native Triton binders and derived compiled artifacts.

No driver activation, compiler invocation, handle initialization, cache write or
kernel launch. Observed artifact hashes are separate from the stable producer
identity: checkpoint/recovery must additionally bind those actual artifacts.
"""
from copy import copy
from collections import defaultdict
import builtins
import hashlib
import importlib
from importlib.machinery import ExtensionFileLoader
import inspect
import io
import json
import math
from pathlib import Path
import sys
from types import BuiltinFunctionType, FunctionType

from .modular_sampling.progressive_effect_identity import _code_identity
from .patch_stack_policy import UnverifiedModelStack
from .res_kitchen_identity import _SourceAudit, _digest


_PINS = {
    "triton.runtime.jit": "0b70358d901afd70937a344a7811e4a2bcd10edca8a7c241e47842dc9ac660c8",
    "triton.compiler.compiler": "b6e1ab8e5c5c154b73d7a4c552411d7d830d6d73d02dcd15f30cb8ba6dca1017",
    "triton.backends.compiler": "c5d3403b88b5bb728f8ac78c55163198e418430f126c842892c68cb4b6baaa78",
    "triton.backends.nvidia.compiler": "f0754e1c22109e9fe774d6fd99d602341946c90385aba5e9cd4376e933140c06",
}
_NATIVE_BINARY = "8e4f083ab4af2a2dec0d8a2a7ed1fb9f1812e678fa688d7b8691bb55250e7108"


def _require(condition, message):
    if not condition:
        raise UnverifiedModelStack("RES Triton cache " + message)


def _plain(value, depth=0):
    _require(depth <= 12, "metadata nesting exceeds its bounded data contract")
    if type(value) in (str, bool, int, float, type(None)):
        if type(value) is str:
            _require(len(value) <= 131072, "metadata string exceeds its budget")
        if type(value) is float:
            _require(math.isfinite(value), "metadata contains a nonfinite value")
        return value
    if type(value) is dict:
        _require(len(value) <= 512 and all(type(key) is str for key in value), "metadata map has another owner")
        return {key: _plain(item, depth + 1) for key, item in sorted(value.items())}
    if type(value) in (tuple, list):
        _require(len(value) <= 512, "metadata sequence exceeds its budget")
        return [_plain(item, depth + 1) for item in value]
    raise UnverifiedModelStack("RES Triton cache metadata has an opaque executable value")


def _runtime():
    audit = _SourceAudit(_PINS)
    modules = {name: importlib.import_module(name) for name in _PINS}
    for name, module in modules.items():
        audit.module(module)
        # _SourceAudit's original source-pin gate is kitchen-specific. Never
        # mistake an extra recorded source SHA for verification of this epoch.
        _require(audit.sources[name] == _PINS[name], "runtime source requires a fresh audit: " + name)
    jit = modules["triton.runtime.jit"]
    compiler = modules["triton.compiler.compiler"]
    base = modules["triton.backends.compiler"]
    cuda = modules["triton.backends.nvidia.compiler"]
    for module, classes in ((compiler, (compiler.ASTSource, compiler.CompiledKernel, compiler.AsmDict)),
                            (base, (base.BaseBackend,)), (cuda, (cuda.CUDABackend,))):
        for cls in classes:
            for descriptor in vars(cls).values():
                value = descriptor.__func__ if type(descriptor) in (staticmethod, classmethod) else descriptor
                functions = ([value] if type(value) is FunctionType else
                             [part for part in (value.fget, value.fset) if part is not None]
                             if type(value) is property else [])
                for function in functions:
                    audit.function(function, module, dependencies=False)
    for name in ("create_function_from_signature", "compute_cache_key"):
        audit.function(getattr(jit, name), jit, dependencies=False)
    _require(compiler.GPUTarget is base.GPUTarget and compiler.Language is base.Language,
             "compiler target/language owners changed")
    native = importlib.import_module("triton._C.libtriton")
    _require(sys.modules.get(native.__name__) is native
             and type(native.__spec__.loader) is ExtensionFileLoader
             and type(jit.native_specialize_impl) is BuiltinFunctionType
             and jit.native_specialize_impl is native.native_specialize_impl
             and jit.native_specialize_impl.__self__ is native,
             "generated binder specialization has another native binary owner")
    _require(hashlib.sha256(_file(native.__file__, 128 * 1024**2)).hexdigest() == _NATIVE_BINARY,
             "native specialization binary requires a fresh audit")
    return audit, jit, compiler, base, cuda


def _target(value, base):
    _require(type(value) is base.GPUTarget, "compiled target has another owner")
    data = vars(value)
    _require(set(data) == {"backend", "arch", "warp_size"}
             and type(data["backend"]) is str and data["backend"] == "cuda" and type(data["arch"]) is int
             and data["arch"] == 89 and type(data["warp_size"]) is int and data["warp_size"] == 32,
             "compiled target is not the audited SM89 branch")
    return dict(data)


def _binder(kernel, backend, binder, jit, cuda):
    _require(type(backend) is cuda.CUDABackend and type(binder) is FunctionType,
             "binder or backend has another executable owner")
    _require(not (set(vars(backend)) & {"parse_options", "pack_metadata", "parse_attr", "hash"}),
             "backend has an instance selector override")
    namespace = binder.__globals__
    _require(namespace.get("backend") is backend and namespace.get("JITCallable") is jit.JITCallable
             and namespace.get("specialize_impl") is jit.native_specialize_impl,
             "generated binder dependencies changed")
    # Never fill native KernelParam cached properties on the executing kernel.
    private_params = []
    for parameter in kernel.params:
        private = copy(parameter)
        private.__dict__ = dict(vars(parameter))
        private_params.append(private)
    expected = jit.create_function_from_signature(kernel.signature, private_params, backend)
    _require(_code_identity(binder.__code__) == _code_identity(expected.__code__)
             and _plain(binder.__defaults__) == _plain(expected.__defaults__)
             and _plain(binder.__kwdefaults__) == _plain(expected.__kwdefaults__)
             and set(namespace) == set(expected.__globals__), "generated binder code/defaults/namespace changed")
    _require(namespace.get("dynamic_func") is binder and namespace.get("__builtins__") is vars(builtins),
             "generated binder namespace has another self/builtins owner")
    for name, value in namespace.items():
        if name not in ("__builtins__", "dynamic_func"):
            _require(value is expected.__globals__[name], "generated binder global value changed")
    return _digest(_code_identity(binder.__code__))


def _file(path, maximum):
    path = Path(path)
    _require(not path.is_symlink() and path.is_file(), "compiled artifact is not a regular native file")
    before = path.stat()
    _require(0 < before.st_size <= maximum, "compiled artifact exceeds its bounded file budget")
    data = path.read_bytes()
    after = path.stat()
    _require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
             and len(data) == before.st_size, "compiled artifact changed during inspection")
    return data


def _tuple_map(value, arguments):
    _require(type(value) is dict and len(value) <= len(arguments), "source constants/attributes have another owner")
    result = []
    for key, item in value.items():
        _require(type(key) is tuple and len(key) == 1 and type(key[0]) is int
                 and 0 <= key[0] < len(arguments), "source constant/attribute index changed")
        result.append([list(key), _plain(item)])
    return result


def _specialization(key, kernel, backend):
    specification = key[0]
    _require(len(specification) == len(kernel.arg_names), "specialization argument count changed")
    signature, constants, attrs = {}, {}, {}
    for index, item in enumerate(specification):
        _require(type(item) is tuple and len(item) == 2 and type(item[0]) is str,
                 "specialization has unsupported nested or opaque arguments")
        _plain(item)
        signature[kernel.arg_names[index]] = item[0]
        if item[0] == "constexpr":
            constants[(index,)] = item[1]
        if type(item[1]) is str:
            # This already-authenticated native static parser is pure CPU;
            # unlike JIT._pack_args it does not parse options or query a driver.
            attrs[(index,)] = backend.parse_attr(item[1])
    return signature, constants, attrs


def _program(program, kernel, target, compiler, base, expected_inputs, specialization_key, launcher_runtime,
             dependency_hash, option_repr, compile_runtime):
    _require(type(program) is compiler.CompiledKernel and type(program.src) is compiler.ASTSource,
             "compiled cache entry is opaque or asynchronous")
    _require(program.src.fn is kernel and type(program.src.name) is str
             and program.src.name == kernel.fn.__name__ and type(program.src.ext) is str
             and program.src.ext == "ttir" and program.src.language is base.Language.TRITON,
             "compiled source does not belong to its selected native JIT")
    signature = _plain(program.src.signature)
    _require(list(program.src.signature) == kernel.arg_names
             and all(type(value) is str for value in signature.values()), "compiled source signature changed")
    constants = _tuple_map(program.src.constants, kernel.arg_names)
    attrs = _tuple_map(program.src.attrs, kernel.arg_names)
    expected_signature, expected_constants, expected_attrs = expected_inputs
    _require(signature == expected_signature and constants == _tuple_map(expected_constants, kernel.arg_names)
             and attrs == _tuple_map(expected_attrs, kernel.arg_names),
             "compiled source signature/constants/attributes differ from the specialization")
    _require(type(program.metadata_group) is dict and 2 <= len(program.metadata_group) <= 16,
             "compiled metadata group has another owner or exceeds budget")
    groups = program.metadata_group
    name = program.src.name[:150]
    _require(name + ".json" in groups and name + ".cubin" in groups
             and all(type(key) is str and type(value) is str and Path(value).name == key
                     and key.startswith(name + ".") for key, value in groups.items()),
             "compiled artifact names do not match the actual source")
    paths = [Path(value).resolve() for value in groups.values()]
    _require(len({path.parent for path in paths}) == 1, "compiled group spans multiple cache directories")
    metadata = json.loads(_file(groups[name + ".json"], 1024**2))
    metadata = _plain(metadata)
    _require(type(metadata) is dict and metadata.get("target") == target
             and type(program.hash) is str and len(program.hash) == 64
             and all(char in "0123456789abcdef" for char in program.hash)
             and type(program.name) is str and metadata.get("hash") == program.hash
             and metadata.get("name") == program.name,
             "compiled metadata source/target/hash differs from its native program")
    fields = inspect.getattr_static(type(program.metadata), "_fields", None)
    _require(type(program.metadata).__bases__ == (tuple,) and type(fields) is tuple
             and all(type(field) is str for field in fields) and set(fields) == set(metadata),
             "compiled runtime metadata has another tuple owner")
    values = list(tuple.__iter__(program.metadata))
    _require(len(fields) == len(values), "compiled runtime metadata size changed")
    runtime = {field: (_target(value, base) if field == "target" else _plain(value))
               for field, value in zip(fields, values, strict=True)}
    _require(runtime == metadata, "compiled runtime metadata differs from the native file")
    from .res_triton_compile_identity import inspect_compile_key
    compile_key = inspect_compile_key(program, dependency_hash, option_repr, target, metadata, *compile_runtime)
    _require(_plain(program.packed_metadata) == [metadata["num_warps"], metadata["num_ctas"], metadata["shared"]],
             "compiled packed launch metadata changed")
    _require(type(program.asm) is compiler.AsmDict and type(program.kernel) is bytes,
             "compiled binary storage has another owner")
    binary = _file(groups[name + ".cubin"], 32 * 1024**2)
    memory_binary = dict.__getitem__(program.asm, "cubin")
    _require(type(memory_binary) is bytes and memory_binary == binary and program.kernel == binary,
             "compiled binary differs between memory and disk")
    artifacts = {}
    for filename, path in groups.items():
        if filename.endswith(".json"):
            data = _file(path, 1024**2)
        else:
            data = _file(path, 32 * 1024**2)
            suffix = filename.rsplit(".", 1)[-1]
            if suffix in program.asm:
                actual = dict.__getitem__(program.asm, suffix)
                # Native CompiledKernel uses Path.read_text(), including the
                # platform encoding and universal newline conversion. Raw disk
                # bytes stay independently SHA-bound below; do not rewrite IR.
                expected = data if suffix == "cubin" else io.TextIOWrapper(io.BytesIO(data)).read()
                _require(type(actual) is type(expected) and actual == expected,
                         "compiled IR differs between memory and disk")
        artifacts[filename] = {"path": str(Path(path).resolve()), "bytes": len(data),
                               "sha256": hashlib.sha256(data).hexdigest()}
    _require(not (set(vars(program)) & {"run", "launch_metadata", "_init_handles", "__getitem__"}),
             "compiled program has an execution instance override")
    from .res_triton_launcher_identity import inspect_launcher
    launcher = inspect_launcher(program, metadata, *launcher_runtime)
    # Never access program.run: its property initializes device handles. The
    # launcher/loaded handle still requires its separate actual CUDA certificate.
    return {"source_name": program.src.name, "compile_hash": program.hash,
            "target": target, "metadata_sha256": _digest(metadata),
            "specialization_key_sha256": hashlib.sha256(specialization_key.encode()).hexdigest(),
            "source_signature": signature, "source_constants": constants, "source_attributes": attrs,
            "compiler_key_independently_derived": True, "actual_compile_key": compile_key,
            "native_files_and_memory_content_consistent": True, "artifacts": artifacts,
            "actual_launcher": launcher,
            "native_launcher_and_loaded_handle_qualified": False}


def inspect_device_cache(kernel, dependency_hash):
    """Return stable producer evidence and separately observed actual artifacts.

    Caller must first authenticate the selected JIT's source/native state. This
    helper is not itself a portable recovery permit or CUDA numerical proof.
    """
    audit, jit, compiler, base, cuda = _runtime()
    from .res_triton_launcher_identity import runtime as launcher_runtime
    launcher_producer, launcher_driver, launcher_build = launcher_runtime()
    from .res_triton_compile_identity import runtime as compiler_runtime
    compiler_producer, compiler_root = compiler_runtime()
    _require(type(kernel) is jit.JITFunction, "selected JIT has another owner")
    # The native default-factory owner was authenticated by res_triton_identity.
    _require(type(kernel.device_caches) is defaultdict and len(kernel.device_caches) <= 8,
             "device cache has another owner or exceeds budget")
    programs, binders = [], []
    for device, entry in kernel.device_caches.items():
        _require(type(device) is int and 0 <= device <= 64 and type(entry) is tuple and len(entry) == 5,
                 "device cache row has another owner")
        cached, key_cache, target, backend, binder = entry
        target_data = _target(target, base)
        _require(type(backend) is cuda.CUDABackend and vars(backend).get("target") is target
                 and type(backend.binary_ext) is str and backend.binary_ext == "cubin",
                 "binder target/backend pair changed")
        binders.append(_binder(kernel, backend, binder, jit, cuda))
        _require(type(cached) is dict and type(key_cache) is dict
                 and len(cached) <= 128 and len(key_cache) <= 512, "specialization cache has another owner/budget")
        inputs = {}
        for key, value in key_cache.items():
            _require(type(key) is tuple and len(key) == 2 and type(key[0]) is tuple and type(key[1]) is str
                     and type(value) is str, "specialization key has an opaque value")
            _plain(key)
            _require(jit.compute_cache_key({}, list(key[0]), key[1]) == value,
                     "native specialization key differs from its actual data")
            source_inputs = _specialization(key, kernel, backend)
            _require(value not in inputs or inputs[value] == (source_inputs, key[1]),
                     "different specializations have a colliding cache key")
            inputs[value] = (source_inputs, key[1])
        _require(all(type(key) is str and key in key_cache.values() for key in cached),
                 "compiled cache has an unbound specialization key")
        for key, program in cached.items():
            source_inputs, option_repr = inputs[key]
            programs.append(_program(program, kernel, target_data, compiler, base, source_inputs, key,
                (launcher_driver, launcher_build), dependency_hash, option_repr, (compiler_producer, compiler_root)))
        for name, expected in (("CompiledKernel", compiler.CompiledKernel), ("compile", compiler.compile),
                               ("ASTSource", compiler.ASTSource)):
            _require(getattr(kernel, name, None) is expected, "native compile factory was replaced")
    producer = {"schema": "t8.minimax_h3.RES_Triton_derived_cache.v1",
                "sources": audit.sources, "executables": audit.functions,
                "native_specialization_binary_sha256": _NATIVE_BINARY,
                "launcher_producer": launcher_producer,
                "compiler_producer": compiler_producer,
                "inspection_writes_cache_or_initializes_driver": False,
                "compiled_device_cache_content_qualified": False,
                "provider_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    return producer, {"observed_native_binders": binders, "observed_programs": programs,
                      "CUDA_kernel_launched_by_inspection": False}
