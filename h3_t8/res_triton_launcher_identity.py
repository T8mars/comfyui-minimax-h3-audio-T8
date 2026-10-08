"""Read-only native CUDA launcher content for RES, without compiling/loading it.

Generating the authenticated C source is CPU string formatting, not compiling
or launching it. Already-loaded extensions are inspected, never imported from
an artifact path. Handle types are observations, NOT numerical qualification.
"""
import ast
import base64
import functools
import hashlib
import importlib
from importlib.machinery import ExtensionFileLoader, ModuleSpec
from pathlib import Path
import re
from types import BuiltinFunctionType, FunctionType, ModuleType

from .res_kitchen_identity import _SourceAudit
from .res_triton_cache_identity import _file, _plain, _require


_PINS = {
    "triton.backends.nvidia.driver": "e7e76aa1ec22f8974ae1f518819cb12787d4ed42bf6907772fd6e87c2e4d15cd",
    "triton.runtime.build": "8e1d02742d887b110638ae73cf13bcebec1f76ea75c138acc022182eb3abd9dd",
    "triton.runtime.cache": "03f1234ed00cb98dfd2409974ca0f65edc427a248456c1815c000a0c7d111b09",
    "triton.knobs": "53776bdbeb3c06da277b4fac1ae6ffacdc3ddea5a9eefc2a4987ca7ad426dac8",
}


def runtime():
    audit = _SourceAudit(_PINS)
    modules = {name: importlib.import_module(name) for name in _PINS}
    for name, module in modules.items():
        audit.module(module)
        _require(audit.sources[name] == _PINS[name], "launcher source requires a fresh audit: " + name)
    driver, build, cache, knobs = (modules[name] for name in _PINS)
    for name in ("make_launcher", "ty_to_cpp", "wrap_handle_tensordesc"):
        audit.function(getattr(driver, name), driver, dependencies=False)
    for cls in (driver.CudaLauncher, knobs.HookChain):
        module = driver if cls is driver.CudaLauncher else knobs
        for descriptor in vars(cls).values():
            functions = ([descriptor] if type(descriptor) is FunctionType else
                         [part for part in (descriptor.fget, descriptor.fset) if part is not None]
                         if type(descriptor) is property else [])
            for function in functions:
                audit.function(function, module, dependencies=False)
    _require(set(vars(driver.CudaLauncher)) <= {"__module__", "__init__", "__call__",
                 "__dict__", "__weakref__", "__doc__"}, "launcher class has another descriptor owner")
    for name in ("compile_module_from_src", "_load_module_from_path"):
        audit.function(getattr(build, name), build, dependencies=False)
    _require(type(build.platform_key) is functools._lru_cache_wrapper,
             "launcher platform key has another cache owner")
    audit.function(build.platform_key.__wrapped__, build, dependencies=False)
    audit.function(cache._base32, cache, dependencies=False)
    _require(driver.compile_module_from_src is build.compile_module_from_src
             and driver.re is re and cache.base64 is base64 and build.hashlib is hashlib,
             "launcher calculation dependencies changed")
    source = Path(driver.__file__).read_bytes()
    literals = {}
    for node in ast.parse(source).body:
        if type(node) is ast.Assign and len(node.targets) == 1 and type(node.targets[0]) is ast.Name:
            name = node.targets[0].id
            if name in ("FLOAT_STORAGE_TYPE", "FLOAT_PACK_FUNCTION", "_BASE_ARGS_FORMAT"):
                literals[name] = ast.literal_eval(node.value)
    for name, expected in literals.items():
        _require(_plain(getattr(driver, name)) == expected, "launcher literal dependency changed: " + name)
    _require(type(driver._BASE_ARGS_FORMAT_LEN) is int
             and driver._BASE_ARGS_FORMAT_LEN == len(literals["_BASE_ARGS_FORMAT"]),
             "launcher argument-format length changed")
    hooks = {}
    for name in ("launch_enter_hook", "launch_exit_hook", "kernel_load_start_hook", "kernel_load_end_hook"):
        hook = getattr(knobs.runtime, name)
        _require(type(hook) is knobs.HookChain and set(vars(hook)) == {"calls", "reversed"}
                 and type(hook.calls) is list and not hook.calls and type(hook.reversed) is bool,
                 "launcher has an additional global execution hook: " + name)
        hooks[name] = dict(calls=[], reversed=hook.reversed)
    producer = dict(schema="t8.minimax_h3.RES_Triton_launcher_content.v1", sources=audit.sources,
                    executables=audit.functions, literals=literals, empty_runtime_hooks=hooks,
                    provider_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    driver_compiler_or_launcher_called=False)
    return producer, driver, build


def generated_source(source, metadata, driver):
    """Selected source constants/signature were type-checked by the cache reader."""
    _require(all(type(value) is str and not value.startswith("tensordesc")
                 and value != "nvTmaDesc" for value in source.signature.values()),
             "launcher tensor-descriptor/nested signature is not qualified")
    descriptor_metadata = metadata.get("tensordesc_meta")
    _require(descriptor_metadata is None or (type(descriptor_metadata) is list and not descriptor_metadata),
             "launcher tensor-descriptor metadata is not qualified")
    result = driver.make_launcher(dict(source.constants), dict(source.signature), descriptor_metadata)
    _require(type(result) is str and len(result.encode()) <= 1024**2, "generated launcher source exceeds budget")
    return result


def extension_content(function, source_text, expected_name, build):
    """Inspect only an already-loaded native function and its selected file."""
    _require(type(function) is BuiltinFunctionType and function.__name__ == "launch",
             "launcher call is not the original native export")
    module = function.__self__
    _require(type(module) is ModuleType and module.__name__ == expected_name
             and vars(module).get("launch") is function and type(module.__spec__) is ModuleSpec
             and type(module.__spec__.loader) is ExtensionFileLoader,
             "launcher export has another extension owner")
    _require(all(type(value) is str for value in (module.__file__, module.__spec__.origin,
                                                module.__spec__.loader.path)),
             "launcher extension file declaration has another owner")
    path = Path(module.__file__).resolve()
    _require(Path(module.__spec__.origin).resolve() == path
             and Path(module.__spec__.loader.path).resolve() == path
             and path.name.startswith(expected_name) and path.suffix == ".pyd",
             "launcher extension file/loader differs")
    # Call the authenticated uncached native platform helper, not its mutable
    # process LRU. On this Windows branch it only reads standard platform data.
    platform_key = build.platform_key.__wrapped__()
    _require(type(platform_key) is str and len(platform_key) <= 1024,
             "launcher platform data is not bounded")
    cache_key = hashlib.sha256((source_text + platform_key).encode()).hexdigest()
    directory_key = base64.b32encode(bytes.fromhex(cache_key)).decode().rstrip("=")
    _require(path.parent.name == directory_key, "loaded launcher does not belong to its generated source cache key")
    binary = _file(path, 32 * 1024**2)
    return dict(path=str(path), bytes=len(binary), sha256=hashlib.sha256(binary).hexdigest(),
                native_export="launch", generated_C_sha256=hashlib.sha256(source_text.encode()).hexdigest(),
                native_build_cache_key=cache_key, native_platform_key=platform_key)


def inspect_launcher(program, metadata, driver, build):
    c_source = generated_source(program.src, metadata, driver)
    report = dict(generated_C_sha256=hashlib.sha256(c_source.encode()).hexdigest(),
                  native_launcher_extension_content_bound=False,
                  loaded_GPU_handle_types_observed=False, CUDA_numerical_execution_qualified=False)
    launcher = vars(program).get("_run")
    if launcher is None:
        _require(vars(program).get("module") is None and vars(program).get("function") is None,
                 "compiled program has handles without a native launcher")
        return report
    _require(type(launcher) is driver.CudaLauncher, "loaded launcher has an opaque execution owner")
    expected_fields = {"launch", "num_ctas", "global_scratch_size", "global_scratch_align",
                       "profile_scratch_size", "profile_scratch_align", "launch_cooperative_grid", "launch_pdl"}
    _require(set(vars(launcher)) == expected_fields, "loaded launcher fields or invocation owner changed")
    for name in expected_fields - {"launch"}:
        value = vars(launcher)[name]
        _require(type(value) in (int, bool) and type(value) is type(metadata.get(name))
                 and value == metadata[name], "loaded launcher metadata differs: " + name)
    extension = extension_content(launcher.launch, c_source, "__triton_launcher", build)
    _require(all(type(vars(program).get(name)) is int and vars(program)[name] > 0
                 for name in ("module", "function")), "loaded native kernel handles are missing")
    report.update(native_launcher_extension_content_bound=True, extension=extension,
                  loaded_GPU_handle_types_observed=True)
    return report
