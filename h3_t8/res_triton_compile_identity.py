"""Derive the installed SM89 Triton compile key without compiling or a driver.

The source hash uses the authenticated private JIT dependency hash, never the
selected JIT's mutating cache_key property. Native tuple/default/extern-library
semantics are preserved; JSON metadata alone is not a compile-key certificate.
"""
import ast
import base64
from dataclasses import Field
import functools
import hashlib
import importlib
import inspect
from pathlib import Path
import subprocess
from types import BuiltinFunctionType

from .res_kitchen_identity import _SourceAudit
from .patch_stack_policy import UnverifiedModelStack
from .res_triton_cache_identity import _file, _plain, _require, _NATIVE_BINARY, _PINS as _COMPILER_PINS


_PINS = {**_COMPILER_PINS,
    "triton.runtime.cache": "03f1234ed00cb98dfd2409974ca0f65edc427a248456c1815c000a0c7d111b09",
    "triton.knobs": "53776bdbeb3c06da277b4fac1ae6ffacdc3ddea5a9eefc2a4987ca7ad426dac8"}
_PTXAS = "7f6f04d25e5a7ab2c07c15e4ef26a07cec08104ebfebf364331fafa8f3c443d6"
_LIBDEVICE = "5c2fae37c86e68c3a38605a95f512d7d12d5f3db986310be47f57304aa72a5ee"
# Exact pkgutil.walk_packages order for the installed in-tree Python modules.
# Do not call walk_packages: it imports package initializers during inspection.
_PACKAGE_PATHS = (
    "compiler/code_generator.py", "compiler/compiler.py", "compiler/errors.py", "compiler/make_launcher.py",
    "backends/amd/__init__.py", "backends/amd/compiler.py", "backends/amd/driver.py",
    "backends/compiler.py", "backends/driver.py", "backends/nvidia/__init__.py",
    "backends/nvidia/compiler.py", "backends/nvidia/driver.py",
    "language/core.py", "language/extra/__init__.py", "language/extra/cuda/__init__.py",
    "language/extra/cuda/gdc.py", "language/extra/cuda/libdevice.py", "language/extra/cuda/utils.py",
    "language/extra/hip/__init__.py", "language/extra/hip/libdevice.py", "language/extra/hip/utils.py",
    "language/extra/libdevice.py", "language/math.py", "language/random.py", "language/semantic.py",
    "language/standard.py", "language/target_info.py")


def _knob(knobs, group_name, name, key, default=None, *, boolean=False):
    group = vars(knobs)[group_name]
    _require(type(group) is vars(knobs)[group_name + "_knobs"], "compile knob group has another owner")
    descriptor = inspect.getattr_static(type(group), name)
    expected_class = knobs.env_bool if boolean else knobs.env_opt_str
    _require(type(descriptor) is expected_class and vars(descriptor) ==
             {"key": key, "name": name, **({"default": default} if boolean else {})},
             "compile knob descriptor changed: " + name)
    if name in vars(group):
        value = vars(group)[name]
    else:
        value = knobs.getenv_bool(key, default) if boolean else knobs.getenv(key)
    _require(type(value) is bool if boolean else type(value) in (str, type(None)),
             "compile knob value has an opaque owner: " + name)
    return value


@functools.lru_cache(maxsize=4)
def _version(path, content_sha256):
    """Only the content-pinned bundled executable's read-only --version query."""
    _require(content_sha256 == _PTXAS, "ptxas executable requires a fresh audit")
    try:
        result = subprocess.run([path, "--version"], check=True, capture_output=True, timeout=5,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        value = result.stdout.decode("utf8")
    except (OSError, subprocess.SubprocessError, UnicodeError) as error:
        raise UnverifiedModelStack("RES native ptxas version query unavailable; ordinary sampling remains unqualified") from error
    _require(0 < len(value) <= 8192 and "release 12.8, V12.8.93" in value,
             "ptxas full version differs from the installed source epoch")
    return value


def _package_key(root, version):
    _require(type(version) is str and version == "3.6.0", "native compiler version requires a fresh audit")
    excluded = {"compiler/__init__.py", "backends/__init__.py", "language/__init__.py"}
    actual = {path.relative_to(root).as_posix() for directory in ("compiler", "backends", "language")
              for path in (root / directory).rglob("*.py")}
    _require(actual == set(_PACKAGE_PATHS) | excluded, "native compiler package module set changed")
    paths = ("runtime/cache.py", *_PACKAGE_PATHS[:12], "_C/libtriton.pyd", *_PACKAGE_PATHS[12:])
    files = {}
    for name in paths:
        path = root / name
        if name.endswith(".py"):
            # Empty package initializers are legitimate source inputs to native
            # triton_key. Keep the compiled IR/binary nonempty guard unchanged.
            _require(path.is_file() and not path.is_symlink(), "package source is not a regular file")
            before = path.stat()
            _require(before.st_size <= 1024**2, "package source exceeds its bounded budget")
            data = path.read_bytes()
            after = path.stat()
            _require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
                     and len(data) == before.st_size, "package source changed during inspection")
        else:
            data = _file(path, 128 * 1024**2)
        files[name] = hashlib.sha256(data).hexdigest()
    _require(files["_C/libtriton.pyd"] == _NATIVE_BINARY, "native compiler binary requires a fresh audit")
    return version + "-".join(files.values()), files


def runtime():
    audit = _SourceAudit(_PINS)
    modules = {name: importlib.import_module(name) for name in _PINS}
    for name, module in modules.items():
        audit.module(module)
        _require(audit.sources[name] == _PINS[name], "compile-key source requires a fresh audit: " + name)
    cuda, cache, knobs = (modules[name] for name in
                         ("triton.backends.nvidia.compiler", "triton.runtime.cache", "triton.knobs"))
    for function, module in ((cuda.CUDAOptions.__post_init__, cuda), (cuda.CUDAOptions.hash, cuda),
            (cuda.CUDABackend.parse_options, cuda), (cuda.CUDABackend._parse_arch, cuda),
            (cuda.get_ptxas, cuda), (cache.get_cache_key, cache),
            (knobs.env_base.__get__, knobs), (knobs.env_base.transform, knobs),
            (knobs.env_opt_str.get, knobs), (knobs.env_bool.get, knobs)):
        audit.function(function, module, dependencies=False)
    for wrapped, module in ((cache.triton_key, cache), (cuda.get_ptxas_version, cuda), (cuda.CUDABackend.hash, cuda)):
        _require(type(wrapped) is functools._lru_cache_wrapper, "compiler hash has another cache owner")
        audit.function(wrapped.__wrapped__, module, dependencies=False)
    native = importlib.import_module("triton._C.libtriton")
    compiler = modules["triton.compiler.compiler"]
    for function, name in ((knobs.getenv, "getenv"), (knobs.getenv_bool, "getenv_bool"),
                           (compiler.get_cache_invalidating_env_vars, "get_cache_invalidating_env_vars")):
        _require(type(function) is BuiltinFunctionType and function is vars(native).get(name)
                 and function.__name__ == name and function.__module__ == native.__name__
                 and (function.__self__ is native or (name == "get_cache_invalidating_env_vars"
                      and type(function.__self__).__module__ == "pybind11_builtins"
                      and type(function.__self__).__name__ == "pybind11_detail_function_record_v1_msvc_md_mscver19")),
                 "compiler environment reader has another native owner")
    _require(cuda.knobs is knobs and cuda.hashlib is hashlib and cache.hashlib is hashlib,
             "compiler calculation dependency changed")
    root = Path(cache.__file__).resolve().parents[1]
    package_key, files = _package_key(root, cache.__version__)
    defaults = {}
    source_class = next(node for node in ast.parse(Path(cuda.__file__).read_bytes()).body
                        if type(node) is ast.ClassDef and node.name == "CUDAOptions")
    for node in source_class.body:
        if type(node) is ast.AnnAssign:
            name = node.target.id
            actual = vars(cuda.CUDAOptions).get(name)
            _plain(actual)
            expected = actual if name == "ptx_options" else ast.literal_eval(node.value)
            _require(type(actual) is type(expected) and actual == expected, "CUDAOptions native default changed: " + name)
            defaults[name] = actual
    fields = vars(cuda.CUDAOptions).get("__dataclass_fields__")
    _require(type(fields) is dict and list(fields) == list(defaults) and all(type(field) is Field
             and type(field.default) is type(defaults[name]) and field.default == defaults[name]
             for name, field in fields.items()), "CUDAOptions dataclass defaults changed")
    controls = {
        "arch": _knob(knobs, "runtime", "override_arch", "TRITON_OVERRIDE_ARCH"),
        "fp_fusion": _knob(knobs, "language", "default_fp_fusion", "TRITON_DEFAULT_FP_FUSION", True, boolean=True),
        "libdevice": _knob(knobs, "nvidia", "libdevice_path", "TRITON_LIBDEVICE_PATH"),
        "mock_ptx_version": _knob(knobs, "nvidia", "mock_ptx_version", "TRITON_MOCK_PTX_VERSION"),
        "override": _knob(knobs, "compilation", "override", "TRITON_KERNEL_OVERRIDE", False, boolean=True),
        "store_binary_only": _knob(knobs, "compilation", "store_binary_only", "TRITON_STORE_BINARY_ONLY", False, boolean=True)}
    _require(controls["arch"] in (None, "sm89") and controls["mock_ptx_version"] is None
             and not controls["override"] and not controls["store_binary_only"],
             "alternate compiler target/mock/override/binary-only mode is not qualified")
    _require(inspect.getattr_static(type(knobs.compilation), "listener") is None
             and vars(knobs.compilation).get("listener") is None
             and inspect.getattr_static(type(knobs.runtime), "add_stages_inspection_hook") is None
             and vars(knobs.runtime).get("add_stages_inspection_hook") is None,
             "compiler has another listener or pipeline stage owner")
    ptxas_descriptor = inspect.getattr_static(type(knobs.nvidia), "ptxas")
    _require(type(ptxas_descriptor) is knobs.env_nvidia_tool and vars(ptxas_descriptor) ==
             {"key": "TRITON_PTXAS.EXE_PATH", "name": "ptxas", "binary": "ptxas.exe"},
             "native ptxas descriptor changed")
    ptxas = root / "backends/nvidia/bin/ptxas.exe"
    selected = vars(knobs.nvidia).get("ptxas", knobs.getenv(ptxas_descriptor.key))
    _require(type(selected) in (str, type(None)) and (selected is None or Path(selected).resolve() == ptxas.resolve()),
             "alternate ptxas executable is not content-qualified")
    ptxas_sha = hashlib.sha256(_file(ptxas, 32 * 1024**2)).hexdigest()
    full_version = _version(str(ptxas), ptxas_sha)
    env = compiler.get_cache_invalidating_env_vars()
    _plain(env)
    _require(type(env) is dict and all(type(key) is str and type(value) in (str, bool, int) for key, value in env.items()),
             "native invalidating environment has an unsupported value")
    producer = dict(schema="t8.minimax_h3.RES_Triton_compile_key.v1", sources=audit.sources,
        executables=audit.functions, package_content=files, package_key=package_key, defaults=defaults,
        controls=controls, invalidating_environment=env,
        ptxas=dict(sha256=ptxas_sha, full_version_stdout=full_version),
        provider_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        inspection_compiles_or_initializes_driver=False)
    return producer, root


def derive_key(source, dependency_hash, option_repr, target, producer, root):
    """Reconstruct native compiler hash from authenticated literal inputs."""
    _require(type(option_repr) is str and len(option_repr) <= 131072, "specialization options exceed budget")
    try:
        original = ast.literal_eval(option_repr)
    except (SyntaxError, ValueError, TypeError, RecursionError) as error:
        raise UnverifiedModelStack("RES specialization options must be bounded literal data") from error
    _plain(original)
    _require(type(original) is dict and str(original) == option_repr, "specialization options are not native literal kwargs")
    options = dict(producer["defaults"])
    options["arch"] = producer["controls"]["arch"] or "sm89"
    options.update({name: value for name, value in original.items() if name in options and value is not None})
    if "supported_fp8_dtypes" not in original or original["supported_fp8_dtypes"] is None:
        options["supported_fp8_dtypes"] = tuple(sorted(set(options["supported_fp8_dtypes"])))
    if "enable_fp_fusion" not in original or original["enable_fp_fusion"] is None:
        options["enable_fp_fusion"] = producer["controls"]["fp_fusion"]
    options["max_num_imprecise_acc_default"] = 0
    _require(options["arch"] == "sm89" and options["num_ctas"] == 1 and options["warp_size"] == 32
             and type(options["num_warps"]) is int and options["num_warps"] > 0
             and options["num_warps"] & (options["num_warps"] - 1) == 0
             and options["ir_override"] is None and options["instrumentation_mode"] == "",
             "native compiler option branch is not qualified")
    libs = options["extern_libs"]
    _require(libs is None or type(libs) is dict, "native extern library map has another owner")
    libs = dict(libs or {})
    if not libs.get("libdevice"):
        libs["libdevice"] = producer["controls"]["libdevice"] or str(root / "backends/nvidia/lib/libdevice.10.bc")
    _require(len(libs) <= 16 and all(type(name) is str and type(path) is str for name, path in libs.items()),
             "native extern libraries exceed the literal file contract")
    options["extern_libs"] = tuple(libs.items())
    contents = {name: {"path": str(Path(path).resolve()), "sha256": hashlib.sha256(_file(path, 32 * 1024**2)).hexdigest()}
                for name, path in libs.items()}
    if Path(libs["libdevice"]).resolve() == (root / "backends/nvidia/lib/libdevice.10.bc").resolve():
        _require(contents["libdevice"]["sha256"] == _LIBDEVICE, "bundled libdevice content requires a fresh audit")
    hashed_options = {**options, "extern_libs": tuple((name, contents[name]["sha256"]) for name in sorted(libs))}
    options_key = "_".join(f"{name}-{value}" for name, value in sorted(hashed_options.items()))
    options_hash = hashlib.sha256(options_key.encode()).hexdigest()
    sorted_signature = [value for name, value in sorted(source.signature.items())]
    constant_key = "-".join(str(value) for key, value in sorted(source.constants.items()))
    source_key = f"{dependency_hash}-{str(source.attrs)}-{sorted_signature}-{constant_key}"
    source_hash = hashlib.sha256(source_key.encode()).hexdigest()
    backend_hash = producer["ptxas"]["full_version_stdout"] + "-" + str(target["arch"])
    key = (f"{producer['package_key']}-{source_hash}-{backend_hash}-{options_hash}-"
           f"{str(sorted(producer['invalidating_environment'].items()))}")
    return dict(compile_hash=hashlib.sha256(key.encode()).hexdigest(), options=options, extern_library_content=contents,
                source_hash=source_hash, options_hash=options_hash)


def inspect_compile_key(program, dependency_hash, option_repr, target, metadata, producer, root):
    value = derive_key(program.src, dependency_hash, option_repr, target, producer, root)
    _require(value["compile_hash"] == program.hash, "native compiled key differs from source/options/compiler/environment")
    _require(all(name in metadata and _plain(metadata[name]) == _plain(item) for name, item in value["options"].items())
             and all(metadata.get(name) == item for name, item in producer["invalidating_environment"].items()),
             "compiled metadata options/environment differ from the actual specialization")
    directory = base64.b32encode(bytes.fromhex(value["compile_hash"])).decode().rstrip("=")
    _require(all(Path(path).parent.name == directory for path in program.metadata_group.values()),
             "compiled artifacts do not belong to their derived native cache key")
    return {**value, "compiler_key_independently_derived": True,
            "compile_key_is_not_a_CUDA_execution_certificate": True}
