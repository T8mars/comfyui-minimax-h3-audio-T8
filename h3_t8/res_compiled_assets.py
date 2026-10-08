"""Bound, data-only RES compiled assets; no import, compile or device launch.

Dynamic cache observations belong to the boundary, not stable MODEL identity.
File integrity is necessary but is not a CUDA or portable-resume certificate.
"""
import base64
import hashlib
import importlib
import inspect
import json
from pathlib import Path
from types import BuiltinFunctionType

from .res_kitchen_identity import _SourceAudit, _digest
from .res_triton_cache_identity import _file, _plain, _require

SCHEMA = "t8.minimax_h3.RES_compiled_assets.v1"
MAX_JSON_BYTES = 1024**2
MAX_FILES = 512


def native_asset_roots():
    """Read authenticated native cache configuration without calling getters.

    A custom manager/factory is not imported or executed. Inspection of an
    unknown configuration remains unqualified, not a ban on full sampling.
    """
    from .res_triton_compile_identity import _PINS
    knobs = importlib.import_module("triton.knobs")
    cache = importlib.import_module("triton.runtime.cache")
    native = importlib.import_module("triton._C.libtriton")
    audit = _SourceAudit({name: _PINS[name] for name in (knobs.__name__, cache.__name__)})
    for module in (knobs, cache):
        audit.module(module)
        _require(audit.sources[module.__name__] == _PINS[module.__name__], "asset configuration source changed")
    _require(type(knobs.cache) is knobs.cache_knobs and type(knobs.getenv) is BuiltinFunctionType
             and knobs.getenv is native.getenv and knobs.getenv.__self__ is native,
             "asset configuration has another native owner")
    group = knobs.cache
    home = inspect.getattr_static(type(group), "home_dir")
    directory = inspect.getattr_static(type(group), "dir")
    _require(type(home) is knobs.env_str and set(vars(home)) == {"key", "name", "default"}
             and vars(home)["key"] == "TRITON_HOME" and vars(home)["name"] == "home_dir"
             and type(vars(home)["default"]) is str,
             "asset home descriptor changed")
    _require(type(directory) is knobs.env_str_callable_default
             and set(vars(directory)) == {"key", "name", "default_factory"}
             and vars(directory)["key"] == "TRITON_CACHE_DIR" and vars(directory)["name"] == "dir",
             "asset cache descriptor changed")
    # Authenticate, but never invoke, the default lambda or native path method.
    audit.function(vars(directory)["default_factory"], knobs, dependencies=False)
    audit.function(knobs.cache_knobs.get_triton_dir, knobs, dependencies=False)
    _require("get_triton_dir" not in vars(group), "asset cache path method has an instance override")
    for name, key in (("manager_class", "TRITON_CACHE_MANAGER"),
                      ("remote_manager_class", "TRITON_REMOTE_CACHE_BACKEND")):
        _require(vars(group).get(name) is None and knobs.getenv(key) is None,
                 "asset custom manager is not qualified")
    home_value = vars(group).get("home_dir", knobs.getenv(vars(home)["key"], vars(home)["default"]))
    root_value = vars(group).get("dir", knobs.getenv(vars(directory)["key"]))
    _require(type(home_value) is str and type(root_value) in (str, type(None)),
             "asset cache path has an opaque value")
    root = root_value if root_value is not None else str(Path(home_value) / ".triton/cache")
    _require(bool(root) and len(root) <= 4096 and "\x00" not in root, "asset cache path exceeds budget")
    return Path(root).resolve(), Path(cache.__file__).resolve().parents[1]


def freeze_compiled_assets(observed):
    """Freeze only authenticated observations; never retain a live object."""
    _require(type(observed) is dict and len(observed) <= 32, "asset producer count exceeds budget")
    kernels = {}
    for name, entry in observed.items():
        _require(type(name) is str and len(name) <= 512 and type(entry) is dict,
                 "asset producer has an opaque owner")
        _require(entry.get("CUDA_kernel_launched_by_inspection") is False,
                 "asset observer unexpectedly launched a kernel")
        kernels[name] = {"producer_sha256": entry["producer_sha256"],
                         "programs": _plain(entry["observed_programs"])}
    result = {"schema": SCHEMA, "kernels": kernels,
              "CUDA_execution_qualified": False, "portable_resume_qualified": False}
    result["sha256"] = _digest(result)
    validate_compiled_assets_data(result)
    return result


def _hex(value):
    return type(value) is str and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def validate_compiled_assets_data(snapshot):
    """Check bounded literal metadata before tensor loading; no filesystem I/O."""
    snapshot = _plain(snapshot)
    _require(type(snapshot) is dict and set(snapshot) == {"schema", "kernels", "sha256",
                 "CUDA_execution_qualified", "portable_resume_qualified"}
             and snapshot["schema"] == SCHEMA and snapshot["CUDA_execution_qualified"] is False
             and snapshot["portable_resume_qualified"] is False,
             "asset snapshot is not a data-only boundary")
    _require(_hex(snapshot["sha256"]) and snapshot["sha256"] ==
             _digest({key: value for key, value in snapshot.items() if key != "sha256"}),
             "asset snapshot digest mismatch")
    _require(len(json.dumps(snapshot).encode()) <= MAX_JSON_BYTES,
             "asset snapshot exceeds its metadata budget")
    kernels = snapshot["kernels"]
    _require(type(kernels) is dict and len(kernels) <= 32, "asset producer count exceeds budget")
    files = 0
    for name, entry in kernels.items():
        _require(type(name) is str and len(name) <= 512 and type(entry) is dict
                 and set(entry) == {"producer_sha256", "programs"} and _hex(entry["producer_sha256"]),
                 "asset producer identity changed")
        _require(type(entry["programs"]) is list and len(entry["programs"]) <= 128,
                 "asset program count exceeds budget")
        for program in entry["programs"]:
            _require(type(program) is dict and _hex(program.get("compile_hash"))
                     and type(program.get("source_name")) is str
                     and program.get("compiler_key_independently_derived") is True
                     and program.get("native_files_and_memory_content_consistent") is True
                     and program.get("native_launcher_and_loaded_handle_qualified") is False,
                     "asset program lacks authenticated content observations")
            assets = program.get("artifacts")
            _require(type(assets) is dict and 2 <= len(assets) <= 16,
                     "asset file group exceeds budget")
            for filename, record in assets.items():
                _require(type(filename) is str and type(record) is dict
                         and set(record) == {"path", "bytes", "sha256"}
                         and type(record["path"]) is str and 0 < len(record["path"]) <= 4096
                         and type(record["bytes"]) is int and 0 < record["bytes"] <= 32 * 1024**2
                         and _hex(record["sha256"]), "asset file record is invalid")
                files += 1
    _require(files <= MAX_FILES, "asset file count exceeds budget")
    return snapshot


def _path_spelling(path):
    # Windows native cache records use extended-length names. Normalize only
    # that spelling, not content or filesystem identity, before root checks.
    value = str(path)
    if value.startswith("\\\\?\\UNC\\"):
        value = "\\\\" + value[8:]
    elif value.startswith("\\\\?\\"):
        value = value[4:]
    return Path(value)


def _bound_file(record, root, *, directory=None, filename=None):
    path = _path_spelling(record["path"])
    _require(path.is_absolute() and path.name == (filename or path.name), "asset filename changed")
    resolved = _path_spelling(path.resolve())
    root = _path_spelling(root.resolve())
    _require(root in resolved.parents and (directory is None or resolved.parent.name == directory),
             "asset path is outside its current native root")
    current = root
    for part in path.relative_to(root).parts:
        current = current / part
        _require(not current.is_symlink() and not getattr(current, "is_junction", lambda: False)(),
                 "asset path traverses a link")
    data = _file(path, 32 * 1024**2)
    _require(("bytes" not in record or len(data) == record["bytes"])
             and hashlib.sha256(data).hexdigest() == record["sha256"], "bound asset file content changed")


def validate_bound_compiled_assets(snapshot, current_observed):
    """Validate a saved warm manifest against cold producers and native files.

    Does not insert programs in device caches, load a pyd, or initialize handles.
    A process address is neither stored nor compared.
    """
    snapshot = validate_compiled_assets_data(snapshot)
    current = freeze_compiled_assets(current_observed)
    _require(set(snapshot["kernels"]) == set(current["kernels"]), "bound asset producer set changed")
    for name, saved in snapshot["kernels"].items():
        _require(saved["producer_sha256"] == current["kernels"][name]["producer_sha256"],
                 "bound asset producer content changed")
    programs = [program for entry in snapshot["kernels"].values() for program in entry["programs"]]
    if not programs:
        return {"compiled_programs": 0, "file_contents_verified": True,
                "portable_resume_qualified": False, "kernel_launched": False}
    cache_root, compiler_root = native_asset_roots()
    for program in programs:
        directory = base64.b32encode(bytes.fromhex(program["compile_hash"])).decode().rstrip("=")
        for filename, record in program["artifacts"].items():
            _require(filename.startswith(program["source_name"][:150] + "."), "asset source filename changed")
            _bound_file(record, cache_root, directory=directory, filename=filename)
        for record in program["actual_compile_key"]["extern_library_content"].values():
            _bound_file(record, compiler_root)
        launcher = program["actual_launcher"]
        if launcher["native_launcher_extension_content_bound"]:
            extension = launcher["extension"]
            _require(_hex(extension["native_build_cache_key"]), "asset launcher cache key changed")
            launcher_directory = base64.b32encode(bytes.fromhex(extension["native_build_cache_key"])).decode().rstrip("=")
            _bound_file(extension, cache_root, directory=launcher_directory)
    return {"compiled_programs": len(programs), "file_contents_verified": True,
            "portable_resume_qualified": False, "kernel_launched": False}
