"""Read-only RES identity of the installed Sol custom-op/registry/binary chain.

This binds execution content, not CUDA numerical or performance qualification.
It never calls a selected attention kernel, changes a registry, or initializes
CUDA. Unknown registration owners remain available to ordinary full sampling.
"""
import hashlib
import importlib
from importlib.machinery import ExtensionFileLoader
import inspect
import json
from pathlib import Path
import sys
from types import FunctionType

import torch

from .long_video_dual_identity import content_identity
from .modular_sampling.progressive_effect_identity import _code_identity
from .patch_stack_policy import UnverifiedModelStack
from .relay_kj_backend import _codes, _live_code_matches


# Audited installed source epoch, not package-name/version declarations. A new
# implementation needs a new audit; ordinary sampling is not gated by these.
_SOURCE_PINS = {
    "comfy_kitchen": "a6cc073c46db51f1ff055065daa809ab217d642adfc8b0f552be049bbe18b4a3",
    "comfy_kitchen.registry": "e7eb061c1d52222af2a8d738102bfe9f4e85d5cc5a9f9fe4bd443d0185b0f77f",
    "comfy_kitchen.constraints": "89d7b3b51e62b8cf91757335c870f356941a5f4d6cccc2eb967bc01fc9ab640f",
    "comfy_kitchen.backends.eager": "296053492c0e1fecc56341b3a4df58259b371c369c4d25bb0fd8415c6e3fc679",
    "comfy_kitchen.backends.eager.sol_attn": "fd71051a57eae170af415eda3375ceaeeb219b48eb804386d3922e8db6e2ca88",
    "comfy_kitchen.backends.cuda": "27ef9c1063a3dcdee6e05928e07588ed82f076b64c07c131d987d065ab4c50ac",
}


def _require(condition, message):
    if not condition:
        raise UnverifiedModelStack("RES kitchen " + message)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _data(value):
    if type(value) is frozenset:
        items = [_data(item) for item in value]
        return {"type": "frozenset", "items": sorted(items, key=lambda item: json.dumps(item, sort_keys=True))}
    return content_identity(value)


class _SourceAudit:
    def __init__(self, extra_source_pins=None):
        self.source_pins = dict(_SOURCE_PINS)
        for name, digest in (extra_source_pins or {}).items():
            _require(name not in self.source_pins, "new inspector must not replace an existing source pin")
            self.source_pins[name] = digest
        self.sources = {}
        self.functions = {}
        self.constants = {}
        self._compiled = {}

    def module(self, module):
        _require(sys.modules.get(module.__name__) is module, "module is detached from its loaded owner")
        name = module.__name__
        if name not in self._compiled:
            path = Path(module.__file__).resolve()
            _require(path.suffix == ".py", "expected Python source, not an opaque module")
            source = path.read_bytes()
            digest = hashlib.sha256(source).hexdigest()
            if name.startswith("comfy_kitchen"):
                _require(self.source_pins.get(name) == digest, "source requires a fresh kernel-chain audit: " + name)
            self.sources[name] = digest
            self._compiled[name] = tuple(_codes(compile(source, str(path), "exec", dont_inherit=True)))
        return self._compiled[name]

    def function(self, function, module=None, *, dependencies=True):
        _require(type(function) is FunctionType, "selected Python owner is opaque")
        if module is None:
            # functools.wraps can replace __module__; actual globals own code.
            module = sys.modules.get(function.__globals__.get("__name__"))
        _require(module is not None and _live_code_matches(function, self.module(module), module),
                 "live executable differs from source or has foreign globals")
        key = module.__name__ + ":" + function.__code__.co_qualname
        record = {"code_sha256": _digest(_code_identity(function.__code__)),
                  "defaults": _data(function.__defaults__), "kwdefaults": _data(function.__kwdefaults__)}
        if key in self.functions:
            _require(self.functions[key] == record, "same code owner has different live defaults")
            return record
        self.functions[key] = record
        if dependencies:
            names = {name for code in _codes(function.__code__) for name in code.co_names}
            for name in sorted(names):
                if name not in function.__globals__:
                    continue
                value = function.__globals__[name]
                if type(value) is FunctionType:
                    _require(value.__globals__.get("__name__", "").startswith("comfy_kitchen"),
                             "calculation helper is outside the audited package: " + name)
                    self.function(value)
                elif type(value) in (int, float, bool, str, tuple, frozenset) or value is None:
                    self.constants[module.__name__ + ":" + name] = _data(value)
                elif name == "torch":
                    _require(value is torch, "torch dependency was replaced")
        return record


def _custom_op(audit, ck, eager, registry, operation="sol_attn"):
    import torch._compile as compile_module
    import torch._library.custom_ops as custom_module

    op = getattr(eager, "_op_" + operation)
    _require(type(op) is custom_module.CustomOpDef
             and custom_module.OPDEFS.get("comfy_kitchen::" + operation) is op
             and getattr(torch.ops.comfy_kitchen, operation).default is op._opoverload,
             "custom-op registration is not the actual selected definition")
    _require(op._namespace == "comfy_kitchen" and op._name == operation
             and set(op._backend_fns) == {None} and not op._disabled_kernel,
             "custom-op has another device-specific/disabled kernel")
    for name in ("_torch_dispatch_fns", "_used_triton_kernels"):
        _require(not getattr(op, name), "custom-op has another dispatch owner")
    for name in ("_setup_context_fn", "_backward_fn", "_vmap_fn", "_autocast_cpu_dtype", "_autocast_cuda_dtype"):
        _require(getattr(op, name) is None, "custom-op has an additional execution rule")
    _require(not op._opoverload.py_kernels and not op._opoverload.python_key_table,
             "operator overload has an extra Python dispatch rule")
    _require(eager.registry is registry, "custom-op uses another backend registry")
    audit.function(getattr(ck, operation), ck)
    audit.function(op._init_fn, eager)
    audit.function(op._abstract_fn, eager)
    outer = op._backend_fns[None]
    audit.function(outer, compile_module, dependencies=False)
    _require(outer.__code__.co_qualname == "_disable_dynamo.<locals>.inner",
             "default backend is not the native wrapper code")
    cells = inspect.getclosurevars(outer).nonlocals
    _require(set(cells) == {"fn", "recursive"} and cells["recursive"] is True,
             "default backend is not the native disable-dynamo wrapper")
    wrapped = cells["fn"]
    audit.function(wrapped, custom_module, dependencies=False)
    _require(wrapped.__code__.co_qualname == "CustomOpDef.register_kernel.<locals>.inner.<locals>.wrapped_fn",
             "default backend is not the native kernel wrapper code")
    cells = inspect.getclosurevars(wrapped).nonlocals
    _require(set(cells) == {"device_type", "fn", "self"} and cells["device_type"] is None
             and cells["self"] is op and cells["fn"] is op._init_fn,
             "default backend wrapper delegates to another computation")
    cached = vars(wrapped).get("__dynamo_disable")
    _require(not (set(vars(wrapped)) - {"__dynamo_disable"}), "backend wrapper has foreign attributes")
    if cached is not None:
        # This native cache appears only after a real op call. Authenticate it
        # without invoking it; cache presence must not change portable identity.
        import torch._dynamo.eval_frame as eval_frame
        _require(type(cached) is FunctionType
                 and _live_code_matches(cached, audit.module(eval_frame), eval_frame)
                 and cached.__code__.co_qualname == "DisableContext.__call__.<locals>._fn"
                 and cached.__defaults__ is None and cached.__kwdefaults__ is None,
                 "hot default backend cache is not native disable-context code")
        cells = inspect.getclosurevars(cached).nonlocals
        _require(set(cells) == {"fn", "self"} and cells["fn"] is wrapped
                 and type(cells["self"]) is eval_frame.DisableContext
                 and cells["self"].callback is None and cells["self"].wrapping is False
                 and cells["self"].msg is None,
                 "hot default backend cache delegates to a foreign execution owner")
        _require(vars(cached) == {"_torchdynamo_disable": True, "_torchdynamo_disable_msg": None,
                                 "_torchdynamo_orig_callable": wrapped, "_torchdynamo_disable_recursive": True},
                 "hot default backend cache has foreign callable attributes")
    # Computed CPU/CUDA dispatch must still be the native registered backend.
    table = torch._C._dispatch_dump_table("comfy_kitchen::" + operation)
    for device in ("CPU", "CUDA"):
        row = next((row for row in table.splitlines() if row.startswith(device + ":")), "")
        _require("custom_ops.py:" in row and "[default backend kernel]" in row,
                 "native dispatcher has a replacement " + device + " kernel")
    # Bind the wrapper implementations also for a cold process. Actual hot
    # caches are validated above, but are not themselves algorithm settings.
    eval_frame = importlib.import_module("torch._dynamo.eval_frame")
    audit.function(eval_frame.DisableContext.__call__, eval_frame, dependencies=False)
    return {"schema": op._schema, "tags": [str(tag) for tag in op._tags],
            "default_backend": "native_custom_op_disable_dynamo",
            "extra_dispatch_rules": False}


def _constraint(audit, rule, constraints_module):
    allowed = (constraints_module.FunctionConstraints, constraints_module.ParamConstraint,
               constraints_module.ExactDims, constraints_module.MinDims, constraints_module.DivisibleBy)
    _require(type(rule) in allowed, "constraint has an unknown validation owner")
    state = {}
    for name, value in sorted(vars(rule).items()):
        if name == "params":
            _require(type(value) is dict and all(type(key) is str for key in value), "constraint params are malformed")
            state[name] = {key: _constraint(audit, item, constraints_module) for key, item in sorted(value.items())}
        elif name == "shape_rules":
            state[name] = [_constraint(audit, item, constraints_module) for item in value]
        elif name == "call_rules":
            state[name] = []
            for item in value:
                _require(type(item) is FunctionType, "constraint call rule has an opaque owner")
                owner = sys.modules.get(item.__globals__.get("__name__"))
                # FunctionConstraints is defined in constraints.py, but native
                # backend-specific call rules live in their backend module.
                # Authenticate the real pinned source/global owner; never call
                # the rule or manufacture backend availability during inspection.
                _require(owner is not None and owner.__name__ in audit.source_pins,
                         "constraint call rule is outside the audited package")
                state[name].append(audit.function(item, owner))
        else:
            state[name] = _data(value)
    for name, value in vars(type(rule)).items():
        if type(value) is FunctionType and name in ("check", "check_dtype", "check_device", "check_shape"):
            audit.function(value, constraints_module)
    return {"class": type(rule).__qualname__, "state": state}


def _binary(cuda, export_names=("sol_attn", "sol_attn_plan")):
    extension = cuda._C
    _require(cuda._EXT_AVAILABLE and extension is sys.modules.get("comfy_kitchen.backends.cuda._C")
             and isinstance(extension.__spec__.loader, ExtensionFileLoader),
             "CUDA extension is not its loaded native binary")
    path = Path(extension.__file__).resolve()
    _require(path.parent == Path(cuda.__file__).resolve().parent and path.name.startswith("_C."),
             "CUDA extension path has a foreign owner")
    exports = {}
    for name in export_names:
        function = getattr(extension, name)
        _require(type(function).__module__ == "nanobind" and type(function).__name__ == "nb_func"
                 and function.__module__ == extension.__name__ and function.__name__ == name,
                 "native extension export was replaced: " + name)
        exports[name] = {"type": "nanobind.nb_func", "module": extension.__name__}
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    _require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
             "native binary changed during inspection")
    return {"sha256": digest.hexdigest(), "bytes": before.st_size, "exports": exports}


def _inspect_sol_kitchen(ck):
    """Bind actual selection state and both installed Sol computations, no launch."""
    _require(ck is not None and ck is sys.modules.get("comfy_kitchen"), "Sol uses another kitchen module")
    registry_module = importlib.import_module("comfy_kitchen.registry")
    constraints = importlib.import_module("comfy_kitchen.constraints")
    eager = importlib.import_module("comfy_kitchen.backends.eager.sol_attn")
    cuda = importlib.import_module("comfy_kitchen.backends.cuda")
    registry = ck.registry
    _require(registry is registry_module.registry and type(registry) is registry_module.BackendRegistry,
             "selected registry has a foreign owner")
    audit = _SourceAudit()
    for module in (ck, registry_module, constraints, eager, cuda):
        audit.module(module)
    for name in ("get_implementation", "get_capable_backend", "validate_backend_for_call", "is_available"):
        _require(name not in vars(registry), "registry instance overrides native selection: " + name)
        audit.function(getattr(type(registry), name), registry_module)
    audit.function(type(registry)._compute_capability.func, registry_module, dependencies=False)
    _require(type(registry._priority) is list and all(type(name) is str for name in registry._priority)
             and type(registry._disabled) is set and all(type(name) is str for name in registry._disabled),
             "selection priority/disabled state is malformed")
    candidates = {}
    for name, backend in registry._backends.items():
        if "sol_attn" not in registry._capabilities.get(name, ()):
            continue
        expected = sys.modules.get("comfy_kitchen.backends." + name)
        _require(name in ("cuda", "eager") and backend is expected, "selected Sol backend is unknown")
        audit.module(backend)
        candidates[name] = {"implementation": audit.function(backend.sol_attn),
                            "constraints": _constraint(audit, registry._constraints[(name, "sol_attn")], constraints)}
    # Bind the installed CUDA implementation even in a CPU inspection process.
    # Availability is recorded below, not fabricated by registering a backend.
    audit.function(cuda.sol_attn)
    audit.function(eager.sol_attn)
    op = _custom_op(audit, ck, eager, registry)
    value = {"schema": "t8.minimax_h3.RES_Sol_kitchen_chain.v1", "custom_op": op,
             "priority": list(registry._priority), "disabled": sorted(registry._disabled),
             "thread_override": _data(getattr(registry._thread_local, "backend_override", None)),
             "candidates": candidates, "sources": audit.sources,
             "executables": audit.functions, "constants": audit.constants,
             "CUDA_binary": _binary(cuda), "dispatch_content_verified": True,
             "CUDA_execution_qualified": False,
             "provider_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    value["sha256"] = _digest(value)
    return value


def inspect_sol_kitchen(ck):
    """Unknown/unavailable owners yield an unverified identity, not a full-run ban."""
    try:
        return _inspect_sol_kitchen(ck)
    except UnverifiedModelStack:
        raise
    except (AttributeError, ImportError, KeyError, OSError, TypeError, ValueError) as error:
        raise UnverifiedModelStack("RES kitchen identity unavailable: " + str(error)) from error
