"""Actual KJ in-place RMS/partial split-half RoPE dispatch content for RES."""
import hashlib
import importlib
from pathlib import Path
import sys

from .patch_stack_policy import UnverifiedModelStack
from .res_kitchen_identity import _SourceAudit, _custom_op, _constraint, _binary, _data, _digest, _require
from .res_triton_identity import inspect_native_jit


_PINS = {
    "comfy_kitchen._rope_utils": "53436ba6213778061740d87d95f38d72d8bda21eb2c9cf34a1a64b3931e33f37",
    "comfy_kitchen.backends.eager.rope": "8bc585b0400e2d53d6cd1d921a991d57358697dd6f48ba0cb3ffa3f796c619e5",
    "comfy_kitchen.backends.triton": "9d85006409b9b7d3702e8c97150ebe606b704ee4f84faf2a66fc832777a1016a",
    "comfy_kitchen.backends.triton.rms_rope": "0d20124f20213d4855be7490db6d6852ea62bc99fd4d89751b96dfa405acc465",
}


def _inspect(ck, observed_artifacts):
    _require(ck is not None and ck is sys.modules.get("comfy_kitchen"), "RoPE uses another module")
    registry_module = importlib.import_module("comfy_kitchen.registry")
    constraints = importlib.import_module("comfy_kitchen.constraints")
    eager = importlib.import_module("comfy_kitchen.backends.eager.rope")
    cuda = importlib.import_module("comfy_kitchen.backends.cuda")
    registry = ck.registry
    _require(type(registry) is registry_module.BackendRegistry and registry is registry_module.registry,
             "RoPE uses another backend registry")
    audit = _SourceAudit(_PINS)
    for name in ("get_implementation", "get_capable_backend", "validate_backend_for_call", "is_available"):
        _require(name not in vars(registry), "RoPE registry has a foreign selector")
        audit.function(getattr(type(registry), name), registry_module)
    op = _custom_op(audit, ck, eager, registry, "rms_rope_split_half_")
    candidates = {}
    for name, backend in registry._backends.items():
        if "rms_rope_split_half_" not in registry._capabilities.get(name, ()):
            continue
        _require(name in ("cuda", "eager", "triton")
                 and backend is sys.modules.get("comfy_kitchen.backends." + name), "RoPE backend has another owner")
        audit.module(backend)
        candidates[name] = {"implementation": audit.function(backend.rms_rope_split_half_),
            "constraints": _constraint(audit, registry._constraints[(name, "rms_rope_split_half_")], constraints)}
    # Both installed CPU/CUDA paths and the CUDA branch's eager partial fallback
    # are bound even while inspecting under --cpu. No fake registration occurs.
    _require(cuda._eager_rope is eager, "CUDA partial-RoPE fallback has another owner")
    audit.function(cuda.rms_rope_split_half_)
    audit.function(eager.rms_rope_split_half_)
    audit.function(eager.rms_rope_split_half)
    jit = {}
    if "triton" in candidates:
        module = importlib.import_module("comfy_kitchen.backends.triton.rms_rope")
        _require(module._eager_rope is eager, "Triton partial-RoPE fallback has another owner")
        jit = inspect_native_jit(module.rms_rope_kernel, module, _PINS[module.__name__],
                                 observed_artifacts=observed_artifacts)
    value = {"schema": "t8.minimax_h3.RES_RMS_RoPE_content.v1", "custom_op": op,
             "priority": list(registry._priority), "disabled": sorted(registry._disabled),
             "thread_override": _data(getattr(registry._thread_local, "backend_override", None)),
             "candidates": candidates, "executables": audit.functions, "sources": audit.sources,
             "constants": audit.constants, "CUDA_binary": _binary(cuda, ("rms_rope", "rms_rope1")),
             "Triton": jit, "dispatch_content_verified": True, "CUDA_execution_qualified": False,
             "provider_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    value["sha256"] = _digest(value)
    return value


def inspect_rope_kitchen(ck, *, observed_artifacts=None):
    try:
        return _inspect(ck, observed_artifacts)
    except UnverifiedModelStack:
        raise
    except (AttributeError, ImportError, KeyError, OSError, TypeError, ValueError) as error:
        raise UnverifiedModelStack("RES RoPE content identity unavailable: " + str(error)) from error
