"""Request-local INT8 row scheduling, without Kitchen/global dispatch writes.

Actual QuantizedTensor operators and the actual public fused INT8 call execute
each row range. Module-owned preprocessing and weight preparation stay outside
that split. Full outputs and the network's full K/V still exist.
"""

from __future__ import annotations

import ast
import builtins
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
import inspect
from pathlib import Path
import sys
import threading
import time
from types import CodeType, FunctionType, MethodType

import torch
from torch import nn
from torch.utils._python_dispatch import (
    TorchDispatchMode,
    _get_current_dispatch_mode_stack,
)

import comfy.model_management as mm
import comfy.ops as ops
import comfy_kitchen as kitchen
from comfy.ldm.minimax import model as core
from comfy_kitchen.registry import registry
from comfy_kitchen import constraints
from comfy_kitchen.tensor import QuantizedTensor
from comfy_kitchen.tensor import base as tensor_base
from comfy_kitchen.tensor import int8 as tensor_int8

from .h3_projection_chunks import (
    _module_call_native,
    _source_exact,
    shared_network_lease,
)
from .source_code_identity import executable_code_equal

SCHEMA = "t8.h3.int8_row_chunk.v1"
_ACTIVE = ContextVar("t8_int8_row_request", default=())
_UNSET = object()
_OUT_OF_TARGET = object()
_CONTRACTS = {}
_CONTRACT_LOCK = threading.RLock()
_DISPATCHERS = {}
_LINEAR_OPS = {
    torch.ops.aten.linear.default: 0,
    torch.ops.aten.mm.default: 0,
    torch.ops.aten.addmm.default: 1,
}


@dataclass(frozen=True)
class Int8RowPlan:
    rows: int = 2048
    targets: str = "all"

    def __post_init__(self):
        if type(self.rows) is not int or self.rows < 0:
            raise ValueError(
                "INT8 row count must be a nonnegative integer; zero is off"
            )
        if type(self.targets) is not str or self.targets not in ("all", "mlp"):
            raise ValueError("INT8 targets must be all or mlp")


@dataclass
class Int8Runtime:
    network: object
    plan: Int8RowPlan
    phase: str
    owners: dict = field(default_factory=dict)
    mlp_owners: set = field(default_factory=set)
    calls: dict = field(default_factory=dict)
    delegated: set = field(default_factory=set)
    completed_output_bytes: int = 0
    elapsed: float = 0.0
    network_code: object = None

    def note(self, path, rows, *, chunk):
        item = self.calls.setdefault(
            path, {"calls": 0, "rows": 0, "max_rows": 0, "chunks": 0}
        )
        item["calls"] += 1
        item["rows"] += rows
        item["max_rows"] = max(item["max_rows"], rows)
        item["chunks"] += int(chunk)

    def report(self):
        return {
            "schema": SCHEMA,
            "plan": asdict(self.plan),
            "phase": self.phase,
            "calls": {key: dict(value) for key, value in self.calls.items()},
            "delegated": sorted(self.delegated),
            "partial": bool(self.delegated),
            "completed_output_bytes": self.completed_output_bytes,
            "cpu_dispatch_wall_seconds": self.elapsed,
            "full_output_buffers_retained": True,
            "full_KV_retained": True,
            "allocator_peak_measured": False,
            "performance_qualified": False,
            "portable_cache_identity": False,
            "quality_accepted": False,
        }


def _function(method):
    return getattr(method, "__func__", method)


def _codes(code):
    yield code
    for value in code.co_consts:
        if type(value) is CodeType:
            yield from _codes(value)


def _default(value, node):
    if isinstance(node, ast.Constant):
        return type(value) is type(node.value) and value == node.value
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "torch"
    ):
        return node.attr in ("float32", "float16", "bfloat16") and value is getattr(
            torch, node.attr
        )
    if isinstance(node, ast.Dict):
        return type(value) is dict and not value and not node.keys
    if isinstance(node, (ast.List, ast.Tuple)):
        kind = list if isinstance(node, ast.List) else tuple
        return (
            type(value) is kind
            and len(value) == len(node.elts)
            and all(
                _default(item, declaration)
                for item, declaration in zip(value, node.elts)
            )
        )
    return False


def _exact(function, module, *, allow_closure=False):
    """Read/compile installed trusted source, never execute source expressions."""
    function = _function(function)
    if (
        type(function) is not FunctionType
        or function.__globals__ is not vars(module)
        or (function.__closure__ is not None and not allow_closure)
    ):
        return False
    try:
        source = inspect.getsourcefile(function)
        if source is None or Path(source).resolve() != Path(module.__file__).resolve():
            return False
        with _CONTRACT_LOCK:
            contract = _CONTRACTS.get(module)
            if contract is None:
                raw = Path(source).read_bytes()
                contract = (
                    tuple(_codes(compile(raw, source, "exec", dont_inherit=True))),
                    tuple(
                        node
                        for node in ast.walk(ast.parse(raw, filename=source))
                        if isinstance(node, ast.FunctionDef)
                    ),
                )
                _CONTRACTS[module] = contract
        code = function.__code__
        compiled = [
            candidate
            for candidate in contract[0]
            if candidate.co_qualname == code.co_qualname
            and executable_code_equal(code, candidate)
        ]
        definitions = [
            node
            for node in contract[1]
            if node.name == code.co_name
            and min([node.lineno, *(item.lineno for item in node.decorator_list)])
            == code.co_firstlineno
        ]
        if len(compiled) != 1 or len(definitions) != 1:
            return False
        declaration = definitions[0]
        defaults = function.__defaults__
        declared = declaration.args.defaults
        if declared:
            if (
                type(defaults) is not tuple
                or len(defaults) != len(declared)
                or not all(
                    _default(value, node) for value, node in zip(defaults, declared)
                )
            ):
                return False
        elif defaults is not None:
            return False
        keyword = {
            arg.arg: value
            for arg, value in zip(
                declaration.args.kwonlyargs, declaration.args.kw_defaults
            )
            if value is not None
        }
        live = function.__kwdefaults__
        return (
            live is None
            if not keyword
            else type(live) is dict
            and set(live) == set(keyword)
            and all(_default(live[key], node) for key, node in keyword.items())
        )
    except (AttributeError, OSError, TypeError, SyntaxError, ValueError):
        return False


def _custom_op_known(device):
    """Follow actual Torch dispatch kernels and closures, not __wrapped__ tags."""
    from comfy_kitchen.backends.eager import quantization as eager
    import torch._library.custom_ops as custom
    import torch._compile as compile_module

    definition = eager._op_int8_linear
    if (
        type(definition) is not custom.CustomOpDef
        or type(definition._backend_fns) is not dict
    ):
        return False
    leaf = definition._init_fn
    if not _exact(leaf, eager) or leaf.__code__.co_qualname != "_op_int8_linear":
        return False
    current = definition._backend_fns.get(device, definition._backend_fns.get(None))
    visited = set()
    while current is not leaf:
        if type(current) is not FunctionType or id(current) in visited:
            return False
        visited.add(id(current))
        values = dict(
            zip(
                current.__code__.co_freevars,
                (cell.cell_contents for cell in (current.__closure__ or ())),
            )
        )
        if current.__code__.co_qualname == "_disable_dynamo.<locals>.inner":
            if (
                not _exact(current, compile_module, allow_closure=True)
                or set(values) != {"fn", "recursive"}
                or values["recursive"] is not True
            ):
                return False
            inner = values["fn"]
            cached = getattr(inner, "__dynamo_disable", None)
            if cached is not None:
                eval_module = sys.modules.get("torch._dynamo.eval_frame")
                if (
                    eval_module is None
                    or type(cached) is not FunctionType
                    or not _exact(cached, eval_module, allow_closure=True)
                    or cached.__code__.co_qualname
                    != "DisableContext.__call__.<locals>._fn"
                ):
                    return False
                cells = dict(
                    zip(
                        cached.__code__.co_freevars,
                        (cell.cell_contents for cell in (cached.__closure__ or ())),
                    )
                )
                if (
                    set(cells) != {"fn", "self"}
                    or cells["fn"] is not inner
                    or type(cells["self"]) is not eval_module.DisableContext
                    or cells["self"].callback is not None
                ):
                    return False
            current = inner
        elif (
            current.__code__.co_qualname
            == "CustomOpDef.register_kernel.<locals>.inner.<locals>.wrapped_fn"
        ):
            if (
                not _exact(current, custom, allow_closure=True)
                or set(values) != {"device_type", "fn", "self"}
                or values["self"] is not definition
                or (
                    values["device_type"] is not None
                    and type(values["device_type"]) is not str
                )
            ):
                return False
            current = values["fn"]
        else:
            return False
    return True


def _python_math_known(implementation, owner):
    """Check actual transitive Python math helpers before changing call size."""
    pending, seen = [(implementation, owner)], set()
    kitchen_root = Path(kitchen.__file__).resolve().parent
    aliases = {
        "_apply_input_act": "apply_input_act",
        "_apply_residual": "apply_residual",
        "_input_act_width": "input_act_width",
        "_input_act_code": "input_act_code",
    }
    while pending:
        function, module = pending.pop()
        if id(function) in seen:
            continue
        seen.add(id(function))
        if not _exact(function, module):
            return False
        if function.__globals__.get("torch", torch) is not torch:
            return False
        for code in _codes(function.__code__):
            for name in code.co_names:
                value = function.__globals__.get(name)
                if value is None or not callable(value):
                    continue
                if type(value) is FunctionType:
                    source = sys.modules.get(value.__globals__.get("__name__"))
                    if source is None:
                        return False
                    try:
                        inside = (
                            Path(source.__file__).resolve().is_relative_to(kitchen_root)
                        )
                    except (AttributeError, OSError, TypeError, ValueError):
                        return False
                    if not inside or value.__code__.co_qualname != aliases.get(
                        name, name
                    ):
                        return False
                    pending.append((value, source))
                elif getattr(type(value), "__module__", "").startswith("triton"):
                    # JIT/Autotuner executable owners need separate qualification.
                    return False
                elif (
                    not inspect.isbuiltin(value)
                    or getattr(builtins, name, _UNSET) is not value
                ):
                    return False
    return True


def _backend_known(kwargs, *, custom_op=True):
    registry_module = sys.modules[registry.__class__.__module__]
    activation = kwargs.get("input_act")
    functional = torch.nn.functional
    if activation == "swiglu":
        if not _source_exact(
            functional.silu, functional, "silu"
        ) or not inspect.isbuiltin(torch._C._nn.silu):
            return False
    elif activation == "gelu_tanh":
        if (
            not inspect.isbuiltin(functional.gelu)
            or functional.gelu is not torch._C._nn.gelu
        ):
            return False
    elif activation == "rms_norm":
        if (
            not _source_exact(functional.rms_norm, functional, "rms_norm")
            or not inspect.isbuiltin(torch.rms_norm)
            or torch.rms_norm is not torch._C._VariableFunctions.rms_norm
        ):
            return False
    elif activation == "situ":
        if (
            not inspect.isbuiltin(torch.sigmoid)
            or torch.sigmoid is not torch._C._VariableFunctions.sigmoid
        ):
            return False
    if (
        kitchen.registry is not registry
        or type(registry) is not registry_module.BackendRegistry
        or type(registry._constraints) is not dict
        or registry_module.validate_function_call
        is not constraints.validate_function_call
    ):
        return False
    # Constraint callbacks are executable owners too. Do not run a custom
    # validator an extra time merely to inspect an optional optimization.
    for (_backend, name), declared in registry._constraints.items():
        if name != "int8_linear":
            continue
        if (
            type(declared) is not constraints.FunctionConstraints
            or declared.call_rules
            or type(declared.params) is not dict
            or type(declared.default_devices) is not frozenset
            or any(type(value) is not str for value in declared.default_devices)
        ):
            return False
        if any(
            type(item) is not constraints.ParamConstraint
            for item in declared.params.values()
        ):
            return False
        for item in declared.params.values():
            if (
                type(item.shape_rules) is not tuple
                or (item.dtypes is not None and type(item.dtypes) is not frozenset)
                or (item.devices is not None and type(item.devices) is not frozenset)
            ):
                return False
            if item.dtypes is not None and any(
                type(value) is not torch.dtype
                and not any(
                    value is kind for kind in (bool, int, float, str, type(None))
                )
                for value in item.dtypes
            ):
                return False
            if item.devices is not None and any(
                type(value) is not str for value in item.devices
            ):
                return False
            if not all(
                _exact(getattr(item, name), constraints)
                for name in ("check_dtype", "check_device", "check_shape")
            ):
                return False
            for rule in item.shape_rules:
                if type(rule) not in (
                    constraints.MinDims,
                    constraints.ExactDims,
                    constraints.DivisibleBy,
                ) or not _exact(rule.check, constraints):
                    return False
                if type(rule) is constraints.DivisibleBy:
                    if (
                        type(rule.dim) is not int
                        or type(rule.factor) is not int
                        or rule.factor <= 0
                    ):
                        return False
                elif type(rule.ndim) is not int:
                    return False
    if not all(
        _exact(getattr(constraints, name), constraints)
        for name in ("validate_function_call", "validate_param")
    ):
        return False
    module = sys.modules[registry.__class__.__module__]
    if not all(
        _exact(getattr(registry, name), module)
        for name in (
            "get_implementation",
            "get_capable_backend",
            "validate_backend_for_call",
            "is_available",
        )
    ):
        return False
    if custom_op and not _custom_op_known(kwargs["x"].device.type):
        return False
    try:
        implementation = registry.get_implementation("int8_linear", kwargs=kwargs)
    except sys.modules[registry.__class__.__module__].NoCapableBackendError:
        # The original call will still raise its actual error if also invalid.
        return False
    owner = sys.modules.get(getattr(implementation, "__module__", ""))
    if owner is None:
        return False
    try:
        relative = (
            Path(owner.__file__)
            .resolve()
            .relative_to(Path(kitchen.__file__).resolve().parent)
            .as_posix()
        )
    except (AttributeError, OSError, TypeError, ValueError):
        return False
    if relative == "backends/eager/quantization.py":
        selected_mm = getattr(torch, "int8_mm", torch._int_mm)
        expected_mm = getattr(
            torch._C._VariableFunctions, "int8_mm", torch._C._VariableFunctions._int_mm
        )
        if not inspect.isbuiltin(selected_mm) or selected_mm is not expected_mm:
            return False
    return relative in (
        "backends/eager/quantization.py",
        "backends/cuda/__init__.py",
        "backends/triton/quantization.py",
    ) and _python_math_known(implementation, owner)


def _chunk_backends_known(values, rows, *, custom_op=True):
    x = values["x"]
    count = x.numel() // x.shape[-1]
    if count <= rows:
        return True
    flat = x.reshape(count, x.shape[-1])
    # No optional chunk can turn an accepted full-size API into an error.
    return all(
        _backend_known({**values, "x": flat[start : start + rows]}, custom_op=custom_op)
        for start in range(0, count, rows)
    )


def _owner(runtime):
    """Nearest actual Python module call, not a model name or weight filename.

    This avoids changing a nested different MODEL merely because it was invoked
    inside this diffusion call. Unknown/compiled callers retain the full op.
    """
    frame = inspect.currentframe().f_back
    nearest, root_calls = None, 0
    try:
        while frame is not None:
            value = frame.f_locals.get("self")
            if nearest is None and isinstance(value, nn.Module):
                nearest = value
            if value is runtime.network and frame.f_code is runtime.network_code:
                root_calls += 1
            frame = frame.f_back
        if (
            nearest is None
            or runtime.owners.get(id(nearest), (None,))[0] is not nearest
            or root_calls != 1
        ):
            return None
        if runtime.plan.targets == "mlp" and id(nearest) not in runtime.mlp_owners:
            return _OUT_OF_TARGET
        return runtime.owners[id(nearest)][1] or "<root>"
    finally:
        del frame


def _layout_known(weight):
    descriptor = vars(QuantizedTensor).get("layout_cls")
    return (
        type(tensor_base.LAYOUTS) is dict
        and tensor_base.LAYOUTS.get("TensorWiseINT8Layout")
        is tensor_int8.TensorWiseINT8Layout
        and type(descriptor) is property
        and _exact(descriptor.fget, tensor_base)
        and _exact(tensor_base.get_layout_class, tensor_base)
        and _exact(tensor_base._get_layout_from_args, tensor_base)
        and _exact(tensor_int8.TensorWiseINT8Layout.get_plain_tensors, tensor_int8)
        and _exact(tensor_int8._dtype_code, tensor_int8)
        and tensor_int8.QuantizedTensor is QuantizedTensor
        and tensor_base.QuantizedTensor is QuantizedTensor
        and type(weight._params) is tensor_int8.TensorWiseINT8Layout.Params
        and type(weight._qdata) in (torch.Tensor, nn.Parameter)
        and type(weight._params.scale) in (torch.Tensor, nn.Parameter)
    )


def _selected():
    values = _ACTIVE.get()
    return values[-1] if values else None


class _RowMode(TorchDispatchMode):
    def __init__(self, runtime):
        super().__init__()
        self.runtime = runtime

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        runtime = self.runtime
        if _selected() is not runtime or function not in _LINEAR_OPS:
            return function(*args, **kwargs)
        index = _LINEAR_OPS[function]
        if len(args) <= index + 1:
            return function(*args, **kwargs)
        x, weight = args[index : index + 2]
        if (
            type(weight) is not QuantizedTensor
            or weight._layout_cls != "TensorWiseINT8Layout"
        ):
            return function(*args, **kwargs)
        path = _owner(runtime)
        if path is _OUT_OF_TARGET:
            return function(*args, **kwargs)
        if (
            not _layout_known(weight)
            or type(tensor_base._LAYOUT_DISPATCH_TABLE) is not dict
            or type(tensor_base._DISPATCH_TABLE) is not dict
        ):
            runtime.delegated.add("operator:layout_registry_owner_retained")
            return function(*args, **kwargs)
        mapping = tensor_base._LAYOUT_DISPATCH_TABLE.get(function, {})
        if type(mapping) is not dict:
            runtime.delegated.add("operator:dispatch_mapping_owner_retained")
            return function(*args, **kwargs)
        handler = mapping.get(tensor_int8.TensorWiseINT8Layout)
        if (
            path is None
            or isinstance(x, QuantizedTensor)
            or type(x) not in (torch.Tensor, nn.Parameter)
            or x.ndim != 2
            or not _exact(handler, tensor_int8)
            or not _exact(QuantizedTensor.__torch_dispatch__, tensor_base)
            or function in tensor_base._DISPATCH_TABLE
            or tensor_int8.torch is not torch
            or tensor_base.torch is not torch
            or any(
                not isinstance(mode, _RowMode)
                for mode in _get_current_dispatch_mode_stack()
            )
        ):
            runtime.delegated.add("operator:full_original_owner_or_layout")
            return function(*args, **kwargs)
        if function is not torch.ops.aten.linear.default and not getattr(
            weight._params, "transposed", False
        ):
            runtime.delegated.add("operator:nonstandard_rhs_retained")
            return function(*args, **kwargs)
        bias = args[0] if index == 1 else args[2] if len(args) > 2 else None
        parameters = {
            "x": x,
            "weight": weight._qdata,
            "weight_scale": weight._params.scale,
            "bias": bias,
            "out_dtype": x.dtype,
            "convrot": getattr(weight._params, "convrot", False),
            "convrot_groupsize": getattr(weight._params, "convrot_groupsize", 256),
            "input_act": None,
            "input_act_weight": None,
            "input_act_eps": 0.0,
            "residual": None,
            "residual_scale": None,
        }
        if not _backend_known(parameters) or not _chunk_backends_known(
            parameters, runtime.plan.rows
        ):
            runtime.delegated.add("operator:backend_owner_retained")
            return function(*args, **kwargs)
        rows = x.shape[0]
        if rows <= runtime.plan.rows:
            runtime.note(path + ":" + str(function), rows, chunk=False)
            return function(*args, **kwargs)
        output = None
        for start in range(0, rows, runtime.plan.rows):
            stop = min(rows, start + runtime.plan.rows)
            selected = (*args[:index], x[start:stop], *args[index + 1 :])
            value = function(*selected, **kwargs)
            if output is None:
                output = value.new_empty((rows, *value.shape[1:]))
            output[start:stop] = value
            runtime.note(path + ":" + str(function), stop - start, chunk=True)
        runtime.completed_output_bytes += output.numel() * output.element_size()
        return output


class _Overlay:
    def __init__(self, base, **values):
        self.base, self.values = base, values

    def __getattr__(self, name):
        return self.values[name] if name in self.values else getattr(self.base, name)


def _clone(function, **globals_update):
    copied = FunctionType(
        function.__code__,
        {**function.__globals__, **globals_update},
        function.__name__,
        function.__defaults__,
        function.__closure__,
    )
    copied.__kwdefaults__ = function.__kwdefaults__
    return copied


def _public_fused(runtime, path, *args, **kwargs):
    original = kitchen.int8_linear
    if any(
        not isinstance(mode, _RowMode) for mode in _get_current_dispatch_mode_stack()
    ):
        runtime.delegated.add(path + ":fused_dispatch_mode_owner_retained")
        return original(*args, **kwargs)
    if not _exact(original, kitchen):
        runtime.delegated.add(path + ":fused_public_owner_retained")
        return original(*args, **kwargs)
    bound = inspect.signature(original).bind(*args, **kwargs)
    bound.apply_defaults()
    values = dict(bound.arguments)
    x = values["x"]
    if any(
        torch.is_tensor(value) and type(value) not in (torch.Tensor, nn.Parameter)
        for value in values.values()
    ):
        runtime.delegated.add(path + ":fused_tensor_owner_retained")
        return original(*args, **kwargs)
    if (
        not _backend_known(values, custom_op=False)
        or x.ndim < 2
        or not x.shape[-1]
        or not _chunk_backends_known(values, runtime.plan.rows, custom_op=False)
    ):
        runtime.delegated.add(path + ":fused_full_original_backend")
        return original(*args, **kwargs)
    rows = x.numel() // x.shape[-1]
    if rows <= runtime.plan.rows:
        runtime.note(path + ":fused_int8", rows, chunk=False)
        return original(*args, **kwargs)
    residual = values.get("residual")
    scale = values.get("residual_scale")
    if scale is not None and scale.numel() not in (1, values["weight"].shape[0]):
        runtime.delegated.add(path + ":fused_full_residual_scale")
        return original(*args, **kwargs)
    residual_rows = None
    if residual is not None:
        expected = (*x.shape[:-1], values["weight"].shape[0])
        try:
            residual_rows = torch.broadcast_to(residual, expected).reshape(rows, -1)
        except RuntimeError:
            return original(*args, **kwargs)
    flattened, output = x.reshape(rows, x.shape[-1]), None
    for start in range(0, rows, runtime.plan.rows):
        stop = min(rows, start + runtime.plan.rows)
        part = {**values, "x": flattened[start:stop]}
        if residual_rows is not None:
            part["residual"] = residual_rows[start:stop]
        value = original(**part)
        if output is None:
            output = value.new_empty((rows, value.shape[-1]))
        output[start:stop] = value
        runtime.note(path + ":fused_int8", stop - start, chunk=True)
    runtime.completed_output_bytes += output.numel() * output.element_size()
    return output.reshape(*x.shape[:-1], output.shape[-1])


def _fused_linear(runtime, path, linear, *args, **kwargs):
    # Keep unknown module processing, callbacks and the existing Core fused
    # semantics intact; do not silently force_forward or clear any patch.
    callbacks = any(
        getattr(linear, key, ()) for key in ("weight_function", "bias_function")
    )
    owner_unknown = (
        _owner(runtime) is None
        or core.comfy is not sys.modules["comfy"]
        or core.comfy.ops is not ops
        or ops.QuantizedTensor is not QuantizedTensor
        or tensor_int8.ck is not kitchen
        or not _module_call_native(linear)
        or getattr(linear, "pre_quant_scale", None) is not None
        or callbacks
        or getattr(linear, "weight_lowvram_function", None) is not None
        or getattr(linear, "bias_lowvram_function", None) is not None
        or bool(linear._forward_hooks or linear._forward_pre_hooks)
        or not _source_exact(_function(linear.forward), ops)
    )
    if (
        owner_unknown
        or not _source_exact(ops.linear_input_act, ops, "linear_input_act")
        or not _source_exact(
            ops._try_quantized_linear_input_act, ops, "_try_quantized_linear_input_act"
        )
    ):
        runtime.delegated.add(path + ":fused_full_module_owner")
        return ops.linear_input_act(linear, *args, **kwargs)

    def prepared(
        x, weight, bias, input_act, act_weight, act_eps, residual, residual_scale
    ):
        if (
            mm.in_training
            or type(weight) is not QuantizedTensor
            or not _layout_known(weight)
            or weight.layout_cls is not tensor_int8.TensorWiseINT8Layout
            or not _exact(QuantizedTensor.linear_input_act, tensor_base)
            or not _exact(
                tensor_int8.TensorWiseINT8Layout.linear_input_act, tensor_int8
            )
            or not _exact(
                tensor_int8.TensorWiseINT8Layout.get_plain_tensors, tensor_int8
            )
        ):
            runtime.delegated.add(path + ":fused_full_prepared_owner")
            return ops._try_quantized_linear_input_act(
                weight=weight,
                x=x,
                bias=bias,
                input_act=input_act,
                act_weight=act_weight,
                act_eps=act_eps,
                residual=residual,
                residual_scale=residual_scale,
            )
        with ops._input_act_weight(x, input_act, act_weight, act_eps) as (
            act_weight,
            act_eps,
        ):
            method = _function(tensor_int8.TensorWiseINT8Layout.linear_input_act)
            local = _clone(
                method,
                ck=_Overlay(
                    kitchen,
                    int8_linear=lambda *a, **k: _public_fused(runtime, path, *a, **k),
                ),
            )
            return local(
                tensor_int8.TensorWiseINT8Layout,
                weight,
                x,
                bias,
                input_act,
                act_weight,
                act_eps,
                residual,
                residual_scale,
            )

    return _clone(ops.linear_input_act, _try_quantized_linear_input_act=prepared)(
        linear, *args, **kwargs
    )


@contextmanager
def int8_row_scope(network, plan=None, *, phase="SINGLE"):
    plan = Int8RowPlan() if plan is None else plan
    if type(plan) is not Int8RowPlan:
        raise TypeError("Expected Int8RowPlan")
    runtime = Int8Runtime(network, plan, str(phase))
    started, saved, token = time.perf_counter(), [], None
    if not plan.rows:
        token = _ACTIVE.set((*_ACTIVE.get(), runtime))
        try:
            yield runtime
        finally:
            _ACTIVE.reset(token)
            runtime.elapsed = time.perf_counter() - started
        return
    with shared_network_lease(network) as acquired:
        if not acquired:
            runtime.delegated.add("network:shared_lease_unavailable")
            token = _ACTIVE.set((*_ACTIVE.get(), runtime))
            try:
                yield runtime
            finally:
                _ACTIVE.reset(token)
                runtime.elapsed = time.perf_counter() - started
            return
        try:
            runtime.owners = {
                id(module): (module, name) for name, module in network.named_modules()
            }
            runtime.network_code = getattr(
                _function(network._call_impl), "__code__", None
            )
            for module, path in tuple(runtime.owners.values()):
                if isinstance(getattr(module, "weight", None), QuantizedTensor) and any(
                    getattr(module, key, ())
                    for key in ("weight_function", "bias_function")
                ):
                    runtime.delegated.add(
                        (path or "<root>") + ":actual_weight_callbacks_preserved"
                    )
                if not isinstance(module, core.MLP):
                    continue
                runtime.mlp_owners.update((id(module), id(module.fc1), id(module.fc2)))
                original = module.forward
                function = _function(original)
                existing = _DISPATCHERS.get(id(function))
                if (
                    existing is not None
                    and existing[0] is function
                    and existing[1] is module
                ):
                    continue  # The outer dispatcher selects the innermost request.
                if (
                    not _source_exact(function, core, "MLP.forward")
                    or not _module_call_native(module)
                    or core.comfy is not sys.modules["comfy"]
                    or core.comfy.ops is not ops
                ):
                    runtime.delegated.add(
                        (path or "<root>") + ":fused_mlp_owner_retained"
                    )
                    continue

                def make_dispatch(original, function, path):
                    def dispatch(self, *args, **kwargs):
                        selected = _selected()
                        if (
                            selected is None
                            or selected.network is not network
                            or not selected.plan.rows
                        ):
                            return original(*args, **kwargs)
                        if (
                            core.comfy is not sys.modules["comfy"]
                            or core.comfy.ops is not ops
                            or not _source_exact(function, core, "MLP.forward")
                        ):
                            selected.delegated.add(path + ":changed_mlp_owner_retained")
                            return original(*args, **kwargs)
                        local_ops = _Overlay(
                            ops,
                            linear_input_act=lambda linear, *a, **k: _fused_linear(
                                selected, path, linear, *a, **k
                            ),
                        )
                        return _clone(
                            function, comfy=_Overlay(core.comfy, ops=local_ops)
                        )(self, *args, **kwargs)

                    return dispatch

                previous = vars(module).get("forward", _UNSET)
                dispatcher = MethodType(make_dispatch(original, function, path), module)
                _DISPATCHERS[id(dispatcher.__func__)] = dispatcher.__func__, module
                saved.append((module, previous, dispatcher))
                module.forward = dispatcher
            token = _ACTIVE.set((*_ACTIVE.get(), runtime))
            with _RowMode(runtime):
                yield runtime
        finally:
            if token is not None:
                _ACTIVE.reset(token)
            for module, previous, dispatcher in reversed(saved):
                _DISPATCHERS.pop(id(dispatcher.__func__), None)
                if vars(module).get("forward", _UNSET) is not dispatcher:
                    runtime.delegated.add("scope:later_owner_preserved")
                elif previous is _UNSET:
                    delattr(module, "forward")
                else:
                    module.forward = previous
            runtime.elapsed = time.perf_counter() - started
