"""Opt-in, request-scoped H3 projection working sets, independently authored.

Actual selected modules perform every projection. No raw-weight shortcut,
prepared-weight cache, global class patch, sampling rule or spatial crop lives
here. Full K/V remain visible; only proven dense queries are sliced. Ordinary
requests seeing a temporary dispatcher use the original callable unchanged.
"""

from __future__ import annotations

import ast
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
import inspect
from pathlib import Path
import sys
import threading
import time
from types import CodeType, FunctionType, MethodType
import weakref

import torch
from torch import nn

import comfy.model_management
import comfy.ops
import comfy.quant_ops
from comfy.ldm.minimax import model as core
from comfy.ldm.modules import attention as core_attention

from .source_code_identity import executable_code_equal


SCHEMA = "t8.h3.projection_chunks.v1"
_ACTIVE = ContextVar("t8_projection_requests", default=())
_REGISTRY = {}
_REGISTRY_LOCK = threading.RLock()
_ABSENT = object()
_NATIVE_BLOCK = core.DiTBlock.forward
_NATIVE_ATTN = core.Attention.forward
_NATIVE_FINAL = core.FinalLayer.forward
_NATIVE_ADALN = core.AdalnProj.forward
_SOURCE_CONTRACTS = {}
_CORE_HELPERS = {
    name: getattr(core, name)
    for name in (
        "_mod_row",
        "_mod_scale_shift",
        "_mod_gate",
        "_pdd_head",
        "time_shift_sigma",
    )
}


@dataclass(frozen=True)
class ProjectionPlan:
    rows: int = 1024
    query_rows: int = 1024
    stage_queries: bool = False
    norm_qkv: bool = True
    output: bool = True
    embedding: bool = True
    final: bool = True

    def __post_init__(self):
        for name in ("rows", "query_rows"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(name + " must be a positive integer")
        for name in ("stage_queries", "norm_qkv", "output", "embedding", "final"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(name + " must be boolean")


@dataclass
class ProjectionRuntime:
    plan: ProjectionPlan | None
    phase: str
    calls: dict = field(default_factory=dict)
    delegated: set = field(default_factory=set)
    query_spans: list = field(default_factory=list)
    max_live_norm_rows: int = 0
    persistent_kv_bytes: int = 0
    staged_q_bytes: int = 0
    elapsed: float = 0.0
    lease_wait: float = 0.0

    def note(self, path, rows, *, chunk=False):
        count = self.calls.setdefault(
            path, {"calls": 0, "rows": 0, "max_rows": 0, "chunks": 0}
        )
        count["calls"] += 1
        count["rows"] += rows
        count["max_rows"] = max(count["max_rows"], rows)
        count["chunks"] += int(chunk)

    def report(self):
        return {
            "schema": SCHEMA,
            "phase": self.phase,
            "plan": asdict(self.plan) if self.plan is not None else None,
            "calls": {key: dict(value) for key, value in self.calls.items()},
            "delegated": sorted(self.delegated),
            "partial": bool(self.delegated),
            "portable_cache_identity": False,
            "query_spans": list(self.query_spans),
            "max_live_norm_rows": self.max_live_norm_rows,
            "persistent_kv_bytes": self.persistent_kv_bytes,
            "staged_q_bytes": self.staged_q_bytes,
            "cpu_dispatch_wall_seconds": self.elapsed - self.lease_wait,
            "lease_wait_seconds": self.lease_wait,
            "allocator_peak_measured": False,
            "performance_or_media_qualified": False,
        }


class _Shared:
    def __init__(self):
        self.lock = threading.RLock()
        self.depth = 0
        self.saved = {}
        self.dispatchers = {}


def _selected(shared):
    return next(
        (runtime for owner, runtime in reversed(_ACTIVE.get()) if owner is shared), None
    )


def _function(method):
    return getattr(method, "__func__", method)


def _codes(code):
    yield code
    for value in code.co_consts:
        if type(value) is CodeType:
            yield from _codes(value)


def _default_matches(value, node):
    # Authenticate only plain source literals; never call user-defined equality
    # or execute an expression found in a source default.
    if isinstance(node, ast.Constant):
        expected = node.value
        return type(value) is type(expected) and (
            value is expected if expected is None else value == expected
        )
    if isinstance(node, (ast.Dict, ast.List, ast.Tuple)):
        if isinstance(node, ast.Dict):
            return type(value) is dict and not value and not node.keys
        kind = list if isinstance(node, ast.List) else tuple
        return (
            type(value) is kind
            and len(value) == len(node.elts)
            and all(
                _default_matches(item, declaration)
                for item, declaration in zip(value, node.elts)
            )
        )
    return False


def _defaults_match(function, definition):
    declared = definition.args.defaults
    live = function.__defaults__
    if declared:
        if not (
            type(live) is tuple
            and len(live) == len(declared)
            and all(
                _default_matches(value, node) for value, node in zip(live, declared)
            )
        ):
            return False
    elif live is not None:
        return False
    keyword = {
        arg.arg: value
        for arg, value in zip(definition.args.kwonlyargs, definition.args.kw_defaults)
        if value is not None
    }
    live_keyword = function.__kwdefaults__
    return (
        live_keyword is None
        if not keyword
        else type(live_keyword) is dict
        and set(live_keyword) == set(keyword)
        and all(
            _default_matches(live_keyword[key], value) for key, value in keyword.items()
        )
    )


def _source_exact(function, module, qualname=None):
    """Compile trusted installed source, never execute it or trust live markers.

    Capturing a function at this module's import is insufficient: another owner
    may have changed its code before that import, or later in place while keeping
    the same object and filename. Both cases must delegate the original call.
    """
    if type(function) is not FunctionType or function.__globals__ is not vars(module):
        return False
    try:
        if function.__closure__ is not None:
            if function.__code__.co_freevars != ("__class__",):
                return False
            selected_class = module
            for name in function.__code__.co_qualname.split(".")[:-1]:
                selected_class = vars(selected_class).get(name)
                if selected_class is None:
                    return False
            if (
                len(function.__closure__) != 1
                or function.__closure__[0].cell_contents is not selected_class
            ):
                return False
        source = inspect.getsourcefile(function)
        if source is None or Path(source).resolve() != Path(module.__file__).resolve():
            return False
        with _REGISTRY_LOCK:
            contract = _SOURCE_CONTRACTS.get(module)
            if contract is None:
                payload = Path(source).read_bytes()
                compiled = tuple(
                    _codes(compile(payload, source, "exec", dont_inherit=True))
                )
                definitions = tuple(
                    node
                    for node in ast.walk(ast.parse(payload, filename=source))
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                )
                contract = compiled, definitions
                _SOURCE_CONTRACTS[module] = contract
        code = function.__code__
        if qualname is not None and code.co_qualname != qualname:
            return False
        selected = [
            candidate
            for candidate in contract[0]
            if candidate.co_qualname == code.co_qualname
            and executable_code_equal(code, candidate)
        ]
        if len(selected) != 1:
            return False
        definitions = [
            node
            for node in contract[1]
            if node.name == code.co_name
            and min([node.lineno, *(item.lineno for item in node.decorator_list)])
            == code.co_firstlineno
        ]
        return len(definitions) == 1 and _defaults_match(function, definitions[0])
    except (AttributeError, OSError, TypeError, SyntaxError, ValueError):
        return False


def _module_call_native(module):
    """Native forward code alone does not authenticate a whole module call."""
    if (
        not isinstance(module, nn.Module)
        or getattr(module, "_compiled_call_impl", None) is not None
    ):
        return False
    owner = nn.modules.module
    if not _source_exact(type(module).__call__, owner, "Module._wrapped_call_impl"):
        return False
    for name in ("_call_impl", "_slow_forward"):
        method = getattr(module, name, None)
        if getattr(method, "__self__", None) is not module or not _source_exact(
            _function(method), owner, "Module." + name
        ):
            return False
    return True


def _row_local(method):
    """Exact source owners, not user-selected class or checkpoint names."""
    function = _function(method)
    owner = getattr(method, "__self__", None)
    if not _module_call_native(owner):
        return False
    if isinstance(owner, nn.RMSNorm):
        if len(owner.normalized_shape) != 1:
            return False
    elif not isinstance(owner, nn.Linear) and not (
        isinstance(owner, comfy.ops.MixedPrecisionOp)
        and getattr(function, "__qualname__", None)
        == "mixed_precision_ops.<locals>.MixedPrecisionOps.Linear.forward"
    ):
        return False
    # Actual selected weight transformations must retain their original call
    # count. Arbitrary callables may be stateful or depend on the whole input;
    # no marker, filename or observed one-off result proves otherwise.
    for parameter in ("weight", "bias"):
        callbacks = getattr(owner, parameter + "_function", ())
        if type(callbacks) not in (list, tuple) or callbacks:
            return False
        if getattr(owner, parameter + "_lowvram_function", None) is not None:
            return False
    # The native outer forward dynamically dispatches these actual methods.
    # A user-selected inner input transform may couple rows even while that
    # outer callable remains byte-for-byte native.
    for name in ("forward_comfy_cast_weights", "_forward"):
        delegate = getattr(owner, name, None)
        if delegate is not None:
            function_delegate = _function(delegate)
            delegate_module = sys.modules.get(
                getattr(function_delegate, "__module__", "")
            )
            if delegate_module is not comfy.ops or not _source_exact(
                function_delegate, comfy.ops
            ):
                return False
    module = sys.modules.get(getattr(function, "__module__", ""))
    return any(
        module is allowed
        for allowed in (
            comfy.ops,
            sys.modules[nn.Linear.__module__],
            sys.modules[nn.RMSNorm.__module__],
        )
    ) and _source_exact(function, module)


def _helpers_current():
    return all(
        getattr(core, name) is function and _source_exact(function, core, name)
        for name, function in _CORE_HELPERS.items()
    )


def _hooks(module):
    return bool(
        module._forward_hooks or module._forward_pre_hooks or module._backward_hooks
    )


def _global_hooks():
    return bool(
        nn.modules.module._global_forward_hooks
        or nn.modules.module._global_forward_pre_hooks
    )


def _original(shared, module):
    return shared.saved[module][0] if module in shared.saved else module.forward


def _batch(runtime, path, original, x, *args, **kwargs):
    if x.ndim < 2 or not _row_local(original):
        runtime.delegated.add(path + ":unknown_row_contract")
        return original(x, *args, **kwargs)
    count = x.shape[0]
    if count <= runtime.plan.rows:
        runtime.note(path, count)
        return original(x, *args, **kwargs)
    output = None
    for start in range(0, count, runtime.plan.rows):
        stop = min(count, start + runtime.plan.rows)
        value = original(x[start:stop], *args, **kwargs)
        if output is None:
            output = value.new_empty((count, *value.shape[1:]))
        output[start:stop] = value
        runtime.note(path, stop - start, chunk=True)
    return output


def slice_segment(segment, start, stop):
    """Keep original packed coordinates, slicing tensor row labels relatively."""
    a, b, row = segment
    if not 0 <= a <= start <= stop <= b:
        raise ValueError("Projection subrange must stay inside its original segment")
    if torch.is_tensor(row):
        if row.ndim != 1 or row.numel() != b - a or row.dtype != torch.long:
            raise ValueError("Per-token modulation rows must be a matching LongTensor")
        row = row[start - a : stop - a]
    elif type(row) is not int or row < 0:
        raise ValueError("Modulation row must be a nonnegative integer or LongTensor")
    return start, stop, row


def _segments(segments, total):
    cursor = 0
    for segment in segments:
        a, b, _ = segment
        if a != cursor or not a < b <= total:
            raise ValueError(
                "Modulation segments must contiguously cover all packed rows"
            )
        slice_segment(segment, a, b)
        cursor = b
    if cursor != total:
        raise ValueError("Modulation segments do not cover all packed rows")


def _native_dense(preferred, options):
    from .vdn_attention_compat import _factory_closure

    backend = core_attention.attention_pytorch
    state = _factory_closure(backend, core_attention.wrap_attn, "wrapper")
    backend_current = (
        _source_exact(core_attention.wrap_attn, core_attention, "wrap_attn")
        and state is not None
        and set(state) == {"func", "wrapper"}
        and state["wrapper"] is backend
        and _source_exact(state["func"], core_attention, "attention_pytorch")
        and _source_exact(
            comfy.ops.scaled_dot_product_attention,
            comfy.ops,
            "scaled_dot_product_attention",
        )
        and nn.functional.scaled_dot_product_attention
        is torch._C._nn.scaled_dot_product_attention
        and inspect.isbuiltin(nn.functional.scaled_dot_product_attention)
    )
    selected = getattr(preferred, "function", None)
    patches = options.get("patches")
    return (
        backend_current
        and core.optimized_attention is backend
        and getattr(core.optimized_attention, "container_function", None) is None
        and (selected is None or selected is core_attention.attention_pytorch)
        and "optimized_attention_override" not in options
        and (patches is None or isinstance(patches, dict) and len(patches) == 0)
    )


def dense_attention(
    runtime, q, k, v, heads, *, preferred=None, options=None, mask=None
):
    """Original full K/V and original-index mask slices for qualified SDPA only."""
    options = {} if options is None else options
    total = q.shape[-2]
    allowed = _native_dense(preferred, options)
    if mask is not None and (mask.ndim < 2 or mask.shape[-2] not in (1, total)):
        allowed = False
    if not allowed:
        runtime.delegated.add("attention:full_query_route")
        if runtime.plan.stage_queries:
            runtime.delegated.add("query_staging:full_query_route")
        return core.optimized_attention(
            core_attention.AttentionTensorContainer(q.to(k.device)),
            core_attention.AttentionTensorContainer(k),
            core_attention.AttentionTensorContainer(v),
            heads,
            preferred_attention=preferred,
            mask=mask,
            skip_reshape=True,
            transformer_options=options,
        )
    output = None
    for start in range(0, total, runtime.plan.query_rows):
        stop = min(total, start + runtime.plan.query_rows)
        selected_mask = mask
        if mask is not None and mask.shape[-2] == total:
            selected_mask = mask[..., start:stop, :]
        value = core.optimized_attention(
            core_attention.AttentionTensorContainer(q[..., start:stop, :].to(k.device)),
            core_attention.AttentionTensorContainer(k),
            core_attention.AttentionTensorContainer(v),
            heads,
            preferred_attention=preferred,
            mask=selected_mask,
            skip_reshape=True,
            transformer_options=options,
        )
        if output is None:
            output = value.new_empty((value.shape[0], total, value.shape[-1]))
        output[:, start:stop] = value
        runtime.query_spans.append((start, stop, k.shape[-2]))
        runtime.note("attention.query", stop - start, chunk=True)
    return output


def _normal_qk(attn, q, k, rope, x):
    rows = q.shape[0]
    if rope is None:
        return attn.q_norm(q.view(rows, attn.heads, attn.head_dim)), attn.k_norm(
            k.view(rows, attn.heads, attn.head_dim)
        )
    q = q.view(1, rows, attn.heads, attn.head_dim)
    k = k.view(1, rows, attn.heads, attn.head_dim)
    qw = comfy.model_management.cast_to(attn.q_norm.weight, device=x.device)
    kw = comfy.model_management.cast_to(attn.k_norm.weight, device=x.device)
    if comfy.model_management.in_training:
        q, k = comfy.quant_ops.ck.rms_rope_split_half(
            q, k, rope, qw, kw, epsilon=attn.q_norm.eps, rot_dim=rope.shape[-3] * 2
        )
    else:
        comfy.quant_ops.ck.rms_rope_split_half_(
            q, k, rope, qw, kw, epsilon=attn.q_norm.eps, rot_dim=rope.shape[-3] * 2
        )
    return q[0], k[0]


def _native_rope_owner(rope):
    if rope is None:
        return True
    kitchen = getattr(comfy.quant_ops, "ck", None)
    name = "rms_rope_split_half" if comfy.model_management.in_training else "rms_rope_split_half_"
    # A legal public helper may couple sequence rows. Keep its original full
    # call instead of changing it into one invocation per projection chunk.
    return kitchen is not None and _source_exact(getattr(kitchen, name, None), kitchen, name)


def _block(
    shared,
    runtime,
    block,
    original,
    x,
    t_emb,
    mod_segments,
    rope_freqs,
    transformer_options=None,
    attention=None,
):
    options = {} if transformer_options is None else transformer_options
    attn = block.attn
    trusted = (
        _function(original) is _NATIVE_BLOCK
        and _source_exact(_NATIVE_BLOCK, core, "DiTBlock.forward")
        and _function(_original(shared, attn)) is _NATIVE_ATTN
        and _source_exact(_NATIVE_ATTN, core, "Attention.forward")
        and _helpers_current()
        and _native_rope_owner(rope_freqs)
        and _module_call_native(block)
        and _module_call_native(attn)
        and attention is None
        and all(
            _row_local(_original(shared, module))
            for module in (block.norm1, attn.qkv_proj, attn.q_norm, attn.k_norm)
        )
        and all(
            not _hooks(module)
            for module in (
                block.norm1,
                attn,
                attn.qkv_proj,
                attn.q_norm,
                attn.k_norm,
                attn.out_proj,
            )
        )
    )
    if not runtime.plan.norm_qkv:
        return original(
            x,
            t_emb,
            mod_segments,
            rope_freqs,
            transformer_options=options,
            attention=attention,
        )
    if not trusted or _global_hooks():
        runtime.delegated.add("block:fused_norm_qkv_delegate")
        return original(
            x,
            t_emb,
            mod_segments,
            rope_freqs,
            transformer_options=options,
            attention=attention,
        )
    try:
        _segments(mod_segments, x.shape[0])
    except (TypeError, ValueError):
        runtime.delegated.add("block:nonstandard_mod_segments")
        return original(
            x,
            t_emb,
            mod_segments,
            rope_freqs,
            transformer_options=options,
            attention=attention,
        )
    shift, scale, gate, shift_mlp, scale_mlp, gate_mlp = block.adaln_proj(t_emb)
    count, width = x.shape[0], attn.heads * attn.head_dim
    stage = runtime.plan.stage_queries and _native_dense(attn.comfy_attention, options)
    q_all = k_all = v_all = None
    for segment in mod_segments:
        a, b, _ = segment
        for start in range(a, b, runtime.plan.rows):
            stop = min(b, start + runtime.plan.rows)
            local = slice_segment(segment, start, stop)
            h = block.norm1(x[start:stop])
            row = local[2]
            h.mul_(1 + core._mod_row(scale, row, h.dtype)).add_(
                core._mod_row(shift, row, h.dtype)
            )
            q, k, v = attn.qkv_proj(h).split(width, dim=-1)
            q, k = _normal_qk(
                attn, q, k, None if rope_freqs is None else rope_freqs[:, start:stop], h
            )
            v = v.view(stop - start, attn.heads, attn.head_dim)
            if q_all is None:
                q_all = q.new_empty(
                    (count, attn.heads, attn.head_dim),
                    device="cpu" if stage else q.device,
                )
                k_all = k.new_empty((count, attn.heads, attn.head_dim))
                v_all = v.new_empty((count, attn.heads, attn.head_dim))
            q_all[start:stop].copy_(q)
            k_all[start:stop].copy_(k)
            v_all[start:stop].copy_(v)
            runtime.max_live_norm_rows = max(runtime.max_live_norm_rows, stop - start)
            runtime.note("norm_qkv.fused", stop - start, chunk=True)
            del h, q, k, v
    runtime.persistent_kv_bytes = max(
        runtime.persistent_kv_bytes,
        k_all.numel() * k_all.element_size() + v_all.numel() * v_all.element_size(),
    )
    if stage:
        runtime.staged_q_bytes += q_all.numel() * q_all.element_size()
    result = dense_attention(
        runtime,
        q_all.transpose(0, 1).unsqueeze(0),
        k_all.transpose(0, 1).unsqueeze(0),
        v_all.transpose(0, 1).unsqueeze(0),
        attn.heads,
        preferred=attn.comfy_attention,
        options=options,
    ).squeeze(0)
    del q_all, k_all, v_all
    # Output batching calls the selected projection (and its input processing).
    attended = attn.out_proj(result)
    del result
    x = core._mod_gate(x, gate, attended, mod_segments)
    del attended
    h = core._mod_scale_shift(block.norm2(x), shift_mlp, scale_mlp, mod_segments)
    return core._mod_gate(x, gate_mlp, block.mlp(h), mod_segments)


def _known_final(original):
    if (
        not _source_exact(_NATIVE_FINAL, core, "FinalLayer.forward")
        or not _helpers_current()
    ):
        return False
    if _function(original) is _NATIVE_FINAL:
        return True
    from . import hyperflow_runtime_advanced as hyperflow
    from .vdn_attention_compat import _factory_closure

    state = _factory_closure(original, hyperflow._install_two_time, "final_forward")
    return (
        _source_exact(hyperflow._install_two_time, hyperflow, "_install_two_time")
        and state is not None
        and set(state) == {"owner", "final_original", "remap"}
        and set(vars(original)) == {"_t8_hyperflow_owner", "_t8_hyperflow_inner"}
        and original._t8_hyperflow_owner == state["owner"]
        and original._t8_hyperflow_inner is state["final_original"]
        and _function(state["final_original"]) is _NATIVE_FINAL
    )


def _final(runtime, layer, original, x, t_emb, video_seg, audio_seg, *args, **kwargs):
    if not runtime.plan.final:
        return original(x, t_emb, video_seg, audio_seg, *args, **kwargs)
    if (
        not _known_final(original)
        or _global_hooks()
        or any(_hooks(module) for module in layer.modules())
        or any(not _module_call_native(module) for module in layer.modules())
        or not _row_local(layer.norm.forward)
        or _function(layer.adaln_proj.forward) is not _NATIVE_ADALN
        or not _source_exact(_NATIVE_ADALN, core, "AdalnProj.forward")
        or not all(
            _row_local(module.forward)
            for module in (layer.adaln_proj.linear, layer.video_out, layer.audio_out)
        )
    ):
        runtime.delegated.add("final:original_delegate")
        return original(x, t_emb, video_seg, audio_seg, *args, **kwargs)
    # Bind the original ABI once before execution; never catch/retry an inner TypeError.
    from .h3_core_compat import _final_layer_schedule_parameters

    _final_layer_schedule_parameters(_function(original))
    try:
        va, vb, _ = slice_segment(video_seg, video_seg[0], video_seg[1])
        aa, ab, _ = slice_segment(audio_seg, audio_seg[0], audio_seg[1])
    except (TypeError, ValueError):
        runtime.delegated.add("final:nonstandard_mod_segments")
        return original(x, t_emb, video_seg, audio_seg, *args, **kwargs)
    outputs = [None, None]
    total = max(vb - va, ab - aa)
    for offset in range(0, max(total, 1), runtime.plan.rows):
        vr = slice_segment(
            video_seg, min(va + offset, vb), min(va + offset + runtime.plan.rows, vb)
        )
        ar = slice_segment(
            audio_seg, min(aa + offset, ab), min(aa + offset + runtime.plan.rows, ab)
        )
        values = original(x, t_emb, vr, ar, *args, **kwargs)
        for index, (value, seg, length) in enumerate(
            zip(values, (vr, ar), (vb - va, ab - aa))
        ):
            if outputs[index] is None:
                outputs[index] = value.new_empty((length, *value.shape[1:]))
            start = min(offset, length)
            outputs[index][start : start + seg[1] - seg[0]] = value
        runtime.note("final.delegate", max(vr[1] - vr[0], ar[1] - ar[0]), chunk=True)
    return tuple(outputs)


def _install(shared, network):
    def bind(module, path, role):
        if module in shared.saved:
            return
        original = module.forward
        shared.saved[module] = original, module.__dict__.get("forward", _ABSENT)

        def dispatch(self, *args, **kwargs):
            runtime = _selected(shared)
            if runtime is None or runtime.plan is None:
                return original(*args, **kwargs)
            if role == "block":
                return _block(shared, runtime, self, original, *args, **kwargs)
            if role == "final":
                return _final(runtime, self, original, *args, **kwargs)
            enabled = (
                runtime.plan.embedding
                if role == "embedding"
                else runtime.plan.output
                if role == "output"
                else runtime.plan.norm_qkv
            )
            if not enabled:
                return original(*args, **kwargs)
            if not args or not torch.is_tensor(args[0]):
                runtime.delegated.add(path + ":call_signature_delegate")
                return original(*args, **kwargs)
            return _batch(runtime, path, original, *args, **kwargs)

        module.forward = MethodType(dispatch, module)
        shared.dispatchers[module] = module.forward

    # Only established H3 surfaces; no MLP or prepared-weight state touched.
    named_modules = getattr(network, "named_modules", None)
    if not callable(named_modules):
        return
    for name, module in named_modules():
        if isinstance(module, core.DiTBlock):
            bind(module, name, "block")
        elif isinstance(module, core.FinalLayer):
            bind(module, name, "final")
        if name.endswith(("norm1", "norm2", "q_norm", "k_norm", "final_norm")):
            bind(module, name, "norm")
        elif name.endswith("qkv_proj"):
            bind(module, name, "qkv")
        elif name.endswith("out_proj"):
            bind(module, name, "output")
        elif name in ("video_patch_proj", "audio_patch_proj", "condition_proj"):
            bind(module, name, "embedding")


def _shared_for_network(network):
    """Identity-keyed lease lookup; never invoke a user module's equality."""
    with _REGISTRY_LOCK:
        key = id(network)
        entry = _REGISTRY.get(key)
        if entry is not None and entry[0]() is network:
            shared = entry[1]
        elif getattr(type(network), "__hash__", None) is None:
            # nn.Module.named_modules itself needs hashability to traverse.
            shared = None
        else:

            def release(reference):
                with _REGISTRY_LOCK:
                    current = _REGISTRY.get(key)
                    if current is not None and current[0] is reference:
                        del _REGISTRY[key]

            try:
                reference = weakref.ref(network, release)
            except TypeError:
                shared = None
            else:
                shared = _Shared()
                _REGISTRY[key] = reference, shared
    return shared


@contextmanager
def shared_network_lease(network):
    """Serialize cooperating operators without installing or disabling a plan.

    The boolean indicates only whether this identity-based lease was acquired;
    it is not portable, thread-safe ModelPatcher weight loading, or an operator
    completion receipt. R1/R2 use the same reentrant lock in either order.
    """
    shared = _shared_for_network(network)
    if shared is None:
        yield False
    else:
        with shared.lock:
            yield True


@contextmanager
def projection_scope(network, plan=None, *, phase="SINGLE"):
    """A shared-network lease plus context-local dispatch, not a clone-isolation claim."""
    if plan is not None and type(plan) is not ProjectionPlan:
        raise TypeError("Expected ProjectionPlan or None")
    runtime = ProjectionRuntime(plan, str(phase))
    started = time.perf_counter()
    shared = _shared_for_network(network)
    if shared is None:
        runtime.delegated.add("network:shared_lease_unavailable")
        try:
            yield runtime
        finally:
            runtime.elapsed = time.perf_counter() - started
        return
    with shared.lock:
        runtime.lease_wait = time.perf_counter() - started
        token = None
        try:
            if not shared.saved and plan is not None:
                _install(shared, network)
                if not shared.saved:
                    runtime.delegated.add("network:no_projection_surfaces")
            shared.depth += 1
            token = _ACTIVE.set((*_ACTIVE.get(), (shared, runtime)))
            yield runtime
        finally:
            if token is not None:
                _ACTIVE.reset(token)
                shared.depth -= 1
            if not shared.depth:
                for module, (_original_forward, previous) in reversed(
                    tuple(shared.saved.items())
                ):
                    if (
                        module.__dict__.get("forward", _ABSENT)
                        is not shared.dispatchers[module]
                    ):
                        runtime.delegated.add("scope:later_owner_preserved")
                    elif previous is _ABSENT:
                        delattr(module, "forward")
                    else:
                        module.forward = previous
                shared.saved.clear()
                shared.dispatchers.clear()
            runtime.elapsed = time.perf_counter() - started
