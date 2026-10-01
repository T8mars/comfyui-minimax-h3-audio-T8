"""Scoped KJ Sage selector adapter; never execute an unknown override.

This covers the ordinary KJ selector, not the H3 direct-forward memory patch.
Float Relay bias is never passed to kernels that merely accept **kwargs: the
audited CUDA/auto variants ignore it. Biased rows use the separately audited,
scoped Sage Triton kernel, or explicit SDPA if that kernel is unavailable.
"""
from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass, field
import hashlib
import inspect
from pathlib import Path
import sys
from types import CodeType, FunctionType

import torch
from comfy.ldm.modules import attention as core_attention


_KJ_AST = {
    "get_sage_func": "b21a024db81666aa48b98d43baa144fdfa2cbc58f2bf22f0d524650f5f34b026",
    "PathchSageAttentionKJ": "adaeb6167e3f57019fdf92c1f99c406a757b4e852dbbeac728b1caba9f17ef74",
}
_MASKED_TRITON_AST = "d0dec2f5591419bb9dc431153370a071c308ccbc33a7168c81a3344abe41d7d4"


def _codes(code):
    yield code
    for item in code.co_consts:
        if isinstance(item, CodeType):
            yield from _codes(item)


def _cells(function):
    try:
        return {name: cell.cell_contents for name, cell in
                zip(function.__code__.co_freevars, function.__closure__ or ())}
    except (AttributeError, ValueError):
        return {}


def _ast_hash(node):
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


def _live_code_matches(function, compiled, module):
    from .source_code_identity import executable_code_equal

    return (isinstance(function, FunctionType)
            and function.__globals__ is vars(module)
            and any(executable_code_equal(function.__code__, code) for code in compiled))


def _audited_mask_kernel(kernel):
    """Source and live code evidence, not signature/name/marker guessing."""
    if not isinstance(kernel, FunctionType):
        return False
    try:
        path = inspect.getsourcefile(kernel)
        payload = Path(path).read_bytes()
        definitions = [node for node in ast.parse(payload).body
                       if isinstance(node, ast.FunctionDef) and node.name == kernel.__name__]
        module = sys.modules.get(kernel.__module__)
        return (len(definitions) == 1 and _ast_hash(definitions[0]) == _MASKED_TRITON_AST
                and module is not None
                and _live_code_matches(kernel, _codes(compile(payload, path, "exec", dont_inherit=True)), module))
    except (OSError, TypeError, SyntaxError):
        return False


@dataclass
class KJRelayBackend:
    mode: str
    kernel: object
    source_sha256: str
    mask_supported: bool = False
    masked_kernel: object = None
    counters: Counter = field(default_factory=Counter)

    def report(self):
        return {
            "kind": "audited_kj_selector", "requested_mode": self.mode,
            "source_sha256": self.source_sha256,
            "biased_rows": "audited_sage_triton_scoped" if self.mask_supported else "pytorch_sdpa",
            "masked_launch": "two_stage; natural_log_bias_to_log2" if self.masked_kernel is not None else None,
            "compile_policy": "scoped_adapter_eager_kernel; upstream_compile_wrapper_not_reused",
            "completed_calls": dict(self.counters),
            "measurement_scope": "backend call counts, not end-to-end speed or human qualification",
        }

    def _selected_attention(self, q, k, v, heads, *, mask=None, skip_reshape=False,
                            skip_output_reshape=False, **kwargs):
        # Only verified KJ settings are reconstructed. No foreign closure runs.
        input_dtype = v.dtype
        if torch.float32 in (q.dtype, k.dtype, v.dtype):
            q, k, v = (value.to(torch.float16) for value in (q, k, v))
        batch = q.shape[0]
        if skip_reshape:
            layout = "HND"
            dim = q.shape[-1]
        else:
            layout = "NHD"
            dim = q.shape[-1] // heads
            q, k, v = (value.reshape(batch, -1, heads, dim) for value in (q, k, v))
        if mask is not None:
            if not self.mask_supported:
                raise RuntimeError("KJ Sage selected kernel cannot consume Relay bias")
            if mask.dtype != torch.bool:
                mask = mask.to(q.dtype)
            while mask.ndim < 4:
                mask = mask.unsqueeze(0 if mask.ndim == 2 else 1)
        seq_dim = 2 if layout == "HND" else 1
        if any((value.shape[seq_dim] - 1) * value.stride(seq_dim) >= 2**31 for value in (q, k, v)):
            q, k, v = (value.contiguous() for value in (q, k, v))
        options = {"is_causal": False, "tensor_layout": layout}
        if mask is not None:
            options["attn_mask"] = mask
        if self.mode == "sageattn_qk_int8_pv_fp16_cuda":
            options["pv_accum_dtype"] = "fp32"
        elif self.mode.startswith("sageattn_qk_int8_pv_fp8_cuda"):
            options["pv_accum_dtype"] = "fp32+fp16" if self.mode.endswith("++") else "fp32+fp32"
        if self.mode.startswith("sageattn3"):
            if layout == "NHD":
                q, k, v = (value.transpose(1, 2) for value in (q, k, v))
            options.pop("tensor_layout")
            options["per_block_mean"] = self.mode == "sageattn3_per_block_mean"
        kernel = self.masked_kernel if mask is not None and self.masked_kernel is not None else self.kernel
        output = kernel(q, k, v, **options).to(input_dtype)
        if self.mode.startswith("sageattn3") or layout == "HND":
            return output if skip_output_reshape else output.transpose(1, 2).reshape(batch, -1, heads * dim)
        return output.transpose(1, 2) if skip_output_reshape else output.reshape(batch, -1, heads * dim)

    def attention(self, q, k, v, heads, *, mask=None, **kwargs):
        reason = None
        if kwargs.get("low_precision_attention", True) is False:
            reason = "low_precision_disabled"
        elif q.device.type != "cuda":
            reason = "non_cuda"
        elif mask is not None and not self.mask_supported:
            reason = "no_audited_bias_kernel"
        elif kwargs.get("scale") is not None or kwargs.get("enable_gqa", False):
            reason = "custom_scale_or_gqa"
        if reason:
            output = core_attention.attention_pytorch(q, k, v, heads, mask=mask,
                                                      **{**kwargs, "_inside_attn_wrapper": True})
            self.counters["pytorch:" + reason] += 1
            return output
        # Kernel errors (including cancellation/OOM) propagate; no hidden retry
        # or success counter on a failed kernel invocation.
        output = self._selected_attention(q, k, v, heads, mask=mask, **kwargs)
        self.counters["sage:biased" if mask is not None else "sage:unbiased"] += 1
        return output


def capture_kj_relay_backend(override):
    """Recognize only the audited ordinary KJ selector, returning None otherwise.

    Source AST pins only relevant definitions, leaving unrelated KJ updates free.
    Live code, globals, closure contents and kernel identity are checked too.
    No KJ module is imported/started and no kernel is called by this inspection.
    """
    if not isinstance(override, FunctionType):
        return None
    cells = _cells(override)
    if set(cells) != {"new_attention"}:
        return None
    module = sys.modules.get(override.__module__)
    if module is None or override.__globals__ is not vars(module):
        return None
    try:
        path = inspect.getsourcefile(override)
        payload = Path(path).read_bytes()
        definitions = {node.name: node for node in ast.parse(payload).body
                       if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in _KJ_AST}
        if set(definitions) != set(_KJ_AST) or any(_ast_hash(node) != _KJ_AST[name] for name, node in definitions.items()):
            return None
        compiled = tuple(_codes(compile(payload, path, "exec", dont_inherit=True)))
        if not _live_code_matches(override, compiled, module):
            return None
        factory = getattr(module, "get_sage_func", None)
        if (not _live_code_matches(factory, compiled, module)
                or factory.__defaults__ != (False,)
                or getattr(module, "wrap_attn", None) is not core_attention.wrap_attn
                or getattr(module, "attention_pytorch", None) is not core_attention.attention_pytorch
                or getattr(module, "torch", None) is not torch):
            return None
        raw_attention = getattr(cells["new_attention"], "__wrapped__", None)
        if not _live_code_matches(raw_attention, compiled, module):
            return None
        raw_cells = _cells(raw_attention)
        if set(raw_cells) != {"sage_func"}:
            return None
        # Unwrap only to inspect: never invoke a decorator supplied by MODEL.
        selected = inspect.unwrap(raw_cells["sage_func"])
        if not _live_code_matches(selected, compiled, module):
            return None
        kernel_cells = _cells(selected)
        names = set(kernel_cells) - {"sage_attention"}
        if len(names) != 1:
            return None
        name = next(iter(names))
        package = sys.modules.get("sageattn3" if name == "sageattn3_blackwell" else "sageattention")
        kernel = kernel_cells[name]
        if package is None or kernel is not getattr(package, name, None) or not callable(kernel):
            return None
        if name == "sageattn":
            mode = "auto"
        elif name == "sageattn3_blackwell":
            mode = kernel_cells.get("sage_attention")
            if mode not in {"sageattn3", "sageattn3_per_block_mean"}:
                return None
        elif name in {"sageattn_qk_int8_pv_fp16_cuda", "sageattn_qk_int8_pv_fp16_triton", "sageattn_qk_int8_pv_fp8_cuda"}:
            mode = name + ("++" if "fp32+fp16" in selected.__code__.co_consts else "")
        else:
            return None
        # The KJ CUDA/auto choice remains active for unbiased rows. Biased
        # Relay rows explicitly use the separately audited Sage Triton variant
        # when already installed, otherwise SDPA with a reported reason.
        mask_candidate = getattr(sys.modules.get("sageattention"), "sageattn_qk_int8_pv_fp16_triton", None)
        mask_supported = _audited_mask_kernel(mask_candidate)
        masked_kernel = None
        if mask_supported:
            from .scoped_sage_triton import build_scoped_mask_kernel
            masked_kernel = build_scoped_mask_kernel(mask_candidate)
            mask_supported = masked_kernel is not None
        return KJRelayBackend(mode, kernel, hashlib.sha256(payload).hexdigest(), mask_supported, masked_kernel)
    except (OSError, TypeError, ValueError, SyntaxError, AttributeError):
        return None
