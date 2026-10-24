"""Request-local approximate DiT reference-input reuse, independently authored.

This is not Qwen prefix caching or an exact cache of diffusion evaluations.
Reference inputs frozen at a refresh no longer follow the changing targets.
All selected projection/normalization modules and full K/V still execute.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
from pathlib import Path
from types import CodeType, FunctionType

import torch
from comfy.ldm.minimax import model as core

from .reference_temporal_pack import condition_identity, tensor_record
from .reference_temporal_pack_advanced import _native_source_current
from .source_code_identity import executable_code_equal


@lru_cache(maxsize=32)
def _source_codes(path, stamp, size):
    """Compile source only, never execute imported/foreign code to classify it.

    The runtime separately hashes these files pre/post each actual evaluation;
    the stat-keyed compile cache only avoids recompiling Torch for every block.
    """
    def codes(code):
        yield code
        for item in code.co_consts:
            if type(item) is CodeType:
                yield from codes(item)
    return {code.co_qualname: code for code in codes(compile(Path(path).read_bytes(), path, "exec", dont_inherit=True))}


def source_current(function, module):
    function = getattr(function, "__func__", function)
    if type(function) is not FunctionType or function.__globals__ is not vars(module):
        return False
    path = str(Path(module.__file__).resolve())
    status = Path(path).stat()
    expected = _source_codes(path, status.st_mtime_ns, status.st_size).get(function.__code__.co_qualname)
    return executable_code_equal(function.__code__, expected)


def module_call_native(module):
    from torch.nn.modules import module as owner
    return (getattr(module, "_compiled_call_impl", None) is None
        and type(module).__call__ is torch.nn.Module.__call__
        and getattr(getattr(module, "_call_impl", None), "__func__", None) is torch.nn.Module._call_impl
        and source_current(torch.nn.Module.__call__, owner)
        and source_current(torch.nn.Module._call_impl, owner))


def attention_source_current(function):
    from comfy.ldm.modules import attention
    if not source_current(function, attention):
        return False
    # Core's decorator chooses between default and configured attention. Verify
    # its actual closed function too; a live __wrapped__ label proves nothing.
    names = function.__code__.co_freevars
    if set(names) == {"func", "wrapper"}:
        cells = dict(zip(names, function.__closure__ or (), strict=True))
        container = getattr(function, "container_function", None)
        return (cells["wrapper"].cell_contents is function
                and source_current(cells["func"].cell_contents, attention)
                and (container is None or source_current(container, attention)))
    return not names


@dataclass(frozen=True)
class ReferenceStepPlan:
    refresh_every: int = 2
    include_text: bool = False

    def __post_init__(self):
        if type(self.refresh_every) is not int or self.refresh_every < 1:
            raise ValueError("Reference refresh interval must be a positive logical-step count")
        if type(self.include_text) is not bool:
            raise ValueError("include_text must be an explicit boolean")


def parameter_versions(network):
    result = []
    for name, value in list(network.named_parameters()) + list(network.named_buffers()):
        try:
            version = value._version
        except RuntimeError:
            version = None  # Inference tensors: final content check still applies.
        result.append((name, tuple(value.shape), str(value.dtype), version))
    return result


def module_contract(network):
    """Bind executable structure, not verified Core VBAR residency metadata.

    Only the three fields written by Core's cast resolver are transport state.
    Unknown owners retain their complete scalar contract; this classification
    never selects/replaces a model, callable, LoRA, or its execution path.
    """
    from comfy import ops
    transport = {"_v_weight", "_v_bias", "_v_signature"}
    cast_owner = getattr(ops, "CastWeightBiasOp", None)
    core_cast_current = (type(cast_owner) is type
                         and source_current(getattr(ops, "cast_bias_weight", None), ops)
                         and source_current(getattr(ops, "resolve_cast_module_with_vbar", None), ops))
    result = []
    for name, module in network.named_modules():
        forward = getattr(module, "forward", None)
        function = getattr(forward, "__func__", forward)
        core_transport = (core_cast_current and isinstance(module, cast_owner)
                          and module_call_native(module) and source_current(forward, ops))
        attributes = {key: value for key, value in vars(module).items()
                      if type(value) in (str, bool, int, float, type(None))
                      and key != "training" and not (core_transport and key in transport)}
        result.append((name, id(module), id(function), id(getattr(function, "__code__", None)), attributes))
    return result


def weight_digest(network):
    """Original selected live storage, no dequantization or file-name inference.

    Unsupported opaque storage disables this optional cache, not generation.
    Known floating NaN is a real error. Called at first/last logical evaluations,
    not a claimed free performance measurement.
    """
    digest = hashlib.sha256()
    for name, value in network.state_dict().items():
        if type(value) not in (torch.Tensor, torch.nn.Parameter) or value.is_meta or value.is_quantized:
            return None
        digest.update(name.encode())
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        flat = value.detach().contiguous().reshape(-1).view(torch.uint8).reshape(-1)
        for start in range(0, flat.numel(), 8 * 1024**2):
            from comfy.model_management import throw_exception_if_processing_interrupted
            throw_exception_if_processing_interrupted()
            raw = flat[start:start + 8 * 1024**2].cpu()
            if value.is_floating_point():
                floating = raw.view(value.dtype)
                # CPU isfinite lacks float8 kernels; widening those values is
                # lossless. Do not narrow float64: a valid selected checkpoint
                # may contain finite values outside float32's representable range.
                if str(value.dtype).startswith("torch.float8_"):
                    floating = floating.float()
                if not torch.isfinite(floating).all():
                    raise ValueError("Reference cache selected MODEL contains non-finite weights")
            digest.update(raw.numpy().tobytes())
    return digest.hexdigest()


def native_network(network):
    return _native_source_current(getattr(network, "_forward", None), "MiniMaxH3Model._forward")


def native_block(block):
    import comfy.ops
    from torch.nn.modules import module as module_owner
    from torch.nn.modules import normalization
    if module_owner._global_forward_pre_hooks or module_owner._global_forward_hooks:
        return False
    if not (_native_source_current(getattr(block, "forward", None), "DiTBlock.forward")
            and _native_source_current(getattr(getattr(block, "attn", None), "forward", None), "Attention.forward")
            and _native_source_current(getattr(getattr(block, "mlp", None), "forward", None), "MLP.forward")
            and _native_source_current(getattr(getattr(block, "adaln_proj", None), "forward", None), "AdalnProj.forward")):
        return False
    for name in ("norm1", "norm2"):
        function = getattr(getattr(getattr(block, name), "forward", None), "__func__", None)
        if not any(function is value for value in (comfy.ops.disable_weight_init.RMSNorm.forward, torch.nn.RMSNorm.forward)):
            return False
        if not (source_current(comfy.ops.disable_weight_init.RMSNorm.forward, comfy.ops)
                and source_current(torch.nn.RMSNorm.forward, normalization)
                and source_current(torch.nn.functional.rms_norm, torch.nn.functional)):
            return False
    for module in block.modules():
        if not module_call_native(module):
            return False
        if getattr(module, "weight_function", ()) or getattr(module, "bias_function", ()):
            return False
        if isinstance(module, torch.nn.Linear):
            function = getattr(getattr(module, "forward", None), "__func__", None)
            if not any(function is value for value in (comfy.ops.disable_weight_init.Linear.forward, torch.nn.Linear.forward)):
                return False
            if not source_current(function, comfy.ops if function is comfy.ops.disable_weight_init.Linear.forward
                                  else __import__("torch.nn.modules.linear", fromlist=["Linear"])):
                return False
    preferred = getattr(block.attn.comfy_attention, "function", None)
    if preferred is not None and not attention_source_current(preferred):
        return False
    return not any(getattr(module, "_forward_pre_hooks", {}) or getattr(module, "_forward_hooks", {})
                   for module in block.modules())


def native_attention_backend():
    from comfy.ldm.modules import attention
    return (attention_source_current(core.optimized_attention)
        and any(core.optimized_attention is getattr(attention, name, None) for name in (
        "optimized_attention", "attention_pytorch", "attention_sage", "attention3_sage", "attention_flash",
        "attention_xformers", "attention_basic", "attention_split", "attention_sub_quad", "attention_comfy_kitchen_int8")))


def logical_step(options):
    grid, now = options.get("sample_sigmas"), options.get("sigmas")
    if not isinstance(grid, torch.Tensor) or not isinstance(now, torch.Tensor):
        return None
    if grid.ndim != 1 or grid.numel() < 2 or now.numel() < 1:
        raise ValueError("Reference cache needs a real one-dimensional sampler grid/current sigma")
    values = tuple(float(x) for x in grid.detach().cpu().tolist())
    if any(not torch.isfinite(torch.tensor(x)) or x < 0 for x in values):
        raise ValueError("Reference cache sampler grid must be finite and nonnegative")
    if any(a <= b for a, b in zip(values, values[1:])):
        return None  # A different executable schedule is not a cache admission ban.
    current = now.detach().reshape(-1).cpu()
    if not torch.isfinite(current).all():
        raise ValueError("Reference cache current sigma is non-finite")
    if not bool((current == current[0]).all()):
        return None
    matches = [i for i, value in enumerate(values[:-1]) if value == float(current[0])]
    return (values, matches[0]) if len(matches) == 1 else None


def row_indices(layout, include_text=False):
    if not hasattr(layout, "segments") or not hasattr(layout, "position_ids"):
        return None
    kinds = {"cond", "cond_audio", "ref_img", "ref_audio"}
    if include_text:
        kinds.add("text")
    selected = torch.zeros(layout.seq_len, dtype=torch.bool)
    cursor = 0
    for start, stop, kind in layout.segments:
        if type(start) is not int or type(stop) is not int or start != cursor or stop < start:
            raise ValueError("Reference cache layout has invalid packed segment boundaries")
        selected[start:stop] = kind in kinds
        cursor = stop
    if cursor != layout.seq_len or layout.position_ids.shape != (cursor, 3):
        raise ValueError("Reference cache layout rows/physical positions disagree")
    tensor_record(layout.position_ids)
    return selected.nonzero().flatten(), (~selected).nonzero().flatten()


def select_segments(segments, indices, total):
    """Map actual per-row modulation indices, including soft-mask vector labels."""
    result, source_cursor, destination = [], 0, 0
    for start, stop, row in segments:
        if start != source_cursor or not start < stop <= total:
            raise ValueError("Reference cache modulation rows do not cover the actual packed sequence")
        chosen = indices[(indices >= start) & (indices < stop)] - start
        count = chosen.numel()
        if isinstance(row, torch.Tensor):
            if row.ndim != 1 or row.numel() != stop - start or row.dtype != torch.long:
                raise ValueError("Reference cache modulation tensor has wrong row geometry")
            new_row = row.index_select(0, chosen.to(row.device))
        elif type(row) is int and row >= 0:
            new_row = row
        else:
            raise ValueError("Reference cache modulation label must be an actual integer row")
        if count:
            result.append((destination, destination + count, new_row))
        destination += count
        source_cursor = stop
    if source_cursor != total or destination != indices.numel():
        raise ValueError("Reference cache modulation sequence is incomplete")
    return result


def condition_key(context, payload, layout):
    metadata = {key: value for key, value in payload.items() if key != "layout"}
    metadata["physical_layout"] = {"signature": tuple(layout.signature), "segments": layout.segments,
                                   "positions": tensor_record(layout.position_ids)}
    return condition_identity([[context, metadata]])["sha256"]


class ReferenceAttention:
    """Native full-order K/V, only active queries; no global attention patches."""
    def __init__(self, attention, cached, cached_indices, active_indices, full_rope, total):
        self.attention, self.cached = attention, cached
        self.cached_indices, self.active_indices = cached_indices, active_indices
        self.full_rope, self.total = full_rope, total

    def __call__(self, x, rope_freqs=None, transformer_options=None):
        import comfy.model_management
        import comfy.quant_ops
        from comfy.ldm.modules.attention import AttentionTensorContainer
        attention = self.attention
        full = torch.empty(self.total, x.shape[1], device=x.device, dtype=x.dtype)
        full[self.active_indices.to(x.device)] = x
        full[self.cached_indices.to(x.device)] = self.cached
        q, k, v = attention.qkv_proj(full).split(attention.heads * attention.head_dim, dim=-1)
        q = q.view(1, self.total, attention.heads, attention.head_dim)
        k = k.view(1, self.total, attention.heads, attention.head_dim)
        qw = comfy.model_management.cast_to(attention.q_norm.weight, device=x.device)
        kw = comfy.model_management.cast_to(attention.k_norm.weight, device=x.device)
        comfy.quant_ops.ck.rms_rope_split_half_(
            q, k, self.full_rope, qw, kw, epsilon=attention.q_norm.eps,
            rot_dim=self.full_rope.shape[-3] * 2)
        q = q[0].index_select(0, self.active_indices.to(q.device))
        q = AttentionTensorContainer(q.transpose(0, 1).unsqueeze(0))
        k = AttentionTensorContainer(k[0].transpose(0, 1).unsqueeze(0))
        v = AttentionTensorContainer(v.view(self.total, attention.heads, attention.head_dim).transpose(0, 1).unsqueeze(0))
        result = core.optimized_attention(q, k, v, attention.heads, preferred_attention=attention.comfy_attention,
                                          mask=None, skip_reshape=True, transformer_options=transformer_options or {})
        return attention.out_proj(result.squeeze(0))


def cached_block(block, args, extra, stored, cached, active):
    h, rope = args["img"], args["rope_freqs"]
    if h.ndim != 2 or h.shape[0] != cached.numel() + active.numel() or not torch.isfinite(h).all():
        raise ValueError("Reference cache actual block input geometry/content is invalid")
    if rope.ndim != 6 or rope.shape[1] != h.shape[0]:
        raise ValueError("Reference cache RoPE does not cover the actual packed rows")
    inside = select_segments(args["mod_segments"], cached, h.shape[0])
    outside = select_segments(args["mod_segments"], active, h.shape[0])
    shift, scale = block.adaln_proj(args["t_emb"])[:2]
    normalized = core._mod_scale_shift(block.norm1(stored.to(h.device, h.dtype)), shift, scale, inside)
    attention = ReferenceAttention(block.attn, normalized, cached, active, rope, h.shape[0])
    output = extra["original_block"]({**args, "img": h.index_select(0, active.to(h.device)),
        "mod_segments": outside, "rope_freqs": rope.index_select(1, active.to(rope.device)), "attention": attention})["img"]
    if output.shape != (active.numel(), h.shape[1]) or not torch.isfinite(output).all():
        raise ValueError("Reference cache active block result has invalid shape/content")
    result = h.clone()
    result[active.to(h.device)] = output
    return {"img": result}
