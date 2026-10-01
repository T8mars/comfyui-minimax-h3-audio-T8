"""Bounded FETA statistics around the native V2 chunked VSA producer.

No kernel replacement, full QKV materialization, global monkeypatch, or Dense
fallback. FETA is measured from extra, bounded native target-video projections;
the original sparse plan, pooled statistics, gate, sinks and kernel stay intact.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import torch
import comfy.model_management
import comfy.quant_ops
from torch import nn

from .. import enhance_a_video_advanced as feta
from .. import fast_h3_v2_advanced as v2
from ..patch_stack_policy import warn_patch_stack
from ..vdn_attention_compat import _factory_closure


def projected_cfi(attn, hidden, rope_freqs, route):
    """Same temporal statistic, streaming spatial columns of the video grid.

    The budget covers a conservative estimate of owned gather/QKV/RoPE/score
    intermediates, not model weights or the sparse kernel's own workspace.
    Never gather a whole-video QKV tensor just to obtain this scalar.
    """
    frames, spatial = int(route["frames"]), int(route["spatial_tokens"])
    heads, dim = int(attn.heads), int(attn.head_dim)
    start, end = int(route["video_start"]), int(route["video_end"])
    if (hidden.ndim != 2 or frames < 2 or min(spatial, heads, dim) < 1
            or end - start != frames * spatial or not 0 <= start < end <= hidden.shape[0]):
        raise ValueError("Sparse FETA target-video layout is invalid")
    if rope_freqs is None or rope_freqs.ndim != 6 or rope_freqs.shape[:2] != (1, hidden.shape[0]):
        raise ValueError("Sparse FETA requires the original native packed RoPE table")
    # Reserve for eager RMS/RoPE intermediates too; CUDA often needs less.
    projection_bytes = frames * (hidden.shape[-1] * hidden.element_size() + 16 * heads * dim * 4)
    rope_bytes = frames * (rope_freqs[0, 0].numel() * rope_freqs.element_size() + 8)
    scores_bytes = heads * frames * frames * 12
    per_column = projection_bytes + rope_bytes + scores_bytes
    budget = int(route["max_workspace_mib"]) * 1024 * 1024
    columns = min(spatial, budget // per_column)
    if columns < 1:
        raise ValueError("Sparse FETA workspace cannot fit one temporal column; increase max_workspace_mib")
    trace = torch.zeros((), device=hidden.device, dtype=torch.float64)
    qw = comfy.model_management.cast_to(attn.q_norm.weight, device=hidden.device)
    kw = comfy.model_management.cast_to(attn.k_norm.weight, device=hidden.device)
    for left in range(0, spatial, columns):
        comfy.model_management.throw_exception_if_processing_interrupted()
        width = min(columns, spatial - left)
        indices = (torch.arange(frames, device=hidden.device)[:, None] * spatial
                   + torch.arange(left, left + width, device=hidden.device)[None, :] + start).reshape(-1)
        gathered = hidden.index_select(0, indices)
        fused = attn.qkv_proj(gathered)
        if fused.shape != (frames * width, 3 * heads * dim):
            raise ValueError("Sparse FETA QKV projection differs from the native head layout")
        q, k, _ = fused.split(heads * dim, dim=-1)
        q = q.view(1, frames * width, heads, dim)
        k = k.view(1, frames * width, heads, dim)
        freqs = rope_freqs.index_select(1, indices)
        comfy.quant_ops.ck.rms_rope_split_half_(q, k, freqs, qw, kw,
                                              epsilon=attn.q_norm.eps, rot_dim=freqs.shape[-3] * 2)
        query = q[0].view(frames, width, heads, dim).permute(1, 2, 0, 3) * (float(dim) ** -.5)
        key = k[0].view(frames, width, heads, dim).permute(1, 2, 0, 3)
        logits = torch.matmul(query, key.transpose(-2, -1)).to(torch.float32)
        probabilities = torch.softmax(logits, dim=-1)
        trace += torch.diagonal(probabilities, dim1=-2, dim2=-1).sum(dtype=torch.float64)
        del gathered, fused, q, k, query, key, logits, probabilities, freqs, indices
    count = spatial * heads
    cfi = ((float(count * frames) - trace) / float(count * frames * (frames - 1))).float()
    return cfi, int(columns), int(columns * per_column)


def _gain(attn, hidden, rope_freqs, route):
    cfi, columns, workspace = projected_cfi(attn, hidden, rope_freqs, route)
    gain = torch.clamp_min((float(route["frames"]) + float(route["tau"])) * cfi, 1.)
    g, value = float(gain.detach().cpu()), float(cfi.detach().cpu())
    if not math.isfinite(g) or g < 1 or g > float(route["g_hard_limit"]):
        raise RuntimeError("Sparse H3 EAV enhancement factor exceeded its configured hard limit or is invalid")
    route["runtime"].record(int(route["forward_index"]), g=g, cfi=value, chunk_rows=columns, workspace=workspace)
    return gain


def wrap_producer(original, owner, block, index, runtime):
    """Wrap only a source-authenticated native V2 attention producer.

    The original hook still chooses eligibility and invokes original_block.
    Only its authenticated sparse attention callback is adapted; foreign
    callbacks/DiT producers are delegated as-is and cannot acquire fake coverage.
    """
    state = _factory_closure(original, v2._V2Runtime.block_patch, "patch")
    if (state is None or state.get("self") is not owner or state.get("block") is not block
            or state.get("index") != index):
        warn_patch_stack("Stage sparse EAV retains an unverified DiT producer; sparse FETA coverage is not certified")
        return original
    original_attention = state["attention"]

    def sparse_attention(hidden, rope_freqs=None, transformer_options=None):
        options = transformer_options or {}
        route = options.get(feta.EAV_RUNTIME_KEY)
        if route is None:
            return original_attention(hidden, rope_freqs, options)
        if not route["active"]:
            output = original_attention(hidden, rope_freqs, options)
            runtime.sparse_calls += 1
            return output
        attn = block.attn
        if getattr(attn, "to_gate_compress", None) is None:
            raise RuntimeError("Stage sparse EAV lost the trained V2 coarse gate")
        gain = _gain(attn, hidden, rope_freqs, route)
        if route["mode"] not in ("report_only", "apply_exp"):
            raise ValueError("Unknown sparse EAV mode")

        def projection(source):
            def apply(output):
                if route["mode"] == "apply_exp":
                    output[int(route["video_start"]):int(route["video_end"])].mul_(gain.to(output))
                return source(output)
            return apply

        if owner.head_chunks == 1:
            view = SimpleNamespace(heads=attn.heads, head_dim=attn.head_dim, q_norm=attn.q_norm,
                k_norm=attn.k_norm, qkv_proj=attn.qkv_proj, to_gate_compress=attn.to_gate_compress,
                out_proj=projection(attn.out_proj))
            output = owner.sparse.h3_sparse_attention(view, hidden, rope_freqs, options, owner.patch, index)
        else:
            outputs = []
            width = math.ceil(attn.heads / owner.head_chunks)
            for first in range(0, attn.heads, width):
                last = min(first + width, attn.heads)
                view = SimpleNamespace(heads=last-first, head_dim=attn.head_dim, q_norm=attn.q_norm,
                    k_norm=attn.k_norm, out_proj=projection(nn.Identity()),
                    qkv_proj=v2._HeadProjection(attn.qkv_proj, attn.heads, attn.head_dim, first, last),
                    to_gate_compress=v2._HeadProjection(attn.to_gate_compress, attn.heads,
                                                       attn.head_dim, first, last, gate=True))
                outputs.append(owner.sparse.h3_sparse_attention(view, hidden, rope_freqs, options,
                                                                 owner.patch, (index, first, last)))
            output = attn.out_proj(torch.cat(outputs, dim=-1))
        owner.counts["vsa"] += 1
        runtime.sparse_calls += 1
        return output

    def wrapped(args, extra):
        def original_block(actual):
            if actual.get("attention") is original_attention:
                actual = {**actual, "attention": sparse_attention}
            return extra["original_block"](actual)
        return original(args, {**extra, "original_block": original_block})

    return wrapped


def install_sparse_eav(model, owner, runtime):
    blocks = model.get_model_object("diffusion_model").blocks
    existing = dict(model.model_options["transformer_options"].get("patches_replace", {}).get("dit", {}))
    for index, block in enumerate(blocks):
        key = ("double_block", index)
        original = existing.get(key)
        adapted = wrap_producer(original, owner, block, index, runtime)
        if adapted is not original:
            model.set_model_patch_replace(adapted, "dit", "double_block", index)
