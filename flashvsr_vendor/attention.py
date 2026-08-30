"""Block-sparse attention bridge used by the vendored FlashVSR core.

The LCSA mask construction follows the public Apache-2.0 FlashVSR/TE-Speed
implementation.  Kernel execution is delegated to the separately installed
``spas_sage_attn`` wheel; importing this module never imports the CUDA wheel so
ComfyUI can still start and show an actionable dependency error at execution.
"""

from __future__ import annotations

import math

import torch
from einops import rearrange


def uses_split_k_mask(k_windows: torch.Tensor) -> bool:
    """Return whether the installed sparse kernel's 128Q/64K mask is valid."""

    return k_windows.ndim == 3 and int(k_windows.shape[1]) == 128


@torch.no_grad()
def generate_sparge_mask(
    batch_size: int,
    nheads: int,
    seqlen: int,
    q_w: torch.Tensor,
    k_w: torch.Tensor,
    topk: int = 10,
    local_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build an exact-cardinality LCSA mask for 128-token Q/64-token K blocks."""

    if batch_size != 1:
        raise ValueError("FlashVSR sparse attention currently supports batch_size=1")
    if local_mask is None:
        raise ValueError("FlashVSR sparse attention requires a local LCSA mask")
    if q_w.ndim != 3 or k_w.ndim != 3 or q_w.shape[1] != 128 or k_w.shape[1] != 128:
        raise ValueError(
            "FlashVSR sparse attention expects Q/K windows shaped [blocks, 128, channels]"
        )
    if q_w.shape[-1] != k_w.shape[-1] or q_w.shape[-1] % nheads:
        raise ValueError("FlashVSR sparse attention received incompatible head dimensions")

    q_summary = q_w.mean(dim=1)
    q_summary = rearrange(q_summary, "s (h d) -> h s d", h=nheads)

    # The CUDA kernel consumes K blocks of 64 tokens.  Preserve the temporal /
    # spatial block ordering and split each 128-token window into two summaries.
    k_summary = k_w.reshape(k_w.shape[0], 2, 64, k_w.shape[2]).mean(dim=2)
    k_summary = rearrange(k_summary, "s two (h d) -> h (s two) d", h=nheads)

    scores = torch.einsum("hqd,hkd->hqk", q_summary, k_summary)
    scores = scores / math.sqrt(float(q_summary.shape[-1]))

    spatial_q, spatial_k = map(int, local_mask.shape)
    if scores.shape[1] % spatial_q or scores.shape[2] % (2 * spatial_k):
        raise ValueError(
            "FlashVSR LCSA geometry mismatch: "
            f"score={tuple(scores.shape)} local={tuple(local_mask.shape)}"
        )
    q_time = scores.shape[1] // spatial_q
    k_time = scores.shape[2] // (2 * spatial_k)
    expanded_local = local_mask.to(device=scores.device, dtype=torch.bool)
    expanded_local = expanded_local.unsqueeze(0).unsqueeze(2)
    expanded_local = expanded_local.repeat(q_time, 1, k_time, 1)
    expanded_local = rearrange(expanded_local, "qt q kt k -> (qt q) (kt k)")
    expanded_local = expanded_local.repeat_interleave(2, dim=1)
    expanded_local = expanded_local.unsqueeze(0).expand(nheads, -1, -1)
    scores = scores.masked_fill(~expanded_local, float("-inf"))

    probabilities = torch.softmax(scores.float(), dim=-1)
    grouped = rearrange(
        probabilities,
        "h (t q) k -> (h t) q k",
        t=seqlen,
    )
    flat = grouped.flatten(1)
    keep = max(0, min(int(topk), flat.shape[1]))
    selected = torch.zeros_like(flat, dtype=torch.bool)
    if keep:
        indices = torch.topk(flat, k=keep, dim=1, largest=True, sorted=False).indices
        selected.scatter_(1, indices, True)
    selected = selected.reshape_as(grouped)
    selected = rearrange(selected, "(h t) q k -> h (t q) k", t=seqlen)
    return selected.unsqueeze(0).to(torch.int8)


def sparge_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: torch.Tensor,
    num_heads: int,
) -> torch.Tensor:
    """Run the maintained Spas-SageAttn CUDA wheel with the LCSA mask."""

    try:
        from spas_sage_attn import block_sparse_sage2_attn_cuda
    except Exception as exc:  # pragma: no cover - depends on optional CUDA wheel
        raise RuntimeError(
            "FlashVSR requires the optional 'spas_sage_attn' CUDA wheel. "
            "Install the build matching your PyTorch/CUDA version, then restart ComfyUI."
        ) from exc

    q_hnd = rearrange(q, "b s (h d) -> b h s d", h=num_heads).contiguous()
    k_hnd = rearrange(k, "b s (h d) -> b h s d", h=num_heads).contiguous()
    v_hnd = rearrange(v, "b s (h d) -> b h s d", h=num_heads).contiguous()
    result = block_sparse_sage2_attn_cuda(
        q_hnd,
        k_hnd,
        v_hnd,
        mask_id=attention_mask.to(device=q.device, dtype=torch.int8),
        dropout_p=0.0,
        scale=None,
        smooth_k=True,
        pvthreshd=1_000_000.0,
        attention_sink=False,
        tensor_layout="HND",
        output_dtype=q.dtype,
        return_sparsity=False,
    )
    if isinstance(result, tuple):
        result = result[0]
    if not isinstance(result, torch.Tensor) or result.shape != q_hnd.shape:
        raise RuntimeError(
            "spas_sage_attn returned an unexpected result for FlashVSR: "
            f"expected {tuple(q_hnd.shape)}, got {getattr(result, 'shape', type(result))}"
        )
    return rearrange(result, "b h s d -> b s (h d)")
