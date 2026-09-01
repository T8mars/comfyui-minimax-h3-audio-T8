"""Block-sparse attention forward kernel for MiniMax-H3.

Vendored and reduced from LightX2V (Apache-2.0):
  lightx2v/common/ops/attn/kernels/sla_kernel_ar.py  -- ``_attn_fwd``
  https://github.com/ModelTC/LightX2V

Only the forward pass survives the port; sampling runs under ``no_grad`` so the
backward half and the ``autograd.Function`` wrapper are dead weight here.

The ``_ar`` name upstream refers to the models it was written for, not to any
causal structure: the kernel walks whatever key blocks the lookup table names
and applies no triangular mask. That is what makes it usable for H3, whose
packed ``[text | cond/ref | audio | video]`` sequence is fully bidirectional.

Layout is BLHD -- ``(B, L, H, D)`` -- which is deliberate. H3 materialises q/k/v
as ``[S, H, D]`` and only then transposes to ``[1, H, S, D]`` for the attention
call, so transposing back is free, while a BHSD kernel would force a real
``.contiguous()`` copy costing ~1.3 GB per tensor at 768p/15s.

Two changes against upstream, both marked FIX below: masked loads now pass
``other=0.0``, because Triton leaves masked lanes undefined and ``0 * NaN`` is
NaN, not zero -- the sequence-tail block can poison a whole row otherwise.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _attn_fwd(
    Q,
    K,
    V,
    qk_scale: tl.constexpr,
    topk: tl.constexpr,
    LUT,
    OS,
    H: tl.constexpr,
    LQ: tl.constexpr,
    LK: tl.constexpr,
    M_BLOCKS: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    idx_m = tl.program_id(0).to(tl.int64)
    idx_bh = tl.program_id(1).to(tl.int64)

    idx_b = idx_bh // H
    idx_h = idx_bh % H

    HD: tl.constexpr = H * D

    # Q/K/V/O: (B, L, H, D) contiguous.
    q_offset = idx_b * LQ * HD + idx_h * D
    kv_offset = idx_b * LK * HD + idx_h * D
    lut_offset = (idx_bh * M_BLOCKS + idx_m) * topk

    offs_m = idx_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)

    Q_ptrs = Q + q_offset + offs_m[:, None] * HD + offs_d[None, :]
    OS_ptrs = OS + q_offset + offs_m[:, None] * HD + offs_d[None, :]
    LUT_ptr = LUT + lut_offset

    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    o_s = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    # FIX vs upstream: other=0.0 on every masked load.
    q = tl.load(Q_ptrs, mask=offs_m[:, None] < LQ, other=0.0)
    for block_idx in tl.range(topk):
        idx_n = tl.load(LUT_ptr + block_idx).to(tl.int64)
        k_start = idx_n * BLOCK_N
        k_mask = (k_start + offs_n) < LK

        K_ptrs = K + kv_offset + (k_start + offs_n)[None, :] * HD + offs_d[:, None]
        V_ptrs = V + kv_offset + (k_start + offs_n)[:, None] * HD + offs_d[None, :]

        k = tl.load(K_ptrs, mask=k_mask[None, :], other=0.0)
        qk = tl.dot(q, k) * (qk_scale * 1.4426950408889634)  # 1/ln(2), for exp2
        qk = tl.where(k_mask[None, :], qk, float("-inf"))

        v = tl.load(V_ptrs, mask=k_mask[:, None], other=0.0)
        local_m = tl.max(qk, 1)
        new_m = tl.maximum(m_i, local_m)
        qk = qk - new_m[:, None]

        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - new_m)
        o_s = o_s * alpha[:, None]
        o_s += tl.dot(p.to(v.dtype), v)

        l_i = l_i * alpha + l_ij
        m_i = new_m

    o_s = o_s / l_i[:, None]
    tl.store(OS_ptrs, o_s.to(OS.type.element_ty), mask=offs_m[:, None] < LQ)


# (BLOCK_M, BLOCK_N) -> ladder of (num_warps, num_stages) to try, best first.
#
# These are not upstream's numbers. LightX2V hardcodes num_warps=4, num_stages=3,
# which on sm_120 needs 160 KB of shared memory against a 99 KB limit -- it does
# not merely run slowly, it fails to launch. Consumer Blackwell has less shared
# memory than the datacentre parts these kernels were tuned on, so we probe and
# fall back. Measured on a 5090 at S=32768, H=56, D=128 (ms, lower better):
#   128x64  w8 s3 -> 22.4   128x64  w4 s3 -> 23.9   128x128 w8 s2 -> 25.4
#   64x64   w4 s1 -> 26.8   64x128  w4 s2 -> 30.5   128x128 w4 s1 -> 40.5
_LADDER = {
    (128, 64): ((8, 3), (4, 3), (8, 2), (4, 1)),
    (128, 128): ((8, 2), (4, 2), (8, 1), (4, 1)),
    (64, 128): ((4, 2), (8, 2), (4, 1)),
    (64, 64): ((4, 1), (4, 3), (8, 3), (8, 1)),
    # UNBENCHMARKED on this kernel/hardware -- no measured (num_warps,
    # num_stages) exists for 32x32 here. (4, 2) is first based on external
    # evidence, not a local measurement: published Triton flash-attention
    # autotune sweeps that include a 32x32 tile consistently pair it with
    # num_warps=4, num_stages=2 rather than lower warp counts (e.g.
    # triton-lang/triton discussion #8261's BLOCK_BR=32,BLOCK_BC=32 config).
    # A local benchmark run measured 32x32 at the same speed as 64x64 with
    # the earlier (2, 1)-first ordering, so this reordering has not been
    # re-verified to actually help -- treat as still unproven, not a fix.
    (32, 32): ((4, 2), (2, 1), (4, 1), (2, 2)),
}
_CHOSEN: dict = {}


def block_sparse_attention(q, k, v, lut, topk, BLOCK_M, BLOCK_N, qk_scale=None):
    """Attend each query block to only the key blocks named in ``lut``.

    ``q``/``k``/``v`` are ``(B, L, H, D)`` contiguous; ``lut`` is
    ``(B, H, M_BLOCKS, topk)`` int32, contiguous. Returns ``(B, L, H, D)``.
    """
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    assert lut.is_contiguous()
    assert BLOCK_M in (32, 64, 128) and BLOCK_N in (32, 64, 128)

    B, LQ, H, D = q.shape
    LK = k.shape[1]
    if qk_scale is None:
        qk_scale = D**-0.5

    M_BLOCKS = triton.cdiv(LQ, BLOCK_M)
    o_s = torch.empty_like(q)
    grid = (M_BLOCKS, B * H)

    key = (BLOCK_M, BLOCK_N, D)
    ladder = (_CHOSEN[key],) if key in _CHOSEN else _LADDER[(BLOCK_M, BLOCK_N)]

    last = None
    for cfg in ladder:
        num_warps, num_stages = cfg
        try:
            _attn_fwd[grid](
                q, k, v, qk_scale, topk, lut, o_s,
                H, LQ, LK, M_BLOCKS, D, BLOCK_M, BLOCK_N,
                num_warps=num_warps, num_stages=num_stages,
            )
        except triton.runtime.errors.OutOfResources as exc:
            last = exc
            continue
        _CHOSEN[key] = cfg
        return o_s

    raise last if last is not None else RuntimeError("no viable launch config")
