# SPDX-License-Identifier: MIT
# Vendored from veda-sparse/Miowtion at 3bc5e5ae53e8c0d4539f2bc872265164947d013a.
"""Fused Triton kernels for tile permutations and predictor pooling.

The Veda paths permute q / k / v into tile order per head group, pool q and
k per tile (masked mean / max / min over the valid-prefix rows) and scatter
the output back. In torch this is an advanced-indexing gather per tensor, a
zero fill of the padding slots, three reductions and a masked recompute of
the partial tiles: about 16 ms per layer on RTX 4090 at 38k tokens, 30% of
the sparse attention time. Here one program per (tile, head) loads the
tile's 128 rows once through the permutation and writes the gathered rows
and, for q / k, the pooled features.

Numerics: gathers and scatters are exact; max / min are exact; the mean sums
in fp32 in a different order than torch (tolerance, not bitwise).
"""

from __future__ import annotations

import functools

import torch

from .veda import tiling

_TILE = tiling.TILE_SIZE


@functools.cache
def _triton():
    try:
        import triton  # pylint: disable=import-outside-toplevel
        import triton.language as tl  # pylint: disable=import-outside-toplevel
    except ImportError:
        return None
    return triton, tl


def available() -> bool:
    return torch.cuda.is_available() and _triton() is not None


@functools.cache
def _kernels():
    triton, tl = _triton()

    @triton.jit
    def gather_kernel(x_ptr, index_ptr, valid_ptr, heads_ptr, out_ptr,
                      feats_ptr, count_ptr, stride_xn, stride_xh,
                      stride_on, stride_oh, stride_fh, stride_fn,
                      HEAD_DIM: tl.constexpr, TILE: tl.constexpr,
                      POOL: tl.constexpr):
        tile = tl.program_id(0)
        h = tl.program_id(1)
        head = tl.load(heads_ptr + h)
        slots = tile * TILE + tl.arange(0, TILE)
        dims = tl.arange(0, HEAD_DIM)
        src = tl.load(index_ptr + slots)
        valid = tl.load(valid_ptr + slots) != 0
        x = tl.load(x_ptr + src[:, None] * stride_xn + head * stride_xh
                    + dims[None, :], mask=valid[:, None], other=0.0)
        tl.store(out_ptr + slots[:, None] * stride_on + h * stride_oh
                 + dims[None, :], x)
        if POOL:
            count = tl.load(count_ptr + tile)
            xf = x.to(tl.float32)
            mean = tl.sum(xf, axis=0) / tl.maximum(count, 1).to(tl.float32)
            tmax = tl.max(tl.where(valid[:, None], xf, float('-inf')), axis=0)
            tmin = tl.min(tl.where(valid[:, None], xf, float('inf')), axis=0)
            empty = count == 0
            base = feats_ptr + h * stride_fh + tile * stride_fn
            tl.store(base + dims, tl.where(empty, 0.0, mean))
            tl.store(base + HEAD_DIM + dims, tl.where(empty, 0.0, tmax))
            tl.store(base + 2 * HEAD_DIM + dims, tl.where(empty, 0.0, tmin))

    @triton.jit
    def scatter_kernel(tiled_ptr, index_ptr, valid_ptr, heads_ptr, out_ptr,
                       stride_tn, stride_th, stride_on, stride_oh,
                       HEAD_DIM: tl.constexpr, TILE: tl.constexpr):
        tile = tl.program_id(0)
        h = tl.program_id(1)
        head = tl.load(heads_ptr + h)
        slots = tile * TILE + tl.arange(0, TILE)
        dims = tl.arange(0, HEAD_DIM)
        dst = tl.load(index_ptr + slots)
        valid = tl.load(valid_ptr + slots) != 0
        x = tl.load(tiled_ptr + slots[:, None] * stride_tn + h * stride_th
                    + dims[None, :])
        tl.store(out_ptr + dst[:, None] * stride_on + head * stride_oh
                 + dims[None, :], x, mask=valid[:, None])

    return gather_kernel, scatter_kernel


def _heads(x: torch.Tensor, heads: torch.Tensor | None) -> torch.Tensor:
    if heads is None:
        return torch.arange(x.shape[1], device=x.device)
    return heads


def _check(x: torch.Tensor) -> None:
    if x.shape[-1] not in (64, 128) or x.stride(-1) != 1:
        raise ValueError('tile kernels need contiguous head_dim 64 or 128')


def gather_tiles(x: torch.Tensor, layout: tiling.TileLayout,
                 heads: torch.Tensor | None) -> torch.Tensor:
    """Same contract as tiling.gather_tiles."""
    return _gather(x, layout, heads, pool=False)[0]


def gather_and_pool(x: torch.Tensor, layout: tiling.TileLayout,
                    heads: torch.Tensor | None
                    ) -> tuple[torch.Tensor, torch.Tensor]:
    """tiling.gather_tiles and predictor.pool_tiles in one pass.

    Returns:
        ([N, H', D] tile-ordered rows, [H', n_tiles, 3D] fp32 features).
    """
    return _gather(x, layout, heads, pool=True)


def _gather(x, layout, heads, pool):
    _check(x)
    heads = _heads(x, heads)
    num_heads, dim = heads.numel(), x.shape[-1]
    out = torch.empty(layout.num_slots, num_heads, dim, dtype=x.dtype,
                      device=x.device)
    feats = (torch.empty(num_heads, layout.n_tiles, 3 * dim,
                         dtype=torch.float32, device=x.device)
             if pool else out)
    gather_kernel, _ = _kernels()
    gather_kernel[(layout.n_tiles, num_heads)](
        x, layout.gather_index, layout.slot_valid, heads, out, feats,
        layout.valid_count, x.stride(0), x.stride(1), out.stride(0),
        out.stride(1), feats.stride(0) if pool else 0,
        feats.stride(1) if pool else 0, HEAD_DIM=dim, TILE=_TILE, POOL=pool,
        num_warps=4)
    return out, (feats if pool else None)


def scatter_tiles_(out: torch.Tensor, tiled: torch.Tensor,
                   layout: tiling.TileLayout,
                   heads: torch.Tensor | None) -> None:
    """Same contract as tiling.scatter_tiles_ (padding slots are dropped)."""
    _check(out)
    if out.shape[0] != layout.seq_len + 1:
        raise ValueError(f'scatter buffer needs {layout.seq_len + 1} rows')
    heads = _heads(out, heads)
    _, scatter_kernel = _kernels()
    scatter_kernel[(layout.n_tiles, heads.numel())](
        tiled, layout.scatter_index, layout.slot_valid, heads, out,
        tiled.stride(0), tiled.stride(1), out.stride(0), out.stride(1),
        HEAD_DIM=out.shape[-1], TILE=_TILE, num_warps=4)
