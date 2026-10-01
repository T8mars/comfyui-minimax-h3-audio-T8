# SPDX-License-Identifier: MIT
# Vendored from veda-sparse/Miowtion at 3bc5e5ae53e8c0d4539f2bc872265164947d013a.
"""Block masks from tile scores.

Rules (the tile search's oracle uses exactly the same rules):
  1. Global tiles are dense in both directions: every query tile sees every
     global key tile, and global query tiles see every tile.
  2. Only the video -> video quadrant is sparse (top-k per query tile).
  3. Budgets are equal-kernel-cost: with n_ideal = ceil(real tokens / 128),
     a query tile keeps ratio * n_ideal^2 / n_tiles key tiles. The kernel
     pays for every kept tile as a full 128x128 block, so a padded plan must
     not get a larger budget than an unpadded one.
  4. The fractional part of the budget is spread by a Bresenham pattern
     indexed by the query tile number, so the mean kept count per row equals
     the budget exactly (rounding the budget moves real sparsity by 20-50%
     at high sparsity).
  5. A query tile's own tile is forced in (+inf score) and consumes budget.
  6. Empty key tiles (valid_count == 0) are never selected.
  7. With condition spans tiled, the video columns split into a reference
     block [0, n_ref_tiles) and a target block [n_ref_tiles, n_video_tiles).
     Each block runs its own top-k with its own budget; the diagonal is only
     forced inside the block that contains it. A single pooled top-k starves
     the reference block (spatial neighbours in the target always win).
"""

from __future__ import annotations

import dataclasses
import functools
import math

import torch

from . import tiling

_NEG_INF = float('-inf')
_POS_INF = float('inf')


@dataclasses.dataclass(frozen=True)
class Budget:
    """Keep budget of one column block: a ratio or an absolute tile count.

    Attributes:
        ratio: Keep ratio of the equal-cost budget; >= 1 keeps everything.
        tiles: Absolute key tiles per query tile (overrides ratio).
    """

    ratio: float | None = None
    tiles: float | None = None

    def __post_init__(self):
        if (self.ratio is None) == (self.tiles is None):
            raise ValueError('set exactly one of ratio / tiles')
        value = self.ratio if self.tiles is None else self.tiles
        if value <= 0:
            raise ValueError(f'budget must be positive: {self}')

    @property
    def keeps_all(self) -> bool:
        return self.ratio is not None and self.ratio >= 1.0

    def per_row(self, real_tokens: int, n_tiles: int) -> float:
        if self.tiles is not None:
            return float(self.tiles)
        n_ideal = math.ceil(real_tokens / tiling.TILE_SIZE)
        return self.ratio * n_ideal * n_ideal / n_tiles


def split_budget(budget: float, n_cols: int) -> tuple[int, int, float]:
    """(k_lo, k_hi, frac) with 1 <= k_lo <= k_hi <= n_cols."""
    if budget >= n_cols:
        return n_cols, n_cols, 0.0
    k_lo = min(max(1, math.floor(budget)), n_cols)
    # Rounded so that e.g. 2.3 - 2 gives 0.3, not 0.29999999999999982,
    # which would drop one extra tile per 1/frac rows.
    frac = round(min(max(budget - k_lo, 0.0), 1.0), 12)
    return k_lo, min(k_lo + 1, n_cols), frac


@functools.lru_cache(maxsize=256)
def _bresenham_cpu(n_rows: int, frac: float) -> torch.Tensor:
    ramp = torch.floor(torch.arange(n_rows + 1, dtype=torch.float64) * frac)
    return ramp[1:] > ramp[:-1]


def bresenham_extra(n_rows: int, frac: float,
                    device: torch.device | str) -> torch.Tensor:
    """[n_rows] bool: row r keeps k_hi iff floor((r+1)f) > floor(r f)."""
    return _bresenham_cpu(n_rows, float(frac)).to(device)


@dataclasses.dataclass(frozen=True)
class ColumnBlock:
    """A sparse column block of the video quadrant."""

    start: int
    stop: int
    budget: Budget
    real_tokens: int


def column_blocks(layout: tiling.TileLayout, target_budget: Budget,
                  ref_budget: Budget | None = None) -> list[ColumnBlock]:
    """Target block, preceded by a reference block when references are tiled.

    Raises:
        ValueError: If references are tiled but no ref budget is given.
    """
    target = ColumnBlock(layout.n_ref_tiles, layout.n_video_tiles,
                         target_budget, layout.target_tokens)
    if layout.n_ref_tiles == 0:
        return [target]
    if ref_budget is None:
        raise ValueError('tiled references need their own budget')
    return [ColumnBlock(0, layout.n_ref_tiles, ref_budget, layout.ref_tokens),
            target]


@dataclasses.dataclass
class Selection:
    """Selected video key tiles per video query tile.

    Attributes:
        index: [H', n_video, K] int64 global key tile ids (all blocks
            concatenated).
        keep: [H', n_video, K] bool, which entries are selected.
    """

    index: torch.Tensor
    keep: torch.Tensor


@torch.no_grad()
def select_video_blocks(scores: torch.Tensor, layout: tiling.TileLayout,
                        blocks: list[ColumnBlock],
                        rows: torch.Tensor | None = None) -> Selection:
    """Top-k over the video quadrant of `scores` (rules 2-7).

    Args:
        scores: [H', R, >= n_video] block scores of query tiles `rows` (any
            score monotone in importance: logits or heat).
        layout: Tile layout.
        blocks: Column blocks from column_blocks().
        rows: [R] int64 video query tile ids; None means all n_video tiles.
            The Bresenham pattern and the diagonal follow the tile id, not
            the position within `rows`.

    Returns:
        The selection for the R query tiles.
    """
    heads = scores.shape[0]
    n_video = layout.n_video_tiles
    device = scores.device
    if rows is None:
        rows = torch.arange(n_video, device=device)
    num_rows = rows.numel()
    indices, keeps = [], []
    for block in blocks:
        n_cols = block.stop - block.start
        if block.budget.keeps_all:
            idx = torch.arange(block.start, block.stop, device=device)
            indices.append(idx.expand(heads, num_rows, n_cols))
            keeps.append(layout.kv_ok[block.start:block.stop].expand(
                heads, num_rows, n_cols))
            continue
        s = scores[:, :, block.start:block.stop].float().clone()
        s.masked_fill_(~layout.kv_ok[None, None, block.start:block.stop],
                       _NEG_INF)
        own = torch.nonzero((rows >= block.start)
                            & (rows < block.stop)).view(-1)
        s[:, own, rows[own] - block.start] = _POS_INF
        budget = block.budget.per_row(block.real_tokens, n_cols)
        k_lo, k_hi, frac = split_budget(budget, n_cols)
        vals, idx = torch.topk(s, k_hi, dim=-1, sorted=True)
        extra = bresenham_extra(n_video, frac, device).index_select(0, rows)
        allowed = k_lo + extra.to(torch.long)
        keep = (torch.arange(k_hi, device=device)[None, None, :]
                < allowed[None, :, None]) & (vals > _NEG_INF)
        indices.append(idx + block.start)
        keeps.append(keep)
    return Selection(torch.cat(indices, -1), torch.cat(keeps, -1))


@torch.no_grad()
def dense_block_mask(selection: Selection, layout: tiling.TileLayout,
                     global_rows: bool = True) -> torch.Tensor:
    """Block mask of a selection (rules 1-7), the kernels' input.

    Args:
        selection: Video query tiles' selected key tiles (all n_video rows,
            or a subset of R rows).
        layout: Tile layout.
        global_rows: Append the global query rows, which see every non-empty
            tile (the full-sequence case). False yields the selection's R
            rows only, e.g. to run sampled query tiles.

    Returns:
        [H', n_tiles, n_tiles] bool, or [H', R, n_tiles] without global rows.
        Global key columns are always kept.
    """
    heads, num_rows, _ = selection.index.shape
    n, n_video = layout.n_tiles, layout.n_video_tiles
    if global_rows and num_rows != n_video:
        raise ValueError('global_rows needs a selection of all video rows')
    mask = torch.zeros(heads, n if global_rows else num_rows, n,
                       dtype=torch.bool, device=selection.index.device)
    # Global columns start after the layout's video tiles, whatever the
    # number of selected rows.
    mask[:, :num_rows].scatter_(2, selection.index, selection.keep)
    mask[:, :, n_video:] = layout.kv_ok[n_video:]
    if global_rows:
        mask[:, n_video:, :] = layout.kv_ok
    return mask


def kept_tiles_per_row(selection: Selection) -> torch.Tensor:
    """[H', n_video] number of selected video tiles (budget accounting)."""
    return selection.keep.sum(-1)
