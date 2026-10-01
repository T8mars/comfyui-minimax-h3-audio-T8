# SPDX-License-Identifier: MIT
# Vendored from veda-sparse/Miowtion at 3bc5e5ae53e8c0d4539f2bc872265164947d013a.
"""Tile permutations: reorder packed rows so every 128 rows form a 3D tile.

A tile shape (tt, th, tw) with tt*th*tw = 128 cuts a (T, H, W) token grid
into boxes. Each video-like span of the sequence is tiled with its own grid
and shape; all other real rows (text, audio, untiled conditions) are
"global" and are chunked into 128-row tiles in sequence order, after all
video tiles.

Tile shapes are dynamic in two ways, and both must be handled exactly:
  * Per head: heads of one layer may use different shapes (at most two per
    layer), each with its own permutation.
  * Per tile: a grid that is not a multiple of the shape is padded; padded
    slots hold -1. Inside every tile the real rows are stably moved to the
    front, so a tile's real rows are always a prefix of length
    valid_count[tile] (0..128). Kernels and pooling rely on this prefix
    property instead of per-row masks.
"""

from __future__ import annotations

import dataclasses
import itertools
from collections.abc import Sequence

import torch

TILE_SIZE = 128


@dataclasses.dataclass(frozen=True, order=True)
class TileShape:
    """A (t, h, w) box of TILE_SIZE tokens."""

    t: int
    h: int
    w: int

    def __post_init__(self):
        if self.t * self.h * self.w != TILE_SIZE:
            raise ValueError(f'tile {self} does not hold {TILE_SIZE} tokens')

    def __str__(self) -> str:
        return f'{self.t}x{self.h}x{self.w}'

    @classmethod
    def parse(cls, text: str) -> TileShape:
        t, h, w = (int(v) for v in text.split('x'))
        return cls(t, h, w)

    def transposed(self) -> TileShape:
        """Swaps the h and w extents."""
        return TileShape(self.t, self.w, self.h)

    def padded_grid(self, grid: Sequence[int]) -> tuple[int, int, int]:
        return tuple(-(-g // s) * s for g, s in zip(grid, (self.t, self.h,
                                                          self.w)))

    def num_tiles(self, grid: Sequence[int]) -> int:
        tp, hp, wp = self.padded_grid(grid)
        return tp * hp * wp // TILE_SIZE

    def padding_ratio(self, grid: Sequence[int]) -> float:
        """Padded volume / real volume - 1."""
        t, h, w = grid
        return self.num_tiles(grid) * TILE_SIZE / (t * h * w) - 1.0

    def aspect_spread(self) -> float:
        """max / min extent; 1.0 for the most cubic shape."""
        return max(self.t, self.h, self.w) / min(self.t, self.h, self.w)


def all_shapes() -> list[TileShape]:
    """All 36 power-of-two triples with product TILE_SIZE."""
    exponent = TILE_SIZE.bit_length() - 1
    shapes = []
    for i, j in itertools.product(range(exponent + 1), repeat=2):
        if i + j <= exponent:
            shapes.append(TileShape(2**i, 2**j, 2**(exponent - i - j)))
    return sorted(shapes)


def candidate_shapes(grid: Sequence[int]) -> list[TileShape]:
    """Shapes whose extents fit inside `grid`."""
    return [s for s in all_shapes()
            if s.t <= grid[0] and s.h <= grid[1] and s.w <= grid[2]]


def least_padding_shape(grid: Sequence[int]) -> TileShape:
    """Least padding; ties broken by the most cubic, then lexicographic."""
    shapes = candidate_shapes(grid) or all_shapes()
    return min(shapes, key=lambda s: (s.num_tiles(grid), s.aspect_spread(),
                                      (s.t, s.h, s.w)))


@dataclasses.dataclass(frozen=True)
class TiledSpan:
    """A span of packed rows [start, start + T*H*W) tiled with `shape`."""

    start: int
    grid: tuple[int, int, int]
    shape: TileShape

    @property
    def num_rows(self) -> int:
        t, h, w = self.grid
        return t * h * w


def span_tiles(span: TiledSpan) -> torch.Tensor:
    """[n_tiles, 128] packed row ids of one span, -1 on padding.

    Tile order is (h-block, w-block, t-block) outer to inner; rows inside a
    tile are t, h, w row-major, then stably compacted to the front.
    """
    t, h, w = span.grid
    s = span.shape
    tp, hp, wp = s.padded_grid(span.grid)
    grid = torch.full((tp, hp, wp), -1, dtype=torch.long)
    grid[:t, :h, :w] = span.start + torch.arange(t * h * w).view(t, h, w)
    tiles = grid.view(tp // s.t, s.t, hp // s.h, s.h, wp // s.w, s.w)
    tiles = tiles.permute(2, 4, 0, 1, 3, 5).reshape(-1, TILE_SIZE)
    order = torch.argsort((tiles < 0).to(torch.int8), dim=1, stable=True)
    return torch.gather(tiles, 1, order)


@dataclasses.dataclass
class TileLayout:
    """One permutation of the packed sequence and its derived constants.

    All tensors live on `device`; they are built once per (spans, shapes,
    device) and reused on every call, because recomputing them (nonzero,
    tolist) forces device synchronizations on the hot path.

    Attributes:
        perm: [N] int64 packed row of every permuted slot, -1 on padding;
            N = n_tiles * 128.
        valid_count: [n_tiles] int32 real rows per tile (a prefix).
        n_video_tiles: Tiles of all tiled spans (the video quadrant).
        n_ref_tiles: Leading video tiles that belong to condition spans;
            tiles [n_ref_tiles, n_video_tiles) belong to the target.
        n_global_tiles: Trailing tiles of global rows.
        ref_tokens: Real rows of condition spans.
        target_tokens: Real rows of the target span.
        used: Real rows [0, used) of the packed sequence.
        seq_len: Packed length; `scatter_index` sends padding to row seq_len.
        gather_index: [N] perm with padding redirected to row 0.
        scatter_index: [N] perm with padding redirected to row seq_len.
        pad_slots: [P] slots of perm that are padding.
        partial_tiles: [Q] tiles with 0 < valid_count < 128.
        kv_ok: [n_tiles] bool, tile has at least one real row.
        full_tile: [n_tiles] bool, valid_count == 128.
        slot_valid: [N] int32, 1 where the slot holds a real row.
    """

    perm: torch.Tensor
    valid_count: torch.Tensor
    n_video_tiles: int
    n_ref_tiles: int
    n_global_tiles: int
    ref_tokens: int
    target_tokens: int
    used: int
    seq_len: int
    gather_index: torch.Tensor
    scatter_index: torch.Tensor
    pad_slots: torch.Tensor
    partial_tiles: torch.Tensor
    kv_ok: torch.Tensor
    full_tile: torch.Tensor
    slot_valid: torch.Tensor

    @property
    def n_tiles(self) -> int:
        return self.n_video_tiles + self.n_global_tiles

    @property
    def num_slots(self) -> int:
        return self.n_tiles * TILE_SIZE


def build_tile_layout(spans: Sequence[TiledSpan], used: int, seq_len: int,
                      device: torch.device | str = 'cpu') -> TileLayout:
    """Builds the permutation for tiled spans; all other real rows are global.

    Args:
        spans: Tiled spans in packed order; the last one is the target.
        used: Real rows [0, used); padding rows [used, seq_len) are excluded.
        seq_len: Packed sequence length.
        device: Device of the derived tensors.

    Returns:
        The tile layout.

    Raises:
        ValueError: If spans overlap, are out of order or exceed `used`.
    """
    if not spans:
        raise ValueError('at least the target span must be tiled')
    covered = torch.zeros(used, dtype=torch.bool)
    tiles = []
    prev_stop = 0
    for span in spans:
        if span.start < prev_stop or span.start + span.num_rows > used:
            raise ValueError(f'span {span} overlaps or exceeds used={used}')
        prev_stop = span.start + span.num_rows
        covered[span.start:prev_stop] = True
        tiles.append(span_tiles(span))
    n_ref_tiles = sum(t.shape[0] for t in tiles[:-1])
    video = torch.cat(tiles)
    global_rows = torch.nonzero(~covered).view(-1)
    n_global = -(-global_rows.numel() // TILE_SIZE)
    global_tiles = torch.full((n_global * TILE_SIZE,), -1, dtype=torch.long)
    global_tiles[:global_rows.numel()] = global_rows
    perm = torch.cat([video.view(-1), global_tiles])
    valid_count = (perm.view(-1, TILE_SIZE) >= 0).sum(1).to(torch.int32)
    ref_tokens = sum(s.num_rows for s in spans[:-1])
    return TileLayout(
        perm=perm.to(device),
        valid_count=valid_count.to(device),
        n_video_tiles=video.shape[0],
        n_ref_tiles=n_ref_tiles,
        n_global_tiles=n_global,
        ref_tokens=ref_tokens,
        target_tokens=spans[-1].num_rows,
        used=used,
        seq_len=seq_len,
        gather_index=perm.clamp(min=0).to(device),
        scatter_index=torch.where(perm < 0, seq_len, perm).to(device),
        pad_slots=torch.nonzero(perm < 0).view(-1).to(device),
        partial_tiles=torch.nonzero(
            (valid_count > 0) & (valid_count < TILE_SIZE)).view(-1).to(device),
        kv_ok=(valid_count > 0).to(device),
        full_tile=(valid_count == TILE_SIZE).to(device),
        slot_valid=(perm >= 0).to(torch.int32).to(device),
    )


def gather_tiles(x: torch.Tensor, layout: TileLayout,
                 heads: torch.Tensor | None) -> torch.Tensor:
    """Permutes rows of x into tile order, zeroing padding slots.

    Args:
        x: [S, H, D] packed tensor.
        layout: Tile layout of this head group.
        heads: [H'] int64 head indices of the group, None for all heads.

    Returns:
        [N, H', D] in tile order (FA4-native seq-major layout). A tile view
        is `out.view(n_tiles, 128, H', D)`.
    """
    if heads is None:
        out = x.index_select(0, layout.gather_index)
    else:
        out = x[layout.gather_index[:, None], heads[None, :]]
    if layout.pad_slots.numel():
        out.index_fill_(0, layout.pad_slots, 0)
    return out


def scatter_tiles_(out: torch.Tensor, tiled: torch.Tensor,
                   layout: TileLayout, heads: torch.Tensor | None) -> None:
    """Writes tile-ordered rows back into out[S + 1, H, D] in place.

    Padding slots land in the extra row out[S], so the inverse permutation
    is a single index write with no masking.
    """
    if out.shape[0] != layout.seq_len + 1:
        raise ValueError(f'scatter buffer needs {layout.seq_len + 1} rows')
    if heads is None:
        out.index_copy_(0, layout.scatter_index, tiled)
    else:
        out[layout.scatter_index[:, None], heads[None, :]] = tiled
