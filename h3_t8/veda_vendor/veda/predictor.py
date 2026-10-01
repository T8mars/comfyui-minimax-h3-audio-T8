# SPDX-License-Identifier: MIT
# Vendored from veda-sparse/Miowtion at 3bc5e5ae53e8c0d4539f2bc872265164947d013a.
"""Tile-score predictor: which key tiles each query tile should attend.

Per tile, q and k are pooled into [mean | max | min] (3D features). Per
layer and head a residual projection P maps them to D dims:
    q_hat = pool_q @ P_q[h] + mean_q,   k_hat likewise,
and block logits are q_hat . k_hat / sqrt(D) in fp32.

P is initialized N(0, 1e-4), so an untrained predictor already equals
mean-pooled QK, a usable block score.

The predictor is a side branch: pooling runs on detached activations under
no_grad, and gradients never reach the trunk.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from . import tiling

INIT_STD = 1e-4


@torch.no_grad()
def pool_tiles(x: torch.Tensor, layout: tiling.TileLayout) -> torch.Tensor:
    """Masked mean/max/min over the real rows of every tile.

    Args:
        x: [N, H', D] bf16 tile-ordered rows (padding slots are zero), as
            returned by tiling.gather_tiles.
        layout: Its tile layout.

    Returns:
        [H', n_tiles, 3D] fp32 features; exactly 0 for empty tiles.
    """
    n = layout.n_tiles
    tiles = x.detach().view(n, tiling.TILE_SIZE, *x.shape[1:])  # [n,128,H,D]
    count = layout.valid_count.clamp(min=1).to(torch.float32)
    # Sum bf16 with fp32 accumulation instead of upcasting the whole tensor.
    mean = tiles.sum(dim=1, dtype=torch.float32) / count[:, None, None]
    # Padding rows are 0, which is a legal value, so max/min are recomputed
    # with masking only for the partial tiles.
    tmax = tiles.amax(dim=1)
    tmin = tiles.amin(dim=1)
    partial = layout.partial_tiles
    if partial.numel():
        sub = tiles.index_select(0, partial)
        valid = (torch.arange(tiling.TILE_SIZE, device=x.device)[None, :]
                 < layout.valid_count.index_select(0, partial)[:, None])
        valid = valid[:, :, None, None]
        tmax.index_copy_(0, partial, sub.masked_fill(
            ~valid, float('-inf')).amax(dim=1))
        tmin.index_copy_(0, partial, sub.masked_fill(
            ~valid, float('inf')).amin(dim=1))
    feats = torch.cat([mean, tmax.float(), tmin.float()], dim=-1)
    # where (not multiply): -inf * 0 would be NaN for empty tiles.
    feats = torch.where(layout.kv_ok[:, None, None], feats, 0.0)
    return feats.permute(1, 0, 2).contiguous()  # [H', n, 3D]


class LayerPredictor(nn.Module):
    """Projections of one layer: proj_q, proj_k [num_heads, 3D, D]."""

    def __init__(self, num_heads: int, head_dim: int):
        super().__init__()
        self.head_dim = head_dim
        self.proj_q = nn.Parameter(torch.empty(num_heads, 3 * head_dim,
                                               head_dim))
        self.proj_k = nn.Parameter(torch.empty(num_heads, 3 * head_dim,
                                               head_dim))
        nn.init.normal_(self.proj_q, std=INIT_STD)
        nn.init.normal_(self.proj_k, std=INIT_STD)

    def embed(self, feats: torch.Tensor, heads: torch.Tensor,
              proj: torch.Tensor) -> torch.Tensor:
        """[H', n, 3D] fp32 features -> [H', n, D] tile embeddings."""
        mean = feats[..., :self.head_dim]
        return torch.bmm(feats, proj.index_select(0, heads).float()) + mean

    def forward(self, feats_q: torch.Tensor, feats_k: torch.Tensor,
                heads: torch.Tensor) -> torch.Tensor:
        """Block logits [H', n_q, n_k] fp32.

        Args:
            feats_q: [H', n_q, 3D] pooled query features.
            feats_k: [H', n_k, 3D] pooled key features.
            heads: [H'] int64 global head indices of the group.
        """
        q_hat = self.embed(feats_q, heads, self.proj_q)
        k_hat = self.embed(feats_k, heads, self.proj_k)
        return torch.bmm(q_hat, k_hat.transpose(1, 2)) / math.sqrt(
            self.head_dim)


class TileScorePredictor(nn.Module):
    """All layers: `layers.{i}.proj_q` / `layers.{i}.proj_k`.

    With 50 layers x 56 heads x 384 x 128 x 2 this is 275M parameters. It is
    kept fully replicated (never FSDP-sharded): dim 0 is the head axis and
    per-head indexing of a sharded DTensor is not supported; replication
    also costs one gradient all-reduce instead of an all-gather per layer.
    """

    def __init__(self, num_layers: int, num_heads: int, head_dim: int):
        super().__init__()
        self.layers = nn.ModuleList(LayerPredictor(num_heads, head_dim)
                                    for _ in range(num_layers))

    def scores(self, layer: int, q_tiles: torch.Tensor, k_tiles: torch.Tensor,
               layout: tiling.TileLayout, heads: torch.Tensor) -> torch.Tensor:
        """Block logits of one head group.

        Args:
            layer: Layer index.
            q_tiles: [N, H', D] tile-ordered queries.
            k_tiles: [N, H', D] tile-ordered keys.
            layout: Tile layout of the group.
            heads: [H'] global head indices.

        Returns:
            [H', n_tiles, n_tiles] fp32 logits.
        """
        feats_q = pool_tiles(q_tiles, layout)
        feats_k = pool_tiles(k_tiles, layout)
        return self.layers[layer](feats_q, feats_k, heads)
