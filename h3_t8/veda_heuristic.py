"""Independent 64-token TripPool sparse-attention experiment for native H3.

This is the model-free, head-aware route from the BSAI VedaSparse interface,
not the released Miowtion 128-token trained predictor or its FA4 kernel.
Core supplies the exact packed video geometry; no global Core monkeypatch or
silent scorer-weight fallback is used.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .veda_flex import compiled_flex, flex_block_mask, require_flex_runtime


TILE_SIZE = 64
PRESETS = {
    "balanced": ((4, 4, 4),),
    "dual_fast": ((4, 4, 4), (8, 4, 2)),
    "head_aware": ((4, 4, 4), (8, 4, 2), (2, 4, 8), (4, 8, 2)),
    "temporal_first": ((8, 4, 2),),
    "spatial_first": ((2, 4, 8),),
    "extreme_spatial": ((1, 8, 8),),
}
POOL_MODES = ("triplet", "maxmin", "avg")
SINK_MODES = ("exact_kv_and_rows", "off")


def parse_shapes(preset: str, custom: str = "") -> tuple[tuple[int, int, int], ...]:
    if preset == "custom":
        parts = [part.strip() for part in custom.split(";") if part.strip()]
        try:
            shapes = tuple(tuple(int(value.strip()) for value in part.split(","))
                           for part in parts)
        except ValueError as error:
            raise ValueError("Veda heuristic custom tiles must be t,h,w triples") from error
    elif preset in PRESETS:
        shapes = PRESETS[preset]
    else:
        raise ValueError("Unknown Veda heuristic head-tiling preset")
    if not shapes or len(shapes) > 16 or any(
            len(shape) != 3 or any(value <= 0 for value in shape)
            or math.prod(shape) != TILE_SIZE for shape in shapes):
        raise ValueError("Veda heuristic tile extents must be positive and multiply to 64")
    return shapes


@dataclass(frozen=True)
class Layout64:
    gather_index: torch.Tensor
    scatter_index: torch.Tensor
    pad_slots: torch.Tensor
    valid_count: torch.Tensor
    kv_ok: torch.Tensor
    video_tiles: int
    seq_len: int

    @property
    def num_tiles(self) -> int:
        return self.valid_count.numel()

    @property
    def num_slots(self) -> int:
        return self.num_tiles * TILE_SIZE


def build_layout(grid: tuple[int, int, int], video_start: int,
                 seq_len: int, shape: tuple[int, int, int],
                 device: torch.device | str = "cpu") -> Layout64:
    if (len(grid) != 3 or any(value <= 0 for value in grid)
            or video_start < 0 or video_start + math.prod(grid) != seq_len
            or len(shape) != 3 or math.prod(shape) != TILE_SIZE
            or any(value <= 0 for value in shape)):
        raise ValueError("Veda heuristic layout does not match the native target grid")
    t, h, w = grid
    pt, ph, pw = shape
    tp, hp, wp = (math.ceil(size / part) * part
                  for size, part in zip(grid, shape))
    rows = torch.full((tp, hp, wp), -1, dtype=torch.long)
    rows[:t, :h, :w] = video_start + torch.arange(math.prod(grid)).view(grid)
    video = rows.view(tp // pt, pt, hp // ph, ph, wp // pw, pw)
    video = video.permute(0, 2, 4, 1, 3, 5).reshape(-1, TILE_SIZE)
    order = torch.argsort(video < 0, dim=1, stable=True)
    video = torch.gather(video, 1, order)
    globals_ = torch.full((math.ceil(video_start / TILE_SIZE) * TILE_SIZE,),
                          -1, dtype=torch.long)
    globals_[:video_start] = torch.arange(video_start)
    perm = torch.cat((video.flatten(), globals_))
    valid_count = (perm.view(-1, TILE_SIZE) >= 0).sum(1).to(torch.int32)
    return Layout64(
        gather_index=perm.clamp(min=0).to(device),
        scatter_index=torch.where(perm < 0, seq_len, perm).to(device),
        pad_slots=torch.nonzero(perm < 0).view(-1).to(device),
        valid_count=valid_count.to(device),
        kv_ok=(valid_count > 0).to(device),
        video_tiles=video.shape[0], seq_len=seq_len,
    )


def _gather(x: torch.Tensor, layout: Layout64, heads: torch.Tensor) -> torch.Tensor:
    result = x[layout.gather_index[:, None], heads[None, :]]
    if layout.pad_slots.numel():
        result.index_fill_(0, layout.pad_slots, 0)
    return result


def _pool(tiled: torch.Tensor, layout: Layout64, mode: str) -> torch.Tensor:
    if mode not in POOL_MODES:
        raise ValueError("Unsupported TripPool mode")
    rows = tiled.detach().view(layout.num_tiles, TILE_SIZE, *tiled.shape[1:])
    valid = (torch.arange(TILE_SIZE, device=rows.device)[None, :]
             < layout.valid_count[:, None])[:, :, None, None]
    count = layout.valid_count.clamp(min=1).float()
    mean = rows.sum(dim=1, dtype=torch.float32) / count[:, None, None]
    features = [mean]
    if mode != "avg":
        maximum = rows.masked_fill(~valid, float("-inf")).amax(dim=1).float()
        minimum = rows.masked_fill(~valid, float("inf")).amin(dim=1).float()
        features = ([mean, maximum, minimum] if mode == "triplet"
                    else [maximum, minimum])
    pooled = torch.cat(features, dim=-1)
    pooled = torch.where(layout.kv_ok[:, None, None], pooled, 0.0)
    return pooled.permute(1, 0, 2).contiguous()


def _selected_blocks(q_features: torch.Tensor, k_features: torch.Tensor,
                     layout: Layout64, keep_ratio: float,
                     sink_conditioning: str = "exact_kv_and_rows") -> torch.Tensor:
    heads, count, features = q_features.shape
    if (k_features.shape != q_features.shape or not 0 < keep_ratio <= 1
            or sink_conditioning not in SINK_MODES):
        raise ValueError("Veda heuristic score inputs or keep ratio are invalid")
    n_video = layout.video_tiles
    scores = torch.bmm(q_features[:, :n_video],
                       k_features[:, :n_video].transpose(-1, -2)) / math.sqrt(features)
    scores = scores.masked_fill(~layout.kv_ok[None, None, :n_video], float("-inf"))
    indices = torch.arange(n_video, device=scores.device)
    scores[:, indices, indices] = float("inf")
    topk = min(n_video, max(1, math.ceil(n_video * keep_ratio)))
    chosen = scores.topk(topk, dim=-1).indices
    blocks = torch.zeros(heads, count, count, dtype=torch.bool, device=scores.device)
    blocks[:, :n_video, :n_video].scatter_(2, chosen, True)
    if sink_conditioning == "exact_kv_and_rows":
        blocks[:, :, n_video:] = layout.kv_ok[n_video:]
    blocks[:, n_video:, :] = layout.kv_ok
    return blocks


@torch.no_grad()
def attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *,
           video_start: int, grid: tuple[int, int, int], keep_ratio: float,
           shapes: tuple[tuple[int, int, int], ...], mode: str = "triplet",
           max_copy_mib: int = 16, tile_cache: dict | None = None,
           sink_conditioning: str = "exact_kv_and_rows") -> torch.Tensor:
    """Return [S,H,D] using 64-token heuristic video blocks and exact globals."""
    if q.ndim != 3 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("Veda heuristic requires matching [sequence, heads, dim] QKV")
    sequence, heads, dim = q.shape
    if (not shapes or not 0 < keep_ratio <= 1 or not 1 <= max_copy_mib <= 256
            or mode not in POOL_MODES or sink_conditioning not in SINK_MODES
            or video_start + math.prod(grid) != sequence):
        raise ValueError("Veda heuristic configuration does not match the H3 layout")
    require_flex_runtime(q.device)
    cache = {} if tile_cache is None else tile_cache
    output = q.new_zeros(sequence + 1, heads, dim)
    groups: dict[tuple[int, int, int], list[int]] = {}
    for head in range(heads):
        groups.setdefault(shapes[head % len(shapes)], []).append(head)
    for shape, group in groups.items():
        key = (shape, grid, video_start, sequence, q.device)
        layout = cache.get(key)
        if layout is None:
            layout = build_layout(grid, video_start, sequence, shape, q.device)
            cache[key] = layout
        bytes_per_head = layout.num_slots * dim * q.element_size()
        chunk_size = min(len(group), max(1, max_copy_mib * 1024 * 1024 // bytes_per_head))
        group_heads = torch.tensor(group, dtype=torch.long, device=q.device)
        for chunk in group_heads.split(chunk_size):
            q_tiles = _gather(q, layout, chunk)
            k_tiles = _gather(k, layout, chunk)
            v_tiles = _gather(v, layout, chunk)
            q_features = _pool(q_tiles, layout, mode)
            k_features = _pool(k_tiles, layout, mode)
            selected = _selected_blocks(
                q_features, k_features, layout, keep_ratio, sink_conditioning)
            real_heads = len(chunk)
            missing = chunk_size - real_heads
            if missing:
                q_tiles = torch.cat((q_tiles, q_tiles[:, -1:, :].expand(-1, missing, -1)), dim=1)
                k_tiles = torch.cat((k_tiles, k_tiles[:, -1:, :].expand(-1, missing, -1)), dim=1)
                v_tiles = torch.cat((v_tiles, v_tiles[:, -1:, :].expand(-1, missing, -1)), dim=1)
                selected = torch.cat((selected, selected[-1:, :, :].expand(missing, -1, -1)), dim=0)
            block_mask = flex_block_mask(selected, layout.valid_count, TILE_SIZE)
            result = compiled_flex()(
                q_tiles.transpose(0, 1)[None],
                k_tiles.transpose(0, 1)[None],
                v_tiles.transpose(0, 1)[None],
                block_mask=block_mask,
                kernel_options={"BLOCK_M": TILE_SIZE, "BLOCK_N": TILE_SIZE},
            )
            output[layout.scatter_index[:, None], chunk[None, :]] = (
                result[0].transpose(0, 1)[:, :real_heads])
    return output[:sequence]
