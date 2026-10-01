"""Opt-in FlexAttention kernel for the pinned Veda tile predictor.

The author tile plans, predictor and selection rules are vendored unchanged
apart from package-relative imports. This file replaces *only* FA4's block
kernel on Windows; it does not claim to be the author's FA4 implementation.
"""

from __future__ import annotations

import importlib.util
import locale
import os

import torch
from torch.nn.attention.flex_attention import BlockMask, flex_attention

from .veda_vendor import tile_gather_triton
from .veda_vendor.veda import mask as veda_mask
from .veda_vendor.veda import predictor as veda_predictor
from .veda_vendor.veda import tiling


TILE_SIZE = tiling.TILE_SIZE
_COMPILED = None


def _windows_inductor_text_encoding_supported(platform_name: str, encoding: str) -> bool:
    return platform_name != "nt" or encoding.lower().replace("-", "") == "utf8"


def require_flex_runtime(device: torch.device) -> None:
    """Fail before changing MODEL when the selected CUDA kernel cannot run."""
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Veda Flex execution requires an NVIDIA CUDA device")
    if importlib.util.find_spec("triton") is None:
        raise RuntimeError(
            "Veda Flex execution needs a compatible Triton installation "
            "(on Windows, use triton-windows matching the current Torch/CUDA); "
            "the original H3 attention has not been replaced"
        )
    if not _windows_inductor_text_encoding_supported(
        os.name, locale.getpreferredencoding(False)
    ):
        raise RuntimeError(
            "Veda Flex on Windows requires a UTF-8 Python process for Torch "
            "Inductor's source templates. Set PYTHONUTF8=1 before launching "
            "ComfyUI, then restart it; the original MODEL has not been changed"
        )


def flex_block_mask(blocks: torch.Tensor, valid_count: torch.Tensor,
                    tile_size: int = TILE_SIZE) -> BlockMask:
    """Convert author selected tiles and valid prefixes without densifying tokens."""
    if blocks.ndim != 3 or blocks.dtype != torch.bool:
        raise ValueError("Veda selected tiles must be [head, query, key] bool")
    heads, n_query, n_key = blocks.shape
    if heads < 1 or n_query != n_key or valid_count.numel() != n_key or tile_size not in (64, 128):
        raise ValueError("Veda tile-mask dimensions differ")
    if blocks.device != valid_count.device:
        raise ValueError("Veda tile mask and valid counts must share a device")
    full = blocks & (valid_count == tile_size)[None, None, :]
    partial = blocks & (valid_count != tile_size)[None, None, :]

    def packed(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        counts = values.sum(-1).to(torch.int32).unsqueeze(0)
        # Keeping an n_key-width index buffer is small on the tile grid and
        # avoids synchronizing to the host to discover the per-row maximum.
        indices = torch.argsort(~values, dim=-1, stable=True).to(torch.int32).unsqueeze(0)
        return counts, indices

    partial_count, partial_indices = packed(partial)
    full_count, full_indices = packed(full)

    def valid_key(batch, head, q_idx, kv_idx):
        del batch, head, q_idx
        return kv_idx % tile_size < valid_count[kv_idx // tile_size]

    return BlockMask.from_kv_blocks(
        partial_count, partial_indices, full_count, full_indices,
        BLOCK_SIZE=tile_size, mask_mod=valid_key,
        seq_lengths=(n_query * tile_size, n_key * tile_size),
        compute_q_blocks=False,
    )


def compiled_flex():
    global _COMPILED
    if _COMPILED is None:
        # Author head groups have different padded sequence lengths. Start
        # with a dynamic graph: Dynamo's default static-first/generalize-later
        # policy can change Flex numerics between the cold and repeated call
        # even when QKV and selected blocks are identical.
        _COMPILED = torch.compile(flex_attention, fullgraph=True, dynamic=True)
    return _COMPILED


def _head_chunks(tile_layout: tiling.TileLayout, dtype: torch.dtype,
                 head_dim: int, max_copy_mib: int) -> int:
    bytes_per_head = tile_layout.num_slots * head_dim * torch.tensor([], dtype=dtype).element_size()
    return max(1, max_copy_mib * 1024 * 1024 // bytes_per_head)


@torch.no_grad()
def attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *,
           layer_index: int, target_start: int, target_grid: tuple[int, int, int],
           plan, predictor, keep_ratio: float, max_copy_mib: int = 64,
           tile_cache: dict | None = None,
           fused_tile_io: bool = False) -> torch.Tensor:
    """Return [S,H,D] on Core's post-QKNorm/post-RoPE QKV; T2VA only.

    The input remains Core's unpadded [S,H,D]. Tile slots have their own
    -1 padding and valid prefixes, and the scatter discards those slots.
    """
    if q.ndim != 3 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("Veda requires equal [sequence, heads, head_dim] Q/K/V")
    seq_len, heads, dim = q.shape
    if (not 0 <= target_start < seq_len or target_start +
            target_grid[0] * target_grid[1] * target_grid[2] != seq_len):
        raise ValueError("Veda target video must be the final contiguous H3 segment")
    if tuple(plan.grid) != tuple(target_grid):
        raise ValueError("Veda plan grid does not match the live Core video token grid")
    if not 0 < keep_ratio <= 1 or max_copy_mib < 1:
        raise ValueError("Veda keep ratio or copy budget is invalid")
    if not 0 <= layer_index < plan.num_layers or heads != len(plan.head_shape[layer_index]):
        raise ValueError("Veda predictor plan does not match the live attention layer")
    require_flex_runtime(q.device)
    if fused_tile_io and not tile_gather_triton.available():
        raise RuntimeError("Veda fused tile I/O requires the pinned Triton CUDA kernel")
    cache = {} if tile_cache is None else tile_cache
    out = q.new_zeros(seq_len + 1, heads, dim)
    layer = predictor.layers[layer_index]
    if next(layer.parameters()).device != q.device:
        layer.to(q.device)
        staged = True
    else:
        staged = False
    try:
        for group in plan.head_groups(layer_index, q.device):
            shape = group.shape
            key = (shape, seq_len, target_start, tuple(target_grid), q.device)
            tile_layout = cache.get(key)
            if tile_layout is None:
                tile_layout = tiling.build_tile_layout(
                    [tiling.TiledSpan(target_start, target_grid, shape)],
                    seq_len, seq_len, q.device,
                )
                cache[key] = tile_layout
            blocks = veda_mask.column_blocks(tile_layout, veda_mask.Budget(ratio=keep_ratio))
            chunk_heads = min(heads, _head_chunks(tile_layout, q.dtype, dim, max_copy_mib))
            for chunk in group.heads.split(
                    chunk_heads):
                if fused_tile_io:
                    q_tiles, q_features = tile_gather_triton.gather_and_pool(
                        q, tile_layout, chunk)
                    k_tiles, k_features = tile_gather_triton.gather_and_pool(
                        k, tile_layout, chunk)
                    v_tiles = tile_gather_triton.gather_tiles(v, tile_layout, chunk)
                else:
                    q_tiles = tiling.gather_tiles(q, tile_layout, chunk)
                    k_tiles = tiling.gather_tiles(k, tile_layout, chunk)
                    v_tiles = tiling.gather_tiles(v, tile_layout, chunk)
                    q_features = veda_predictor.pool_tiles(q_tiles, tile_layout)
                    k_features = veda_predictor.pool_tiles(k_tiles, tile_layout)
                scores = layer(q_features, k_features, chunk)
                selection = veda_mask.select_video_blocks(
                    scores[:, :tile_layout.n_video_tiles], tile_layout, blocks)
                selected = veda_mask.dense_block_mask(selection, tile_layout)
                # The official plan's per-shape head groups have many tail
                # sizes. Keeping a fixed head count per tile layout avoids
                # Dynamo's eight-specialization hard failure. Each dummy
                # head simply duplicates the last real head, and is dropped
                # before scattering; attention never mixes heads.
                real_heads = len(chunk)
                missing = chunk_heads - real_heads
                if missing:
                    q_tiles = torch.cat((q_tiles, q_tiles[:, -1:, :].expand(-1, missing, -1)), dim=1)
                    k_tiles = torch.cat((k_tiles, k_tiles[:, -1:, :].expand(-1, missing, -1)), dim=1)
                    v_tiles = torch.cat((v_tiles, v_tiles[:, -1:, :].expand(-1, missing, -1)), dim=1)
                    selected = torch.cat((selected, selected[-1:, :, :].expand(missing, -1, -1)), dim=0)
                block_mask = flex_block_mask(selected, tile_layout.valid_count)
                result = compiled_flex()(
                    q_tiles.transpose(0, 1)[None],
                    k_tiles.transpose(0, 1)[None],
                    v_tiles.transpose(0, 1)[None],
                    block_mask=block_mask,
                )
                scatter = (tile_gather_triton.scatter_tiles_ if fused_tile_io
                           else tiling.scatter_tiles_)
                scatter(out, result[0].transpose(0, 1)[:, :real_heads],
                        tile_layout, chunk)
        return out[:seq_len]
    finally:
        if staged:
            layer.to("cpu")
