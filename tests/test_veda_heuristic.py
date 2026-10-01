"""Model-free 64-token Veda heuristic geometry and mask contracts."""

import pytest
import torch

from h3_audio_t8_pkg.veda_flex import flex_block_mask
from h3_audio_t8_pkg.veda_heuristic import (
    TILE_SIZE, _gather, _pool, _selected_blocks, build_layout, parse_shapes,
)


def test_parse_head_tiling_presets_and_custom():
    assert parse_shapes("dual_fast") == ((4, 4, 4), (8, 4, 2))
    assert parse_shapes("custom", "4,4,4; 1,8,8") == ((4, 4, 4), (1, 8, 8))
    with pytest.raises(ValueError, match="multiply to 64"):
        parse_shapes("custom", "2,2,2")
    with pytest.raises(ValueError, match="triples"):
        parse_shapes("custom", "a,b,c")


@pytest.mark.parametrize("grid,shape", [
    ((7, 9, 16), (4, 4, 4)),
    ((37, 24, 24), (8, 4, 2)),
    ((3, 7, 11), (1, 8, 8)),
])
def test_layout_covers_each_real_packed_row_once(grid, shape):
    prefix = 141
    sequence = prefix + grid[0] * grid[1] * grid[2]
    layout = build_layout(grid, prefix, sequence, shape)
    real = layout.scatter_index[layout.scatter_index < sequence]
    assert torch.equal(real.sort().values, torch.arange(sequence))
    assert layout.video_tiles * TILE_SIZE >= sequence - prefix
    assert layout.valid_count.sum().item() == sequence
    data = torch.arange(sequence * 2, dtype=torch.float32).view(sequence, 2, 1)
    heads = torch.tensor([1, 0])
    tiled = _gather(data, layout, heads)
    output = torch.empty(sequence + 1, 2, 1)
    output[layout.scatter_index[:, None], heads[None, :]] = tiled
    assert torch.equal(output[:sequence], data)


def test_tripool_masks_partial_tiles_and_preserves_global_rows():
    layout = build_layout((1, 3, 5), 3, 18, (4, 4, 4))
    tensor = torch.arange(18, dtype=torch.float32).view(18, 1, 1)
    tiled = _gather(tensor, layout, torch.tensor([0]))
    features = _pool(tiled, layout, "triplet")
    assert features.shape == (1, layout.num_tiles, 3)
    assert features[0, 0].tolist() == pytest.approx([9.5, 16., 3.])
    assert features[0, layout.video_tiles].tolist() == pytest.approx([1., 2., 0.])
    selected = _selected_blocks(features, features, layout, 0.05)
    assert selected.shape == (1, layout.num_tiles, layout.num_tiles)
    assert selected[0, 0, 0] and selected[0, 1, 1]
    assert selected[0, :, layout.video_tiles].all()
    assert selected[0, layout.video_tiles].all()
    block_mask = flex_block_mask(selected, layout.valid_count, TILE_SIZE)
    assert block_mask.BLOCK_SIZE == (64, 64)
    assert torch.equal(block_mask.to_dense()[0, 0].bool(), selected[0])


def test_optional_sink_off_removes_only_global_kv_for_video_queries():
    layout = build_layout((1, 3, 5), 3, 18, (4, 4, 4))
    tensor = torch.arange(18, dtype=torch.float32).view(18, 1, 1)
    features = _pool(_gather(tensor, layout, torch.tensor([0])), layout, "triplet")
    selected = _selected_blocks(features, features, layout, 0.05, "off")
    global_row = layout.video_tiles
    assert not selected[0, :global_row, global_row:].any()
    assert selected[0, global_row:, :].all()
    assert selected[0, 0, 0] and selected[0, 1, 1]
    with pytest.raises(ValueError, match="score inputs or keep ratio"):
        _selected_blocks(features, features, layout, 0.05, "unknown")


def test_layout_rejects_guessed_geometry():
    with pytest.raises(ValueError, match="native target grid"):
        build_layout((7, 9, 16), 12, 1000, (4, 4, 4))
