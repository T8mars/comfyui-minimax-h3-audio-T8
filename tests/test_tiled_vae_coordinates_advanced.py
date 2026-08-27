from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
import torch

from h3_audio_t8_pkg.tiled_vae_coordinates_advanced import (
    _compile_decoder_forward,
    _compile_tiled_decode,
    build_tiled_vae_coordinate_compatibility,
    create_token_ids_global,
    probe_native_tiled_coordinates,
)


def create_token_ids(patch_dims, device, dtype):
    coords = []
    for dim_size in patch_dims:
        axis = torch.arange(0.5, dim_size, dtype=dtype, device=device)
        coords.append(2.0 * axis / dim_size - 1.0)
    return torch.stack(torch.meshgrid(*coords, indexing="ij"), dim=-1).flatten(0, 2).unsqueeze(0)


class ViT3DDecoder(torch.nn.Module):
    def forward(self, x):
        batch, _channels, latent_t, latent_h, latent_w = x.shape
        return create_token_ids(
            (latent_t, latent_h, latent_w), x.device, x.dtype
        ).expand(batch, -1, -1)


class MiniMaxH3VideoVAE(torch.nn.Module):
    vae_ratio = 1

    def __init__(self):
        super().__init__()
        self.decoder = ViT3DDecoder()
        self.post_quant_conv = torch.nn.Identity()

    def _decode_pixels(self, z):
        return self.decoder(self.post_quant_conv(z))

    def split_tiles(self, size):
        if size <= 2:
            return [0], [size], []
        return [0, 2], [2, size - 2], [0]

    def tiled_decode(self, z):
        height, width = z.shape[-2] * self.vae_ratio, z.shape[-1] * self.vae_ratio
        y_idx, y_len, _ = self.split_tiles(height)
        x_idx, x_len, _ = self.split_tiles(width)
        tiles = []
        for i_pos, i_len in zip(y_idx, y_len):
            zi, zl = i_pos // self.vae_ratio, i_len // self.vae_ratio
            for j_pos, j_len in zip(x_idx, x_len):
                zj, zw = j_pos // self.vae_ratio, j_len // self.vae_ratio
                tile = self._decode_pixels(z[..., zi:zi + zl, zj:zj + zw])
                tiles.append(tile)
        return tiles


class _VAEWrapper:
    def __init__(self):
        self.first_stage_model = MiniMaxH3VideoVAE()
        self.patcher = object()


def test_global_token_ids_match_full_grid_slices():
    full = create_token_ids_global((1, 4, 6), "cpu", torch.float32)
    tile = create_token_ids_global(
        (1, 2, 3),
        "cpu",
        torch.float32,
        full_dims=(1, 4, 6),
        offset=(0, 2, 3),
    )
    expected = full.reshape(1, 1, 4, 6, 3)[:, :, 2:4, 3:6].reshape(1, -1, 3)
    assert torch.equal(tile, expected)


def test_default_global_token_ids_equal_original_formula():
    expected = create_token_ids((2, 3, 4), "cpu", torch.float32)
    actual = create_token_ids_global((2, 3, 4), "cpu", torch.float32)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    "full_dims,offset",
    [((1, 2), (0, 0, 0)), ((1, 2, 2), (0, -1, 0)), ((1, 2, 2), (0, 1, 1))],
)
def test_invalid_global_coordinate_requests_fail_closed(full_dims, offset):
    with pytest.raises(ValueError):
        create_token_ids_global(
            (1, 2, 2), "cpu", torch.float32, full_dims=full_dims, offset=offset
        )


def test_decoder_and_tiled_source_transformers_supply_full_grid_coordinates():
    decoder_forward, decoder_source = _compile_decoder_forward(ViT3DDecoder.forward)
    tiled_decode, tiled_source = _compile_tiled_decode(MiniMaxH3VideoVAE.tiled_decode)
    assert "full_dims=None" in decoder_source.replace(" ", "")
    assert "_t8_create_token_ids_global" in decoder_source
    assert "offset=(0, zi, zj)" in tiled_source

    stage = MiniMaxH3VideoVAE()
    stage.decoder.forward = decoder_forward.__get__(stage.decoder, type(stage.decoder))
    stage._decode_pixels = (
        lambda z, full_dims=None, offset=None: stage.decoder(
            z, full_dims=full_dims, offset=offset
        )
    )
    tiles = tiled_decode(stage, torch.zeros(1, 1, 1, 4, 4))
    assert len(tiles) == 4
    assert not torch.equal(tiles[0], tiles[1])
    assert not torch.equal(tiles[0], tiles[2])


def test_builder_clones_methods_without_copying_weights_or_mutating_source():
    source = _VAEWrapper()
    original_stage = source.first_stage_model
    original_decoder = original_stage.decoder
    patched, report_json = build_tiled_vae_coordinate_compatibility(
        source, "apply_global_coordinates_exp"
    )
    report = json.loads(report_json)
    assert patched is not source
    assert patched.first_stage_model is not original_stage
    assert patched.first_stage_model.decoder is not original_decoder
    assert patched.first_stage_model.decoder._modules is not original_decoder._modules
    assert patched.patcher is source.patcher
    assert report["status"] == "compatibility_clone_ready"
    assert report["source_vae_unchanged"] is True
    assert probe_native_tiled_coordinates(patched)["available"] is True
    assert probe_native_tiled_coordinates(source)["available"] is False


def test_builder_defaults_to_report_only_after_negative_real_validation():
    source = _VAEWrapper()
    output, report_json = build_tiled_vae_coordinate_compatibility(source)
    report = json.loads(report_json)
    assert output is source
    assert report["status"] == "report_only_abstain_real_validation_failed"
    assert report["candidate_applied"] is False
    assert "stronger grid/stripe artifacts" in report["warning"]


def test_registration_is_append_only():
    from h3_audio_t8_pkg.nodes import MiniMaxH3AudioT8Extension

    node_ids = [
        node.define_schema().node_id
        for node in asyncio.run(MiniMaxH3AudioT8Extension().get_node_list())
    ]
    assert node_ids[-1] == "MiniMaxH3GlobalCoordinateTiledVAET8Advanced"
    from h3_audio_t8_pkg.nodes_tiled_vae_coordinates_advanced import (
        MiniMaxH3GlobalCoordinateTiledVAET8Advanced,
    )

    schema = MiniMaxH3GlobalCoordinateTiledVAET8Advanced.define_schema()
    mode = next(item for item in schema.inputs if item.id == "mode")
    assert schema.is_experimental is True
    assert mode.default == "report_only"


def test_official_core_compatibility_workflow_is_frontend_importable_and_documented():
    path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "workflows"
        / "20-core-compatibility"
        / "2026-08-28_H3_Official_Core_Compatibility_Advanced.json"
    )
    workflow = json.loads(path.read_text(encoding="utf-8"))
    node_types = [node["type"] for node in workflow["nodes"]]
    assert workflow["version"] == 0.4
    assert workflow["last_node_id"] == max(node["id"] for node in workflow["nodes"])
    assert workflow["last_link_id"] == max(link[0] for link in workflow["links"])
    for required in (
        "MiniMaxH3AVLatentBuilderT8Advanced",
        "MiniMaxH3AttentionHooksT8Advanced",
        "MiniMaxH3ForwardSyncOptimizationT8Advanced",
        "MiniMaxH3GlobalCoordinateTiledVAET8Advanced",
    ):
        assert node_types.count(required) == 1
    notes = "\n".join(
        str(node["widgets_values"])
        for node in workflow["nodes"]
        if node["type"] == "MarkdownNote"
    )
    assert "不要全部强行串联" in notes
    assert "不会因为用户模型指纹不同而报错" in notes
