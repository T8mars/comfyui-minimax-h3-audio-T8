from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from h3_audio_t8_pkg import fast_h3_vsa_advanced as vsa
from h3_audio_t8_pkg import h3_lora_compat_advanced as lora


def test_tile_geometry_is_bijective_and_pads_each_segment_independently():
    destination, lengths, prefix_blocks, padded_rows = vsa._geometry_cpu(
        (65, 3), (5, 3, 5)
    )
    source_rows = 65 + 3 + 5 * 3 * 5
    assert destination.numel() == source_rows
    assert torch.unique(destination).numel() == source_rows
    assert int(destination.min()) == 0
    assert int(destination.max()) < padded_rows
    assert lengths.dtype == torch.int32
    assert lengths.tolist()[:3] == [64, 1, 3]
    assert prefix_blocks == 3
    # Every live source lands in the live prefix of exactly one 64-row tile.
    for source, target in enumerate(destination.tolist()):
        block = target // vsa.FAST_H3_VSA_TILE_ROWS
        assert target % vsa.FAST_H3_VSA_TILE_ROWS < int(lengths[block])


def test_plain_t2va_geometry_rejects_reference_layout_without_crashing_dense_path():
    layout = SimpleNamespace(
        segments=[(0, 4, "text"), (4, 8, "ref_img"), (8, 10, "audio"), (10, 14, "video")],
        signature=(4, 1, 4, 4, 1),
    )
    with pytest.raises(vsa._NotPlainT2VALayout, match="plain T2VA"):
        vsa._plain_t2va_geometry(layout, 14, torch.device("cpu"))


def test_fastvideo_vsa_gate_extraction_is_structural_not_filename_based():
    gate = torch.zeros(8, 6)
    state = {
        "transformer_blocks.0.attn.to_gate_compress.set_weight": gate,
        "transformer_blocks.0.attn.to_q.lora_A.weight": torch.ones(2, 6),
    }
    gates, remaining = lora._extract_fastvideo_vsa_gate_weights(state)
    assert list(gates) == [0]
    assert gates[0] is gate
    assert list(remaining) == ["transformer_blocks.0.attn.to_q.lora_A.weight"]


def test_gate_attachment_requires_every_live_h3_block_and_scales_weights():
    class _Model:
        def __init__(self, diffusion, object_patches=None, attachments=None):
            self.diffusion = diffusion
            self.object_patches = dict(object_patches or {})
            self.attachments = dict(attachments or {})

        def get_model_object(self, name):
            assert name == "diffusion_model"
            return self.diffusion

        def clone(self):
            return _Model(self.diffusion, self.object_patches, self.attachments)

        def add_object_patch(self, path, value):
            self.object_patches[path] = value

        def set_attachments(self, key, value):
            self.attachments[key] = value

    blocks = [
        SimpleNamespace(attn=SimpleNamespace(qkv_proj=nn.Linear(6, 24, bias=False), heads=2, head_dim=4)),
        SimpleNamespace(attn=SimpleNamespace(qkv_proj=nn.Linear(6, 24, bias=False), heads=2, head_dim=4)),
    ]
    model = _Model(SimpleNamespace(blocks=blocks))
    weights = {0: torch.ones(8, 6), 1: torch.full((8, 6), 2.0)}
    patched, receipt = lora._attach_fastvideo_vsa_gates(model, weights, 0.5)
    assert receipt["attached_gate_count"] == 2
    first = patched.object_patches["diffusion_model.blocks.0.attn.to_gate_compress"]
    second = patched.object_patches["diffusion_model.blocks.1.attn.to_gate_compress"]
    assert torch.equal(first.weight, torch.full((8, 6), 0.5))
    assert torch.equal(second.weight, torch.ones(8, 6))
    with pytest.raises(ValueError, match="exactly one"):
        lora._attach_fastvideo_vsa_gates(model, {0: weights[0]}, 1.0)


def test_vsa_kernel_output_is_fused_back_to_h3_hidden_width(monkeypatch):
    class _Out(nn.Module):
        def forward(self, value):
            assert value.shape == (3, 4)
            return value

    attention = SimpleNamespace(
        heads=2,
        head_dim=2,
        qkv_proj=nn.Linear(4, 12, bias=False),
        q_norm=nn.LayerNorm(2),
        k_norm=nn.LayerNorm(2),
        to_gate_compress=nn.Linear(4, 4, bias=False),
        out_proj=_Out(),
    )
    monkeypatch.setattr(
        vsa,
        "_plain_t2va_geometry",
        lambda _layout, _sequence, _device: (
            torch.arange(3),
            torch.tensor([3], dtype=torch.int32),
            0,
            3,
        ),
    )
    monkeypatch.setattr(vsa, "_sol_attn_function", lambda: lambda q, k, value, **_kwargs: value)
    output = vsa._fast_h3_vsa_attention(
        attention,
        torch.randn(3, 4),
        None,
        {},
        object(),
    )
    assert output.shape == (3, 4)


def test_fast_h3_vsa_frontend_workflow_is_explicit_t2va_and_mirrored():
    root = Path(__file__).resolve().parents[1]
    relative = Path("10-speed") / (
        "2026-08-30_H3_FastH3_VSA_T2VA_4Step_0p4MP_Advanced_EXP.json"
    )
    source = root / "examples" / "workflows" / relative
    mirror = (
        root.parents[1]
        / "user"
        / "default"
        / "workflows"
        / "MiniMax H3 T8"
        / relative
    )
    assert source.read_bytes() == mirror.read_bytes()
    workflow = json.loads(source.read_text(encoding="utf-8"))
    nodes = {node["type"]: node for node in workflow["nodes"]}
    loader = nodes["MiniMaxH3LoRACompatibilityLoaderT8Advanced"]
    setup = nodes["MiniMaxH3FastH34StepSetupT8Advanced"]
    conditioning = nodes["MiniMaxH3AudioConditioningT8"]
    assert loader["widgets_values"] == [
        "FastH3-VSA\\vsa-datafree\\adapter_model.safetensors",
        1.0,
    ]
    assert setup["widgets_values"] == ["t2va_only", "external_vsa_if_available"]
    assert conditioning["widgets_values"][1:4] == [832, 480, 124]
    assert any(node["type"] == "MarkdownNote" for node in workflow["nodes"])
