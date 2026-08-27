from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
import torch

from h3_audio_t8_pkg import attention_hooks_advanced as hooks
from h3_audio_t8_pkg.nodes import MiniMaxH3AudioT8Extension
from h3_audio_t8_pkg.nodes_attention_hooks_advanced import (
    MiniMaxH3AttentionHooksT8Advanced,
)


class _QKV(torch.nn.Module):
    def forward(self, x):
        return torch.cat((x, x + 1, x + 2), dim=-1)


def _attention():
    attention = hooks.minimax_model.Attention(4, 2, 2, 1e-6, operations=torch.nn)
    attention.qkv_proj = _QKV()
    attention.q_norm = torch.nn.Identity()
    attention.k_norm = torch.nn.Identity()
    attention.out_proj = torch.nn.Identity()
    return attention


def test_compatibility_forward_matches_unpatched_path_without_hooks(monkeypatch):
    attention = _attention()

    def fake_attention(q, k, v, *_args, **_kwargs):
        del q, k
        value = v.peek()
        return value.transpose(1, 2).reshape(1, value.shape[2], -1)

    monkeypatch.setattr(hooks.minimax_model, "optimized_attention", fake_attention)
    x = torch.zeros(2, 4)
    expected = attention(x, transformer_options={})
    actual = hooks._hooked_attention_forward(
        attention,
        x,
        None,
        {},
        block_index=3,
        block_type="double",
        total_blocks=50,
    )
    assert torch.equal(actual, expected)


def test_tuple_mapping_and_output_hooks_receive_standard_metadata(monkeypatch):
    attention = _attention()
    seen = {}

    def tuple_patch(q, k, v, extra_options):
        assert extra_options["block_index"] == 3
        assert extra_options["block_type"] == "double"
        assert extra_options["total_blocks"] == 50
        return q + 1, k + 1, v + 1

    def mapping_patch(q, k, v, pe=None, attn_mask=None, extra_options=None):
        assert pe is None
        assert attn_mask is None
        assert extra_options["n_heads"] == 2
        return {"q": q * 2, "v": v * 3}

    def output_patch(output, extra_options):
        assert extra_options["dim_head"] == 2
        return output + 4

    def fake_attention(q, k, v, *_args, **_kwargs):
        q_value, k_value, v_value = q.peek(), k.peek(), v.peek()
        seen.update(q=q_value, k=k_value, v=v_value)
        return v_value.transpose(1, 2).reshape(1, v_value.shape[2], -1)

    monkeypatch.setattr(hooks.minimax_model, "optimized_attention", fake_attention)
    output = hooks._hooked_attention_forward(
        attention,
        torch.zeros(2, 4),
        None,
        {
            "patches": {
                "attn1_patch": [tuple_patch, mapping_patch],
                "attn1_output_patch": [output_patch],
            }
        },
        block_index=3,
        block_type="double",
        total_blocks=50,
    )

    assert torch.equal(seen["q"], torch.full((1, 2, 2, 2), 2.0))
    assert torch.equal(seen["k"], torch.full((1, 2, 2, 2), 2.0))
    assert torch.equal(seen["v"], torch.full((1, 2, 2, 2), 9.0))
    assert torch.equal(output, torch.full((2, 4), 13.0))


class MiniMaxH3Model:
    def __init__(self):
        self.blocks = [SimpleNamespace(attn=_attention()) for _ in range(3)]
        self.token_refiner = SimpleNamespace(
            blocks=[SimpleNamespace(attn=_attention()) for _ in range(2)]
        )


class _Patcher:
    def __init__(self, diffusion=None):
        self.diffusion = diffusion or MiniMaxH3Model()
        self.object_patches = {}
        self.model_options = {"transformer_options": {}}
        self.attachments = {}

    def clone(self):
        cloned = _Patcher(self.diffusion)
        cloned.object_patches = dict(self.object_patches)
        cloned.model_options = {
            "transformer_options": dict(self.model_options["transformer_options"])
        }
        return cloned

    def get_model_object(self, name):
        assert name == "diffusion_model"
        return self.diffusion

    def add_object_patch(self, name, value):
        self.object_patches[name] = value

    def set_attachments(self, name, value):
        self.attachments[name] = value


def test_builder_patches_main_and_refiner_attention_on_legacy_core(monkeypatch):
    monkeypatch.setattr(
        hooks,
        "probe_native_attention_hooks",
        lambda diffusion: {
            "available": False,
            "attention_hooks": False,
            "block_metadata": False,
            "main_block_count": len(diffusion.blocks),
            "policy": "test",
        },
    )
    original = _Patcher()

    patched, report_json = hooks.build_attention_hook_compatibility(original)
    report = json.loads(report_json)

    assert original.object_patches == {}
    assert len(patched.object_patches) == 5
    assert report["main_blocks"] == 3
    assert report["token_refiner_blocks"] == 2
    assert report["object_patch_scope"] == "model_patcher_clone_only"


def test_builder_uses_native_core_without_object_patches(monkeypatch):
    monkeypatch.setattr(
        hooks,
        "probe_native_attention_hooks",
        lambda _diffusion: {
            "available": True,
            "attention_hooks": True,
            "block_metadata": True,
            "main_block_count": 3,
            "policy": "test",
        },
    )
    patched, report_json = hooks.build_attention_hook_compatibility(_Patcher())

    assert patched.object_patches == {}
    assert json.loads(report_json)["status"] == "native"


def test_builder_refuses_to_overwrite_existing_attention_object_patch(monkeypatch):
    monkeypatch.setattr(
        hooks,
        "probe_native_attention_hooks",
        lambda _diffusion: {
            "available": False,
            "attention_hooks": False,
            "block_metadata": False,
            "main_block_count": 3,
            "policy": "test",
        },
    )
    model = _Patcher()
    model.object_patches["diffusion_model.blocks.0.attn.forward"] = object()

    with pytest.raises(RuntimeError, match="will not replace"):
        hooks.build_attention_hook_compatibility(model)


def test_schema_and_append_only_registration():
    schema = MiniMaxH3AttentionHooksT8Advanced.define_schema()
    assert schema.node_id == "MiniMaxH3AttentionHooksT8Advanced"
    node_list = asyncio.run(MiniMaxH3AudioT8Extension().get_node_list())
    assert MiniMaxH3AttentionHooksT8Advanced in node_list
    assert node_list.index(MiniMaxH3AttentionHooksT8Advanced) > node_list.index(
        next(
            node
            for node in node_list
            if node.define_schema().node_id == "MiniMaxH3AVLatentBuilderT8Advanced"
        )
    )
