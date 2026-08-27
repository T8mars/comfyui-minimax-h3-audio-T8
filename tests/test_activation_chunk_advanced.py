from __future__ import annotations

import copy
import json

import pytest
import torch
from torch import nn

import h3_audio_t8_pkg.activation_chunk_advanced as activation_chunk
from h3_audio_t8_pkg.activation_chunk_advanced import (
    ACTIVATION_CHUNK_SCHEMA,
    ATTACHMENT_KEY,
    H3MLPActivationChunkPatch,
    configure_activation_chunk,
)
from h3_audio_t8_pkg.nodes_activation_chunk_advanced import (
    MiniMaxH3ActivationChunkT8Advanced,
)


class _AdaLN(nn.Module):
    def __init__(self, rows: int, hidden: int):
        super().__init__()
        generator = torch.Generator().manual_seed(7)
        self.values = nn.ParameterList(
            [
                nn.Parameter(torch.randn(rows, hidden, generator=generator) * 0.05)
                for _ in range(6)
            ]
        )

    def forward(self, _t_emb):
        return tuple(self.values)


class _Attention(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.linear = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x, rope_freqs=None, transformer_options=None):
        assert rope_freqs is not None
        assert isinstance(transformer_options, dict)
        return self.linear(x)


class _MLP(nn.Module):
    def __init__(self, hidden: int, ffn: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden, ffn * 2, bias=False)
        self.fc2 = nn.Linear(ffn, hidden, bias=False)

    def forward(self, x):
        gate, value = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(torch.nn.functional.silu(gate) * value)


class _Block(nn.Module):
    def __init__(self, hidden=12, ffn=20, modulation_rows=6):
        super().__init__()
        self.adaln_proj = _AdaLN(modulation_rows, hidden)
        self.norm1 = nn.RMSNorm(hidden)
        self.attn = _Attention(hidden)
        self.norm2 = nn.RMSNorm(hidden)
        self.mlp = _MLP(hidden, ffn)

    def forward(self, x, t_emb, mod_segments, rope_freqs, transformer_options=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(t_emb)
        h = x.new_empty(x.shape)
        for start, stop, row in mod_segments:
            h[start:stop] = self.norm1(x[start:stop])
            h[start:stop].mul_(1 + scale_msa[row]).add_(shift_msa[row])
        attention = self.attn(
            h,
            rope_freqs=rope_freqs,
            transformer_options=transformer_options,
        )
        for start, stop, row in mod_segments:
            x[start:stop].addcmul_(attention[start:stop], gate_msa[row])
        for start, stop, row in mod_segments:
            h = self.norm2(x[start:stop])
            h.mul_(1 + scale_mlp[row]).add_(shift_mlp[row])
            x[start:stop].addcmul_(self.mlp(h), gate_mlp[row])
        return x


def _callbacks(block):
    def original_block(args):
        return {
            "img": block(
                args["img"],
                args["t_emb"],
                args["mod_segments"],
                args["rope_freqs"],
                transformer_options=args["transformer_options"],
            )
        }

    return original_block


def test_chunk_patch_matches_the_native_token_local_formula_within_cpu_tolerance():
    torch.manual_seed(9)
    block = _Block()
    original = _callbacks(block)
    args = {
        "img": torch.randn(23, 12),
        "t_emb": torch.zeros(2, 4),
        "mod_segments": [(0, 7, 0), (7, 18, 1), (18, 23, 2)],
        "rope_freqs": torch.zeros(1),
        "transformer_options": {},
    }
    expected_args = copy.deepcopy(args)
    expected = original(expected_args)["img"]
    patch = H3MLPActivationChunkPatch(0, chunk_rows=4, preserve_short_path=False)
    actual = patch(copy.deepcopy(args), {"original_block": original})["img"]
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)


def test_chunk_patch_preserves_exact_short_path_when_requested():
    block = _Block()
    original = _callbacks(block)
    args = {
        "img": torch.randn(8, 12),
        "t_emb": torch.zeros(2, 4),
        "mod_segments": [(0, 8, 0)],
        "rope_freqs": torch.zeros(1),
        "transformer_options": {},
    }
    expected = original(copy.deepcopy(args))["img"]
    actual = H3MLPActivationChunkPatch(0, 16, True)(
        copy.deepcopy(args),
        {"original_block": original},
    )["img"]
    assert torch.equal(actual, expected)


def test_chunk_patch_rejects_noncontiguous_segments_and_unknown_callback_contract():
    block = _Block()
    args = {
        "img": torch.randn(8, 12),
        "t_emb": torch.zeros(2, 4),
        "mod_segments": [(0, 4, 0), (5, 8, 1)],
        "rope_freqs": torch.zeros(1),
        "transformer_options": {},
    }
    patch = H3MLPActivationChunkPatch(0, 2, False)
    with pytest.raises(ValueError, match="contiguous"):
        patch(copy.deepcopy(args), {"original_block": _callbacks(block)})
    with pytest.raises(RuntimeError, match="closure"):
        patch(copy.deepcopy(args), {"original_block": lambda value: value})


class _FakeDiffusion:
    def __init__(self, count=4):
        self.blocks = nn.ModuleList([_Block() for _ in range(count)])


class _FakeModel:
    def __init__(self, count=4):
        self.diffusion = _FakeDiffusion(count)
        self.model_options = {"transformer_options": {}}
        self.attachments = {}

    def clone(self):
        cloned = _FakeModel(0)
        cloned.diffusion = self.diffusion
        cloned.model_options = copy.deepcopy(self.model_options)
        cloned.attachments = copy.deepcopy(self.attachments)
        return cloned

    def get_model_object(self, name):
        assert name == "diffusion_model"
        return self.diffusion

    def set_model_patch_replace(self, patch, name, block_name, number, transformer_index=None):
        assert name == "dit"
        assert transformer_index is None
        patches = self.model_options["transformer_options"].setdefault("patches_replace", {})
        patches.setdefault("dit", {})[(block_name, number)] = patch

    def set_attachments(self, key, value):
        self.attachments[key] = value


@pytest.fixture
def supported_core(monkeypatch):
    monkeypatch.setattr(
        activation_chunk,
        "core_contract",
        lambda: {"supported": True, "hashes": {"probe": "ok"}},
    )


def test_configure_report_only_returns_identical_model(supported_core):
    model = _FakeModel()
    returned, report = configure_activation_chunk(
        model, "report_only", 256, 0, 3, True, 736, 416, 124, 1
    )
    assert returned is model
    assert report["schema"] == ACTIVATION_CHUNK_SCHEMA
    assert report["status"] == "report_only"
    assert report["applied"] is False
    assert report["memory_safe_claim"] is False
    assert report["mlp_backend"]["kind"] == "eager_or_runtime_dependent"
    assert report["activation_proxy"]["proxy_applies_to_detected_backend"] is True


def test_report_marks_tensorwise_int8_fused_proxy_as_inapplicable(
    supported_core, monkeypatch
):
    monkeypatch.setattr(
        activation_chunk,
        "_mlp_backend_profile",
        lambda _block: {
            "kind": "tensorwise_int8_fused_swiglu",
            "fc2_weight_layout": "TensorWiseINT8Layout",
            "fc2_weight_function_count": 0,
            "native_full_fc1_intermediate_expected": False,
            "reason": "fused test",
        },
    )
    _returned, report = configure_activation_chunk(
        _FakeModel(), "report_only", 256, 0, 3, True, 736, 416, 124, 0
    )
    assert report["expected_memory_benefit"] == (
        "low_or_none_on_detected_int8_fused_path"
    )
    assert report["activation_proxy"]["proxy_applies_to_detected_backend"] is False
    assert report["activation_proxy"]["expected_proxy_reduction_mib"] == 0.0
    assert report["activation_proxy"]["theoretical_proxy_reduction_mib"] > 0.0


def test_configure_apply_clones_and_adds_only_selected_block_replacements(supported_core):
    model = _FakeModel()
    returned, report = configure_activation_chunk(
        model, "apply_exp", 128, 1, 3, True, 736, 416, 124, 0
    )
    assert returned is not model
    assert "patches_replace" not in model.model_options["transformer_options"]
    entries = returned.model_options["transformer_options"]["patches_replace"]["dit"]
    assert sorted(entries) == [("double_block", 1), ("double_block", 2), ("double_block", 3)]
    assert report["applied"] is True
    assert returned.attachments[ATTACHMENT_KEY]["chunk_rows"] == 128


def test_configure_rejects_existing_double_block_owner_without_overwriting(supported_core):
    model = _FakeModel()
    existing = object()
    model.model_options["transformer_options"]["patches_replace"] = {
        "dit": {("double_block", 2): existing}
    }
    with pytest.raises(RuntimeError, match="conflicts"):
        configure_activation_chunk(
            model, "apply_exp", 128, 0, 3, True, 736, 416, 124, 0
        )
    assert (
        model.model_options["transformer_options"]["patches_replace"]["dit"][("double_block", 2)]
        is existing
    )


def test_configure_unknown_core_is_reported_or_blocked(monkeypatch):
    monkeypatch.setattr(
        activation_chunk,
        "core_contract",
        lambda: {"supported": False, "hashes": {"probe": "unknown"}},
    )
    model = _FakeModel()
    returned, report = configure_activation_chunk(
        model, "report_only", 128, 0, 3, True, 736, 416, 124, 0
    )
    assert returned is model
    assert report["status"] == "unsupported_core"
    with pytest.raises(RuntimeError, match="unknown ComfyUI H3 source contract"):
        configure_activation_chunk(
            model, "apply_exp", 128, 0, 3, True, 736, 416, 124, 0
        )


def test_current_comfy_h3_activation_contract_is_semantically_supported():
    contract = activation_chunk.core_contract()
    assert contract["supported"] is True
    assert contract["source_hash_policy"] == "diagnostic_only_not_a_compatibility_gate"
    assert contract["semantic_contract"]["status"] == "semantic_contract_validated"


def test_activation_chunk_node_schema_is_safe_by_default():
    schema = MiniMaxH3ActivationChunkT8Advanced.define_schema()
    inputs = {item.id: item for item in schema.inputs}
    assert schema.node_id.endswith("Advanced")
    assert schema.is_experimental is True
    assert inputs["mode"].default == "report_only"
    assert inputs["chunk_rows"].default == 256


def test_activation_chunk_node_report_is_json(monkeypatch):
    model = object()
    monkeypatch.setattr(
        "h3_audio_t8_pkg.nodes_activation_chunk_advanced.configure_activation_chunk",
        lambda *_args: (model, {"schema": ACTIVATION_CHUNK_SCHEMA, "applied": False}),
    )
    output = MiniMaxH3ActivationChunkT8Advanced.execute(
        model, "report_only", 256, 0, 49, True, 736, 416, 124, 0
    )
    assert output[0] is model
    assert json.loads(output[1])["schema"] == ACTIVATION_CHUNK_SCHEMA
