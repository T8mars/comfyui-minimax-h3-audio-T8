from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools" / "convert_openvdn_h3_turbo_for_pruned_curve.py"
SPEC = importlib.util.spec_from_file_location("openvdn_pruned_curve_tool", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
tool = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = tool
SPEC.loader.exec_module(tool)


def test_named_openvdn_turbo_keys_are_normalized_without_touching_values():
    value = torch.ones(2, 3)
    state = {
        "transformer_blocks.0.attn.orig.to_q.lora_A.turbo.weight": value,
        "transformer_blocks.0.attn.orig.to_q.lora_B.turbo.weight": value.T,
    }
    normalized = tool.normalize_openvdn_turbo(state)
    assert set(normalized) == {
        "transformer_blocks.0.attn.to_q.lora_A.weight",
        "transformer_blocks.0.attn.to_q.lora_B.weight",
    }
    assert normalized["transformer_blocks.0.attn.to_q.lora_A.weight"] is value


def test_comfy_source_adds_exactly_one_prefix_and_preserves_conversion_report():
    state = {"source": torch.ones(1)}
    fake = SimpleNamespace(
        convert_fastvideo_h3_adapter=lambda normalized: (
            {"blocks.0.mlp.fc1.lora_A.weight": normalized["source"]},
            {"conversion": "fake"},
        )
    )
    converted, report = tool.comfy_source_state(state, fake)
    assert set(converted) == {
        "diffusion_model.blocks.0.mlp.fc1.lora_A.weight"
    }
    assert report == {"conversion": "fake"}


def test_openvdn_source_validation_accepts_fused_rank_and_rejects_bad_target():
    expected_modules = {
        "blocks.0.attn.qkv_proj": ((64, 4), (12, 64)),
        "blocks.0.adaln_proj.linear": ((2, 4), (8, 2)),
    }
    curve = SimpleNamespace(
        EXPECTED_MODULES=expected_modules,
        ADALN_MODULES=("blocks.0.adaln_proj.linear",),
        SOURCE_TENSOR_COUNT=4,
        module_key=lambda module, suffix: f"diffusion_model.{module}.{suffix}",
    )
    source = {
        "diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight": torch.ones(
            6, 4, dtype=torch.bfloat16
        ),
        "diffusion_model.blocks.0.attn.qkv_proj.lora_B.weight": torch.ones(
            12, 6, dtype=torch.bfloat16
        ),
        "diffusion_model.blocks.0.adaln_proj.linear.lora_A.weight": torch.ones(
            2, 4, dtype=torch.bfloat16
        ),
        "diffusion_model.blocks.0.adaln_proj.linear.lora_B.weight": torch.ones(
            8, 2, dtype=torch.bfloat16
        ),
    }
    tool.validate_openvdn_source(source, curve)
    source["diffusion_model.blocks.0.attn.qkv_proj.lora_B.weight"] = torch.ones(
        11, 6, dtype=torch.bfloat16
    )
    with pytest.raises(ValueError, match="target dimensions differ"):
        tool.validate_openvdn_source(source, curve)
