from __future__ import annotations

import json

import pytest
import torch

from h3_audio_t8_pkg.external_blockswap_advanced import (
    H3ExternalBlockSwapConfig,
    build_external_blockswap_config,
)
from h3_audio_t8_pkg.nodes_external_blockswap_advanced import (
    MiniMaxH3ExternalBlockSwapBridgeT8Advanced,
)


def _build(**overrides):
    values = {
        "profile": "upstream_default_auto",
        "block_to_swap": 1,
        "hot_blocks": 9,
        "prefetch": False,
        "prefetch_count": 8,
        "pin_memory": False,
        "disk_workers": 9,
        "auto_vram": False,
        "vram_reserve_mb": 999.0,
        "offload_dit": True,
        "dtype": "float32",
        "require_external_runtime": False,
    }
    values.update(overrides)
    return build_external_blockswap_config(**values)


def test_upstream_default_profile_is_exact_and_ignores_manual_widgets():
    config, report = _build()
    assert isinstance(config, H3ExternalBlockSwapConfig)
    assert config.block_to_swap == 47
    assert config.window_size(50) == 3
    assert config.prefetch_count == 2
    assert config.hot_blocks == 0
    assert config.auto_vram is True
    assert config.offload_dit is False
    assert config.torch_dtype is torch.bfloat16
    payload = json.loads(report)
    assert payload["derived"]["requested_gpu_slots_before_auto_vram"] == 5
    assert payload["claims"]["universal_oom_prevention"] is False
    assert payload["compatibility"]["official_comfy_model_supported"] is False


def test_manual_profile_preserves_explicit_values_and_duck_contract():
    config, _report = _build(
        profile="manual",
        block_to_swap=40,
        hot_blocks=3,
        prefetch=True,
        prefetch_count=1,
        auto_vram=False,
        vram_reserve_mb=2048,
        dtype="float16",
    )
    assert config.window_size(50) == 10
    assert config.hot_blocks == 3
    assert config.vram_reserve_mb == 2048
    assert config.torch_dtype is torch.float16


def test_external_runtime_guard_fails_before_false_compatibility_claim(monkeypatch):
    monkeypatch.setattr(
        "h3_audio_t8_pkg.external_blockswap_advanced.external_blockswap_available",
        lambda: False,
    )
    with pytest.raises(RuntimeError, match="does not alter ComfyUI's official MODEL"):
        _build(require_external_runtime=True)


def test_bridge_schema_is_isolated_and_safe_by_default():
    schema = MiniMaxH3ExternalBlockSwapBridgeT8Advanced.define_schema()
    inputs = {item.id: item for item in schema.inputs}
    assert schema.is_experimental is True
    assert schema.category == "T8/MiniMax H3/Models/Experimental"
    assert inputs["profile"].default == "upstream_default_auto"
    assert inputs["auto_vram"].default is True
    assert inputs["require_external_runtime"].default is True
    assert schema.outputs[0].io_type == "MINIMAX_H3_SWAP"
