"""Synthetic software checks for the Transformer weight format (not video QA)."""
import json

import pytest
import torch
from safetensors.torch import save_file

from h3_audio_t8_pkg import semantic_bridge as sb
from h3_audio_t8_pkg.semantic_bridge_trans import TransBridge, TransSpec


def test_transformer_weight_and_fixed_application_contract(tmp_path):
    spec = TransSpec(64, 1, 4, 64, True, 1.0, None)
    model = TransBridge(spec).eval()
    path = tmp_path / "synthetic_trans.safetensors"
    contract = {
        "schema": 1, "alpha_min": 1.0, "alpha_max": 1.0,
        "magnitude_match": "per_token", "token_span": "all", "tail_ratio": 1.0,
        "chunk_tokens": 0, "auto_alpha": False, "guard": False,
        "allow_dim_mismatch": False,
    }
    save_file({"net." + key: value.contiguous() for key, value in model.state_dict().items()},
              path, metadata={"arch": "trans", "dim": "5120", "hidden": "64",
                              "layers": "1", "heads": "4", "max_tokens": "64",
                              "residual_skip": "True", "residual_scale": "1.0",
                              "extra_json": json.dumps({"application_contract": contract})})
    native = torch.randn(1, 5, 5120, generator=torch.Generator().manual_seed(20260929))
    source = [[native, {"start_percent": 0.25}]]
    cfg = sb.BridgeConfig(str(path), sb.file_sha(path), alpha=1.0, chunk_tokens=0)
    result, report = sb.apply_bridge(source, cfg, cancel=lambda: None)
    normalized = native / (native.square().mean(-1, keepdim=True) + 1e-6).sqrt()
    predicted = model(normalized)
    matched = predicted * ((native.square().mean(-1, keepdim=True) + 1e-8).sqrt() /
                           (predicted.square().mean(-1, keepdim=True) + 1e-8).sqrt())
    expected = native + (matched - native)
    torch.testing.assert_close(result[0][0], expected, rtol=1e-5, atol=1e-6)
    assert report["applied"] and source[0][1] == {"start_percent": 0.25}
    assert result[0][1]["start_percent"] == 0.25
    with pytest.raises(ValueError, match="application contract mismatch"):
        sb.apply_bridge(source, sb.BridgeConfig(str(path), sb.file_sha(path),
                                                alpha=0.1, chunk_tokens=0), cancel=lambda: None)


def test_transformer_rejects_invalid_tensor(tmp_path):
    path = tmp_path / "bad.safetensors"
    save_file({"net.in_proj.weight": torch.zeros(64, 5120)}, path,
              metadata={"arch": "trans", "dim": "5120", "hidden": "64",
                        "layers": "1", "heads": "4"})
    with pytest.raises(ValueError, match="tensor keys"):
        sb.read_weights(path)
