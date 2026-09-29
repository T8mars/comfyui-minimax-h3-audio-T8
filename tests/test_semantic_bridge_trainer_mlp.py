"""Synthetic loader checks for trainer-exported MLP weights (not video QA)."""
import json

import pytest
import torch
from safetensors.torch import save_file

from h3_audio_t8_pkg import semantic_bridge as sb


@pytest.mark.parametrize("residual_skip", [False, True])
def test_trainer_mlp_variable_width_and_residual(tmp_path, residual_skip):
    hidden = 64
    state = {
        "net.fc1.weight": torch.randn(hidden, 5120) * 0.002,
        "net.fc1.bias": torch.randn(hidden) * 0.002,
        "net.fc2.weight": torch.randn(hidden, hidden) * 0.002,
        "net.fc2.bias": torch.randn(hidden) * 0.002,
        "net.fc3.weight": torch.randn(5120, hidden) * 0.002,
        "net.fc3.bias": torch.randn(5120) * 0.002,
    }
    path = tmp_path / "trainer_mlp.safetensors"
    save_file(state, path, metadata={"arch": "mlp", "dim": "5120", "hidden": str(hidden),
                                     "residual_skip": str(residual_skip), "residual_scale": "0.1",
                                     "extra_json": json.dumps({})})
    source = torch.randn(1, 5, 5120)
    result, report = sb.apply_bridge(
        [[source, {"start_percent": 0.25}]],
        sb.BridgeConfig(str(path), sb.file_sha(path), alpha=0.12, chunk_tokens=0),
        cancel=lambda: None,
    )
    x = source / (source.square().mean(-1, keepdim=True) + 1e-6).sqrt()
    correction = torch.nn.functional.silu(torch.nn.functional.linear(x, state["net.fc1.weight"], state["net.fc1.bias"]))
    correction = torch.nn.functional.silu(torch.nn.functional.linear(correction, state["net.fc2.weight"], state["net.fc2.bias"]))
    correction = torch.nn.functional.linear(correction, state["net.fc3.weight"], state["net.fc3.bias"])
    pred = x + 0.1 * correction if residual_skip else correction
    pred = pred * ((source.square().mean(-1, keepdim=True) + 1e-8).sqrt() /
                   (pred.square().mean(-1, keepdim=True) + 1e-8).sqrt())
    expected = source + 0.12 * (pred - source)
    torch.testing.assert_close(result[0][0], expected, rtol=1e-5, atol=2e-6)
    assert report["applied"]
