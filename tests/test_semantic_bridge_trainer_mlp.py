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


@pytest.mark.parametrize("rms_scale", [0.1, 1., 3., 10.])
@pytest.mark.parametrize("mode", ["per_token", "global", "none"])
@pytest.mark.parametrize("chunk", [0, 1, 4])
def test_residual_base_is_normalized_at_all_input_scales(tmp_path, rms_scale, mode, chunk):
    generator = torch.Generator().manual_seed(930)
    hidden = 8
    shapes = {"fc1.weight": (hidden, 5120), "fc1.bias": (hidden,),
              "fc2.weight": (hidden, hidden), "fc2.bias": (hidden,),
              "fc3.weight": (5120, hidden), "fc3.bias": (5120,)}
    state = {key: torch.randn(shape, generator=generator) * .05 for key, shape in shapes.items()}
    path = tmp_path / "residual-export.safetensors"
    save_file({"net." + key: value for key, value in state.items()}, path,
              metadata={"arch": "mlp", "dim": "5120", "hidden": str(hidden),
                        "residual_skip": "True", "residual_scale": ".3"})
    source = torch.randn(1, 5, 5120, generator=generator) * rms_scale
    result, _ = sb.apply_bridge([[source, {}]], sb.BridgeConfig(str(path), sb.file_sha(path),
        alpha=.12, chunk_tokens=chunk, magnitude_match=mode), cancel=lambda: None)
    rows = source.reshape(-1, 5120)
    predictions = []
    for h in rows.split(chunk or len(rows)):
        x = h / (h.square().mean(-1, keepdim=True) + 1e-6).sqrt()
        net = torch.nn.functional.silu(torch.nn.functional.linear(x, state["fc1.weight"], state["fc1.bias"]))
        net = torch.nn.functional.silu(torch.nn.functional.linear(net, state["fc2.weight"], state["fc2.bias"]))
        net = torch.nn.functional.linear(net, state["fc3.weight"], state["fc3.bias"])
        predictions.append(x + .3 * net)
    pred = torch.cat(predictions)
    if mode == "per_token":
        pred = pred * ((rows.square().mean(-1, keepdim=True) + 1e-8).sqrt() /
                       (pred.square().mean(-1, keepdim=True) + 1e-8).sqrt())
    elif mode == "global":
        numerator = sum((h.double().square().sum() for h in rows.split(chunk or len(rows))),
                        torch.zeros((), dtype=torch.float64))
        denominator = sum((p.double().square().sum() for p in predictions),
                          torch.zeros((), dtype=torch.float64))
        pred = pred * ((numerator / rows.numel() + 1e-8) /
                       (denominator / rows.numel() + 1e-8)).sqrt().float()
    expected = (rows + .12 * (pred - rows)).reshape_as(source)
    torch.testing.assert_close(result[0][0], expected, rtol=0, atol=0)
