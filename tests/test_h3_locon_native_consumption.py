"""Pointwise LoCon/Tucker factors on a real H3 Linear, not style qualification."""

import json

import pytest
import torch
from safetensors.torch import save_file

from h3_audio_t8_pkg.h3_lora_compat_advanced import load_minimax_h3_lora_model
from test_h21_lora_consumption import tiny_model, actual_projection, KEY


@pytest.mark.parametrize("strength", [0.0, 0.7, -0.6])
@pytest.mark.parametrize("mid", [False, True])
def test_pointwise_LoCon_real_projection_alpha_and_full_restore(
    tmp_path, strength, mid
):
    selected = tiny_model()
    layer = selected.model.diffusion_model.blocks[0].attn.qkv_proj
    weight = layer.weight.detach().clone()
    generator = torch.Generator().manual_seed(12112)
    rank = 2
    alpha = 0.7
    up = torch.randn(weight.shape[0], rank, generator=generator) * 0.03
    down = torch.randn(rank, weight.shape[1], generator=generator) * 0.03
    prefix = KEY.removesuffix(".weight")
    payload = {
        prefix + ".lora_up.weight": up[:, :, None, None],
        prefix + ".lora_down.weight": down[:, :, None, None],
        prefix + ".alpha": torch.tensor(alpha),
    }
    if mid:
        center = torch.randn(rank, rank, generator=generator) * 0.03
        payload[prefix + ".lora_mid.weight"] = center[:, :, None, None]
        delta = up @ center @ down
    else:
        delta = up @ down
    path = tmp_path / "pointwise-locon.safetensors"
    save_file(payload, str(path))
    values = torch.randn(4, weight.shape[1], generator=generator)
    original = actual_projection(selected, values)
    for _ in range(2):
        patched, report = load_minimax_h3_lora_model(selected, path, strength)
        assert json.loads(report)["applied_patch_count"] == 1
        actual = actual_projection(patched, values)
        expected = torch.nn.functional.linear(
            values, weight + strength * alpha / rank * delta, layer.bias
        )
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)
        assert torch.equal(layer.weight, weight)
        assert torch.equal(actual_projection(selected, values), original)
    assert not torch.cuda.is_initialized()
