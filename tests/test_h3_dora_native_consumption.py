"""Actual Core DoRA magnitude axes and patch restoration; no style-model claim."""
import json

import pytest
import torch
from safetensors.torch import save_file

from h3_audio_t8_pkg.h3_lora_compat_advanced import load_minimax_h3_lora_model
from test_h21_lora_consumption import actual_projection, tiny_model


@pytest.mark.parametrize("axis", ["output", "input"])
@pytest.mark.parametrize("strength", [.4, -1.])
def test_native_dora_magnitude_and_actual_projection_do_not_leak_to_shared_base(tmp_path, axis, strength):
    base = tiny_model()
    layer = base.model.diffusion_model.blocks[0].attn.qkv_proj
    original = layer.weight.detach().clone()
    generator = torch.Generator().manual_seed(81109)
    rank, alpha = 2, .7
    up = torch.randn(original.shape[0], rank, generator=generator) * .03
    down = torch.randn(rank, original.shape[1], generator=generator) * .03
    magnitude = torch.linspace(.4, 1.1, original.shape[0] if axis == "output" else original.shape[1])
    magnitude = magnitude[:, None] if axis == "output" else magnitude[None, :]
    prefix = "diffusion_model.blocks.0.attn.qkv_proj"
    path = tmp_path / "user-selected-dora.safetensors"
    save_file({prefix + ".lora_up.weight": up, prefix + ".lora_down.weight": down,
               prefix + ".alpha": torch.tensor(alpha), prefix + ".dora_scale": magnitude}, str(path))
    patched, raw = load_minimax_h3_lora_model(base, path, strength)
    report = json.loads(raw)
    assert report["applied_patch_count"] == 1 and report["missed_patch_target_count"] == 0
    updated = original + (up @ down) * (alpha / rank)
    # Core's output-axis contract uses the original row norms. Its input-axis
    # contract normalizes the updated columns. Exercise both actual consumers.
    norm = (original.norm(dim=1, keepdim=True) if axis == "output"
            else updated.norm(dim=0, keepdim=True)) + torch.finfo(original.dtype).eps
    decomposed = updated * (magnitude / norm)
    expected_weight = original + strength * (decomposed - original)
    x = torch.randn(3, original.shape[1], generator=generator)
    expected = torch.nn.functional.linear(x, expected_weight, layer.bias)
    for _ in range(2):
        torch.testing.assert_close(actual_projection(patched, x), expected, rtol=1e-6, atol=1e-6)
        assert torch.equal(layer.weight, original)
        assert not base.patches and not base.backup
    assert not torch.cuda.is_initialized()
