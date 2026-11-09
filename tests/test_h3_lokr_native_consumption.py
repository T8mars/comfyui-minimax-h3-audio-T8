"""Native LoKr algebra/alpha and patch lifetime, independent of style training."""
import json

import pytest
import torch
from safetensors.torch import save_file

from h3_audio_t8_pkg.h3_lora_compat_advanced import load_minimax_h3_lora_model
from test_h21_lora_consumption import tiny_model, actual_projection, KEY


@pytest.mark.parametrize("factored", [False, True])
@pytest.mark.parametrize("alpha", [.5, 3.])
def test_lokr_file_alpha_and_real_projection_survive_unpatch_and_reload(tmp_path, factored, alpha):
    selected = tiny_model()
    layer = selected.model.diffusion_model.blocks[0].attn.qkv_proj
    original = layer.weight.detach().clone()
    out_features, in_features = original.shape
    assert out_features % 2 == in_features % 2 == 0
    generator = torch.Generator().manual_seed(81108)
    w1 = torch.randn(2, 2, generator=generator) * .03
    prefix = 'diffusion_model.blocks.0.attn.qkv_proj'
    payload = {prefix+'.lokr_w1': w1, prefix+'.alpha': torch.tensor(alpha)}
    if factored:
        a = torch.randn(out_features//2, 2, generator=generator) * .03
        b = torch.randn(2, in_features//2, generator=generator) * .03
        payload.update({prefix+'.lokr_w2_a': a, prefix+'.lokr_w2_b': b})
        delta = torch.kron(w1, a@b) * (alpha/2)
    else:
        w2 = torch.randn(out_features//2, in_features//2, generator=generator) * .03
        payload[prefix+'.lokr_w2'] = w2
        delta = torch.kron(w1, w2)  # Direct factors have no rank divisor in native LoKr.
    path = tmp_path/'actual-lokr.safetensors'
    save_file(payload, str(path))
    x = torch.randn(3, in_features, generator=generator)
    unpatched = actual_projection(selected, x)
    for strength in (0., .7, -1.):
        patched, raw = load_minimax_h3_lora_model(selected, path, strength)
        report = json.loads(raw)
        assert report['applied_patch_count'] == 1 and report['missed_patch_target_count'] == 0
        assert KEY in patched.patches
        result = actual_projection(patched, x)
        expected = torch.nn.functional.linear(x, original+delta*strength, layer.bias)
        torch.testing.assert_close(result, expected, rtol=1e-6, atol=1e-7)
        assert torch.equal(layer.weight, original)
        assert torch.equal(actual_projection(selected, x), unpatched)
        assert not selected.patches and not selected.backup
    assert not torch.cuda.is_initialized()
