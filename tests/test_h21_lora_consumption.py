"""H21 actual LoRA consumption contracts, with tiny CPU tensors only.

The real Core parser, LoRA weight calculation, safetensors reader and
ModelPatcher are used. These tests do not qualify a quantized CUDA kernel,
an AWQ forward, a whole-model image/audio result, or arbitrary PTD runtimes.
"""
from __future__ import annotations

import json

import comfy.lora
import pytest
import torch
import torch.nn.functional as functional
from safetensors.torch import save_file

from h3_audio_t8_pkg import h3_lora_compat_advanced as compat
from test_progressive_sampling_runtime import tiny_model


FUSED = "blocks.0.attn.qkv_proj"
KEY = "diffusion_model." + FUSED + ".weight"


def test_direct_aliases_follow_actual_h3_state_not_selected_python_class_name():
    actual = tiny_model().model

    class UserSelectedH3Facade:
        def state_dict(self):
            return actual.state_dict()

    facade = UserSelectedH3Facade()
    aliases = compat.add_minimax_h3_direct_lora_keys(facade)
    assert not isinstance(facade, compat.comfy.model_base.MiniMaxH3)
    assert aliases[FUSED] == KEY
    assert aliases["blocks.0.mlp.fc1"] == "diffusion_model.blocks.0.mlp.fc1.weight"
    assert not torch.cuda.is_initialized()


def test_structural_direct_alias_does_not_overwrite_existing_core_target():
    actual = tiny_model().model
    original = {FUSED: "core_explicit_different_target.weight", "already.native": "kept.weight"}
    before = dict(original)
    aliases = compat.add_minimax_h3_direct_lora_keys(actual, original)
    assert aliases[FUSED] == original[FUSED]
    assert aliases["already.native"] == original["already.native"]
    assert aliases["blocks.0.mlp.fc1"] == "diffusion_model.blocks.0.mlp.fc1.weight"
    assert original == before


def test_optional_alias_extension_keeps_core_mapping_without_a_state_dict_provider():
    original = {"native.mapped": "actual.weight"}
    assert compat.add_minimax_h3_direct_lora_keys(object(), original) == original
    assert original == {"native.mapped": "actual.weight"}


def split_qkv(seed=0, *, alpha=True):
    """Different ranks/alphas make lost per-projection scaling observable."""
    generator = torch.Generator(device="cpu").manual_seed(2100 + seed)
    source, updates = {}, []
    for projection, rank, scale in zip(("q", "k", "v"), (1, 2, 3), (0., -1.5, 4.5), strict=True):
        prefix = f"transformer_blocks.0.attn.to_{projection}"
        down = torch.randn(rank, 8, generator=generator) * .1
        up = torch.randn(16, rank, generator=generator) * .1
        source[prefix + ".lora_A.weight"] = down
        source[prefix + ".lora_B.weight"] = up
        if alpha:
            source[prefix + ".alpha"] = torch.tensor(scale)
        updates.append((up @ down) * (scale / rank if alpha else 1.))
    return source, torch.cat(updates)


def native_delta(converted, alias, shape):
    key = "diffusion_model." + alias + ".weight"
    patches = comfy.lora.load_lora(converted, {alias: key}, log_missing=False)
    assert key in patches, "conversion must reach the actual Core parser"
    return comfy.lora.calculate_weight(
        [(1., patches[key], 1., None, None)], torch.zeros(shape), key,
    )


@pytest.mark.parametrize("alpha", [False, True])
def test_real_core_fused_qkv_preserves_independent_rank_alpha_and_forward(alpha):
    source, expected = split_qkv(alpha=alpha)
    originals = {key: value.clone() for key, value in source.items()}
    converted, _ = compat.convert_fastvideo_h3_adapter(source)
    actual = native_delta(converted, FUSED, (48, 8))
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    x = torch.arange(40, dtype=torch.float32).reshape(5, 8) / 23
    torch.testing.assert_close(
        functional.linear(x, actual), functional.linear(x, expected), rtol=1e-6, atol=1e-7,
    )
    assert source.keys() == originals.keys()
    assert all(torch.equal(value, originals[key]) for key, value in source.items())
    assert not torch.cuda.is_initialized()


def test_real_core_swiglu_conversion_preserves_alpha_and_half_swap():
    source, _ = split_qkv(alpha=False)
    prefix = "transformer_blocks.0.ff.net.0.proj"
    down = torch.arange(16, dtype=torch.float32).reshape(2, 8) / 17
    up = torch.arange(64, dtype=torch.float32).reshape(32, 2) / 67
    source.update({prefix + ".lora_A.weight": down, prefix + ".lora_B.weight": up,
                   prefix + ".alpha": torch.tensor(1.5)})
    converted, _ = compat.convert_fastvideo_h3_adapter(source)
    expected = torch.cat((up[16:], up[:16])) @ down * (1.5 / 2)
    actual = native_delta(converted, "blocks.0.mlp.fc1", expected.shape)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    assert not torch.cuda.is_initialized()


def test_real_core_fused_qkv_preserves_equal_alpha_rank_after_rank_concatenation():
    source, unscaled = split_qkv(alpha=False)
    for projection, rank in zip(("q", "k", "v"), (1, 2, 3), strict=True):
        source[f"transformer_blocks.0.attn.to_{projection}.alpha"] = torch.tensor(.5 * rank)
    converted, _ = compat.convert_fastvideo_h3_adapter(source)
    actual = native_delta(converted, FUSED, (48, 8))
    torch.testing.assert_close(actual, unscaled * .5, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("extras", ["unknown_tensor", "unknown_complete_pair", "unconverted_dora_scale"])
def test_mixed_fastvideo_unknown_payload_reaches_real_loader_unused_report(tmp_path, extras):
    source, expected = split_qkv(alpha=False)
    if extras == "unknown_tensor":
        unknown = {"future_adapter.custom_tensor": torch.tensor([2., 3.])}
    elif extras == "unknown_complete_pair":
        prefix = "transformer_blocks.0.future_projection"
        unknown = {prefix + ".lora_A.weight": torch.ones(2, 8),
                   prefix + ".lora_B.weight": torch.ones(3, 2),
                   prefix + ".alpha": torch.tensor(2.)}
    else:
        # Retaining this source key is not a claim that per-Q/K/V DoRA was
        # converted into one equivalent fused-QKV DoRA normalization.
        unknown = {"transformer_blocks.0.attn.to_q.dora_scale": torch.ones(16)}
    source.update(unknown)
    converted, _ = compat.convert_fastvideo_h3_adapter(source)
    assert all(converted[key] is value for key, value in unknown.items())
    torch.testing.assert_close(native_delta(converted, FUSED, (48, 8)), expected)
    path = tmp_path / "user_selected_arbitrary_name.safetensors"
    save_file(source, str(path))
    original = tiny_model()
    patched, text = compat.load_minimax_h3_lora_model(original, path, .75)
    report = json.loads(text)
    assert report["status"] == "applied"
    assert report["selected_strength"] == .75
    assert report["applied_patch_count"] == 1
    assert set(report["mapping_diagnostics"]["converted_unmapped_keys"]) == set(unknown)
    assert report["mapping_diagnostics"]["registration_is_execution"] is False
    assert report["full_algorithm_verified"] is False
    assert report["warnings"] and not original.patches
    assert KEY in patched.patches
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("damage", ["missing_q_b", "bad_q_pair", "different_k_width", "missing_fc1_b"])
def test_known_broken_pairs_and_qkv_widths_still_raise_before_native_parser(damage):
    source, _ = split_qkv(alpha=False)
    q = "transformer_blocks.0.attn.to_q"
    if damage == "missing_q_b":
        source.pop(q + ".lora_B.weight")
    elif damage == "bad_q_pair":
        source[q + ".lora_B.weight"] = torch.ones(16, 2)
    elif damage == "different_k_width":
        source["transformer_blocks.0.attn.to_k.lora_A.weight"] = torch.ones(2, 7)
    else:
        source["transformer_blocks.0.ff.net.0.proj.lora_A.weight"] = torch.ones(2, 8)
    with pytest.raises(ValueError, match="incomplete|invalid|width|pair"):
        compat.convert_fastvideo_h3_adapter(source)


def test_conversion_rejects_ambiguous_existing_fused_target_without_silent_overwrite():
    source, _ = split_qkv(alpha=False)
    source[FUSED + ".lora_A.weight"] = torch.full((2, 8), 10.)
    source[FUSED + ".lora_B.weight"] = torch.full((48, 2), 20.)
    with pytest.raises(ValueError, match="(?i)collision|ambiguous|conflict|duplicate"):
        compat.convert_fastvideo_h3_adapter(source)


def actual_projection(patcher, x):
    """Exercise genuine Core patch/unpatch and the layer's real forward."""
    try:
        if KEY in patcher.patches:
            patcher.patch_weight_to_device(KEY, device_to=torch.device("cpu"))
        with torch.inference_mode():
            return patcher.model.diffusion_model.blocks[0].attn.qkv_proj(x).clone()
    finally:
        patcher.unpatch_model()


def test_shared_base_independent_stage_loras_then_all_off_matches_unpatched_forward(tmp_path):
    base = tiny_model()
    layer = base.model.diffusion_model.blocks[0].attn.qkv_proj
    original = layer.weight.detach().clone()
    x = torch.arange(24, dtype=torch.float32).reshape(3, 8) / 29
    control = actual_projection(base, x)
    files, deltas = [], []
    for index in range(3):
        source, delta = split_qkv(index, alpha=False)
        path = tmp_path / f"selected_{index}.safetensors"
        save_file(source, str(path))
        files.append(path)
        deltas.append(delta)
    low, low_a = compat.load_minimax_h3_lora_model(base, files[0], .7)
    low, low_b = compat.load_minimax_h3_lora_model(low, files[1], -.3)
    high, high_a = compat.load_minimax_h3_lora_model(base, files[2], .2)
    assert [patch[0] for patch in low.patches[KEY]] == [.7, -.3]
    assert [patch[0] for patch in high.patches[KEY]] == [.2]
    assert [json.loads(text)["selected_strength"] for text in (low_a, low_b, high_a)] == [.7, -.3, .2]
    assert not base.patches
    observed_low = actual_projection(low, x)
    torch.testing.assert_close(
        observed_low, functional.linear(x, original + .7 * deltas[0] - .3 * deltas[1], layer.bias),
        rtol=1e-6, atol=1e-7,
    )
    assert torch.equal(layer.weight, original)
    observed_high = actual_projection(high, x)
    torch.testing.assert_close(
        observed_high, functional.linear(x, original + .2 * deltas[2], layer.bias),
        rtol=1e-6, atol=1e-7,
    )
    assert not torch.equal(observed_low, observed_high)
    assert torch.equal(actual_projection(base, x), control)
    assert torch.equal(layer.weight, original)
    assert not base.patches and not base.backup
    assert not torch.cuda.is_initialized()
