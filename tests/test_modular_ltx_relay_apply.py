"""Real Core LTX CrossAttention/ModelPatcher; no trained-weight claim."""
from contextlib import contextmanager
import json

import pytest
import torch
from comfy.ldm.lightricks.model import CrossAttention
from comfy.ldm.lightricks.symmetric_patchifier import SymmetricPatchifier
from comfy.model_patcher import ModelPatcher

from h3_audio_t8_pkg.modular_sampling import ltx_relay_apply as relay
from h3_audio_t8_pkg.modular_sampling.ltx_relay_plan import build_ltx_relay_plan
from h3_audio_t8_pkg.modular_sampling.ltx_relay_text import encode_ltx_relay_conditioning


class TinyClip:
    def tokenize(self, text):
        return text

    def encode_from_tokens_scheduled(self, tokens):
        length = len(tokens.split()) + 2
        return [[torch.arange(length * 4, dtype=torch.float32).reshape(1, length, 4) / 10 + length,
                 {"unprocessed_ltxav_embeds": True}]]


def fixture(*, gated=False, causal=False):
    torch.manual_seed(806)
    attention = CrossAttention(8, context_dim=4, heads=2, dim_head=4,
                               apply_gated_attention=gated, operations=torch.nn,
                               dtype=torch.float32, device="cpu").eval()
    block = torch.nn.Module()
    block.attn2 = attention
    block.audio_attn2 = CrossAttention(8, context_dim=4, heads=2, dim_head=4,
                                       operations=torch.nn).eval()
    diffusion = torch.nn.Module()
    diffusion.patchifier = SymmetricPatchifier(1, start_end=True)
    diffusion.vae_scale_factors = (8, 32, 32)
    diffusion.causal_temporal_positioning = causal
    diffusion.transformer_blocks = torch.nn.ModuleList([block])
    root = torch.nn.Module()
    root.diffusion_model = diffusion
    model = ModelPatcher(root, torch.device("cpu"), torch.device("cpu"))
    latent = {"samples": torch.zeros(1, 128, 3, 2, 2)}
    plan, *_ = build_ltx_relay_plan(latent, "Cinematic night street",
                                    "Woman arrives\nWoman leaves", "auto_equal", "",
                                    24., .1, False, False)
    positive, binding, _ = encode_ltx_relay_conditioning(TinyClip(), plan)
    context = positive[0][0]
    x = torch.randn(1, 12, 8)
    sigmas = torch.tensor([.9, .5, 0.])
    return model, latent, sigmas, positive, binding, context, x, attention


@contextmanager
def installed(model):
    model.patch_model(load_weights=False)
    try:
        yield model.model.diffusion_model.transformer_blocks[0].attn2
    finally:
        model.unpatch_model(unpatch_weights=False)


@pytest.mark.parametrize("mode", relay.MODES)
@pytest.mark.parametrize("gated", [False, True])
def test_native_video_forward_identity_or_nonidentity(mode, gated):
    model, latent, sigmas, positive, binding, context, x, original = fixture(gated=gated)
    options = {"sigmas": sigmas[1:2], "cond_or_uncond": [0]}
    with torch.no_grad():
        baseline = original(x, context=context, transformer_options=options)
        patched, forwarded, runtime, _ = relay.apply_ltx_relay(
            model, latent, sigmas, positive, binding, mode)
        assert forwarded is positive
        if mode != "disabled":
            patched.prepare_state(sigmas[:1], patched.model_options)
        with installed(patched) as live:
            actual = live(x, context=context, transformer_options=options)
    if mode == "apply_exp":
        assert not torch.equal(actual, baseline)
        assert runtime.applied_calls == [1] and runtime.bias_chunks >= 1
    else:
        assert torch.equal(actual, baseline)
    candidate, raw = relay.audit_ltx_relay(patched, latent, runtime)
    assert candidate is latent and json.loads(raw)["candidate_provenance_verified"] is False
    assert model.model.diffusion_model.transformer_blocks[0].audio_attn2.forward.__func__ is CrossAttention.forward


def test_negative_cfg_row_exact_and_prior_backend_mask_delegated():
    model, latent, sigmas, positive, binding, context, x, original = fixture()
    x = x.repeat(2, 1, 1)
    context = context.repeat(2, 1, 1)
    seen = []
    def previous(func, q, k, v, heads, **kwargs):
        seen.append(kwargs.get("mask"))
        return func(q, k, v, heads, **kwargs)
    options = {"sigmas": sigmas[1:2], "cond_or_uncond": [0, 1],
               "optimized_attention_override": previous}
    with torch.no_grad():
        baseline = original(x, context=context, transformer_options=options)
        seen.clear()
        patched, _, runtime, _ = relay.apply_ltx_relay(model, latent, sigmas, positive,
                                                        binding, "apply_exp")
        with installed(patched) as live:
            actual = live(x, context=context, transformer_options=options)
    assert not torch.equal(actual[0], baseline[0])
    assert torch.equal(actual[1], baseline[1])
    assert seen[0] is None and any(isinstance(mask, torch.Tensor) for mask in seen[1:])
    assert runtime.backend_delegate_calls >= 2 and runtime.reasons["existing_override_backend_unverified"] == 0
    assert options["optimized_attention_override"] is previous


@pytest.mark.parametrize("mask_kind", ["bool", "additive"])
def test_original_attention_mask_is_composed_not_dropped(mask_kind):
    model, latent, sigmas, positive, binding, context, x, original = fixture()
    keys = context.shape[1]
    mask = torch.ones(1, 1, 1, keys, dtype=torch.bool) if mask_kind == "bool" else (
        torch.zeros(1, 1, 1, keys))
    if mask_kind == "bool":
        mask[..., -1] = False
    else:
        mask[..., -1] = float("-inf")
    masks = []
    def previous(func, q, k, v, heads, **kwargs):
        masks.append(kwargs.get("mask"))
        return func(q, k, v, heads, **kwargs)
    options = {"sigmas": sigmas[1:2], "cond_or_uncond": [0],
               "optimized_attention_override": previous}
    with torch.no_grad():
        baseline = original(x, context=context, mask=mask, transformer_options=options)
        patched, _, runtime, _ = relay.apply_ltx_relay(model, latent, sigmas, positive,
                                                        binding, "apply_exp")
        masks.clear()
        with installed(patched) as live:
            actual = live(x, context=context, mask=mask, transformer_options=options)
    assert not torch.equal(actual, baseline)
    assert masks and masks[0].shape == (1, 1, 12, keys)
    assert torch.isneginf(masks[0][..., -1]).all()
    assert runtime.applied_calls == [1]


def test_unknown_cfg_and_non_delegating_foreign_forward_are_unverified_passthrough():
    model, latent, sigmas, positive, binding, context, x, original = fixture()
    with torch.no_grad():
        expected = original(x, context=context)
    patched, _, runtime, _ = relay.apply_ltx_relay(model, latent, sigmas, positive,
                                                    binding, "apply_exp")
    with torch.no_grad(), installed(patched) as live:
        actual = live(x, context=context, transformer_options={"sigmas": sigmas[1:2]})
    assert torch.equal(actual, expected)
    assert runtime.reasons["unknown_cfg_branch_layout"] == 1
    model, latent, sigmas, positive, binding, context, x, _ = fixture()
    path = "diffusion_model.transformer_blocks.0.attn2.forward"
    def foreign(x, context=None, mask=None, pe=None, k_pe=None, transformer_options={}):
        return x
    model.add_object_patch(path, foreign)
    patched, _, runtime, _ = relay.apply_ltx_relay(model, latent, sigmas, positive,
                                                    binding, "apply_exp")
    with installed(patched) as live:
        assert live(x, context=context, transformer_options={"sigmas": sigmas[1:2],
                                                              "cond_or_uncond": [0]}) is x
    assert runtime.reasons["producer_bypassed_selector"] == 1
    assert runtime.applied_calls == [0]


def test_eav_and_relay_patch_distinct_attention_and_keep_origins():
    from h3_audio_t8_pkg.modular_sampling.eav import EAVConfig
    from h3_audio_t8_pkg.modular_sampling import ltx_eav
    model, latent, sigmas, positive, binding, context, x, _ = fixture()
    block = model.model.diffusion_model.transformer_blocks[0]
    block.attn1 = CrossAttention(8, heads=2, dim_head=4, operations=torch.nn).eval()
    eav_config = EAVConfig("report_only", .5, 0., 1., 4, 3.)
    eav_model, eav_runtime, _ = ltx_eav.apply_ltx_eav(model, latent, sigmas, eav_config)
    patched, _, runtime, _ = relay.apply_ltx_relay(eav_model, latent, sigmas, positive,
                                                    binding, "report_only")
    assert "diffusion_model.transformer_blocks.0.attn1.forward" in patched.object_patches
    assert "diffusion_model.transformer_blocks.0.attn2.forward" in patched.object_patches
    with torch.no_grad(), installed(patched) as live:
        block.attn1(x, transformer_options={"sigmas": sigmas[1:2]})
        live(x, context=context, transformer_options={"sigmas": sigmas[1:2],
                                                      "cond_or_uncond": [0]})
    assert eav_runtime.measured == [1] and runtime.observed_calls == [1]
    assert model.object_patches == {}


@pytest.mark.parametrize("causal", [False, True])
def test_native_temporal_coordinates_follow_selected_model(causal):
    model, latent, sigmas, positive, binding, _, _, _ = fixture(causal=causal)
    patched, _, runtime, _ = relay.apply_ltx_relay(model, latent, sigmas, positive,
                                                    binding, "report_only")
    assert patched is not model and runtime.times.numel() == 12
    assert runtime.times[0] == 0
    assert runtime.times[4] == pytest.approx((1 if causal else 8) * 5 / 3)


def test_wrong_binding_and_later_owner_are_reported_not_hidden():
    model, latent, sigmas, positive, binding, context, x, original = fixture()
    positive[0][0][0, 0, 0] += 1
    with pytest.raises(ValueError, match="not the encoded binding"):
        relay.apply_ltx_relay(model, latent, sigmas, positive, binding, "apply_exp")
    model, latent, sigmas, positive, binding, context, x, original = fixture()
    patched, _, runtime, _ = relay.apply_ltx_relay(model, latent, sigmas, positive,
                                                    binding, "apply_exp")
    later = patched.clone()
    later.add_object_patch(runtime.paths[0], original.forward)
    _, raw = relay.audit_ltx_relay(later, latent, runtime)
    assert json.loads(raw)["replaced_forward_paths"] == list(runtime.paths)


class TinyLTXClip:
    def tokenize(self, text):
        return text

    def encode_from_tokens_scheduled(self, tokens):
        length = len(tokens.split()) + 2
        data = torch.arange(length * 16, dtype=torch.float32).reshape(1, length, 16) / 100
        return [[data + len(tokens), {"frame_rate": 24.,
                                      "attention_mask": torch.ones(1, length, dtype=torch.int64)}]]


def tiny_ltx_model():
    import comfy.latent_formats
    import comfy.model_base
    import comfy.ops
    import comfy.supported_models_base
    class Config(comfy.supported_models_base.BASE):
        latent_format = comfy.latent_formats.LTXV
        unet_extra_config = {}
        sampling_settings = {"shift": 1.}
        custom_operations = comfy.ops.disable_weight_init
    cfg = Config(dict(in_channels=128, cross_attention_dim=48, attention_head_dim=24,
        num_attention_heads=2, caption_channels=16, num_layers=2, dtype=torch.float32))
    base = comfy.model_base.LTXV(cfg, device=torch.device("cpu"))
    generator = torch.Generator().manual_seed(8132)
    with torch.no_grad():
        for parameter in base.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * .08)
    return ModelPatcher(base, torch.device("cpu"), torch.device("cpu"))


def native_sample(model, sigmas, source, positive):
    import comfy.samplers
    from comfy_extras.nodes_custom_sampler import BasicGuider, RandomNoise, SamplerCustomAdvanced
    guider = BasicGuider.execute(model, positive).result[0]
    noise = RandomNoise.execute(438).result[0]
    return SamplerCustomAdvanced.execute(noise, guider, comfy.samplers.sampler_object("euler"),
                                         sigmas, source).result[0]


@pytest.mark.parametrize("identity", [False, True])
@pytest.mark.parametrize("mode", relay.MODES)
@pytest.mark.parametrize("lora_strength", [0., .7])
def test_real_core_tiny_ltx_euler_two_setups_and_rank_one_lora(identity, mode, lora_strength):
    from h3_audio_t8_pkg import sol_engine_h3_super_advanced as sol
    model = tiny_ltx_model()
    weight_key = "diffusion_model.transformer_blocks.0.attn2.to_v.weight"
    weight = model.model.state_dict()[weight_key]
    up = torch.ones(weight.shape[0], 1) * .2
    down = torch.ones(1, weight.shape[1]) * .1
    assert model.add_patches({weight_key: ("diff", (up @ down,))}, lora_strength) == [weight_key]
    setup = sol.setup_ltx_identity_preserve_refiner if identity else sol.setup_ltx_stage2_refiner
    model, sigmas, _, _ = setup(model, attention_backend="dense_reference")
    latent = {"samples": torch.zeros(1, 128, 3, 2, 2)}
    plan, *_ = build_ltx_relay_plan(latent, "A stable night street",
                                    "A red coat enters\nA red coat leaves", "auto_equal", "",
                                    24., .1, False, False)
    positive, binding, _ = encode_ltx_relay_conditioning(TinyLTXClip(), plan)
    baseline = native_sample(model, sigmas, latent, positive)
    patched, forwarded, runtime, _ = relay.apply_ltx_relay(model, latent, sigmas,
                                                            positive, binding, mode)
    actual = native_sample(patched, sigmas, latent, forwarded)
    assert torch.isfinite(actual["samples"]).all()
    assert actual["samples"].shape == latent["samples"].shape
    if mode == "apply_exp":
        assert not torch.equal(actual["samples"], baseline["samples"])
        assert runtime.observed_calls == [3, 3] and runtime.applied_calls == [3, 3]
    else:
        assert torch.equal(actual["samples"], baseline["samples"])
    if mode != "disabled":
        assert runtime.closed and runtime.lifecycle_observed and runtime.run_count == 1
        repeated = native_sample(patched, sigmas, latent, forwarded)
        assert torch.equal(repeated["samples"], actual["samples"])
        assert runtime.run_count == 2 and runtime.observed_calls == [3, 3]
    assert patched.patches == model.patches
    assert torch.equal(native_sample(model, sigmas, latent, positive)["samples"], baseline["samples"])
