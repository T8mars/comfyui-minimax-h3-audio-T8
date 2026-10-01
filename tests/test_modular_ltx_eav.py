"""Real Core CrossAttention + ModelPatcher, tiny initialized weights on CPU.

No stand-in attention kernel; not a trained-model or complete-media test.
"""
import asyncio
from contextlib import contextmanager
from dataclasses import replace
import json

import pytest
import torch
from comfy.ldm.lightricks.model import CrossAttention, GuideAttentionMask
from comfy.ldm.lightricks.symmetric_patchifier import SymmetricPatchifier
from comfy.model_patcher import ModelPatcher

import h3_audio_t8_pkg
from h3_audio_t8_pkg.modular_sampling.eav import EAVConfig
from h3_audio_t8_pkg.modular_sampling import ltx_eav as eav, ltx_effect_nodes

PATH = "diffusion_model.transformer_blocks.0.attn1.forward"
CONFIG = EAVConfig("apply_exp", .5, 0., 1., 4, 3.)


def fixture(*, uniform=True, gated=True, batch=1):
    torch.manual_seed(230)
    attention = CrossAttention(8, heads=2, dim_head=4, apply_gated_attention=gated,
                               operations=torch.nn, dtype=torch.float32, device="cpu").eval()
    if uniform:
        with torch.no_grad():
            attention.to_q.weight.zero_()
            attention.to_q.bias.zero_()
            attention.to_k.weight.zero_()
            attention.to_k.bias.zero_()
    block = torch.nn.Module()
    block.attn1 = attention
    block.audio_attn1 = CrossAttention(8, heads=2, dim_head=4, operations=torch.nn).eval()
    diffusion = torch.nn.Module()
    diffusion.patchifier = SymmetricPatchifier(1, start_end=True)
    diffusion.transformer_blocks = torch.nn.ModuleList([block])
    root = torch.nn.Module()
    root.diffusion_model = diffusion
    model = ModelPatcher(root, torch.device("cpu"), torch.device("cpu"))
    latent = {"samples": torch.zeros(batch, 128, 3, 2, 2)}
    sigmas = torch.tensor([.9, .5, 0.])
    return model, latent, sigmas, torch.randn(batch, 12, 8), attention


@contextmanager
def installed(model):
    model.patch_model(load_weights=False)
    try:
        yield model.model.diffusion_model.transformer_blocks[0].attn1
    finally:
        model.unpatch_model(unpatch_weights=False)


def config(mode):
    return replace(CONFIG, mode=mode)


@pytest.mark.parametrize("mode", ["disabled", "report_only", "apply_exp"])
@pytest.mark.parametrize("gated", [False, True])
def test_native_forward_identity_or_exact_post_projection_gain(mode, gated):
    model, latent, sigmas, x, original = fixture(gated=gated)
    options = {"sigmas": torch.tensor([.5]), "sentinel": object()}
    with torch.no_grad():
        baseline = original(x, transformer_options=options)
        patched, runtime, initial = eav.apply_ltx_eav(model, latent, sigmas, config(mode))
        if mode == "disabled":
            assert patched is model and model.get_attachment(eav.KEY) is None
        else:
            assert json.loads(initial)["status"] == "unverified_no_effect_observed"
            patched.prepare_state(sigmas[:1], patched.model_options)
        with installed(patched) as live:
            actual = live(x, transformer_options=options)
        if mode == "apply_exp":
            torch.testing.assert_close(actual, baseline * (1 + .5 / 3), rtol=1e-6, atol=1e-6)
            assert not torch.equal(actual, baseline)
            assert runtime.applied == [1]
        else:
            assert torch.equal(actual, baseline)
        assert set(options) == {"sigmas", "sentinel"}
        assert not model.object_patches and original.forward.__func__ is CrossAttention.forward
        candidate, report = eav.audit_ltx_eav(patched, latent, runtime)
        assert candidate is latent
        assert json.loads(report)["sampler_completion_verified"] is False
        assert json.loads(report)["candidate_provenance_verified"] is False


@pytest.mark.parametrize("batch", [1, 2])
def test_bounded_statistic_matches_full_temporal_reference(batch):
    torch.manual_seed(34)
    q, k = torch.randn(batch, 3, 20, 7), torch.randn(batch, 3, 20, 7)
    a = q.reshape(batch, 3, 5, 4, 7).permute(0, 1, 3, 2, 4)
    b = k.reshape(batch, 3, 5, 4, 7).permute(0, 1, 3, 2, 4)
    p = ((a @ b.transpose(-1, -2)) * (7 ** -.5)).softmax(-1)
    reference = (p.sum() - p.diagonal(dim1=-2, dim2=-1).sum()) / (batch * 3 * 4 * 5 * 4)
    calls = []
    actual, rows, workspace = eav.chunked_cfi(q, k, 5, 4, 2000, lambda: calls.append(True))
    assert actual == pytest.approx(float(reference), abs=2e-8)
    assert rows == 1 and workspace <= 2000 and len(calls) == batch * 3 * 4 + 1


def test_statistic_resource_limit_cancel_and_nan_are_real_errors():
    q = torch.zeros(1, 2, 12, 4)
    with pytest.raises(ValueError, match="workspace"):
        eav.chunked_cfi(q, q, 3, 4, 1, lambda: None)
    def cancel():
        raise InterruptedError("cancel between statistics")
    with pytest.raises(InterruptedError):
        eav.chunked_cfi(q, q, 3, 4, 10000, cancel)
    q[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        eav.chunked_cfi(q, q, 3, 4, 10000, lambda: None)


@pytest.mark.parametrize("guided", [False, True])
def test_masks_backend_delegate_and_gates_retained_guide_tail_unchanged(guided):
    model, latent, sigmas, x, original = fixture()
    if guided:
        x = torch.cat((x, torch.randn(1, 3, 8)), dim=1)
        mask = GuideAttentionMask(15, 12, 2, torch.tensor([.5, 0.]))
    else:
        mask = torch.zeros(1, 1, 12, 12)
        mask[..., -1] = float("-inf")
    seen = []
    def previous(func, q, k, v, heads, *args, **kwargs):
        seen.append(kwargs.get("mask"))
        return func(q, k, v, heads, *args, **kwargs)
    options = {"sigmas": sigmas[1:2], "optimized_attention_override": previous}
    with torch.no_grad():
        baseline = original(x, mask=mask, transformer_options=options)
        seen.clear()
        patched, runtime, _ = eav.apply_ltx_eav(model, latent, sigmas, CONFIG)
        with installed(patched) as live:
            actual = live(x, mask=mask, transformer_options=options)
    torch.testing.assert_close(actual[:, :12], baseline[:, :12] * (1 + .5 / 3))
    if guided:
        assert torch.equal(actual[:, 12:], baseline[:, 12:])
        assert seen[0] is mask.noisy_mask and seen[1] is mask.tracked_mask and seen[2] is None
    else:
        assert seen == [mask]
    assert runtime.measured == [1] and runtime.applied == [1]
    assert options["optimized_attention_override"] is previous


def test_real_rope_and_cfg_batch_are_delegated_in_report_only():
    model, latent, sigmas, x, original = fixture(uniform=False)
    x = x.repeat(2, 1, 1)
    angle = torch.randn(2, 12, 2, 2)
    rotation = torch.stack((angle.cos(), -angle.sin(), angle.sin(), angle.cos()), -1).reshape(2, 12, 2, 2, 2, 2)
    options = {"sigmas": torch.tensor([.5, .5])}
    with torch.no_grad():
        baseline = original(x, pe=(rotation, False), transformer_options=options)
        patched, runtime, _ = eav.apply_ltx_eav(model, latent, sigmas, config("report_only"))
        with installed(patched) as live:
            actual = live(x, pe=(rotation, False), transformer_options=options)
    assert torch.equal(actual, baseline) and runtime.measured == [1]


@pytest.mark.parametrize("options,reason", [
    ({"sigmas": torch.tensor([.5]), "stg_skip_self_attn": True}, "stg_or_cross_attention_bypass"),
    ({}, "missing_native_sigma"),
    ({"sigmas": torch.tensor([.5, .4])}, "mixed_or_invalid_sigma"),
    ({"sigmas": torch.tensor([1.])}, "outside_bound_sigma_window"),
    ({"sigmas": torch.tensor([.1])}, "outside_effect_window"),
])
def test_bypass_paths_keep_native_output_exact(options, reason):
    model, latent, sigmas, x, original = fixture()
    cfg = replace(CONFIG, start_video_progress=.15, end_video_progress=.85)
    with torch.no_grad():
        baseline = original(x, transformer_options=options)
        patched, runtime, _ = eav.apply_ltx_eav(model, latent, sigmas, cfg)
        with installed(patched) as live:
            actual = live(x, transformer_options=options)
    assert torch.equal(actual, baseline) and runtime.reasons[reason] == 1
    assert runtime.measured == [0] and runtime.applied == [0]


def test_original_forward_weight_patches_callbacks_and_audio_preserved():
    model, latent, sigmas, x, original = fixture()
    original_forward = original.forward
    original_calls, callback_calls = [], []
    def foreign(x, context=None, mask=None, pe=None, k_pe=None, transformer_options={}):
        original_calls.append(True)
        return original_forward(x, context, mask, pe, k_pe, transformer_options) + .25
    model.add_object_patch(PATH, foreign)
    model.patches["sentinel_weight_patch"] = [(1., object())]
    model.add_callback_with_key(eav.extension.CallbacksMP.ON_PREPARE_STATE, "foreign", lambda *_: callback_calls.append(True))
    audio_forward = model.model.diffusion_model.transformer_blocks[0].audio_attn1.forward
    patched, runtime, _ = eav.apply_ltx_eav(model, latent, sigmas, CONFIG)
    assert patched.patches == model.patches and model.object_patches[PATH] is foreign
    patched.prepare_state(sigmas[:1], patched.model_options)
    with installed(patched) as live, torch.no_grad():
        result = live(x, transformer_options={"sigmas": sigmas[1:2]})
    expected = (original_forward(x) + .25) * (1 + .5 / 3)
    torch.testing.assert_close(result, expected)
    assert original_calls == [True] and callback_calls == [True]
    assert model.model.diffusion_model.transformer_blocks[0].audio_attn1.forward == audio_forward
    assert runtime.lifecycle_observed and runtime.run_count == 1
    runtime.cleanup(patched)
    patched.prepare_state(sigmas[:1], patched.model_options)
    assert runtime.run_count == 2 and runtime.measured == [0]


def test_opaque_producer_and_later_patch_are_uncovered_not_forbidden():
    model, latent, sigmas, x, _ = fixture()
    def foreign(x, transformer_options={}):
        return x
    model.add_object_patch(PATH, foreign)
    patched, runtime, _ = eav.apply_ltx_eav(model, latent, sigmas, CONFIG)
    with installed(patched) as live:
        assert live(x, transformer_options={"sigmas": sigmas[1:2]}) is x
    assert runtime.reasons["producer_bypassed_selector"] == 1
    later = patched.clone()
    later.add_object_patch(PATH, foreign)
    candidate, raw = eav.audit_ltx_eav(later, latent, runtime)
    assert candidate is latent and json.loads(raw)["replaced_forward_paths"] == [PATH]
    assert json.loads(raw)["status"] == "unverified_no_effect_observed"


def test_pending_parent_replacement_not_shadowed_by_earlier_child_patch():
    model, latent, sigmas, x, _ = fixture()
    other, _, _, _, replacement = fixture(uniform=False)
    def old_child(*_args, **_kwargs):
        raise AssertionError("parent replacement superseded this child")
    model.add_object_patch(PATH, old_child)
    model.add_object_patch("diffusion_model.transformer_blocks", other.model.diffusion_model.transformer_blocks)
    assert eav.effective_object(model, PATH) == replacement.forward
    patched, runtime, _ = eav.apply_ltx_eav(model, latent, sigmas, config("report_only"))
    with installed(patched) as live, torch.no_grad():
        actual = live(x, transformer_options={"sigmas": sigmas[1:2]})
    assert torch.equal(actual, replacement(x))
    assert runtime.measured == [1]


@pytest.mark.parametrize("mode", ["report_only", "apply_exp"])
def test_gain_limit_and_foreign_exceptions_not_swallowed(mode):
    model, latent, sigmas, x, _ = fixture()
    patched, _, _ = eav.apply_ltx_eav(model, latent, sigmas, replace(config(mode), tau=4., g_hard_limit=1.5))
    with installed(patched) as live, pytest.raises(RuntimeError, match="hard limit"):
        live(x, transformer_options={"sigmas": sigmas[1:2]})
    def fail(*args, **kwargs):
        raise RuntimeError("original producer failed")
    model.add_object_patch(PATH, fail)
    patched, _, _ = eav.apply_ltx_eav(model, latent, sigmas, CONFIG)
    with installed(patched) as live, pytest.raises(RuntimeError, match="original producer failed"):
        live(x)


def test_wrong_runtime_candidate_schedule_and_duplicate_owner_rejected():
    model, latent, sigmas, _, _ = fixture()
    for invalid in [torch.tensor([.1, .8]), torch.tensor([float("nan"), 0.]), torch.tensor([1.2, 0.])]:
        with pytest.raises(ValueError, match="SIGMAS"):
            eav.apply_ltx_eav(model, latent, invalid, CONFIG)
    patched, runtime, _ = eav.apply_ltx_eav(model, latent, sigmas, CONFIG)
    with pytest.raises(ValueError, match="already"):
        eav.apply_ltx_eav(patched, latent, sigmas, CONFIG)
    with pytest.raises(ValueError, match="connected MODEL"):
        eav.audit_ltx_eav(model, latent, runtime)
    with pytest.raises(ValueError, match="geometry"):
        eav.audit_ltx_eav(patched, {"samples": torch.zeros(1, 128, 4, 2, 2)}, runtime)
    with pytest.raises(ValueError, match="non-finite"):
        eav.audit_ltx_eav(patched, {"samples": latent["samples"] + float("nan")}, runtime)


def test_registration_appends_complete_561_prefix_and_nodes_are_callable():
    previous = [*asyncio.run(h3_audio_t8_pkg._HyperFlowLongVideoExtension().get_node_list()),
        *h3_audio_t8_pkg._modular_node_classes(), *h3_audio_t8_pkg._hyper_vae_2x_node_classes,
        *h3_audio_t8_pkg._audio_refine_effect_node_classes, *h3_audio_t8_pkg._ltx_rgb_source_node_classes,
        *h3_audio_t8_pkg._serial_video_io_node_classes]
    actual = asyncio.run(h3_audio_t8_pkg.comfy_entrypoint().get_node_list())
    assert len(previous) == 561
    assert actual[:567] == previous + ltx_effect_nodes.NODES + h3_audio_t8_pkg._ltx_relay_node_classes
    assert len(actual) == len({node.define_schema().node_id for node in actual})
    model, latent, sigmas, _, _ = fixture()
    result = ltx_effect_nodes.MiniMaxH3LTXEAVApplyEXPT8.execute(model, latent, sigmas, config("disabled")).result
    audit = ltx_effect_nodes.MiniMaxH3LTXEAVAuditEXPT8.execute(model, latent, result[1]).result
    assert result[0] is model and audit[0] is latent


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


def native_sample(model, sigmas, source):
    import comfy.samplers
    from comfy_extras.nodes_custom_sampler import BasicGuider, RandomNoise, SamplerCustomAdvanced
    generator = torch.Generator().manual_seed(32)
    positive = [[torch.randn(1, 4, 16, generator=generator),
                 {"frame_rate": 24., "attention_mask": torch.ones(1, 4, dtype=torch.int64)}]]
    guider = BasicGuider.execute(model, positive).result[0]
    noise = RandomNoise.execute(438).result[0]
    return SamplerCustomAdvanced.execute(noise, guider, comfy.samplers.sampler_object("euler"), sigmas, source).result[0]


@pytest.mark.parametrize("identity", [False, True])
@pytest.mark.parametrize("mode", ["disabled", "report_only", "apply_exp"])
@pytest.mark.parametrize("lora_strength", [0., .7])
def test_actual_core_tiny_ltx_sampler_setup_lora_and_repeat_lifecycle(identity, mode, lora_strength):
    from h3_audio_t8_pkg import sol_engine_h3_super_advanced as sol
    model = tiny_ltx_model()
    # Actual rank-one weight delta through Core patching, not an inert sentinel.
    weight_key = "diffusion_model.transformer_blocks.0.attn1.to_v.weight"
    weight = model.model.state_dict()[weight_key]
    up = torch.ones(weight.shape[0], 1) * .2
    down = torch.ones(1, weight.shape[1]) * .1
    assert model.add_patches({weight_key: ("diff", (up @ down,))}, lora_strength) == [weight_key]
    setup = sol.setup_ltx_identity_preserve_refiner if identity else sol.setup_ltx_stage2_refiner
    model, sigmas, _, _ = setup(model, attention_backend="dense_reference")
    latent = {"samples": torch.zeros(1, 128, 3, 2, 2)}
    baseline = native_sample(model, sigmas, latent)
    patched, runtime, _ = eav.apply_ltx_eav(model, latent, sigmas, config(mode))
    actual = native_sample(patched, sigmas, latent)
    assert torch.isfinite(actual["samples"]).all()
    assert actual["samples"].shape == latent["samples"].shape
    if mode == "apply_exp":
        assert not torch.equal(actual["samples"], baseline["samples"])
        assert runtime.measured == [3, 3] and runtime.applied == [3, 3]
        assert runtime.max_gain > 1.
    else:
        assert torch.equal(actual["samples"], baseline["samples"])
    if mode != "disabled":
        assert runtime.closed and runtime.lifecycle_observed and runtime.run_count == 1
        assert runtime.reasons == {}
        repeated = native_sample(patched, sigmas, latent)
        assert torch.equal(repeated["samples"], actual["samples"])
        assert runtime.run_count == 2 and runtime.measured == [3, 3]
    assert patched.patches == model.patches
    # The incoming model remains usable after the effect branch unpatches.
    after = native_sample(model, sigmas, latent)
    assert torch.equal(after["samples"], baseline["samples"])
