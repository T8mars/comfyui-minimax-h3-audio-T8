"""Independent baked-PDD identity negatives using actual tiny Core files/math.

No trained checkpoint, media quality, GPU or new sampler algorithm is qualified.
Do not duplicate the parent lifecycle/parity scope in test_h21_pdd_lifecycle.py.
"""
from __future__ import annotations

import json

import comfy.model_management
from comfy.ldm.minimax import model as native
from comfy.ldm.modules import attention
from comfy.weight_adapter.lora import LoRAAdapter
from comfy_extras.nodes_custom_sampler import BasicGuider, RandomNoise
import pytest
import torch

from h3_audio_t8_pkg import pdd_advanced as pdd
from h3_audio_t8_pkg.modular_sampling import pdd_stages as split
from h3_audio_t8_pkg.modular_sampling.results import sample_stage, selected_model_identity
from h3_audio_t8_pkg.modular_sampling.storage import load_stage, save_stage
from h3_audio_t8_pkg.patch_stack_policy import UnverifiedModelStack
from test_fast_h3_v2_core_sampler import model
from test_h21_pdd_lifecycle import baked_file, load_cpu
from test_modular_rf_restart import source_latent
from test_progressive_sampling_runtime import conditioning


@pytest.fixture
def baked(tmp_path, monkeypatch):
    monkeypatch.setattr(native, "optimized_attention", attention.attention_pytorch)
    path, _, _ = baked_file(tmp_path)
    return load_cpu(path)


def run(bound, sampler, sigmas, context, source, seed=211038):
    noise = RandomNoise.execute(seed).result[0]
    guider = BasicGuider.execute(bound, conditioning()).result[0]
    return sample_stage(noise, guider, sampler, sigmas, source, context)[2]


def test_baked_two_head_loras_order_strength_and_load_identity_remain_distinct(baked):
    key = "diffusion_model.final_layer.video_out.weight"
    raw = baked.model_state_dict()[key].detach().clone()
    adapters = [LoRAAdapter({"first"}, (torch.full((raw.shape[0], 1), .013),
                                       torch.full((1, raw.shape[1]), .014), 1., None, None, None)),
                LoRAAdapter({"second"}, (torch.full((raw.shape[0], 1), -.017),
                                        torch.full((1, raw.shape[1]), .009), 1., None, None, None))]
    source = source_latent(True)
    branches = []
    for order, amounts in (((0, 1), (.7, -.4)), ((1, 0), (.7, -.4)), ((0, 1), (.6, -.4))):
        selected = baked.clone()
        for index in order:
            assert selected.add_patches({key: adapters[index]}, amounts[index]) == [key]
        bound, _, _, context, _ = split.build_stage(selected, source, pdd.pdd_runtime_sigmas())
        patches = bound.patches[key]
        assert [item[1] for item in patches] == [adapters[index] for index in order]
        assert [item[0] for item in patches] == [amounts[index] for index in order]
        branches.append((bound, context, selected_model_identity(bound)))
    assert len({identity["sha256"] for _, _, identity in branches}) == 3
    contracts = [json.loads(context.profile)["pdd"] for _, context, _ in branches]
    assert contracts[0] == contracts[1] == contracts[2]
    assert contracts[0]["mode"] == split.BAKED and contracts[0]["loader_receipt"] is None
    bound, _, identity_before = branches[0]
    try:
        comfy.model_management.load_model_gpu(bound)
        assert not torch.equal(bound.model_state_dict()[key], raw)
        assert selected_model_identity(bound) == identity_before
        assert split._head_contract(bound) == contracts[0]
        assert [item[1] for item in bound.patches[key]] == adapters
    finally:
        bound.unpatch_model()
    torch.testing.assert_close(baked.model_state_dict()[key], raw, rtol=0, atol=0)
    assert baked.get_attachment(pdd.PDD_ATTACHMENT_KEY) is None
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("damage", ["video_nan", "audio_bias_nan", "video_rows", "finite_bank_change"])
def test_baked_actual_bank_damage_after_setup_cannot_keep_old_stage_identity(baked, damage):
    source = source_latent(True)
    bound, _, sigmas, context, _ = split.build_stage(baked, source, pdd.pdd_runtime_sigmas())
    final = bound.get_model_object("diffusion_model.final_layer")
    with torch.no_grad():
        if damage == "video_nan":
            final.video_out.weight[0, 0] = float("nan")
        elif damage == "audio_bias_nan":
            final.audio_out.bias[0] = float("nan")
        elif damage == "video_rows":
            final.video_out.weight = torch.nn.Parameter(final.video_out.weight[:-1], requires_grad=False)
        else:
            final.video_out.weight[0, 0] += .1
    with pytest.raises(ValueError, match="invalid shape or values|heads or loader contract changed"):
        split.validate_stage(bound, sigmas, source, context)


def test_forged_config_without_actual_32_head_rows_does_not_create_baked_identity():
    ordinary = model()
    ordinary.model.model_config.unet_config["pdd_steps"] = 32
    with pytest.raises(ValueError, match="invalid shape or values"):
        split.build_stage(ordinary, source_latent(True), pdd.pdd_runtime_sigmas())
    assert ordinary.get_attachment(pdd.PDD_ATTACHMENT_KEY) is None
    assert not ordinary.patches


def test_actual_unknown_baked_forward_runs_but_saved_artifact_is_not_portable(baked, tmp_path):
    source = source_latent(True)
    bound, sampler, sigmas, context, _ = split.build_stage(baked, source, pdd.pdd_runtime_sigmas())
    final = bound.get_model_object("diffusion_model.final_layer")
    calls = []

    def foreign_hook(module, inputs, outputs):
        assert module is final and all(torch.isfinite(value).all() for value in outputs)
        calls.append(True)

    handle = final.register_forward_hook(foreign_hook)
    try:
        result = run(bound, sampler, sigmas, context, source)
        receipt = result.verify()
        assert calls == [True] * 4
        assert receipt["execution"]["denoiser_evaluations"] == 4
        assert receipt["portable_identity"] is False
        assert bound.get_attachment(pdd.PDD_ATTACHMENT_KEY) is None
        # Known sampler completion does not certify opaque user execution. It
        # can be saved for inspection, but must not become a portable cache.
        assert receipt["verified_recipe_completion"] is True
        path, digest, report_json = save_stage(result, tmp_path / "saved")
        assert json.loads(report_json)["portable_identity"] is False
        with pytest.raises(ValueError, match="unverified executable identity"):
            load_stage(tmp_path / "saved", path, digest, context.stage)
        assert foreign_hook in final._forward_hooks.values()
    finally:
        handle.remove()


def test_unverified_plain_stage_cannot_be_saved_as_completed_artifact(tmp_path):
    source = source_latent(True)
    bound, sampler, sigmas, context, _ = split.build_stage(model(), source, pdd.pdd_runtime_sigmas())
    assert not split.sampler_is_known(bound, sampler, context)
    with pytest.raises(UnverifiedModelStack, match="no content-bound PDD"):
        split.project_identity(bound)
    result = run(bound, sampler, sigmas, context, source, 211039)
    receipt = result.verify()
    assert not receipt["portable_identity"] and not receipt["verified_recipe_completion"]
    with pytest.raises(ValueError, match="incomplete or unknown recipe"):
        save_stage(result, tmp_path / "must-not-publish")
    assert not (tmp_path / "must-not-publish").exists()
