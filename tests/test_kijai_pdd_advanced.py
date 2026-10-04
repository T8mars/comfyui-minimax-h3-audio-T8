"""Actual tiny Core relative-head algebra/lifecycle/split effects; not trained quality."""
import json

import pytest
import torch
import comfy.lora
from comfy.weight_adapter.lora import LoRAAdapter
from comfy_extras.nodes_custom_sampler import BasicGuider, RandomNoise

from h3_audio_t8_pkg import kijai_pdd_advanced as kijai, pdd_advanced as pdd
from h3_audio_t8_pkg.modular_sampling import pdd_stages as split, eav
from h3_audio_t8_pkg.modular_sampling.results import sample_stage
from h3_audio_t8_pkg.modular_sampling.storage import load_stage, save_stage
from test_fast_h3_v2_core_sampler import model
from test_modular_rf_restart import source_latent
from test_progressive_sampling_runtime import conditioning
from test_progressive_relay import paired


def original_state(bare, add_backbone=True, head_aliases="up_down"):
    generator = torch.Generator().manual_seed(261003)
    state = {}
    for key in kijai.HEAD_KEYS:
        value = bare.model_state_dict()[key]
        target = key.removesuffix(".weight") if key.endswith(".weight") else key
        shape = (32 * value.shape[0], *value.shape[1:])
        delta = torch.randn(shape, generator=generator) * .003
        if key.endswith(".weight"):
            a, b = delta, torch.eye(shape[0], dtype=torch.bfloat16)
        else:
            a, b = torch.ones(1, 1), delta[:, None]
        a_suffix, b_suffix = ((".lora_down.weight", ".lora_up.weight") if head_aliases == "up_down"
                              else (".lora_A.weight", ".lora_B.weight"))
        state.update({target + a_suffix: a, target + b_suffix: b,
                      target + ".reshape_weight": torch.tensor(shape, dtype=torch.int64)})
    if add_backbone:
        for target in ("diffusion_model.blocks.0.attn.qkv_proj", "diffusion_model.blocks.0.adaln_proj.linear"):
            value = bare.model_state_dict()[target + ".weight"]
            state.update({target + ".lora_A.weight": torch.full((2, value.shape[1]), .02, dtype=torch.bfloat16),
                          target + ".lora_B.weight": torch.full((value.shape[0], 2), .03, dtype=torch.bfloat16),
                          target + ".alpha": torch.tensor(2.)})
            if target + ".bias" in bare.model_state_dict():
                state[target + ".diff_b"] = torch.full((value.shape[0],), .004)
    return state


def setup(stage=split.STAGES[0], strength=1.0, effects=False):
    bare = model()
    prepared, descriptor = kijai.apply_native_state(bare, original_state(bare), strength)
    prepared.set_attachments(pdd.PDD_ATTACHMENT_KEY, {"schema": "t8_minimax_h3_pdd_8step_setup_v2", "lora": descriptor})
    positive = conditioning()
    if effects:
        prepared, positive, _ = paired(prepared)
    source = source_latent(True)
    prepared, sampler, sigmas, context, _ = split.build_stage(prepared, source, pdd.pdd_runtime_sigmas(), stage)
    runtime = None
    if effects:
        prepared, runtime, _ = eav.apply_stage_eav(prepared, sigmas, source, context,
            eav.EAVConfig("apply_exp", tau=.2, start_video_progress=0., end_video_progress=1., g_hard_limit=3.))
    return (RandomNoise.execute(261003).result[0], BasicGuider.execute(prepared, positive).result[0],
            sampler, sigmas, source, context), runtime


@pytest.mark.parametrize("strength", [0., .3, 1.])
@pytest.mark.parametrize("head_aliases", ["up_down", "A_B"])
def test_all_four_relative_heads_exactly_match_original_core_lora_with_padding(strength, head_aliases):
    bare = model()
    state = original_state(bare, head_aliases=head_aliases)
    source = comfy.lora.load_lora(state, comfy.lora.model_lora_keys_unet(bare.model, {}), log_missing=False)
    _, heads, count = kijai.compile_native_patches(bare, state)
    assert count == 7  # two backbone plus one bias plus all four heads
    for key in kijai.HEAD_KEYS:
        value = bare.model_state_dict()[key]
        reference = comfy.lora.calculate_weight([(strength, source[key], 1., None, None)], value.clone(), key)
        actual = comfy.lora.calculate_weight([(strength, heads[key], 1., None, None)], value.clone(), key)
        assert torch.equal(actual, reference)
        assert comfy.lora.calculate_shape([(strength, heads[key], 1., None, None)], value, key) == reference.shape
    assert not bare.patches and not bare.injections


def test_nonidentity_head_b_and_alpha_are_not_discarded():
    a = torch.arange(48, dtype=torch.float32).reshape(8, 6) / 100
    b = torch.eye(8) * 2
    b[0, 1] = .125
    adapter = LoRAAdapter(set(), (b, a, 4., None, None, [8, 6]))
    diff = kijai.relative_head_diff(adapter, (8, 6), "test.weight")[1][0]
    assert torch.equal(diff, (b @ a) * .5)


@pytest.mark.parametrize("stage", split.STAGES)
@pytest.mark.parametrize("strength", [0., .3, 1.])
@pytest.mark.parametrize("effects", [False, True])
def test_real_relative_heads_hot_rebuilt_saved_and_absolute_4nfe(stage, strength, effects, tmp_path, monkeypatch):
    from comfy.ldm.minimax import model as native
    original = native._pdd_head
    seen = []
    def observe(head, hidden, count, start, stop, shift):
        seen.append((count, start, stop, shift))
        return original(head, hidden, count, start, stop, shift)
    monkeypatch.setattr(native, "_pdd_head", observe)
    args, runtime = setup(stage, strength, effects)
    result = sample_stage(*args)[2]
    expected = [(32, i * 4, (i + 1) * 4, shift) for i in range(args[-1].start, args[-1].end) for shift in (12., 3.)]
    assert seen == expected
    receipt = result.verify()
    assert receipt["portable_identity"] and receipt["verified_recipe_completion"]
    assert receipt["execution"]["denoiser_evaluations"] == 4
    assert sample_stage(*args)[2].verify()["request_sha256"] == receipt["request_sha256"]
    assert sample_stage(*setup(stage, strength, effects)[0])[2].verify()["request_sha256"] == receipt["request_sha256"]
    if runtime:
        assert runtime.snapshot()["status"] == "observed_apply_exp"
    path, digest, _ = save_stage(result, tmp_path)
    loaded = load_stage(tmp_path, path, digest, stage)[0]
    assert all(torch.equal(a, b) for a, b in zip(loaded["samples"].unbind(), result.output["samples"].unbind()))
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("damage", ["missing_audio_bias", "unknown_key", "wrong_adaln", "bad_reshape", "nan"])
def test_actual_missing_or_corrupt_source_is_not_silently_loaded(damage):
    bare = model()
    state = original_state(bare)
    if damage == "missing_audio_bias":
        del state["diffusion_model.final_layer.audio_out.bias.lora_down.weight"]
    elif damage == "unknown_key":
        state["unknown.lora_A.weight"] = torch.ones(1, 1)
    elif damage == "wrong_adaln":
        state["diffusion_model.blocks.0.adaln_proj.linear.lora_A.weight"] = torch.ones(2, 8)
    elif damage == "bad_reshape":
        state["diffusion_model.final_layer.video_out.reshape_weight"][0] -= 1
    else:
        state["diffusion_model.final_layer.audio_out.bias.lora_up.weight"][0, 0] = torch.nan
    with pytest.raises(ValueError):
        kijai.apply_native_state(bare, state)
    assert not bare.patches and not bare.injections


@pytest.mark.parametrize("damage", ["strength", "tensor", "base_strength", "receipt", "duplicate"])
def test_bound_relative_head_receipt_damage_is_detected(damage):
    args, _ = setup()
    selected = args[1].model_patcher
    key = kijai.HEAD_KEYS[0]
    patch = selected.patches[key][-1]
    if damage == "tensor":
        patch[1][1][0][0, 0] += .1
    elif damage == "strength":
        selected.patches[key][-1] = (.4, *patch[1:])
    elif damage == "base_strength":
        selected.patches[key][-1] = (patch[0], patch[1], 0., *patch[3:])
    elif damage == "receipt":
        selected.get_attachment(pdd.PDD_ATTACHMENT_KEY)["lora"]["native_head_differences"] = {}
    else:
        selected.patches[key].append(patch)
    with pytest.raises(ValueError):
        split.capture_owner(selected)


def test_existing_bypass_lora_owner_is_retained_not_overwritten():
    from test_modular_core_bypass_identity import prepared
    bare = prepared()
    original = list(bare.get_injections("bypass_lora"))
    patched, descriptor = kijai.apply_native_state(bare, original_state(bare), .7)
    assert patched.get_injections("bypass_lora") == original == bare.get_injections("bypass_lora")
    assert descriptor["backbone_mode"] == "native_weight_patches_existing_bypass_preserved"
    assert "diffusion_model.blocks.0.attn.qkv_proj.weight" in patched.patches


def test_new_node_is_an_explicit_distinct_schema_not_an_old_pdd_migration():
    from h3_audio_t8_pkg.nodes_kijai_pdd_advanced import MiniMaxH3KijaiPDD8StepSetupEXPT8 as node
    from h3_audio_t8_pkg.nodes_pdd_advanced import MiniMaxH3PDD8StepSetupT8Advanced as old
    assert node.define_schema().node_id != old.define_schema().node_id
    assert json.loads(json.dumps(kijai.HEAD_KEYS)) == list(kijai.HEAD_KEYS)


def test_actual_registration_appends_new_setup_and_metadata_matches():
    import asyncio
    from pathlib import Path
    import h3_audio_t8_pkg
    ids = [node.define_schema().node_id for node in asyncio.run(h3_audio_t8_pkg.comfy_entrypoint().get_node_list())]
    assert len(ids) == len(set(ids)) == 620
    assert ids[618:] == ["MiniMaxH3HyperFlowCurveBaseLoaderEXPT8", "MiniMaxH3KijaiPDD8StepSetupEXPT8"]
    assert json.loads((Path(__file__).resolve().parents[1] / "features.json").read_text(encoding="utf8"))["nodes"] == ids


def test_same_loaded_base_rebuilds_only_high_and_preserves_saved_low(tmp_path):
    """One requested hot case, not a new parameter/video qualification matrix."""
    import hashlib
    import comfy.model_management

    bare = model()
    state = original_state(bare)  # Capture before Core temporarily expands heads.
    original_heads = {key: bare.model_state_dict()[key].clone() for key in kijai.HEAD_KEYS}
    targets = [bare.model.diffusion_model.blocks[0].attn.qkv_proj,
               bare.model.diffusion_model.blocks[0].adaln_proj.linear]
    original_forwards = [target.forward for target in targets]
    source = source_latent(True)

    def arguments(stage, strength, latent):
        selected, descriptor = kijai.apply_native_state(bare.clone(), state, strength)
        selected.set_attachments(pdd.PDD_ATTACHMENT_KEY,
            {"schema": "t8_minimax_h3_pdd_8step_setup_v2", "lora": descriptor})
        selected, sampler, sigmas, context, _ = split.build_stage(
            selected, latent, pdd.pdd_runtime_sigmas(), stage)
        assert selected.model is bare.model and selected.backup is bare.backup
        return (RandomNoise.execute(261003).result[0],
                BasicGuider.execute(selected, conditioning()).result[0],
                sampler, sigmas, latent, context)

    def hashes():
        return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in tmp_path.rglob("*") if path.is_file() and path.name != ".stage.lock"}

    try:
        low_args = arguments(split.STAGES[0], 1., source)
        low = sample_stage(*low_args)[2]
        low_receipt = low.verify()
        assert low_receipt["portable_identity"] and low_receipt["execution"]["denoiser_evaluations"] == 4
        path, digest, _ = save_stage(low, tmp_path)
        frozen_low = hashes()
        assert set(kijai.HEAD_KEYS).issubset(bare.backup)
        assert bare.model.diffusion_model.final_layer.video_out.weight.shape[0] == 32 * original_heads[kijai.HEAD_KEYS[0]].shape[0]
        assert bare.model.current_patcher is None  # Normal cleanup, not unload.

        high_args = arguments(split.STAGES[1], 1., low.denoised_output)
        high = sample_stage(*high_args)[2]
        changed_args = arguments(split.STAGES[1], .3, low.denoised_output)
        changed = sample_stage(*changed_args)[2]
        for result in (high, changed):
            receipt = result.verify()
            assert receipt["portable_identity"] and receipt["execution"]["denoiser_evaluations"] == 4
            assert all(bool(torch.isfinite(value).all()) for value in result.output["samples"].unbind())
        assert high.verify()["request_sha256"] != changed.verify()["request_sha256"]
        loaded = load_stage(tmp_path, path, digest, split.STAGES[0])[0]
        assert hashes() == frozen_low and low.verify() == low_receipt
        assert all(torch.equal(a, b) for a, b in zip(loaded["samples"].unbind(), low.output["samples"].unbind()))
    finally:
        # Only this isolated tiny test process; never unload a user's server.
        comfy.model_management.unload_all_models()
    assert all(torch.equal(bare.model_state_dict()[key], value) for key, value in original_heads.items())
    assert [target.forward for target in targets] == original_forwards
    assert not torch.cuda.is_initialized()
