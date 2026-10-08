"""Actual Core RES/LoRA identities; no pretrained/GPU/quality claims."""
from types import MethodType

import pytest
import torch
from comfy.weight_adapter.lora import LoRAAdapter

from h3_audio_t8_pkg import res_history_exp as res
from h3_audio_t8_pkg import res_history_setup as setup
from h3_audio_t8_pkg.res_model_identity import loaded_model_identity
from test_fast_h3_v2_core_sampler import model
from test_modular_rf_restart import source_latent
from test_res_history_setup import options, run


def prepared(bare, source, tmp_path, **kwargs):
    return setup.setup_res_history_sampling(bare, source, **options(tmp_path, **kwargs))


def test_independent_actual_Core_models_have_equal_identity_not_filename_or_address(tmp_path):
    source, first, second = source_latent(), model(), model()
    branch_a = prepared(first, source, tmp_path, mode="disabled")[0]
    branch_b = prepared(second, source, tmp_path, mode="disabled")[0]
    before = [value.detach().clone() for value in first.model.parameters()]
    identity = loaded_model_identity(branch_a)
    assert identity == loaded_model_identity(branch_b)
    assert identity["portable_cache_reuse"] and identity["automatic_loaded_weights_verified"]
    assert identity["model_filename_trusted"] is False
    with torch.no_grad():
        next(second.model.parameters()).reshape(-1)[0].add_(.001)
    assert loaded_model_identity(branch_b) != identity
    assert all(torch.equal(a, b) for a, b in zip(before, first.model.parameters(), strict=True))
    assert not first.object_patches and not first.patches and not torch.cuda.is_initialized()


def test_actual_native_LoRA_tensor_bytes_order_and_strength_are_bound(tmp_path):
    bare, source = model(), source_latent()
    key = "diffusion_model.blocks.0.attn.out_proj.weight"
    weight = bare.model_state_dict()[key]
    rows, columns = weight.shape
    adapter = LoRAAdapter({"up", "down"}, (
        torch.ones(rows, 1) * .01, torch.ones(1, columns) * .02, 1., None, None, None))
    branch = bare.clone()
    assert branch.add_patches({key: adapter}, .25) == [key]
    selected = prepared(branch, source, tmp_path, mode="disabled")[0]
    first = loaded_model_identity(selected)
    assert first["portable_cache_reuse"] and first["weights"]["lora_target_count"] == 1
    adapter.weights[0][0, 0] += .01
    changed_bytes = loaded_model_identity(selected)
    assert changed_bytes != first
    selected.patches[key][0] = (.75, *selected.patches[key][0][1:])
    changed_strength = loaded_model_identity(selected)
    assert changed_strength != changed_bytes
    assert selected.add_patches({key: ("diff", (torch.ones_like(weight) * .001,))}, .5) == [key]
    ordered = loaded_model_identity(selected)
    selected.patches[key].reverse()
    assert loaded_model_identity(selected) != ordered
    assert not bare.patches and not torch.cuda.is_initialized()


def test_actual_RES_file_rejects_changed_loaded_base_despite_same_text_contract(tmp_path):
    bare, source = model(), source_latent(masked=True)
    checkpoint = prepared(bare, source, tmp_path, mode="checkpoint", confirm_checkpoint_write=True)
    run(*checkpoint[:3], source)
    saved = res.read_checkpoint(tmp_path / "boundaries", "native.h3res.safetensors")
    assert saved["payload"]["run_contract"]["loaded_weight_fingerprint_verified"] is True
    with torch.no_grad():
        next(bare.model.parameters()).reshape(-1)[0].add_(.01)
    restored = prepared(bare, source, tmp_path, mode="resume")
    with pytest.raises(ValueError, match="contract changed"):
        run(*restored[:3], source)
    assert res.read_checkpoint(tmp_path / "boundaries", "native.h3res.safetensors")["file_sha256"] == saved["file_sha256"]
    assert not torch.cuda.is_initialized()


def test_unknown_real_forward_is_retained_for_full_run_not_certified_for_recovery(tmp_path):
    bare, source = model(), source_latent(masked=True)
    network = bare.model.diffusion_model
    original, calls = network.forward, []

    def foreign(self, *args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    network.forward = MethodType(foreign, network)
    checkpoint = prepared(bare, source, tmp_path, mode="checkpoint", confirm_checkpoint_write=True)
    assert loaded_model_identity(checkpoint[0])["portable_cache_reuse"] is False
    result = run(*checkpoint[:3], source)
    assert len(calls) == 8
    assert all(torch.isfinite(value).all() for value in result[0]["samples"].unbind())
    restored = prepared(bare, source, tmp_path, mode="resume")
    with pytest.raises(ValueError, match="no portable RES identity"):
        run(*restored[:3], source)
    assert network.forward.__func__ is foreign and not torch.cuda.is_initialized()


def test_disabled_sampler_does_not_scan_MODEL_identity_or_write_checkpoint(tmp_path, monkeypatch):
    def forbidden(model):
        raise AssertionError("disabled mode must not scan MODEL weights")
    monkeypatch.setattr(setup, "loaded_model_identity", forbidden)
    bare, source = model(), source_latent()
    disabled = prepared(bare, source, tmp_path, mode="disabled")
    run(*disabled[:3], source)
    assert not (tmp_path / "boundaries").exists() and not torch.cuda.is_initialized()
