"""Actual Core packed AV / inpaint wrapper, not pretrained media quality."""
import json
from types import SimpleNamespace

import pytest
import torch
from comfy_extras.nodes_custom_sampler import BasicGuider, RandomNoise, SamplerCustomAdvanced

from h3_audio_t8_pkg import res_history_exp as res
from h3_audio_t8_pkg import res_history_setup as setup
from h3_audio_t8_pkg.nodes_res_history_exp import MiniMaxH3RESHistoryEXPT8
from h3_audio_t8_pkg.sampling import setup_dual_clock_sampling
from test_fast_h3_v2_core_sampler import model
from test_modular_rf_restart import source_latent
from test_progressive_sampling_runtime import conditioning


def options(tmp_path, **kwargs):
    return {"storage_root": tmp_path / "boundaries", "steps": 8, "shift_video": 12., "shift_audio": 3.,
            "checkpoint_step": 4, "checkpoint_path": "native.h3res.safetensors",
            "model_contract_id": "actual_test_Core_tiny_same_base_no_LoRA_not_pretrained",
            "run_contract_json": '{"purpose":"actual_Core_tiny_AV_with_mask"}',
            "hash_chunk_megabytes": 1, **kwargs}


def run(branch, sampler, sigmas, source, *, seed=42, positive=None):
    return SamplerCustomAdvanced.execute(RandomNoise.execute(seed).result[0],
        BasicGuider.execute(branch, conditioning() if positive is None else positive).result[0],
        sampler, sigmas, source).result


def test_actual_Core_packed_masked_AV_full_and_history_resume_outputs_are_exact(tmp_path):
    bare, source = model(), source_latent(masked=True)
    # Preserve the existing native H3 batch-one boundary, not a fabricated
    # multi-batch claim from a permissive synthetic solver fixture.
    with pytest.raises(ValueError, match="batch size 1"):
        setup.setup_res_history_sampling(bare, source_latent(masked=True, batch=2),
                                        **options(tmp_path, mode="disabled"))
    original = [value.clone() for value in source["samples"].unbind()]
    masks = [value.clone() for value in source["noise_mask"].unbind()]
    core_branch, core_sampler, full = setup_dual_clock_sampling(bare, source, 8, 12., 3.,
        "res_multistep", "native_flow")
    reference = run(core_branch, core_sampler, full, source)
    branch, sampler, sigmas, _, _, report = setup.setup_res_history_sampling(bare, source,
        **options(tmp_path, mode="checkpoint", confirm_checkpoint_write=True))
    assert not (tmp_path / "boundaries").exists() and not json.loads(report)["setup_writes_files"]
    checkpointed = run(branch, sampler, sigmas, source)
    saved = res.read_checkpoint(tmp_path / "boundaries", "native.h3res.safetensors")
    assert saved["history"].completed_steps == 4
    assert saved["payload"]["run_contract"]["actual_conditioning"]["portable_condition_identity"]
    resumed_branch, resumed_sampler, remaining, *_ = setup.setup_res_history_sampling(bare, source,
        **options(tmp_path, mode="resume"))
    assert torch.equal(remaining, full[4:])
    resumed = run(resumed_branch, resumed_sampler, remaining, source)
    for expected, uninterrupted, continued in zip(reference, checkpointed, resumed, strict=True):
        for a, b, c in zip(expected["samples"].unbind(), uninterrupted["samples"].unbind(),
                            continued["samples"].unbind(), strict=True):
            assert torch.equal(a, b) and torch.equal(a, c)
    assert res.read_checkpoint(tmp_path / "boundaries", "native.h3res.safetensors")["file_sha256"] == saved["file_sha256"]
    assert all(torch.equal(a, b) for a, b in zip(source["samples"].unbind(), original, strict=True))
    assert all(torch.equal(a, b) for a, b in zip(source["noise_mask"].unbind(), masks, strict=True))
    assert not bare.object_patches and not torch.cuda.is_initialized()


def test_actual_resume_rejects_changed_seed_condition_and_original_input_without_writing(tmp_path):
    bare, source = model(), source_latent(masked=True)
    first = setup.setup_res_history_sampling(bare, source,
        **options(tmp_path, mode="checkpoint", confirm_checkpoint_write=True))
    run(*first[:3], source)
    original_sha = res.read_checkpoint(tmp_path / "boundaries", "native.h3res.safetensors")["file_sha256"]
    restored = setup.setup_res_history_sampling(bare, source, **options(tmp_path, mode="resume"))
    with pytest.raises(ValueError, match="contract changed"):
        run(*restored[:3], source, seed=43)
    changed_condition = conditioning()
    changed_condition[0][0] = changed_condition[0][0] + .01
    restored = setup.setup_res_history_sampling(bare, source, **options(tmp_path, mode="resume"))
    with pytest.raises(ValueError, match="contract changed"):
        run(*restored[:3], source, positive=changed_condition)
    source["samples"].unbind()[0].add_(.01)
    restored = setup.setup_res_history_sampling(bare, source, **options(tmp_path, mode="resume"))
    with pytest.raises(ValueError, match="current latent differs"):
        run(*restored[:3], source)
    assert res.read_checkpoint(tmp_path / "boundaries", "native.h3res.safetensors")["file_sha256"] == original_sha
    assert not torch.cuda.is_initialized()


def test_new_schema_disabled_and_opaque_condition_has_no_forged_portable_identity(tmp_path):
    schema = MiniMaxH3RESHistoryEXPT8.define_schema().get_v1_info(MiniMaxH3RESHistoryEXPT8)
    assert schema.input["required"]["mode"][1]["default"] == "disabled"
    assert schema.input["required"]["confirm_checkpoint_write"][1]["default"] is False
    bare, source = model(), source_latent()
    selected = setup.setup_res_history_sampling(bare, source, **options(tmp_path, mode="disabled"))
    assert not (tmp_path / "boundaries").exists()
    report = json.loads(selected[-1])
    assert report["portable_Stage_qualified"] is False
    assert report["weights_identity"] == "not_checked_disabled_mode"
    runtime = SimpleNamespace(inner_model=SimpleNamespace(original_conds={"positive": [
        {"cross_attn": torch.ones(1, 2), "external": object(), "uuid": object()}]}))
    identity = setup._conditioning_identity(runtime, 1024)
    assert identity["opaque_paths"] and not identity["portable_condition_identity"]
    with pytest.raises(ValueError, match="write confirmation"):
        setup.setup_res_history_sampling(bare, source, **options(tmp_path, mode="checkpoint"))
    assert not (tmp_path / "boundaries").exists()
