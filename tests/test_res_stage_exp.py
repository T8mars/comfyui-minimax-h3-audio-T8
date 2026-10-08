"""Actual tiny Core RES Stage behavior. No pretrained/GPU or quality approval."""
import asyncio
import json
from types import MethodType

import pytest
import torch
from comfy_extras.nodes_custom_sampler import BasicGuider, RandomNoise

from h3_audio_t8_pkg import res_history_exp as history
from h3_audio_t8_pkg import res_stage_exp as stage
from h3_audio_t8_pkg.modular_sampling.results import StageResult, canonical, sha
from h3_audio_t8_pkg.modular_sampling.storage import load_stage, save_stage
from h3_audio_t8_pkg.nodes_res_stage_exp import MiniMaxH3RESStageSamplerEXPT8, MiniMaxH3RESStageLoadEXPT8
from h3_audio_t8_pkg.res_history_setup import setup_res_history_sampling
from test_fast_h3_v2_core_sampler import model
from test_modular_rf_restart import source_latent
from test_progressive_sampling_runtime import conditioning
from test_res_history_setup import options, run


def sample(selected, source):
    branch, sampler, sigmas = selected[:3]
    return stage.sample_res_stage(RandomNoise.execute(42).result[0],
        BasicGuider.execute(branch, conditioning()).result[0], sampler, sigmas, source)


def assert_outputs_exact(first, second):
    for a, b in zip(first[:2], second[:2], strict=True):
        for av_a, av_b in zip(a["samples"].unbind(), b["samples"].unbind(), strict=True):
            assert torch.equal(av_a, av_b)


def test_actual_full_RES_outputs_completion_save_and_exact_readonly_load(tmp_path):
    bare, source = model(), source_latent(masked=True)
    selected = setup_res_history_sampling(bare, source, **options(tmp_path, mode="disabled"))
    reference = run(*selected[:3], source)
    actual = sample(selected, source)
    assert_outputs_exact(reference, actual)
    receipt = actual[2].verify()
    assert receipt["verified_recipe_completion"] and receipt["portable_identity"]
    assert receipt["execution"]["callbacks"] == list(range(8))
    assert receipt["execution"]["denoiser_evaluations"] == 8
    assert receipt["execution"]["native_sampler_returns"] == 1
    assert actual[3].stage == "res_complete"
    assert receipt["request"]["res_result_qualification"]["generic_MODEL_portability_granted"] is False
    artifact, digest, _ = save_stage(actual[2], tmp_path / "stages", "res")
    loaded = load_stage(tmp_path / "stages", artifact, digest, "res_complete")
    assert_outputs_exact(actual, loaded)
    assert loaded[3].receipt_json == actual[2].receipt_json
    assert not bare.object_patches and not torch.cuda.is_initialized()


def test_actual_checkpoint_and_remaining4_result_have_distinct_interval_same_final_AV(tmp_path):
    bare, source = model(), source_latent(masked=True)
    first = setup_res_history_sampling(bare, source,
        **options(tmp_path, mode="checkpoint", confirm_checkpoint_write=True))
    complete = sample(first, source)
    digest = history.read_checkpoint(tmp_path / "boundaries", "native.h3res.safetensors")["file_sha256"]
    second = setup_res_history_sampling(bare, source, **options(tmp_path, mode="resume"))
    remaining = sample(second, source)
    assert_outputs_exact(complete, remaining)
    receipt = remaining[2].verify()
    assert remaining[3].stage == "res_remaining_complete" and remaining[3].start == 4
    assert receipt["execution"]["callbacks"] == [0, 1, 2, 3]
    assert receipt["execution"]["denoiser_evaluations"] == 4
    assert receipt["verified_recipe_completion"] and receipt["portable_identity"]
    assert history.read_checkpoint(tmp_path / "boundaries", "native.h3res.safetensors")["file_sha256"] == digest
    artifact, expected, _ = save_stage(remaining[2], tmp_path / "stages", "remaining")
    loaded = load_stage(tmp_path / "stages", artifact, expected, "res_remaining_complete")
    assert_outputs_exact(remaining, loaded)
    with pytest.raises(ValueError, match="expected stage"):
        load_stage(tmp_path / "stages", artifact, expected, "res_complete")
    assert not torch.cuda.is_initialized()


def test_ordinary_native_boundary_and_Stage_remaining4_are_bidirectionally_compatible(tmp_path):
    bare, source = model(), source_latent(masked=True)
    ordinary = setup_res_history_sampling(bare, source,
        **options(tmp_path, mode="checkpoint", confirm_checkpoint_write=True))
    reference = run(*ordinary[:3], source)
    second = setup_res_history_sampling(bare, source, **options(tmp_path, mode="resume"))
    actual = sample(second, source)
    assert_outputs_exact(reference, actual)
    assert actual[2].verify()["execution"]["global_post_denoiser_callbacks"] == [4, 5, 6, 7]
    assert actual[2].verify()["portable_identity"]
    different = {**options(tmp_path, mode="checkpoint", confirm_checkpoint_write=True),
                 "checkpoint_path": "stage_native.h3res.safetensors"}
    stage_complete = sample(setup_res_history_sampling(bare, source, **different), source)
    native_resume = setup_res_history_sampling(bare, source,
        **{**options(tmp_path, mode="resume"), "checkpoint_path": "stage_native.h3res.safetensors"})
    assert_outputs_exact(stage_complete, run(*native_resume[:3], source))
    assert not torch.cuda.is_initialized()


def test_unknown_forward_is_executed_preserved_and_frozen_reuse_remains_unqualified(tmp_path):
    bare, source = model(), source_latent(masked=True)
    network = bare.model.diffusion_model
    original, calls = network.forward, []

    def foreign(self, *args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    network.forward = MethodType(foreign, network)
    actual = sample(setup_res_history_sampling(bare, source, **options(tmp_path, mode="disabled")), source)
    assert len(calls) == 8 and network.forward.__func__ is foreign
    assert actual[2].verify()["verified_recipe_completion"]
    assert actual[2].verify()["portable_identity"] is False
    artifact, digest, _ = save_stage(actual[2], tmp_path / "stages", "opaque")
    with pytest.raises(ValueError, match="unverified executable identity"):
        load_stage(tmp_path / "stages", artifact, digest, "res_complete")
    assert not torch.cuda.is_initialized()


def test_actual_sampling_exception_cannot_return_or_save_completed_stage(tmp_path):
    bare, source = model(), source_latent()
    calls = []

    def failed(self, *args, **kwargs):
        calls.append(1)
        raise RuntimeError("actual model execution failed")

    bare.model.diffusion_model.forward = MethodType(failed, bare.model.diffusion_model)
    selected = setup_res_history_sampling(bare, source, **options(tmp_path, mode="disabled"))
    with pytest.raises(RuntimeError, match="actual model execution failed"):
        sample(selected, source)
    assert calls == [1] and not (tmp_path / "stages").exists()
    assert not torch.cuda.is_initialized()


def test_self_consistent_rehashed_receipt_cannot_claim_missing_NFE_completion(tmp_path):
    bare, source = model(), source_latent()
    actual = sample(setup_res_history_sampling(bare, source, **options(tmp_path, mode="disabled")), source)
    forged = json.loads(actual[2].receipt_json)
    forged["execution"]["denoiser_evaluations"] = 7
    forged.pop("receipt_sha256")
    forged["receipt_sha256"] = sha(forged)
    invalid = StageResult(actual[0], actual[1], canonical(forged))
    with pytest.raises(ValueError, match="literal NFE"):
        save_stage(invalid, tmp_path / "forged", "res")
    assert not (tmp_path / "forged").exists()
    actual[0]["samples"].unbind()[0].add_(.01)
    with pytest.raises(ValueError, match="mutated"):
        actual[2].verify()
    assert not torch.cuda.is_initialized()


def test_new_node_schemas_and_selected_schedule_guard_do_not_change_legacy_setup(tmp_path):
    first = MiniMaxH3RESStageSamplerEXPT8.define_schema().get_v1_info(MiniMaxH3RESStageSamplerEXPT8)
    second = MiniMaxH3RESStageLoadEXPT8.define_schema().get_v1_info(MiniMaxH3RESStageLoadEXPT8)
    assert list(first.input["required"]) == ["noise", "guider", "sampler", "sigmas", "latent_image"]
    assert list(first.output[:2]) == ["LATENT", "LATENT"]
    assert second.input["required"]["expected_stage"][1]["options"] == list(stage.STAGES)
    bare, source = model(), source_latent()
    selected = setup_res_history_sampling(bare, source, **options(tmp_path, mode="disabled"))
    broken = selected[2].clone()
    broken[1] *= .95
    with pytest.raises(ValueError, match="remaining schedule"):
        stage.sample_res_stage(RandomNoise.execute(42).result[0],
            BasicGuider.execute(selected[0], conditioning()).result[0], selected[1], broken, source)
    assert json.loads(selected[-1])["portable_Stage_qualified"] is False
    assert not (tmp_path / "boundaries").exists() and not torch.cuda.is_initialized()


def test_actual_package_entrypoint_appends_both_RES_Stage_nodes_without_duplicate_IDs():
    import h3_audio_t8_pkg as package
    classes = asyncio.run(package.comfy_entrypoint().get_node_list())
    identifiers = [kind.define_schema().node_id for kind in classes]
    assert len(identifiers) == len(set(identifiers))
    stage_index = identifiers.index("MiniMaxH3RESStageSamplerEXPT8")
    assert identifiers[stage_index:stage_index + 3] == ["MiniMaxH3RESStageSamplerEXPT8", "MiniMaxH3RESStageLoadEXPT8",
                               "MiniMaxH3RESEffectsBindEXPT8"]
    assert "MiniMaxH3RESHistoryEXPT8" in identifiers[:-2]
    assert "MiniMaxH3StageSamplerEXPT8" in identifiers[:-2]
    assert not torch.cuda.is_initialized()
