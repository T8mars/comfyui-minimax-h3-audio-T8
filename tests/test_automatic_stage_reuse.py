"""Execute Core, not a fake sampler: transparent binding and Director wiring."""
from copy import deepcopy
import json

import pytest
import torch
from comfy_extras.nodes_custom_sampler import SamplerCustomAdvanced

from h3_audio_t8_pkg.modular_sampling import automatic_reuse as auto, results
from h3_audio_t8_pkg.director_stage_reuse import apply_stage_reuse, reuse_reports
from h3_audio_t8_pkg.director_sampling_settings import normalize_sampling
from test_modular_native_explicit import inputs


def execute(root, args, **kwargs):
    value = auto.sample(*args, root, mode="reuse_completed", **kwargs)
    return value, json.loads(value[3])


@pytest.mark.parametrize("role", ["native_low", "native_high"])
def test_real_native_parity_hit_and_unmodified_inputs(tmp_path, monkeypatch, role):
    _, _, original = inputs(stage=role)
    selected = original[1].model_patcher
    attachments = dict(selected.attachments)
    expected = SamplerCustomAdvanced.execute(*original).result
    first, report = execute(tmp_path, original, stage_role=role)
    assert report["status"] == "miss_saved"
    assert report["current_denoiser_evaluations"] == 4
    for actual, reference in zip(first[:2], expected):
        assert all(torch.equal(a, b) for a, b in zip(actual["samples"].unbind(), reference["samples"].unbind()))
    assert selected.attachments == attachments
    monkeypatch.setattr(results, "sample_stage", lambda *a: pytest.fail("hit must not sample"))
    second, hit = execute(tmp_path, original, stage_role=role)
    assert hit["status"] == "verified_hit" and hit["current_denoiser_evaluations"] == 0
    assert results._input_identity(first[0]) == results._input_identity(second[0])


def test_unknown_sampler_runs_once_with_original_objects_and_no_cache(tmp_path):
    _, _, args = inputs()
    calls = []
    def foreign(denoiser, x, sigmas, **kwargs):
        calls.append(True)
        return x
    args[2].sampler_function = foreign
    output, report = execute(tmp_path, args)
    assert calls == [True]
    assert output[2] is None and report["status"] == "uncertified_executed"
    assert not tmp_path.joinpath("reuse-index").exists()


def test_sampler_failure_propagates_without_retry_or_completion(tmp_path):
    _, _, args = inputs()
    calls = []
    def broken(*a, **k):
        calls.append(True)
        raise RuntimeError("actual kernel failure")
    args[2].sampler_function = broken
    with pytest.raises(RuntimeError, match="actual kernel failure"):
        execute(tmp_path, args)
    assert calls == [True] and not list(tmp_path.rglob("entry.json"))


def test_disabled_exact_native_and_no_store(tmp_path):
    _, _, args = inputs()
    output = auto.sample(*args, tmp_path / "absent")
    assert output[2] is None and json.loads(output[3])["status"] == "disabled_executed"
    assert not (tmp_path / "absent").exists()


def test_old_settings_unchanged_and_invalid_reuse_rejected():
    assert normalize_sampling(None) == normalize_sampling({"mode": "single"}) == {"mode": "single"}
    assert normalize_sampling({"mode": "single", "reuse_mode": "reuse_completed"})["reuse_mode"] == "reuse_completed"
    with pytest.raises(ValueError, match="复用"):
        normalize_sampling({"reuse_mode": "pretend_hit"})


def test_compiler_preserves_all_wires_slots_and_explicit_contexts():
    native = dict(noise=["8", 0], guider=["7", 0], sampler=["6", 1],
                  sigmas=["6", 2], latent_image=["5", 1])
    graph = {"9": {"class_type": "SamplerCustomAdvanced", "inputs": native},
             "20": {"class_type": "SamplerCustomAdvanced", "inputs": deepcopy(native)},
             "21": {"class_type": "MiniMaxH3StageSamplerEXPT8", "inputs": {**native, "stage_context": ["19", 3]}},
             "22": {"class_type": "Decode", "inputs": {"latent": ["20", 0]}},
             "23": {"class_type": "StageSave", "inputs": {"stage_result": ["21", 2]}}}
    old = deepcopy(graph)
    assert apply_stage_reuse(graph)["sampler_nodes"] == [] and graph == old
    assert apply_stage_reuse(graph, "reuse_completed")["sampler_nodes"] == ["9", "20", "21"]
    for key in ("9", "20", "21"):
        assert graph[key]["class_type"] == "MiniMaxH3StageReuseEXPT8"
        assert all(graph[key]["inputs"][k] == v for k, v in old[key]["inputs"].items())
    assert graph["9"]["inputs"]["stage_role"] == "native_low"
    assert graph["20"]["inputs"]["stage_role"] == "native_high"
    assert graph["22"] == old["22"] and graph["23"] == old["23"]
    assert reuse_reports({"9": {"t8_stage_reuse": [{"status": "verified_hit"}]}}) == [
        {"node_id": "9", "status": "verified_hit"}]


def test_dedicated_recipe_is_preserved_and_does_not_claim_automatic_reuse():
    graph = {"9": {"class_type": "MiniMaxH3HyperFlowSplitT8Advanced", "inputs": {"seed": 31}}}
    original = deepcopy(graph)
    report = apply_stage_reuse(graph, "reuse_completed")
    assert graph == original and report["sampler_nodes"] == []
    assert "未启用自动" in report["warning"]


def test_creator_actual_sampling_receipt_and_unrelated_edit_keeps_hit(tmp_path):
    from test_creator_workspace_advanced import _timeline, _override
    from h3_audio_t8_pkg.creator_workspace_advanced import compile_creator_workspace
    from h3_audio_t8_pkg.creator_runtime_advanced import record_creator_run_receipt
    timeline = _timeline()
    workspace = compile_creator_workspace(timeline, 0, -1, False, "")[0]
    other_edit = compile_creator_workspace(timeline, 0, -1, False, "", _override(timeline))[0]
    assert workspace["workspace_hash"] != other_edit["workspace_hash"]
    _, _, args = inputs()
    args[0].seed = workspace["shots"][0]["variant_seeds"][0]
    output, first = execute(tmp_path, args, creator_workspace=workspace)
    assert first["status"] == "miss_saved"
    _, hit = execute(tmp_path, args, creator_workspace=other_edit)
    assert hit["status"] == "verified_hit"
    retention = compile_creator_workspace(timeline, 0, -1, False, "", _override(timeline,
        shot_index=0, compiled_prompt_override="", use_seed_override=False, variant_count=1,
        media_roles_json="{}", retention_policy="keep_all", hold_policy="none", hold_frames=0))[0]
    _, retained = execute(tmp_path, args, creator_workspace=retention)
    assert retained["status"] == "verified_hit"
    ledger, event, _, report = record_creator_run_receipt(
        workspace, 0, 0, 1, "failed", "test", "unknown", "{}", "decode failed",
        stage_result=output[2])
    assert json.loads(event)["completed_stage"]["stage_completion_verified"] is True
    assert json.loads(report)["stage_completion_verified"] is True
    assert ledger["events"][0]["outcome"] == "failed"
    with pytest.raises(ValueError, match="different Creator"):
        record_creator_run_receipt(workspace, 1, 0, 1, "failed", "test", "unknown", "{}", "",
                                  stage_result=output[2])
    args[0].seed += 1
    with pytest.raises(ValueError, match="variant seed"):
        execute(tmp_path, args, creator_workspace=workspace)


@pytest.mark.parametrize("two_pass", [False, True])
def test_actual_director_compiler_only_changes_sampler_boundary(tmp_path, monkeypatch, two_pass):
    from test_director_generation import _patch_director_models, _store
    from h3_audio_t8_pkg import director_generation as dg
    from h3_audio_t8_pkg.director_project import new_project
    _patch_director_models(monkeypatch)
    monkeypatch.setattr(dg, "resolve_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(dg.folder_paths, "get_filename_list", lambda folder: {
        "latent_upscale_models": ["minimax_h3_latent_upscaler_3d_fp16.safetensors"],
    }.get(folder, []))
    monkeypatch.setattr(dg.folder_paths, "get_full_path", lambda folder, name: name)
    project = new_project()
    project["doc"]["shots"][0]["simplePrompt"] = "A steady shot."
    if two_pass:
        project["doc"]["sampling"] = {"mode": "two_pass", "output_mp": .4}
    first = dg.build_director_generation_prompt(project, project["current"], _store(tmp_path))
    project["doc"]["sampling"]["reuse_mode"] = "reuse_completed"
    wrapped = dg.build_director_generation_prompt(project, project["current"], _store(tmp_path))
    actual = deepcopy(wrapped["prompt"])
    samplers = wrapped["report"]["stage_reuse"]["sampler_nodes"]
    assert len(samplers) == (2 if two_pass else 1)
    for key in samplers:
        assert actual[key]["class_type"] == "MiniMaxH3StageReuseEXPT8"
        actual[key]["class_type"] = "SamplerCustomAdvanced"
        actual[key]["inputs"].pop("mode")
        actual[key]["inputs"].pop("stage_role")
    assert actual == first["prompt"]
