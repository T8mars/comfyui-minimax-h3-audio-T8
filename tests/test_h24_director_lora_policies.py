"""Director keeps explicit per-row interpretation without changing legacy graphs."""
from copy import deepcopy
from pathlib import Path
import shutil
import subprocess

import pytest

from h3_audio_t8_pkg import director_generation as generation
from h3_audio_t8_pkg import director_hyperflow
from h3_audio_t8_pkg.director_project import ProjectStore, generation_settings, new_project
from h3_audio_t8_pkg.director_sampling_settings import normalize_sampling
from test_director_generation import _patch_director_models


POLICIES = {"file_alpha_policy": "metadata_alpha_per_rank", "prefix_policy": "unique_double_prefix"}


def row(name="a.safetensors", **extra):
    return {"id": name, "name": name, "strength": .4, "enabled": True, **extra}


@pytest.mark.parametrize("mode", ["single", "two_pass", "hyperflow"])
def test_active_rows_preserve_explicit_policies_and_old_rows_stay_absent(mode):
    raw = {"mode": mode, "loras": [row(**POLICIES), row("b.safetensors")],
           "low_loras": [row(**POLICIES), row("b.safetensors")], "high_loras": [],
           "hyperflow_file": "hyperflow/test.safetensors"}
    before = deepcopy(raw)
    result = normalize_sampling(raw)
    rows = result["loras" if mode == "single" else "low_loras"]
    assert {key: rows[0].get(key) for key in POLICIES} == POLICIES
    assert all(key not in rows[1] for key in POLICIES)
    assert raw == before


def test_generation_normalizers_and_save_reopen_preserve_disabled_and_legacy_rows(tmp_path):
    project = new_project()
    project["doc"]["generation"].update(lora_mode="manual", loras=[
        row(**POLICIES), row("disabled.safetensors", enabled=False, **POLICIES), row("b.safetensors")])
    for result in (generation_settings(project["doc"]), generation._director_generation_settings(project)):
        assert [item["id"] for item in result["loras"]] == [item["id"] for item in project["doc"]["generation"]["loras"]]
        assert {key: result["loras"][0].get(key) for key in POLICIES} == POLICIES
        assert {key: result["loras"][1].get(key) for key in POLICIES} == POLICIES
        assert all(key not in result["loras"][2] for key in POLICIES)
    store = ProjectStore(tmp_path / "user", tmp_path / "input")
    store.save(project, 0)
    assert store.load(project["id"])["doc"]["generation"]["loras"] == project["doc"]["generation"]["loras"]


def test_legacy_generation_rows_without_ids_do_not_gain_invented_identity():
    project = new_project()
    project["doc"]["generation"] = {"lora": "a.safetensors", "lora_strength": 0.}
    for result in (generation_settings(project["doc"]), generation._director_generation_settings(project)):
        assert result["loras"] == [{"name": "a.safetensors", "strength": 0., "enabled": True}]


@pytest.mark.parametrize("field,value", [("file_alpha_policy", "guess"), ("prefix_policy", []),
                                         ("file_alpha_policy", False), ("prefix_policy", None)])
def test_explicit_invalid_policy_is_an_input_error_not_silently_dropped(field, value):
    with pytest.raises(ValueError, match=field):
        normalize_sampling({"mode": "single", "loras": [row(**{field: value})]})


def test_inactive_high_draft_kept_by_project_but_not_executed_or_validated(tmp_path):
    project = new_project()
    project["doc"]["sampling"] = {
        "mode": "hyperflow", "variant": "single8", "hyperflow_file": "hyperflow/test.safetensors",
        "low_loras": [row(**POLICIES)], "high_loras": [row(prefix_policy="future-draft")]}
    store = ProjectStore(tmp_path / "user", tmp_path / "input")
    store.save(project, 0)
    loaded = store.load(project["id"])
    assert loaded["doc"]["sampling"]["high_loras"] == project["doc"]["sampling"]["high_loras"]
    assert normalize_sampling(loaded["doc"]["sampling"])["high_loras"] == []


@pytest.fixture
def graph_project(tmp_path, monkeypatch):
    _patch_director_models(monkeypatch)
    monkeypatch.setattr(generation.folder_paths, "get_filename_list", lambda folder: {
        "loras": ["a.safetensors", "b.safetensors"],
        "diffusion_models": ["minimax_h3_fl2va_int8_convrot.safetensors"],
        "latent_upscale_models": ["minimax_h3_latent_upscaler_3d_fp16.safetensors"],
    }.get(folder, []))
    monkeypatch.setattr(generation.folder_paths, "get_full_path", lambda _folder, name: name)
    original = tmp_path / "hf.safetensors"
    original.write_bytes(b"graph fixture only, never executed as weights")
    monkeypatch.setattr(director_hyperflow, "_resolve", lambda _name: original)
    project = new_project()
    project["doc"]["generation"]["unet"] = "minimax_h3_fl2va_int8_convrot.safetensors"
    project["doc"]["shots"][0]["simplePrompt"] = "A quiet room."
    return project, ProjectStore(tmp_path / "user", tmp_path / "input")


@pytest.mark.parametrize("route", ["generation", "single", "two_pass", "single8", "continuous4plus4",
                                    "continuous4plus4separate", "resume_tail", "upscale8plus4", "upscale4plus4"])
def test_each_compiled_stage_keeps_order_strength_and_explicit_interpretation(graph_project, route, tmp_path, monkeypatch):
    project, store = graph_project
    low = [row(**POLICIES), row("b.safetensors", strength=-.2)]
    high = [row("a.safetensors", strength=.7, prefix_policy="unique_double_prefix")]
    if route == "generation":
        project["doc"]["generation"].update(lora_mode="manual", loras=low)
    elif route == "single":
        project["doc"]["sampling"] = {"mode": "single", "lora_mode": "manual", "loras": low}
    else:
        project["doc"]["sampling"] = {
            "mode": "two_pass" if route == "two_pass" else "hyperflow", "variant": route,
            "hyperflow_file": "hyperflow/hf.safetensors", "output_mp": .4,
            "low_loras": low, "high_loras": high}
        if route == "resume_tail":
            from test_modular_hyperflow import inputs, head
            from h3_audio_t8_pkg.modular_sampling.hyperflow_storage import save_boundary

            low_model, _, source, positive = inputs(monkeypatch)
            boundary = head(low_model, source, positive)
            monkeypatch.setattr(generation.folder_paths, "get_output_directory", lambda: str(tmp_path))
            path, digest, _ = save_boundary(boundary, tmp_path / "MiniMaxH3" / "hyperflow_stage_artifacts")
            project["doc"]["sampling"].update(variant="continuous4plus4separate",
                stage_checkpoint={"mode": "resume_tail", "artifact_path": path, "artifact_sha256": digest})
            low[0]["prefix_policy"] = "unused-head-draft"
    built = generation.build_director_generation_prompt(project, project["current"], store)
    loaders = [node["inputs"] for node in built["prompt"].values()
               if node["class_type"] == "MiniMaxH3LoRACompatibilityLoaderT8Advanced"]
    expected = ([] if route == "resume_tail" else low) + ([] if route in {"generation", "single", "single8"} else high)
    assert [(item["lora_name"], item["strength_model"]) for item in loaders] == [(item["name"], item["strength"]) for item in expected]
    for actual, wanted in zip(loaders, expected):
        assert {key: actual[key] for key in POLICIES if key in actual} == {key: wanted[key] for key in POLICIES if key in wanted}
    if route == "generation":
        assert built["turbo_lora"] == [("a.safetensors", .4), ("b.safetensors", -.2)]
    assert sum(node["class_type"] == "UNETLoader" for node in built["prompt"].values()) == 1
    bindings = built["report"]["lora_loaders"]
    assert len(bindings) == len(expected)
    assert {row["node_id"] for row in bindings} == {key for key, node in built["prompt"].items()
        if node["class_type"] == "MiniMaxH3LoRACompatibilityLoaderT8Advanced"}
    for binding, wanted in zip(bindings, expected):
        actual = built["prompt"][binding["node_id"]]["inputs"]
        assert actual["lora_name"] == wanted["name"]
        assert binding["stage"] in {"SINGLE", "LOW", "HIGH"}
        assert binding["row_index"] >= 1
    if route == "resume_tail":
        assert all(row["stage"] == "HIGH" for row in bindings)


def test_explicit_legacy_defaults_do_not_change_other_graph_inputs(graph_project):
    project, store = graph_project
    project["doc"]["sampling"] = {"mode": "two_pass", "output_mp": .4,
                                  "low_loras": [row()], "high_loras": [row("b.safetensors")]}
    before = generation.build_director_generation_prompt(project, project["current"], store)["prompt"]
    for stage in ("low_loras", "high_loras"):
        project["doc"]["sampling"][stage][0].update(file_alpha_policy="legacy_tensor_only", prefix_policy="legacy")
    after = generation.build_director_generation_prompt(project, project["current"], store)["prompt"]
    for node in after.values():
        if node["class_type"] == "MiniMaxH3LoRACompatibilityLoaderT8Advanced":
            for key in POLICIES:
                node["inputs"].pop(key, None)
    assert before == after


def test_high_only_policy_change_does_not_rewrite_low_graph_or_old_project(graph_project):
    project, store = graph_project
    project['doc']['sampling'] = {'mode': 'two_pass', 'output_mp': .4,
        'low_loras': [row(**POLICIES)], 'high_loras': [row('b.safetensors')]}
    original = deepcopy(project)
    before = generation.build_director_generation_prompt(project, project['current'], store)['prompt']
    assert project == original
    project['doc']['sampling']['high_loras'][0]['prefix_policy'] = 'unique_double_prefix'
    after = generation.build_director_generation_prompt(project, project['current'], store)['prompt']
    changed = [key for key in before if before[key] != after[key]]
    assert len(changed) == 1
    node = after[changed[0]]
    assert node['class_type'] == 'MiniMaxH3LoRACompatibilityLoaderT8Advanced'
    assert node['inputs']['lora_name'] == 'b.safetensors'
    old = deepcopy(node)
    old['inputs'].pop('prefix_policy')
    assert old == before[changed[0]]


def test_frozen_version_copy_keeps_both_stage_policies_without_mutating_source(tmp_path):
    import uuid
    from h3_audio_t8_pkg.director_project import atomic_json, sha
    from h3_audio_t8_pkg.director_routes import copy_version_project, director_result_snapshot

    store = ProjectStore(tmp_path / 'user', tmp_path / 'input')
    project = new_project()
    project['doc']['sampling'] = {'mode': 'two_pass', 'output_mp': .4,
        'low_loras': [row(**POLICIES)], 'high_loras': [row('b.safetensors', enabled=False, **POLICIES)]}
    project = store.save(project, 0)
    before = deepcopy(project)
    request_id, target_id = str(uuid.uuid4()), str(uuid.uuid4())
    snapshot = {'project': project, 'shot_id': project['current'], 'seed': 42}
    atomic_json(store.root / 'requests' / f'{request_id}.json', {
        'project_id': project['id'], 'shot_id': project['current'],
        'snapshot': snapshot, 'snapshot_sha256': sha(snapshot)})
    copy_version_project(store, project['id'], request_id, target_id)
    copied = store.load(target_id)
    assert copied['doc']['sampling'] == project['doc']['sampling']
    assert store.load(project['id']) == before
    assert director_result_snapshot(store, project['id'], request_id)['snapshot'] == snapshot


def test_frontend_active_input_policy_sequences():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required")
    suite = Path(__file__).with_name("director_h24_lora_policies.mjs")
    result = subprocess.run([node, "--test", str(suite)], capture_output=True, text=True,
                            encoding="utf8", timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr

@pytest.mark.parametrize("mode", ("generation", "single", "two_pass"))
def test_loader_bindings_keep_original_row_number_identity_and_owner(graph_project, mode):
    project, store = graph_project
    rows = [row("off.safetensors", enabled=False, id="off"),
            row(id="selected-row", strength=3.)]
    if mode == "generation":
        project["doc"]["generation"].update(lora_mode="manual", loras=rows)
    else:
        project["doc"]["sampling"] = {"mode": mode, "lora_mode": "manual",
            "loras": rows, "low_loras": rows, "high_loras": [], "output_mp": .4}
    built = generation.build_director_generation_prompt(project, project["current"], store)
    bindings = built["report"]["lora_loaders"]
    assert len(bindings) == 1
    assert bindings[0]["row_index"] == 2
    assert bindings[0]["row_id"] == "selected-row"
    assert bindings[0]["source"] == ("generation_rows" if mode == "generation" else "sampling_rows")
