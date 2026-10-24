"""Factory forwarding matrix, not model/task compatibility or GPU qualification.

All former 55 task/audio ban cases remain, now checking implemented phase input
forwarding. The real project resolver maps recordings to Ref2VA/Hybrid; these
unit cases deliberately exercise the factory contract rather than renaming that
resolver's output to T2VA. Missing input/sampler APIs still have negative cases.
"""

from __future__ import annotations

import pytest
from copy import deepcopy

from h3_audio_t8_pkg import director_generation, director_hyperflow
from h3_audio_t8_pkg.director_generation import build_director_generation_prompt
from h3_audio_t8_pkg.director_hyperflow import apply_hyperflow_graph
from h3_audio_t8_pkg.director_project import ProjectStore, new_project
from h3_audio_t8_pkg.director_sampling_settings import normalize_sampling, two_pass_canvas
from test_director_generation import _patch_director_models


VARIANTS = ("single8", "continuous4plus4", "continuous4plus4separate",
            "upscale8plus4", "upscale4plus4")
TASKS = ("t2va", "i2va", "l2va", "fl2va", "ref2va", "hybrid")
AUDIO_MODES = ("native", "lock_source")
EXPERIMENTAL_PAIRS = tuple(
    (task, audio_mode)
    for task in TASKS
    for audio_mode in AUDIO_MODES
    if (task, audio_mode) != ("t2va", "native")
)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("task,audio_mode", EXPERIMENTAL_PAIRS)
def test_explicit_task_audio_inputs_are_forwarded_to_every_real_phase_factory(
    tmp_path, monkeypatch, variant, task, audio_mode,
):
    _patch_director_models(monkeypatch)
    monkeypatch.setattr(director_generation.folder_paths, "get_filename_list", lambda folder: {
        "latent_upscale_models": ["minimax_h3_latent_upscaler_3d_fp16.safetensors"],
    }.get(folder, []))
    monkeypatch.setattr(director_generation.folder_paths, "get_full_path", lambda _folder, name: name)
    weight = tmp_path / "hf.safetensors"
    weight.write_bytes(b"factory contract only; no inference")
    monkeypatch.setattr(director_hyperflow, "_resolve", lambda _selection: weight)
    project = new_project()
    project["doc"]["shots"][0]["simplePrompt"] = "An explicit native factory contract."
    store = ProjectStore(tmp_path / "user", tmp_path / "input")
    built = build_director_generation_prompt(project, project["current"], store)
    graph, shot = built["prompt"], built["shot"]
    shot.update(task_type=task, audio_mode=audio_mode, two_pass_canvas=two_pass_canvas(16 / 9, .4))
    graph["5"]["inputs"].update(task_type={"ref2va": "Ref2VA", "hybrid": "Hybrid"}.get(task, task.upper()),
        audio_mode=audio_mode, audio_denoise_strength=0. if audio_mode == "lock_source" else 1.)
    original_inputs = deepcopy(graph["5"]["inputs"])
    sampling = normalize_sampling({"mode": "hyperflow", "variant": variant,
        "hyperflow_file": "hyperflow/hf.safetensors", "output_mp": .4})
    result = apply_hyperflow_graph(graph, shot, sampling, store, 0,
        {"memory": {}, "semantic_bridge": {}, "prompt_relay": {}, "fast_h3_v2": {}})
    factories = [node["inputs"] for node in graph.values() if node["class_type"] == "MiniMaxH3AudioConditioningT8"]
    assert len(factories) == (1 if variant == "single8" else 2)
    for inputs in factories:
        for field in ("task_type", "audio_mode", "audio_denoise_strength", "prompt", "clip", "video_vae", "audio_vae"):
            assert inputs[field] == original_inputs[field]
    assert result["stage_nfe"] == ([8] if variant == "single8" else [8, 4] if variant == "upscale8plus4" else [4, 4])
    assert result["quality_status"] == "experimental_not_human_reviewed"
    assert graph["10"]["inputs"]["av_latent"][0] in graph


@pytest.mark.parametrize("feature", ("prompt_relay", "fast_h3_v2"))
def test_relay_is_paired_and_competing_fast_sampler_is_explicit(
    tmp_path, monkeypatch, feature
):
    _patch_director_models(monkeypatch)
    monkeypatch.setattr(director_generation, "resolve_ffmpeg", lambda: "ffmpeg")
    project = new_project()
    shot = project["doc"]["shots"][0]
    shot["simplePrompt"] = "A single steady shot in a bright room."
    project["doc"]["sampling"] = {
        "mode": "hyperflow", "variant": "single8",
        "hyperflow_file": "hyperflow/placeholder.safetensors",
    }
    project["doc"]["d3"][feature] = {"enabled": True}
    store = ProjectStore(tmp_path / "user", tmp_path / "input")
    if feature == "fast_h3_v2":
        with pytest.raises(ValueError, match="SAMPLER/SIGMAS.*联合采样协议"):
            build_director_generation_prompt(project, shot["id"], store)
    else:
        weight = tmp_path / "hf.safetensors"
        weight.write_bytes(b"factory preflight only")
        monkeypatch.setattr(director_hyperflow, "_resolve", lambda _selection: weight)
        graph = build_director_generation_prompt(project, shot["id"], store)["prompt"]
        setup = graph[graph["7"]["inputs"]["model"][0]]
        condition_id = setup["inputs"]["model"][0]
        assert graph[condition_id]["class_type"] == "MiniMaxH3PromptRelayConditioningT8Advanced"
        assert graph["7"]["inputs"]["conditioning"] == [condition_id, 1]
        assert graph["9"]["inputs"]["latent_image"] == [condition_id, 2]


def test_actual_missing_sampler_contract_is_not_a_model_ban():
    with pytest.raises(ValueError, match="SAMPLER/SIGMAS.*联合网格协议"):
        apply_hyperflow_graph({}, {"task_type": "unknown_user_model"}, {}, None, 0, {})
