from __future__ import annotations

import uuid
import wave
from copy import deepcopy

import pytest
from PIL import Image

from h3_audio_t8_pkg.director_generation import build_director_generation_prompt
from h3_audio_t8_pkg.director_project import ProjectStore, new_project


def _store(tmp_path):
    return ProjectStore(tmp_path / "user", tmp_path / "input")


def test_missing_encoder_fails_before_model_selection(tmp_path, monkeypatch):
    from h3_audio_t8_pkg import director_generation

    project = new_project()
    project["doc"]["shots"][0]["simplePrompt"] = "A quiet scene."
    def missing():
        raise RuntimeError("未找到 FFmpeg")
    monkeypatch.setattr(director_generation, "resolve_ffmpeg", missing)
    monkeypatch.setattr(director_generation, "_pick_requested", lambda *a: pytest.fail("must preflight encoder first"))
    with pytest.raises(RuntimeError, match="未找到 FFmpeg"):
        build_director_generation_prompt(project, project["current"], _store(tmp_path))


@pytest.mark.parametrize("global_on", [False, True])
def test_explicit_global_d3_overrides_stale_local_and_compiles(tmp_path, monkeypatch, global_on):
    from h3_audio_t8_pkg.director_generation import _director_d3_settings

    _patch_director_models(monkeypatch)
    project = new_project()
    shot = project["doc"]["shots"][0]
    shot["simplePrompt"] = "A quiet cinematic portrait."
    shot["d3Inherit"] = True
    shot["d3"] = {"memory": {"low_vram": not global_on}}
    project["doc"]["d3"] = {"memory": {"low_vram": global_on}}
    assert _director_d3_settings(project, shot)["memory"]["low_vram"] is global_on
    built = build_director_generation_prompt(project, shot["id"], _store(tmp_path))
    types = {node["class_type"] for node in built["prompt"].values()}
    assert ("MiniMaxH3LowVRAMAttentionT8Advanced" in types) is global_on


def test_local_and_legacy_d3_scope_preserves_compatibility():
    from h3_audio_t8_pkg.director_generation import _director_d3_settings

    project = {"doc": {"d3": {"semantic_bridge": {"enabled": True}}}}
    local = {"d3Inherit": False, "d3": {"semantic_bridge": {"enabled": False}}}
    assert _director_d3_settings(project, local)["semantic_bridge"]["enabled"] is False
    assert _director_d3_settings(project, {"d3Inherit": True})["semantic_bridge"]["enabled"] is True
    legacy = {"d3": {"memory": {"low_vram": True}}}
    assert _director_d3_settings(project, legacy)["memory"]["low_vram"] is True


def _image(store, name="frame.png"):
    asset_id = str(uuid.uuid4())
    path = store.input_root / "t8_director" / asset_id / "source.png"
    path.parent.mkdir(parents=True)
    Image.new("RGB", (512, 768), (40, 80, 120)).save(path)
    return store.register_asset(path, asset_id, name)


def _audio(store, name="voice.wav", seconds=4.0):
    asset_id = str(uuid.uuid4())
    path = store.input_root / "t8_director" / asset_id / "source.wav"
    path.parent.mkdir(parents=True)
    frames = int(16_000 * seconds)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16_000)
        stream.writeframes(b"\x00\x00" * frames)
    return store.register_asset(path, asset_id, name)


def test_d2a_builds_real_native_t2va_prompt_without_queue(tmp_path):
    store = _store(tmp_path)
    project = new_project()
    project["doc"]["shots"][0]["simplePrompt"] = "A quiet cinematic room, one continuous shot."
    built = build_director_generation_prompt(project, project["current"], store)
    graph = built["prompt"]
    assert graph["5"]["class_type"] == "MiniMaxH3AudioConditioningT8"
    assert graph["5"]["inputs"]["task_type"] == "T2VA"
    assert graph["6"]["class_type"] == "MiniMaxH3DualClockSamplerT8"
    assert graph["12"]["class_type"] == "MiniMaxH3SafeAVSaveT8Advanced"
    assert built["recipe"].startswith("director_")


def test_d2a_binds_prepared_first_frame(tmp_path):
    store = _store(tmp_path)
    frame = _image(store)
    project = new_project()
    shot = project["doc"]["shots"][0]
    shot.update(mode="first", first=frame["id"], tray=[frame["id"]], simplePrompt="A portrait moves gently.")
    project["assets"] = [frame]
    built = build_director_generation_prompt(project, shot["id"], store)
    assert built["prompt"]["5"]["inputs"]["first_frame"] == ["14", 0]
    assert built["prompt"]["14"]["class_type"] == "LoadImage"
    assert built["prompt"]["14"]["inputs"]["image"].startswith("t8_director/")


def test_director_recipe_contract_is_not_downgraded_for_unsupported_assets(tmp_path):
    store = _store(tmp_path)
    project = new_project()
    shot = project["doc"]["shots"][0]
    shot.update(mode="refs", simplePrompt="A scene.")
    with pytest.raises(ValueError, match="参考素材"):
        build_director_generation_prompt(project, shot["id"], store)


def test_shared_ref_native_audio_current_shot_ignores_unfinished_sibling_drafts(tmp_path, monkeypatch):
    from h3_audio_t8_pkg.director_project import compile_project
    from h3_audio_t8_pkg import nodes_director
    from h3_audio_t8_pkg.director_d3 import inspect_d3_routes
    import json

    _patch_director_models(monkeypatch)
    store = _store(tmp_path)
    ref = _image(store)
    project = new_project()
    shot = project["doc"]["shots"][0]
    shot.update(mode="refs", sound="native", simplePrompt="女人在微笑", first=None, last=None, audio=None)
    project["assets"] = [ref]
    project["doc"]["sharedRefs"] = [ref["id"]]
    ends = deepcopy(shot)
    ends.update(id=str(uuid.uuid4()), name="未完成首尾", mode="ends", simplePrompt="")
    avatar = deepcopy(shot)
    avatar.update(id=str(uuid.uuid4()), name="未完成录音", mode="first", sound="record", simplePrompt="")
    project["doc"]["shots"].extend([ends, avatar])
    original = deepcopy(project)

    full = compile_project(project, store)
    assert not full["ready"]
    assert {error["shot_number"] for error in full["errors"]} == {2, 3}
    scoped = compile_project(project, store, shot_id=shot["id"])
    assert scoped["ready"] and len(scoped["shots"]) == 1
    assert scoped["selection"] == {"scope": "shot", "shot_id": shot["id"]}
    built = build_director_generation_prompt(project, shot["id"], store)
    inputs = built["prompt"]["5"]["inputs"]
    assert inputs["task_type"] == "Ref2VA"
    assert inputs["prompt"] == "女人在微笑"
    assert "ref_images.ref_image_0" in inputs
    assert not any(key in inputs for key in ("first_frame", "last_frame", "drive_audio", "final_audio"))
    assert not any(node["class_type"] in {"LoadAudio", "MiniMaxH3AudioWindowT8"} for node in built["prompt"].values())
    monkeypatch.setattr(nodes_director, "get_store", lambda: store)
    result = nodes_director.MiniMaxH3DirectorProjectT8.execute(json.dumps(project), shot["id"]).result
    assert result[0] == "女人在微笑"
    routes = inspect_d3_routes(project, shot["id"], store, node_ids=[], model_inventory={})
    assert routes["compile"]["ready"]
    assert project == original

    with pytest.raises(ValueError, match="第 2 镜.*尾帧"):
        build_director_generation_prompt(project, ends["id"], store)
    with pytest.raises(ValueError, match="有效的镜头"):
        compile_project(project, store, shot_id=str(uuid.uuid4()))


def test_d2b_binds_ends_and_ref2va_with_native_task_labels(tmp_path):
    store = _store(tmp_path)
    first, last, ref = _image(store, "first.png"), _image(store, "last.png"), _image(store, "ref.png")
    project = new_project()
    shot = project["doc"]["shots"][0]
    shot.update(mode="ends", first=first["id"], last=last["id"], tray=[first["id"], last["id"]], simplePrompt="A clean transition.")
    project["assets"] = [first, last]
    built = build_director_generation_prompt(project, shot["id"], store)
    assert built["prompt"]["5"]["inputs"]["task_type"] == "FL2VA"
    assert built["prompt"]["5"]["inputs"]["first_frame"] == ["14", 0]
    assert built["prompt"]["5"]["inputs"]["last_frame"] == ["15", 0]

    project = new_project()
    shot = project["doc"]["shots"][0]
    shot.update(mode="refs", refs=[ref["id"]], tray=[ref["id"]], simplePrompt="Keep the character identity.")
    project["assets"] = [ref]
    built = build_director_generation_prompt(project, shot["id"], store)
    assert built["prompt"]["5"]["inputs"]["task_type"] == "Ref2VA"
    assert built["prompt"]["5"]["inputs"]["ref_images.ref_image_0"] == ["14", 0]

    project = new_project()
    shot = project["doc"]["shots"][0]
    shot.update(mode="first", first=first["id"], refs=[ref["id"]],
                tray=[first["id"], ref["id"]], simplePrompt="A person turns toward the reference.")
    project["assets"] = [first, ref]
    built = build_director_generation_prompt(project, shot["id"], store)
    assert built["prompt"]["5"]["inputs"]["task_type"] == "Hybrid"
    assert "ref2va" in built["prompt"]["1"]["inputs"]["unet_name"].lower()
    assert built["prompt"]["5"]["inputs"]["first_frame"] == ["14", 0]
    assert built["prompt"]["5"]["inputs"]["ref_images.ref_image_0"] == ["15", 0]


def test_d2c_record_uses_windowed_drive_audio_as_delivery_track(tmp_path):
    store = _store(tmp_path)
    frame, audio = _image(store), _audio(store)
    project = new_project()
    shot = project["doc"]["shots"][0]
    shot.update(mode="first", first=frame["id"], tray=[frame["id"], audio["id"]],
                sound="record", audio=audio["id"], start=0, end=4, duration=4, manualDuration=4,
                simplePrompt="A person speaks naturally.")
    project["assets"] = [frame, audio]
    built = build_director_generation_prompt(project, shot["id"], store)
    graph = built["prompt"]
    assert built["recipe"].endswith("record")
    assert built["turbo_lora"] is None
    assert graph["6"]["inputs"]["steps"] == 8
    assert graph["5"]["inputs"]["task_type"] == "Hybrid"
    assert "ref2va" in graph["1"]["inputs"]["unet_name"].lower()
    assert graph["5"]["inputs"]["drive_audio"] == ["16", 0]
    assert graph["5"]["inputs"]["final_audio"] == ["16", 0]
    assert graph["5"]["inputs"]["length"] == ["16", 1]
    assert graph["11"]["inputs"]["audio"] == ["5", 2]
    assert graph["11"]["inputs"]["start_seconds"] == ["16", 2]
    assert graph["16"]["class_type"] == "MiniMaxH3AudioWindowT8"


def test_d2c_reference_voice_uses_audio_reference_not_delivery_audio(tmp_path):
    store = _store(tmp_path)
    audio = _audio(store)
    project = new_project()
    shot = project["doc"]["shots"][0]
    shot.update(sound="voice", audio=audio["id"], start=0, end=4,
                tray=[audio["id"]], simplePrompt="Say the new line.")
    project["assets"] = [audio]
    built = build_director_generation_prompt(project, shot["id"], store)
    graph = built["prompt"]
    assert graph["5"]["inputs"]["task_type"] == "Ref2VA"
    assert graph["5"]["inputs"]["ref_audios.ref_audio_0"] == ["14", 0]
    assert graph["11"]["inputs"]["audio"] == ["10", 1]


def test_d2b_reference_video_extracts_images_and_audio_components(tmp_path, monkeypatch):
    store = _store(tmp_path)
    import uuid as _uuid

    video_id = str(_uuid.uuid4())
    video = {
        "id": video_id,
        "name": "reference.mp4",
        "kind": "video",
        "width": 512,
        "height": 512,
        "duration": 3.0,
        "has_audio": True,
        "server_path": "t8_director/reference.mp4",
        "sha256": "video-sha",
        "size": 1,
    }
    monkeypatch.setattr(store, "asset", lambda asset_id, verify=True: video if asset_id == video_id else None)
    project = new_project()
    shot = project["doc"]["shots"][0]
    shot.update(mode="refs", refs=[video_id], tray=[video_id], simplePrompt="Use the motion reference.")
    project["assets"] = [video]
    built = build_director_generation_prompt(project, shot["id"], store)
    graph = built["prompt"]
    assert graph["14"]["class_type"] == "LoadVideo"
    assert graph["15"]["class_type"] == "GetVideoComponents"
    assert graph["15"]["inputs"]["video"] == ["14", 0]
    assert graph["5"]["inputs"]["ref_videos.ref_video_0"] == ["15", 0]
    assert graph["5"]["inputs"]["ref_video_audios.ref_video_audio_0"] == ["15", 1]


def _patch_director_models(monkeypatch):
    from h3_audio_t8_pkg import director_generation

    monkeypatch.setattr(
        director_generation,
        "_pick",
        lambda _folder, candidates, _label: candidates[0],
    )
    monkeypatch.setattr(director_generation, "_optional_turbo_lora", lambda: None)
    monkeypatch.setattr(
        director_generation,
        "_optional_semantic_bridge_model",
        lambda: "t8_compat/semantic_bridge.safetensors",
    )


def test_two_pass_stages_are_independent_and_only_high_is_decoded(tmp_path, monkeypatch):
    from h3_audio_t8_pkg import director_generation

    _patch_director_models(monkeypatch)
    monkeypatch.setattr(director_generation.folder_paths, "get_filename_list", lambda folder: {
        "loras": ["a.safetensors", "b.safetensors"],
        "latent_upscale_models": ["minimax_h3_latent_upscaler_3d_fp16.safetensors"],
    }.get(folder, []))
    monkeypatch.setattr(director_generation.folder_paths, "get_full_path", lambda folder, name: name)
    project = new_project()
    shot = project["doc"]["shots"][0]
    shot["simplePrompt"] = "A subject walks toward camera."
    project["doc"]["sampling"] = {
        "mode": "two_pass", "preset": "standard_4plus4_v1", "output_mp": 0.4,
        "low_loras": [{"id": "l1", "name": "a.safetensors", "strength": 0.8, "enabled": True},
                      {"id": "l2", "name": "a.safetensors", "strength": 0.4, "enabled": True}],
        "high_loras": [{"id": "h1", "name": "b.safetensors", "strength": -0.2, "enabled": True}],
    }
    result = build_director_generation_prompt(project, shot["id"], _store(tmp_path))
    graph = result["prompt"]
    loaders = [(key, node) for key, node in graph.items() if node["class_type"] == "MiniMaxH3LoRACompatibilityLoaderT8Advanced"]
    assert len(loaders) == 3
    assert loaders[0][1]["inputs"]["model"] == ["1", 0]
    assert loaders[1][1]["inputs"]["model"] == [loaders[0][0], 0]
    assert loaders[2][1]["inputs"]["model"] == ["1", 0]
    assert graph["6"]["inputs"]["model"] == [loaders[1][0], 0]
    assert graph["6"]["inputs"]["steps"] == 8
    assert graph["9"]["inputs"]["sigmas"][1] == 0
    upscale_id = next(k for k, v in graph.items() if v["class_type"] == "MiniMaxH3LearnedLatentUpscaleT8Advanced")
    assert graph[upscale_id]["inputs"]["av_latent"] == ["9", 1]
    high_condition_id = "5"
    assert graph[high_condition_id]["inputs"]["width"] == [upscale_id, 1]
    assert graph[high_condition_id]["inputs"]["height"] == [upscale_id, 2]
    mixer_id = next(k for k, v in graph.items() if v["class_type"] == "MiniMaxH3TwoPassDetailMixerT8Advanced")
    assert graph[mixer_id]["inputs"]["model"] == [loaders[2][0], 0]
    assert graph["10"]["inputs"]["av_latent"] == [result["sampling"]["high_output_node"], 0]


def test_sampling_v1_migration_and_per_shot_independence(tmp_path):
    from h3_audio_t8_pkg.director_project import validate_project, compile_project

    project = new_project()
    shot = project["doc"]["shots"][0]
    shot["simplePrompt"] = "A quiet portrait."
    project["version"] = 1
    project["doc"].pop("sampling")
    migrated = validate_project(project)
    assert migrated["version"] == 2
    assert migrated["doc"]["sampling"] == {"mode": "single"}
    assert compile_project(migrated, _store(tmp_path))["shots"][0]["sampling"] == {"mode": "single"}

    migrated["doc"]["sampling"] = {"mode": "two_pass", "output_mp": 0.4}
    shot = migrated["doc"]["shots"][0]
    shot["samplingInherit"] = False
    shot["sampling"] = {"mode": "single", "resolution_mp": 0.5, "lora_mode": "none", "loras": []}
    local = compile_project(migrated, _store(tmp_path))["shots"][0]
    assert local["sampling"]["mode"] == "single"
    assert local["canvas"]["width"] * local["canvas"]["height"] > 450_000
    shot["samplingInherit"] = True
    inherited = compile_project(migrated, _store(tmp_path))["shots"][0]
    assert inherited["sampling"]["mode"] == "two_pass"
    assert inherited["two_pass_canvas"]["low_width"] < inherited["canvas"]["width"]


def test_observed_pruned_ema_patch_error_is_warned_not_silently_accepted(tmp_path, monkeypatch):
    from h3_audio_t8_pkg import director_generation

    _patch_director_models(monkeypatch)
    monkeypatch.setattr(director_generation, "_pick_requested", lambda _folder, requested, candidates, _label: candidates[0] if requested == "auto" else requested)
    monkeypatch.setattr(director_generation.folder_paths, "get_filename_list", lambda folder: {
        "loras": ["minimax_h3_turbo_v4_step600_ema_comfyui_B.safetensors"],
        "latent_upscale_models": ["minimax_h3_latent_upscaler_3d_fp16.safetensors"],
    }.get(folder, []))
    monkeypatch.setattr(director_generation.folder_paths, "get_full_path", lambda folder, name: name)
    project = new_project()
    project["doc"]["shots"][0]["simplePrompt"] = "A quiet portrait."
    project["doc"]["generation"]["unet"] = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
    project["doc"]["sampling"] = {"mode": "two_pass", "output_mp": 0.4,
        "low_loras": [{"id": "low", "name": "minimax_h3_turbo_v4_step600_ema_comfyui_B.safetensors", "strength": 1, "enabled": True}],
        "high_loras": []}
    built = build_director_generation_prompt(project, project["current"], _store(tmp_path))
    assert any("AdaLN patch" in warning["message"] for warning in built["report"]["warnings"])


def test_declared_merged_acceleration_warns_and_skips_only_automatic_extra_lora(tmp_path, monkeypatch):
    from h3_audio_t8_pkg import director_generation
    from safetensors.numpy import save_file
    import numpy as np

    _patch_director_models(monkeypatch)
    unet = tmp_path / "arbitrary-base.safetensors"
    save_file({"dummy": np.zeros(1, dtype=np.float32)}, str(unet),
              metadata={"merged_loras": "Turbo step adapter"})
    monkeypatch.setattr(director_generation.folder_paths, "get_full_path",
                        lambda folder, _name: str(unet) if folder == "diffusion_models" else "selected.safetensors")
    monkeypatch.setattr(director_generation.folder_paths, "get_filename_list",
                        lambda folder: ["selected.safetensors"] if folder == "loras" else [])
    monkeypatch.setattr(director_generation, "_optional_turbo_lora", lambda: "selected.safetensors")
    project = new_project()
    project["doc"]["shots"][0]["simplePrompt"] = "A quiet portrait."
    built = build_director_generation_prompt(project, project["current"], _store(tmp_path))
    assert built["turbo_lora"] is None
    assert any("元数据声明" in row["message"] for row in built["report"]["warnings"])
    # Explicit user choice remains intact even when metadata declares a merge.
    project["doc"]["generation"].update(lora_mode="manual", loras=[
        {"name": "selected.safetensors", "enabled": True, "strength": 0.5}])
    chosen = build_director_generation_prompt(project, project["current"], _store(tmp_path))
    assert chosen["turbo_lora"] == [("selected.safetensors", 0.5)]


def test_hyperflow_single_ignores_inactive_high_draft_but_dual_validates_it():
    from h3_audio_t8_pkg.director_sampling_settings import normalize_sampling

    raw = {"mode": "hyperflow", "variant": "single8", "hyperflow_file": "hyperflow/weight.safetensors",
           "high_loras": [{"id": "draft", "name": "", "strength": 200, "enabled": True}]}
    assert normalize_sampling(raw)["high_loras"] == []
    assert raw["high_loras"][0]["strength"] == 200
    raw["variant"] = "continuous4plus4"
    with pytest.raises(ValueError, match="LoRA 强度"):
        normalize_sampling(raw)
    raw["variant"] = "continuous4plus4separate"
    with pytest.raises(ValueError, match="LoRA 强度"):
        normalize_sampling(raw)


@pytest.mark.parametrize("variant,recipe", [
    ("single8", "hyperflow8_single_v1"),
    ("continuous4plus4", "hyperflow8_continuous_split_exp_v1"),
    ("continuous4plus4separate", "hyperflow8_continuous_separate_stages_exp_v1"),
    ("upscale8plus4", "hyperflow8plus4_new_noise_upscale_exp_v1"),
    ("upscale4plus4", "hyperflow4plus4_partial_x0_upscale_exp_v1"),
])
def test_director_hyperflow_explicit_graph_keeps_legacy_recipe_separate(tmp_path, monkeypatch, variant, recipe):
    from h3_audio_t8_pkg import director_generation, director_hyperflow

    _patch_director_models(monkeypatch)
    monkeypatch.setattr(director_generation.folder_paths, "get_filename_list", lambda folder: {
        "loras": ["content.safetensors"],
        "diffusion_models": ["minimax_h3_fl2va_int8_convrot.safetensors"],
        "latent_upscale_models": ["minimax_h3_latent_upscaler_3d_fp16.safetensors"],
    }.get(folder, []))
    monkeypatch.setattr(director_generation.folder_paths, "get_full_path", lambda _folder, name: name)
    monkeypatch.setattr(director_hyperflow, "_resolve", lambda _selection: tmp_path / "hf.safetensors")
    (tmp_path / "hf.safetensors").write_bytes(b"probe-only")
    project = new_project()
    project["doc"]["shots"][0]["simplePrompt"] = "A quiet portrait."
    project["doc"]["generation"]["unet"] = "minimax_h3_fl2va_int8_convrot.safetensors"
    project["doc"]["sampling"] = {
        "mode": "hyperflow", "variant": variant,
        "hyperflow_file": "hyperflow/hf.safetensors", "output_mp": 0.4,
        "upscaler": "auto", "low_loras": [{"id": "l1", "name": "content.safetensors", "enabled": True, "strength": 0.4}],
        "high_loras": [{"id": "h1", "name": "content.safetensors", "enabled": True, "strength": 0.2}],
    }
    built = build_director_generation_prompt(project, project["current"], _store(tmp_path))
    graph = built["prompt"]
    types = [node["class_type"] for node in graph.values()]
    assert built["sampling"]["recipe"] == recipe
    expected_nfe = 12 if variant == "upscale8plus4" else 8
    assert built["sampling"]["total_nfe"] == expected_nfe
    assert len(built["sampling"]["trained_grid_contract"]["raw_sigmas"]) == 9
    assert len(built["sampling"]["trained_grid_contract_sha256"]) == 64
    assert types.count("MiniMaxH3HyperFlowLoaderT8Advanced") == (1 if variant == "single8" else 2)
    assert types.count("MiniMaxH3LoRACompatibilityLoaderT8Advanced") == (1 if variant == "single8" else 2)
    if variant in {"upscale8plus4", "upscale4plus4"}:
        assert graph["10"]["inputs"]["av_latent"] != ["9", 0]
    if variant == "continuous4plus4":
        assert graph["9"]["class_type"] == "MiniMaxH3HyperFlowSplitT8Advanced"
    if variant == "continuous4plus4separate":
        assert graph["9"]["class_type"] == "MiniMaxH3HyperFlowHeadStageEXPT8"
        assert "MiniMaxH3HyperFlowSplitT8Advanced" not in types
        tails = [(key, node) for key, node in graph.items()
                 if node["class_type"] == "MiniMaxH3HyperFlowTailStageEXPT8"]
        assert len(tails) == 1
        tail_id, tail = tails[0]
        head_audit_id = next(key for key, node in graph.items()
                             if node["class_type"] == "MiniMaxH3HyperFlowHeadEffectsAuditEXPT8")
        tail_audit_id = next(key for key, node in graph.items()
                             if node["class_type"] == "MiniMaxH3HyperFlowTailEffectsAuditEXPT8")
        assert graph[head_audit_id]["inputs"]["continuous_boundary"] == ["9", 0]
        assert tail["inputs"]["continuous_boundary"] == [head_audit_id, 0]
        assert tail["inputs"]["positive"][0] == tail["inputs"]["negative"][0]
        assert (tail["inputs"]["positive"][1], tail["inputs"]["negative"][1]) == (1, 2)
        assert tail["inputs"]["positive"] != graph["9"]["inputs"]["positive"]
        assert graph[tail_audit_id]["inputs"]["completed_result"] == [tail_id, 2]
        assert graph["10"]["inputs"]["av_latent"] == [tail_audit_id, 1]
        assert graph["9"]["inputs"]["noise"] == ["8", 0]
        assert [node["class_type"] for node in graph.values()].count("MiniMaxH3AudioConditioningT8") == 2
        assert types.count("MiniMaxH3HyperFlowHeadEffectsBindEXPT8") == 1
        assert types.count("MiniMaxH3HyperFlowTailEffectsBindEXPT8") == 1
        assert types.count("MiniMaxH3StageEAVConfigEXPT8") == 2
        assert types.count("MiniMaxH3StageEAVApplyEXPT8") == 2
    if variant in {"upscale8plus4", "upscale4plus4"}:
        assert "MiniMaxH3LearnedLatentUpscaleT8Advanced" in types
        assert ("MiniMaxH3HyperFlowPartialRefineSamplerT8Advanced" if variant == "upscale4plus4"
                else "MiniMaxH3HyperFlowRefineSamplerT8Advanced") in types
    if variant == "upscale4plus4":
        assert "MiniMaxH3HyperFlowHeadPlanT8Advanced" in types
        assert "MiniMaxH3HyperFlowCoarseSamplerT8Advanced" in types
        upscaler = next(node for node in graph.values()
                        if node["class_type"] == "MiniMaxH3LearnedLatentUpscaleT8Advanced")
        assert upscaler["inputs"]["av_latent"] == ["9", 1]


@pytest.mark.parametrize("relay", [False, True])
@pytest.mark.parametrize("checkpoint_mode", ["off", "save", "resume_tail"])
def test_director_hyperflow_separate_head_tail_validates_in_core(tmp_path, monkeypatch, relay, checkpoint_mode):
    import json
    import os
    from pathlib import Path
    import subprocess
    import sys

    from h3_audio_t8_pkg import director_generation, director_hyperflow
    from h3_audio_t8_pkg.modular_sampling import hyperflow_storage

    _patch_director_models(monkeypatch)
    monkeypatch.setattr(director_generation.folder_paths, "get_filename_list", lambda folder: {
        "diffusion_models": ["minimax_h3_fl2va_int8_convrot.safetensors"],
        "vae": ["minimax_h3_video_vae_fp16.safetensors", "minimax_h3_audio_vae_fp32.safetensors"],
        "clip": ["qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"],
        "text_encoders": ["qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"],
        "hyperflow": ["hf.safetensors"],
    }.get(folder, []))
    monkeypatch.setattr(director_generation.folder_paths, "get_full_path", lambda _folder, name: name)
    weight = tmp_path / "hf.safetensors"
    weight.write_bytes(b"preflight-only")
    monkeypatch.setattr(director_hyperflow, "_resolve", lambda _selection: weight)
    monkeypatch.setattr(director_hyperflow.folder_paths, "get_output_directory", lambda: str(tmp_path))
    project = new_project()
    project["doc"]["shots"][0]["simplePrompt"] = "A continuous scene."
    project["doc"]["generation"]["unet"] = "minimax_h3_fl2va_int8_convrot.safetensors"
    project["doc"]["sampling"] = {"mode": "hyperflow", "variant": "continuous4plus4separate",
                                  "hyperflow_file": "hyperflow/hf.safetensors", "output_mp": 0.4}
    if checkpoint_mode == "save":
        project["doc"]["sampling"]["stage_checkpoint"] = {"mode": "save"}
    elif checkpoint_mode == "resume_tail":
        from test_modular_hyperflow import inputs, head

        low, _, source, positive = inputs(monkeypatch)
        boundary = head(low, source, positive)
        root = tmp_path / "MiniMaxH3" / "hyperflow_stage_artifacts"
        path, digest, _ = hyperflow_storage.save_boundary(boundary, root)
        project["doc"]["sampling"]["stage_checkpoint"] = {
            "mode": "resume_tail", "artifact_path": path, "artifact_sha256": digest}
    if relay:
        project["doc"]["d3"] = {"prompt_relay": {"enabled": True, "execution_mode": "apply_exp"}}
        project["doc"]["sampling"]["stage_relay"] = {
            "head": {"mode": "custom", "global_prompt": "Head stage portrait"},
            "tail": {"mode": "custom", "global_prompt": "Tail stage portrait"},
        }
    graph = build_director_generation_prompt(project, project["current"], _store(tmp_path))["prompt"]
    root = Path(__file__).resolve().parents[1]
    core = root.parents[1]
    script = """
import asyncio
import importlib.util
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
sys.path.insert(0, str(root.parents[1]))
from comfy.cli_args import args
args.cpu = True
import folder_paths
folder_paths.get_filename_list = lambda folder: {
    'diffusion_models': ['minimax_h3_fl2va_int8_convrot.safetensors'],
    'vae': ['minimax_h3_video_vae_fp16.safetensors', 'minimax_h3_audio_vae_fp32.safetensors'],
    'clip': ['qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors'],
    'text_encoders': ['qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors'],
    'hyperflow': ['hf.safetensors'],
}.get(folder, [])
folder_paths.get_full_path = lambda _folder, name: name
spec = importlib.util.spec_from_file_location('h3_audio_t8_pkg', root / '__init__.py',
                                              submodule_search_locations=[str(root)])
package = importlib.util.module_from_spec(spec)
sys.modules['h3_audio_t8_pkg'] = package
spec.loader.exec_module(package)
import execution
import nodes
from comfy_extras import nodes_custom_sampler
graph = json.load(sys.stdin)
classes = asyncio.run(package.comfy_entrypoint().get_node_list())
required = {item['class_type'] for item in graph.values()}
for cls in classes:
    if cls.__name__ in required:
        nodes.NODE_CLASS_MAPPINGS[cls.__name__] = cls
for cls in (nodes_custom_sampler.BasicGuider, nodes_custom_sampler.RandomNoise,
            nodes_custom_sampler.SamplerCustomAdvanced):
    nodes.NODE_CLASS_MAPPINGS[cls.__name__] = cls
valid = asyncio.run(execution.validate_prompt('director-hyperflow-separate', graph, None))
print(json.dumps({'valid': valid[0], 'errors': valid[1]}, default=str))
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(core) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run([sys.executable, "-c", script, str(root)], input=json.dumps(graph),
                            text=True, capture_output=True, env=env, cwd=root, timeout=60, check=True)
    validation = json.loads(result.stdout.splitlines()[-1])
    assert validation["valid"], validation["errors"]


def test_director_hyperflow_separate_clones_semantic_bridge_per_stage(tmp_path, monkeypatch):
    from h3_audio_t8_pkg import director_generation, director_hyperflow

    _patch_director_models(monkeypatch)
    monkeypatch.setattr(director_hyperflow, "_resolve", lambda _selection: tmp_path / "hf.safetensors")
    (tmp_path / "hf.safetensors").write_bytes(b"preflight-only")
    project = new_project()
    project["doc"]["shots"][0]["simplePrompt"] = "A continuous scene."
    project["doc"]["sampling"] = {"mode": "hyperflow", "variant": "continuous4plus4separate",
                                  "hyperflow_file": "hyperflow/hf.safetensors", "output_mp": 0.4}
    project["doc"]["d3"] = {"semantic_bridge": {"enabled": True, "model_name": "bridge.safetensors"}}
    monkeypatch.setattr(director_generation, "_optional_semantic_bridge_model", lambda: "bridge.safetensors")
    graph = build_director_generation_prompt(project, project["current"], _store(tmp_path))["prompt"]
    bridge_ids = {key for key, node in graph.items()
                  if node["class_type"] == "MiniMaxH3SemanticBridgeApplyT8"}
    assert len(bridge_ids) == 2
    head_bind = next(node for node in graph.values()
                     if node["class_type"] == "MiniMaxH3HyperFlowHeadEffectsBindEXPT8")
    tail_bind = next(node for node in graph.values()
                     if node["class_type"] == "MiniMaxH3HyperFlowTailEffectsBindEXPT8")
    head_positive = head_bind["inputs"]["positive"]
    tail_positive = tail_bind["inputs"]["positive"]
    assert head_positive[0] in bridge_ids and tail_positive[0] in bridge_ids
    assert head_positive != tail_positive
    assert graph[tail_positive[0]]["inputs"]["conditioning"][0] != "5"


def test_director_hyperflow_stage_checkpoint_validation_is_opt_in():
    from h3_audio_t8_pkg.director_sampling_settings import normalize_sampling

    raw = {"mode": "hyperflow", "variant": "continuous4plus4separate",
           "hyperflow_file": "hyperflow/hf.safetensors"}
    assert normalize_sampling(raw)["stage_checkpoint"] == {"mode": "off"}
    assert normalize_sampling({**raw, "stage_checkpoint": {"mode": "save"}})["stage_checkpoint"] == {"mode": "save"}
    for checkpoint in ({"mode": "unknown"}, {"mode": "resume_tail"},
                       {"mode": "resume_tail", "artifact_path": "head.safetensors", "artifact_sha256": "x" * 64}):
        with pytest.raises(ValueError, match="HEAD|TAIL"):
            normalize_sampling({**raw, "stage_checkpoint": checkpoint})
    old = {**raw, "variant": "continuous4plus4", "stage_checkpoint": {"mode": "unknown"}}
    assert "stage_checkpoint" not in normalize_sampling(old)
    old["stage_relay"] = {"head": {"mode": "invalid"}}
    assert "stage_relay" not in normalize_sampling(old)
    resume = {**raw, "stage_checkpoint": {"mode": "resume_tail", "artifact_path": "frozen/hyperflow-head.safetensors",
                                            "artifact_sha256": "a" * 64},
              "stage_eav": {"head": {"mode": "invalid"}, "tail": {"mode": "report_only", "tau": 2.0}},
              "stage_relay": {"head": {"mode": "invalid"},
                              "tail": {"mode": "custom", "global_prompt": "New TAIL"}}}
    normalized = normalize_sampling(resume)
    assert normalized["stage_eav"]["head"]["mode"] == "disabled"
    assert normalized["stage_eav"]["tail"]["mode"] == "report_only"
    assert normalized["stage_relay"]["head"] == {"mode": "inherit"}
    assert normalized["stage_relay"]["tail"]["global_prompt"] == "New TAIL"


def test_director_hyperflow_save_and_exact_tail_resume_graph(tmp_path, monkeypatch):
    from h3_audio_t8_pkg import director_hyperflow
    from h3_audio_t8_pkg.modular_sampling import hyperflow_storage
    from test_modular_hyperflow import inputs, head

    _patch_director_models(monkeypatch)
    weight = tmp_path / "hf.safetensors"
    weight.write_bytes(b"preflight-only")
    monkeypatch.setattr(director_hyperflow, "_resolve", lambda _selection: weight)
    monkeypatch.setattr(director_hyperflow.folder_paths, "get_output_directory", lambda: str(tmp_path))
    project = new_project()
    project["doc"]["shots"][0]["simplePrompt"] = "A continuous scene."
    sampling = project["doc"]["sampling"] = {
        "mode": "hyperflow", "variant": "continuous4plus4separate",
        "hyperflow_file": "hyperflow/hf.safetensors", "output_mp": 0.4,
        "stage_checkpoint": {"mode": "save"},
    }
    save_graph = build_director_generation_prompt(project, project["current"], _store(tmp_path))["prompt"]
    save_id = next(key for key, node in save_graph.items()
                   if node["class_type"] == "MiniMaxH3HyperFlowHeadSaveEXPT8")
    head_audit_id = next(key for key, node in save_graph.items()
                         if node["class_type"] == "MiniMaxH3HyperFlowHeadEffectsAuditEXPT8")
    tail_id = next(key for key, node in save_graph.items()
                   if node["class_type"] == "MiniMaxH3HyperFlowTailStageEXPT8")
    assert save_graph[save_id]["inputs"]["continuous_boundary"] == [head_audit_id, 0]
    assert save_graph[tail_id]["inputs"]["continuous_boundary"] == [save_id, 0]
    assert save_graph["12"]["class_type"] == "MiniMaxH3SafeAVSaveT8Advanced"

    low, _, source, positive = inputs(monkeypatch)
    boundary = head(low, source, positive)
    root = tmp_path / "MiniMaxH3" / "hyperflow_stage_artifacts"
    path, digest, report = hyperflow_storage.save_boundary(boundary, root)
    assert '"portable_identity":true' in report
    sampling["stage_checkpoint"] = {"mode": "resume_tail", "artifact_path": path,
                                    "artifact_sha256": digest.upper()}
    sampling["low_loras"] = [{"enabled": True, "name": "deleted-low-draft.safetensors", "strength": 9}]
    resume = build_director_generation_prompt(project, project["current"], _store(tmp_path))
    graph = resume["prompt"]
    kinds = [node["class_type"] for node in graph.values()]
    assert kinds.count("MiniMaxH3HyperFlowHeadLoadEXPT8") == 1
    assert "MiniMaxH3HyperFlowHeadStageEXPT8" not in kinds
    assert "MiniMaxH3HyperFlowHeadEffectsBindEXPT8" not in kinds
    assert "MiniMaxH3HyperFlowHeadSaveEXPT8" not in kinds
    assert "RandomNoise" not in kinds
    load_id = next(key for key, node in graph.items()
                   if node["class_type"] == "MiniMaxH3HyperFlowHeadLoadEXPT8")
    tail_id = next(key for key, node in graph.items()
                   if node["class_type"] == "MiniMaxH3HyperFlowTailStageEXPT8")
    assert graph[tail_id]["inputs"]["continuous_boundary"] == [load_id, 0]
    assert graph[load_id]["inputs"]["artifact_sha256"] == digest
    assert resume["sampling"]["total_nfe"] == 4
    sampling["stage_checkpoint"]["artifact_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA256"):
        build_director_generation_prompt(project, project["current"], _store(tmp_path))


def test_director_hyperflow_separate_stage_eav_and_relay_are_independent(tmp_path, monkeypatch):
    from h3_audio_t8_pkg import director_hyperflow
    from h3_audio_t8_pkg.director_sampling_settings import normalize_sampling

    _patch_director_models(monkeypatch)
    monkeypatch.setattr(director_hyperflow, "_resolve", lambda _selection: tmp_path / "hf.safetensors")
    (tmp_path / "hf.safetensors").write_bytes(b"preflight-only")
    project = new_project()
    project["doc"]["shots"][0]["simplePrompt"] = "A continuous scene."
    project["doc"]["sampling"] = {"mode": "hyperflow", "variant": "continuous4plus4separate",
        "hyperflow_file": "hyperflow/hf.safetensors", "output_mp": 0.4,
        "stage_eav": {"head": {"mode": "report_only", "tau": 2.0},
                      "tail": {"mode": "apply_exp", "tau": 4.0}},
        "stage_relay": {
            "head": {"mode": "custom", "global_prompt": "First stage city",
                     "local_prompts": "A person starts walking\nCamera follows", "timing_mode": "auto_equal"},
            "tail": {"mode": "custom", "global_prompt": "Second stage city",
                     "local_prompts": "A person keeps walking\nCamera closes in", "timing_mode": "auto_equal"},
        }}
    project["doc"]["d3"] = {"prompt_relay": {"enabled": True, "execution_mode": "apply_exp"}}
    graph = build_director_generation_prompt(project, project["current"], _store(tmp_path))["prompt"]
    def by_type(kind):
        return [(key, node) for key, node in graph.items() if node["class_type"] == kind]
    plans = by_type("MiniMaxH3PromptRelayPlanT8Advanced")
    relays = by_type("MiniMaxH3PromptRelayConditioningT8Advanced")
    configs = by_type("MiniMaxH3StageEAVConfigEXPT8")
    applies = by_type("MiniMaxH3StageEAVApplyEXPT8")
    assert len(plans) == len(relays) == len(configs) == len(applies) == 2
    assert {node["inputs"]["mode"] for _, node in configs} == {"report_only", "apply_exp"}
    assert {node["inputs"]["tau"] for _, node in configs} == {2.0, 4.0}
    assert relays[0][1]["inputs"]["prompt_relay_plan"] != relays[1][1]["inputs"]["prompt_relay_plan"]
    assert relays[0][1]["inputs"]["model"] != relays[1][1]["inputs"]["model"]
    assert {node["inputs"]["global_prompt"] for _, node in plans} == {
        "First stage city", "Second stage city"}
    assert {node["inputs"]["local_prompts"] for _, node in plans} == {
        "A person starts walking\nCamera follows", "A person keeps walking\nCamera closes in"}
    assert graph["9"]["inputs"]["noise"] == ["8", 0]
    for bad in ({"mode": "unknown"}, {"tau": float("nan")},
                {"start_video_progress": .9, "end_video_progress": .2},
                {"max_workspace_mib": 513}):
        raw = {**project["doc"]["sampling"], "stage_eav": {"head": bad}}
        with pytest.raises(ValueError, match="EAV"):
            normalize_sampling(raw)
    for bad in ({"mode": "custom", "global_prompt": " "},
                {"mode": "custom", "global_prompt": "Valid", "timing_mode": "unknown"},
                {"mode": "custom", "global_prompt": "Valid", "time_ranges": "0-23"}):
        raw = {**project["doc"]["sampling"], "stage_relay": {"tail": bad}}
        with pytest.raises(ValueError, match="Relay"):
            normalize_sampling(raw)
    invalid_timeline = deepcopy(project)
    invalid_timeline["doc"]["sampling"]["stage_relay"]["tail"].update(
        timing_mode="frames", time_ranges="0-23")
    with pytest.raises(ValueError, match="one time range per local prompt"):
        build_director_generation_prompt(invalid_timeline, invalid_timeline["current"], _store(tmp_path))


@pytest.mark.parametrize("relay", [False, True])
def test_director_hyperflow_separate_real_core_tail_eav_edit_keeps_head_cached(tmp_path, monkeypatch, relay):
    import asyncio
    from pathlib import Path
    from types import SimpleNamespace

    import torch
    import comfy.nested_tensor
    from comfy_api.latest import io

    from h3_audio_t8_pkg import director_hyperflow
    from h3_audio_t8_pkg.modular_sampling import hyperflow as stages
    from h3_audio_t8_pkg.modular_sampling import hyperflow_nodes, hyperflow_effect_nodes
    from h3_audio_t8_pkg.modular_sampling import hyperflow_storage, hyperflow_storage_nodes
    from h3_audio_t8_pkg.modular_sampling import nodes as effect_nodes
    from test_hyperflow_advanced import _tiny_native_model
    from test_modular_hyperflow_core_cache import TinyHyperBase, SyntheticHyperLoader, HyperResultSink
    from test_modular_progressive_core_cache import TinyCondition
    from test_progressive_sampling_runtime import tiny_model
    from test_progressive_relay import paired
    from tools import build_modular_hyperflow_workflow as builder

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3]))
    import execution
    import nodes
    from comfy_extras.nodes_custom_sampler import RandomNoise

    _patch_director_models(monkeypatch)
    monkeypatch.setattr(director_hyperflow, "_resolve", lambda _selection: tmp_path / "hf.safetensors")
    monkeypatch.setattr(director_hyperflow.folder_paths, "get_output_directory", lambda: str(tmp_path))
    (tmp_path / "hf.safetensors").write_bytes(b"preflight-only")
    project = new_project()
    project["doc"]["shots"][0]["simplePrompt"] = "A continuous scene."
    project["doc"]["sampling"] = {"mode": "hyperflow", "variant": "continuous4plus4separate",
        "hyperflow_file": "hyperflow/hf.safetensors", "output_mp": 0.4,
        "stage_checkpoint": {"mode": "save"},
        "stage_eav": {"head": {"mode": "report_only", "tau": .2, "g_hard_limit": 3.},
                      "tail": {"mode": "report_only", "tau": .2, "g_hard_limit": 3.}}}
    if relay:
        project["doc"]["d3"] = {"prompt_relay": {"enabled": True, "execution_mode": "apply_exp"}}
        project["doc"]["sampling"]["stage_relay"] = {
            "head": {"mode": "custom", "global_prompt": "Head stage portrait"},
            "tail": {"mode": "custom", "global_prompt": "Tail stage portrait"},
        }
    graph = build_director_generation_prompt(project, project["current"], _store(tmp_path))["prompt"]

    class TinyDirectorRelay(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(node_id=cls.__name__, inputs=[io.Model.Input("model"),
                io.Combo.Input("route", options=["joint_av_exp", "video_only_paper"]),
                io.Float.Input("prompt", default=0.)],
                outputs=[io.Model.Output(), io.Conditioning.Output(), io.Latent.Output()])

        @classmethod
        def execute(cls, model, route, prompt):
            selected, positive, _ = paired(model, query_route=route)
            positive[0][0].add_(prompt)
            source = {"samples": comfy.nested_tensor.NestedTensor((
                torch.zeros(1, 24, 2, 4, 8), torch.zeros(1, 32, 2, 8)))}
            return io.NodeOutput(selected, positive, source)

    _, diffusion = _tiny_native_model(monkeypatch)
    base = tiny_model()
    base.model.diffusion_model = diffusion
    monkeypatch.setattr(TinyHyperBase, "execute", classmethod(lambda cls: io.NodeOutput(base)))
    graph["1"] = {"class_type": "TinyHyperBase", "inputs": {}}
    for key, node in list(graph.items()):
        if node["class_type"] == "MiniMaxH3HyperFlowLoaderT8Advanced":
            graph[key] = {"class_type": "SyntheticHyperLoader", "inputs": {
                "model": node["inputs"]["model"], "patch": 0.}}
        elif node["class_type"] == "MiniMaxH3AudioConditioningT8":
            graph[key] = {"class_type": "TinyCondition", "inputs": {
                "width": 64, "height": 64, "prompt": 0.}}
        elif node["class_type"] == "MiniMaxH3PromptRelayConditioningT8Advanced":
            plan = graph[node["inputs"]["prompt_relay_plan"][0]]["inputs"]
            graph[key] = {"class_type": "TinyDirectorRelay", "inputs": {
                "model": node["inputs"]["model"], "route": "joint_av_exp",
                "prompt": .1 if plan["global_prompt"].startswith("Head stage") else .2}}
    audit_id = next(key for key, node in graph.items()
                    if node["class_type"] == "MiniMaxH3HyperFlowTailEffectsAuditEXPT8")
    tail_bind_id = next(key for key, node in graph.items()
                        if node["class_type"] == "MiniMaxH3HyperFlowTailEffectsBindEXPT8")
    tail_apply_id = next(key for key, node in graph.items()
                         if node["class_type"] == "MiniMaxH3StageEAVApplyEXPT8"
                         and node["inputs"]["model"] == [tail_bind_id, 0])
    tail_config_id = graph[tail_apply_id]["inputs"]["eav_config"][0]
    graph["999"] = {"class_type": "HyperResultSink", "inputs": {"result": [audit_id, 0]}}
    graph = builder.common.prune(graph, ["999"])
    for cls in (*hyperflow_nodes.NODES, *hyperflow_effect_nodes.NODES, *hyperflow_storage_nodes.NODES, TinyHyperBase,
                SyntheticHyperLoader, TinyCondition, TinyDirectorRelay, HyperResultSink, RandomNoise,
                effect_nodes.MiniMaxH3StageEAVApplyEXPT8, effect_nodes.MiniMaxH3StageEAVConfigEXPT8):
        monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, cls.__name__, cls)
    assert asyncio.run(execution.validate_prompt("director-stage-effects", graph, None))[0]
    server = SimpleNamespace(client_id=None, last_node_id=None, sockets_metadata={}, send_sync=lambda *a, **k: None)
    executor = execution.PromptExecutor(server, cache_args={"ram": 0., "ram_inactive": 0.},
                                        asset_manager=SimpleNamespace(enabled=False))
    calls, receipts = [], []
    original = stages._run

    def counted(model, plan, *args, **kwargs):
        calls.append("tail" if plan.start_interval else "head")
        output = original(model, plan, *args, **kwargs)
        receipts.append(output[3]["effects"])
        return output

    monkeypatch.setattr(stages, "_run", counted)

    def run(label):
        executor.execute(deepcopy(graph), label, execute_outputs=["999"])
        assert executor.success, executor.status_messages

    run("first")
    assert calls == ["head", "tail"]
    root = tmp_path / "MiniMaxH3" / "hyperflow_stage_artifacts"
    head_file = next(root.rglob("hyperflow-head.safetensors"))
    frozen_head_sha = hyperflow_storage.file_sha(head_file)
    assert all(receipt["composition_verified"] for receipt in receipts)
    if relay:
        assert all(receipt["relay"]["routed_attention_calls"] > 0 for receipt in receipts)
    calls.clear()
    run("same")
    assert calls == []
    graph[tail_config_id]["inputs"]["mode"] = "apply_exp"
    run("tail-eav-only")
    assert calls == ["tail"]
    assert receipts[-1]["eav"]["status"] == "observed_apply_exp"
    if relay:
        calls.clear()
        tail_relay_id = next(key for key, node in graph.items()
                             if node["class_type"] == "TinyDirectorRelay"
                             and node["inputs"]["prompt"] == .2)
        graph[tail_relay_id]["inputs"]["prompt"] = .3
        run("tail-relay-text-only")
        assert calls == ["tail"]
        assert receipts[-1]["relay"]["routed_attention_calls"] > 0
    assert hyperflow_storage.file_sha(head_file) == frozen_head_sha
    assert len(list(root.rglob("hyperflow-head.safetensors"))) == 1
    path = head_file.relative_to(root).as_posix()
    digest = hyperflow_storage.file_sha(head_file)
    project["doc"]["sampling"]["stage_checkpoint"] = {
        "mode": "resume_tail", "artifact_path": path, "artifact_sha256": digest}
    resumed = build_director_generation_prompt(project, project["current"], _store(tmp_path))["prompt"]
    for key, node in list(resumed.items()):
        if node["class_type"] == "MiniMaxH3HyperFlowLoaderT8Advanced":
            resumed[key] = {"class_type": "SyntheticHyperLoader", "inputs": {
                "model": node["inputs"]["model"], "patch": 0.}}
        elif node["class_type"] == "MiniMaxH3AudioConditioningT8":
            resumed[key] = {"class_type": "TinyCondition", "inputs": {
                "width": 64, "height": 64, "prompt": 0.}}
        elif node["class_type"] == "MiniMaxH3PromptRelayConditioningT8Advanced":
            plan = resumed[node["inputs"]["prompt_relay_plan"][0]]["inputs"]
            resumed[key] = {"class_type": "TinyDirectorRelay", "inputs": {
                "model": node["inputs"]["model"], "route": "joint_av_exp",
                "prompt": .1 if plan["global_prompt"].startswith("Head stage") else .2}}
    resumed["1"] = {"class_type": "TinyHyperBase", "inputs": {}}
    audit_id = next(key for key, node in resumed.items()
                    if node["class_type"] == "MiniMaxH3HyperFlowTailEffectsAuditEXPT8")
    resumed["999"] = {"class_type": "HyperResultSink", "inputs": {"result": [audit_id, 0]}}
    resumed = builder.common.prune(resumed, ["999"])
    assert asyncio.run(execution.validate_prompt("director-resume-tail", resumed, None))[0]
    calls.clear()
    cold_executor = execution.PromptExecutor(server, cache_args={"ram": 0., "ram_inactive": 0.},
                                             asset_manager=SimpleNamespace(enabled=False))
    cold_executor.execute(deepcopy(resumed), "director-cold-tail", execute_outputs=["999"])
    assert cold_executor.success, cold_executor.status_messages
    assert calls == ["tail"]
    assert receipts[-1]["composition_verified"]
    assert not torch.cuda.is_initialized()


def test_hyperflow_stage_memory_and_bridge_are_preserved_and_high_seed_wraps(tmp_path, monkeypatch):
    from h3_audio_t8_pkg import director_generation, director_hyperflow

    _patch_director_models(monkeypatch)
    monkeypatch.setattr(director_generation.folder_paths, "get_filename_list", lambda folder: {
        "diffusion_models": ["minimax_h3_fl2va_int8_convrot.safetensors"],
        "latent_upscale_models": ["minimax_h3_latent_upscaler_3d_fp16.safetensors"],
    }.get(folder, []))
    monkeypatch.setattr(director_generation.folder_paths, "get_full_path", lambda _folder, name: name)
    monkeypatch.setattr(director_hyperflow, "_resolve", lambda _selection: tmp_path / "hf.safetensors")
    (tmp_path / "hf.safetensors").write_bytes(b"probe-only")
    project = new_project()
    project["doc"]["shots"][0]["simplePrompt"] = "A quiet portrait."
    project["doc"]["generation"]["unet"] = "minimax_h3_fl2va_int8_convrot.safetensors"
    project["doc"]["sampling"] = {"mode": "hyperflow", "variant": "upscale8plus4",
                                  "hyperflow_file": "hyperflow/hf.safetensors", "output_mp": 0.4}
    project["doc"]["d3"] = {"semantic_bridge": {"enabled": True},
                             "memory": {"low_vram": True, "chunk_ffn": True}}
    built = build_director_generation_prompt(project, project["current"], _store(tmp_path), seed=2**64 - 1)
    graph = built["prompt"]
    kinds = [node["class_type"] for node in graph.values()]
    assert kinds.count("MiniMaxH3LowVRAMAttentionT8Advanced") >= 2
    assert kinds.count("MiniMaxH3ChunkFeedForwardT8Advanced") >= 2
    assert kinds.count("MiniMaxH3SemanticBridgeApplyT8") == 2
    assert any(node["class_type"] == "RandomNoise" and node["inputs"]["noise_seed"] == 0
               for node in graph.values())
    assert any("叠加尚未" in warning["message"] for warning in built["report"]["warnings"])


@pytest.mark.parametrize("copied_file", [False, True])
def test_hyperflow_source_cannot_be_reused_as_stage_content_lora(tmp_path, monkeypatch, copied_file):
    import torch
    from safetensors.torch import save_file
    from h3_audio_t8_pkg import director_generation, director_hyperflow

    _patch_director_models(monkeypatch)
    original = tmp_path / "hyperflow-original.safetensors"
    save_file({"probe.weight": torch.zeros(1)}, str(original), metadata={"hyperflow": "true"})
    content = tmp_path / "content-copy.safetensors" if copied_file else original
    if copied_file:
        content.write_bytes(original.read_bytes())
    monkeypatch.setattr(director_generation.folder_paths, "get_filename_list", lambda folder: {
        "loras": ["content.safetensors"],
        "diffusion_models": ["minimax_h3_fl2va_int8_convrot.safetensors"],
    }.get(folder, []))
    monkeypatch.setattr(director_generation.folder_paths, "get_full_path",
                        lambda folder, name: str(content) if folder == "loras" else name)
    monkeypatch.setattr(director_hyperflow, "_resolve", lambda _selection: original)
    project = new_project()
    project["doc"]["shots"][0]["simplePrompt"] = "A quiet portrait."
    project["doc"]["generation"]["unet"] = "minimax_h3_fl2va_int8_convrot.safetensors"
    project["doc"]["sampling"] = {
        "mode": "hyperflow", "variant": "single8", "hyperflow_file": "hyperflow/original.safetensors",
        "low_loras": [{"id": "l1", "name": "content.safetensors", "enabled": True, "strength": 0.4}],
    }
    with pytest.raises(ValueError, match="专用 HyperFlow 权重栏"):
        build_director_generation_prompt(project, project["current"], _store(tmp_path))


def test_hyperflow_dual_compiler_crash_preflight_is_explicit(tmp_path, monkeypatch):
    import comfy.cli_args
    import comfy.memory_management

    _patch_director_models(monkeypatch)
    project = new_project()
    project["doc"]["shots"][0]["simplePrompt"] = "A quiet portrait."
    project["doc"]["sampling"] = {"mode": "hyperflow", "variant": "continuous4plus4",
                                  "hyperflow_file": "hyperflow/weight.safetensors"}
    monkeypatch.setattr(comfy.memory_management, "aimdo_enabled", True)
    monkeypatch.setattr(comfy.cli_args.args, "disable_comfy_compiler", False)
    with pytest.raises(ValueError, match="--disable-comfy-compiler"):
        build_director_generation_prompt(project, project["current"], _store(tmp_path))


def test_two_pass_rebuilds_first_frame_from_source_and_keeps_recording(tmp_path, monkeypatch):
    from h3_audio_t8_pkg import director_generation

    _patch_director_models(monkeypatch)
    monkeypatch.setattr(director_generation.folder_paths, "get_filename_list", lambda folder: {
        "latent_upscale_models": ["minimax_h3_latent_upscaler_3d_fp16.safetensors"],
    }.get(folder, []))
    monkeypatch.setattr(director_generation.folder_paths, "get_full_path", lambda folder, name: name)
    store = _store(tmp_path)
    first, audio = _image(store), _audio(store)
    project = new_project()
    shot = project["doc"]["shots"][0]
    shot.update(mode="first", first=first["id"], tray=[first["id"], audio["id"]],
                sound="record", audio=audio["id"], start=0, end=4,
                simplePrompt="The woman speaks while looking at camera.")
    project["assets"] = [first, audio]
    project["doc"]["sampling"] = {"mode": "two_pass", "output_mp": 0.4}
    built = build_director_generation_prompt(project, shot["id"], store)
    graph = built["prompt"]
    conditions = [(key, node) for key, node in graph.items() if node["class_type"] == "MiniMaxH3AudioConditioningT8"]
    assert len(conditions) == 2
    high = graph["5"]["inputs"]
    low_id, low_node = next((key, node) for key, node in conditions if key != "5")
    low = low_node["inputs"]
    assert low["first_frame"] != high["first_frame"]
    assert low["drive_audio"] == high["drive_audio"]
    assert low["final_audio"] == high["final_audio"]
    assert low["length"] == high["length"]
    assert graph["11"]["inputs"]["audio"] == ["5", 2]
    window_id = high["drive_audio"][0]
    assert graph["11"]["inputs"]["start_seconds"] == [window_id, 2]
    assert graph["10"]["inputs"]["av_latent"] == [built["sampling"]["high_output_node"], 0]
    assert low_id != "5"


@pytest.mark.parametrize("ratio", [16 / 9, 9 / 16, 2 / 3, 1])
@pytest.mark.parametrize("mp", [0.4, 0.5, "auto"])
def test_two_pass_canvas_uses_real_learned_geometry(ratio, mp):
    from h3_audio_t8_pkg.director_sampling_settings import two_pass_canvas
    from h3_audio_t8_pkg.learned_latent_upscale_advanced import learned_upscale_geometry

    plan = two_pass_canvas(ratio, mp)
    actual = learned_upscale_geometry(plan["low_width"] // 16, plan["low_height"] // 16,
                                      "target_megapixels", 2.0, plan["actual_megapixels"],
                                      plan["width"], plan["height"], "preserve_source", 1.05)
    assert (actual["output_width"], actual["output_height"]) == (plan["width"], plan["height"])
    assert plan["width"] % 32 == plan["height"] % 32 == 0


@pytest.mark.parametrize("d3", [{}, {"prompt_relay": {"enabled": True}},
                               {"memory": {"low_vram": True, "chunk_ffn": True}}])
def test_two_pass_prompt_validates_against_actual_core_schemas(tmp_path, monkeypatch, d3):
    import asyncio
    from pathlib import Path
    import folder_paths
    from h3_audio_t8_pkg import director_generation
    from h3_audio_t8_pkg.nodes import comfy_entrypoint

    monkeypatch.syspath_prepend(str(Path(folder_paths.__file__).resolve().parent))
    import execution
    import nodes
    from comfy_extras import nodes_custom_sampler

    _patch_director_models(monkeypatch)
    monkeypatch.setattr(director_generation.folder_paths, "get_filename_list", lambda folder: {
        "loras": ["style_a.safetensors", "style_b.safetensors"], "latent_upscale_models": ["minimax_h3_latent_upscaler_3d_fp16.safetensors"],
        "diffusion_models": ["minimax_h3_fl2va_int8_convrot.safetensors"],
        "clip": ["qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"],
        "text_encoders": ["qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"],
        "vae": ["minimax_h3_video_vae_fp16.safetensors", "minimax_h3_audio_vae_fp32.safetensors"],
    }.get(folder, []))
    monkeypatch.setattr(director_generation.folder_paths, "get_full_path", lambda folder, name: name)
    project = new_project()
    project["doc"]["shots"][0]["simplePrompt"] = "A woman smiles at the camera."
    project["doc"]["shots"][0]["d3Inherit"] = False
    project["doc"]["shots"][0]["d3"] = d3
    project["doc"]["sampling"] = {
        "mode": "two_pass", "output_mp": 0.4,
        "low_loras": [{"id": "a", "name": "style_a.safetensors", "strength": 0.8, "enabled": True},
                      {"id": "b", "name": "style_a.safetensors", "strength": 0.25, "enabled": True}],
        "high_loras": [{"id": "c", "name": "style_b.safetensors", "strength": -0.3, "enabled": True}],
    }
    graph = build_director_generation_prompt(project, project["current"], _store(tmp_path))["prompt"]
    classes = asyncio.run(comfy_entrypoint().get_node_list())
    for cls in classes:
        if cls.__name__ in {item["class_type"] for item in graph.values()}:
            monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, cls.__name__, cls)
    for cls in (nodes_custom_sampler.BasicGuider, nodes_custom_sampler.RandomNoise,
                nodes_custom_sampler.SamplerCustomAdvanced):
        monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, cls.__name__, cls)
    valid = asyncio.run(execution.validate_prompt("director-two-pass-schema", graph, None))
    assert valid[0], valid[1]


def test_d3_semantic_bridge_is_compiled_into_native_conditioning(tmp_path, monkeypatch):
    _patch_director_models(monkeypatch)
    store = _store(tmp_path)
    project = new_project()
    project["doc"]["shots"][0]["d3Inherit"] = False
    project["doc"]["shots"][0]["simplePrompt"] = "A stable portrait with gentle motion."
    project["doc"]["shots"][0]["d3"] = {
        "semantic_bridge": {"enabled": True, "alpha": 0.12},
    }
    built = build_director_generation_prompt(project, project["current"], store)
    graph = built["prompt"]
    bridge = next(node for node in graph.values() if node["class_type"] == "MiniMaxH3SemanticBridgeConfigT8")
    apply = next(node for node in graph.values() if node["class_type"] == "MiniMaxH3SemanticBridgeApplyT8")
    assert bridge["inputs"]["alpha"] == 0.12
    assert apply["inputs"]["conditioning"] == ["5", 0]
    assert graph["7"]["inputs"]["conditioning"] == [next(k for k, v in graph.items() if v is apply), 0]


def test_semantic_bridge_prefers_converted_minimax_compat_asset(monkeypatch):
    from h3_audio_t8_pkg import director_generation, nodes_semantic_bridge

    monkeypatch.setattr(
        nodes_semantic_bridge,
        "model_paths",
        lambda: {
            "bunny/BUNNY_H3_ActionLogic_Bridge_V1.safetensors": "bunny",
            "t8_compat/BUNNY_H3_ActionLogic_Bridge_V1_T8_Compat.safetensors": "bunny-compat",
            "t8_compat/MiniMaxH3_SemanticBridge_v1_T8_Compat.safetensors": "minimax-compat",
        },
    )
    monkeypatch.setattr(director_generation.folder_paths, "get_filename_list", lambda _folder: [])
    assert director_generation._optional_semantic_bridge_model() == (
        "t8_compat/MiniMaxH3_SemanticBridge_v1_T8_Compat.safetensors"
    )


def test_d3_prompt_relay_replaces_conditioning_and_preserves_media_slots(tmp_path, monkeypatch):
    _patch_director_models(monkeypatch)
    store = _store(tmp_path)
    project = new_project()
    shot = project["doc"]["shots"][0]
    shot.update(
        writingMode="advanced",
        d3Inherit=False,
        prompt="",
        global_prompt="",
        events=[
            {"id": str(uuid.uuid4()), "start": 0, "end": 2, "text": "The subject looks left."},
            {"id": str(uuid.uuid4()), "start": 2, "end": 4, "text": "The subject looks right."},
        ],
        simplePrompt="",
        d3={"prompt_relay": {"enabled": True, "execution_mode": "apply_exp"}},
    )
    project["doc"]["global"] = "A cinematic continuous shot."
    built = build_director_generation_prompt(project, shot["id"], store)
    graph = built["prompt"]
    plan_id = next(k for k, v in graph.items() if v["class_type"] == "MiniMaxH3PromptRelayPlanT8Advanced")
    relay_id = next(k for k, v in graph.items() if v["class_type"] == "MiniMaxH3PromptRelayConditioningT8Advanced")
    assert graph[plan_id]["inputs"]["timing_mode"] == "frames"
    assert graph[relay_id]["inputs"]["prompt_relay_plan"] == [plan_id, 0]
    assert graph["6"]["inputs"]["model"] == [relay_id, 0]
    assert graph["7"]["inputs"]["conditioning"] == [relay_id, 1]
    assert "MiniMaxH3AudioConditioningT8" not in {node["class_type"] for node in graph.values()}


def test_d3_fast_h3_and_memory_nodes_are_chained_without_changing_default_graph(tmp_path, monkeypatch):
    _patch_director_models(monkeypatch)
    store = _store(tmp_path)
    project = new_project()
    shot = project["doc"]["shots"][0]
    shot["d3Inherit"] = False
    shot["simplePrompt"] = "A calm subject speaks to camera."
    shot["d3"] = {
        "fast_h3_v2": {"enabled": True, "profile": "dense_compat_exp", "min_tokens": 8192},
        "memory": {"low_vram": True, "head_chunks": 4, "chunk_ffn": True, "chunks": 2, "seq_threshold": 4096},
    }
    built = build_director_generation_prompt(project, shot["id"], store)
    graph = built["prompt"]
    types = [node["class_type"] for node in graph.values()]
    assert "MiniMaxH3LowVRAMAttentionT8Advanced" in types
    assert "MiniMaxH3ChunkFeedForwardT8Advanced" in types
    assert "MiniMaxH3FastH3V2SetupEXPT8" in types
    assert "MiniMaxH3FastH3V2RuntimeAuditEXPT8" in types
    assert graph["9"]["class_type"] == "SamplerCustomAdvanced"
    assert graph["1"]["inputs"]["unet_name"] == "fastvideo_fasth3_8step_v2_pruned_int8_convrot.safetensors"
    assert built["turbo_lora"] is None


def test_d3_fast_h3_relay_requires_explicit_dense_profile(tmp_path, monkeypatch):
    _patch_director_models(monkeypatch)
    store = _store(tmp_path)
    project = new_project()
    shot = project["doc"]["shots"][0]
    shot["d3Inherit"] = False
    shot["writingMode"] = "advanced"
    shot["events"] = [
        {"id": str(uuid.uuid4()), "start": 0, "end": 2, "text": "She turns left."},
        {"id": str(uuid.uuid4()), "start": 2, "end": 4, "text": "She turns right."},
    ]
    project["doc"]["global"] = "A woman turns toward the camera."
    shot["d3"] = {
        "fast_h3_v2": {"enabled": True},
        "prompt_relay": {"enabled": True, "execution_mode": "apply_exp"},
    }
    with pytest.raises(ValueError, match="训练 VSA 无法执行 Prompt Relay"):
        build_director_generation_prompt(project, shot["id"], store)

    shot["d3"]["prompt_relay"]["execution_mode"] = "report_only"
    report_only = build_director_generation_prompt(project, shot["id"], store)["prompt"]
    assert any(node["class_type"] == "MiniMaxH3FastH3V2SetupEXPT8" for node in report_only.values())

    shot["d3"]["prompt_relay"]["execution_mode"] = "apply_exp"
    shot["events"] = [{"id": str(uuid.uuid4()), "start": 0, "end": 4, "text": "She turns left."}]
    assert any(node["class_type"] == "MiniMaxH3FastH3V2SetupEXPT8"
               for node in build_director_generation_prompt(project, shot["id"], store)["prompt"].values())
    shot["events"] = []
    shot["writingMode"] = "simple"
    shot["simplePrompt"] = "A woman turns toward the camera."
    assert any(node["class_type"] == "MiniMaxH3FastH3V2SetupEXPT8"
               for node in build_director_generation_prompt(project, shot["id"], store)["prompt"].values())
    shot["writingMode"] = "advanced"
    shot["events"] = [{"id": str(uuid.uuid4()), "start": 0, "end": 4,
                       "text": "She turns left.|She turns right."}]
    with pytest.raises(ValueError, match="训练 VSA 无法执行 Prompt Relay"):
        build_director_generation_prompt(project, shot["id"], store)
    shot["events"] = [
        {"id": str(uuid.uuid4()), "start": 0, "end": 2, "text": "She turns left."},
        {"id": str(uuid.uuid4()), "start": 2, "end": 4, "text": "She turns right."},
    ]
    shot["d3"]["fast_h3_v2"]["profile"] = "dense_compat_exp"
    graph = build_director_generation_prompt(project, shot["id"], store)["prompt"]
    kinds = {node["class_type"] for node in graph.values()}
    assert "MiniMaxH3FastH3V2SetupEXPT8" in kinds
    assert "MiniMaxH3PromptRelayConditioningT8Advanced" in kinds


def test_generation_settings_support_multiple_loras_and_total_pixel_resolution(tmp_path, monkeypatch):
    from h3_audio_t8_pkg import director_generation

    _patch_director_models(monkeypatch)
    monkeypatch.setattr(
        director_generation,
        "_pick_requested",
        lambda _folder, requested, candidates, _label: candidates[0] if requested == "auto" else requested,
    )
    store = _store(tmp_path)
    project = new_project()
    project["doc"]["shots"][0]["simplePrompt"] = "A stable cinematic portrait."
    project["doc"]["generation"].update(
        lora=["style_a.safetensors", "motion_b.safetensors"],
        lora_strength=0.65,
        resolution_mp=0.4,
    )
    built = director_generation.build_director_generation_prompt(project, project["current"], store)
    lora_nodes = [node for node in built["prompt"].values() if node["class_type"] == "MiniMaxH3LoRACompatibilityLoaderT8Advanced"]
    assert [node["inputs"]["lora_name"] for node in lora_nodes] == ["style_a.safetensors", "motion_b.safetensors"]
    assert all(node["inputs"]["strength_model"] == 0.65 for node in lora_nodes)
    canvas = built["report"]["shots"][0]["canvas"]
    assert canvas["width"] % 32 == canvas["height"] % 32 == 0
    assert 0.35 <= canvas["width"] * canvas["height"] / 1_000_000 <= 0.45
