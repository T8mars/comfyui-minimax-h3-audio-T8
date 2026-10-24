"""Real file-backed Director graph contracts; not trained HyperFlow inference."""
import asyncio
from copy import deepcopy

import pytest

import h3_audio_t8_pkg as package
from h3_audio_t8_pkg import director_generation as generation, director_hyperflow
from test_director_generation import _patch_director_models
from test_director_generation import _audio
from test_director_keyframes import scene as shared_scene

scene = shared_scene
FACTORY = "MiniMaxH3DirectorKeyframeConditioningEXPT8"
VARIANTS = ("single8", "continuous4plus4", "continuous4plus4separate", "upscale8plus4", "upscale4plus4")


def stage_pair(graph, inputs):
    model = inputs["model"]
    positive = inputs.get("positive", inputs.get("conditioning"))
    if graph[positive[0]]["class_type"] == "MiniMaxH3SemanticBridgeApplyT8":
        positive = graph[positive[0]]["inputs"]["conditioning"]
    assert graph[model[0]]["class_type"] == FACTORY and model[1] == 0
    assert positive == [model[0], 1]
    return model[0]


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("relay", [False, True])
@pytest.mark.parametrize("bridge", [False, True])
def test_middle_factory_is_paired_per_hyperflow_stage_and_validates_in_real_core(
    scene, tmp_path, monkeypatch, variant, relay, bridge,
):
    import execution
    import nodes
    from comfy_extras import nodes_custom_sampler

    store, project, shot, images = scene
    _patch_director_models(monkeypatch)
    names = {
        "diffusion_models": ["minimax_h3_fl2va_int8_convrot.safetensors"],
        "vae": ["minimax_h3_video_vae_fp16.safetensors", "minimax_h3_audio_vae_fp32.safetensors"],
        "clip": ["qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"],
        "text_encoders": ["qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"],
        "loras": ["low-a.safetensors", "high-b.safetensors"],
        "hyperflow": ["hf.safetensors"],
        "latent_upscale_models": ["minimax_h3_latent_upscaler_3d_fp16.safetensors"],
    }
    monkeypatch.setattr(generation.folder_paths, "get_filename_list", lambda folder: names.get(folder, []))
    monkeypatch.setattr(generation.folder_paths, "get_full_path", lambda _folder, name: name)
    monkeypatch.setattr(generation.folder_paths, "get_input_directory", lambda: str(store.input_root))
    weight = tmp_path / "hf.safetensors"
    weight.write_bytes(b"graph preflight only; no model inference")
    monkeypatch.setattr(director_hyperflow, "_resolve", lambda _selection: weight)
    project["doc"]["generation"]["unet"] = names["diffusion_models"][0]
    project["doc"]["d3"] = {"semantic_bridge": {"enabled": bridge},
        "prompt_relay": {"enabled": relay, "execution_mode": "apply_exp"}}
    project["doc"]["sampling"] = {"mode": "hyperflow", "variant": variant,
        "hyperflow_file": "hyperflow/hf.safetensors", "output_mp": .4,
        "low_loras": [{"id": "l1", "name": "low-a.safetensors", "strength": .8, "enabled": True},
                      {"id": "l2", "name": "low-a.safetensors", "strength": .3, "enabled": True}],
        "high_loras": [{"id": "h1", "name": "high-b.safetensors", "strength": -.2, "enabled": True}]}
    source = deepcopy(project)
    built = generation.build_director_generation_prompt(project, shot["id"], store)
    graph = built["prompt"]
    factories = {key: node for key, node in graph.items() if node["class_type"] == FACTORY}
    assert len(factories) == (1 if variant == "single8" else 2)
    assert len({tuple(node["inputs"]["keyframe_plan"]) for node in factories.values()}) == 1
    assert all(("prompt_relay_plan" in node["inputs"]) == relay for node in factories.values())
    assert project == source
    assert built["sampling"]["total_nfe"] == (12 if variant == "upscale8plus4" else 8)

    if variant == "continuous4plus4":
        split = graph["9"]["inputs"]
        low_id = stage_pair(graph, {"model": split["model_low"], "positive": split["positive"]})
        high_id = stage_pair(graph, {"model": split["model_high"], "positive": split["positive_high"]})
        assert split["av_latent"] == [low_id, 2]
    elif variant == "continuous4plus4separate":
        head = next(node for node in graph.values() if node["class_type"] == "MiniMaxH3HyperFlowHeadEffectsBindEXPT8")
        tail = next(node for node in graph.values() if node["class_type"] == "MiniMaxH3HyperFlowTailEffectsBindEXPT8")
        low_id, high_id = stage_pair(graph, head["inputs"]), stage_pair(graph, tail["inputs"])
        assert graph["9"]["inputs"]["av_latent"] == [low_id, 2]
    else:
        setup = graph[graph["7"]["inputs"]["model"][0]]["inputs"]
        low_id = stage_pair(graph, {"model": setup["model"], "positive": graph["7"]["inputs"]["conditioning"]})
        assert graph["9"]["inputs"]["latent_image"] == [low_id, 2]
        high_id = None
        if variant.startswith("upscale"):
            high_sampler = graph[graph["10"]["inputs"]["av_latent"][0]]["inputs"]
            high_guider = graph[high_sampler["guider"][0]]["inputs"]
            high_setup = graph[high_guider["model"][0]]["inputs"]
            reconcile_id = high_sampler["latent_image"][0]
            reconcile = graph[reconcile_id]
            assert reconcile["class_type"] == "MiniMaxH3TwoPassLatentReconcileT8Advanced"
            assert high_setup["av_latent"] == [reconcile_id, 0]
            assert high_guider["conditioning"] == [reconcile_id, 1]
            high_id = stage_pair(graph, {"model": high_setup["model"], "positive": reconcile["inputs"]["positive"]})
            up_id = reconcile["inputs"]["learned_latent"][0]
            assert graph[up_id]["inputs"]["av_latent"] == ["9", 1 if variant == "upscale4plus4" else 0]
            assert (factories[high_id]["inputs"]["width"], factories[high_id]["inputs"]["height"]) == ([up_id, 1], [up_id, 2])
            assert factories[low_id]["inputs"]["width"] == built["shot"]["two_pass_canvas"]["low_width"]
            assert factories[low_id]["inputs"]["first_frame"] != factories[high_id]["inputs"]["first_frame"]
    assert high_id != low_id
    expected_stacks = [(low_id, "low-a.safetensors", .3)]
    if high_id:
        expected_stacks.append((high_id, "high-b.safetensors", -.2))
    for condition_id, expected_name, expected_strength in expected_stacks:
        loader = graph[factories[condition_id]["inputs"]["model"][0]]
        assert loader["class_type"] == "MiniMaxH3HyperFlowFreshLoaderEXPT8"
        lora = graph[loader["inputs"]["model"][0]]
        assert lora["inputs"]["lora_name"] == expected_name and lora["inputs"]["strength_model"] == expected_strength
        if condition_id == low_id:
            assert graph[lora["inputs"]["model"][0]]["inputs"]["strength_model"] == .8
    assert graph["11"]["inputs"]["audio"] == ["10", 1]
    assert graph[next(iter(factories.values()))["inputs"]["keyframe_plan"][0]]["inputs"]["position"] == 90
    assert any(node["class_type"] == "LoadImage" and node["inputs"]["image"] == images[2]["server_path"] for node in graph.values())
    if not bridge:
        for cls in asyncio.run(package.comfy_entrypoint().get_node_list()):
            monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, cls.__name__, cls)
        for cls in (nodes_custom_sampler.BasicGuider, nodes_custom_sampler.RandomNoise, nodes_custom_sampler.SamplerCustomAdvanced):
            monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, cls.__name__, cls)
        valid = asyncio.run(execution.validate_prompt("h21-hyperflow-middle", graph, None))
        assert valid[0], valid[1:]


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("relay", [False, True])
def test_real_recording_window_middle_and_original_mux_are_preserved(scene, tmp_path, monkeypatch, variant, relay):
    store, project, shot, _ = scene
    _patch_director_models(monkeypatch)
    audio = _audio(store, seconds=4.)
    project["assets"].append(audio)
    # Preserve the Director's existing first-frame recording recipe. The
    # separate native-audio matrix above covers first+last+middle factories.
    shot.update(mode="first", last=None, sound="record", audio=audio["id"], start=0., end=4.,
                simplePrompt="The person speaks naturally.")
    names = {"diffusion_models": ["minimax_h3_fl2va_int8_convrot.safetensors"],
        "loras": [], "hyperflow": ["hf.safetensors"],
        "latent_upscale_models": ["minimax_h3_latent_upscaler_3d_fp16.safetensors"]}
    monkeypatch.setattr(generation.folder_paths, "get_filename_list", lambda folder: names.get(folder, []))
    monkeypatch.setattr(generation.folder_paths, "get_full_path", lambda _folder, name: name)
    weight = tmp_path / "hf.safetensors"
    weight.write_bytes(b"recording graph preflight only")
    monkeypatch.setattr(director_hyperflow, "_resolve", lambda _selection: weight)
    project["doc"]["generation"]["unet"] = names["diffusion_models"][0]
    project["doc"]["sampling"] = {"mode": "hyperflow", "variant": variant,
        "hyperflow_file": "hyperflow/hf.safetensors", "output_mp": .4}
    project["doc"]["d3"] = {"prompt_relay": {"enabled": relay, "execution_mode": "apply_exp"}}
    original = deepcopy(project)
    built = generation.build_director_generation_prompt(project, shot["id"], store)
    graph = built["prompt"]
    assert built["shot"]["task_type"].lower() == "hybrid" and built["shot"]["audio_mode"] == "lock_source"
    factories = {key: node["inputs"] for key, node in graph.items() if node["class_type"] == FACTORY}
    assert len(factories) == (1 if variant == "single8" else 2)
    assert any(node["class_type"] == "MiniMaxH3AudioWindowT8" for node in graph.values()), {
        "media_roles": [(item["role"], item["asset_id"]) for item in built["shot"]["media_map"]],
        "node_types": [node["class_type"] for node in graph.values()]}
    window_id = next(key for key, node in graph.items() if node["class_type"] == "MiniMaxH3AudioWindowT8")
    source = graph[window_id]["inputs"]["audio"]
    assert graph[source[0]]["inputs"]["audio"] == audio["server_path"]
    for inputs in factories.values():
        assert inputs["drive_audio"] == inputs["final_audio"] == [window_id, 0]
        assert inputs["audio_mode"] == "lock_source" and inputs["audio_denoise_strength"] == 0.
        assert inputs["add_source_as_reference"] is True
    mux_id, mux_port = graph["11"]["inputs"]["audio"]
    assert mux_id in factories and mux_port == 3
    assert graph["11"]["inputs"]["start_seconds"] == [window_id, 2]
    assert built["shot"]["delivery_audio"] == "original_selected_recording"
    assert project == original
    if variant == "continuous4plus4separate":
        from h3_audio_t8_pkg.modular_sampling import hyperflow_storage

        # Graph/warning unit fixture only. Actual SHA, safetensors and cold
        # sampling are independently exercised by the continuation tests.
        digest = "a" * 64
        monkeypatch.setattr(hyperflow_storage, "fingerprint", lambda *_a: digest)
        project["doc"]["sampling"]["stage_checkpoint"] = {"mode": "resume_tail",
            "artifact_path": "frozen/hyperflow-head.safetensors", "artifact_sha256": digest}
        resumed = generation.build_director_generation_prompt(project, shot["id"], store)
        notices = [item for item in resumed["report"]["warnings"] if "frozen_head_sha256" in item]
        assert len(notices) == 1 and notices[0]["frozen_head_sha256"] == digest
        assert notices[0]["current_recording_sha256"] == audio["sha256"]
        assert "不证明" in notices[0]["message"] and "重新生成HEAD" in notices[0]["message"]
        graph = resumed["prompt"]
        factories = {key for key, node in graph.items() if node["class_type"] == FACTORY}
        assert len(factories) == 1 and graph["11"]["inputs"]["audio"] == [next(iter(factories)), 3]
        assert not any(node["class_type"] in {"RandomNoise", "MiniMaxH3HyperFlowHeadStageEXPT8"} for node in graph.values())
        assert resumed["sampling"]["stage_nfe"] == [4]
