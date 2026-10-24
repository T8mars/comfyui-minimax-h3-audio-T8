"""Actual graph/port validation; model-name fixtures do not execute weights."""
import asyncio
from copy import deepcopy

import pytest

import h3_audio_t8_pkg as package
from h3_audio_t8_pkg.director_generation import build_director_generation_prompt
from test_director_generation import _patch_director_models
from test_director_keyframes import scene as shared_scene

scene = shared_scene  # Register the shared real-PNG pytest fixture in this module.


KEYFRAME = "MiniMaxH3DirectorKeyframeConditioningEXPT8"


@pytest.mark.parametrize("relay", [False, True])
@pytest.mark.parametrize("bridge", [False, True])
@pytest.mark.parametrize("dual", [False, True])
def test_actual_compiled_models_conditions_plans_and_lora_stacks_pair_by_stage(scene, monkeypatch, relay, bridge, dual):
    import execution
    import nodes
    from comfy_extras import nodes_custom_sampler
    from h3_audio_t8_pkg import director_generation as generation

    store, project, shot, images = scene
    _patch_director_models(monkeypatch)
    names = {
        "diffusion_models": ["minimax_h3_fl2va_int8_convrot.safetensors"],
        "vae": ["minimax_h3_video_vae_fp16.safetensors", "minimax_h3_audio_vae_fp32.safetensors"],
        "clip": ["qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"],
        "text_encoders": ["qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"],
        "loras": ["low-a.safetensors", "high-b.safetensors"],
        "latent_upscale_models": ["minimax_h3_latent_upscaler_3d_fp16.safetensors"],
    }
    monkeypatch.setattr(generation.folder_paths, "get_filename_list", lambda folder: names.get(folder, []))
    monkeypatch.setattr(generation.folder_paths, "get_full_path", lambda folder, name: name)
    monkeypatch.setattr(generation.folder_paths, "get_input_directory", lambda: str(store.input_root))
    project["doc"]["d3"] = {"semantic_bridge": {"enabled": bridge},
        "prompt_relay": {"enabled": relay, "execution_mode": "apply_exp"}}
    project["doc"]["generation"]["unet"] = names["diffusion_models"][0]
    if dual:
        project["doc"]["sampling"] = {"mode": "two_pass", "output_mp": .4,
            "low_loras": [{"id": "low-1", "name": "low-a.safetensors", "strength": .8, "enabled": True},
                          {"id": "low-2", "name": "low-a.safetensors", "strength": .3, "enabled": True}],
            "high_loras": [{"id": "high-1", "name": "high-b.safetensors", "strength": -.2, "enabled": True}]}
    source = deepcopy(project)
    built = build_director_generation_prompt(project, shot["id"], store)
    graph = built["prompt"]
    conditions = {key: value for key, value in graph.items() if value["class_type"] == KEYFRAME}
    assert len(conditions) == (2 if dual else 1)
    assert len([value for value in graph.values() if value["class_type"] == "MiniMaxH3KeyframePlanT8Advanced"]) == 1
    plan_id = next(key for key, value in graph.items() if value["class_type"] == "MiniMaxH3KeyframePlanT8Advanced")
    plan = graph[plan_id]["inputs"]
    assert plan["position_mode"] == "frame" and plan["position"] == 90
    assert graph[plan["image"][0]]["inputs"]["image"] == images[2]["server_path"]
    assert all(value["inputs"]["keyframe_plan"] == [plan_id, 0] for value in conditions.values())
    assert all(("prompt_relay_plan" in value["inputs"]) == relay for value in conditions.values())
    low_condition_id = graph["6"]["inputs"]["model"][0]
    assert low_condition_id in conditions and graph["9"]["inputs"]["latent_image"] == [low_condition_id, 2]
    low_positive = graph["7"]["inputs"]["conditioning"]
    if bridge and not relay:
        assert graph[low_positive[0]]["class_type"] == "MiniMaxH3SemanticBridgeApplyT8"
        assert graph[low_positive[0]]["inputs"]["conditioning"] == [low_condition_id, 1]
    else:
        assert low_positive == [low_condition_id, 1]
    if dual:
        high_id = next(key for key in conditions if key != low_condition_id)
        low_inputs, high_inputs = conditions[low_condition_id]["inputs"], conditions[high_id]["inputs"]
        low_lora, high_lora = graph[low_inputs["model"][0]], graph[high_inputs["model"][0]]
        assert low_lora["inputs"]["lora_name"] == "low-a.safetensors" and low_lora["inputs"]["strength_model"] == .3
        assert graph[low_lora["inputs"]["model"][0]]["inputs"]["strength_model"] == .8
        assert high_lora["inputs"]["lora_name"] == "high-b.safetensors" and high_lora["inputs"]["strength_model"] == -.2
        mixer = next(node for node in graph.values() if node["class_type"] == "MiniMaxH3TwoPassDetailMixerT8Advanced")
        assert mixer["inputs"]["model"] == [high_id, 0]
        up_id = next(key for key, node in graph.items() if node["class_type"] == "MiniMaxH3LearnedLatentUpscaleT8Advanced")
        assert graph[up_id]["inputs"]["av_latent"] == ["9", 1]
        assert (high_inputs["width"], high_inputs["height"]) == ([up_id, 1], [up_id, 2])
        assert isinstance(low_inputs["width"], int) and low_inputs["width"] == built["shot"]["two_pass_canvas"]["low_width"]
        assert low_inputs["first_frame"] != high_inputs["first_frame"]
        assert low_inputs["last_frame"] != high_inputs["last_frame"]
    assert graph["11"]["inputs"]["audio"] == ["10", 1] and project == source
    if not bridge:
        # Validate real Core schemas and file-backed LoadImage inputs, not
        # mocked node return values or a queue/inference operation.
        for cls in asyncio.run(package.comfy_entrypoint().get_node_list()):
            monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, cls.__name__, cls)
        for cls in (nodes_custom_sampler.BasicGuider, nodes_custom_sampler.RandomNoise, nodes_custom_sampler.SamplerCustomAdvanced):
            monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, cls.__name__, cls)
        valid = asyncio.run(execution.validate_prompt("director-keyframe-graph", graph, None))
        assert valid[0], valid[1:]
