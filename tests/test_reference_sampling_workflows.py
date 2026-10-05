"""Real finalized Core/project schemas and validator, no queue or model load."""
import asyncio
from copy import deepcopy
import json
from pathlib import Path

import pytest

import h3_audio_t8_pkg
from tools.build_reference_sampling_workflows import make_workflow, RELAY, APPLY
from tools.check_windows_paths import validate_paths


@pytest.fixture
def actual_info(monkeypatch):
    import folder_paths
    # conftest exposes h3_t8 for package tests; select Core before importing
    # its equally named nodes module, rather than silently testing the shim.
    core = Path(folder_paths.__file__).resolve().parent
    monkeypatch.syspath_prepend(str(core))
    import nodes
    assert Path(nodes.__file__).resolve().parent == core
    from h3_audio_t8_pkg import nodes_reference_package
    from tools.build_reference_sampling_workflows import MODEL, CLIP, VIDEO_VAE, AUDIO_VAE, LORA, UPSCALER
    original_files = folder_paths.get_filename_list
    selected_files = {"diffusion_models": [MODEL], "unet": [MODEL], "text_encoders": [CLIP],
        "clip": [CLIP], "vae": [VIDEO_VAE, AUDIO_VAE], "loras": [LORA], "latent_upscale_models": [UPSCALER]}
    monkeypatch.setattr(folder_paths, "get_filename_list", lambda category:
        selected_files[category] if category in selected_files else original_files(category))
    monkeypatch.setattr(nodes_reference_package, "installed_package_names", lambda *_: ["A.safetensors", "B.safetensors"])
    from comfy_extras.nodes_custom_sampler import BasicGuider, RandomNoise, SamplerCustomAdvanced
    classes = asyncio.run(h3_audio_t8_pkg.comfy_entrypoint().get_node_list())
    selected = {cls.define_schema().node_id: cls for cls in classes}
    selected.update({cls.__name__: cls for cls in (BasicGuider, RandomNoise, SamplerCustomAdvanced)})
    for name in ("UNETLoader", "CLIPLoader", "VAELoader", "PreviewAny"):
        if name in nodes.NODE_CLASS_MAPPINGS:
            selected[name] = nodes.NODE_CLASS_MAPPINGS[name]
    # PreviewAny is needed by the readback builder's contract, not an output
    # in these sampling graphs. Use its actual class if Core has not registered
    # extras in this fresh CPU process yet.
    if "PreviewAny" not in selected:
        from comfy_extras.nodes_preview_any import PreviewAny
        selected["PreviewAny"] = PreviewAny
    info = {}
    for name, cls in selected.items():
        monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, name, cls)
        if hasattr(cls, "GET_NODE_INFO_V1"):
            info[name] = cls.GET_NODE_INFO_V1()
        else:
            inputs = cls.INPUT_TYPES()
            info[name] = {"input": inputs, "input_order": {key: list(group) for key, group in inputs.items()},
                "output": cls.RETURN_TYPES, "output_name": cls.RETURN_TYPES}
    return info


@pytest.mark.parametrize("variant", ["ordinary", "split", "cold"])
def test_reference_candidate_real_schemas_typed_edges_and_core_validation(actual_info, variant):
    kwargs = dict(voice_name="A.safetensors", image_name="B.safetensors", voice_sha="a"*64, image_sha="b"*64)
    workflow, graph = make_workflow(actual_info, **kwargs, variant=variant)
    import execution
    validation = asyncio.run(execution.validate_prompt("reference-schema-only", deepcopy(graph), None))
    assert validation[0], validation
    ids = {key: index+1 for index, key in enumerate(graph)}
    by_id = {item["id"]: item for item in workflow["nodes"]}
    links = {item[0]: item for item in workflow["links"]}
    for key, source in graph.items():
        for field, value in source["inputs"].items():
            if isinstance(value, list) and len(value) == 2 and str(value[0]) in graph:
                pin = next(item for item in by_id[ids[key]]["inputs"] if item["name"] == field)
                edge = links[pin["link"]]
                assert edge[1:3] == [ids[value[0]], value[1]] and edge[3] == ids[key]
                expected_type = actual_info[graph[value[0]]["class_type"]]["output"][value[1]]
                assert pin["type"] == expected_type
    assert graph["trim"]["inputs"]["duration_seconds"] == 5.
    assert json.loads(graph["5"]["inputs"]["roles_json"])[0]["visual"] is False
    assert graph["5"]["inputs"]["packages.package_0"] == ["1", 0]
    assert graph["5"]["inputs"]["packages.package_1"] == ["2", 0]
    assert not workflow["extra"]["radar_r6_reference_sampling"]["gpu_verified"]
    assert not validate_paths(["examples/workflows/75-radar-r6-reference/Ref_"+variant+"_EXP.json"])
    if variant == "ordinary":
        assert graph["single_apply"]["class_type"] == APPLY and "low_sample" not in graph
    else:
        assert graph["high_apply"]["class_type"] == RELAY
        assert graph["high_apply"]["inputs"]["width"] == ["upscale", 1]
        assert graph["upscale"]["inputs"]["scale_by"] == 1.2
        assert graph["handoff"]["inputs"]["second_audio_source"] == "auto"
        assert graph["high_eav_config"]["inputs"]["mode"] == "report_only"
        if variant == "cold":
            assert graph["low_load"]["inputs"]["expected_stage"] == "dual_low_4"
            assert not any(key.startswith("low_") and key != "low_load" for key in graph)
            assert sum(item["class_type"] == "MiniMaxH3StageSamplerEXPT8" for item in graph.values()) == 1
            assert not any(item["class_type"] == "MiniMaxH3StageUNETLoaderAfterEXPT8" for item in graph.values())
        else:
            assert graph["low_apply"]["class_type"] == RELAY
            assert graph["high_model"]["inputs"]["completed_stage"] == ["low_sample", 2]
            assert graph["upscale"]["inputs"]["av_latent"] == ["low_audit", 0]
            assert graph["low_audit"]["inputs"]["av_latent"] == ["low_save", 1]


def test_missing_selected_weight_or_new_service_schema_is_not_guessed(actual_info):
    args = dict(voice_name="A.safetensors", image_name="B.safetensors", voice_sha="a"*64, image_sha="b"*64)
    del actual_info[RELAY]
    with pytest.raises(ValueError, match="not loaded"):
        make_workflow(actual_info, **args)
