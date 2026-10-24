"""Actual Director dependency graphs; no trained control/GPU quality claim."""
import asyncio
from copy import deepcopy
import json
import uuid

import pytest

import h3_audio_t8_pkg as package
from h3_audio_t8_pkg import director_generation as generation, director_hyperflow
from h3_audio_t8_pkg.director_batch import capture_resources, verify_resources
from test_director_generation import _audio, _patch_director_models
from test_director_keyframes import scene as shared_scene

scene = shared_scene
MODES = ("single", "two_pass", "single8", "continuous4plus4",
         "continuous4plus4separate", "upscale8plus4", "upscale4plus4")
APPLY = "MiniMaxH3FunControlRegionApplyEXPT8"
SOURCE = "MiniMaxH3DirectorControlSourceEXPT8"
AUDIT = "MiniMaxH3ControlRegionAuditEXPT8"
HEAD_AUDIT = "MiniMaxH3DirectorControlHeadAuditEXPT8"


def configure(scene, tmp_path, monkeypatch, mode, relay=False, middle=True):
    store, project, shot, images = scene
    _patch_director_models(monkeypatch)
    names = {
        "diffusion_models": ["minimax_h3_fl2va_int8_convrot.safetensors"],
        "vae": ["minimax_h3_video_vae_fp16.safetensors", "minimax_h3_audio_vae_fp32.safetensors"],
        "clip": ["qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"],
        "text_encoders": ["qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"],
        "loras": [], "hyperflow": ["hf.safetensors"],
        "latent_upscale_models": ["minimax_h3_latent_upscaler_3d_fp16.safetensors"],
        "controlnet": [], "model_patches": ["unfamiliar/control.safetensors"],
    }
    weights = tmp_path / "selected-model-bytes.bin"
    weights.write_bytes(b"graph validation only, never inference weights")
    monkeypatch.setattr(generation.folder_paths, "get_filename_list", lambda folder: names.get(folder, []))
    monkeypatch.setattr(generation.folder_paths, "get_full_path", lambda folder, name:
        str(weights) if name in names.get(folder, []) else None)
    monkeypatch.setattr(generation.folder_paths, "get_input_directory", lambda: str(store.input_root))
    monkeypatch.setattr(director_hyperflow, "_resolve", lambda _selection: weights)
    project["doc"]["generation"]["unet"] = names["diffusion_models"][0]
    project["doc"]["d3"] = {"prompt_relay": {"enabled": relay, "execution_mode": "apply_exp"}}
    if mode in {"single", "two_pass"}:
        project["doc"]["sampling"] = {"mode": mode, "output_mp": .4}
    else:
        project["doc"]["sampling"] = {"mode": "hyperflow", "variant": mode,
            "hyperflow_file": "hyperflow/hf.safetensors", "output_mp": .4}
    if not middle:
        shot.pop("middleKeyframes")
        shot["simplePrompt"] = "The person turns naturally."
    shot["controlRegions"] = {"version": 1, "items": [
        {"id": str(uuid.uuid4()), "asset_id": images[2]["id"],
         "control_net_name": "unfamiliar/control.safetensors", "strength": strength,
         "region": {"mode": "rect", "xyxy_normalized": [.1, .2, .7, .9]},
         "crop": {"enabled": True, "xyxy_normalized": [-.1, .1, .8, 1.2]}}
        for strength in (7., -3.5)]}
    return store, project, shot, images, weights


def closure(graph, output="12"):
    reached = set()

    def visit(key):
        if key in reached:
            return
        reached.add(key)
        for value in graph[key]["inputs"].values():
            if isinstance(value, list) and len(value) == 2 and value[0] in graph:
                visit(value[0])

    visit(output)
    return reached


def registered(monkeypatch):
    import nodes
    from comfy_extras import nodes_custom_sampler
    from comfy_extras.nodes_video import LoadVideo, GetVideoComponents

    for cls in asyncio.run(package.comfy_entrypoint().get_node_list()):
        monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, cls.__name__, cls)
    for cls in (nodes_custom_sampler.BasicGuider, nodes_custom_sampler.RandomNoise,
                nodes_custom_sampler.SamplerCustomAdvanced, LoadVideo, GetVideoComponents):
        monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, cls.__name__, cls)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("relay", [False, True])
@pytest.mark.parametrize("middle", [False, True])
def test_real_control_pairs_stage_canvas_and_audits_are_on_output_closure(
    scene, tmp_path, monkeypatch, mode, relay, middle,
):
    import execution

    store, project, shot, images, _ = configure(scene, tmp_path, monkeypatch, mode, relay, middle)
    before = deepcopy(project)
    built = generation.build_director_generation_prompt(project, shot["id"], store)
    graph = built["prompt"]
    active_stages = {"SINGLE"} if mode in {"single", "single8"} else {"LOW", "HIGH"}
    applies = {key: node["inputs"] for key, node in graph.items() if node["class_type"] == APPLY}
    assert len(applies) == 2 * len(active_stages)
    assert {item["stage"] for item in applies.values()} == active_stages
    used = closure(graph)
    for key, inputs in applies.items():
        assert key in used
        assert inputs["vae"] == ["2", 0]
        source = graph[inputs["control_video"][0]]
        assert source["class_type"] == SOURCE
        assert source["inputs"]["width"] == inputs["width"]
        assert source["inputs"]["height"] == inputs["height"]
        assert source["inputs"]["length"] == inputs["length"]
        assert source["inputs"]["stage"] == inputs["stage"]
        assert inputs["region_mask"] == [inputs["control_video"][0], 1]
        assert inputs["crop_plan_json"] == [inputs["control_video"][0], 2]
        raw = graph[source["inputs"]["control_video"][0]]
        assert raw["class_type"] == "LoadImage"
        assert raw["inputs"]["image"] == images[2]["server_path"]
        spec = json.loads(source["inputs"]["spec_json"])
        assert spec["crop"]["xyxy_normalized"] == [-.1, .1, .8, 1.2]
        assert spec["strength"] == inputs["strength"]
        if graph[inputs["model"][0]]["class_type"] == APPLY:
            assert inputs["positive"] == [inputs["model"][0], 1]
    audits = {key: node for key, node in graph.items() if node["class_type"] in {AUDIT, HEAD_AUDIT}}
    assert len(audits) == len(applies)
    assert all(key in used for key in audits)
    assert {tuple(node["inputs"]["region_state"]) for node in audits.values()} == {
        (key, 2) for key in applies}
    assert sum(node["class_type"] == HEAD_AUDIT for node in audits.values()) == (
        2 if mode == "continuous4plus4separate" else 0)
    assert graph["11"]["inputs"]["audio"] == ["10", 1]
    assert project == before
    if mode in {"two_pass", "upscale8plus4", "upscale4plus4"}:
        up_id, up = next((key, node) for key, node in graph.items()
                        if node["class_type"] == "MiniMaxH3LearnedLatentUpscaleT8Advanced")
        cursor = up["inputs"]["av_latent"]
        while graph[cursor[0]]["class_type"] == AUDIT:
            cursor = graph[cursor[0]]["inputs"]["after_sampling"]
        assert cursor == ["9", 0 if mode == "upscale8plus4" else 1]
        high = [inputs for inputs in applies.values() if inputs["stage"] == "HIGH"]
        assert all((item["width"], item["height"]) == ([up_id, 1], [up_id, 2]) for item in high)
    registered(monkeypatch)
    valid = asyncio.run(execution.validate_prompt("h21-control-graph", graph, None))
    assert valid[0], valid[1:]


def test_disabled_controls_preserve_original_graph_and_do_not_load_model(scene, tmp_path, monkeypatch):
    store, project, shot, _, _ = configure(scene, tmp_path, monkeypatch, "two_pass")
    for item in shot["controlRegions"]["items"]:
        item.update(enabled=False, asset_id=None, control_net_name="")
    disabled = generation.build_director_generation_prompt(project, shot["id"], store)
    del shot["controlRegions"]
    baseline = generation.build_director_generation_prompt(project, shot["id"], store)
    assert disabled["prompt"] == baseline["prompt"]


def test_fresh_hyperflow_and_control_model_bytes_are_frozen_and_change_rejected(scene, tmp_path, monkeypatch):
    store, project, shot, images, weights = configure(scene, tmp_path, monkeypatch, "single8")
    built = generation.build_director_generation_prompt(project, shot["id"], store)
    frozen = capture_resources(store, project, [built], lambda *_: weights)
    assert "hyperflow_selection:hyperflow/hf.safetensors" in frozen["models"]
    assert "fun_control_selection:unfamiliar/control.safetensors" in frozen["models"]
    assert images[2]["id"] in frozen["assets"]
    verify_resources(store, frozen, lambda *_: weights)
    weights.write_bytes(b"different selected bytes")
    with pytest.raises(ValueError, match="模型已变化"):
        verify_resources(store, frozen, lambda *_: weights)


def test_inactive_control_draft_is_not_a_frozen_batch_resource(scene, tmp_path, monkeypatch):
    store, project, shot, _, weights = configure(scene, tmp_path, monkeypatch, "two_pass", middle=False)
    for row in shot["controlRegions"]["items"]:
        row.update(asset_id=str(uuid.uuid4()), control_net_name="missing.safetensors",
                   stages={"low": False, "high": False})
    built = generation.build_director_generation_prompt(project, shot["id"], store)
    frozen = capture_resources(store, project, [built], lambda *_: weights)
    assert not any(key.startswith("fun_control_selection:") for key in frozen["models"])
    assert not any(row["asset_id"] in frozen["assets"] for row in shot["controlRegions"]["items"])


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("relay", [False, True])
def test_original_recording_routes_keep_actual_window_grid_and_mux(scene, tmp_path, monkeypatch, mode, relay):
    store, project, shot, _, _ = configure(scene, tmp_path, monkeypatch, mode, relay, middle=False)
    recording = _audio(store)
    project["assets"].append(recording)
    shot.update(mode="first", last=None, sound="record", audio=recording["id"], start=0., end=4.)
    original = deepcopy(project)
    built = generation.build_director_generation_prompt(project, shot["id"], store)
    graph = built["prompt"]
    window = next(key for key, node in graph.items() if node["class_type"] == "MiniMaxH3AudioWindowT8")
    mux, port = graph["11"]["inputs"]["audio"]
    assert graph[mux]["class_type"] in {"MiniMaxH3AudioConditioningT8", "MiniMaxH3PromptRelayConditioningT8Advanced"}
    assert port == (3 if relay else 2)
    assert graph[mux]["inputs"]["drive_audio"] == graph[mux]["inputs"]["final_audio"] == [window, 0]
    assert graph["11"]["inputs"]["start_seconds"] == [window, 2]
    sources = [node["inputs"] for node in graph.values() if node["class_type"] == SOURCE]
    if not relay:
        assert all(item["length"] == [window, 1] for item in sources)
    else:
        plans = [node["inputs"]["length"] for node in graph.values()
                 if node["class_type"] == "MiniMaxH3PromptRelayPlanT8Advanced"]
        assert all(item["length"] in plans for item in sources)
    assert project == original


def test_resume_tail_excludes_low_controls_and_has_no_head_noise_or_audit(scene, tmp_path, monkeypatch):
    from h3_audio_t8_pkg.modular_sampling import hyperflow_storage

    store, project, shot, _, weights = configure(scene, tmp_path, monkeypatch, "continuous4plus4separate")
    digest = "a" * 64
    monkeypatch.setattr(hyperflow_storage, "fingerprint", lambda *_: digest)
    project["doc"]["sampling"]["stage_checkpoint"] = {"mode": "resume_tail",
        "artifact_path": "frozen/head.safetensors", "artifact_sha256": digest}
    shot["controlRegions"]["items"].append({"id": str(uuid.uuid4()), "asset_id": str(uuid.uuid4()),
        "control_net_name": "does-not-exist", "stages": {"low": True, "high": False}})
    built = generation.build_director_generation_prompt(project, shot["id"], store)
    graph = built["prompt"]
    assert {row["stage"] for row in built["shot"]["control_regions"]} == {"HIGH"}
    assert not any(node["class_type"] in {HEAD_AUDIT, "RandomNoise", "MiniMaxH3HyperFlowHeadStageEXPT8"}
                   for node in graph.values())
    assert all(node["inputs"]["stage"] == "HIGH" for node in graph.values() if node["class_type"] == APPLY)
    assert built["sampling"]["stage_nfe"] == [4]
    frozen = capture_resources(store, project, [built], lambda *_: weights)
    assert not any("does-not-exist" in key for key in frozen["models"])


def test_control_video_uses_original_frames_and_native_fps_without_an_audio_edge(scene, tmp_path, monkeypatch):
    import av
    import numpy as np

    store, project, shot, _, _ = configure(scene, tmp_path, monkeypatch, "single", middle=False)
    aid = str(uuid.uuid4())
    path = store.input_root / "t8_director" / aid / "source.mp4"
    path.parent.mkdir(parents=True)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=12)
        stream.width, stream.height, stream.pix_fmt = 64, 32, "yuv420p"
        stream.codec_context.thread_count = 1
        for value in (10, 40, 80, 130):
            frame = av.VideoFrame.from_ndarray(np.full((32, 64, 3), value, np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    video = store.register_asset(path, aid, "prepared depth.mp4")
    project["assets"].append(video)
    for row in shot["controlRegions"]["items"]:
        row["asset_id"] = aid
    built = generation.build_director_generation_prompt(project, shot["id"], store)
    graph = built["prompt"]
    sources = [node["inputs"] for node in graph.values() if node["class_type"] == SOURCE]
    for source in sources:
        component = source["control_video"][0]
        assert source["source_kind"] == "video" and source["source_fps"] == [component, 2]
        assert graph[component]["class_type"] == "GetVideoComponents"
        loaded = graph[graph[component]["inputs"]["video"][0]]
        assert loaded["class_type"] == "LoadVideo" and loaded["inputs"]["file"] == video["server_path"]
        assert not any(value == [component, 1] for node in graph.values() for value in node["inputs"].values())
    assert aid not in {row["asset_id"] for row in built["shot"]["media_map"]}
    registered(monkeypatch)
    import execution

    valid = asyncio.run(execution.validate_prompt("h21-control-video", graph, None))
    assert valid[0], valid[1:]


def test_model_catalog_and_batch_resolver_use_both_actual_control_folders(scene, tmp_path, monkeypatch):
    from h3_audio_t8_pkg.director_routes import _resolve_director_resource

    _, _, _, _, weights = configure(scene, tmp_path, monkeypatch, "single")
    catalog = generation.director_model_catalog()
    assert catalog["fun_control_names"] == [{"value": "unfamiliar/control.safetensors", "label": "unfamiliar/control.safetensors"}]
    assert _resolve_director_resource("fun_control_selection", "unfamiliar/control.safetensors") == str(weights)


@pytest.mark.parametrize("mode", MODES)
def test_bridge_positive_remains_paired_before_regional_controls(scene, tmp_path, monkeypatch, mode):
    from h3_audio_t8_pkg import nodes_semantic_bridge
    import execution
    import torch
    from safetensors.torch import save_file

    store, project, shot, _, _ = configure(scene, tmp_path, monkeypatch, mode, middle=False)
    weights = tmp_path / "bridge.safetensors"
    shapes = {"fc1.weight": (8, 5120), "fc1.bias": (8,), "fc2.weight": (8, 8),
              "fc2.bias": (8,), "fc3.weight": (5120, 8), "fc3.bias": (5120,)}
    save_file({"net." + key: torch.zeros(shape) for key, shape in shapes.items()}, weights,
              metadata={"arch": "mlp", "dim": "5120", "hidden": "8", "residual_skip": "True"})
    monkeypatch.setattr(nodes_semantic_bridge, "model_paths", lambda: {
        "t8_compat/semantic_bridge.safetensors": str(weights)})
    project["doc"]["d3"]["semantic_bridge"] = {"enabled": True}
    graph = generation.build_director_generation_prompt(project, shot["id"], store)["prompt"]
    for node in graph.values():
        if node["class_type"] != APPLY or graph[node["inputs"]["model"][0]]["class_type"] == APPLY:
            continue
        positive = graph[node["inputs"]["positive"][0]]
        if positive["class_type"] == "MiniMaxH3TwoPassLatentReconcileT8Advanced":
            positive = graph[positive["inputs"]["positive"][0]]
        assert positive["class_type"] == "MiniMaxH3SemanticBridgeApplyT8"
    registered(monkeypatch)
    valid = asyncio.run(execution.validate_prompt("h21-control-bridge", graph, None))
    assert valid[0], valid[1:]


def test_fast_setup_keeps_selected_sampler_and_actual_runtime_audit(scene, tmp_path, monkeypatch):
    store, project, shot, _, _ = configure(scene, tmp_path, monkeypatch, "single", middle=False)
    shot.update(mode="text", first=None, last=None)
    project["doc"]["d3"]["fast_h3_v2"] = {"enabled": True, "profile": "dense_compat_exp", "min_tokens": 8192}
    graph = generation.build_director_generation_prompt(project, shot["id"], store)["prompt"]
    setup_id, setup = next((key, node) for key, node in graph.items()
                          if node["class_type"] == "MiniMaxH3FastH3V2SetupEXPT8")
    assert graph[setup["inputs"]["model"][0]]["class_type"] == APPLY
    assert graph["7"]["inputs"]["model"] == [setup_id, 0]
    assert graph["9"]["inputs"]["sampler"] == [setup_id, 1]
    assert graph["9"]["inputs"]["sigmas"] == [setup_id, 2]
    link = graph["10"]["inputs"]["av_latent"]
    while graph[link[0]]["class_type"] == AUDIT:
        link = graph[link[0]]["inputs"]["after_sampling"]
    assert graph[link[0]]["class_type"] == "MiniMaxH3FastH3V2RuntimeAuditEXPT8"
    assert graph[link[0]]["inputs"]["sampled_av_latent"] == ["9", 0]


@pytest.mark.parametrize("resume", [False, True])
def test_editable_export_keeps_actual_control_and_typed_head_save_dependencies(scene, tmp_path, monkeypatch, resume):
    from h3_audio_t8_pkg.director_workflow_export import export_director_split_workflow
    from h3_audio_t8_pkg.modular_sampling import hyperflow_storage
    import nodes

    store, project, shot, _, _ = configure(scene, tmp_path, monkeypatch, "continuous4plus4separate")
    checkpoint = {"mode": "save"}
    if resume:
        digest = "a" * 64
        monkeypatch.setattr(hyperflow_storage, "fingerprint", lambda *_: digest)
        checkpoint = {"mode": "resume_tail", "artifact_path": "frozen/head.safetensors", "artifact_sha256": digest}
    project["doc"]["sampling"]["stage_checkpoint"] = checkpoint
    registered(monkeypatch)
    result = export_director_split_workflow(project, shot["id"], store, registry=nodes.NODE_CLASS_MAPPINGS)
    graph = result["api_snapshot"]
    assert len(result["workflow"]["nodes"]) == len(graph)
    assert len([node for node in graph.values() if node["class_type"] == APPLY]) == (2 if resume else 4)
    assert len([node for node in graph.values() if node["class_type"] == HEAD_AUDIT]) == (0 if resume else 2)
    assert closure(graph) == set(graph)
    if not resume:
        save = next(node for node in graph.values() if node["class_type"] == "MiniMaxH3HyperFlowHeadSaveEXPT8")
        assert graph[save["inputs"]["continuous_boundary"][0]]["class_type"] == HEAD_AUDIT
