"""Actual CPU raw API graph validation, not trained model execution evidence."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import uuid

from PIL import Image
import pytest

from test_h21_control_probe import CORE, probe, tiny_header


@pytest.fixture(scope="module")
def raw_config(tmp_path_factory):
    root = tmp_path_factory.mktemp("h21-raw-api")
    roots = {}
    for category in probe.MODEL_FOLDERS:
        folder = root / "models" / category
        folder.mkdir(parents=True)
        roots[category] = [str(folder)]
    selected = {"diffusion_models": "unlisted-raw.safetensors",
                "text_encoders": "raw-clip.safetensors", "vae": "raw-video.safetensors"}
    for category, name in selected.items():
        tiny_header(Path(roots[category][0]) / name)
    source = root / "owned.png"
    Image.new("RGB", (13, 17), "red").save(source)
    asset = str(uuid.uuid4())
    graph = {
        "1": {"class_type": "UNETLoader", "inputs": {
            "unet_name": selected["diffusion_models"], "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {
            "clip_name": selected["text_encoders"], "type": "minimax", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": selected["vae"]}},
        "4": {"class_type": "LoadImage", "inputs": {"image": "original-owned.png"}},
        "5": {"class_type": "ImageFromBatch", "inputs": {
            "image": ["4", 0], "batch_index": 0, "length": 1}},
        "6": {"class_type": "PreviewAny", "inputs": {"source": ["5", 0]}},
        "7": {"class_type": "MiniMaxH3AudioConditioningT8", "inputs": {
            "clip": ["2", 0], "video_vae": ["3", 0], "audio_vae": ["3", 0],
            "prompt": "An empty room with quiet ambient sound.",
            "width": 512, "height": 288, "length": 124,
            "task_type": "T2VA", "audio_mode": "native", "audio_denoise_strength": .35,
            "add_source_as_reference": True, "prompt_primary_audio_ordinal": 0,
            "strict_prompt_tags": True, "ref_image_size": "match",
            "reference_video_policy": "official_2_to_15s"}},
        "8": {"class_type": "MiniMaxH3DualClockSamplerT8", "inputs": {
            "model": ["1", 0], "av_latent": ["7", 1], "steps": 40,
            "shift_video": 12, "shift_audio": 3,
            "sampler_name": "dual_clock_euler", "scheduler": "native_flow"}},
        "9": {"class_type": "PreviewAny", "inputs": {"source": ["8", 2]}},
    }
    api_file = root / "original.json"
    api_file.write_text(json.dumps(graph), encoding="utf-8")
    config = {"schema": "t8.h21.raw_api_config.v1", "mode": "raw_api",
              "model_directories": roots, "assets": {
                  asset: {"source": str(source), "sha256": probe.file_identity(source)["sha256"],
                          "private_relative": "raw/" + asset + "/source.png",
                          "nodes": [{"node": "4", "input": "image"}]}},
              "models": [
                  {"node": "1", "input": "unet_name", "folder": "diffusion_models",
                   "name": selected["diffusion_models"]},
                  {"node": "2", "input": "clip_name", "folder": "text_encoders",
                   "name": selected["text_encoders"]},
                  {"node": "3", "input": "vae_name", "folder": "vae", "name": selected["vae"]}],
              "raw_api_cases": [{"name": "native40", "api": str(api_file),
                                 "api_sha256": probe.file_identity(api_file)["sha256"]}]}
    path = root / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return root, path, config, graph


@pytest.fixture(scope="module")
def raw_prepared(raw_config):
    root, config_file, config, original = raw_config
    run = root / "owned-prepared"
    result = probe.prepare(config_file, run, CORE, python=sys.executable)
    return run, result, config, original


def test_raw_actual_core_graph_preserves_real40_and_freezes_original_bytes(raw_prepared):
    run, plan, config, original = raw_prepared
    assert plan["graph_origin"] == "explicit_raw_api"
    assert plan["model_execution"] is False and plan["cuda_initialized"] is False
    assert not (run / "source-project.json").exists()
    case = plan["cases"][0]
    assert probe.read_json(run / case["validation"])["valid"] is True
    assert (run / case["original_api"]).read_bytes() == Path(config["raw_api_cases"][0]["api"]).read_bytes()
    graph = probe.read_json(run / case["base_prompt"])
    expected = deepcopy(original)
    expected["4"]["inputs"]["image"] = next(iter(config["assets"].values()))["private_relative"]
    assert graph == expected and graph["8"]["inputs"]["steps"] == 40
    assert probe.verify_prepared(run / "manifest.json")["run_id"] == plan["run_id"]
    assert not (run / "attempts").exists()
    assert not plan["gpu_started"] and plan["project_identity"] is None


def test_raw_resolved_unknown_selected_names_are_hashed_not_replaced(raw_prepared):
    _, plan, _, _ = raw_prepared
    captured = plan["resources"]["models"]
    assert len(captured) == 3
    for row in captured.values():
        real = probe.file_identity(row["path"])
        assert real["sha256"] == row["sha256"] and real["bytes"] == row["size"]


@pytest.mark.parametrize("field", ["sampling", "steps", "project", "cases"])
def test_unapplied_dummy_director_or_steps_fields_rejected(raw_config, field):
    config = deepcopy(raw_config[2])
    config[field] = {} if field != "steps" else 8
    with pytest.raises(ValueError, match="unapplied"):
        probe._raw_api_helper().validate_config(config)


def test_case_sha_is_required_without_reading_or_running_graph(raw_config):
    config = deepcopy(raw_config[2])
    config["raw_api_cases"][0].pop("api_sha256")
    with pytest.raises(ValueError, match="full SHA"):
        probe._raw_api_helper().validate_config(config)


def test_asset_identity_is_explicit_not_invented(raw_config):
    config = deepcopy(raw_config[2])
    config["assets"]["not-a-uuid"] = config["assets"].pop(next(iter(config["assets"])))
    with pytest.raises(ValueError, match="UUID"):
        probe._raw_api_helper().validate_config(config)


@pytest.mark.parametrize("mutation, expected", [
    ("graph_sha", "Raw graph source SHA differs"),
    ("asset_sha", "Raw asset source SHA differs"),
    ("model_resolution", "resolves differently"),
    ("missing_model_binding", "lacks an explicit resource binding"),
    ("escaping_asset", "relative path"),
])
def test_actual_cpu_prepare_rejects_corrupt_or_misbound_raw_evidence_without_gpu(
        raw_config, mutation, expected):
    root, _, original, _ = raw_config
    config = deepcopy(original)
    if mutation == "graph_sha":
        config["raw_api_cases"][0]["api_sha256"] = "0" * 64
    elif mutation == "asset_sha":
        next(iter(config["assets"].values()))["sha256"] = "0" * 64
    elif mutation == "model_resolution":
        config["models"][0]["folder"] = "loras"
        tiny_header(Path(config["model_directories"]["loras"][0]) / config["models"][0]["name"])
    elif mutation == "missing_model_binding":
        config["models"].pop(0)
    else:
        next(iter(config["assets"].values()))["private_relative"] = "../outside.png"
    config_file = root / (mutation + ".json")
    config_file.write_text(json.dumps(config), encoding="utf-8")
    run = root / mutation
    with pytest.raises(RuntimeError, match="CPU graph preparation failed"):
        probe.prepare(config_file, run, CORE, python=sys.executable)
    assert expected in (run / "prepare.stderr.log").read_text(encoding="utf-8")
    assert not (run / "manifest.json").exists() and not (run / "attempts").exists()
    assert not (root / "outside.png").exists()
