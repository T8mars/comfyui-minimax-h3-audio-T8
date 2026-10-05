"""Builder mechanical contracts only; actual native media proof is separate."""
import json

import pytest

from tools.build_reference_asset_workflow import CREATE, SAVE, make_workflow as encode_workflow
from tools.build_reference_readback_workflow import LOAD, ROUTE, ROLES, make_workflow as readback_workflow


def definition(inputs, outputs):
    return {"input": {"required": inputs}, "input_order": {"required": list(inputs)},
            "output": outputs, "output_name": outputs}


def info(*, v3_combo=False):
    def combo(values):
        return ["COMBO", {"options": values}] if v3_combo else [values, {}]
    return {
        "LoadVideo": definition({"file": combo(["master.mp4"])}, ["VIDEO"]),
        "GetVideoComponents": definition({"video": ["VIDEO", {}]}, ["IMAGE", "AUDIO", "FLOAT"]),
        "ImageFromBatch": definition({"image": ["IMAGE", {}], "batch_index": ["INT", {}],
            "length": ["INT", {}]}, ["IMAGE"]),
        "VAELoader": definition({"vae_name": combo(["video.safetensors", "audio.safetensors"])}, ["VAE"]),
        CREATE: {**definition({"role_id": ["STRING", {}], "kind": combo(["image", "video", "audio"]),
            "width": ["INT", {}], "height": ["INT", {}], "frame_limit": ["INT", {}]},
            ["T8_H3_REFERENCE_PACKAGE", "STRING"]),
            "input": {"required": {"role_id": ["STRING", {}], "kind": combo(["image", "video", "audio"]),
                "width": ["INT", {}], "height": ["INT", {}], "frame_limit": ["INT", {}]},
                "optional": {"frames": ["IMAGE", {}], "audio": ["AUDIO", {}],
                    "video_vae": ["VAE", {}], "audio_vae": ["VAE", {}]}}},
        SAVE: definition({"reference_package": ["T8_H3_REFERENCE_PACKAGE", {}],
            "filename": ["STRING", {}], "confirm_save": ["BOOLEAN", {}]},
            ["T8_H3_REFERENCE_PACKAGE", "STRING", "STRING", "STRING"]),
        LOAD: definition({"filename": combo(["voice.safetensors", "image.safetensors"]),
            "expected_sha256": ["STRING", {}]}, ["T8_H3_REFERENCE_PACKAGE", "STRING"]),
        ROUTE: definition({"packages": ["COMFY_AUTOGROW_V3", {"template": {
            "prefix": "package_", "min": 1, "max": 15, "input": {
                "required": {"package": ["T8_H3_REFERENCE_PACKAGE", {}]}}}}],
            "roles_json": ["STRING", {}]}, ["T8_H3_REFERENCE_SET", "STRING"]),
        "PreviewAny": definition({"source": ["*", {}]}, ["STRING"]),
    }


@pytest.mark.parametrize('v3_combo', [False, True])
def test_encode_builder_requires_explicit_assets_and_preserves_no_sampler_contract(v3_combo):
    inputs = info(v3_combo=v3_combo)
    args = dict(source_video="master.mp4", video_vae="video.safetensors", audio_vae="audio.safetensors",
                voice_filename="A.safetensors", image_filename="B.safetensors")
    workflow, graph = encode_workflow(inputs, **args)
    assert len(workflow["nodes"]) == 10
    assert graph["6"]["inputs"]["audio"] == ["2", 1]
    assert graph["7"]["inputs"]["frames"] == ["3", 0]
    assert graph["8"]["inputs"]["confirm_save"] is graph["9"]["inputs"]["confirm_save"] is False
    assert all("sampl" not in node["class_type"].lower() for node in graph.values())
    assert workflow["extra"]["radar_r6_reference_assets"]["pretrained_runtime_verified"] is False
    for change in ({"source_video": "unknown.mp4"}, {"audio_vae": "unknown.safetensors"},
                   {"width": 33}, {"confirm": 1}, {"image_filename": "A.safetensors"},
                   {"voice_filename": "../A.safetensors"}):
        with pytest.raises(ValueError):
            encode_workflow(inputs, **{**args, **change})


@pytest.mark.parametrize('v3_combo', [False, True])
def test_readback_builder_preserves_native_autogrow_links_sha_and_unconfirmed_save(v3_combo):
    inputs = info(v3_combo=v3_combo)
    args = dict(voice_name="voice.safetensors", image_name="image.safetensors", voice_sha="a"*64, image_sha="b"*64)
    workflow, graph = readback_workflow(inputs, **args)
    route = next(node for node in workflow["nodes"] if node["type"] == ROUTE)
    sockets = {entry["name"]: entry for entry in route["inputs"]}
    for key in ("packages.package_0", "packages.package_1"):
        assert sockets[key]["link"] is not None and "widget" not in sockets[key]
        assert sockets[key]["type"] == "T8_H3_REFERENCE_PACKAGE"
    assert route["widgets_values"] == [ROLES]
    assert json.loads(ROLES)[0] == {"role_id": "A", "visual": False, "voice": True}
    assert graph["3"]["inputs"]["confirm_save"] is graph["4"]["inputs"]["confirm_save"] is False
    assert graph["1"]["inputs"]["expected_sha256"] == "a"*64
    assert len(workflow["links"]) == 5
    for change in ({"voice_name": "missing.safetensors"}, {"voice_sha": ""}, {"image_name": "voice.safetensors"}):
        with pytest.raises(ValueError):
            readback_workflow(inputs, **{**args, **change})
