"""Fast single-stage EXP preserves real non-text conditions and selected recipe.

Graph files are real registered media; model-name fixtures are not trained
weights. Separate actual random tiny Core sampler cases test native consumption.
No GPU, user service, external browser or inference queue is used.
"""

import asyncio
from copy import deepcopy
import json
import uuid

import pytest
import torch

from h3_audio_t8_pkg import director_generation as generation
from h3_audio_t8_pkg import fast_h3_v2_advanced as fast
from h3_audio_t8_pkg.director_project import compile_project, new_project
from test_director_generation import _audio, _image, _patch_director_models, _store
from test_h21_director_control_graph import closure, registered


KINDS = (
    "first",
    "ends",
    "refs",
    "hybrid",
    "record",
    "voice",
    "middle",
    "record_middle",
)


def graph_case(
    tmp_path,
    monkeypatch,
    kind,
    profile="dense_compat_exp",
    *,
    relay=False,
    control=False,
):
    store = _store(tmp_path)
    images = [
        _image(store, name) for name in ("first.png", "last.png", "reference.png")
    ]
    project = new_project()
    project["assets"] = images.copy()
    shot = project["doc"]["shots"][0]
    shot.update(
        mode="first",
        first=images[0]["id"],
        simplePrompt="A person turns naturally.",
        autoDuration=False,
        manualDuration=4,
        duration=4,
        d3Inherit=False,
    )
    if kind in {"ends", "middle"}:
        shot.update(mode="ends", last=images[1]["id"])
    elif kind == "refs":
        shot.update(mode="refs", first=None, refs=[images[2]["id"]])
    elif kind == "hybrid":
        shot["refs"] = [images[2]["id"]]
    if kind in {"record", "voice", "record_middle"}:
        audio = _audio(store)
        project["assets"].append(audio)
        shot.update(
            sound="voice" if kind == "voice" else "record",
            audio=audio["id"],
            start=0.0,
            end=4.0,
        )
    if kind in {"middle", "record_middle"}:
        shot["middleKeyframes"] = {
            "version": 1,
            "items": [
                {
                    "id": str(uuid.uuid4()),
                    "asset_id": images[2]["id"],
                    "position_mode": "percent",
                    "position": 50,
                    "enabled": True,
                    "visual_noise_aug": 0.91,
                    "resize_mode": "stretch",
                }
            ],
        }
    if control:
        shot["controlRegions"] = {
            "version": 1,
            "items": [
                {
                    "id": str(uuid.uuid4()),
                    "asset_id": images[1]["id"],
                    "control_net_name": "user-control.safetensors",
                    "enabled": True,
                    "strength": -3.5,
                    "region": {"mode": "rect", "xyxy_normalized": [0.1, 0.2, 0.7, 0.9]},
                }
            ],
        }
    shot["d3"] = {
        "fast_h3_v2": {"enabled": True, "profile": profile, "min_tokens": 8192},
        "prompt_relay": {"enabled": relay, "execution_mode": "apply_exp"},
    }
    _patch_director_models(monkeypatch)
    names = {
        "diffusion_models": [fast.MODEL_FILE, "user-other-model.safetensors"],
        "vae": [
            "minimax_h3_video_vae_fp16.safetensors",
            "minimax_h3_audio_vae_fp32.safetensors",
        ],
        "clip": ["qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"],
        "text_encoders": ["qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"],
        "model_patches": ["user-control.safetensors"],
        "controlnet": [],
        "loras": ["user-a.safetensors", "user-b.safetensors"],
    }
    weights = tmp_path / "schema-only-not-inference-weights.bin"
    weights.write_bytes(b"Only catalog/schema proof, not executable model weights")
    monkeypatch.setattr(
        generation.folder_paths,
        "get_filename_list",
        lambda folder: names.get(folder, []),
    )
    monkeypatch.setattr(
        generation.folder_paths,
        "get_full_path",
        lambda folder, name: str(weights) if name in names.get(folder, []) else None,
    )
    monkeypatch.setattr(
        generation.folder_paths, "get_input_directory", lambda: str(store.input_root)
    )
    return store, project, shot


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("profile", fast.PROFILES)
def test_nontext_fast_graph_warns_exp_without_erasing_inputs_or_changing_sampler(
    tmp_path,
    monkeypatch,
    kind,
    profile,
):
    import execution

    store, project, shot = graph_case(tmp_path, monkeypatch, kind, profile)
    before = deepcopy(project)
    preflight = compile_project(project, store, shot_id=shot["id"])
    assert preflight["ready"], preflight["errors"]
    built = generation.build_director_generation_prompt(project, shot["id"], store)
    graph = built["prompt"]
    setup_id, setup = next(
        (key, node)
        for key, node in graph.items()
        if node["class_type"] == "MiniMaxH3FastH3V2SetupEXPT8"
    )
    assert (
        setup["inputs"]["profile"] == profile and setup["inputs"]["min_tokens"] == 8192
    )
    assert graph["7"]["inputs"]["model"] == [setup_id, 0]
    assert graph["9"]["inputs"]["sampler"] == [setup_id, 1]
    assert graph["9"]["inputs"]["sigmas"] == [setup_id, 2]
    assert graph["1"]["inputs"]["unet_name"] == fast.MODEL_FILE
    assert not any(
        node["class_type"] == "MiniMaxH3DualClockSamplerT8" for node in graph.values()
    )
    condition_id, condition = next(
        (key, node)
        for key, node in graph.items()
        if node["class_type"]
        in {
            "MiniMaxH3AudioConditioningT8",
            "MiniMaxH3DirectorKeyframeConditioningEXPT8",
        }
    )
    expected_type = {
        "first": "I2VA",
        "ends": "FL2VA",
        "refs": "Ref2VA",
        "hybrid": "Hybrid",
        "record": "Hybrid",
        "voice": "Hybrid",
        "middle": "FL2VA",
        "record_middle": "Hybrid",
    }[kind]
    assert condition["inputs"]["task_type"] == expected_type
    assert ("first_frame" in condition["inputs"]) is (kind != "refs")
    assert ("last_frame" in condition["inputs"]) is (kind in {"ends", "middle"})
    if kind in {"refs", "hybrid"}:
        assert "ref_images.ref_image_0" in condition["inputs"]
    if kind == "voice":
        assert "ref_audios.ref_audio_0" in condition["inputs"]
    if kind in {"middle", "record_middle"}:
        assert setup["inputs"]["model"] == [condition_id, 0]
        assert graph["7"]["inputs"]["conditioning"] == [condition_id, 1]
        assert graph["9"]["inputs"]["latent_image"] == [condition_id, 2]
        assert (
            graph[condition["inputs"]["keyframe_plan"][0]]["inputs"]["visual_noise_aug"]
            == 0.91
        )
    if kind in {"record", "record_middle"}:
        assert condition["inputs"]["audio_mode"] == "lock_source"
        assert condition["inputs"]["drive_audio"] == condition["inputs"]["final_audio"]
        assert graph["11"]["inputs"]["audio"] == [
            condition_id,
            3 if "middle" in kind else 2,
        ]
        assert (
            graph["11"]["inputs"]["start_seconds"][0]
            == condition["inputs"]["drive_audio"][0]
        )
    else:
        assert graph["11"]["inputs"]["audio"] == ["10", 1]
    assert any(
        "FastH3 V2" in row["message"] and "EXP" in row["message"]
        for row in built["report"]["warnings"]
    )
    assert closure(graph) == set(graph)
    assert project == before
    registered(monkeypatch)
    import nodes
    from comfy_extras.nodes_audio import LoadAudio

    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "LoadAudio", LoadAudio)
    result = asyncio.run(
        execution.validate_prompt("h21-fast-nontext-" + kind, graph, None)
    )
    assert result[0], result[1:]


@pytest.mark.parametrize("kind", ("ends", "refs", "record_middle"))
def test_explicit_dense_relay_controls_and_ordered_loras_stay_paired(
    tmp_path, monkeypatch, kind
):
    store, project, shot = graph_case(
        tmp_path, monkeypatch, kind, relay=True, control=True
    )
    project["doc"]["generation"].update(
        unet="user-other-model.safetensors",
        lora_mode="manual",
        loras=[
            {"name": "user-a.safetensors", "enabled": True, "strength": 1.4},
            {"name": "user-b.safetensors", "enabled": True, "strength": -0.2},
        ],
    )
    before = deepcopy(project)
    built = generation.build_director_generation_prompt(project, shot["id"], store)
    graph = built["prompt"]
    setup = next(
        node
        for node in graph.values()
        if node["class_type"] == "MiniMaxH3FastH3V2SetupEXPT8"
    )
    control = graph[setup["inputs"]["model"][0]]
    assert control["class_type"] == "MiniMaxH3FunControlRegionApplyEXPT8"
    assert graph["7"]["inputs"]["conditioning"] == [setup["inputs"]["model"][0], 1]
    condition_id = control["inputs"]["model"][0]
    condition = graph[condition_id]
    assert condition["class_type"] == (
        "MiniMaxH3DirectorKeyframeConditioningEXPT8"
        if "middle" in kind
        else "MiniMaxH3PromptRelayConditioningT8Advanced"
    )
    assert control["inputs"]["positive"] == [condition_id, 1]
    assert "prompt_relay_plan" in condition["inputs"]
    loras = [
        node["inputs"]
        for node in graph.values()
        if node["class_type"] == "MiniMaxH3LoRACompatibilityLoaderT8Advanced"
    ]
    assert [(row["lora_name"], row["strength_model"]) for row in loras] == [
        ("user-a.safetensors", 1.4),
        ("user-b.safetensors", -0.2),
    ]
    assert graph["1"]["inputs"]["unet_name"] == "user-other-model.safetensors"
    assert project == before


def test_actual_reference_video_components_reach_fast_conditioning_and_core_validation(
    tmp_path, monkeypatch
):
    import av
    import execution
    import numpy as np

    store, project, shot = graph_case(tmp_path, monkeypatch, "refs", "trained_vsa_exp")
    aid = str(uuid.uuid4())
    path = store.input_root / "t8_director" / aid / "source.mp4"
    path.parent.mkdir(parents=True)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=12)
        stream.width, stream.height, stream.pix_fmt = 64, 32, "yuv420p"
        stream.codec_context.thread_count = 1
        for index in range(24):
            frame = av.VideoFrame.from_ndarray(
                np.full((32, 64, 3), index * 10, np.uint8), format="rgb24"
            )
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    video = store.register_asset(path, aid, "actual-reference.mp4")
    project["assets"].append(video)
    shot["refs"] = [aid]
    built = generation.build_director_generation_prompt(project, shot["id"], store)
    graph = built["prompt"]
    condition = graph["5"]["inputs"]
    component_id = condition["ref_videos.ref_video_0"][0]
    component = graph[component_id]
    assert component["class_type"] == "GetVideoComponents"
    assert (
        graph[component["inputs"]["video"][0]]["inputs"]["file"] == video["server_path"]
    )
    assert condition["task_type"] == "Ref2VA" and graph["11"]["inputs"]["audio"] == [
        "10",
        1,
    ]
    assert any(
        node["class_type"] == "MiniMaxH3FastH3V2SetupEXPT8" for node in graph.values()
    )
    registered(monkeypatch)
    checked = asyncio.run(execution.validate_prompt("h21-fast-ref-video", graph, None))
    assert checked[0], checked[1:]


@pytest.mark.parametrize("mode", ("two_pass", "hyperflow"))
def test_real_sampler_protocol_conflicts_remain_errors(tmp_path, monkeypatch, mode):
    store, project, shot = graph_case(tmp_path, monkeypatch, "first")
    project["doc"]["sampling"] = {
        "mode": mode,
        **(
            {"hyperflow_file": "hyperflow/hf.safetensors"}
            if mode == "hyperflow"
            else {}
        ),
    }
    with pytest.raises(ValueError, match="4\\+4 双采|SAMPLER/SIGMAS"):
        generation.build_director_generation_prompt(project, shot["id"], store)


def test_trained_vsa_multi_event_bias_conflict_is_not_silently_disabled(
    tmp_path, monkeypatch
):
    store, project, shot = graph_case(
        tmp_path, monkeypatch, "ends", "trained_vsa_exp", relay=True
    )
    shot.update(
        writingMode="advanced",
        simplePrompt="",
        events=[
            {"id": str(uuid.uuid4()), "start": 0, "end": 2, "text": "Turns left."},
            {"id": str(uuid.uuid4()), "start": 2, "end": 4, "text": "Turns right."},
        ],
    )
    before = deepcopy(project)
    with pytest.raises(ValueError, match="训练 VSA 无法执行 Prompt Relay"):
        generation.build_director_generation_prompt(project, shot["id"], store)
    assert project == before


@pytest.mark.parametrize("damage", ("missing_first", "missing_ref", "missing_audio"))
def test_actual_missing_media_is_not_converted_into_an_experimental_warning(
    tmp_path, monkeypatch, damage
):
    kind = {"missing_first": "first", "missing_ref": "refs", "missing_audio": "record"}[
        damage
    ]
    store, project, shot = graph_case(tmp_path, monkeypatch, kind)
    shot[
        {"missing_first": "first", "missing_ref": "refs", "missing_audio": "audio"}[
            damage
        ]
    ] = [] if damage == "missing_ref" else None
    with pytest.raises(ValueError, match="预检未通过"):
        generation.build_director_generation_prompt(project, shot["id"], store)


@pytest.mark.parametrize(
    "task", ("i2va", "fl2va", "ref2va", "hybrid", "middle", "record_middle")
)
@pytest.mark.parametrize("profile", ("dense_compat_exp", "trained_vsa_exp"))
def test_actual_tiny_fast_sampler_consumes_nontext_conditioning_without_new_rungs(
    monkeypatch, task, profile
):
    from comfy.ldm.minimax import model as native
    from comfy.ldm.modules import attention
    from comfy_extras.nodes_custom_sampler import (
        BasicGuider,
        RandomNoise,
        SamplerCustomAdvanced,
    )
    from h3_audio_t8_pkg.multikeyframe_advanced import (
        append_keyframe_plan,
        build_multikeyframe_conditioning,
    )
    from test_fast_h3_v2_core_sampler import model
    from test_h21_control_regions import RealTinyEncoderOwner
    from test_prompt_relay_advanced import NativeLikeFakeClip
    from helpers import FakeAudioVAE, make_audio

    monkeypatch.setattr(native, "optimized_attention", attention.attention_pytorch)
    bare = model()
    image = torch.linspace(0.05, 0.95, 32 * 32 * 3).reshape(1, 32, 32, 3)
    inputs = dict(
        model=bare,
        clip=NativeLikeFakeClip(),
        video_vae=RealTinyEncoderOwner(),
        audio_vae=FakeAudioVAE(),
        prompt="A person turns.",
        width=32,
        height=32,
        length=22 if "middle" in task else 5,
        task_type="auto"
        if task == "record_middle"
        else "FL2VA"
        if task == "middle"
        else task,
        audio_mode="native",
        audio_denoise_strength=0.35,
        add_source_as_reference=False,
        prompt_primary_audio_ordinal=0,
        strict_prompt_tags=True,
        ref_image_size="match",
        reference_video_policy="official_2_to_15s",
    )
    if task in {"i2va", "fl2va", "hybrid", "middle", "record_middle"}:
        inputs["first_frame"] = image
    if task in {"fl2va", "middle"}:
        inputs["last_frame"] = image * 0.8
    if task in {"ref2va", "hybrid"}:
        inputs["ref_images"] = {"ref_image_0": image * 0.7}
    if "middle" in task:
        inputs["keyframe_plan"] = append_keyframe_plan(
            None, image * 0.9, "frame", 1, 0.91, "stretch", True
        )
    if task == "record_middle":
        inputs.update(
            audio_mode="lock_source",
            audio_denoise_strength=0.0,
            add_source_as_reference=True,
            drive_audio=make_audio(22 / 24, value=0.13),
            final_audio=make_audio(22 / 24, value=0.23),
        )
    selected, positive, source = build_multikeyframe_conditioning(**inputs)[:3]
    if "middle" in task:
        assert len(positive[0][1]["minimax_keyframes"]) == (
            2 if task == "record_middle" else 3
        )
    if task == "record_middle":
        assert source["noise_mask"].unbind()[1].count_nonzero() == 0
    originals = [part.clone() for part in source["samples"].unbind()]
    prepared, sampler, sigmas, text = fast.build_fast_h3_v2_setup(
        selected, source, profile, 8192
    )
    calls = []
    hook = prepared.model.diffusion_model.register_forward_pre_hook(
        lambda *_: calls.append(1)
    )
    try:
        noise = RandomNoise.execute(211021401).result[0]
        guider = BasicGuider.execute(prepared, positive).result[0]
        output = SamplerCustomAdvanced.execute(
            noise, guider, sampler, sigmas, source
        ).result[0]
    finally:
        hook.remove()
    assert len(calls) == 8 and torch.equal(sigmas, fast.dmd_sigmas())
    assert all(torch.isfinite(part).all() for part in output["samples"].unbind())
    assert all(
        torch.equal(before, after)
        for before, after in zip(originals, source["samples"].unbind(), strict=True)
    )
    assert json.loads(text)["nfe"] == 8 and "EXP" in json.loads(text)["trained_scope"]
    assert fast.capture_fast_h3_v2_owner(bare) is None and not bare.object_patches
    assert not torch.cuda.is_initialized()
