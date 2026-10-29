"""Visible delivery coordinates through compiler, native payload and stage graphs."""

import json
import math
import uuid
import wave

import pytest

from h3_audio_t8_pkg.director_generation import build_director_generation_prompt
from h3_audio_t8_pkg.director_keyframes import FIELD
from h3_audio_t8_pkg.director_project import compile_project
from h3_audio_t8_pkg.multikeyframe_advanced import build_multikeyframe_conditioning
from h3_audio_t8_pkg.nodes_director_keyframes import (
    MiniMaxH3DirectorKeyframeConditioningEXPT8,
)
from h3_audio_t8_pkg.timing import make_timing_plan
from test_director_generation import _patch_director_models
import test_h21_director_keyframes as keyframe_cases

from test_multikeyframe_advanced import make_image, make_model_patcher
from helpers import FakeAudioVAE, FakeClip, FakeVideoVAE

scene, row, plan = keyframe_cases.scene, keyframe_cases.row, keyframe_cases.plan


def recording(store, seconds=8):
    aid = str(uuid.uuid4())
    path = store.input_root / "t8_director" / aid / "source.wav"
    path.parent.mkdir(parents=True)
    with wave.open(str(path), "wb") as output:
        output.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
        output.writeframes(bytes(seconds * 8000 * 2))
    return store.register_asset(path, aid, "source.wav")


@pytest.mark.parametrize("seconds", [1.0, 4.0, 5.1, 5.1667, 5.2])
@pytest.mark.parametrize("sound", ["native", "record"])
def test_visible_endpoints_share_actual_audio_window_clock(scene, seconds, sound):
    store, project, shot, images = scene
    shot.update(manualDuration=seconds, sound=sound)
    if sound == "record":
        audio = recording(store)
        project["assets"].append(audio)
        shot.update(audio=audio["id"], start=0.25, end=0.25 + seconds)
        expected = make_timing_plan(
            0.25, seconds, ensure_minimum_context=True, source_duration_seconds=8
        )
        start = round(expected.final_trim_start_seconds * 24)
        generated = expected.frame_count
    else:
        start = 0
        requested = max(5, math.ceil(seconds * 24))
        generated = requested + ((5 - requested) % 17)
    visible = math.ceil(seconds * 24)
    # A generated-canvas middle guide deliberately lies AFTER the visible tail.
    hidden = generated - 2
    shot[FIELD] = plan([row(images[2]["id"], hidden)], "generated_canvas")
    compiled = compile_project(project, store)
    assert compiled["ready"], compiled["errors"]
    result = compiled["shots"][0]
    assert result["time"]["aligned_frames"] == generated
    assert result["time"]["delivery_start_frame"] == start
    assert result["time"]["delivery_end_frame"] == start + visible
    physical = sorted(
        [
            (start, "first_frame"),
            (hidden, "middle_keyframe"),
            (start + visible - 1, "last_frame"),
        ],
        key=lambda x: x[0],
    )
    assert [
        (m["role"], m["native"])
        for m in result["media_map"]
        if m["native"].startswith("<Picture ")
    ] == [(role, f"<Picture {i}>") for i, (_, role) in enumerate(physical, 1)]
    assert result["middle_keyframes"][0]["frame_index"] == hidden
    assert result["middle_keyframes"][0]["visible_in_delivery"] == (
        start <= hidden < start + visible
    )


def test_record_visible_middle_is_shifted_by_render_padding_not_source_offset(scene):
    store, project, shot, images = scene
    audio = recording(store)
    project["assets"].append(audio)
    shot.update(
        sound="record", audio=audio["id"], start=2.0, end=3.0, manualDuration=1.0
    )
    shot[FIELD] = plan([row(images[2]["id"], 12)])
    compiled = compile_project(project, store)["shots"][0]
    expected = make_timing_plan(2.0, 1.0, source_duration_seconds=8)
    offset = round(expected.final_trim_start_seconds * 24)
    assert compiled["middle_keyframes"][0]["frame_index"] == offset + 12
    assert compiled["time"]["delivery_start_frame"] == offset != round(2 * 24)


@pytest.mark.parametrize(
    "route",
    [
        "single",
        "two_pass",
        "single8",
        "continuous4plus4",
        "continuous4plus4separate",
        "upscale8plus4",
        "upscale4plus4",
    ],
)
@pytest.mark.parametrize("relay", [False, True])
@pytest.mark.parametrize("record", [False, True])
def test_each_fresh_stage_factory_receives_visible_endpoint_without_changing_steps(
    scene, tmp_path, monkeypatch, route, relay, record
):
    from h3_audio_t8_pkg import director_generation, director_hyperflow

    store, project, shot, images = scene
    _patch_director_models(monkeypatch)
    monkeypatch.setattr(
        director_generation.folder_paths,
        "get_filename_list",
        lambda folder: {
            "diffusion_models": ["minimax_h3_fl2va_int8_convrot.safetensors"],
            "latent_upscale_models": ["minimax_h3_latent_upscaler_3d_fp16.safetensors"],
        }.get(folder, []),
    )
    monkeypatch.setattr(
        director_generation.folder_paths, "get_full_path", lambda folder, name: name
    )
    hf = tmp_path / "hf.safetensors"
    hf.write_bytes(b"probe-only")
    monkeypatch.setattr(director_hyperflow, "_resolve", lambda selection: hf)
    project["doc"]["generation"]["unet_name"] = (
        "minimax_h3_fl2va_int8_convrot.safetensors"
    )
    if record:
        audio = recording(store)
        project["assets"].append(audio)
        shot.update(sound="record", audio=audio["id"], start=0.25, end=4.25)
    shot[FIELD] = plan([])  # Tail alone must select the correctly paired wrapper.
    project["doc"]["d3"]["prompt_relay"]["enabled"] = relay
    if route in ("single", "two_pass"):
        project["doc"]["generation"]["sampling"] = {
            "mode": route,
            "output_mp": ".4",
            "upscaler": "auto",
        }
    else:
        project["doc"]["generation"]["sampling"] = {
            "mode": "hyperflow",
            "variant": route,
            "hyperflow_file": hf.name,
            "output_mp": ".4",
            "upscaler": "auto",
        }
    built = build_director_generation_prompt(project, shot["id"], store)
    graph = built["prompt"]
    factories = [
        n
        for n in graph.values()
        if n["class_type"] == "MiniMaxH3DirectorKeyframeConditioningEXPT8"
    ]
    assert factories
    for factory in factories:
        assert (
            factory["inputs"]["first_frame_index"]
            == built["shot"]["time"]["delivery_start_frame"]
        )
        assert (
            factory["inputs"]["last_frame_index"]
            == built["shot"]["time"]["delivery_end_frame"] - 1
        )
        length = factory["inputs"]["length"]
        if isinstance(length, list):
            window = graph[length[0]]["inputs"]
            length = make_timing_plan(
                window["scene_start_seconds"], window["scene_duration_seconds"]
            ).frame_count
        assert length == built["shot"]["time"]["aligned_frames"]
    assert sum(n["class_type"] == "MiniMaxH3AVDecodeT8" for n in graph.values()) == 1
    assert graph["11"]["inputs"]["frame_count"] == 96


@pytest.mark.parametrize("first_index,last_index", [(0, 95), (13, 95)])
def test_explicit_endpoint_is_encoded_in_actual_native_payload(first_index, last_index):
    first, last = make_image(0.2, 32, 64), make_image(0.8, 32, 64)
    built = build_multikeyframe_conditioning(
        make_model_patcher(),
        FakeClip(),
        FakeVideoVAE(),
        FakeAudioVAE(),
        "A continuous shot.",
        64,
        32,
        124,
        task_type="FL2VA",
        audio_mode="native",
        first_frame=first,
        last_frame=last,
        first_frame_index=first_index,
        last_frame_index=last_index,
    )
    assert [x["resolved_frame_index"] for x in built[1][0][1]["minimax_keyframes"]] == [
        first_index,
        last_index,
    ]
    assert [x["frame_index"] for x in json.loads(built[6])] == [first_index, last_index]


def test_old_default_factory_keeps_stable_identity_and_natural_endpoint():
    model = make_model_patcher()
    built = build_multikeyframe_conditioning(
        model,
        FakeClip(),
        FakeVideoVAE(),
        FakeAudioVAE(),
        "A continuous shot.",
        64,
        32,
        124,
        task_type="L2VA",
        audio_mode="native",
        last_frame=make_image(0.8, 32, 64),
    )
    assert built[0] is model
    assert built[1][0][1]["minimax_keyframes"][0]["resolved_frame_index"] == 123
    schema = MiniMaxH3DirectorKeyframeConditioningEXPT8.define_schema()
    task = next(x for x in schema.inputs if x.id == "task_type")
    assert "L2VA" in task.options
