"""Both reference videos audible: real loaders, aliases and encoder boundaries."""

from copy import deepcopy
import json

import pytest
import torch
import folder_paths

from h3_audio_t8_pkg.director_generation import build_director_generation_prompt
from h3_audio_t8_pkg.director_project import ProjectStore, compile_project
from test_h24_director_audio_bindings import (
    _consume,
    _load_media,
    _project,
    _video,
    _tone,
)
from test_director_generation import _image, _patch_director_models, _store


@pytest.fixture
def sources(tmp_path, monkeypatch):
    store = _store(tmp_path)
    assets = {
        "a": _video(store, "red", True, hz=431),
        "b": _video(store, "blue", True, hz=997),
        "v": _tone(store, 743),
        "d": _tone(store, 1193),
        "first": _image(store, "first.png"),
        "last": _image(store, "last.png"),
    }
    monkeypatch.setattr(
        folder_paths, "get_input_directory", lambda: str(store.input_root)
    )
    _patch_director_models(monkeypatch)
    monkeypatch.setattr(
        folder_paths,
        "get_filename_list",
        lambda folder: {
            "latent_upscale_models": ["minimax_h3_latent_upscaler_3d_fp16.safetensors"]
        }.get(folder, []),
    )
    monkeypatch.setattr(folder_paths, "get_full_path", lambda folder, name: name)
    return store, assets


@pytest.mark.parametrize("order", [("a", "b"), ("b", "a")])
@pytest.mark.parametrize("sound", ["native", "voice", "record"])
def test_two_audible_videos_keep_actual_audio_ordinals_across_voice_and_record(
    sources, order, sound
):
    store, assets = sources
    assert assets["a"]["has_audio"] and assets["b"]["has_audio"]
    project, shot = _project(assets, sound=sound, mode="refs", order=order)
    if sound != "voice":
        shot["refs"].append(assets["v"]["id"])
    shot["simplePrompt"] = (
        "<Audio 1> and <Audio 2> are the ordered video soundtracks. "
        "@audio1 is the separate voice reference. "
        + ("@audio2 drives the performance." if sound == "record" else "")
    )
    saved = store.save(project, 0)
    reopened = ProjectStore(store.root.parent, store.input_root).load(saved["id"])
    before = deepcopy(reopened)
    compiled = compile_project(reopened, store, shot_id=shot["id"])
    assert compiled["ready"], compiled["errors"]
    shot_plan = compiled["shots"][0]
    audio_rows = [row for row in shot_plan["media_map"] if "audio" in row["role"]]
    expected_ids = [assets[key]["id"] for key in order]
    expected_ids += ([assets["d"]["id"]] if sound == "record" else []) + [
        assets["v"]["id"]
    ]
    assert [row["asset_id"] for row in audio_rows] == expected_ids
    assert [row["native"] for row in audio_rows] == [
        f"<Audio {i + 1}>" for i in range(len(expected_ids))
    ]
    graph = build_director_generation_prompt(reopened, shot["id"], store)["prompt"]
    (positive, latent, final, prompt, mapping, _), inputs, vae = _consume(
        graph, _load_media(graph)
    )
    assert f"<Audio {4 if sound == 'record' else 3}> is the separate voice" in prompt
    video_sound = [inputs["ref_video_audios"][f"ref_video_audio_{i}"] for i in range(2)]
    assert not torch.equal(video_sound[0]["waveform"], video_sound[1]["waveform"])
    expected_audio = (
        video_sound
        + ([inputs["drive_audio"]] if sound == "record" else [])
        + [inputs["ref_audios"]["ref_audio_0"]]
    )
    assert len(vae.encode_calls) == len(expected_audio)
    for actual, expected in zip(vae.encode_calls, expected_audio, strict=True):
        assert expected["sample_rate"] == 32000
        assert torch.equal(actual, expected["waveform"].movedim(1, -1))
    actual_video_refs = [
        row
        for row in positive[0][1]["minimax_refs"]
        if row["kind"] in {"video", "video_audio"}
    ]
    assert len(actual_video_refs) == 2 and all(
        row["kind"] == "video_audio" for row in actual_video_refs
    )
    # Generated audio remains generated for native/voice; record alone locks it.
    if sound == "record":
        assert json.loads(mapping)["source_audio_ordinal"] == 3
        assert "<Audio 3> drives the performance." in prompt
        assert final is inputs["drive_audio"] is inputs["final_audio"]
        assert torch.count_nonzero(latent["noise_mask"].unbind()[1]) == 0
    else:
        assert json.loads(mapping)["source_audio_ordinal"] is None and final is None
        assert "drive_audio" not in inputs and "final_audio" not in inputs
        assert "noise_mask" not in latent and graph["11"]["inputs"]["audio"] == [
            "10",
            1,
        ]
    assert reopened == before
