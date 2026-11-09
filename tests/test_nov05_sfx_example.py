"""Actual Core PCM/video consumers; synthetic fixtures are not review media."""
import asyncio
from copy import deepcopy
import json
from pathlib import Path

import av
import numpy as np
import pytest
import torch

from h3_audio_t8_pkg.nodes_h3_av_delivery import MiniMaxH3SafeAVSaveT8Advanced
from tools import build_nov05_sfx_example as builder


def write_wave(path, waveform, rate):
    with av.open(str(path), "w") as container:
        stream = container.add_stream("pcm_f32le", rate=rate)
        stream.layout = "mono"
        frame = av.AudioFrame.from_ndarray(waveform.numpy(), format="fltp", layout="mono")
        frame.sample_rate = rate
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


@pytest.fixture
def actual_media(tmp_path, monkeypatch):
    import folder_paths
    inputs, outputs = tmp_path / "input", tmp_path / "output"
    inputs.mkdir()
    outputs.mkdir()
    monkeypatch.setattr(folder_paths, "input_directory", str(inputs))
    monkeypatch.setattr(folder_paths, "output_directory", str(outputs))
    rate = 32000
    time = torch.arange(rate * 5 + 768) / rate
    original = (.02 * torch.sin(time * (2 * np.pi * 220))).reshape(1, 1, -1)
    # A real float WAV has 768 extra samples. The visible Core trim must remove
    # them before BOTH source/protected inputs; this is not a voiced recording.
    write_wave(inputs / builder.MEDIA["source_audio"], original[0], rate)
    for field, frequency, candidate_rate in (("external_audio", 470, 44100), ("h3_audio", 690, 32000)):
        values = .1 * torch.sin(torch.arange(candidate_rate * 5) / candidate_rate * (2 * np.pi * frequency))
        write_wave(inputs / builder.MEDIA[field], values.reshape(1, -1), candidate_rate)
    images = torch.linspace(0., 1., 120 * 32 * 64 * 3).reshape(120, 32, 64, 3)
    video = MiniMaxH3SafeAVSaveT8Advanced.execute(images, {
        "waveform": original[..., :5 * rate], "sample_rate": rate}, "source", 18, False).result
    (inputs / builder.MEDIA["source_video"]).write_bytes(Path(video[1]).read_bytes())
    return inputs, outputs, original


def test_every_example_output_validates_with_real_core_files(actual_media):
    import execution
    graph, workflow, audit, info = builder.build(builder.load_live_info())
    assert graph["7"]["inputs"]["level_policy"] == "target_rms_ceiling"
    assert graph["7"]["inputs"]["target_rms_dbfs"] == -56.
    validation = asyncio.run(execution.validate_prompt("nov05-sfx-example", graph, None))
    assert validation[0] and not validation[3], validation
    assert set(validation[2]) == {"8", "9", "10", "11"}
    assert audit["nodes"] == 11 and len(workflow["nodes"]) == 12
    assert "不是分离" in workflow["nodes"][-1]["widgets_values"][0]
    assert set(info) == {node["class_type"] for node in graph.values()}
    controls = info["MiniMaxH3LocalSFXCandidatesT8"]["input"]["required"]
    assert controls["fade_seconds"][1]["default"] == .01
    for name, resolution in (("fade_seconds", .001), ("start_seconds", .01),
                             ("end_seconds", .01), ("candidate_start_seconds", .01)):
        assert controls[name][1]["round"] == controls[name][1]["step"] == resolution
    # Current native LoadVideo adds this empty preview widget asynchronously.
    # Real Core accepts the exported UI-only field without losing any output.
    graph["1"]["inputs"]["video-preview"] = ""
    native_valid = asyncio.run(execution.validate_prompt("nov05-sfx-native-preview", graph, None))
    assert native_valid[0] and not native_valid[3] and set(native_valid[2]) == {"8", "9", "10", "11"}
    assert not torch.cuda.is_initialized()


def test_actual_core_decode_trim_three_selections_and_saves(actual_media):
    import nodes
    builder.load_live_info()
    graph = builder.recipe()
    cache = {}

    def consume(key):
        if key in cache:
            return cache[key]
        node = graph[key]
        kwargs = {name: consume(str(value[0]))[value[1]]
                  if isinstance(value, list) else value for name, value in node["inputs"].items()}
        result = nodes.NODE_CLASS_MAPPINGS[node["class_type"]].execute(**kwargs).result
        cache[key] = result
        return result

    source = consume("4")[0]
    assert consume("3")[0]["waveform"].shape[-1] == 160768
    assert source["waveform"].shape[-1] == 160000
    torch.testing.assert_close(source["waveform"], actual_media[2][..., :160000], rtol=0, atol=0)
    expected_indices = {"original": None, "external": 1, "h3": 2}
    comparison = None
    for selection, slot in expected_indices.items():
        graph["7"]["inputs"]["selection"] = selection
        cache.pop("7", None)
        result = consume("7")
        if comparison is None:
            comparison = result
        expected = source if slot is None else comparison[slot]
        torch.testing.assert_close(result[0]["waveform"], expected["waveform"], rtol=0, atol=0)
        report = json.loads(result[3])
        assert report["start_sample"] == 38400 and report["end_sample"] == 112000
        assert report["crossfade_samples"] == 320 and report["source_residual_rms"] == 0
        assert report["claims"]["source_separation_performed"] is False
        assert report["candidates"]["external"]["resampled"] is True
        for arm in result[1:3]:
            torch.testing.assert_close(arm["waveform"][..., :38400], source["waveform"][..., :38400], rtol=0, atol=0)
            torch.testing.assert_close(arm["waveform"][..., 112000:], source["waveform"][..., 112000:], rtol=0, atol=0)
            assert not torch.equal(arm["waveform"][..., 39000:111000], source["waveform"][..., 39000:111000])
    reports = []
    for key in ("8", "9", "10", "11"):
        output = consume(key)
        assert Path(output[1]).is_file()
        reports.append(json.loads(output[2]))
        with av.open(output[1]) as container:
            assert len(list(container.decode(video=0))) == 120
        with av.open(output[1]) as container:
            assert container.streams.audio and container.streams.audio[0].duration
    assert len({report["source_rgb8_sha256"] for report in reports}) == 1
    assert not torch.cuda.is_initialized()


def test_missing_candidate_is_actual_input_failure_not_silent_original(actual_media):
    import execution
    graph = builder.recipe()
    builder.load_live_info()
    (actual_media[0] / builder.MEDIA["h3_audio"]).unlink()
    graph["7"]["inputs"]["selection"] = "h3"
    valid = asyncio.run(execution.validate_prompt("nov05-sfx-missing", deepcopy(graph), None))
    # The independent original output can remain valid, while all H3-dependent
    # comparison/selected outputs must be excluded by real Core validation.
    assert set(valid[2]) == {"8"} and "6" in valid[3], valid


def test_checked_in_example_matches_current_editable_recipe(actual_media):
    from tools.audit_progressive_workflows import audit_candidate
    graph, _, _, info = builder.build(builder.load_live_info())
    example = builder.ROOT / "examples/workflows/96-nov05-exp/Local_SFX_AB_EXP.json"
    saved = json.loads(example.read_text(encoding="utf-8"))
    assert audit_candidate(graph, saved, info)["nodes"] == 11
    assert graph["7"]["inputs"]["source_audio"] == graph["7"]["inputs"]["protected_voice_audio"] == ["4", 0]
    assert all(graph[key]["inputs"]["images"] == ["2", 0] for key in ("8", "9", "10", "11"))
