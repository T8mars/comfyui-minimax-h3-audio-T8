"""Actual tiny RGB/PCM/PTS execution, not a shape-only provenance assertion."""
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction
import json

import av
import numpy as np
import pytest
import torch
from comfy_api.latest import InputImpl, Types

from h3_audio_t8_pkg import source_conform as c
from h3_audio_t8_pkg.source_av import prepare_source_media_window
from h3_audio_t8_pkg.modular_sampling.video_io import save_serial
from h3_audio_t8_pkg.nodes_source_conform import NODES


def images(count=12):
    return torch.arange(count).float().div(count).reshape(count, 1, 1, 1).expand(-1, 32, 64, 3).clone()


def prepared(source=None, **options):
    source = images() if source is None else source
    return c.conform(source, options.pop("source_fps", 48.), 64, 32,
                     options.pop("length", 5), **options)


@pytest.mark.parametrize("fps,start,expected", [
    (48., 0., (0, 2, 4, 6, 8)), (12., 0., (0, 0, 1, 2, 2)),
    (24., 1/48, (0, 2, 2, 4, 4)), (24., 1/24, (1, 2, 3, 4, 5))])
def test_map_uses_actual_legacy_indices_and_default_five_outputs(fps, start, expected):
    source = images()
    legacy = prepare_source_media_window(source, fps, 64, 32, 5, start, "strict", "pad_silence")
    selected, audio, count, duration, mapping, report = prepared(source, source_fps=fps, start_seconds=start)
    assert len(legacy) == 5
    assert torch.equal(selected, legacy[0])
    assert torch.equal(audio["waveform"], legacy[1]["waveform"])
    assert (count, duration) == legacy[2:4]
    assert report["legacy_report"] == json.loads(legacy[4])
    assert mapping.indices == expected
    assert report["clock_status"] == "declared_fps_unverified"
    assert report["trace"]["actual_selected_pts"] is None
    assert report["legacy_report"]["facts"]["audio_source"] == "generated_silence"
    with pytest.raises(FrozenInstanceError):
        mapping.indices = (0,)


def test_holds_and_audio_crop_padding_are_captured_without_a_second_call(monkeypatch):
    real = c.prepare_source_media_window
    calls = []
    def counted(*args, **kwargs):
        calls.append(kwargs)
        return real(*args, **kwargs)
    monkeypatch.setattr(c, "prepare_source_media_window", counted)
    source = images(3)
    audio = {"waveform": torch.linspace(-.1, .1, 1000)[None, None], "sample_rate": 16000}
    result = prepared(source, source_fps=24., short_video_policy="hold_last_frame", source_audio=audio)
    mapping, report = result[-2:]
    assert len(calls) == 1 and calls[0] == {"return_frame_map": True}
    assert mapping.indices == (0, 1, 2, 2, 2)
    assert report["trace"]["facts"]["held_video_frames"] == 2
    assert report["trace"]["source_audio_identity"] == list(c.audio_identity(audio))
    assert report["trace"]["facts"]["padded_audio_samples"] == round(5/24*32000) - 2000
    assert torch.equal(result[1]["waveform"][:, 0], result[1]["waveform"][:, 1])


def test_owned_rgb_and_manual_mask_share_map_but_low_high_geometry_is_separate():
    source = images()
    mapping = prepared(source)[4]
    mask = torch.zeros(source.shape[:3])
    mask[2:5, 6:12, 10:20] = 1.
    control, rgb_lineage = c.derive_control(source, "identity_rgb")
    marked, mask_lineage = c.derive_control(source, "manual_mask", mask)
    low, low_report = c.map_control(mapping, control, lineage=rgb_lineage)
    high, high_report = c.map_control(mapping, control, width=96, height=64, lineage=rgb_lineage)
    mapped_mask, mask_report = c.map_control(mapping, marked, "mask", 96, 64, mask_lineage)
    assert torch.equal(low, mapping._output)
    assert high.shape == (5, 64, 96, 3) and mapped_mask.shape == (5, 64, 96)
    assert low_report["map_sha"] == high_report["map_sha"] == mask_report["map_sha"]
    assert low_report["stage_geometry"] != high_report["stage_geometry"]
    assert high_report["output_sha"] != low_report["output_sha"]
    assert high_report["lineage_status"] == "owned_preprocess_verified"
    assert mask_report["annotation_semantics_verified"] is False
    assert torch.count_nonzero(mapped_mask[0]) == 0 and torch.count_nonzero(mapped_mask[-1]) == 0
    assert set(mapped_mask.unique().tolist()) == {0., 1.}


def test_opaque_controls_allowed_but_shape_and_json_never_prove_origin():
    source = images()
    mapping = prepared(source)[4]
    control = torch.rand_like(source)
    _, report = c.map_control(mapping, control)
    assert report["lineage_status"] == "external_origin_unverified"
    with pytest.raises(ValueError, match="actual"):
        c.map_control(mapping, control, lineage=json.dumps({"source_sha": mapping.source_sha}))
    with pytest.raises(ValueError, match="JSON"):
        c.map_control(json.loads(c.canonical(c._map_payload(mapping))), control)


@pytest.mark.parametrize("target", ["source", "output", "audio", "indices"])
def test_mutated_owned_content_or_map_rejected(target):
    source = images()
    selected, audio, _, _, mapping, _ = prepared(source)
    if target == "source":
        source[0, 0, 0, 0] += .01
    elif target == "output":
        selected[0, 0, 0, 0] += .01
    elif target == "audio":
        audio["waveform"][0, 0, 0] = .01
    else:
        mapping = replace(mapping, indices=(0,) * len(mapping.indices))
    with pytest.raises(ValueError, match="changed"):
        c.validate_map(mapping)


def test_derived_controls_bind_actual_source_and_output_not_just_dimensions():
    source = images()
    mapping = prepared(source)[4]
    control, lineage = c.derive_control(source)
    c.map_control(mapping, control, lineage=lineage)
    bad, wrong = c.derive_control(source + .001)
    with pytest.raises(ValueError, match="different"):
        c.map_control(mapping, bad, lineage=wrong)
    control[2, 0, 0, 0] += .01
    with pytest.raises(ValueError, match="modified"):
        c.map_control(mapping, control, lineage=lineage)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 0., -1.])
def test_invalid_clock_does_not_allocate_a_prepared_video(bad):
    with pytest.raises(ValueError, match="finite"):
        prepared(source_fps=bad)


def test_bad_frame_count_geometry_mask_and_budgets_rejected(monkeypatch):
    source = images()
    with pytest.raises(ValueError, match="4096"):
        prepared(source, length=4096)
    mapping = prepared(source)[4]
    with pytest.raises(ValueError, match="grid differs"):
        c.map_control(mapping, source[:5])
    with pytest.raises(ValueError, match="no implicit broadcast"):
        c.derive_control(source, "manual_mask", torch.zeros(1, 32, 64))
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        c.derive_control(source, "manual_mask", torch.full(source.shape[:3], 2.))
    monkeypatch.setattr(c, "MAX_TENSOR_BYTES", 100)
    with pytest.raises(ValueError, match="budget"):
        prepared(source)


@pytest.mark.parametrize("trim,crop", [(0., None), (2/24, (0, 0, 32, 32))])
def test_native_clock_records_same_decode_cfr_rgb_audio_trim_crop(tmp_path, trim, crop):
    source = images(12)
    time = torch.arange(16000).float() / 16000
    audio = {"waveform": (.03 * torch.sin(2 * torch.pi * 330 * time))[None, None].repeat(1, 2, 1),
             "sample_rate": 16000}
    parts = Types.VideoComponents(images=source, audio=audio, frame_rate=Fraction(24))
    path, _ = save_serial(InputImpl.VideoFromComponents(parts), tmp_path, "source")
    video = InputImpl.VideoFromFile(str(path), start_time=trim, duration=8/24, crop=crop)
    components, clock, evidence = c.decode_clock(video)
    native = video.get_components()
    assert torch.equal(components.images, native.images)
    assert torch.equal(components.audio["waveform"], native.audio["waveform"])
    assert clock.status == "observed_CFR" and len(clock.pts_seconds) == 8
    assert clock.pts_seconds[0] == Fraction(round(trim * 24), 24)
    assert evidence["audio_pts_alignment_certified"] is False
    result = c.conform(components.images, 24., 64, 32, 5, source_audio=components.audio, clock=clock)
    assert result[-1]["trace"]["actual_selected_pts"] == [str(value) for value in clock.pts_seconds[:5]]
    with pytest.raises(ValueError, match="different"):
        c.conform(components.images + .01, 24., 64, 32, 5, source_audio=components.audio, clock=clock)
    with pytest.raises(ValueError, match="record changed"):
        c.conform(components.images, 24., 64, 32, 5, source_audio=components.audio,
                  clock=replace(clock, status="pretend_verified"))


def test_real_variable_pts_not_certified_as_cfr(tmp_path):
    path = tmp_path / "vfr.mp4"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=24)
        stream.width, stream.height, stream.pix_fmt = 64, 32, "yuv420p"
        stream.codec_context.thread_count = 1
        for index, pts in enumerate([0, 1, 2, 4, 5, 8]):
            frame = av.VideoFrame.from_ndarray(np.full((32, 64, 3), index * 30, dtype=np.uint8), format="rgb24")
            frame.pts, frame.time_base = pts, Fraction(1, 24)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    components, clock, _ = c.decode_clock(InputImpl.VideoFromFile(str(path)))
    assert clock.status == "clock_unverified" and len(clock.pts_seconds) == 6
    result = c.conform(components.images, float(clock.frame_rate), 64, 32, 5,
                       short_video_policy="hold_last_frame", clock=clock)
    assert result[-1]["clock_status"] == "clock_unverified"
    assert result[-1]["trace"]["actual_selected_pts"] is None


def test_materialized_external_components_remain_usable_unverified_and_cancellation_propagates():
    video = InputImpl.VideoFromComponents(Types.VideoComponents(images=images(), frame_rate=Fraction(24)))
    components, clock, evidence = c.decode_clock(video)
    assert evidence["decoder"] == "external_unverified" and clock.status == "clock_unverified"
    assert components.images is video.get_components().images
    def cancel():
        raise InterruptedError("cancelled")
    with pytest.raises(InterruptedError):
        c.decode_clock(video, cancel)


def test_schema_uses_distinct_optional_typed_lineage_ports_and_no_cache_certification():
    ids = [cls.GET_SCHEMA().node_id for cls in NODES]
    assert len(ids) == len(set(ids)) == 4
    for cls in NODES:
        assert cls.GET_SCHEMA().is_experimental
        first, second = cls.fingerprint_inputs(), cls.fingerprint_inputs()
        assert isinstance(first, str) and len(first) == 32 and first != second
        assert json.loads(json.dumps({"is_changed": first}, allow_nan=False))["is_changed"] == first
