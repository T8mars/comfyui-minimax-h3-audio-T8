"""Explicit media boundaries; no diffusion model or quality claims."""
import asyncio
from fractions import Fraction
import json
import subprocess
from types import SimpleNamespace

import av
import pytest
import torch
from comfy_api.latest import InputImpl, Types

import h3_audio_t8_pkg
from h3_audio_t8_pkg.modular_sampling import video_io, video_io_nodes


def components(audio=True, frames=17):
    x = torch.linspace(0, 1, 96)[None, None, :, None].expand(frames, 64, 96, 3).clone()
    x[:, :, :, 1] = torch.linspace(0, .8, frames)[:, None, None]
    sound = None
    if audio:
        t = torch.arange(16000).float() / 16000
        sound = {"waveform": (.1 * torch.sin(2 * torch.pi * 440 * t))[None, None].repeat(1, 2, 1),
                 "sample_rate": 16000}
    return Types.VideoComponents(images=x, audio=sound, frame_rate=Fraction(24))


def strict(path):
    subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-i", str(path), "-f", "null", "-"],
                   capture_output=True, check=True, timeout=30)
    return json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json",
                                               str(path)], timeout=30))


@pytest.mark.parametrize("depth,color", [(8, "sRGB"), (10, "sRGB"), (10, "HDR"), (10, "HDR PQ")])
@pytest.mark.parametrize("audio", [False, True])
def test_serial_save_preserves_depth_geometry_rate_color_audio_and_metadata(tmp_path, depth, color, audio):
    parts = components(audio)
    video = InputImpl.VideoFromComponents(parts, bit_depth=depth, color_space=color)
    before = parts.images.clone()
    original_open = av.open
    path, report = video_io.save_serial(video, tmp_path, "test/video", {"prompt": {"test": "中文"}})
    assert av.open is original_open
    assert torch.equal(before, parts.images)
    assert report["frames"] == 17 and report["video_encoder_threads"] == 1
    assert report["bit_depth"] == depth and report["color_space"] == color
    assert report["strict_decode_verified"] is False and report["quality_accepted"] is False
    metadata = strict(path)
    streams = metadata["streams"]
    picture = next(s for s in streams if s["codec_type"] == "video")
    assert (picture["width"], picture["height"], picture["nb_frames"], picture["avg_frame_rate"]) == (96, 64, "17", "24/1")
    assert picture["pix_fmt"] == ("yuv420p10le" if depth == 10 else "yuv420p")
    assert picture["color_transfer"] == {"sRGB": "iec61966-2-1", "HDR": "arib-std-b67", "HDR PQ": "smpte2084"}[color]
    sound = [s for s in streams if s["codec_type"] == "audio"]
    assert bool(sound) == audio
    if audio:
        assert (sound[0]["sample_rate"], sound[0]["channels"]) == ("16000", 2)
    assert json.loads(metadata["format"]["tags"]["prompt"])["test"] == "中文"
    again, second_report = video_io.save_serial(video, tmp_path, "test/video")
    assert again != path and path.is_file()
    assert report["color_conversion_threads"] == 1
    for field in ("input_identity", "quantized_rgb_sha256", "converted_yuv_sha256"):
        assert report[field] == second_report[field]


def test_serial_read_matches_native_valid_file_and_retains_crop_trim(tmp_path):
    parts = components()
    path, _ = video_io.save_serial(InputImpl.VideoFromComponents(parts), tmp_path, "source")
    original_open = av.open
    file = InputImpl.VideoFromFile(str(path), start_time=2/24, duration=8/24, crop=(0, 0, 64, 32))
    a, ra = video_io.components_serial(file)
    b, rb = video_io.components_serial(file)
    native = file.get_components()
    assert av.open is original_open
    assert ra == rb and ra["decoder_threads"] == 1
    assert torch.equal(a.images, b.images) and torch.equal(a.images, native.images)
    assert torch.equal(a.audio["waveform"], b.audio["waveform"])
    assert a.frame_rate == native.frame_rate == Fraction(24)
    assert a.images.shape[0] == 8 and a.images.shape[1:3] == native.images.shape[1:3]
    memory_video = InputImpl.VideoFromComponents(parts)
    original, report = video_io.components_serial(memory_video)
    assert original.images is parts.images and original.audio is parts.audio
    assert report["backend"] == "materialized_components"


def test_bad_data_path_or_cancellation_cannot_publish_a_completed_video(tmp_path):
    parts = components()
    video = InputImpl.VideoFromComponents(parts)
    with pytest.raises(ValueError):
        video_io.save_serial(video, tmp_path, "../escape")
    parts.images[3, 1, 1, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        video_io.save_serial(video, tmp_path, "bad")
    assert not list(tmp_path.rglob("video.mp4"))
    parts.images.nan_to_num_(0)
    calls = 0
    def cancel():
        nonlocal calls
        calls += 1
        if calls >= 10:
            raise InterruptedError("test cancellation")
    with pytest.raises(InterruptedError):
        video_io.save_serial(video, tmp_path, "cancel", interrupt=cancel)
    assert not list(tmp_path.rglob("video.mp4"))
    # x264 may buffer every frame before producing its first packet, so early
    # cancellation need not create a partial file. Never fabricate an empty one.
    assert len(list(tmp_path.glob("cancel-*"))) == 1


def test_failure_after_partial_write_preserves_evidence_without_publishing(tmp_path, monkeypatch):
    def failing_writer(_components, path, *_args):
        path.write_bytes(b"interrupted-media-evidence")
        raise OSError("injected write failure")
    monkeypatch.setattr(video_io, "write_serial", failing_writer)
    with pytest.raises(OSError, match="injected"):
        video_io.save_serial(InputImpl.VideoFromComponents(components()), tmp_path, "write-error")
    assert not list(tmp_path.rglob("video.mp4"))
    assert [p.read_bytes() for p in tmp_path.rglob("video.partial.mp4")] == [b"interrupted-media-evidence"]


def test_component_identity_detects_rgb_audio_and_rate_changes_without_mutation():
    parts = components()
    a = video_io.component_identity(parts)
    assert video_io.component_identity(parts) == a
    parts.images[0, 0, 0, 0] += .1
    b = video_io.component_identity(parts)
    assert b["images"] != a["images"] and b["audio"] == a["audio"]
    parts.audio["waveform"][0, 0, 0] += .1
    c = video_io.component_identity(parts)
    assert c["audio"] != b["audio"] and c["images"] == b["images"]


def test_serial_reader_checks_each_packet_and_does_not_swallow_invalid_data():
    calls = []
    packet = SimpleNamespace(stream="video", decode=lambda: iter((1, 2)))
    container = SimpleNamespace(metadata={"unchanged": True}, demux=lambda *_: iter((packet,)))
    checked = video_io._CheckedContainer(container, lambda: calls.append(True))
    wrapped = next(checked.demux("video"))
    assert checked.metadata is container.metadata and wrapped.stream == "video"
    assert list(wrapped.decode()) == [1, 2] and len(calls) == 4
    def corrupt():
        raise av.error.InvalidDataError(1, "bad packet")
    packet.decode = corrupt
    with pytest.raises(ValueError, match="failed to decode"):
        list(wrapped.decode())
    def cancel():
        raise InterruptedError("cancel between frames")
    with pytest.raises(InterruptedError):
        list(video_io._CheckedPacket(packet, cancel).decode())


def test_serial_nodes_append_after_entire_old_prefix_and_preserve_component_slots():
    before = [*asyncio.run(h3_audio_t8_pkg._HyperFlowLongVideoExtension().get_node_list()),
        *h3_audio_t8_pkg._modular_node_classes(), *h3_audio_t8_pkg._hyper_vae_2x_node_classes,
        *h3_audio_t8_pkg._audio_refine_effect_node_classes, *h3_audio_t8_pkg._ltx_rgb_source_node_classes]
    actual = asyncio.run(h3_audio_t8_pkg.comfy_entrypoint().get_node_list())
    prefix = (before + video_io_nodes.NODES + h3_audio_t8_pkg._ltx_effect_node_classes
              + h3_audio_t8_pkg._ltx_relay_node_classes)
    assert actual[:len(prefix)] == prefix
    assert actual[len(prefix):] == [*h3_audio_t8_pkg._veda_sparse_node_classes,
        *h3_audio_t8_pkg._veda_heuristic_node_classes,
        *h3_audio_t8_pkg._prepared_ltx_effect_node_classes,
        *h3_audio_t8_pkg._prepared_ltx_relay_cache_node_classes,
        *h3_audio_t8_pkg._semantic_bridge_extra_node_classes,
        *h3_audio_t8_pkg._ltx_load_policy_node_classes,
        *h3_audio_t8_pkg._face_source_node_classes]
    assert len(actual) == len({n.define_schema().node_id for n in actual})
    video = InputImpl.VideoFromComponents(components(), bit_depth=10, color_space="HDR")
    result = video_io_nodes.MiniMaxH3VideoComponentsSerialEXPT8.execute(video).result
    assert result[2:5] == (24., 10, "HDR")
    assert json.loads(result[5])["sampler_executed"] is False
