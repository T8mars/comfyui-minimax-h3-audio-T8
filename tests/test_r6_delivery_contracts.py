"""N05 existing Core crop + durable master: CPU mechanics, not AV quality."""
import hashlib
import importlib
from pathlib import Path

import av
import comfy.utils
import pytest
import torch

from h3_audio_t8_pkg.h3_av_delivery import save_h3_av_safe
from h3_audio_t8_pkg import long_video_delivery


@pytest.fixture
def native_crop(monkeypatch):
    # conftest's historical h3_t8 path must not shadow Core's top-level nodes.
    core = Path(comfy.utils.__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(core))
    native = importlib.import_module("comfy_extras.nodes_images")
    assert Path(native.nodes.__file__).resolve() == core / "nodes.py"
    return native.BoundingBox, native.ImageCropV2


def decoded_tracks(path):
    video_pts, pcm = [], hashlib.sha256()
    with av.open(str(path)) as container:
        video = container.streams.video[0]
        assert video.average_rate == 24
        dimensions = (video.width, video.height)
        for frame in container.decode(video):
            video_pts.append((frame.pts, str(frame.time_base)))
    with av.open(str(path)) as container:
        audio = container.streams.audio[0]
        clock = (audio.codec_context.sample_rate, audio.codec_context.layout.name)
        audio_pts = []
        for frame in container.decode(audio):
            pcm.update(frame.to_ndarray().tobytes())
            audio_pts.append((frame.pts, str(frame.time_base), frame.samples))
    return dimensions, video_pts, clock, audio_pts, pcm.hexdigest()


def test_exact_1080_core_bounding_box_after_decoded_pixels_not_latent(native_crop):
    BoundingBox, ImageCropV2 = native_crop
    image = (torch.arange(1088, dtype=torch.float32) / 1088)[None, :, None, None].expand(2, 1088, 1920, 3)
    schema = ImageCropV2.GET_SCHEMA().get_v1_info(ImageCropV2)
    assert list(schema.input["required"]) == ["image", "crop_region"]
    region = BoundingBox.execute(x=0, y=4, width=1920, height=1080).result[0]
    result = ImageCropV2.execute(image, region).result[0]
    assert tuple(result.shape) == (2, 1080, 1920, 3)
    assert torch.equal(result, image[:, 4:1084])
    assert torch.equal(image[:, 0], torch.zeros_like(image[:, 0]))


def test_same_queue_original_pcm_passthrough_and_failed_postprocess_keeps_master(tmp_path, monkeypatch, native_crop):
    BoundingBox, ImageCropV2 = native_crop
    frames = torch.linspace(0, 1, 24 * 32 * 64 * 3).reshape(24, 32, 64, 3)
    t = torch.arange(32000, dtype=torch.float32) / 32000
    audio = {"sample_rate": 32000, "waveform": torch.stack([
        .03 * torch.sin(2 * torch.pi * 440 * t), .04 * torch.sin(2 * torch.pi * 660 * t)])[None]}
    original_pcm = audio["waveform"].clone()
    master = tmp_path / "master.mp4"
    master_report = save_h3_av_safe(frames, audio, master)
    master_sha = hashlib.sha256(master.read_bytes()).hexdigest()
    assert master_report["output_sha256"] == master_sha
    crop = ImageCropV2.execute(frames, BoundingBox.execute(0, 4, 64, 24).result[0]).result[0]
    output = tmp_path / "crop.mp4"
    save_h3_av_safe(crop, audio, output)
    before, after = decoded_tracks(master), decoded_tracks(output)
    assert before[0] == (64, 32) and after[0] == (64, 24)
    assert before[1:] == after[1:]  # actual video PTS and decoded audio PCM/PTS unchanged
    assert torch.equal(audio["waveform"], original_pcm)

    def encoder_failure(*args, **kwargs):
        raise RuntimeError("injected downstream encoder failure")

    monkeypatch.setattr(long_video_delivery, "_encode_rgb_frames_isolated", encoder_failure)
    failed_output = tmp_path / "failed-postprocess.mp4"
    try:
        save_h3_av_safe(crop, audio, failed_output)
    except RuntimeError as error:
        assert "downstream encoder failure" in str(error)
    else:
        raise AssertionError("An encoder failure must not be reported as enhanced success")
    assert not failed_output.exists()
    assert hashlib.sha256(master.read_bytes()).hexdigest() == master_sha
    assert not list(tmp_path.glob(".h3-av-encode-*"))
