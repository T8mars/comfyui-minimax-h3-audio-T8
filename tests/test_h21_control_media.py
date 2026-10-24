"""Real CPU codecs and actual Director Source; no trained model execution."""

from fractions import Fraction
import importlib.util
import json
import os
from pathlib import Path
import uuid

import av
import numpy as np
from PIL import Image
import pytest
import torch

from h3_audio_t8_pkg.director_control_source import prepare_director_control_source


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("h21_media_tested", ROOT / "tools/prepare_h21_control_media.py")
media = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(media)


@pytest.fixture
def originals(tmp_path):
    directory = tmp_path / "originals"
    directory.mkdir()
    rng = np.random.default_rng(1224)
    arrays = [rng.integers(0, 256, (16, 24, 3), dtype=np.uint8) for _ in range(5)]
    source = directory / "original_rgb.mp4"
    media._encode(source, arrays, Fraction(24))
    rows = []
    for number, array in enumerate(arrays):
        path = directory / f"{number:05d}.png"
        Image.fromarray(array).save(path)
        rows.append({"frame": number, "path": str(path), "sha256": media.file_sha(path)})
    manifest = {"source": str(source), "source_sha256": media.file_sha(source), "frames": 5, "fps": 24, "width": 24, "height": 16, "images": rows}
    path = directory / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    before = {p: p.read_bytes() for p in directory.iterdir()}
    return path, manifest, arrays, before


def test_real_lossless_codec_preserves_every_RGB_byte_and_rational_timestamp(originals, tmp_path):
    path, manifest, arrays, before = originals
    report = media.prepare(path, tmp_path / "prepared", expected_manifest_sha256=media.file_sha(path))
    video = report["control_video"]
    assert video["frames"] == 5 and video["fps"] == "24" and video["audio"] is False
    assert video["width"] == 24 and video["height"] == 16
    assert len(report["decoded_control_frames"]) == 5
    assert [r["rgb_sha256"] for r in report["decoded_control_frames"]] == [media.rgb_sha(a) for a in arrays]
    for role, index in (("first", 0), ("middle", 2), ("last", 4)):
        frame = report["source_images"][role]
        assert frame["frame"] == index
        assert frame["rgb_sha256"] == media.rgb_sha(arrays[index])
    assert manifest["source_sha256"] == report["raw_rgb"]["sha256"]
    assert {p: p.read_bytes() for p in before} == before
    assert all(value is False for value in report["actions"].values())


def test_actual_source_owns_hold_last_not_the_lossless_stored_movie(originals, tmp_path):
    path, _, arrays, _ = originals
    report = media.prepare(path, tmp_path / "prepared")
    with av.open(report["control_video"]["path"]) as stream:
        decoded = np.stack([frame.to_ndarray(format="rgb24") for frame in stream.decode(video=0)])
    frames = torch.from_numpy(decoded).float() / 255
    row = {"id": str(uuid.uuid4()), "asset_id": str(uuid.uuid4()), "control_net_name": "explicit-user-file.safetensors", "source": {"start_seconds": 0, "short_video_policy": "hold_last_frame"}}
    selected, _, _, receipt = prepare_director_control_source(frames, row, width=96, height=64, length=22, source_kind="video", source_fps=24)
    assert selected.shape == (22, 16, 24, 3)  # Source never silently spatially resizes.
    torch.testing.assert_close(selected[:5], frames, rtol=0, atol=0)
    torch.testing.assert_close(selected[5:], frames[-1:].expand(17, -1, -1, -1), rtol=0, atol=0)
    assert receipt["held_video_frames"] == 17
    assert report["control_video"]["frames"] == len(arrays)


@pytest.mark.parametrize("change,match", [
    ("manifest_sha", "manifest SHA"),
    ("png", "image SHA"),
    ("source", "source SHA"),
    ("order", "order"),
    ("count", "count"),
    ("geometry", "geometry"),
    ("repeat", "repeat"),
])
def test_bad_provenance_is_rejected_without_touching_originals_or_starting_output(originals, tmp_path, change, match):
    path, manifest, _, _ = originals
    expected = None
    if change == "manifest_sha":
        expected = "0" * 64
    elif change == "png":
        Path(manifest["images"][1]["path"]).write_bytes(b"changed")
    elif change == "source":
        Path(manifest["source"]).write_bytes(b"changed")
    elif change == "order":
        manifest["images"][0], manifest["images"][1] = manifest["images"][1], manifest["images"][0]
    elif change == "count":
        manifest["frames"] += 1
    elif change == "geometry":
        manifest["width"] += 1
    elif change == "repeat":
        manifest["images"][1]["path"] = manifest["images"][0]["path"]
    path.write_text(json.dumps(manifest), encoding="utf-8")
    before = {p: p.read_bytes() for p in path.parent.iterdir()}
    with pytest.raises(ValueError, match=match):
        media.prepare(path, tmp_path / "failed", expected_manifest_sha256=expected)
    assert not (tmp_path / "failed").exists()
    assert {p: p.read_bytes() for p in before} == before


def test_existing_success_or_failure_directory_is_never_overwritten(originals, tmp_path):
    path, _, _, _ = originals
    output = tmp_path / "existing"
    output.mkdir()
    preserved = output / "failure-evidence.txt"
    preserved.write_bytes(b"preserve")
    with pytest.raises(FileExistsError):
        media.prepare(path, output)
    assert preserved.read_bytes() == b"preserve"
    assert list(output.iterdir()) == [preserved]


def test_verifier_refuses_a_truncated_or_lossy_movie(originals, tmp_path):
    path, _, arrays, _ = originals
    report = media.prepare(path, tmp_path / "prepared")
    with pytest.raises(ValueError, match="incomplete"):
        media.verify_video(report["control_video"]["path"], [*arrays, arrays[-1]], Fraction(24))
    altered = [array.copy() for array in arrays]
    altered[2][3, 4, 0] ^= 1
    with pytest.raises(ValueError, match="RGB check failed at control frame 2"):
        media.verify_video(report["control_video"]["path"], altered, Fraction(24))


def test_explicit_original_audio_is_float_PCM_bit_exact_without_resample_or_trim(tmp_path):
    source = tmp_path / "original_audio.wav"
    rng = np.random.default_rng(32)
    expected = rng.uniform(-0.25, 0.25, (2, 1031)).astype(np.float32)
    with av.open(str(source), mode="w", format="wav") as container:
        stream = container.add_stream("pcm_f32le", rate=32000)
        stream.layout = "stereo"
        frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(expected.T).reshape(1, -1), format="flt", layout="stereo")
        frame.sample_rate = 32000
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    before = source.read_bytes()
    report = media.extract_float_audio(source, tmp_path / "audio", expected_source_sha256=media.file_sha(source))
    assert report["audio"]["samples"] == 1031
    assert report["audio"]["channels"] == 2 and report["audio"]["sample_rate"] == 32000
    assert report["audio"]["source_decoded_pcm_sha256"] == media.rgb_sha(expected)
    assert report["audio"]["wav_decoded_pcm_sha256"] == media.rgb_sha(expected)
    assert source.read_bytes() == before
    with pytest.raises(FileExistsError):
        media.extract_float_audio(source, tmp_path / "audio", expected_source_sha256=media.file_sha(source))
    with pytest.raises(ValueError, match="SHA changed"):
        media.extract_float_audio(source, tmp_path / "absent", expected_source_sha256="0" * 64)
    assert not (tmp_path / "absent").exists()


@pytest.mark.skipif(not os.environ.get("T8_H21_CONTROL_MEDIA_REPORT"), reason="Explicit private H07 media opt-in; no fixture download or generation")
def test_explicit_actual_H07_media_120_to_native124_uses_only_four_held_frames():
    from h3_audio_t8_pkg.core import temporal_shape
    from h3_audio_t8_pkg.director_project import director_canvas

    report = json.loads(Path(os.environ["T8_H21_CONTROL_MEDIA_REPORT"]).read_text(encoding="utf-8"))
    video = report["control_video"]
    assert media.file_sha(video["path"]) == video["sha256"]
    with av.open(video["path"]) as container:
        decoded = np.stack([frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)])
    assert decoded.shape == (120, 144, 256, 3)
    assert [media.rgb_sha(frame) for frame in decoded] == [row["rgb_sha256"] for row in report["input_pngs"]]
    frames = torch.from_numpy(decoded).float() / 255
    row = {"id": str(uuid.uuid4()), "asset_id": str(uuid.uuid4()), "control_net_name": "MiniMax-H3-Fun-Controlnet-Union.safetensors", "source": {"start_seconds": 0, "short_video_policy": "hold_last_frame"}}
    length, latent_t, audio_t = temporal_shape(120)
    width, height = director_canvas(16 / 9, 0.6)
    assert (length, latent_t, audio_t, width, height) == (124, 37, 207, 1024, 576)
    selected, _, _, receipt = prepare_director_control_source(frames, row, width=width, height=height, length=length, source_kind="video", source_fps=24)
    assert selected.shape == (124, 144, 256, 3)
    assert receipt["source_indices"] == [*range(120), *([119] * 4)]
    assert receipt["held_video_frames"] == 4 and receipt["source_audio_consumed"] is False
    assert torch.equal(selected[:120], frames)
    assert torch.equal(selected[120:], frames[-1:].expand(4, -1, -1, -1))
    assert not torch.cuda.is_initialized()
