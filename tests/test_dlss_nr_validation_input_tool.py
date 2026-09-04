from __future__ import annotations

from fractions import Fraction
import hashlib
import json
from pathlib import Path

import av
import numpy as np
from PIL import Image
import pytest

from h3_audio_t8_pkg.tools import prepare_dlss_nr_validation_inputs as inputs


def _contract(*, frames: int = 124, width: int = 960, height: int = 544):
    return {
        "width": width,
        "height": height,
        "frame_count": frames,
        "rate": Fraction(24, 1),
        "audio_packets": [{"payload_sha256": "packet"}],
        "audio_pcm": [{"pcm_sha256": "pcm"}],
    }


def _write_cut_video(path: Path) -> None:
    with av.open(str(path), mode="w", format="mp4") as container:
        stream = container.add_stream("libx264", rate=24)
        stream.width = 96
        stream.height = 54
        stream.pix_fmt = "yuv420p"
        stream.codec_context.max_b_frames = 0
        stream.codec_context.time_base = Fraction(1, 24)
        stream.time_base = Fraction(1, 24)
        for index in range(inputs.SPEECH_FRAME_COUNT):
            pixels = np.zeros((54, 96, 3), dtype=np.uint8)
            pixels[..., 0 if index < inputs.HARD_CUT_FRAME else 2] = 230
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts = index
            frame.time_base = Fraction(1, 24)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def test_speech_source_uses_strict_rate_contract_and_exact_phrase(monkeypatch, tmp_path):
    source = tmp_path / "speech.mp4"
    source.write_bytes(b"source")
    monkeypatch.setattr(inputs, "_strict_video_contract", lambda _path: _contract())

    report = inputs._validate_speech_source(source, inputs.SPEECH_PHRASE)
    assert report["rate"] == Fraction(24, 1)

    with pytest.raises(ValueError, match="operator must confirm"):
        inputs._validate_speech_source(source, "你在那里")


@pytest.mark.parametrize(
    ("contract", "message"),
    [
        (_contract(frames=123), "exactly 124 frames"),
        (_contract(width=320, height=192), "approximately 0.5 MP"),
    ],
)
def test_speech_source_rejects_wrong_real_gate(monkeypatch, tmp_path, contract, message):
    source = tmp_path / "speech.mp4"
    source.write_bytes(b"source")
    monkeypatch.setattr(inputs, "_strict_video_contract", lambda _path: contract)
    with pytest.raises(ValueError, match=message):
        inputs._validate_speech_source(source, inputs.SPEECH_PHRASE)


def test_hard_cut_screen_checks_adjacent_pairs_and_exact_join(tmp_path):
    source = tmp_path / "cut.mp4"
    _write_cut_video(source)
    report = inputs._hard_cut_screen(source)
    assert report["expected_cut_frame"] == 62
    assert report["strongest_cut_frame"] == 62
    assert report["cut_mean_absolute_delta"] >= 0.08
    assert report["mechanical_hard_cut"] is True


def test_hard_cut_command_is_fixed_serial_media_contract(tmp_path):
    command = inputs._hard_cut_command(
        tmp_path / "ffmpeg.exe",
        tmp_path / "first.mp4",
        tmp_path / "second.mp4",
        tmp_path / "out.mp4",
    )
    graph = command[command.index("-filter_complex") + 1]
    assert "trim=start_frame=0:end_frame=62" in graph
    assert graph.count("trim=start_frame=0:end_frame=62") == 2
    assert command[command.index("-frames:v") + 1] == "124"
    assert command[command.index("-c:a") + 1] == "copy"
    assert command[command.index("-threads") + 1] == "1"


def test_fine_texture_overlay_is_deterministic_and_contains_one_pixel_detail(
    tmp_path,
):
    font = Path(r"C:\Windows\Fonts\consola.ttf")
    if not font.is_file():
        pytest.skip("Windows Consolas font is not installed")
    target = tmp_path / "overlay.png"
    report = inputs._write_fine_texture_overlay(target, font)
    image = np.asarray(Image.open(target).convert("RGBA"))
    assert report["geometry"] == [960, 544]
    assert report["patterns"] == [
        "alternating_2px_vertical_lines",
        "2px_checkerboard",
    ]
    assert image.shape == (544, 960, 4)
    assert np.any(image[..., 3] == 0)
    assert len(np.unique(image[28:76, 660:932, 0])) >= 2


def test_existing_bundle_is_never_overwritten(monkeypatch, tmp_path):
    speech = tmp_path / "speech.mp4"
    second = tmp_path / "second.mp4"
    ffmpeg = tmp_path / "ffmpeg.exe"
    font = tmp_path / "font.ttf"
    for path in (speech, second, ffmpeg, font):
        path.write_bytes(b"input")
    output = tmp_path / "existing"
    output.mkdir()
    (output / "evidence.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(inputs, "_validate_speech_source", lambda *_args: _contract())
    monkeypatch.setattr(inputs, "_strict_video_contract", lambda _path: _contract())

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        inputs.prepare_validation_inputs(
            speech_video=speech,
            hard_cut_second_video=second,
            output_dir=output,
            ffmpeg=ffmpeg,
            font_file=font,
            speech_phrase_confirmation=inputs.SPEECH_PHRASE,
        )
    assert (output / "evidence.json").read_text(encoding="utf-8") == "{}"


def test_bundle_publish_is_atomic_and_writes_manifest_checksum(monkeypatch, tmp_path):
    speech = tmp_path / "speech.mp4"
    second = tmp_path / "second.mp4"
    ffmpeg = tmp_path / "ffmpeg.exe"
    font = tmp_path / "font.ttf"
    for path in (speech, second, ffmpeg, font):
        path.write_bytes(path.name.encode("ascii"))
    contract = _contract()
    monkeypatch.setattr(inputs, "_validate_speech_source", lambda *_args: contract)
    monkeypatch.setattr(inputs, "_strict_video_contract", lambda _path: contract)

    def extract(_source, target, frame_index):
        Image.new("RGB", (960, 544), "black").save(target)
        return {
            "source_frame_index": frame_index,
            "width": 960,
            "height": 544,
            "megapixels": 0.52224,
            "rgb_bridge": "decoded_rgb8_png",
        }

    def execute(command, *, label):
        Path(command[-1]).write_bytes(label.encode("utf-8"))
        return {"returncode": 0, "elapsed_seconds": 0.0, "stderr_tail": ""}

    def overlay(target, _font):
        Image.new("RGBA", (960, 544), (0, 0, 0, 0)).save(target)
        return {
            "geometry": [960, 544],
            "text": ["test"],
            "patterns": ["test"],
            "purpose": "test",
        }

    monkeypatch.setattr(inputs, "_extract_frame", extract)
    monkeypatch.setattr(inputs, "_run", execute)
    monkeypatch.setattr(inputs, "_write_fine_texture_overlay", overlay)
    monkeypatch.setattr(
        inputs,
        "_hard_cut_screen",
        lambda _path: {
            "expected_cut_frame": 62,
            "strongest_cut_frame": 62,
            "mechanical_hard_cut": True,
        },
    )
    output = tmp_path / "bundle"
    report = inputs.prepare_validation_inputs(
        speech_video=speech,
        hard_cut_second_video=second,
        output_dir=output,
        ffmpeg=ffmpeg,
        font_file=font,
        speech_phrase_confirmation=inputs.SPEECH_PHRASE,
    )

    manifest = output / "validation_inputs.json"
    checksum = hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert report["status"] == "PREPARED_NOT_DLSS_TESTED"
    assert json.loads(manifest.read_text(encoding="utf-8"))["schema"] == inputs.SCHEMA
    assert (output / "validation_inputs.sha256").read_text(encoding="ascii") == (
        f"{checksum}  validation_inputs.json\n"
    )
    assert not list(tmp_path.glob(".bundle.tmp-*"))
