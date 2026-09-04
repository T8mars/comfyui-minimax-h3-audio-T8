from __future__ import annotations

from collections import Counter
from fractions import Fraction
import json
from pathlib import Path
import shutil
import subprocess

import av
import numpy as np
import pytest

from h3_audio_t8_pkg.tools import build_dlss_nr_blind_review as blind


def _sha(path: Path) -> str:
    return blind.validation_tool._sha256_file(path)


def _contract(path: Path) -> dict:
    candidate = "candidate" in path.name
    return {
        "frame_count": 4,
        "width": 128 if candidate else 64,
        "height": 96 if candidate else 48,
        "bit_depth": 8,
        "rate": Fraction(24, 1),
        "time_base": Fraction(1, 24),
        "pts": [0, 1, 2, 3],
        "reported_rate": 24.0,
        "sdr": {"passed": True},
        "cfr": {"passed": True},
        "audio_packets": [],
        "audio_pcm": [],
        "ffprobe": {},
    }


def _manifest(tmp_path: Path, *, clip_count: int = 3) -> dict:
    clip_types = ["speech", "hard_cut", "fine_texture", "speech"]
    clips = []
    for index in range(clip_count):
        source = tmp_path / f"source-{index}.mp4"
        source.write_bytes(f"source-{index}".encode())
        methods = {}
        for method in blind.METHODS:
            candidate = tmp_path / f"candidate-{index}-{method}.mp4"
            candidate.write_bytes(f"candidate-{index}-{method}".encode())
            methods[method] = {
                "path": candidate.name,
                "source_sha256": _sha(source),
                "candidate_sha256": _sha(candidate),
                "profile": blind.METHOD_PROFILES[method],
            }
        clips.append(
            {
                "clip_id": f"clip-{index}",
                "label": f"第 {index + 1} 组",
                "clip_type": clip_types[index],
                "source": source.name,
                "source_sha256": _sha(source),
                "methods": methods,
            }
        )
    return {
        "schema": blind.MANIFEST_SCHEMA,
        "review_id": "dlss-p4-v1",
        "bitrate_kbps": 8000,
        "clips": clips,
    }


def _fake_normalize(**kwargs):
    target = kwargs["target_path"]
    target.write_bytes(b"normalized:" + kwargs["candidate_path"].name.encode())
    return {
        "path": str(target),
        "sha256": _sha(target),
        "encoding_contract": blind._encoding_contract(kwargs["bitrate_kbps"]),
        "source_audio_packet_exact": True,
        "source_audio_pcm_exact": True,
    }


def test_p4_package_is_four_way_hash_bound_deterministic_and_blind(
    monkeypatch, tmp_path
):
    manifest = _manifest(tmp_path)
    manifest["clips"][0]["clip_id"] = "dlss_nr_private_source_id"
    manifest["clips"][0]["label"] = "FlashVSR private source note"
    monkeypatch.setattr(blind, "_media_contract", _contract)
    monkeypatch.setattr(blind, "_normalize_candidate", _fake_normalize)
    monkeypatch.setattr(
        blind.validation_tool,
        "_video_screen",
        lambda *_args, **_kwargs: {"quality_ranking": None},
    )
    first = tmp_path / "blind-first"
    second = tmp_path / "blind-second"
    key1 = blind.build_package(manifest, tmp_path, first, 1234)
    key2 = blind.build_package(manifest, tmp_path, second, 1234)

    assert [
        [(side["code"], side["method"]) for side in clip["sides"]]
        for clip in key1["clips"]
    ] == [
        [(side["code"], side["method"]) for side in clip["sides"]]
        for clip in key2["clips"]
    ]
    assert key1["encoding_contract"]["video_encoder"] == "libx264"
    assert key1["encoding_contract"]["bitrate_kbps"] == 8000
    assert key1["encoding_contract"]["audio"].startswith("packet-copy")
    for clip in key1["clips"]:
        assert {side["code"] for side in clip["sides"]} == set(blind.CODES)
        assert {side["method"] for side in clip["sides"]} == set(blind.METHODS)

    page = (first / "blind_review.html").read_text(encoding="utf-8")
    for secret in (
        "realbasicvsr",
        "flashvsr",
        "lanczos_2x",
        "quality_locked",
        "standard",
        "dlss_nr_private_source_id",
        "FlashVSR private source note",
        str(tmp_path),
    ):
        assert secret not in page
    assert blind.REVIEW_SCHEMA in page
    assert "A–D" in page
    assert "已完整观看" in page
    assert "mouth_lipsync" in page
    assert "blocking_failure" in page
    node = shutil.which("node")
    if node:
        script = page.split("<script>", 1)[1].split("</script>", 1)[0]
        checked = subprocess.run(
            [node, "--check", "-"],
            input=script,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
        assert checked.returncode == 0, checked.stderr
    screening = json.loads(
        (first / "mechanical_screening.json").read_text(encoding="utf-8")
    )
    assert screening["quality_ranking"] is None
    assert {
        side["method"]
        for clip in screening["clips"]
        for side in clip["sides"]
    } == set(blind.METHODS)


def test_latin_order_balances_every_method_across_four_positions():
    orders = blind._latin_orders(4321, "review", 4)
    for order in orders:
        assert set(order) == set(blind.METHODS)
    for position in range(4):
        assert Counter(order[position] for order in orders) == Counter(
            {method: 1 for method in blind.METHODS}
        )


def test_manifest_rejects_wrong_profile_source_binding_and_missing_clip_type(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(blind, "_media_contract", _contract)
    manifest = _manifest(tmp_path)
    manifest["clips"][0]["methods"]["dlss_nr"]["profile"] = "strong"
    with pytest.raises(ValueError, match="profile must be"):
        blind._prepare_manifest(manifest, tmp_path)

    manifest = _manifest(tmp_path)
    manifest["clips"][0]["methods"]["dlss_nr"]["source_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="same source hash"):
        blind._prepare_manifest(manifest, tmp_path)

    manifest = _manifest(tmp_path)
    manifest["clips"] = manifest["clips"][:2]
    with pytest.raises(ValueError, match="missing required clip types"):
        blind._prepare_manifest(manifest, tmp_path)


def test_manifest_rejects_non_2x_candidate(monkeypatch, tmp_path):
    manifest = _manifest(tmp_path)

    def wrong_contract(path):
        value = _contract(path)
        if "candidate" in path.name:
            value["width"] = 127
        return value

    monkeypatch.setattr(blind, "_media_contract", wrong_contract)
    with pytest.raises(ValueError, match="exact 2x"):
        blind._prepare_manifest(manifest, tmp_path)


def test_p4_output_evidence_is_never_overwritten(monkeypatch, tmp_path):
    manifest = _manifest(tmp_path)
    monkeypatch.setattr(blind, "_media_contract", _contract)
    output = tmp_path / "existing"
    output.mkdir()
    (output / "old-review.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        blind.build_package(manifest, tmp_path, output, 1)


def _write_video(path: Path, width: int, height: int, *, with_audio: bool) -> None:
    with av.open(str(path), mode="w", format="mp4") as container:
        video = container.add_stream("libx264", rate=24)
        video.width = width
        video.height = height
        video.pix_fmt = "yuv420p"
        video.codec_context.max_b_frames = 0
        video.codec_context.thread_count = 1
        video.codec_context.time_base = Fraction(1, 24)
        video.time_base = Fraction(1, 24)
        video.codec_context.color_primaries = 1
        video.codec_context.color_trc = 1
        video.codec_context.colorspace = 1
        audio = container.add_stream("aac", rate=32000) if with_audio else None
        if audio is not None:
            audio.layout = "mono"
        for index in range(4):
            pixels = np.full((height, width, 3), 30 + index * 30, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts = index
            frame.time_base = Fraction(1, 24)
            for packet in video.encode(frame):
                container.mux(packet)
        for packet in video.encode():
            container.mux(packet)
        if audio is not None:
            samples = np.zeros((1, 4096), dtype=np.float32)
            for offset in range(0, 4096, 1024):
                frame = av.AudioFrame.from_ndarray(
                    samples[:, offset : offset + 1024], format="fltp", layout="mono"
                )
                frame.sample_rate = 32000
                frame.pts = offset
                frame.time_base = Fraction(1, 32000)
                for packet in audio.encode(frame):
                    container.mux(packet)
            for packet in audio.encode():
                container.mux(packet)


def test_real_normalizer_uses_fixed_encoder_and_packet_exact_source_audio(tmp_path):
    source = tmp_path / "source.mp4"
    candidate = tmp_path / "candidate.mp4"
    target = tmp_path / "A.mp4"
    _write_video(source, 64, 48, with_audio=True)
    _write_video(candidate, 128, 96, with_audio=False)
    source_contract = blind._media_contract(source)
    candidate_contract = blind._media_contract(candidate)
    blind._validate_method_contract(
        clip_id="tiny",
        source_contract=source_contract,
        candidate_contract=candidate_contract,
    )
    report = blind._normalize_candidate(
        source_path=source,
        source_contract=source_contract,
        candidate_path=candidate,
        target_path=target,
        bitrate_kbps=1000,
    )
    assert target.is_file()
    assert report["stream"]["codec_name"] == "h264"
    assert report["stream"]["pix_fmt"] == "yuv420p"
    assert report["encoding_contract"]["bitrate_kbps"] == 1000
    assert report["source_audio_packet_exact"] is True
    assert report["source_audio_pcm_exact"] is True
    assert report["validation"]["decoded_video_frames"] == 4
    assert not list(tmp_path.glob("*.partial-*.mp4"))
    assert not list(tmp_path.glob("*.video-only-*.mp4"))
