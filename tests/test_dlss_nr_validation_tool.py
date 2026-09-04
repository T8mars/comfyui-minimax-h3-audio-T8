from __future__ import annotations

from fractions import Fraction
import json
from pathlib import Path
import shutil
import subprocess

import av
import numpy as np
from PIL import Image
import pytest
import torch

from h3_audio_t8_pkg.tools import run_dlss_nr_validation as validation


def _write_image(path: Path, width: int = 800, height: int = 640) -> None:
    x = np.linspace(0, 255, width, dtype=np.uint8)
    pixels = np.broadcast_to(x[None, :, None], (height, width, 3)).copy()
    Image.fromarray(pixels, mode="RGB").save(path)


def _write_cut_video(path: Path, frame_count: int = 8) -> None:
    with av.open(str(path), mode="w", format="mp4") as container:
        stream = container.add_stream("libx264", rate=24)
        stream.width = 64
        stream.height = 48
        stream.pix_fmt = "yuv420p"
        stream.codec_context.max_b_frames = 0
        stream.codec_context.time_base = Fraction(1, 24)
        stream.time_base = Fraction(1, 24)
        for index in range(frame_count):
            pixels = np.zeros((48, 64, 3), dtype=np.uint8)
            pixels[..., 0 if index < frame_count // 2 else 2] = 220
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts = index
            frame.time_base = Fraction(1, 24)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def test_p2_runs_exact_real_matrix_serially_and_records_uint8_outputs(
    monkeypatch, tmp_path
):
    source_path = tmp_path / "source.png"
    _write_image(source_path)
    order = []

    def revalidate(_runtime):
        order.append("revalidate")
        return {"device_mapping": {"nvidia": {"index": 0}}}

    def process(_runtime, source, *, mode, scale):
        order.append((mode, scale))
        if scale == 1.0:
            candidate = source.clone()
        else:
            candidate = (
                torch.nn.functional.interpolate(
                    source.permute(0, 3, 1, 2), scale_factor=scale, mode="nearest"
                )
                .permute(0, 2, 3, 1)
                .contiguous()
            )
        return candidate, source, {"mode": mode, "scale": scale}

    def measure(gpu_index, operation):
        assert gpu_index == 0
        return operation(), {"peak_used_mib": 123.0, "minimum_free_mib": 456.0}

    monkeypatch.setattr(validation.dlss, "revalidate_runtime_handle", revalidate)
    monkeypatch.setattr(validation.dlss, "process_image_batch", process)
    monkeypatch.setattr(validation, "_measure_vram", measure)
    output = tmp_path / "out"
    output.mkdir()
    report = validation._run_p2({}, source_path, output)

    assert order == [
        "revalidate",
        ("nr_only", 1.0),
        "revalidate",
        ("sr_only", 2.0),
        "revalidate",
        ("sr_nr", 2.0),
    ]
    assert report["status"] == "REAL_MECHANICAL_PASS_HUMAN_REVIEW_REQUIRED"
    assert report["human_review_complete"] is False
    assert report["source"]["megapixels"] == 0.512
    assert report["baseline"]["width"] == 1600
    assert report["baseline"]["height"] == 1280
    assert [run["name"] for run in report["runs"]] == [
        "dlss_nr_1x",
        "dlss_sr_2x",
        "dlss_sr_nr_2x",
    ]
    for run in report["runs"]:
        assert run["output"]["bits_per_channel"] == 8
        assert run["output"]["rgb_quantized_to_uint8_grid"] is True


def test_p2_rejects_nonrepresentative_image_before_execution(tmp_path):
    source = tmp_path / "small.png"
    _write_image(source, 64, 64)
    output = tmp_path / "out"
    output.mkdir()
    with pytest.raises(ValueError, match="approximately 0.5 MP"):
        validation._run_p2({}, source, output)


def test_blocked_runtime_writes_audit_but_never_starts_media(monkeypatch, tmp_path):
    calls = []
    audit = {"status": "BLOCKED", "errors": ["driver", "runtime"]}
    monkeypatch.setattr(
        validation.dlss,
        "audit_dlss_nr_runtime",
        lambda *_args, **_kwargs: (False, audit),
    )
    monkeypatch.setattr(validation, "_run_p2", lambda *_args: calls.append("p2"))
    output = tmp_path / "blocked"
    with pytest.raises(RuntimeError, match="real feature gate"):
        validation.run_validation(
            models_dir=tmp_path / "models",
            runtime_version="1.2",
            output_dir=output,
            stage="p2",
            accept_external_runtime_license=True,
            dxgi_adapter_index=0,
            cuda_device_index=0,
            image=tmp_path / "unused.png",
        )
    assert calls == []
    assert json.loads((output / "runtime_audit.json").read_text(encoding="utf-8")) == audit
    assert not (output / "validation_report.json").exists()


def _write_input_manifest(bundle: Path, image: Path) -> tuple[Path, str]:
    manifest = bundle / "validation_inputs.json"
    document = {
        "schema": validation.INPUT_MANIFEST_SCHEMA,
        "status": "PREPARED_NOT_DLSS_TESTED",
        "gate_effect": {
            "p2_complete": False,
            "p3_complete": False,
            "p4_complete": False,
            "automatic_promotion": False,
        },
        "operator_confirmations": {
            "speech_phrase": validation.SPEECH_PHRASE,
            "speech_phrase_clearly_audible": True,
            "hard_cut_is_intentional": True,
            "fine_texture_overlay_is_intentional": True,
        },
        "prepared_inputs": {
            "p2_image": {
                "path": image.name,
                "bytes": image.stat().st_size,
                "sha256": validation._sha256_file(image),
            }
        },
    }
    manifest.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest, validation._sha256_file(manifest)


def test_real_runner_hash_binds_prepared_input_manifest(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    image = bundle / "image.png"
    _write_image(image)
    manifest, manifest_sha = _write_input_manifest(bundle, image)

    report = validation._bind_prepared_inputs(
        manifest,
        manifest_sha,
        image=image,
        speech_video=None,
        hard_cut_video=None,
        fine_texture_video=None,
    )
    assert report["manifest_sha256"] == manifest_sha
    assert report["bound_inputs"]["p2_image"]["sha256"] == validation._sha256_file(
        image
    )
    assert report["p2_p4_claimed_complete"] is False


def test_real_runner_rejects_changed_prepared_input(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    image = bundle / "image.png"
    _write_image(image)
    manifest, manifest_sha = _write_input_manifest(bundle, image)
    image.write_bytes(image.read_bytes() + b"changed")

    with pytest.raises(ValueError, match="no longer matches"):
        validation._bind_prepared_inputs(
            manifest,
            manifest_sha,
            image=image,
            speech_video=None,
            hard_cut_video=None,
            fine_texture_video=None,
        )


def test_ready_runtime_requires_hash_bound_inputs_before_media(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        validation.dlss,
        "audit_dlss_nr_runtime",
        lambda *_args, **_kwargs: (True, {"status": "READY"}),
    )
    monkeypatch.setattr(validation, "_run_p2", lambda *_args: calls.append("p2"))
    image = tmp_path / "image.png"
    _write_image(image)

    with pytest.raises(ValueError, match="hash-bound --input-manifest"):
        validation.run_validation(
            models_dir=tmp_path / "models",
            runtime_version="1.2",
            output_dir=tmp_path / "output",
            stage="p2",
            accept_external_runtime_license=True,
            dxgi_adapter_index=0,
            cuda_device_index=0,
            image=image,
        )
    assert calls == []


def test_output_evidence_is_never_overwritten(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    (output / "evidence.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        validation._new_output_directory(output)


def test_file_video_and_hard_cut_screen_are_frame_exact(tmp_path):
    source = tmp_path / "source.mp4"
    candidate = tmp_path / "candidate.mp4"
    _write_cut_video(source)
    _write_cut_video(candidate)
    video = validation.FileBackedVideo(source)
    screen = validation._video_screen(source, candidate, hard_cut=True)

    assert video.get_frame_count() == 8
    assert video.get_dimensions() == (64, 48)
    assert video.get_frame_rate() == 24.0
    assert video.get_active_trim_window() == (0.0, 0.0)
    assert screen["decoded_frame_count"] == 8
    assert screen["black_regression_frames"] == []
    assert screen["freeze_regression_frames"] == []
    assert screen["hard_cut"]["source_has_mechanical_hard_cut"] is True
    assert screen["hard_cut"]["candidate_preserves_cut_transition"] is True
    assert (
        screen["hard_cut"]["post_cut_closer_to_current_source_than_previous_source"]
        is True
    )
    assert screen["quality_ranking"] is None


def test_p3_requires_explicit_phrase_confirmation_before_any_output(tmp_path):
    output = tmp_path / "out"
    output.mkdir()
    with pytest.raises(ValueError, match="operator confirmation"):
        validation._run_p3(
            {},
            {
                "speech": tmp_path / "missing-speech.mp4",
                "hard_cut": tmp_path / "missing-cut.mp4",
                "fine_texture": tmp_path / "missing-text.mp4",
            },
            output,
            speech_phrase_confirmation="",
            hard_cut_source_confirmed=False,
            fine_texture_source_confirmed=False,
            motion_engine="auto",
            crf=18.0,
        )
    assert not (output / "p3-video").exists()


def test_p3_rejects_speech_source_without_audio_before_runtime_execution(
    monkeypatch, tmp_path
):
    speech = tmp_path / "speech.mp4"
    speech.write_bytes(b"placeholder")

    class FakeVideo:
        def __init__(self, path):
            self.path = path.resolve()
            self.width = 960
            self.height = 544
            self.frame_count = 124
            self.rate = 24.0

    monkeypatch.setattr(validation, "FileBackedVideo", FakeVideo)
    monkeypatch.setattr(
        validation.dlss,
        "_file_source_contract",
        lambda _video: (speech.resolve(), {"audio_packets": []}),
    )
    monkeypatch.setattr(
        validation.dlss,
        "revalidate_runtime_handle",
        lambda _runtime: pytest.fail("runtime execution must not start"),
    )
    output = tmp_path / "out"
    output.mkdir()
    with pytest.raises(ValueError, match="audio stream"):
        validation._run_p3(
            {},
            {
                "speech": speech,
                "hard_cut": tmp_path / "unused-cut.mp4",
                "fine_texture": tmp_path / "unused-text.mp4",
            },
            output,
            speech_phrase_confirmation=validation.SPEECH_PHRASE,
            hard_cut_source_confirmed=True,
            fine_texture_source_confirmed=True,
            motion_engine="auto",
            crf=18.0,
        )


def test_p3_review_requires_all_human_quality_dimensions():
    document = validation._p3_review_document(
        [
            {
                "id": "speech",
                "label": "speech",
                "source": "source.mp4",
                "candidate": "candidate.mp4",
            }
        ]
    )
    assert validation.P3_REVIEW_SCHEMA in document
    for value in (
        "mouth_lipsync",
        "face_identity_skin",
        "text_fine_texture",
        "color",
        "temporal_stability",
        "cut_history",
        "audio",
    ):
        assert value in document
    assert "机械 PASS 不代表" in document

    node = shutil.which("node")
    if node:
        script = document.split("<script>", 1)[1].split("</script>", 1)[0]
        checked = subprocess.run(
            [node, "--check", "-"],
            input=script,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
        assert checked.returncode == 0, checked.stderr


def test_vram_sampler_is_monitoring_only_and_stops(monkeypatch):
    samples = iter(
        [
            {"total_mib": 16000.0, "used_mib": 1000.0, "free_mib": 15000.0},
            {"total_mib": 16000.0, "used_mib": 1200.0, "free_mib": 14800.0},
        ]
    )
    last = {"total_mib": 16000.0, "used_mib": 1200.0, "free_mib": 14800.0}

    def sample(_index):
        return next(samples, last)

    monkeypatch.setattr(validation, "_nvidia_memory_sample", sample)
    result, report = validation._measure_vram(0, lambda: "done")
    assert result == "done"
    assert report["sample_count"] >= 2
    assert report["peak_used_mib"] == 1200.0
    assert report["minimum_free_mib"] == 14800.0
