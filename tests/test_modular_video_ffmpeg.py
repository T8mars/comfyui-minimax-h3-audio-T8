"""Real owned codec subprocesses; separate from sampling or human quality."""
import json
from fractions import Fraction
import os
from pathlib import Path
import threading
import time
import sys
from types import SimpleNamespace

import pytest
import torch
from comfy_api.latest import InputImpl, Types

from h3_audio_t8_pkg.modular_sampling import video_ffmpeg as exporter
from h3_audio_t8_pkg.modular_sampling import video_ffmpeg_worker as worker
from h3_audio_t8_pkg.modular_sampling.video_io import component_identity
from tests.test_modular_video_io import components, strict

pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows Job-owned codec")


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64, torch.bfloat16])
def test_bounded_identity_preserves_logical_hash_for_noncontiguous_views(dtype, monkeypatch):
    # Larger than the chunk limit, with noncontiguous RGB and channel/time views.
    rgb = torch.zeros((3, 256, 512, 6), dtype=dtype)[..., ::2]
    wave = torch.arange(320000, dtype=torch.float32).reshape(1, 2, -1)[..., ::2] / 320000
    parts = Types.VideoComponents(images=rgb, audio={"waveform": wave, "sample_rate": 48000},
                                  frame_rate=Fraction(30000, 1001))
    expected = component_identity(parts)
    real = torch.Tensor.contiguous
    copies, checkpoints = [], []
    def bounded(tensor, *args, **kwargs):
        copies.append(tensor.numel() * tensor.element_size())
        assert copies[-1] <= 4 * 1024**2
        return real(tensor, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, "contiguous", bounded)
    assert exporter.bounded_component_identity(parts, lambda: checkpoints.append(True)) == expected
    assert copies and len(checkpoints) >= len(copies)
    def cancelled():
        raise InterruptedError("identity cancelled")
    with pytest.raises(InterruptedError, match="identity cancelled"):
        exporter.bounded_component_identity(parts, cancelled)


@pytest.mark.parametrize("channels", [1, 6])
def test_fractional_rate_and_native_audio_channel_count(tmp_path, channels):
    parts = components()
    parts = Types.VideoComponents(images=parts.images, frame_rate=Fraction(30000, 1001),
        audio={"waveform": parts.audio["waveform"][:, :1].repeat(1, channels, 1), "sample_rate": 16000})
    path, report = exporter.save_isolated(InputImpl.VideoFromComponents(parts), tmp_path, "fractional")
    streams = strict(path)["streams"]
    picture = next(s for s in streams if s["codec_type"] == "video")
    sound = next(s for s in streams if s["codec_type"] == "audio")
    assert picture["avg_frame_rate"] == "2997/100"
    assert sound["channels"] == channels and sound["sample_rate"] == "16000"
    assert report["input_identity"] == component_identity(parts)


@pytest.mark.parametrize("depth,color", [(8, "sRGB"), (8, "HDR"), (8, "HDR PQ"), (10, "sRGB"), (10, "HDR"), (10, "HDR PQ")])
@pytest.mark.parametrize("audio", [False, True])
def test_owned_export_preserves_depth_color_audio_and_large_metadata(tmp_path, depth, color, audio):
    parts = components(audio)
    video = InputImpl.VideoFromComponents(parts, bit_depth=depth, color_space=color)
    metadata = {"prompt": {"unicode": "中文\\#=;\n", "large": "x" * 40000}}
    path, report = exporter.save_isolated(video, tmp_path, "new/video", metadata)
    assert path.name == "video.mp4" and report["strict_decode_verified"] is True
    assert report["quality_accepted"] is False
    assert report["worker"]["decoded_video_format"] == ("rgb48le" if depth == 10 else "rgb24")
    assert report["isolation"]["job_assigned_before_task"] and report["isolation"]["active_after_cleanup"] == 0
    details = strict(path)
    picture = next(s for s in details["streams"] if s["codec_type"] == "video")
    assert (picture["width"], picture["height"], picture["nb_frames"], picture["avg_frame_rate"]) == (96, 64, "17", "24/1")
    assert picture["pix_fmt"] == ("yuv420p10le" if depth == 10 else "yuv420p")
    assert picture["color_transfer"] == {"sRGB": "iec61966-2-1", "HDR": "arib-std-b67", "HDR PQ": "smpte2084"}[color]
    assert json.loads(details["format"]["tags"]["prompt"]) == metadata["prompt"]
    assert not list(path.parent.glob("*.yuv")) and not list(path.parent.glob("*.f32le"))
    assert bool([s for s in details["streams"] if s["codec_type"] == "audio"]) == audio
    again, second = exporter.save_isolated(video, tmp_path, "new/video", metadata)
    assert path != again and path.is_file()
    assert report["worker"]["decoded_sha256"] == second["worker"]["decoded_sha256"]
    assert report["converted_yuv_sha256"] == second["converted_yuv_sha256"]


def test_preflight_rejects_bad_budget_path_memory_or_disk_without_worker(tmp_path, monkeypatch):
    video = InputImpl.VideoFromComponents(components())
    monkeypatch.setattr(exporter, "run_isolated", lambda *a, **k: pytest.fail("No worker should start"))
    for kwargs in ({"timeout": float("nan")}, {"timeout": 0}, {"max_staging_gib": 1e-9}):
        with pytest.raises(ValueError):
            exporter.save_isolated(video, tmp_path, "fail", **kwargs)
    with pytest.raises(ValueError):
        exporter.save_isolated(video, tmp_path, "../escape")
    monkeypatch.setattr(exporter.psutil, "virtual_memory", lambda: SimpleNamespace(available=1))
    with pytest.raises(MemoryError):
        exporter.save_isolated(video, tmp_path, "memory")
    monkeypatch.setattr(exporter.psutil, "virtual_memory", lambda: SimpleNamespace(available=2**40))
    monkeypatch.setattr(exporter.shutil, "disk_usage", lambda _: SimpleNamespace(free=1))
    with pytest.raises(OSError, match="free disk"):
        exporter.save_isolated(video, tmp_path, "disk")
    assert not list(tmp_path.rglob("video.mp4"))


def test_worker_rejects_request_tampering_before_start(tmp_path):
    request = tmp_path / "request.json"
    request.write_text("{}", encoding="utf8")
    with pytest.raises(ValueError, match="identity"):
        worker.execute(request, "a" * 64)


def test_codec_subprocess_drain_is_bounded_and_retains_stage_error():
    command = [sys.executable, "-I", "-S", "-c"]
    with pytest.raises(ValueError, match="bounded response"):
        worker.run([*command, "import sys; sys.stdout.buffer.write(b'x' * (5 * 1024**2))"], "probe")
    with pytest.raises(RuntimeError, match="synthetic_encode exited 7") as error:
        worker.run([*command, "import sys; sys.stderr.buffer.write(b'x' * (5 * 1024**2)); sys.exit(7)"], "synthetic_encode")
    assert len(str(error.value)) < 4200


@pytest.mark.parametrize("mode", ["strict_false", "changed_file", "changed_source"])
def test_bad_worker_or_changed_source_is_never_published_or_retried(tmp_path, monkeypatch, mode):
    parts = components()
    calls = []
    def fake(_script, arguments, **_kwargs):
        calls.append(True)
        owned = Path(arguments[0]).parent
        partial = owned / "video.partial.mp4"
        partial.write_bytes(b"synthetic worker output")
        report = {"status": "validated_not_published", "request_sha": arguments[1],
                  "strict_decode_verified": mode != "strict_false", "file_sha256": exporter.file_sha(partial)}
        (owned / "worker-report.json").write_text(json.dumps(report), encoding="utf8")
        if mode == "changed_file":
            partial.write_bytes(b"changed")
        elif mode == "changed_source":
            parts.images[0, 0, 0, 0] += .1
        return {"status": "complete", "active_after_cleanup": 0}
    monkeypatch.setattr(exporter, "run_isolated", fake)
    with pytest.raises(ValueError, match="changed before publication"):
        exporter.save_isolated(InputImpl.VideoFromComponents(parts), tmp_path, mode)
    assert len(calls) == 1 and not list(tmp_path.rglob("video.mp4"))
    assert len(list(tmp_path.rglob("failure.json"))) == 1


@pytest.mark.parametrize("mode", ["error", "tree_hang"])
def test_failed_or_timed_out_owned_tree_never_publishes(tmp_path, monkeypatch, mode):
    real_run = exporter.run_isolated
    fixture = Path(__file__).parent / "fixtures/dlss_fi_isolation_task.py"
    def faulty(_script, _arguments, **kwargs):
        return real_run(fixture, [mode], timeout=1, check=kwargs["check"])
    monkeypatch.setattr(exporter, "run_isolated", faulty)
    with pytest.raises(exporter.IsolatedTaskError):
        exporter.save_isolated(InputImpl.VideoFromComponents(components()), tmp_path, "failure")
    failure = json.loads(next(tmp_path.rglob("failure.json")).read_text(encoding="utf8"))
    assert failure["isolation"]["active_after_cleanup"] == 0
    assert failure["isolation"]["status"] == ("child_failed" if mode == "error" else "timeout")
    assert not list(tmp_path.rglob("video.mp4"))
    assert list(tmp_path.rglob("frames.yuv"))  # Failure inputs retained, never a completed video.


def test_cancellation_during_owned_worker_preserves_original_exception_and_cleans(tmp_path, monkeypatch):
    real_run = exporter.run_isolated
    fixture = Path(__file__).parent / "fixtures/dlss_fi_isolation_task.py"
    cancel = threading.Event()
    def faulty(_script, _arguments, **kwargs):
        timer = threading.Timer(.5, cancel.set)
        timer.start()
        try:
            return real_run(fixture, ["tree_hang"], timeout=10, check=kwargs["check"])
        finally:
            timer.cancel()
    def check():
        if cancel.is_set():
            raise InterruptedError("original interrupt")
    monkeypatch.setattr(exporter, "run_isolated", faulty)
    started = time.monotonic()
    with pytest.raises(InterruptedError, match="original interrupt"):
        exporter.save_isolated(InputImpl.VideoFromComponents(components()), tmp_path, "cancel", interrupt=check)
    failure = json.loads(next(tmp_path.rglob("failure.json")).read_text(encoding="utf8"))
    assert failure["isolation"]["active_after_cleanup"] == 0 and time.monotonic() - started < 8
    assert not list(tmp_path.rglob("video.mp4"))


@pytest.mark.parametrize("mode,status", [("complete", "complete"), ("flood", "complete"),
                                         ("error", "child_failed"), ("orphan", "child_left_descendants")])
def test_actual_runtime_job_helper_not_old_tools_prototype(mode, status):
    fixture = Path(__file__).parent / "fixtures/dlss_fi_isolation_task.py"
    if status == "complete":
        receipt = exporter.run_isolated(fixture, [mode], timeout=10)
    else:
        with pytest.raises(exporter.IsolatedTaskError) as error:
            exporter.run_isolated(fixture, [mode], timeout=10)
        receipt = error.value.receipt
    assert receipt["status"] == status and receipt["active_after_cleanup"] == 0
    assert receipt["job_assigned_before_task"]
    assert len(receipt["stdout_tail"]) <= 131072 and len(receipt["stderr_tail"]) <= 131072
