"""Real AudioWindow/OutputTrim and codec qualification, not learned VAE QA."""
import json
import math
import subprocess
import wave
from fractions import Fraction

import pytest
import torch
import av
import numpy as np

from comfy_extras.nodes_audio import LoadAudio
import folder_paths

from h3_audio_t8_pkg.nodes import MiniMaxH3AudioWindowT8, MiniMaxH3OutputTrimT8
from h3_audio_t8_pkg.h3_av_delivery import save_h3_av_safe
from h3_audio_t8_pkg.ffmpeg_utils import resolve_ffmpeg
from h3_audio_t8_pkg.director_project import file_sha
from h3_audio_t8_pkg.timing import TimingPlan, make_timing_plan, window_audio
from test_h24_audio_media_timeline import _source


@pytest.mark.parametrize("rate", (32000, 44100, 48000))
@pytest.mark.parametrize("start,duration,source_seconds,warmup,cooldown,minimum", [
    (25 / 24, 101 / 24, 8, 0, 0, True),
    (1 / 24, 101 / 24, 1, 0, 0, True),
    (0, 1, 1, 0, 0, True),
    (.25, 1, 2, .5, .25, True),
    (1.01, 2, 5, 1 / 24, .25, False),
    (8, 1, 1, 0, 0, True),
    (math.nextafter(1 / 24, 0), 101 / 24, 1, 0, 0, True),
    (math.nextafter(25 / 24, math.inf), 101 / 24, 8, 0, 0, True),
    (3 / 24, 123 / 24, 1, 0, 0, True),
])
def test_window_then_trim_keeps_requested_integer_source_origin(
        rate, start, duration, source_seconds, warmup, cooldown, minimum):
    # Unique integer-index samples expose a one-sample shift hidden by silence.
    source = torch.arange(round(source_seconds * rate), dtype=torch.float64)
    audio = {"waveform": torch.stack((source, -source)).unsqueeze(0), "sample_rate": rate}
    original = audio["waveform"].clone()
    context, length, trim, seconds, _, raw = MiniMaxH3AudioWindowT8.execute(
        audio, start, duration, warmup, cooldown, minimum).result
    images = torch.arange(length).reshape(-1, 1, 1, 1).float()
    frames, result, _ = MiniMaxH3OutputTrimT8.execute(images, trim, seconds, 24, context).result
    count, origin = round(duration * rate), round(start * rate)
    expected = original[..., origin:origin + count]
    expected = torch.nn.functional.pad(expected, (0, count - expected.shape[-1]))
    assert torch.equal(result["waveform"], expected)
    assert torch.equal(audio["waveform"], original)
    assert frames[0].item() == round(trim * 24)
    assert frames.shape[0] == max(1, round(duration * 24))
    assert context["sample_rate"] == rate
    assert context["waveform"].shape[-1] == round(length * rate / 24)
    report = json.loads(raw)
    assert report["final_trim_start_seconds"] == trim
    assert report["scene_start_seconds"] == start
    assert Fraction(length, 24) >= Fraction(str(duration))


def _decoded_audio(path, samples):
    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        frames = list(container.decode(audio=0))
        cursor = Fraction(0)
        for frame in frames:
            assert Fraction(frame.pts) * frame.time_base == cursor
            cursor += Fraction(frame.samples, frame.sample_rate)
        # AAC's declared duration may round one sample at a fractional frame
        # end. Compare exact sample counts, not two float seconds whose binary
        # error can falsely reject a difference of precisely one sample.
        assert abs(Fraction(stream.duration) * stream.time_base * stream.rate - samples) <= 1
        decoded = np.concatenate([frame.to_ndarray() for frame in frames], axis=1)
        assert decoded.shape[0] == 2 and decoded.shape[1] >= samples
        assert np.isfinite(decoded).all()
        return decoded[:, :samples]


@pytest.mark.parametrize("rate", (32000, 44100, 48000))
@pytest.mark.parametrize("source_frames,start_frame,keep_frames", [(30, 6, 24), (240, 25, 101)])
def test_real_file_window_trim_and_safe_av_match_independent_source_selection(
        tmp_path, monkeypatch, rate, source_frames, start_frame, keep_frames):
    pcm = _source(rate, source_frames)["waveform"][0].numpy()
    pcm16 = np.ascontiguousarray((pcm.T * 32767).round(), dtype="<i2")
    source_path = tmp_path / "source.wav"
    with wave.open(str(source_path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm16.tobytes())
    original_sha = file_sha(source_path)
    # Actual installed Core loader and actual T8 node entry points, not stubs.
    monkeypatch.setattr(folder_paths, "get_input_directory", lambda: str(tmp_path))
    loaded = LoadAudio.execute(source_path.name).result[0]
    expected_source = torch.from_numpy(pcm16.astype(np.float32).T.copy()).unsqueeze(0) / 32768
    assert torch.equal(loaded["waveform"], expected_source)
    context, length, trim, duration, _, report = MiniMaxH3AudioWindowT8.execute(
        loaded, start_frame / 24, keep_frames / 24, 0, 0, True).result
    index = torch.arange(length).reshape(-1, 1, 1, 1)
    images = (index / length).expand(-1, 32, 64, 3).float().clone()
    final_images, final_audio, _ = MiniMaxH3OutputTrimT8.execute(images, trim, duration, 24, context).result
    count, start = round(duration * rate), round(start_frame * rate / 24)
    expected = expected_source[..., start:start + count]
    expected = torch.nn.functional.pad(expected, (0, count - expected.shape[-1]))
    assert torch.equal(final_audio["waveform"], expected)
    assert final_images.shape[0] == keep_frames
    output = tmp_path / "selected.mp4"
    saved = save_h3_av_safe(final_images, final_audio, output)
    intended = math.ceil(keep_frames * rate / 24)
    expected = torch.nn.functional.pad(expected, (0, max(0, intended - count)))[..., :intended]
    control = tmp_path / "source-selection.m4a"
    seconds = f"{keep_frames / 24:.12f}"
    raw = np.ascontiguousarray(expected[0].numpy().T, dtype="<f4").tobytes()
    subprocess.run([resolve_ffmpeg(), "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
                    "-f", "f32le", "-ar", str(rate), "-ac", "2", "-i", "pipe:0",
                    "-c:a", "aac", "-af", f"atrim=end={seconds}", "-t", seconds,
                    "-movflags", "+faststart", "-movie_timescale", str(math.lcm(24, rate)), str(control)], input=raw, capture_output=True,
                   check=True, timeout=30, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    delivered, independent = _decoded_audio(output, intended), _decoded_audio(control, intended)
    assert np.array_equal(delivered, independent)
    assert not np.array_equal(delivered, independent[::-1])
    with av.open(str(output)) as container:
        decoded = list(container.decode(video=0))
        assert len(decoded) == keep_frames
        assert [Fraction(f.pts) * f.time_base for f in decoded] == [Fraction(i, 24) for i in range(keep_frames)]
    assert file_sha(source_path) == original_sha
    assert torch.equal(loaded["waveform"], expected_source)
    (tmp_path / "selection-delivery-evidence.json").write_text(json.dumps({
        "sample_rate": rate, "start_frame": start_frame, "keep_frames": keep_frames,
        "source_sha256": original_sha, "AudioWindow": json.loads(report), "saved": saved,
        "source_selection_PCM_exact": True, "independent_codec_PCM_equal": True,
        "complete_audio_PTS_contiguous": True, "complete_video_PTS_contiguous": True,
        "trained_Audio_VAE_or_human_quality_qualified": False,
    }, indent=2), encoding="utf8")


def test_historical_report_keys_can_still_construct_a_timing_plan():
    plan = make_timing_plan(0, 1, source_duration_seconds=1)
    old = json.loads(plan.report())
    old.pop("scene_start_seconds")
    restored = TimingPlan(**old)
    assert restored.scene_start_seconds is None
    audio = {"waveform": torch.arange(44100).reshape(1, 1, -1).float(), "sample_rate": 44100}
    assert torch.equal(window_audio(audio, plan)["waveform"], window_audio(audio, restored)["waveform"])
