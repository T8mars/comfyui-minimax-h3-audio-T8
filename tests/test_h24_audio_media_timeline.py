"""Real CPU resample/save/export/decode, not trained Audio VAE or listening QA."""
from copy import deepcopy
from fractions import Fraction
import json
import math
import subprocess
import uuid

import av
import numpy as np
import pytest
import torch
import torchaudio

from h3_audio_t8_pkg.core import encode_audio_once
from h3_audio_t8_pkg.director_film import prepare_film
from h3_audio_t8_pkg.director_film_export import POLICY, reserve_export, run_export, export_file
from h3_audio_t8_pkg.director_project import ProjectStore, file_sha, new_project
from h3_audio_t8_pkg.h3_av_delivery import save_h3_av_safe
from h3_audio_t8_pkg.ffmpeg_utils import resolve_ffmpeg
from h3_audio_t8_pkg.source_audio_plan import make_plan, sample_at, window


class CaptureAudioVAE:
    """Observe the actual core resampler input boundary; no learned encoding."""
    audio_sample_rate = 32000

    def encode(self, value):
        self.pcm = value.movedim(-1, 1).clone()
        return torch.zeros(1, 32, 2, math.ceil(value.shape[1] / 800))


def _source(rate, frames=240):
    time = torch.arange(sample_at(frames, rate), dtype=torch.float32) / rate
    channels = []
    for hz, centers in ((431, (.12, 1.2, 4.18, 4.25, 6.7, 8.17, 8.25, 9.87)),
                        (743, (.22, 1.7, 4.16, 4.28, 6.9, 8.15, 8.28, 9.91))):
        envelope = .025 + sum(.2 * torch.exp(-((time - center) / .022) ** 2) for center in centers)
        channels.append(envelope * torch.sin(2 * torch.pi * (hz * time + 3 * time ** 2)))
    return {"waveform": torch.stack(channels).unsqueeze(0), "sample_rate": rate}


def _decode(path):
    with av.open(str(path)) as media:
        frames = list(media.decode(video=0))
        assert len(frames) == 240
        pts = [float(frame.pts * frame.time_base) for frame in frames]
        assert pts == pytest.approx(np.arange(240) / 24, abs=1e-6)
        picture = [frame.to_ndarray(format="rgb24").mean() for frame in frames]
    with av.open(str(path)) as media:
        stream = media.streams.audio[0]
        assert stream.codec_context.channels == 2
        rate = stream.codec_context.sample_rate
        duration = float(stream.duration * stream.time_base)
        audio_frames = list(media.decode(audio=0))
        assert float(audio_frames[0].pts * audio_frames[0].time_base) == pytest.approx(0, abs=1 / rate)
        pcm = np.concatenate([frame.to_ndarray() for frame in audio_frames], axis=1)
    assert abs(duration - 10) <= 1 / rate
    assert pcm.shape[1] >= rate * 10 and np.isfinite(pcm).all()
    return pcm[:, :rate * 10], rate, duration, picture


def _audio_only_control(root, master, cuts):
    """Independent audio-only codec chain, with no video/saver/export code.

    AAC changes transient energy shape. Compare pipeline timing to the same
    codec policy, not a sub-ms energy-centroid promise for lossy compression.
    """
    root.mkdir()
    ffmpeg = resolve_ffmpeg()

    def run(arguments, raw=None):
        subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-n", *arguments],
                       input=raw, check=True, capture_output=True, timeout=30,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))

    clips, waves = [], []
    for index, (lo, hi) in enumerate(zip(cuts, cuts[1:])):
        seconds = f"{(hi-lo)/24:.12f}"
        pcm = master[0, :, sample_at(lo, 32000):sample_at(hi, 32000)]
        expected_samples = math.ceil((hi-lo)*32000/24)
        pcm = torch.nn.functional.pad(pcm, (0, expected_samples-pcm.shape[-1]))
        raw = np.ascontiguousarray(pcm.numpy().T, dtype="<f4").tobytes()
        clip = root / f"audio-{index}.m4a"
        run(["-f", "f32le", "-ar", "32000", "-ac", "2", "-i", "pipe:0", "-c:a", "aac",
             "-af", f"atrim=end={seconds}", "-t", seconds, "-movflags", "+faststart",
             "-movie_timescale", str(math.lcm(24, 32000)), str(clip)], raw)
        clips.append(clip)
        if len(cuts) > 2:
            wave = root / f"audio-{index}.wav"
            filters = (f"atrim=start=0.000000000000:duration={seconds},asetpts=PTS-STARTPTS,"
                       f"aresample=48000,apad,atrim=duration={seconds}")
            run(["-i", str(clip), "-af", filters, "-c:a", "pcm_s16le", "-ar", "48000", "-ac", "2",
                 "-t", seconds, "-threads", "1", "-filter_threads", "1", str(wave)])
            waves.append(wave)
    path = clips[0]
    if waves:
        listing = root / "audio.txt"
        listing.write_text("".join(f"file '{wave.name}'\n" for wave in waves), encoding="utf8")
        path = root / "audio-concatenated.m4a"
        run(["-f", "concat", "-safe", "1", "-i", str(listing), "-c:a", "aac", "-b:a", "192k",
             "-ar", "48000", "-ac", "2", "-threads", "1", "-movflags", "+faststart", str(path)])
    with av.open(str(path)) as media:
        rate = media.streams.audio[0].codec_context.sample_rate
        decoded = np.concatenate([frame.to_ndarray() for frame in media.decode(audio=0)], axis=1)
    return decoded[:, :rate * 10], path


@pytest.mark.parametrize("rate", (32000, 44100, 48000))
@pytest.mark.parametrize("segmented", (False, True))
def test_real_resample_save_and_three_clip_export_keep_global_stereo_timeline(tmp_path, rate, segmented):
    source = _source(rate)
    original = source["waveform"].clone()
    encoder = CaptureAudioVAE()
    encode_audio_once(encoder, source)
    assert encoder.pcm.shape == (1, 2, 320000)
    assert torch.equal(source["waveform"], original)
    master = {"waveform": encoder.pcm, "sample_rate": 32000}
    cuts = [0, 101, 197, 240] if segmented else [0, 240]
    plan, timeline = make_plan(master, cuts, [], [{"role_id": "A", "reference_ordinal": 1}])
    finals, reports, records, inputs = [], [], [], []
    store = ProjectStore(tmp_path / "user", tmp_path / "input")
    output = tmp_path / "output"
    output.mkdir()
    project = new_project()
    template = deepcopy(project["doc"]["shots"][0])
    project["doc"]["shots"] = []
    for index, (lo, hi) in enumerate(zip(cuts, cuts[1:])):
        drive, final, row = window(plan, master, index)
        assert final["waveform"].shape[-1] == sample_at(hi, 32000) - sample_at(lo, 32000)
        assert torch.equal(drive["waveform"][..., :row["keep_samples"]], final["waveform"])
        assert not drive["waveform"][..., row["keep_samples"]:].count_nonzero()
        finals.append(final["waveform"])
        images = torch.full((hi - lo, 32, 64, 3), .2 + index * .2)
        path = output / f"clip-{index}.mp4"
        reports.append(save_h3_av_safe(images, final, path))
        inputs.append((path, file_sha(path)))
        shot = {**deepcopy(template), "id": str(uuid.uuid4()), "name": f"part-{index}",
                "adoptedResultId": str(uuid.uuid4()), "duration": (hi - lo) / 24}
        project["doc"]["shots"].append(shot)
        records.append({"shot_id": shot["id"], "prompt_id": shot["adoptedResultId"], "state": "success",
                        "outputs": {"save": {"images": [{"filename": path.name, "type": "output"}]}}})
    assert torch.equal(torch.cat(finals, -1), master["waveform"])
    if segmented:
        project["current"] = project["doc"]["shots"][0]["id"]
        film = prepare_film(store, project, records, output)
        assert film["ready"] and len(film["entries"]) == 3
        job_id = str(uuid.uuid4())
        reserve_export(store, project["id"], film["id"], job_id,
                       {"policy": POLICY, "confirm_encoding": True, "width": 64, "height": 32})
        result = run_export(store, project["id"], film["id"], job_id)
        assert result["state"] == "success", result.get("error")
        delivered = export_file(store, project["id"], film["id"], job_id)
    else:
        delivered = inputs[0][0]
    decoded, final_rate, duration, picture = _decode(delivered)
    codec_control, control_path = _audio_only_control(tmp_path / "audio-only-control", master["waveform"], cuts)
    # This checks every intended sample, including both cut boundaries and tail.
    assert np.array_equal(decoded, codec_control)
    assert not np.array_equal(decoded, np.roll(codec_control, round(.01 * final_rate), axis=1))
    assert not np.array_equal(decoded, codec_control[::-1])
    expected = master["waveform"]
    if final_rate != 32000:
        expected = torchaudio.functional.resample(expected, 32000, final_rate)
    expected = expected[0].numpy()
    evidence = []
    # Regions include both awkward cut boundaries and the final marker.
    for channel in range(2):
        for lo, hi in ((0, 1), (4.10, 4.34), (8.08, 8.36), (9.8, 10)):
            start, end = round(lo * final_rate), round(hi * final_rate)
            reference, actual = expected[channel, start:end], decoded[channel, start:end]
            correlation = float(np.corrcoef(reference, actual)[0, 1])
            wrong = float(np.corrcoef(reference, decoded[1 - channel, start:end])[0, 1])
            axis = np.arange(end - start)
            def center(values):
                return np.sum(axis * values ** 2) / np.sum(values ** 2)
            raw_centroid_difference = float(abs(center(reference) - center(actual)))
            drift_samples = float(abs(center(codec_control[channel, start:end]) - center(actual)))
            assert correlation > .94 and abs(wrong) < .3
            assert drift_samples < final_rate / 1000
            evidence.append(dict(channel=channel, start=lo, end=hi, correlation=correlation,
                                 wrong_channel_correlation=wrong, codec_control_center_drift_samples=drift_samples,
                                 raw_PCM_energy_centroid_difference_samples=raw_centroid_difference))
    for index, cut in enumerate(cuts[:-1]):
        assert picture[cut] == pytest.approx((.2 + index * .2) * 255, abs=4)
    assert all(file_sha(path) == digest for path, digest in inputs)
    (tmp_path / "audio-timeline-evidence.json").write_text(json.dumps({
        "input_rate": rate, "input_samples": original.shape[-1], "resampled_samples": 320000,
        "final_rate": final_rate, "stream_duration": duration, "frames": 240, "cuts": cuts,
        "timeline": timeline, "saver_reports": reports, "marker_measurements": evidence,
        "output": str(delivered), "output_sha256": file_sha(delivered),
        "independent_audio_only_control": str(control_path), "decoded_codec_control_sample_equal": True,
        "trained_Audio_VAE_or_perceptual_quality_qualified": False,
    }, indent=2), encoding="utf8")


def test_88200_samples_are_resampled_not_relabelled_or_silently_trimmed():
    source = _source(44100, 48)
    correct, mislabelled = CaptureAudioVAE(), CaptureAudioVAE()
    encode_audio_once(correct, source)
    encode_audio_once(mislabelled, {**source, "sample_rate": 32000})
    assert source["waveform"].shape[-1] == 88200
    assert correct.pcm.shape[-1] == 64000
    assert mislabelled.pcm.shape[-1] == 88200
    assert Fraction(mislabelled.pcm.shape[-1] - correct.pcm.shape[-1], 32000) == Fraction(121, 160)
