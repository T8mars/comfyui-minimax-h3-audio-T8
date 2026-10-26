"""Real MOV/MP4 movie-clock controls; never alters sampling or codec choices."""
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess

import av
import numpy as np

from h3_audio_t8_pkg import director_film_export as exporter
from h3_audio_t8_pkg.director_project import file_sha
from h3_audio_t8_pkg.ffmpeg_utils import resolve_ffmpeg
from test_h24_film_export_recovery import case, reserve, run


def probe(path):
    report = {"sha256": file_sha(path)}
    with av.open(str(path)) as media:
        frames = list(media.decode(video=0))
        report["video_pts"] = [str(Fraction(f.pts) * f.time_base) for f in frames]
    with av.open(str(path)) as media:
        stream = media.streams.audio[0]
        frames = list(media.decode(audio=0))
        pcm = np.concatenate([f.to_ndarray() for f in frames], axis=-1)
        report["audio"] = {
            "rate": stream.rate,
            "declared_samples": str(Fraction(stream.duration) * stream.time_base * stream.rate),
            # Packed s16 stereo is [1, 2*N], planar AAC is [2, N]. The
            # decoder's per-channel frame.samples is the common sample clock.
            "decoded_samples": sum(frame.samples for frame in frames),
            "format": frames[0].format.name,
            "channels": len(stream.layout.channels),
            "pcm_sha256": hashlib.sha256(pcm.tobytes()).hexdigest(),
            "pts": [str(Fraction(f.pts) * f.time_base) for f in frames],
        }
    with av.open(str(path)) as media:
        packets = [bytes(packet) for packet in media.demux(audio=0) if packet.size]
        report["audio"]["packets_sha256"] = hashlib.sha256(b"".join(packets)).hexdigest()
    with av.open(str(path)) as media:
        packets = [bytes(packet) for packet in media.demux(video=0) if packet.size]
        report["video_packets_sha256"] = hashlib.sha256(b"".join(packets)).hexdigest()
    return report


def test_real_intermediate_and_final_movie_clock_matrix(tmp_path, monkeypatch):
    store, output, project, film, options = case(tmp_path)
    originals = {path: file_sha(path) for path in output.glob("*.mp4")}
    sources = {path.name: probe(path) for path in originals}
    original_run = exporter._run
    variants = {}
    for intermediate, final in ((1000, 1000), (48000, 1000), (1000, 48000), (48000, 48000)):
        observed = {}

        def encode(arguments, log):
            args = list(arguments)
            # Explicit test-only controls also remain useful after production
            # adopts the qualified clock. The original 1000-tick negative is
            # retained, not a relaxation of the independent 82000 assertion.
            while "-movie_timescale" in args:
                index = args.index("-movie_timescale")
                del args[index:index + 2]
            name = Path(args[-1]).name
            clock = final if name == "film.mp4" else intermediate
            args[-1:-1] = ["-movie_timescale", str(clock)]
            original_run(args, log)
            observed[name] = probe(Path(args[-1]))

        with monkeypatch.context() as patch:
            patch.setattr(exporter, "_run", encode)
            job = reserve(store, project, film, options)
            result = run(store, project, film, job)
            assert result["state"] == "success", result
        variants[f"{intermediate}/{final}"] = observed
        assert all(file_sha(path) == digest for path, digest in originals.items())
    executable = resolve_ffmpeg()
    version = subprocess.run([executable, "-version"], check=True, capture_output=True, text=True, timeout=10,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
    evidence = {"sources": sources, "variants": variants, "source_files_unchanged": True,
                "ffmpeg": {"path": executable, "sha256": file_sha(Path(executable)), "version": version}}
    (tmp_path / "movie-clock-evidence.json").write_text(json.dumps(evidence, indent=2), encoding="utf8")
    for name, observed in variants.items():
        assert observed["film.mp4"]["video_pts"] == [str(Fraction(i, 24)) for i in range(41)]
        expected = 82000 if name.endswith("/48000") else 81984
        assert Fraction(observed["film.mp4"]["audio"]["declared_samples"]) == expected
    # Intermediates already contain all requested PCM even with the old
    # millisecond movie clock. Only the final movie clock controls this red.
    for index, expected in enumerate((24000, 24000, 34000)):
        rows = [row[f"clip-{index:04}.mov"]["audio"] for row in variants.values()]
        assert {Fraction(row["declared_samples"]) for row in rows} == {expected}
        assert {row["decoded_samples"] for row in rows} == {expected}
        assert len({row["pcm_sha256"] for row in rows}) == 1
    finals = [row["film.mp4"]["audio"] for row in variants.values()]
    assert len({row["pcm_sha256"] for row in finals}) == 1
    assert len({row["packets_sha256"] for row in finals}) == 1
    assert len({row["film.mp4"]["video_packets_sha256"] for row in variants.values()}) == 1
