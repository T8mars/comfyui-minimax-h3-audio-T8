from __future__ import annotations

import struct

import pytest

from h3_audio_t8_pkg.mp4_timing import (apply_delta, constant_delta, dominant_delta,
                                        frame_rate, repair_delta, video_timing)


def box(kind, *payload):
    return struct.pack('>I4s', 8 + sum(len(part) for part in payload), kind) + b''.join(payload)


def stbl(stts):
    return box(b'stbl', stts)


def sample_table(entries):
    return box(b'stts', b'\x00\x00\x00\x00', struct.pack('>I', len(entries)),
               b''.join(struct.pack('>II', *entry) for entry in entries))


def movie(timescale, duration, entries):
    mdhd = box(b'mdhd', b'\x00\x00\x00\x00', struct.pack('>IIII', 0, 0, timescale, duration))
    hdlr = box(b'hdlr', b'\x00\x00\x00\x00', b'\x00' * 4, b'vide', b'\x00' * 12)
    trak = box(b'trak', box(b'mdia', mdhd, hdlr,
        box(b'minf', stbl(sample_table(entries)))))
    return box(b'ftyp', b'isom' + b'\x00\x00\x02\x00') + box(b'moov', trak)


def write(tmp_path, data):
    path = tmp_path / 'clip.mp4'
    path.write_bytes(data)
    return path


def test_one_duration_run_is_already_constant_rate(tmp_path):
    source = write(tmp_path, movie(12288, 977 * 512, [(977, 512)]))
    timing = video_timing(source)
    assert constant_delta(timing) == 512
    assert frame_rate(timing, 512) == 24
    # A constant table is also explained by one delta; the node passes it through
    # on constant_delta, so this never reaches a rewrite.
    assert repair_delta(timing) == 512


def test_one_miswritten_frame_duration_is_repaired_without_touching_frames(tmp_path):
    # The exact shape of a bad splice: one 516-tick frame shifts every later timestamp.
    source = write(tmp_path, movie(12288, 977 * 512 + 4, [(174, 512), (1, 516), (802, 512)]))
    timing = video_timing(source)
    assert constant_delta(timing) is None
    assert dominant_delta(timing) == 512
    assert repair_delta(timing) == 512

    destination = tmp_path / 'fixed.mp4'
    apply_delta(source, destination, timing, 512)
    fixed = video_timing(destination)
    # Runs are rewritten in place rather than merged, since merging would resize the
    # table and move every following box. What matters is the table is now constant.
    assert fixed['entries'] == [(174, 512), (1, 512), (802, 512)]
    assert constant_delta(fixed) == 512
    assert fixed['duration_ticks'] == 977 * 512
    assert destination.stat().st_size == source.stat().st_size
    # Every rewritten byte has to sit in the duration table, never in the media payload.
    original, patched = source.read_bytes(), destination.read_bytes()
    table = range(timing['deltas_at'], timing['deltas_at'] + 8 * len(timing['entries']))
    duration = range(timing['duration_at'], timing['duration_at'] + timing['duration_width'])
    touched = [i for i in range(len(original)) if original[i] != patched[i]]
    assert touched
    assert all(i in table or i in duration for i in touched)


def test_genuine_variable_rate_is_never_silently_rewritten(tmp_path):
    # Deltas that no single rate can explain must fall through to a re-encode.
    entries = [(100, 512), (100, 513), (100, 40960), (100, 20481)]
    duration = sum(count * delta for count, delta in entries)
    timing = video_timing(write(tmp_path, movie(12288, duration, entries)))
    assert constant_delta(timing) is None
    assert repair_delta(timing) is None
    assert dominant_delta(timing) == 512


def test_rate_change_beyond_the_drift_limit_is_not_treated_as_a_typo(tmp_path):
    # Half the clip at 24 fps and half at 30 fps: no single rate explains the duration.
    entries = [(300, 512), (300, 341)]
    timing = video_timing(write(tmp_path, movie(12288, sum(c * d for c, d in entries), entries)))
    assert constant_delta(timing) is None
    assert dominant_delta(timing) == 341
    assert repair_delta(timing) is None


def test_non_iso_bmff_source_reports_no_timing(tmp_path):
    source = tmp_path / 'clip.mkv'
    source.write_bytes(b'\x1a\x45\xdf\xa3' + b'\x00' * 64)
    assert video_timing(source) is None


def test_a_track_without_a_sample_table_is_skipped(tmp_path):
    sound = box(b'trak', box(b'mdia',
        box(b'mdhd', b'\x00\x00\x00\x00', struct.pack('>IIII', 0, 0, 48000, 48000)),
        box(b'hdlr', b'\x00\x00\x00\x00', b'\x00' * 4, b'soun', b'\x00' * 12),
        box(b'minf', stbl(box(b'stco', b'\x00\x00\x00\x00', struct.pack('>I', 0))))))
    data = box(b'ftyp', b'isom' + b'\x00\x00\x02\x00') + box(b'moov', sound)
    assert video_timing(write(tmp_path, data)) is None


def test_audio_only_track_does_not_answer_for_a_video_request(tmp_path):
    timing = video_timing(write(tmp_path, movie(12288, 512, [(1, 512)])))
    assert timing['frames'] == 1
    with pytest.raises(ZeroDivisionError):
        frame_rate(timing, 0)
