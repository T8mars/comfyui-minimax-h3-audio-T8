"""Constant-rate repair for an ISO-BMFF sample table; no re-encode, no frame rewrite.

A muxer that writes one wrong frame duration shifts every later timestamp, and
that single number is the only thing standing between the source and the official
Topaz route.  Rewriting the table copies the file and patches a few bytes, so the
encoded frames themselves are never touched.
"""
from fractions import Fraction
from pathlib import Path
import shutil
import struct


# A genuinely variable-rate source spans many deltas over its whole duration; one
# miswritten run does not, so the total duration still has to be explainable.
DURATION_DRIFT_LIMIT = Fraction(1, 1000)


def _boxes(stream, start, end):
    while start + 8 <= end:
        stream.seek(start)
        header = stream.read(8)
        if len(header) < 8:
            return
        size, kind = struct.unpack('>I4s', header)
        body = start + 8
        if size == 1:
            extended = stream.read(8)
            if len(extended) < 8:
                return
            size, body = struct.unpack('>Q', extended)[0], start + 16
        elif size == 0:
            size = end - start
        if size < body - start or start + size > end:
            return
        yield kind, body, start + size
        start += size


def _find(stream, start, end, *path):
    """Descend a box path, returning (body_start, end) of the last box."""
    for kind in path:
        found = next(((b, s) for k, b, s in _boxes(stream, start, end) if k == kind), None)
        if found is None:
            return None
        start, end = found
    return start, end


def _load(stream, span):
    stream.seek(span[0])
    return stream.read(span[1] - span[0])


def _mdhd(body):
    version = body[0]
    at = 12 if version == 0 else 20
    return struct.unpack('>I', body[at:at + 4])[0], at + 4, 4 if version == 0 else 8


def _track_timing(stream, start, end):
    mdia = _find(stream, start, end, b'mdia')
    if mdia is None:
        return None
    handler = _find(stream, *mdia, b'hdlr')
    if handler is None or _load(stream, handler)[8:12] != b'vide':
        return None
    mdhd = _find(stream, *mdia, b'mdhd')
    stts = _find(stream, *mdia, b'minf', b'stbl', b'stts')
    if mdhd is None or stts is None:
        return None
    timescale, duration_at, duration_width = _mdhd(_load(stream, mdhd))
    table = _load(stream, stts)
    count = struct.unpack('>I', table[4:8])[0]
    entries = [struct.unpack('>II', table[8 + i * 8:16 + i * 8]) for i in range(count)]
    return {'timescale': timescale, 'entries': entries,
        'deltas_at': stts[0] + 8, 'duration_at': mdhd[0] + duration_at,
        'duration_width': duration_width, 'frames': sum(count for count, _ in entries),
        'duration_ticks': sum(count * delta for count, delta in entries)}


def video_timing(source):
    """Describe the first video track sample table, or None for a non ISO-BMFF source."""
    path = Path(source)
    with path.open('rb') as stream:
        moov = _find(stream, 0, path.stat().st_size, b'moov')
        if moov is None:
            return None
        for kind, track, end in _boxes(stream, *moov):
            if kind == b'trak':
                timing = _track_timing(stream, track, end)
                if timing is not None:
                    return timing
    return None


def constant_delta(timing):
    """The single frame duration of an already constant-rate table."""
    deltas = {delta for _, delta in timing['entries']}
    return deltas.pop() if len(deltas) == 1 else None


def dominant_delta(timing):
    """The duration covering most frames, i.e. this source's closest constant rate."""
    weights = {}
    for count, delta in timing['entries']:
        weights[delta] = weights.get(delta, 0) + count
    return max(weights, key=lambda delta: (weights[delta], -delta))


def repair_delta(timing):
    """The dominant delta when it also explains the whole track, else None for real VFR."""
    if timing['frames'] < 2 or timing['duration_ticks'] <= 0:
        return None
    delta = dominant_delta(timing)
    if abs(timing['frames'] * delta - timing['duration_ticks']) > timing['duration_ticks'] * DURATION_DRIFT_LIMIT:
        return None
    return delta


def frame_rate(timing, delta):
    return Fraction(timing['timescale'], delta)


def apply_delta(source, destination, timing, delta):
    """Copy the file, then rewrite only its stts deltas and the track duration."""
    shutil.copyfile(source, destination)
    with destination.open('r+b') as stream:
        at = timing['deltas_at']
        for index in range(len(timing['entries'])):
            stream.seek(at + index * 8 + 4)
            stream.write(struct.pack('>I', delta))
        stream.seek(timing['duration_at'])
        stream.write((timing['frames'] * delta).to_bytes(timing['duration_width'], 'big'))
