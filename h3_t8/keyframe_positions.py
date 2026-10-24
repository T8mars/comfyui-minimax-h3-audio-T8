"""Pure native H3 keyframe position math, shared without model imports."""
from __future__ import annotations

import math


def resolve_position(mode, value, frame_count, *, fps=24, label="keyframe"):
    if type(frame_count) is not int or frame_count < 3:
        raise ValueError(f"{label} needs an actual frame count with an interior")
    if not isinstance(mode, str) or mode not in {"frame", "seconds", "percent"}:
        raise ValueError(f"{label} has invalid position_mode")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{label} position must be finite")
    if mode == "frame":
        rounded = round(value)
        if not math.isclose(value, rounded, rel_tol=0, abs_tol=1e-9):
            raise ValueError(f"{label} frame position must be an integer")
        frame = int(rounded)
    elif mode == "seconds":
        if value < 0:
            raise ValueError(f"{label} seconds must be non-negative")
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"{label} needs a finite positive frame rate")
        scaled = value * fps
        if not math.isfinite(scaled):
            raise ValueError(f"{label} seconds resolve to a nonfinite frame")
        frame = int(math.floor(scaled + .5))
    else:
        if not 0 <= value <= 100:
            raise ValueError(f"{label} percent must be between 0 and 100")
        frame = int(math.floor(value / 100 * (frame_count - 1) + .5))
    if frame <= 0 or frame >= frame_count - 1:
        raise ValueError(f"{label} resolves to frame {frame}; middle keyframes must be in 1..{frame_count - 2}")
    return frame
