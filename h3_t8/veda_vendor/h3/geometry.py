# SPDX-License-Identifier: MIT
# Vendored from veda-sparse/Miowtion at 3bc5e5ae53e8c0d4539f2bc872265164947d013a.
"""Request geometry: canvas size, frame count and latent grid sizes.

Tile plans and predictors are only valid for the exact token grid a request
produces, so every rounding rule here is part of the contract.
"""

from __future__ import annotations

import dataclasses
import math

FPS = 24
BASE_SHORT_EDGE = 768
MAX_PIXELS = BASE_SHORT_EDGE * 1344
CANVAS_MULTIPLE = 32
MIN_ASPECT_RATIO = 1.0 / 4.0
MAX_ASPECT_RATIO = 4.0
# Pixels per latent cell (VAE f16) and latent cells per token (patch 1x2x2).
VAE_SPATIAL_STRIDE = 16
PATCH_HW = 2
AUDIO_LATENT_HZ = 40.0
AUDIO_CHANNELS = 2
MAX_DURATION_SECONDS = 15.0


def _nearest_multiple(value: float, multiple: int) -> int:
    return max(multiple, int(round(float(value) / multiple)) * multiple)


def resolve_canvas(aspect_w: int, aspect_h: int,
                   short_edge: int = BASE_SHORT_EDGE) -> tuple[int, int]:
    """Returns the pixel canvas (width, height) for an aspect ratio.

    Args:
        aspect_w: Aspect ratio numerator, e.g. 16.
        aspect_h: Aspect ratio denominator, e.g. 9.
        short_edge: Nominal short edge in pixels.

    Returns:
        (width, height), both multiples of 32.

    Raises:
        ValueError: If the ratio is outside [1:4, 4:1].
    """
    ratio = float(aspect_w) / float(aspect_h)
    if not MIN_ASPECT_RATIO <= ratio <= MAX_ASPECT_RATIO:
        raise ValueError(f'aspect ratio {aspect_w}:{aspect_h} out of range')
    if ratio >= 1.0:
        width, height = float(short_edge) * ratio, float(short_edge)
    else:
        width, height = float(short_edge), float(short_edge) / ratio
    area = width * height
    if area > MAX_PIXELS:
        scale = math.sqrt(float(MAX_PIXELS) / area)
        width *= scale
        height *= scale
    return (_nearest_multiple(width, CANVAS_MULTIPLE),
            _nearest_multiple(height, CANVAS_MULTIPLE))


def align_frame_count(frame_count: int) -> int:
    """Snaps a frame count up to the 17n+5 boundary."""
    if frame_count <= 0:
        return 1
    return int(frame_count) + (5 - int(frame_count)) % 17


def video_latent_t(frame_count: int) -> int:
    if frame_count <= 5:
        return 2
    return ((int(frame_count) - 5) // 17) * 5 + 2


def frame_count_from_latent_t(latent_t: int) -> int:
    if latent_t < 2 or (latent_t - 2) % 5:
        raise ValueError(f'latent_t {latent_t} is not of the form 5n+2')
    return 17 * ((latent_t - 2) // 5) + 5


def audio_latent_t(duration_seconds: float) -> int:
    return int(round(float(duration_seconds) * AUDIO_LATENT_HZ))


@dataclasses.dataclass(frozen=True)
class Geometry:
    """A fully resolved t2va target geometry.

    Attributes:
        aspect: Aspect ratio string, e.g. '16:9'.
        width: Canvas width in pixels.
        height: Canvas height in pixels.
        frame_count: Output frame count (17n+5).
        latent_t: Video latent frames.
        latent_h: Video latent height (pixels / 16).
        latent_w: Video latent width (pixels / 16).
        audio_t: Audio latent steps per channel (40 Hz).
    """

    aspect: str
    width: int
    height: int
    frame_count: int
    latent_t: int
    latent_h: int
    latent_w: int
    audio_t: int

    @property
    def duration_seconds(self) -> float:
        return self.frame_count / FPS

    @property
    def video_grid(self) -> tuple[int, int, int]:
        """Token grid (T, H, W) of the target video after 1x2x2 patching."""
        return (self.latent_t, self.latent_h // PATCH_HW,
                self.latent_w // PATCH_HW)

    @property
    def num_video_tokens(self) -> int:
        t, h, w = self.video_grid
        return t * h * w

    @property
    def num_audio_rows(self) -> int:
        return AUDIO_CHANNELS * self.audio_t

    @property
    def name(self) -> str:
        """Stable key, e.g. '16x9_t37' (aspect and latent frames)."""
        return f'{self.aspect.replace(":", "x")}_t{self.latent_t}'


def resolve_geometry(aspect: str, duration_seconds: float,
                     short_edge: int = BASE_SHORT_EDGE) -> Geometry:
    """Resolves (aspect, requested duration) into a target geometry."""
    aspect_w, aspect_h = (int(v) for v in aspect.split(':'))
    width, height = resolve_canvas(aspect_w, aspect_h, short_edge)
    frame_count = align_frame_count(int(round(float(duration_seconds) * FPS)))
    return Geometry(
        aspect=aspect,
        width=width,
        height=height,
        frame_count=frame_count,
        latent_t=video_latent_t(frame_count),
        latent_h=height // VAE_SPATIAL_STRIDE,
        latent_w=width // VAE_SPATIAL_STRIDE,
        audio_t=audio_latent_t(frame_count / FPS),
    )


def geometry_from_latent_t(aspect: str, latent_t: int,
                           short_edge: int = BASE_SHORT_EDGE) -> Geometry:
    """Geometry whose request duration is exactly frame_count / FPS."""
    frame_count = frame_count_from_latent_t(latent_t)
    return resolve_geometry(aspect, frame_count / FPS, short_edge)


def latent_t_ladder(min_seconds: float = 5.0,
                    max_seconds: float = MAX_DURATION_SECONDS) -> list[int]:
    """All reachable latent_t whose actual duration is in [min, max]."""
    ladder = []
    latent_t = 2
    while True:
        seconds = frame_count_from_latent_t(latent_t) / FPS
        if seconds > max_seconds:
            return ladder
        if seconds >= min_seconds:
            ladder.append(latent_t)
        latent_t += 5
