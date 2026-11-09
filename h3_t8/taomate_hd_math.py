# Adapted by T8 on 2026-11-05 from slmonker/ComfyUI-TaoMateHD
# 1e4fb7c76ed15918a89492f78dbf2250484d1c3f, progressive.py.
# MiniMax H3 Community License; see THIRD_PARTY_NOTICES/TAOMATE-HD-LICENSE.txt.
# Numerical reference helpers retained; the T8 runtime is separate.
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import torch
import torch.nn.functional as functional

CANVAS_MULTIPLE = 32
VIDEO_LATENT_SPATIAL_COMPRESSION = 16
VIDEO_CHANNELS = 24
AUDIO_CHANNELS = 32
AUDIO_STREAMS = 2
TEACHER_STATE_INDICES = (0, 16, 33, 49)
SWITCH_STATE_INDEX = 33
DEFAULT_LOW_SCALE = 480.0 / 768.0


@dataclass(frozen=True)
class ProgressiveCanvas:
    width: int
    height: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.width, bool)
            or isinstance(self.height, bool)
            or not isinstance(self.width, int)
            or not isinstance(self.height, int)
            or self.width <= 0
            or self.height <= 0
            or self.width % CANVAS_MULTIPLE
            or self.height % CANVAS_MULTIPLE
        ):
            raise ValueError(
                f"canvas dimensions must be positive multiples of {CANVAS_MULTIPLE}, "
                f"got {self.width}x{self.height}"
            )

    @classmethod
    def from_video_latent(cls, video: torch.Tensor) -> "ProgressiveCanvas":
        if video.ndim != 5 or int(video.shape[1]) != VIDEO_CHANNELS:
            raise ValueError(
                "expected H3 video latent [B,24,T,H,W], got "
                f"{tuple(int(v) for v in video.shape)}"
            )
        return cls(
            width=int(video.shape[-1]) * VIDEO_LATENT_SPATIAL_COMPRESSION,
            height=int(video.shape[-2]) * VIDEO_LATENT_SPATIAL_COMPRESSION,
        )

    @property
    def latent_width(self) -> int:
        return self.width // VIDEO_LATENT_SPATIAL_COMPRESSION

    @property
    def latent_height(self) -> int:
        return self.height // VIDEO_LATENT_SPATIAL_COMPRESSION

    def as_dict(self) -> dict[str, int]:
        return {
            "width": self.width,
            "height": self.height,
            "latent_width": self.latent_width,
            "latent_height": self.latent_height,
        }


def derive_low_canvas(high: ProgressiveCanvas, low_scale: float) -> ProgressiveCanvas:
    if not math.isfinite(low_scale) or not 0.0 < low_scale < 1.0:
        raise ValueError(f"low_scale must be finite and inside (0,1), got {low_scale!r}")
    low_width = max(
        CANVAS_MULTIPLE,
        int(round((high.width * low_scale) / CANVAS_MULTIPLE)) * CANVAS_MULTIPLE,
    )
    low_height = max(
        CANVAS_MULTIPLE,
        int(round((high.height * low_scale) / CANVAS_MULTIPLE)) * CANVAS_MULTIPLE,
    )
    if low_width >= high.width or low_height >= high.height:
        raise ValueError(
            f"low canvas {low_width}x{low_height} must be smaller than high canvas "
            f"{high.width}x{high.height} on both axes"
        )
    low = ProgressiveCanvas(width=low_width, height=low_height)
    half = CANVAS_MULTIPLE / 2.0
    low_aspect = ((low.width - half) / (low.height + half), (low.width + half) / (low.height - half))
    high_aspect = (
        (high.width - half) / (high.height + half),
        (high.width + half) / (high.height - half),
    )
    if max(low_aspect[0], high_aspect[0]) > min(low_aspect[1], high_aspect[1]):
        raise ValueError(
            f"low canvas {low_width}x{low_height} does not preserve the high canvas aspect"
        )
    return low


def teacher_sigmas(*, video_shift: float = 12.0) -> tuple[float, float, float, float]:
    """Return the four TaoMate teacher states 0, 16, 33, 49."""
    if not math.isfinite(video_shift) or video_shift <= 0:
        raise ValueError(f"video_shift must be positive and finite, got {video_shift!r}")
    base = torch.linspace(1.0, 0.0, 50, dtype=torch.float32)
    shifted = video_shift * base / (1.0 + (video_shift - 1.0) * base)
    values = tuple(float(shifted[index]) for index in TEACHER_STATE_INDICES)
    return values  # type: ignore[return-value]


def progressive_stage_sigmas(
    *, video_shift: float = 12.0
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    sigma_0, sigma_16, sigma_33, sigma_49 = teacher_sigmas(video_shift=video_shift)
    return (
        (sigma_0, sigma_16),
        (sigma_16, sigma_33),
        (sigma_33, sigma_49),
    )


def pixel_frames_to_latent_t(frame_count: int) -> int:
    if frame_count < 5 or frame_count % 17 != 5:
        raise ValueError(f"H3 frame count must be 17k+5, got {frame_count}")
    return 2 if frame_count <= 5 else ((frame_count - 5) // 17) * 5 + 2


def latent_t_to_pixel_frames(latent_t: int) -> int:
    if latent_t < 2 or (latent_t - 2) % 5:
        raise ValueError(f"H3 video latent length must be 5k+2, got {latent_t}")
    if latent_t == 2:
        return 5
    return 17 * ((latent_t - 2) // 5) + 5


def _resize_spatial(tensor: torch.Tensor, *, height: int, width: int) -> torch.Tensor:
    if tensor.ndim != 5:
        raise ValueError(f"spatial resize expects rank 5, got {tuple(tensor.shape)}")
    batch, channels, temporal, source_h, source_w = tensor.shape
    if source_h == height and source_w == width:
        return tensor
    frames = tensor.permute(0, 2, 1, 3, 4).reshape(
        batch * temporal, channels, source_h, source_w
    )
    resized = functional.interpolate(
        frames, size=(height, width), mode="bilinear", align_corners=False
    )
    return resized.reshape(batch, temporal, channels, height, width).permute(
        0, 2, 1, 3, 4
    ).contiguous()


@lru_cache(maxsize=32)
def _orthonormal_lowpass_projection(low_size: int, high_size: int) -> torch.Tensor:
    if low_size <= 0 or high_size <= low_size:
        raise ValueError(
            f"projection requires 0 < low_size < high_size, got {low_size}, {high_size}"
        )
    basis = torch.eye(high_size, dtype=torch.float64).reshape(high_size, 1, high_size)
    resize = functional.interpolate(
        basis, size=low_size, mode="linear", align_corners=False
    )[:, 0, :].transpose(0, 1)
    gram = resize @ resize.transpose(0, 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    inverse_sqrt = (
        eigenvectors
        @ torch.diag(eigenvalues.clamp_min(1e-12).rsqrt())
        @ eigenvectors.transpose(0, 1)
    )
    return (inverse_sqrt @ resize).to(dtype=torch.float32).contiguous()


def coupled_video_noise(
    *,
    low_canvas: ProgressiveCanvas,
    high_canvas: ProgressiveCanvas,
    temporal_length: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create TaoMate's coupled low/high Gaussian video noise.

    Returns ``(low_noise, high_noise)`` in native latent layout
    ``[1,24,T,H,W]``. High noise is sampled per temporal latent unit, matching
    the upstream full-clip and streaming reconstruction rule.
    """
    if temporal_length <= 0:
        raise ValueError(f"temporal_length must be positive, got {temporal_length!r}")
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed!r}")
    high_slices = []
    for temporal_index in range(temporal_length):
        generator = torch.Generator(device="cpu").manual_seed(
            (int(seed) + temporal_index * 1_000_003) % (2**63)
        )
        high_slices.append(
            torch.randn(
                1,
                VIDEO_CHANNELS,
                1,
                high_canvas.latent_height,
                high_canvas.latent_width,
                generator=generator,
                dtype=torch.float32,
                device="cpu",
            )
        )
    high_noise = torch.cat(high_slices, dim=2)
    height_projection = _orthonormal_lowpass_projection(
        low_canvas.latent_height, high_canvas.latent_height
    )
    width_projection = _orthonormal_lowpass_projection(
        low_canvas.latent_width, high_canvas.latent_width
    )
    low_height = torch.einsum("lh,bcthw->bctlw", height_projection, high_noise)
    low_noise = torch.einsum("kw,bctlw->bctlk", width_projection, low_height).contiguous()
    return low_noise.float(), high_noise.float()


def audio_noise(
    *,
    audio_shape: tuple[int, ...],
    high_canvas: ProgressiveCanvas,
    temporal_length: int,
    seed: int,
) -> torch.Tensor:
    """Create the stereo H3 audio noise consumed by the native latent layout."""
    if len(audio_shape) != 4:
        raise ValueError(f"expected audio latent [B,32,2,T], got {audio_shape}")
    batch, channels, streams, audio_t = (int(value) for value in audio_shape)
    if batch != 1 or channels != AUDIO_CHANNELS or streams != AUDIO_STREAMS or audio_t <= 0:
        raise ValueError(
            "expected H3 audio latent [1,32,2,T], got "
            f"{tuple(int(v) for v in audio_shape)}"
        )
    generator = torch.Generator(device="cpu").manual_seed(int(seed) % (2**63))
    # Upstream consumes one full high-video draw before creating audio noise.
    torch.randn(
        1,
        VIDEO_CHANNELS,
        temporal_length,
        high_canvas.latent_height,
        high_canvas.latent_width,
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    )
    rows = torch.randn(
        audio_t * AUDIO_STREAMS,
        AUDIO_CHANNELS,
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    )
    return rows.reshape(AUDIO_STREAMS, audio_t, AUDIO_CHANNELS).permute(2, 0, 1).unsqueeze(0)


def lift_video_latent(
    *,
    low_video: torch.Tensor,
    low_noise: torch.Tensor,
    high_noise: torch.Tensor,
    sigma: float,
    low_canvas: ProgressiveCanvas,
    high_canvas: ProgressiveCanvas,
) -> torch.Tensor:
    """Lift the low-resolution state to the high grid with coupled noise."""
    if low_video.ndim != 5 or low_noise.ndim != 5 or high_noise.ndim != 5:
        raise ValueError("lift inputs must be rank-5 video latents")
    if tuple(low_video.shape) != tuple(low_noise.shape):
        raise ValueError(
            f"low state/noise shapes differ: {tuple(low_video.shape)} vs {tuple(low_noise.shape)}"
        )
    expected_high = (
        int(low_video.shape[0]),
        VIDEO_CHANNELS,
        int(low_video.shape[2]),
        high_canvas.latent_height,
        high_canvas.latent_width,
    )
    if tuple(high_noise.shape) != expected_high:
        raise ValueError(
            f"high noise shape {tuple(high_noise.shape)} != expected {expected_high}"
        )
    interpolated = _resize_spatial(
        low_video.float(), height=high_canvas.latent_height, width=high_canvas.latent_width
    )
    projected_low_noise = _resize_spatial(
        low_noise.float(), height=high_canvas.latent_height, width=high_canvas.latent_width
    )
    lifted = interpolated + float(sigma) * (high_noise.float() - projected_low_noise)
    return lifted.contiguous()
