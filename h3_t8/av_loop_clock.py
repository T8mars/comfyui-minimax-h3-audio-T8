"""Explicit experimental periodic AV geometry; no node/default registration.

Algorithm reference: pxl-pshr/ComfyUI-MiniMaxH3-Loop, commit
7583d64c315ad8ce6e8e74d989c0b730de10012e (__init__.py).
Source LICENSE is GNU GPL version 3, blob
f288702d2fa16d3cdf0035b15a9fcbc552cd88e7. This is an independently written
thin Core adapter, not the upstream node/interpolator framework. Model weights,
example media and optional interpolation packages are not included.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import torch


SOURCE = {
    "repository": "https://github.com/pxl-pshr/ComfyUI-MiniMaxH3-Loop",
    "commit": "7583d64c315ad8ce6e8e74d989c0b730de10012e",
    "source_utf8_sha256": "65fab9b84c94483f5025fdd7cafd3cb990b32887a00209789556fc22ae144e68",
    "license_commit": "1df6dba2383e4065f328f879f2ddecc1c8ed08ac",
    "license_git_blob": "f288702d2fa16d3cdf0035b15a9fcbc552cd88e7",
    "license": "GNU GPL version 3",
}


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def rational(value):
    value = Fraction(value)
    return {"numerator": value.numerator, "denominator": value.denominator}


@dataclass(frozen=True)
class LoopPlan:
    length: int
    pad_clips: int = 1
    views: int = 2

    def __post_init__(self):
        _integer(self.length, "length", 22)
        _integer(self.pad_clips, "pad_clips", 1)
        _integer(self.views, "views", 1)
        if (self.length - 5) % 17:
            raise ValueError("periodic H3 requires an actual 17k+5 frame input")
        if self.views not in (1, 2):
            raise ValueError("the prototype implements one or two views")

    @property
    def clips(self):
        return (self.length - 5) // 17

    @property
    def frames(self):
        return self.length - 5

    @property
    def video_ticks(self):
        return self.clips * 5

    @property
    def audio_ticks(self):
        return round(Fraction(self.frames * 40, 24))

    @property
    def input_audio_ticks(self):
        return round(Fraction(self.length * 40, 24))

    @property
    def video_pad(self):
        return self.pad_clips * 5

    @property
    def audio_pad(self):
        return round(Fraction(self.pad_clips * 17 * 40, 24))

    @property
    def shifts(self):
        return (0, self.clips // 2) if self.views == 2 and self.clips > 1 else (0,)

    def view(self, shift):
        _integer(shift, "shift")
        if shift >= self.clips:
            raise ValueError("view rotation must be inside one loop")
        exact_audio = Fraction(shift * 17 * 40, 24)
        return {
            "shift_clips": shift, "shift_video_ticks": shift * 5,
            "shift_frames": shift * 17, "shift_audio_ticks": round(exact_audio),
            "audio_rotation_error_ticks": rational(round(exact_audio) - exact_audio),
            "is_exact_half_turn": 2 * shift == self.clips,
        }

    def report(self):
        exact = Fraction(self.frames * 40, 24)
        return {
            "input_frames": self.length, "loop_frames": self.frames,
            "video_ticks": self.video_ticks, "audio_ticks": self.audio_ticks,
            "input_audio_ticks": self.input_audio_ticks,
            "pad_video_ticks": self.video_pad, "pad_audio_ticks": self.audio_pad,
            "audio_padding_error_ticks": rational(
                self.audio_pad - Fraction(self.pad_clips * 17 * 40, 24)),
            "duration": rational(Fraction(self.frames, 24)),
            "audio_period_error_ticks": rational(self.audio_ticks - exact),
            "latent_periods_exact": self.audio_ticks == exact,
            "views": [self.view(shift) for shift in self.shifts],
            "video_view": 0, "second_view_usage": "audio_only",
        }


def periodic(value, before, after, dim, *, shift=0):
    """Index a periodic tensor without changing its owner or global RNG."""
    _integer(before, "before")
    _integer(after, "after")
    if not isinstance(value, torch.Tensor) or value.shape[dim] < 1:
        raise ValueError("periodic input must have a nonempty time dimension")
    size = value.shape[dim]
    indices = (torch.arange(-before, size + after, device=value.device) - shift) % size
    return value.index_select(dim, indices)


def pcm_plan(plan, rate, padded_ticks, actual_samples):
    """Frame-clock crop, explicitly distinct from exact latent periodicity.

No decoder stride is inferred by integer division. Non-integral sample rates
use rational boundaries. Cropping into wrapped context does not time-stretch
the latent audio, and therefore cannot qualify a non-exact tick loop as seamless.
"""
    _integer(rate, "sample_rate", 1)
    expected = round(Fraction(padded_ticks * rate, 40))
    if actual_samples != expected:
        raise ValueError(f"audio decoder returned {actual_samples} PCM samples; "
                         f"{padded_ticks} ticks at 40Hz/{rate}Hz require {expected}")
    start = round(Fraction(plan.audio_pad * rate, 40))
    count = round(Fraction(plan.frames * rate, 24))
    if start + count > actual_samples:
        raise ValueError("decoded wrapped audio does not cover the frame-clock crop")
    return {
        "start_sample": start, "sample_count": count, "end_sample": start + count,
        "sample_rate": rate, "decoded_samples": actual_samples,
        "latent_period_samples": rational(Fraction(plan.audio_ticks * rate, 40)),
        "frame_period_samples": rational(Fraction(plan.frames * rate, 24)),
        "pcm_rounding_error_samples": rational(count - Fraction(plan.frames * rate, 24)),
        "clock_policy": "explicit_frame_clock_crop_no_resampling",
        "latent_seam_crossed": count > Fraction(plan.audio_ticks * rate, 40),
        "phase_continuity_qualified": False,
    }


def repeat_pcm(waveform, sample_rate, frames, cycles=3):
    """Repeat PCM with cumulative rational boundaries, reporting 1-sample edits.

This is a diagnostic repeated-play timeline, not a hidden loop-quality repair.
It keeps left/right samples together and does not resample or crossfade.
"""
    _integer(sample_rate, "sample_rate", 1)
    _integer(frames, "frames", 1)
    _integer(cycles, "cycles", 1)
    period = Fraction(frames * sample_rate, 24)
    if (not isinstance(waveform, torch.Tensor) or waveform.ndim != 3
            or waveform.shape[:2] != (1, 2) or waveform.shape[-1] != round(period)
            or not bool(torch.isfinite(waveform).all())):
        raise ValueError("repeat PCM requires finite [1,2,N] at the declared frame-clock length")
    boundaries = [round(i * period) for i in range(cycles + 1)]
    sizes = [b - a for a, b in zip(boundaries, boundaries[1:])]
    parts = [waveform.index_select(-1, torch.arange(size, device=waveform.device)
                                  % waveform.shape[-1]) for size in sizes]
    return torch.cat(parts, dim=-1), {
        "cycles": cycles, "frames": frames * cycles, "sample_boundaries": boundaries,
        "cycle_sample_adjustments": [size - waveform.shape[-1] for size in sizes],
        "total_rounding_error_samples": rational(boundaries[-1] - cycles * period),
        "video_time_base": rational(Fraction(1, 24)),
        "audio_time_base": rational(Fraction(1, sample_rate)),
        "resampled": False, "crossfaded": False, "quality_accepted": False,
    }
