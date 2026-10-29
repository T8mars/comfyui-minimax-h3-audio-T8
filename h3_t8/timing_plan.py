"""Pure timing contract shared by AudioWindow and the Director compiler."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math

from .frame_grid import FPS, MIN_TRAINED_FRAMES, align_frame_count


@dataclass(frozen=True)
class TimingPlan:
    frame_count: int
    render_duration_seconds: float
    source_slice_start_seconds: float
    source_slice_duration_seconds: float
    left_pad_seconds: float
    right_pad_seconds: float
    final_trim_start_seconds: float
    final_duration_seconds: float
    main_prompt_start_seconds: float
    main_prompt_end_seconds: float
    # Retain the caller's original float rather than reconstructing a half-
    # sample boundary through subtractive context/padding arithmetic. Optional
    # for callers that still construct a plan from the historical report keys.
    scene_start_seconds: float | None = None

    def report(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    def prompt_note(self) -> str:
        return (
            f"Timing: context starts at 0.00s; main requested action starts at "
            f"{self.main_prompt_start_seconds:.2f}s and ends at {self.main_prompt_end_seconds:.2f}s."
        )


def make_timing_plan(
    scene_start_seconds: float,
    scene_duration_seconds: float,
    warmup_seconds: float = 0.0,
    cooldown_seconds: float = 0.0,
    ensure_minimum_context: bool = True,
    source_duration_seconds: float = 0.0,
) -> TimingPlan:
    if scene_start_seconds < 0:
        raise ValueError("scene_start_seconds cannot be negative")
    if scene_duration_seconds <= 0:
        raise ValueError("scene_duration_seconds must be positive")
    if warmup_seconds < 0 or cooldown_seconds < 0:
        raise ValueError("warmup_seconds and cooldown_seconds cannot be negative")

    required = warmup_seconds + scene_duration_seconds + cooldown_seconds
    if ensure_minimum_context:
        required = max(required, MIN_TRAINED_FRAMES / FPS)
    extra = max(
        0.0, required - (warmup_seconds + scene_duration_seconds + cooldown_seconds)
    )
    ideal_start = scene_start_seconds - warmup_seconds - extra / 2.0
    ideal_end = (
        scene_start_seconds + scene_duration_seconds + cooldown_seconds + extra / 2.0
    )

    frame_count = align_frame_count(math.ceil((ideal_end - ideal_start) * FPS - 1e-9))
    render_duration = frame_count / FPS
    # Grid alignment extends the tail, never shifts the user's requested scene.
    ideal_end = ideal_start + render_duration
    slice_start = max(0.0, ideal_start)
    left_pad = max(0.0, -ideal_start)
    available_end = (
        source_duration_seconds if source_duration_seconds > 0 else ideal_end
    )
    slice_end = min(ideal_end, available_end)
    slice_duration = max(0.0, slice_end - slice_start)
    right_pad = max(0.0, render_duration - left_pad - slice_duration)
    trim_start = scene_start_seconds - ideal_start

    return TimingPlan(
        frame_count=frame_count,
        render_duration_seconds=render_duration,
        source_slice_start_seconds=slice_start,
        source_slice_duration_seconds=slice_duration,
        left_pad_seconds=left_pad,
        right_pad_seconds=right_pad,
        final_trim_start_seconds=trim_start,
        final_duration_seconds=scene_duration_seconds,
        main_prompt_start_seconds=trim_start,
        main_prompt_end_seconds=trim_start + scene_duration_seconds,
        scene_start_seconds=scene_start_seconds,
    )
