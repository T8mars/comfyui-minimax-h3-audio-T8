from __future__ import annotations

from .core import validate_audio
from .timing_plan import TimingPlan, make_timing_plan as make_timing_plan


def window_audio(audio: dict, plan: TimingPlan) -> dict:
    waveform, sample_rate = validate_audio(audio)
    scene_start = plan.scene_start_seconds
    if scene_start is None:
        scene_start = (plan.source_slice_start_seconds + plan.final_trim_start_seconds
                       - plan.left_pad_seconds)
    # OutputTrim rounds its local start. Anchor this window to the same integer
    # sample so separate rounding cannot shift the user's selected source by 1.
    origin = round(scene_start * sample_rate) - round(plan.final_trim_start_seconds * sample_rate)
    start = max(0, origin)
    source_end = round((plan.source_slice_start_seconds + plan.source_slice_duration_seconds) * sample_rate)
    count = max(0, source_end - start)
    sliced = waveform[..., start : start + count]
    left_count = max(0, -origin)
    target_count = round(plan.render_duration_seconds * sample_rate)
    output = waveform.new_zeros((waveform.shape[0], waveform.shape[1], target_count))
    writable = min(sliced.shape[-1], max(0, target_count - left_count))
    if writable:
        output[..., left_count : left_count + writable] = sliced[..., :writable]
    return {"waveform": output, "sample_rate": sample_rate}
