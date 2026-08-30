from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any

import torch

from .detail_sampling_advanced import build_tail_detail_schedule


LONG_VIDEO_SAMPLING_PLAN_TYPE = "H3_T8_LONG_VIDEO_SAMPLING_PLAN"
PLAN_MODES = ("disabled", "tail_subdivide", "manual_second_pass")
TAIL_SPACING = ("video_sigma_linear", "video_sigma_cosine", "base_flow_linear")


@dataclass(frozen=True)
class LongVideoSamplingPlan:
    mode: str
    extra_tail_steps: int
    tail_spacing: str
    manual_sigmas: tuple[float, ...]

    def contract(self) -> dict[str, Any]:
        return {
            "schema": "t8.minimax_h3.long_video_sampling_plan.v1",
            "mode": self.mode,
            "extra_tail_steps": self.extra_tail_steps,
            "tail_spacing": self.tail_spacing,
            "manual_sigmas": list(self.manual_sigmas),
            "audio_policy": "joint_av_dual_clock",
            "preview_cache_policy": "one_callback_state_per_segment_across_passes",
        }


def _parse_manual_sigmas(value: str) -> tuple[float, ...]:
    normalized = str(value).replace("→", ",").replace(";", ",")
    try:
        values = tuple(float(token.strip()) for token in normalized.split(",") if token.strip())
    except ValueError as error:
        raise ValueError(f"invalid long-video manual sigma schedule: {value!r}") from error
    if len(values) < 2:
        raise ValueError("manual second-pass schedule needs at least two sigma values")
    if not all(math.isfinite(item) and 0.0 <= item <= 1.0 for item in values):
        raise ValueError("manual second-pass sigmas must be finite and inside [0, 1]")
    if values[0] <= 0.0 or any(left <= right for left, right in zip(values, values[1:])):
        raise ValueError("manual second-pass sigmas must be strictly descending from above zero")
    if not math.isclose(values[-1], 0.0, rel_tol=0.0, abs_tol=1.0e-8):
        raise ValueError("manual second-pass sigmas must end at exactly zero")
    return (*values[:-1], 0.0)


def build_long_video_sampling_plan(
    mode: str,
    extra_tail_steps: int,
    tail_spacing: str,
    manual_sigmas: str,
) -> tuple[LongVideoSamplingPlan, str]:
    if mode not in PLAN_MODES:
        raise ValueError(f"unknown long-video sampling mode: {mode!r}")
    if tail_spacing not in TAIL_SPACING:
        raise ValueError(f"unknown long-video tail spacing: {tail_spacing!r}")
    if not 0 <= int(extra_tail_steps) <= 8:
        raise ValueError("extra_tail_steps must be between 0 and 8")
    parsed = _parse_manual_sigmas(manual_sigmas)
    plan = LongVideoSamplingPlan(
        mode=str(mode),
        extra_tail_steps=int(extra_tail_steps),
        tail_spacing=str(tail_spacing),
        manual_sigmas=parsed,
    )
    report = {
        **plan.contract(),
        "status": "disabled" if mode == "disabled" else "configured_exp",
        "first_pass_extra_nfe": int(extra_tail_steps) if mode == "tail_subdivide" else 0,
        "second_pass_nfe": len(parsed) - 1 if mode == "manual_second_pass" else 0,
        "scientific_boundary": (
            "Both modes change the joint video/audio trajectory and require complete visual "
            "review plus listening. They do not promise more detail or unchanged speech."
        ),
    }
    return plan, json.dumps(report, ensure_ascii=False, indent=2)


def validate_long_video_sampling_plan(value) -> LongVideoSamplingPlan | None:
    if value is None:
        return None
    if not isinstance(value, LongVideoSamplingPlan):
        raise TypeError("long_video_sampling_plan must come from the T8 plan node")
    # Rebuild through the public validator so forged instances cannot bypass checks.
    rebuilt, _report = build_long_video_sampling_plan(
        value.mode,
        value.extra_tail_steps,
        value.tail_spacing,
        ",".join(str(item) for item in value.manual_sigmas),
    )
    return rebuilt


def resolve_long_video_sample_schedules(
    base_sigmas: torch.Tensor,
    plan: LongVideoSamplingPlan | None,
    *,
    shift_video: float,
    shift_audio: float,
) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, Any]]:
    plan = validate_long_video_sampling_plan(plan)
    if plan is None or plan.mode == "disabled":
        return base_sigmas, None, {
            "mode": "disabled",
            "first_pass_nfe": int(base_sigmas.numel() - 1),
            "second_pass_nfe": 0,
        }
    if plan.mode == "tail_subdivide":
        first, nfe, report_json = build_tail_detail_schedule(
            base_sigmas,
            extra_tail_steps=plan.extra_tail_steps,
            spacing=plan.tail_spacing,
            shift_video=float(shift_video),
            shift_audio=float(shift_audio),
            profile="custom_strict",
        )
        return first, None, {
            "mode": plan.mode,
            "first_pass_nfe": int(nfe),
            "second_pass_nfe": 0,
            "tail": json.loads(report_json),
        }
    second = torch.tensor(
        plan.manual_sigmas, dtype=base_sigmas.dtype, device=base_sigmas.device
    )
    return base_sigmas, second, {
        "mode": plan.mode,
        "first_pass_nfe": int(base_sigmas.numel() - 1),
        "second_pass_nfe": int(second.numel() - 1),
        "manual_second_pass_sigmas": list(plan.manual_sigmas),
    }
