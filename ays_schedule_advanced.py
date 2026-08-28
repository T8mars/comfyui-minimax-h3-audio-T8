from __future__ import annotations

import json
import math
from collections.abc import Sequence

import torch

from .sampling import (
    DEFAULT_SAMPLER_NAME,
    shift_sigma,
    setup_dual_clock_sampling,
)


AYS_CONTRACT_VERSION = "minimax_h3_ays_schedule_contract_t8_v1"
SCHEDULE_PROFILES = ("native_flow_baseline", "manual_h3_calibrated")


def _parse_manual_base_sigmas(value: str | Sequence[float], steps: int) -> torch.Tensor:
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("manual_base_sigmas must be a JSON array") from exc
    else:
        decoded = value
    if not isinstance(decoded, Sequence) or isinstance(decoded, (str, bytes)):
        raise ValueError("manual_base_sigmas must be a JSON array")
    try:
        sigmas = torch.tensor([float(item) for item in decoded], dtype=torch.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("manual_base_sigmas must contain only finite numbers") from exc
    if sigmas.numel() != int(steps) + 1:
        raise ValueError(
            "manual_base_sigmas must contain steps + 1 entries; "
            f"expected {int(steps) + 1}, got {sigmas.numel()}"
        )
    if not bool(torch.isfinite(sigmas).all()):
        raise ValueError("manual_base_sigmas must contain only finite numbers")
    if not math.isclose(float(sigmas[0]), 1.0, rel_tol=0.0, abs_tol=1.0e-7):
        raise ValueError("manual_base_sigmas must start at exactly 1.0")
    if not math.isclose(float(sigmas[-1]), 0.0, rel_tol=0.0, abs_tol=1.0e-7):
        raise ValueError("manual_base_sigmas must end at exactly 0.0")
    if bool(((sigmas[:-1] - sigmas[1:]) <= 0.0).any()):
        raise ValueError("manual_base_sigmas must be strictly descending")
    if bool(((sigmas < 0.0) | (sigmas > 1.0)).any()):
        raise ValueError("manual_base_sigmas must stay within [0, 1]")
    return sigmas


def build_dual_clock_schedule_contract(
    model,
    av_latent: dict,
    steps: int,
    shift_video: float,
    shift_audio: float,
    schedule_profile: str,
    manual_base_sigmas: str,
    schedule_label: str,
    sampler_name: str = DEFAULT_SAMPLER_NAME,
):
    if schedule_profile not in SCHEDULE_PROFILES:
        raise ValueError(f"unsupported schedule_profile: {schedule_profile!r}")
    if not math.isfinite(float(shift_video)) or float(shift_video) <= 0.0:
        raise ValueError("shift_video must be finite and greater than zero")
    if not math.isfinite(float(shift_audio)) or float(shift_audio) <= 0.0:
        raise ValueError("shift_audio must be finite and greater than zero")

    if schedule_profile == "native_flow_baseline":
        base_sigmas = torch.linspace(1.0, 0.0, int(steps) + 1, dtype=torch.float32)
        source = "MiniMax H3 native uniform flow baseline"
        ays_validated = False
    else:
        base_sigmas = _parse_manual_base_sigmas(manual_base_sigmas, int(steps))
        source = str(schedule_label).strip() or "user supplied H3 calibration"
        # A user label is provenance, not proof that the paper's KLUB optimization ran.
        ays_validated = False

    patched_model, sampler, _native_sigmas = setup_dual_clock_sampling(
        model,
        av_latent,
        int(steps),
        float(shift_video),
        float(shift_audio),
        sampler_name,
        "native_flow",
    )
    video_sigmas = shift_sigma(base_sigmas, float(shift_video)).to(torch.float32)
    audio_sigmas = shift_sigma(base_sigmas, float(shift_audio)).to(torch.float32)
    report = {
        "contract_version": AYS_CONTRACT_VERSION,
        "status": "experimental_schedule_contract",
        "schedule_profile": schedule_profile,
        "schedule_label": source,
        "sampler_name": sampler_name,
        "steps": int(steps),
        "shift_video": float(shift_video),
        "shift_audio": float(shift_audio),
        "base_sigmas": [float(value) for value in base_sigmas],
        "video_sigmas": [float(value) for value in video_sigmas],
        "audio_sigmas": [float(value) for value in audio_sigmas],
        "audio_and_video_share_base_knots": True,
        "audio_conditioning_changed": False,
        "ays_klub_optimized_for_minimax_h3": ays_validated,
        "scientific_boundary": (
            "Align Your Steps schedules are model, dataset and solver specific. "
            "This node never relabels SD/SDXL/SVD schedules as MiniMax H3 schedules; "
            "manual_h3_calibrated only imports externally calibrated base-flow knots."
        ),
    }
    return patched_model, sampler, video_sigmas, json.dumps(
        report, ensure_ascii=False, indent=2
    )

