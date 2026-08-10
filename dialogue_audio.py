from __future__ import annotations

import json
import math
from typing import Mapping

import torch
import torch.nn.functional as torch_functional
import torchaudio

from .core import validate_audio


DIALOGUE_SAFE_MASTER_SCHEMA = "minimax_h3_t8_dialogue_safe_master_v1"
BED_FIT_POLICIES = ("strict", "pad_or_trim", "loop_crossfade")
SFX_FIT_POLICIES = ("strict", "pad_or_trim")


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def _to_stereo(waveform: torch.Tensor) -> torch.Tensor:
    if waveform.shape[1] == 2:
        return waveform
    if waveform.shape[1] == 1:
        return waveform.expand(-1, 2, -1)
    return waveform.mean(dim=1, keepdim=True).expand(-1, 2, -1)


def _prepare_audio(audio: Mapping, name: str, output_sample_rate: int) -> torch.Tensor:
    waveform, sample_rate = validate_audio(audio, name)
    output = waveform.detach().to(device="cpu", dtype=torch.float32)
    if sample_rate != output_sample_rate:
        output = torchaudio.functional.resample(output, sample_rate, output_sample_rate)
    return _to_stereo(output).contiguous()


def _fade_curve(length: int, fade_in: bool) -> torch.Tensor:
    if length <= 0:
        return torch.empty(0, dtype=torch.float32)
    phase = torch.linspace(0.0, math.pi / 2.0, length, dtype=torch.float32)
    curve = phase.sin().square()
    return curve if fade_in else curve.flip(0)


def _loop_crossfade(
    waveform: torch.Tensor,
    target_samples: int,
    crossfade_samples: int,
) -> tuple[torch.Tensor, int, int]:
    source_samples = waveform.shape[-1]
    if source_samples >= target_samples:
        return waveform[..., :target_samples].clone(), 1, 0
    overlap = min(crossfade_samples, source_samples // 4)
    if overlap <= 0:
        raise ValueError("loop_crossfade requires a connected stem long enough for a crossfade")
    output = waveform.clone()
    repeats = 1
    while output.shape[-1] < target_samples:
        actual_overlap = min(overlap, output.shape[-1], waveform.shape[-1])
        cross = (
            output[..., -actual_overlap:] * _fade_curve(actual_overlap, False)
            + waveform[..., :actual_overlap] * _fade_curve(actual_overlap, True)
        )
        output = torch.cat(
            (output[..., :-actual_overlap], cross, waveform[..., actual_overlap:]),
            dim=-1,
        )
        repeats += 1
    return output[..., :target_samples].contiguous(), repeats, overlap


def _fit_bed(
    audio: Mapping | None,
    name: str,
    policy: str,
    output_sample_rate: int,
    target_samples: int,
    loop_crossfade_seconds: float,
) -> tuple[torch.Tensor, dict]:
    if audio is None:
        return torch.zeros((1, 2, target_samples), dtype=torch.float32), {
            "connected": False,
            "policy": policy,
            "action": "missing_stem_is_silence",
            "input_samples": 0,
            "output_samples": target_samples,
        }
    waveform = _prepare_audio(audio, name, output_sample_rate)
    source_samples = waveform.shape[-1]
    if source_samples == target_samples:
        return waveform, {
            "connected": True,
            "policy": policy,
            "action": "exact",
            "input_samples": source_samples,
            "output_samples": target_samples,
        }
    if policy == "strict":
        raise ValueError(
            f"{name} has {source_samples} samples after resampling but the target master requires "
            f"{target_samples}; choose an explicit fit policy"
        )
    if policy == "pad_or_trim":
        if source_samples < target_samples:
            output = torch_functional.pad(waveform, (0, target_samples - source_samples))
            action = f"padded_{target_samples - source_samples}_samples_with_silence"
        else:
            output = waveform[..., :target_samples].contiguous()
            action = f"trimmed_{source_samples - target_samples}_tail_samples"
        return output, {
            "connected": True,
            "policy": policy,
            "action": action,
            "input_samples": source_samples,
            "output_samples": target_samples,
        }
    if policy == "loop_crossfade":
        output, repeats, overlap = _loop_crossfade(
            waveform,
            target_samples,
            round(loop_crossfade_seconds * output_sample_rate),
        )
        return output, {
            "connected": True,
            "policy": policy,
            "action": "looped_with_crossfade",
            "input_samples": source_samples,
            "output_samples": target_samples,
            "repeat_count": repeats,
            "crossfade_samples": overlap,
            "crossfade_seconds": overlap / output_sample_rate,
        }
    raise ValueError(f"unknown {name} fit policy: {policy}")


def _dbfs(value: float) -> float:
    return 20.0 * math.log10(max(float(value), 1e-12))


def build_dialogue_safe_master(
    speech_audio: Mapping,
    speech_accepted: bool,
    target_duration_seconds: float,
    speech_start_seconds: float = 0.0,
    output_sample_rate: int = 32000,
    music_fit_policy: str = "strict",
    ambience_fit_policy: str = "strict",
    sfx_fit_policy: str = "strict",
    loop_crossfade_seconds: float = 0.25,
    speech_gain_db: float = 0.0,
    music_gain_db: float = 0.0,
    ambience_gain_db: float = 0.0,
    sfx_gain_db: float = 0.0,
    duck_background: float = 0.0,
    peak_limit_dbfs: float = -1.0,
    music_audio: Mapping | None = None,
    ambience_audio: Mapping | None = None,
    sfx_audio: Mapping | None = None,
):
    if not bool(speech_accepted):
        raise ValueError(
            "Dialogue Safe Master requires a true accepted signal from exact speech verification; "
            "it will not guess whether a speech stem is clean"
        )
    if output_sample_rate not in {32000, 44100, 48000}:
        raise ValueError("output_sample_rate must be 32000, 44100, or 48000")
    if not math.isfinite(float(target_duration_seconds)) or target_duration_seconds <= 0.0:
        raise ValueError("target_duration_seconds must be positive")
    if not math.isfinite(float(speech_start_seconds)) or speech_start_seconds < 0.0:
        raise ValueError("speech_start_seconds must be nonnegative")
    if music_fit_policy not in BED_FIT_POLICIES:
        raise ValueError("unknown music_fit_policy")
    if ambience_fit_policy not in BED_FIT_POLICIES:
        raise ValueError("unknown ambience_fit_policy")
    if sfx_fit_policy not in SFX_FIT_POLICIES:
        raise ValueError("sfx_fit_policy must be strict or pad_or_trim; sound effects are never looped")
    if not 0.0 <= float(loop_crossfade_seconds) <= 5.0:
        raise ValueError("loop_crossfade_seconds must be between 0 and 5")
    if not 0.0 <= float(duck_background) <= 1.0:
        raise ValueError("duck_background must be between 0 and 1")
    if not -24.0 <= float(peak_limit_dbfs) <= 0.0:
        raise ValueError("peak_limit_dbfs must be between -24 and 0")
    gains = {
        "speech": float(speech_gain_db),
        "music": float(music_gain_db),
        "ambience": float(ambience_gain_db),
        "sfx": float(sfx_gain_db),
    }
    if any(not -60.0 <= gain <= 24.0 for gain in gains.values()):
        raise ValueError("stem gains must be between -60 and +24 dB")

    target_samples = round(float(target_duration_seconds) * output_sample_rate)
    if target_samples <= 0:
        raise ValueError("target duration produced no audio samples")
    speech = _prepare_audio(speech_audio, "speech_audio", output_sample_rate)
    speech_start = round(float(speech_start_seconds) * output_sample_rate)
    speech_end = speech_start + speech.shape[-1]
    if speech_end > target_samples:
        raise ValueError(
            f"verified speech ends at sample {speech_end}, beyond target master sample "
            f"{target_samples}; regenerate, shorten the line, or move it earlier"
        )
    speech_stem = torch.zeros((1, 2, target_samples), dtype=torch.float32)
    speech_stem[..., speech_start:speech_end] = speech
    speech_stem *= 10.0 ** (gains["speech"] / 20.0)

    music, music_report = _fit_bed(
        music_audio,
        "music_audio",
        music_fit_policy,
        output_sample_rate,
        target_samples,
        loop_crossfade_seconds,
    )
    ambience, ambience_report = _fit_bed(
        ambience_audio,
        "ambience_audio",
        ambience_fit_policy,
        output_sample_rate,
        target_samples,
        loop_crossfade_seconds,
    )
    sfx, sfx_report = _fit_bed(
        sfx_audio,
        "sfx_audio",
        sfx_fit_policy,
        output_sample_rate,
        target_samples,
        loop_crossfade_seconds,
    )
    music *= 10.0 ** (gains["music"] / 20.0)
    ambience *= 10.0 ** (gains["ambience"] / 20.0)
    sfx *= 10.0 ** (gains["sfx"] / 20.0)
    background = music + ambience + sfx

    if duck_background > 0.0:
        envelope = speech_stem.abs().mean(dim=1, keepdim=True)
        window = max(1, round(output_sample_rate * 0.02))
        envelope = torch_functional.avg_pool1d(
            envelope,
            window,
            stride=1,
            padding=window // 2,
        )[..., :target_samples]
        activity = (envelope / 0.05).clamp(0.0, 1.0)
        background *= 1.0 - float(duck_background) * activity

    master = speech_stem + background
    peak_before = float(master.abs().amax())
    limit = 10.0 ** (float(peak_limit_dbfs) / 20.0)
    limiter_gain = min(1.0, limit / max(peak_before, 1e-12))
    speech_stem *= limiter_gain
    background *= limiter_gain
    master = speech_stem + background
    reconstruction_error = float((master - (speech_stem + background)).abs().amax())
    report = {
        "schema": DIALOGUE_SAFE_MASTER_SCHEMA,
        "status": "sample_exact_stem_master",
        "sample_rate": output_sample_rate,
        "target_duration_seconds": float(target_duration_seconds),
        "target_samples": target_samples,
        "output_samples": int(master.shape[-1]),
        "exact_sample_error": int(master.shape[-1]) - target_samples,
        "speech": {
            "accepted_input_required": True,
            "start_sample": speech_start,
            "end_sample": speech_end,
            "start_seconds": speech_start / output_sample_rate,
            "end_seconds": speech_end / output_sample_rate,
            "input_samples_after_resample": int(speech.shape[-1]),
            "tail_samples_after_speech": target_samples - speech_end,
            "action": "placed_without_trimming_or_time_stretch",
        },
        "beds": {
            "music": music_report,
            "ambience": ambience_report,
            "sfx": sfx_report,
        },
        "gains_db": gains,
        "duck_background": float(duck_background),
        "peak_limit": {
            "limit_dbfs": float(peak_limit_dbfs),
            "peak_before_dbfs": _dbfs(peak_before),
            "limiter_gain": limiter_gain,
            "attenuation_db": _dbfs(limiter_gain),
        },
        "stem_reconstruction_max_abs_error": reconstruction_error,
        "claims": {
            "master_tail_was_truncated_at_dialogue_end": False,
            "background_was_source_separated_from_a_mix": False,
            "speech_text_was_inferred_by_this_node": False,
            "sample_exact_duration": True,
            "lip_sync_verified": False,
        },
        "limitations": [
            "Connected music, ambience, and SFX must already be independent stems.",
            "This node does not remove speech from an existing mixed master.",
            "An upstream accepted boolean is an assertion from ASR/QA, not proof supplied by this mixer.",
        ],
    }
    return (
        {"waveform": master.contiguous(), "sample_rate": output_sample_rate},
        {"waveform": speech_stem.contiguous(), "sample_rate": output_sample_rate},
        {"waveform": background.contiguous(), "sample_rate": output_sample_rate},
        _json(report),
    )
