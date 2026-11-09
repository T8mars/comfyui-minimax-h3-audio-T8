"""Local PCM candidates, without inventing H3 latents or source separation."""
from __future__ import annotations

import hashlib
import json
import math

import torch

from .audio_ops import _absolute_sample_tick, _match_channels, _resample
from .core import validate_audio


def _pcm(audio, name):
    waveform, rate = validate_audio(audio, name)
    if type(audio["sample_rate"]) is not int or rate <= 0:
        raise ValueError(f"{name} requires a positive integer sample rate")
    if not waveform.is_floating_point() or not bool(torch.isfinite(waveform).all()):
        raise ValueError(f"{name} requires finite floating-point PCM")
    return waveform.detach().cpu(), rate


def _rms(waveform):
    return float(waveform.double().square().mean().sqrt())


def _identity(waveform):
    raw = waveform.contiguous().view(torch.uint8).numpy().tobytes()
    return {"sha256": hashlib.sha256(raw).hexdigest(), "shape": list(waveform.shape),
            "dtype": str(waveform.dtype)}


def _candidate_headroom(fixed, contribution):
    """Largest gain in [0,1] fitting PCM full scale, leaving fixed voice intact."""
    fixed, contribution = fixed.double(), contribution.double()
    active = contribution != 0
    if bool((fixed[~active].abs() > 1).any()):
        raise ValueError("Candidate attenuation cannot fit the protected voice within PCM full scale")
    if not bool(active.any()):
        return 1.0
    a, b = (-1-fixed[active])/contribution[active], (1-fixed[active])/contribution[active]
    low = max(0.0, float(torch.minimum(a, b).amax()))
    high = min(1.0, float(torch.maximum(a, b).amin()))
    if low > high:
        raise ValueError("Candidate attenuation cannot fit the protected voice within PCM full scale")
    # Allow room for the final source-dtype addition, without changing voice.
    return max(low, high * (1.0 - 8 * torch.finfo(torch.float32).eps)) if high < 1 else 1.0


def prepare_local_sfx_candidates(
    source_audio, external_sfx_audio, start_seconds, end_seconds, *,
    protected_voice_audio=None, h3_sfx_audio=None, candidate_start_seconds=0.0,
    fade_seconds=0.01, level_policy="match_source_rms", target_rms_dbfs=-24.0,
    selection="original", headroom_policy="none",
):
    """Return selected, external preview, H3 preview, and a transparent report.

    Optional voice is an existing aligned stem, never an inferred separation.
    Candidate inputs are local PCM (or longer PCM with an explicit offset).
    Selection is the user's choice; numeric metrics never auto-select a winner.
    """
    for name, value in (("start_seconds", start_seconds), ("end_seconds", end_seconds),
                        ("candidate_start_seconds", candidate_start_seconds), ("fade_seconds", fade_seconds),
                        ("target_rms_dbfs", target_rms_dbfs)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number")
    if start_seconds < 0 or end_seconds <= start_seconds or candidate_start_seconds < 0 or fade_seconds < 0:
        raise ValueError("SFX requires an ordered nonnegative window, offset and fade")
    if selection not in ("original", "external", "h3"):
        raise ValueError("selection must be original, external or h3")
    if selection == "h3" and h3_sfx_audio is None:
        raise ValueError("H3 candidate is absent; connect actual H3 PCM before selecting it")
    if external_sfx_audio is None:
        raise ValueError("Connect actual external PCM before preparing SFX candidates")
    if level_policy not in ("match_source_rms", "target_rms", "target_rms_ceiling"):
        raise ValueError("level_policy must be match_source_rms, target_rms or target_rms_ceiling")
    if headroom_policy not in ("none", "attenuate_candidate"):
        raise ValueError("headroom_policy must be none or attenuate_candidate")
    if not -120 <= target_rms_dbfs <= 0:
        raise ValueError("target_rms_dbfs must be between -120 and 0 dBFS")

    source, rate = _pcm(source_audio, "source_audio")
    start, end = _absolute_sample_tick(start_seconds, rate), _absolute_sample_tick(end_seconds, rate)
    if not 0 <= start < end <= source.shape[-1]:
        raise ValueError("SFX sample window must lie inside the source PCM; no padding or looping")
    fade = _absolute_sample_tick(fade_seconds, rate)
    if 2 * fade > end-start:
        raise ValueError("The two crossfades must fit inside the selected sample window")
    if protected_voice_audio is None:
        voice = torch.zeros_like(source)
    else:
        voice, voice_rate = _pcm(protected_voice_audio, "protected_voice_audio")
        if voice_rate != rate or voice.shape != source.shape or voice.dtype != source.dtype:
            raise ValueError("Protected voice must share source PCM rate, shape, dtype and timeline; it is never resampled")
    residual = source - voice
    if not bool(torch.isfinite(residual).all()):
        raise ValueError("Source-minus-voice residual overflows the PCM dtype")
    reconstruction_error = float((source - (voice + residual)).abs().amax())
    tolerance = 8 * torch.finfo(source.dtype).eps * max(1., float(source.abs().amax()), float(voice.abs().amax()))
    if reconstruction_error > tolerance:
        raise ValueError("Source and protected voice cannot numerically reconstruct the original PCM")
    window = residual[..., start:end]
    reference_rms = _rms(window)
    target_rms = reference_rms if level_policy == "match_source_rms" else 10 ** (target_rms_dbfs / 20)
    weight = torch.ones(end-start, dtype=source.dtype)
    if fade:
        ramp = torch.linspace(0, math.pi/2, fade, dtype=source.dtype).sin().square()
        weight[:fade], weight[-fade:] = ramp, ramp.flip(0)
    offset = _absolute_sample_tick(candidate_start_seconds, rate)
    rows = {}

    def preview(audio, name):
        if audio is None:
            rows[name] = {"status": "absent", "preview_is_original_bypass": True}
            return source_audio
        candidate, input_rate = _pcm(audio, name)
        identity = _identity(candidate)
        original_channels = candidate.shape[1]
        candidate = _resample(candidate.to(dtype=source.dtype), input_rate, rate)
        candidate = _match_channels(candidate, source.shape[1])
        if offset + end-start > candidate.shape[-1]:
            raise ValueError(f"{name} PCM is too short for the local window and offset; no padding or looping")
        candidate = candidate[..., offset:offset + end-start]
        if not bool(torch.isfinite(candidate).all()):
            raise ValueError(f"{name} resampled PCM is nonfinite")
        candidate_rms = _rms(candidate)
        if target_rms == 0 and candidate_rms > 0:
            raise ValueError("Source residual is silent; choose explicit target_rms to add audible SFX")
        if candidate_rms == 0 and target_rms > 0 and level_policy != "target_rms_ceiling":
            raise ValueError(f"{name} is silent and cannot match a nonzero RMS target")
        requested_gain = target_rms / candidate_rms if candidate_rms else 1.
        gain = min(1., requested_gain) if level_policy == "target_rms_ceiling" else requested_gain
        matched = candidate * gain
        # The fixed component contains the original at transition edges and
        # the untouched voice at full replacement. Only candidate is attenuated.
        fixed = source[..., start:end] * (1-weight) + voice[..., start:end] * weight
        contribution = matched * weight
        headroom_gain = _candidate_headroom(fixed, contribution) if headroom_policy == "attenuate_candidate" else 1.
        changed = fixed + contribution * headroom_gain
        if not bool(torch.isfinite(changed).all()):
            raise ValueError(f"{name} gain or mixing produced nonfinite PCM")
        output = source.clone()
        output[..., start:end] = changed
        peak = float(output.abs().amax())
        rows[name] = {
            "status": "candidate", "input": identity, "input_rate": input_rate,
            "resampled": input_rate != rate, "input_channels": original_channels,
            "output_channels": source.shape[1], "candidate_local_start_sample": offset,
            "candidate_local_end_sample": offset+end-start, "input_window_rms": candidate_rms,
            "matched_window_rms_before_fade": _rms(matched), "match_gain": gain,
            "requested_match_gain": requested_gain,
            "upward_gain_prevented": level_policy == "target_rms_ceiling" and requested_gain > 1.,
            "headroom_gain": headroom_gain, "candidate_rms_after_headroom": _rms(matched * headroom_gain),
            "output_peak": peak, "exceeds_pcm_full_scale": peak > 1,
            "output": _identity(output), "outside_window_bit_exact": True,
        }
        return {"waveform": output, "sample_rate": rate}

    external, h3 = preview(external_sfx_audio, "external"), preview(h3_sfx_audio, "h3")
    selected = {"original": source_audio, "external": external, "h3": h3}[selection]
    report = {
        "schema": "t8.local_sfx_pcm.v1", "selection": selection, "sample_rate": rate,
        "start_sample": start, "end_sample": end, "crossfade_samples": fade,
        "level_policy": level_policy, "level_measurement": "unweighted_window_rms_not_LUFS",
        "level_behavior": "attenuate_only_ceiling" if level_policy == "target_rms_ceiling" else "normalize_to_target",
        "source_residual_rms": reference_rms, "target_rms": target_rms,
        "headroom_policy": headroom_policy, "source": _identity(source),
        "protected_voice": _identity(voice) if protected_voice_audio is not None else None,
        "stem_reconstruction_max_abs_error": reconstruction_error,
        "voice_stem_gain": 1.0 if protected_voice_audio is not None else None,
        "candidates": rows,
        "claims": {"source_separation_performed": False, "voice_separation_quality_verified": False,
                   "candidate_speech_absence_verified": False, "perceptual_loudness_equal_verified": False,
                   "human_quality_reviewed": False, "video_processed": False, "av_latent_created": False},
        "limitations": ["Provided voice is an aligned stem assertion; source-minus-voice may still contain speech.",
                        "RMS matching does not establish equal perceived loudness or semantic synchronization.",
                        "A candidate can contain new speech; listen before choosing it.",
                        "With no voice stem, replacement can remove speech inside the selected window.",
                        "Connect the unchanged source IMAGE to the existing saver; this node handles PCM only."],
    }
    return selected, external, h3, json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
