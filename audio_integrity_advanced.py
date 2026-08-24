from __future__ import annotations

from collections.abc import Mapping
import json
import math
import re

import torch

from .core import validate_audio
from .speech import audio_quality_facts, validate_speech_plan, validate_voice_profile


AUDIO_INTEGRITY_SCHEMA = "minimax_h3_t8_audio_integrity_audit_v1"
PERCEPTUAL_DRIFT_SCHEMA = "minimax_h3_t8_audio_perceptual_drift_audit_v1"
SPEAKER_ROUTING_SCHEMA = "minimax_h3_t8_speaker_routing_audit_v1"


def _json(value: Mapping) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def _dbfs(value: float) -> float:
    return 20.0 * math.log10(max(float(value), 1e-12))


def _window_rms_dbfs(value: torch.Tensor) -> float:
    if value.numel() == 0:
        return -240.0
    return _dbfs(float(value.square().mean().sqrt()))


def analyze_audio_integrity(
    audio: Mapping,
    video_frame_count: int = 0,
    fps: float = 24.0,
    opening_window_ms: float = 40.0,
    comparison_window_ms: float = 250.0,
    pop_jump_threshold: float = 0.15,
    dc_jump_threshold: float = 0.02,
    wrap_correlation_threshold: float = 0.985,
    clipping_ratio_threshold: float = 0.001,
    max_av_delta_ms: float = 21.0,
) -> tuple[Mapping, bool, str, int, float, float, str]:
    """Report conservative boundary anomalies without changing the input audio."""

    waveform, sample_rate = validate_audio(audio, "audio")
    raw = waveform.detach().to(device="cpu", dtype=torch.float32)
    canonical = torch.nan_to_num(raw)
    sample_count = int(canonical.shape[-1])
    duration_seconds = sample_count / int(sample_rate)
    if fps <= 0.0:
        raise ValueError("fps must be greater than zero")
    if video_frame_count < 0:
        raise ValueError("video_frame_count cannot be negative")

    findings: list[dict] = []
    checks: dict[str, dict] = {}
    nonfinite = bool(not torch.isfinite(raw).all())
    if nonfinite:
        findings.append(
            {
                "code": "nonfinite_samples",
                "severity": "abstain",
                "message": "Audio contains NaN or Inf samples.",
            }
        )

    opening_samples = max(2, round(opening_window_ms * sample_rate / 1000.0))
    comparison_samples = max(8, round(comparison_window_ms * sample_rate / 1000.0))
    derivative = (canonical[..., 1:] - canonical[..., :-1]).abs()
    opening_count = min(opening_samples, int(derivative.shape[-1]))
    opening_derivative = derivative[..., :opening_count]
    baseline_start = min(opening_count, int(derivative.shape[-1]))
    baseline_end = min(int(derivative.shape[-1]), baseline_start + max(comparison_samples, opening_count))
    baseline_derivative = derivative[..., baseline_start:baseline_end]
    opening_max_jump = float(opening_derivative.amax()) if opening_derivative.numel() else 0.0
    baseline_p995 = (
        float(torch.quantile(baseline_derivative.flatten(), 0.995))
        if baseline_derivative.numel()
        else 0.0
    )
    opening_jump_ratio = opening_max_jump / max(baseline_p995, 1e-8)
    first_sample_abs = float(canonical[..., 0].abs().amax()) if sample_count else 0.0
    opening_flag = bool(
        (opening_max_jump >= pop_jump_threshold and opening_jump_ratio >= 4.0)
        or first_sample_abs >= max(0.25, pop_jump_threshold * 2.0)
    )
    checks["opening_transient"] = {
        "evaluated": bool(opening_derivative.numel() and baseline_derivative.numel()),
        "window_ms": float(opening_window_ms),
        "maximum_adjacent_sample_jump": opening_max_jump,
        "later_p995_adjacent_sample_jump": baseline_p995,
        "opening_to_later_jump_ratio": opening_jump_ratio,
        "first_sample_absolute_amplitude": first_sample_abs,
        "threshold": float(pop_jump_threshold),
        "suspected": opening_flag,
    }
    if opening_flag:
        findings.append(
            {
                "code": "suspected_opening_pop_or_cut",
                "severity": "abstain",
                "message": (
                    "The opening contains an unusually large discontinuity; this is a signal "
                    "heuristic, not proof that the model produced a pop."
                ),
            }
        )

    block_samples = max(1, round(sample_rate * 0.010))
    block_count = sample_count // block_samples
    dc_context_blocks = max(3, round(0.100 * sample_rate / block_samples))
    if block_count >= dc_context_blocks * 2:
        blocks = canonical[..., : block_count * block_samples].reshape(
            *canonical.shape[:-1], block_count, block_samples
        )
        block_dc = blocks.mean(dim=-1)
        context_means = block_dc.unfold(-1, dc_context_blocks, 1).mean(dim=-1)
        comparison_count = block_count - 2 * dc_context_blocks + 1
        before = context_means[..., :comparison_count]
        after = context_means[
            ..., dc_context_blocks : dc_context_blocks + comparison_count
        ]
        dc_steps = (after - before).abs()
        max_dc_jump = float(dc_steps.amax())
        max_dc_flat = int(dc_steps.reshape(-1).argmax())
        time_axis_index = max_dc_flat % int(dc_steps.shape[-1])
        max_dc_time = (
            time_axis_index + dc_context_blocks
        ) * block_samples / sample_rate
        dc_flag = max_dc_jump >= dc_jump_threshold
    else:
        max_dc_jump = 0.0
        max_dc_time = 0.0
        dc_flag = False
    checks["dc_discontinuity"] = {
        "evaluated": block_count >= dc_context_blocks * 2,
        "block_ms": 10.0,
        "context_ms_per_side": dc_context_blocks * block_samples * 1000.0 / sample_rate,
        "maximum_persistent_context_mean_jump": max_dc_jump,
        "maximum_jump_time_seconds": max_dc_time,
        "threshold": float(dc_jump_threshold),
        "suspected": dc_flag,
    }
    if dc_flag:
        findings.append(
            {
                "code": "suspected_dc_jump",
                "severity": "abstain",
                "message": (
                    "The persistent mean before and after a candidate boundary exceeds the "
                    "configured DC-jump threshold."
                ),
            }
        )

    compare_count = min(comparison_samples, sample_count // 2)
    wrap_evaluated = compare_count >= max(32, round(sample_rate * 0.025))
    correlation = 0.0
    relative_rmse = float("inf")
    head_rms_dbfs = -240.0
    tail_rms_dbfs = -240.0
    if wrap_evaluated:
        head = canonical[..., :compare_count].flatten().float()
        tail = canonical[..., -compare_count:].flatten().float()
        head_rms_dbfs = _window_rms_dbfs(head)
        tail_rms_dbfs = _window_rms_dbfs(tail)
        head_centered = head - head.mean()
        tail_centered = tail - tail.mean()
        denominator = float(head_centered.norm() * tail_centered.norm())
        if denominator > 1e-10:
            correlation = float(torch.dot(head_centered, tail_centered) / denominator)
        reference_rms = max(float(head.square().mean().sqrt()), 1e-8)
        relative_rmse = float((head - tail).square().mean().sqrt()) / reference_rms
    wrap_flag = bool(
        wrap_evaluated
        and min(head_rms_dbfs, tail_rms_dbfs) > -50.0
        and correlation >= wrap_correlation_threshold
        and relative_rmse <= 0.25
    )
    checks["tail_to_head_similarity"] = {
        "evaluated": wrap_evaluated,
        "window_ms": compare_count * 1000.0 / sample_rate if sample_rate else 0.0,
        "normalized_correlation": correlation,
        "relative_rmse": relative_rmse if math.isfinite(relative_rmse) else None,
        "head_rms_dbfs": head_rms_dbfs,
        "tail_rms_dbfs": tail_rms_dbfs,
        "correlation_threshold": float(wrap_correlation_threshold),
        "suspected": wrap_flag,
    }
    if wrap_flag:
        findings.append(
            {
                "code": "suspected_tail_wrapped_to_head",
                "severity": "abstain",
                "message": (
                    "The ending is unusually similar to the opening. Periodic music can also "
                    "trigger this heuristic, so human listening is required."
                ),
            }
        )

    quality = audio_quality_facts(audio)
    clipping_flag = quality["clipping_sample_ratio"] > clipping_ratio_threshold
    checks["clipping"] = {
        "sample_ratio": quality["clipping_sample_ratio"],
        "threshold": float(clipping_ratio_threshold),
        "suspected": clipping_flag,
    }
    if clipping_flag:
        findings.append(
            {
                "code": "clipping_ratio_exceeded",
                "severity": "abstain",
                "message": "The clipped-sample ratio exceeds the configured threshold.",
            }
        )

    av_delta_ms = 0.0
    if video_frame_count > 0:
        expected_samples = round(video_frame_count * sample_rate / fps)
        delta_samples = sample_count - expected_samples
        av_delta_ms = delta_samples * 1000.0 / sample_rate
        av_flag = abs(av_delta_ms) > max_av_delta_ms
        checks["audio_video_boundary"] = {
            "evaluated": True,
            "video_frame_count": int(video_frame_count),
            "fps": float(fps),
            "expected_audio_samples": int(expected_samples),
            "actual_audio_samples": sample_count,
            "delta_samples": int(delta_samples),
            "delta_ms": av_delta_ms,
            "maximum_absolute_delta_ms": float(max_av_delta_ms),
            "suspected": av_flag,
        }
        if av_flag:
            findings.append(
                {
                    "code": "audio_video_boundary_mismatch",
                    "severity": "abstain",
                    "message": "Audio duration does not match the declared video boundary.",
                }
            )
    else:
        checks["audio_video_boundary"] = {
            "evaluated": False,
            "reason": "video_frame_count_is_zero",
            "actual_audio_samples": sample_count,
        }

    decision = "ABSTAIN" if findings else "PASS"
    report = {
        "schema": AUDIO_INTEGRITY_SCHEMA,
        "decision": decision,
        "report_only": True,
        "audio_mutated": False,
        "facts": quality,
        "checks": checks,
        "findings": findings,
        "limitations": [
            "Signal heuristics cannot identify model-level speaker leakage or prove causality.",
            "A tail/head match may be an intentional loop or periodic music; listen before rejecting.",
            "PASS means no configured heuristic fired, not perceptual audio certification.",
        ],
    }
    return audio, not findings, decision, sample_count, duration_seconds, av_delta_ms, _json(report)


def _mono_cpu(value: torch.Tensor) -> torch.Tensor:
    canonical = torch.nan_to_num(value.detach().to(device="cpu", dtype=torch.float32))
    return canonical.reshape(-1, canonical.shape[-1]).mean(dim=0)


def _windowed_acoustic_envelope(
    mono: torch.Tensor,
    sample_rate: int,
    window_samples: int,
    hop_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    windows = mono.unfold(0, window_samples, hop_samples)
    taper = torch.hann_window(window_samples, periodic=False, dtype=torch.float32)
    n_fft = 1 << (window_samples - 1).bit_length()
    frequencies = torch.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    upper_hz = min(8000.0, sample_rate * 0.475)
    edges = torch.logspace(math.log10(80.0), math.log10(upper_hz), 25)
    band_masks = []
    for index in range(24):
        selected = (frequencies >= edges[index]) & (frequencies < edges[index + 1])
        if not bool(selected.any()):
            raise ValueError("sample rate/window combination leaves an empty analysis band")
        band_masks.append(selected)
    envelope_chunks = []
    rms_chunks = []
    # Bound FFT workspace for long dialogue/audio. The input waveform is already resident in
    # ComfyUI, but no chunk creates a full clip x FFT-bin matrix.
    for start in range(0, int(windows.shape[0]), 128):
        chunk = windows[start : start + 128]
        rms_chunks.append(
            20.0
            * torch.log10(chunk.square().mean(dim=-1).sqrt().clamp_min(1e-12))
        )
        power = (
            torch.fft.rfft(chunk * taper, n=n_fft, dim=-1)
            .abs()
            .square()
            .clamp_min(1e-12)
        )
        bands = [
            torch.log10(power[:, selected].mean(dim=-1).clamp_min(1e-12))
            for selected in band_masks
        ]
        envelope = torch.stack(bands, dim=-1)
        envelope_chunks.append(envelope - envelope.mean(dim=-1, keepdim=True))
    return torch.cat(envelope_chunks, dim=0), torch.cat(rms_chunks, dim=0)


def _persistent_sections(
    suspected: torch.Tensor,
    spectral_drift: torch.Tensor,
    level_delta_db: torch.Tensor,
    hop_samples: int,
    window_samples: int,
    sample_rate: int,
    minimum_windows: int,
) -> list[dict]:
    sections = []
    start = None
    values = [bool(item) for item in suspected.tolist()]
    for index in range(len(values) + 1):
        active = values[index] if index < len(values) else False
        if active and start is None:
            start = index
        if not active and start is not None:
            end = index - 1
            if end - start + 1 >= minimum_windows:
                section_spectral = spectral_drift[start : end + 1]
                section_level = level_delta_db[start : end + 1]
                sections.append(
                    {
                        "start_seconds": start * hop_samples / sample_rate,
                        "end_seconds": (end * hop_samples + window_samples) / sample_rate,
                        "window_count": end - start + 1,
                        "peak_spectral_drift": float(section_spectral.max()),
                        "peak_absolute_level_delta_db": float(section_level.abs().max()),
                        "median_level_delta_db": float(section_level.median()),
                    }
                )
            start = None
    return sections


def analyze_audio_perceptual_drift(
    reference_audio: Mapping,
    candidate_audio: Mapping,
    analysis_window_ms: float = 500.0,
    hop_ms: float = 100.0,
    active_rms_floor_dbfs: float = -50.0,
    spectral_drift_threshold: float = 0.30,
    level_delta_threshold_db: float = 4.0,
    persistent_window_count: int = 3,
    max_duration_delta_ms: float = 21.0,
) -> tuple[Mapping, bool, str, float, float, float, float, str]:
    """Compare aligned audio to a reference without claiming perceptual diagnosis."""

    reference_waveform, reference_rate = validate_audio(reference_audio, "reference_audio")
    candidate_waveform, candidate_rate = validate_audio(candidate_audio, "candidate_audio")
    if analysis_window_ms <= 0.0 or hop_ms <= 0.0:
        raise ValueError("analysis_window_ms and hop_ms must be positive")
    if spectral_drift_threshold <= 0.0 or level_delta_threshold_db <= 0.0:
        raise ValueError("drift thresholds must be positive")
    if persistent_window_count < 1:
        raise ValueError("persistent_window_count must be at least one")
    if max_duration_delta_ms < 0.0:
        raise ValueError("max_duration_delta_ms cannot be negative")

    findings: list[dict] = []
    warnings: list[dict] = []
    reference_samples = int(reference_waveform.shape[-1])
    candidate_samples = int(candidate_waveform.shape[-1])
    duration_delta_ms = (
        candidate_samples * 1000.0 / candidate_rate
        - reference_samples * 1000.0 / reference_rate
    )
    if reference_rate != candidate_rate:
        findings.append(
            {
                "code": "sample_rate_mismatch",
                "reference_sample_rate": int(reference_rate),
                "candidate_sample_rate": int(candidate_rate),
                "message": "Use synchronized audio with the same sample rate; the node will not resample silently.",
            }
        )
    if abs(duration_delta_ms) > max_duration_delta_ms:
        findings.append(
            {
                "code": "duration_mismatch",
                "duration_delta_ms": duration_delta_ms,
                "maximum_absolute_delta_ms": float(max_duration_delta_ms),
            }
        )
    if reference_waveform.shape[:-1] != candidate_waveform.shape[:-1]:
        warnings.append(
            {
                "code": "channel_or_batch_shape_mismatch",
                "reference_shape": list(reference_waveform.shape),
                "candidate_shape": list(candidate_waveform.shape),
                "message": "Both inputs are folded to mono for analysis; the candidate output is unchanged.",
            }
        )
    if not torch.isfinite(reference_waveform).all() or not torch.isfinite(candidate_waveform).all():
        findings.append(
            {
                "code": "nonfinite_samples",
                "message": "At least one input contains NaN or Inf samples.",
            }
        )

    checks: dict[str, object] = {}
    maximum_spectral_drift = 0.0
    maximum_level_delta_db = 0.0
    first_suspected_start = -1.0
    last_suspected_end = -1.0
    correlation = None
    sections: list[dict] = []
    if reference_rate == candidate_rate:
        sample_rate = int(reference_rate)
        common_samples = min(reference_samples, candidate_samples)
        window_samples = max(32, round(analysis_window_ms * sample_rate / 1000.0))
        hop_samples = max(1, round(hop_ms * sample_rate / 1000.0))
        if sample_rate < 4000 or common_samples < window_samples:
            findings.append(
                {
                    "code": "insufficient_audio_for_acoustic_drift",
                    "sample_rate": sample_rate,
                    "common_samples": common_samples,
                    "required_window_samples": window_samples,
                }
            )
        else:
            reference_mono = _mono_cpu(reference_waveform)[..., :common_samples]
            candidate_mono = _mono_cpu(candidate_waveform)[..., :common_samples]
            reference_envelope, reference_rms = _windowed_acoustic_envelope(
                reference_mono, sample_rate, window_samples, hop_samples
            )
            candidate_envelope, candidate_rms = _windowed_acoustic_envelope(
                candidate_mono, sample_rate, window_samples, hop_samples
            )
            spectral_drift = (
                (candidate_envelope - reference_envelope).square().mean(dim=-1).sqrt()
            )
            level_delta_db = candidate_rms - reference_rms
            active = reference_rms >= active_rms_floor_dbfs
            active_count = int(active.sum())
            if active_count < persistent_window_count:
                findings.append(
                    {
                        "code": "insufficient_active_reference_audio",
                        "active_window_count": active_count,
                        "required_window_count": int(persistent_window_count),
                    }
                )
            if active_count:
                active_spectral = spectral_drift[active]
                active_level = level_delta_db[active]
                maximum_spectral_drift = float(active_spectral.max())
                maximum_level_delta_db = float(active_level.abs().max())
                spectral_p90 = float(torch.quantile(active_spectral, 0.90))
                level_p90 = float(torch.quantile(active_level.abs(), 0.90))
            else:
                spectral_p90 = 0.0
                level_p90 = 0.0
            suspected = active & (
                (
                    (spectral_drift >= spectral_drift_threshold)
                    & (level_delta_db.abs() >= level_delta_threshold_db)
                )
                | (spectral_drift >= spectral_drift_threshold * 1.75)
                | (level_delta_db.abs() >= level_delta_threshold_db * 2.0)
            )
            sections = _persistent_sections(
                suspected,
                spectral_drift,
                level_delta_db,
                hop_samples,
                window_samples,
                sample_rate,
                persistent_window_count,
            )
            if sections:
                first_suspected_start = float(sections[0]["start_seconds"])
                last_suspected_end = float(sections[-1]["end_seconds"])
                findings.append(
                    {
                        "code": "reference_relative_acoustic_drift",
                        "sections": sections,
                        "message": (
                            "Persistent reference-relative spectral/level drift was detected. "
                            "This is a review cue, not proof of distance, reverb or speaker change."
                        ),
                    }
                )
            correlation_stride = max(1, math.ceil(common_samples / 1_000_000))
            reference_correlation = reference_mono[::correlation_stride]
            candidate_correlation = candidate_mono[::correlation_stride]
            reference_centered = reference_correlation - reference_correlation.mean()
            candidate_centered = candidate_correlation - candidate_correlation.mean()
            denominator = float(reference_centered.norm() * candidate_centered.norm())
            correlation = (
                float(torch.dot(reference_centered, candidate_centered) / denominator)
                if denominator > 1e-10
                else None
            )
            if correlation is not None:
                correlation = max(-1.0, min(1.0, correlation))
            checks["reference_relative_acoustic_drift"] = {
                "evaluated": True,
                "analysis_window_ms": float(analysis_window_ms),
                "hop_ms": float(hop_ms),
                "active_rms_floor_dbfs": float(active_rms_floor_dbfs),
                "spectral_drift_definition": (
                    "RMS difference between gain-normalized 24-band log10 power envelopes "
                    "from 80Hz to min(8kHz, 0.475*sample_rate)"
                ),
                "spectral_drift_threshold": float(spectral_drift_threshold),
                "level_delta_threshold_db": float(level_delta_threshold_db),
                "persistent_window_count": int(persistent_window_count),
                "active_window_count": active_count,
                "suspected_window_count": int(suspected.sum()),
                "spectral_drift_p90": spectral_p90,
                "absolute_level_delta_db_p90": level_p90,
                "maximum_spectral_drift": maximum_spectral_drift,
                "maximum_absolute_level_delta_db": maximum_level_delta_db,
                "waveform_correlation": correlation,
                "waveform_correlation_sample_stride": correlation_stride,
                "analysis_chunk_windows": 128,
                "sections": sections,
            }
    if "reference_relative_acoustic_drift" not in checks:
        checks["reference_relative_acoustic_drift"] = {
            "evaluated": False,
            "reason": "incompatible_or_insufficient_inputs",
        }

    decision = "ABSTAIN" if findings else "PASS"
    report = {
        "schema": PERCEPTUAL_DRIFT_SCHEMA,
        "decision": decision,
        "report_only": True,
        "candidate_audio_mutated": False,
        "reference": {
            "sample_rate": int(reference_rate),
            "sample_count": reference_samples,
            "shape": list(reference_waveform.shape),
        },
        "candidate": {
            "sample_rate": int(candidate_rate),
            "sample_count": candidate_samples,
            "shape": list(candidate_waveform.shape),
        },
        "duration_delta_ms": duration_delta_ms,
        "checks": checks,
        "findings": findings,
        "warnings": warnings,
        "limitations": [
            "The inputs must be time-aligned versions of the same intended content.",
            "The metric is an acoustic proxy; it cannot prove perceived distance, reverb, identity or model causality.",
            "Speech timing, performance or sound-design changes may legitimately trigger ABSTAIN.",
            "PASS means no configured persistent reference-relative drift fired, not perceptual equivalence.",
        ],
    }
    return (
        candidate_audio,
        not findings,
        decision,
        maximum_spectral_drift,
        maximum_level_delta_db,
        first_suspected_start,
        last_suspected_end,
        _json(report),
    )


_VOCALIZATION_RE = re.compile(
    r"(?:\[|\(|（)\s*(?:laugh|laughter|giggle|gasp|sigh|breath|pant|chuckle|"
    r"笑|笑声|大笑|轻笑|喘息|喘气|叹气|吸气|呼气|抽泣)\s*(?:\]|\)|）)|"
    r"(?:哈哈哈+|呵呵呵+|嘿嘿嘿+)",
    flags=re.IGNORECASE,
)
_FEMALE_MARKERS = ("female", "woman", "girl", "女声", "女性", "女人", "女孩", "少女")
_MALE_MARKERS = ("male", "man", "boy", "男声", "男性", "男人", "男孩", "少年")


def _descriptor_gender_marker(value: str) -> str | None:
    lowered = str(value or "").casefold()
    female = any(marker in lowered for marker in _FEMALE_MARKERS)
    male = any(marker in lowered for marker in _MALE_MARKERS)
    if female == male:
        return None
    return "female_descriptor" if female else "male_descriptor"


def _descriptor_tokens(value: str) -> set[str]:
    normalized = re.sub(r"[^\w\u3400-\u9fff]+", " ", str(value or "").casefold()).strip()
    words = {item for item in normalized.split() if len(item) >= 2}
    cjk = "".join(re.findall(r"[\u3400-\u9fff]", normalized))
    words.update(cjk[index : index + 2] for index in range(max(0, len(cjk) - 1)))
    return words


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def audit_speaker_routing(
    dialogue_plan: Mapping,
    require_reference_voice: bool = True,
    descriptor_similarity_threshold: float = 0.75,
) -> tuple[Mapping, bool, str, str, str]:
    """Compile speaker/audio ordinals and abstain on deterministic routing ambiguity."""

    plan = validate_speech_plan(dialogue_plan)
    if plan.get("kind") != "dialogue":
        raise ValueError("Speaker Routing Audit requires a dialogue plan")
    profiles = {
        str(key): validate_voice_profile(value)
        for key, value in dict(plan.get("profiles") or {}).items()
    }
    turns = list(plan.get("segments") or [])
    findings: list[dict] = []
    warnings: list[dict] = []
    seen_order: list[str] = []
    for index, turn in enumerate(turns):
        speaker_id = str(turn.get("speaker_id") or "")
        if speaker_id not in profiles:
            findings.append(
                {
                    "code": "turn_has_no_bound_profile",
                    "turn_index": index,
                    "speaker_id": speaker_id,
                }
            )
            continue
        if speaker_id not in seen_order:
            seen_order.append(speaker_id)
        text = str(turn.get("text") or "")
        matches = [match.group(0) for match in _VOCALIZATION_RE.finditer(text)]
        if matches:
            findings.append(
                {
                    "code": "unstructured_vocalization_in_spoken_text",
                    "turn_index": index,
                    "speaker_id": speaker_id,
                    "matches": matches,
                    "message": (
                        "Move laughter, gasps or breathing into the turn direction/performance "
                        "field unless those literal sounds are intended as spoken text."
                    ),
                }
            )

    unused_profiles = sorted(set(profiles) - set(seen_order))
    if unused_profiles:
        warnings.append(
            {
                "code": "unused_voice_profiles",
                "speaker_ids": unused_profiles,
            }
        )

    bindings = []
    fingerprint_owners: dict[str, list[str]] = {}
    for ordinal, speaker_id in enumerate(seen_order, 1):
        profile = profiles[speaker_id]
        mode = str(profile.get("mode") or "")
        fingerprint = str(profile.get("reference_sha256") or "")
        if fingerprint:
            fingerprint_owners.setdefault(fingerprint, []).append(speaker_id)
        if require_reference_voice and mode != "reference_voice":
            findings.append(
                {
                    "code": "speaker_has_no_reference_voice",
                    "speaker_id": speaker_id,
                    "mode": mode,
                }
            )
        bindings.append(
            {
                "speaker_ordinal": ordinal,
                "speaker_id": speaker_id,
                "audio_tag": f"<Audio {ordinal}>",
                "profile_mode": mode,
                "reference_sha256": fingerprint or None,
                "voice_description": profile.get("voice_description"),
                "language": profile.get("language"),
            }
        )

    for fingerprint, owners in fingerprint_owners.items():
        if len(owners) > 1:
            findings.append(
                {
                    "code": "duplicate_reference_audio_assignment",
                    "speaker_ids": owners,
                    "reference_sha256": fingerprint,
                    "message": "Multiple speaker IDs resolve to the same reference waveform.",
                }
            )

    for left_index, left_id in enumerate(seen_order):
        for right_id in seen_order[left_index + 1 :]:
            left = profiles[left_id]
            right = profiles[right_id]
            marker = _descriptor_gender_marker(left.get("voice_description", ""))
            if not marker or marker != _descriptor_gender_marker(right.get("voice_description", "")):
                continue
            similarity = _jaccard(
                _descriptor_tokens(left.get("voice_description", "")),
                _descriptor_tokens(right.get("voice_description", "")),
            )
            unique_references = bool(
                left.get("reference_sha256")
                and right.get("reference_sha256")
                and left.get("reference_sha256") != right.get("reference_sha256")
            )
            if similarity >= descriptor_similarity_threshold and not unique_references:
                findings.append(
                    {
                        "code": "same_gender_descriptor_assignment_ambiguity",
                        "speaker_ids": [left_id, right_id],
                        "descriptor_group": marker,
                        "descriptor_jaccard": similarity,
                        "message": (
                            "The profiles use highly similar same-gender wording without two "
                            "distinct reference fingerprints. Add explicit, differentiating voice "
                            "traits or unique reference audio."
                        ),
                    }
                )

    binding_payload = {
        "schema": SPEAKER_ROUTING_SCHEMA,
        "bindings": bindings,
        "turn_route": [
            {
                "turn_index": index,
                "speaker_id": turn.get("speaker_id"),
                "audio_tag": next(
                    (
                        binding["audio_tag"]
                        for binding in bindings
                        if binding["speaker_id"] == turn.get("speaker_id")
                    ),
                    None,
                ),
            }
            for index, turn in enumerate(turns)
        ],
    }
    decision = "ABSTAIN" if findings else "PASS"
    report = {
        **binding_payload,
        "decision": decision,
        "report_only": True,
        "plan_mutated": False,
        "require_reference_voice": bool(require_reference_voice),
        "descriptor_similarity_threshold": float(descriptor_similarity_threshold),
        "findings": findings,
        "warnings": warnings,
        "limitations": [
            "Descriptor matching is a conservative text heuristic, not gender classification.",
            "Unique hashes prove different waveform bytes, not speaker identity or clone fidelity.",
            "PASS does not prove that the generated H3 dialogue will avoid voice swapping.",
        ],
    }
    return dialogue_plan, not findings, decision, _json(binding_payload), _json(report)
