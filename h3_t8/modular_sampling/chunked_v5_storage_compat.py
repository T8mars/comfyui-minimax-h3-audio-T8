"""Exact reviewed historical formats for explicitly selected v5 windows.

This is not automatic cache admission or MODEL/conditioning certification.
All five sampling/effect sources must still match the current reader, and
every original source/noise/plan/content/file/lease check remains mandatory.
"""
from __future__ import annotations

import hashlib

from .results import _input_identity, canonical


STORAGE_SOURCE = "chunked_v5_storage.py"
COMPAT_SOURCE = "chunked_v5_storage_compat.py"
LEGACY_SOURCES = frozenset({
    STORAGE_SOURCE, "chunked_two_pass_parity.py", "chunked_two_pass_upscale_advanced.py",
    "chunked_v5.py", "chunked_v5_effects.py", "chunked_v5_relay.py",
})
LEGACY_FULL_REPORT_SHA = "a60d85616918df6691ed7523f30cf642c9763cf8d602324de0eef3b600b7d614"
HISTORICAL_STABLE_SHA = "518a7d43910f799c0a545e7f07d8cce574045fe6816d6891eafbfde85bb1048d"
FULL_REPORT_PROFILE = "historical_full_report_v1"
STABLE_PROFILE = "historical_stable_telemetry_v1"
CURRENT_PROFILE = "current_stable_telemetry_v1"


def implementation_profile(implementation, current):
    """Recognize only exact source inventories; never ignore changed owners."""
    if (type(implementation) is not dict or type(current) is not dict or
            set(current) != LEGACY_SOURCES | {COMPAT_SOURCE} or
            any(type(value) is not str or len(value) != 64 or
                any(char not in "0123456789abcdef" for char in value)
                for value in (*implementation.values(), *current.values()))):
        raise ValueError("Unknown, stale or incomplete v5 frozen manifest implementation")
    if implementation == current:
        return CURRENT_PROFILE
    profiles = {LEGACY_FULL_REPORT_SHA: FULL_REPORT_PROFILE,
                HISTORICAL_STABLE_SHA: STABLE_PROFILE}
    profile = profiles.get(implementation.get(STORAGE_SOURCE))
    if (set(implementation) != LEGACY_SOURCES or profile is None or
            any(implementation[key] != current[key]
                for key in LEGACY_SOURCES - {STORAGE_SOURCE})):
        raise ValueError("Unknown, stale or incomplete v5 frozen manifest implementation")
    return profile


def prepared_sha(prepared, profile, stable_sha):
    if profile in {CURRENT_PROFILE, STABLE_PROFILE}:
        return stable_sha
    if profile != FULL_REPORT_PROFILE:
        raise ValueError("Unknown v5 frozen preparation identity profile")
    # The historical full report is intentional: old caches do NOT gain the
    # later telemetry relaxation, nor can their missing report be reconstructed.
    return hashlib.sha256(canonical({
        "video_noise": _input_identity(prepared.video_noise),
        "audio_noise": _input_identity(prepared.audio_noise),
        "noise_report": _input_identity(prepared.noise_report),
        "noise_seed": prepared.noise_seed,
        "upscale_report": _input_identity(prepared.lift.upscale_report),
    }).encode("utf-8")).hexdigest()
