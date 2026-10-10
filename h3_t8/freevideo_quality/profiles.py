"""Exact CPU clocks for the four author plans, not arbitrary step interpolation."""
from __future__ import annotations

FREEVIDEO_REVISION = "e7eb66326a038344ba241fa31355c15b258b98cb"
AUDIO_FREEVIDEO_REVISION = "566251c88707322e4674ba1c6f587834e07a91b6"
LEGACY_RUNTIME = "t8-freevideo-runtime-v2"
AUDIO_RUNTIME = "t8-freevideo-runtime-audio-v3"


def binding(schema=LEGACY_RUNTIME):
    """Explicit producer family; defaults remain the original immutable v2."""
    if schema == LEGACY_RUNTIME:
        return dict(runtime=schema, revision=FREEVIDEO_REVISION, reference_audio_t=0.,
                    request="t8-freevideo-quality-request-v2", stage="t8-freevideo-quality-stage-v2",
                    saved="t8-freevideo-quality-saved-v2")
    if schema == AUDIO_RUNTIME:
        return dict(runtime=schema, revision=AUDIO_FREEVIDEO_REVISION, reference_audio_t=1.,
                    request="t8-freevideo-quality-request-audio-v3", stage="t8-freevideo-quality-stage-audio-v3",
                    saved="t8-freevideo-quality-saved-audio-v3")
    raise ValueError("Unknown FreeVideo runtime producer family")


def receipt_binding(schema):
    for runtime in (LEGACY_RUNTIME, AUDIO_RUNTIME):
        value = binding(runtime)
        if schema == value["stage"]:
            return value
    raise ValueError("Unknown FreeVideo stage producer family")


COMMUNITY = "community-sigma3-v1"
PROFILES = {"light": 8, "medium": 12, "high": 16, "max": 20}
LABELS = {
    "Light / 轻量8+3（先LOW8，再独立HIGH3）": "light",
    "Medium / 标准单采12": "medium",
    "High / 精细单采16": "high",
    "Max / 极致单采20": "max",
}
TASKS = ("t2va", "i2va", "l2va", "fl2va", "ref2va", "ref2va_audio", "ref2va_av")


def raw_grid(n):
    import torch
    raw = torch.linspace(1., 0., n + 1, dtype=torch.float32, device="cpu")
    if n == 20:
        # The pinned published bank was produced with raw[9]=FP32(.55).
        # Some CPU torch.linspace builds yield its immediately lower neighbor.
        # Exact table identity requires an explicit, documented new-only clock.
        raw[9] = torch.tensor(.55, dtype=torch.float32)
    return raw


def profile(value):
    value = LABELS.get(value, value)
    if value not in PROFILES:
        raise ValueError("Choose an exact FreeVideo quality profile: light/medium/high/max")
    return value


def plan(value, role=None):
    value = profile(value)
    expected = "LOW" if value == "light" else "SINGLE"
    role = expected if role is None else role
    if role != expected and not (value == "light" and role == "HIGH"):
        raise ValueError("Community HIGH3 requires Light completed LOW8; single-pass and MID are not valid inputs")
    base = PROFILES[value]
    return dict(profile=value, role=role, base_steps=base, nfe=3 if role == "HIGH" else base,
                schedule=COMMUNITY if role == "HIGH" else "original-full-grid",
                audio_policy="completed_LOW_preserved_with_audio_clock_conditioning" if role == "HIGH" else "joint_AV_updates",
                stage_complete=True, plan_complete=role != "LOW", plan_total_nfe=11 if value == "light" else base,
                pending_HIGH_nfe=3 if role == "LOW" else 0,
                clock_contract="published_20_FP32_raw_index9_v1" if base == 20 else "author_native_grid_v1")


def clock(value, role=None, task="t2va", *, reference_audio_t=0.):
    import torch
    selected = plan(value, role)
    if task not in TASKS:
        raise ValueError("Unsupported actual FreeVideo task")
    if type(reference_audio_t) not in (int, float) or reference_audio_t not in (0., 1.):
        raise ValueError("Reference audio clock must belong to an explicit producer family")
    if selected["role"] == "HIGH":
        # Exact author independent schedule. Leading sigma=1 is initialization only.
        video = torch.tensor((1., .9035, .6316, .3158, 0.), dtype=torch.float32, device="cpu")
        raw = video / (12. + (1. - 12.) * video)
        audio = 3. * raw / (1. + 2. * raw)
        video, audio = video[1:], audio[1:]
    else:
        raw = raw_grid(selected["nfe"])
        video = 12. * raw / (1. + 11. * raw)
        audio = 3. * raw / (1. + 2. * raw)
    video_t, audio_t = 1. - video[:-1], 1. - audio[:-1]
    rows = []
    visual = task in ("i2va", "l2va", "fl2va", "ref2va", "ref2va_av")
    reference_audio = task in ("ref2va_audio", "ref2va_av")
    for vt, at in zip(video_t, audio_t):
        times = [vt, at]
        if visual:
            times.append(torch.maximum(vt, torch.tensor(.999, dtype=torch.float32)))
        if reference_audio:
            times.append(torch.tensor(reference_audio_t, dtype=torch.float32))
        rows.append(torch.unique(torch.stack(times), sorted=True).tolist())
    return dict(video_sigmas=video.tolist(), audio_sigmas=audio.tolist(),
                video_timesteps=video_t.tolist(), audio_timesteps=audio_t.tolist(), modulation_timesteps=rows,
                convention="H3_t_equals_1_minus_sigma", leading_initialization_not_NFE=selected["role"] == "HIGH")


def table_identity(weights, value, role, task, *, reference_audio_t=0.):
    return dict(format="freevideo-adaln-v2", contract="minimax-h3-adaln-silu-linear-3x6-v1", weights=weights,
                timesteps=clock(value, role, task, reference_audio_t=reference_audio_t)["modulation_timesteps"], channels=5376, dtype="BF16",
                modality_rows=3, components=6)


def validate_clock(observed, value, role, task, *, reference_audio_t=0.):
    if observed != clock(value, role, task, reference_audio_t=reference_audio_t):
        raise ValueError("FreeVideo quality clock differs from the exact author profile/task")
