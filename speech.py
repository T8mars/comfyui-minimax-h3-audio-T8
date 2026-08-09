from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
import math
import re
import unicodedata

import torch
import torch.nn.functional as torch_functional
import torchaudio

from comfy_extras.nodes_audio import vae_decode_audio

from .conditioning import build_conditioning
from .core import FPS, align_frame_count, nested_av_parts, sorted_autogrow_values, validate_audio


VOICE_PROFILE_TYPE = "H3_T8_VOICE_PROFILE"
SPEECH_PLAN_TYPE = "H3_T8_SPEECH_PLAN"
VOICE_PROFILE_SCHEMA = "minimax_h3_t8_voice_profile_v1"
SPEECH_PLAN_SCHEMA = "minimax_h3_t8_speech_plan_v1"
SPEECH_CATEGORY = "T8/MiniMax H3/Speech/Experimental"

H3_AUDIO_SAMPLE_RATE = 32000
H3_MIN_REFERENCE_SECONDS = 2.0
H3_MAX_REFERENCE_SECONDS = 15.0
H3_MIN_RENDER_FRAMES = 124
H3_MAX_RENDER_FRAMES = 362

SUPPORTED_DIALOGUE_LANGUAGES = (
    "Arabic",
    "Chinese",
    "English",
    "French",
    "German",
    "Italian",
    "Japanese",
    "Korean",
    "Portuguese",
    "Russian",
    "Spanish",
)

SPACE_DESCRIPTIONS = {
    "studio": "a quiet professional voice booth with dry close-mic acoustics",
    "close": (
        "a real furnished room, physically close to the listener, with subtle natural "
        "early reflections and intimate microphone perspective"
    ),
    "across_table": (
        "a quiet furnished room seated across a small table from the listener, with "
        "believable conversational distance and restrained room reflections"
    ),
    "living_room": (
        "a comfortable furnished living room near the listener, with soft realistic "
        "room tone and short warm reflections"
    ),
    "bedside": (
        "a quiet bedroom beside the listener, very close but naturally voiced, with "
        "soft fabric absorption and tiny spatial reflections"
    ),
    "stage": (
        "a small treated performance stage facing the listener, with controlled "
        "professional room reflections and clear direct sound"
    ),
}

_CONTROL_TAG = re.compile(
    r"</?d(?:\s[^>]*)?>|<(?:scenetrans|cutoff)\s*/?>|"
    r"<(?:Picture|Video|Audio)\s+\d+>",
    flags=re.IGNORECASE,
)
_CJK_LANGUAGE_NAMES = {"chinese", "japanese", "korean"}
_CJK_CHAR = re.compile(r"[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]")


def _json(data) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, default=str)


def safe_speaker_id(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or "")).strip()
    normalized = re.sub(r"\s+", "_", normalized)
    normalized = re.sub(r"[^\w\-\u3400-\u9fff]", "", normalized, flags=re.UNICODE)
    normalized = normalized.strip("_-")
    if not normalized:
        raise ValueError("speaker_id must contain at least one letter, number, or CJK character")
    return normalized[:64]


def _plain_direction(value: str) -> str:
    value = _CONTROL_TAG.sub("", str(value or ""))
    return " ".join(value.split()).strip()


def validate_spoken_text(value: str) -> str:
    value = str(value or "").strip()
    if not value:
        raise ValueError("spoken text cannot be empty")
    if _CONTROL_TAG.search(value):
        raise ValueError(
            "spoken text must be plain text; H3 dialogue/media control tags are added by the node"
        )
    return value


def _to_stereo(waveform: torch.Tensor) -> torch.Tensor:
    channels = int(waveform.shape[1])
    if channels == 2:
        return waveform
    if channels == 1:
        return waveform.expand(-1, 2, -1)
    mono = waveform.mean(dim=1, keepdim=True)
    return mono.expand(-1, 2, -1)


def _dbfs(value: torch.Tensor | float) -> float:
    if isinstance(value, torch.Tensor):
        value = float(value.detach().cpu())
    return 20.0 * math.log10(max(float(value), 1e-12))


def _audio_fingerprint(waveform: torch.Tensor, sample_rate: int) -> str:
    canonical = waveform.detach().to(device="cpu", dtype=torch.float32).contiguous()
    digest = hashlib.sha256()
    digest.update(f"f32le:{sample_rate}:{tuple(canonical.shape)}".encode("ascii"))
    digest.update(canonical.numpy().tobytes(order="C"))
    return digest.hexdigest()


def audio_quality_facts(audio: Mapping) -> dict:
    waveform, sample_rate = validate_audio(audio)
    canonical = torch.nan_to_num(waveform.detach().to(device="cpu", dtype=torch.float32))
    peak = canonical.abs().amax()
    rms = canonical.square().mean().sqrt()
    clipping = (canonical.abs() >= 0.999).float().mean()
    silence = (canonical.abs() <= 10.0 ** (-50.0 / 20.0)).float().mean()
    return {
        "sample_rate": int(sample_rate),
        "channels": int(canonical.shape[1]),
        "samples": int(canonical.shape[-1]),
        "duration_seconds": canonical.shape[-1] / int(sample_rate),
        "peak_dbfs": _dbfs(peak),
        "rms_dbfs": _dbfs(rms),
        "clipping_sample_ratio": float(clipping),
        "below_minus_50_dbfs_sample_ratio": float(silence),
        "contains_nonfinite": bool(not torch.isfinite(waveform).all()),
    }


def prepare_reference_audio(
    audio: Mapping,
    start_seconds: float = 0.0,
    duration_seconds: float = 0.0,
    highpass_60hz: bool = True,
    peak_limit_minus_3_dbfs: bool = True,
) -> tuple[dict, dict]:
    waveform, sample_rate = validate_audio(audio, "reference_audio")
    if start_seconds < 0.0 or duration_seconds < 0.0:
        raise ValueError("reference start/duration cannot be negative")

    source = torch.nan_to_num(waveform.detach().to(device="cpu", dtype=torch.float32))
    source_duration = source.shape[-1] / sample_rate
    if start_seconds >= source_duration:
        raise ValueError("reference start is beyond the connected audio")
    available = source_duration - start_seconds
    requested_or_available = available if duration_seconds == 0.0 else duration_seconds
    selected_seconds = min(H3_MAX_REFERENCE_SECONDS, available, requested_or_available)
    if selected_seconds < H3_MIN_REFERENCE_SECONDS:
        raise ValueError(
            "MiniMax H3 reference audio must provide at least 2.0 seconds after cropping"
        )

    start_sample = round(start_seconds * sample_rate)
    sample_count = round(selected_seconds * sample_rate)
    selected = source[..., start_sample : start_sample + sample_count]
    if selected.shape[-1] < round(H3_MIN_REFERENCE_SECONDS * sample_rate):
        raise ValueError("reference crop contains less than 2.0 seconds of audio")
    if sample_rate != H3_AUDIO_SAMPLE_RATE:
        selected = torchaudio.functional.resample(
            selected, sample_rate, H3_AUDIO_SAMPLE_RATE
        )
    selected = _to_stereo(selected)
    selected = selected - selected.mean(dim=-1, keepdim=True)
    if highpass_60hz:
        selected = torchaudio.functional.highpass_biquad(
            selected, H3_AUDIO_SAMPLE_RATE, 60.0
        )
    if peak_limit_minus_3_dbfs:
        peak = selected.abs().amax().clamp_min(1e-8)
        target_peak = 10.0 ** (-3.0 / 20.0)
        selected = selected * min(1.0, target_peak / float(peak))

    output = {
        "waveform": selected.contiguous(),
        "sample_rate": H3_AUDIO_SAMPLE_RATE,
    }
    facts = audio_quality_facts(output)
    facts.update(
        {
            "source_duration_seconds": source_duration,
            "selected_start_seconds": start_seconds,
            "requested_duration_seconds": duration_seconds,
            "highpass_60hz": bool(highpass_60hz),
            "peak_limit_minus_3_dbfs": bool(peak_limit_minus_3_dbfs),
            "reference_sha256": _audio_fingerprint(
                output["waveform"], H3_AUDIO_SAMPLE_RATE
            ),
        }
    )
    warnings = []
    if facts["rms_dbfs"] < -45.0:
        warnings.append("reference is very quiet after deterministic preparation")
    if facts["below_minus_50_dbfs_sample_ratio"] > 0.8:
        warnings.append("more than 80% of reference samples are near silence")
    if facts["clipping_sample_ratio"] > 0.001:
        warnings.append("reference contains a material clipped-sample ratio")
    facts["warnings"] = warnings
    return output, facts


def make_voice_profile(
    mode: str,
    speaker_id: str,
    voice_description: str,
    language: str,
    rights_confirmed: bool,
    reference_audio: Mapping | None = None,
    reference_start_seconds: float = 0.0,
    reference_duration_seconds: float = 0.0,
    highpass_60hz: bool = True,
    peak_limit_minus_3_dbfs: bool = True,
) -> tuple[dict, dict | None, str]:
    mode = str(mode).lower()
    if mode not in {"described_voice", "reference_voice"}:
        raise ValueError("voice mode must be described_voice or reference_voice")
    speaker_id = safe_speaker_id(speaker_id)
    description = _plain_direction(voice_description)
    if not description:
        raise ValueError("voice_description cannot be empty")
    if language not in SUPPORTED_DIALOGUE_LANGUAGES:
        raise ValueError(f"unsupported H3 dialogue language: {language}")

    prepared = None
    reference_report = None
    if mode == "reference_voice":
        if not rights_confirmed:
            raise ValueError(
                "reference_voice requires explicit confirmation that you have the right "
                "and consent to use this voice"
            )
        if reference_audio is None:
            raise ValueError("reference_voice requires a connected reference_audio")
        prepared, reference_report = prepare_reference_audio(
            reference_audio,
            reference_start_seconds,
            reference_duration_seconds,
            highpass_60hz,
            peak_limit_minus_3_dbfs,
        )

    profile = {
        "schema": VOICE_PROFILE_SCHEMA,
        "mode": mode,
        "speaker_id": speaker_id,
        "voice_description": description,
        "language": language,
        "rights_confirmed": bool(rights_confirmed),
        "reference_audio": prepared,
        "reference_sha256": (
            reference_report["reference_sha256"] if reference_report else None
        ),
        "reference_facts": reference_report,
        "persistence": "workflow_memory_only",
    }
    report = {
        "schema": VOICE_PROFILE_SCHEMA,
        "mode": mode,
        "speaker_id": speaker_id,
        "language": language,
        "rights_confirmed": bool(rights_confirmed),
        "reference": reference_report,
        "limitations": [
            "No speaker-identity model has verified this profile.",
            "Profile creation does not prove generated clone fidelity.",
            "The profile is not persisted by this node.",
        ],
    }
    return profile, prepared, _json(report)


def validate_voice_profile(profile: Mapping) -> dict:
    if not isinstance(profile, Mapping) or profile.get("schema") != VOICE_PROFILE_SCHEMA:
        raise ValueError("expected an H3 T8 voice profile")
    mode = profile.get("mode")
    if mode not in {"described_voice", "reference_voice"}:
        raise ValueError("voice profile has an unknown mode")
    if not profile.get("speaker_id") or not profile.get("voice_description"):
        raise ValueError("voice profile is missing speaker identity metadata")
    if mode == "reference_voice":
        if not profile.get("rights_confirmed"):
            raise ValueError("reference voice profile has no rights confirmation")
        validate_audio(profile.get("reference_audio"), "profile reference_audio")
    return dict(profile)


def _is_cjk_language(language: str, text: str) -> bool:
    return language.casefold() in _CJK_LANGUAGE_NAMES or bool(_CJK_CHAR.search(text))


def _split_cjk_piece(piece: str, max_units: int) -> list[str]:
    compact = re.sub(r"\s+", "", piece)
    return [compact[index : index + max_units] for index in range(0, len(compact), max_units)]


def _split_word_piece(piece: str, max_units: int) -> list[str]:
    words = piece.split()
    return [
        " ".join(words[index : index + max_units])
        for index in range(0, len(words), max_units)
    ]


def split_speech_text(
    text: str,
    language: str,
    target_units: int = 18,
    max_units: int = 24,
) -> list[str]:
    text = validate_spoken_text(text)
    if target_units < 1 or max_units < target_units:
        raise ValueError("chunk target_units must be positive and no greater than max_units")
    cjk = _is_cjk_language(language, text)
    normalized = re.sub(r"\s+", "" if cjk else " ", text).strip()
    sentences = re.split(
        r"(?<=[。！？!?])" if cjk else r"(?<=[.!?])\s+",
        normalized,
    )
    pieces: list[str] = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        size = len(sentence) if cjk else len(sentence.split())
        if size <= max_units:
            pieces.append(sentence)
            continue
        clauses = re.split(
            r"(?<=[，、；：,;:—])" if cjk else r"(?<=[,;:—])\s+",
            sentence,
        )
        for clause in clauses:
            clause = clause.strip()
            if not clause:
                continue
            size = len(clause) if cjk else len(clause.split())
            if size <= max_units:
                pieces.append(clause)
            elif cjk:
                pieces.extend(_split_cjk_piece(clause, max_units))
            else:
                pieces.extend(_split_word_piece(clause, max_units))

    chunks: list[str] = []
    pending = ""
    for piece in pieces:
        joiner = "" if cjk else " "
        candidate = f"{pending}{joiner}{piece}".strip()
        candidate_size = len(candidate) if cjk else len(candidate.split())
        if pending and candidate_size > max_units:
            chunks.append(pending)
            pending = piece
        else:
            pending = candidate
        pending_size = len(pending) if cjk else len(pending.split())
        if pending_size >= target_units and pending[-1:] in ".!?。！？":
            chunks.append(pending)
            pending = ""
    if pending:
        chunks.append(pending)
    return chunks


def make_speech_plan(
    text: str,
    profile: Mapping,
    language: str,
    direction: str,
    emotion: str,
    emotion_intensity: float,
    space: str,
    chunking: str,
    target_units: int,
    max_units: int,
) -> tuple[dict, str]:
    profile = validate_voice_profile(profile)
    if language not in SUPPORTED_DIALOGUE_LANGUAGES:
        raise ValueError(f"unsupported H3 dialogue language: {language}")
    if space not in SPACE_DESCRIPTIONS:
        raise ValueError(f"unknown speech space: {space}")
    if not 0.0 <= emotion_intensity <= 1.0:
        raise ValueError("emotion_intensity must be between 0 and 1")
    if chunking == "single_segment":
        chunks = [validate_spoken_text(text)]
    elif chunking == "language_aware":
        chunks = split_speech_text(text, language, target_units, max_units)
    else:
        raise ValueError("chunking must be single_segment or language_aware")
    clean_direction = _plain_direction(direction)
    clean_emotion = _plain_direction(emotion) or "neutral"
    segments = []
    for index, chunk in enumerate(chunks):
        segments.append(
            {
                "index": index,
                "speaker_id": profile["speaker_id"],
                "text": chunk,
                "language": language,
                "direction": clean_direction,
                "emotion": clean_emotion,
                "emotion_intensity": float(emotion_intensity),
                "space": space,
                "pause_before_seconds": 0.0,
                "overlap_seconds": 0.0,
                "gain_db": 0.0,
                "pan": 0.0,
            }
        )
    plan = {
        "schema": SPEECH_PLAN_SCHEMA,
        "kind": "speech",
        "segments": segments,
        "profiles": {profile["speaker_id"]: profile},
        "chunking": chunking,
        "timing_status": "planned_only_until_audio_is_rendered",
    }
    report = public_plan(plan)
    report["limitations"] = [
        "Segment duration is not guessed by the planner; render_seconds stays explicit.",
        "Emotion intensity is prompt strength, not a calibrated acoustic control.",
        "Actual transcript and speaker identity remain unverified until QA is run.",
    ]
    return plan, _json(report)


def validate_speech_plan(plan: Mapping) -> dict:
    if not isinstance(plan, Mapping) or plan.get("schema") != SPEECH_PLAN_SCHEMA:
        raise ValueError("expected an H3 T8 speech plan")
    segments = plan.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError("speech plan contains no segments")
    for expected_index, segment in enumerate(segments):
        if not isinstance(segment, Mapping):
            raise ValueError("speech plan segment must be an object")
        if int(segment.get("index", -1)) != expected_index:
            raise ValueError("speech plan segment indices must be contiguous from zero")
        validate_spoken_text(segment.get("text", ""))
        if not segment.get("speaker_id"):
            raise ValueError("speech plan segment has no speaker_id")
    return dict(plan)


def public_plan(plan: Mapping) -> dict:
    output = deepcopy(dict(plan))
    profiles = output.get("profiles", {})
    output["profiles"] = {
        speaker_id: {
            key: value
            for key, value in dict(profile).items()
            if key not in {"reference_audio", "reference_facts"}
        }
        for speaker_id, profile in profiles.items()
    }
    return output


def _segment_direction(segment: Mapping, profile: Mapping) -> str:
    parts = [
        _plain_direction(profile.get("voice_description", "")),
        _plain_direction(segment.get("direction", "")),
    ]
    emotion = _plain_direction(segment.get("emotion", ""))
    intensity = float(segment.get("emotion_intensity", 0.5))
    if emotion and emotion.casefold() != "neutral":
        parts.append(
            f"a {emotion} emotional reading with prompt intensity {intensity:.2f}"
        )
    return "; ".join(part.rstrip(".;") for part in parts if part).strip()


def build_speech_prompt(profile: Mapping, segment: Mapping) -> str:
    profile = validate_voice_profile(profile)
    text = validate_spoken_text(segment.get("text", ""))
    language = str(segment.get("language") or profile.get("language") or "English")
    direction = _segment_direction(segment, profile) or "natural conversational delivery"
    location = SPACE_DESCRIPTIONS.get(str(segment.get("space", "close")), SPACE_DESCRIPTIONS["close"])
    if profile["mode"] == "reference_voice":
        return f"""subject_definitions:
<Picture 1> is a blank dark frame and carries no content.
<Audio 1> is the canonical voice identity of Speaker 1 (S1).

summary:
[audio reuse] Treat <Audio 1> as a strict voice-identity reference. Preserve the same adult speaker: timbre, breath, natural pace, microphone distance, accent, and human realism. Do not repeat its words.

retention_analysis:
<Audio 1>: voice_timbre_and_recording_style - preserve this speaker and their natural human micro-pauses for a new line.
<Picture 1>: weak_reference - contributes nothing to the target audio.

detailed_description:
[Shot 1] A still, dark scene. Speaker 1 (S1) is the same adult speaker from <Audio 1>. Voice identity and acting direction only, never spoken aloud: {direction}. The speaker is in {location}. The delivery is emotionally present, connected, and naturally conversational. The only audible words in the entire target are inside the following single dialogue block. Speaker 1 (S1) says exactly and only: <d>[{language}] {text}</d> After the final word, Speaker 1 closes their mouth. No voice description, acting note, instruction, label, metadata, or other prompt text is spoken before or after this dialogue.

overall_soundscape:
Subtle believable non-verbal ambience and microphone perspective from the specified space. No music, effects, other voices, or speech outside the marked dialogue.

non_diegetic_music:
N/A"""

    return f"""integrated_multimodal_description:
[Shot 1] A static featureless dark scene in {location}. A single adult speaker (S1) performs one clean voice take. Voice identity and acting direction only, never spoken aloud: {direction}. The performance is emotionally present, spontaneous, and unmistakably human, with connected phrasing, natural breaths, small timing variations, and no announcer or synthetic TTS cadence. Only the words inside the following dialogue block are audible. Speaker 1 (S1) says exactly and only: <d>[{language}] {text}</d> After the final word, the speaker closes their mouth. No label, voice description, acting note, instruction, or other prompt prose is spoken.

overall_soundscape:
Subtle room tone and believable close-microphone presence from the specified space. No music, effects, other voices, or speech outside the marked dialogue.

non_diegetic_music:
N/A"""


def render_frame_count(render_seconds: float) -> int:
    if not math.isfinite(render_seconds) or render_seconds <= 0.0:
        raise ValueError("render_seconds must be positive")
    frames = align_frame_count(math.ceil(render_seconds * FPS - 1e-9))
    if not H3_MIN_RENDER_FRAMES <= frames <= H3_MAX_RENDER_FRAMES:
        raise ValueError(
            f"speech render window aligns to {frames} frames; keep it within the current "
            f"H3 trained-range baseline {H3_MIN_RENDER_FRAMES}-{H3_MAX_RENDER_FRAMES} frames"
        )
    return frames


def build_speech_conditioning(
    clip,
    video_vae,
    audio_vae,
    profile: Mapping,
    plan: Mapping,
    segment_index: int,
    render_seconds: float,
    resolution: int,
):
    profile = validate_voice_profile(profile)
    plan = validate_speech_plan(plan)
    segments = plan["segments"]
    if not 0 <= int(segment_index) < len(segments):
        raise ValueError(
            f"segment_index {segment_index} is outside plan range 0-{len(segments) - 1}"
        )
    segment = dict(segments[int(segment_index)])
    if segment["speaker_id"] != profile["speaker_id"]:
        raise ValueError(
            "selected speech segment belongs to a different speaker profile; use Dialogue Turn Select"
        )
    if resolution not in {32, 64, 128}:
        raise ValueError("speech canvas must be 32, 64, or 128 pixels")
    frame_count = render_frame_count(render_seconds)
    prompt = build_speech_prompt(profile, segment)

    if profile["mode"] == "reference_voice":
        dark = torch.full((1, 64, 64, 3), 8.0 / 255.0, dtype=torch.float32)
        ref_images = {"ref_image_0": dark}
        ref_audios = {"ref_audio_0": profile["reference_audio"]}
        task_type = "Ref2VA"
        primary_audio = 1
    else:
        ref_images = None
        ref_audios = None
        task_type = "T2VA"
        primary_audio = 0

    conditioning, latent, _, conditioned_prompt, media_map, base_report = build_conditioning(
        clip=clip,
        video_vae=video_vae,
        audio_vae=audio_vae,
        prompt=prompt,
        width=resolution,
        height=resolution,
        length=frame_count,
        task_type=task_type,
        audio_mode="native",
        audio_denoise_strength=1.0,
        add_source_as_reference=False,
        prompt_primary_audio_ordinal=primary_audio,
        strict_prompt_tags=True,
        ref_image_size="match",
        reference_video_policy="official_2_to_15s",
        ref_images=ref_images,
        ref_audios=ref_audios,
    )
    report = {
        "schema": "minimax_h3_t8_speech_conditioning_v1",
        "mode": profile["mode"],
        "speaker_id": profile["speaker_id"],
        "segment_index": int(segment_index),
        "segment_count": len(segments),
        "text": segment["text"],
        "language": segment["language"],
        "requested_render_seconds": float(render_seconds),
        "aligned_frames": frame_count,
        "aligned_render_seconds": frame_count / FPS,
        "canvas": [resolution, resolution],
        "task_type": task_type,
        "media_map_json": media_map,
        "base_conditioning_report": base_report,
        "quality_status": "unverified_generation_input",
        "limitations": [
            "The H3 model still performs a joint AV forward on a tiny video canvas.",
            "Text accuracy, clone identity, acting quality, runtime, and VRAM are not inferred here.",
        ],
    }
    return (
        conditioning,
        latent,
        conditioned_prompt,
        segment["text"],
        _json(public_plan(plan)),
        _json(report),
    )


def _energy_trim(
    audio: Mapping,
    threshold_dbfs: float,
    padding_seconds: float,
) -> tuple[dict, dict]:
    waveform, sample_rate = validate_audio(audio, "generated_audio")
    threshold = 10.0 ** (threshold_dbfs / 20.0)
    mono = waveform.abs().mean(dim=1, keepdim=True)
    window = max(1, round(sample_rate * 0.02))
    envelope = torch_functional.avg_pool1d(
        mono, window, stride=1, padding=window // 2
    )[..., : waveform.shape[-1]]
    active = torch.nonzero(envelope[0, 0] >= threshold).flatten()
    if not active.numel():
        return dict(audio), {
            "applied": False,
            "reason": "no samples exceeded the energy threshold",
        }
    padding = round(padding_seconds * sample_rate)
    start = max(0, int(active[0]) - padding)
    end = min(waveform.shape[-1], int(active[-1]) + 1 + padding)
    if end - start < round(0.25 * sample_rate):
        return dict(audio), {
            "applied": False,
            "reason": "detected active region was shorter than 0.25 seconds",
        }
    trimmed = waveform[..., start:end]
    return {"waveform": trimmed, "sample_rate": sample_rate}, {
        "applied": True,
        "start_sample": start,
        "end_sample": end,
        "start_seconds": start / sample_rate,
        "end_seconds": end / sample_rate,
        "threshold_dbfs": threshold_dbfs,
        "padding_seconds": padding_seconds,
    }


def decode_speech_audio(
    av_latent: Mapping,
    audio_vae,
    trim_mode: str = "none",
    energy_threshold_dbfs: float = -50.0,
    trim_padding_seconds: float = 0.10,
):
    _, audio_latent = nested_av_parts(dict(av_latent))
    decoded = vae_decode_audio(audio_vae, {"samples": audio_latent})
    if trim_mode == "none":
        output = decoded
        trim_report = {"applied": False, "reason": "trim_mode=none"}
    elif trim_mode == "conservative_energy":
        output, trim_report = _energy_trim(
            decoded, energy_threshold_dbfs, trim_padding_seconds
        )
    else:
        raise ValueError("trim_mode must be none or conservative_energy")
    report = {
        "schema": "minimax_h3_t8_speech_decode_v1",
        "trim_mode": trim_mode,
        "trim": trim_report,
        "output_facts": audio_quality_facts(output),
        "transcript_status": "not_run",
        "speaker_identity_status": "not_run",
        "warning": (
            "Energy trim cannot detect copied reference words or verify requested speech; "
            "use actual ASR/speaker QA before accepting a clone."
        ),
    }
    return output, _json(report)


def _pan_stereo(waveform: torch.Tensor, pan: float) -> torch.Tensor:
    pan = max(-1.0, min(1.0, float(pan)))
    angle = (pan + 1.0) * math.pi / 4.0
    gains = waveform.new_tensor([math.cos(angle), math.sin(angle)]).view(1, 2, 1)
    return waveform * gains * math.sqrt(2.0)


def _fade_curve(length: int, device, dtype, fade_in: bool) -> torch.Tensor:
    if length <= 0:
        return torch.ones((0,), device=device, dtype=dtype)
    phase = torch.linspace(0.0, math.pi / 2.0, length, device=device, dtype=dtype)
    curve = torch.sin(phase).square()
    return curve if fade_in else curve.flip(0)


def _srt_timestamp(seconds: float, separator: str = ",") -> str:
    total_ms = max(0, round(seconds * 1000))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{millis:03d}"


def assemble_speech_audio(
    plan: Mapping,
    audio_segments: Mapping | None,
    output_sample_rate: int = H3_AUDIO_SAMPLE_RATE,
    crossfade_seconds: float = 0.06,
    peak_limit_dbfs: float = -1.0,
):
    plan = validate_speech_plan(plan)
    audios = sorted_autogrow_values(audio_segments)
    segments = plan["segments"]
    if len(audios) != len(segments):
        raise ValueError(
            f"speech plan has {len(segments)} segments but {len(audios)} AUDIO inputs are connected"
        )
    if output_sample_rate not in {32000, 44100, 48000}:
        raise ValueError("output_sample_rate must be 32000, 44100, or 48000")
    if not 0.0 <= crossfade_seconds <= 0.5:
        raise ValueError("crossfade_seconds must be between 0 and 0.5")

    prepared: list[torch.Tensor] = []
    for segment, audio in zip(segments, audios, strict=True):
        waveform, sample_rate = validate_audio(audio, "audio_segment")
        waveform = waveform.detach().to(device="cpu", dtype=torch.float32)
        if sample_rate != output_sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, output_sample_rate
            )
        waveform = _to_stereo(waveform)
        waveform = waveform * (10.0 ** (float(segment.get("gain_db", 0.0)) / 20.0))
        waveform = _pan_stereo(waveform, float(segment.get("pan", 0.0)))
        prepared.append(waveform)

    starts: list[int] = []
    ends: list[int] = []
    cursor = 0
    fade_samples = round(crossfade_seconds * output_sample_rate)
    auto_crossfades: list[int] = [0] * len(prepared)
    for index, (segment, waveform) in enumerate(zip(segments, prepared, strict=True)):
        pause = round(float(segment.get("pause_before_seconds", 0.0)) * output_sample_rate)
        requested_overlap = round(
            float(segment.get("overlap_seconds", 0.0)) * output_sample_rate
        )
        if pause < 0 or requested_overlap < 0:
            raise ValueError("pause_before_seconds and overlap_seconds cannot be negative")
        if index == 0:
            start = pause
        else:
            automatic = fade_samples if pause == 0 and requested_overlap == 0 else 0
            overlap = max(requested_overlap, automatic)
            overlap = min(overlap, prepared[index - 1].shape[-1], waveform.shape[-1])
            start = max(0, cursor + pause - overlap)
            if automatic:
                auto_crossfades[index] = overlap
        end = start + waveform.shape[-1]
        starts.append(start)
        ends.append(end)
        cursor = max(cursor, end)

    output = torch.zeros((1, 2, max(1, max(ends))), dtype=torch.float32)
    for index, waveform in enumerate(prepared):
        overlap = auto_crossfades[index]
        if overlap:
            waveform = waveform.clone()
            waveform[..., :overlap] *= _fade_curve(
                overlap, waveform.device, waveform.dtype, True
            )
            previous_overlap = min(overlap, prepared[index - 1].shape[-1])
            region_start = max(0, starts[index])
            output[..., region_start : region_start + previous_overlap] *= _fade_curve(
                previous_overlap, output.device, output.dtype, False
            )
        output[..., starts[index] : ends[index]] += waveform

    limit = 10.0 ** (float(peak_limit_dbfs) / 20.0)
    peak_before = float(output.abs().amax())
    limiter_gain = min(1.0, limit / max(peak_before, 1e-12))
    output *= limiter_gain

    timeline = []
    srt_blocks = []
    vtt_blocks = ["WEBVTT", ""]
    for index, (segment, start, end) in enumerate(zip(segments, starts, ends, strict=True)):
        item = {
            "index": index,
            "speaker_id": segment["speaker_id"],
            "text": segment["text"],
            "start_sample": start,
            "end_sample": end,
            "start_seconds": start / output_sample_rate,
            "end_seconds": end / output_sample_rate,
        }
        timeline.append(item)
        srt_blocks.extend(
            [
                str(index + 1),
                f"{_srt_timestamp(item['start_seconds'])} --> {_srt_timestamp(item['end_seconds'])}",
                f"{item['speaker_id']}: {item['text']}",
                "",
            ]
        )
        vtt_blocks.extend(
            [
                f"{_srt_timestamp(item['start_seconds'], '.')} --> {_srt_timestamp(item['end_seconds'], '.')}",
                f"{item['speaker_id']}: {item['text']}",
                "",
            ]
        )

    result_audio = {"waveform": output, "sample_rate": output_sample_rate}
    report = {
        "schema": "minimax_h3_t8_speech_timeline_v1",
        "sample_rate": output_sample_rate,
        "total_samples": int(output.shape[-1]),
        "total_seconds": output.shape[-1] / output_sample_rate,
        "segment_count": len(segments),
        "crossfade_seconds": crossfade_seconds,
        "peak_before_limit_dbfs": _dbfs(peak_before),
        "limiter_gain": limiter_gain,
        "timeline": timeline,
        "subtitle_status": "actual_audio_boundaries_with_planned_text_not_asr_verified",
    }
    return result_audio, _json(report), "\n".join(srt_blocks), "\n".join(vtt_blocks)


def _dialogue_profile_map(profiles: Mapping | None) -> tuple[list[dict], dict[str, dict]]:
    values = [validate_voice_profile(value) for value in sorted_autogrow_values(profiles)]
    if not 2 <= len(values) <= 3:
        raise ValueError("Dialogue Script requires 2 or 3 connected voice profiles")
    mapping = {profile["speaker_id"]: profile for profile in values}
    if len(mapping) != len(values):
        raise ValueError("Dialogue voice profiles must use unique speaker_id values")
    for index, profile in enumerate(values, 1):
        mapping[f"S{index}"] = profile
    return values, mapping


def make_dialogue_plan(
    script: str,
    script_format: str,
    profiles: Mapping | None,
    default_language: str,
    default_space: str,
) -> tuple[dict, str]:
    values, aliases = _dialogue_profile_map(profiles)
    if default_language not in SUPPORTED_DIALOGUE_LANGUAGES:
        raise ValueError(f"unsupported H3 dialogue language: {default_language}")
    if default_space not in SPACE_DESCRIPTIONS:
        raise ValueError(f"unknown dialogue space: {default_space}")
    script = str(script or "").strip()
    if not script:
        raise ValueError("dialogue script cannot be empty")
    if script_format == "auto":
        script_format = "json" if script[:1] in "[{" else "speaker_lines"

    raw_turns = []
    if script_format == "speaker_lines":
        for line_number, line in enumerate(script.splitlines(), 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            match = re.match(r"^([^:：]+)[:：]\s*(.+)$", line)
            if not match:
                raise ValueError(
                    f"dialogue line {line_number} must use 'speaker: spoken text'"
                )
            raw_turns.append({"speaker_id": match.group(1).strip(), "text": match.group(2)})
    elif script_format == "json":
        try:
            payload = json.loads(script)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid dialogue JSON: {error}") from error
        raw_turns = payload.get("turns") if isinstance(payload, dict) else payload
        if not isinstance(raw_turns, list):
            raise ValueError("dialogue JSON must be a list or an object with a turns list")
    else:
        raise ValueError("script_format must be auto, speaker_lines, or json")
    if not raw_turns:
        raise ValueError("dialogue script contains no turns")

    casefold_aliases = {key.casefold(): value for key, value in aliases.items()}
    turns = []
    for index, raw in enumerate(raw_turns):
        if not isinstance(raw, Mapping):
            raise ValueError(f"dialogue turn {index} must be an object")
        requested = str(raw.get("speaker_id") or raw.get("speaker") or "").strip()
        profile = casefold_aliases.get(requested.casefold())
        if profile is None:
            known = ", ".join(profile["speaker_id"] for profile in values)
            raise ValueError(f"unknown dialogue speaker '{requested}'; connected speakers: {known}")
        text = validate_spoken_text(raw.get("text", ""))
        turn = {
            "index": index,
            "speaker_id": profile["speaker_id"],
            "text": text,
            "language": str(raw.get("language") or default_language),
            "direction": _plain_direction(raw.get("direction") or raw.get("subtext") or ""),
            "emotion": _plain_direction(raw.get("emotion") or "neutral"),
            "emotion_intensity": float(raw.get("emotion_intensity", 0.5)),
            "space": str(raw.get("space") or default_space),
            "pause_before_seconds": float(raw.get("pause_before_seconds", 0.0)),
            "overlap_seconds": float(raw.get("overlap_seconds", 0.0)),
            "gain_db": float(raw.get("gain_db", 0.0)),
            "pan": float(raw.get("pan", 0.0)),
        }
        if turn["language"] not in SUPPORTED_DIALOGUE_LANGUAGES:
            raise ValueError(f"dialogue turn {index} has an unsupported language")
        if turn["space"] not in SPACE_DESCRIPTIONS:
            raise ValueError(f"dialogue turn {index} has an unknown space")
        if not 0.0 <= turn["emotion_intensity"] <= 1.0:
            raise ValueError(f"dialogue turn {index} emotion_intensity must be 0-1")
        if turn["pause_before_seconds"] < 0 or turn["overlap_seconds"] < 0:
            raise ValueError(f"dialogue turn {index} pause/overlap cannot be negative")
        if not -1.0 <= turn["pan"] <= 1.0:
            raise ValueError(f"dialogue turn {index} pan must be -1 to 1")
        turns.append(turn)

    canonical_profiles = {profile["speaker_id"]: profile for profile in values}
    plan = {
        "schema": SPEECH_PLAN_SCHEMA,
        "kind": "dialogue",
        "segments": turns,
        "profiles": canonical_profiles,
        "script_format": script_format,
        "render_strategy": "one_turn_at_a_time",
        "joint_multi_speaker_status": "not_used",
        "timing_status": "planned_until_actual_turn_audio_is_assembled",
    }
    report = public_plan(plan)
    report["limitations"] = [
        "Each turn is generated independently to reduce speaker swapping.",
        "Overlap and pan are deterministic mix controls, not evidence of joint H3 dialogue quality.",
        "Subtitles remain planned text until ASR verification is added.",
    ]
    return plan, _json(report)


def select_dialogue_turn(plan: Mapping, turn_index: int):
    plan = validate_speech_plan(plan)
    if plan.get("kind") != "dialogue":
        raise ValueError("Dialogue Turn Select requires a dialogue plan")
    segments = plan["segments"]
    if not 0 <= int(turn_index) < len(segments):
        raise ValueError(f"turn_index must be between 0 and {len(segments) - 1}")
    segment = deepcopy(segments[int(turn_index)])
    profile = validate_voice_profile(plan["profiles"][segment["speaker_id"]])
    speech_plan = {
        "schema": SPEECH_PLAN_SCHEMA,
        "kind": "dialogue_turn",
        "segments": [{**segment, "index": 0}],
        "profiles": {profile["speaker_id"]: profile},
        "source_turn_index": int(turn_index),
        "source_turn_count": len(segments),
    }
    report = {
        "turn_index": int(turn_index),
        "turn_count": len(segments),
        "speaker_id": profile["speaker_id"],
        "text": segment["text"],
    }
    return profile, speech_plan, segment["text"], _json(report)
