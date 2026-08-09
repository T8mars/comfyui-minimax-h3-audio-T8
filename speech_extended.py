from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time
import unicodedata
import re
import uuid

import folder_paths
from safetensors import safe_open
from safetensors.torch import save_file
import torch
import torchaudio

from .core import validate_audio
from .conditioning import build_conditioning
from .long_video_delivery import _open_advisory_lock, _release_advisory_lock, _try_advisory_lock
from .speech import (
    H3_AUDIO_SAMPLE_RATE,
    assemble_speech_audio,
    public_plan,
    render_frame_count,
    validate_speech_plan,
    validate_voice_profile,
)


SPEECH_SESSION_TYPE = "H3_T8_SPEECH_LONGFORM_SESSION"
SPEECH_SESSION_SCHEMA = "minimax_h3_t8_speech_longform_session_v1"
SPEECH_MANIFEST_SCHEMA = "minimax_h3_t8_speech_longform_manifest_v1"
VOICE_LIBRARY_SCHEMA = "minimax_h3_t8_voice_library_entry_v1"


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def _safe_name(value: str, label: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or "").strip())
    normalized = re.sub(r"[^0-9A-Za-z._\-\u3400-\u9fff]+", "_", normalized).strip("._-")
    if not normalized:
        raise ValueError(f"{label} must contain a letter, number, or CJK character")
    if len(normalized) > 80:
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:10]
        normalized = f"{normalized[:64]}_{digest}"
    return normalized


def _atomic_write_json(path: Path, value: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _voice_library_root() -> Path:
    configured = os.environ.get("H3_T8_VOICE_LIBRARY_DIR", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    getter = getattr(folder_paths, "get_user_directory", None)
    base = Path(getter() if getter else folder_paths.get_output_directory()).resolve()
    return (base / "minimax_h3_t8" / "voice_profiles").resolve()


def save_voice_profile(profile: Mapping, library_name: str, replace_existing: bool = False) -> tuple[dict, str]:
    profile = validate_voice_profile(profile)
    name = _safe_name(library_name, "library_name")
    root = _voice_library_root()
    entry = (root / name).resolve()
    if root != entry and root not in entry.parents:
        raise ValueError("voice library entry escaped its root")
    if entry.exists() and not replace_existing:
        raise FileExistsError(
            f"voice profile '{name}' already exists; enable replace_existing explicitly"
        )
    entry.mkdir(parents=True, exist_ok=True)
    public = deepcopy(dict(profile))
    reference = public.pop("reference_audio", None)
    public.pop("reference_facts", None)
    payload = {
        "schema": VOICE_LIBRARY_SCHEMA,
        "saved_unix": time.time(),
        "library_name": name,
        "profile": public,
        "contains_reference_audio": reference is not None,
        "consent_record": {
            "rights_confirmed": bool(profile.get("rights_confirmed")),
            "source_hash": profile.get("reference_sha256"),
            "note": "A checkbox is a provenance record, not legal advice or proof of consent.",
        },
    }
    tensors = {}
    if reference is not None:
        waveform, sample_rate = validate_audio(reference, "profile reference_audio")
        tensors["reference_waveform"] = waveform.detach().to(device="cpu", dtype=torch.float32).contiguous()
        payload["reference_sample_rate"] = int(sample_rate)
    temporary = entry / f".profile.{uuid.uuid4().hex}.tmp"
    final = entry / "profile.safetensors"
    save_file(tensors or {"empty": torch.empty(0)}, str(temporary), metadata={"entry_json": json.dumps(payload, ensure_ascii=False)})
    os.replace(temporary, final)
    payload["file_sha256"] = _sha256_file(final)
    _atomic_write_json(entry / "entry.json", payload)
    loaded, _ = load_voice_profile(name)
    return loaded, _json({**payload, "path": str(entry), "operation": "saved"})


def load_voice_profile(library_name: str) -> tuple[dict, str]:
    name = _safe_name(library_name, "library_name")
    root = _voice_library_root()
    entry = (root / name).resolve()
    path = entry / "profile.safetensors"
    if not path.is_file():
        raise FileNotFoundError(f"voice profile '{name}' does not exist")
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        payload = json.loads(metadata.get("entry_json", "{}"))
        profile = dict(payload.get("profile") or {})
        if payload.get("contains_reference_audio"):
            profile["reference_audio"] = {
                "waveform": handle.get_tensor("reference_waveform"),
                "sample_rate": int(payload["reference_sample_rate"]),
            }
        else:
            profile["reference_audio"] = None
    profile["persistence"] = "explicit_local_voice_library"
    validate_voice_profile(profile)
    report = {
        "schema": VOICE_LIBRARY_SCHEMA,
        "operation": "loaded",
        "library_name": name,
        "path": str(entry),
        "file_sha256": _sha256_file(path),
        "rights_confirmed": bool(profile.get("rights_confirmed")),
        "privacy_warning": "Delete the entry when the reference voice no longer needs to be retained.",
    }
    return profile, _json(report)


def delete_voice_profile(library_name: str, confirm_delete: bool) -> str:
    if not confirm_delete:
        raise ValueError("confirm_delete must be enabled before moving a voice profile to trash")
    name = _safe_name(library_name, "library_name")
    root = _voice_library_root()
    entry = (root / name).resolve()
    if not entry.is_dir():
        raise FileNotFoundError(f"voice profile '{name}' does not exist")
    trash = root / ".trash"
    trash.mkdir(parents=True, exist_ok=True)
    target = trash / f"{name}.{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}.{uuid.uuid4().hex[:8]}"
    shutil.move(str(entry), str(target))
    return _json(
        {
            "schema": VOICE_LIBRARY_SCHEMA,
            "operation": "moved_to_trash",
            "library_name": name,
            "recoverable_path": str(target),
            "permanently_deleted": False,
        }
    )


def apply_performance_direction(
    plan: Mapping,
    segment_index: int,
    emotion: str,
    prompt_intensity: float,
    pace: str,
    pitch: str,
    energy: str,
    nonverbal_direction: str,
) -> tuple[dict, str]:
    plan = validate_speech_plan(plan)
    if not 0.0 <= float(prompt_intensity) <= 1.0:
        raise ValueError("prompt_intensity must be between 0 and 1")
    if pace not in {"very_slow", "slow", "natural", "fast", "very_fast"}:
        raise ValueError("unknown pace preset")
    if pitch not in {"very_low", "low", "natural", "high", "very_high"}:
        raise ValueError("unknown pitch preset")
    if energy not in {"restrained", "low", "natural", "high", "intense"}:
        raise ValueError("unknown energy preset")
    output = deepcopy(plan)
    indices = range(len(output["segments"])) if int(segment_index) < 0 else [int(segment_index)]
    for index in indices:
        if not 0 <= index < len(output["segments"]):
            raise ValueError("segment_index is outside the speech plan")
        segment = output["segments"][index]
        additions = [
            f"pace direction: {pace.replace('_', ' ')}",
            f"vocal pitch direction: {pitch.replace('_', ' ')}",
            f"performance energy: {energy}",
        ]
        clean_nonverbal = " ".join(str(nonverbal_direction or "").split())
        if clean_nonverbal:
            additions.append(f"nonverbal performance: {clean_nonverbal}")
        existing = str(segment.get("direction", "")).strip(" ;")
        segment["direction"] = "; ".join(([existing] if existing else []) + additions)
        segment["emotion"] = " ".join(str(emotion or "neutral").split())
        segment["emotion_intensity"] = float(prompt_intensity)
        segment["performance_controls"] = {
            "pace": pace,
            "pitch": pitch,
            "energy": energy,
            "nonverbal_direction": clean_nonverbal,
            "control_kind": "uncalibrated_prompt_direction",
        }
    report = {
        "schema": "minimax_h3_t8_speech_performance_v1",
        "modified_segments": list(indices),
        "control_kind": "uncalibrated_prompt_direction",
        "warning": (
            "These values steer H3 prompt interpretation; they are not measured speech-rate, "
            "F0, or emotion-intensity controls. Use the ADR/DSP node for deterministic duration/pitch changes."
        ),
    }
    return output, _json(report)


def fit_audio_for_adr(
    audio: Mapping,
    target_duration_seconds: float,
    fit_mode: str = "safe_time_stretch",
    minimum_rate: float = 0.90,
    maximum_rate: float = 1.10,
    pitch_semitones: float = 0.0,
) -> tuple[dict, str]:
    waveform, sample_rate = validate_audio(audio, "adr_audio")
    source = waveform.detach().to(device="cpu", dtype=torch.float32)
    target_samples = int(round(float(target_duration_seconds) * sample_rate))
    if target_samples <= 0:
        raise ValueError("target_duration_seconds must produce at least one sample")
    source_samples = int(source.shape[-1])
    rate = source_samples / target_samples
    if minimum_rate <= 0 or maximum_rate < minimum_rate:
        raise ValueError("invalid ADR rate bounds")
    if fit_mode not in {"refuse_if_mismatch", "pad_or_trim", "safe_time_stretch"}:
        raise ValueError("unknown ADR fit_mode")
    if fit_mode == "refuse_if_mismatch" and source_samples != target_samples:
        raise ValueError(
            f"ADR source has {source_samples} samples but target requires {target_samples}"
        )
    if fit_mode == "safe_time_stretch" and not minimum_rate <= rate <= maximum_rate:
        raise ValueError(
            f"required time-stretch rate {rate:.4f} is outside the explicit safe range "
            f"{minimum_rate:.4f}-{maximum_rate:.4f}; regenerate or widen it knowingly"
        )

    output = source
    if fit_mode == "safe_time_stretch" and source_samples != target_samples:
        n_fft = 1024
        hop_length = 256
        flat = source.reshape(-1, source_samples)
        window = torch.hann_window(n_fft, dtype=flat.dtype)
        spectrum = torch.stft(
            flat,
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
            return_complex=True,
        )
        phase_advance = torch.linspace(0, math.pi * hop_length, spectrum.shape[-2], dtype=flat.dtype)[..., None]
        stretched = torchaudio.functional.phase_vocoder(spectrum, rate=rate, phase_advance=phase_advance)
        flat = torch.istft(
            stretched,
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
            length=target_samples,
        )
        output = flat.reshape(source.shape[0], source.shape[1], target_samples)
    elif source_samples < target_samples:
        output = torch.nn.functional.pad(source, (0, target_samples - source_samples))
    elif source_samples > target_samples:
        output = source[..., :target_samples]

    if float(pitch_semitones) != 0.0:
        if not -12.0 <= float(pitch_semitones) <= 12.0:
            raise ValueError("pitch_semitones must stay within -12 to +12")
        output = torchaudio.functional.pitch_shift(output, sample_rate, float(pitch_semitones))
    if output.shape[-1] < target_samples:
        output = torch.nn.functional.pad(output, (0, target_samples - output.shape[-1]))
    output = output[..., :target_samples].contiguous()
    report = {
        "schema": "minimax_h3_t8_speech_adr_fit_v1",
        "fit_mode": fit_mode,
        "sample_rate": int(sample_rate),
        "source_samples": source_samples,
        "target_samples": target_samples,
        "output_samples": int(output.shape[-1]),
        "exact_sample_error": int(output.shape[-1]) - target_samples,
        "required_rate": rate,
        "pitch_semitones": float(pitch_semitones),
        "timing_status": "sample_exact",
        "lip_sync_status": "not_verified",
        "warning": "Sample-exact duration does not prove phoneme alignment or visual lip sync.",
    }
    return {"waveform": output, "sample_rate": int(sample_rate)}, _json(report)


def _speech_job_root(job_id: str) -> Path:
    safe = _safe_name(job_id, "job_id")
    output = Path(folder_paths.get_output_directory()).resolve()
    root = (output / "minimax_h3_t8" / "speech_jobs" / safe).resolve()
    if output != root and output not in root.parents:
        raise ValueError("speech job path escaped the ComfyUI output directory")
    return root


def _plan_hash(plan: Mapping) -> str:
    encoded = json.dumps(public_plan(plan), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _manifest_path(root: Path) -> Path:
    return root / "manifest.json"


def speech_manifest_fingerprint(job_id: str) -> str:
    """Return a cache key that advances only after durable manifest changes."""
    root = _speech_job_root(job_id)
    path = _manifest_path(root)
    if not path.is_file():
        return f"missing:{_safe_name(job_id, 'job_id')}"
    stat = path.stat()
    return f"{stat.st_mtime_ns}:{stat.st_size}:{_sha256_file(path)}"


def _locked(root: Path):
    class _Lock:
        def __enter__(self):
            root.mkdir(parents=True, exist_ok=True)
            self.handle = _open_advisory_lock(root / "manifest.lock")
            if not _try_advisory_lock(self.handle):
                self.handle.close()
                raise RuntimeError("speech long-form session is owned by another ComfyUI process")
            return self

        def __exit__(self, exc_type, exc, tb):
            try:
                _release_advisory_lock(self.handle)
            finally:
                self.handle.close()

    return _Lock()


def _load_manifest(root: Path) -> dict | None:
    path = _manifest_path(root)
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != SPEECH_MANIFEST_SCHEMA:
        raise ValueError("speech long-form manifest is unsupported or corrupt")
    return value


def start_or_resume_longform(plan: Mapping, job_id: str) -> tuple[dict, dict, int, str, str]:
    plan = validate_speech_plan(plan)
    root = _speech_job_root(job_id)
    digest = _plan_hash(plan)
    with _locked(root):
        manifest = _load_manifest(root)
        if manifest is None:
            manifest = {
                "schema": SPEECH_MANIFEST_SCHEMA,
                "job_id": _safe_name(job_id, "job_id"),
                "plan_sha256": digest,
                "plan": public_plan(plan),
                "created_unix": time.time(),
                "updated_unix": time.time(),
                "state": "active",
                "cancel_requested": False,
                "segments": [
                    {"index": index, "status": "pending", "text": segment["text"]}
                    for index, segment in enumerate(plan["segments"])
                ],
            }
            _atomic_write_json(_manifest_path(root), manifest)
        elif manifest.get("plan_sha256") != digest:
            raise ValueError("job_id already belongs to a different speech plan")
        if manifest.get("cancel_requested"):
            state = "cancelled"
            next_index = -1
        else:
            pending = [entry["index"] for entry in manifest["segments"] if entry.get("status") != "accepted"]
            next_index = int(pending[0]) if pending else -1
            state = "complete" if next_index < 0 else "active"
        manifest["state"] = state
        manifest["updated_unix"] = time.time()
        _atomic_write_json(_manifest_path(root), manifest)
    session = {
        "schema": SPEECH_SESSION_SCHEMA,
        "job_id": manifest["job_id"],
        "root": str(root),
        "plan_sha256": digest,
        "next_index": next_index,
        "state": state,
    }
    if next_index >= 0:
        segment = dict(plan["segments"][next_index])
        profile = dict(plan["profiles"][segment["speaker_id"]])
        text = str(segment["text"])
    else:
        profile = dict(next(iter(plan["profiles"].values())))
        text = ""
    report = {
        "schema": SPEECH_SESSION_SCHEMA,
        "job_id": manifest["job_id"],
        "state": state,
        "next_index": next_index,
        "accepted_segments": sum(entry.get("status") == "accepted" for entry in manifest["segments"]),
        "segment_count": len(manifest["segments"]),
        "manifest_path": str(_manifest_path(root)),
        "resume_contract": "accepted segments are immutable unless replace_existing is explicit",
        "streaming_contract": "each accepted segment becomes a chunk-ready file; this is not token/frame realtime streaming",
    }
    return session, profile, next_index, text, _json(report)


def accept_longform_segment(
    session: Mapping,
    plan: Mapping,
    segment_index: int,
    audio: Mapping,
    transcript: str,
    text_similarity: float,
    speaker_similarity: float,
    accepted: bool,
    replace_existing: bool = False,
) -> tuple[dict, str]:
    if session.get("schema") != SPEECH_SESSION_SCHEMA:
        raise ValueError("expected an H3 T8 speech long-form session")
    plan = validate_speech_plan(plan)
    if session.get("plan_sha256") != _plan_hash(plan):
        raise ValueError("speech session and plan do not match")
    index = int(segment_index)
    if not 0 <= index < len(plan["segments"]):
        raise ValueError("segment_index is outside the speech plan")
    if not accepted:
        raise ValueError("rejected speech must not be committed to the accepted manifest")
    waveform, sample_rate = validate_audio(audio, "accepted_speech_audio")
    if int(waveform.shape[0]) != 1:
        raise ValueError("long-form accepted AUDIO currently requires batch size 1")
    root = Path(str(session["root"])).resolve()
    expected_root = _speech_job_root(str(session["job_id"])).resolve()
    if root != expected_root:
        raise ValueError("speech session root is invalid")
    segment_path = root / "accepted" / f"segment_{index:05d}.safetensors"
    preview_path = root / "accepted" / f"segment_{index:05d}.flac"
    segment_path.parent.mkdir(parents=True, exist_ok=True)
    with _locked(root):
        manifest = _load_manifest(root)
        if manifest is None or manifest.get("plan_sha256") != session.get("plan_sha256"):
            raise ValueError("speech manifest is missing or does not match the session")
        if manifest.get("cancel_requested"):
            raise RuntimeError("speech session is cancelled; clear cancellation before accepting")
        current = manifest["segments"][index]
        if current.get("status") == "accepted" and not replace_existing:
            raise FileExistsError("segment is already accepted; enable replace_existing explicitly")
        temporary = segment_path.with_suffix(f".{uuid.uuid4().hex}.tmp")
        metadata = {
            "sample_rate": str(int(sample_rate)),
            "segment_index": str(index),
            "expected_text": str(plan["segments"][index]["text"]),
            "transcript": str(transcript or ""),
        }
        save_file(
            {"waveform": waveform.detach().to(device="cpu", dtype=torch.float32).contiguous()},
            str(temporary),
            metadata=metadata,
        )
        file_hash = _sha256_file(temporary)
        os.replace(temporary, segment_path)
        preview_temporary = preview_path.with_name(
            f".{preview_path.stem}.{uuid.uuid4().hex}.tmp.flac"
        )
        try:
            import soundfile

            soundfile.write(
                str(preview_temporary),
                waveform[0]
                .detach()
                .to(device="cpu", dtype=torch.float32)
                .transpose(0, 1)
                .numpy(),
                int(sample_rate),
                format="FLAC",
                subtype="PCM_24",
            )
            os.replace(preview_temporary, preview_path)
        finally:
            preview_temporary.unlink(missing_ok=True)
        current.update(
            {
                "status": "accepted",
                "file": segment_path.relative_to(root).as_posix(),
                "file_sha256": file_hash,
                "preview_file": preview_path.relative_to(root).as_posix(),
                "sample_rate": int(sample_rate),
                "samples": int(waveform.shape[-1]),
                "duration_seconds": waveform.shape[-1] / int(sample_rate),
                "transcript": str(transcript or ""),
                "text_similarity": float(text_similarity),
                "speaker_similarity": float(speaker_similarity),
                "accepted_unix": time.time(),
            }
        )
        pending = [entry["index"] for entry in manifest["segments"] if entry.get("status") != "accepted"]
        manifest["state"] = "complete" if not pending else "active"
        manifest["updated_unix"] = time.time()
        _atomic_write_json(_manifest_path(root), manifest)
    report = {
        "schema": SPEECH_MANIFEST_SCHEMA,
        "operation": "accepted",
        "segment_index": index,
        "state": manifest["state"],
        "next_index": int(pending[0]) if pending else -1,
        "chunk_ready_path": str(segment_path),
        "chunk_ready_preview_path": str(preview_path),
        "file_sha256": file_hash,
        "crash_contract": "audio file was atomically committed before the manifest advanced",
    }
    return dict(audio), _json(report)


def control_longform_session(job_id: str, action: str, confirm_reset: bool = False) -> str:
    if action not in {"status", "request_cancel", "clear_cancel", "reset_to_trash"}:
        raise ValueError("unknown long-form control action")
    root = _speech_job_root(job_id)
    if action == "reset_to_trash":
        if not confirm_reset:
            raise ValueError("confirm_reset must be enabled before moving a session to trash")
        if not root.exists():
            raise FileNotFoundError("speech session does not exist")
        trash = root.parent / ".trash"
        trash.mkdir(parents=True, exist_ok=True)
        target = trash / f"{root.name}.{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}.{uuid.uuid4().hex[:8]}"
        shutil.move(str(root), str(target))
        return _json({"operation": "moved_to_trash", "recoverable_path": str(target)})
    with _locked(root):
        manifest = _load_manifest(root)
        if manifest is None:
            raise FileNotFoundError("speech session does not exist")
        if action == "request_cancel":
            manifest["cancel_requested"] = True
            manifest["state"] = "cancelled"
        elif action == "clear_cancel":
            manifest["cancel_requested"] = False
            manifest["state"] = "active"
        manifest["updated_unix"] = time.time()
        _atomic_write_json(_manifest_path(root), manifest)
    return _json(manifest)


def compose_longform_session(plan: Mapping, job_id: str, crossfade_seconds: float, peak_limit_dbfs: float):
    plan = validate_speech_plan(plan)
    root = _speech_job_root(job_id)
    manifest = _load_manifest(root)
    if manifest is None or manifest.get("plan_sha256") != _plan_hash(plan):
        raise ValueError("speech job does not match the connected plan")
    missing = [entry["index"] for entry in manifest["segments"] if entry.get("status") != "accepted"]
    if missing:
        raise ValueError(f"speech job is incomplete; pending segments: {missing}")
    audios = {}
    for entry in manifest["segments"]:
        path = (root / entry["file"]).resolve()
        if root not in path.parents or not path.is_file():
            raise ValueError(f"accepted segment {entry['index']} file is missing or escaped the job root")
        if _sha256_file(path) != entry["file_sha256"]:
            raise ValueError(f"accepted segment {entry['index']} failed SHA256 verification")
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
            audios[f"audio_segment_{entry['index']}"] = {
                "waveform": handle.get_tensor("waveform"),
                "sample_rate": int(metadata["sample_rate"]),
            }
    audio, timeline, srt, vtt = assemble_speech_audio(
        plan,
        audios,
        H3_AUDIO_SAMPLE_RATE,
        crossfade_seconds,
        peak_limit_dbfs,
    )
    report = json.loads(timeline)
    report["longform"] = {
        "job_id": manifest["job_id"],
        "manifest_path": str(_manifest_path(root)),
        "segment_hashes_verified": True,
        "crash_resumed": True,
    }
    return audio, _json(report), srt, vtt


def build_joint_dialogue_conditioning(
    clip,
    video_vae,
    audio_vae,
    dialogue_plan: Mapping,
    start_turn: int,
    turn_count: int,
    render_seconds: float,
    resolution: int,
):
    """Build an explicitly unverified 2-3 speaker joint Ref2VA experiment."""
    plan = validate_speech_plan(dialogue_plan)
    if plan.get("kind") != "dialogue":
        raise ValueError("joint dialogue conditioning requires a dialogue plan")
    start = int(start_turn)
    selected = plan["segments"][start : start + int(turn_count)]
    if not 2 <= len(selected) <= 3:
        raise ValueError("joint dialogue EXP requires exactly 2 or 3 selected turns")
    speaker_ids = []
    for segment in selected:
        speaker_id = segment["speaker_id"]
        if speaker_id not in speaker_ids:
            speaker_ids.append(speaker_id)
    if not 2 <= len(speaker_ids) <= 3:
        raise ValueError("joint dialogue EXP requires 2 or 3 distinct speakers")
    profiles = [validate_voice_profile(plan["profiles"][speaker_id]) for speaker_id in speaker_ids]
    if any(profile["mode"] != "reference_voice" for profile in profiles):
        raise ValueError("joint dialogue EXP requires reference_voice profiles for every speaker")
    if resolution not in {32, 64, 128}:
        raise ValueError("speech canvas must be 32, 64, or 128 pixels")
    frames = render_frame_count(render_seconds)
    ordinal = {speaker_id: index + 1 for index, speaker_id in enumerate(speaker_ids)}
    definitions = ["<Picture 1> is a blank dark frame and carries no content."]
    for profile in profiles:
        index = ordinal[profile["speaker_id"]]
        definitions.append(
            f"<Audio {index}> is the canonical voice identity of Speaker {index} (S{index})."
        )
    lines = []
    for segment in selected:
        index = ordinal[segment["speaker_id"]]
        language = segment.get("language") or plan["profiles"][segment["speaker_id"]].get("language", "English")
        text = str(segment["text"])
        lines.append(f"Speaker {index} (S{index}) says exactly and only: <d>[{language}] {text}</d>")
    prompt = f"""subject_definitions:
{chr(10).join(definitions)}

summary:
A static dark-frame recording of a quiet two-person or three-person spoken dialogue.

detailed_description:
[Shot 1] A static featureless dark scene in a quiet furnished room. The numbered speakers perform one dialogue take in this order:
{chr(10).join(lines)}

overall_soundscape:
Subtle dry room tone.

non_diegetic_music:
N/A"""
    dark = torch.full((1, 64, 64, 3), 8.0 / 255.0, dtype=torch.float32)
    ref_audios = {
        f"ref_audio_{index}": profile["reference_audio"]
        for index, profile in enumerate(profiles)
    }
    result = build_conditioning(
        clip=clip,
        video_vae=video_vae,
        audio_vae=audio_vae,
        prompt=prompt,
        width=resolution,
        height=resolution,
        length=frames,
        task_type="Ref2VA",
        audio_mode="native",
        audio_denoise_strength=1.0,
        add_source_as_reference=False,
        prompt_primary_audio_ordinal=0,
        strict_prompt_tags=True,
        ref_image_size="match",
        reference_video_policy="official_2_to_15s",
        ref_images={"ref_image_0": dark},
        ref_audios=ref_audios,
    )
    conditioning, latent, _, conditioned_prompt, media_map, base_report = result
    report = {
        "schema": "minimax_h3_t8_joint_dialogue_conditioning_v1",
        "status": "experimental_unverified",
        "speaker_ids": speaker_ids,
        "selected_turn_indices": [int(segment["index"]) for segment in selected],
        "aligned_frames": frames,
        "canvas": [resolution, resolution],
        "media_map_json": media_map,
        "base_conditioning_report": base_report,
        "denial": (
            "This node does not establish speaker binding. Do not use it as a stable multi-speaker "
            "or high-fidelity-clone path until leakage, identity-swap, and blind-listening gates pass."
        ),
    }
    return conditioning, latent, conditioned_prompt, _json(report)
