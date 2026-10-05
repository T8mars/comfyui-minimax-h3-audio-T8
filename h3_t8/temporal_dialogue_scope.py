"""Opt-in Chunk window text compiler; no sampling or audio modification.

Inputs are explicit timed dialogue and *actual snapped* window boundaries.
The result is raw text to encode natively, never an embedding slice. Receipts
bind published intervals/content, not ASR or a promise that a line was spoken.
Legacy LongVideo and Chunk executors do not implicitly consume this module.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re


POLICY = "t8.temporal-dialogue.window-text.v2"
NATIVE_FPS = 24


def _digest(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")
    # Structured fields are plain text. Native <d> pairs are inserted by us;
    # malformed or nested pairs must not leak into the native tokenizer.
    if re.search(r"</?d(?=[\s>/]|$)", value, re.IGNORECASE):
        raise ValueError(f"{name}: move <d> dialogue into its explicit timed event")
    return value


def _sha(value, name):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA256")


@dataclass(frozen=True)
class DialogueEvent:
    event_id: str
    speaker: str
    utterance: str
    start_frame: int
    end_frame: int
    requested_start_seconds: float
    requested_end_seconds: float


@dataclass(frozen=True)
class PerformanceEvent:
    event_id: str
    cue: str
    start_frame: int
    end_frame: int
    requested_start_seconds: float
    requested_end_seconds: float


@dataclass(frozen=True)
class DialoguePlan:
    policy: str
    global_prompt: str
    total_frames: int
    events: tuple[DialogueEvent, ...]
    sha256: str
    performance_events: tuple[PerformanceEvent, ...] = ()


def _plan_payload(plan):
    return {key: value for key, value in asdict(plan).items() if key != "sha256"}


def _performance_events(events, total_frames):
    if isinstance(events, str):
        try:
            events = json.loads(events)
        except (TypeError, ValueError) as exc:
            raise ValueError("performance_events_json is not valid JSON") from exc
    if type(events) is not list:
        raise ValueError("performance events must be an explicit JSON array")
    normalized, seen = [], set()
    required = {"event_id", "cue", "start_seconds", "end_seconds"}
    for event in events:
        if type(event) is not dict or set(event) != required:
            raise ValueError("Each performance event needs exactly event_id, cue, start_seconds, end_seconds")
        event_id = _text(event["event_id"], "performance event_id")
        if event_id in seen:
            raise ValueError("Performance event_id must be unique")
        seen.add(event_id)
        times = [event[key] for key in ("start_seconds", "end_seconds")]
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in times):
            raise ValueError("Performance times must be finite numbers")
        start, end = map(float, times)
        if not 0 <= start < end <= total_frames / NATIVE_FPS:
            raise ValueError("Performance interval is outside the complete source timeline")
        sf, ef = round(start * NATIVE_FPS), round(end * NATIVE_FPS)
        if sf >= ef:
            raise ValueError("Performance interval is empty after native frame quantization")
        normalized.append(PerformanceEvent(event_id, _text(event["cue"], "performance cue"),
                                           sf, ef, start, end))
    return tuple(sorted(normalized, key=lambda item: (item.start_frame, item.end_frame, item.event_id)))


def build_dialogue_plan(global_prompt, events, total_frames, *, performance_events="[]"):
    """Round explicit seconds to native 24fps; keep requested times in identity."""
    _integer(total_frames, "total_frames", 1)
    global_prompt = _text(global_prompt, "Global prompt")
    if isinstance(events, str):
        try:
            events = json.loads(events)
        except (TypeError, ValueError) as exc:
            raise ValueError("dialogue_events_json is not valid JSON") from exc
    if type(events) is not list:
        raise ValueError("dialogue events must be an explicit JSON array")
    normalized, seen = [], set()
    required = {"event_id", "speaker", "utterance", "start_seconds", "end_seconds"}
    for event in events:
        if type(event) is not dict or set(event) != required:
            raise ValueError("Each dialogue event needs exactly event_id, speaker, utterance, start_seconds, end_seconds")
        event_id = _text(event["event_id"], "event_id")
        if event_id in seen:
            raise ValueError("Dialogue event_id must be unique; equal utterances may use distinct IDs")
        seen.add(event_id)
        times = []
        for key in ("start_seconds", "end_seconds"):
            value = event[key]
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{key} must be a finite number")
            times.append(float(value))
        start, end = times
        if not 0 <= start < end <= total_frames / NATIVE_FPS:
            raise ValueError("Dialogue interval is outside the complete source timeline")
        sf, ef = round(start * NATIVE_FPS), round(end * NATIVE_FPS)
        if sf >= ef:
            raise ValueError("Dialogue interval is empty after native frame quantization")
        normalized.append(DialogueEvent(event_id, _text(event["speaker"], "speaker"),
                                        _text(event["utterance"], "utterance"),
                                        sf, ef, start, end))
    normalized.sort(key=lambda item: (item.start_frame, item.end_frame, item.event_id))
    performance = _performance_events(performance_events, total_frames)
    draft = DialoguePlan(POLICY, global_prompt, total_frames, tuple(normalized), "", performance)
    return DialoguePlan(POLICY, global_prompt, total_frames, draft.events,
                        _digest(_plan_payload(draft)), performance)


def validate_dialogue_plan(plan):
    if type(plan) is not DialoguePlan or plan.policy != POLICY:
        raise ValueError("Expected an explicit Temporal Dialogue Plan")
    if type(plan.events) is not tuple or any(type(item) is not DialogueEvent for item in plan.events):
        raise ValueError("Temporal Dialogue Plan events are not typed immutable data")
    if (type(plan.performance_events) is not tuple
            or any(type(item) is not PerformanceEvent for item in plan.performance_events)):
        raise ValueError("Temporal performance events are not typed immutable data")
    _sha(plan.sha256, "plan.sha256")
    raw = [{"event_id": item.event_id, "speaker": item.speaker,
            "utterance": item.utterance, "start_seconds": item.requested_start_seconds,
            "end_seconds": item.requested_end_seconds} for item in plan.events]
    performance = [{"event_id": item.event_id, "cue": item.cue,
                    "start_seconds": item.requested_start_seconds,
                    "end_seconds": item.requested_end_seconds} for item in plan.performance_events]
    if build_dialogue_plan(plan.global_prompt, raw, plan.total_frames,
                           performance_events=performance) != plan:
        raise ValueError("Temporal Dialogue Plan content or SHA changed")
    return plan


@dataclass(frozen=True)
class WindowDescriptor:
    index: int
    count: int
    start_frame: int
    end_frame: int
    owned_start_frame: int
    audio_start: int
    audio_stop: int
    owned_audio_start: int
    total_frames: int
    audio_length: int
    audio_padding_policy: str = "include_last"


def window_descriptors(segments, audio_bounds, total_frames, audio_length,
                       audio_padding_policy="include_last"):
    """Adapt real executor spans, not a guessed chunk/overlap time formula."""
    _integer(total_frames, "total_frames", 1)
    _integer(audio_length, "audio_length", 1)
    if audio_padding_policy not in {"include_last", "retain_source_tail"}:
        raise ValueError("Unknown actual executor audio padding policy")
    if not segments or len(segments) != len(audio_bounds):
        raise ValueError("Actual video/audio window spans are missing or differ")
    windows, frame_stop, audio_stop, video_stop = [], 0, 0, 0
    for index, (segment, bounds) in enumerate(zip(segments, audio_bounds)):
        if len(segment) != 4 or len(bounds) != 2:
            raise ValueError("Invalid executor window descriptor")
        vs, sf, ve, ef = segment
        astart, astop = bounds
        for value in (*segment, *bounds):
            _integer(value, "window bound")
        if (not vs < ve or not sf < ef <= total_frames or not astart < astop <= audio_length
                or sf > frame_stop or vs > video_stop or astart > audio_stop
                or ef <= frame_stop or ve <= video_stop or astop <= audio_stop
                or (index == 0 and (vs != 0 or sf != 0 or astart != 0))):
            raise ValueError("Windows must advance, overlap or touch, and cover the source without gaps")
        if astart != round(sf * 5 / 3):
            raise ValueError("Audio start differs from the native absolute window clock")
        expected_stop = (audio_length if ef == total_frames and audio_padding_policy == "include_last"
                         else min(audio_length, round(ef * 5 / 3)))
        if astop != expected_stop:
            raise ValueError("Audio stop differs from the native absolute window clock/padding")
        windows.append(WindowDescriptor(index, len(segments), sf, ef, frame_stop,
                                        astart, astop, audio_stop, total_frames, audio_length,
                                        audio_padding_policy))
        frame_stop, audio_stop, video_stop = ef, astop, ve
    expected_final_audio = (audio_length if audio_padding_policy == "include_last"
                            else min(audio_length, round(total_frames * 5 / 3)))
    if frame_stop != total_frames or audio_stop != expected_final_audio:
        raise ValueError("Actual windows do not cover the complete source AV")
    return tuple(windows)


def _validate_window(window):
    if type(window) is not WindowDescriptor:
        raise ValueError("Expected an actual snapped WindowDescriptor")
    for key, value in asdict(window).items():
        if key != "audio_padding_policy":
            _integer(value, key)
    if window.audio_padding_policy not in {"include_last", "retain_source_tail"}:
        raise ValueError("Unknown actual executor audio padding policy")
    if (not 0 <= window.index < window.count
            or not window.start_frame <= window.owned_start_frame < window.end_frame <= window.total_frames
            or not window.audio_start <= window.owned_audio_start < window.audio_stop <= window.audio_length
            or (window.index == 0 and (window.start_frame != 0 or window.owned_start_frame != 0
                                       or window.audio_start != 0 or window.owned_audio_start != 0))):
        raise ValueError("Window ownership bounds are invalid")
    expected_stop = (window.audio_length if window.end_frame == window.total_frames
                     and window.audio_padding_policy == "include_last"
                     else min(window.audio_length, round(window.end_frame * 5 / 3)))
    if (window.audio_start != round(window.start_frame * 5 / 3)
            or window.owned_audio_start != round(window.owned_start_frame * 5 / 3)
            or window.audio_stop != expected_stop
            or ((window.index == window.count - 1) != (window.end_frame == window.total_frames))):
        raise ValueError("Window descriptor native clocks or terminal coverage differ")


@dataclass(frozen=True)
class ProjectedDialogue:
    event_id: str
    state: str
    global_start_frame: int
    global_end_frame: int
    local_start_frame: int
    local_end_frame: int
    prompt_char_start: int
    prompt_char_end: int


@dataclass(frozen=True)
class CompiledWindowText:
    policy: str
    plan_sha256: str
    window: WindowDescriptor
    prompt: str
    dialogue: tuple[ProjectedDialogue, ...]
    omitted_event_ids: tuple[str, ...]
    sha256: str
    performance_event_ids: tuple[str, ...] = ()


def compile_window_text(plan, window):
    """Keep Global verbatim; select speech on writable ownership, even in W0.

    This is a precompilable *text* plan, not permission to reuse an unverified
    previous audio prefix or a hard attention-time isolation guarantee.
    """
    validate_dialogue_plan(plan)
    _validate_window(window)
    if window.total_frames != plan.total_frames:
        raise ValueError("Dialogue Plan and actual source timeline differ")
    prompt, projected, omitted = f"Global scene: {plan.global_prompt}", [], []
    performance_ids = []
    for event in plan.performance_events:
        # Visual context follows the render window, not speech ownership.
        if event.end_frame <= window.start_frame or event.start_frame >= window.end_frame:
            continue
        performance_ids.append(event.event_id)
        ls, le = event.start_frame - window.start_frame, event.end_frame - window.start_frame
        prompt += (f"\nPerformance context {event.event_id}, "
                   f"local interval {ls / NATIVE_FPS:.6f}–{le / NATIVE_FPS:.6f}s: {event.cue}")
    for event in plan.events:
        if event.end_frame <= window.owned_start_frame or event.start_frame >= window.end_frame:
            omitted.append(event.event_id)
            continue
        state = "continuation" if event.start_frame < window.owned_start_frame else "onset"
        ls, le = event.start_frame - window.start_frame, event.end_frame - window.start_frame
        # Preserve the full event duration and native media labels; clipping
        # the utterance or resetting a crossing event to a new onset is wrong.
        line = (f"Dialogue {event.event_id}, speaker {event.speaker}, "
                f"local interval {ls / NATIVE_FPS:.6f}–{le / NATIVE_FPS:.6f}s, "
                f"{state}: <d>{event.utterance}</d>")
        if state == "continuation":
            line += " Continue the already-started line from the accepted audio context; do not restart it."
        prompt += "\n"
        start = len(prompt)
        prompt += line
        projected.append(ProjectedDialogue(event.event_id, state, event.start_frame,
                                            event.end_frame, ls, le, start, len(prompt)))
    draft = CompiledWindowText(POLICY, plan.sha256, window, prompt,
                               tuple(projected), tuple(omitted), "", tuple(performance_ids))
    payload = {key: value for key, value in asdict(draft).items() if key != "sha256"}
    return CompiledWindowText(POLICY, plan.sha256, window, prompt, draft.dialogue,
                              draft.omitted_event_ids, _digest(payload), draft.performance_event_ids)


def validate_compiled_window(plan, compiled):
    if type(compiled) is not CompiledWindowText or compile_window_text(plan, compiled.window) != compiled:
        raise ValueError("Compiled window text or ownership identity changed")


@dataclass(frozen=True)
class DialogueReceipt:
    policy: str
    plan_sha256: str
    text_sha256: str
    window: WindowDescriptor
    published_audio_sha256: str
    previous_receipt_sha256: str | None
    sha256: str


def _validate_receipt(receipt):
    if type(receipt) is not DialogueReceipt or receipt.policy != POLICY:
        raise ValueError("A scoped completed window receipt is required; legacy cache is not ownership proof")
    _validate_window(receipt.window)
    for key in ("plan_sha256", "text_sha256", "published_audio_sha256", "sha256"):
        _sha(getattr(receipt, key), key)
    if receipt.previous_receipt_sha256 is not None:
        _sha(receipt.previous_receipt_sha256, "previous_receipt_sha256")
    payload = {key: value for key, value in asdict(receipt).items() if key != "sha256"}
    if _digest(payload) != receipt.sha256:
        raise ValueError("Scoped completed window receipt content/SHA changed")


def verify_previous_window(plan, window, previous, actual_published_audio_sha256):
    """Caller hashes the real published audio; never infer completion from text."""
    validate_dialogue_plan(plan)
    _validate_window(window)
    if window.total_frames != plan.total_frames:
        raise ValueError("Dialogue Plan and source timeline differ")
    if window.index == 0:
        if previous is not None or actual_published_audio_sha256 is not None:
            raise ValueError("First window must not have an accepted previous audio prefix")
        return
    _validate_receipt(previous)
    _sha(actual_published_audio_sha256, "actual_published_audio_sha256")
    pw = previous.window
    if (previous.plan_sha256 != plan.sha256 or pw.index != window.index - 1
            or pw.count != window.count or pw.total_frames != window.total_frames
            or pw.audio_length != window.audio_length
            or pw.audio_padding_policy != window.audio_padding_policy
            or pw.end_frame != window.owned_start_frame
            or pw.audio_stop != window.owned_audio_start
            or previous.published_audio_sha256 != actual_published_audio_sha256):
        raise ValueError("Previous window policy, order, ownership or actual audio prefix differs")


def bind_published_window(plan, compiled, published_audio_sha256, previous=None,
                          previous_audio_sha256=None):
    """Bind the adapter's real output hash, not a semantic/quality acceptance."""
    validate_compiled_window(plan, compiled)
    verify_previous_window(plan, compiled.window, previous, previous_audio_sha256)
    _sha(published_audio_sha256, "published_audio_sha256")
    draft = DialogueReceipt(POLICY, plan.sha256, compiled.sha256, compiled.window,
                            published_audio_sha256, previous.sha256 if previous else None, "")
    payload = {key: value for key, value in asdict(draft).items() if key != "sha256"}
    return DialogueReceipt(POLICY, plan.sha256, compiled.sha256, compiled.window,
                           published_audio_sha256, draft.previous_receipt_sha256, _digest(payload))
