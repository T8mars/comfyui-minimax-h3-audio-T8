from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math

from .core import FPS, MAX_TRAINED_FRAMES, MIN_TRAINED_FRAMES, align_frame_count
from .long_video import CONTEXT_FRAME_STEPS, LongVideoPlan, make_long_video_plan, sanitize_chain_id
from .long_video_delivery import load_delivery_manifest


ORCHESTRATION_SCHEMA = 1
MAX_UINT64 = 0xFFFFFFFFFFFFFFFF


@dataclass(frozen=True)
class OrchestratedSegment:
    index: int
    prompt: str
    seed: int
    note: str
    plan: LongVideoPlan

    def as_report(self) -> dict:
        return {
            "index": self.index,
            "prompt": self.prompt,
            "seed": self.seed,
            "note": self.note,
            **asdict(self.plan),
        }


@dataclass(frozen=True)
class OrchestrationResult:
    chain_id: str
    requested_total_duration_seconds: float
    total_frame_count: int
    quantized_total_duration_seconds: float
    render_window_frames: int
    context_frames: int
    seed_policy: str
    steps: int
    shift_video: float
    shift_audio: float
    sampler_name: str
    scheduler: str
    sampling_summary: str
    segments: tuple[OrchestratedSegment, ...]
    accepted_count: int
    manifest_revision: int
    manifest_source: str
    warnings: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return self.accepted_count == len(self.segments)

    @property
    def next_segment(self) -> OrchestratedSegment | None:
        if self.complete:
            return None
        return self.segments[self.accepted_count]

    @property
    def progress(self) -> float:
        return self.accepted_count / len(self.segments)

    def report(self) -> str:
        return json.dumps(
            {
                "schema": ORCHESTRATION_SCHEMA,
                "chain_id": self.chain_id,
                "requested_total_duration_seconds": self.requested_total_duration_seconds,
                "total_frame_count": self.total_frame_count,
                "quantized_total_duration_seconds": self.quantized_total_duration_seconds,
                "render_window_frames": self.render_window_frames,
                "context_frames": self.context_frames,
                "segment_count": len(self.segments),
                "accepted_count": self.accepted_count,
                "next_segment_index": (
                    self.next_segment.index if self.next_segment is not None else None
                ),
                "complete": self.complete,
                "progress": self.progress,
                "manifest_revision": self.manifest_revision,
                "manifest_source": self.manifest_source,
                "seed_policy": self.seed_policy,
                "steps": self.steps,
                "shift_video": self.shift_video,
                "shift_audio": self.shift_audio,
                "sampler_name": self.sampler_name,
                "scheduler": self.scheduler,
                "sampling_summary": self.sampling_summary,
                "warnings": list(self.warnings),
            },
            ensure_ascii=False,
            indent=2,
        )

    def plan_json(self, manifest: dict | None = None) -> str:
        accepted = manifest.get("segments", []) if manifest is not None else []
        items = []
        for segment in self.segments:
            item = segment.as_report()
            if segment.index < self.accepted_count:
                entry = accepted[segment.index]
                item.update(
                    {
                        "status": "accepted",
                        "accepted_candidate_id": entry.get("candidate_id", ""),
                        "accepted_video_path": entry.get("video_path", ""),
                        "actual_prompt": entry.get("prompt", ""),
                        "actual_seed": entry.get("seed"),
                    }
                )
            elif segment.index == self.accepted_count:
                item["status"] = "next" if not self.complete else "complete"
            else:
                item["status"] = "pending"
            items.append(item)
        payload = {
            "schema": ORCHESTRATION_SCHEMA,
            "chain_id": self.chain_id,
            "requested_total_duration_seconds": self.requested_total_duration_seconds,
            "total_frame_count": self.total_frame_count,
            "quantized_total_duration_seconds": self.quantized_total_duration_seconds,
            "render_window_frames": self.render_window_frames,
            "context_frames": self.context_frames,
            "steps": self.steps,
            "shift_video": self.shift_video,
            "shift_audio": self.shift_audio,
            "sampler_name": self.sampler_name,
            "scheduler": self.scheduler,
            "sampling_summary": self.sampling_summary,
            "segment_count": len(items),
            "accepted_count": self.accepted_count,
            "complete": self.complete,
            "manifest_revision": self.manifest_revision,
            "segments": items,
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)


def _derived_seed(chain_id: str, base_seed: int, index: int, policy: str) -> int:
    base_seed = int(base_seed)
    if not 0 <= base_seed <= MAX_UINT64:
        raise ValueError("base_seed must be between 0 and 2^64-1")
    if policy == "fixed":
        return base_seed
    if policy == "increment":
        return (base_seed + index) & MAX_UINT64
    if policy == "hash_chain_segment":
        value = f"{sanitize_chain_id(chain_id)}\0{base_seed}\0{index}".encode("utf-8")
        return int.from_bytes(hashlib.sha256(value).digest()[:8], "little")
    raise ValueError("seed_policy must be fixed, increment, or hash_chain_segment")


def _sampling_identity(
    steps: int,
    shift_video: float,
    shift_audio: float,
    sampler_name: str,
    scheduler: str,
) -> tuple[int, float, float, str, str, str]:
    steps = int(steps)
    shift_video = float(shift_video)
    shift_audio = float(shift_audio)
    sampler_name = str(sampler_name or "").strip()
    scheduler = str(scheduler or "").strip()
    if not 1 <= steps <= 1000:
        raise ValueError("steps must be between 1 and 1000")
    if not math.isfinite(shift_video) or shift_video <= 0:
        raise ValueError("shift_video must be a finite positive value")
    if not math.isfinite(shift_audio) or shift_audio <= 0:
        raise ValueError("shift_audio must be a finite positive value")
    if not sampler_name:
        raise ValueError("sampler_name cannot be empty")
    if not scheduler:
        raise ValueError("scheduler cannot be empty")
    summary = (
        f"{steps}-step {sampler_name}/{scheduler} "
        f"shift{shift_video:g}/{shift_audio:g}"
    )
    return steps, shift_video, shift_audio, sampler_name, scheduler, summary


def _segment_overrides(value: str) -> dict[int, dict]:
    text = str(value or "").strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"segment_prompts_json is invalid JSON: {error}") from error
    if isinstance(payload, dict) and "segments" in payload:
        payload = payload["segments"]
    if isinstance(payload, list):
        source = enumerate(payload)
    elif isinstance(payload, dict):
        source = payload.items()
    else:
        raise ValueError("segment_prompts_json must be a list, index object, or {segments: ...}")

    overrides: dict[int, dict] = {}
    for raw_index, raw_item in source:
        try:
            index = int(raw_index)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Segment override index is not an integer: {raw_index!r}") from error
        if index < 0:
            raise ValueError("Segment override indices cannot be negative")
        if isinstance(raw_item, str):
            item = {"prompt": raw_item}
        elif isinstance(raw_item, dict):
            item = dict(raw_item)
        elif raw_item is None:
            item = {}
        else:
            raise ValueError(f"Segment override {index} must be a string or object")
        if "prompt" in item and not isinstance(item["prompt"], str):
            raise ValueError(f"Segment override {index} prompt must be a string")
        if "note" in item and not isinstance(item["note"], str):
            raise ValueError(f"Segment override {index} note must be a string")
        if "seed" in item:
            seed = int(item["seed"])
            if not 0 <= seed <= MAX_UINT64:
                raise ValueError(f"Segment override {index} seed must be between 0 and 2^64-1")
            item["seed"] = seed
        overrides[index] = item
    return overrides


def build_long_video_chain_plan(
    chain_id: str,
    total_duration_seconds: float,
    render_window_frames: int = MIN_TRAINED_FRAMES,
    context_frames: int = 22,
    global_prompt: str = "",
    segment_prompts_json: str = "",
    base_seed: int = 0,
    seed_policy: str = "increment",
) -> tuple[OrchestratedSegment, ...]:
    safe_chain = sanitize_chain_id(chain_id)
    total_duration_seconds = float(total_duration_seconds)
    if not math.isfinite(total_duration_seconds) or total_duration_seconds <= 0:
        raise ValueError("total_duration_seconds must be a finite positive value")
    total_frames = max(1, round(total_duration_seconds * FPS))

    render_window_frames = int(render_window_frames)
    if not MIN_TRAINED_FRAMES <= render_window_frames <= MAX_TRAINED_FRAMES:
        raise ValueError(
            f"render_window_frames must stay in the current H3 range "
            f"{MIN_TRAINED_FRAMES}..{MAX_TRAINED_FRAMES}"
        )
    if align_frame_count(render_window_frames) != render_window_frames:
        raise ValueError("render_window_frames must be on the MiniMax H3 17n+5 grid")
    context_frames = int(context_frames)
    if context_frames not in CONTEXT_FRAME_STEPS:
        raise ValueError("context_frames must be 5, 22, or 39")
    if context_frames >= render_window_frames:
        raise ValueError("context_frames must be shorter than render_window_frames")

    overrides = _segment_overrides(segment_prompts_json)
    segments: list[OrchestratedSegment] = []
    timeline_frame = 0
    index = 0
    while timeline_frame < total_frames:
        capacity = render_window_frames if index == 0 else render_window_frames - context_frames
        remaining = total_frames - timeline_frame
        new_frames = min(capacity, remaining)
        is_final = new_frames == remaining
        plan = make_long_video_plan(
            safe_chain,
            index,
            new_frames / FPS,
            context_frames,
            render_window_frames,
            timeline_frame / FPS,
            is_final,
        )
        if plan.render_frames != render_window_frames:
            raise RuntimeError(
                "Chain planner exceeded its fixed render window; this would invalidate the "
                "bounded-memory contract"
            )
        if plan.final_frame_count != new_frames:
            raise RuntimeError("Chain planner produced an unexpected effective segment length")
        override = overrides.get(index, {})
        prompt = str(override.get("prompt", "") or global_prompt)
        note = str(override.get("note", ""))
        seed = int(
            override.get("seed", _derived_seed(safe_chain, base_seed, index, seed_policy))
        )
        segments.append(OrchestratedSegment(index, prompt, seed, note, plan))
        timeline_frame += new_frames
        index += 1
        if index > 10000:
            raise ValueError("Long-video chain would exceed 10,000 segments")

    unused = sorted(set(overrides) - set(range(len(segments))))
    if unused:
        raise ValueError(
            "segment_prompts_json contains indices outside this chain: "
            + ", ".join(map(str, unused))
        )
    if timeline_frame != total_frames or not segments[-1].plan.is_final_segment:
        raise RuntimeError("Long-video chain planning did not terminate on the exact final frame")
    return tuple(segments)


def _validate_manifest_against_plan(
    manifest: dict,
    segments: tuple[OrchestratedSegment, ...],
    sampling_summary: str,
) -> None:
    accepted = manifest["segments"]
    if len(accepted) > len(segments):
        raise ValueError(
            "Accepted manifest is longer than the requested total duration; use the original "
            "settings or a new chain_id"
        )
    fields = (
        "frame_count", "timeline_start_frame", "timeline_end_frame", "is_final_segment"
    )
    for index, entry in enumerate(accepted):
        planned = segments[index].plan
        expected = {
            "frame_count": planned.final_frame_count,
            "timeline_start_frame": round(planned.timeline_start_seconds * FPS),
            "timeline_end_frame": round(planned.timeline_end_seconds * FPS),
            "is_final_segment": planned.is_final_segment,
        }
        mismatches = [field for field in fields if entry.get(field) != expected[field]]
        if int(entry.get("fps", -1)) != FPS:
            mismatches.append("fps")
        if entry.get("sampling_summary") != sampling_summary:
            mismatches.append("sampling_summary")
        if mismatches:
            raise ValueError(
                f"Accepted segment {index} conflicts with this total-duration plan: "
                + ", ".join(mismatches)
                + ". Restore the original settings or use a new chain_id."
            )


def resolve_long_video_orchestration(
    chain_id: str,
    total_duration_seconds: float,
    render_window_frames: int = MIN_TRAINED_FRAMES,
    context_frames: int = 22,
    global_prompt: str = "",
    segment_prompts_json: str = "",
    base_seed: int = 0,
    seed_policy: str = "increment",
    steps: int = 4,
    shift_video: float = 12.0,
    shift_audio: float = 3.0,
    sampler_name: str = "dual_clock_euler",
    scheduler: str = "native_flow",
) -> tuple[OrchestrationResult, dict | None]:
    safe_chain = sanitize_chain_id(chain_id)
    segments = build_long_video_chain_plan(
        safe_chain,
        total_duration_seconds,
        render_window_frames,
        context_frames,
        global_prompt,
        segment_prompts_json,
        base_seed,
        seed_policy,
    )
    (
        steps,
        shift_video,
        shift_audio,
        sampler_name,
        scheduler,
        sampling_summary,
    ) = _sampling_identity(steps, shift_video, shift_audio, sampler_name, scheduler)
    warnings = []
    try:
        manifest, manifest_source = load_delivery_manifest(safe_chain)
        _validate_manifest_against_plan(manifest, segments, sampling_summary)
        accepted_count = len(manifest["segments"])
        manifest_revision = int(manifest["revision"])
        if manifest_source == "backup":
            warnings.append(
                "The primary manifest was invalid; orchestration resumed from the last valid backup."
            )
    except FileNotFoundError:
        manifest = None
        manifest_source = "new_chain"
        accepted_count = 0
        manifest_revision = 0

    requested = float(total_duration_seconds)
    total_frame_count = sum(segment.plan.final_frame_count for segment in segments)
    quantized = total_frame_count / FPS
    if not math.isclose(requested, quantized, abs_tol=1e-9):
        warnings.append(
            f"Requested duration was quantized to the nearest 24fps frame: {quantized:.6f}s."
        )
    result = OrchestrationResult(
        safe_chain,
        requested,
        total_frame_count,
        quantized,
        int(render_window_frames),
        int(context_frames),
        str(seed_policy),
        steps,
        shift_video,
        shift_audio,
        sampler_name,
        scheduler,
        sampling_summary,
        segments,
        accepted_count,
        manifest_revision,
        manifest_source,
        tuple(warnings),
    )
    return result, manifest
