from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time

import torch

import comfy.model_management
import comfy.sample
import comfy.samplers
import comfy.utils
import latent_preview

from .audio_ops import decode_av_latent, trim_av_output
from .core import FPS, validate_audio
from .long_video import build_long_video_conditioning, patch_long_video_model
from .long_video_delivery import (
    accept_long_video_candidate,
    compose_accepted_long_video,
    load_accepted_context,
    load_delivery_manifest,
    load_long_video_candidate_descriptor,
    long_video_chain_root,
    save_long_video_candidate,
)
from .long_video_orchestration import resolve_long_video_orchestration
from .sampling import setup_dual_clock_sampling


LOOP_SCHEMA = 1
LOOP_FORMAT = "minimax_h3_t8_in_node_loop"
LOOP_STATE_NAME = "in_node_loop_state.json"
LOOP_LOCK_NAME = "in_node_loop.lock"


class _SingleConditionGuider(comfy.samplers.CFGGuider):
    def set_conditioning(self, positive) -> None:
        self.inner_set_conds({"positive": positive})


def _canonical_json(payload: object) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_json(payload: object) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        with temporary.open("r+b") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_loop_state(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"In-node long-video state is corrupt: {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("In-node long-video state root must be an object")
    if int(payload.get("schema", -1)) != LOOP_SCHEMA or payload.get("format") != LOOP_FORMAT:
        raise ValueError("In-node long-video state schema is unsupported")
    return payload


def _tensor_signature(value) -> object:
    """Return a bounded media signature without copying a full video tensor.

    The sampled values are a change detector, not a content-authentication hash. Accepted
    segment files and continuation tensors still use their existing full SHA-256 contracts.
    """
    if not isinstance(value, torch.Tensor):
        return None
    shape = [int(item) for item in value.shape]
    count = int(value.numel())
    sampled_sha256 = ""
    if count:
        sample_count = min(64, count)
        indices = torch.linspace(0, count - 1, sample_count, dtype=torch.int64)
        flat = value.detach().reshape(-1)
        sampled = flat.index_select(0, indices.to(flat.device)).float().cpu().contiguous()
        sampled_sha256 = hashlib.sha256(sampled.numpy().tobytes()).hexdigest()
    return {
        "shape": shape,
        "dtype": str(value.dtype),
        "sample_count": min(64, count),
        "sampled_sha256": sampled_sha256,
    }


def _media_signature(value) -> object:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return _tensor_signature(value)
    if isinstance(value, Mapping):
        if "waveform" in value:
            return {
                "waveform": _tensor_signature(value.get("waveform")),
                "sample_rate": int(value.get("sample_rate", 0)),
            }
        return {
            str(key): _media_signature(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_media_signature(item) for item in value]
    return {"type": type(value).__name__, "value": str(value)}


def _job_contract(
    *,
    chain_id: str,
    total_duration_seconds: float,
    render_window_frames: int,
    context_frames: int,
    global_prompt: str,
    segment_prompts_json: str,
    base_seed: int,
    seed_policy: str,
    steps: int,
    shift_video: float,
    shift_audio: float,
    sampler_name: str,
    scheduler: str,
    width: int,
    height: int,
    task_type: str,
    context_audio: str,
    audio_mode: str,
    audio_denoise_strength: float,
    add_source_as_reference: bool,
    prompt_primary_audio_ordinal: int,
    strict_prompt_tags: bool,
    ref_image_size: str,
    reference_video_policy: str,
    first_frame_reuse: str,
    persistent_identity_strategy: str,
    persistent_identity_interval: int,
    model_id: str,
    drive_audio,
    final_audio,
    first_frame,
    last_frame,
    persistent_identity_image,
    ref_images,
    ref_videos,
    ref_video_audios,
    ref_audios,
) -> dict:
    return {
        "chain_id": str(chain_id),
        "total_duration_seconds": float(total_duration_seconds),
        "render_window_frames": int(render_window_frames),
        "context_frames": int(context_frames),
        "global_prompt": str(global_prompt),
        "segment_prompts_json": str(segment_prompts_json),
        "base_seed": int(base_seed),
        "seed_policy": str(seed_policy),
        "steps": int(steps),
        "shift_video": float(shift_video),
        "shift_audio": float(shift_audio),
        "sampler_name": str(sampler_name),
        "scheduler": str(scheduler),
        "width": int(width),
        "height": int(height),
        "task_type": str(task_type),
        "context_audio": str(context_audio),
        "audio_mode": str(audio_mode),
        "audio_denoise_strength": float(audio_denoise_strength),
        "add_source_as_reference": bool(add_source_as_reference),
        "prompt_primary_audio_ordinal": int(prompt_primary_audio_ordinal),
        "strict_prompt_tags": bool(strict_prompt_tags),
        "ref_image_size": str(ref_image_size),
        "reference_video_policy": str(reference_video_policy),
        "first_frame_reuse": str(first_frame_reuse),
        "persistent_identity_strategy": str(persistent_identity_strategy),
        "persistent_identity_interval": int(persistent_identity_interval),
        "model_id": str(model_id or "unknown"),
        "media": {
            "drive_audio": _media_signature(drive_audio),
            "final_audio": _media_signature(final_audio),
            "first_frame": _media_signature(first_frame),
            "last_frame": _media_signature(last_frame),
            "persistent_identity_image": _media_signature(persistent_identity_image),
            "ref_images": _media_signature(ref_images),
            "ref_videos": _media_signature(ref_videos),
            "ref_video_audios": _media_signature(ref_video_audios),
            "ref_audios": _media_signature(ref_audios),
        },
    }


@contextmanager
def _exclusive_loop_lock(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    path = root / LOOP_LOCK_NAME
    handle = path.open("a+b")
    if path.stat().st_size == 0:
        handle.write(b"0")
        handle.flush()
    handle.seek(0)
    try:
        if os.name == "nt":
            import msvcrt

            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise RuntimeError(
                    "Another in-node long-video runner is already using this chain_id"
                ) from error
        else:
            import fcntl

            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                raise RuntimeError(
                    "Another in-node long-video runner is already using this chain_id"
                ) from error
        yield
    finally:
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _check_interrupted() -> None:
    comfy.model_management.throw_exception_if_processing_interrupted()


def _sample_one_segment(
    model,
    positive,
    av_latent: dict,
    *,
    seed: int,
    steps: int,
    shift_video: float,
    shift_audio: float,
    sampler_name: str,
    scheduler: str,
) -> dict:
    sampled_model, sampler, sigmas = setup_dual_clock_sampling(
        model,
        av_latent,
        steps,
        shift_video,
        shift_audio,
        sampler_name,
        scheduler,
    )
    latent = dict(av_latent)
    latent_image = comfy.sample.fix_empty_latent_channels(
        sampled_model,
        latent["samples"],
        latent.get("downscale_ratio_spacial"),
        latent.get("downscale_ratio_temporal"),
    )
    latent["samples"] = latent_image
    noise = comfy.sample.prepare_noise(
        latent_image,
        int(seed),
        latent.get("batch_index"),
    )
    guider = _SingleConditionGuider(sampled_model)
    guider.set_conditioning(positive)
    preview_callback = latent_preview.prepare_callback(
        sampled_model, sigmas.shape[-1] - 1, {}
    )

    def callback(_step, _x0, _x, _total_steps):
        _check_interrupted()
        preview_callback(_step, _x0, _x, _total_steps)

    samples = guider.sample(
        noise,
        latent_image,
        sampler,
        sigmas,
        denoise_mask=latent.get("noise_mask"),
        callback=callback,
        disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED,
        seed=int(seed),
    )
    # Match ComfyUI SamplerCustomAdvanced exactly: move to the intermediate device
    # without an additional dtype conversion that could change accepted segment bytes.
    samples = samples.to(device=comfy.model_management.intermediate_device())
    output = dict(latent)
    output.pop("downscale_ratio_spacial", None)
    output.pop("downscale_ratio_temporal", None)
    output["samples"] = samples
    return output


def _window_segment_audio(audio, plan, *, name: str):
    """Select the exact render-window audio for one global-timeline segment.

    Continuation segments reconstruct ``context_frames`` before their new timeline range.
    Feeding the whole source to every iteration would incorrectly restart source speech at
    zero, so the controller slices from ``timeline_start - context`` and zero-pads only when
    the connected source does not cover the requested window.
    """
    if audio is None:
        return None
    waveform, sample_rate = validate_audio(audio, name)
    render_samples = round(int(plan.render_frames) / FPS * sample_rate)
    window_start_seconds = float(plan.timeline_start_seconds) - int(plan.context_frames) / FPS
    window_start_sample = round(window_start_seconds * sample_rate)
    source_start = max(0, window_start_sample)
    target_start = max(0, -window_start_sample)
    output = waveform.new_zeros(
        (waveform.shape[0], waveform.shape[1], render_samples)
    )
    writable = min(
        max(0, int(waveform.shape[-1]) - source_start),
        max(0, render_samples - target_start),
    )
    if writable:
        output[..., target_start : target_start + writable] = waveform[
            ..., source_start : source_start + writable
        ]
    return {"waveform": output, "sample_rate": sample_rate}


def _candidate_contract_matches(candidate: Mapping, expected: Mapping) -> bool:
    fields = (
        "chain_id",
        "index",
        "parent_candidate_id",
        "parent_manifest_revision",
        "frame_count",
        "timeline_start_frame",
        "timeline_end_frame",
        "is_final_segment",
        "model_id",
        "sampling_summary",
        "seed",
        "width",
        "height",
    )
    return all(candidate.get(field) == expected.get(field) for field in fields)


def _candidate_base_id(expected: Mapping, contract_sha256: str) -> str:
    payload = {key: expected.get(key) for key in sorted(expected)}
    digest = _sha256_json(payload)[:12]
    return (
        f"loop_s{int(expected['index']):05d}_r"
        f"{int(expected['parent_manifest_revision']):05d}_{contract_sha256[:8]}_{digest}"
    )


def _candidate_descriptor_path(root: Path, index: int, candidate_id: str) -> Path:
    return (
        root
        / "candidates"
        / f"segment_{int(index):05d}"
        / candidate_id
        / "candidate.json"
    )


def _reusable_candidate(root: Path, expected: Mapping, candidate_id: str) -> str | None:
    segment_root = root / "candidates" / f"segment_{int(expected['index']):05d}"
    if not segment_root.is_dir():
        return None
    candidate_dirs = []
    primary = segment_root / candidate_id
    if primary.is_dir():
        candidate_dirs.append(primary)
    retry_prefix = f"{candidate_id}_retry"
    candidate_dirs.extend(
        sorted(
            (
                path
                for path in segment_root.iterdir()
                if path.is_dir()
                and path.name.startswith(retry_prefix)
                and path.name[len(retry_prefix) :].isdigit()
            ),
            key=lambda path: int(path.name[len(retry_prefix) :]),
        )
    )
    for candidate_dir in candidate_dirs:
        descriptor = candidate_dir / "candidate.json"
        if not descriptor.is_file():
            continue
        try:
            candidate, _video_path = load_long_video_candidate_descriptor(str(descriptor))
        except (OSError, ValueError):
            continue
        if _candidate_contract_matches(candidate, expected):
            return str(descriptor)
    return None


def _available_candidate_id(root: Path, index: int, base_id: str) -> str:
    for attempt in range(10000):
        candidate_id = base_id if attempt == 0 else f"{base_id}_retry{attempt:04d}"
        folder = _candidate_descriptor_path(root, index, candidate_id).parent
        if not folder.exists():
            return candidate_id
    raise RuntimeError("In-node long-video candidate retry namespace is exhausted")


def _has_saved_candidate(root: Path) -> bool:
    candidate_root = root / "candidates"
    return candidate_root.is_dir() and next(candidate_root.rglob("candidate.json"), None) is not None


def _release_segment_memory() -> None:
    gc.collect()
    comfy.model_management.soft_empty_cache()


def _state_payload(
    *,
    chain_id: str,
    contract_sha256: str,
    status: str,
    segment_count: int,
    accepted_count: int,
    manifest_revision: int,
    current_segment_index: int | None,
    final_video_path: str = "",
    final_video_sha256: str = "",
    last_error: str = "",
    adopted_existing_manifest: bool = False,
    created_unix: float | None = None,
) -> dict:
    now = time.time()
    return {
        "schema": LOOP_SCHEMA,
        "format": LOOP_FORMAT,
        "chain_id": chain_id,
        "contract_sha256": contract_sha256,
        "status": status,
        "segment_count": int(segment_count),
        "accepted_count": int(accepted_count),
        "manifest_revision": int(manifest_revision),
        "current_segment_index": current_segment_index,
        "final_video_path": str(final_video_path),
        "final_video_sha256": str(final_video_sha256),
        "last_error": str(last_error),
        "adopted_existing_manifest": bool(adopted_existing_manifest),
        "created_unix": float(created_unix if created_unix is not None else now),
        "updated_unix": now,
        "memory_contract": (
            "strictly sequential batch=1; current segment AV latent/decoded media only; "
            "accepted history is file-backed"
        ),
    }


def _existing_complete_output(state: Mapping | None, manifest_revision: int) -> str:
    if not state or state.get("status") != "complete":
        return ""
    if int(state.get("manifest_revision", -1)) != int(manifest_revision):
        return ""
    value = str(state.get("final_video_path", ""))
    expected_hash = str(state.get("final_video_sha256", ""))
    path = Path(value)
    if not value or not expected_hash or not path.is_file():
        return ""
    if _sha256_file(path) != expected_hash:
        return ""
    return str(path)


def run_long_video_in_node_loop(
    model,
    clip,
    video_vae,
    audio_vae,
    *,
    chain_id: str,
    total_duration_seconds: float,
    render_window_frames: int,
    context_frames: int,
    global_prompt: str,
    segment_prompts_json: str,
    base_seed: int,
    seed_policy: str,
    steps: int,
    shift_video: float,
    shift_audio: float,
    sampler_name: str,
    scheduler: str,
    width: int,
    height: int,
    task_type: str,
    context_audio: str,
    audio_mode: str,
    audio_denoise_strength: float,
    add_source_as_reference: bool,
    prompt_primary_audio_ordinal: int,
    strict_prompt_tags: bool,
    ref_image_size: str,
    reference_video_policy: str,
    first_frame_reuse: str,
    persistent_identity_strategy: str,
    persistent_identity_interval: int,
    resume_existing: bool,
    filename_prefix: str,
    audio_seam_policy: str,
    bridge_ms: float,
    bit_depth: int,
    crf: int,
    model_id: str,
    drive_audio=None,
    final_audio=None,
    first_frame=None,
    last_frame=None,
    persistent_identity_image=None,
    ref_images=None,
    ref_videos=None,
    ref_video_audios=None,
    ref_audios=None,
) -> tuple[str, str, int, str, str]:
    if width % 32 or height % 32:
        raise ValueError("MiniMax H3 in-node long-video width and height must be divisible by 32")
    if width * height > 2_088_960:
        raise ValueError("MiniMax H3 in-node long-video canvas exceeds the 1920x1088 2.0MP cap")
    if bit_depth not in {8, 10}:
        raise ValueError("bit_depth must be 8 or 10")
    if not 0 <= crf <= 51:
        raise ValueError("crf must be between 0 and 51")
    if audio_seam_policy not in {"cosine_bridge", "none"}:
        raise ValueError("audio_seam_policy must be cosine_bridge or none")
    if not math.isfinite(float(bridge_ms)) or not 0 <= float(bridge_ms) <= 50:
        raise ValueError("bridge_ms must be between 0 and 50")

    orchestration, manifest = resolve_long_video_orchestration(
        chain_id,
        total_duration_seconds,
        render_window_frames,
        context_frames,
        global_prompt,
        segment_prompts_json,
        base_seed,
        seed_policy,
        steps,
        shift_video,
        shift_audio,
        sampler_name,
        scheduler,
    )
    safe_chain = orchestration.chain_id
    root = long_video_chain_root(safe_chain)
    state_path = root / LOOP_STATE_NAME
    contract = _job_contract(
        chain_id=safe_chain,
        total_duration_seconds=total_duration_seconds,
        render_window_frames=render_window_frames,
        context_frames=context_frames,
        global_prompt=global_prompt,
        segment_prompts_json=segment_prompts_json,
        base_seed=base_seed,
        seed_policy=seed_policy,
        steps=steps,
        shift_video=shift_video,
        shift_audio=shift_audio,
        sampler_name=sampler_name,
        scheduler=scheduler,
        width=width,
        height=height,
        task_type=task_type,
        context_audio=context_audio,
        audio_mode=audio_mode,
        audio_denoise_strength=audio_denoise_strength,
        add_source_as_reference=add_source_as_reference,
        prompt_primary_audio_ordinal=prompt_primary_audio_ordinal,
        strict_prompt_tags=strict_prompt_tags,
        ref_image_size=ref_image_size,
        reference_video_policy=reference_video_policy,
        first_frame_reuse=first_frame_reuse,
        persistent_identity_strategy=persistent_identity_strategy,
        persistent_identity_interval=persistent_identity_interval,
        model_id=model_id,
        drive_audio=drive_audio,
        final_audio=final_audio,
        first_frame=first_frame,
        last_frame=last_frame,
        persistent_identity_image=persistent_identity_image,
        ref_images=ref_images,
        ref_videos=ref_videos,
        ref_video_audios=ref_video_audios,
        ref_audios=ref_audios,
    )
    contract_sha256 = _sha256_json(contract)
    segment_count = len(orchestration.segments)

    with _exclusive_loop_lock(root):
        state = _load_loop_state(state_path)
        if state is not None and state.get("chain_id") != safe_chain:
            raise ValueError("In-node long-video state chain_id is inconsistent")
        if not resume_existing and (
            state is not None
            or orchestration.accepted_count
            or _has_saved_candidate(root)
        ):
            raise ValueError(
                "resume_existing is false but this chain_id already contains saved loop "
                "state or candidates; use a new chain_id or enable resume_existing"
            )
        if state is not None and state.get("contract_sha256") != contract_sha256:
            accepted = int(orchestration.accepted_count)
            if accepted or int(state.get("accepted_count", 0)):
                raise ValueError(
                    "This chain_id already contains accepted segments from a different in-node "
                    "loop contract. Restore the original settings or use a new chain_id."
                )
            state = None
        adopted = state is None and orchestration.accepted_count > 0
        created_unix = state.get("created_unix") if state else None
        existing_output = _existing_complete_output(state, orchestration.manifest_revision)
        if orchestration.complete and existing_output:
            report = dict(state)
            report.update(
                {
                    "resume_action": "returned_verified_existing_final",
                    "state_path": str(state_path),
                    "manifest_path": str(root / "manifest.json"),
                }
            )
            return (
                existing_output,
                str(root / "manifest.json"),
                segment_count,
                "complete",
                json.dumps(report, ensure_ascii=False, indent=2),
            )

        current_index = orchestration.accepted_count if not orchestration.complete else None
        state = _state_payload(
            chain_id=safe_chain,
            contract_sha256=contract_sha256,
            status="running" if current_index is not None else "composing",
            segment_count=segment_count,
            accepted_count=orchestration.accepted_count,
            manifest_revision=orchestration.manifest_revision,
            current_segment_index=current_index,
            adopted_existing_manifest=adopted,
            created_unix=created_unix,
        )
        _atomic_write_json(state_path, state)

        try:
            long_video_model = patch_long_video_model(model)
            for segment in orchestration.segments[orchestration.accepted_count :]:
                _check_interrupted()
                manifest_now, _manifest_source = load_delivery_manifest(
                    safe_chain, allow_new=True
                )
                if len(manifest_now["segments"]) != segment.index:
                    raise RuntimeError(
                        "Accepted manifest changed while the in-node loop was running; "
                        "the current segment was not generated"
                    )
                context, _has_context, parent_candidate_id, parent_revision, _context_report = (
                    load_accepted_context(safe_chain, segment.index)
                )
                plan = segment.plan
                expected = {
                    "chain_id": safe_chain,
                    "index": segment.index,
                    "parent_candidate_id": parent_candidate_id,
                    "parent_manifest_revision": parent_revision,
                    "frame_count": plan.final_frame_count,
                    "timeline_start_frame": round(plan.timeline_start_seconds * 24),
                    "timeline_end_frame": round(plan.timeline_end_seconds * 24),
                    "is_final_segment": plan.is_final_segment,
                    "model_id": str(model_id or "unknown"),
                    "sampling_summary": orchestration.sampling_summary,
                    "prompt": segment.prompt,
                    "seed": segment.seed,
                    "width": int(width),
                    "height": int(height),
                }
                base_candidate_id = _candidate_base_id(expected, contract_sha256)
                candidate_json_path = _reusable_candidate(
                    root, expected, base_candidate_id
                )
                if candidate_json_path is None:
                    candidate_id = _available_candidate_id(
                        root, segment.index, base_candidate_id
                    )
                    try:
                        segment_drive_audio = _window_segment_audio(
                            drive_audio, plan, name="drive_audio"
                        )
                        segment_final_audio = _window_segment_audio(
                            final_audio, plan, name="final_audio"
                        )
                        (
                            positive,
                            av_latent,
                            mux_audio,
                            conditioned_prompt,
                            _media_map_json,
                            _conditioning_report,
                        ) = build_long_video_conditioning(
                            clip,
                            video_vae,
                            audio_vae,
                            context,
                            segment.index,
                            plan.context_frames,
                            context_audio,
                            segment.prompt,
                            width,
                            height,
                            plan.render_frames,
                            task_type,
                            audio_mode,
                            audio_denoise_strength,
                            add_source_as_reference,
                            prompt_primary_audio_ordinal,
                            strict_prompt_tags,
                            ref_image_size,
                            reference_video_policy,
                            segment_drive_audio,
                            segment_final_audio,
                            first_frame,
                            last_frame if plan.is_final_segment else None,
                            ref_images,
                            ref_videos,
                            ref_video_audios,
                            ref_audios,
                            first_frame_reuse,
                            persistent_identity_image,
                            persistent_identity_strategy,
                            persistent_identity_interval,
                        )
                        sampled = _sample_one_segment(
                            long_video_model,
                            positive,
                            av_latent,
                            seed=segment.seed,
                            steps=orchestration.steps,
                            shift_video=orchestration.shift_video,
                            shift_audio=orchestration.shift_audio,
                            sampler_name=orchestration.sampler_name,
                            scheduler=orchestration.scheduler,
                        )
                        frames, generated_audio, _video_latent, _audio_latent = (
                            decode_av_latent(sampled, video_vae, audio_vae)
                        )
                        delivery_audio = mux_audio if mux_audio is not None else generated_audio
                        trimmed_frames, trimmed_audio, _trim_report = trim_av_output(
                            frames,
                            plan.trim_start_seconds,
                            plan.final_duration_seconds,
                            delivery_audio,
                            24.0,
                        )
                        if trimmed_audio is None:
                            raise RuntimeError(
                                "In-node long-video segment has no audio delivery value"
                            )
                        candidate_json_path, _candidate_video, _save_report = (
                            save_long_video_candidate(
                                trimmed_frames,
                                trimmed_audio,
                                sampled,
                                safe_chain,
                                segment.index,
                                plan.timeline_start_seconds,
                                plan.save_context,
                                parent_candidate_id,
                                parent_revision,
                                candidate_id,
                                model_id,
                                orchestration.sampling_summary,
                                conditioned_prompt,
                                segment.seed,
                                24,
                                bit_depth,
                                crf,
                            )
                        )
                    finally:
                        _release_segment_memory()

                _check_interrupted()
                _accepted_video, accepted, manifest_path, _accept_report = (
                    accept_long_video_candidate(
                        candidate_json_path,
                        True,
                        "reject_existing",
                        True,
                    )
                )
                if not accepted:
                    raise RuntimeError(
                        f"In-node long-video segment {segment.index} was not accepted"
                    )
                manifest_now, _manifest_source = load_delivery_manifest(safe_chain)
                state = _state_payload(
                    chain_id=safe_chain,
                    contract_sha256=contract_sha256,
                    status=(
                        "composing"
                        if len(manifest_now["segments"]) == segment_count
                        else "running"
                    ),
                    segment_count=segment_count,
                    accepted_count=len(manifest_now["segments"]),
                    manifest_revision=int(manifest_now["revision"]),
                    current_segment_index=(
                        None
                        if len(manifest_now["segments"]) == segment_count
                        else len(manifest_now["segments"])
                    ),
                    adopted_existing_manifest=adopted,
                    created_unix=created_unix,
                )
                _atomic_write_json(state_path, state)

            manifest_now, _manifest_source = load_delivery_manifest(safe_chain)
            if len(manifest_now["segments"]) != segment_count:
                raise RuntimeError("In-node long-video loop ended before all segments were accepted")
            final_video_path, compose_report_json = compose_accepted_long_video(
                safe_chain,
                filename_prefix,
                True,
                audio_seam_policy,
                bridge_ms,
                crf,
            )
            compose_report = json.loads(compose_report_json)
            state = _state_payload(
                chain_id=safe_chain,
                contract_sha256=contract_sha256,
                status="complete",
                segment_count=segment_count,
                accepted_count=segment_count,
                manifest_revision=int(manifest_now["revision"]),
                current_segment_index=None,
                final_video_path=final_video_path,
                final_video_sha256=str(compose_report["output_sha256"]),
                adopted_existing_manifest=adopted,
                created_unix=created_unix,
            )
            _atomic_write_json(state_path, state)
            report = {
                **state,
                "state_path": str(state_path),
                "manifest_path": str(root / "manifest.json"),
                "sampling_summary": orchestration.sampling_summary,
                "resume_action": (
                    "adopted_existing_then_completed"
                    if adopted
                    else "generated_or_resumed_then_completed"
                ),
                "compose": compose_report,
                "media_contract_note": (
                    "connected media use bounded sampled change signatures; accepted files and "
                    "continuation tensors retain full SHA-256 validation"
                ),
                "timeline_audio_note": (
                    "connected drive/final audio is windowed on the global timeline for every "
                    "segment; non-final segments do not receive the global last_frame"
                ),
            }
            return (
                final_video_path,
                str(root / "manifest.json"),
                segment_count,
                "complete",
                json.dumps(report, ensure_ascii=False, indent=2),
            )
        except BaseException as error:
            try:
                manifest_now, _manifest_source = load_delivery_manifest(
                    safe_chain, allow_new=True
                )
                accepted_count = len(manifest_now["segments"])
                revision = int(manifest_now["revision"])
            except BaseException:
                accepted_count = orchestration.accepted_count
                revision = orchestration.manifest_revision
            failed_state = _state_payload(
                chain_id=safe_chain,
                contract_sha256=contract_sha256,
                status=(
                    "interrupted"
                    if isinstance(error, comfy.model_management.InterruptProcessingException)
                    else "failed"
                ),
                segment_count=segment_count,
                accepted_count=accepted_count,
                manifest_revision=revision,
                current_segment_index=(
                    accepted_count if accepted_count < segment_count else None
                ),
                last_error=f"{type(error).__name__}: {error}",
                adopted_existing_manifest=adopted,
                created_unix=created_unix,
            )
            try:
                _atomic_write_json(state_path, failed_state)
            except OSError as state_error:
                add_note = getattr(error, "add_note", None)
                if callable(add_note):
                    add_note(f"Could not persist in-node loop failure state: {state_error}")
            raise
        finally:
            _release_segment_memory()
