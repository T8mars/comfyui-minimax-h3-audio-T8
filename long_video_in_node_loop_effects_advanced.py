from __future__ import annotations

from collections.abc import Mapping
import json
import math
from pathlib import Path

import torch

import comfy.model_management
import comfy.sample
import comfy.utils
import latent_preview

from .audio_ops import decode_av_latent, trim_av_output
from .enhance_a_video_advanced import (
    build_eav_long_video_model,
    build_eav_prompt_relay_long_video_model,
    finalize_eav_runtime,
)
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
from .long_video_in_node_loop_advanced import (
    _SingleConditionGuider,
    _atomic_write_json,
    _available_candidate_id,
    _candidate_base_id,
    _check_interrupted,
    _existing_complete_output,
    _exclusive_loop_lock,
    _has_saved_candidate,
    _job_contract,
    _release_segment_memory,
    _reusable_candidate,
    _sha256_json,
    _state_payload,
    _window_segment_audio,
)
from .long_video_orchestration import resolve_long_video_orchestration
from .prompt_relay_advanced import (
    EXECUTION_MODES as PROMPT_RELAY_EXECUTION_MODES,
    _validate_plan as validate_prompt_relay_plan,
)
from .prompt_relay_long_video_advanced import (
    build_prompt_relay_long_video_conditioning,
    project_prompt_relay_plan_to_long_video_window,
)
from .sampling import setup_dual_clock_sampling


EFFECTS_LOOP_SCHEMA = 1
EFFECTS_LOOP_FORMAT = "minimax_h3_t8_in_node_loop_effects"
EFFECTS_LOOP_STATE_NAME = "in_node_loop_effects_state.json"
EFFECTS_AUDIT_SCHEMA = 1
EFFECTS_AUDIT_FORMAT = "minimax_h3_t8_in_node_loop_effects_segment"
EFFECTS_AUDIT_NAME = "effects_audit.json"
PROMPT_RELAY_MODES = ("disabled", *PROMPT_RELAY_EXECUTION_MODES)


class _ResourceHeadroomError(RuntimeError):
    pass


def _load_effects_state(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"In-node effects state is corrupt: {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("In-node effects state root must be an object")
    if (
        int(payload.get("schema", -1)) != EFFECTS_LOOP_SCHEMA
        or payload.get("format") != EFFECTS_LOOP_FORMAT
    ):
        raise ValueError("In-node effects state schema is unsupported")
    return payload


def _effects_state_payload(**kwargs) -> dict:
    payload = _state_payload(**kwargs)
    payload.update(
        {
            "schema": EFFECTS_LOOP_SCHEMA,
            "format": EFFECTS_LOOP_FORMAT,
            "memory_contract": (
                "strictly sequential batch=1; one segment-local Prompt Relay projection and "
                "one fresh EAV runtime; accepted history is file-backed"
            ),
        }
    )
    return payload


def _resource_snapshot(minimum_free_vram_mib: int) -> dict:
    snapshot = {
        "minimum_free_vram_mib": int(minimum_free_vram_mib),
        "telemetry_available": False,
        "device": "unknown",
        "free_mib": None,
        "passed": True,
    }
    if not torch.cuda.is_available():
        snapshot["device"] = "cpu_or_non_cuda"
        return snapshot
    try:
        device = comfy.model_management.get_torch_device()
        free_mib = float(comfy.model_management.get_free_memory(device)) / (1024**2)
    except Exception as error:
        snapshot["telemetry_error"] = f"{type(error).__name__}: {error}"
        return snapshot
    snapshot.update(
        {
            "telemetry_available": True,
            "device": str(device),
            "free_mib": free_mib,
            "passed": free_mib >= float(minimum_free_vram_mib),
        }
    )
    if not snapshot["passed"]:
        raise _ResourceHeadroomError(
            "MiniMax H3 in-node effects loop paused before the next segment: "
            f"{free_mib:.1f} MiB free VRAM is below the configured "
            f"{int(minimum_free_vram_mib)} MiB start floor"
        )
    return snapshot


def _validate_effect_modes(
    *,
    prompt_relay_mode: str,
    prompt_relay_plan,
    eav_mode: str,
    steps: int,
    sampler_name: str,
    scheduler: str,
    global_prompt: str,
    segment_prompts_json: str,
) -> tuple[dict | None, str]:
    prompt_relay_mode = str(prompt_relay_mode)
    eav_mode = str(eav_mode)
    if prompt_relay_mode not in PROMPT_RELAY_MODES:
        raise ValueError(f"Unknown in-node Prompt Relay mode {prompt_relay_mode!r}")
    if eav_mode not in {"disabled", "report_only", "apply_exp"}:
        raise ValueError(f"Unknown in-node Enhance-A-Video mode {eav_mode!r}")
    relay_plan = None
    resolved_prompt = str(global_prompt)
    if prompt_relay_mode != "disabled":
        if prompt_relay_plan is None:
            raise ValueError(
                "Prompt Relay mode requires a connected global H3 Prompt Relay Plan"
            )
        relay_plan = validate_prompt_relay_plan(prompt_relay_plan)
        if str(segment_prompts_json).strip():
            raise ValueError(
                "Prompt Relay owns the global text timeline; clear segment_prompts_json "
                "instead of overriding prompts per segment"
            )
        plan_prompt = str(relay_plan["global_prompt"])
        if str(global_prompt).strip() and str(global_prompt).strip() != plan_prompt.strip():
            raise ValueError(
                "global_prompt differs from the connected Prompt Relay Plan; clear it or "
                "use the same global text"
            )
        resolved_prompt = plan_prompt
        if int(steps) not in {8, 20}:
            raise ValueError(
                "In-node Prompt Relay currently admits the reviewed 8-step or Stock20 routes"
            )
        if sampler_name != "dual_clock_euler" or scheduler != "native_flow":
            raise ValueError(
                "In-node Prompt Relay currently requires dual_clock_euler/native_flow"
            )
    elif prompt_relay_plan is not None:
        raise ValueError(
            "A Prompt Relay Plan is connected while prompt_relay_mode is disabled; "
            "disconnect it or select report_only/apply_exp"
        )
    if eav_mode != "disabled":
        if int(steps) != 20:
            raise ValueError("In-node Enhance-A-Video currently requires Stock20")
        if sampler_name != "dual_clock_euler" or scheduler != "native_flow":
            raise ValueError(
                "In-node Enhance-A-Video currently requires dual_clock_euler/native_flow"
            )
    return relay_plan, resolved_prompt


def _sample_prepared_segment(
    model,
    positive,
    av_latent: dict,
    *,
    sampler,
    sigmas: torch.Tensor,
    seed: int,
) -> dict:
    latent = dict(av_latent)
    latent_image = comfy.sample.fix_empty_latent_channels(
        model,
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
    guider = _SingleConditionGuider(model)
    guider.set_conditioning(positive)
    preview_callback = latent_preview.prepare_callback(
        model, sigmas.shape[-1] - 1, {}
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
    samples = samples.to(device=comfy.model_management.intermediate_device())
    output = dict(latent)
    output.pop("downscale_ratio_spacial", None)
    output.pop("downscale_ratio_temporal", None)
    output["samples"] = samples
    return output


def _audit_sidecar_path(candidate_json_path: str | Path) -> Path:
    return Path(candidate_json_path).resolve().parent / EFFECTS_AUDIT_NAME


def _write_effects_audit(candidate_json_path: str, payload: Mapping) -> dict:
    unsigned = dict(payload)
    unsigned.update(
        {
            "schema": EFFECTS_AUDIT_SCHEMA,
            "format": EFFECTS_AUDIT_FORMAT,
        }
    )
    unsigned.pop("audit_sha256", None)
    unsigned["audit_sha256"] = _sha256_json(unsigned)
    _atomic_write_json(_audit_sidecar_path(candidate_json_path), unsigned)
    return unsigned


def _load_effects_audit(
    candidate_json_path: str | Path,
    *,
    contract_sha256: str,
    segment_index: int,
    candidate_id: str,
) -> dict:
    path = _audit_sidecar_path(candidate_json_path)
    if not path.is_file():
        raise ValueError(
            "In-node effects candidate has no verified effects audit sidecar; "
            "use a new chain_id or regenerate this segment"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"In-node effects audit is corrupt: {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("In-node effects audit root must be an object")
    claimed = str(payload.get("audit_sha256", ""))
    unsigned = dict(payload)
    unsigned.pop("audit_sha256", None)
    if claimed != _sha256_json(unsigned):
        raise ValueError("In-node effects audit SHA-256 is invalid")
    expected = {
        "schema": EFFECTS_AUDIT_SCHEMA,
        "format": EFFECTS_AUDIT_FORMAT,
        "contract_sha256": str(contract_sha256),
        "segment_index": int(segment_index),
        "candidate_id": str(candidate_id),
    }
    mismatches = [key for key, value in expected.items() if payload.get(key) != value]
    if mismatches:
        raise ValueError(
            "In-node effects audit does not match this job: " + ", ".join(mismatches)
        )
    return payload


def _accepted_effect_audits(root: Path, manifest: Mapping, contract_sha256: str) -> list[dict]:
    audits = []
    for segment in manifest.get("segments", []):
        index = int(segment["index"])
        candidate_id = str(segment["candidate_id"])
        candidate_json = (
            root
            / "candidates"
            / f"segment_{index:05d}"
            / candidate_id
            / "candidate.json"
        )
        candidate, _video = load_long_video_candidate_descriptor(str(candidate_json))
        if candidate.get("video_sha256") != segment.get("video_sha256"):
            raise ValueError(
                f"Accepted segment {index} no longer matches its effects candidate"
            )
        audits.append(
            _load_effects_audit(
                candidate_json,
                contract_sha256=contract_sha256,
                segment_index=index,
                candidate_id=candidate_id,
            )
        )
    return audits


def _effects_summary(
    *,
    base_summary: str,
    prompt_relay_mode: str,
    eav_mode: str,
) -> str:
    return (
        f"{base_summary}; in_node_effects_v1 prompt_relay={prompt_relay_mode} "
        f"eav={eav_mode}"
    )


def _json_object(value) -> dict:
    if isinstance(value, Mapping):
        return dict(value)
    parsed = json.loads(str(value))
    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object report")
    return parsed


def run_long_video_in_node_loop_effects(
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
    prompt_relay_mode: str,
    query_chunk_rows: int,
    eav_mode: str,
    eav_tau: float,
    eav_start_video_progress: float,
    eav_end_video_progress: float,
    eav_max_workspace_mib: int,
    eav_g_hard_limit: float,
    minimum_free_vram_mib: int,
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
    prompt_relay_plan=None,
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
        raise ValueError("MiniMax H3 in-node effects width and height must be divisible by 32")
    if width * height > 2_088_960:
        raise ValueError("MiniMax H3 in-node effects canvas exceeds 1920x1088 / 2.0MP")
    if bit_depth not in {8, 10}:
        raise ValueError("bit_depth must be 8 or 10")
    if not 0 <= crf <= 51:
        raise ValueError("crf must be between 0 and 51")
    if audio_seam_policy not in {"cosine_bridge", "none"}:
        raise ValueError("audio_seam_policy must be cosine_bridge or none")
    if not math.isfinite(float(bridge_ms)) or not 0 <= float(bridge_ms) <= 50:
        raise ValueError("bridge_ms must be between 0 and 50")
    if not 0 <= int(minimum_free_vram_mib) <= 65536:
        raise ValueError("minimum_free_vram_mib must be between 0 and 65536")
    if not 32 <= int(query_chunk_rows) <= 2048:
        raise ValueError("query_chunk_rows must be between 32 and 2048")

    relay_plan, resolved_global_prompt = _validate_effect_modes(
        prompt_relay_mode=prompt_relay_mode,
        prompt_relay_plan=prompt_relay_plan,
        eav_mode=eav_mode,
        steps=steps,
        sampler_name=sampler_name,
        scheduler=scheduler,
        global_prompt=global_prompt,
        segment_prompts_json=segment_prompts_json,
    )
    orchestration, manifest = resolve_long_video_orchestration(
        chain_id,
        total_duration_seconds,
        render_window_frames,
        context_frames,
        resolved_global_prompt,
        segment_prompts_json,
        base_seed,
        seed_policy,
        steps,
        shift_video,
        shift_audio,
        sampler_name,
        scheduler,
    )
    if relay_plan is not None:
        required_global_frames = max(
            round(float(segment.plan.timeline_end_seconds) * 24)
            for segment in orchestration.segments
        )
        if int(relay_plan["frame_count"]) < required_global_frames:
            raise ValueError(
                "The global Prompt Relay Plan is shorter than the requested long video: "
                f"plan={int(relay_plan['frame_count'])} frames, "
                f"required={required_global_frames} frames"
            )
    safe_chain = orchestration.chain_id
    root = long_video_chain_root(safe_chain)
    state_path = root / EFFECTS_LOOP_STATE_NAME
    base_contract = _job_contract(
        chain_id=safe_chain,
        total_duration_seconds=total_duration_seconds,
        render_window_frames=render_window_frames,
        context_frames=context_frames,
        global_prompt=resolved_global_prompt,
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
    contract = {
        "schema": EFFECTS_LOOP_SCHEMA,
        "format": EFFECTS_LOOP_FORMAT,
        "base": base_contract,
        "effects": {
            "prompt_relay_mode": str(prompt_relay_mode),
            "prompt_relay_plan_hash": (
                str(relay_plan["plan_hash"]) if relay_plan is not None else ""
            ),
            "query_chunk_rows": int(query_chunk_rows),
            "eav_mode": str(eav_mode),
            "eav_tau": float(eav_tau),
            "eav_start_video_progress": float(eav_start_video_progress),
            "eav_end_video_progress": float(eav_end_video_progress),
            "eav_max_workspace_mib": int(eav_max_workspace_mib),
            "eav_g_hard_limit": float(eav_g_hard_limit),
            "minimum_free_vram_mib": int(minimum_free_vram_mib),
        },
    }
    contract_sha256 = _sha256_json(contract)
    segment_count = len(orchestration.segments)
    sampling_summary = _effects_summary(
        base_summary=orchestration.sampling_summary,
        prompt_relay_mode=prompt_relay_mode,
        eav_mode=eav_mode,
    )

    with _exclusive_loop_lock(root):
        state = _load_effects_state(state_path)
        if state is not None and state.get("chain_id") != safe_chain:
            raise ValueError("In-node effects state chain_id is inconsistent")
        if not resume_existing and (
            state is not None or orchestration.accepted_count or _has_saved_candidate(root)
        ):
            raise ValueError(
                "resume_existing is false but this chain_id already contains saved state"
            )
        if state is not None and state.get("contract_sha256") != contract_sha256:
            accepted = int(orchestration.accepted_count)
            if accepted or int(state.get("accepted_count", 0)):
                raise ValueError(
                    "This chain_id contains accepted segments from a different in-node "
                    "effects contract; restore the settings or use a new chain_id"
                )
            state = None
        if orchestration.accepted_count:
            _accepted_effect_audits(root, manifest, contract_sha256)
        adopted = state is None and orchestration.accepted_count > 0
        created_unix = state.get("created_unix") if state else None
        existing_output = _existing_complete_output(
            state, orchestration.manifest_revision
        )
        if orchestration.complete and existing_output:
            report = dict(state)
            report.update(
                {
                    "resume_action": "returned_verified_existing_final",
                    "state_path": str(state_path),
                    "manifest_path": str(root / "manifest.json"),
                    "segment_audits": _accepted_effect_audits(
                        root, manifest, contract_sha256
                    ),
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
        state = _effects_state_payload(
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
            plain_long_video_model = patch_long_video_model(model)
            for segment in orchestration.segments[orchestration.accepted_count :]:
                _check_interrupted()
                preflight = _resource_snapshot(int(minimum_free_vram_mib))
                manifest_now, _source = load_delivery_manifest(safe_chain, allow_new=True)
                if len(manifest_now["segments"]) != segment.index:
                    raise RuntimeError(
                        "Accepted manifest changed while the in-node effects loop was running"
                    )
                context, _has_context, parent_candidate_id, parent_revision, _context_report = (
                    load_accepted_context(safe_chain, segment.index)
                )
                plan = segment.plan
                projected_plan = None
                projection_report = None
                if relay_plan is not None:
                    projected_plan, _compiled, projection_report_json = (
                        project_prompt_relay_plan_to_long_video_window(
                            relay_plan,
                            segment.index,
                            plan.render_frames,
                            plan.context_frames,
                            plan.timeline_start_seconds,
                            plan.timeline_end_seconds,
                        )
                    )
                    projection_report = json.loads(projection_report_json)
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
                    "sampling_summary": sampling_summary,
                    "prompt": (
                        str(projected_plan["compiled_prompt"])
                        if projected_plan is not None
                        else segment.prompt
                    ),
                    "seed": segment.seed,
                    "width": int(width),
                    "height": int(height),
                    "effects_contract_sha256": contract_sha256,
                    "projected_plan_hash": (
                        str(projected_plan["plan_hash"])
                        if projected_plan is not None
                        else ""
                    ),
                }
                base_candidate_id = _candidate_base_id(expected, contract_sha256)
                candidate_json_path = _reusable_candidate(
                    root, expected, base_candidate_id
                )
                if candidate_json_path is not None:
                    try:
                        candidate, _video = load_long_video_candidate_descriptor(
                            candidate_json_path
                        )
                        _load_effects_audit(
                            candidate_json_path,
                            contract_sha256=contract_sha256,
                            segment_index=segment.index,
                            candidate_id=str(candidate["candidate_id"]),
                        )
                    except (OSError, ValueError):
                        # Keep the incomplete/tampered candidate as forensic evidence and
                        # allocate a non-overwriting retry namespace below. This is the exact
                        # crash window between candidate save and effects-audit publication.
                        candidate_json_path = None
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
                        relay_report = {
                            "status": "disabled",
                            "global_plan_hash": "",
                            "projected_plan_hash": "",
                        }
                        if projected_plan is None:
                            (
                                positive,
                                av_latent,
                                mux_audio,
                                conditioned_prompt,
                                _media_map_json,
                                conditioning_report_json,
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
                            segment_model = plain_long_video_model
                        else:
                            relay_result = build_prompt_relay_long_video_conditioning(
                                model=model,
                                clip=clip,
                                video_vae=video_vae,
                                audio_vae=audio_vae,
                                context=context,
                                prompt_relay_plan=projected_plan,
                                segment_index=segment.index,
                                context_frames=plan.context_frames,
                                context_audio=context_audio,
                                width=width,
                                height=height,
                                length=plan.render_frames,
                                task_type=task_type,
                                audio_mode=audio_mode,
                                audio_denoise_strength=audio_denoise_strength,
                                add_source_as_reference=add_source_as_reference,
                                prompt_primary_audio_ordinal=prompt_primary_audio_ordinal,
                                strict_prompt_tags=strict_prompt_tags,
                                ref_image_size=ref_image_size,
                                reference_video_policy=reference_video_policy,
                                execution_mode=prompt_relay_mode,
                                query_chunk_rows=query_chunk_rows,
                                drive_audio=segment_drive_audio,
                                final_audio=segment_final_audio,
                                first_frame=first_frame,
                                last_frame=(last_frame if plan.is_final_segment else None),
                                ref_images=ref_images,
                                ref_videos=ref_videos,
                                ref_video_audios=ref_video_audios,
                                ref_audios=ref_audios,
                                first_frame_reuse=first_frame_reuse,
                                persistent_identity_image=persistent_identity_image,
                                persistent_identity_strategy=persistent_identity_strategy,
                                persistent_identity_interval=persistent_identity_interval,
                            )
                            (
                                segment_model,
                                positive,
                                av_latent,
                                mux_audio,
                                conditioned_prompt,
                                _media_map_json,
                                relay_report_json,
                            ) = relay_result
                            relay_report = json.loads(relay_report_json)
                            conditioning_report_json = relay_report["long_video_report"]
                            # report_only does not install the Relay/LongVideo model patch;
                            # sampling still needs the scoped continuation layout repair.
                            segment_model = patch_long_video_model(segment_model)

                        sampled_model, sampler, sigmas = setup_dual_clock_sampling(
                            segment_model,
                            av_latent,
                            orchestration.steps,
                            orchestration.shift_video,
                            orchestration.shift_audio,
                            orchestration.sampler_name,
                            orchestration.scheduler,
                        )
                        eav_runtime = None
                        eav_setup_report = {"status": "disabled"}
                        if eav_mode != "disabled":
                            relay_applied = relay_report.get("status") == "applied_exp"
                            if relay_applied:
                                sampled_model, eav_runtime, eav_setup_json = (
                                    build_eav_prompt_relay_long_video_model(
                                        sampled_model,
                                        sigmas,
                                        segment_index=segment.index,
                                        context_frames=plan.context_frames,
                                        mode=eav_mode,
                                        tau=eav_tau,
                                        start_video_progress=eav_start_video_progress,
                                        end_video_progress=eav_end_video_progress,
                                        max_workspace_mib=eav_max_workspace_mib,
                                        g_hard_limit=eav_g_hard_limit,
                                    )
                                )
                            else:
                                sampled_model, eav_runtime, eav_setup_json = (
                                    build_eav_long_video_model(
                                        sampled_model,
                                        sigmas,
                                        segment_index=segment.index,
                                        context_frames=plan.context_frames,
                                        mode=eav_mode,
                                        tau=eav_tau,
                                        start_video_progress=eav_start_video_progress,
                                        end_video_progress=eav_end_video_progress,
                                        max_workspace_mib=eav_max_workspace_mib,
                                        g_hard_limit=eav_g_hard_limit,
                                    )
                                )
                            eav_setup_report = json.loads(eav_setup_json)
                        sampled = _sample_prepared_segment(
                            sampled_model,
                            positive,
                            av_latent,
                            sampler=sampler,
                            sigmas=sigmas,
                            seed=segment.seed,
                        )
                        eav_audit_report = {"status": "disabled"}
                        if eav_runtime is not None:
                            sampled, eav_audit_json = finalize_eav_runtime(
                                sampled, eav_runtime
                            )
                            eav_audit_report = json.loads(eav_audit_json)
                        frames, generated_audio, _video_latent, _audio_latent = (
                            decode_av_latent(sampled, video_vae, audio_vae)
                        )
                        delivery_audio = (
                            mux_audio if mux_audio is not None else generated_audio
                        )
                        trimmed_frames, trimmed_audio, trim_report_json = trim_av_output(
                            frames,
                            plan.trim_start_seconds,
                            plan.final_duration_seconds,
                            delivery_audio,
                            24.0,
                        )
                        if trimmed_audio is None:
                            raise RuntimeError(
                                "In-node effects segment has no audio delivery value"
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
                                sampling_summary,
                                conditioned_prompt,
                                segment.seed,
                                24,
                                bit_depth,
                                crf,
                            )
                        )
                        _write_effects_audit(
                            candidate_json_path,
                            {
                                "contract_sha256": contract_sha256,
                                "segment_index": int(segment.index),
                                "candidate_id": candidate_id,
                                "global_plan_hash": (
                                    str(relay_plan["plan_hash"])
                                    if relay_plan is not None
                                    else ""
                                ),
                                "projected_plan_hash": (
                                    str(projected_plan["plan_hash"])
                                    if projected_plan is not None
                                    else ""
                                ),
                                "prompt_relay": relay_report,
                                "projection": projection_report,
                                "enhance_a_video_setup": eav_setup_report,
                                "enhance_a_video_audit": eav_audit_report,
                                "conditioning": _json_object(conditioning_report_json),
                                "trim": _json_object(trim_report_json),
                                "resource_preflight": preflight,
                            },
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
                        f"In-node effects segment {segment.index} was not accepted"
                    )
                manifest_now, _manifest_source = load_delivery_manifest(safe_chain)
                state = _effects_state_payload(
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
                raise RuntimeError("In-node effects loop ended before all segments were accepted")
            audits = _accepted_effect_audits(root, manifest_now, contract_sha256)
            final_video_path, compose_report_json = compose_accepted_long_video(
                safe_chain,
                filename_prefix,
                True,
                audio_seam_policy,
                bridge_ms,
                crf,
            )
            compose_report = json.loads(compose_report_json)
            state = _effects_state_payload(
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
                "sampling_summary": sampling_summary,
                "effect_contract": contract["effects"],
                "segment_audits": audits,
                "resume_action": (
                    "adopted_existing_then_completed"
                    if adopted
                    else "generated_or_resumed_then_completed"
                ),
                "compose": compose_report,
                "audio_boundary": (
                    "FETA does not directly scale audio rows; Prompt Relay joint_av_exp and "
                    "later joint-AV layers can still alter audio and require listening"
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
            if isinstance(error, comfy.model_management.InterruptProcessingException):
                failure_status = "interrupted"
            elif isinstance(error, _ResourceHeadroomError):
                failure_status = "waiting_resources"
            else:
                failure_status = "failed"
            failed_state = _effects_state_payload(
                chain_id=safe_chain,
                contract_sha256=contract_sha256,
                status=failure_status,
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
                    add_note(f"Could not persist in-node effects failure state: {state_error}")
            raise
        finally:
            _release_segment_memory()
