from __future__ import annotations

import json
import math
from collections.abc import Mapping

from .core import (
    AUDIO_LATENT_FPS,
    CANVAS_MULTIPLE,
    FPS,
    MAX_PIXELS,
    VRAM_CAUTION_PIXELS,
    align_frame_count,
    align_frame_count_down,
    video_latent_t,
)
from .prompt_relay_advanced import _validate_plan


PRECISION_BYTES = {
    "bf16_fp16": 2,
    "fp32": 4,
}
QUERY_ROUTES = {"video_only_paper", "joint_av_exp"}


def _checked_int(name: str, value, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"Prompt Relay Resource Estimate {name} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"Prompt Relay Resource Estimate {name} must be an integer"
        ) from exc
    if parsed != value or not minimum <= parsed <= maximum:
        raise ValueError(
            f"Prompt Relay Resource Estimate {name} must be between "
            f"{minimum} and {maximum}"
        )
    return parsed


def _checked_float(name: str, value, minimum: float, maximum: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"Prompt Relay Resource Estimate {name} must be numeric")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"Prompt Relay Resource Estimate {name} must be numeric"
        ) from exc
    if not math.isfinite(parsed) or not minimum <= parsed <= maximum:
        raise ValueError(
            f"Prompt Relay Resource Estimate {name} must be between "
            f"{minimum} and {maximum}"
        )
    return parsed


def _mib(value: int) -> float:
    return float(value) / float(1024**2)


def _gib(value: int) -> float:
    return float(value) / float(1024**3)


def estimate_prompt_relay_resources(
    prompt_relay_plan: Mapping,
    width: int,
    height: int,
    query_chunk_rows: int,
    precision: str,
    keyframe_stills: int,
    reference_images_match: int,
    reference_video_count: int,
    reference_video_frames_each: int,
    reference_video_has_audio: bool,
    reference_video_audio_seconds_each: float,
    standalone_reference_audio_count: int,
    standalone_reference_audio_seconds_each: float,
    additional_text_rows: int,
    manual_extra_packed_rows: int,
) -> tuple[dict, int, float, str, str]:
    """Estimate explicit Prompt Relay bias allocation without loading any model.

    This intentionally does not estimate total attention activation, model weights,
    VAE/CLIP workspaces, backend scratch buffers, VBAR residency, or allocator
    fragmentation. It is a planning aid, never a memory-safety certificate.
    """

    plan = _validate_plan(prompt_relay_plan)
    width = _checked_int("width", width, CANVAS_MULTIPLE, 16384)
    height = _checked_int("height", height, CANVAS_MULTIPLE, 16384)
    if width % CANVAS_MULTIPLE or height % CANVAS_MULTIPLE:
        raise ValueError(
            "Prompt Relay Resource Estimate width and height must both be divisible by 32"
        )
    pixel_area = width * height
    if pixel_area > MAX_PIXELS:
        raise ValueError(
            "Prompt Relay Resource Estimate canvas exceeds the MiniMax H3 "
            f"{MAX_PIXELS:,}-pixel limit"
        )

    query_chunk_rows = _checked_int("query_chunk_rows", query_chunk_rows, 32, 2048)
    precision = str(precision)
    if precision not in PRECISION_BYTES:
        raise ValueError(f"Unknown Prompt Relay Resource Estimate precision {precision!r}")
    dtype_bytes = PRECISION_BYTES[precision]

    keyframe_stills = _checked_int("keyframe_stills", keyframe_stills, 0, 16)
    reference_images_match = _checked_int(
        "reference_images_match", reference_images_match, 0, 16
    )
    reference_video_count = _checked_int(
        "reference_video_count", reference_video_count, 0, 3
    )
    reference_video_frames_each = _checked_int(
        "reference_video_frames_each", reference_video_frames_each, 5, 3600
    )
    reference_video_audio_seconds_each = _checked_float(
        "reference_video_audio_seconds_each",
        reference_video_audio_seconds_each,
        0.0,
        900.0,
    )
    standalone_reference_audio_count = _checked_int(
        "standalone_reference_audio_count", standalone_reference_audio_count, 0, 3
    )
    standalone_reference_audio_seconds_each = _checked_float(
        "standalone_reference_audio_seconds_each",
        standalone_reference_audio_seconds_each,
        0.0,
        900.0,
    )
    additional_text_rows = _checked_int(
        "additional_text_rows", additional_text_rows, 0, 1_000_000
    )
    manual_extra_packed_rows = _checked_int(
        "manual_extra_packed_rows", manual_extra_packed_rows, 0, 10_000_000
    )

    frame_count = plan.get("frame_count")
    if isinstance(frame_count, bool) or not isinstance(frame_count, int):
        raise ValueError("Prompt Relay Resource Estimate requires an integer frame_count")
    if frame_count < 5 or frame_count > 3600 or align_frame_count(frame_count) != frame_count:
        raise ValueError(
            "Prompt Relay Resource Estimate requires a 5..3600 frame H3 17n+5 timeline"
        )
    if not math.isclose(float(plan.get("fps", 0.0)), float(FPS), abs_tol=1e-12):
        raise ValueError("Prompt Relay Resource Estimate requires the native 24fps timeline")
    events = plan.get("events")
    if not isinstance(events, list):
        raise ValueError("Prompt Relay Resource Estimate requires a Plan event list")
    event_count = len(events)
    relay_active = event_count > 1

    query_route = str(plan.get("query_route", "video_only_paper"))
    if query_route not in QUERY_ROUTES:
        raise ValueError(
            f"Unknown Prompt Relay Resource Estimate query route {query_route!r}"
        )

    frame_rows = (width // CANVAS_MULTIPLE) * (height // CANVAS_MULTIPLE)
    target_video_latent_t = video_latent_t(frame_count)
    target_video_rows = target_video_latent_t * frame_rows
    duration_seconds = frame_count / FPS
    target_audio_latent_t = round(duration_seconds * AUDIO_LATENT_FPS)
    target_audio_rows = target_audio_latent_t * 2

    keyframe_rows = keyframe_stills * frame_rows
    reference_image_rows = reference_images_match * frame_rows
    effective_reference_video_frames = align_frame_count_down(
        reference_video_frames_each
    )
    reference_video_latent_t = video_latent_t(effective_reference_video_frames)
    reference_video_rows = (
        reference_video_count * reference_video_latent_t * frame_rows
    )
    reference_video_audio_rows = 0
    if bool(reference_video_has_audio) and reference_video_count:
        reference_video_audio_rows = (
            reference_video_count
            * round(reference_video_audio_seconds_each * AUDIO_LATENT_FPS)
            * 2
        )
    standalone_reference_audio_rows = (
        standalone_reference_audio_count
        * round(standalone_reference_audio_seconds_each * AUDIO_LATENT_FPS)
        * 2
    )
    conditioning_rows = (
        keyframe_rows
        + reference_image_rows
        + reference_video_rows
        + reference_video_audio_rows
        + standalone_reference_audio_rows
    )

    compiled_prompt = str(plan.get("compiled_prompt", ""))
    prompt_utf8_bytes_proxy = len(compiled_prompt.encode("utf-8"))
    estimated_text_rows = prompt_utf8_bytes_proxy + additional_text_rows
    estimated_seq_len = (
        estimated_text_rows
        + conditioning_rows
        + manual_extra_packed_rows
        + target_audio_rows
        + target_video_rows
    )

    potential_query_rows = target_video_rows
    route_chunk_count_if_active = math.ceil(target_video_rows / query_chunk_rows)
    largest_query_group = target_video_rows
    if query_route == "joint_av_exp":
        potential_query_rows += target_audio_rows
        route_chunk_count_if_active += math.ceil(target_audio_rows / query_chunk_rows)
        largest_query_group = max(target_video_rows, target_audio_rows)

    routed_query_rows = potential_query_rows if relay_active else 0
    route_chunk_count = route_chunk_count_if_active if relay_active else 0
    peak_query_rows = min(query_chunk_rows, largest_query_group) if relay_active else 0
    peak_explicit_bias_bytes = peak_query_rows * estimated_seq_len * dtype_bytes
    hypothetical_dense_bytes = estimated_seq_len * estimated_seq_len * dtype_bytes

    standard_canvas_matrix = []
    for profile_id, profile_width, profile_height in (
        ("preview_0p3m", 736, 416),
        ("balanced_0p7m", 1152, 640),
        ("maximum_2p0m", 1920, 1088),
    ):
        profile_frame_rows = (
            profile_width // CANVAS_MULTIPLE
        ) * (profile_height // CANVAS_MULTIPLE)
        profile_target_video_rows = target_video_latent_t * profile_frame_rows
        profile_keyframe_rows = keyframe_stills * profile_frame_rows
        profile_reference_image_rows = reference_images_match * profile_frame_rows
        profile_reference_video_rows = (
            reference_video_count
            * reference_video_latent_t
            * profile_frame_rows
        )
        profile_conditioning_rows = (
            profile_keyframe_rows
            + profile_reference_image_rows
            + profile_reference_video_rows
            + reference_video_audio_rows
            + standalone_reference_audio_rows
        )
        profile_seq_len = (
            estimated_text_rows
            + profile_conditioning_rows
            + manual_extra_packed_rows
            + target_audio_rows
            + profile_target_video_rows
        )
        profile_potential_query_rows = profile_target_video_rows
        profile_chunk_count_if_active = math.ceil(
            profile_target_video_rows / query_chunk_rows
        )
        profile_largest_query_group = profile_target_video_rows
        if query_route == "joint_av_exp":
            profile_potential_query_rows += target_audio_rows
            profile_chunk_count_if_active += math.ceil(
                target_audio_rows / query_chunk_rows
            )
            profile_largest_query_group = max(
                profile_target_video_rows, target_audio_rows
            )
        profile_peak_query_rows = (
            min(query_chunk_rows, profile_largest_query_group)
            if relay_active
            else 0
        )
        profile_bias_bytes = (
            profile_peak_query_rows * profile_seq_len * dtype_bytes
        )
        profile_dense_bytes = profile_seq_len * profile_seq_len * dtype_bytes
        standard_canvas_matrix.append(
            {
                "profile": profile_id,
                "width": profile_width,
                "height": profile_height,
                "pixel_area": profile_width * profile_height,
                "selected_canvas": width == profile_width and height == profile_height,
                "frame_rows": profile_frame_rows,
                "target_video_rows": profile_target_video_rows,
                "target_audio_rows": target_audio_rows,
                "condition_reference_rows": profile_conditioning_rows,
                "estimated_seq_len": profile_seq_len,
                "potential_routed_query_rows": profile_potential_query_rows,
                "route_chunk_count_if_active": profile_chunk_count_if_active,
                "route_chunk_count": (
                    profile_chunk_count_if_active if relay_active else 0
                ),
                "peak_explicit_bias_bytes": profile_bias_bytes,
                "peak_explicit_bias_mib": _mib(profile_bias_bytes),
                "hypothetical_dense_sxs_gib_not_allocated": _gib(
                    profile_dense_bytes
                ),
            }
        )

    warnings = [
        "This estimate covers only Prompt Relay's explicit chunk bias tensor and packed-row planning proxy.",
        "It does not estimate H3 weights, full attention activations, CLIP/VAE workspaces, backend scratch buffers, VBAR/DynamicVRAM residency, allocator fragmentation, or total GPU headroom.",
        "UTF-8 prompt bytes plus additional_text_rows is a planning proxy, not the authoritative Qwen token count; media/system tokens may differ.",
        "reference_images_match assumes match-size reference images; use manual_extra_packed_rows for max-size or otherwise unrepresented conditioning rows.",
        "The result is not a universal 16GB safety claim; run the existing real preflight/probe for the exact workflow and backend.",
    ]
    if not relay_active:
        warnings.append(
            "Zero or one active local event uses Prompt Relay's no-patch passthrough, so actual Relay bias allocation is zero."
        )
    if pixel_area > VRAM_CAUTION_PIXELS:
        warnings.append(
            "Canvas exceeds the project's 1.032MP caution threshold; total H3 activation pressure can be high even when the Relay bias estimate is small."
        )
    if peak_explicit_bias_bytes > 64 * 1024**2:
        warnings.append(
            "The estimated explicit Relay bias alone exceeds 64MiB; reduce query_chunk_rows or packed conditioning before a real run."
        )
    if query_route == "joint_av_exp":
        warnings.append(
            "joint_av_exp adds target-audio query chunks and is an H3 extension, not a mode validated by the Prompt Relay paper."
        )

    report = {
        "status": "prompt_relay_resource_estimate_ready",
        "experimental": True,
        "scope": "explicit_prompt_relay_bias_and_packed_rows_proxy_only",
        "plan_hash": plan["plan_hash"],
        "event_count": event_count,
        "relay_active": relay_active,
        "query_route": query_route,
        "target": {
            "width": width,
            "height": height,
            "pixel_area": pixel_area,
            "frame_count": frame_count,
            "fps": FPS,
            "duration_seconds": duration_seconds,
            "frame_rows": frame_rows,
            "video_latent_t": target_video_latent_t,
            "target_video_rows": target_video_rows,
            "audio_latent_t": target_audio_latent_t,
            "target_audio_rows": target_audio_rows,
        },
        "text_proxy": {
            "prompt_utf8_bytes_proxy": prompt_utf8_bytes_proxy,
            "additional_text_rows": additional_text_rows,
            "estimated_text_rows": estimated_text_rows,
            "authoritative_tokenizer_executed": False,
        },
        "conditioning_rows": {
            "keyframe_stills": keyframe_stills,
            "keyframe_rows": keyframe_rows,
            "reference_images_match": reference_images_match,
            "reference_image_rows": reference_image_rows,
            "reference_video_count": reference_video_count,
            "reference_video_frames_each_requested": reference_video_frames_each,
            "reference_video_frames_each_effective": effective_reference_video_frames,
            "reference_video_latent_t_each": reference_video_latent_t,
            "reference_video_rows": reference_video_rows,
            "reference_video_has_audio": bool(reference_video_has_audio),
            "reference_video_audio_rows": reference_video_audio_rows,
            "standalone_reference_audio_count": standalone_reference_audio_count,
            "standalone_reference_audio_rows": standalone_reference_audio_rows,
            "conditioning_rows_total": conditioning_rows,
            "manual_extra_packed_rows": manual_extra_packed_rows,
        },
        "packed_sequence": {
            "estimated_seq_len": estimated_seq_len,
            "is_runtime_measurement": False,
        },
        "standard_canvas_matrix": standard_canvas_matrix,
        "relay_bias": {
            "query_chunk_rows": query_chunk_rows,
            "precision": precision,
            "dtype_bytes": dtype_bytes,
            "potential_routed_query_rows": potential_query_rows,
            "routed_query_rows": routed_query_rows,
            "route_chunk_count_if_active": route_chunk_count_if_active,
            "route_chunk_count": route_chunk_count,
            "peak_query_rows": peak_query_rows,
            "peak_explicit_bias_bytes": peak_explicit_bias_bytes,
            "peak_explicit_bias_mib": _mib(peak_explicit_bias_bytes),
            "hypothetical_dense_sxs_bytes_not_allocated": hypothetical_dense_bytes,
            "hypothetical_dense_sxs_mib_not_allocated": _mib(hypothetical_dense_bytes),
            "hypothetical_dense_sxs_gib_not_allocated": _gib(hypothetical_dense_bytes),
            "implementation_allocates_dense_sxs": False,
        },
        "execution": {
            "model_loaded": False,
            "media_encoded": False,
            "sampling_executed": False,
        },
        "warnings": warnings,
    }
    summary_lines = [
        "Prompt Relay Resource Estimate: READY (planning proxy)",
        (
            f"{width}x{height} | {frame_count} frames @ {FPS}fps | "
            f"events={event_count} | relay_active={'yes' if relay_active else 'no'}"
        ),
        (
            f"frame_rows={frame_rows:,} | video_rows={target_video_rows:,} | "
            f"audio_rows={target_audio_rows:,}"
        ),
        (
            f"estimated packed rows={estimated_seq_len:,} | route={query_route} | "
            f"chunks={route_chunk_count}"
        ),
        (
            f"peak explicit bias={_mib(peak_explicit_bias_bytes):.2f}MiB "
            f"({precision}, chunk={query_chunk_rows})"
        ),
        (
            "standard matrix bias: "
            + " | ".join(
                f"{item['width']}x{item['height']}={item['peak_explicit_bias_mib']:.2f}MiB"
                for item in standard_canvas_matrix
            )
        ),
        "Scope: explicit Relay bias only; this is NOT total VRAM or a 16GB safety certificate.",
    ]
    return (
        plan,
        estimated_seq_len,
        _mib(peak_explicit_bias_bytes),
        "\n".join(summary_lines),
        json.dumps(report, ensure_ascii=False, indent=2),
    )
