from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from typing import Any

import torch

import comfy.nested_tensor

from .core import AUDIO_LATENT_FPS, FPS, nested_av_parts, sorted_autogrow_values
from .long_video import CONTEXT_FRAME_STEPS, LONG_VIDEO_SCHEMA, pixel_frames_from_latent_t


VIDEO_PREFIX_LATENT_STEPS = 2
VIDEO_PREFIX_FRAMES = 5
RESUME_MANIFEST_SCHEMA = "t8.minimax_h3.native_latent_resume_manifest.v1"
CONTINUATION_CONCAT_SCHEMA = "t8.minimax_h3.native_latent_continuation_concat.v1"
_VOLATILE_RESUME_METADATA_KEYS = {
    "t8_native_latent_checkpoint",
    "t8_native_latent_timeline_concat",
    "t8_native_latent_resume_manifest",
}


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _tensor_bytes(value: torch.Tensor) -> int:
    return int(value.numel() * value.element_size())


def _mask_parts(latent: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor] | None:
    masks = latent.get("noise_mask")
    if masks is None:
        return None
    if not getattr(masks, "is_nested", False):
        raise ValueError(
            "Native latent timeline concat requires a nested AV noise_mask or no noise_mask; "
            "legacy video-only masks are ambiguous"
        )
    parts = tuple(masks.unbind())
    if len(parts) != 2 or parts[0].ndim != 5 or parts[1].ndim != 4:
        raise ValueError("Native latent timeline concat received an invalid nested AV noise_mask")
    return parts[0], parts[1]


def _metadata_signature(latent: Mapping[str, Any]) -> dict[str, str]:
    result = {}
    for key, value in latent.items():
        if key in {"samples", "noise_mask", "t8_native_latent_timeline_concat"}:
            continue
        if torch.is_tensor(value):
            result[str(key)] = f"tensor:{tuple(value.shape)}:{value.dtype}:{value.device}"
        else:
            try:
                result[str(key)] = _json(value)
            except (TypeError, ValueError):
                result[str(key)] = repr(value)
    return result


def _tensor_content_digest(
    value: torch.Tensor,
    *,
    max_chunk_bytes: int,
) -> tuple[str, int]:
    if value.layout != torch.strided:
        raise ValueError(f"resume manifest cannot hash non-strided tensor layout {value.layout}")
    if max_chunk_bytes < 1:
        raise ValueError("resume manifest hash chunk size must be positive")
    tensor = value.detach()
    byte_count = _tensor_bytes(tensor)
    hasher = hashlib.sha256()
    hasher.update(
        _json(
            {
                "kind": "tensor",
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
            }
        ).encode("utf-8")
    )

    def visit(part: torch.Tensor) -> None:
        part_bytes = _tensor_bytes(part)
        if part_bytes <= max_chunk_bytes or part.ndim == 0 or part.numel() == 0:
            cpu = part.to(device="cpu").contiguous()
            if cpu.numel():
                hasher.update(cpu.view(torch.uint8).numpy().tobytes(order="C"))
            return
        split_dim = next((index for index, size in enumerate(part.shape) if size > 1), None)
        if split_dim is None:
            cpu = part.to(device="cpu").contiguous()
            hasher.update(cpu.view(torch.uint8).numpy().tobytes(order="C"))
            return
        dimension = int(part.shape[split_dim])
        bytes_per_index = max(1, (part_bytes + dimension - 1) // dimension)
        indices_per_chunk = max(1, max_chunk_bytes // bytes_per_index)
        for start in range(0, dimension, indices_per_chunk):
            visit(part.narrow(split_dim, start, min(indices_per_chunk, dimension - start)))

    visit(tensor)
    return hasher.hexdigest().upper(), byte_count


def _stable_value_digest(
    value: Any,
    *,
    max_chunk_bytes: int,
    path: str,
) -> tuple[str, int]:
    if torch.is_tensor(value):
        return _tensor_content_digest(value, max_chunk_bytes=max_chunk_bytes)
    if getattr(value, "is_nested", False) and hasattr(value, "unbind"):
        parts = tuple(value.unbind())
        part_digests = []
        tensor_bytes = 0
        for index, part in enumerate(parts):
            digest, part_bytes = _stable_value_digest(
                part,
                max_chunk_bytes=max_chunk_bytes,
                path=f"{path}[{index}]",
            )
            part_digests.append(digest)
            tensor_bytes += part_bytes
        return (
            hashlib.sha256(_json(["nested", part_digests]).encode("utf-8")).hexdigest().upper(),
            tensor_bytes,
        )
    if value is None or isinstance(value, (bool, int, float, str)):
        try:
            encoded = json.dumps(
                ["scalar", type(value).__name__, value],
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"resume manifest cannot serialize metadata at {path}: {exc}") from exc
        return hashlib.sha256(encoded).hexdigest().upper(), 0
    if isinstance(value, bytes):
        return hashlib.sha256(b"bytes\x00" + value).hexdigest().upper(), 0
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError(f"resume manifest requires string metadata keys at {path}")
        items = []
        tensor_bytes = 0
        for key in sorted(value):
            digest, child_bytes = _stable_value_digest(
                value[key],
                max_chunk_bytes=max_chunk_bytes,
                path=f"{path}.{key}",
            )
            items.append([key, digest])
            tensor_bytes += child_bytes
        return (
            hashlib.sha256(_json(["mapping", items]).encode("utf-8")).hexdigest().upper(),
            tensor_bytes,
        )
    if isinstance(value, (list, tuple)):
        items = []
        tensor_bytes = 0
        for index, child in enumerate(value):
            digest, child_bytes = _stable_value_digest(
                child,
                max_chunk_bytes=max_chunk_bytes,
                path=f"{path}[{index}]",
            )
            items.append(digest)
            tensor_bytes += child_bytes
        container = "tuple" if isinstance(value, tuple) else "list"
        return (
            hashlib.sha256(_json([container, items]).encode("utf-8")).hexdigest().upper(),
            tensor_bytes,
        )
    raise ValueError(
        f"resume manifest cannot safely fingerprint metadata at {path}: "
        f"unsupported type {type(value).__name__}"
    )


def _manifest_identity(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "checkpoint_id": manifest.get("checkpoint_id"),
        "content_sha256": manifest.get("content_sha256"),
        "frame_count": manifest.get("frame_count"),
        "video_shape": manifest.get("video_shape"),
        "video_dtype": manifest.get("video_dtype"),
        "audio_shape": manifest.get("audio_shape"),
        "audio_dtype": manifest.get("audio_dtype"),
        "noise_mask_present": manifest.get("noise_mask_present"),
    }


def audit_native_h3_av_latent_resume_manifest(
    av_latent: Mapping[str, Any],
    checkpoint_id: str = "timeline_checkpoint",
    expected_manifest_json: str = "",
    mismatch_policy: str = "error",
    hash_chunk_megabytes: int = 8,
) -> tuple[str, bool, str, str]:
    if not isinstance(av_latent, Mapping):
        raise ValueError("Native latent resume manifest requires one LATENT mapping")
    checkpoint_id = str(checkpoint_id).strip()
    if not checkpoint_id or len(checkpoint_id) > 128:
        raise ValueError("checkpoint_id must contain 1 to 128 non-whitespace characters")
    if mismatch_policy not in {"error", "report_only"}:
        raise ValueError(f"unsupported resume manifest mismatch_policy: {mismatch_policy!r}")
    if not 1 <= int(hash_chunk_megabytes) <= 64:
        raise ValueError("hash_chunk_megabytes must be between 1 and 64")
    max_chunk_bytes = int(hash_chunk_megabytes) * 1024 * 1024

    video, audio = nested_av_parts(dict(av_latent))
    if video.shape[0] != 1 or audio.shape[0] != 1:
        raise ValueError("Native latent resume manifest currently requires batch 1")
    if video.shape[1] != 24 or audio.shape[1:3] != (32, 2):
        raise ValueError(
            "Native latent resume manifest requires the MiniMax H3 "
            "[1,24,T,H,W] + [1,32,2,T] contract"
        )
    video_t = int(video.shape[2])
    if video_t < VIDEO_PREFIX_LATENT_STEPS or (video_t - 2) % 5:
        raise ValueError(f"video latent T={video_t} is not a complete native H3 5n+2 grid")
    frame_count = pixel_frames_from_latent_t(video_t)
    expected_audio_t = round(frame_count / FPS * AUDIO_LATENT_FPS)
    if int(audio.shape[-1]) != expected_audio_t:
        raise ValueError(
            f"audio latent T={audio.shape[-1]} does not match "
            f"round({frame_count}/24*40)={expected_audio_t}"
        )

    components = []
    total_tensor_bytes = 0
    for name, tensor in (("samples.video", video), ("samples.audio", audio)):
        digest, tensor_bytes = _tensor_content_digest(
            tensor,
            max_chunk_bytes=max_chunk_bytes,
        )
        components.append({"name": name, "sha256": digest, "tensor_bytes": tensor_bytes})
        total_tensor_bytes += tensor_bytes

    masks = _mask_parts(av_latent)
    if masks is not None:
        if masks[0].shape != video.shape or masks[1].shape != audio.shape:
            raise ValueError("Native latent resume manifest noise_mask shapes do not match AV samples")
        for name, tensor in (("noise_mask.video", masks[0]), ("noise_mask.audio", masks[1])):
            digest, tensor_bytes = _tensor_content_digest(
                tensor,
                max_chunk_bytes=max_chunk_bytes,
            )
            components.append({"name": name, "sha256": digest, "tensor_bytes": tensor_bytes})
            total_tensor_bytes += tensor_bytes

    metadata_keys = [
        key
        for key in av_latent
        if key not in {"samples", "noise_mask", *_VOLATILE_RESUME_METADATA_KEYS}
    ]
    if any(not isinstance(key, str) for key in metadata_keys):
        raise ValueError("Native latent resume manifest requires string top-level metadata keys")
    metadata_keys = sorted(metadata_keys)
    for key in metadata_keys:
        digest, tensor_bytes = _stable_value_digest(
            av_latent[key],
            max_chunk_bytes=max_chunk_bytes,
            path=f"metadata.{key}",
        )
        components.append(
            {"name": f"metadata.{key}", "sha256": digest, "tensor_bytes": tensor_bytes}
        )
        total_tensor_bytes += tensor_bytes

    content_sha256 = hashlib.sha256(_json(components).encode("utf-8")).hexdigest().upper()
    manifest: dict[str, Any] = {
        "schema": RESUME_MANIFEST_SCHEMA,
        "checkpoint_id": checkpoint_id,
        "content_sha256": content_sha256,
        "frame_count": frame_count,
        "video_shape": list(video.shape),
        "video_dtype": str(video.dtype),
        "audio_shape": list(audio.shape),
        "audio_dtype": str(audio.dtype),
        "noise_mask_present": masks is not None,
        "metadata_keys_hashed": metadata_keys,
        "volatile_metadata_keys_excluded": sorted(
            key for key in _VOLATILE_RESUME_METADATA_KEYS if key in av_latent
        ),
        "hash_algorithm": "SHA-256",
        "hash_scope": "AV samples, optional nested AV noise_mask, and supported non-volatile metadata",
        "hash_chunk_megabytes": int(hash_chunk_megabytes),
        "tensor_bytes_hashed": total_tensor_bytes,
        "components": components,
        "sampling_executed": False,
        "vae_decode_executed": False,
        "files_written": False,
    }

    expected_text = str(expected_manifest_json or "").strip()
    resume_verified = False
    status = "BASELINE_CREATED"
    comparison: dict[str, Any] = {"expected_manifest_supplied": bool(expected_text)}
    if expected_text:
        try:
            expected = json.loads(expected_text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"expected_manifest_json is invalid JSON: {exc}") from exc
        if not isinstance(expected, Mapping) or expected.get("schema") != RESUME_MANIFEST_SCHEMA:
            raise ValueError(f"expected_manifest_json must use schema {RESUME_MANIFEST_SCHEMA}")
        expected_identity = _manifest_identity(expected)
        actual_identity = _manifest_identity(manifest)
        mismatches = [
            {
                "field": field,
                "expected": expected_identity[field],
                "actual": actual_identity[field],
            }
            for field in actual_identity
            if expected_identity[field] != actual_identity[field]
        ]
        resume_verified = not mismatches
        status = "MATCH" if resume_verified else "MISMATCH"
        comparison.update(
            {
                "expected_content_sha256": expected_identity["content_sha256"],
                "mismatches": mismatches,
            }
        )
        if mismatches and mismatch_policy == "error":
            fields = ", ".join(item["field"] for item in mismatches)
            raise ValueError(f"Native latent resume manifest mismatch: {fields}")
    manifest["status"] = status
    manifest["resume_verified"] = resume_verified
    manifest["comparison"] = comparison
    manifest["scientific_boundary"] = (
        "MATCH proves byte-exact latent, mask and supported metadata identity for this checkpoint. "
        "It does not persist tensors, resume diffusion internals, validate perceptual continuity, "
        "or reduce ComfyUI caching/VRAM. Save both the latent and this manifest with an external "
        "authorized checkpoint workflow before a crash can be recovered."
    )
    return status, resume_verified, content_sha256, _json(manifest)


def concat_native_h3_av_latents(
    first_segment: Mapping[str, Any],
    second_segment: Mapping[str, Any],
    output_device: str = "cpu",
    require_identical_metadata: bool = False,
    additional_segments: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], int, int, int, str]:
    if output_device not in {"cpu", "preserve_first"}:
        raise ValueError(f"unsupported native latent concat output_device: {output_device!r}")
    segments = [first_segment, second_segment, *sorted_autogrow_values(additional_segments)]
    if len(segments) < 2:
        raise ValueError("Native latent timeline concat requires at least two AV latent segments")

    parsed = []
    mask_presence = []
    metadata_signatures = []
    for index, latent in enumerate(segments):
        if not isinstance(latent, Mapping):
            raise ValueError(f"segment {index} is not a LATENT mapping")
        video, audio = nested_av_parts(dict(latent))
        if video.shape[1] != 24 or audio.shape[1:3] != (32, 2):
            raise ValueError(
                f"segment {index} is not the MiniMax H3 [1,24,T,H,W] + [1,32,2,T] contract"
            )
        video_t = int(video.shape[2])
        if video_t < VIDEO_PREFIX_LATENT_STEPS or (video_t - 2) % 5:
            raise ValueError(
                f"segment {index} video latent T={video_t} is not a complete native H3 5n+2 grid"
            )
        frames = pixel_frames_from_latent_t(video_t)
        expected_audio_t = round(frames / FPS * AUDIO_LATENT_FPS)
        if int(audio.shape[-1]) != expected_audio_t:
            raise ValueError(
                f"segment {index} audio latent T={audio.shape[-1]} does not match "
                f"round({frames}/24*40)={expected_audio_t}"
            )
        masks = _mask_parts(latent)
        if masks is not None and (masks[0].shape != video.shape or masks[1].shape != audio.shape):
            raise ValueError(f"segment {index} noise_mask shapes do not match its AV samples")
        parsed.append((video, audio, masks, frames))
        mask_presence.append(masks is not None)
        metadata_signatures.append(_metadata_signature(latent))

    first_video, first_audio, _, _ = parsed[0]
    spatial = tuple(first_video.shape[-2:])
    video_dtype = first_video.dtype
    audio_dtype = first_audio.dtype
    for index, (video, audio, _masks, _frames) in enumerate(parsed[1:], 1):
        if tuple(video.shape[-2:]) != spatial:
            raise ValueError(
                f"segment {index} canvas mismatch: latent {tuple(video.shape[-2:])}, expected {spatial}"
            )
        if video.dtype != video_dtype or audio.dtype != audio_dtype:
            raise ValueError(
                f"segment {index} dtype mismatch: video/audio {video.dtype}/{audio.dtype}, "
                f"expected {video_dtype}/{audio_dtype}"
            )
    if any(mask_presence) and not all(mask_presence):
        raise ValueError("All segments must either carry nested AV noise masks or carry none")
    if require_identical_metadata and any(
        signature != metadata_signatures[0] for signature in metadata_signatures[1:]
    ):
        raise ValueError("Segment metadata differs while require_identical_metadata is enabled")

    target_device = first_video.device if output_device == "preserve_first" else torch.device("cpu")
    video_parts = [first_video.to(device=target_device)]
    audio_parts = [first_audio.to(device=target_device)]
    video_mask_parts = []
    audio_mask_parts = []
    if all(mask_presence):
        first_video_mask, first_audio_mask = parsed[0][2]
        video_mask_parts.append(first_video_mask.to(device=target_device))
        audio_mask_parts.append(first_audio_mask.to(device=target_device))

    cumulative_frames = parsed[0][3]
    cumulative_audio_t = int(first_audio.shape[-1])
    segment_reports = [
        {
            "index": 0,
            "source_frames": parsed[0][3],
            "source_video_t": int(first_video.shape[2]),
            "source_audio_t": int(first_audio.shape[-1]),
            "drop_video_t": 0,
            "drop_audio_t": 0,
            "append_frames": parsed[0][3],
            "append_video_t": int(first_video.shape[2]),
            "append_audio_t": int(first_audio.shape[-1]),
        }
    ]
    for index, (video, audio, masks, frames) in enumerate(parsed[1:], 1):
        next_frames = cumulative_frames + frames - VIDEO_PREFIX_FRAMES
        next_audio_t = round(next_frames / FPS * AUDIO_LATENT_FPS)
        append_audio_t = next_audio_t - cumulative_audio_t
        drop_audio_t = int(audio.shape[-1]) - append_audio_t
        if append_audio_t < 1 or not 0 <= drop_audio_t < int(audio.shape[-1]):
            raise RuntimeError(
                f"segment {index} produced an impossible audio phase: "
                f"append={append_audio_t}, drop={drop_audio_t}"
            )
        video_parts.append(video[:, :, VIDEO_PREFIX_LATENT_STEPS:].to(device=target_device))
        audio_parts.append(audio[..., drop_audio_t:].to(device=target_device))
        if masks is not None:
            video_mask_parts.append(
                masks[0][:, :, VIDEO_PREFIX_LATENT_STEPS:].to(device=target_device)
            )
            audio_mask_parts.append(masks[1][..., drop_audio_t:].to(device=target_device))
        segment_reports.append(
            {
                "index": index,
                "source_frames": frames,
                "source_video_t": int(video.shape[2]),
                "source_audio_t": int(audio.shape[-1]),
                "drop_video_t": VIDEO_PREFIX_LATENT_STEPS,
                "drop_audio_t": drop_audio_t,
                "append_frames": frames - VIDEO_PREFIX_FRAMES,
                "append_video_t": int(video.shape[2]) - VIDEO_PREFIX_LATENT_STEPS,
                "append_audio_t": append_audio_t,
            }
        )
        cumulative_frames = next_frames
        cumulative_audio_t = next_audio_t

    output_video = torch.cat(video_parts, dim=2).contiguous()
    output_audio = torch.cat(audio_parts, dim=-1).contiguous()
    expected_video_t = sum(int(video.shape[2]) for video, *_ in parsed) - (
        len(parsed) - 1
    ) * VIDEO_PREFIX_LATENT_STEPS
    if int(output_video.shape[2]) != expected_video_t:
        raise RuntimeError("Native latent concat video T accounting changed unexpectedly")
    if pixel_frames_from_latent_t(expected_video_t) != cumulative_frames:
        raise RuntimeError("Native latent concat video phase does not map to the planned frame count")
    if int(output_audio.shape[-1]) != cumulative_audio_t:
        raise RuntimeError("Native latent concat audio phase does not match the cumulative clock")

    output = dict(first_segment)
    output["samples"] = comfy.nested_tensor.NestedTensor((output_video, output_audio))
    if all(mask_presence):
        output["noise_mask"] = comfy.nested_tensor.NestedTensor(
            (
                torch.cat(video_mask_parts, dim=2).contiguous(),
                torch.cat(audio_mask_parts, dim=-1).contiguous(),
            )
        )
    else:
        output.pop("noise_mask", None)

    metadata_equal = all(
        signature == metadata_signatures[0] for signature in metadata_signatures[1:]
    )
    report = {
        "schema": "t8.minimax_h3.native_latent_timeline_concat.v1",
        "experimental": True,
        "segment_count": len(parsed),
        "total_frame_count": cumulative_frames,
        "total_video_latent_t": int(output_video.shape[2]),
        "total_audio_latent_t": int(output_audio.shape[-1]),
        "fps": FPS,
        "audio_latent_fps": AUDIO_LATENT_FPS,
        "video_prefix_removed_per_later_segment": {
            "frames": VIDEO_PREFIX_FRAMES,
            "latent_steps": VIDEO_PREFIX_LATENT_STEPS,
        },
        "segments": segment_reports,
        "spatial_latent": list(spatial),
        "canvas_pixels": [spatial[1] * 16, spatial[0] * 16],
        "output_device": str(target_device),
        "output_bytes": _tensor_bytes(output_video) + _tensor_bytes(output_audio),
        "noise_masks_preserved": all(mask_presence),
        "metadata_equal": metadata_equal,
        "metadata_preserved_from": "first_segment",
        "sampling_executed": False,
        "vae_decode_executed": False,
        "scientific_boundary": (
            "The 5-frame/2-latent video prefix and cumulative 24fps-to-40Hz audio phase are "
            "mechanically exact for complete H3 grids. This does not prove that independently "
            "sampled segment states are seamless or that one VAE decode improves perceptual cuts."
        ),
    }
    output["t8_native_latent_timeline_concat"] = report
    return output, cumulative_frames, len(parsed), int(output_audio.shape[-1]), _json(report)


def _long_video_proof(value: str, *, label: str) -> dict[str, Any]:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be connected directly from the matching Long Video node")
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain one JSON object")
    try:
        schema = int(payload.get("schema", -1))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} has an invalid schema") from exc
    if schema != LONG_VIDEO_SCHEMA:
        raise ValueError(
            f"{label} uses Long Video schema {schema}; expected {LONG_VIDEO_SCHEMA}"
        )
    return payload


def _complete_native_av_latent(
    latent: Mapping[str, Any],
    *,
    label: str,
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None, int]:
    if not isinstance(latent, Mapping):
        raise ValueError(f"{label} is not a LATENT mapping")
    video, audio = nested_av_parts(dict(latent))
    if (
        video.ndim != 5
        or audio.ndim != 4
        or video.shape[0] != 1
        or audio.shape[0] != 1
        or video.shape[1] != 24
        or audio.shape[1:3] != (32, 2)
    ):
        raise ValueError(
            f"{label} is not the MiniMax H3 [1,24,T,H,W] + [1,32,2,T] contract"
        )
    video_t = int(video.shape[2])
    if video_t < VIDEO_PREFIX_LATENT_STEPS or (video_t - 2) % 5:
        raise ValueError(f"{label} video latent T={video_t} is not a complete H3 5n+2 grid")
    frames = pixel_frames_from_latent_t(video_t)
    expected_audio_t = round(frames / FPS * AUDIO_LATENT_FPS)
    if int(audio.shape[-1]) != expected_audio_t:
        raise ValueError(
            f"{label} audio latent T={audio.shape[-1]} does not match "
            f"round({frames}/24*40)={expected_audio_t}"
        )
    masks = _mask_parts(latent)
    if masks is not None and (masks[0].shape != video.shape or masks[1].shape != audio.shape):
        raise ValueError(f"{label} noise_mask shapes do not match its AV samples")
    return video, audio, masks, frames


def concat_native_h3_av_continuation(
    timeline_latent: Mapping[str, Any],
    continuation_segment: Mapping[str, Any],
    planner_report_json: str,
    conditioning_report_json: str,
    output_device: str = "cpu",
    audio_context_policy: str = "require_video_and_audio",
) -> tuple[dict[str, Any], int, int, int, int, str]:
    """Append one proven Long Video continuation without duplicating its head context.

    Long Video Conditioning re-injects the previous 5/22/39-frame tail as per-token
    MiniMax keyframes. This operation removes that full re-injected span, rather than
    the ordinary five-frame H3 prefix used by ``concat_native_h3_av_latents``.
    """

    if output_device not in {"cpu", "preserve_first"}:
        raise ValueError(f"unsupported continuation concat output_device: {output_device!r}")
    if audio_context_policy not in {"require_video_and_audio", "allow_video_only"}:
        raise ValueError(f"unsupported audio_context_policy: {audio_context_policy!r}")

    planner = _long_video_proof(planner_report_json, label="planner_report_json")
    conditioning = _long_video_proof(
        conditioning_report_json, label="conditioning_report_json"
    )
    try:
        segment_index = int(planner["segment_index"])
        context_frames = int(planner["context_frames"])
        render_frames = int(planner["render_frames"])
        final_frame_count = int(planner["final_frame_count"])
        hidden_tail_frames = int(planner.get("hidden_tail_frames", 0))
        timeline_start_seconds = float(planner["timeline_start_seconds"])
        timeline_end_seconds = float(planner["timeline_end_seconds"])
        trim_start_seconds = float(planner["trim_start_seconds"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("planner_report_json is missing required numeric continuation fields") from exc
    chain_id = str(planner.get("chain_id", "")).strip()
    if not chain_id:
        raise ValueError("planner_report_json is missing chain_id")
    if segment_index < 1:
        raise ValueError("Native continuation concat requires a continuation segment_index >= 1")
    if context_frames not in CONTEXT_FRAME_STEPS:
        raise ValueError("Long Video continuation context_frames must be 5, 22, or 39")
    context_video_t = CONTEXT_FRAME_STEPS[context_frames]

    try:
        conditioning_segment = int(conditioning["segment_index"])
        conditioning_context_frames = int(conditioning["context_frames"])
        conditioning_render_frames = int(conditioning["render_frames"])
        motion_keyframes = int(conditioning["motion_keyframes"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "conditioning_report_json is missing required continuation fields"
        ) from exc
    if conditioning_segment != segment_index:
        raise ValueError("Planner and Conditioning segment_index values do not match")
    if conditioning_context_frames != context_frames:
        raise ValueError("Planner and Conditioning context_frames values do not match")
    if conditioning.get("context_active") is not True:
        raise ValueError("Conditioning report does not prove an active previous-segment context")
    if motion_keyframes != context_video_t:
        raise ValueError(
            "Conditioning report motion_keyframes do not match the native context token count"
        )
    task = str(conditioning.get("task", ""))
    if not task.endswith("-motion"):
        raise ValueError("Conditioning report does not identify a Long Video motion task")
    context_audio = str(conditioning.get("context_audio", ""))
    timeline_audio_ref = conditioning.get("timeline_audio_ref") is True
    if context_audio not in {"video_and_audio", "video_only"}:
        raise ValueError("Conditioning report has an invalid context_audio mode")
    if audio_context_policy == "require_video_and_audio" and (
        context_audio != "video_and_audio" or not timeline_audio_ref
    ):
        raise ValueError(
            "audio_context_policy requires Long Video video_and_audio context with timeline_audio_ref"
        )
    if context_audio == "video_and_audio" and not timeline_audio_ref:
        raise ValueError("Conditioning report is missing its promised timeline audio reference")
    if context_audio == "video_only" and timeline_audio_ref:
        raise ValueError("Conditioning report contradicts its video_only audio context")

    timeline_video, timeline_audio, timeline_masks, timeline_frames = (
        _complete_native_av_latent(timeline_latent, label="timeline_latent")
    )
    segment_video, segment_audio, segment_masks, segment_frames = (
        _complete_native_av_latent(continuation_segment, label="continuation_segment")
    )
    if render_frames != segment_frames or conditioning_render_frames != segment_frames:
        raise ValueError(
            "Planner/Conditioning render_frames do not match continuation_segment geometry"
        )
    if context_frames >= segment_frames or context_video_t >= int(segment_video.shape[2]):
        raise ValueError("Continuation context must be shorter than continuation_segment")
    if tuple(timeline_video.shape[-2:]) != tuple(segment_video.shape[-2:]):
        raise ValueError("Continuation timeline and segment canvases do not match")
    if timeline_video.dtype != segment_video.dtype or timeline_audio.dtype != segment_audio.dtype:
        raise ValueError("Continuation timeline and segment AV dtypes do not match")
    if (timeline_masks is None) != (segment_masks is None):
        raise ValueError("Timeline and continuation must either both carry nested AV masks or neither")

    if abs(timeline_start_seconds * FPS - timeline_frames) > 1e-6:
        raise ValueError(
            "Planner timeline_start_seconds does not match the current timeline latent length"
        )
    if abs(trim_start_seconds * FPS - context_frames) > 1e-6:
        raise ValueError("Planner trim_start_seconds does not match context_frames")

    physical_append_frames = segment_frames - context_frames
    if final_frame_count < 1 or hidden_tail_frames < 0:
        raise ValueError("Planner final-frame accounting must remain positive")
    if final_frame_count + hidden_tail_frames != physical_append_frames:
        raise ValueError(
            "Planner final_frame_count + hidden_tail_frames does not equal the post-context span"
        )
    is_final_segment = bool(planner.get("is_final_segment", False))
    if not is_final_segment and hidden_tail_frames:
        raise ValueError("Only a final segment may carry hidden tail frames")
    if bool(planner.get("save_context", False)) == is_final_segment:
        raise ValueError("Planner save_context/is_final_segment contract is inconsistent")
    visible_frame_count = timeline_frames + final_frame_count
    if abs(timeline_end_seconds * FPS - visible_frame_count) > 1e-6:
        raise ValueError("Planner timeline_end_seconds does not match the visible timeline")

    previous = timeline_latent.get("t8_native_latent_continuation_concat")
    if previous is None:
        if segment_index != 1:
            raise ValueError(
                "A timeline without continuation provenance may only append segment_index 1"
            )
        previous_segment_count = 1
    else:
        if not isinstance(previous, Mapping) or previous.get("schema") != CONTINUATION_CONCAT_SCHEMA:
            raise ValueError("Timeline continuation provenance is invalid")
        if str(previous.get("chain_id", "")) != chain_id:
            raise ValueError("Continuation chain_id does not match the existing timeline")
        if int(previous.get("last_segment_index", -1)) + 1 != segment_index:
            raise ValueError("Continuation segment_index is not next in the existing timeline")
        if bool(previous.get("chain_closed", False)):
            raise ValueError("The existing continuation timeline is already closed by a final segment")
        if int(previous.get("physical_frame_count", -1)) != timeline_frames:
            raise ValueError("Timeline continuation provenance frame count is stale")
        previous_segment_count = int(previous.get("segment_count", 0))
        if previous_segment_count != segment_index:
            raise ValueError("Timeline continuation provenance segment count is inconsistent")

    physical_frame_count = timeline_frames + physical_append_frames
    expected_video_t = (
        int(timeline_video.shape[2]) + int(segment_video.shape[2]) - context_video_t
    )
    if pixel_frames_from_latent_t(expected_video_t) != physical_frame_count:
        raise RuntimeError("Continuation video phase does not map to the planned physical frames")
    expected_audio_t = round(physical_frame_count / FPS * AUDIO_LATENT_FPS)
    append_audio_t = expected_audio_t - int(timeline_audio.shape[-1])
    drop_audio_t = int(segment_audio.shape[-1]) - append_audio_t
    if append_audio_t < 1 or not 0 <= drop_audio_t < int(segment_audio.shape[-1]):
        raise RuntimeError(
            f"Continuation produced an impossible audio phase: append={append_audio_t}, "
            f"drop={drop_audio_t}"
        )

    target_device = (
        timeline_video.device if output_device == "preserve_first" else torch.device("cpu")
    )
    output_video = torch.cat(
        (
            timeline_video.to(device=target_device),
            segment_video[:, :, context_video_t:].to(device=target_device),
        ),
        dim=2,
    ).contiguous()
    output_audio = torch.cat(
        (
            timeline_audio.to(device=target_device),
            segment_audio[..., drop_audio_t:].to(device=target_device),
        ),
        dim=-1,
    ).contiguous()
    if int(output_video.shape[2]) != expected_video_t or int(output_audio.shape[-1]) != expected_audio_t:
        raise RuntimeError("Continuation concat tensor accounting changed unexpectedly")

    output = dict(timeline_latent)
    output["samples"] = comfy.nested_tensor.NestedTensor((output_video, output_audio))
    if timeline_masks is not None and segment_masks is not None:
        output["noise_mask"] = comfy.nested_tensor.NestedTensor(
            (
                torch.cat(
                    (
                        timeline_masks[0].to(device=target_device),
                        segment_masks[0][:, :, context_video_t:].to(device=target_device),
                    ),
                    dim=2,
                ).contiguous(),
                torch.cat(
                    (
                        timeline_masks[1].to(device=target_device),
                        segment_masks[1][..., drop_audio_t:].to(device=target_device),
                    ),
                    dim=-1,
                ).contiguous(),
            )
        )
    else:
        output.pop("noise_mask", None)

    report = {
        "schema": CONTINUATION_CONCAT_SCHEMA,
        "experimental": True,
        "chain_id": chain_id,
        "segment_count": previous_segment_count + 1,
        "last_segment_index": segment_index,
        "chain_closed": is_final_segment,
        "physical_frame_count": physical_frame_count,
        "visible_frame_count": visible_frame_count,
        "trim_tail_frames_after_decode": hidden_tail_frames,
        "total_video_latent_t": int(output_video.shape[2]),
        "total_audio_latent_t": int(output_audio.shape[-1]),
        "context_removed": {
            "frames": context_frames,
            "video_latent_steps": context_video_t,
            "audio_latent_steps": drop_audio_t,
            "audio_context_mode": context_audio,
        },
        "appended_segment": {
            "segment_index": segment_index,
            "source_frames": segment_frames,
            "source_video_t": int(segment_video.shape[2]),
            "source_audio_t": int(segment_audio.shape[-1]),
            "physical_append_frames": physical_append_frames,
            "visible_append_frames": final_frame_count,
            "append_video_t": int(segment_video.shape[2]) - context_video_t,
            "append_audio_t": append_audio_t,
        },
        "proof": {
            "planner_report_sha256": hashlib.sha256(
                planner_report_json.encode("utf-8")
            ).hexdigest().upper(),
            "conditioning_report_sha256": hashlib.sha256(
                conditioning_report_json.encode("utf-8")
            ).hexdigest().upper(),
            "motion_keyframes": motion_keyframes,
            "task": task,
        },
        "canvas_pixels": [int(output_video.shape[-1]) * 16, int(output_video.shape[-2]) * 16],
        "output_device": str(target_device),
        "output_bytes": _tensor_bytes(output_video) + _tensor_bytes(output_audio),
        "noise_masks_preserved": timeline_masks is not None,
        "metadata_preserved_from": "timeline_latent",
        "sampling_executed": False,
        "vae_decode_executed": False,
        "scientific_boundary": (
            "Planner and Conditioning reports prove the structural 5/22/39-frame Long Video "
            "contract, and the matching native video/audio span is removed exactly. Report JSON "
            "is wiring evidence rather than a cryptographic proof that a third-party sampler used "
            "the supplied conditioning. Human continuity, audio quality, lower VRAM and interrupted "
            "diffusion-NFE recovery remain unproven."
        ),
    }
    output["t8_native_latent_continuation_concat"] = report
    return (
        output,
        physical_frame_count,
        visible_frame_count,
        hidden_tail_frames,
        int(output_audio.shape[-1]),
        _json(report),
    )
