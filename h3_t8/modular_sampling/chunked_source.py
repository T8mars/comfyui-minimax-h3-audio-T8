"""Expose one existing Chunked first-pass time slice without running PASS2.

This is source preparation only. A later learned lift and chunk PASS2 must be
independently connected; the slice is not a completed-stage/cache receipt.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

import comfy.nested_tensor
import torch

from .. import chunked_two_pass_upscale_advanced as legacy
from .results import _input_identity


SCHEMAS = frozenset({
    legacy.PLAN_SCHEMA_V1,
    legacy.PLAN_SCHEMA_GLOBAL_NOISE_V2,
    legacy.PLAN_SCHEMA_LOW_SIGMA_V3,
    legacy.PLAN_SCHEMA_MASKED_LOW_SIGMA_V4,
})


@dataclass(frozen=True)
class ChunkedSourceSegment:
    plan_schema: str
    plan_sha256: str
    source_identity: dict[str, Any]
    slice_identity: dict[str, Any]
    index: int
    count: int
    start_token: int
    end_token: int
    start_frame: int
    end_frame: int
    audio_start: int
    audio_end: int
    target_width: int
    target_height: int


def _plan_sha(plan):
    return hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def slice_chunked_source(latent, plan, segment_index):
    """Return the exact pre-upscale AV chunk consumed by the legacy executor."""
    if type(plan) is not dict or plan.get("schema") not in SCHEMAS:
        raise ValueError("Expected a v1-v4 Chunked Two-Pass Plan")
    legacy._validate_chunked_source_geometry(plan, latent)
    samples = latent.get("samples") if type(latent) is dict else None
    video, audio = legacy._nested_parts(samples, name="source_latent")
    if (video.ndim != 5 or video.shape[0] != 1 or video.shape[1] != 24
            or audio.ndim != 4 or audio.shape[1:3] != (32, 2)):
        raise ValueError("Expected one native MiniMax H3 AV source")
    frames = legacy.frames_for_tokens(int(video.shape[2]))
    if (plan["schema"] != legacy.PLAN_SCHEMA_V1
            and plan.get("temporal_strategy", "full_clip_safe") == "full_clip_safe"):
        segments = [(0, 0, int(video.shape[2]), frames)]
    else:
        segments, _ = legacy.compute_temporal_segments(
            int(video.shape[2]), int(plan["temporal_chunk_frames"]),
            int(plan["temporal_overlap_frames"]),
        )
    if type(segment_index) is not int or not 0 <= segment_index < len(segments):
        raise ValueError("Chunked segment index is outside this plan's source timeline")
    start_token, start_frame, end_token, end_frame = segments[segment_index]
    audio_start = round(start_frame * legacy.FRAME_RESCALE)
    audio_end = min(audio.shape[-1], round(end_frame * legacy.FRAME_RESCALE))
    chunk_video = video[:, :, start_token:end_token].contiguous()
    chunk_audio = audio[..., audio_start:audio_end].contiguous()
    chunk = {"samples": comfy.nested_tensor.NestedTensor((chunk_video, chunk_audio))}
    mask_report = None
    if plan["schema"] == legacy.PLAN_SCHEMA_MASKED_LOW_SIGMA_V4:
        inherited, mask_report = legacy._normalize_inherited_video_mask(
            latent, video, policy=plan.get("video_mask_policy", "inherit_required"),
        )
        if inherited is not None:
            chunk["noise_mask"] = comfy.nested_tensor.NestedTensor((
                inherited[:, :, start_token:end_token].contiguous(),
                torch.ones_like(chunk_audio),
            ))
    spec = ChunkedSourceSegment(
        plan_schema=plan["schema"], plan_sha256=_plan_sha(plan),
        source_identity=_input_identity(latent), slice_identity=_input_identity(chunk),
        index=segment_index, count=len(segments), start_token=start_token,
        end_token=end_token, start_frame=start_frame, end_frame=end_frame,
        audio_start=audio_start, audio_end=audio_end,
        target_width=int(plan["target_width"]), target_height=int(plan["target_height"]),
    )
    report = {"schema": "t8.modular-sampling.chunked-source-slice.v1",
              "plan_schema": plan["schema"], "plan_sha256": spec.plan_sha256,
              "segment_index": segment_index, "segment_count": len(segments),
              "video_tokens": [start_token, end_token],
              "pixel_frames": [start_frame, end_frame],
              "audio_tokens_read_only": [audio_start, audio_end],
              "inherited_video_mask": mask_report,
              "sampled": False, "portable_stage_result": False}
    return chunk, spec, json.dumps(report, ensure_ascii=False, sort_keys=True)
