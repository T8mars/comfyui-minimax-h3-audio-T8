"""Strict opt-in frozen first-pass source for cold Audio Refine tails."""
from __future__ import annotations

import json
import re

from comfy_api.latest import io

from ..core import nested_av_parts
from ..native_latent_checkpoint_advanced import load_native_h3_av_checkpoint
from ..nodes_native_latent_checkpoint_advanced import native_h3_checkpoint_storage_root


CATEGORY = "T8/MiniMax H3/Modular Sampling/Experimental"
FIRST_PASS_CHECKPOINT_ID = "audio_refine_firstpass"
SHA256 = re.compile(r"[0-9a-fA-F]{64}\Z")


class MiniMaxH3AudioRefineFrozenFirstPassLoadEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3AudioRefineFrozenFirstPassLoadEXPT8",
            display_name="H3 Audio Refine · Load Verified First Pass (T8 EXP)",
            category=CATEGORY, is_experimental=True,
            description="Load only a complete native H3 AV checkpoint whose explicit "
                        "external manifest, whole-file SHA-256 and audio_refine_firstpass "
                        "ID match. Empty evidence fails before any tail sampling.",
            inputs=[
                io.String.Input("checkpoint_path", default=""),
                io.String.Input("expected_manifest_json", default="", multiline=True),
                io.String.Input("expected_file_sha256", default=""),
                io.Int.Input("hash_chunk_megabytes", default=8, min=1, max=64, step=1),
            ],
            outputs=[
                io.Latent.Output("av_latent"), io.String.Output("status"),
                io.Boolean.Output("resume_verified"), io.String.Output("checkpoint_id"),
                io.String.Output("content_sha256"), io.String.Output("file_sha256"),
                io.String.Output("manifest_json"), io.String.Output("report_json"),
                io.Int.Output("video_width"), io.Int.Output("video_height"),
            ],
        )

    @classmethod
    def execute(cls, checkpoint_path, expected_manifest_json, expected_file_sha256,
                hash_chunk_megabytes=8):
        if not isinstance(checkpoint_path, str) or not checkpoint_path.strip():
            raise ValueError("Audio Refine frozen first pass requires checkpoint_path")
        if not isinstance(expected_manifest_json, str) or not expected_manifest_json.strip():
            raise ValueError("Audio Refine frozen first pass requires an external manifest")
        try:
            expected = json.loads(expected_manifest_json)
        except json.JSONDecodeError as error:
            raise ValueError("Audio Refine external manifest is not valid JSON") from error
        if not isinstance(expected, dict):
            raise ValueError("Audio Refine external manifest must be an object")
        if not isinstance(expected_file_sha256, str) or not SHA256.fullmatch(expected_file_sha256):
            raise ValueError("Audio Refine frozen first pass requires exact file SHA-256")
        loaded = load_native_h3_av_checkpoint(
            storage_root=native_h3_checkpoint_storage_root(),
            checkpoint_path=checkpoint_path,
            expected_manifest_json=expected_manifest_json,
            expected_file_sha256=expected_file_sha256,
            hash_chunk_megabytes=hash_chunk_megabytes,
        )
        if loaded[3] != FIRST_PASS_CHECKPOINT_ID:
            raise ValueError("Audio Refine checkpoint ID is not audio_refine_firstpass")
        if not loaded[2]:
            raise ValueError("Audio Refine first-pass checkpoint did not verify")
        video, _ = nested_av_parts(loaded[0])
        return io.NodeOutput(*loaded, int(video.shape[-1]) * 16,
                             int(video.shape[-2]) * 16)


class MiniMaxH3AudioRefineFrozenAVFrameGuardEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3AudioRefineFrozenAVFrameGuardEXPT8",
            display_name="H3 Audio Refine · Match Frozen Video Frames (T8 EXP)",
            category=CATEGORY, is_experimental=True,
            description="Fail before audio refinement when the verified frozen H3 AV "
                        "does not have this workflow's expected video frame count.",
            inputs=[io.Latent.Input("av_latent"),
                    io.Int.Input("expected_video_frame_count", default=124,
                                 min=1, max=3600, step=1)],
            outputs=[io.Latent.Output("av_latent"),
                     io.Int.Output("video_width"), io.Int.Output("video_height"),
                     io.Int.Output("video_frame_count")],
        )

    @classmethod
    def execute(cls, av_latent, expected_video_frame_count):
        video, _ = nested_av_parts(av_latent)
        latent_frames = int(video.shape[-3])
        if latent_frames == 1:
            video_frames = 1
        elif latent_frames >= 2 and (latent_frames - 2) % 5 == 0:
            video_frames = 5 + 17 * ((latent_frames - 2) // 5)
        else:
            raise ValueError("Audio Refine checkpoint has non-native H3 video frame geometry")
        if video_frames != expected_video_frame_count:
            raise ValueError("Audio Refine checkpoint video frame count differs from this resume graph")
        return io.NodeOutput(av_latent, int(video.shape[-1]) * 16,
                             int(video.shape[-2]) * 16, video_frames)


class MiniMaxH3AudioRefineContextCommitGateEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3AudioRefineContextCommitGateEXPT8",
            display_name="H3 Audio Refine · Commit Long Context After Freeze (T8 EXP)",
            category=CATEGORY, is_experimental=True,
            description="Only let the original long-video context saver write after the "
                        "frozen native AV checkpoint has been explicitly saved and verified.",
            inputs=[io.String.Input("checkpoint_status", force_input=True),
                    io.Boolean.Input("planned_save_context", force_input=True)],
            outputs=[io.Boolean.Output("save_context")],
        )

    @classmethod
    def execute(cls, checkpoint_status, planned_save_context):
        if checkpoint_status == "SAVED_VERIFIED":
            return io.NodeOutput(bool(planned_save_context))
        if checkpoint_status == "NOT_SAVED":
            return io.NodeOutput(False)
        raise ValueError("Audio Refine long context requires verified frozen checkpoint Save")


NODES = [MiniMaxH3AudioRefineFrozenFirstPassLoadEXPT8,
         MiniMaxH3AudioRefineFrozenAVFrameGuardEXPT8,
         MiniMaxH3AudioRefineContextCommitGateEXPT8]
