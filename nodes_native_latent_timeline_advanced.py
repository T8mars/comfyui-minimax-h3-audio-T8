from __future__ import annotations

from comfy_api.latest import io

from .native_latent_timeline_advanced import (
    audit_native_h3_av_latent_resume_manifest,
    concat_native_h3_av_continuation,
    concat_native_h3_av_latents,
)


CATEGORY = "T8/MiniMax H3/Long Video/Experimental"


class MiniMaxH3NativeLatentTimelineConcatT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3NativeLatentTimelineConcatT8Advanced",
            display_name="MiniMax H3 Native Latent Timeline Concat / 原生时格拼接 (Advanced EXP/T8)",
            description=(
                "Concatenates independently sampled complete H3 AV latents on the native temporal "
                "grid: later video segments drop the exact 5-frame/2-latent prefix and audio drops "
                "a cumulative 24fps-to-40Hz phase amount. No sampling or VAE decode is performed."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Latent.Input("first_segment"),
                io.Latent.Input("second_segment"),
                io.Combo.Input(
                    "output_device",
                    options=["cpu", "preserve_first"],
                    default="cpu",
                    tooltip=(
                        "cpu avoids allocating the combined output on GPU. Input tensors can still "
                        "remain cached by ComfyUI, so this is not a global unload operation."
                    ),
                ),
                io.Boolean.Input(
                    "require_identical_metadata",
                    default=False,
                    advanced=True,
                    tooltip=(
                        "When enabled, non-sample metadata must match exactly. Otherwise the first "
                        "segment's metadata is preserved and differences are reported."
                    ),
                ),
                io.Autogrow.Input(
                    "additional_segments",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Latent.Input("segment"), prefix="segment_", min=0, max=30
                    ),
                ),
            ],
            outputs=[
                io.Latent.Output("av_latent"),
                io.Int.Output("total_frame_count"),
                io.Int.Output("segment_count"),
                io.Int.Output("audio_latent_steps"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*concat_native_h3_av_latents(**kwargs))


class MiniMaxH3NativeLatentResumeManifestT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3NativeLatentResumeManifestT8Advanced",
            display_name=(
                "MiniMax H3 Native Latent Resume Manifest / 原生Latent恢复清单 "
                "(Advanced EXP/T8)"
            ),
            description=(
                "Builds a chunked exact SHA-256 manifest for one complete H3 AV latent. When a "
                "previous manifest is supplied, the default policy fails closed on any content, "
                "shape, dtype, mask or checkpoint-ID mismatch. It writes no files."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Latent.Input("av_latent"),
                io.String.Input(
                    "checkpoint_id",
                    default="timeline_checkpoint",
                    tooltip="Logical checkpoint identity; must match during resume verification.",
                ),
                io.String.Input(
                    "expected_manifest_json",
                    default="",
                    multiline=True,
                    advanced=True,
                    tooltip=(
                        "Leave empty to create a baseline. Paste the previously saved manifest to "
                        "verify a reloaded latent before continuing."
                    ),
                ),
                io.Combo.Input(
                    "mismatch_policy",
                    options=["error", "report_only"],
                    default="error",
                    advanced=True,
                    tooltip=(
                        "error is the safe resume default. report_only exposes a MISMATCH report "
                        "without allowing it to be mistaken for a verified checkpoint."
                    ),
                ),
                io.Int.Input(
                    "hash_chunk_megabytes",
                    default=8,
                    min=1,
                    max=64,
                    step=1,
                    advanced=True,
                    tooltip=(
                        "Upper bound for each temporary CPU hash chunk. It affects speed/memory, "
                        "not the resulting content SHA-256."
                    ),
                ),
            ],
            outputs=[
                io.String.Output("status"),
                io.Boolean.Output("resume_verified"),
                io.String.Output("content_sha256"),
                io.String.Output("manifest_json"),
            ],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*audit_native_h3_av_latent_resume_manifest(**kwargs))


class MiniMaxH3NativeLatentContinuationConcatT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3NativeLatentContinuationConcatT8Advanced",
            display_name=(
                "MiniMax H3 Native Latent Continuation Concat / "
                "原生续段去重拼接 (Advanced EXP/T8)"
            ),
            description=(
                "Appends one H3 Long Video continuation and removes its complete proven "
                "5/22/39-frame head context on the native video/audio clocks. Planner and "
                "Conditioning reports must be connected directly; no sampling or VAE decode runs."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Latent.Input(
                    "timeline_latent",
                    tooltip="Segment 0, or the output of this node from the previous continuation.",
                ),
                io.Latent.Input(
                    "continuation_segment",
                    tooltip="The sampled latent produced by the matching Long Video Conditioning.",
                ),
                io.String.Input(
                    "planner_report_json",
                    force_input=True,
                    tooltip="Connect Long Video Segment Planner report_json for this segment.",
                ),
                io.String.Input(
                    "conditioning_report_json",
                    force_input=True,
                    tooltip="Connect Long Video Conditioning report_json used for this segment.",
                ),
                io.Combo.Input(
                    "output_device",
                    options=["cpu", "preserve_first"],
                    default="cpu",
                    tooltip=(
                        "cpu keeps the combined output off GPU, but does not evict ComfyUI's "
                        "cached input latents or models."
                    ),
                ),
                io.Combo.Input(
                    "audio_context_policy",
                    options=["require_video_and_audio", "allow_video_only"],
                    default="require_video_and_audio",
                    advanced=True,
                    tooltip=(
                        "The safe default requires the previous audio tail to have been supplied "
                        "to Long Video Conditioning. allow_video_only still removes the overlap "
                        "duration but cannot claim audio continuity."
                    ),
                ),
            ],
            outputs=[
                io.Latent.Output("av_latent"),
                io.Int.Output("physical_frame_count"),
                io.Int.Output("visible_frame_count"),
                io.Int.Output("trim_tail_frames_after_decode"),
                io.Int.Output("audio_latent_steps"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*concat_native_h3_av_continuation(**kwargs))


NATIVE_LATENT_TIMELINE_ADVANCED_NODE_CLASSES = [
    MiniMaxH3NativeLatentTimelineConcatT8Advanced,
]


# Registered separately at the very end of nodes.py so positions 0-179 remain immutable.
NATIVE_LATENT_RESUME_ADVANCED_NODE_CLASSES = [
    MiniMaxH3NativeLatentResumeManifestT8Advanced,
]


# Registered after the checkpoint Save/Load nodes so all first 183 positions remain immutable.
NATIVE_LATENT_CONTINUATION_ADVANCED_NODE_CLASSES = [
    MiniMaxH3NativeLatentContinuationConcatT8Advanced,
]
