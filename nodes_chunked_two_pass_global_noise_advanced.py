from __future__ import annotations

from comfy_api.latest import io

from .chunked_two_pass_upscale_advanced import (
    build_chunked_two_pass_global_noise_plan,
    build_chunked_two_pass_low_sigma_plan,
    build_chunked_two_pass_masked_low_sigma_plan,
)
from .learned_latent_upscale_advanced import PRECISIONS, RELEASE_POLICIES
from .nodes_chunked_two_pass_upscale_advanced import (
    CATEGORY,
    PLAN_TYPE,
    _upscaler_options,
)


class MiniMaxH3ChunkedTwoPassGlobalNoisePlanT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3ChunkedTwoPassGlobalNoisePlanT8Advanced",
            display_name=(
                "MiniMax H3 Chunked Two-Pass Global Noise Plan (v2 Advanced EXP/T8)"
            ),
            description=(
                "Opt-in v2 plan: creates one full target-resolution video-noise field "
                "and slices it by exact coordinates. Temporal overlap inherits the "
                "previous output only in the explicit guarded_overlap_exp route. The "
                "default full_clip_safe route never stitches temporal trajectories. "
                "Use full_frame_safe for quality: independent spatial H3 Transformer "
                "tiles change local coordinates and remain diagnostic EXP only. Old v1 "
                "plans and workflows are unchanged. Audio noise stays zero and the "
                "executor returns the exact input audio tensor."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Combo.Input("model_name", options=_upscaler_options()),
                io.Int.Input("target_width", default=1280, min=32, max=16384, step=32),
                io.Int.Input("target_height", default=704, min=32, max=16384, step=32),
                io.Int.Input(
                    "temporal_chunk_frames", default=136, min=17, max=3600, step=17
                ),
                io.Int.Input(
                    "temporal_overlap_frames", default=17, min=0, max=1700, step=17
                ),
                io.Float.Input(
                    "anchor_strength", default=0.999, min=0.0, max=1.0, step=0.001
                ),
                io.Int.Input("tile_width", default=512, min=32, max=16384, step=32),
                io.Int.Input("tile_height", default=512, min=32, max=16384, step=32),
                io.Int.Input(
                    "spatial_overlap", default=128, min=0, max=4096, step=32
                ),
                io.Int.Input("spatial_fade", default=32, min=0, max=4096, step=32),
                io.Int.Input(
                    "minimum_tile_size", default=256, min=32, max=4096, step=32
                ),
                io.Combo.Input(
                    "overlap_blend",
                    options=["smoothstep", "linear"],
                    default="smoothstep",
                ),
                io.Combo.Input("precision", options=list(PRECISIONS), default="fp16"),
                io.Combo.Input(
                    "release_policy",
                    options=list(RELEASE_POLICIES),
                    default="offload_after",
                ),
                io.Combo.Input(
                    "spatial_strategy",
                    options=["full_frame_safe", "independent_tiles_exp"],
                    default="full_frame_safe",
                ),
                io.Combo.Input(
                    "temporal_strategy",
                    options=["full_clip_safe", "guarded_overlap_exp"],
                    default="full_clip_safe",
                ),
            ],
            outputs=[PLAN_TYPE.Output("plan"), io.String.Output("report_json")],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*build_chunked_two_pass_global_noise_plan(**kwargs))


class MiniMaxH3ChunkedTwoPassLowSigmaPlanT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3ChunkedTwoPassLowSigmaPlanT8Advanced",
            display_name=(
                "MiniMax H3 Complete First Pass + Low-Sigma Refine Plan "
                "(v3 Advanced EXP/T8)"
            ),
            description=(
                "Corrected append-only v3 route. Complete the first-pass trajectory "
                "to sigma zero before learned latent upscale, then connect a separate "
                "low-noise schedule. The published workflow uses ComfyUI "
                "BasicScheduler(simple, 3 steps, denoise 0.30). The default keeps one "
                "full-frame/full-clip H3 trajectory. Joint AV context is sampled during "
                "refine, while the exact first-pass audio tensor is returned for delivery. "
                "Old v1/v2 nodes and workflows are unchanged."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Combo.Input("model_name", options=_upscaler_options()),
                io.Int.Input("target_width", default=1152, min=32, max=16384, step=32),
                io.Int.Input("target_height", default=640, min=32, max=16384, step=32),
                io.Int.Input(
                    "temporal_chunk_frames", default=136, min=17, max=3600, step=17
                ),
                io.Int.Input(
                    "temporal_overlap_frames", default=17, min=0, max=1700, step=17
                ),
                io.Float.Input(
                    "anchor_strength", default=0.999, min=0.0, max=1.0, step=0.001
                ),
                io.Int.Input("tile_width", default=1152, min=32, max=16384, step=32),
                io.Int.Input("tile_height", default=640, min=32, max=16384, step=32),
                io.Int.Input(
                    "spatial_overlap", default=0, min=0, max=4096, step=32
                ),
                io.Int.Input("spatial_fade", default=0, min=0, max=4096, step=32),
                io.Int.Input(
                    "minimum_tile_size", default=256, min=32, max=4096, step=32
                ),
                io.Combo.Input(
                    "overlap_blend",
                    options=["smoothstep", "linear"],
                    default="smoothstep",
                ),
                io.Combo.Input("precision", options=list(PRECISIONS), default="fp16"),
                io.Combo.Input(
                    "release_policy",
                    options=list(RELEASE_POLICIES),
                    default="offload_after",
                ),
                io.Combo.Input(
                    "spatial_strategy",
                    options=["full_frame_safe", "independent_tiles_exp"],
                    default="full_frame_safe",
                ),
                io.Combo.Input(
                    "temporal_strategy",
                    options=["full_clip_safe", "guarded_overlap_exp"],
                    default="full_clip_safe",
                ),
                io.Combo.Input(
                    "second_pass_audio_policy",
                    options=["joint_av_preserve_input", "locked_input_audio"],
                    default="joint_av_preserve_input",
                    tooltip=(
                        "joint_av_preserve_input matches the upstream model-context "
                        "behavior but discards refined audio at output; locked_input_audio "
                        "is a diagnostic alternative."
                    ),
                ),
            ],
            outputs=[PLAN_TYPE.Output("plan"), io.String.Output("report_json")],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*build_chunked_two_pass_low_sigma_plan(**kwargs))


class MiniMaxH3ChunkedTwoPassMaskedLowSigmaPlanT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3ChunkedTwoPassMaskedLowSigmaPlanT8Advanced",
            display_name=(
                "MiniMax H3 Mask-Preserving Low-Sigma Refine Plan "
                "(v4 Advanced EXP/T8)"
            ),
            description=(
                "Append-only v4 route for a first-pass H3 latent that already carries "
                "a nested video noise mask. The learned upscaler resizes only the mask's "
                "spatial grid with nearest-exact; the second pass multiplies that mask "
                "with spatial and temporal ownership. Mask value 0 protects the original "
                "region and 1 permits refinement. Dynamic mismatched time masks are "
                "rejected instead of interpolated. The exact first-pass audio is still "
                "returned. Old v1/v2/v3 nodes and workflows are unchanged."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Combo.Input("model_name", options=_upscaler_options()),
                io.Int.Input("target_width", default=1152, min=32, max=16384, step=32),
                io.Int.Input("target_height", default=640, min=32, max=16384, step=32),
                io.Int.Input(
                    "temporal_chunk_frames", default=136, min=17, max=3600, step=17
                ),
                io.Int.Input(
                    "temporal_overlap_frames", default=17, min=0, max=1700, step=17
                ),
                io.Float.Input(
                    "anchor_strength", default=0.999, min=0.0, max=1.0, step=0.001
                ),
                io.Int.Input("tile_width", default=1152, min=32, max=16384, step=32),
                io.Int.Input("tile_height", default=640, min=32, max=16384, step=32),
                io.Int.Input("spatial_overlap", default=0, min=0, max=4096, step=32),
                io.Int.Input("spatial_fade", default=0, min=0, max=4096, step=32),
                io.Int.Input(
                    "minimum_tile_size", default=256, min=32, max=4096, step=32
                ),
                io.Combo.Input(
                    "overlap_blend",
                    options=["smoothstep", "linear"],
                    default="smoothstep",
                ),
                io.Combo.Input("precision", options=list(PRECISIONS), default="fp16"),
                io.Combo.Input(
                    "release_policy",
                    options=list(RELEASE_POLICIES),
                    default="offload_after",
                ),
                io.Combo.Input(
                    "spatial_strategy",
                    options=["full_frame_safe", "independent_tiles_exp"],
                    default="full_frame_safe",
                ),
                io.Combo.Input(
                    "temporal_strategy",
                    options=["full_clip_safe", "guarded_overlap_exp"],
                    default="full_clip_safe",
                ),
                io.Combo.Input(
                    "second_pass_audio_policy",
                    options=["joint_av_preserve_input", "locked_input_audio"],
                    default="joint_av_preserve_input",
                ),
                io.Combo.Input(
                    "video_mask_policy",
                    options=[
                        "inherit_required",
                        "inherit_if_present_else_generate_all",
                        "disabled",
                    ],
                    default="inherit_required",
                    tooltip=(
                        "Recommended: inherit_required. It refuses an unmasked input "
                        "instead of silently reopening the full background."
                    ),
                ),
            ],
            outputs=[PLAN_TYPE.Output("plan"), io.String.Output("report_json")],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*build_chunked_two_pass_masked_low_sigma_plan(**kwargs))


CHUNKED_TWO_PASS_GLOBAL_NOISE_ADVANCED_NODE_CLASSES = [
    MiniMaxH3ChunkedTwoPassGlobalNoisePlanT8Advanced,
    MiniMaxH3ChunkedTwoPassLowSigmaPlanT8Advanced,
    MiniMaxH3ChunkedTwoPassMaskedLowSigmaPlanT8Advanced,
]
