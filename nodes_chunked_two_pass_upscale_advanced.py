from __future__ import annotations

import folder_paths
from comfy_api.latest import io

from .chunked_two_pass_upscale_advanced import (
    build_chunked_two_pass_plan,
    execute_chunked_two_pass_upscale,
)
from .learned_latent_upscale_advanced import PRECISIONS, RELEASE_POLICIES


CATEGORY = "T8/MiniMax H3/Upscale/Advanced"
PLAN_TYPE = io.Custom("T8_H3_CHUNKED_TWO_PASS_PLAN")


def _upscaler_options() -> list[str]:
    names = list(folder_paths.get_filename_list("latent_upscale_models"))
    return names or ["minimax_h3_latent_upscaler_3d_fp16.safetensors"]


class MiniMaxH3ChunkedTwoPassPlanT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3ChunkedTwoPassPlanT8Advanced",
            display_name="MiniMax H3 Chunked Two-Pass Plan (Advanced EXP/T8)",
            description=(
                "Plans learned 3D latent upscale followed by temporal-chunk H3 "
                "re-sampling. The safe default keeps each temporal chunk full-frame; "
                "independent spatial canvases remain explicit EXP. There is no project "
                "pixel-area ceiling; memory and runtime remain user-owned."
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
            ],
            outputs=[PLAN_TYPE.Output("plan"), io.String.Output("report_json")],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*build_chunked_two_pass_plan(**kwargs))


class MiniMaxH3ChunkedTwoPassUpscaleT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3ChunkedTwoPassUpscaleT8Advanced",
            display_name="MiniMax H3 Chunked Two-Pass Upscale (Advanced EXP/T8)",
            description=(
                "Runs learned latent upscale per temporal chunk and restores the exact "
                "input audio tensor. Full-frame mode preserves global H3 spatial context; "
                "independent spatial tiles are research-only because their content can diverge."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Model.Input("model"),
                io.Conditioning.Input("conditioning"),
                io.Latent.Input("latent"),
                io.Noise.Input("noise"),
                io.Sampler.Input("sampler"),
                io.Sigmas.Input("sigmas"),
                PLAN_TYPE.Input("plan"),
                io.Conditioning.Input("negative", optional=True),
                io.Float.Input("cfg", default=1.0, min=0.0, max=100.0, step=0.1),
            ],
            outputs=[io.Latent.Output("latent"), io.String.Output("report_json")],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*execute_chunked_two_pass_upscale(**kwargs))


CHUNKED_TWO_PASS_UPSCALE_ADVANCED_NODE_CLASSES = [
    MiniMaxH3ChunkedTwoPassPlanT8Advanced,
    MiniMaxH3ChunkedTwoPassUpscaleT8Advanced,
]
