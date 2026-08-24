from __future__ import annotations

from comfy_api.latest import io

from .skin_finish_p2 import guard_skin_finish_candidate


CATEGORY = "T8/MiniMax H3/Post FX/Experimental"


class MiniMaxH3SkinFinishTextureGuardT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SkinFinishTextureGuardT8Advanced",
            display_name=(
                "MiniMax H3 Skin Finish Texture Guard / 肤质纹理保护 (Advanced EXP)"
            ),
            description=(
                "Append-only fail-closed guard for an existing Skin Finish candidate. Source "
                "deep shadows and near-clipped highlights are preserved; frames that introduce "
                "clipping or fall below a source-relative high-frequency floor return to source. "
                "It is not a beauty score, semantic parser, deblur or pore generator."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Image.Input("source_frames"),
                io.Image.Input("candidate_frames"),
                io.Mask.Input("used_skin_mask"),
                io.Float.Input(
                    "shadow_protection",
                    default=0.10,
                    min=0.0,
                    max=0.45,
                    step=0.01,
                    tooltip="Source luma at or below this value is preserved.",
                ),
                io.Float.Input(
                    "highlight_protection",
                    default=0.94,
                    min=0.55,
                    max=1.0,
                    step=0.01,
                    tooltip="Source luma at or above this value is preserved.",
                ),
                io.Float.Input(
                    "transition_width", default=0.06, min=0.001, max=0.25, step=0.005
                ),
                io.Float.Input(
                    "minimum_texture_ratio",
                    default=0.78,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "Per-frame high-pass RMS floor relative to source. This is a hard "
                        "retention guard, not a perceptual quality score."
                    ),
                ),
                io.Float.Input(
                    "minimum_reference_texture",
                    default=0.003,
                    min=0.0,
                    max=0.25,
                    step=0.001,
                    advanced=True,
                ),
                io.Float.Input(
                    "maximum_new_clipped_fraction",
                    default=0.0005,
                    min=0.0,
                    max=0.25,
                    step=0.0005,
                    tooltip="Maximum newly clipped masked-pixel fraction before frame fallback.",
                ),
                io.Float.Input(
                    "clipping_epsilon",
                    default=1.0 / 255.0,
                    min=0.0001,
                    max=0.05,
                    step=0.0001,
                    advanced=True,
                ),
                io.Int.Input("texture_radius", default=1, min=1, max=8, advanced=True),
                io.Int.Input("chunk_frames", default=4, min=1, max=32, advanced=True),
                io.Boolean.Input(
                    "accept_candidate",
                    default=False,
                    tooltip="Source remains selected until the guarded candidate is reviewed.",
                ),
                io.Audio.Input("audio", optional=True),
            ],
            outputs=[
                io.Image.Output("guarded_candidate"),
                io.Image.Output("source"),
                io.Image.Output("selected"),
                io.Audio.Output("audio"),
                io.Mask.Output("effective_mask"),
                io.Mask.Output("rejected_mask"),
                io.Image.Output("difference"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*guard_skin_finish_candidate(**kwargs))


SKIN_FINISH_P2_NODE_CLASSES = [MiniMaxH3SkinFinishTextureGuardT8Advanced]
