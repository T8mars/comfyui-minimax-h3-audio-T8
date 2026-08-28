from __future__ import annotations

from comfy_api.latest import io

from .freenoise_advanced import MODES, build_free_noise_model


CATEGORY = "T8/MiniMax H3/Long Video/Experimental"


class MiniMaxH3FreeNoiseLongVideoT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3FreeNoiseLongVideoT8Advanced",
            display_name="MiniMax H3 FreeNoise Long Video / 长视频噪声重排 (Advanced EXP)",
            description=(
                "Adds deterministic FreeNoise-style video-noise rescheduling to either T8 "
                "in-node long-video runner. It composes with Prompt Relay and Enhance-A-Video "
                "because it does not own attention. Audio noise remains native and independent."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input("mode", options=list(MODES), default="paper_permutation"),
                io.Int.Input("base_seed", default=123456789, min=0, max=0xFFFFFFFFFFFFFFFF),
                io.Float.Input(
                    "reuse_ratio",
                    default=0.65,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "Used only by variance_preserving_blend. paper_permutation always uses "
                        "the paper-style full shared pool (1.0)."
                    ),
                ),
            ],
            outputs=[io.Model.Output("model"), io.String.Output("report_json")],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*build_free_noise_model(**kwargs))


FREENOISE_ADVANCED_NODE_CLASSES = [MiniMaxH3FreeNoiseLongVideoT8Advanced]

