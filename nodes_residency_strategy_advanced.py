from __future__ import annotations

from comfy_api.latest import io

from .residency_strategy_advanced import (
    RESIDENCY_PRESETS,
    build_h3_residency_strategy,
)
from .vram_policy import VRAM_POLICY_TYPE


CATEGORY = "T8/MiniMax H3/Models/Advanced"
VRAMPolicyIO = io.Custom(VRAM_POLICY_TYPE)


class MiniMaxH3ResidencyStrategyT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3ResidencyStrategyT8Advanced",
            display_name="MiniMax H3 Residency Strategy / 低显存驻留策略 (Advanced EXP/T8)",
            description=(
                "Compiles a side-effect-free report_only/minimum_memory/balanced/faster policy "
                "from current ComfyUI and AIMDO telemetry. Applying it remains explicit through "
                "a compatible loader; this node never unloads other models or monkey-patches globally."
            ),
            category=CATEGORY,
            inputs=[
                io.Combo.Input(
                    "strategy",
                    options=list(RESIDENCY_PRESETS),
                    default="report_only",
                ),
                io.Int.Input("policy_epoch", default=0, min=0, max=0x7FFFFFFF, advanced=True),
                io.Model.Input("model", optional=True),
                io.Clip.Input("clip", optional=True),
            ],
            outputs=[
                VRAMPolicyIO.Output("vram_policy"),
                io.Float.Output("recommended_reserve_gib"),
                io.Boolean.Output("current_gate_pass"),
                io.String.Output("report_json"),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, strategy, policy_epoch, model=None, clip=None):
        return io.NodeOutput(
            *build_h3_residency_strategy(strategy, policy_epoch, model, clip)
        )


RESIDENCY_STRATEGY_ADVANCED_NODE_CLASSES = [
    MiniMaxH3ResidencyStrategyT8Advanced,
]
