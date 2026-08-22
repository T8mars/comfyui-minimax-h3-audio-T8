from __future__ import annotations

from comfy_api.latest import io

from .external_blockswap_advanced import build_external_blockswap_config


CATEGORY = "T8/MiniMax H3/Models/Experimental"
ExternalSwapIO = io.Custom("MINIMAX_H3_SWAP")


class MiniMaxH3ExternalBlockSwapBridgeT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3ExternalBlockSwapBridgeT8Advanced",
            display_name="MiniMax H3 External BlockSwap Bridge / 流式驻留桥接 (Advanced)",
            description=(
                "Builds a compatible BlockSwap payload for the optional external streaming "
                "MiniMaxH3KSampler. It never patches or replaces ComfyUI's official MODEL path."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Combo.Input(
                    "profile",
                    options=["upstream_default_auto", "manual"],
                    default="upstream_default_auto",
                    tooltip="默认档严格复现上游当前默认值；manual才读取下方数值。",
                ),
                io.Int.Input("block_to_swap", default=47, min=0, max=50, step=1),
                io.Int.Input("hot_blocks", default=0, min=0, max=50, step=1),
                io.Boolean.Input("prefetch", default=True),
                io.Int.Input("prefetch_count", default=2, min=1, max=8, step=1),
                io.Boolean.Input("pin_memory", default=True),
                io.Int.Input("disk_workers", default=2, min=1, max=16, step=1),
                io.Boolean.Input(
                    "auto_vram",
                    default=True,
                    tooltip="让上游运行时根据当前序列与可用显存缩小驻留窗口。",
                ),
                io.Float.Input(
                    "vram_reserve_mb",
                    default=0.0,
                    min=0.0,
                    max=131072.0,
                    step=128.0,
                    advanced=True,
                ),
                io.Boolean.Input("offload_dit", default=False, advanced=True),
                io.Combo.Input(
                    "dtype",
                    options=["bfloat16", "float16", "float32"],
                    default="bfloat16",
                ),
                io.Boolean.Input(
                    "require_external_runtime",
                    default=True,
                    tooltip="上游流式加载器未安装时提前报错。",
                ),
            ],
            outputs=[
                ExternalSwapIO.Output("block_swap_args"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*build_external_blockswap_config(**kwargs))


EXTERNAL_BLOCKSWAP_ADVANCED_NODE_CLASSES = [
    MiniMaxH3ExternalBlockSwapBridgeT8Advanced
]
