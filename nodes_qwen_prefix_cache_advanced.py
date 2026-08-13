from __future__ import annotations

from comfy_api.latest import io

from .qwen_prefix_cache_advanced import build_prefix_cache_clip, canonical_json


CATEGORY = "T8/MiniMax H3/Conditioning/Experimental"
CACHE_TYPE = "H3_T8_QWEN_PREFIX_CACHE"
CacheIO = io.Custom(CACHE_TYPE)


class MiniMaxH3QwenReferencePrefixCacheT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3QwenReferencePrefixCacheT8Advanced",
            display_name="MiniMax H3 Qwen Reference Prefix Cache / 参考前缀缓存 (Advanced)",
            description=(
                "Opt-in, bounded CPU-memory cache for the exact visual-reference prefix of the "
                "native H3 Qwen3-VL encoder. Prompt text is still recomputed. It wraps only this "
                "CLIP output and never patches ComfyUI core files or the input CLIP object."
            ),
            category=CATEGORY,
            inputs=[
                io.Clip.Input("clip"),
                io.Combo.Input(
                    "mode",
                    options=["report_only", "memory_lru_exp"],
                    default="report_only",
                ),
                io.Int.Input("max_entries", default=1, min=1, max=16, step=1, advanced=True),
                io.Float.Input(
                    "maximum_cache_mib",
                    default=1024.0,
                    min=64.0,
                    max=65536.0,
                    step=64.0,
                    advanced=True,
                ),
                io.Int.Input(
                    "cache_epoch",
                    default=0,
                    min=0,
                    max=0x7FFFFFFF,
                    step=1,
                    advanced=True,
                    tooltip="Increment to create a fresh empty cache without mutating the input CLIP.",
                ),
            ],
            outputs=[
                io.Clip.Output("clip"),
                CacheIO.Output("cache_handle"),
                io.String.Output("report_json"),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, clip, mode, max_entries, maximum_cache_mib, cache_epoch):
        output, cache, report = build_prefix_cache_clip(
            clip,
            mode,
            max_entries,
            maximum_cache_mib,
            cache_epoch,
        )
        return io.NodeOutput(output, cache, canonical_json(report))


class MiniMaxH3QwenPrefixCacheStatsT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3QwenPrefixCacheStatsT8Advanced",
            display_name="MiniMax H3 Qwen Prefix Cache Stats / 前缀缓存统计 (Advanced)",
            description=(
                "Reads hit/miss/size counters from one bounded in-memory Qwen prefix cache. "
                "Connect Conditioning.report to after_report to force this node to run later."
            ),
            category=CATEGORY,
            inputs=[
                CacheIO.Input("cache_handle"),
                io.String.Input("after_report", optional=True, force_input=True),
            ],
            outputs=[io.String.Output("report_json")],
            is_experimental=True,
            is_output_node=True,
        )

    @classmethod
    def execute(cls, cache_handle, after_report=None):
        del after_report
        report_json = canonical_json(cache_handle.report())
        return io.NodeOutput(report_json, ui={"text": (report_json,)})

    @classmethod
    def fingerprint_inputs(cls, **_kwargs):
        return float("nan")


QWEN_PREFIX_CACHE_ADVANCED_NODE_CLASSES = [
    MiniMaxH3QwenReferencePrefixCacheT8Advanced,
    MiniMaxH3QwenPrefixCacheStatsT8Advanced,
]
