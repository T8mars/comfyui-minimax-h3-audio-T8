from __future__ import annotations

from comfy_api.latest import io

from .prompt_rewriter_8b import (
    BASE_REPOSITORY,
    DEFAULT_ADAPTER_FOLDER,
    DEFAULT_BASE_FOLDER,
    TASK_LABELS,
    UPSTREAM_REPOSITORY,
    rewrite_prompt_8b,
    unload_prompt_rewriter_cache,
)


CATEGORY = "T8/MiniMax H3/Conditioning/Experimental"


class MiniMaxH3PromptRewriter8BT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3PromptRewriter8BT8Advanced",
            display_name="MiniMax H3 Prompt Rewriter 8B / 多模态提示词改写 (Advanced)",
            description=(
                "Runs the pinned LightX2V Qwen3-VL-8B LoRA for T2VA/I2VA/L2VA/FL2VA. "
                "It is separate from the H3 32B CLIP because their parameter shapes are "
                "not LoRA-compatible. Unload-after-generate is enabled by default."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.String.Input(
                    "prompt",
                    multiline=True,
                    dynamic_prompts=True,
                    tooltip="原始短提示词；对白、歌词和画面文字会要求保持原文。",
                ),
                io.Combo.Input(
                    "task",
                    options=list(TASK_LABELS),
                    default="T2VA — 文生音视频",
                ),
                io.Combo.Input(
                    "resolution",
                    options=["adaptive", "21:9", "16:9", "4:3", "1:1", "3:4", "9:16"],
                    default="16:9",
                ),
                io.Int.Input("duration", default=10, min=4, max=15, step=1),
                io.String.Input(
                    "base_model_path",
                    default=DEFAULT_BASE_FOLDER,
                    tooltip=(
                        "相对路径从 ComfyUI/models/text_encoders 解析，也可填绝对目录。"
                        f"精确基座为 {BASE_REPOSITORY}。"
                    ),
                ),
                io.String.Input(
                    "adapter_path",
                    default=DEFAULT_ADAPTER_FOLDER,
                    tooltip=(
                        "相对路径从 ComfyUI/models/loras 解析，也可填绝对目录。"
                        f"精确LoRA为 {UPSTREAM_REPOSITORY}。"
                    ),
                ),
                io.Combo.Input(
                    "load_policy",
                    options=["auto_cpu_offload", "gpu_only", "cpu_only"],
                    default="auto_cpu_offload",
                    tooltip="16GB显卡建议自动CPU分片；gpu_only需要足够显存。",
                ),
                io.Combo.Input("dtype", options=["bfloat16", "float16"], default="bfloat16"),
                io.Combo.Input("decoding", options=["greedy", "sample"], default="greedy"),
                io.Int.Input("seed", default=42, min=0, max=0x7FFFFFFFFFFFFFFF),
                io.Int.Input(
                    "max_new_tokens",
                    default=1024,
                    min=128,
                    max=8192,
                    step=128,
                    tooltip=(
                        "默认1024兼顾完整度与16GB分片速度；过低可能截断三段结构，"
                        "过高只提高上限而不会强迫模型生成到上限。"
                    ),
                ),
                io.Float.Input("temperature", default=0.7, min=0.01, max=2.0, step=0.01, advanced=True),
                io.Float.Input("top_p", default=0.8, min=0.01, max=1.0, step=0.01, advanced=True),
                io.Int.Input(
                    "min_image_pixels",
                    default=256 * 256,
                    min=32 * 32,
                    max=2048 * 2048,
                    step=1024,
                    advanced=True,
                ),
                io.Int.Input(
                    "max_image_pixels",
                    default=1024 * 1024,
                    min=32 * 32,
                    max=4096 * 4096,
                    step=1024,
                    advanced=True,
                ),
                io.Boolean.Input(
                    "unload_after_generate",
                    default=True,
                    tooltip="生成、报错或OOM后都尝试只释放本节点的模型和缓存。",
                ),
                io.Boolean.Input(
                    "free_comfy_models_before_load",
                    default=True,
                    tooltip="加载8B前释放ComfyUI当前模型，为16GB显卡留空间；不会在生成后全局卸载。",
                ),
                io.Boolean.Input(
                    "allow_hub_download",
                    default=False,
                    advanced=True,
                    tooltip="默认只允许本地模型，避免运行时意外下载大文件。",
                ),
                io.Image.Input("first_frame", optional=True),
                io.Image.Input("last_frame", optional=True),
            ],
            outputs=[
                io.String.Output("enhanced_prompt"),
                io.String.Output("integrated_multimodal_description"),
                io.String.Output("overall_soundscape"),
                io.String.Output("non_diegetic_music"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*rewrite_prompt_8b(**kwargs))


class MiniMaxH3PromptRewriterUnloadT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3PromptRewriterUnloadT8Advanced",
            display_name="MiniMax H3 Prompt Rewriter Unload / 手动卸载 (Advanced)",
            description="Releases only the optional Prompt Rewriter 8B cache retained by this node pack.",
            category=CATEGORY,
            is_experimental=True,
            is_output_node=True,
            inputs=[io.Boolean.Input("unload_now", default=True)],
            outputs=[io.String.Output("report_json")],
        )

    @classmethod
    def execute(cls, unload_now):
        if not unload_now:
            return io.NodeOutput('{"status":"bypass"}')
        return io.NodeOutput(unload_prompt_rewriter_cache())


PROMPT_REWRITER_8B_ADVANCED_NODE_CLASSES = [
    MiniMaxH3PromptRewriter8BT8Advanced,
    MiniMaxH3PromptRewriterUnloadT8Advanced,
]
