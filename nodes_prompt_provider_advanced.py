from __future__ import annotations

from comfy_api.latest import io

from .prompt_provider_advanced import PROVIDER_MODES, rewrite_prompt_provider
from .prompt_rewriter_8b import TASK_LABELS


CATEGORY = "T8/MiniMax H3/Prompt/Advanced"


class MiniMaxH3PromptProviderRouterT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3PromptProviderRouterT8Advanced",
            display_name="MiniMax H3 Prompt Provider Router / 提示词服务路由 (Advanced/T8)",
            description=(
                "Uses the pinned H3 three-field rewriting contract with local passthrough, an "
                "OpenAI-compatible server (OpenAI/LM Studio/llama.cpp), or Ollama. Network use "
                "is explicit; keys come only from environment variables."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Combo.Input(
                    "provider_mode",
                    options=list(PROVIDER_MODES),
                    default=PROVIDER_MODES[0],
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
                    "endpoint",
                    default="",
                    tooltip=(
                        "Empty uses 127.0.0.1:1234 for OpenAI-compatible or 127.0.0.1:11434 "
                        "for Ollama. Non-loopback endpoints require HTTPS and explicit permission."
                    ),
                ),
                io.String.Input("provider_model", default="local-model"),
                io.String.Input(
                    "api_key_env",
                    default="",
                    tooltip="Environment-variable name only; never paste the secret value here.",
                ),
                io.Boolean.Input("confirm_provider_request", default=False),
                io.Boolean.Input("allow_remote_endpoint", default=False, advanced=True),
                io.Int.Input("max_new_tokens", default=1024, min=1, max=32768, step=1),
                io.Float.Input("temperature", default=0.0, min=0.0, max=2.0, step=0.01),
                io.Float.Input("top_p", default=1.0, min=0.01, max=1.0, step=0.01),
                io.Int.Input(
                    "maximum_image_edge", default=768, min=128, max=2048, advanced=True
                ),
                io.Int.Input("jpeg_quality", default=85, min=30, max=95, advanced=True),
                io.Float.Input(
                    "timeout_seconds", default=120.0, min=1.0, max=600.0, advanced=True
                ),
                io.Int.Input(
                    "maximum_response_bytes",
                    default=262144,
                    min=4096,
                    max=4194304,
                    advanced=True,
                ),
                io.Boolean.Input("strict_output_contract", default=True),
                io.String.Input(
                    "ollama_keep_alive",
                    default="0",
                    advanced=True,
                    tooltip="0 asks Ollama to unload after the response; ignored by other providers.",
                ),
                io.Image.Input("first_frame", optional=True),
                io.Image.Input("last_frame", optional=True),
                io.Int.Input(
                    "contract_repair_attempts",
                    default=0,
                    min=0,
                    max=2,
                    step=1,
                    advanced=True,
                    tooltip=(
                        "Optional deterministic repair requests after contract validation fails. "
                        "Default 0 preserves old workflow request count; retries do not resend images."
                    ),
                ),
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
        return io.NodeOutput(*rewrite_prompt_provider(**kwargs))


PROMPT_PROVIDER_ADVANCED_NODE_CLASSES = [MiniMaxH3PromptProviderRouterT8Advanced]
