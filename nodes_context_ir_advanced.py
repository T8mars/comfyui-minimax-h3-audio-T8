from __future__ import annotations

from comfy_api.latest import io

from .context_ir_advanced import (
    PROVIDER_MODES,
    build_context_ir,
    canonical_json,
    context_ir_creative_brief,
)
from .studio_advanced import BACKENDS, compile_prompt_packet
from .nodes_studio_advanced import CastIO, PromptPacketIO, SoundCanvasIO


CATEGORY = "T8/MiniMax H3/Studio/Experimental"
ContextIRIO = io.Custom("H3_T8_CONTEXT_IR")


class MiniMaxH3ContextIRProviderT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3ContextIRProviderT8Advanced",
            display_name="T8 Context IR Provider / 参考语义理解 (Advanced)",
            description=(
                "Validates a local Context IR by default. External OpenAI-compatible visual "
                "analysis is opt-in, requires explicit upload confirmation and reads its API key "
                "only from an environment variable. Raw audio is never uploaded."
            ),
            category=CATEGORY,
            is_experimental=True,
            is_output_node=True,
            inputs=[
                io.Combo.Input(
                    "provider_mode",
                    options=list(PROVIDER_MODES),
                    default="validate_local",
                ),
                io.String.Input(
                    "local_context_ir_json",
                    multiline=True,
                    default='{"context":{"scene":"describe the verified reference scene"}}',
                ),
                io.String.Input("creative_brief", multiline=True, dynamic_prompts=True),
                io.String.Input("exact_dialogue", multiline=True, default="", optional=True),
                io.String.Input("audio_transcript", multiline=True, default="", optional=True),
                io.String.Input(
                    "endpoint",
                    default="https://api.openai.com/v1/chat/completions",
                    advanced=True,
                ),
                io.String.Input("provider_model", default="gpt-4.1-mini", advanced=True),
                io.String.Input("api_key_env", default="OPENAI_API_KEY", advanced=True),
                io.Boolean.Input("confirm_external_upload", default=False),
                io.Int.Input("maximum_visual_frames", default=8, min=1, max=32, advanced=True),
                io.Int.Input("maximum_image_edge", default=768, min=128, max=2048, advanced=True),
                io.Int.Input("jpeg_quality", default=85, min=30, max=95, advanced=True),
                io.Float.Input("timeout_seconds", default=120.0, min=1.0, max=600.0, advanced=True),
                io.Int.Input(
                    "maximum_response_bytes",
                    default=262144,
                    min=4096,
                    max=4194304,
                    advanced=True,
                ),
                io.Image.Input("reference_images", optional=True),
            ],
            outputs=[
                ContextIRIO.Output("context_ir"),
                io.String.Output("context_ir_json"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(
        cls,
        provider_mode,
        local_context_ir_json,
        creative_brief,
        exact_dialogue,
        audio_transcript,
        endpoint,
        provider_model,
        api_key_env,
        confirm_external_upload,
        maximum_visual_frames,
        maximum_image_edge,
        jpeg_quality,
        timeout_seconds,
        maximum_response_bytes,
        reference_images=None,
    ):
        result, report = build_context_ir(
            provider_mode,
            local_context_ir_json,
            creative_brief,
            exact_dialogue,
            audio_transcript,
            endpoint,
            provider_model,
            api_key_env,
            confirm_external_upload,
            maximum_visual_frames,
            maximum_image_edge,
            jpeg_quality,
            timeout_seconds,
            maximum_response_bytes,
            reference_images,
        )
        return io.NodeOutput(result, canonical_json(result), canonical_json(report))


class MiniMaxH3ContextIRPromptCompilerT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3ContextIRPromptCompilerT8Advanced",
            display_name="T8 Context IR Prompt Compiler / IR提示词编译 (Advanced)",
            description=(
                "Deterministically compiles one reviewed Context IR through the existing T8 "
                "Prompt Packet contract. The provider cannot alter local paths or sampling."
            ),
            category=CATEGORY,
            is_experimental=True,
            is_output_node=True,
            inputs=[
                ContextIRIO.Input("context_ir"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Combo.Input("backend", options=list(BACKENDS), default="minimax_h3"),
                io.Float.Input(
                    "duration_seconds",
                    default=5.167,
                    min=0.001,
                    max=86400.0,
                    step=0.001,
                ),
                io.String.Input("aspect_ratio", default="16:9"),
                io.String.Input("dialogue", multiline=True, default="", optional=True),
                io.String.Input("negative_prompt", multiline=True, default="", optional=True),
                io.Boolean.Input("strict_exact_dialogue", default=True),
                CastIO.Input("cast", optional=True),
                SoundCanvasIO.Input("sound_canvas", optional=True),
                io.String.Input("cast_ids", default="", optional=True, advanced=True),
            ],
            outputs=[
                PromptPacketIO.Output("prompt_packet"),
                io.String.Output("compiled_prompt"),
                io.String.Output("negative_prompt"),
                io.String.Output("audio_prompt"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(
        cls,
        context_ir,
        prompt,
        backend,
        duration_seconds,
        aspect_ratio,
        dialogue,
        negative_prompt,
        strict_exact_dialogue,
        cast=None,
        sound_canvas=None,
        cast_ids="",
    ):
        exact = str(context_ir.get("exact_dialogue", ""))
        if exact and str(dialogue).strip() != exact:
            raise ValueError("dialogue must exactly match the Context IR exact_dialogue")
        ids = [value.strip() for value in str(cast_ids).split(",") if value.strip()]
        packet = compile_prompt_packet(
            prompt,
            backend,
            duration_seconds,
            aspect_ratio,
            dialogue,
            negative_prompt,
            strict_exact_dialogue,
            cast,
            ids,
            sound_canvas,
            context_ir_creative_brief(context_ir),
        )
        report = {
            "schema": context_ir["schema"],
            "context_ir_provenance": context_ir["provenance"],
            "packet_hash": packet["packet_hash"],
            "remote_control_authority": False,
            "api_key_serialized": False,
        }
        return io.NodeOutput(
            packet,
            packet["compiled_prompt"],
            packet["negative_prompt"],
            packet["audio_prompt"],
            canonical_json(report),
        )


CONTEXT_IR_ADVANCED_NODE_CLASSES = [
    MiniMaxH3ContextIRProviderT8Advanced,
    MiniMaxH3ContextIRPromptCompilerT8Advanced,
]
