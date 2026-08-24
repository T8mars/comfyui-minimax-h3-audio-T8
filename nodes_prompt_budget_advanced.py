from __future__ import annotations

from comfy_api.latest import io

from .prompt_budget_advanced import compile_prompt_budget


CATEGORY = "T8/MiniMax H3/Prompt/Advanced"


class MiniMaxH3PromptBudgetCompilerT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3PromptBudgetCompilerT8Advanced",
            display_name="MiniMax H3 Prompt Budget + Role Compiler / 提示词预算与角色编译 (Advanced/T8)",
            description=(
                "Never truncates. The default 7000-character audit ceiling matches the current "
                "official H3 CLI submission rule without claiming a local tokenizer hard limit. "
                "Also audits token budgets, media count/order, assignment coverage and "
                "subject-to-picture/video/audio bindings."
            ),
            category=CATEGORY,
            is_experimental=True,
            is_output_node=True,
            inputs=[
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Int.Input(
                    "character_limit",
                    default=7000,
                    min=1,
                    max=1000000,
                    tooltip=(
                        "Default 7000 matches the current official MiniMax H3 CLI submission "
                        "limit. It is not a detected hard limit of the local open-weight "
                        "tokenizer. Raising it opts out of official submission compatibility; "
                        "the report warns and the prompt is never truncated."
                    ),
                ),
                io.Int.Input(
                    "token_limit",
                    default=0,
                    min=0,
                    max=1000000,
                    tooltip="0 reports token counts without applying a token ceiling.",
                ),
                io.Int.Input("picture_count", default=0, min=0, max=99),
                io.Int.Input("video_count", default=0, min=0, max=99),
                io.Int.Input("audio_count", default=0, min=0, max=99),
                io.String.Input(
                    "media_assignments_json",
                    multiline=True,
                    default=(
                        '{"subjects":[{"subject_id":"lead","picture_ordinal":1,'
                        '"audio_ordinal":1,"role":"primary_character"}]}'
                    ),
                ),
                io.Boolean.Input("append_role_bindings", default=True),
                io.Boolean.Input("allow_shared_audio", default=False, advanced=True),
                io.Boolean.Input("require_exact_token_count", default=False, advanced=True),
                io.Clip.Input("clip", optional=True),
            ],
            outputs=[
                io.String.Output("compiled_prompt"),
                io.Boolean.Output("pass_audit"),
                io.String.Output("decision"),
                io.Int.Output("character_count"),
                io.Int.Output("estimated_token_count"),
                io.Int.Output("exact_token_count"),
                io.String.Output("media_map_json"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*compile_prompt_budget(**kwargs))


PROMPT_BUDGET_ADVANCED_NODE_CLASSES = [MiniMaxH3PromptBudgetCompilerT8Advanced]
