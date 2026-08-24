from __future__ import annotations

from comfy_api.latest import io

from .prompt_semantic_audit_advanced import audit_prompt_semantics


CATEGORY = "T8/MiniMax H3/Prompt/Advanced"


class MiniMaxH3PromptSemanticContractAuditT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3PromptSemanticContractAuditT8Advanced",
            display_name=(
                "MiniMax H3 Prompt Semantic Contract Audit / 提示词语义合同审计 "
                "(Advanced/T8)"
            ),
            description=(
                "Checks explicit required/forbidden phrase groups, exact dialogue blocks and "
                "media tags. It defaults to the original prompt until a mechanical pass is "
                "human-reviewed and explicitly accepted; it does not claim semantic equivalence."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.String.Input("original_prompt", multiline=True, dynamic_prompts=True),
                io.String.Input("candidate_prompt", multiline=True, dynamic_prompts=True),
                io.String.Input(
                    "semantic_contract_json",
                    multiline=True,
                    default=(
                        '{\n  "required_groups": [\n    {"id": "turn_motion", '
                        '"any_of": ["turns", "turning", "rotates", "spins", '
                        '"转身", "旋转"], "scope": "integrated"}\n  ],\n  '
                        '"forbidden_groups": [\n    {"id": "stillness", '
                        '"any_of": ["stands still", "motionless", "静止不动"], '
                        '"scope": "integrated"}\n  ]\n}'
                    ),
                    tooltip=(
                        "Strict local JSON. Each group requires id, any_of and scope. Scope is "
                        "full, integrated, soundscape or music. Empty anchors always ABSTAIN."
                    ),
                ),
                io.Boolean.Input("accept_candidate_after_review", default=False),
                io.Boolean.Input("preserve_exact_dialogue", default=True),
                io.Boolean.Input("preserve_source_media_tags", default=True),
                io.Boolean.Input("allow_new_media_tags", default=True, advanced=True),
            ],
            outputs=[
                io.String.Output("safe_prompt"),
                io.String.Output("candidate_prompt"),
                io.Boolean.Output("mechanical_pass"),
                io.String.Output("decision"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*audit_prompt_semantics(**kwargs))


PROMPT_SEMANTIC_AUDIT_ADVANCED_NODE_CLASSES = [
    MiniMaxH3PromptSemanticContractAuditT8Advanced
]
