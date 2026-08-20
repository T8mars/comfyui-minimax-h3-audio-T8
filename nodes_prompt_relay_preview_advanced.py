from __future__ import annotations

from comfy_api.latest import io

from .prompt_relay_advanced import PROMPT_RELAY_PLAN_TYPE
from .prompt_relay_preview_advanced import preview_prompt_relay_plan


CATEGORY = "T8/MiniMax H3/Conditioning/Experimental"
PromptRelayPlanIO = io.Custom(PROMPT_RELAY_PLAN_TYPE)


class MiniMaxH3PromptRelayPreviewT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3PromptRelayPreviewT8Advanced",
            display_name="MiniMax H3 Prompt Relay Preview / 时间线预检 (Advanced)",
            description=(
                "Validates an authenticated Prompt Relay plan and shows every event's "
                "frame/second range before any H3 model is loaded or sampling begins. "
                "Its plan output is an unchanged pass-through for inline use."
            ),
            category=CATEGORY,
            is_experimental=True,
            is_output_node=True,
            inputs=[PromptRelayPlanIO.Input("prompt_relay_plan")],
            outputs=[
                PromptRelayPlanIO.Output("prompt_relay_plan"),
                io.Boolean.Output("ready"),
                io.Int.Output("event_count"),
                io.String.Output("timeline_text"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, prompt_relay_plan):
        values = preview_prompt_relay_plan(prompt_relay_plan)
        return io.NodeOutput(*values, ui={"text": (values[3],)})


PROMPT_RELAY_PREVIEW_ADVANCED_NODE_CLASSES = [
    MiniMaxH3PromptRelayPreviewT8Advanced,
]
