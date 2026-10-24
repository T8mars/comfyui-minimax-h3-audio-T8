"""Additive combined fresh-encoding entry for Director and ordinary graphs."""
import json

from comfy_api.latest import io

from .multikeyframe_advanced import build_multikeyframe_conditioning
from .nodes_multikeyframe_advanced import MiniMaxH3MultiKeyframeConditioningT8Advanced
from .nodes_prompt_relay_advanced import PromptRelayPlanIO
from .nodes_semantic_bridge import BridgeIO
from .prompt_relay_advanced import build_prompt_relay_conditioning


class MiniMaxH3DirectorKeyframeConditioningEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        original = MiniMaxH3MultiKeyframeConditioningT8Advanced.define_schema()
        return io.Schema(node_id=cls.__name__, display_name="H3 中间关键帧条件 · 可选 Relay (EXP/T8)",
            category=original.category, is_experimental=True,
            description="普通提示词无需Relay。连接Plan时复用同一次中间图新编码，绑定实际物理位置；MODEL和positive必须成对使用。旧节点和接口保持。",
            inputs=[*original.inputs, PromptRelayPlanIO.Input("prompt_relay_plan", optional=True),
                BridgeIO.Input("semantic_bridge", optional=True),
                io.Combo.Input("execution_mode", options=["report_only", "apply_exp"], default="report_only", advanced=True),
                io.Int.Input("query_chunk_rows", default=256, min=32, max=2048, step=32, advanced=True)],
            outputs=[io.Model.Output("model"), io.Conditioning.Output("positive"), io.Latent.Output("av_latent"),
                io.Audio.Output("mux_audio"), io.String.Output("conditioned_prompt"),
                io.String.Output("media_map_json"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, prompt_relay_plan=None, execution_mode="report_only", query_chunk_rows=256, **kwargs):
        if prompt_relay_plan is None:
            result = build_multikeyframe_conditioning(**kwargs)
            report = json.loads(result[7])
            report["keyframes"] = json.loads(result[6])
            return io.NodeOutput(*result[:6], json.dumps(report, ensure_ascii=False, sort_keys=True))
        kwargs.pop("prompt")  # The connected Relay plan is authoritative.
        kwargs.pop("length")
        options = {key: kwargs.pop(key) for key in (
            "first_frame_noise_aug", "last_frame_noise_aug", "reference_visual_noise_aug") if key in kwargs}
        return io.NodeOutput(*build_prompt_relay_conditioning(**kwargs, prompt_relay_plan=prompt_relay_plan,
            execution_mode=execution_mode, query_chunk_rows=query_chunk_rows, keyframe_options=options))


NODES = [MiniMaxH3DirectorKeyframeConditioningEXPT8]
