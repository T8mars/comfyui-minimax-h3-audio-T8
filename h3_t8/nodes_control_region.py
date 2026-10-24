"""Append-only regional controls; old Fun loader/apply schemas are untouched."""
from collections.abc import Mapping
from copy import copy

from comfy_api.latest import io

from .control_region_runtime import ControlRegionRuntime, apply_h3_fun_control_region
from .control_regions import canonical
from .nodes_h3_fun_control_advanced import MiniMaxH3FunControlApplyT8Advanced


STATE = io.Custom("H3_T8_CONTROL_REGION_STATE")
CATEGORY = "T8/MiniMax H3/Control/Experimental"


class MiniMaxH3FunControlRegionApplyEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        original = MiniMaxH3FunControlApplyT8Advanced.define_schema()
        inputs = [copy(item) for item in original.inputs]
        for item in inputs:
            # Only the new regional entry drops inherited arbitrary widget
            # caps. Keep legacy Apply/defaults and real finite/shape checks.
            if item.id in {"width", "height"}:
                item.max = None
            elif item.id == "strength":
                item.min = item.max = None
        return io.Schema(node_id=cls.__name__, display_name="H3 Fun Control · 区域与裁框 (EXP/T8)",
            category=CATEGORY, is_experimental=True,
            description="可选目标区域MASK与源像素crop JSON。只门控目标视频控制残差，保留原控制塔及其它行贡献；不是最终像素/音频锁。LOW/HIGH各按实际画布接入。普通控制仍用原Apply。",
            inputs=[*inputs,
                    io.Mask.Input("region_mask", optional=True),
                    io.String.Input("crop_plan_json", default="", multiline=True, advanced=True),
                    io.String.Input("stage", default="SINGLE", advanced=True)],
            outputs=[io.Model.Output("model"), io.Conditioning.Output("positive"),
                     STATE.Output("region_state"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, model, positive, control_net, vae, control_video, width, height, length,
                control_kind, fit_mode, strength, start_percent, end_percent,
                region_mask=None, crop_plan_json="", stage="SINGLE"):
        return io.NodeOutput(*apply_h3_fun_control_region(
            model, positive, control_net, vae, control_video, width, height, length,
            control_kind, fit_mode, strength, start_percent, end_percent,
            region_mask=region_mask, crop_plan_json=crop_plan_json, stage=stage))


class MiniMaxH3ControlRegionAuditEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, display_name="H3 区域控制 · 采样后审计 (EXP/T8)",
            category=CATEGORY, is_experimental=True, is_output_node=True,
            description="after_sampling接对应采样LATENT，再接解码。核实际残差调用/区域行覆盖和未应用原因，不把配置成功冒充采样或画质通过。",
            inputs=[STATE.Input("region_state"), io.Latent.Input("after_sampling")],
            outputs=[io.Latent.Output("samples"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, region_state, after_sampling):
        if not isinstance(region_state, ControlRegionRuntime):
            raise TypeError("Connect the matching regional control runtime")
        if not isinstance(after_sampling, Mapping) or "samples" not in after_sampling:
            raise ValueError("after_sampling must be the actual sampler LATENT dependency")
        text = canonical(region_state.report())
        return io.NodeOutput(after_sampling, text, ui={"text": [text]})

    @classmethod
    def fingerprint_inputs(cls, **_kwargs):
        return float("nan")


NODES = [MiniMaxH3FunControlRegionApplyEXPT8, MiniMaxH3ControlRegionAuditEXPT8]
