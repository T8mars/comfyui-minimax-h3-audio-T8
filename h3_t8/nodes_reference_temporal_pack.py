"""Opt-in paired outputs; original empty/0/5 route stays untouched."""
from comfy_api.latest import io

from .reference_temporal_pack import canonical
from .reference_temporal_pack_advanced import (
    audit_reference_temporal_pack, configure_reference_temporal_pack,
)


class MiniMaxH3ReferenceTemporalPackEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.__name__, display_name="H3 参考时间抽样 · Reference Temporal Pack (EXP/T8)",
            category="T8/MiniMax H3/References/Experimental", is_experimental=True,
            description=("只抽样已编码参考视频latent时间帧，保留原物理RoPE。0/5原对象直通；"
                         "不改变生成latent、Qwen取帧、参考音频、采样器或步数。"
                         "MODEL与positive必须成对使用，LOW/HIGH分别接各自实际条件。"
                         "未知执行路径保留原图并警告；不保证速度、显存或画质，不认证便携缓存。"),
            inputs=[io.Model.Input("model"), io.Conditioning.Input("positive"),
                    io.Combo.Input("ratio", options=["0/5", "1/5", "2/5", "3/5"], default="0/5"),
                    io.String.Input("phase", default="SINGLE", advanced=True)],
            outputs=[io.Model.Output("model"), io.Conditioning.Output("positive"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, model, positive, ratio="0/5", phase="SINGLE"):
        selected, packed, report = configure_reference_temporal_pack(model, positive, ratio, phase=phase)
        return io.NodeOutput(selected, packed, canonical(report))


class MiniMaxH3ReferenceTemporalPackAuditEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.__name__, display_name="H3 参考时间抽样 · 实际布局审计 (EXP/T8)",
            category="T8/MiniMax H3/References/Experimental", is_experimental=True, is_output_node=True,
            description="after_sampling仅建立顺序。报告实际MODEL调用和布局，不推断NFE、GPU性能或画质通过。",
            inputs=[io.Model.Input("model"), io.AnyType.Input("after_sampling")],
            outputs=[io.String.Output("report_json")])

    @classmethod
    def execute(cls, model, after_sampling):
        report = canonical(audit_reference_temporal_pack(model))
        return io.NodeOutput(report, ui={"text": [report]})

    @classmethod
    def fingerprint_inputs(cls, **_kwargs):
        return float("nan")


NODES = [MiniMaxH3ReferenceTemporalPackEXPT8, MiniMaxH3ReferenceTemporalPackAuditEXPT8]
