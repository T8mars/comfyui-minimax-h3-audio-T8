"""Explicit opt-in MODEL projection working sets, never a replacement sampler."""
from comfy_api.latest import io

from .h3_projection_chunks import ProjectionPlan
from .projection_chunks_advanced import (
    audit_projection_chunks, canonical, configure_projection_chunks,
)


CATEGORY = "T8/MiniMax H3/Models/Experimental"


class MiniMaxH3ProjectionChunksEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.__name__, display_name="H3 投影分块 · Projection Chunks (EXP/T8)",
            category=CATEGORY, is_experimental=True,
            description=(
                "可选分块norm/QKV、输出投影、输入和最终层；完整K/V仍可见。"
                "Relay及未知注意力保留完整Q委托，不清模型或LoRA。"
                "LOW/HIGH可从共用底模各接一份，独立设置；不会改变采样器、步数、音频或3D放大器。"
                "按实际调用报告覆盖，不保证省显存或提速，当前不认证跨进程缓存。"
            ),
            inputs=[
                io.Model.Input("model"),
                io.Boolean.Input("enabled", default=True),
                io.Int.Input("rows", default=1024, min=1, step=1),
                io.Int.Input("query_rows", default=1024, min=1, step=1),
                io.Boolean.Input("stage_queries", default=False, advanced=True,
                                 tooltip="只对已核实dense路径把Q暂存CPU；增加主存与传输，不缩减K/V。"),
                io.Boolean.Input("norm_qkv", default=True, advanced=True),
                io.Boolean.Input("output", default=True, advanced=True),
                io.Boolean.Input("embedding", default=True, advanced=True),
                io.Boolean.Input("final", default=True, advanced=True),
                io.String.Input("phase", default="SINGLE", advanced=True),
            ],
            outputs=[io.Model.Output("model"), io.String.Output("report_json")],
        )

    @classmethod
    def execute(cls, model, enabled=True, rows=1024, query_rows=1024,
                stage_queries=False, norm_qkv=True, output=True, embedding=True,
                final=True, phase="SINGLE"):
        plan = ProjectionPlan(rows=rows, query_rows=query_rows, stage_queries=stage_queries,
                              norm_qkv=norm_qkv, output=output, embedding=embedding, final=final)
        selected, report = configure_projection_chunks(model, plan, enabled=enabled, phase=phase)
        return io.NodeOutput(selected, canonical(report))


class MiniMaxH3ProjectionChunksAuditEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.__name__, display_name="H3 投影分块 · 实际调用审计 (EXP/T8)",
            category=CATEGORY, is_experimental=True, is_output_node=True,
            description=(
                "model接所选分块MODEL；after_sampling接对应采样结果，仅建立执行顺序。"
                "读取真实调用、行数、委托和清理状态，不从LATENT或Stage类型推断采样成功、NFE、画质或峰值显存。"
            ),
            inputs=[io.Model.Input("model"), io.AnyType.Input("after_sampling")],
            outputs=[io.String.Output("report_json")],
        )

    @classmethod
    def execute(cls, model, after_sampling):
        report = canonical(audit_projection_chunks(model))
        return io.NodeOutput(report, ui={"text": [report]})

    @classmethod
    def fingerprint_inputs(cls, **_kwargs):
        return float("nan")


NODES = [MiniMaxH3ProjectionChunksEXPT8, MiniMaxH3ProjectionChunksAuditEXPT8]
