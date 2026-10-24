"""Explicit MODEL-local INT8 row selection and actual-call observation."""

import json

from comfy_api.latest import io

from .int8_row_chunk import Int8RowPlan
from .int8_row_chunk_advanced import (
    audit_int8_row_chunks,
    configure_int8_row_chunks,
)


CATEGORY = "T8/MiniMax H3/Models/Experimental"


def _report(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


class MiniMaxH3Int8RowChunksEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.__name__,
            display_name="H3 INT8 行分块 · INT8 Row Chunks (EXP/T8)",
            category=CATEGORY,
            is_experimental=True,
            description=(
                "只在所接 MODEL 的实际 INT8 投影调用中分行；rows=0 关闭。"
                "all 覆盖可识别投影，mlp 只选前馈层，完整输出和 K/V 仍保留。"
                "LOW/HIGH 可从共用底模分别接入，保留各自多个 LoRA 及原顺序。"
                "未知算子、补丁和后端保留原调用，不改全局 Kitchen 配置、采样步数或音频。"
                "实际覆盖看审计；不保证显存下降或提速，不认证跨进程缓存。"
            ),
            inputs=[
                io.Model.Input("model"),
                io.Boolean.Input("enabled", default=True),
                io.Int.Input(
                    "rows", default=2048, min=0, step=1,
                    tooltip="每次 INT8 投影的行数；0 关闭，不限制整个序列长度。",
                ),
                io.Combo.Input("targets", options=["all", "mlp"], default="all"),
                io.String.Input("phase", default="SINGLE", advanced=True),
            ],
            outputs=[io.Model.Output("model"), io.String.Output("report_json")],
        )

    @classmethod
    def execute(cls, model, enabled=True, rows=2048, targets="all", phase="SINGLE"):
        selected, report = configure_int8_row_chunks(
            model, Int8RowPlan(rows=rows, targets=targets),
            phase=phase, enabled=enabled,
        )
        return io.NodeOutput(selected, _report(report))


class MiniMaxH3Int8RowChunksAuditEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.__name__,
            display_name="H3 INT8 行分块 · 实际调用审计 (EXP/T8)",
            category=CATEGORY,
            is_experimental=True,
            is_output_node=True,
            description=(
                "model 接对应行分块 MODEL，after_sampling 接该路采样结果以建立执行顺序。"
                "仅报告真实投影调用、行数、委托和清理状态；连接结果本身不证明采样完成、"
                "NFE、画质、峰值显存或性能。"
            ),
            inputs=[io.Model.Input("model"), io.AnyType.Input("after_sampling")],
            outputs=[io.String.Output("report_json")],
        )

    @classmethod
    def execute(cls, model, after_sampling):
        report = _report(audit_int8_row_chunks(model))
        return io.NodeOutput(report, ui={"text": [report]})

    @classmethod
    def fingerprint_inputs(cls, **_kwargs):
        return float("nan")


NODES = [MiniMaxH3Int8RowChunksEXPT8, MiniMaxH3Int8RowChunksAuditEXPT8]
