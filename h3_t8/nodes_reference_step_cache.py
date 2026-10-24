"""New opt-in nodes only; registration is appended by the project owner."""
from comfy_api.latest import io

from .reference_temporal_pack import canonical
from .reference_step_cache_advanced import configure_reference_step_cache, audit_reference_step_cache


class MiniMaxH3ReferenceStepCacheEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, display_name="H3 参考跨步缓存 · Reference Step Cache (EXP/T8)",
            category="T8/MiniMax H3/References/Experimental", is_experimental=True,
            description="off原对象直通。approximate复用上次刷新时的参考输入，可能改变画质；不是Qwen缓存。"
                        "MODEL/positive必须成对使用，LOW/HIGH独立接入；同sigma重复评估完全重算。"
                        "未知owner保留执行并报告旁路，不承诺速度、显存或便携恢复。",
            inputs=[io.Model.Input("model"), io.Conditioning.Input("positive"),
                    io.Combo.Input("mode", options=["off", "approximate"], default="off"),
                    io.Int.Input("refresh_every", default=2, min=1),
                    io.Boolean.Input("include_text", default=False, advanced=True),
                    io.String.Input("phase", default="SINGLE", advanced=True)],
            outputs=[io.Model.Output("model"), io.Conditioning.Output("positive"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, model, positive, mode="off", refresh_every=2, include_text=False, phase="SINGLE"):
        selected, output, report = configure_reference_step_cache(
            model, positive, mode, refresh_every, include_text, phase=phase)
        return io.NodeOutput(selected, output, canonical(report))


class MiniMaxH3ReferenceStepCacheAuditEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, display_name="H3 参考跨步缓存 · 实际消费审计 (EXP/T8)",
            category="T8/MiniMax H3/References/Experimental", is_experimental=True, is_output_node=True,
            description="依赖实际采样输出，报告真正命中/完全重算/旁路与清理；不从配置推测NFE或收益。",
            inputs=[io.Model.Input("model"), io.AnyType.Input("after_sampling")],
            outputs=[io.String.Output("report_json")])

    @classmethod
    def execute(cls, model, after_sampling):
        raw = canonical(audit_reference_step_cache(model))
        return io.NodeOutput(raw, ui={"text": [raw]})

    @classmethod
    def fingerprint_inputs(cls, **_kwargs):
        return float("nan")


NODES = [MiniMaxH3ReferenceStepCacheEXPT8, MiniMaxH3ReferenceStepCacheAuditEXPT8]
