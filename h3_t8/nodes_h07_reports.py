"""Append-only read-only H07 diagnostics. No director draft replacement or sampling."""
from comfy_api.latest import io
from .h07_reports import canonical, reference_summary, continuity_lint


class MiniMaxH3ReferenceExecutionSummaryEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, display_name="H3 参考编号／实际执行摘要（只读）",
            category="T8/MiniMax H3/Diagnostics", is_experimental=True, is_output_node=True,
            description="接实际media_map；角色／Subject／S只接受显式声明，不按顺序猜。SIGMAS数量不是实际NFE，只有经校验完成Stage才报完成计数。不改条件或提示词。",
            inputs=[io.String.Input("media_map_json", default='{"pictures":{},"videos":{},"audios":{}}', multiline=True),
                    io.String.Input("roles_json", default="[]", multiline=True),
                    io.Sigmas.Input("sigmas", optional=True),
                    io.Custom("H3_T8_FREEVIDEO_QUALITY_STAGE").Input("completed_quality_stage", optional=True)],
            outputs=[io.String.Output("report_json")])

    @classmethod
    def execute(cls, **kwargs):
        report = canonical(reference_summary(**kwargs))
        return io.NodeOutput(report, ui={"text": [report]})


class MiniMaxH3PropContinuityLintEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, display_name="H3 道具ID／左右手连续性检查（只读）",
            category="T8/MiniMax H3/Diagnostics", is_experimental=True, is_output_node=True,
            description="独立文本配方与冲突报告；相同kind不同ID不会合并。全局秒数、显式交接；不识别画面、不替换导演台草稿、不自动采用或排队。",
            inputs=[io.String.Input("states_json", default="[]", multiline=True),
                    io.String.Input("allowed_transfers_json", default="[]", multiline=True)],
            outputs=[io.String.Output("template"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, **kwargs):
        result = continuity_lint(**kwargs)
        report = canonical(result)
        return io.NodeOutput(result["template"], report, ui={"text": [report]})


class MiniMaxH3ContinuationCardEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, display_name="H3 预览选片·继续／恢复卡（只读）",
            category="T8/MiniMax H3/Diagnostics", is_experimental=True, is_output_node=True,
            description="只接一个经校验Stage/MID或明确RES历史path+SHA。完成态读取0NFE；LOW8→HIGH3、MID4→后4、RES POST4→同轨余4分别显示。不是采样器，不从MP4恢复latent。",
            inputs=[io.String.Input("res_checkpoint_path", default=""),
                    io.String.Input("res_checkpoint_sha256", default=""),
                    io.Custom("H3_T8_FREEVIDEO_QUALITY_STAGE").Input("completed_quality_stage", optional=True),
                    io.Custom("H3_T8_FREEVIDEO_STAGE").Input("completed_freevideo_stage", optional=True),
                    io.Custom("H3_T8_FREEVIDEO_MID").Input("freevideo_mid", optional=True),
                    io.Custom("T8_STAGE_RESULT").Input("completed_modular_stage", optional=True)],
            outputs=[io.String.Output("continue_restore_card"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, **kwargs):
        from .h07_continuation import continuation_card
        from .nodes_res_history_exp import storage_root
        result = continuation_card(checkpoint_root=storage_root(), **kwargs)
        return io.NodeOutput(result["text"], canonical(result), ui={"text": [result["text"], canonical(result)]})

    @classmethod
    def fingerprint_inputs(cls, res_checkpoint_path="", **kwargs):
        if res_checkpoint_path:
            # Re-read the selected file; a stale cached card is not current evidence.
            return float("nan")
        return None


NODES = [MiniMaxH3ReferenceExecutionSummaryEXPT8, MiniMaxH3PropContinuityLintEXPT8,
         MiniMaxH3ContinuationCardEXPT8]
