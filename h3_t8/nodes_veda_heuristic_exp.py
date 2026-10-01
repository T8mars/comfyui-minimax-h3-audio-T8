"""Separate model-free TripPool tile-64 Veda experiment and execution audit."""

from __future__ import annotations

import json

from comfy_api.latest import io

from .veda_heuristic import POOL_MODES, PRESETS, SINK_MODES
from .veda_heuristic_runtime import HeuristicRuntime, apply_heuristic


class MiniMaxH3VedaHeuristicApplyEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3VedaHeuristicApplyEXPT8",
            display_name="H3 Veda · Heuristic TripPool 64 Apply (EXP)",
            category="T8/MiniMax H3/Acceleration/Experimental",
            is_experimental=True,
            description="Independent, model-free tile-64 TripPool route inspired by BSAI; "
                        "not the released Miowtion predictor or FA4. Exact native H3 grid, "
                        "global conditioning rows/columns, per-MODEL clone and history audit. "
                        "Default observes dense; optional sigma window delegates outside "
                        "the selected interval. Sink off removes global K/V only for video "
                        "queries and is not quality approved. No speed guarantee.",
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input("mode", options=["report_only", "apply_exp"], default="report_only"),
                io.Float.Input("keep_percent", default=5.0, min=1.0, max=100.0, step=1.0),
                io.Combo.Input("head_tiling", options=[*PRESETS, "custom"], default="dual_fast"),
                io.String.Input("custom_tiling", default="4,4,4;8,4,2;2,4,8;4,8,2", advanced=True),
                io.Combo.Input("pool_mode", options=list(POOL_MODES), default="triplet"),
                io.Int.Input("min_tokens", default=4096, min=0, max=1_000_000),
                io.Int.Input("max_copy_mib", default=16, min=1, max=256, advanced=True),
                io.Float.Input("start_percent", default=0.0, min=0.0, max=1.0,
                               step=0.01, optional=True, advanced=True),
                io.Float.Input("end_percent", default=1.0, min=0.0, max=1.0,
                               step=0.01, optional=True, advanced=True),
                io.Combo.Input("sink_conditioning", options=list(SINK_MODES),
                               default="exact_kv_and_rows", optional=True, advanced=True),
            ],
            outputs=[io.Model.Output("model"),
                     io.Custom("T8_VEDA_HEURISTIC_RUNTIME").Output("runtime")],
        )

    @classmethod
    def execute(cls, model, mode="report_only", keep_percent=5.0,
                head_tiling="dual_fast", custom_tiling="4,4,4;8,4,2;2,4,8;4,8,2",
                pool_mode="triplet", min_tokens=4096, max_copy_mib=16,
                start_percent=0.0, end_percent=1.0,
                sink_conditioning="exact_kv_and_rows"):
        patched, runtime = apply_heuristic(
            model, mode=mode, keep_percent=keep_percent,
            head_tiling=head_tiling, custom_tiling=custom_tiling,
            pool_mode=pool_mode, min_tokens=min_tokens, max_copy_mib=max_copy_mib,
            start_percent=start_percent, end_percent=end_percent,
            sink_conditioning=sink_conditioning)
        return io.NodeOutput(patched, runtime)


class MiniMaxH3VedaHeuristicAuditEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3VedaHeuristicAuditEXPT8",
            display_name="H3 Veda · Heuristic Audit After Sample (EXP)",
            category="T8/MiniMax H3/Acceleration/Experimental",
            is_experimental=True,
            is_output_node=True,
            description="Connect after the actual sampler. Sparse calls are heuristic, "
                        "never evidence of trained predictor use, quality, or speedup.",
            inputs=[io.Latent.Input("av_latent"),
                    io.Custom("T8_VEDA_HEURISTIC_RUNTIME").Input("runtime")],
            outputs=[io.Latent.Output("av_latent"), io.String.Output("report_json")],
        )

    @classmethod
    def execute(cls, av_latent, runtime: HeuristicRuntime):
        if type(runtime) is not HeuristicRuntime:
            raise TypeError("Use the paired H3 Veda Heuristic Apply runtime")
        report = json.dumps(runtime.snapshot(), ensure_ascii=False, indent=2)
        return io.NodeOutput(av_latent, report, ui={"text": (report,)})


VEDA_HEURISTIC_NODE_CLASSES = [
    MiniMaxH3VedaHeuristicApplyEXPT8,
    MiniMaxH3VedaHeuristicAuditEXPT8,
]
