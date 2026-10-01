"""Opt-in verified LOW x0 port for editable V2 learned-upscale workflows."""

from comfy_api.latest import io

from .fast_h3_v2_handoff import completed_low_x0


class MiniMaxH3FastH3V2CompletedLowX0EXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3FastH3V2CompletedLowX0EXPT8",
            display_name="FastH3 V2 · Completed LOW x0 for Learned Handoff (T8 EXP)",
            category="T8/MiniMax H3/Modular Sampling/Experimental", is_experimental=True,
            description="Verify the actual completed V2 LOW 0:4 StageResult and expose its "
                        "denoised prediction_x0, never unfinished x_sigma. Connect the output to "
                        "the existing learned 3D latent upscaler, then external reconcile and HIGH 4:8. "
                        "Accepts an exact frozen LOW result for HIGH-only resume; no sampling here.",
            inputs=[io.Custom("T8_STAGE_RESULT").Input("low_stage_result")],
            outputs=[io.Latent.Output("low_x0"), io.String.Output("report_json")],
        )

    @classmethod
    def execute(cls, low_stage_result):
        return io.NodeOutput(*completed_low_x0(low_stage_result))


NODES = (MiniMaxH3FastH3V2CompletedLowX0EXPT8,)
