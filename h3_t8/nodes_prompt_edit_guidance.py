from comfy_api.latest import io

from .prompt_edit_guidance import PromptEditGuider


class MiniMaxH3PromptEditGuiderT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3PromptEditGuiderT8",
                         display_name="H3 Prompt Edit Guidance / 提示词定向编辑 (EXP/T8)",
                         category="T8/MiniMax H3/Sampling/Experimental", is_experimental=True,
                         description="Use original and edited prompts with the same actual media. "
                                     "0 uses original; 1 uses edited; other strengths evaluate two branches. "
                                     "Different text lengths use native position layouts.",
                         inputs=[io.Model.Input("model"), io.Conditioning.Input("original"),
                                 io.Conditioning.Input("edited"),
                                 io.Float.Input("edit_strength", default=1.1, min=-100.0, max=100.0, step=0.05),
                                 io.Int.Input("first_step", default=0, min=0, max=1000000)],
                         outputs=[io.Guider.Output("guider")])

    @classmethod
    def execute(cls, model, original, edited, edit_strength, first_step):
        return io.NodeOutput(PromptEditGuider(model, original, edited, edit_strength, first_step))


class MiniMaxH3PromptEditAuditT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3PromptEditAuditT8",
                         display_name="H3 Prompt Edit Report / 定向编辑运行报告 (EXP/T8)",
                         category="T8/MiniMax H3/Sampling/Experimental", is_experimental=True,
                         inputs=[io.Latent.Input("av_latent"), io.Guider.Input("guider")],
                         outputs=[io.Latent.Output("av_latent"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, av_latent, guider):
        if not isinstance(guider, PromptEditGuider):
            raise TypeError("Connect the T8 prompt-edit guider whose execution is being reported")
        return io.NodeOutput(av_latent, guider.report())


NODES = [MiniMaxH3PromptEditGuiderT8, MiniMaxH3PromptEditAuditT8]
