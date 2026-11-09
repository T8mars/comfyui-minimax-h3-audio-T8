from comfy_api.latest import io

from .taomate_hd_runtime import sample_taomate_hd


class MiniMaxH3TaoMateHDEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3TaoMateHDEXPT8",
            display_name="H3 TaoMate-HD · LOW2 + HIGH1 (T8 EXP)",
            category="T8/MiniMax H3/Sampling/Experimental", is_experimental=True,
            description="Teacher-coupled progressive generation. Load TaoMate-HD LoRA with the existing loader. "
                        "At CFG=1, 3 solver evaluations use 2 LOW + 1 HIGH. CFG>1 can call the MODEL more often; "
                        "the report separates solver evaluations and observed MODEL calls. Larger counts are experimental subdivisions. "
                        "No streaming or eight-GPU acceleration claim; MODEL and LoRAs remain user-selected.",
            inputs=[io.Model.Input("model"), io.Conditioning.Input("positive"), io.Latent.Input("latent"),
                    io.Int.Input("seed", default=8301, min=0, max=0xffffffffffffffff, control_after_generate=True),
                    io.Int.Input("steps", default=3, min=3, max=None),
                    io.Float.Input("cfg", default=1.0, min=0., max=None),
                    io.Float.Input("low_scale", default=.625, min=.001, max=.999, step=.001),
                    io.Conditioning.Input("negative", optional=True), io.Model.Input("high_model", optional=True)],
            outputs=[io.Latent.Output("av_latent"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, model, positive, latent, seed=8301, steps=3, cfg=1., low_scale=.625,
                negative=None, high_model=None):
        return io.NodeOutput(*sample_taomate_hd(model, positive, latent, negative=negative,
            seed=seed, steps=steps, cfg=cfg, low_scale=low_scale, high_model=high_model))


NODES = [MiniMaxH3TaoMateHDEXPT8]
