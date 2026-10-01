"""Explicit continuous HyperFlow stages; old combined split remains available."""
from comfy_api.latest import io

from . import hyperflow as stages
from .progressive_nodes import _generate_noise, _noise_seed

BOUNDARY = "T8_HYPERFLOW_CONTINUOUS_BOUNDARY"
RESULT = "T8_HYPERFLOW_COMPLETED_AV"
CATEGORY = "T8/MiniMax H3/Modular Sampling/HyperFlow Experimental"


def _progress(start, stop):
    import comfy.utils
    progress = comfy.utils.ProgressBar(stop - start)
    def update(index, _prediction, _state, _total):
        progress.update_absolute(index - start + 1, stop - start)
    return update


class MiniMaxH3HyperFlowHeadStageEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, display_name="H3 HyperFlow · Continuous HEAD Only (T8 EXP)",
            category=CATEGORY, is_experimental=True,
            description="Only absolute0:split, capturing the exact raw model-space AV x_sigma. Independent "
                "MODEL/content LoRAs/conditions/NOISE. No TAIL or upscale. Requires dedicated HyperFlow Loader. "
                "No denoise masks in this continuous contract. Boundary is not clean x0 or a Progressive boundary. "
                "Current identity is process-local; disk resume and stage EAV/Relay adapters remain pending.",
            inputs=[io.Model.Input("model"), io.Latent.Input("av_latent"), io.Noise.Input("noise"),
                io.Conditioning.Input("positive"), io.Conditioning.Input("negative"),
                io.Int.Input("split_interval", default=4, min=1, max=7),
                io.Float.Input("cfg", default=1., min=0., max=100., step=.1),
                io.Int.Input("reserve_vram_mib", default=1024, min=512, max=65536)],
            outputs=[io.Custom(BOUNDARY).Output("continuous_boundary"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, model, av_latent, noise, positive, negative, split_interval=4, cfg=1., reserve_vram_mib=1024):
        result = stages.sample_head(model, av_latent, positive, negative, _generate_noise(noise, av_latent),
            seed=_noise_seed(noise), split_interval=split_interval, cfg=cfg, reserve_vram_mib=reserve_vram_mib,
            callback=_progress(0, split_interval))
        return io.NodeOutput(result, result.receipt_json)


class MiniMaxH3HyperFlowTailStageEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, display_name="H3 HyperFlow · Continuous TAIL Only (T8 EXP)",
            category=CATEGORY, is_experimental=True,
            description="Only absolute split:8 on the captured x_sigma, with independent MODEL/content LoRAs "
                "and conditions. Clone the same full H3 base and use the same original HyperFlow adapter. "
                "No fresh noise, resize or HEAD rerun. Seed is a model execution seed, not a new noise seed. "
                "This is NOT fresh-noise8+4/partial4+4/P7; no persisted-effect/GPU/media quality claim.",
            inputs=[io.Custom(BOUNDARY).Input("continuous_boundary"), io.Model.Input("model"),
                io.Conditioning.Input("positive"), io.Conditioning.Input("negative"),
                io.Int.Input("seed", default=26092301, min=0, max=2**64-1),
                io.Float.Input("cfg", default=1., min=0., max=100., step=.1),
                io.Int.Input("reserve_vram_mib", default=1024, min=512, max=65536)],
            outputs=[io.Latent.Output("av_latent"), io.String.Output("report_json"), io.Custom(RESULT).Output("completed_result")])

    @classmethod
    def execute(cls, continuous_boundary, model, positive, negative, seed=26092301, cfg=1., reserve_vram_mib=1024):
        start = continuous_boundary.verify()["request"]["plan"]["absolute_interval"][1]
        result, report = stages.sample_tail_result(continuous_boundary, model, positive, negative,
            seed=seed, cfg=cfg, reserve_vram_mib=reserve_vram_mib, callback=_progress(start, 8))
        return io.NodeOutput(result.output, report, result)


NODES = [MiniMaxH3HyperFlowHeadStageEXPT8, MiniMaxH3HyperFlowTailStageEXPT8]
