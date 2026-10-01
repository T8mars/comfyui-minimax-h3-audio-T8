"""Additive Audio Refine effect ports; the seven old split nodes are untouched."""
from comfy_api.latest import io

from .audio_refine_nodes import BOUNDARY, CATEGORY
from .audio_refine_effects import bind_tail_effects, tail_effects_guider

CONTEXT = io.Custom("T8_STAGE_CONTEXT")


class MiniMaxH3AudioRefineEffectsBindEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__,
            display_name="H3 Audio Refine · External Effects Context (T8 EXP)",
            category=CATEGORY, is_experimental=True,
            description="After the existing Audio Refine Stage Bind, expose a source-bound context for "
                        "separate Relay, Stage EAV and Effects Guider nodes. Never samples or changes "
                        "noise/masks/SIGMAS. Keep the original Quality Gate. Long tails need their exact window.",
            inputs=[BOUNDARY.Input("stage_boundary"), io.Model.Input("model"), io.Noise.Input("noise"),
                    io.Guider.Input("guider"), io.Sampler.Input("sampler"), io.Sigmas.Input("sigmas"),
                    io.Latent.Input("stage_latent"), io.Int.Input("segment_index", default=0, min=0),
                    io.Int.Input("context_frames", default=0, min=0, max=39)],
            outputs=[io.Model.Output("model"), CONTEXT.Output("stage_context"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, stage_boundary, model, noise, guider, sampler, sigmas, stage_latent,
                segment_index=0, context_frames=0):
        return io.NodeOutput(*bind_tail_effects(stage_boundary, model, noise, guider,
            sampler, sigmas, stage_latent, segment_index, context_frames))


class MiniMaxH3AudioRefineEffectsGuiderEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__,
            display_name="H3 Audio Refine · External Effects Guider (T8 EXP)",
            category=CATEGORY, is_experimental=True,
            description="Connect the effect-patched MODEL to the old external Core sampler. Optional "
                        "positive accepts separately encoded Relay conditioning paired with this MODEL. "
                        "Retains the frozen source AV, original sampling controls and cfg=1; no quality approval.",
            inputs=[io.Model.Input("model"), CONTEXT.Input("stage_context"), io.Sigmas.Input("sigmas"),
                    io.Latent.Input("stage_latent"), io.Conditioning.Input("positive", optional=True)],
            outputs=[io.Guider.Output("guider"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, model, stage_context, sigmas, stage_latent, positive=None):
        return io.NodeOutput(*tail_effects_guider(model, stage_context, sigmas, stage_latent, positive))


NODES = [MiniMaxH3AudioRefineEffectsBindEXPT8, MiniMaxH3AudioRefineEffectsGuiderEXPT8]
