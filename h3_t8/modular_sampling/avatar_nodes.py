"""Avatar's recording-specific boundaries; common Progressive HIGH/effects stay external."""
from comfy_api.latest import io

from . import avatar
from . import progressive_nodes as shared

SOURCE = "T8_AVATAR_STAGE_SOURCE"


def schema(node, label, description, inputs, outputs, *, output=False):
    return io.Schema(node_id=node.__name__, display_name="H3 Avatar · " + label + " (T8 EXP)",
        category="T8/MiniMax H3/Modular Sampling/Avatar Experimental", is_experimental=True,
        description=description, inputs=inputs, outputs=outputs, is_output_node=output)


class MiniMaxH3AvatarSourceBindEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(cls, "Bind Encoded Source + Recording", "Bind your selected encoded AV and original PCM. "
            "Requires explicit nested AV masks with audio=0, e.g. Audio Latent Control lock. No encoding/sampling; "
            "this records your selected pair, not VAE provenance. Keep this neutral source independent of HIGH prompts.",
            [io.Latent.Input("high_source"), io.Audio.Input("original_recording")],
            [io.Latent.Output("high_source"), io.Custom(SOURCE).Output("avatar_source"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, high_source, original_recording):
        source = avatar.bind_source(high_source, original_recording)
        return io.NodeOutput(source.source, source, source.contract_json)


class MiniMaxH3AvatarLowStageEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        generic = shared.MiniMaxH3ProgressiveLowStageEXPT8.define_schema()
        return schema(cls, "LOW Only — Recording Bound", "Execute only initialized Avatar LOW. "
            "Use common Progressive Plan in initialized_av_exp mode and independent LOW EAV/Relay adapters. "
            "The completed boundary binds original recording/encoded source; no hidden HIGH or lift.",
            [*generic.inputs, io.Custom(SOURCE).Input("avatar_source")], generic.outputs)

    @classmethod
    def execute(cls, model, sampler, noise, plan, low_source, positive, negative, avatar_source,
                cfg=1., reserve_vram_mib=1024):
        result = avatar.sample_low(avatar_source, model, sampler, plan, low_source, positive, negative,
            shared._generate_noise(noise, low_source), seed=shared._noise_seed(noise), cfg=cfg,
            reserve_vram_mib=reserve_vram_mib, callback=shared._stage_progress(plan, "low"))
        return io.NodeOutput(result, result.receipt_json)


class MiniMaxH3AvatarHighHandoffEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(cls, "HIGH Handoff — Recording Anchor", "Restore frozen Avatar LOW and external learned lift. "
            "Default HIGH source is the bound recording source. Optional HIGH video source/mask may change, but "
            "clean audio and zero audio mask must match. Feed common Progressive HIGH/EAV/Relay nodes. No sampling.",
            [io.Custom(SOURCE).Input("avatar_source"), io.Custom(shared.BOUNDARY).Input("low_boundary"),
             io.Model.Input("model"), io.Sampler.Input("sampler"), io.Latent.Input("lifted_av"),
             io.Noise.Input("video_noise"), io.Latent.Input("high_source", optional=True),
             io.Sigmas.Input("high_sigmas", optional=True)],
            [io.Custom(shared.RESTART).Output("high_restart"), io.Custom(shared.PLAN).Output("plan"),
             io.String.Output("report_json")])

    @classmethod
    def execute(cls, avatar_source, low_boundary, model, sampler, lifted_av, video_noise,
                high_source=None, high_sigmas=None):
        video, _ = shared.stages.masks._av_parts(lifted_av["samples"], "Avatar lifted AV")
        state = avatar.prepare_high(avatar_source, low_boundary, model, sampler, lifted_av,
            shared._generate_noise(video_noise, {"samples": video}), high_source=high_source, high_sigmas=high_sigmas)
        return io.NodeOutput(state, state.plan, shared.stages.canonical(state.verify()))


class MiniMaxH3AvatarDeliveryAuditEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(cls, "Deliver Original Recording + Audit", "Verify the completed HIGH's recording binding. "
            "Return original selected PCM unchanged for explicit final mux, not generated/decoded audio. "
            "Accepts saved HIGH without source re-encoding or sampling. Reports latent anchor difference; "
            "does not certify trained lip sync, voice identity or quality.",
            [io.Custom(shared.HIGH_RESULT).Input("high_result"), io.Audio.Input("original_recording")],
            [io.Latent.Output("av_latent"), io.Audio.Output("original_audio"), io.String.Output("report_json")], output=True)

    @classmethod
    def execute(cls, high_result, original_recording):
        output, audio, report = avatar.deliver(high_result, original_recording)
        return io.NodeOutput(output, audio, report, ui={"text": (report,)})


NODES = [MiniMaxH3AvatarSourceBindEXPT8, MiniMaxH3AvatarLowStageEXPT8,
         MiniMaxH3AvatarHighHandoffEXPT8, MiniMaxH3AvatarDeliveryAuditEXPT8]
