"""Append-only public P7 accepted-parent and context preparation interfaces.

These nodes only read an existing accepted chain. They do not run diffusion,
accept a segment, or modify the historical P7 delivery/cache formats.
"""
import folder_paths
from comfy_api.latest import io

from .. import long_video_delivery as delivery
from . import hyperflow_p7 as p7


SOURCE = "T8_HYPERFLOW_P7_ACCEPTED_PARENT"
CONTEXTS = "T8_HYPERFLOW_P7_CONTEXTS"
PHASE = "T8_HYPERFLOW_P7_PREPARED_PHASE"
LOW_RESULT = "T8_HYPERFLOW_P7_LOW_RESULT"
LIFT = "T8_HYPERFLOW_P7_LEARNED_LIFT"
HIGH_INPUT = "T8_HYPERFLOW_P7_HIGH_INPUT"
HIGH_RESULT = "T8_HYPERFLOW_P7_HIGH_RESULT"
CATEGORY = "T8/MiniMax H3/Modular Sampling/HyperFlow P7 Experimental"


def schema(cls, title, description, inputs, outputs):
    return io.Schema(node_id=cls.__name__, display_name=f"H3 HyperFlow P7 · {title} (T8 EXP)",
        category=CATEGORY, is_experimental=True, description=description,
        inputs=inputs, outputs=outputs)


class MiniMaxH3HyperFlowP7InitialSegmentEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(cls, "Select Empty Segment 0",
            "Use the original P7 empty first-segment context on a new or still-empty chain. "
            "The context_frames widget selects later-segment capacity; actual segment 0 conditions use 0. "
            "Read-only; refuses an already accepted segment or a changed manifest. Does not create "
            "a chain, prepare VAE, or sample. Connect output to two independent phase conditions.",
            [io.String.Input("chain_id", default="h3_hyperflow_long_video_exp_new"),
             io.Combo.Input("context_frames", options=[5, 22, 39], default=22),
             io.Int.Input("width", default=896, min=64, max=16384, step=32),
             io.Int.Input("height", default=448, min=64, max=16384, step=32),
             io.Int.Input("low_width", default=448, min=32, max=16352, step=32),
             io.Int.Input("low_height", default=224, min=32, max=16352, step=32)],
            [io.Custom(CONTEXTS).Output("contexts"), io.String.Output("report_json"),
             io.Int.Output("width"), io.Int.Output("height"),
             io.Int.Output("low_width"), io.Int.Output("low_height")])

    @classmethod
    def execute(cls, chain_id, context_frames=22, width=896, height=448,
                low_width=448, low_height=224):
        value = p7.capture_initial(delivery.long_video_chain_root(chain_id), chain_id=chain_id,
            context_frames=context_frames, width=width, height=height,
            low_width=low_width, low_height=low_height)
        return io.NodeOutput(value, value.binding_json, width, height, low_width, low_height)

    @classmethod
    def fingerprint_inputs(cls, **inputs):
        try:
            return cls.execute(**inputs).result[0].verify()["sha256"]
        except (OSError, ValueError, RuntimeError):
            return float("nan")


class MiniMaxH3HyperFlowP7AcceptedParentEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(cls, "Select Existing Accepted Parent",
            "Read an existing P7 accepted immediate predecessor. Verify revision, candidate, actual "
            "MP4, LOW/HIGH contexts, completed audio and previous execution contract. The previous "
            "job SHA authenticates that parent, not today's edited models or prompts. No sampling, "
            "chain creation, acceptance or implicit reuse of the old P7 stage cache.",
            [io.String.Input("chain_id", default="h3_hyperflow_long_video_exp"),
             io.Int.Input("segment_index", default=1, min=1, max=99999),
             io.String.Input("parent_candidate_id", default=""),
             io.Int.Input("parent_revision", default=1, min=1),
             io.String.Input("previous_job_sha256", default=""),
             io.Combo.Input("context_frames", options=[5, 22, 39], default=22),
             io.Int.Input("width", default=896, min=64, max=16384, step=32),
             io.Int.Input("height", default=448, min=64, max=16384, step=32),
             io.Int.Input("low_width", default=448, min=32, max=16352, step=32),
             io.Int.Input("low_height", default=224, min=32, max=16352, step=32)],
            [io.Custom(SOURCE).Output("accepted_parent"), io.String.Output("report_json"),
             io.Int.Output("width"), io.Int.Output("height"),
             io.Int.Output("low_width"), io.Int.Output("low_height")])

    @classmethod
    def execute(cls, chain_id, segment_index, parent_candidate_id, parent_revision,
                previous_job_sha256, context_frames=22, width=896, height=448,
                low_width=448, low_height=224):
        parent = p7.capture_parent(delivery.long_video_chain_root(chain_id), chain_id=chain_id,
            segment_index=segment_index, parent_candidate_id=parent_candidate_id,
            parent_revision=parent_revision, previous_job_sha256=previous_job_sha256,
            context_frames=context_frames, width=width, height=height,
            low_width=low_width, low_height=low_height)
        return io.NodeOutput(parent, parent.binding_json, width, height, low_width, low_height)

    @classmethod
    def fingerprint_inputs(cls, **inputs):
        try:
            return cls.execute(**inputs).result[0].sha256
        except (OSError, ValueError, RuntimeError):
            return float("nan")


class MiniMaxH3HyperFlowP7PrepareContextsEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(cls, "Prepare LOW/HIGH Continuation Contexts",
            "Use the accepted parent's last 39 decoded RGB24 frames, original resize and current "
            "VAE for LOW picture motion guides only. HIGH context and completed audio stay unchanged. "
            "A new explicit stage graph must consume these typed contexts; this node does not sample.",
            [io.Custom(SOURCE).Input("accepted_parent"), io.Vae.Input("video_vae")],
            [io.Custom(CONTEXTS).Output("contexts"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, accepted_parent, video_vae):
        if type(accepted_parent) is not p7.P7Parent:
            raise ValueError("Select an authenticated P7 accepted parent first")
        contexts = accepted_parent.prepare_contexts(video_vae)
        return io.NodeOutput(contexts, contexts.contract_json)


class MiniMaxH3HyperFlowP7ConditioningEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(cls, "One Phase Native Conditions",
            "Prepare independently editable LOW or HIGH T2VA/native prompt and original P7 motion/audio "
            "conditions. HIGH receives the accepted motion reference without a new prefix lock. "
            "No sampling, MODEL patch, "
            "Relay or EAV is performed here; those require separate downstream stage nodes.",
            [io.Custom(CONTEXTS).Input("contexts"),
             io.Combo.Input("phase", options=["low", "high"], default="low"),
             io.Clip.Input("clip"), io.Vae.Input("video_vae"), io.Vae.Input("audio_vae"),
             io.String.Input("prompt", multiline=True, default=""),
             io.Int.Input("length", default=124, min=5, step=17)],
            [io.Custom(PHASE).Output("prepared_phase"), io.Conditioning.Output("positive"),
             io.Conditioning.Output("negative"), io.Latent.Output("source_av"),
             io.Audio.Output("mux_audio"), io.String.Output("conditioned_prompt"),
             io.String.Output("media_map_json"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, contexts, phase, clip, video_vae, audio_vae, prompt, length=124):
        prepared = p7.prepare_phase(contexts, phase, clip=clip, video_vae=video_vae,
                                    audio_vae=audio_vae, prompt=prompt, length=length)
        positive, latent, audio, text, media_map, _, _ = prepared.result
        return io.NodeOutput(prepared, positive, positive, latent, audio, text,
                             media_map, prepared.contract_json)


class MiniMaxH3HyperFlowP7LowSetupEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(cls, "LOW 0:4 Setup Only",
            "Bind an independently selected Fresh HyperFlow MODEL/LoRA and LOW phase to the "
            "original long-video extra conditions and absolute 0:4 sampler. External positive "
            "may carry a paired Prompt Relay; protected motion guides cannot be replaced. "
            "Connect optional Stage EAV, Basic Guider, and ONE Stage Sampler. No sampling here.",
            [io.Custom(PHASE).Input("prepared_phase"), io.Model.Input("model"),
             io.Conditioning.Input("positive"), io.Conditioning.Input("negative")],
            [io.Model.Output("model"), io.Sampler.Output("sampler"),
             io.Sigmas.Output("sigmas"), io.Latent.Output("source_av"),
             io.Custom("T8_STAGE_CONTEXT").Output("stage_context"),
             io.Conditioning.Output("positive"), io.Conditioning.Output("negative"),
             io.String.Output("report_json")])

    @classmethod
    def execute(cls, prepared_phase, model, positive, negative):
        return io.NodeOutput(*p7.setup_low(prepared_phase, model, positive, negative))


class MiniMaxH3HyperFlowP7LowResultEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(cls, "Bind Completed LOW Result",
            "Verify the actual Stage Sampler 0:4 receipt came from this P7 phase, including "
            "the accepted parent and source tensor identity. Select the real denoised_output "
            "for external learned3D, not nonterminal output. A saved/loaded LOW Stage Result "
            "can be rebound for HIGH-only work after source revalidation. No sampling here.",
            [io.Custom(PHASE).Input("prepared_phase"),
             io.Custom("T8_STAGE_RESULT").Input("stage_result")],
            [io.Custom(LOW_RESULT).Output("low_result"),
             io.Latent.Output("low_x0"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, prepared_phase, stage_result):
        result = p7.bind_low(prepared_phase, stage_result)
        return io.NodeOutput(result, result.sampled.denoised_output, result.contract_json)


class MiniMaxH3HyperFlowP7LearnedLiftEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(cls, "Original Learned 3D Lift",
            "Run the original P7 target-dimensions learned3D on authenticated LOW denoised_output. "
            "Audio is preserved; this is a standalone upscale, not a diffusion stage.",
            [io.Custom(LOW_RESULT).Input("low_result"),
             io.Combo.Input("upscaler_model", options=folder_paths.get_filename_list("latent_upscale_models"))],
            [io.Custom(LIFT).Output("learned_lift"), io.Latent.Output("enlarged_av"),
             io.String.Output("report_json")])

    @classmethod
    def execute(cls, low_result, upscaler_model):
        value = p7.lift_low(low_result, upscaler_model)
        return io.NodeOutput(value, value.latent, value.contract_json)


class MiniMaxH3HyperFlowP7HighHandoffEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(cls, "Original HIGH Audio Reconcile",
            "Join a verified LOW learned lift with independently prepared HIGH reference conditions. "
            "Use the original audio policy: LOW coarse generated region, HIGH locked template region. "
            "External paired Relay may feed positive; protected motion guides must remain intact.",
            [io.Custom(LIFT).Input("learned_lift"), io.Custom(PHASE).Input("high_phase"),
             io.Conditioning.Input("positive"), io.Conditioning.Input("negative")],
            [io.Custom(HIGH_INPUT).Output("high_handoff"), io.Latent.Output("high_source_av"),
             io.Conditioning.Output("positive"), io.Conditioning.Output("negative"),
             io.String.Output("report_json")])

    @classmethod
    def execute(cls, learned_lift, high_phase, positive, negative):
        value = p7.handoff_high(learned_lift, high_phase, positive, negative)
        return io.NodeOutput(value, value.source, value.positive, value.negative, value.contract_json)


class MiniMaxH3HyperFlowP7HighSetupEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(cls, "HIGH 4:8 Setup Only",
            "Patch the selected independent HIGH MODEL and build original absolute fresh 4:8 stage. "
            "Connect optional external Stage EAV, Basic Guider and ONE Stage Sampler.",
            [io.Custom(HIGH_INPUT).Input("high_handoff"), io.Model.Input("model")],
            [io.Model.Output("model"), io.Sampler.Output("sampler"), io.Sigmas.Output("sigmas"),
             io.Latent.Output("source_av"), io.Custom("T8_STAGE_CONTEXT").Output("stage_context"),
             io.Conditioning.Output("positive"), io.Conditioning.Output("negative"),
             io.String.Output("report_json")])

    @classmethod
    def execute(cls, high_handoff, model):
        return io.NodeOutput(*p7.setup_high(high_handoff, model))


class MiniMaxH3HyperFlowP7HighResultEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(cls, "Bind Completed HIGH Result",
            "Verify actual fresh 4:8 Stage Sampler completion and source identity. Output is the "
            "terminal joint AV; delivery/acceptance remains a separate explicit step.",
            [io.Custom(HIGH_INPUT).Input("high_handoff"),
             io.Custom("T8_STAGE_RESULT").Input("stage_result")],
            [io.Custom(HIGH_RESULT).Output("high_result"), io.Latent.Output("completed_av"),
             io.String.Output("report_json")])

    @classmethod
    def execute(cls, high_handoff, stage_result):
        value = p7.bind_high(high_handoff, stage_result)
        return io.NodeOutput(value, value.sampled.output, value.contract_json)


NODES = [MiniMaxH3HyperFlowP7AcceptedParentEXPT8,
         MiniMaxH3HyperFlowP7PrepareContextsEXPT8,
         MiniMaxH3HyperFlowP7ConditioningEXPT8,
         MiniMaxH3HyperFlowP7InitialSegmentEXPT8,
         MiniMaxH3HyperFlowP7LowSetupEXPT8,
         MiniMaxH3HyperFlowP7LowResultEXPT8,
         MiniMaxH3HyperFlowP7LearnedLiftEXPT8,
         MiniMaxH3HyperFlowP7HighHandoffEXPT8,
         MiniMaxH3HyperFlowP7HighSetupEXPT8,
         MiniMaxH3HyperFlowP7HighResultEXPT8]
