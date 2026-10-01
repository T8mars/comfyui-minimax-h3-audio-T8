"""Explicit accepted-parent continuation graph stages; no hidden loop/accept."""
from comfy_api.latest import io

from ..nodes_long_video_exp import MiniMaxH3LongVideoConditioningT8
from ..prompt_relay_advanced import PROMPT_RELAY_PLAN_TYPE
from . import continuation as stages
from . import continuation_effects as effects
from . import progressive_nodes as shared
from .eav import CONFIG_TYPE

SOURCE = "T8_CONTINUATION_ACCEPTED_SOURCE"
CONTEXTS = "T8_CONTINUATION_STAGE_CONTEXTS"
PHASE = "T8_CONTINUATION_PREPARED_PHASE"
RELAY = "T8_CONTINUATION_PROJECTED_RELAY"


def schema(node, title, description, inputs, outputs, *, output=False):
    return io.Schema(node_id=node.__name__, display_name="H3 Continuation · " + title + " (T8 EXP)",
        category="T8/MiniMax H3/Modular Sampling/Continuation Experimental", is_experimental=True,
        description=description, inputs=inputs, outputs=outputs, is_output_node=output)


class MiniMaxH3ContinuationSourceEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(cls, "Select Accepted Parent", "Select the existing immediate accepted parent in an output "
            "chain. Validates revision, candidate, actual MP4/context bytes and previous job SHA. No VAE, sampling "
            "or chain creation. previous_job_sha256 authenticates the selected previous job only; it is NOT proof "
            "that today's edited stage models/prompts are the same job. Does not auto-accept or compose.",
            [io.String.Input("chain_id", default="h3_progressive_long_video"),
             io.Int.Input("segment_index", default=1, min=1, max=99999),
             io.String.Input("parent_candidate_id", default=""), io.Int.Input("parent_revision", default=1, min=1),
             io.String.Input("previous_job_sha256", default=""),
             io.Combo.Input("context_frames", options=[5, 22, 39], default=22),
             io.Int.Input("width", default=896, min=64, max=16384, step=32),
             io.Int.Input("height", default=448, min=64, max=16384, step=32),
             io.Int.Input("low_width", default=448, min=32, max=16352, step=32),
             io.Int.Input("low_height", default=224, min=32, max=16352, step=32)],
            [io.Custom(SOURCE).Output("accepted_source"), io.String.Output("report_json"),
             io.Int.Output("width"), io.Int.Output("height"), io.Int.Output("low_width"), io.Int.Output("low_height")])

    @classmethod
    def execute(cls, chain_id, segment_index, parent_candidate_id, parent_revision, previous_job_sha256,
                context_frames=22, width=896, height=448, low_width=448, low_height=224):
        source = stages.capture_source(stages.delivery.long_video_chain_root(chain_id), chain_id=chain_id,
            segment_index=segment_index, parent_candidate_id=parent_candidate_id, parent_revision=parent_revision,
            job_sha256=previous_job_sha256, context_frames=context_frames, width=width, height=height,
            low_width=low_width, low_height=low_height)
        return io.NodeOutput(source, source.binding_json, width, height, low_width, low_height)

    @classmethod
    def fingerprint_inputs(cls, **inputs):
        try:
            return cls.execute(**inputs).result[0].sha256
        except (OSError, ValueError, RuntimeError):
            return float("nan")


class MiniMaxH3ContinuationContextsEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(cls, "Prepare Accepted Contexts", "Decode accepted MP4 last39 RGB frames, existing resize "
            "then current VAE for LOW motion guides only. HIGH completed AV and shared audio are unchanged. "
            "Neutral preparation: no HIGH prompt, model, sampling, accept or timeline update.",
            [io.Custom(SOURCE).Input("accepted_source"), io.Vae.Input("video_vae")],
            [io.Custom(CONTEXTS).Output("contexts"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, accepted_source, video_vae):
        contexts = stages.prepare_contexts(accepted_source, video_vae)
        return io.NodeOutput(contexts, contexts.contract_json)


class MiniMaxH3ContinuationPlanEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(cls, "Geometry + Stage Plan", "One native initialized Euler schedule and accepted-parent "
            "geometry. No HIGH prompt or condition dependency. Independent stages keep original AV clocks; "
            "not FastH3/HyperFlow/VDN math. Connect common Progressive Lift Input and external learned3D upscale.",
            [io.Custom(CONTEXTS).Input("contexts"), io.Sigmas.Input("full_sigmas"),
             io.Int.Input("length", default=124, min=5, step=17),
             io.Int.Input("low_evaluations", default=4, min=1, max=999),
             io.Float.Input("low_scale", default=.5, min=.25, max=.99, step=.01)],
            [io.Custom(shared.PLAN).Output("plan"), io.Sigmas.Output("low_sigmas"),
             io.Sigmas.Output("high_sigmas"), io.Int.Output("length"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, contexts, full_sigmas, length=124, low_evaluations=4, low_scale=.5):
        plan = stages.build_plan(contexts, length, full_sigmas, low_evaluations, low_scale)
        return io.NodeOutput(plan, full_sigmas[:low_evaluations+1].detach().clone(),
            full_sigmas[low_evaluations:].detach().clone(), length, stages.stages.canonical(plan.report()))


class MiniMaxH3ContinuationConditioningEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        old = MiniMaxH3LongVideoConditioningT8.define_schema()
        removed = {"model", "context", "segment_index", "context_frames", "width", "height"}
        inherited = [item for item in old.inputs if item.id not in removed]
        for item in inherited:
            if item.id == "length":
                item.force_input = False  # this new schema accepts a widget or an external frame count
        return schema(cls, "ONE Phase Conditions", "Use independent LOW and HIGH nodes/prompts/references. "
            "Reuse the original native motion builder without resizing LOW guides twice. Only HIGH locks the "
            "completed video prefix. Both phases must retain matching source audio. Accepts external Bridge; "
            "for Relay connect projected compiled_prompt, then Continuation Relay Apply.",
            [io.Custom(CONTEXTS).Input("contexts"), io.Combo.Input("phase", options=["low", "high"], default="low"),
             *inherited],
            [io.Custom(PHASE).Output("prepared_phase"), io.Conditioning.Output("positive"),
             io.Conditioning.Output("negative"), io.Latent.Output("source_av"), io.Audio.Output("mux_audio"),
             io.String.Output("conditioned_prompt"), io.String.Output("media_map_json"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, contexts, phase, clip, video_vae, audio_vae, prompt, length, **options):
        prepared = stages.prepare_phase(contexts, phase, clip=clip, video_vae=video_vae,
            audio_vae=audio_vae, prompt=prompt, length=length, **options)
        positive, latent, audio, prompt, media_map, _, _ = prepared.result
        return io.NodeOutput(prepared, positive, positive, latent, audio, prompt, media_map, prepared.contract_json)


class MiniMaxH3ContinuationLowStageEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(cls, "LOW Sampler Only", "Only this prepared LOW stage. Original accepted-picture motion "
            "guides and native CFG1. Independent MODEL/LoRA/NOISE/conditions; no hidden HIGH/learned lift.",
            [io.Custom(PHASE).Input("prepared_phase"), io.Model.Input("model"), io.Sampler.Input("sampler"),
             io.Custom(shared.PLAN).Input("plan"), io.Noise.Input("noise"),
             io.Conditioning.Input("positive"), io.Conditioning.Input("negative"),
             io.Int.Input("reserve_vram_mib", default=1024, min=512, max=65536)],
            [io.Custom(shared.BOUNDARY).Output("low_boundary"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, prepared_phase, model, sampler, plan, noise, positive, negative, reserve_vram_mib=1024):
        result = stages.sample_low(prepared_phase, model, sampler, plan,
            shared._generate_noise(noise, prepared_phase.result[1]), positive=positive, negative=negative,
            seed=shared._noise_seed(noise), reserve_vram_mib=reserve_vram_mib, callback=shared._stage_progress(plan, "low"))
        return io.NodeOutput(result, result.receipt_json)


class MiniMaxH3ContinuationHighHandoffEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        old = shared.MiniMaxH3ProgressiveHighHandoffEXPT8.define_schema()
        return schema(cls, "HIGH Handoff Only", "Bind frozen LOW, external learned3D output and this prepared HIGH. "
            "Preserves original video renoise and evolving audio; HIGH's clean prefix is not the noisy restart. "
            "No sampling. Original video noise policy: LOW seed+1 modulo uint64. Optional HIGH tail starts at frozen sigma.",
            [io.Custom(PHASE).Input("prepared_phase"), *[item for item in old.inputs if item.id != "high_source"]], old.outputs)

    @classmethod
    def execute(cls, prepared_phase, low_boundary, model, sampler, lifted_av, video_noise, high_sigmas=None):
        video, _ = stages.stages.masks._av_parts(lifted_av["samples"], "continuation learned output")
        restart = stages.prepare_high(prepared_phase, low_boundary, model, sampler, lifted_av,
            shared._generate_noise(video_noise, {"samples": video}), high_sigmas=high_sigmas)
        return io.NodeOutput(restart, restart.plan, stages.stages.canonical(restart.verify()))


class MiniMaxH3ContinuationHighStageEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        old = shared.MiniMaxH3ProgressiveHighStageEXPT8.define_schema()
        return schema(cls, "HIGH Sampler Only", "Only the selected continuation HIGH. Independent MODEL/LoRA, "
            "conditions and external phase effects. Native CFG1, original AV/motion clocks. No LOW, lift, loop or auto-accept.",
            [io.Custom(PHASE).Input("prepared_phase"), *[item for item in old.inputs if item.id != "cfg"]], old.outputs)

    @classmethod
    def execute(cls, prepared_phase, high_restart, model, sampler, positive, negative, seed=1234, reserve_vram_mib=1024):
        result, report = stages.sample_high(prepared_phase, high_restart, model, sampler,
            positive=positive, negative=negative, seed=seed, reserve_vram_mib=reserve_vram_mib,
            callback=shared._stage_progress(high_restart.plan, "high"))
        return io.NodeOutput(result.output, report, result)


class MiniMaxH3ContinuationRelayProjectEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(cls, "External Relay Window", "Project one external global Relay plan to this accepted "
            "window using original absolute motion time. LOW/HIGH may use separate plan nodes. Connect compiled_prompt "
            "to the corresponding ONE Phase Conditions before Relay Apply. No encoding/sampling.",
            [io.Custom(CONTEXTS).Input("contexts"), io.Custom(PROMPT_RELAY_PLAN_TYPE).Input("global_plan"),
             io.Int.Input("length", default=124, min=5, step=17),
             io.Int.Input("accepted_end_frame", default=-1, min=-1, max=10000000)],
            [io.Custom(RELAY).Output("projected_relay"), io.String.Output("compiled_prompt"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, contexts, global_plan, length=124, accepted_end_frame=-1):
        projected = effects.project_relay(contexts, global_plan, length,
            accepted_end_frame=None if accepted_end_frame == -1 else accepted_end_frame)
        return io.NodeOutput(projected, projected.projected["compiled_prompt"], projected.contract_json)


class MiniMaxH3ContinuationRelayApplyEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(cls, "External Relay ONE Phase", "Bind only this phase's prepared motion layout and projected "
            "events to its actual MODEL and CONDITIONING. Keep LOW and HIGH independent; no opposite-phase dependency. "
            "Apply before continuation EAV. Actual effect calls are recorded by common Progressive Effects Audit nodes.",
            [io.Model.Input("model"), io.Custom(PHASE).Input("prepared_phase"), io.Custom(shared.PLAN).Input("plan"),
             io.Clip.Input("clip"), io.Custom(RELAY).Input("projected_relay"),
             io.Int.Input("query_chunk_rows", default=256, min=32, max=2048),
             io.Combo.Input("mode", options=["disabled", "apply_exp"], default="apply_exp")],
            [io.Model.Output("model"), io.Conditioning.Output("positive"), io.Conditioning.Output("negative")])

    @classmethod
    def execute(cls, model, prepared_phase, plan, clip, projected_relay, query_chunk_rows=256, mode="apply_exp"):
        return io.NodeOutput(*effects.apply_relay(model, prepared_phase, plan, clip, projected_relay,
            query_chunk_rows=query_chunk_rows, mode=mode))


class MiniMaxH3ContinuationLowEAVApplyEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(cls, "External LOW EAV", "Stage EAV Config applied only to this accepted-parent LOW. "
            "Retains native motion guides/masks and global 1-sigma clock. disabled preserves the input MODEL object.",
            [io.Model.Input("model"), io.Custom(PHASE).Input("prepared_phase"),
             io.Custom(shared.PLAN).Input("plan"), io.Custom(CONFIG_TYPE).Input("eav_config")], [io.Model.Output("model")])

    @classmethod
    def execute(cls, model, prepared_phase, plan, eav_config):
        if prepared_phase.phase != "low":
            raise ValueError("LOW EAV requires LOW prepared conditions")
        return io.NodeOutput(effects.apply_eav(model, eav_config, prepared_phase, plan))


class MiniMaxH3ContinuationHighEAVApplyEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(cls, "External HIGH EAV", "Stage EAV Config applied only to this HIGH's typed restart and "
            "original clean-source mask. No LOW dependency. Keeps the accepted completed-prefix motion contract.",
            [io.Model.Input("model"), io.Custom(PHASE).Input("prepared_phase"),
             io.Custom(shared.RESTART).Input("high_restart"), io.Custom(CONFIG_TYPE).Input("eav_config")], [io.Model.Output("model")])

    @classmethod
    def execute(cls, model, prepared_phase, high_restart, eav_config):
        if prepared_phase.phase != "high":
            raise ValueError("HIGH EAV requires HIGH prepared conditions")
        return io.NodeOutput(effects.apply_eav(model, eav_config, prepared_phase, high_restart.plan, restart=high_restart))


class MiniMaxH3ContinuationDeliveryEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(cls, "Deliver Completed Window", "Revalidate the selected accepted parent and return completed "
            "HIGH unchanged. Saved HIGH can load without VAE preparation/LOW/HIGH sampling. mux_audio None means use "
            "native decoded generated audio; an explicitly selected PCM is preserved. Does not write candidate, accept, "
            "append or compose; no equivalence claim for today's edited recipe and previous job.",
            [io.Custom(shared.HIGH_RESULT).Input("high_result"), io.Custom(SOURCE).Input("accepted_source")],
            [io.Latent.Output("av_latent"), io.Audio.Output("mux_audio"), io.String.Output("conditioned_prompt"),
             io.String.Output("media_map_json"), io.String.Output("report_json")], output=True)

    @classmethod
    def execute(cls, high_result, accepted_source):
        result = stages.deliver(high_result, accepted_source)
        return io.NodeOutput(*result, ui={"text": (result[-1],)})


NODES = [MiniMaxH3ContinuationSourceEXPT8, MiniMaxH3ContinuationContextsEXPT8,
    MiniMaxH3ContinuationPlanEXPT8, MiniMaxH3ContinuationConditioningEXPT8,
    MiniMaxH3ContinuationLowStageEXPT8, MiniMaxH3ContinuationHighHandoffEXPT8, MiniMaxH3ContinuationHighStageEXPT8,
    MiniMaxH3ContinuationRelayProjectEXPT8, MiniMaxH3ContinuationRelayApplyEXPT8,
    MiniMaxH3ContinuationLowEAVApplyEXPT8, MiniMaxH3ContinuationHighEAVApplyEXPT8, MiniMaxH3ContinuationDeliveryEXPT8]
