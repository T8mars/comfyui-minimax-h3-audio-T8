"""Append-only public SPEED stage surfaces; legacy SPEED nodes stay unchanged."""
from __future__ import annotations

import json

from comfy_api.latest import io

from ..prompt_relay_advanced import PROMPT_RELAY_PLAN_TYPE
from .speed_effects import apply_relay
from .speed_stages import prepare_speed_stage, sample_speed_stage, handoff_speed_stage


CATEGORY = "T8/MiniMax H3/Modular Sampling/SPEED Experimental"
PLAN = "H3_T8_SPEED_PLAN"
SOURCE = "H3_T8_SPEED_SOURCE"
SPEC = "T8_SPEED_STAGE_SPEC"
RESULT = "T8_SPEED_STAGE_RESULT"


class MiniMaxH3SPEEDStageSetupEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.__name__, display_name="H3 SPEED · Prepare ONE Stage (T8 EXP)",
            category=CATEGORY, is_experimental=True,
            description="Rebuilds exactly one planned H3 AV canvas/conditioning and its native-flow Euler schedule. "
                        "Independent MODEL/LoRA branch per stage. The previous spec enables original T2VA text reuse; "
                        "set reuse=false to rebuild edited stage text. No sampling or hidden fallback. "
                        "External Stage EAV and SPEED Relay adapters are available. Explicit Save/Load accepts "
                        "only completed stages with authenticated execution and effect-call evidence; "
                        "automatic cache reuse is disabled.",
            inputs=[io.Model.Input("model"), io.Custom(PLAN).Input("speed_plan"),
                    io.Custom(SOURCE).Input("speed_source"),
                    io.Int.Input("stage_index", default=0, min=0, max=99),
                    io.Float.Input("shift_audio", default=3.0, min=0.01, max=100.0, step=0.01),
                    io.Int.Input("seed", default=2608184001, min=0, max=0xFFFFFFFFFFFFFFFF),
                    io.Combo.Input("execution_scope", options=["strict_t2va_stock20", "multimodal_research_exp",
                                                               "turbo8_t2va_research_exp"], default="strict_t2va_stock20"),
                    io.Boolean.Input("reuse_t2va_text", default=True),
                    io.Custom(SPEC).Input("previous_spec", optional=True)],
            outputs=[io.Model.Output("model"), io.Conditioning.Output("positive"),
                     io.Latent.Output("av_latent"), io.Sampler.Output("sampler"),
                     io.Sigmas.Output("sigmas"), io.Custom(SPEC).Output("stage_spec"),
                     io.Audio.Output("mux_audio"), io.String.Output("conditioned_prompt"),
                     io.String.Output("media_map_json"), io.String.Output("report_json"),
                     io.Custom("T8_STAGE_CONTEXT").Output("stage_context")],
        )

    @classmethod
    def execute(cls, model, speed_plan, speed_source, stage_index=0, shift_audio=3.0,
                seed=2608184001, execution_scope="strict_t2va_stock20", reuse_t2va_text=True,
                previous_spec=None):
        result = prepare_speed_stage(
            model, speed_plan, speed_source, stage_index=stage_index,
            shift_audio=shift_audio, seed=seed, execution_scope=execution_scope,
            previous_spec=previous_spec, reuse_t2va_text=reuse_t2va_text,
        )
        return io.NodeOutput(*result[:9], json.dumps(result[9], ensure_ascii=False), result[10])


class MiniMaxH3SPEEDStageSampleEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.__name__, display_name="H3 SPEED · Sample ONE Stage (T8 EXP)",
            category=CATEGORY, is_experimental=True,
            description="Calls the native H3 Guider.sample once for this stage's sigmas. MODEL, CONDITIONING, "
                        "SAMPLER and NOISE remain externally wired. Output is the public-flow state used by "
                        "DCT Transition, not a decoded x0. A live result is not portable persistent cache proof.",
            inputs=[io.Model.Input("model"), io.Conditioning.Input("positive"),
                    io.Latent.Input("av_latent"), io.Sampler.Input("sampler"),
                    io.Sigmas.Input("sigmas"), io.Noise.Input("noise"),
                    io.Custom(SPEC).Input("stage_spec")],
            outputs=[io.Latent.Output("av_latent"), io.Custom(RESULT).Output("stage_result"),
                     io.String.Output("report_json")],
        )

    @classmethod
    def execute(cls, model, positive, av_latent, sampler, sigmas, noise, stage_spec):
        output, result, report = sample_speed_stage(model, positive, av_latent, sampler, sigmas, noise, stage_spec)
        return io.NodeOutput(output, result, json.dumps(report, ensure_ascii=False))


class MiniMaxH3SPEEDDCTTransitionEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.__name__, display_name="H3 SPEED · DCT + AV Transition (T8 EXP)",
            category=CATEGORY, is_experimental=True,
            description="Consumes one completed stage and the next independently prepared target. "
                        "Official DCT high-frequency expansion plus H3 audio flow reindex; solves the "
                        "next segment's NOISE. Does not sample or silently reuse old stages.",
            inputs=[io.Custom(RESULT).Input("completed_stage"),
                    io.Custom(SPEC).Input("next_stage"),
                    io.Int.Input("dct_chunk_size", default=64, min=1, max=1024)],
            outputs=[io.Noise.Output("noise"), io.String.Output("report_json")],
        )

    @classmethod
    def execute(cls, completed_stage, next_stage, dct_chunk_size=64):
        noise, report = handoff_speed_stage(completed_stage, next_stage, dct_chunk_size=dct_chunk_size)
        return io.NodeOutput(noise, json.dumps(report, ensure_ascii=False))


class MiniMaxH3SPEEDRelayApplyEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.__name__, display_name="H3 SPEED · External Relay for ONE Stage (T8 EXP)",
            category=CATEGORY, is_experimental=True,
            description="Re-encodes one external Prompt Relay Plan at this stage's exact canvas, "
                        "checks the rebuilt AV target against Stage Setup, and binds MODEL with CONDITIONING. "
                        "Connect its outputs to this stage's Sampler, optionally through Stage EAV Apply. "
                        "Inspect downstream actual-call audit; this node alone is not effect execution proof.",
            inputs=[io.Model.Input("model"), io.Custom(SPEC).Input("stage_spec"),
                    io.Custom(SOURCE).Input("speed_source"),
                    io.Custom(PROMPT_RELAY_PLAN_TYPE).Input("prompt_relay_plan"),
                    io.Combo.Input("execution_mode", options=["report_only", "apply_exp"], default="report_only"),
                    io.Int.Input("query_chunk_rows", default=256, min=32, max=2048)],
            outputs=[io.Model.Output("model"), io.Conditioning.Output("positive"),
                     io.Latent.Output("av_latent"), io.Audio.Output("mux_audio"),
                     io.String.Output("conditioned_prompt"), io.String.Output("media_map_json"),
                     io.String.Output("report_json")],
        )

    @classmethod
    def execute(cls, model, stage_spec, speed_source, prompt_relay_plan,
                execution_mode="report_only", query_chunk_rows=256):
        return io.NodeOutput(*apply_relay(
            model, stage_spec, speed_source, prompt_relay_plan,
            execution_mode=execution_mode, query_chunk_rows=query_chunk_rows,
        ))


NODES = [MiniMaxH3SPEEDStageSetupEXPT8, MiniMaxH3SPEEDStageSampleEXPT8,
         MiniMaxH3SPEEDDCTTransitionEXPT8, MiniMaxH3SPEEDRelayApplyEXPT8]
