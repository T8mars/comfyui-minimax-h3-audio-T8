from __future__ import annotations

from comfy_api.latest import ComfyExtension, io

from .audio_ops import decode_av_latent, inject_audio_latent, mix_audio, trim_av_output
from .conditioning import build_conditioning
from .nodes_multirate_exp import MiniMaxH3MultiRateSamplerEXPT8
from .nodes_still_exp import (
    MiniMaxH3StillConditioningT8,
    MiniMaxH3StillDecodeT8,
    MiniMaxH3StillPreflightT8,
)
from .preflight import run_preflight
from .prompt_tags import prepare_prompt
from .sampling import (
    DEFAULT_SAMPLER_NAME,
    DEFAULT_SCHEDULER_NAME,
    SAMPLER_OPTIONS,
    SCHEDULER_OPTIONS,
    setup_dual_clock_sampling,
)
from .timing import make_timing_plan, window_audio


CATEGORY = "T8/MiniMax H3/Audio"
MAX_RESOLUTION = 16384


class MiniMaxH3AudioConditioningT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3AudioConditioningT8",
            display_name="MiniMax H3 Audio Conditioning (T8)",
            description="Unified native H3 T2VA/I2VA/FL2VA/L2VA/Ref2VA/hybrid conditioning with correct media tags and source-audio control.",
            category=CATEGORY,
            inputs=[
                io.Clip.Input("clip", tooltip="Native MiniMax H3 Qwen3-VL CLIP."),
                io.Vae.Input("video_vae", tooltip="MiniMax H3 video VAE."),
                io.Vae.Input("audio_vae", tooltip="MiniMax H3 audio VAE."),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Int.Input("width", default=1344, min=32, max=MAX_RESOLUTION, step=32),
                io.Int.Input("height", default=768, min=32, max=MAX_RESOLUTION, step=32),
                io.Int.Input("length", default=124, min=5, max=3600, step=17, tooltip="24fps; snapped up to the 17n+5 H3 grid."),
                io.Combo.Input("task_type", options=["auto", "T2VA", "I2VA", "FL2VA", "L2VA", "Ref2VA", "Hybrid"], default="auto"),
                io.Combo.Input("audio_mode", options=["lock_source", "remix_source", "reference_only", "native"], default="lock_source", tooltip="lock_source preserves source latent; remix_source denoises it; reference_only/native generate target audio."),
                io.Float.Input("audio_denoise_strength", default=0.35, min=0.0, max=1.0, step=0.01, advanced=True),
                io.Boolean.Input("add_source_as_reference", default=True, tooltip="Presents drive_audio to Qwen/DiT as an official <Audio N> reference."),
                io.Int.Input("prompt_primary_audio_ordinal", default=1, min=0, max=9, step=1, tooltip="Prompt audio ordinal intended as the primary source; remapped after video soundtracks. Use 0 to disable.", advanced=True),
                io.Boolean.Input("strict_prompt_tags", default=True, advanced=True),
                io.Combo.Input("ref_image_size", options=["match", "max"], default="match", advanced=True),
                io.Combo.Input("reference_video_policy", options=["official_2_to_15s", "model_minimum"], default="official_2_to_15s", advanced=True),
                io.Audio.Input("drive_audio", optional=True),
                io.Audio.Input("final_audio", optional=True, tooltip="Optional clean/stem track passed through for final mux; defaults to drive_audio."),
                io.Image.Input("first_frame", optional=True),
                io.Image.Input("last_frame", optional=True),
                io.Autogrow.Input("ref_images", optional=True, template=io.Autogrow.TemplatePrefix(input=io.Image.Input("ref_image"), prefix="ref_image_", min=0, max=9)),
                io.Autogrow.Input("ref_videos", optional=True, template=io.Autogrow.TemplatePrefix(input=io.Image.Input("ref_video", tooltip="IMAGE frame batch at 24fps."), prefix="ref_video_", min=0, max=3)),
                io.Autogrow.Input("ref_video_audios", optional=True, template=io.Autogrow.TemplatePrefix(input=io.Audio.Input("ref_video_audio"), prefix="ref_video_audio_", min=0, max=3)),
                io.Autogrow.Input("ref_audios", optional=True, template=io.Autogrow.TemplatePrefix(input=io.Audio.Input("ref_audio"), prefix="ref_audio_", min=0, max=3)),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Latent.Output(display_name="av_latent"),
                io.Audio.Output(display_name="mux_audio"),
                io.String.Output(display_name="conditioned_prompt"),
                io.String.Output(display_name="media_map_json"),
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def execute(cls, clip, video_vae, audio_vae, prompt, width, height, length, task_type, audio_mode,
                audio_denoise_strength, add_source_as_reference, prompt_primary_audio_ordinal,
                strict_prompt_tags, ref_image_size, reference_video_policy, drive_audio=None,
                final_audio=None, first_frame=None, last_frame=None, ref_images=None, ref_videos=None,
                ref_video_audios=None, ref_audios=None):
        return io.NodeOutput(*build_conditioning(
            clip, video_vae, audio_vae, prompt, width, height, length, task_type, audio_mode,
            audio_denoise_strength, add_source_as_reference, prompt_primary_audio_ordinal,
            strict_prompt_tags, ref_image_size, reference_video_policy, drive_audio, final_audio,
            first_frame, last_frame, ref_images, ref_videos, ref_video_audios, ref_audios,
        ))


class MiniMaxH3AudioLatentControlT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3AudioLatentControlT8",
            display_name="MiniMax H3 Audio Latent Control (T8)",
            description="Injects source audio once and preserves an existing video noise mask.",
            category=CATEGORY,
            inputs=[
                io.Latent.Input("av_latent"), io.Audio.Input("source_audio"), io.Vae.Input("audio_vae"),
                io.Combo.Input("mode", options=["lock", "remix"], default="lock"),
                io.Float.Input("strength", default=0.35, min=0.0, max=1.0, step=0.01),
            ],
            outputs=[io.Latent.Output(display_name="av_latent"), io.Audio.Output(display_name="source_audio")],
        )

    @classmethod
    def execute(cls, av_latent, source_audio, audio_vae, mode, strength):
        return io.NodeOutput(*inject_audio_latent(av_latent, source_audio, audio_vae, mode, strength))


class MiniMaxH3DurationPlannerT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3DurationPlannerT8",
            display_name="MiniMax H3 Duration Planner (T8)",
            category=CATEGORY,
            inputs=[
                io.Float.Input("scene_start_seconds", default=0.0, min=0.0, max=86400.0, step=0.01),
                io.Float.Input("scene_duration_seconds", default=5.0, min=0.04, max=900.0, step=0.01),
                io.Float.Input("warmup_seconds", default=0.0, min=0.0, max=60.0, step=0.01),
                io.Float.Input("cooldown_seconds", default=0.0, min=0.0, max=60.0, step=0.01),
                io.Boolean.Input("ensure_minimum_context", default=True),
                io.Float.Input("source_duration_seconds", default=0.0, min=0.0, max=86400.0, step=0.01, advanced=True, tooltip="0 means unknown; the Audio Window node reads it from AUDIO."),
            ],
            outputs=[
                io.Int.Output("length"), io.Float.Output("render_duration_seconds"),
                io.Float.Output("source_slice_start_seconds"), io.Float.Output("source_slice_duration_seconds"),
                io.Float.Output("final_trim_start_seconds"), io.Float.Output("final_duration_seconds"),
                io.String.Output("prompt_timing_note"), io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, scene_start_seconds, scene_duration_seconds, warmup_seconds, cooldown_seconds,
                ensure_minimum_context, source_duration_seconds):
        plan = make_timing_plan(scene_start_seconds, scene_duration_seconds, warmup_seconds,
                                cooldown_seconds, ensure_minimum_context, source_duration_seconds)
        return io.NodeOutput(plan.frame_count, plan.render_duration_seconds, plan.source_slice_start_seconds,
                             plan.source_slice_duration_seconds, plan.final_trim_start_seconds,
                             plan.final_duration_seconds, plan.prompt_note(), plan.report())


class MiniMaxH3AudioWindowT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3AudioWindowT8",
            display_name="MiniMax H3 Audio Window (T8)",
            description="Slices/pads source AUDIO to an aligned H3 context and returns exact final trim metadata.",
            category=CATEGORY,
            inputs=[
                io.Audio.Input("audio"),
                io.Float.Input("scene_start_seconds", default=0.0, min=0.0, max=86400.0, step=0.01),
                io.Float.Input("scene_duration_seconds", default=5.0, min=0.04, max=900.0, step=0.01),
                io.Float.Input("warmup_seconds", default=0.0, min=0.0, max=60.0, step=0.01),
                io.Float.Input("cooldown_seconds", default=0.0, min=0.0, max=60.0, step=0.01),
                io.Boolean.Input("ensure_minimum_context", default=True),
            ],
            outputs=[io.Audio.Output("context_audio"), io.Int.Output("length"),
                     io.Float.Output("final_trim_start_seconds"), io.Float.Output("final_duration_seconds"),
                     io.String.Output("prompt_timing_note"), io.String.Output("report_json")],
        )

    @classmethod
    def execute(cls, audio, scene_start_seconds, scene_duration_seconds, warmup_seconds, cooldown_seconds,
                ensure_minimum_context):
        source_duration = audio["waveform"].shape[-1] / int(audio["sample_rate"])
        plan = make_timing_plan(scene_start_seconds, scene_duration_seconds, warmup_seconds,
                                cooldown_seconds, ensure_minimum_context, source_duration)
        return io.NodeOutput(window_audio(audio, plan), plan.frame_count, plan.final_trim_start_seconds,
                             plan.final_duration_seconds, plan.prompt_note(), plan.report())


class MiniMaxH3PromptTagsT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3PromptTagsT8", display_name="MiniMax H3 Prompt Tags (T8)", category=CATEGORY,
            inputs=[
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Int.Input("picture_count", default=0, min=0, max=11),
                io.Int.Input("video_count", default=0, min=0, max=3),
                io.Int.Input("audio_count", default=1, min=0, max=9),
                io.Int.Input("source_audio_ordinal", default=1, min=0, max=9),
                io.Int.Input("prompt_primary_audio_ordinal", default=1, min=0, max=9),
                io.Boolean.Input("strict", default=True),
            ], outputs=[io.String.Output("prompt"), io.String.Output("report")],
        )

    @classmethod
    def execute(cls, prompt, picture_count, video_count, audio_count, source_audio_ordinal,
                prompt_primary_audio_ordinal, strict):
        normalized, warnings = prepare_prompt(prompt, {"pictures": picture_count, "videos": video_count,
                                                       "audios": audio_count}, source_audio_ordinal,
                                              prompt_primary_audio_ordinal, strict)
        return io.NodeOutput(normalized, "OK" if not warnings else "\n".join(warnings))


class MiniMaxH3AVDecodeT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3AVDecodeT8", display_name="MiniMax H3 AV Decode (T8)", category=CATEGORY,
            inputs=[io.Latent.Input("av_latent"), io.Vae.Input("video_vae"), io.Vae.Input("audio_vae")],
            outputs=[io.Image.Output("frames"), io.Audio.Output("generated_audio"),
                     io.Latent.Output("video_latent"), io.Latent.Output("audio_latent")],
        )

    @classmethod
    def execute(cls, av_latent, video_vae, audio_vae):
        return io.NodeOutput(*decode_av_latent(av_latent, video_vae, audio_vae))


class MiniMaxH3AudioMixT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3AudioMixT8", display_name="MiniMax H3 Audio Mix (T8)", category=CATEGORY,
            inputs=[
                io.Audio.Input("source_audio"), io.Audio.Input("generated_audio"),
                io.Float.Input("source_gain_db", default=0.0, min=-60.0, max=24.0, step=0.1),
                io.Float.Input("generated_gain_db", default=-6.0, min=-60.0, max=24.0, step=0.1),
                io.Float.Input("duck_generated", default=0.5, min=0.0, max=1.0, step=0.01),
                io.Combo.Input("output_sample_rate", options=["source", "generated", "48000", "44100", "32000"], default="source"),
                io.Float.Input("peak_limit_dbfs", default=-1.0, min=-12.0, max=0.0, step=0.1),
            ], outputs=[io.Audio.Output("mixed_audio")],
        )

    @classmethod
    def execute(cls, source_audio, generated_audio, source_gain_db, generated_gain_db,
                duck_generated, output_sample_rate, peak_limit_dbfs):
        return io.NodeOutput(mix_audio(source_audio, generated_audio, source_gain_db, generated_gain_db,
                                       duck_generated, output_sample_rate, peak_limit_dbfs))


class MiniMaxH3OutputTrimT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3OutputTrimT8", display_name="MiniMax H3 Output Trim (T8)", category=CATEGORY,
            description="Applies Duration Planner trim metadata to decoded IMAGE frames and optional AUDIO.",
            inputs=[
                io.Image.Input("frames"),
                io.Float.Input("start_seconds", default=0.0, min=0.0, max=900.0, step=0.001),
                io.Float.Input("duration_seconds", default=5.0, min=0.04, max=900.0, step=0.001),
                io.Float.Input("fps", default=24.0, min=1.0, max=240.0, step=0.001, advanced=True),
                io.Audio.Input("audio", optional=True),
            ],
            outputs=[io.Image.Output("frames"), io.Audio.Output("audio"), io.String.Output("report_json")],
        )

    @classmethod
    def execute(cls, frames, start_seconds, duration_seconds, fps, audio=None):
        return io.NodeOutput(*trim_av_output(frames, start_seconds, duration_seconds, audio, fps))


class MiniMaxH3PreflightT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3PreflightT8", display_name="MiniMax H3 Preflight (T8)", category=CATEGORY,
            inputs=[
                io.Int.Input("width", default=1344, min=32, max=MAX_RESOLUTION, step=32),
                io.Int.Input("height", default=768, min=32, max=MAX_RESOLUTION, step=32),
                io.Int.Input("length", default=124, min=5, max=3600),
                io.Combo.Input("audio_mode", options=["lock_source", "remix_source", "reference_only", "native"], default="lock_source"),
                io.Model.Input("model", optional=True), io.Vae.Input("video_vae", optional=True),
                io.Vae.Input("audio_vae", optional=True), io.Audio.Input("drive_audio", optional=True),
                io.Autogrow.Input("ref_images", optional=True, template=io.Autogrow.TemplatePrefix(input=io.Image.Input("ref_image"), prefix="ref_image_", min=0, max=9)),
                io.Autogrow.Input("ref_videos", optional=True, template=io.Autogrow.TemplatePrefix(input=io.Image.Input("ref_video"), prefix="ref_video_", min=0, max=3)),
                io.Autogrow.Input("ref_audios", optional=True, template=io.Autogrow.TemplatePrefix(input=io.Audio.Input("ref_audio"), prefix="ref_audio_", min=0, max=3)),
            ], outputs=[io.Boolean.Output("ready"), io.Int.Output("warning_count"), io.String.Output("report_json")],
        )

    @classmethod
    def execute(cls, width, height, length, audio_mode, model=None, video_vae=None, audio_vae=None,
                drive_audio=None, ref_images=None, ref_videos=None, ref_audios=None):
        return io.NodeOutput(*run_preflight(width, height, length, audio_mode, model, video_vae, audio_vae,
                                            drive_audio, ref_images, ref_videos, ref_audios))


class MiniMaxH3DualClockSamplerT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3DualClockSamplerT8",
            display_name="MiniMax H3 Dual-Clock Sampler (T8)",
            description=(
                "MiniMax H3 sampling setup with separate video/audio clocks. "
                "The default dual_clock_euler + native_flow path is unchanged; other ComfyUI "
                "samplers use native FLOW_AV support."
            ),
            category=CATEGORY,
            inputs=[
                io.Model.Input("model"),
                io.Latent.Input("av_latent"),
                io.Int.Input("steps", default=4, min=1, max=1000),
                io.Float.Input("shift_video", default=12.0, min=0.01, max=100.0, step=0.01, advanced=True),
                io.Float.Input("shift_audio", default=3.0, min=0.01, max=100.0, step=0.01, advanced=True),
                io.Combo.Input(
                    "sampler_name",
                    options=SAMPLER_OPTIONS,
                    default=DEFAULT_SAMPLER_NAME,
                    optional=True,
                    display_name="sampler / 采样器",
                    tooltip=(
                        "dual_clock_euler preserves the original T8 explicit dual-clock path. "
                        "Other choices use ComfyUI's native MiniMax H3 FLOW_AV protocol."
                    ),
                ),
                io.Combo.Input(
                    "scheduler",
                    options=SCHEDULER_OPTIONS,
                    default=DEFAULT_SCHEDULER_NAME,
                    optional=True,
                    display_name="scheduler / 调度器",
                    tooltip=(
                        "native_flow preserves the original shifted uniform H3 flow schedule. "
                        "Other choices use ComfyUI's built-in scheduler implementation."
                    ),
                ),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
                io.Sampler.Output(display_name="sampler"),
                io.Sigmas.Output(display_name="sigmas"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model,
        av_latent,
        steps,
        shift_video,
        shift_audio,
        sampler_name=DEFAULT_SAMPLER_NAME,
        scheduler=DEFAULT_SCHEDULER_NAME,
    ):
        return io.NodeOutput(*setup_dual_clock_sampling(
            model,
            av_latent,
            steps,
            shift_video,
            shift_audio,
            sampler_name,
            scheduler,
        ))


class MiniMaxH3AudioT8Extension(ComfyExtension):
    async def get_node_list(self):
        return [MiniMaxH3AudioConditioningT8, MiniMaxH3AudioLatentControlT8,
                MiniMaxH3DurationPlannerT8, MiniMaxH3AudioWindowT8, MiniMaxH3PromptTagsT8,
                MiniMaxH3AVDecodeT8, MiniMaxH3AudioMixT8, MiniMaxH3OutputTrimT8,
                MiniMaxH3PreflightT8, MiniMaxH3DualClockSamplerT8,
                MiniMaxH3MultiRateSamplerEXPT8, MiniMaxH3StillConditioningT8,
                MiniMaxH3StillPreflightT8, MiniMaxH3StillDecodeT8]


def comfy_entrypoint():
    return MiniMaxH3AudioT8Extension()
