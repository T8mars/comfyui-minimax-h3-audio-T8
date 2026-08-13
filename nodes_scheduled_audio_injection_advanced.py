from __future__ import annotations

from comfy_api.latest import io

from .sampling import DEFAULT_SCHEDULER_NAME, SCHEDULER_OPTIONS
from .scheduled_audio_injection_advanced import (
    ENVELOPES,
    MODES,
    setup_scheduled_drive_audio_injection,
)


CATEGORY = "T8/MiniMax H3/Audio/Experimental"


class MiniMaxH3ScheduledDriveAudioInjectionT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3ScheduledDriveAudioInjectionT8Advanced",
            display_name="MiniMax H3 Scheduled Drive Audio Injection (T8 Advanced)",
            description=(
                "Experimental dual-clock sampler that repeatedly anchors the complete supplied "
                "drive-audio latent on its own sigma path. It does not isolate speech from music "
                "or effects. report_only is an unchanged dual-clock bypass."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Model.Input("model"),
                io.Latent.Input("av_latent"),
                io.Audio.Input("drive_audio"),
                io.Vae.Input("audio_vae"),
                io.Int.Input("steps", default=4, min=1, max=1000),
                io.Float.Input(
                    "shift_video",
                    default=12.0,
                    min=0.01,
                    max=100.0,
                    step=0.01,
                    advanced=True,
                ),
                io.Float.Input(
                    "shift_audio",
                    default=3.0,
                    min=0.01,
                    max=100.0,
                    step=0.01,
                    advanced=True,
                ),
                io.Combo.Input("mode", options=list(MODES), default="report_only"),
                io.Float.Input(
                    "start_percent",
                    default=0.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="0 is the first/high-noise denoise step.",
                ),
                io.Float.Input(
                    "end_percent",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="1 is the last/low-noise denoise step.",
                ),
                io.Float.Input("strength", default=1.0, min=0.0, max=1.0, step=0.01),
                io.Combo.Input("envelope", options=list(ENVELOPES), default="constant"),
                io.Int.Input(
                    "injection_seed",
                    default=0,
                    min=0,
                    max=0x7FFFFFFFFFFFFFFF,
                    control_after_generate=True,
                ),
                io.Boolean.Input(
                    "lock_final_audio",
                    default=False,
                    tooltip=(
                        "Replace final generated audio latent with encoded drive audio. "
                        "Use mux_audio for unchanged source PCM."
                    ),
                ),
                io.Combo.Input(
                    "scheduler",
                    options=SCHEDULER_OPTIONS,
                    default=DEFAULT_SCHEDULER_NAME,
                    optional=True,
                    display_name="scheduler / 调度器",
                ),
                io.Boolean.Input(
                    "allow_unverified_patch_stack",
                    default=False,
                    advanced=True,
                    tooltip=(
                        "Allow patches_replace/LongVideo/MultiKeyframe combinations that have "
                        "not passed the Scheduled Audio validation matrix."
                    ),
                ),
                io.Audio.Input("final_audio", optional=True),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
                io.Sampler.Output(display_name="sampler"),
                io.Sigmas.Output(display_name="sigmas"),
                io.Latent.Output(display_name="av_latent"),
                io.Audio.Output(display_name="mux_audio"),
                io.String.Output(display_name="report_json"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model,
        av_latent,
        drive_audio,
        audio_vae,
        steps,
        shift_video,
        shift_audio,
        mode,
        start_percent,
        end_percent,
        strength,
        envelope,
        injection_seed,
        lock_final_audio,
        scheduler=DEFAULT_SCHEDULER_NAME,
        allow_unverified_patch_stack=False,
        final_audio=None,
    ):
        return io.NodeOutput(*setup_scheduled_drive_audio_injection(
            model,
            av_latent,
            drive_audio,
            audio_vae,
            steps,
            shift_video,
            shift_audio,
            mode,
            start_percent,
            end_percent,
            strength,
            envelope,
            injection_seed,
            lock_final_audio,
            scheduler,
            allow_unverified_patch_stack,
            final_audio,
        ))


SCHEDULED_AUDIO_INJECTION_ADVANCED_NODE_CLASSES = [
    MiniMaxH3ScheduledDriveAudioInjectionT8Advanced
]
