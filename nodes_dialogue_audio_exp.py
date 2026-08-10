from __future__ import annotations

from comfy_api.latest import io

from .dialogue_audio import (
    BED_FIT_POLICIES,
    SFX_FIT_POLICIES,
    build_dialogue_safe_master,
)
from .speech import SPEECH_CATEGORY
from .speech_verification import ASR_LANGUAGE_CODES, analyze_dialogue_boundary
from .timed_audio_latent import (
    AUDIO_LATENT_FIT_POLICIES,
    build_timed_audio_bed_lock,
)


class MiniMaxH3DialogueBoundaryAnalyzerT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3DialogueBoundaryAnalyzerT8",
            display_name="MiniMax H3 Dialogue Boundary Analyzer / 对白边界分析 (EXP/T8)",
            description=(
                "Runs local CPU faster-whisper and reports a boundary only when exactly one "
                "contiguous exact target-text sequence is found. It never edits audio and does "
                "not infer that tail energy is speech."
            ),
            category=SPEECH_CATEGORY,
            is_experimental=True,
            inputs=[
                io.Audio.Input("audio"),
                io.String.Input("expected_text", multiline=True, force_input=True),
                io.String.Input(
                    "asr_model_directory",
                    default="",
                    tooltip=(
                        "Absolute faster-whisper CTranslate2 directory, or a folder under "
                        "ComfyUI/models/TTS. No model is downloaded automatically."
                    ),
                ),
                io.Combo.Input(
                    "language",
                    options=list(ASR_LANGUAGE_CODES),
                    default="auto",
                ),
                io.Int.Input("beam_size", default=5, min=1, max=20, advanced=True),
                io.Int.Input("cpu_threads", default=8, min=1, max=64, advanced=True),
                io.Boolean.Input("unload_after_analyze", default=True),
                io.Float.Input(
                    "tail_activity_threshold_dbfs",
                    default=-45.0,
                    min=-120.0,
                    max=0.0,
                    step=0.5,
                    advanced=True,
                    tooltip=(
                        "Only reports post-target signal activity. This is not a VAD or a "
                        "speech/non-speech decision."
                    ),
                ),
            ],
            outputs=[
                io.String.Output("transcript"),
                io.Boolean.Output("unique_target_found"),
                io.Boolean.Output("clean_exact"),
                io.Float.Output("speech_start_seconds"),
                io.Float.Output("speech_end_seconds"),
                io.Int.Output("extra_before_units"),
                io.Int.Output("extra_after_units"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(
        cls,
        audio,
        expected_text,
        asr_model_directory,
        language,
        beam_size,
        cpu_threads,
        unload_after_analyze,
        tail_activity_threshold_dbfs,
    ):
        return io.NodeOutput(
            *analyze_dialogue_boundary(
                audio,
                expected_text,
                asr_model_directory,
                language,
                beam_size,
                cpu_threads,
                unload_after_analyze,
                tail_activity_threshold_dbfs,
            )
        )


class MiniMaxH3DialogueSafeMasterT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3DialogueSafeMasterT8",
            display_name="MiniMax H3 Dialogue Safe Master / 对白安全混音 (EXP/T8)",
            description=(
                "Builds an exact-duration master from an already verified speech stem plus "
                "independent music, ambience, and SFX stems. Speech is never silently trimmed, "
                "and the master is never cut at the dialogue end. This is not source separation."
            ),
            category=SPEECH_CATEGORY,
            is_experimental=True,
            inputs=[
                io.Audio.Input("speech_audio"),
                io.Boolean.Input(
                    "speech_accepted",
                    default=False,
                    force_input=True,
                    tooltip=(
                        "Connect accepted from Speech Verify after exact-target alignment. "
                        "False is rejected; this mixer will not guess speech cleanliness."
                    ),
                ),
                io.Float.Input(
                    "target_duration_seconds",
                    default=10.0,
                    min=0.001,
                    max=36000.0,
                    step=0.001,
                ),
                io.Float.Input(
                    "speech_start_seconds",
                    default=0.0,
                    min=0.0,
                    max=36000.0,
                    step=0.001,
                ),
                io.Combo.Input(
                    "output_sample_rate",
                    options=[32000, 44100, 48000],
                    default=32000,
                ),
                io.Combo.Input(
                    "music_fit_policy",
                    options=list(BED_FIT_POLICIES),
                    default="strict",
                ),
                io.Combo.Input(
                    "ambience_fit_policy",
                    options=list(BED_FIT_POLICIES),
                    default="strict",
                ),
                io.Combo.Input(
                    "sfx_fit_policy",
                    options=list(SFX_FIT_POLICIES),
                    default="strict",
                    tooltip="SFX can be padded/trimmed explicitly but is never looped.",
                ),
                io.Float.Input(
                    "loop_crossfade_seconds",
                    default=0.25,
                    min=0.0,
                    max=5.0,
                    step=0.01,
                    advanced=True,
                ),
                io.Float.Input("speech_gain_db", default=0.0, min=-60.0, max=24.0, step=0.1),
                io.Float.Input("music_gain_db", default=0.0, min=-60.0, max=24.0, step=0.1),
                io.Float.Input("ambience_gain_db", default=0.0, min=-60.0, max=24.0, step=0.1),
                io.Float.Input("sfx_gain_db", default=0.0, min=-60.0, max=24.0, step=0.1),
                io.Float.Input(
                    "duck_background",
                    default=0.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Deterministically attenuates all background stems while speech is active.",
                ),
                io.Float.Input(
                    "peak_limit_dbfs",
                    default=-1.0,
                    min=-24.0,
                    max=0.0,
                    step=0.1,
                ),
                io.Audio.Input("music_audio", optional=True),
                io.Audio.Input("ambience_audio", optional=True),
                io.Audio.Input("sfx_audio", optional=True),
            ],
            outputs=[
                io.Audio.Output("master_audio"),
                io.Audio.Output("speech_stem"),
                io.Audio.Output("background_stem"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(
        cls,
        speech_audio,
        speech_accepted,
        target_duration_seconds,
        speech_start_seconds,
        output_sample_rate,
        music_fit_policy,
        ambience_fit_policy,
        sfx_fit_policy,
        loop_crossfade_seconds,
        speech_gain_db,
        music_gain_db,
        ambience_gain_db,
        sfx_gain_db,
        duck_background,
        peak_limit_dbfs,
        music_audio=None,
        ambience_audio=None,
        sfx_audio=None,
    ):
        return io.NodeOutput(
            *build_dialogue_safe_master(
                speech_audio,
                speech_accepted,
                target_duration_seconds,
                speech_start_seconds,
                output_sample_rate,
                music_fit_policy,
                ambience_fit_policy,
                sfx_fit_policy,
                loop_crossfade_seconds,
                speech_gain_db,
                music_gain_db,
                ambience_gain_db,
                sfx_gain_db,
                duck_background,
                peak_limit_dbfs,
                music_audio,
                ambience_audio,
                sfx_audio,
            )
        )


class MiniMaxH3TimedAudioBedLockT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3TimedAudioBedLockT8",
            display_name=(
                "MiniMax H3 Timed Background Bed Lock / 分时背景底轨锁定 (EXP/T8)"
            ),
            description=(
                "Two-pass H3 helper: encodes an independent full-duration music/ambience/SFX "
                "bed into the AV latent and locks its tail after an explicit dialogue boundary. "
                "It preserves the video stream and does not separate speech from a mixed master."
            ),
            category=SPEECH_CATEGORY,
            is_experimental=True,
            inputs=[
                io.Latent.Input("av_latent"),
                io.Audio.Input("background_audio"),
                io.Vae.Input("audio_vae"),
                io.Float.Input(
                    "tail_lock_start_seconds",
                    default=5.0,
                    min=0.0,
                    max=3600.0,
                    step=0.001,
                    force_input=True,
                    tooltip=(
                        "Connect a verified speech_end_seconds or an explicit planned boundary. "
                        "It is quantized upward to the H3 40Hz audio-latent grid."
                    ),
                ),
                io.Float.Input(
                    "head_denoise_strength",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                ),
                io.Float.Input(
                    "tail_denoise_strength",
                    default=0.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Use 0 to request an unchanged background-bed latent endpoint.",
                ),
                io.Float.Input(
                    "transition_seconds",
                    default=0.0,
                    min=0.0,
                    max=10.0,
                    step=0.025,
                    advanced=True,
                    tooltip=(
                        "Default 0 makes the lock boundary explicit. A nonzero transition delays "
                        "the fully locked tail and is reported, not assumed perceptually safer."
                    ),
                ),
                io.Combo.Input(
                    "audio_latent_fit_policy",
                    options=list(AUDIO_LATENT_FIT_POLICIES),
                    default="strict",
                    tooltip=(
                        "Strict rejects a duration mismatch. fit_reported explicitly trims or "
                        "zero-pads the encoded latent and records the action."
                    ),
                ),
            ],
            outputs=[
                io.Latent.Output("av_latent"),
                io.Latent.Output("audio_latent"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(
        cls,
        av_latent,
        background_audio,
        audio_vae,
        tail_lock_start_seconds,
        head_denoise_strength,
        tail_denoise_strength,
        transition_seconds,
        audio_latent_fit_policy,
    ):
        return io.NodeOutput(
            *build_timed_audio_bed_lock(
                av_latent,
                background_audio,
                audio_vae,
                tail_lock_start_seconds,
                head_denoise_strength,
                tail_denoise_strength,
                transition_seconds,
                audio_latent_fit_policy,
            )
        )


DIALOGUE_AUDIO_NODE_CLASSES = [
    MiniMaxH3DialogueBoundaryAnalyzerT8,
    MiniMaxH3DialogueSafeMasterT8,
    MiniMaxH3TimedAudioBedLockT8,
]
