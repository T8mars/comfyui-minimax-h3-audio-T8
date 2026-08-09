from __future__ import annotations

import hashlib
import json
from pathlib import Path

import folder_paths

from comfy_api.latest import io
from comfy_execution.graph_utils import GraphBuilder

from .long_video_background import ComfyQueueRuntime, RELEASE_POLICIES
from .sampling import (
    DEFAULT_SAMPLER_NAME,
    DEFAULT_SCHEDULER_NAME,
    SAMPLER_OPTIONS,
    SCHEDULER_OPTIONS,
)
from .speech import (
    SPACE_DESCRIPTIONS,
    SPEECH_CATEGORY,
    SPEECH_PLAN_TYPE,
    SUPPORTED_DIALOGUE_LANGUAGES,
    VOICE_PROFILE_TYPE,
    assemble_speech_audio,
    build_speech_conditioning,
    decode_speech_audio,
    make_dialogue_plan,
    make_speech_plan,
    make_voice_profile,
    public_plan,
    select_dialogue_turn,
    validate_speech_plan,
)
from .speech_verification import ASR_LANGUAGE_CODES, verify_speech_audio
from .speech_reliability import (
    SPEECH_GUARD_TYPE,
    arm_speech_guard,
    complete_speech_guard,
    vram_preflight,
)
from .speech_extended import (
    SPEECH_SESSION_TYPE,
    accept_longform_segment,
    apply_performance_direction,
    build_joint_dialogue_conditioning,
    compose_longform_session,
    control_longform_session,
    delete_voice_profile,
    fit_audio_for_adr,
    load_voice_profile,
    save_voice_profile,
    speech_manifest_fingerprint,
    start_or_resume_longform,
)


VoiceProfile = io.Custom(VOICE_PROFILE_TYPE)
SpeechPlan = io.Custom(SPEECH_PLAN_TYPE)
SpeechGuard = io.Custom(SPEECH_GUARD_TYPE)
SpeechSession = io.Custom(SPEECH_SESSION_TYPE)

SPEECH_SAMPLERS = list(SAMPLER_OPTIONS)
if "res_multistep" in SPEECH_SAMPLERS:
    SPEECH_SAMPLERS.remove("res_multistep")
    SPEECH_SAMPLERS.insert(0, "res_multistep")
SPEECH_SCHEDULERS = list(SCHEDULER_OPTIONS)
if "simple" in SPEECH_SCHEDULERS:
    SPEECH_SCHEDULERS.remove("simple")
    SPEECH_SCHEDULERS.insert(0, "simple")


class MiniMaxH3VoiceProfileT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3VoiceProfileT8",
            display_name="MiniMax H3 Voice Profile / 音色资料 (EXP/T8)",
            description=(
                "Creates a workflow-memory voice profile. Reference mode prepares a "
                "2-15s 32kHz stereo anchor and requires explicit rights confirmation."
            ),
            category=SPEECH_CATEGORY,
            inputs=[
                io.Combo.Input(
                    "voice_mode",
                    options=["described_voice", "reference_voice"],
                    default="described_voice",
                ),
                io.String.Input("speaker_id", default="speaker_1"),
                io.String.Input(
                    "voice_description",
                    default=(
                        "an adult speaker with a natural warm voice, clear diction, "
                        "human micro-pauses, and close conversational delivery"
                    ),
                    multiline=True,
                ),
                io.Combo.Input(
                    "language",
                    options=list(SUPPORTED_DIALOGUE_LANGUAGES),
                    default="Chinese",
                ),
                io.Boolean.Input(
                    "rights_confirmed",
                    default=False,
                    tooltip=(
                        "Required for reference_voice. Confirm that you have consent or "
                        "another lawful right to use the connected person's voice."
                    ),
                ),
                io.Float.Input(
                    "reference_start_seconds",
                    default=0.0,
                    min=0.0,
                    max=86400.0,
                    step=0.01,
                    advanced=True,
                ),
                io.Float.Input(
                    "reference_duration_seconds",
                    default=0.0,
                    min=0.0,
                    max=15.0,
                    step=0.01,
                    tooltip="0 selects up to the first available 15 seconds after start.",
                    advanced=True,
                ),
                io.Boolean.Input("highpass_60hz", default=True, advanced=True),
                io.Boolean.Input(
                    "peak_limit_minus_3_dbfs",
                    default=True,
                    tooltip="Only attenuates peaks above -3 dBFS; it never boosts quiet references.",
                    advanced=True,
                ),
                io.Audio.Input("reference_audio", optional=True),
            ],
            outputs=[
                VoiceProfile.Output("voice_profile"),
                io.Audio.Output("prepared_reference_audio"),
                io.String.Output("report_json"),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(
        cls,
        voice_mode,
        speaker_id,
        voice_description,
        language,
        rights_confirmed,
        reference_start_seconds,
        reference_duration_seconds,
        highpass_60hz,
        peak_limit_minus_3_dbfs,
        reference_audio=None,
    ):
        return io.NodeOutput(
            *make_voice_profile(
                voice_mode,
                speaker_id,
                voice_description,
                language,
                rights_confirmed,
                reference_audio,
                reference_start_seconds,
                reference_duration_seconds,
                highpass_60hz,
                peak_limit_minus_3_dbfs,
            )
        )


class MiniMaxH3SpeechPlanT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SpeechPlanT8",
            display_name="MiniMax H3 Speech Plan / 语音规划 (EXP/T8)",
            description=(
                "Separates spoken text from acting direction and performs language-aware "
                "chunking. It never guesses the render duration."
            ),
            category=SPEECH_CATEGORY,
            inputs=[
                VoiceProfile.Input("voice_profile"),
                io.String.Input("text", multiline=True, dynamic_prompts=True),
                io.Combo.Input(
                    "language",
                    options=list(SUPPORTED_DIALOGUE_LANGUAGES),
                    default="Chinese",
                ),
                io.String.Input(
                    "acting_direction",
                    default="natural, emotionally connected, with clear diction",
                    multiline=True,
                ),
                io.String.Input("emotion", default="neutral"),
                io.Float.Input(
                    "emotion_intensity",
                    default=0.5,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    tooltip="Prompt strength only; this is not a calibrated acoustic control.",
                ),
                io.Combo.Input(
                    "space",
                    options=list(SPACE_DESCRIPTIONS),
                    default="close",
                ),
                io.Combo.Input(
                    "chunking",
                    options=["single_segment", "language_aware"],
                    default="single_segment",
                ),
                io.Int.Input(
                    "target_units",
                    default=18,
                    min=1,
                    max=200,
                    advanced=True,
                    tooltip="Words for Latin text; characters for CJK text.",
                ),
                io.Int.Input(
                    "max_units",
                    default=24,
                    min=1,
                    max=300,
                    advanced=True,
                ),
            ],
            outputs=[SpeechPlan.Output("speech_plan"), io.String.Output("plan_json")],
            is_experimental=True,
        )

    @classmethod
    def execute(
        cls,
        voice_profile,
        text,
        language,
        acting_direction,
        emotion,
        emotion_intensity,
        space,
        chunking,
        target_units,
        max_units,
    ):
        return io.NodeOutput(
            *make_speech_plan(
                text,
                voice_profile,
                language,
                acting_direction,
                emotion,
                emotion_intensity,
                space,
                chunking,
                target_units,
                max_units,
            )
        )


class MiniMaxH3SpeechConditioningT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SpeechConditioningT8",
            display_name="MiniMax H3 Speech Conditioning / 语音条件 (EXP/T8)",
            description=(
                "Builds native H3 audio-first conditioning without loading a model. "
                "Described voice uses T2VA; reference voice uses Ref2VA with a dark image."
            ),
            category=SPEECH_CATEGORY,
            inputs=[
                io.Clip.Input("clip"),
                io.Vae.Input("video_vae"),
                io.Vae.Input("audio_vae"),
                VoiceProfile.Input("voice_profile"),
                SpeechPlan.Input("speech_plan"),
                io.Int.Input("segment_index", default=0, min=0, max=9999),
                io.Float.Input(
                    "render_seconds",
                    default=10.0,
                    min=5.17,
                    max=15.08,
                    step=0.01,
                    tooltip=(
                        "Explicit H3 render window. It is aligned to 17n+5 frames and is "
                        "not inferred from text length."
                    ),
                ),
                io.Combo.Input("resolution", options=[32, 64, 128], default=32),
                SpeechGuard.Input("speech_guard", optional=True),
            ],
            outputs=[
                io.Conditioning.Output("positive"),
                io.Latent.Output("av_latent"),
                io.String.Output("conditioned_prompt"),
                io.String.Output("spoken_text"),
                io.String.Output("plan_json"),
                io.String.Output("report_json"),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(
        cls,
        clip,
        video_vae,
        audio_vae,
        voice_profile,
        speech_plan,
        segment_index,
        render_seconds,
        resolution,
        speech_guard=None,
    ):
        return io.NodeOutput(
            *build_speech_conditioning(
                clip,
                video_vae,
                audio_vae,
                voice_profile,
                speech_plan,
                segment_index,
                render_seconds,
                resolution,
            )
        )


class MiniMaxH3SpeechDecodeT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SpeechDecodeT8",
            display_name="MiniMax H3 Speech Decode / 仅解码语音 (EXP/T8)",
            description=(
                "Decodes only the H3 audio latent. Conservative energy trim is optional "
                "and does not replace ASR or speaker verification."
            ),
            category=SPEECH_CATEGORY,
            inputs=[
                io.Latent.Input("av_latent"),
                io.Vae.Input("audio_vae"),
                io.Combo.Input(
                    "trim_mode",
                    options=["none", "conservative_energy"],
                    default="none",
                ),
                io.Float.Input(
                    "energy_threshold_dbfs",
                    default=-50.0,
                    min=-80.0,
                    max=-20.0,
                    step=1.0,
                    advanced=True,
                ),
                io.Float.Input(
                    "trim_padding_seconds",
                    default=0.10,
                    min=0.0,
                    max=2.0,
                    step=0.01,
                    advanced=True,
                ),
            ],
            outputs=[io.Audio.Output("audio"), io.String.Output("report_json")],
            is_experimental=True,
        )

    @classmethod
    def execute(
        cls,
        av_latent,
        audio_vae,
        trim_mode,
        energy_threshold_dbfs,
        trim_padding_seconds,
    ):
        return io.NodeOutput(
            *decode_speech_audio(
                av_latent,
                audio_vae,
                trim_mode,
                energy_threshold_dbfs,
                trim_padding_seconds,
            )
        )


class MiniMaxH3SpeechAssembleT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SpeechAssembleT8",
            display_name="MiniMax H3 Speech Assemble / 语音时间线 (EXP/T8)",
            description=(
                "Assembles rendered turns on exact sample boundaries and emits actual "
                "audio-boundary timeline plus planned-text SRT/VTT."
            ),
            category=SPEECH_CATEGORY,
            inputs=[
                SpeechPlan.Input("speech_plan"),
                io.Combo.Input(
                    "output_sample_rate",
                    options=[32000, 44100, 48000],
                    default=32000,
                ),
                io.Float.Input(
                    "crossfade_seconds",
                    default=0.06,
                    min=0.0,
                    max=0.5,
                    step=0.005,
                ),
                io.Float.Input(
                    "peak_limit_dbfs",
                    default=-1.0,
                    min=-12.0,
                    max=0.0,
                    step=0.1,
                ),
                io.Autogrow.Input(
                    "audio_segments",
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input("audio_segment"),
                        prefix="audio_segment_",
                        min=1,
                        max=100,
                    ),
                ),
            ],
            outputs=[
                io.Audio.Output("audio"),
                io.String.Output("timeline_json"),
                io.String.Output("srt"),
                io.String.Output("vtt"),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(
        cls,
        speech_plan,
        output_sample_rate,
        crossfade_seconds,
        peak_limit_dbfs,
        audio_segments=None,
    ):
        return io.NodeOutput(
            *assemble_speech_audio(
                speech_plan,
                audio_segments,
                int(output_sample_rate),
                crossfade_seconds,
                peak_limit_dbfs,
            )
        )


class MiniMaxH3SpeechVerifyT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SpeechVerifyT8",
            display_name="MiniMax H3 Speech Verify & Align / ASR校验裁切 (EXP/T8)",
            description=(
                "Runs optional CPU faster-whisper verification. Exact-target mode removes "
                "Ref2VA lead-in only when the complete requested text is found in order."
            ),
            category=SPEECH_CATEGORY,
            inputs=[
                io.Audio.Input("audio"),
                io.String.Input("expected_text", multiline=True, force_input=True),
                io.Combo.Input(
                    "verify_mode",
                    options=["off", "verify_only", "trim_exact_target"],
                    default="off",
                ),
                io.String.Input(
                    "asr_model_directory",
                    default="",
                    tooltip=(
                        "Absolute faster-whisper CTranslate2 directory, or a folder name "
                        "under ComfyUI/models/TTS. Required only when verification is enabled."
                    ),
                ),
                io.Combo.Input(
                    "language",
                    options=list(ASR_LANGUAGE_CODES),
                    default="auto",
                ),
                io.Float.Input(
                    "min_similarity",
                    default=0.85,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                ),
                io.Int.Input("beam_size", default=5, min=1, max=20, advanced=True),
                io.Int.Input("cpu_threads", default=8, min=1, max=64, advanced=True),
                io.Boolean.Input("unload_after_verify", default=True),
                io.Boolean.Input("strict", default=False, advanced=True),
                io.Float.Input(
                    "pre_padding_seconds",
                    default=0.12,
                    min=0.0,
                    max=2.0,
                    step=0.01,
                    advanced=True,
                ),
                io.Float.Input(
                    "post_padding_seconds",
                    default=0.25,
                    min=0.0,
                    max=2.0,
                    step=0.01,
                    advanced=True,
                ),
                VoiceProfile.Input("voice_profile", optional=True),
                io.Combo.Input(
                    "speaker_check_mode",
                    options=["off", "report_cosine", "require_threshold"],
                    default="off",
                ),
                io.String.Input(
                    "speaker_model_directory",
                    default="",
                    tooltip=(
                        "Local WavLM X-Vector model directory. A voice_profile with a "
                        "reference is also required when speaker checking is enabled."
                    ),
                    advanced=True,
                ),
                io.Float.Input(
                    "min_speaker_similarity",
                    default=0.86,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    advanced=True,
                    tooltip="Dataset-dependent EXP threshold; report_cosine does not gate output.",
                ),
                io.Boolean.Input(
                    "unload_speaker_after_verify",
                    default=True,
                    advanced=True,
                ),
                io.Float.Input(
                    "peak_limit_dbfs",
                    default=-1.0,
                    min=-24.0,
                    max=0.0,
                    step=0.1,
                    tooltip="Final attenuation-only peak protection; never boosts quiet speech.",
                ),
            ],
            outputs=[
                io.Audio.Output("audio"),
                io.String.Output("transcript"),
                io.Float.Output("text_similarity"),
                io.Float.Output("speaker_similarity"),
                io.Boolean.Output("accepted"),
                io.String.Output("report_json"),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(
        cls,
        audio,
        expected_text,
        verify_mode,
        asr_model_directory,
        language,
        min_similarity,
        beam_size,
        cpu_threads,
        unload_after_verify,
        strict,
        pre_padding_seconds,
        post_padding_seconds,
        voice_profile=None,
        speaker_check_mode="off",
        speaker_model_directory="",
        min_speaker_similarity=0.86,
        unload_speaker_after_verify=True,
        peak_limit_dbfs=-1.0,
    ):
        reference_audio = None
        if isinstance(voice_profile, dict):
            reference_audio = voice_profile.get("reference_audio")
        return io.NodeOutput(
            *verify_speech_audio(
                audio,
                expected_text,
                verify_mode,
                asr_model_directory,
                language,
                min_similarity,
                beam_size,
                cpu_threads,
                unload_after_verify,
                strict,
                pre_padding_seconds,
                post_padding_seconds,
                reference_audio,
                speaker_check_mode,
                speaker_model_directory,
                min_speaker_similarity,
                unload_speaker_after_verify,
                peak_limit_dbfs,
            )
        )


class MiniMaxH3DialogueScriptT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3DialogueScriptT8",
            display_name="MiniMax H3 Dialogue Script / 对话脚本 (EXP/T8)",
            description=(
                "Builds a two- or three-speaker, one-turn-at-a-time dialogue plan. "
                "Use 'S1: text' lines or structured JSON for per-turn acting and timing."
            ),
            category=SPEECH_CATEGORY,
            inputs=[
                io.String.Input("script", multiline=True, dynamic_prompts=True),
                io.Combo.Input(
                    "script_format",
                    options=["auto", "speaker_lines", "json"],
                    default="auto",
                ),
                io.Combo.Input(
                    "default_language",
                    options=list(SUPPORTED_DIALOGUE_LANGUAGES),
                    default="Chinese",
                ),
                io.Combo.Input(
                    "default_space",
                    options=list(SPACE_DESCRIPTIONS),
                    default="close",
                ),
                io.Autogrow.Input(
                    "voice_profiles",
                    template=io.Autogrow.TemplatePrefix(
                        input=VoiceProfile.Input("voice_profile"),
                        prefix="voice_profile_",
                        min=2,
                        max=3,
                    ),
                ),
            ],
            outputs=[SpeechPlan.Output("dialogue_plan"), io.String.Output("plan_json")],
            is_experimental=True,
        )

    @classmethod
    def execute(
        cls,
        script,
        script_format,
        default_language,
        default_space,
        voice_profiles=None,
    ):
        return io.NodeOutput(
            *make_dialogue_plan(
                script,
                script_format,
                voice_profiles,
                default_language,
                default_space,
            )
        )


class MiniMaxH3DialogueTurnSelectT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3DialogueTurnSelectT8",
            display_name="MiniMax H3 Dialogue Turn Select / 选择对白句 (EXP/T8)",
            description=(
                "Selects one dialogue turn and its bound voice profile for independent "
                "generation, retry, and replacement."
            ),
            category=SPEECH_CATEGORY,
            inputs=[
                SpeechPlan.Input("dialogue_plan"),
                io.Int.Input("turn_index", default=0, min=0, max=9999),
            ],
            outputs=[
                VoiceProfile.Output("voice_profile"),
                SpeechPlan.Output("speech_plan"),
                io.String.Output("spoken_text"),
                io.String.Output("report_json"),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, dialogue_plan, turn_index):
        return io.NodeOutput(*select_dialogue_turn(dialogue_plan, turn_index))


def request_speech_release(release_policy: str) -> dict:
    if release_policy not in RELEASE_POLICIES:
        raise ValueError(f"unknown release policy: {release_policy}")
    if release_policy != "keep_loaded":
        ComfyQueueRuntime().request_release(release_policy)
    return {
        "release_policy": release_policy,
        "scope": (
            "none"
            if release_policy == "keep_loaded"
            else (
                "global_comfyui_models"
                if release_policy == "unload_all_models"
                else "execution_cache_and_soft_memory"
            )
        ),
        "requested": release_policy != "keep_loaded",
    }


class MiniMaxH3SpeechGuardT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SpeechGuardT8",
            display_name="MiniMax H3 Speech Abnormal-Exit Guard / 异常释放保护 (EXP/T8)",
            description=(
                "Arms a prompt-lifecycle guard before speech conditioning. If OOM, cancellation, "
                "or another upstream exception prevents Finalize, the selected error release is requested."
            ),
            category=SPEECH_CATEGORY,
            inputs=[
                io.Combo.Input(
                    "error_release_policy",
                    options=list(RELEASE_POLICIES),
                    default="unload_all_models",
                    tooltip=(
                        "unload_all_models is global. Current ComfyUI already does this for recognized "
                        "CUDA OOM; the guard closes non-OOM/cancel gaps."
                    ),
                ),
            ],
            outputs=[SpeechGuard.Output("speech_guard"), io.String.Output("report_json")],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, error_release_policy):
        guard = arm_speech_guard(error_release_policy)
        report = {
            "schema": guard["schema"],
            "armed": True,
            "prompt_id": guard["prompt_id"],
            "error_release_policy": error_release_policy,
            "scope": (
                "global_comfyui_models"
                if error_release_policy == "unload_all_models"
                else (
                    "execution_cache_and_soft_memory"
                    if error_release_policy == "clear_execution_cache"
                    else "none"
                )
            ),
            "recognized_oom_note": "Current ComfyUI also unloads all models immediately for recognized CUDA OOM.",
        }
        return io.NodeOutput(guard, json.dumps(report, ensure_ascii=False, indent=2))


class MiniMaxH3SpeechVRAMPreflightT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SpeechVRAMPreflightT8",
            display_name="MiniMax H3 Speech VRAM Preflight / 显存预检 (EXP/T8)",
            description=(
                "Reports present whole-device free VRAM and DynamicVRAM configuration. "
                "It cannot predict sampler peak and never grants a memory_safe label."
            ),
            category=SPEECH_CATEGORY,
            inputs=[
                io.Model.Input("model"),
                io.Float.Input(
                    "minimum_headroom_mib",
                    default=512.0,
                    min=0.0,
                    max=16384.0,
                    step=16.0,
                ),
                io.Boolean.Input("block_when_currently_below_gate", default=False),
            ],
            outputs=[
                io.Model.Output("model"),
                io.Boolean.Output("current_gate_pass"),
                io.String.Output("report_json"),
            ],
            is_output_node=True,
            is_experimental=True,
        )

    @classmethod
    def execute(cls, model, minimum_headroom_mib, block_when_currently_below_gate):
        report = vram_preflight(minimum_headroom_mib)
        passed = bool(report.get("current_gate_pass"))
        if block_when_currently_below_gate and not passed:
            raise RuntimeError(
                "Speech VRAM preflight is below the explicit current-headroom gate: "
                + json.dumps(report, ensure_ascii=False)
            )
        return io.NodeOutput(model, passed, json.dumps(report, ensure_ascii=False, indent=2))


class MiniMaxH3SpeechFinalizeT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SpeechFinalizeT8",
            display_name="MiniMax H3 Speech Finalize & Release / 完成并释放 (EXP/T8)",
            description=(
                "Passes AUDIO through, then requests the selected ComfyUI release policy. "
                "unload_all_models is global and is never described as H3-only."
            ),
            category=SPEECH_CATEGORY,
            inputs=[
                io.Audio.Input("audio"),
                io.Combo.Input(
                    "release_policy",
                    options=list(RELEASE_POLICIES),
                    default="clear_execution_cache",
                ),
                io.String.Input("upstream_report", optional=True, force_input=True),
                SpeechGuard.Input("speech_guard", optional=True),
            ],
            outputs=[io.Audio.Output("audio"), io.String.Output("report_json")],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, audio, release_policy, upstream_report=None, speech_guard=None):
        release = request_speech_release(release_policy)
        guard_completion = complete_speech_guard(speech_guard)
        report = {
            "schema": "minimax_h3_t8_speech_finalize_v2",
            "release": release,
            "abnormal_exit_guard": guard_completion,
        }
        if upstream_report:
            try:
                report["upstream"] = json.loads(upstream_report)
            except (TypeError, json.JSONDecodeError):
                report["upstream"] = str(upstream_report)
        return io.NodeOutput(audio, json.dumps(report, ensure_ascii=False, indent=2))


class MiniMaxH3SpeechStudioT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SpeechStudioT8",
            display_name="MiniMax H3 Speech Studio / 一站式语音 (EXP/T8)",
            description=(
                "Native ComfyUI speech graph: conditioning -> sampler -> audio-only decode -> "
                "explicit release. Stock res_multistep/simple is the initial quality baseline."
            ),
            category=SPEECH_CATEGORY,
            enable_expand=True,
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input("clip"),
                io.Vae.Input("video_vae"),
                io.Vae.Input("audio_vae"),
                VoiceProfile.Input("voice_profile"),
                SpeechPlan.Input("speech_plan"),
                io.Int.Input("segment_index", default=0, min=0, max=9999),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF),
                io.Float.Input("render_seconds", default=10.0, min=5.17, max=15.08, step=0.01),
                io.Combo.Input("resolution", options=[32, 64, 128], default=32),
                SpeechGuard.Input("speech_guard", optional=True),
                io.Int.Input("steps", default=20, min=1, max=1000),
                io.Combo.Input(
                    "sampler_name",
                    options=SPEECH_SAMPLERS,
                    default=(
                        "res_multistep"
                        if "res_multistep" in SPEECH_SAMPLERS
                        else DEFAULT_SAMPLER_NAME
                    ),
                ),
                io.Combo.Input(
                    "scheduler",
                    options=SPEECH_SCHEDULERS,
                    default=(
                        "simple"
                        if "simple" in SPEECH_SCHEDULERS
                        else DEFAULT_SCHEDULER_NAME
                    ),
                ),
                io.Float.Input("shift_video", default=12.0, min=0.01, max=100.0, step=0.01, advanced=True),
                io.Float.Input("shift_audio", default=3.0, min=0.01, max=100.0, step=0.01, advanced=True),
                io.Combo.Input(
                    "trim_mode",
                    options=["none", "conservative_energy"],
                    default="none",
                ),
                io.Combo.Input(
                    "verify_mode",
                    options=["off", "verify_only", "trim_exact_target"],
                    default="off",
                ),
                io.String.Input(
                    "asr_model_directory",
                    default="",
                    tooltip=(
                        "Optional CPU faster-whisper CTranslate2 model directory. "
                        "Required only when verify_mode is not off."
                    ),
                    advanced=True,
                ),
                io.Combo.Input(
                    "asr_language",
                    options=list(ASR_LANGUAGE_CODES),
                    default="auto",
                    advanced=True,
                ),
                io.Float.Input(
                    "min_similarity",
                    default=0.85,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    advanced=True,
                ),
                io.Boolean.Input("unload_asr_after_verify", default=True, advanced=True),
                io.Combo.Input(
                    "speaker_check_mode",
                    options=["off", "report_cosine", "require_threshold"],
                    default="off",
                    advanced=True,
                ),
                io.String.Input(
                    "speaker_model_directory",
                    default="",
                    advanced=True,
                ),
                io.Float.Input(
                    "min_speaker_similarity",
                    default=0.86,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    advanced=True,
                ),
                io.Boolean.Input(
                    "unload_speaker_after_verify",
                    default=True,
                    advanced=True,
                ),
                io.Float.Input(
                    "peak_limit_dbfs",
                    default=-1.0,
                    min=-24.0,
                    max=0.0,
                    step=0.1,
                    advanced=True,
                ),
                io.Combo.Input(
                    "release_policy",
                    options=list(RELEASE_POLICIES),
                    default="clear_execution_cache",
                ),
            ],
            outputs=[
                io.Audio.Output("audio"),
                io.String.Output("conditioned_prompt"),
                io.String.Output("report_json"),
                io.Latent.Output("generated_av_latent"),
                io.String.Output("transcript"),
                io.Float.Output("text_similarity"),
                io.Float.Output("speaker_similarity"),
                io.Boolean.Output("accepted"),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(
        cls,
        model,
        clip,
        video_vae,
        audio_vae,
        voice_profile,
        speech_plan,
        segment_index,
        seed,
        render_seconds,
        resolution,
        steps,
        sampler_name,
        scheduler,
        shift_video,
        shift_audio,
        trim_mode,
        verify_mode="off",
        asr_model_directory="",
        asr_language="auto",
        min_similarity=0.85,
        unload_asr_after_verify=True,
        speaker_check_mode="off",
        speaker_model_directory="",
        min_speaker_similarity=0.86,
        unload_speaker_after_verify=True,
        peak_limit_dbfs=-1.0,
        release_policy="clear_execution_cache",
    ):
        graph = GraphBuilder()
        error_release_policy = (
            "keep_loaded" if release_policy == "keep_loaded" else "unload_all_models"
        )
        guard = graph.node(
            "MiniMaxH3SpeechGuardT8",
            id="speech_abnormal_exit_guard",
            error_release_policy=error_release_policy,
        )
        conditioning = graph.node(
            "MiniMaxH3SpeechConditioningT8",
            id="speech_conditioning",
            clip=clip,
            video_vae=video_vae,
            audio_vae=audio_vae,
            voice_profile=voice_profile,
            speech_plan=speech_plan,
            segment_index=segment_index,
            render_seconds=render_seconds,
            resolution=resolution,
            speech_guard=guard.out(0),
        )
        sampling = graph.node(
            "MiniMaxH3DualClockSamplerT8",
            id="speech_sampling_setup",
            model=model,
            av_latent=conditioning.out(1),
            steps=steps,
            shift_video=shift_video,
            shift_audio=shift_audio,
            sampler_name=sampler_name,
            scheduler=scheduler,
        )
        guider = graph.node(
            "BasicGuider",
            id="speech_guider",
            model=sampling.out(0),
            conditioning=conditioning.out(0),
        )
        noise = graph.node("RandomNoise", id="speech_noise", noise_seed=seed)
        sampler = graph.node(
            "SamplerCustomAdvanced",
            id="speech_sampler",
            noise=noise.out(0),
            guider=guider.out(0),
            sampler=sampling.out(1),
            sigmas=sampling.out(2),
            latent_image=conditioning.out(1),
        )
        decoded = graph.node(
            "MiniMaxH3SpeechDecodeT8",
            id="speech_decode",
            av_latent=sampler.out(0),
            audio_vae=audio_vae,
            trim_mode=trim_mode,
            energy_threshold_dbfs=-50.0,
            trim_padding_seconds=0.10,
        )
        verified = graph.node(
            "MiniMaxH3SpeechVerifyT8",
            id="speech_verify",
            audio=decoded.out(0),
            expected_text=conditioning.out(3),
            verify_mode=verify_mode,
            asr_model_directory=asr_model_directory,
            language=asr_language,
            min_similarity=min_similarity,
            beam_size=5,
            cpu_threads=8,
            unload_after_verify=unload_asr_after_verify,
            strict=False,
            pre_padding_seconds=0.12,
            post_padding_seconds=0.25,
            voice_profile=voice_profile,
            speaker_check_mode=speaker_check_mode,
            speaker_model_directory=speaker_model_directory,
            min_speaker_similarity=min_speaker_similarity,
            unload_speaker_after_verify=unload_speaker_after_verify,
            peak_limit_dbfs=peak_limit_dbfs,
        )
        finalized = graph.node(
            "MiniMaxH3SpeechFinalizeT8",
            id="speech_finalize",
            audio=verified.out(0),
            release_policy=release_policy,
            upstream_report=verified.out(5),
            speech_guard=guard.out(0),
        )
        return io.NodeOutput(
            finalized.out(0),
            conditioning.out(2),
            finalized.out(1),
            sampler.out(0),
            verified.out(1),
            verified.out(2),
            verified.out(3),
            verified.out(4),
            expand=graph.finalize(),
        )


class MiniMaxH3VoiceLibrarySaveT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3VoiceLibrarySaveT8",
            display_name="MiniMax H3 Voice Library Save / 保存音色 (EXP/T8)",
            description="Explicitly persists a voice profile and any reference audio in the local ComfyUI user directory.",
            category=SPEECH_CATEGORY,
            inputs=[
                VoiceProfile.Input("voice_profile"),
                io.String.Input("library_name", default="my_voice"),
                io.Boolean.Input("replace_existing", default=False),
            ],
            outputs=[VoiceProfile.Output("voice_profile"), io.String.Output("report_json")],
            is_output_node=True,
            is_experimental=True,
        )

    @classmethod
    def execute(cls, voice_profile, library_name, replace_existing):
        return io.NodeOutput(*save_voice_profile(voice_profile, library_name, replace_existing))


class MiniMaxH3VoiceLibraryLoadT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3VoiceLibraryLoadT8",
            display_name="MiniMax H3 Voice Library Load / 加载音色 (EXP/T8)",
            category=SPEECH_CATEGORY,
            inputs=[io.String.Input("library_name", default="my_voice")],
            outputs=[VoiceProfile.Output("voice_profile"), io.String.Output("report_json")],
            is_output_node=True,
            is_experimental=True,
        )

    @classmethod
    def execute(cls, library_name):
        return io.NodeOutput(*load_voice_profile(library_name))


class MiniMaxH3VoiceLibraryDeleteT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3VoiceLibraryDeleteT8",
            display_name="MiniMax H3 Voice Library Delete / 删除音色 (EXP/T8)",
            description="Moves an entry to a recoverable local trash folder; it does not permanently erase it.",
            category=SPEECH_CATEGORY,
            inputs=[
                io.String.Input("library_name", default="my_voice"),
                io.Boolean.Input("confirm_delete", default=False),
            ],
            outputs=[io.String.Output("report_json")],
            is_output_node=True,
            is_experimental=True,
        )

    @classmethod
    def execute(cls, library_name, confirm_delete):
        return io.NodeOutput(delete_voice_profile(library_name, confirm_delete))


class MiniMaxH3SpeechPerformanceT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SpeechPerformanceT8",
            display_name="MiniMax H3 Speech Performance Direction / 演绎控制 (EXP/T8)",
            description=(
                "Adds emotion, pace, pitch, energy and nonverbal prompt direction. "
                "These model controls remain uncalibrated and are reported as such."
            ),
            category=SPEECH_CATEGORY,
            inputs=[
                SpeechPlan.Input("speech_plan"),
                io.Int.Input("segment_index", default=-1, min=-1, max=9999, tooltip="-1 applies to all segments."),
                io.String.Input("emotion", default="neutral"),
                io.Float.Input("prompt_intensity", default=0.5, min=0.0, max=1.0, step=0.05),
                io.Combo.Input("pace", options=["very_slow", "slow", "natural", "fast", "very_fast"], default="natural"),
                io.Combo.Input("pitch", options=["very_low", "low", "natural", "high", "very_high"], default="natural"),
                io.Combo.Input("energy", options=["restrained", "low", "natural", "high", "intense"], default="natural"),
                io.String.Input("nonverbal_direction", default="", multiline=True),
            ],
            outputs=[SpeechPlan.Output("speech_plan"), io.String.Output("report_json")],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, speech_plan, segment_index, emotion, prompt_intensity, pace, pitch, energy, nonverbal_direction):
        return io.NodeOutput(
            *apply_performance_direction(
                speech_plan,
                segment_index,
                emotion,
                prompt_intensity,
                pace,
                pitch,
                energy,
                nonverbal_direction,
            )
        )


class MiniMaxH3SpeechADRFitT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SpeechADRFitT8",
            display_name="MiniMax H3 Speech ADR Exact Fit / 配音精确时长 (EXP/T8)",
            description=(
                "Fits AUDIO to an exact sample count using refusal, pad/trim, or bounded phase-vocoder stretch. "
                "Exact duration is not a lip-sync claim."
            ),
            category=SPEECH_CATEGORY,
            inputs=[
                io.Audio.Input("audio"),
                io.Float.Input("target_duration_seconds", default=5.0, min=0.001, max=36000.0, step=0.001),
                io.Combo.Input(
                    "fit_mode",
                    options=["refuse_if_mismatch", "pad_or_trim", "safe_time_stretch"],
                    default="safe_time_stretch",
                ),
                io.Float.Input("minimum_rate", default=0.90, min=0.10, max=4.0, step=0.01, advanced=True),
                io.Float.Input("maximum_rate", default=1.10, min=0.10, max=4.0, step=0.01, advanced=True),
                io.Float.Input("pitch_semitones", default=0.0, min=-12.0, max=12.0, step=0.1, advanced=True),
            ],
            outputs=[io.Audio.Output("audio"), io.String.Output("report_json")],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, audio, target_duration_seconds, fit_mode, minimum_rate, maximum_rate, pitch_semitones):
        return io.NodeOutput(
            *fit_audio_for_adr(
                audio,
                target_duration_seconds,
                fit_mode,
                minimum_rate,
                maximum_rate,
                pitch_semitones,
            )
        )


class MiniMaxH3SpeechLongFormStartT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SpeechLongFormStartT8",
            display_name="MiniMax H3 Speech Long Form Start/Resume / 长文本恢复 (EXP/T8)",
            description=(
                "Creates or resumes an atomic accepted-segment manifest. Render only next_index per prompt; "
                "this avoids retaining every long-form latent in one execution."
            ),
            category=SPEECH_CATEGORY,
            inputs=[
                SpeechPlan.Input("speech_plan"),
                io.String.Input("job_id", default="speech_job_001"),
            ],
            outputs=[
                SpeechSession.Output("session"),
                VoiceProfile.Output("next_voice_profile"),
                io.Int.Output("next_index"),
                io.String.Output("next_text"),
                io.Boolean.Output("complete"),
                io.String.Output("report_json"),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, speech_plan, job_id):
        session, profile, next_index, text, report = start_or_resume_longform(speech_plan, job_id)
        return io.NodeOutput(session, profile, next_index, text, next_index < 0 and session["state"] == "complete", report)

    @classmethod
    def fingerprint_inputs(cls, speech_plan, job_id):
        try:
            plan_digest = hashlib.sha256(
                json.dumps(public_plan(validate_speech_plan(speech_plan)), ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest()
            return f"{plan_digest}:{speech_manifest_fingerprint(job_id)}"
        except (TypeError, ValueError):
            return f"unresolved:{job_id!r}"


class MiniMaxH3SpeechLongFormAcceptT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SpeechLongFormAcceptT8",
            display_name="MiniMax H3 Speech Long Form Accept / 接受语音分段 (EXP/T8)",
            description="Atomically stores one accepted chunk before advancing the manifest; safe to resume after a crash.",
            category=SPEECH_CATEGORY,
            inputs=[
                SpeechSession.Input("session"),
                SpeechPlan.Input("speech_plan"),
                io.Int.Input("segment_index", default=0, min=0, max=99999),
                io.Audio.Input("audio"),
                io.String.Input("transcript", default="", force_input=True),
                io.Float.Input("text_similarity", default=0.0, min=0.0, max=1.0, force_input=True),
                io.Float.Input("speaker_similarity", default=0.0, min=-1.0, max=1.0, force_input=True),
                io.Boolean.Input("accepted", default=False, force_input=True),
                io.Boolean.Input("replace_existing", default=False, advanced=True),
            ],
            outputs=[io.Audio.Output("chunk_ready_audio"), io.String.Output("report_json")],
            is_output_node=True,
            is_experimental=True,
        )

    @classmethod
    def execute(
        cls,
        session,
        speech_plan,
        segment_index,
        audio,
        transcript,
        text_similarity,
        speaker_similarity,
        accepted,
        replace_existing,
    ):
        output_audio, report_json = accept_longform_segment(
                session,
                speech_plan,
                segment_index,
                audio,
                transcript,
                text_similarity,
                speaker_similarity,
                accepted,
                replace_existing,
            )
        report = json.loads(report_json)
        preview_path = Path(report["chunk_ready_preview_path"]).resolve()
        output_root = Path(folder_paths.get_output_directory()).resolve()
        relative = preview_path.relative_to(output_root)
        return io.NodeOutput(
            output_audio,
            report_json,
            ui={
                "audio": [
                    {
                        "filename": relative.name,
                        "subfolder": relative.parent.as_posix(),
                        "type": "output",
                    }
                ]
            },
        )


class MiniMaxH3SpeechLongFormControlT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SpeechLongFormControlT8",
            display_name="MiniMax H3 Speech Long Form Control / 长文本控制 (EXP/T8)",
            category=SPEECH_CATEGORY,
            inputs=[
                io.String.Input("job_id", default="speech_job_001"),
                io.Combo.Input(
                    "action",
                    options=["status", "request_cancel", "clear_cancel", "reset_to_trash"],
                    default="status",
                ),
                io.Boolean.Input("confirm_reset", default=False),
            ],
            outputs=[io.String.Output("report_json")],
            is_output_node=True,
            is_experimental=True,
        )

    @classmethod
    def execute(cls, job_id, action, confirm_reset):
        return io.NodeOutput(control_longform_session(job_id, action, confirm_reset))


class MiniMaxH3SpeechLongFormComposeT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SpeechLongFormComposeT8",
            display_name="MiniMax H3 Speech Long Form Compose / 合成长文本 (EXP/T8)",
            category=SPEECH_CATEGORY,
            inputs=[
                SpeechPlan.Input("speech_plan"),
                io.String.Input("job_id", default="speech_job_001"),
                io.Float.Input("crossfade_seconds", default=0.06, min=0.0, max=0.5, step=0.005),
                io.Float.Input("peak_limit_dbfs", default=-1.0, min=-12.0, max=0.0, step=0.1),
            ],
            outputs=[
                io.Audio.Output("audio"),
                io.String.Output("timeline_json"),
                io.String.Output("srt"),
                io.String.Output("vtt"),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, speech_plan, job_id, crossfade_seconds, peak_limit_dbfs):
        return io.NodeOutput(
            *compose_longform_session(
                speech_plan,
                job_id,
                crossfade_seconds,
                peak_limit_dbfs,
            )
        )


class MiniMaxH3JointDialogueConditioningT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3JointDialogueConditioningT8",
            display_name="MiniMax H3 Joint Dialogue Conditioning / 多人同段条件 (EXP/T8)",
            description=(
                "Builds an unverified 2-3 speaker joint Ref2VA experiment. It is not the stable dialogue path "
                "and must not be described as identity-safe until leakage and identity-swap tests pass."
            ),
            category=SPEECH_CATEGORY,
            inputs=[
                io.Clip.Input("clip"),
                io.Vae.Input("video_vae"),
                io.Vae.Input("audio_vae"),
                SpeechPlan.Input("dialogue_plan"),
                io.Int.Input("start_turn", default=0, min=0, max=9999),
                io.Int.Input("turn_count", default=2, min=2, max=3),
                io.Float.Input("render_seconds", default=10.0, min=5.17, max=15.08, step=0.01),
                io.Combo.Input("resolution", options=[32, 64, 128], default=32),
                SpeechGuard.Input("speech_guard", optional=True),
            ],
            outputs=[
                io.Conditioning.Output("positive"),
                io.Latent.Output("av_latent"),
                io.String.Output("conditioned_prompt"),
                io.String.Output("report_json"),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(
        cls,
        clip,
        video_vae,
        audio_vae,
        dialogue_plan,
        start_turn,
        turn_count,
        render_seconds,
        resolution,
        speech_guard=None,
    ):
        del speech_guard  # Graph dependency only: arm abnormal-exit cleanup before conditioning.
        return io.NodeOutput(
            *build_joint_dialogue_conditioning(
                clip,
                video_vae,
                audio_vae,
                dialogue_plan,
                start_turn,
                turn_count,
                render_seconds,
                resolution,
            )
        )


SPEECH_NODE_CLASSES = [
    MiniMaxH3VoiceProfileT8,
    MiniMaxH3SpeechPlanT8,
    MiniMaxH3SpeechConditioningT8,
    MiniMaxH3SpeechDecodeT8,
    MiniMaxH3SpeechVerifyT8,
    MiniMaxH3SpeechAssembleT8,
    MiniMaxH3DialogueScriptT8,
    MiniMaxH3DialogueTurnSelectT8,
    MiniMaxH3SpeechFinalizeT8,
    MiniMaxH3SpeechStudioT8,
    MiniMaxH3SpeechGuardT8,
    MiniMaxH3SpeechVRAMPreflightT8,
    MiniMaxH3VoiceLibrarySaveT8,
    MiniMaxH3VoiceLibraryLoadT8,
    MiniMaxH3VoiceLibraryDeleteT8,
    MiniMaxH3SpeechPerformanceT8,
    MiniMaxH3SpeechADRFitT8,
    MiniMaxH3SpeechLongFormStartT8,
    MiniMaxH3SpeechLongFormAcceptT8,
    MiniMaxH3SpeechLongFormControlT8,
    MiniMaxH3SpeechLongFormComposeT8,
    MiniMaxH3JointDialogueConditioningT8,
]
