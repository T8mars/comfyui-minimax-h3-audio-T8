import json

import comfy.model_management
from comfy_api.latest import io

from .final_audio_audit import audit_final_audio
from .speech_verification import ASR_LANGUAGE_CODES


class MiniMaxH3FinalAudioAuditEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3FinalAudioAuditEXPT8",
            display_name="H3 Final Audio Audit / 最终保留音轨核对 (EXP/T8)",
            category="T8/MiniMax H3/Audio/Experimental",
            is_experimental=True,
            description="Offline CPU ASR on the final retained PCM. Returns the same AUDIO object: no peak limiter, "
            "trim, retime or automatic retry. Select an existing local faster-whisper directory; "
            "expected windows use final playback time, not discarded generation overlaps.",
            inputs=[
                io.Audio.Input("final_audio"),
                io.String.Input("expected_text", multiline=True, default=""),
                io.String.Input("asr_model_directory", default=""),
                io.Combo.Input("language", options=list(ASR_LANGUAGE_CODES)),
                io.String.Input(
                    "windows_json", multiline=True, default="[]", advanced=True
                ),
                io.Int.Input("beam_size", default=5, min=1, max=100, advanced=True),
                io.Int.Input("cpu_threads", default=2, min=1, max=128, advanced=True),
                io.Boolean.Input("unload_after", default=True, advanced=True),
                io.Combo.Input(
                    "transcription_scope",
                    options=["whole_final_audio", "explicit_final_windows"],
                    default="whole_final_audio",
                    advanced=True,
                ),
            ],
            outputs=[io.Audio.Output("audio"), io.String.Output("report_json")],
        )

    @classmethod
    def execute(
        cls,
        final_audio,
        expected_text,
        asr_model_directory,
        language="auto",
        windows_json="[]",
        beam_size=5,
        cpu_threads=2,
        unload_after=True,
        transcription_scope="whole_final_audio",
    ):
        return io.NodeOutput(
            *audit_final_audio(
                final_audio,
                expected_text,
                asr_model_directory,
                language,
                json.loads(windows_json),
                beam_size,
                cpu_threads,
                unload_after,
                comfy.model_management.throw_exception_if_processing_interrupted,
                transcription_scope=transcription_scope,
            )
        )


NODES = [MiniMaxH3FinalAudioAuditEXPT8]
