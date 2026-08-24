from __future__ import annotations

from comfy_api.latest import io

from .audio_integrity_advanced import (
    analyze_audio_integrity,
    analyze_audio_perceptual_drift,
    audit_speaker_routing,
)
from .speech import SPEECH_PLAN_TYPE


AudioPlan = io.Custom(SPEECH_PLAN_TYPE)
CATEGORY = "T8/MiniMax H3/Audio/Advanced"


class MiniMaxH3AudioIntegrityAuditT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3AudioIntegrityAuditT8Advanced",
            display_name="MiniMax H3 Audio Integrity Audit / 音频完整性审计 (Advanced/T8)",
            description=(
                "CPU-only, report-only audit for opening discontinuities, tail/head similarity, "
                "DC jumps, clipping, exact samples and optional video-boundary mismatch. It "
                "never repairs or modifies the audio."
            ),
            category=CATEGORY,
            is_experimental=True,
            is_output_node=True,
            inputs=[
                io.Audio.Input("audio"),
                io.Int.Input(
                    "video_frame_count",
                    default=0,
                    min=0,
                    max=1000000,
                    tooltip="0 skips A/V boundary comparison; otherwise use the decoded frame count.",
                ),
                io.Float.Input("fps", default=24.0, min=0.001, max=1000.0, step=0.001),
                io.Float.Input(
                    "opening_window_ms", default=40.0, min=1.0, max=1000.0, step=1.0, advanced=True
                ),
                io.Float.Input(
                    "comparison_window_ms", default=250.0, min=25.0, max=5000.0, step=5.0, advanced=True
                ),
                io.Float.Input(
                    "pop_jump_threshold", default=0.15, min=0.001, max=2.0, step=0.001, advanced=True
                ),
                io.Float.Input(
                    "dc_jump_threshold", default=0.02, min=0.0001, max=1.0, step=0.0001, advanced=True
                ),
                io.Float.Input(
                    "wrap_correlation_threshold", default=0.985, min=0.0, max=1.0, step=0.001, advanced=True
                ),
                io.Float.Input(
                    "clipping_ratio_threshold", default=0.001, min=0.0, max=1.0, step=0.0001, advanced=True
                ),
                io.Float.Input(
                    "max_av_delta_ms", default=21.0, min=0.0, max=10000.0, step=0.1, advanced=True
                ),
            ],
            outputs=[
                io.Audio.Output("audio"),
                io.Boolean.Output("pass_audit"),
                io.String.Output("decision"),
                io.Int.Output("exact_samples"),
                io.Float.Output("duration_seconds"),
                io.Float.Output("av_delta_ms"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*analyze_audio_integrity(**kwargs))


class MiniMaxH3SpeakerRoutingAuditT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SpeakerRoutingAuditT8Advanced",
            display_name="MiniMax H3 Speaker Routing Audit / 多人对白路由预检 (Advanced/T8)",
            description=(
                "Compiles dialogue speakers to deterministic <Audio N> ordinals and abstains "
                "on missing/duplicate references, unstructured vocalizations or ambiguous "
                "same-gender descriptors. It never changes the dialogue plan."
            ),
            category=CATEGORY,
            is_experimental=True,
            is_output_node=True,
            inputs=[
                AudioPlan.Input("dialogue_plan"),
                io.Boolean.Input("require_reference_voice", default=True),
                io.Float.Input(
                    "descriptor_similarity_threshold",
                    default=0.75,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    advanced=True,
                ),
            ],
            outputs=[
                AudioPlan.Output("dialogue_plan"),
                io.Boolean.Output("pass_audit"),
                io.String.Output("decision"),
                io.String.Output("binding_json"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, dialogue_plan, require_reference_voice, descriptor_similarity_threshold):
        return io.NodeOutput(
            *audit_speaker_routing(
                dialogue_plan,
                require_reference_voice,
                descriptor_similarity_threshold,
            )
        )


class MiniMaxH3AudioPerceptualDriftAuditT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3AudioPerceptualDriftAuditT8Advanced",
            display_name=(
                "MiniMax H3 Audio Perceptual Drift Audit / 音色远近漂移审计 "
                "(Advanced/T8)"
            ),
            description=(
                "CPU-only, report-only comparison of a synchronized candidate against a "
                "reference. It marks persistent spectral-envelope and level drift for human "
                "review; it does not diagnose distance, reverb or speaker identity."
            ),
            category=CATEGORY,
            is_experimental=True,
            is_output_node=True,
            inputs=[
                io.Audio.Input("reference_audio"),
                io.Audio.Input("candidate_audio"),
                io.Float.Input(
                    "analysis_window_ms", default=500.0, min=100.0, max=2000.0, step=10.0
                ),
                io.Float.Input("hop_ms", default=100.0, min=20.0, max=1000.0, step=10.0),
                io.Float.Input(
                    "active_rms_floor_dbfs", default=-50.0, min=-100.0, max=0.0, step=1.0
                ),
                io.Float.Input(
                    "spectral_drift_threshold",
                    default=0.30,
                    min=0.01,
                    max=4.0,
                    step=0.01,
                    advanced=True,
                ),
                io.Float.Input(
                    "level_delta_threshold_db",
                    default=4.0,
                    min=0.1,
                    max=40.0,
                    step=0.1,
                    advanced=True,
                ),
                io.Int.Input(
                    "persistent_window_count",
                    default=3,
                    min=1,
                    max=100,
                    step=1,
                    advanced=True,
                ),
                io.Float.Input(
                    "max_duration_delta_ms",
                    default=21.0,
                    min=0.0,
                    max=10000.0,
                    step=0.1,
                    advanced=True,
                ),
            ],
            outputs=[
                io.Audio.Output("candidate_audio"),
                io.Boolean.Output("pass_audit"),
                io.String.Output("decision"),
                io.Float.Output("maximum_spectral_drift"),
                io.Float.Output("maximum_level_delta_db"),
                io.Float.Output("suspected_start_seconds"),
                io.Float.Output("suspected_end_seconds"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*analyze_audio_perceptual_drift(**kwargs))


AUDIO_INTEGRITY_ADVANCED_NODE_CLASSES = [
    MiniMaxH3AudioIntegrityAuditT8Advanced,
    MiniMaxH3SpeakerRoutingAuditT8Advanced,
]

AUDIO_PERCEPTUAL_DRIFT_ADVANCED_NODE_CLASSES = [
    MiniMaxH3AudioPerceptualDriftAuditT8Advanced,
]
