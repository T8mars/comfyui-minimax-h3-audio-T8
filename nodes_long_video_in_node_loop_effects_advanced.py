from __future__ import annotations

from pathlib import Path

import folder_paths
from comfy_api.latest import InputImpl, io, ui

from .long_video_in_node_loop_effects_advanced import (
    PROMPT_RELAY_MODES,
    run_long_video_in_node_loop_effects,
)
from .nodes_long_video_sampling_plan_advanced import LongVideoSamplingPlanIO
from .prompt_relay_advanced import PROMPT_RELAY_PLAN_TYPE
from .sampling import (
    DEFAULT_SAMPLER_NAME,
    DEFAULT_SCHEDULER_NAME,
    SAMPLER_OPTIONS,
    SCHEDULER_OPTIONS,
)


CATEGORY = "T8/MiniMax H3/Long Video/Experimental"
PromptRelayPlanIO = io.Custom(PROMPT_RELAY_PLAN_TYPE)


def _preview_video(path_value: str):
    path = Path(path_value).resolve()
    output_root = Path(folder_paths.get_output_directory()).resolve()
    if output_root not in path.parents:
        raise ValueError("In-node effects preview is outside the ComfyUI output directory")
    relative = path.relative_to(output_root)
    saved = ui.SavedResult(relative.name, relative.parent.as_posix(), io.FolderType.output)
    return InputImpl.VideoFromFile(str(path)), ui.PreviewVideo([saved])


class MiniMaxH3LongVideoInNodeLoopEffectsT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3LongVideoInNodeLoopEffectsT8Advanced",
            display_name=(
                "MiniMax H3 In-Node Long Video + Relay/EAV / 内循环长视频增强 "
                "(Advanced EXP/T8)"
            ),
            description=(
                "Append-only long-video runner that projects one global Prompt Relay timeline "
                "into every segment and/or creates one fresh Enhance-A-Video audit runtime per "
                "segment. Relay+EAV uses one Relay-then-FETA attention owner. The original "
                "in-node loop node and every old workflow remain unchanged."
            ),
            category=CATEGORY,
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input("clip", tooltip="Native MiniMax H3 Qwen3-VL CLIP."),
                io.Vae.Input("video_vae"),
                io.Vae.Input("audio_vae"),
                io.String.Input("chain_id", default="my_h3_in_node_effects_video"),
                io.Float.Input(
                    "total_duration_seconds",
                    default=30.0,
                    min=0.04,
                    max=3600.0,
                    step=0.01,
                ),
                io.Int.Input("width", default=736, min=32, max=16384, step=32),
                io.Int.Input("height", default=416, min=32, max=16384, step=32),
                io.Int.Input(
                    "render_window_frames",
                    default=124,
                    min=124,
                    max=362,
                    step=17,
                ),
                io.Combo.Input("context_frames", options=[5, 22, 39], default=22),
                io.String.Input(
                    "global_prompt",
                    default="",
                    multiline=True,
                    dynamic_prompts=True,
                    tooltip=(
                        "Used when Prompt Relay is disabled. With Relay, leave empty or copy "
                        "the Plan global prompt exactly."
                    ),
                ),
                io.String.Input(
                    "segment_prompts_json",
                    default="",
                    multiline=True,
                    advanced=True,
                    tooltip="Prompt overrides must be empty when Prompt Relay owns the timeline.",
                ),
                io.Combo.Input(
                    "prompt_relay_mode",
                    options=list(PROMPT_RELAY_MODES),
                    default="disabled",
                    tooltip=(
                        "disabled is exact bypass; report_only compiles/projects Relay without "
                        "attention bias; apply_exp enables the projected route."
                    ),
                ),
                io.Int.Input(
                    "query_chunk_rows",
                    default=256,
                    min=32,
                    max=2048,
                    step=32,
                    advanced=True,
                ),
                io.Combo.Input(
                    "eav_mode",
                    options=["disabled", "report_only", "apply_exp"],
                    default="disabled",
                    tooltip=(
                        "Stock20 only. report_only audits CFI/g without modifying attention; "
                        "apply_exp enables target-video FETA gain."
                    ),
                ),
                io.Float.Input(
                    "eav_tau",
                    default=4.0,
                    min=-32.0,
                    max=32.0,
                    step=0.25,
                ),
                io.Float.Input(
                    "eav_start_video_progress",
                    default=0.0,
                    min=0.0,
                    max=0.99,
                    step=0.01,
                    advanced=True,
                ),
                io.Float.Input(
                    "eav_end_video_progress",
                    default=1.0,
                    min=0.01,
                    max=1.0,
                    step=0.01,
                    advanced=True,
                ),
                io.Int.Input(
                    "eav_max_workspace_mib",
                    default=32,
                    min=4,
                    max=512,
                    step=4,
                    advanced=True,
                ),
                io.Float.Input(
                    "eav_g_hard_limit",
                    default=1.5,
                    min=1.0,
                    max=3.0,
                    step=0.01,
                    advanced=True,
                ),
                io.Int.Input(
                    "minimum_free_vram_mib",
                    default=512,
                    min=0,
                    max=65536,
                    step=64,
                    advanced=True,
                    tooltip="Rechecked before every segment; this is a start floor, not a peak guarantee.",
                ),
                io.Int.Input(
                    "base_seed", default=123456789, min=0, max=0xFFFFFFFFFFFFFFFF
                ),
                io.Combo.Input(
                    "seed_policy",
                    options=["increment", "fixed", "hash_chain_segment"],
                    default="increment",
                ),
                io.Int.Input("steps", default=20, min=1, max=1000),
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
                io.Combo.Input(
                    "sampler_name",
                    options=SAMPLER_OPTIONS,
                    default=DEFAULT_SAMPLER_NAME,
                ),
                io.Combo.Input(
                    "scheduler",
                    options=SCHEDULER_OPTIONS,
                    default=DEFAULT_SCHEDULER_NAME,
                ),
                io.Combo.Input(
                    "task_type",
                    options=["auto", "T2VA", "I2VA", "FL2VA", "L2VA", "Ref2VA", "Hybrid"],
                    default="auto",
                ),
                io.Combo.Input(
                    "context_audio",
                    options=["video_and_audio", "video_only"],
                    default="video_and_audio",
                ),
                io.Combo.Input(
                    "audio_mode",
                    options=["lock_source", "remix_source", "reference_only", "native"],
                    default="native",
                ),
                io.Float.Input(
                    "audio_denoise_strength",
                    default=0.35,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    advanced=True,
                ),
                io.Boolean.Input("add_source_as_reference", default=True, advanced=True),
                io.Int.Input(
                    "prompt_primary_audio_ordinal",
                    default=0,
                    min=0,
                    max=9,
                    advanced=True,
                ),
                io.Boolean.Input("strict_prompt_tags", default=True, advanced=True),
                io.Combo.Input(
                    "ref_image_size",
                    options=["match", "max"],
                    default="match",
                    advanced=True,
                ),
                io.Combo.Input(
                    "reference_video_policy",
                    options=["official_2_to_15s", "model_minimum"],
                    default="official_2_to_15s",
                    advanced=True,
                ),
                io.Combo.Input(
                    "first_frame_reuse",
                    options=["segment0_only", "persistent_identity_reference"],
                    default="segment0_only",
                    advanced=True,
                ),
                io.Combo.Input(
                    "persistent_identity_strategy",
                    options=["single_reference", "scene_plus_identity"],
                    default="single_reference",
                    advanced=True,
                ),
                io.Int.Input(
                    "persistent_identity_interval",
                    default=1,
                    min=1,
                    max=32,
                    advanced=True,
                ),
                io.Boolean.Input("resume_existing", default=True),
                io.String.Input("filename_prefix", default="H3_In_Node_Effects_Long_Video"),
                io.Combo.Input(
                    "audio_seam_policy",
                    options=["cosine_bridge", "none"],
                    default="cosine_bridge",
                ),
                io.Float.Input(
                    "bridge_ms",
                    default=5.0,
                    min=0.0,
                    max=50.0,
                    step=0.1,
                    advanced=True,
                ),
                io.Combo.Input("bit_depth", options=[8, 10], default=8, advanced=True),
                io.Int.Input("crf", default=18, min=0, max=51, advanced=True),
                io.String.Input("model_id", default="unknown", advanced=True),
                PromptRelayPlanIO.Input("prompt_relay_plan", optional=True),
                io.Audio.Input("drive_audio", optional=True),
                io.Audio.Input("final_audio", optional=True),
                io.Image.Input("first_frame", optional=True),
                io.Image.Input("last_frame", optional=True),
                io.Image.Input("persistent_identity_image", optional=True, advanced=True),
                io.Autogrow.Input(
                    "ref_images",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("ref_image"), prefix="ref_image_", min=0, max=9
                    ),
                ),
                io.Autogrow.Input(
                    "ref_videos",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("ref_video"), prefix="ref_video_", min=0, max=3
                    ),
                ),
                io.Autogrow.Input(
                    "ref_video_audios",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input("ref_video_audio"),
                        prefix="ref_video_audio_",
                        min=0,
                        max=3,
                    ),
                ),
                io.Autogrow.Input(
                    "ref_audios",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input("ref_audio"), prefix="ref_audio_", min=0, max=3
                    ),
                ),
                LongVideoSamplingPlanIO.Input(
                    "long_video_sampling_plan",
                    optional=True,
                    tooltip=(
                        "Optional Tail/manual second-pass plan. Prompt Relay remains scoped; "
                        "EAV audits pass 1 only for manual second-pass mode."
                    ),
                ),
            ],
            outputs=[
                io.Video.Output("video"),
                io.String.Output("video_path"),
                io.String.Output("manifest_path"),
                io.Int.Output("completed_segments"),
                io.String.Output("status"),
                io.String.Output("report_json"),
            ],
            is_output_node=True,
            is_experimental=True,
        )

    @classmethod
    def execute(cls, **kwargs):
        video_path, manifest_path, completed, status, report = (
            run_long_video_in_node_loop_effects(**kwargs)
        )
        video, preview = _preview_video(video_path)
        return io.NodeOutput(
            video,
            video_path,
            manifest_path,
            completed,
            status,
            report,
            ui=preview,
        )


LONG_VIDEO_IN_NODE_LOOP_EFFECTS_ADVANCED_NODE_CLASSES = [
    MiniMaxH3LongVideoInNodeLoopEffectsT8Advanced,
]
