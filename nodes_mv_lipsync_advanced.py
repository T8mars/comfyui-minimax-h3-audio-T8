from __future__ import annotations

from pathlib import Path

import folder_paths
from comfy_api.latest import InputImpl, io, ui

from .mv_lipsync_advanced import (
    MV_PROMPT_PLAN_TYPE,
    MV_SCENE_PLAN_TYPE,
    build_mv_prompt_plan,
    build_mv_scene_plan,
    run_local_mv_in_node_loop,
)
from .prompt_relay_events_advanced import PROMPT_RELAY_EVENTS_TYPE
from .sampling import (
    DEFAULT_SAMPLER_NAME,
    DEFAULT_SCHEDULER_NAME,
    SAMPLER_OPTIONS,
    SCHEDULER_OPTIONS,
)


CATEGORY = "T8/MiniMax H3/MV & Lip Sync/Experimental"
MVScenePlanIO = io.Custom(MV_SCENE_PLAN_TYPE)
MVPromptPlanIO = io.Custom(MV_PROMPT_PLAN_TYPE)
PromptRelayEventsIO = io.Custom(PROMPT_RELAY_EVENTS_TYPE)


def _preview_video(path_value: str):
    path = Path(path_value).resolve()
    output_root = Path(folder_paths.get_output_directory()).resolve()
    if output_root not in path.parents:
        raise ValueError("Local MV preview is outside the ComfyUI output directory")
    relative = path.relative_to(output_root)
    saved = ui.SavedResult(relative.name, relative.parent.as_posix(), io.FolderType.output)
    return InputImpl.VideoFromFile(str(path)), ui.PreviewVideo([saved])


class MiniMaxH3MVVocalScenePlannerT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3MVVocalScenePlannerT8Advanced",
            display_name=(
                "MiniMax H3 MV Vocal Scene Planner / 本地歌曲分镜 "
                "(Advanced EXP/T8)"
            ),
            description=(
                "Analyzes local AUDIO with deterministic CPU tensor math and chooses 5–10s "
                "scene boundaries before generation. A local vocal stem is optional. No remote "
                "LLM, TTS, music service or external API is called."
            ),
            category=CATEGORY,
            inputs=[
                io.Audio.Input("full_song", tooltip="最终成片使用的完整原曲。"),
                io.Float.Input(
                    "min_scene_seconds", default=5.0, min=1.0, max=15.0, step=0.1
                ),
                io.Float.Input(
                    "target_scene_seconds", default=7.0, min=1.0, max=15.0, step=0.1
                ),
                io.Float.Input(
                    "max_scene_seconds", default=10.0, min=1.0, max=15.0, step=0.1
                ),
                io.Int.Input(
                    "analysis_hop_ms",
                    default=100,
                    min=20,
                    max=500,
                    step=10,
                    advanced=True,
                ),
                io.Combo.Input(
                    "vocal_policy",
                    options=["assume_vocal", "energy_proxy", "vocal_stem_required"],
                    default="assume_vocal",
                    tooltip=(
                        "没有人声干声时推荐 assume_vocal；energy_proxy 仅按能量估计；"
                        "vocal_stem_required 要求连接本地干声。"
                    ),
                ),
                io.String.Input(
                    "manual_boundaries_json",
                    default="",
                    multiline=True,
                    advanced=True,
                    tooltip='可选秒数列表，例如 [5.2, 11.8]；留空自动分析。',
                ),
                io.Audio.Input("vocal_stem", optional=True, tooltip="可选的本地人声干声。"),
            ],
            outputs=[
                MVScenePlanIO.Output("scene_plan"),
                io.Int.Output("scene_count"),
                io.Float.Output("duration_seconds"),
                io.String.Output("timeline_json"),
                io.String.Output("report_json"),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*build_mv_scene_plan(**kwargs))


class MiniMaxH3MVRef2VAPromptCompilerT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3MVRef2VAPromptCompilerT8Advanced",
            display_name=(
                "MiniMax H3 MV Ref2VA Prompt Compiler / 本地口型分镜提示词 "
                "(Advanced EXP/T8)"
            ),
            description=(
                "Deterministically compiles one six-section Ref2VA prompt per local scene. "
                "It uses <Picture 1>/<Audio 1>, never guesses lyrics and also exposes typed "
                "Prompt Relay events. No LLM or external API is used."
            ),
            category=CATEGORY,
            inputs=[
                MVScenePlanIO.Input("scene_plan"),
                io.String.Input(
                    "global_creative_prompt",
                    default=(
                        "A singer performs through a coherent cinematic music video with "
                        "natural expression and intentional scene changes."
                    ),
                    multiline=True,
                    dynamic_prompts=True,
                ),
                io.String.Input(
                    "performer_description",
                    default="the same lead performer shown in the reference picture",
                    multiline=True,
                ),
                io.String.Input(
                    "visual_style",
                    default="cinematic music video, natural skin, realistic light and texture",
                    multiline=True,
                ),
                io.String.Input(
                    "camera_pattern",
                    default=(
                        "stable medium close-up with subtle handheld movement\n"
                        "smooth lateral tracking medium shot\n"
                        "restrained slow push-in with a stable background"
                    ),
                    multiline=True,
                    tooltip="每行或每个 | 一种镜头，按场景循环使用。",
                ),
                io.String.Input(
                    "non_vocal_action",
                    default="keeps the mouth naturally closed and moves with the rhythm",
                    multiline=True,
                ),
                io.String.Input(
                    "exact_lyrics_json",
                    default="",
                    multiline=True,
                    advanced=True,
                    tooltip=(
                        "可选：按场景提供精确歌词字符串列表。留空时绝不猜歌词，也不生成字幕。"
                    ),
                ),
            ],
            outputs=[
                MVPromptPlanIO.Output("mv_prompt_plan"),
                io.String.Output("segment_prompts_json"),
                PromptRelayEventsIO.Output("prompt_relay_events"),
                io.String.Output("prompt_preview"),
                io.String.Output("report_json"),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*build_mv_prompt_plan(**kwargs))


class MiniMaxH3LocalMVInNodeRendererT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3LocalMVInNodeRendererT8Advanced",
            display_name=(
                "MiniMax H3 Local MV In-Node Renderer / 全本地MV内循环生成 "
                "(Advanced EXP/T8)"
            ),
            description=(
                "Generates every Ref2VA scene serially through the connected local H3 MODEL, "
                "atomically resumes accepted scenes, assembles video with bounded memory and "
                "muxes the original full song once. It never submits ComfyUI /prompt HTTP jobs "
                "and never calls a remote video, LLM, TTS or music API."
            ),
            category=CATEGORY,
            inputs=[
                io.Model.Input("model", tooltip="本地 MiniMax H3 MODEL。"),
                io.Clip.Input("clip", tooltip="本地 MiniMax H3 Qwen3-VL CLIP。"),
                io.Vae.Input("video_vae"),
                io.Vae.Input("audio_vae"),
                io.Image.Input("reference_image", tooltip="歌手/人物身份参考图。"),
                io.Audio.Input("full_song", tooltip="驱动表演并最终一次性混入成片的原曲。"),
                MVPromptPlanIO.Input("mv_prompt_plan"),
                io.String.Input("chain_id", default="my_h3_local_mv"),
                io.Int.Input("width", default=1056, min=32, max=16384, step=32),
                io.Int.Input("height", default=608, min=32, max=16384, step=32),
                io.Int.Input("base_seed", default=123456789, min=0, max=0xFFFFFFFFFFFFFFFF),
                io.Int.Input("steps", default=8, min=1, max=1000),
                io.Float.Input(
                    "shift_video",
                    default=6.0,
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
                io.Boolean.Input("resume_existing", default=True),
                io.String.Input("filename_prefix", default="H3_Local_MV"),
                io.Combo.Input("bit_depth", options=[8, 10], default=8, advanced=True),
                io.Int.Input("crf", default=18, min=0, max=51, advanced=True),
                io.String.Input(
                    "model_id",
                    default="user-selected-local-h3",
                    advanced=True,
                    tooltip="只写入审计报告，不校验文件名、大小或哈希。",
                ),
            ],
            outputs=[
                io.Video.Output("video"),
                io.String.Output("video_path"),
                io.String.Output("manifest_path"),
                io.Int.Output("completed_scenes"),
                io.String.Output("status"),
                io.String.Output("report_json"),
            ],
            is_output_node=True,
            is_experimental=True,
        )

    @classmethod
    def execute(cls, **kwargs):
        video_path, manifest_path, completed, status, report = run_local_mv_in_node_loop(
            **kwargs
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


MV_LIPSYNC_ADVANCED_NODE_CLASSES = [
    MiniMaxH3MVVocalScenePlannerT8Advanced,
    MiniMaxH3MVRef2VAPromptCompilerT8Advanced,
    MiniMaxH3LocalMVInNodeRendererT8Advanced,
]
