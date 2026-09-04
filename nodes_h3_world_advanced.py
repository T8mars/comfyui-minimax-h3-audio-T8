from __future__ import annotations

import json
from pathlib import Path

import folder_paths
from comfy_api.latest import InputImpl, io, ui

from .h3_world_advanced import (
    PLAN_TYPE,
    PRESETS,
    build_h3_world_i2va_conditioning,
    compile_action_plan,
    compose_h3_world_model,
    save_h3_world_video_safe,
)


CATEGORY = "T8/MiniMax H3/World"
ActionPlanIO = io.Custom(PLAN_TYPE)


def _lora_options() -> list[str]:
    names = list(folder_paths.get_filename_list("loras"))
    preferred = "minimax\\H3-World\\step-10000.safetensors"
    return sorted(names, key=lambda name: (name.replace("/", "\\") != preferred, name)) or [
        preferred
    ]


class MiniMaxH3WorldActionTimelineT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3WorldActionTimelineT8Advanced",
            display_name="MiniMax H3-World Action Timeline / 动作时间线 (T8)",
            description=(
                "Builds the exact 37-step H3-World action script for one 832x480, "
                "124-frame clip. Custom ranges use end-exclusive latent indices."
            ),
            category=CATEGORY,
            is_experimental=False,
            inputs=[
                io.Combo.Input(
                    "action_preset",
                    options=[*PRESETS, "custom"],
                    default="forward",
                    tooltip="WASD controls the character; IJKL controls the camera; F makes pan fast.",
                ),
                io.String.Input(
                    "custom_timeline_json",
                    multiline=True,
                    dynamic_prompts=False,
                    default=(
                        '[\n  {"start_latent":0,"end_latent":12,"keys":["W"]},\n'
                        '  {"start_latent":12,"end_latent":25,"keys":["L"]},\n'
                        '  {"start_latent":25,"end_latent":37,"keys":[]}\n]'
                    ),
                    tooltip="Only used when preset=custom. Segments must tile 0..37 without gaps.",
                ),
            ],
            outputs=[
                ActionPlanIO.Output("action_plan"),
                io.String.Output("action_script_json"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, action_preset, custom_timeline_json):
        return io.NodeOutput(*compile_action_plan(action_preset, custom_timeline_json))


class MiniMaxH3WorldModelComposerT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3WorldModelComposerT8Advanced",
            display_name="MiniMax H3-World Model Composer / 模型组合 (T8)",
            description=(
                "Loads the H3-World LoRA and installs its required per-action text "
                "refiner segmentation plus directed FlexAttention runtime."
            ),
            category=CATEGORY,
            is_experimental=False,
            inputs=[
                io.Model.Input("model", tooltip="Native ComfyUI MiniMax H3 FL2VA MODEL."),
                io.Combo.Input("lora_name", options=_lora_options()),
                io.Float.Input(
                    "strength_model", default=1.0, min=0.0, max=2.0, step=0.01
                ),
                io.Boolean.Input(
                    "compile_flex_attention",
                    default=True,
                    advanced=True,
                    tooltip="Recommended. First run compiles one shape-specific FlexAttention kernel.",
                ),
            ],
            outputs=[io.Model.Output("model"), io.String.Output("report_json")],
        )

    @classmethod
    def execute(
        cls,
        model,
        lora_name,
        strength_model=1.0,
        compile_flex_attention=True,
    ):
        path = folder_paths.get_full_path_or_raise("loras", lora_name)
        return io.NodeOutput(
            *compose_h3_world_model(
                model,
                path,
                strength_model=strength_model,
                compile_flex_attention=compile_flex_attention,
            )
        )


class MiniMaxH3WorldI2VAConditioningT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3WorldI2VAConditioningT8Advanced",
            display_name="MiniMax H3-World I2VA 832×480×124 (T8)",
            description=(
                "First-frame H3-World conditioning. Action sentences are encoded and "
                "refined independently, then bound one-to-one to 37 video latents."
            ),
            category=CATEGORY,
            is_experimental=False,
            inputs=[
                io.Clip.Input("clip", tooltip="Native MiniMax H3 Qwen3-VL CLIP."),
                io.Vae.Input("video_vae"),
                io.Vae.Input("audio_vae"),
                io.Image.Input("first_frame"),
                io.String.Input(
                    "prompt",
                    multiline=True,
                    dynamic_prompts=True,
                    default=(
                        "A man in a yellow floral shirt stands in a dim, multi-level "
                        "concrete parking garage."
                    ),
                ),
                ActionPlanIO.Input("action_plan"),
            ],
            outputs=[
                io.Conditioning.Output("positive"),
                io.Latent.Output("av_latent"),
                io.String.Output("conditioned_prompt"),
                io.String.Output("action_script_json"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, clip, video_vae, audio_vae, first_frame, prompt, action_plan):
        return io.NodeOutput(
            *build_h3_world_i2va_conditioning(
                clip, video_vae, audio_vae, first_frame, prompt, action_plan
            )
        )


class MiniMaxH3WorldSafeVideoSaveT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3WorldSafeVideoSaveT8Advanced",
            display_name="MiniMax H3-World Safe Video Save / 安全视频保存 (T8)",
            description=(
                "Saves the fixed 832x480x124 H3-World result with an isolated, "
                "single-thread libx264 process. The MP4 is published only after strict "
                "video and audio decoding succeeds."
            ),
            category=CATEGORY,
            is_experimental=False,
            is_output_node=True,
            inputs=[
                io.Image.Input("images"),
                io.Audio.Input("audio"),
                io.String.Input(
                    "filename_prefix",
                    default="MiniMaxH3/H3-World/h3_world_i2va_832x480_124f",
                ),
                io.Int.Input("crf", default=18, min=0, max=51, advanced=True),
            ],
            outputs=[
                io.Video.Output("video"),
                io.String.Output("saved_path"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, images, audio, filename_prefix, crf=18):
        full_output_folder, filename, counter, subfolder, _prefix = (
            folder_paths.get_save_image_path(
                filename_prefix,
                folder_paths.get_output_directory(),
                832,
                480,
            )
        )
        saved_name = f"{filename}_{counter:05}_.mp4"
        output_path, report = save_h3_world_video_safe(
            images,
            audio,
            Path(full_output_folder) / saved_name,
            crf=crf,
        )
        preview = ui.PreviewVideo(
            [ui.SavedResult(saved_name, subfolder, io.FolderType.output)]
        )
        return io.NodeOutput(
            InputImpl.VideoFromFile(str(output_path)),
            str(output_path),
            json.dumps(report, ensure_ascii=False, indent=2),
            ui=preview,
        )


H3_WORLD_ADVANCED_NODE_CLASSES = [
    MiniMaxH3WorldActionTimelineT8Advanced,
    MiniMaxH3WorldModelComposerT8Advanced,
    MiniMaxH3WorldI2VAConditioningT8Advanced,
    MiniMaxH3WorldSafeVideoSaveT8Advanced,
]
