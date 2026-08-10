from __future__ import annotations

from comfy_api.latest import io

from .source_av import (
    AUDIO_FIT_POLICIES,
    DTYPE_DEVICE_POLICIES,
    SHORT_AUDIO_POLICIES,
    SHORT_VIDEO_POLICIES,
    STREAM_MODES,
    prepare_source_media_window,
    prepare_source_av_latent,
    separate_source_av_latent,
)


CATEGORY = "T8/MiniMax H3/Source AV/Experimental"


class MiniMaxH3SourceMediaWindowT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SourceMediaWindowT8",
            display_name="MiniMax H3 Source Media Window / 来源视频窗口 (EXP/T8)",
            description=(
                "Selects a 24fps, 17n+5 source window, resizes it to an H3 canvas, and creates "
                "an exactly timed 32kHz stereo audio window for the standard video/audio VAE "
                "encoders. The full IMAGE input is already resident; this is not streaming decode "
                "and does not claim a memory-safe long-video path."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Image.Input("frames"),
                io.Float.Input("source_fps", default=24.0, min=0.01, max=240.0, step=0.001),
                io.Int.Input("width", default=736, min=32, max=1920, step=32),
                io.Int.Input("height", default=416, min=32, max=1088, step=32),
                io.Int.Input(
                    "length",
                    default=124,
                    min=5,
                    max=362,
                    step=17,
                    tooltip="Snapped up to H3's 17n+5 target grid.",
                ),
                io.Float.Input(
                    "start_seconds",
                    default=0.0,
                    min=0.0,
                    max=86400.0,
                    step=0.001,
                ),
                io.Combo.Input(
                    "short_video_policy",
                    options=list(SHORT_VIDEO_POLICIES),
                    default="strict",
                    display_name="short video / 视频不足",
                ),
                io.Combo.Input(
                    "short_audio_policy",
                    options=list(SHORT_AUDIO_POLICIES),
                    default="pad_silence",
                    display_name="short audio / 音频不足",
                ),
                io.Audio.Input(
                    "source_audio",
                    optional=True,
                    tooltip="Optional source soundtrack. Missing audio becomes reported stereo silence.",
                ),
            ],
            outputs=[
                io.Image.Output("frames"),
                io.Audio.Output("audio"),
                io.Int.Output("frame_count"),
                io.Float.Output("duration_seconds"),
                io.String.Output("report"),
            ],
        )

    @classmethod
    def execute(
        cls,
        frames,
        source_fps,
        width,
        height,
        length,
        start_seconds,
        short_video_policy,
        short_audio_policy,
        source_audio=None,
    ):
        return io.NodeOutput(
            *prepare_source_media_window(
                frames,
                source_fps,
                width,
                height,
                length,
                start_seconds,
                short_video_policy,
                short_audio_policy,
                source_audio,
            )
        )


class MiniMaxH3SourceAVPrepareT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SourceAVPrepareT8",
            display_name="MiniMax H3 Source AV Prepare / 来源音画重绘准备 (EXP/T8)",
            description=(
                "Strictly assembles H3 video/audio latents, preserves metadata and masks, "
                "aligns the audio clock only under an explicit policy, and creates separate "
                "video/audio denoise masks. This is stream assembly, not temporal concat; "
                "denoise strength is experimental and is not claimed to be a linear redraw weight."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Latent.Input(
                    "video_latent",
                    tooltip="H3 video latent [B,24,T,H,W], or an existing H3 joint AV latent.",
                ),
                io.Latent.Input(
                    "audio_latent",
                    tooltip="H3 audio latent [B,32,2,T], or an H3 joint AV latent whose audio stream will be used.",
                ),
                io.Combo.Input(
                    "video_mode",
                    options=list(STREAM_MODES),
                    default="remix",
                    display_name="video mode / 画面模式",
                    tooltip="lock=mask 0; remix=use strength; regenerate=mask 1.",
                ),
                io.Float.Input(
                    "video_denoise_strength",
                    default=0.5,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    display_name="video denoise / 画面重绘强度",
                    tooltip="Used only by remix. No linear visual-strength claim is made before real A/B calibration.",
                ),
                io.Combo.Input(
                    "audio_mode",
                    options=list(STREAM_MODES),
                    default="lock",
                    display_name="audio mode / 音频模式",
                    tooltip="lock=preserve source latent; remix=use strength; regenerate=mask 1.",
                ),
                io.Float.Input(
                    "audio_denoise_strength",
                    default=0.35,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    display_name="audio denoise / 音频重绘强度",
                    tooltip="Used only by remix; regenerated or padded audio still requires perceptual validation.",
                ),
                io.Combo.Input(
                    "audio_fit_policy",
                    options=list(AUDIO_FIT_POLICIES),
                    default="fit_to_video_generate_tail",
                    display_name="audio fit / 音频时钟对齐",
                    tooltip=(
                        "strict never adjusts. trim/pad policies are explicit; a padded tail is zero latent "
                        "with mask 1 so H3 may generate it. Every adjustment is reported."
                    ),
                ),
                io.Combo.Input(
                    "dtype_device_policy",
                    options=list(DTYPE_DEVICE_POLICIES),
                    default="match_video",
                    display_name="dtype/device / 精度与设备",
                    advanced=True,
                    tooltip="match_video converts only the smaller audio stream; strict refuses a mismatch.",
                ),
            ],
            outputs=[
                io.Latent.Output("av_latent"),
                io.Latent.Output("video_latent"),
                io.Latent.Output("audio_latent"),
                io.String.Output("report"),
            ],
        )

    @classmethod
    def execute(
        cls,
        video_latent,
        audio_latent,
        video_mode,
        video_denoise_strength,
        audio_mode,
        audio_denoise_strength,
        audio_fit_policy,
        dtype_device_policy,
    ):
        return io.NodeOutput(
            *prepare_source_av_latent(
                video_latent,
                audio_latent,
                video_mode,
                video_denoise_strength,
                audio_mode,
                audio_denoise_strength,
                audio_fit_policy,
                dtype_device_policy,
            )
        )


class MiniMaxH3AVLatentSeparateT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3AVLatentSeparateT8",
            display_name="MiniMax H3 AV Latent Separate / 联合潜空间拆分 (EXP/T8)",
            description=(
                "Validates and separates a joint H3 AV latent without invoking either VAE. "
                "It preserves metadata and per-stream noise masks, so it is cheaper than decoding "
                "when only latent routing or replacement is needed."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[io.Latent.Input("av_latent")],
            outputs=[
                io.Latent.Output("video_latent"),
                io.Latent.Output("audio_latent"),
                io.String.Output("report"),
            ],
        )

    @classmethod
    def execute(cls, av_latent):
        return io.NodeOutput(*separate_source_av_latent(av_latent))


SOURCE_AV_NODE_CLASSES = [
    MiniMaxH3SourceMediaWindowT8,
    MiniMaxH3SourceAVPrepareT8,
    MiniMaxH3AVLatentSeparateT8,
]
