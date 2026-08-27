from __future__ import annotations

import torch

import comfy.nested_tensor
from comfy_api.latest import io


CATEGORY = "T8/MiniMax H3/Latent/Advanced"
FPS = 24
AUDIO_LATENT_FPS = 40
FRAME_PER_TOKEN = (1, 4, 4, 4, 4)


def _stream_tensor(latent: object, name: str) -> torch.Tensor:
    samples = latent.get("samples") if isinstance(latent, dict) else None
    if not torch.is_tensor(samples):
        raise ValueError(f"{name} must contain a tensor in samples")
    return samples


def _video_frame_count(video_token_count: int) -> int:
    cycles, remainder = divmod(int(video_token_count), len(FRAME_PER_TOKEN))
    return cycles * sum(FRAME_PER_TOKEN) + sum(FRAME_PER_TOKEN[:remainder])


def build_h3_av_latent(video_latent: object, audio_latent: object) -> dict:
    """Combine separately encoded H3 streams without changing either tensor."""
    video = _stream_tensor(video_latent, "video_latent")
    audio = _stream_tensor(audio_latent, "audio_latent")

    if (
        video.ndim != 5
        or video.shape[1] != 24
        or any(size <= 0 for size in (video.shape[0], *video.shape[2:]))
    ):
        raise ValueError(
            "MiniMax H3 video latent must have shape [B, 24, T, H, W] "
            f"with positive dimensions, got {tuple(video.shape)}"
        )
    if (
        audio.ndim != 4
        or audio.shape[1] != 32
        or audio.shape[2] != 2
        or any(size <= 0 for size in (audio.shape[0], audio.shape[3]))
    ):
        raise ValueError(
            "MiniMax H3 audio latent must have shape [B, 32, 2, T] "
            f"with positive dimensions, got {tuple(audio.shape)}"
        )
    if video.shape[0] != audio.shape[0]:
        raise ValueError(
            "MiniMax H3 video and audio latent batch sizes must match, "
            f"got {video.shape[0]} and {audio.shape[0]}"
        )

    frame_count = _video_frame_count(video.shape[2])
    expected_audio_t = round(frame_count / FPS * AUDIO_LATENT_FPS)
    if abs(audio.shape[3] - expected_audio_t) > 1:
        raise ValueError(
            "MiniMax H3 video/audio latent timelines do not match: "
            f"video T={video.shape[2]} represents {frame_count} frames and "
            f"expects audio T={expected_audio_t} (±1), got {audio.shape[3]}"
        )

    return {"samples": comfy.nested_tensor.NestedTensor((video, audio))}


class MiniMaxH3AVLatentBuilderT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3AVLatentBuilderT8Advanced",
            display_name="MiniMax H3 AV Latent Builder (T8 Advanced)",
            category=CATEGORY,
            description=(
                "Combine separately encoded MiniMax H3 video and audio latents. "
                "Validates the official stream shapes and shared timeline before sampling."
            ),
            inputs=[
                io.Latent.Input("video_latent"),
                io.Latent.Input("audio_latent"),
            ],
            outputs=[io.Latent.Output(display_name="av_latent")],
        )

    @classmethod
    def execute(cls, video_latent, audio_latent) -> io.NodeOutput:
        return io.NodeOutput(build_h3_av_latent(video_latent, audio_latent))


AV_LATENT_BUILDER_ADVANCED_NODE_CLASSES = [MiniMaxH3AVLatentBuilderT8Advanced]
