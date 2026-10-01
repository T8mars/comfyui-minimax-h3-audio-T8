"""Exact opt-in native checkpoint for Motion's cold pass-2 source."""
import re

from comfy_api.latest import io

from ..core import AUDIO_LATENT_FPS, FPS, align_frame_count, nested_av_parts, video_latent_t
from ..native_latent_checkpoint_advanced import (
    fingerprint_native_h3_checkpoint_file, load_native_h3_av_checkpoint,
)
from ..nodes_native_latent_checkpoint_advanced import native_h3_checkpoint_storage_root

CATEGORY = "T8/MiniMax H3/Modular Sampling/Experimental"
FIRST_PASS_CHECKPOINT_ID = "motion_recovery_firstpass"
SHA256 = re.compile(r"[0-9a-fA-F]{64}\Z")


class MiniMaxH3MotionFrozenFirstPassLoadEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3MotionFrozenFirstPassLoadEXPT8",
            display_name="H3 Motion Recovery · Load Verified First Pass (T8 EXP)",
            category=CATEGORY, is_experimental=True,
            description="Explicit cold Motion source. Requires an independently retained native AV "
                        "manifest and whole-file SHA, exact first-pass ID and original H3 geometry. "
                        "Never samples or silently accepts a different frozen source.",
            inputs=[io.String.Input("checkpoint_path", default=""),
                    io.String.Input("expected_manifest_json", default="", multiline=True),
                    io.String.Input("expected_file_sha256", default=""),
                    io.Int.Input("expected_frame_count", default=124, min=5, max=3600, step=1),
                    io.Int.Input("expected_width", default=512, min=32, max=8192, step=32),
                    io.Int.Input("expected_height", default=768, min=32, max=8192, step=32),
                    io.Int.Input("hash_chunk_megabytes", default=8, min=1, max=64, step=1)],
            outputs=[io.Latent.Output("av_latent"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, checkpoint_path, expected_manifest_json, expected_file_sha256,
                expected_frame_count, expected_width, expected_height, hash_chunk_megabytes=8):
        if not isinstance(checkpoint_path, str) or not checkpoint_path.strip():
            raise ValueError("Motion cold pass 2 requires an explicit first-pass checkpoint path")
        if not isinstance(expected_manifest_json, str) or not expected_manifest_json.strip():
            raise ValueError("Motion cold pass 2 requires an external first-pass manifest")
        if not isinstance(expected_file_sha256, str) or not SHA256.fullmatch(expected_file_sha256):
            raise ValueError("Motion cold pass 2 requires exact first-pass file SHA-256")
        if (type(expected_frame_count) is not int or expected_frame_count < 5
                or align_frame_count(expected_frame_count) != expected_frame_count
                or type(expected_width) is not int or type(expected_height) is not int
                or expected_width < 32 or expected_height < 32
                or expected_width % 32 or expected_height % 32):
            raise ValueError("Motion cold pass 2 requires native H3 frame/canvas geometry")
        loaded = load_native_h3_av_checkpoint(
            storage_root=native_h3_checkpoint_storage_root(), checkpoint_path=checkpoint_path,
            expected_manifest_json=expected_manifest_json,
            expected_file_sha256=expected_file_sha256,
            hash_chunk_megabytes=hash_chunk_megabytes)
        if loaded[1] != "MATCH_EXTERNAL" or loaded[2] is not True or loaded[3] != FIRST_PASS_CHECKPOINT_ID:
            raise ValueError("Motion first-pass checkpoint did not match the external receipt and ID")
        video, audio = nested_av_parts(loaded[0])
        if (tuple(video.shape) != (1, 24, video_latent_t(expected_frame_count),
                                   expected_height // 16, expected_width // 16)
                or tuple(audio.shape) != (1, 32, 2,
                                          round(expected_frame_count / FPS * AUDIO_LATENT_FPS))):
            raise ValueError("Motion frozen first pass differs from expected AV geometry")
        return io.NodeOutput(loaded[0], loaded[7])

    @classmethod
    def fingerprint_inputs(cls, checkpoint_path, expected_manifest_json="",
                           expected_file_sha256="", expected_frame_count=124,
                           expected_width=512, expected_height=768, hash_chunk_megabytes=8):
        try:
            fingerprint = fingerprint_native_h3_checkpoint_file(
                native_h3_checkpoint_storage_root(), checkpoint_path)
        except (FileNotFoundError, ValueError):
            fingerprint = f"unresolved:{checkpoint_path}"
        return (f"{fingerprint}:{expected_file_sha256}:{expected_manifest_json}:"
                f"{expected_frame_count}:{expected_width}:{expected_height}:"
                f"{int(hash_chunk_megabytes)}")
