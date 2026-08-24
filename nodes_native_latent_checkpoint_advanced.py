from __future__ import annotations

from pathlib import Path

import folder_paths
from comfy_api.latest import io

from .native_latent_checkpoint_advanced import (
    fingerprint_native_h3_checkpoint_file,
    load_native_h3_av_checkpoint,
    save_native_h3_av_checkpoint,
)


CATEGORY = "T8/MiniMax H3/Long Video/Experimental"


def native_h3_checkpoint_storage_root() -> Path:
    output_root = Path(folder_paths.get_output_directory()).resolve()
    return (output_root / "MiniMaxH3" / "latent_checkpoints").resolve()


class MiniMaxH3NativeLatentCheckpointSaveT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3NativeLatentCheckpointSaveT8Advanced",
            display_name=(
                "MiniMax H3 Native AV Latent Checkpoint Save / "
                "原生音画Latent检查点保存 (Advanced EXP/T8)"
            ),
            description=(
                "Atomically saves one complete H3 nested video/audio latent as a no-pickle "
                ".h3latent.safetensors checkpoint under the ComfyUI output directory. "
                "confirm_save is false by default and existing files are never overwritten."
            ),
            category=CATEGORY,
            is_experimental=True,
            is_output_node=True,
            inputs=[
                io.Latent.Input("av_latent"),
                io.String.Input(
                    "filename_prefix",
                    default="h3_native_latent",
                    tooltip=(
                        "Relative filename prefix under output/MiniMaxH3/latent_checkpoints. "
                        "A unique suffix is always added; absolute and traversal paths are rejected."
                    ),
                ),
                io.String.Input(
                    "checkpoint_id",
                    default="timeline_checkpoint",
                    tooltip="Logical identity that must match when the checkpoint is reloaded.",
                ),
                io.Boolean.Input(
                    "confirm_save",
                    default=False,
                    tooltip=(
                        "Safety gate. False only computes the exact content manifest and writes "
                        "nothing; enable explicitly after reviewing the target and checkpoint ID."
                    ),
                ),
                io.Boolean.Input(
                    "verify_after_write",
                    default=True,
                    advanced=True,
                    tooltip=(
                        "Reloads the temporary safetensors file and verifies its embedded exact "
                        "AV manifest before atomic placement. Keep enabled for resumable work."
                    ),
                ),
                io.Int.Input(
                    "hash_chunk_megabytes",
                    default=8,
                    min=1,
                    max=64,
                    step=1,
                    advanced=True,
                    tooltip=(
                        "Maximum temporary CPU hash chunk. This changes memory/speed only, not "
                        "the checkpoint content digest."
                    ),
                ),
            ],
            outputs=[
                io.Latent.Output("av_latent"),
                io.String.Output("status"),
                io.String.Output("checkpoint_path"),
                io.String.Output("file_sha256"),
                io.String.Output("manifest_json"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(
            *save_native_h3_av_checkpoint(
                storage_root=native_h3_checkpoint_storage_root(),
                **kwargs,
            )
        )


class MiniMaxH3NativeLatentCheckpointLoadT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3NativeLatentCheckpointLoadT8Advanced",
            display_name=(
                "MiniMax H3 Native AV Latent Checkpoint Load / "
                "原生音画Latent检查点加载 (Advanced EXP/T8)"
            ),
            description=(
                "Loads one H3 no-pickle AV latent checkpoint from the bounded ComfyUI output "
                "store, verifies its embedded exact-content manifest, and optionally requires "
                "an independently saved manifest and whole-file SHA-256."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.String.Input(
                    "checkpoint_path",
                    default="",
                    tooltip=(
                        "Relative .h3latent.safetensors path returned by the Save node. It may be "
                        "pasted after a restart or connected directly in the same graph."
                    ),
                ),
                io.String.Input(
                    "expected_manifest_json",
                    default="",
                    multiline=True,
                    advanced=True,
                    tooltip=(
                        "Optional independently retained manifest. When supplied, any latent, "
                        "mask, metadata, dtype, shape or checkpoint-ID mismatch fails closed."
                    ),
                ),
                io.String.Input(
                    "expected_file_sha256",
                    default="",
                    advanced=True,
                    tooltip=(
                        "Optional whole-file SHA-256 returned by Save. Use it to detect replacement "
                        "of both payload and embedded manifest."
                    ),
                ),
                io.Int.Input(
                    "hash_chunk_megabytes",
                    default=8,
                    min=1,
                    max=64,
                    step=1,
                    advanced=True,
                    tooltip="CPU hash chunk bound; it does not change the resulting digest.",
                ),
            ],
            outputs=[
                io.Latent.Output("av_latent"),
                io.String.Output("status"),
                io.Boolean.Output("resume_verified"),
                io.String.Output("checkpoint_id"),
                io.String.Output("content_sha256"),
                io.String.Output("file_sha256"),
                io.String.Output("manifest_json"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(
            *load_native_h3_av_checkpoint(
                storage_root=native_h3_checkpoint_storage_root(),
                **kwargs,
            )
        )

    @classmethod
    def fingerprint_inputs(
        cls,
        checkpoint_path,
        expected_manifest_json="",
        expected_file_sha256="",
        hash_chunk_megabytes=8,
    ):
        try:
            file_fingerprint = fingerprint_native_h3_checkpoint_file(
                native_h3_checkpoint_storage_root(),
                checkpoint_path,
            )
        except (FileNotFoundError, ValueError):
            file_fingerprint = f"unresolved:{checkpoint_path}"
        return (
            f"{file_fingerprint}:{expected_file_sha256}:"
            f"{expected_manifest_json}:{int(hash_chunk_megabytes)}"
        )


NATIVE_LATENT_CHECKPOINT_ADVANCED_NODE_CLASSES = [
    MiniMaxH3NativeLatentCheckpointSaveT8Advanced,
    MiniMaxH3NativeLatentCheckpointLoadT8Advanced,
]
