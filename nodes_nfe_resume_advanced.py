from __future__ import annotations

from pathlib import Path

import folder_paths
from comfy_api.latest import io

from .nfe_resume_advanced import (
    NFE_RESUME_MODES,
    fingerprint_nfe_resume_checkpoint,
    setup_nfe_resume_sampling,
)


CATEGORY = "T8/MiniMax H3/Long Video/Experimental"


def nfe_checkpoint_storage_root() -> Path:
    output_root = Path(folder_paths.get_output_directory()).resolve()
    return (output_root / "MiniMaxH3" / "nfe_checkpoints").resolve()


class MiniMaxH3NFEResumeSamplerT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3NFEResumeSamplerT8Advanced",
            display_name=(
                "MiniMax H3 Dual-Clock NFE Checkpoint + Resume / "
                "双时钟步边界断点恢复 (Advanced EXP/T8)"
            ),
            description=(
                "Append-only experimental sampler setup for exact completed-step checkpoints "
                "of T8 dual_clock_euler/native_flow. It atomically stores the packed AV state, "
                "original noise/latent/mask and complete sigma table after every finished NFE. "
                "Other samplers, mid-forward recovery and undeclared model/conditioning changes "
                "fail closed. Existing samplers and workflows are unchanged."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Model.Input("model"),
                io.Latent.Input("av_latent"),
                io.Int.Input("steps", default=8, min=1, max=100, step=1),
                io.Float.Input(
                    "shift_video",
                    default=12.0,
                    min=0.01,
                    max=100.0,
                    step=0.01,
                ),
                io.Float.Input(
                    "shift_audio",
                    default=3.0,
                    min=0.01,
                    max=100.0,
                    step=0.01,
                ),
                io.Combo.Input(
                    "mode",
                    options=list(NFE_RESUME_MODES),
                    default="disabled",
                    tooltip=(
                        "disabled performs the same Euler math without file I/O; "
                        "checkpoint_each_step writes after every completed NFE; resume loads "
                        "the last valid boundary and emits only its remaining sigmas."
                    ),
                ),
                io.String.Input(
                    "checkpoint_path",
                    default="h3_nfe_resume.h3nfe.safetensors",
                    tooltip=(
                        "Relative path under output/MiniMaxH3/nfe_checkpoints. Absolute paths, "
                        "traversal and symlinks are rejected. Use a unique name per render."
                    ),
                ),
                io.String.Input(
                    "model_contract_id",
                    default="",
                    multiline=True,
                    tooltip=(
                        "Required outside disabled mode. Record the exact base checkpoint/hash, "
                        "all LoRAs and strengths, and attention/model wrappers. A mismatch blocks "
                        "resume; the node does not pretend this text hashes all loaded weights."
                    ),
                ),
                io.String.Input(
                    "run_contract_json",
                    default="{}",
                    multiline=True,
                    tooltip=(
                        "Required non-empty JSON outside disabled mode. Connect the MiniMax H3 "
                        "NFE Run Contract output, or paste another strict immutable JSON object. "
                        "The plain Conditioning report is text, not valid JSON."
                    ),
                ),
                io.Boolean.Input(
                    "confirm_checkpoint_write",
                    default=False,
                    tooltip=(
                        "Safety gate. Required for checkpoint_each_step. In resume mode, false "
                        "continues read-only; true also advances the same atomic checkpoint."
                    ),
                ),
                io.Boolean.Input(
                    "allow_replace_existing",
                    default=False,
                    advanced=True,
                    tooltip=(
                        "Only for intentionally starting a new session at an existing path. "
                        "Resume of the already verified same session does not need this switch."
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
                        "Upper CPU chunk used to verify every checkpoint tensor. It changes "
                        "verification memory/speed, never the mathematical state."
                    ),
                ),
            ],
            outputs=[
                io.Model.Output("model"),
                io.Sampler.Output("sampler"),
                io.Sigmas.Output("sigmas"),
                io.String.Output("status"),
                io.String.Output("checkpoint_path"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(
            *setup_nfe_resume_sampling(
                storage_root=nfe_checkpoint_storage_root(),
                **kwargs,
            )
        )

    @classmethod
    def fingerprint_inputs(cls, mode, checkpoint_path, **kwargs):
        if mode != "resume":
            return f"{mode}:{checkpoint_path}:{kwargs!r}"
        try:
            file_sha256 = fingerprint_nfe_resume_checkpoint(
                nfe_checkpoint_storage_root(),
                checkpoint_path,
            )
        except (FileNotFoundError, ValueError):
            file_sha256 = f"unresolved:{checkpoint_path}"
        return f"resume:{file_sha256}:{kwargs!r}"


NFE_RESUME_ADVANCED_NODE_CLASSES = [MiniMaxH3NFEResumeSamplerT8Advanced]
