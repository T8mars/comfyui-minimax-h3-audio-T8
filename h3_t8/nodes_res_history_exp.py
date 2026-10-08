"""Append-only opt-in RES boundary setup; ordinary/Euler nodes are unchanged."""
from pathlib import Path

import folder_paths
from comfy_api.latest import io

from .res_history_exp import read_checkpoint
from .res_history_setup import MODES, setup_res_history_sampling


def storage_root():
    return Path(folder_paths.get_output_directory()) / "MiniMaxH3" / "res_checkpoints"


class MiniMaxH3RESHistoryEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3RESHistoryEXPT8",
            display_name="MiniMax H3 RES History / RES历史步边界 (EXP/T8)",
            category="T8/MiniMax H3/Sampling/Experimental", is_experimental=True,
            description=("Deterministic RES on one unchanged sigma table. Saves post-step x plus "
                         "solver history at one chosen boundary, continues the full run, or resumes "
                         "read-only from that boundary. Not LOW/HIGH upscaling or universal Stage; "
                         "actual loaded weights/ordered LoRA/AV coordinates are bound at execution; "
                         "unknown owners remain usable for full sampling, not portable recovery. Legacy Euler/graphs unchanged."),
            inputs=[io.Model.Input("model"), io.Latent.Input("av_latent"),
                    io.Int.Input("steps", default=8, min=1, max=100),
                    io.Float.Input("shift_video", default=12., min=.01, max=100., step=.01),
                    io.Float.Input("shift_audio", default=3., min=.01, max=100., step=.01),
                    io.Combo.Input("mode", options=list(MODES), default="disabled"),
                    io.Int.Input("checkpoint_step", default=4, min=1, max=100),
                    io.String.Input("checkpoint_path", default="res_step4.h3res.safetensors",
                                    tooltip="Relative to output/MiniMaxH3/res_checkpoints. Create-only; unique name per render."),
                    io.String.Input("model_contract_id", default="", multiline=True,
                                    tooltip="Human-readable base/LoRA/strength declaration. Actual loaded MODEL identity is separately checked at execution."),
                    io.String.Input("run_contract_json", default="{}", multiline=True,
                                    tooltip="Connect final NFE Run Contract; required outside disabled mode."),
                    io.Boolean.Input("confirm_checkpoint_write", default=False),
                    io.Int.Input("hash_chunk_megabytes", default=8, min=1, max=64, advanced=True)],
            outputs=[io.Model.Output("model"), io.Sampler.Output("sampler"), io.Sigmas.Output("sigmas"),
                     io.String.Output("status"), io.String.Output("checkpoint_path"),
                     io.String.Output("report_json")])

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*setup_res_history_sampling(storage_root=storage_root(), **kwargs))

    @classmethod
    def fingerprint_inputs(cls, mode, checkpoint_path, **kwargs):
        if mode == "checkpoint":
            # Do not reuse a cached setup to overwrite/reinterpret a file.
            return float("nan")
        if mode == "resume":
            try:
                identity = read_checkpoint(storage_root(), checkpoint_path)["file_sha256"]
            except (OSError, ValueError):
                identity = "unresolved"
            return f"resume:{identity}:{kwargs!r}"
        return f"disabled:{kwargs!r}"


NODES = [MiniMaxH3RESHistoryEXPT8]
