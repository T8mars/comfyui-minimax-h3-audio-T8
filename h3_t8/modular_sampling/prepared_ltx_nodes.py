"""Opt-in public prepared LTX generation, explicit load and decode nodes."""

from copy import deepcopy
import json
from pathlib import Path

import folder_paths
from comfy_api.latest import InputImpl, io, ui

from ..prepared_identity import absolute_path

CATEGORY = "T8/MiniMax H3/Modular Sampling/Experimental"
BUNDLE = io.Custom("T8_PREPARED_GENERATION_BUNDLE")
GENERATION = io.Custom("T8_MODULAR_PREPARED_LTX_GENERATION")


class MiniMaxH3PreparedLTXGenerateEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3PreparedLTXGenerateEXPT8",
            display_name="Prepared LTX · Generate Refined Latent (T8 EXP)",
            category=CATEGORY, is_experimental=True,
            description="Run only the prepared LTX refinement worker. Commit a SHA-bound latent receipt; "
                        "never start VAE decode or implicitly run another sampler.",
            inputs=[BUNDLE.Input("prepared_bundle"),
                io.Int.Input("noise_seed", default=8301, min=0, max=0xffffffffffffffff),
                io.String.Input("chain_id", default="ltx_split_trial_01"),
                io.Boolean.Input("resume_existing", default=True),
                io.String.Input("serial_lease_path", default="", advanced=True)],
            outputs=[GENERATION.Output("generation_receipt"),
                     io.String.Output("latent_path"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, prepared_bundle, noise_seed, chain_id, resume_existing, serial_lease_path):
        from comfy.model_management import throw_exception_if_processing_interrupted
        from .prepared_ltx_stage import run_generation
        lease = Path(absolute_path(serial_lease_path))
        receipt, report = run_generation(deepcopy(prepared_bundle),
            output_directory=folder_paths.get_output_directory(), chain_id=chain_id,
            noise_seed=noise_seed, resume_existing=resume_existing,
            lease_path=lease, interrupt=throw_exception_if_processing_interrupted)
        return io.NodeOutput(receipt, receipt["latent_path"],
                             json.dumps(report, ensure_ascii=False, indent=2))

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return float("nan")


class MiniMaxH3PreparedLTXLoadGenerationEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3PreparedLTXLoadGenerationEXPT8",
            display_name="Prepared LTX · Load Frozen Refined Latent (T8 EXP)",
            category=CATEGORY, is_experimental=True,
            description="Explicitly load a committed generation SHA after a restart. No GPU worker, "
                        "sampling or decode is started; current bundle/seed/engine identity must match.",
            inputs=[BUNDLE.Input("prepared_bundle"),
                io.Int.Input("noise_seed", default=8301, min=0, max=0xffffffffffffffff),
                io.String.Input("chain_id", default="ltx_split_trial_01"),
                io.String.Input("expected_sha256", default="")],
            outputs=[GENERATION.Output("generation_receipt"),
                     io.String.Output("latent_path"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, prepared_bundle, noise_seed, chain_id, expected_sha256):
        from comfy.model_management import throw_exception_if_processing_interrupted
        from .prepared_ltx_stage import load_generation
        receipt, report = load_generation(deepcopy(prepared_bundle),
            output_directory=folder_paths.get_output_directory(), chain_id=chain_id,
            noise_seed=noise_seed, expected_sha256=expected_sha256,
            interrupt=throw_exception_if_processing_interrupted)
        return io.NodeOutput(receipt, receipt["latent_path"],
                             json.dumps(report, ensure_ascii=False, indent=2))

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return float("nan")


class MiniMaxH3PreparedLTXDecodeEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3PreparedLTXDecodeEXPT8",
            display_name="Prepared LTX · Decode Frozen Latent (T8 EXP)",
            category=CATEGORY, is_experimental=True, is_output_node=True,
            description="Run only the LTX VAE/movie worker from a matching committed latent receipt. "
                        "Original audio is retained by the existing worker; generation is never implicit.",
            inputs=[BUNDLE.Input("prepared_bundle"), GENERATION.Input("generation_receipt"),
                io.String.Input("serial_lease_path", default="", advanced=True)],
            outputs=[io.Video.Output("video"), io.String.Output("saved_path"),
                     io.String.Output("report_json")])

    @classmethod
    def execute(cls, prepared_bundle, generation_receipt, serial_lease_path):
        from comfy.model_management import throw_exception_if_processing_interrupted
        from .prepared_ltx_stage import run_decode
        lease = Path(absolute_path(serial_lease_path))
        path, report = run_decode(deepcopy(prepared_bundle), deepcopy(generation_receipt),
            output_directory=folder_paths.get_output_directory(), lease_path=lease,
            interrupt=throw_exception_if_processing_interrupted)
        relative = Path(path).resolve().relative_to(Path(folder_paths.get_output_directory()).resolve())
        return io.NodeOutput(InputImpl.VideoFromFile(str(path)), str(path),
            json.dumps(report, ensure_ascii=False, indent=2),
            ui=ui.PreviewVideo([ui.SavedResult(relative.name, str(relative.parent), io.FolderType.output)]))

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return float("nan")


NODES = [MiniMaxH3PreparedLTXGenerateEXPT8,
         MiniMaxH3PreparedLTXLoadGenerationEXPT8,
         MiniMaxH3PreparedLTXDecodeEXPT8]
