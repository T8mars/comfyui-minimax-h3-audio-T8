"""Additive source Save/Load for the two existing external RGB/LTX samplers."""
from pathlib import Path
import json

import folder_paths
from comfy_api.latest import io

from . import ltx_rgb_source_storage as storage
from .storage import fingerprint_stage

CATEGORY = "T8/MiniMax H3/Modular Sampling/Experimental"


def _root():
    return Path(folder_paths.get_output_directory()) / "MiniMaxH3" / "ltx_rgb_sources"


def _outputs():
    return [io.Image.Output("source_frames"), io.Audio.Output("source_audio"),
            io.Image.Output("prepared_frames"), io.String.Output("prep_report_json"),
            io.Latent.Output("ltx_latent"), io.Float.Output("fps"),
            io.Float.Output("duration_seconds"), io.Combo.Output("bit_depth")]


def _values(values):
    report = json.loads(values["prep_report_json"])
    return (*(values[name] for name in
              ("source_frames", "source_audio", "prepared_frames", "prep_report_json", "ltx_latent")),
            report["fps"], report["output_duration_seconds"], values["bit_depth"])


class MiniMaxH3LTXRGBSourceSaveEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, display_name="H3→LTX RGB · Save Frozen Input (T8 EXP)",
            category=CATEGORY, is_experimental=True, is_output_node=True,
            description="Explicit data-only handoff before LTX sampling. Save original RGB/audio, prepared "
                        "RGB and lifted LTX LATENT. No model or sampler completion is certified.",
            inputs=[io.Image.Input("source_frames"), io.Audio.Input("source_audio"),
                    io.Image.Input("prepared_frames"), io.String.Input("prep_report_json"),
                    io.Latent.Input("ltx_latent"), io.Combo.Input("bit_depth", options=[8, 10], default=8),
                    io.String.Input("filename_prefix", default="rgb_source"),
                    io.Boolean.Input("confirm_save", default=False)],
            outputs=[*_outputs(), io.String.Output("artifact_path"), io.String.Output("artifact_sha256"),
                     io.String.Output("report_json")])

    @classmethod
    def execute(cls, source_frames, source_audio, prepared_frames, prep_report_json, ltx_latent,
                bit_depth=8, filename_prefix="rgb_source", confirm_save=False):
        from comfy.model_management import throw_exception_if_processing_interrupted as interrupt
        values = dict(source_frames=source_frames, source_audio=source_audio, prepared_frames=prepared_frames,
                      prep_report_json=prep_report_json, ltx_latent=ltx_latent, bit_depth=bit_depth)
        if type(confirm_save) is not bool:
            raise ValueError("confirm_save must be a boolean")
        storage.validate_source(values)
        if not confirm_save:
            return io.NodeOutput(*_values(values), "", "", storage._report_result("save_disabled_passthrough", ""))
        path, sha, report = storage.save_source(values, _root(), filename_prefix, interrupt=interrupt)
        return io.NodeOutput(*_values(values), path, sha, report)


class MiniMaxH3LTXRGBSourceLoadEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, display_name="H3→LTX RGB · Load Frozen Input (T8 EXP)",
            category=CATEGORY, is_experimental=True,
            description="Read exact selected input data without video loading, VAE encoding, upscaling or "
                        "H3 sampling. Rebuild current LTX MODEL/conditions and bind a new stage separately.",
            inputs=[io.String.Input("artifact_path", default=""), io.String.Input("artifact_sha256", default="")],
            outputs=[*_outputs(), io.String.Output("report_json")])

    @classmethod
    def execute(cls, artifact_path, artifact_sha256):
        from comfy.model_management import throw_exception_if_processing_interrupted as interrupt
        values, report = storage.load_source(_root(), artifact_path, artifact_sha256, interrupt=interrupt)
        return io.NodeOutput(*_values(values), report)

    @classmethod
    def fingerprint_inputs(cls, artifact_path, artifact_sha256):
        # Same bounded manifest/state-file envelope, but not a StageResult.
        # Contents, not mtime, invalidate cached consumers. Missing/bad files
        # cause Core to invalidate and then execute the strict Load failure.
        storage._digest(artifact_sha256)
        return fingerprint_stage(_root(), artifact_path)


NODES = [MiniMaxH3LTXRGBSourceSaveEXPT8, MiniMaxH3LTXRGBSourceLoadEXPT8]
