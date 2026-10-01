"""Additive Multi-Face source Save/Load. Old graphs and audits stay unchanged."""
from pathlib import Path

import folder_paths
from comfy_api.latest import io

from . import face_source_storage as storage
from .storage import fingerprint_stage

CATEGORY = "T8/MiniMax H3/Modular Sampling/Experimental"
PLAN = io.Custom("H3_T8_FACE_REFINE_PARITY_PLAN")
RESULT = io.Custom("T8_STAGE_RESULT")


def _root():
    return Path(folder_paths.get_output_directory()) / "MiniMaxH3" / "multiface_sources"


def _outputs():
    return [PLAN.Output("face_plan"), io.Image.Output("source_frames"), io.Latent.Output("av_latent")]


class MiniMaxH3MultiFaceSourceSaveEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, display_name="H3 Multi-Face · Save Exact Source Inputs (T8 EXP)",
            category=CATEGORY, is_experimental=True, is_output_node=True,
            description="Explicitly save one character's original plan, source window and encoded AV after "
                        "verified sampling. Binds exact parent RGB/audio and independent StageResult. No approval.",
            inputs=[RESULT.Input("stage_result"), PLAN.Input("face_plan"), io.Image.Input("source_frames"),
                    io.Image.Input("parent_frames"), io.Latent.Input("av_latent"), io.Audio.Input("source_audio"),
                    io.String.Input("filename_prefix", default="multiface_source"),
                    io.Boolean.Input("confirm_save", default=False)],
            outputs=[*_outputs(), io.String.Output("artifact_path"), io.String.Output("artifact_sha256"),
                     io.String.Output("report_json")])

    @classmethod
    def execute(cls, stage_result, face_plan, source_frames, parent_frames, av_latent, source_audio,
                filename_prefix="multiface_source", confirm_save=False):
        if type(confirm_save) is not bool:
            raise ValueError("confirm_save must be a boolean")
        values = storage.source_values(stage_result, face_plan, source_frames, parent_frames, av_latent, source_audio)
        path = digest = ""
        report = storage.report("save_disabled_passthrough", "")
        if confirm_save:
            from comfy.model_management import throw_exception_if_processing_interrupted as interrupt
            path, digest, report = storage.save_source(values, stage_result, parent_frames, source_audio,
                                                       _root(), filename_prefix, interrupt=interrupt)
        return io.NodeOutput(face_plan, source_frames, av_latent, path, digest, report)


class MiniMaxH3MultiFaceSourceLoadEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, display_name="H3 Multi-Face · Load Exact Source Inputs (T8 EXP)",
            category=CATEGORY, is_experimental=True,
            description="Restore selected original plan/window/AV without rerunning SAM, VAE encode or sampling. "
                        "Requires exact independent completed stage and current full parent RGB/audio. No approval.",
            inputs=[RESULT.Input("stage_result"), io.Image.Input("parent_frames"), io.Audio.Input("source_audio"),
                    io.String.Input("artifact_path", default=""), io.String.Input("artifact_sha256", default="")],
            outputs=[*_outputs(), io.String.Output("report_json")])

    @classmethod
    def execute(cls, stage_result, parent_frames, source_audio, artifact_path, artifact_sha256):
        from comfy.model_management import throw_exception_if_processing_interrupted as interrupt
        values, report = storage.load_source(_root(), artifact_path, artifact_sha256, stage_result,
                                             parent_frames, source_audio, interrupt=interrupt)
        return io.NodeOutput(*(values[name] for name in ("face_plan", "source_frames", "av_latent")), report)

    @classmethod
    def fingerprint_inputs(cls, artifact_path, artifact_sha256, **_inputs):
        storage._digest(artifact_sha256)
        return fingerprint_stage(_root(), artifact_path)


NODES = [MiniMaxH3MultiFaceSourceSaveEXPT8, MiniMaxH3MultiFaceSourceLoadEXPT8]
