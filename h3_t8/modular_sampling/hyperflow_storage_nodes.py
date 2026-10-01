"""Append-only public storage nodes for exact continuous HyperFlow stages."""
from pathlib import Path

import folder_paths
from comfy_api.latest import io

from . import hyperflow_storage as storage
from .hyperflow_nodes import BOUNDARY, RESULT, CATEGORY


def _store_root():
    return Path(folder_paths.get_output_directory()) / "MiniMaxH3" / "hyperflow_stage_artifacts"


def _schema(cls, title, description, inputs, outputs, *, output_node=False):
    return io.Schema(node_id=cls.__name__, display_name="H3 HyperFlow · " + title,
        category=CATEGORY, is_experimental=True, description=description,
        inputs=inputs, outputs=outputs, is_output_node=output_node)


def _fingerprint(path, kind):
    try:
        return storage.fingerprint(_store_root(), path, kind)
    except (OSError, ValueError, RuntimeError):
        return float("nan")


class MiniMaxH3HyperFlowHeadSaveEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return _schema(cls, "Save Frozen HEAD (T8 EXP)",
            "Save exact raw x_sigma and separate Core scaffold; NOT completed clean x0. "
            "New unique artifact under output/MiniMaxH3/hyperflow_stage_artifacts. Save both path and SHA. "
            "Unknown user owners may be archived but cannot acquire certified cross-process reuse.",
            [io.Custom(BOUNDARY).Input("continuous_boundary"), io.String.Input("prefix", default="HEAD")],
            [io.Custom(BOUNDARY).Output("continuous_boundary"), io.String.Output("artifact_path"),
             io.String.Output("artifact_sha256"), io.String.Output("report_json")], output_node=True)

    @classmethod
    def execute(cls, continuous_boundary, prefix="HEAD"):
        path, digest, report = storage.save_boundary(continuous_boundary, _store_root(), prefix)
        return io.NodeOutput(continuous_boundary, path, digest, report,
            ui={"text": ("artifact_path: " + path, "artifact_sha256: " + digest, report)})


class MiniMaxH3HyperFlowHeadLoadEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return _schema(cls, "Load Exact Frozen HEAD (T8 EXP)",
            "Load the selected path and SHA without MODEL loading or HEAD execution. Connect directly to "
            "continuous TAIL; same original base/HyperFlow adapter contents are checked there. TAIL conditions "
            "and content LoRAs remain independent. Remove HEAD output branches in a TAIL-only graph. "
            "Missing/corrupt/busy artifacts fail; no hidden regeneration. Not fresh-noise8+4/partial4+4/P7.",
            [io.String.Input("artifact_path", default=""), io.String.Input("artifact_sha256", default="")],
            [io.Custom(BOUNDARY).Output("continuous_boundary"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, artifact_path, artifact_sha256):
        result, report = storage.load_boundary(_store_root(), artifact_path, artifact_sha256)
        return io.NodeOutput(result, report)

    @classmethod
    def fingerprint_inputs(cls, artifact_path, artifact_sha256):
        return _fingerprint(artifact_path, "head")


class MiniMaxH3HyperFlowTailSaveEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return _schema(cls, "Save Completed TAIL AV (T8 EXP)",
            "Save completed AV and its actual continuous TAIL receipt, no extra sampling. "
            "Unique new file; no overwrite, automatic cache lookup or quality claim. Save both path and SHA.",
            [io.Custom(RESULT).Input("completed_result"), io.String.Input("prefix", default="TAIL")],
            [io.Latent.Output("av_latent"), io.String.Output("artifact_path"),
             io.String.Output("artifact_sha256"), io.String.Output("report_json")], output_node=True)

    @classmethod
    def execute(cls, completed_result, prefix="TAIL"):
        path, digest, report = storage.save_result(completed_result, _store_root(), prefix)
        return io.NodeOutput(completed_result.output, path, digest, report,
            ui={"text": ("artifact_path: " + path, "artifact_sha256: " + digest, report)})


class MiniMaxH3HyperFlowTailLoadEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return _schema(cls, "Load Exact Completed TAIL AV (T8 EXP)",
            "Load frozen completed AV for decoding/delivery without diffusion MODEL, CLIP, HEAD or TAIL. "
            "Requires exact path/SHA. Does not claim today's changed settings match this selected artifact.",
            [io.String.Input("artifact_path", default=""), io.String.Input("artifact_sha256", default="")],
            [io.Latent.Output("av_latent"), io.Custom(RESULT).Output("completed_result"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, artifact_path, artifact_sha256):
        result, report = storage.load_result(_store_root(), artifact_path, artifact_sha256)
        return io.NodeOutput(result.output, result, report)

    @classmethod
    def fingerprint_inputs(cls, artifact_path, artifact_sha256):
        return _fingerprint(artifact_path, "tail")


NODES = [MiniMaxH3HyperFlowHeadSaveEXPT8, MiniMaxH3HyperFlowHeadLoadEXPT8,
         MiniMaxH3HyperFlowTailSaveEXPT8, MiniMaxH3HyperFlowTailLoadEXPT8]
