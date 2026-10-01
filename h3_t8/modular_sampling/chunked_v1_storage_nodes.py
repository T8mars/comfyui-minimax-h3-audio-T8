"""Append-only explicit freeze/load nodes for a completed Chunked v1 segment."""
from pathlib import Path

from comfy_api.latest import io
import folder_paths

from .chunked_stage_nodes import CATEGORY, CONTEXT, PLAN, RESULT, SPEC
from .chunked_v1_storage import load_segment, save_segment, verify_segment
from .results import canonical
from .storage import fingerprint_stage


def _store_root():
    return Path(folder_paths.get_output_directory()) / "MiniMaxH3" / "chunked_v1_segment_artifacts"


class MiniMaxH3ChunkedV1SegmentSaveEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.__name__, display_name="H3 Chunked v1 · Freeze ONE Segment (T8 EXP)",
            category=CATEGORY, is_experimental=True, is_output_node=True,
            description="Explicitly freeze one completed v1 cumulative video and its original full-audio "
                        "identity. Copy the returned manifest path and exact SHA. This is not automatic "
                        "MODEL/conditioning cache reuse. Writing is off by default.",
            inputs=[io.Custom(RESULT).Input("segment_result"),
                    io.Latent.Input("source_segment"), io.Custom(SPEC).Input("segment_spec"),
                    io.Custom(CONTEXT).Input("pass2_context"), io.Custom(PLAN).Input("plan"),
                    io.Boolean.Input("confirm_save", default=False)],
            outputs=[io.Latent.Output("cumulative_av_latent"),
                     io.Custom(RESULT).Output("segment_result"),
                     io.String.Output("artifact_path"), io.String.Output("artifact_sha256"),
                     io.String.Output("report_json")],
        )

    @classmethod
    def execute(cls, segment_result, source_segment, segment_spec, pass2_context,
                plan, confirm_save=False):
        if not confirm_save:
            verify_segment(segment_result, source_segment, segment_spec, pass2_context, plan)
            report = canonical({"status": "not_saved_confirm_save_false",
                                "automatic_cache_reuse": False})
            return io.NodeOutput(segment_result.output_latent, segment_result, "", "", report,
                                 ui={"text": (report,)})
        output, result, path, digest, report = save_segment(
            segment_result, source_segment, segment_spec, pass2_context, plan,
            _store_root(),
        )
        return io.NodeOutput(output, result, path, digest, report,
                             ui={"text": ("artifact_path: " + path,
                                          "artifact_sha256: " + digest, report)})


class MiniMaxH3ChunkedV1SegmentLoadEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.__name__, display_name="H3 Chunked v1 · Load Exact Frozen Segment (T8 EXP)",
            category=CATEGORY, is_experimental=True,
            description="Load the explicitly selected completed v1 segment by exact manifest SHA. "
                        "Recheck the current source/plan/segment and rebind current original full audio. "
                        "The next segment can run without resampling the frozen one; changed MODEL or "
                        "CONDITIONING is not treated as an automatic cache hit.",
            inputs=[io.Latent.Input("source_segment"), io.Custom(SPEC).Input("segment_spec"),
                    io.Custom(CONTEXT).Input("pass2_context"), io.Custom(PLAN).Input("plan"),
                    io.String.Input("artifact_path", default=""),
                    io.String.Input("artifact_sha256", default="")],
            outputs=[io.Latent.Output("cumulative_av_latent"),
                     io.Custom(RESULT).Output("segment_result"),
                     io.String.Output("report_json")],
        )

    @classmethod
    def execute(cls, source_segment, segment_spec, pass2_context, plan,
                artifact_path, artifact_sha256):
        return io.NodeOutput(*load_segment(
            source_segment, segment_spec, pass2_context, plan,
            _store_root(), artifact_path, artifact_sha256,
        ))

    @classmethod
    def fingerprint_inputs(cls, artifact_path, artifact_sha256, **_kwargs):
        try:
            return fingerprint_stage(_store_root(), artifact_path)
        except (ValueError, OSError, RuntimeError):
            return float("nan")


NODES = [MiniMaxH3ChunkedV1SegmentSaveEXPT8, MiniMaxH3ChunkedV1SegmentLoadEXPT8]
