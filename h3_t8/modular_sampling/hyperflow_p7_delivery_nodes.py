"""Append-only candidate and explicit review/accept nodes for separated P7."""
from comfy_api.latest import io

from ..nodes_long_video_delivery_exp import _preview_video
from . import hyperflow_p7_delivery as p7_delivery
from .hyperflow_p7_nodes import HIGH_RESULT, CATEGORY


class MiniMaxH3HyperFlowP7CandidateSaveEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__,
            display_name="H3 HyperFlow P7 · Save Separate Candidate (T8 EXP)",
            category=CATEGORY, is_experimental=True, is_output_node=True,
            description="Decode the authenticated completed HIGH AV, trim the context, optionally color-match "
                "the accepted predecessor, and save an immutable candidate. A continuable segment also "
                "writes LOW video with completed HIGH audio and a 4+4 effects audit. This does not accept it.",
            inputs=[io.Custom(HIGH_RESULT).Input("high_result"), io.Vae.Input("video_vae"),
                io.Vae.Input("audio_vae"),
                io.String.Input("candidate_id", default=""),
                io.Boolean.Input("is_final_segment", default=False),
                io.Int.Input("final_frame_count", default=0, min=0, max=100000),
                io.String.Input("model_id", default="h3_p7_modular_exp"),
                io.Int.Input("seed", default=0, min=0, max=2**64 - 1),
                io.Boolean.Input("color_match", default=True),
                io.Combo.Input("bit_depth", options=[8, 10], default=8),
                io.Int.Input("crf", default=18, min=0, max=51)],
            outputs=[io.String.Output("candidate_json_path"),
                io.Video.Output("candidate_video"), io.String.Output("job_sha256"),
                io.String.Output("report_json")])

    @classmethod
    def execute(cls, high_result, video_vae, audio_vae, candidate_id="", is_final_segment=False,
                final_frame_count=0, model_id="h3_p7_modular_exp", seed=0, color_match=True,
                bit_depth=8, crf=18):
        candidate_json, movie, job_sha256, report = p7_delivery.save_candidate(
            high_result, video_vae, audio_vae, candidate_id=candidate_id,
            is_final_segment=is_final_segment, final_frame_count=final_frame_count,
            model_id=model_id, seed=seed, color_match=color_match, bit_depth=bit_depth, crf=crf)
        video, preview = _preview_video(movie)
        return io.NodeOutput(candidate_json, video, job_sha256, report, ui=preview)


class MiniMaxH3HyperFlowP7CandidateAcceptEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__,
            display_name="H3 HyperFlow P7 · Review & Accept Separate Candidate (T8 EXP)",
            category=CATEGORY, is_experimental=True, is_output_node=True,
            description="Default is preview only. Before explicit acceptance, verify candidate MP4/HIGH "
                "context, P7 LOW context with completed audio and the audit job SHA. Preserve old "
                "manifest rules, reject replacements and never auto-accept an incomplete candidate. "
                "Connect returned ID/revision/job SHA to the next Accepted Parent selection.",
            inputs=[io.String.Input("candidate_json_path", default="", force_input=True),
                io.String.Input("job_sha256", default="", force_input=True),
                io.Boolean.Input("accept_candidate", default=False)],
            outputs=[io.Video.Output("video"), io.Boolean.Output("accepted"),
                io.String.Output("manifest_path"), io.String.Output("parent_candidate_id"),
                io.Int.Output("parent_revision"), io.String.Output("previous_job_sha256"),
                io.String.Output("report_json")])

    @classmethod
    def execute(cls, candidate_json_path, job_sha256, accept_candidate=False):
        movie, accepted, manifest, parent, revision, report = p7_delivery.accept_candidate(
            candidate_json_path, job_sha256, accept=accept_candidate)
        video, preview = _preview_video(movie)
        return io.NodeOutput(video, accepted, manifest, parent, revision,
                             job_sha256 if accepted else "", report, ui=preview)


NODES = [MiniMaxH3HyperFlowP7CandidateSaveEXPT8,
         MiniMaxH3HyperFlowP7CandidateAcceptEXPT8]
