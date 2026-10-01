"""Append-only Chunked first-pass source extraction; no hidden sampling."""
from comfy_api.latest import io

from .chunked_source import slice_chunked_source


class MiniMaxH3ChunkedSourceSegmentEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.__name__, display_name="H3 Chunked · Slice ONE First-Pass Segment (T8 EXP)",
            category="T8/MiniMax H3/Modular Sampling/Chunked Experimental", is_experimental=True,
            description="Extracts exactly one pre-upscale AV time segment from a supplied first-pass latent. "
                        "Use its latent with an external learned upscaler, then a separate PASS2 stage. "
                        "This node does not sample or prove the supplied first pass completed; "
                        "v5 joint-4+4 uses a different global-upscale contract and is rejected.",
            inputs=[io.Latent.Input("first_pass_latent"),
                    io.Custom("T8_H3_CHUNKED_TWO_PASS_PLAN").Input("plan"),
                    io.Int.Input("segment_index", default=0, min=0, max=9999)],
            outputs=[io.Latent.Output("segment_latent"),
                     io.Custom("T8_CHUNKED_SOURCE_SEGMENT").Output("segment_spec"),
                     io.String.Output("report_json")],
        )

    @classmethod
    def execute(cls, first_pass_latent, plan, segment_index=0):
        return io.NodeOutput(*slice_chunked_source(first_pass_latent, plan, segment_index))


NODES = [MiniMaxH3ChunkedSourceSegmentEXPT8]
