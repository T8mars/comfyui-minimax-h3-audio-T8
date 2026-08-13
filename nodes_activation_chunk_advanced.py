from __future__ import annotations

from comfy_api.latest import io

from .activation_chunk_advanced import canonical_json, configure_activation_chunk


CATEGORY = "T8/MiniMax H3/Models/Experimental"


class MiniMaxH3ActivationChunkT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3ActivationChunkT8Advanced",
            display_name="MiniMax H3 MLP Activation Chunk / MLP激活分块 (Advanced)",
            description=(
                "Clone-local, fail-closed H3 DiT MLP token chunking. Attention is unchanged. "
                "Current TensorWise INT8 kernels already fuse SwiGLU and may show no memory benefit. "
                "Default report_only makes no MODEL change; apply_exp rejects any existing "
                "dit/double_block owner instead of overwriting Block Cache or Sage wrappers."
            ),
            category=CATEGORY,
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input(
                    "mode",
                    options=["report_only", "apply_exp"],
                    default="report_only",
                ),
                io.Int.Input("chunk_rows", default=256, min=16, max=65536, step=16),
                io.Int.Input("block_start", default=0, min=0, max=49, step=1, advanced=True),
                io.Int.Input("block_end", default=49, min=0, max=49, step=1, advanced=True),
                io.Boolean.Input("preserve_short_path", default=True, advanced=True),
                io.Int.Input("expected_width", default=736, min=32, max=16384, step=32),
                io.Int.Input("expected_height", default=416, min=32, max=16384, step=32),
                io.Int.Input("expected_length", default=124, min=5, max=3600, step=17),
                io.Int.Input(
                    "expected_single_image_references",
                    default=0,
                    min=0,
                    max=32,
                    advanced=True,
                ),
            ],
            outputs=[io.Model.Output("model"), io.String.Output("report_json")],
            is_experimental=True,
        )

    @classmethod
    def execute(
        cls,
        model,
        mode,
        chunk_rows,
        block_start,
        block_end,
        preserve_short_path,
        expected_width,
        expected_height,
        expected_length,
        expected_single_image_references,
    ):
        patched, report = configure_activation_chunk(
            model,
            mode,
            chunk_rows,
            block_start,
            block_end,
            preserve_short_path,
            expected_width,
            expected_height,
            expected_length,
            expected_single_image_references,
        )
        return io.NodeOutput(patched, canonical_json(report))


ACTIVATION_CHUNK_ADVANCED_NODE_CLASSES = [
    MiniMaxH3ActivationChunkT8Advanced,
]
