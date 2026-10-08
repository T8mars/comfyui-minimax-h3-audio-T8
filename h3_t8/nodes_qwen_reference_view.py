"""One separately appended optional view selector; original six schemas stay put."""
from comfy_api.latest import io

from .nodes_reference_package import REFSET, schema
from .qwen_reference_view import select_view
from .reference_package import canonical


class MiniMaxH3ReferenceQwenViewEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(cls, "Qwen View / 独立视觉尺寸", [
            REFSET.Input("reference_set"),
            io.Combo.Input("mode", options=["original", "image_short_edge"], default="original"),
            io.Int.Input("short_edge", default=512, min=32, max=2048, step=32,
                         tooltip="Target SHORT edge; long-edge/pixel caps can lower it. Never enlarge a small image."),
            io.Int.Input("max_long_edge", default=1024, min=32, max=4096, step=32),
            io.Int.Input("max_pixels", default=524288, min=4096, max=4194304, step=1024,
                         tooltip="Per-image pixels supplied to CLIP, not an opaque processor's final token guarantee.")],
            [REFSET.Output("reference_set"), io.String.Output("report_json")],
            "Route -> optional Qwen View -> Apply / Relay Apply. Original is exact old behavior. "
            "Only routed IMAGE reference grounding gets a derived view; stored high-resolution VAE "
            "latent/RGB, video/voice/first/last remain unchanged. Fresh Qwen encode, no embedding edit, "
            "no extra cache or automatic quality/speed claim. Actual dimensions and content hashes reported.")

    @classmethod
    def execute(cls, reference_set, mode="original", short_edge=512, max_long_edge=1024, max_pixels=524288):
        result, report = select_view(reference_set, mode=mode, short_edge=short_edge,
                                   max_long_edge=max_long_edge, max_pixels=max_pixels)
        return io.NodeOutput(result, canonical(report))


NODES = [MiniMaxH3ReferenceQwenViewEXPT8]
