from comfy_api.latest import io
from .refine_endpoint_conditions import condition_refine_endpoints


class MiniMaxH3RefineEndpointsEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3RefineEndpointsEXPT8",
            display_name="H3 Refine · Delivered End Conditions / 交付首尾条件 (T8 EXP)",
            category="T8/MiniMax H3/Refine/Experimental",is_experimental=True,
            description="Wire after the final latent resize and before the sampler. Conditions the source latent "
                        "tokens covering the actual output trim endpoints. Does not paste RGB or bridge sampled latents. "
                        "Delivery frame coordinates must match Output Trim.",
            inputs=[io.Latent.Input("av_latent"),io.Boolean.Input("enabled",default=False),
                    io.Int.Input("delivery_start_frame",default=0,min=0,max=None),
                    io.Int.Input("delivery_frame_count",default=0,min=0,max=None),
                    io.Int.Input("first_frames",default=1,min=0,max=None),
                    io.Int.Input("last_frames",default=1,min=0,max=None),
                    io.Float.Input("endpoint_strength",default=0.,min=0.,max=1.)],
            outputs=[io.Latent.Output("av_latent"),io.String.Output("report_json")])

    @classmethod
    def execute(cls,av_latent,enabled=False,delivery_start_frame=0,delivery_frame_count=0,
                first_frames=1,last_frames=1,endpoint_strength=0.):
        return io.NodeOutput(*condition_refine_endpoints(av_latent,enabled,delivery_start_frame,
            delivery_frame_count,first_frames,last_frames,endpoint_strength))


NODES=[MiniMaxH3RefineEndpointsEXPT8]
